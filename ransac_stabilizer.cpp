#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudacodec.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <future>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
//  RANSAC Stabilizer with ArUco Displacement Tracking
//
//  1. Uses ORB+RANSAC to stabilize background motion
//  2. Tracks ArUco marker (ID 0) displacement after stabilization
//  3. Compares measured displacement to ground truth structural vibration
// ─────────────────────────────────────────────────────────────────────────────

// ── Frame geometry ────────────────────────────────────────────────────────────
static constexpr int ARUCO_SIZE = 150;
static constexpr int EXCLUSION_MARGIN = 40;

// ── Refined Parameters for Sub-pixel Accuracy ───────────────────────────────
// 2500 features: with quality-weighted top-60% selection still yields
// 500–1000 inliers — the affine fit is well-determined.  Cuts ORB cost ~2×.
static constexpr int   ORB_FEATURES   = 2500;
static constexpr float LOWE_RATIO     = 0.65f;  // Tighter ratio test (was 0.7)
static constexpr double RANSAC_THRESH = 3.0;    // Wider threshold for larger displacement
static constexpr int   MIN_MATCHES    = 25;     // More matches required
static constexpr double HOMOGRAPHY_CONFIDENCE = 0.995;  // Higher confidence
// Reference refresh: update reference when inlier ratio drops below this
// to prevent match degradation as the scene drifts from frame 0.
static constexpr double MIN_INLIER_RATIO = 0.10;
static constexpr int    KEYFRAME_INTERVAL = 1000; // refresh keyframe every N frames (long enough to span camera-vibration cycles)
static constexpr int    ROT_SMOOTH_N = 10;        // smooth rotation over N frames
static constexpr int    REFRESH_INLIER_THRESHOLD = 60; // refresh if inliers < this
// Subject mask is expanded by this fraction on every side ("halo") so the
// rectangle's corners don't graze the rotor disk.  Pure mask geometry —
// doesn't filter any frequency band.
static constexpr double SUBJECT_HALO_FRAC = 0.10;
// Keep top fraction of Lowe-passing matches by Harris response; high-response
// keypoints are more localizable, so the affine fit is sharper.  Independent
// ranking — does not filter frequencies.
static constexpr double MATCH_QUALITY_KEEP = 0.60;
// Auto-disable ArUco if no marker is found in the first N frames (rotor
// videos have no marker, saves ~0.5–1 ms/frame).
static constexpr int    ARUCO_PROBE_FRAMES = 100;


struct DisplacementData {
    int frame_id;
    cv::Point2f measured_displacement;
    cv::Point2f ground_truth_displacement;
    double error_magnitude;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Stage 2 — Farneback Dense Optical Flow Residual Correction
//
//  After RANSAC removes the large motion (~50px), there's ~0.8px residual.
//  Farneback computes dense flow between the stabilized frame and the
//  reference (frame 0), takes the median background flow as the residual,
//  and applies a small translational correction.
//
//  This is integrated directly into the RANSAC stabilizer so both stages
//  run in a single pass per frame.
// ─────────────────────────────────────────────────────────────────────────────

class FarnebackRefinement {
public:
    FarnebackRefinement() {
        farneback_ = cv::cuda::FarnebackOpticalFlow::create();
        farneback_->setNumLevels(3);
        farneback_->setPyrScale(0.5);
        farneback_->setWinSize(15);
        farneback_->setNumIters(3);
        farneback_->setPolyN(5);
        farneback_->setPolySigma(1.2);
        farneback_->setFlags(0);
    }

    // Set reference from a GPU-side BGR frame (no CPU round-trip).
    void setReferenceGpu(const cv::cuda::GpuMat& ref_gpu_bgr) {
        cv::cuda::cvtColor(ref_gpu_bgr, ref_gray_gpu_, cv::COLOR_BGR2GRAY);
        ref_set_ = true;
    }

    // Additional region to exclude from the background flow sampling
    // (e.g. vibrating rotor subject). Intersects any existing exclusions.
    void setSubjectRegion(const cv::Rect& r) { subject_rect_ = r; }

    // NVENC path: skip the final cv::Mat download.  Caller reads the
    // result GpuMat via lastGpuOutput() and passes it directly to the
    // NVENC writer.
    void setSkipDownload(bool v) { skip_download_ = v; }
    const cv::cuda::GpuMat& lastGpuOutput() const { return last_output_; }

    // GPU-native refinement: takes the original frame + stage-1 stabilized
    // frame (both on GPU) and the stage-1 transform H_smooth. Computes the
    // residual correction H_corr from dense flow, composes H_final =
    // H_corr * H_smooth, and applies ONE warp to the original frame.
    //
    // Benefits vs. the old cv::Mat refine():
    //   * No intermediate download/upload of the stage-1 result.
    //   * Single interpolation pass (composed warp) instead of two cascaded.
    //   * Reuses preallocated cur_gray_gpu_/flow_gpu_/gpu_final_ buffers.
    //
    // Reference is locked to frame 0 for the whole clip — no periodic refresh,
    // because refreshing at an arbitrary phase of camera vibration bakes that
    // phase into the zero and defeats low-frequency correction.
    cv::Mat refineComposed(const cv::cuda::GpuMat& gpu_frame_bgr,
                           const cv::cuda::GpuMat& gpu_stabilized_bgr,
                           const cv::Mat& H_smooth,
                           double& residual_dx, double& residual_dy) {
        residual_dx = 0.0;
        residual_dy = 0.0;

        if (!ref_set_) {
            setReferenceGpu(gpu_stabilized_bgr);
            last_output_ = gpu_stabilized_bgr;
            cv::Mat out;
            if (!skip_download_) gpu_stabilized_bgr.download(out);
            return out;
        }

        // Convert stabilized (stage-1) to gray on GPU — no CPU trip
        cv::cuda::cvtColor(gpu_stabilized_bgr, cur_gray_gpu_, cv::COLOR_BGR2GRAY);

        // Dense optical flow on GPU
        farneback_->calc(ref_gray_gpu_, cur_gray_gpu_, flow_gpu_);

        flow_gpu_.download(flow_cpu_);
        cv::Mat& flow = flow_cpu_;

        // Build mask: exclude centre marker region, borders, and (if set)
        // the vibrating subject region — otherwise the rigid fit absorbs
        // some of the subject motion as "camera residual."
        if (bg_mask_.empty() || bg_mask_.size() != flow.size()) {
            bg_mask_ = cv::Mat::ones(flow.rows, flow.cols, CV_8U);
            int cx = flow.cols / 2, cy = flow.rows / 2;
            int half = (ARUCO_SIZE + 2 * EXCLUSION_MARGIN) / 2;
            cv::Rect roi(cx - half, cy - half, half * 2, half * 2);
            roi &= cv::Rect(0, 0, flow.cols, flow.rows);
            bg_mask_(roi) = 0;
            bg_mask_(cv::Rect(0, 0, flow.cols, 20)) = 0;
            bg_mask_(cv::Rect(0, flow.rows - 20, flow.cols, 20)) = 0;
            bg_mask_(cv::Rect(0, 0, 20, flow.rows)) = 0;
            bg_mask_(cv::Rect(flow.cols - 20, 0, 20, flow.rows)) = 0;
            if (subject_rect_.area() > 0) {
                cv::Rect s = subject_rect_ & cv::Rect(0, 0, flow.cols, flow.rows);
                if (s.area() > 0) bg_mask_(s) = 0;
            }
        }

        // Collect background flow samples WITH positions (for rotation estimation)
        std::vector<cv::Point2f> src_pts, dst_pts;
        src_pts.reserve(5000);
        dst_pts.reserve(5000);

        // Sample every 8th pixel in the background mask (enough for rigid fit)
        for (int y = 20; y < flow.rows - 20; y += 8) {
            const auto* frow = flow.ptr<cv::Vec2f>(y);
            const auto* mrow = bg_mask_.ptr<uchar>(y);
            for (int x = 20; x < flow.cols - 20; x += 8) {
                if (mrow[x]) {
                    src_pts.push_back(cv::Point2f(x + frow[x][0], y + frow[x][1]));
                    dst_pts.push_back(cv::Point2f(x, y));
                }
            }
        }

        // Fallback if we couldn't collect enough samples: just download stage-1
        if (src_pts.size() < 10) {
            last_output_ = gpu_stabilized_bgr;
            cv::Mat out;
            if (!skip_download_) gpu_stabilized_bgr.download(out);
            return out;
        }

        // Fit rigid transform (translation + rotation) to the flow field
        cv::Mat rigid = cv::estimateAffinePartial2D(src_pts, dst_pts,
                                                     cv::noArray(), cv::RANSAC, 1.5);

        if (rigid.empty()) {
            last_output_ = gpu_stabilized_bgr;
            cv::Mat out;
            if (!skip_download_) gpu_stabilized_bgr.download(out);
            return out;
        }

        double res_tx = rigid.at<double>(0, 2);
        double res_ty = rigid.at<double>(1, 2);
        residual_dx = res_tx;
        residual_dy = res_ty;

        // If residual is below noise floor, skip the composed warp and just
        // return the stage-1 result (download only)
        double residual_mag = std::sqrt(res_tx*res_tx + res_ty*res_ty);
        double res_rot = std::atan2(rigid.at<double>(1, 0), rigid.at<double>(0, 0));
        if (residual_mag < 0.02 && std::abs(res_rot) < 1e-5) {
            last_output_ = gpu_stabilized_bgr;
            cv::Mat out;
            if (!skip_download_) gpu_stabilized_bgr.download(out);
            return out;
        }

        // Build H_corr (3x3) from the 2x3 rigid affine
        cv::Mat H_corr = cv::Mat::eye(3, 3, CV_64F);
        H_corr.at<double>(0, 0) = rigid.at<double>(0, 0);
        H_corr.at<double>(0, 1) = rigid.at<double>(0, 1);
        H_corr.at<double>(0, 2) = rigid.at<double>(0, 2);
        H_corr.at<double>(1, 0) = rigid.at<double>(1, 0);
        H_corr.at<double>(1, 1) = rigid.at<double>(1, 1);
        H_corr.at<double>(1, 2) = rigid.at<double>(1, 2);

        // Compose: final(x) = H_corr(H_smooth(x)) = (H_corr * H_smooth) * x
        // One warp of the ORIGINAL frame replaces two cascaded warps.
        cv::Mat H_final = H_corr * H_smooth;
        cv::cuda::warpPerspective(gpu_frame_bgr, gpu_final_, H_final,
                                  gpu_frame_bgr.size(),
                                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        last_output_ = gpu_final_;
        cv::Mat corrected;
        if (!skip_download_) gpu_final_.download(corrected);
        return corrected;
    }

private:
    cv::Ptr<cv::cuda::FarnebackOpticalFlow> farneback_;
    cv::cuda::GpuMat ref_gray_gpu_;
    cv::cuda::GpuMat cur_gray_gpu_;   // reused each frame
    cv::cuda::GpuMat flow_gpu_;       // reused
    cv::cuda::GpuMat gpu_final_;      // reused
    cv::Mat flow_cpu_;                // reused (gets overwritten each frame)
    cv::Mat bg_mask_;
    cv::Rect subject_rect_;
    cv::cuda::GpuMat last_output_;    // shallow ref to whichever GpuMat is the
                                       // current frame's output (gpu_stabilized_
                                       // or gpu_final_).  For NVENC writer.
    bool ref_set_ = false;
    bool skip_download_ = false;
};


class RansacStabilizer {
public:
    struct Metrics {
        double  stabilization_ms = 0.0;
        double  aruco_detection_ms = 0.0;
        int     keypoints_found  = 0;
        int     good_matches     = 0;
        int     inliers          = 0;
        bool    homography_valid = false;
        bool    aruco_detected   = false;
        cv::Point2f aruco_center = cv::Point2f(0, 0);
        // Decomposed compensation applied to this frame
        double tx = 0.0;
        double ty = 0.0;
        double rotation_deg = 0.0;
        double scale = 1.0;
        // Stage 2 Farneback residual correction
        double farneback_dx = 0.0;
        double farneback_dy = 0.0;
    };

    RansacStabilizer()
    {
        // ── CUDA ORB ──────────────────────────────────────────────────────
        orb_gpu_ = cv::cuda::ORB::create(
            ORB_FEATURES, 1.2f, 12, 15, 0, 2,
            cv::ORB::HARRIS_SCORE, 31, 20, true);

        // ── CUDA BF Matcher ───────────────────────────────────────────────
        matcher_gpu_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

        // ── ArUco (stays on CPU — no CUDA version in OpenCV) ──────────────
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::DetectorParameters det_params;
        det_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        det_params.cornerRefinementWinSize = 5;
        det_params.cornerRefinementMaxIterations = 30;
        det_params.cornerRefinementMinAccuracy = 0.01;
        aruco_detector_ = cv::aruco::ArucoDetector(aruco_dict_, det_params);

        // Mask built on first frame (size not known yet)
        mask_built_ = false;
    }

    void setSubjectRegion(const cv::Rect& r) { subject_rect_ = r; }

    // Video mode pipes the stage-1 result through Farneback which always
    // downloads its own output — so the download inside stabilize() is
    // wasted work.  Set this once in video mode to keep stage-1 result on GPU.
    void setSkipDownload(bool v) { skip_download_ = v; }

    // NVDEC path: caller has already filled gpu_frame_ with BGR data on GPU,
    // so stabilize() must not re-upload from the cv::Mat argument.
    void setSkipUpload(bool v) { skip_upload_ = v; }
    cv::cuda::GpuMat& mutableGpuFrame() { return gpu_frame_; }

    void buildMask(int width, int height) {
        cv::Rect exclusion(
            width / 2 - ARUCO_SIZE / 2 - EXCLUSION_MARGIN,
            height / 2 - ARUCO_SIZE / 2 - EXCLUSION_MARGIN,
            ARUCO_SIZE + 2 * EXCLUSION_MARGIN,
            ARUCO_SIZE + 2 * EXCLUSION_MARGIN);
        cv::Mat mask_cpu = cv::Mat::ones(height, width, CV_8U) * 255;
        mask_cpu(exclusion & cv::Rect(0, 0, width, height)) = 0;
        if (subject_rect_.area() > 0) {
            // Halo: expand the rect by 10% on every side so its corners
            // don't graze rotor pixels.
            int hx = static_cast<int>(subject_rect_.width  * SUBJECT_HALO_FRAC);
            int hy = static_cast<int>(subject_rect_.height * SUBJECT_HALO_FRAC);
            cv::Rect s(subject_rect_.x - hx, subject_rect_.y - hy,
                       subject_rect_.width  + 2 * hx,
                       subject_rect_.height + 2 * hy);
            s &= cv::Rect(0, 0, width, height);
            if (s.area() > 0) mask_cpu(s) = 0;
        }
        mask_gpu_.upload(mask_cpu);
        mask_built_ = true;
    }

    cv::Mat stabilize(const cv::Mat& frame, Metrics& m) {
        auto t_start = cv::getTickCount();

        // Build mask on first frame (adapts to any resolution)
        if (!mask_built_)
            buildMask(frame.cols, frame.rows);

        // ── 1. Upload to GPU and convert to gray (reused buffers) ─────────
        if (!skip_upload_) gpu_frame_.upload(frame);
        cv::cuda::cvtColor(gpu_frame_, gpu_gray_, cv::COLOR_BGR2GRAY);

        // ── 2. CUDA ORB detection ─────────────────────────────────────────
        orb_gpu_->detectAndComputeAsync(gpu_gray_, mask_gpu_,
                                        gpu_kps_mat_, gpu_desc_, false, stream_);
        stream_.waitForCompletion();

        std::vector<cv::KeyPoint> kps;
        orb_gpu_->convert(gpu_kps_mat_, kps);
        m.keypoints_found = static_cast<int>(kps.size());

        if (!initialized_) {
            if (static_cast<int>(kps.size()) < MIN_MATCHES)
                return frame.clone();
            keyframe_kps_ = kps;
            keyframe_desc_gpu_ = gpu_desc_.clone();
            H_accumulated_ = cv::Mat::eye(3, 3, CV_64F);
            H_smooth_ = cv::Mat::eye(3, 3, CV_64F);
            gpu_stabilized_ = gpu_frame_.clone();
            frames_since_keyframe_ = 0;
            initialized_ = true;
            return frame.clone();
        }

        frames_since_keyframe_++;

        // ── 3. Match against keyframe ─────────────────────────────────────
        // H_to_kf: current frame → keyframe (high inliers, keyframe is recent)
        // H_total = H_accumulated_ * H_to_kf: current → frame 0
        cv::Mat H_to_kf;
        if (static_cast<int>(kps.size()) >= MIN_MATCHES) {
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher_gpu_->knnMatch(gpu_desc_, keyframe_desc_gpu_, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            for (const auto& pair : knn_matches) {
                if (pair.size() == 2 && pair[0].distance < LOWE_RATIO * pair[1].distance)
                    good_matches.push_back(pair[0]);
            }
            m.good_matches = static_cast<int>(good_matches.size());

            // Rank Lowe-passing matches by combined Harris response and keep
            // the top MATCH_QUALITY_KEEP fraction.  High-response keypoints
            // are more localizable → less sub-pixel jitter in the fit.
            if (m.good_matches > MIN_MATCHES) {
                std::sort(good_matches.begin(), good_matches.end(),
                          [&](const cv::DMatch& a, const cv::DMatch& b) {
                    float ra = kps[a.queryIdx].response +
                               keyframe_kps_[a.trainIdx].response;
                    float rb = kps[b.queryIdx].response +
                               keyframe_kps_[b.trainIdx].response;
                    return ra > rb;
                });
                int keep = std::max(MIN_MATCHES,
                                    static_cast<int>(good_matches.size() *
                                                     MATCH_QUALITY_KEEP));
                good_matches.resize(keep);
                m.good_matches = keep;
            }

            if (m.good_matches >= MIN_MATCHES) {
                std::vector<cv::Point2f> src_pts, dst_pts;
                for (const auto& dm : good_matches) {
                    src_pts.push_back(kps[dm.queryIdx].pt);
                    dst_pts.push_back(keyframe_kps_[dm.trainIdx].pt);
                }

                // Use affine (4-DOF: tx, ty, rotation, scale) instead of
                // full homography (8-DOF). The perspective terms in a full H
                // are pure noise for a flat scene, and they cause the bottom
                // of the frame to vibrate differently from the top.
                cv::Mat inlier_mask;
                cv::Mat A = cv::estimateAffinePartial2D(src_pts, dst_pts, inlier_mask,
                                                        cv::RANSAC, RANSAC_THRESH,
                                                        2000, HOMOGRAPHY_CONFIDENCE);
                if (!A.empty()) {
                    m.inliers = cv::countNonZero(inlier_mask);
                    double inlier_ratio = static_cast<double>(m.inliers) /
                                         static_cast<double>(m.good_matches);
                    m.homography_valid = m.inliers >= MIN_MATCHES &&
                                        inlier_ratio >= MIN_INLIER_RATIO;
                    if (m.homography_valid) {
                        // Convert 2x3 affine to 3x3 homography for warpPerspective
                        H_to_kf = cv::Mat::eye(3, 3, CV_64F);
                        H_to_kf.at<double>(0, 0) = A.at<double>(0, 0);
                        H_to_kf.at<double>(0, 1) = A.at<double>(0, 1);
                        H_to_kf.at<double>(0, 2) = A.at<double>(0, 2);
                        H_to_kf.at<double>(1, 0) = A.at<double>(1, 0);
                        H_to_kf.at<double>(1, 1) = A.at<double>(1, 1);
                        H_to_kf.at<double>(1, 2) = A.at<double>(1, 2);
                    }
                }
            }
        }

        // ── 4. Compute total H and refresh keyframe if needed ─────────────
        cv::Mat H_total;
        if (m.homography_valid) {
            H_total = H_accumulated_ * H_to_kf;
            last_valid_H_ = H_total.clone();

            // Refresh keyframe every KEYFRAME_INTERVAL frames
            if (frames_since_keyframe_ >= KEYFRAME_INTERVAL) {
                H_accumulated_ = H_total.clone();
                keyframe_kps_ = kps;
                keyframe_desc_gpu_ = gpu_desc_.clone();
                frames_since_keyframe_ = 0;
            }
        } else {
            H_total = last_valid_H_;
        }

        // ── 5. Decompose H ─────────────────────────────────────────────────
        if (!H_total.empty()) {
            m.tx           = H_total.at<double>(0, 2);
            m.ty           = H_total.at<double>(1, 2);
            m.scale        = std::sqrt(H_total.at<double>(0,0)*H_total.at<double>(0,0) +
                                       H_total.at<double>(1,0)*H_total.at<double>(1,0));
            double raw_rot = std::atan2(H_total.at<double>(1,0),
                                        H_total.at<double>(0,0)) * 180.0 / CV_PI;

            // Smooth rotation over ROT_SMOOTH_N frames (rotation changes slowly,
            // but RANSAC estimates it noisily — smoothing reduces edge vibration)
            rot_buffer_.push_back(raw_rot);
            if (static_cast<int>(rot_buffer_.size()) > ROT_SMOOTH_N)
                rot_buffer_.erase(rot_buffer_.begin());
            m.rotation_deg = std::accumulate(rot_buffer_.begin(), rot_buffer_.end(), 0.0)
                           / static_cast<double>(rot_buffer_.size());
        }

        // ── 6. Reconstruct smoothed H and apply warp ──────────────────────
        // H_smooth_ and gpu_stabilized_ are cached as members so the
        // Farneback stage can pick them up without re-uploading or
        // re-downloading anything.
        cv::Mat stabilized;
        if (!H_total.empty()) {
            double theta_rad = m.rotation_deg * CV_PI / 180.0;
            double cos_t = m.scale * std::cos(theta_rad);
            double sin_t = m.scale * std::sin(theta_rad);
            H_smooth_ = cv::Mat::eye(3, 3, CV_64F);
            H_smooth_.at<double>(0, 0) =  cos_t;
            H_smooth_.at<double>(0, 1) = -sin_t;
            H_smooth_.at<double>(0, 2) =  m.tx;
            H_smooth_.at<double>(1, 0) =  sin_t;
            H_smooth_.at<double>(1, 1) =  cos_t;
            H_smooth_.at<double>(1, 2) =  m.ty;

            cv::cuda::warpPerspective(gpu_frame_, gpu_stabilized_, H_smooth_,
                                      frame.size(), cv::INTER_CUBIC,
                                      cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            if (!skip_download_) gpu_stabilized_.download(stabilized);
        } else {
            H_smooth_ = cv::Mat::eye(3, 3, CV_64F);
            gpu_stabilized_ = gpu_frame_.clone();
            if (!skip_download_) stabilized = frame.clone();
        }

        m.stabilization_ms = tickMs(cv::getTickCount() - t_start);
        return stabilized;
    }

    // Accessors used by the Farneback refinement stage to avoid
    // re-uploading the original frame and stage-1 stabilized result.
    const cv::cuda::GpuMat& lastGpuFrame()       const { return gpu_frame_; }
    const cv::cuda::GpuMat& lastGpuStabilized()  const { return gpu_stabilized_; }
    const cv::Mat&          lastHSmooth()        const { return H_smooth_; }

    cv::Point2f detectArUcoCenter(const cv::Mat& frame, Metrics& m) {
        auto t_start = cv::getTickCount();
        
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        aruco_detector_.detectMarkers(frame, corners, ids);
        
        cv::Point2f center(0, 0);
        m.aruco_detected = false;
        
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == 0) {  // Target marker ID
                // Sub-pixel corner refinement for higher accuracy
                cv::Mat gray;
                if (frame.channels() == 3) {
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                } else {
                    gray = frame;
                }
                
                // Refine corners to sub-pixel accuracy
                cv::cornerSubPix(gray, corners[i], cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
                
                // Calculate center from refined corners
                for (const auto& corner : corners[i]) {
                    center += corner;
                }
                center /= 4.0f;
                m.aruco_detected = true;
                m.aruco_center = center;
                break;
            }
        }
        
        m.aruco_detection_ms = tickMs(cv::getTickCount() - t_start);
        return center;
    }

    void reset() {
        initialized_ = false;
        keyframe_kps_.clear();
        keyframe_desc_gpu_ = cv::cuda::GpuMat();
        last_valid_H_ = cv::Mat();
    }

private:
    static double tickMs(int64 ticks) {
        return static_cast<double>(ticks) / cv::getTickFrequency() * 1000.0;
    }

    cv::Ptr<cv::cuda::ORB> orb_gpu_;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_gpu_;
    cv::aruco::Dictionary aruco_dict_;
    cv::aruco::ArucoDetector aruco_detector_;
    cv::cuda::GpuMat mask_gpu_;
    cv::Rect subject_rect_;
    cv::cuda::Stream stream_;

    // Reused per-frame GPU buffers (preallocate → no per-frame cudaMalloc)
    cv::cuda::GpuMat gpu_frame_;
    cv::cuda::GpuMat gpu_gray_;
    cv::cuda::GpuMat gpu_kps_mat_;
    cv::cuda::GpuMat gpu_desc_;
    cv::cuda::GpuMat gpu_stabilized_;  // stage-1 warp result (BGR)
    cv::Mat H_smooth_;                 // stage-1 transform (3x3)

    bool initialized_ = false;
    bool mask_built_ = false;
    bool skip_download_ = false;
    bool skip_upload_ = false;
    int frames_since_keyframe_ = 0;
    std::vector<cv::KeyPoint> keyframe_kps_;
    cv::cuda::GpuMat keyframe_desc_gpu_;
    cv::Mat H_accumulated_;       // keyframe → frame 0
    cv::Mat last_valid_H_;        // last good total H (fallback)
    std::vector<double> rot_buffer_;  // rotation smoothing buffer
};


// ─────────────────────────────────────────────────────────────────────────────

json loadGroundTruth(const std::string& input_dir) {
    std::ifstream file(input_dir + "/ground_truth.json");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ground_truth.json");
    }
    json gt;
    file >> gt;
    return gt;
}

void saveDisplacementData(const std::vector<DisplacementData>& data, const std::string& output_file) {
    json output;
    output["displacement_analysis"] = json::array();
    
    for (const auto& d : data) {
        json entry;
        entry["frame_id"] = d.frame_id;
        entry["measured_displacement"] = {d.measured_displacement.x, d.measured_displacement.y};
        entry["ground_truth_displacement"] = {d.ground_truth_displacement.x, d.ground_truth_displacement.y};
        entry["error_magnitude"] = d.error_magnitude;
        output["displacement_analysis"].push_back(entry);
    }
    
    std::ofstream file(output_file);
    file << output.dump(2);
}

void processDataset(const std::string& input_dir, const std::string& output_dir) {
    RansacStabilizer stabilizer;
    fs::create_directories(output_dir + "/frames");

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,tx_px,ty_px,rotation_deg,scale,"
           "keypoints_found,good_matches,inliers,homography_valid,"
           "gt_sway_x,gt_sway_y,gt_rotation_deg\n";

    // Load ground truth data
    json ground_truth;
    try {
        ground_truth = loadGroundTruth(input_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: " << e.what() << "\n";
    }

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(input_dir + "/frames")) {
        if (e.path().extension() == ".png")
            files.push_back(e.path().filename().string());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No PNG frames found in " << input_dir << "/frames\n";
        return;
    }

    std::cout << "RANSAC Stabilizer with ArUco Displacement Tracking\n";
    std::cout << "Processing " << files.size() << " frames...\n\n";

    std::vector<DisplacementData> displacement_data;
    cv::Point2f reference_center(0, 0);
    bool reference_set = false;

    for (size_t i = 0; i < files.size(); ++i) {
        if (i % 50 == 0) {
            std::cout << "Frame " << i << "/" << files.size() << "\n";
        }

        cv::Mat frame = cv::imread(input_dir + "/frames/" + files[i]);
        if (frame.empty()) continue;

        // Stabilize frame
        RansacStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m);
        cv::imwrite(output_dir + "/frames/" + files[i], stabilized);

        // Detect ArUco in stabilized frame
        cv::Point2f current_center = stabilizer.detectArUcoCenter(stabilized, m);

        // Ground truth camera sway for this frame (CSV)
        cv::Point2f gt_sway(0, 0);
        if (!ground_truth.empty() && ground_truth.contains("camera_sway")) {
            auto& cs = ground_truth["camera_sway"];
            if (cs.contains("displacement_x") && cs.contains("displacement_y")) {
                auto& sx = cs["displacement_x"];
                auto& sy = cs["displacement_y"];
                if (i < sx.size() && i < sy.size()) {
                    gt_sway.x = sx[i];
                    gt_sway.y = sy[i];
                }
            }
        }

        // Ground truth rotation for this frame (CSV)
        double gt_rot_deg = 0.0;
        if (!ground_truth.empty() && ground_truth.contains("rotation")) {
            auto& ra = ground_truth["rotation"]["angles"];
            if (i < ra.size()) {
                gt_rot_deg = ra[i];
            }
        }

        // Ground truth structural vibration (displacement_analysis.json only)
        cv::Point2f gt_disp(0, 0);
        if (!ground_truth.empty() && ground_truth.contains("structural_vibration")) {
            auto& sv = ground_truth["structural_vibration"];
            if (sv.contains("displacement_x") && sv.contains("displacement_y")) {
                auto& dx = sv["displacement_x"];
                auto& dy = sv["displacement_y"];
                if (i < dx.size() && i < dy.size()) {
                    gt_disp.x = dx[i];
                    gt_disp.y = dy[i];
                }
            }
        }

        cv::Point2f measured_disp(0, 0);
        double error_px = 0.0;
        if (m.aruco_detected && reference_set) {
            measured_disp = current_center - reference_center;
            error_px = cv::norm(measured_disp - gt_disp);
        }

        csv << i << ","
            << m.tx << "," << m.ty << ","
            << m.rotation_deg << "," << m.scale << ","
            << m.keypoints_found << "," << m.good_matches << "," << m.inliers << ","
            << m.homography_valid << ","
            << gt_sway.x << "," << gt_sway.y << ","
            << gt_rot_deg << "\n";
        
        if (m.aruco_detected) {
            if (!reference_set) {
                reference_center = current_center;
                reference_set = true;
            } else {
                displacement_data.push_back({static_cast<int>(i), measured_disp, gt_disp, error_px});
            }
        }
    }

    // Save displacement analysis
    if (!displacement_data.empty()) {
        saveDisplacementData(displacement_data, output_dir + "/displacement_analysis.json");
        
        // Calculate statistics
        double mean_error = 0.0;
        for (const auto& d : displacement_data) {
            mean_error += d.error_magnitude;
        }
        mean_error /= displacement_data.size();
        
        std::cout << "\n=== Displacement Analysis ===\n";
        std::cout << "Frames with ArUco detected: " << displacement_data.size() << "/" << files.size() << "\n";
        std::cout << "Mean displacement error: " << mean_error << " px\n";
        std::cout << "Results saved to: " << output_dir << "/displacement_analysis.json\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Video input mode
// ─────────────────────────────────────────────────────────────────────────────

void processVideo(const std::string& video_path, const std::string& output_dir,
                  int max_frames = 0, cv::Rect subject_rect = cv::Rect()) {
    RansacStabilizer stabilizer;
    stabilizer.setSkipDownload(true);
    FarnebackRefinement farneback;
    if (subject_rect.area() > 0) {
        stabilizer.setSubjectRegion(subject_rect);
        farneback.setSubjectRegion(subject_rect);
        std::cout << "Subject mask: x=" << subject_rect.x << " y=" << subject_rect.y
                  << " w=" << subject_rect.width << " h=" << subject_rect.height << "\n";
    }
    fs::create_directories(output_dir);

    // Pull metadata via cv::VideoCapture (cheap; header-only).
    cv::VideoCapture meta_cap(video_path);
    if (!meta_cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << "\n";
        return;
    }
    int total_frames = static_cast<int>(meta_cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps       = meta_cap.get(cv::CAP_PROP_FPS);
    int width        = static_cast<int>(meta_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height       = static_cast<int>(meta_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    meta_cap.release();

    // Try NVDEC first.  Falls back to CPU decode for codecs NVDEC can't
    // handle (e.g. raw uncompressed AVI).
    cv::Ptr<cv::cudacodec::VideoReader> nvdec_reader;
    bool use_nvdec = true;
    try {
        nvdec_reader = cv::cudacodec::createVideoReader(video_path);
    } catch (const cv::Exception&) {
        use_nvdec = false;
    }
    cv::VideoCapture cap;
    if (!use_nvdec) {
        cap.open(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video for CPU decode: " << video_path << "\n";
            return;
        }
    }

    std::cout << "RANSAC Stabilizer — Video Mode " << (use_nvdec ? "(NVDEC)" : "(CPU)") << "\n";
    std::cout << "Input  : " << video_path << "\n";
    std::cout << "Frames : " << total_frames << " @ " << fps << " fps, "
              << width << "x" << height << "\n\n";

    if (use_nvdec) stabilizer.setSkipUpload(true);

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,tx_px,ty_px,rotation_deg,scale,"
           "keypoints_found,good_matches,inliers,homography_valid,"
           "farneback_dx,farneback_dy,"
           "aruco_detected,aruco_center_x,aruco_center_y\n";

    // Try NVENC writer first (H.264-in-AVI); fall back to CPU MJPEG.
    cv::Ptr<cv::cudacodec::VideoWriter> nvenc_writer;
    cv::VideoWriter cpu_writer;
    bool use_nvenc = true;
    try {
        nvenc_writer = cv::cudacodec::createVideoWriter(
            output_dir + "/stabilized.avi",
            cv::Size(width, height),
            cv::cudacodec::Codec::H264,
            fps > 0 ? fps : 1000.0,
            cv::cudacodec::ColorFormat::BGR);
    } catch (const cv::Exception&) {
        use_nvenc = false;
    }
    if (!use_nvenc) {
        cpu_writer.open(output_dir + "/stabilized.avi",
                        cv::VideoWriter::fourcc('M','J','P','G'),
                        fps, cv::Size(width, height));
        if (!cpu_writer.isOpened()) {
            std::cerr << "Warning: CPU VideoWriter failed too — no video output.\n";
        }
    }
    std::cout << "Encoder: " << (use_nvenc ? "NVENC (H.264)" : "CPU (MJPEG)") << "\n";

    int frame_idx = 0;
    int limit = (max_frames > 0) ? max_frames : total_frames;

    // ArUco auto-disable: probe first ARUCO_PROBE_FRAMES frames; if no marker
    // ever shows up, skip the cornerSubPix detector for the rest of the clip.
    bool aruco_active = true;
    int  aruco_hits   = 0;

    // CPU-decode path: pipeline reads in a worker thread so frame N+1's
    // decode overlaps with frame N's GPU compute.
    auto read_next = [&]() -> std::pair<bool, cv::Mat> {
        cv::Mat f;
        bool ok = cap.read(f);
        return {ok, std::move(f)};
    };
    std::future<std::pair<bool, cv::Mat>> prefetch;
    if (!use_nvdec) prefetch = std::async(std::launch::async, read_next);

    // NVDEC path: dummy host header (no allocation needed for stabilize's
    // size queries since skip_upload bypasses the data access).
    cv::Mat nvdec_dummy(height, width, CV_8UC3);
    cv::cuda::GpuMat gpu_bgra;

    while (frame_idx < limit) {
        cv::Mat frame;
        bool has_frame = false;

        if (use_nvdec) {
            if (nvdec_reader->nextFrame(gpu_bgra)) {
                cv::cuda::cvtColor(gpu_bgra, stabilizer.mutableGpuFrame(),
                                   cv::COLOR_BGRA2BGR);
                frame = nvdec_dummy;   // header only — stabilize uses size, not data
                has_frame = true;
            }
        } else {
            auto pair = prefetch.get();
            has_frame = pair.first;
            if (has_frame) {
                frame = std::move(pair.second);
                if (frame_idx + 1 < limit)
                    prefetch = std::async(std::launch::async, read_next);
            }
        }
        if (!has_frame) break;

        if (frame_idx % 100 == 0)
            std::cout << "Frame " << frame_idx << "/" << limit << "\n";

        RansacStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m);

        // We can skip the final cv::Mat download whenever both:
        //   1. The writer is NVENC (consumes GpuMat directly), and
        //   2. ArUco is no longer probing (CPU detector needs cv::Mat).
        // While ArUco is probing the first 100 frames, keep downloading.
        farneback.setSkipDownload(use_nvenc && !aruco_active);

        // Stage 2: Farneback residual correction — GPU-native path.
        // Reuses gpu_frame and gpu_stabilized from stage 1 (no re-upload)
        // and composes a single final warp instead of cascading two.
        stabilized = farneback.refineComposed(
            stabilizer.lastGpuFrame(),
            stabilizer.lastGpuStabilized(),
            stabilizer.lastHSmooth(),
            m.farneback_dx, m.farneback_dy);

        if (use_nvenc) {
            nvenc_writer->write(farneback.lastGpuOutput());
        } else if (cpu_writer.isOpened()) {
            cpu_writer.write(stabilized);
        }

        if (aruco_active) {
            stabilizer.detectArUcoCenter(stabilized, m);
            if (m.aruco_detected) aruco_hits++;
            if (frame_idx + 1 >= ARUCO_PROBE_FRAMES && aruco_hits == 0) {
                aruco_active = false;
                std::cout << "  ArUco: no marker in first " << ARUCO_PROBE_FRAMES
                          << " frames — disabling detector\n";
            }
        }

        csv << frame_idx << ","
            << m.tx << "," << m.ty << ","
            << m.rotation_deg << "," << m.scale << ","
            << m.keypoints_found << "," << m.good_matches << "," << m.inliers << ","
            << m.homography_valid << ","
            << m.farneback_dx << "," << m.farneback_dy << ","
            << m.aruco_detected << ","
            << m.aruco_center.x << "," << m.aruco_center.y << "\n";

        frame_idx++;
    }

    if (use_nvdec) nvdec_reader.release();
    else           cap.release();
    if (use_nvenc) nvenc_writer.release();
    else           cpu_writer.release();
    std::cout << "\nProcessed " << frame_idx << " frames\n";
    std::cout << "Stabilized video → " << output_dir << "/stabilized.avi\n";
}


static bool isVideoFile(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input> <output_dir> [--frames N] [--subject-mask X,Y,W,H]\n";
        std::cerr << "  input            : video file (.mp4/.avi/.mov) OR directory with frames/\n";
        std::cerr << "  --frames N       : process only first N frames (default: all)\n";
        std::cerr << "  --subject-mask   : exclude this rect from stabilizer feature sampling\n";
        std::cerr << "                     (use for non-ArUco videos where the vibrating subject\n";
        std::cerr << "                      would otherwise leak into the background transform)\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output_dir = argv[2];

    int max_frames = 0;
    cv::Rect subject_rect;
    for (int i = 3; i < argc - 1; i++) {
        std::string flag = argv[i];
        if (flag == "--frames") {
            max_frames = std::stoi(argv[i + 1]);
        } else if (flag == "--subject-mask") {
            int x, y, w, h;
            if (std::sscanf(argv[i + 1], "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
                subject_rect = cv::Rect(x, y, w, h);
            } else {
                std::cerr << "Bad --subject-mask value (expected X,Y,W,H): " << argv[i + 1] << "\n";
                return 1;
            }
        }
    }

    try {
        if (isVideoFile(input)) {
            if (!fs::exists(input)) {
                std::cerr << "Video not found: " << input << "\n";
                return 1;
            }
            processVideo(input, output_dir, max_frames, subject_rect);
        } else {
            if (!fs::exists(input + "/frames")) {
                std::cerr << "Frames directory not found: " << input << "/frames\n";
                return 1;
            }
            processDataset(input, output_dir);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}