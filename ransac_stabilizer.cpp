#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
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
static constexpr int   ORB_FEATURES   = 5000;   // More features for stability
static constexpr float LOWE_RATIO     = 0.65f;  // Tighter ratio test (was 0.7)
static constexpr double RANSAC_THRESH = 3.0;    // Wider threshold for larger displacement
static constexpr int   MIN_MATCHES    = 25;     // More matches required
static constexpr double HOMOGRAPHY_CONFIDENCE = 0.995;  // Higher confidence
// Reference refresh: update reference when inlier ratio drops below this
// to prevent match degradation as the scene drifts from frame 0.
static constexpr double MIN_INLIER_RATIO = 0.10;
static constexpr int    KEYFRAME_INTERVAL = 100;  // refresh keyframe every N frames // 10% of good matches must be inliers
static constexpr int    REFRESH_INLIER_THRESHOLD = 60; // refresh if inliers < this


struct DisplacementData {
    int frame_id;
    cv::Point2f measured_displacement;
    cv::Point2f ground_truth_displacement;
    double error_magnitude;
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

    void buildMask(int width, int height) {
        cv::Rect exclusion(
            width / 2 - ARUCO_SIZE / 2 - EXCLUSION_MARGIN,
            height / 2 - ARUCO_SIZE / 2 - EXCLUSION_MARGIN,
            ARUCO_SIZE + 2 * EXCLUSION_MARGIN,
            ARUCO_SIZE + 2 * EXCLUSION_MARGIN);
        cv::Mat mask_cpu = cv::Mat::ones(height, width, CV_8U) * 255;
        mask_cpu(exclusion) = 0;
        mask_gpu_.upload(mask_cpu);
        mask_built_ = true;
    }

    cv::Mat stabilize(const cv::Mat& frame, Metrics& m) {
        auto t_start = cv::getTickCount();

        // Build mask on first frame (adapts to any resolution)
        if (!mask_built_)
            buildMask(frame.cols, frame.rows);

        // ── 1. Upload to GPU and convert to gray ──────────────────────────
        cv::cuda::GpuMat gpu_frame, gpu_gray;
        gpu_frame.upload(frame);
        cv::cuda::cvtColor(gpu_frame, gpu_gray, cv::COLOR_BGR2GRAY);

        // ── 2. CUDA ORB detection ─────────────────────────────────────────
        cv::cuda::GpuMat gpu_kps_mat, gpu_desc;
        orb_gpu_->detectAndComputeAsync(gpu_gray, mask_gpu_,
                                        gpu_kps_mat, gpu_desc, false, stream_);
        stream_.waitForCompletion();

        std::vector<cv::KeyPoint> kps;
        orb_gpu_->convert(gpu_kps_mat, kps);
        m.keypoints_found = static_cast<int>(kps.size());

        if (!initialized_) {
            if (static_cast<int>(kps.size()) < MIN_MATCHES)
                return frame.clone();
            keyframe_kps_ = kps;
            keyframe_desc_gpu_ = gpu_desc.clone();
            H_accumulated_ = cv::Mat::eye(3, 3, CV_64F);
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
            matcher_gpu_->knnMatch(gpu_desc, keyframe_desc_gpu_, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            for (const auto& pair : knn_matches) {
                if (pair.size() == 2 && pair[0].distance < LOWE_RATIO * pair[1].distance)
                    good_matches.push_back(pair[0]);
            }
            m.good_matches = static_cast<int>(good_matches.size());

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
                keyframe_desc_gpu_ = gpu_desc.clone();
                frames_since_keyframe_ = 0;
            }
        } else {
            H_total = last_valid_H_;
        }

        // ── 5. Decompose and apply H ──────────────────────────────────────
        if (!H_total.empty()) {
            m.tx           = H_total.at<double>(0, 2);
            m.ty           = H_total.at<double>(1, 2);
            m.scale        = std::sqrt(H_total.at<double>(0,0)*H_total.at<double>(0,0) +
                                       H_total.at<double>(1,0)*H_total.at<double>(1,0));
            m.rotation_deg = std::atan2(H_total.at<double>(1,0),
                                        H_total.at<double>(0,0)) * 180.0 / CV_PI;
        }

        // ── 6. RANSAC warp ─────────────────────────────────────────────────
        cv::Mat stabilized;
        cv::cuda::GpuMat gpu_stabilized;
        if (!H_total.empty()) {
            cv::cuda::warpPerspective(gpu_frame, gpu_stabilized, H_total, frame.size(),
                                      cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
            gpu_stabilized.download(stabilized);
        } else {
            stabilized = frame.clone();
        }

        // ── 7. ArUco feedback correction ──────────────────────────────────
        // Detect centre marker (ID 0) in RANSAC-stabilized frame.
        // If the marker has drifted from the reference position, apply a
        // small translational correction to push it back.
        // This closes the loop: RANSAC handles large motion (~50px),
        // ArUco feedback corrects the remaining sub-pixel residual.
        {
            Metrics aruco_m;
            cv::Point2f center = detectArUcoCenter(stabilized, aruco_m);
            m.aruco_detected = aruco_m.aruco_detected;
            m.aruco_center   = aruco_m.aruco_center;

            if (aruco_m.aruco_detected) {
                if (!aruco_ref_set_) {
                    aruco_ref_center_ = center;
                    aruco_ref_set_ = true;
                } else {
                    // Residual = how far the marker drifted from reference
                    double residual_x = center.x - aruco_ref_center_.x;
                    double residual_y = center.y - aruco_ref_center_.y;

                    // Only correct if residual is meaningful (> noise floor)
                    if (std::abs(residual_x) > 0.1 || std::abs(residual_y) > 0.1) {
                        // Translational correction to cancel drift
                        cv::Mat H_correction = cv::Mat::eye(3, 3, CV_64F);
                        H_correction.at<double>(0, 2) = -residual_x;
                        H_correction.at<double>(1, 2) = -residual_y;

                        cv::cuda::GpuMat gpu_corrected;
                        gpu_stabilized.upload(stabilized);
                        cv::cuda::warpPerspective(gpu_stabilized, gpu_corrected,
                                                  H_correction, frame.size(),
                                                  cv::INTER_CUBIC, cv::BORDER_CONSTANT,
                                                  cv::Scalar(0,0,0));
                        gpu_corrected.download(stabilized);

                        // Update metrics to reflect final corrected position
                        m.tx += -residual_x;
                        m.ty += -residual_y;
                        m.aruco_center.x -= residual_x;
                        m.aruco_center.y -= residual_y;
                    }
                }
            }
        }

        m.stabilization_ms = tickMs(cv::getTickCount() - t_start);
        return stabilized;
    }

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
    cv::cuda::Stream stream_;

    bool initialized_ = false;
    bool mask_built_ = false;
    int frames_since_keyframe_ = 0;
    std::vector<cv::KeyPoint> keyframe_kps_;
    cv::cuda::GpuMat keyframe_desc_gpu_;
    cv::Mat H_accumulated_;       // keyframe → frame 0
    cv::Mat last_valid_H_;        // last good total H (fallback)
    // ArUco feedback loop state
    cv::Point2f aruco_ref_center_;
    bool aruco_ref_set_ = false;
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

void processVideo(const std::string& video_path, const std::string& output_dir, int max_frames = 0) {
    RansacStabilizer stabilizer;
    fs::create_directories(output_dir);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << "\n";
        return;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps       = cap.get(cv::CAP_PROP_FPS);
    int width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "RANSAC Stabilizer — Video Mode\n";
    std::cout << "Input  : " << video_path << "\n";
    std::cout << "Frames : " << total_frames << " @ " << fps << " fps, "
              << width << "x" << height << "\n\n";

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,tx_px,ty_px,rotation_deg,scale,"
           "keypoints_found,good_matches,inliers,homography_valid,"
           "aruco_detected,aruco_center_x,aruco_center_y\n";

    cv::VideoWriter writer(
        output_dir + "/stabilized.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Warning: VideoWriter failed, trying raw AVI...\n";
        writer.open(output_dir + "/stabilized.avi", 0, 30.0, cv::Size(width, height));
    }
    if (!writer.isOpened()) {
        std::cerr << "Warning: Could not open VideoWriter. Frames will be saved as PNGs only.\n";
    }

    int frame_idx = 0;
    cv::Mat frame;

    int limit = (max_frames > 0) ? max_frames : total_frames;
    while (cap.read(frame) && frame_idx < limit) {
        if (frame_idx % 100 == 0)
            std::cout << "Frame " << frame_idx << "/" << limit << "\n";

        RansacStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m);
        if (writer.isOpened()) writer.write(stabilized);

        cv::Point2f center = stabilizer.detectArUcoCenter(stabilized, m);

        csv << frame_idx << ","
            << m.tx << "," << m.ty << ","
            << m.rotation_deg << "," << m.scale << ","
            << m.keypoints_found << "," << m.good_matches << "," << m.inliers << ","
            << m.homography_valid << ","
            << m.aruco_detected << ","
            << m.aruco_center.x << "," << m.aruco_center.y << "\n";

        frame_idx++;
    }

    cap.release();
    writer.release();
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
        std::cerr << "Usage: " << argv[0] << " <input> <output_dir> [--frames N]\n";
        std::cerr << "  input     : video file (.mp4/.avi/.mov) OR directory with frames/\n";
        std::cerr << "  --frames N: process only first N frames (default: all)\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output_dir = argv[2];

    int max_frames = 0;
    for (int i = 3; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--frames")
            max_frames = std::stoi(argv[i + 1]);
    }

    try {
        if (isVideoFile(input)) {
            if (!fs::exists(input)) {
                std::cerr << "Video not found: " << input << "\n";
                return 1;
            }
            processVideo(input, output_dir, max_frames);
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