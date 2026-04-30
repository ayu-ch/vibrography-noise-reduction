// ─────────────────────────────────────────────────────────────────────────────
//  RL Stabilizer — Phase Correlation + Learned Policy
//
//  Replaces the entire ORB+RANSAC+Farneback pipeline with:
//    1. Phase correlation for base translation (sub-pixel, ~0.5ms on GPU)
//    2. ONNX RL policy for delta refinement (rotation + correction, ~0.1ms)
//    3. GPU warpPerspective for frame stabilization (~0.5ms)
//
//  Total: ~1.1ms per frame → capable of 900fps real-time
//
//  Usage:
//    ./rl_stabilizer <input> <output_dir> --model sac_stabilizer.onnx [--frames N]
//    input : video file (.mp4/.avi/.mov) OR directory with frames/
// ─────────────────────────────────────────────────────────────────────────────

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;


// ── Parameters ───────────────────────────────────────────────────────────────
static constexpr int ARUCO_SIZE = 150;
static constexpr int EXCLUSION_MARGIN = 40;

// RL policy delta bounds (must match training env)
static constexpr double DELTA_TX_MAX    = 2.0;    // px at real resolution
static constexpr double DELTA_TY_MAX    = 2.0;
static constexpr double DELTA_THETA_MAX = 0.1;    // degrees
static constexpr double DELTA_SCALE_MAX = 0.002;

// Observation normalization (must match training env)
static constexpr double TX_NORM = 50.0;
static constexpr double TY_NORM = 20.0;

// Frame difference observation size (must match training)
static constexpr int DIFF_W = 60;
static constexpr int DIFF_H = 45;
static constexpr int DIFF_SIZE = DIFF_W * DIFF_H;

// Total observation size: phase(2) + patch_stats(4) + prev_delta(4) + time(1) + diff(2700)
static constexpr int OBS_DIM = 2 + 4 + 4 + 1 + DIFF_SIZE;


// ─────────────────────────────────────────────────────────────────────────────
//  RL Policy — ONNX Runtime inference
// ─────────────────────────────────────────────────────────────────────────────

class RLPolicy {
public:
    RLPolicy(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "RLStabilizer")
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), opts);
        std::cout << "RL policy loaded: " << model_path << "\n";
    }

    // Returns [delta_tx, delta_ty, delta_theta, delta_scale] in [-1, 1]
    std::array<float, 4> predict(const std::vector<float>& obs) {
        auto mem = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        std::array<int64_t, 2> shape = {1, OBS_DIM};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            mem, const_cast<float*>(obs.data()), obs.size(),
            shape.data(), shape.size());

        const char* input_names[]  = {"observation"};
        const char* output_names[] = {"action"};

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names, &input, 1,
            output_names, 1);

        const float* out = outputs[0].GetTensorData<float>();
        return {out[0], out[1], out[2], out[3]};
    }

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
};


// ─────────────────────────────────────────────────────────────────────────────
//  GPU Phase Correlator — replaces cv::phaseCorrelate (CPU FFT) with cuFFT-
//  backed cv::cuda::dft + cross-power spectrum + IFFT + peak find.
//
//  All buffers are preallocated on first frame; per-frame cost is dominated
//  by the two DFTs (~0.5 ms total at 800×1080) plus a tiny H2D peak download.
// ─────────────────────────────────────────────────────────────────────────────
class GpuPhaseCorrelator {
public:
    void initialize(int W, int H) {
        W_ = W;
        H_ = H;
        cv::Mat hann_cpu;
        cv::createHanningWindow(hann_cpu, cv::Size(W, H), CV_32F);
        hann_gpu_.upload(hann_cpu);
    }

    // ref must be CV_8UC1 on GPU
    void setReferenceGpu(const cv::cuda::GpuMat& ref_gray) {
        applyWindow(ref_gray, ref_w_);
        cv::cuda::dft(ref_w_, F_ref_, cv::Size(W_, H_));
        ref_set_ = true;
    }

    // Returns sub-pixel translation (cur → ref convention, matches
    // cv::phaseCorrelate sign).  cur must be CV_8UC1 on GPU.
    cv::Point2d compute(const cv::cuda::GpuMat& cur_gray) {
        if (!ref_set_) return cv::Point2d(0, 0);

        applyWindow(cur_gray, cur_w_);
        cv::cuda::dft(cur_w_, F_cur_, cv::Size(W_, H_));

        // Cross-power spectrum: F_ref * conj(F_cur)
        cv::cuda::mulSpectrums(F_ref_, F_cur_, C_, 0, true);

        // Normalize so |C(u,v)| = 1 (the "phase" part of phase correlation).
        // Split into Re/Im channels, divide each by magnitude.
        std::vector<cv::cuda::GpuMat> ch;
        cv::cuda::split(C_, ch);
        cv::cuda::magnitude(ch[0], ch[1], mag_);
        cv::cuda::add(mag_, cv::Scalar(1e-10f), mag_);
        cv::cuda::divide(ch[0], mag_, ch[0]);
        cv::cuda::divide(ch[1], mag_, ch[1]);
        cv::cuda::merge(ch, C_);

        // Inverse DFT → real correlation surface
        cv::cuda::dft(C_, corr_, cv::Size(W_, H_),
                      cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        // Find integer peak on GPU
        cv::Point max_loc;
        double max_val;
        cv::cuda::minMaxLoc(corr_, nullptr, &max_val, nullptr, &max_loc);

        // Sub-pixel parabolic refinement: fetch the 3×3 neighbourhood of
        // the peak in one tiny D2H copy, then interpolate on CPU.
        double dx = (max_loc.x > W_ / 2) ? max_loc.x - W_ : max_loc.x;
        double dy = (max_loc.y > H_ / 2) ? max_loc.y - H_ : max_loc.y;

        if (max_loc.x > 0 && max_loc.x < W_ - 1 &&
            max_loc.y > 0 && max_loc.y < H_ - 1) {
            cv::Mat patch;
            corr_(cv::Rect(max_loc.x - 1, max_loc.y - 1, 3, 3)).download(patch);
            float c  = patch.at<float>(1, 1);
            float l  = patch.at<float>(1, 0);
            float r  = patch.at<float>(1, 2);
            float t  = patch.at<float>(0, 1);
            float b  = patch.at<float>(2, 1);
            float dxn = 2.0f * (2.0f * c - l - r);
            float dyn = 2.0f * (2.0f * c - t - b);
            if (std::fabs(dxn) > 1e-10f) dx += (l - r) / dxn;
            if (std::fabs(dyn) > 1e-10f) dy += (t - b) / dyn;
        }

        return cv::Point2d(dx, dy);
    }

private:
    void applyWindow(const cv::cuda::GpuMat& gray_8u, cv::cuda::GpuMat& dst) {
        gray_8u.convertTo(gray_f_, CV_32F, 1.0 / 255.0);
        cv::cuda::multiply(gray_f_, hann_gpu_, dst);
    }

    int W_ = 0, H_ = 0;
    cv::cuda::GpuMat hann_gpu_;
    cv::cuda::GpuMat gray_f_, ref_w_, cur_w_;
    cv::cuda::GpuMat F_ref_, F_cur_, C_, corr_, mag_;
    bool ref_set_ = false;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Phase Correlation Stabilizer
// ─────────────────────────────────────────────────────────────────────────────

class PhaseCorrelationStabilizer {
public:
    struct Metrics {
        double stabilization_ms = 0.0;
        double aruco_detection_ms = 0.0;
        // Phase correlation base
        double phase_dx = 0.0;
        double phase_dy = 0.0;
        // RL delta
        double delta_tx = 0.0;
        double delta_ty = 0.0;
        double delta_theta = 0.0;
        double delta_scale = 0.0;
        // Final correction applied
        double tx = 0.0;
        double ty = 0.0;
        double rotation_deg = 0.0;
        double scale = 1.0;
        // ArUco tracking
        bool aruco_detected = false;
        cv::Point2f aruco_center = cv::Point2f(0, 0);
    };

    PhaseCorrelationStabilizer() {
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::DetectorParameters det_params;
        det_params.adaptiveThreshWinSizeMin = 3;
        det_params.adaptiveThreshWinSizeMax = 53;
        det_params.adaptiveThreshWinSizeStep = 4;
        det_params.adaptiveThreshConstant = 7;
        det_params.minMarkerPerimeterRate = 0.01;
        det_params.maxMarkerPerimeterRate = 4.0;
        det_params.polygonalApproxAccuracyRate = 0.05;
        det_params.minCornerDistanceRate = 0.02;
        det_params.minDistanceToBorder = 1;
        det_params.minOtsuStdDev = 3.0;
        det_params.perspectiveRemovePixelPerCell = 8;
        det_params.perspectiveRemoveIgnoredMarginPerCell = 0.2;
        det_params.maxErroneousBitsInBorderRate = 0.5;
        det_params.errorCorrectionRate = 0.8;
        det_params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        det_params.cornerRefinementWinSize = 5;
        det_params.cornerRefinementMaxIterations = 50;
        det_params.cornerRefinementMinAccuracy = 0.01;
        aruco_detector_ = cv::aruco::ArucoDetector(aruco_dict_, det_params);
    }

    cv::Mat stabilize(const cv::Mat& frame, Metrics& m, RLPolicy* policy) {
        auto t_start = cv::getTickCount();

        // Upload + GPU cvtColor (preallocated buffers — no per-frame alloc)
        gpu_frame_.upload(frame);
        cv::cuda::cvtColor(gpu_frame_, gpu_gray_, cv::COLOR_BGR2GRAY);

        // ── 1. Initialize reference on first frame ────────────────────────
        if (!initialized_) {
            corr_.initialize(frame.cols, frame.rows);
            corr_.setReferenceGpu(gpu_gray_);
            // GPU + CPU copies of the reference for the policy's observation.
            gpu_ref_gray_ = gpu_gray_.clone();
            gpu_gray_.download(ref_gray_);
            initialized_ = true;
            frame_count_ = 0;
            return frame.clone();
        }

        frame_count_++;

        // ── 2. Phase correlation on GPU (cuFFT) ──────────────────────────
        cv::Point2d shift = corr_.compute(gpu_gray_);
        m.phase_dx = shift.x;
        m.phase_dy = shift.y;

        // ── 3. Build observation for RL policy ────────────────────────────
        double act_tx, act_ty, act_theta, act_scale;

        if (policy) {
            // Frame difference (translation-removed) on GPU.
            cv::cuda::GpuMat shifted_gpu;
            if (std::abs(shift.x) > 0.01 || std::abs(shift.y) > 0.01) {
                cv::Mat M_shift = (cv::Mat_<float>(2, 3) <<
                    1, 0, static_cast<float>(-shift.x),
                    0, 1, static_cast<float>(-shift.y));
                cv::cuda::warpAffine(gpu_gray_, shifted_gpu, M_shift, gpu_gray_.size(),
                                     cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
            } else {
                shifted_gpu = gpu_gray_;
            }

            // Absdiff on GPU; only download the small DIFF_W×DIFF_H tile we need.
            cv::cuda::GpuMat diff_gpu, diff_small_gpu;
            cv::cuda::absdiff(gpu_ref_gray_, shifted_gpu, diff_gpu);
            cv::cuda::resize(diff_gpu, diff_small_gpu, cv::Size(DIFF_W, DIFF_H),
                             0, 0, cv::INTER_AREA);
            cv::Mat diff_small;
            diff_small_gpu.download(diff_small);
            diff_small.convertTo(diff_small, CV_32F);
            float diff_max = *std::max_element(diff_small.begin<float>(), diff_small.end<float>());
            if (diff_max > 0) diff_small /= diff_max;

            // Patch stats — small CPU op on the cached ref Mat.
            int cx = ref_gray_.cols / 2, cy = ref_gray_.rows / 2;
            int r = 16;
            cv::Rect patch_roi(cx - r, cy - r, r * 2, r * 2);
            patch_roi &= cv::Rect(0, 0, ref_gray_.cols, ref_gray_.rows);
            cv::Scalar ref_mean, ref_std;
            cv::meanStdDev(ref_gray_(patch_roi), ref_mean, ref_std);

            cv::Mat diff_centre = diff_small(cv::Rect(
                DIFF_W / 2 - 2, DIFF_H / 2 - 2, 4, 4));
            cv::Scalar dc_mean, dc_std;
            cv::meanStdDev(diff_centre, dc_mean, dc_std);

            // Build observation vector
            std::vector<float> obs(OBS_DIM);
            obs[0] = static_cast<float>(shift.x / TX_NORM);
            obs[1] = static_cast<float>(shift.y / TY_NORM);
            obs[2] = static_cast<float>(ref_mean[0] / 255.0);
            obs[3] = static_cast<float>(ref_std[0] / 255.0);
            obs[4] = static_cast<float>(dc_mean[0]);
            obs[5] = static_cast<float>(dc_std[0]);
            obs[6] = prev_delta_[0];
            obs[7] = prev_delta_[1];
            obs[8] = prev_delta_[2];
            obs[9] = prev_delta_[3];
            obs[10] = static_cast<float>(frame_count_) / 5000.0f;

            // Frame difference flattened
            for (int i = 0; i < DIFF_SIZE; i++) {
                obs[11 + i] = diff_small.at<float>(i / DIFF_W, i % DIFF_W);
            }

            // ── 4. Run RL policy ──────────────────────────────────────────
            auto action = policy->predict(obs);
            m.delta_tx    = action[0] * DELTA_TX_MAX;
            m.delta_ty    = action[1] * DELTA_TY_MAX;
            m.delta_theta = action[2] * DELTA_THETA_MAX;
            m.delta_scale = action[3] * DELTA_SCALE_MAX;

            prev_delta_ = action;

            act_tx    = -shift.x + m.delta_tx;
            act_ty    = -shift.y + m.delta_ty;
            act_theta = m.delta_theta;
            act_scale = 1.0 + m.delta_scale;
        } else {
            // No RL model — use phase correlation only
            act_tx    = -shift.x;
            act_ty    = -shift.y;
            act_theta = 0.0;
            act_scale = 1.0;
        }

        m.tx = act_tx;
        m.ty = act_ty;
        m.rotation_deg = act_theta;
        m.scale = act_scale;

        // ── 5. Build correction H and warp on GPU ─────────────────────────
        double theta_rad = act_theta * CV_PI / 180.0;
        double cos_t = act_scale * std::cos(theta_rad);
        double sin_t = act_scale * std::sin(theta_rad);

        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
        H.at<double>(0, 0) =  cos_t;
        H.at<double>(0, 1) = -sin_t;
        H.at<double>(0, 2) =  act_tx;
        H.at<double>(1, 0) =  sin_t;
        H.at<double>(1, 1) =  cos_t;
        H.at<double>(1, 2) =  act_ty;

        // Reuse the gpu_frame_ already uploaded at the top of this call.
        cv::cuda::warpPerspective(gpu_frame_, gpu_stabilized_, H, frame.size(),
                                  cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
        cv::Mat stabilized;
        gpu_stabilized_.download(stabilized);

        m.stabilization_ms = static_cast<double>(cv::getTickCount() - t_start) /
                             cv::getTickFrequency() * 1000.0;
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
            if (ids[i] == 0) {
                cv::Mat gray;
                if (frame.channels() == 3)
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                else
                    gray = frame;
                cv::cornerSubPix(gray, corners[i], cv::Size(5,5), cv::Size(-1,-1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT,
                                     30, 0.01));
                for (const auto& c : corners[i]) center += c;
                center /= 4.0f;
                m.aruco_detected = true;
                m.aruco_center = center;
                break;
            }
        }

        m.aruco_detection_ms = static_cast<double>(cv::getTickCount() - t_start) /
                               cv::getTickFrequency() * 1000.0;
        return center;
    }

private:
    cv::aruco::Dictionary aruco_dict_;
    cv::aruco::ArucoDetector aruco_detector_;

    bool initialized_ = false;
    int frame_count_ = 0;
    cv::Mat ref_gray_;                    // CPU copy of ref (used for patch stats)
    std::array<float, 4> prev_delta_ = {0, 0, 0, 0};

    // Preallocated GPU buffers reused every frame.
    cv::cuda::GpuMat gpu_frame_;          // BGR upload
    cv::cuda::GpuMat gpu_gray_;           // CV_8UC1 grayscale
    cv::cuda::GpuMat gpu_ref_gray_;       // CV_8UC1 reference
    cv::cuda::GpuMat gpu_stabilized_;     // BGR warp output

    GpuPhaseCorrelator corr_;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Video processing
// ─────────────────────────────────────────────────────────────────────────────

void processVideo(const std::string& video_path, const std::string& output_dir,
                  const std::string& model_path, int max_frames = 0) {
    PhaseCorrelationStabilizer stabilizer;

    // Load RL policy if provided
    RLPolicy* policy = nullptr;
    std::unique_ptr<RLPolicy> policy_ptr;
    if (!model_path.empty()) {
        policy_ptr = std::make_unique<RLPolicy>(model_path);
        policy = policy_ptr.get();
        std::cout << "Mode: Phase Correlation + RL Policy\n";
    } else {
        std::cout << "Mode: Phase Correlation Only (no RL model)\n";
    }

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

    std::cout << "Input  : " << video_path << "\n";
    std::cout << "Frames : " << total_frames << " @ " << fps << " fps, "
              << width << "x" << height << "\n\n";

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,phase_dx,phase_dy,"
           "delta_tx,delta_ty,delta_theta,delta_scale,"
           "tx_px,ty_px,rotation_deg,scale,"
           "stabilization_ms,"
           "aruco_detected,aruco_center_x,aruco_center_y\n";

    cv::VideoWriter writer(
        output_dir + "/stabilized.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps, cv::Size(width, height));

    int frame_idx = 0;
    cv::Mat frame;
    int limit = (max_frames > 0) ? max_frames : total_frames;

    double total_ms = 0;

    // ArUco auto-disable: probe first 100 frames; if no marker shows up,
    // skip the (expensive) cornerSubPix detector for the rest of the clip.
    constexpr int ARUCO_PROBE_FRAMES = 100;
    bool aruco_active = true;
    int  aruco_hits   = 0;

    while (cap.read(frame) && frame_idx < limit) {
        if (frame_idx % 100 == 0)
            std::cout << "Frame " << frame_idx << "/" << limit << "\n";

        PhaseCorrelationStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m, policy);

        if (writer.isOpened()) writer.write(stabilized);

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
            << m.phase_dx << "," << m.phase_dy << ","
            << m.delta_tx << "," << m.delta_ty << ","
            << m.delta_theta << "," << m.delta_scale << ","
            << m.tx << "," << m.ty << ","
            << m.rotation_deg << "," << m.scale << ","
            << m.stabilization_ms << ","
            << m.aruco_detected << ","
            << m.aruco_center.x << "," << m.aruco_center.y << "\n";

        total_ms += m.stabilization_ms;
        frame_idx++;
    }

    cap.release();
    writer.release();

    std::cout << "\nProcessed " << frame_idx << " frames\n";
    std::cout << "Avg stabilization: " << (total_ms / frame_idx) << " ms/frame\n";
    std::cout << "Stabilized video → " << output_dir << "/stabilized.avi\n";
    std::cout << "Compensation CSV  → " << output_dir << "/compensation.csv\n";
}


static bool isVideoFile(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input> <output_dir> [--model model.onnx] [--frames N]\n";
        std::cerr << "  input       : video file or frame directory\n";
        std::cerr << "  --model     : ONNX RL policy (optional, uses phase corr only if omitted)\n";
        std::cerr << "  --frames N  : process only first N frames\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output_dir = argv[2];
    std::string model_path;
    int max_frames = 0;

    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (std::string(argv[i]) == "--frames" && i + 1 < argc)
            max_frames = std::stoi(argv[++i]);
    }

    if (!fs::exists(input)) {
        std::cerr << "Input not found: " << input << "\n";
        return 1;
    }

    processVideo(input, output_dir, model_path, max_frames);
    return 0;
}
