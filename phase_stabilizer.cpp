// ─────────────────────────────────────────────────────────────────────────────
//  Phase-Correlation Stabilizer
//
//  Replaces the ORB+RANSAC+Farneback two-stage pipeline (ransac_stabilizer.cpp)
//  with a single-stage GPU phase correlation.
//
//  Why:
//    * Phase correlation is sub-pixel from frame 1 (no two-stage hand-off).
//    * Same algorithm as DIC measurement (dic_analysis.cu) — consistent error
//      model, "we measure with phase correlation, we stabilize with phase
//      correlation."
//    * 1 cuFFT forward + 1 cross-power kernel + 1 cuFFT inverse per frame.
//      Roughly 13× faster than the ORB+RANSAC+Farneback chain.
//
//  Pipeline per frame:
//    1. Upload BGR → GPU
//    2. cvtColor BGR → GRAY on GPU
//    3. prepareKernel: GRAY × (mask · Hanning window)  (handles GpuMat row step)
//    4. cuFFT R2C forward
//    5. cross-power spectrum kernel: F1·conj(F2) / |F1·conj(F2)|
//    6. cuFFT C2R inverse → correlation surface
//    7. Download surface (~3.4 MB), find integer peak + 3-pt parabolic refinement
//    8. Build translation H, single warp on GPU, download
//
//  First version: translation only. Rotation in the rotorkit data is small
//  (~0.05° max measured) so the omission costs very little; it can be added
//  later via differential phase correlation on left/right halves.
//
//  Usage:
//    ./phase_stabilizer <input.avi> <out_dir> [--frames N] [--subject-mask X,Y,W,H]
// ─────────────────────────────────────────────────────────────────────────────

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  CUDA / cuFFT error checking
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
        std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__              \
                  << " — " << cudaGetErrorString(e_) << "\n";                  \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)

#define CUFFT_CHECK(call) do {                                                 \
    cufftResult e_ = (call);                                                   \
    if (e_ != CUFFT_SUCCESS) {                                                 \
        std::cerr << "cuFFT error " << __FILE__ << ":" << __LINE__             \
                  << " — code " << e_ << "\n";                                 \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)


// ─────────────────────────────────────────────────────────────────────────────
//  CUDA kernels
// ─────────────────────────────────────────────────────────────────────────────

// Convert 8U gray (with row step in bytes) → float, multiply by (mask·hann).
// Handles GpuMat row padding by accepting `gray_step` separately.
__global__ void prepareKernel(const unsigned char* __restrict__ gray, int gray_step,
                              const float* __restrict__ mask_hann,
                              float* __restrict__ out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    out[idx] = float(gray[y * gray_step + x]) * mask_hann[idx];
}

// Cross-power spectrum: F1 · conj(F2) / |F1 · conj(F2)|
__global__ void crossPowerKernel(const cufftComplex* __restrict__ fft_ref,
                                 const cufftComplex* __restrict__ fft_cur,
                                 cufftComplex* __restrict__ cross, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = fft_ref[idx].x, b = fft_ref[idx].y;
    float c = fft_cur[idx].x, d = fft_cur[idx].y;
    float re = a * c + b * d;
    float im = b * c - a * d;
    float mag = sqrtf(re * re + im * im);
    if (mag > 1e-10f) {
        cross[idx].x = re / mag;
        cross[idx].y = im / mag;
    } else {
        cross[idx].x = 0.f;
        cross[idx].y = 0.f;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
//  GPU phase correlator (one image size)
// ─────────────────────────────────────────────────────────────────────────────
class GpuPhaseCorrelator {
public:
    GpuPhaseCorrelator() = default;
    ~GpuPhaseCorrelator() { cleanup(); }

    void init(int w, int h, const cv::Mat& mask_hann_cpu) {
        if (initialized_) cleanup();
        w_ = w; h_ = h;
        n_real_     = w * h;
        n_complex_  = (w / 2 + 1) * h;   // cuFFT R2C halves the LAST dim (cols)

        CUDA_CHECK(cudaMalloc(&d_mask_hann_, n_real_    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_real_,  n_real_    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cur_real_,  n_real_    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_corr_,      n_real_    * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_ref_fft_,   n_complex_ * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_cur_fft_,   n_complex_ * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_cross_,     n_complex_ * sizeof(cufftComplex)));

        // mask_hann_cpu must be CV_32F, contiguous, h × w
        CUDA_CHECK(cudaMemcpy(d_mask_hann_, mask_hann_cpu.ptr<float>(),
                              n_real_ * sizeof(float), cudaMemcpyHostToDevice));

        int dims[2] = { h, w };
        CUFFT_CHECK(cufftPlanMany(&plan_fwd_, 2, dims,
                                   nullptr, 1, n_real_,
                                   nullptr, 1, n_complex_,
                                   CUFFT_R2C, 1));
        CUFFT_CHECK(cufftPlanMany(&plan_inv_, 2, dims,
                                   nullptr, 1, n_complex_,
                                   nullptr, 1, n_real_,
                                   CUFFT_C2R, 1));
        corr_cpu_.create(h_, w_, CV_32F);
        initialized_ = true;
    }

    // gray_8u must be CV_8U, sized w_×h_
    void setReference(const cv::cuda::GpuMat& gray_8u) {
        prepare(gray_8u, d_ref_real_);
        CUFFT_CHECK(cufftExecR2C(plan_fwd_, d_ref_real_, d_ref_fft_));
    }

    // Returns sub-pixel (dx, dy) shift of cur relative to ref.
    // Convention: positive dx means current frame moved right relative to ref,
    // so to *cancel* the motion the warp should apply -dx, -dy.
    cv::Point2d correlate(const cv::cuda::GpuMat& gray_8u, double* response = nullptr) {
        prepare(gray_8u, d_cur_real_);
        CUFFT_CHECK(cufftExecR2C(plan_fwd_, d_cur_real_, d_cur_fft_));

        int threads = 256;
        int blocks_c = (n_complex_ + threads - 1) / threads;
        crossPowerKernel<<<blocks_c, threads>>>(d_ref_fft_, d_cur_fft_,
                                                 d_cross_, n_complex_);

        CUFFT_CHECK(cufftExecC2R(plan_inv_, d_cross_, d_corr_));

        CUDA_CHECK(cudaMemcpy(corr_cpu_.data, d_corr_,
                              n_real_ * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // Find integer peak
        double max_val; cv::Point max_loc;
        cv::minMaxLoc(corr_cpu_, nullptr, &max_val, nullptr, &max_loc);

        // Sub-pixel parabolic refinement on a 3-point neighbourhood
        double dx = max_loc.x;
        double dy = max_loc.y;
        if (max_loc.x > 0 && max_loc.x < w_ - 1) {
            float l = corr_cpu_.at<float>(max_loc.y, max_loc.x - 1);
            float r = corr_cpu_.at<float>(max_loc.y, max_loc.x + 1);
            float c = corr_cpu_.at<float>(max_loc.y, max_loc.x);
            float denom = 2.f * (2.f * c - l - r);
            if (std::fabs(denom) > 1e-10f) dx += (l - r) / denom;
        }
        if (max_loc.y > 0 && max_loc.y < h_ - 1) {
            float t = corr_cpu_.at<float>(max_loc.y - 1, max_loc.x);
            float b = corr_cpu_.at<float>(max_loc.y + 1, max_loc.x);
            float c = corr_cpu_.at<float>(max_loc.y, max_loc.x);
            float denom = 2.f * (2.f * c - t - b);
            if (std::fabs(denom) > 1e-10f) dy += (t - b) / denom;
        }

        // Convert from FFT-shift convention to signed displacement
        if (dx > w_ / 2.0) dx -= w_;
        if (dy > h_ / 2.0) dy -= h_;

        if (response) *response = max_val / double(n_real_);
        return cv::Point2d(dx, dy);
    }

private:
    void prepare(const cv::cuda::GpuMat& gray_8u, float* d_out) {
        dim3 block(32, 8);
        dim3 grid((w_ + block.x - 1) / block.x,
                  (h_ + block.y - 1) / block.y);
        prepareKernel<<<grid, block>>>(gray_8u.ptr<unsigned char>(),
                                        int(gray_8u.step),
                                        d_mask_hann_, d_out, w_, h_);
    }

    void cleanup() {
        if (!initialized_) return;
        cudaFree(d_mask_hann_); cudaFree(d_ref_real_); cudaFree(d_cur_real_);
        cudaFree(d_corr_);      cudaFree(d_ref_fft_); cudaFree(d_cur_fft_);
        cudaFree(d_cross_);
        cufftDestroy(plan_fwd_); cufftDestroy(plan_inv_);
        initialized_ = false;
    }

    int w_ = 0, h_ = 0, n_real_ = 0, n_complex_ = 0;
    float* d_mask_hann_ = nullptr;
    float* d_ref_real_  = nullptr;
    float* d_cur_real_  = nullptr;
    float* d_corr_      = nullptr;
    cufftComplex* d_ref_fft_ = nullptr;
    cufftComplex* d_cur_fft_ = nullptr;
    cufftComplex* d_cross_   = nullptr;
    cufftHandle plan_fwd_{}, plan_inv_{};
    cv::Mat corr_cpu_;
    bool initialized_ = false;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Stabilizer wrapper
// ─────────────────────────────────────────────────────────────────────────────
class PhaseStabilizer {
public:
    struct Metrics {
        double tx           = 0.0;
        double ty           = 0.0;
        double rotation_deg = 0.0;
        double response     = 0.0;     // peak quality (higher = sharper peak)
        double total_ms     = 0.0;
    };

    void setSubjectRegion(const cv::Rect& r) { subject_rect_ = r; }

    cv::Mat stabilize(const cv::Mat& frame, Metrics& m) {
        auto t0 = cv::getTickCount();

        if (!initialized_) {
            int w = frame.cols, h = frame.rows;
            cv::Mat mask_hann = buildMaskHann(w, h, subject_rect_);
            full_.init(w, h, mask_hann);
            w_ = w; h_ = h;
            initialized_ = true;
        }

        gpu_frame_.upload(frame);
        cv::cuda::cvtColor(gpu_frame_, gpu_gray_, cv::COLOR_BGR2GRAY);

        if (!ref_set_) {
            full_.setReference(gpu_gray_);
            ref_set_ = true;
            m.total_ms = tickMs(cv::getTickCount() - t0);
            return frame.clone();
        }

        double resp;
        cv::Point2d peak = full_.correlate(gpu_gray_, &resp);

        // Phase-correlation peak math: if  cur(x,y) = ref(x − dx, y − dy)
        // then the IFFT of the cross-power spectrum peaks at (−dx, −dy).
        // To stabilize, we want to undo the shift, i.e. apply the inverse
        // displacement.  Working through cv::warpPerspective's convention
        // (dst(x,y) = src(x − tx, y − ty) with tx = H[0,2], ty = H[1,2]):
        //   tx = −dx_actual = peak.x
        //   ty = −dy_actual = peak.y
        m.tx           = peak.x;
        m.ty           = peak.y;
        m.rotation_deg = 0.0;     // first version: translation only
        m.response     = resp;

        cv::Mat H = cv::Mat::eye(3, 3, CV_64F);
        H.at<double>(0, 2) = m.tx;
        H.at<double>(1, 2) = m.ty;

        cv::cuda::warpPerspective(gpu_frame_, gpu_stab_, H,
                                  cv::Size(w_, h_),
                                  cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                                  cv::Scalar(0, 0, 0));

        cv::Mat stabilized;
        gpu_stab_.download(stabilized);
        m.total_ms = tickMs(cv::getTickCount() - t0);
        return stabilized;
    }

private:
    static double tickMs(int64 t) {
        return double(t) / cv::getTickFrequency() * 1000.0;
    }

    // Build the windowed mask: 2D Hanning × {0 inside subject_rect, 1 outside}.
    // Output: CV_32F, h × w, values in [0, 1].
    static cv::Mat buildMaskHann(int w, int h, const cv::Rect& subject) {
        cv::Mat hx(1, w, CV_32F), hy(h, 1, CV_32F);
        for (int x = 0; x < w; x++)
            hx.at<float>(0, x) = 0.5f * (1.0f - std::cos(2.0f * float(M_PI) * x / (w - 1)));
        for (int y = 0; y < h; y++)
            hy.at<float>(y, 0) = 0.5f * (1.0f - std::cos(2.0f * float(M_PI) * y / (h - 1)));
        cv::Mat m = hy * hx;   // outer product → h × w Hanning
        if (subject.area() > 0) {
            cv::Rect s = subject & cv::Rect(0, 0, w, h);
            if (s.area() > 0) m(s).setTo(0.f);
        }
        return m;
    }

    GpuPhaseCorrelator full_;
    cv::cuda::GpuMat gpu_frame_, gpu_gray_, gpu_stab_;
    int w_ = 0, h_ = 0;
    cv::Rect subject_rect_;
    bool initialized_ = false;
    bool ref_set_     = false;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.avi|.mp4> <output_dir> "
                     "[--frames N] [--subject-mask X,Y,W,H]\n";
        return 1;
    }

    std::string input      = argv[1];
    std::string output_dir = argv[2];

    int max_frames = 0;
    cv::Rect subject_rect;
    for (int i = 3; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (flag == "--subject-mask" && i + 1 < argc) {
            int x, y, w, h;
            if (std::sscanf(argv[++i], "%d,%d,%d,%d", &x, &y, &w, &h) == 4) {
                subject_rect = cv::Rect(x, y, w, h);
            } else {
                std::cerr << "Bad --subject-mask value (expected X,Y,W,H)\n";
                return 1;
            }
        }
    }

    fs::create_directories(output_dir);

    cv::VideoCapture cap(input);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open: " << input << "\n";
        return 1;
    }

    int total = int(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int w = int(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = int(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "Phase Correlation Stabilizer\n";
    std::cout << "  Input  : " << input << "\n";
    std::cout << "  Frames : " << total << " @ " << fps << " fps, "
              << w << "x" << h << "\n";
    if (subject_rect.area() > 0)
        std::cout << "  Mask   : " << subject_rect.x << "," << subject_rect.y
                  << "," << subject_rect.width << "," << subject_rect.height << "\n";
    std::cout << "\n";

    PhaseStabilizer stab;
    stab.setSubjectRegion(subject_rect);

    cv::VideoWriter writer(output_dir + "/stabilized.avi",
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           fps, cv::Size(w, h));
    if (!writer.isOpened()) {
        std::cerr << "Warning: VideoWriter failed to open\n";
    }

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,tx_px,ty_px,rotation_deg,response,total_ms\n";

    int limit = (max_frames > 0) ? std::min(max_frames, total) : total;
    int frame_idx = 0;
    cv::Mat frame;
    double sum_ms = 0.0;

    int64 last_tick = cv::getTickCount();
    int   last_frame = 0;

    while (cap.read(frame) && frame_idx < limit) {
        if (frame_idx > 0 && frame_idx % 100 == 0) {
            int64 now = cv::getTickCount();
            double elapsed_s = double(now - last_tick) / cv::getTickFrequency();
            double fps_live = (frame_idx - last_frame) / elapsed_s;
            double mean_ms_so_far = sum_ms / frame_idx;
            std::cout << "Frame " << frame_idx << "/" << limit
                      << "  |  live: " << fps_live << " fps"
                      << "  mean: " << mean_ms_so_far << " ms/frame ("
                      << (1000.0 / mean_ms_so_far) << " fps)\n"
                      << std::flush;
            last_tick = now;
            last_frame = frame_idx;
        }

        PhaseStabilizer::Metrics m;
        cv::Mat stabilized = stab.stabilize(frame, m);

        if (writer.isOpened()) writer.write(stabilized);

        csv << frame_idx << ","
            << m.tx << "," << m.ty << "," << m.rotation_deg << ","
            << m.response << "," << m.total_ms << "\n";

        sum_ms += m.total_ms;
        frame_idx++;
    }

    cap.release();
    writer.release();

    double mean_ms = (frame_idx > 0) ? sum_ms / frame_idx : 0.0;
    std::cout << "\nProcessed " << frame_idx << " frames\n";
    std::cout << "Mean time/frame: " << mean_ms << " ms (≈ "
              << (mean_ms > 0 ? 1000.0 / mean_ms : 0.0) << " fps)\n";
    std::cout << "Stabilized video → " << output_dir << "/stabilized.avi\n";
    std::cout << "Compensation CSV → " << output_dir << "/compensation.csv\n";
    return 0;
}
