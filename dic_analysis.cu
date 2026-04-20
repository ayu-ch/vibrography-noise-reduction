// ─────────────────────────────────────────────────────────────────────────────
//  Stage 3 — cuFFT DIC: Sub-pixel Vibration Measurement
//
//  Input:  Stabilized frames from Stage 1+2 (camera sway removed)
//  Output: Per-ROI vibration spectrum (frequency, amplitude, phase)
//
//  Pipeline:
//    1. Divide frame into ROI grid (e.g. 32×32 patches)
//    2. For each ROI, compute sub-pixel displacement vs reference (phase correlation)
//    3. Collect displacement time series across N frames
//    4. Batch FFT all ROIs → frequency domain (cuFFT)
//    5. Extract dominant frequency, amplitude, phase per ROI
//
//  All heavy computation on GPU via CUDA + cuFFT.
//
//  Usage:
//    ./dic_analysis <stabilized_video> <output_dir> [--roi-size 32] [--frames 512]
// ─────────────────────────────────────────────────────────────────────────────

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <cmath>

namespace fs = std::filesystem;

// ─────────────────────────────────────────────────────────────────────────────
//  CUDA error checking
// ─────────────────────────────────────────────────────────────────────────────

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " — " << cudaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
} while(0)

#define CUFFT_CHECK(call) do { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__ \
                  << " — code " << err << "\n"; \
        exit(1); \
    } \
} while(0)


// ─────────────────────────────────────────────────────────────────────────────
//  CUDA Kernels
// ─────────────────────────────────────────────────────────────────────────────

// Apply Hanning window to a batch of ROI patches
__global__ void applyHanningKernel(float* data, const float* hann,
                                    int roi_size, int n_rois) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rois * roi_size * roi_size;
    if (idx >= total) return;

    int pixel_in_roi = idx % (roi_size * roi_size);
    data[idx] *= hann[pixel_in_roi];
}

// Compute cross-power spectrum: (F1 * conj(F2)) / |F1 * conj(F2)|
__global__ void crossPowerSpectrumKernel(const cufftComplex* fft_ref,
                                          const cufftComplex* fft_cur,
                                          cufftComplex* cross,
                                          int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float a = fft_ref[idx].x, b = fft_ref[idx].y;
    float c = fft_cur[idx].x, d = fft_cur[idx].y;

    // F1 * conj(F2)
    float re = a * c + b * d;
    float im = b * c - a * d;

    // Normalize
    float mag = sqrtf(re * re + im * im);
    if (mag > 1e-10f) {
        cross[idx].x = re / mag;
        cross[idx].y = im / mag;
    } else {
        cross[idx].x = 0.0f;
        cross[idx].y = 0.0f;
    }
}

// Find sub-pixel peak in inverse FFT result (per ROI)
// Uses 3-point parabolic interpolation around the peak
__global__ void findSubPixelPeakKernel(const float* corr, int roi_size,
                                        int n_rois, float* dx_out, float* dy_out) {
    int roi_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (roi_idx >= n_rois) return;

    const float* roi = corr + roi_idx * roi_size * roi_size;
    int half = roi_size / 2;

    // Find integer peak
    float max_val = -1e30f;
    int peak_x = 0, peak_y = 0;
    for (int y = 0; y < roi_size; y++) {
        for (int x = 0; x < roi_size; x++) {
            float v = roi[y * roi_size + x];
            if (v > max_val) {
                max_val = v;
                peak_x = x;
                peak_y = y;
            }
        }
    }

    // Convert to signed displacement (FFT shift)
    float ix = (peak_x > half) ? (float)(peak_x - roi_size) : (float)peak_x;
    float iy = (peak_y > half) ? (float)(peak_y - roi_size) : (float)peak_y;

    // Sub-pixel refinement via parabolic interpolation
    if (peak_x > 0 && peak_x < roi_size - 1) {
        float left  = roi[peak_y * roi_size + peak_x - 1];
        float right = roi[peak_y * roi_size + peak_x + 1];
        float denom = 2.0f * (2.0f * max_val - left - right);
        if (fabsf(denom) > 1e-10f)
            ix += (left - right) / denom;
    }
    if (peak_y > 0 && peak_y < roi_size - 1) {
        float top = roi[(peak_y - 1) * roi_size + peak_x];
        float bot = roi[(peak_y + 1) * roi_size + peak_x];
        float denom = 2.0f * (2.0f * max_val - top - bot);
        if (fabsf(denom) > 1e-10f)
            iy += (top - bot) / denom;
    }

    dx_out[roi_idx] = ix;
    dy_out[roi_idx] = iy;
}

// Apply Hanning window to 1D displacement time series before spectral FFT
__global__ void applyHanning1DKernel(float* data, int n_frames, int n_rois) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rois * n_frames) return;

    int frame = idx % n_frames;
    float w = 0.5f * (1.0f - cosf(2.0f * M_PI * frame / (n_frames - 1)));
    data[idx] *= w;
}

// High-pass filter: subtract a moving average to remove slow drift
// This reveals the actual vibration signal hidden under the ~2Hz camera drift
__global__ void highPassFilterKernel(float* data, int n_frames, int n_rois, int window) {
    int roi = blockIdx.x * blockDim.x + threadIdx.x;
    if (roi >= n_rois) return;

    float* ts = data + roi * n_frames;

    // Compute moving average and subtract
    for (int i = 0; i < n_frames; i++) {
        float sum = 0;
        int count = 0;
        for (int j = max(0, i - window/2); j < min(n_frames, i + window/2 + 1); j++) {
            sum += ts[j];
            count++;
        }
        // Store filtered value (in-place is OK since we read before write)
        // Actually need a temp buffer — use shared memory or two-pass
    }
    // Two-pass: first compute MA into temp, then subtract
    // For simplicity, do it on CPU instead
}


// ─────────────────────────────────────────────────────────────────────────────
//  DIC Analyzer
// ─────────────────────────────────────────────────────────────────────────────

class DICAnalyzer {
public:
    DICAnalyzer(int roi_size = 32, double fps = 1000.0)
        : roi_size_(roi_size), fps_(fps) {}

    struct ROIResult {
        int roi_x, roi_y;          // ROI position in frame
        double dominant_freq_hz;    // strongest vibration frequency
        double amplitude_px;        // vibration amplitude in pixels
        double phase_rad;           // vibration phase
        double mean_dx, mean_dy;    // mean displacement (DC component)
    };

    // Per-frame displacement data (populated by analyze())
    // Layout: displacement[roi_idx * fft_frames + frame_idx]
    std::vector<float> frame_dx, frame_dy;
    int n_rois_out = 0;
    int fft_frames_out = 0;
    int cols_out = 0, rows_out = 0;

    // Process stabilized video: extract vibration spectrum for each ROI
    std::vector<ROIResult> analyze(const std::string& video_path, int max_frames = 512) {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open: " << video_path << "\n";
            return {};
        }

        int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        int n_frames = std::min(total, max_frames);

        // Round down to power of 2 for efficient FFT
        int fft_frames = 1;
        while (fft_frames * 2 <= n_frames) fft_frames *= 2;

        int cols = width / roi_size_;
        int rows = height / roi_size_;
        int n_rois = cols * rows;

        std::cout << "DIC Analysis\n";
        std::cout << "  Frame: " << width << "×" << height << "\n";
        std::cout << "  ROI: " << roi_size_ << "×" << roi_size_
                  << " → " << cols << "×" << rows << " = " << n_rois << " ROIs\n";
        std::cout << "  Frames: " << fft_frames << " (of " << total << ")\n";
        std::cout << "  FPS: " << fps_ << " → freq resolution: "
                  << fps_ / fft_frames << " Hz\n\n";

        // ── Read frames and extract ROI patches ──────────────────────────
        cv::Mat ref_gray;
        std::vector<float> all_dx(n_rois * fft_frames, 0.0f);
        std::vector<float> all_dy(n_rois * fft_frames, 0.0f);

        // Precompute 2D Hanning window for ROI
        std::vector<float> hann2d(roi_size_ * roi_size_);
        for (int y = 0; y < roi_size_; y++) {
            float wy = 0.5f * (1.0f - std::cos(2.0 * M_PI * y / (roi_size_ - 1)));
            for (int x = 0; x < roi_size_; x++) {
                float wx = 0.5f * (1.0f - std::cos(2.0 * M_PI * x / (roi_size_ - 1)));
                hann2d[y * roi_size_ + x] = wy * wx;
            }
        }

        // GPU allocations for phase correlation
        int roi_pixels = roi_size_ * roi_size_;
        int fft_w = roi_size_ / 2 + 1;
        int fft_complex_size = fft_w * roi_size_;

        float *d_ref_patches, *d_cur_patches, *d_hann, *d_corr;
        cufftComplex *d_fft_ref, *d_fft_cur, *d_cross;
        float *d_dx, *d_dy;

        CUDA_CHECK(cudaMalloc(&d_ref_patches, n_rois * roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cur_patches, n_rois * roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hann, roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_corr, n_rois * roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fft_ref, n_rois * fft_complex_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_fft_cur, n_rois * fft_complex_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_cross, n_rois * fft_complex_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_dx, n_rois * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dy, n_rois * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_hann, hann2d.data(), roi_pixels * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Create batched 2D FFT plans
        int dims[2] = {roi_size_, roi_size_};
        cufftHandle plan_fwd, plan_inv;
        CUFFT_CHECK(cufftPlanMany(&plan_fwd, 2, dims,
                                   NULL, 1, roi_pixels,
                                   NULL, 1, fft_complex_size,
                                   CUFFT_R2C, n_rois));
        CUFFT_CHECK(cufftPlanMany(&plan_inv, 2, dims,
                                   NULL, 1, fft_complex_size,
                                   NULL, 1, roi_pixels,
                                   CUFFT_C2R, n_rois));

        int threads = 256;

        for (int frame_idx = 0; frame_idx < fft_frames; frame_idx++) {
            if (frame_idx % 100 == 0)
                std::cout << "  Phase correlation: frame " << frame_idx << "/" << fft_frames << "\n";

            cv::Mat frame, gray;
            if (!cap.read(frame)) break;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            if (frame_idx == 0) {
                ref_gray = gray.clone();
                // Extract reference patches and FFT them
                extractPatches(ref_gray, d_ref_patches, cols, rows);
                applyHanningKernel<<<(n_rois * roi_pixels + threads - 1) / threads, threads>>>(
                    d_ref_patches, d_hann, roi_size_, n_rois);
                CUFFT_CHECK(cufftExecR2C(plan_fwd, d_ref_patches, d_fft_ref));
                continue;
            }

            // Extract current patches, apply Hanning, FFT
            extractPatches(gray, d_cur_patches, cols, rows);
            applyHanningKernel<<<(n_rois * roi_pixels + threads - 1) / threads, threads>>>(
                d_cur_patches, d_hann, roi_size_, n_rois);
            CUFFT_CHECK(cufftExecR2C(plan_fwd, d_cur_patches, d_fft_cur));

            // Cross-power spectrum
            int total_complex = n_rois * fft_complex_size;
            crossPowerSpectrumKernel<<<(total_complex + threads - 1) / threads, threads>>>(
                d_fft_ref, d_fft_cur, d_cross, total_complex);

            // Inverse FFT → correlation surface
            CUFFT_CHECK(cufftExecC2R(plan_inv, d_cross, d_corr));

            // Find sub-pixel peak per ROI
            findSubPixelPeakKernel<<<(n_rois + threads - 1) / threads, threads>>>(
                d_corr, roi_size_, n_rois, d_dx, d_dy);

            // Download displacements
            std::vector<float> dx(n_rois), dy(n_rois);
            CUDA_CHECK(cudaMemcpy(dx.data(), d_dx, n_rois * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(dy.data(), d_dy, n_rois * sizeof(float), cudaMemcpyDeviceToHost));

            for (int i = 0; i < n_rois; i++) {
                all_dx[i * fft_frames + frame_idx] = dx[i];
                all_dy[i * fft_frames + frame_idx] = dy[i];
            }
        }

        cufftDestroy(plan_fwd);
        cufftDestroy(plan_inv);
        cap.release();

        std::cout << "\n  Spectral analysis (cuFFT batch)...\n";

        // ── Batch FFT of displacement time series ────────────────────────
        // For each ROI, FFT the displacement time series to get vibration spectrum
        float *d_disp_x, *d_disp_y;
        cufftComplex *d_spec_x, *d_spec_y;
        int spec_size = fft_frames / 2 + 1;

        CUDA_CHECK(cudaMalloc(&d_disp_x, n_rois * fft_frames * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_disp_y, n_rois * fft_frames * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_spec_x, n_rois * spec_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_spec_y, n_rois * spec_size * sizeof(cufftComplex)));

        // ── High-pass filter: remove slow drift (<5Hz) on CPU ─────────
        // The ~2Hz camera drift dominates the raw displacement signal.
        // Subtract a moving average (window=200 frames at 1000fps = 5Hz cutoff)
        // to reveal the actual structural vibration.
        int hp_window = static_cast<int>(fps_ / 5.0);  // 5Hz cutoff
        if (hp_window < 3) hp_window = 3;
        std::cout << "  High-pass filter: window=" << hp_window
                  << " frames (>" << fps_/hp_window << " Hz pass)\n";

        for (int roi = 0; roi < n_rois; roi++) {
            float* tx = &all_dx[roi * fft_frames];
            float* ty = &all_dy[roi * fft_frames];
            // Compute moving average
            std::vector<float> ma_x(fft_frames), ma_y(fft_frames);
            for (int i = 0; i < fft_frames; i++) {
                float sx = 0, sy = 0;
                int cnt = 0;
                for (int j = std::max(0, i - hp_window/2);
                     j < std::min(fft_frames, i + hp_window/2 + 1); j++) {
                    sx += tx[j]; sy += ty[j]; cnt++;
                }
                ma_x[i] = sx / cnt;
                ma_y[i] = sy / cnt;
            }
            // Subtract: keep only high-frequency vibration
            for (int i = 0; i < fft_frames; i++) {
                tx[i] -= ma_x[i];
                ty[i] -= ma_y[i];
            }
        }

        CUDA_CHECK(cudaMemcpy(d_disp_x, all_dx.data(),
                              n_rois * fft_frames * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_disp_y, all_dy.data(),
                              n_rois * fft_frames * sizeof(float), cudaMemcpyHostToDevice));

        // Apply Hanning window to time series
        int total_ts = n_rois * fft_frames;
        applyHanning1DKernel<<<(total_ts + threads - 1) / threads, threads>>>(
            d_disp_x, fft_frames, n_rois);
        applyHanning1DKernel<<<(total_ts + threads - 1) / threads, threads>>>(
            d_disp_y, fft_frames, n_rois);

        // Batched 1D FFT: displacement time series → frequency spectrum
        cufftHandle plan_1d;
        CUFFT_CHECK(cufftPlanMany(&plan_1d, 1, &fft_frames,
                                   NULL, 1, fft_frames,
                                   NULL, 1, spec_size,
                                   CUFFT_R2C, n_rois));
        CUFFT_CHECK(cufftExecR2C(plan_1d, d_disp_x, d_spec_x));
        CUFFT_CHECK(cufftExecR2C(plan_1d, d_disp_y, d_spec_y));

        // Download spectra
        std::vector<cufftComplex> spec_x(n_rois * spec_size);
        std::vector<cufftComplex> spec_y(n_rois * spec_size);
        CUDA_CHECK(cudaMemcpy(spec_x.data(), d_spec_x,
                              n_rois * spec_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(spec_y.data(), d_spec_y,
                              n_rois * spec_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost));

        cufftDestroy(plan_1d);

        // ── Extract dominant frequency per ROI ───────────────────────────
        double freq_resolution = fps_ / fft_frames;
        std::vector<ROIResult> results(n_rois);

        // Minimum frequency bin to consider (skip drift below 5Hz)
        int min_bin = std::max(1, static_cast<int>(5.0 / freq_resolution));
        // Border exclusion: skip ROIs within 2 ROI widths of edge (warp artifacts)
        int border_rois = 2;

        for (int roi = 0; roi < n_rois; roi++) {
            int rx = (roi % cols) * roi_size_;
            int ry = (roi / cols) * roi_size_;
            int rc = roi % cols, rr = roi / cols;

            // Skip border ROIs
            bool is_border = (rc < border_rois || rc >= cols - border_rois ||
                              rr < border_rois || rr >= rows - border_rois);

            // Find dominant frequency (skip low frequencies)
            float max_mag = 0;
            int max_bin = min_bin;
            if (!is_border) {
                for (int k = min_bin; k < spec_size; k++) {
                    float mx = spec_x[roi * spec_size + k].x;
                    float my = spec_x[roi * spec_size + k].y;
                    float mag = std::sqrt(mx * mx + my * my);

                    float mxy = spec_y[roi * spec_size + k].x;
                    float myy = spec_y[roi * spec_size + k].y;
                    mag += std::sqrt(mxy * mxy + myy * myy);

                    if (mag > max_mag) {
                        max_mag = mag;
                        max_bin = k;
                    }
                }
            }

            float cx = spec_x[roi * spec_size + max_bin].x;
            float cy_val = spec_x[roi * spec_size + max_bin].y;

            results[roi] = {
                rx, ry,
                is_border ? 0.0 : max_bin * freq_resolution,
                is_border ? 0.0 : max_mag / fft_frames,
                std::atan2(cy_val, cx),
                all_dx[roi * fft_frames],
                all_dy[roi * fft_frames]
            };
        }

        // Store per-frame data for overlay video
        frame_dx = all_dx;
        frame_dy = all_dy;
        n_rois_out = n_rois;
        fft_frames_out = fft_frames;
        cols_out = cols;
        rows_out = rows;

        // Cleanup
        cudaFree(d_ref_patches); cudaFree(d_cur_patches);
        cudaFree(d_hann); cudaFree(d_corr);
        cudaFree(d_fft_ref); cudaFree(d_fft_cur); cudaFree(d_cross);
        cudaFree(d_dx); cudaFree(d_dy);
        cudaFree(d_disp_x); cudaFree(d_disp_y);
        cudaFree(d_spec_x); cudaFree(d_spec_y);

        return results;
    }

private:
    void extractPatches(const cv::Mat& gray, float* d_patches, int cols, int rows) {
        int n_rois = cols * rows;
        int roi_pixels = roi_size_ * roi_size_;
        std::vector<float> patches(n_rois * roi_pixels);

        for (int ry = 0; ry < rows; ry++) {
            for (int rx = 0; rx < cols; rx++) {
                int roi_idx = ry * cols + rx;
                int px = rx * roi_size_;
                int py = ry * roi_size_;
                for (int y = 0; y < roi_size_; y++) {
                    for (int x = 0; x < roi_size_; x++) {
                        patches[roi_idx * roi_pixels + y * roi_size_ + x] =
                            static_cast<float>(gray.at<uchar>(py + y, px + x));
                    }
                }
            }
        }

        CUDA_CHECK(cudaMemcpy(d_patches, patches.data(),
                              n_rois * roi_pixels * sizeof(float), cudaMemcpyHostToDevice));
    }

    int roi_size_;
    double fps_;
};


// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <stabilized_video> <output_dir> [--roi-size 32] [--frames 512] [--fps 1000]\n";
        return 1;
    }

    std::string input = argv[1];
    std::string output_dir = argv[2];
    int roi_size = 32;
    int max_frames = 512;
    double fps = 1000.0;

    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--roi-size" && i + 1 < argc)
            roi_size = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--frames" && i + 1 < argc)
            max_frames = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--fps" && i + 1 < argc)
            fps = std::stod(argv[++i]);
    }

    fs::create_directories(output_dir);

    DICAnalyzer dic(roi_size, fps);
    auto results = dic.analyze(input, max_frames);

    if (results.empty()) {
        std::cerr << "No results\n";
        return 1;
    }

    // ── Save CSV ─────────────────────────────────────────────────────────
    std::string csv_path = output_dir + "/vibration_analysis.csv";
    std::ofstream csv(csv_path);
    csv << "roi_x,roi_y,dominant_freq_hz,amplitude_px,phase_rad,mean_dx,mean_dy\n";
    for (const auto& r : results) {
        csv << r.roi_x << "," << r.roi_y << ","
            << r.dominant_freq_hz << "," << r.amplitude_px << ","
            << r.phase_rad << "," << r.mean_dx << "," << r.mean_dy << "\n";
    }
    std::cout << "\nSaved: " << csv_path << " (" << results.size() << " ROIs)\n";

    // ── Save vibration map as image ──────────────────────────────────────
    cv::VideoCapture cap(input);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cap.release();

    int cols = width / roi_size;
    int rows = height / roi_size;

    // Frequency map (only show meaningful vibration, not edge noise)
    cv::Mat freq_map(rows, cols, CV_64F, cv::Scalar(0));
    cv::Mat amp_map(rows, cols, CV_64F, cv::Scalar(0));
    for (const auto& r : results) {
        int rx = r.roi_x / roi_size;
        int ry = r.roi_y / roi_size;
        if (rx >= 3 && rx < cols - 3 && ry >= 3 && ry < rows - 3 &&
            r.dominant_freq_hz > 10.0 && r.dominant_freq_hz < 200.0 &&
            r.amplitude_px > 0.005 && r.amplitude_px < 2.0) {
            freq_map.at<double>(ry, rx) = r.dominant_freq_hz;
            amp_map.at<double>(ry, rx) = r.amplitude_px;
        }
    }

    // Colorize frequency map
    cv::Mat freq_norm;
    cv::normalize(freq_map, freq_norm, 0, 255, cv::NORM_MINMAX);
    freq_norm.convertTo(freq_norm, CV_8U);
    cv::Mat freq_color;
    cv::applyColorMap(freq_norm, freq_color, cv::COLORMAP_JET);
    cv::resize(freq_color, freq_color, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::imwrite(output_dir + "/frequency_map.png", freq_color);

    // Colorize amplitude map — normalize to 95th percentile so small
    // vibrations on the speckle board are visible (edges don't dominate)
    std::vector<double> amp_values;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (amp_map.at<double>(r, c) > 0)
                amp_values.push_back(amp_map.at<double>(r, c));
    double amp_cap = 0.2;  // default
    if (!amp_values.empty()) {
        std::sort(amp_values.begin(), amp_values.end());
        amp_cap = amp_values[static_cast<int>(amp_values.size() * 0.95)];
        if (amp_cap < 0.01) amp_cap = 0.01;
    }
    cv::Mat amp_norm;
    amp_map.convertTo(amp_norm, CV_64F, 255.0 / amp_cap);
    amp_norm = cv::min(amp_norm, 255.0);
    amp_norm.convertTo(amp_norm, CV_8U);
    cv::Mat amp_color;
    cv::applyColorMap(amp_norm, amp_color, cv::COLORMAP_HOT);
    cv::resize(amp_color, amp_color, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::imwrite(output_dir + "/amplitude_map.png", amp_color);

    std::cout << "Saved: " << output_dir << "/frequency_map.png\n";
    std::cout << "Saved: " << output_dir << "/amplitude_map.png\n";

    // Print summary
    double max_amp = 0;
    double dom_freq = 0;
    for (const auto& r : results) {
        if (r.amplitude_px > max_amp) {
            max_amp = r.amplitude_px;
            dom_freq = r.dominant_freq_hz;
        }
    }
    std::cout << "\n=== Summary ===\n";
    std::cout << "Dominant frequency: " << dom_freq << " Hz\n";
    std::cout << "Max amplitude: " << max_amp << " px\n";
    std::cout << "Frequency resolution: " << fps / max_frames << " Hz\n";

    // ── Generate dynamic overlay video ─────────────────────────────────
    // Each frame shows the per-ROI displacement magnitude as a live heatmap
    // that pulses with the vibration. This makes the vibration VISIBLE.
    std::cout << "\nGenerating dynamic overlay video...\n";

    int n_rois = static_cast<int>(results.size());

    // Read frequencies and amplitudes from results
    std::vector<double> roi_freqs(n_rois), roi_amps(n_rois);
    for (int i = 0; i < n_rois; i++) {
        roi_freqs[i] = results[i].dominant_freq_hz;
        roi_amps[i] = results[i].amplitude_px;
    }

    // (skip CSV re-read — we already have results)
    if (false) {
    std::ifstream csv_in(output_dir + "/vibration_analysis.csv");
    std::string header_line;
    std::getline(csv_in, header_line);
    for (int i = 0; i < n_rois; i++) {
        int rx_tmp, ry_tmp;
        double f, a, p, mdx, mdy;
        char comma;
        csv_in >> rx_tmp >> comma >> ry_tmp >> comma >> f >> comma
               >> a >> comma >> p >> comma >> mdx >> comma >> mdy;
        roi_freqs[i] = f;
        roi_amps[i] = a;
    }
    csv_in.close();
    } // end if(false)

    cv::VideoCapture cap2(input);
    if (!cap2.isOpened()) {
        std::cerr << "Cannot reopen video for overlay\n";
        return 1;
    }

    int fft_frames_used = 1;
    while (fft_frames_used * 2 <= max_frames) fft_frames_used *= 2;

    cv::VideoWriter overlay_writer(
        output_dir + "/vibration_overlay.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        30.0, cv::Size(width, height));

    int frame_idx2 = 0;
    int limit_overlay = std::min(static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_COUNT)),
                                  fft_frames_used);

    // Compute stats from speckle region only (skip edges/background)
    // The speckle boards are the ROIs with ~45Hz signal
    double vis_max_amp = 0;
    double speckle_dom_freq = 0;
    int speckle_count = 0;
    for (int ri = 0; ri < n_rois; ri++) {
        const auto& r = results[ri];
        int src = ri % cols, srr = ri / cols;
        // Only consider centre ROIs with real vibration (not edges)
        if (src >= 3 && src < cols - 3 && srr >= 3 && srr < rows - 3 &&
            r.dominant_freq_hz > 10.0 && r.dominant_freq_hz < 200.0 &&
            r.amplitude_px > 0.005 && r.amplitude_px < 2.0) {
            if (r.amplitude_px > vis_max_amp) {
                vis_max_amp = r.amplitude_px;
                speckle_dom_freq = r.dominant_freq_hz;
            }
            speckle_count++;
        }
    }
    if (vis_max_amp < 0.01) vis_max_amp = 0.01;
    std::cout << "  Speckle ROIs with vibration: " << speckle_count
              << ", dominant: " << speckle_dom_freq << "Hz, max amp: " << vis_max_amp << "px\n";

    cv::Mat overlay_frame;
    while (cap2.read(overlay_frame) && frame_idx2 < limit_overlay) {
        if (frame_idx2 % 100 == 0)
            std::cout << "  Overlay: frame " << frame_idx2 << "/" << limit_overlay << "\n";

        double t = frame_idx2 / fps;
        cv::Mat blended = overlay_frame.clone();

        // Draw displacement arrows on each centre ROI
        double arrow_scale = 50.0;  // amplify sub-pixel displacement for visibility
        int edge_margin = 3;

        if (frame_idx2 < dic.fft_frames_out) {
            for (int roi = 0; roi < dic.n_rois_out; roi++) {
                int rc = roi % cols, rr = roi / cols;
                if (rc < edge_margin || rc >= cols - edge_margin ||
                    rr < edge_margin || rr >= rows - edge_margin) continue;

                float dx = dic.frame_dx[roi * dic.fft_frames_out + frame_idx2];
                float dy = dic.frame_dy[roi * dic.fft_frames_out + frame_idx2];
                float mag = std::sqrt(dx * dx + dy * dy);

                if (mag < 0.005) continue;  // skip zero displacement

                // Arrow centre
                int cx = rc * roi_size + roi_size / 2;
                int cy = rr * roi_size + roi_size / 2;

                // Arrow tip (amplified for visibility)
                int tx = cx + static_cast<int>(dx * arrow_scale);
                int ty = cy + static_cast<int>(dy * arrow_scale);

                // Color by magnitude: green=small, yellow=medium, red=large
                double norm_mag = std::min(1.0, static_cast<double>(mag) / vis_max_amp);
                int r_val = static_cast<int>(255 * norm_mag);
                int g_val = static_cast<int>(255 * (1.0 - norm_mag * 0.5));
                int b_val = 0;
                cv::Scalar color(b_val, g_val, r_val);

                cv::arrowedLine(blended, cv::Point(cx, cy), cv::Point(tx, ty),
                                color, 1, cv::LINE_AA, 0, 0.3);

                // Small dot at ROI centre
                cv::circle(blended, cv::Point(cx, cy), 2, color, -1);
            }
        }

        // Add frequency labels only for centre-region vibration ROIs
        for (const auto& r : results) {
            int lrc = r.roi_x / roi_size, lrr = r.roi_y / roi_size;
            if (lrc < 3 || lrc >= cols - 3 || lrr < 3 || lrr >= rows - 3) continue;
            if (r.amplitude_px > vis_max_amp * 0.15 &&
                r.dominant_freq_hz > 10.0 && r.dominant_freq_hz < 200.0 &&
                r.amplitude_px < 2.0) {
                char label[32];
                snprintf(label, sizeof(label), "%.0fHz", r.dominant_freq_hz);
                cv::putText(blended, label,
                            cv::Point(r.roi_x + 2, r.roi_y + roi_size / 2 + 4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.3,
                            cv::Scalar(255, 255, 255), 1);
            }
        }

        // Info bar
        cv::rectangle(blended, cv::Rect(0, 0, width, 35), cv::Scalar(0, 0, 0), -1);
        char info[256];
        snprintf(info, sizeof(info),
                 "DIC Vibration | ROI: %dx%d | Rotor: %.1fHz | Amp: %.3fpx | t=%.3fs | Frame %d/%d",
                 roi_size, roi_size, speckle_dom_freq, vis_max_amp, t, frame_idx2, limit_overlay);
        cv::putText(blended, info, cv::Point(10, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 200), 1);

        overlay_writer.write(blended);
        frame_idx2++;
    }

    cap2.release();
    overlay_writer.release();

    std::cout << "Saved: " << output_dir << "/vibration_overlay.avi\n";

    return 0;
}
