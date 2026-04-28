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
#include <opencv2/cudacodec.hpp>          // NVDEC hardware video decoder
#include <opencv2/core/cuda_stream_accessor.hpp>
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

// Find sub-pixel peak in inverse FFT result (per ROI), and scatter the result
// directly into the per-ROI time-series buffer at the current frame slot.
// Layout of all_dx / all_dy: [roi * fft_frames + frame_idx]
//
// Folding the scatter into this kernel removes the per-frame
// d_dx/d_dy intermediate plus its host roundtrip — same numerics, no
// CPU loop, no extra memcpy.
__global__ void findSubPixelPeakKernel(const float* corr, int roi_size,
                                        int n_rois, int frame_idx, int fft_frames,
                                        float* all_dx, float* all_dy) {
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

    // Scatter into time-series buffer at this frame's slot
    all_dx[roi_idx * fft_frames + frame_idx] = ix;
    all_dy[roi_idx * fft_frames + frame_idx] = iy;
}

// Apply Hanning window to 1D displacement time series before spectral FFT
__global__ void applyHanning1DKernel(float* data, int n_frames, int n_rois) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_rois * n_frames) return;

    int frame = idx % n_frames;
    float w = 0.5f * (1.0f - cosf(2.0f * M_PI * frame / (n_frames - 1)));
    data[idx] *= w;
}

// High-pass filter (two-pass): compute moving average into a temp buffer,
// then subtract it from the original signal. This removes slow drift
// (e.g. residual ~2Hz camera sway) so the vibration spectrum dominates.
// One thread per (roi, frame) pair — each computes one MA sample.
__global__ void movingAverageKernel(const float* __restrict__ in, float* out,
                                    int n_frames, int n_rois, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_rois * n_frames;
    if (idx >= total) return;

    int roi   = idx / n_frames;
    int frame = idx % n_frames;
    const float* ts = in + roi * n_frames;

    int lo = max(0, frame - window / 2);
    int hi = min(n_frames, frame + window / 2 + 1);
    float sum = 0.0f;
    for (int j = lo; j < hi; j++) sum += ts[j];
    out[idx] = sum / (hi - lo);
}

__global__ void subtractInPlaceKernel(float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] -= b[idx];
}

// Fused patch extraction + Hanning window.
// Reads from a CV_8U cv::cuda::GpuMat (with row step in bytes), converts to
// float, and multiplies by the precomputed 2D Hanning window — all in one
// kernel launch / one pass over the data.  Replaces the old
// extractPatchesKernel + applyHanningKernel pair.
//   grid  = (cols, rows)              one block per ROI
//   block = 256                        threads cooperate to copy roi_size² pixels
__global__ void extractAndWindowKernel(const unsigned char* __restrict__ gray,
                                       int gray_step,
                                       const float* __restrict__ hann,
                                       float* __restrict__ patches,
                                       int roi_size) {
    int rx = blockIdx.x;
    int ry = blockIdx.y;
    int roi_idx  = ry * gridDim.x + rx;
    int n_pixels = roi_size * roi_size;
    int px = rx * roi_size;
    int py = ry * roi_size;
    float* dst = patches + roi_idx * n_pixels;

    for (int idx = threadIdx.x; idx < n_pixels; idx += blockDim.x) {
        int y = idx / roi_size;
        int x = idx - y * roi_size;
        dst[idx] = float(gray[(py + y) * gray_step + (px + x)]) * hann[idx];
    }
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

    // Process stabilized video: extract vibration spectrum for each ROI
    std::vector<ROIResult> analyze(const std::string& video_path, int max_frames = 512) {
        // Pull frame count + resolution via cv::VideoCapture (cheap; doesn't
        // decode anything beyond the header).  Actual decoding happens via
        // NVDEC below.
        cv::VideoCapture meta_cap(video_path);
        if (!meta_cap.isOpened()) {
            std::cerr << "Cannot open: " << video_path << "\n";
            return {};
        }
        int width  = static_cast<int>(meta_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(meta_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        int total  = static_cast<int>(meta_cap.get(cv::CAP_PROP_FRAME_COUNT));
        meta_cap.release();
        int n_frames = std::min(total, max_frames);

        // NVDEC hardware decoder — frames decoded directly into device memory,
        // no CPU MJPEG decode, no CPU→GPU upload.
        cv::Ptr<cv::cudacodec::VideoReader> reader;
        try {
            reader = cv::cudacodec::createVideoReader(video_path);
        } catch (const cv::Exception& e) {
            std::cerr << "NVDEC unavailable (" << e.what() << ").  "
                      << "Build OpenCV with -DWITH_NVCUVID=ON.\n";
            return {};
        }

        // Use every available frame. cuFFT supports arbitrary sizes; it's
        // fastest when the size factors into small primes (2, 3, 5, 7) and
        // falls back to Bluestein's algorithm for other sizes — slower per
        // FFT but still fine here because the temporal 1D FFT runs only
        // twice (X and Y axes), not once per frame.
        int fft_frames = n_frames;

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

        // ── Set up GPU pipeline ──────────────────────────────────────────
        // Per-frame patch extraction now runs on GPU (extractPatchesKernel)
        // and the peak finder writes directly into d_disp_x / d_disp_y at
        // the current frame's slot.  No host roundtrip per frame.

        int roi_pixels = roi_size_ * roi_size_;
        int fft_w = roi_size_ / 2 + 1;
        int fft_complex_size = fft_w * roi_size_;
        int spec_size = fft_frames / 2 + 1;

        // Precompute 2D Hanning window for ROI patches
        std::vector<float> hann2d(roi_pixels);
        for (int y = 0; y < roi_size_; y++) {
            float wy = 0.5f * (1.0f - std::cos(2.0 * M_PI * y / (roi_size_ - 1)));
            for (int x = 0; x < roi_size_; x++) {
                float wx = 0.5f * (1.0f - std::cos(2.0 * M_PI * x / (roi_size_ - 1)));
                hann2d[y * roi_size_ + x] = wy * wx;
            }
        }

        // Device buffers
        float *d_ref_patches, *d_cur_patches, *d_hann, *d_corr;
        cufftComplex *d_fft_ref, *d_fft_cur, *d_cross;
        float *d_disp_x, *d_disp_y;
        cufftComplex *d_spec_x, *d_spec_y;

        CUDA_CHECK(cudaMalloc(&d_ref_patches, n_rois * roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cur_patches, n_rois * roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_hann,                roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_corr,        n_rois * roi_pixels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_fft_ref, n_rois * fft_complex_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_fft_cur, n_rois * fft_complex_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_cross,   n_rois * fft_complex_size * sizeof(cufftComplex)));

        // Time-series buffers — allocated once, written into per-frame.
        // cudaMemset 0 so frame 0's slot stays 0 (frame 0 is the reference,
        // never goes through the peak finder).
        CUDA_CHECK(cudaMalloc(&d_disp_x, n_rois * fft_frames * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_disp_y, n_rois * fft_frames * sizeof(float)));
        CUDA_CHECK(cudaMemset( d_disp_x, 0, n_rois * fft_frames * sizeof(float)));
        CUDA_CHECK(cudaMemset( d_disp_y, 0, n_rois * fft_frames * sizeof(float)));

        // Spectrum buffers (used after the per-frame loop)
        CUDA_CHECK(cudaMalloc(&d_spec_x, n_rois * spec_size * sizeof(cufftComplex)));
        CUDA_CHECK(cudaMalloc(&d_spec_y, n_rois * spec_size * sizeof(cufftComplex)));

        CUDA_CHECK(cudaMemcpy(d_hann, hann2d.data(), roi_pixels * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Create batched 2D FFT plans, bound to the compute stream so cuFFT
        // dispatches asynchronously alongside our custom kernels.
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

        // Two CUDA streams: one for NVDEC decode, one for compute.
        // While stream B does FFT/cross-power/IFFT/peak on frame N, stream A
        // decodes frame N+1 in parallel.  Effective per-frame cost ≈
        // max(decode, compute) instead of sum.
        cv::cuda::Stream s_decode, s_compute;
        cudaStream_t cs_compute = cv::cuda::StreamAccessor::getStream(s_compute);
        CUFFT_CHECK(cufftSetStream(plan_fwd, cs_compute));
        CUFFT_CHECK(cufftSetStream(plan_inv, cs_compute));

        int threads = 256;
        dim3 patch_grid(cols, rows);

        // Double-buffered input frames so decode of frame N+1 can overlap
        // with compute of frame N.
        cv::cuda::GpuMat gpu_frame[2], gpu_gray[2];

        // ── Reference frame (frame 0): synchronous, sets d_fft_ref ──────
        if (!reader->nextFrame(gpu_frame[0], s_decode)) {
            std::cerr << "Cannot decode first frame\n";
            return {};
        }
        s_decode.waitForCompletion();

        cv::cuda::cvtColor(gpu_frame[0], gpu_gray[0], cv::COLOR_BGRA2GRAY,
                           0, s_compute);
        extractAndWindowKernel<<<patch_grid, threads, 0, cs_compute>>>(
            gpu_gray[0].ptr<unsigned char>(), int(gpu_gray[0].step),
            d_hann, d_ref_patches, roi_size_);
        CUFFT_CHECK(cufftExecR2C(plan_fwd, d_ref_patches, d_fft_ref));
        s_compute.waitForCompletion();

        // Pre-decode frame 1 to prime the pipeline
        if (fft_frames > 1) {
            reader->nextFrame(gpu_frame[1], s_decode);
        }

        for (int frame_idx = 1; frame_idx < fft_frames; frame_idx++) {
            if (frame_idx % 100 == 0)
                std::cout << "  Phase correlation: frame " << frame_idx
                          << "/" << fft_frames << "\n";

            int cur_buf = frame_idx & 1;        // current compute buffer
            int nxt_buf = 1 - cur_buf;          // next decode buffer

            // Wait for current frame's decode to finish (was started last iter)
            s_decode.waitForCompletion();

            // Start decoding the next frame in parallel with compute
            if (frame_idx + 1 < fft_frames) {
                if (!reader->nextFrame(gpu_frame[nxt_buf], s_decode)) {
                    // Stream ended early — drop the rest of the loop
                    fft_frames = frame_idx + 1;
                }
            }

            // Compute pipeline on s_compute (sequential within stream)
            cv::cuda::cvtColor(gpu_frame[cur_buf], gpu_gray[cur_buf],
                               cv::COLOR_BGRA2GRAY, 0, s_compute);
            extractAndWindowKernel<<<patch_grid, threads, 0, cs_compute>>>(
                gpu_gray[cur_buf].ptr<unsigned char>(),
                int(gpu_gray[cur_buf].step),
                d_hann, d_cur_patches, roi_size_);
            CUFFT_CHECK(cufftExecR2C(plan_fwd, d_cur_patches, d_fft_cur));

            int total_complex = n_rois * fft_complex_size;
            crossPowerSpectrumKernel<<<(total_complex + threads - 1) / threads,
                                       threads, 0, cs_compute>>>(
                d_fft_ref, d_fft_cur, d_cross, total_complex);

            CUFFT_CHECK(cufftExecC2R(plan_inv, d_cross, d_corr));

            // Find sub-pixel peak per ROI and scatter into the time-series
            // buffers at this frame's slot.  No download per frame.
            findSubPixelPeakKernel<<<(n_rois + threads - 1) / threads,
                                     threads, 0, cs_compute>>>(
                d_corr, roi_size_, n_rois, frame_idx, fft_frames,
                d_disp_x, d_disp_y);

            // Compute must finish before this iter's gpu_gray buffer can be
            // overwritten by the next decode (two iterations later, when
            // cur_buf comes around again).  In practice the cuFFT plans
            // serialise on s_compute so only the final wait is needed.
        }

        // Drain both streams before moving to the spectral stage
        s_decode.waitForCompletion();
        s_compute.waitForCompletion();

        cufftDestroy(plan_fwd);
        cufftDestroy(plan_inv);
        reader.release();

        std::cout << "\n  Spectral analysis (cuFFT batch)...\n";

        // d_disp_x / d_disp_y already populated on device from the per-frame
        // peak kernel — no host upload needed here.

        // ── High-pass filter on GPU: subtract moving average ─────────────
        // The ~2Hz camera drift dominates the raw displacement signal.
        // Subtract a moving average (window = fps/5 → 5Hz cutoff) so the
        // structural vibration band dominates the FFT.
        int hp_window = static_cast<int>(fps_ / 5.0);
        if (hp_window < 3) hp_window = 3;
        std::cout << "  High-pass filter (GPU): window=" << hp_window
                  << " frames (>" << fps_ / hp_window << " Hz pass)\n";

        float* d_ma;
        int total_ts = n_rois * fft_frames;
        CUDA_CHECK(cudaMalloc(&d_ma, total_ts * sizeof(float)));

        movingAverageKernel<<<(total_ts + threads - 1) / threads, threads>>>(
            d_disp_x, d_ma, fft_frames, n_rois, hp_window);
        subtractInPlaceKernel<<<(total_ts + threads - 1) / threads, threads>>>(
            d_disp_x, d_ma, total_ts);

        movingAverageKernel<<<(total_ts + threads - 1) / threads, threads>>>(
            d_disp_y, d_ma, fft_frames, n_rois, hp_window);
        subtractInPlaceKernel<<<(total_ts + threads - 1) / threads, threads>>>(
            d_disp_y, d_ma, total_ts);

        cudaFree(d_ma);

        // Apply Hanning window to time series
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

            // Frame 0 is the reference (no peak finding done for it), so its
            // slot is 0 in the time-series buffers — same as the previous
            // host-side all_dx[roi*fft_frames] and all_dy[roi*fft_frames].
            results[roi] = {
                rx, ry,
                is_border ? 0.0 : max_bin * freq_resolution,
                is_border ? 0.0 : max_mag / fft_frames,
                std::atan2(cy_val, cx),
                0.0,    // mean_dx (was always 0 — frame-0 slot)
                0.0     // mean_dy
            };
        }

        // Cleanup
        cudaFree(d_ref_patches); cudaFree(d_cur_patches);
        cudaFree(d_hann); cudaFree(d_corr);
        cudaFree(d_fft_ref); cudaFree(d_fft_cur); cudaFree(d_cross);
        cudaFree(d_disp_x); cudaFree(d_disp_y);
        cudaFree(d_spec_x); cudaFree(d_spec_y);

        return results;
    }

private:
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
    // Note: actual frequency resolution depends on the number of frames
    // actually used (printed in the DICAnalyzer header), which can be
    // smaller than max_frames if the video is shorter.

    return 0;
}
