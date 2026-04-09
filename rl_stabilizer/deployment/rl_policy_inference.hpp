// ─────────────────────────────────────────────────────────────────────────────
//  RLStabilizer — drop-in replacement for MotionKalman using ONNX Runtime
//
//  Same interface: State update(...) and State predictOnly()
//  Loads the ONNX model exported by export_onnx.py.
//
//  Build: link against onnxruntime
//    -I/path/to/onnxruntime/include -lonnxruntime
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <onnxruntime_cxx_api.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

class RLStabilizer {
public:
    struct State { double tx, ty, theta_deg, scale; };

    static constexpr int HISTORY_LEN = 8;
    static constexpr int OBS_DIM     = 6 + 8 * HISTORY_LEN;  // 70

    // Action scaling (must match StabilizerEnv)
    static constexpr double TX_MAX       = 25.0;
    static constexpr double TY_MAX       = 25.0;
    static constexpr double THETA_MAX    = 0.6;
    static constexpr double SCALE_HALF   = 0.02;
    static constexpr double NORM_T       = 25.0;
    static constexpr double NORM_R       = 0.6;
    static constexpr double INLIER_SAT   = 400.0;

    explicit RLStabilizer(const std::string& onnx_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "RLStabilizer"),
          hist_idx_(0), initialized_(false)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = std::make_unique<Ort::Session>(
            env_, onnx_path.c_str(), opts);

        raw_hist_.fill({});
        act_hist_.fill({});
        prev_action_ = {0.0, 0.0, 0.0, 0.0};
    }

    State update(double tx_m, double ty_m, double theta_m,
                 double scale_m, int inliers) {
        float quality = static_cast<float>(
            std::min(1.0, static_cast<double>(inliers) / INLIER_SAT));
        float valid   = 1.0f;

        auto obs = buildObs(tx_m, ty_m, theta_m, scale_m, quality, valid);
        auto action = runInference(obs);

        // Rescale from [-1,1] to physical units
        State s;
        s.tx        = action[0] * TX_MAX;
        s.ty        = action[1] * TY_MAX;
        s.theta_deg = action[2] * THETA_MAX;
        s.scale     = 1.0 + action[3] * SCALE_HALF;

        // Update history
        pushHistory(tx_m, ty_m, theta_m, scale_m,
                    s.tx, s.ty, s.theta_deg, s.scale);

        prev_action_ = {s.tx, s.ty, s.theta_deg, s.scale};
        initialized_ = true;
        return s;
    }

    State predictOnly() {
        if (!initialized_) return {0.0, 0.0, 0.0, 1.0};

        // Feed zero measurement with valid=0
        auto obs = buildObs(0, 0, 0, 1.0, 0.0f, 0.0f);
        auto action = runInference(obs);

        State s;
        s.tx        = action[0] * TX_MAX;
        s.ty        = action[1] * TY_MAX;
        s.theta_deg = action[2] * THETA_MAX;
        s.scale     = 1.0 + action[3] * SCALE_HALF;

        pushHistory(0, 0, 0, 1.0, s.tx, s.ty, s.theta_deg, s.scale);
        prev_action_ = {s.tx, s.ty, s.theta_deg, s.scale};
        return s;
    }

    bool isInitialized() const { return initialized_; }

private:
    std::vector<float> buildObs(double tx_m, double ty_m, double theta_m,
                                double scale_m, float quality, float valid) {
        std::vector<float> obs(OBS_DIM, 0.0f);

        // Current measurement (normalized)
        obs[0] = static_cast<float>(tx_m / NORM_T);
        obs[1] = static_cast<float>(ty_m / NORM_T);
        obs[2] = static_cast<float>(theta_m / NORM_R);
        obs[3] = static_cast<float>(scale_m - 1.0);
        obs[4] = quality;
        obs[5] = valid;

        // History: oldest first
        int idx = hist_idx_ % HISTORY_LEN;
        for (int i = 0; i < HISTORY_LEN; ++i) {
            int hi = (idx + i) % HISTORY_LEN;
            int base_raw = 6 + i * 4;
            int base_act = 6 + HISTORY_LEN * 4 + i * 4;
            for (int j = 0; j < 4; ++j) {
                obs[base_raw + j] = raw_hist_[hi][j];
                obs[base_act + j] = act_hist_[hi][j];
            }
        }
        return obs;
    }

    std::array<float, 4> runInference(const std::vector<float>& obs) {
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

        const float* out_data = outputs[0].GetTensorData<float>();
        return {out_data[0], out_data[1], out_data[2], out_data[3]};
    }

    void pushHistory(double tx_m, double ty_m, double theta_m, double scale_m,
                     double act_tx, double act_ty, double act_theta, double act_scale) {
        int idx = hist_idx_ % HISTORY_LEN;
        raw_hist_[idx] = {
            static_cast<float>(tx_m / NORM_T),
            static_cast<float>(ty_m / NORM_T),
            static_cast<float>(theta_m / NORM_R),
            static_cast<float>(scale_m - 1.0)
        };
        act_hist_[idx] = {
            static_cast<float>(act_tx / NORM_T),
            static_cast<float>(act_ty / NORM_T),
            static_cast<float>(act_theta / NORM_R),
            static_cast<float>(act_scale - 1.0)
        };
        hist_idx_++;
    }

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;

    std::array<std::array<float, 4>, HISTORY_LEN> raw_hist_;
    std::array<std::array<float, 4>, HISTORY_LEN> act_hist_;
    int hist_idx_;
    std::array<double, 4> prev_action_;
    bool initialized_;
};
