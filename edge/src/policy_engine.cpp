// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/policy_engine.hpp"

#include <algorithm>

#include <spdlog/spdlog.h>

#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

namespace semaforo {

// ── Phase utility ───────────────────────────────────────────────────────────

const char* phase_to_string(Phase phase) {
    switch (phase) {
        case Phase::ALL_RED:        return "ALL_RED";
        case Phase::GREEN_NS:       return "GREEN_NS";
        case Phase::YELLOW_NS:      return "YELLOW_NS";
        case Phase::GREEN_EW:       return "GREEN_EW";
        case Phase::YELLOW_EW:      return "YELLOW_EW";
        case Phase::GREEN_NS_LEFT:   return "GREEN_NS_LEFT";
        case Phase::GREEN_EW_LEFT:   return "GREEN_EW_LEFT";
        case Phase::FLASH_YELLOW_NS: return "FLASH_YELLOW_NS";
        case Phase::FLASH_YELLOW_EW: return "FLASH_YELLOW_EW";
        case Phase::FLASH_ALL_RED:   return "FLASH_ALL_RED";
        default:                     return "UNKNOWN";
    }
}

// ── PIMPL implementation ────────────────────────────────────────────────────

struct PolicyEngine::Impl {
#ifdef ONNXRUNTIME_AVAILABLE
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "semaforo_policy"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
#endif
    bool model_loaded = false;

    /// @brief Encode queue state into a feature vector for the policy model.
    ///
    /// Features per lane: [count, density, avg_wait_s]
    /// Total features: num_lanes * 3 + 2 (total_count, max_wait)
    std::vector<float> encode_state(const QueueState& state) {
        std::vector<float> features;
        features.reserve(state.lanes.size() * 3 + 2);

        for (const auto& lane : state.lanes) {
            features.push_back(static_cast<float>(lane.count));
            features.push_back(lane.density);
            features.push_back(lane.avg_wait_s);
        }

        features.push_back(static_cast<float>(state.total_count));
        features.push_back(state.max_wait_s);

        return features;
    }
};

// ── Constructor / Destructor ────────────────────────────────────────────────

PolicyEngine::PolicyEngine() : impl_(std::make_unique<Impl>()) {}
PolicyEngine::~PolicyEngine() = default;
PolicyEngine::PolicyEngine(PolicyEngine&&) noexcept = default;
PolicyEngine& PolicyEngine::operator=(PolicyEngine&&) noexcept = default;

// ── Public methods ──────────────────────────────────────────────────────────

bool PolicyEngine::init(const std::string& policy_path) {
#ifdef ONNXRUNTIME_AVAILABLE
    try {
        impl_->session_options.SetIntraOpNumThreads(1);
        impl_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, policy_path.c_str(), impl_->session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = impl_->session->GetInputNameAllocated(0, allocator);
        impl_->input_names.push_back(input_name.get());

        auto output_name = impl_->session->GetOutputNameAllocated(0, allocator);
        impl_->output_names.push_back(output_name.get());

        impl_->model_loaded = true;
        spdlog::info("PolicyEngine: model loaded from {}", policy_path);
        return true;

    } catch (const Ort::Exception& ex) {
        spdlog::warn("PolicyEngine: failed to load model: {}", ex.what());
        impl_->model_loaded = false;
        return false;
    }
#else
    spdlog::warn("PolicyEngine: ONNX Runtime unavailable. Path: {}. Using fallback.",
                 policy_path);
    impl_->model_loaded = false;
    return false;
#endif
}

Phase PolicyEngine::decide(const QueueState& state) {
#ifdef ONNXRUNTIME_AVAILABLE
    if (impl_->model_loaded && impl_->session) {
        try {
            auto features = impl_->encode_state(state);

            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);
            std::array<int64_t, 2> input_shape = {
                1, static_cast<int64_t>(features.size())};
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, features.data(), features.size(),
                input_shape.data(), input_shape.size());

            auto output_tensors = impl_->session->Run(
                Ort::RunOptions{nullptr},
                impl_->input_names.data(), &input_tensor, 1,
                impl_->output_names.data(), impl_->output_names.size());

            // Get Q-values from output tensor
            float* q_values = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            int num_actions = static_cast<int>(output_shape.back());

            // Select action with highest Q-value
            int best_action = 0;
            float best_q = q_values[0];
            for (int i = 1; i < num_actions && i < 4; ++i) {
                if (q_values[i] > best_q) {
                    best_q = q_values[i];
                    best_action = i;
                }
            }

            return static_cast<Phase>(best_action);

        } catch (const Ort::Exception& ex) {
            spdlog::error("PolicyEngine: inference error: {}", ex.what());
            return fallback_decide(state);
        }
    }
#endif

    return fallback_decide(state);
}

bool PolicyEngine::is_model_loaded() const {
    return impl_->model_loaded;
}

// ── Private methods ─────────────────────────────────────────────────────────

Phase PolicyEngine::fallback_decide(const QueueState& state) {
    // Simple heuristic: serve the direction with the highest density.
    // Map lane directions to phases:
    //   "NS"       → GREEN_NS
    //   "EW"       → GREEN_EW
    //   "NS_left"  → GREEN_NS_LEFT
    //   "EW_left"  → GREEN_EW_LEFT

    Phase best_phase = Phase::GREEN_NS;
    float best_density = -1.0f;

    for (const auto& lane : state.lanes) {
        // Sum density for each direction group
        // For simplicity, pick the single lane with highest density
        if (lane.density > best_density) {
            best_density = lane.density;

            // TODO: Use lane_config direction field once available
            // For now, just cycle through phases based on density
            if (lane.lane_id.find("NS") != std::string::npos) {
                if (lane.lane_id.find("left") != std::string::npos) {
                    best_phase = Phase::GREEN_NS_LEFT;
                } else {
                    best_phase = Phase::GREEN_NS;
                }
            } else if (lane.lane_id.find("EW") != std::string::npos) {
                if (lane.lane_id.find("left") != std::string::npos) {
                    best_phase = Phase::GREEN_EW_LEFT;
                } else {
                    best_phase = Phase::GREEN_EW;
                }
            }
        }
    }

    spdlog::debug("PolicyEngine: fallback → {} (density={:.2f})",
                  phase_to_string(best_phase), best_density);
    return best_phase;
}

} // namespace semaforo
