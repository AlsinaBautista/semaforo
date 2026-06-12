// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <memory>
#include <string>

#include "semaforo/queue_estimator.hpp"

namespace semaforo {

/// @brief Traffic signal phase definitions for a standard 4-way intersection.
enum class Phase {
    ALL_RED,        ///< All directions red
    GREEN_NS,       ///< North-South through traffic green
    YELLOW_NS,      ///< North-South through traffic yellow
    GREEN_EW,       ///< East-West through traffic green
    YELLOW_EW,      ///< East-West through traffic yellow
    GREEN_NS_LEFT,  ///< North-South left-turn protected green
    GREEN_EW_LEFT,  ///< East-West left-turn protected green
};

/// @brief Convert a Phase enum value to a human-readable string.
/// @param phase  The phase to convert.
/// @return String representation (e.g., "GREEN_NS").
const char* phase_to_string(Phase phase);

/// @brief ONNX Runtime-backed policy engine for adaptive signal timing.
///
/// Uses a trained reinforcement learning model (exported to ONNX) to decide
/// the optimal traffic signal phase given the current queue state. The model
/// takes queue counts, densities, and wait times as input features and
/// outputs a phase selection.
///
/// Falls back to a simple round-robin policy when no model is loaded.
class PolicyEngine {
public:
    PolicyEngine();
    ~PolicyEngine();

    // Non-copyable, movable
    PolicyEngine(const PolicyEngine&) = delete;
    PolicyEngine& operator=(const PolicyEngine&) = delete;
    PolicyEngine(PolicyEngine&&) noexcept;
    PolicyEngine& operator=(PolicyEngine&&) noexcept;

    /// @brief Initialize the policy engine by loading an ONNX policy model.
    ///
    /// If loading fails, the engine falls back to a deterministic round-robin
    /// policy based on maximum queue density.
    ///
    /// @param policy_path  Filesystem path to the .onnx policy model.
    /// @return true if the model was loaded successfully, false if falling back.
    bool init(const std::string& policy_path);

    /// @brief Decide the next traffic signal phase given current queue state.
    ///
    /// Encodes the queue state into a feature vector, runs inference through
    /// the ONNX model, and returns the phase with the highest Q-value.
    /// If no model is loaded, selects the phase serving the lane with the
    /// highest density.
    ///
    /// @param state  Current aggregated queue state from QueueEstimator.
    /// @return The recommended Phase to transition to.
    Phase decide(const QueueState& state);

    /// @brief Check whether a trained policy model is loaded.
    /// @return true if using ONNX model, false if using fallback heuristic.
    bool is_model_loaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_; ///< PIMPL for ONNX Runtime internals

    /// @brief Fallback heuristic: choose phase for the lane with max density.
    /// @param state  Current queue state.
    /// @return Phase selected by heuristic.
    Phase fallback_decide(const QueueState& state);
};

} // namespace semaforo
