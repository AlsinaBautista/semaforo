// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/signal_controller.hpp"
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace semaforo {

// ── SimulationSignalController Implementation ───────────────────────────────

SimulationSignalController::SimulationSignalController() {
    state_.current_phase = Phase::ALL_RED;
    state_.phase_start = std::chrono::steady_clock::now();
    set_emergency_flash();
}

SimulationSignalController::~SimulationSignalController() = default;

bool SimulationSignalController::set_phase(Phase phase) {
    // In a simulation, we just update internal state
    auto now = std::chrono::steady_clock::now();
    state_.current_phase = phase;
    state_.phase_start = now;

    // Reset all
    state_.north_south.red = false;
    state_.north_south.yellow = false;
    state_.north_south.green = false;
    state_.east_west.red = false;
    state_.east_west.yellow = false;
    state_.east_west.green = false;

    switch (phase) {
        case Phase::GREEN_NS:
            state_.north_south.green = true;
            state_.east_west.red = true;
            break;
        case Phase::YELLOW_NS:
            state_.north_south.yellow = true;
            state_.east_west.red = true;
            break;
        case Phase::GREEN_EW:
            state_.north_south.red = true;
            state_.east_west.green = true;
            break;
        case Phase::YELLOW_EW:
            state_.north_south.red = true;
            state_.east_west.yellow = true;
            break;
        case Phase::GREEN_NS_LEFT:
            state_.north_south.left_arrow = true;
            state_.east_west.red = true;
            break;
        case Phase::GREEN_EW_LEFT:
            state_.north_south.red = true;
            state_.east_west.left_arrow = true;
            break;
        case Phase::ALL_RED:
            state_.north_south.red = true;
            state_.east_west.red = true;
            break;
    }

    spdlog::debug("SimulationSignalController: Phase set to {}", static_cast<int>(phase));
    return true;
}

Phase SimulationSignalController::get_current_phase() const {
    return state_.current_phase;
}

SignalState SimulationSignalController::get_signal_state() const {
    return state_;
}

void SimulationSignalController::set_emergency_flash() {
    state_.current_phase = Phase::ALL_RED;
    state_.phase_start = std::chrono::steady_clock::now();
    state_.north_south.red = true;
    state_.north_south.yellow = false;
    state_.north_south.green = false;
    state_.east_west.red = true;
    state_.east_west.yellow = false;
    state_.east_west.green = false;
    spdlog::warn("SimulationSignalController: EMERGENCY FLASH ACTIVATED");
}

bool SimulationSignalController::is_healthy() const {
    return true; // Simulation is always healthy
}

std::chrono::duration<float> SimulationSignalController::elapsed_since_phase_start() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<float>>(now - state_.phase_start);
}

// ── Factory Implementation ──────────────────────────────────────────────────

std::unique_ptr<SignalController> create_signal_controller(const std::string& type) {
    if (type == "simulation") {
        return std::make_unique<SimulationSignalController>();
    } else if (type == "gpio") {
        // TODO: Implement actual GPIO controller for Jetson hardware
        spdlog::warn("GPIO controller not yet implemented. Falling back to simulation.");
        return std::make_unique<SimulationSignalController>();
    } else {
        throw std::invalid_argument("Unknown controller type: " + type);
    }
}

} // namespace semaforo
