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
    // Re-asserting the already-active phase must NOT restart the phase timer:
    // the control loop re-applies the validated phase every tick, and the
    // sustained ~1 Hz flash derives its square wave from phase_start.
    if (phase == state_.current_phase) {
        return true;
    }

    // In a simulation, we just update internal state
    auto now = std::chrono::steady_clock::now();
    state_.current_phase = phase;
    state_.phase_start = now;

    // Reset all
    state_.north_south.red = false;
    state_.north_south.yellow = false;
    state_.north_south.green = false;
    state_.north_south.flashing = false;
    state_.east_west.red = false;
    state_.east_west.yellow = false;
    state_.east_west.green = false;
    state_.east_west.flashing = false;

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
        // Terminal flash patterns (F-1). At most ONE axis may flash yellow;
        // the crossing axis is always restrictive (flashing red). The pattern
        // set is closed — there is no phase that yellow-flashes both axes.
        case Phase::FLASH_YELLOW_NS:
            state_.north_south.yellow = true;
            state_.north_south.flashing = true;
            state_.east_west.red = true;
            state_.east_west.flashing = true;
            break;
        case Phase::FLASH_YELLOW_EW:
            state_.north_south.red = true;
            state_.north_south.flashing = true;
            state_.east_west.yellow = true;
            state_.east_west.flashing = true;
            break;
        case Phase::FLASH_ALL_RED:
            state_.north_south.red = true;
            state_.north_south.flashing = true;
            state_.east_west.red = true;
            state_.east_west.flashing = true;
            break;
    }

    spdlog::debug("SimulationSignalController: Phase set to {}", static_cast<int>(phase));
    return true;
}

Phase SimulationSignalController::get_current_phase() const {
    return state_.current_phase;
}

SignalState SimulationSignalController::get_signal_state() const {
    // Sustained flash: heads marked `flashing` alternate their lamps with a
    // 1 Hz square wave (500 ms lit / 500 ms dark). The `flashing` flag itself
    // stays true through the dark half-cycle so observers can tell a flashing
    // head from a dead one.
    SignalState s = state_;
    const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - s.phase_start);
    if (!flash_lamp_lit(since)) {
        if (s.north_south.flashing) {
            s.north_south.red = false;
            s.north_south.yellow = false;
            s.north_south.green = false;
        }
        if (s.east_west.flashing) {
            s.east_west.red = false;
            s.east_west.yellow = false;
            s.east_west.green = false;
        }
    }
    return s;
}

void SimulationSignalController::set_emergency_flash() {
    state_.current_phase = Phase::FLASH_ALL_RED;
    state_.phase_start = std::chrono::steady_clock::now();
    state_.north_south.red = true;
    state_.north_south.yellow = false;
    state_.north_south.green = false;
    state_.north_south.flashing = true;
    state_.east_west.red = true;
    state_.east_west.yellow = false;
    state_.east_west.green = false;
    state_.east_west.flashing = true;
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
        // ╔════════════════════════════════════════════════════════════════╗
        // ║ TODO(HARDWARE BACKSTOP — NO IMPLEMENTADO):                      ║
        // ║ flash por MMU/conflict-monitor independiente                    ║
        // ╚════════════════════════════════════════════════════════════════╝
        // El modo FLASH terminal de este módulo es COMANDADO POR SOFTWARE:
        // cubre únicamente las fallas detectables con el software vivo
        // (schedule corrupto del ring de emergencia, degradación del Brain).
        // NO cubre el caso de software colgado, muerto o segfaulteado.
        //
        // El backstop real para ese caso es HARDWARE: el MMU del gabinete
        // debe forzar el flash por sí mismo cuando el heartbeat GPIO
        // (semaforo/heartbeat.hpp, pin dead-man) deja de pulsar — relé de
        // conflicto titilando en rojo SIN participación alguna del software.
        //
        // Es una tarea de HARDWARE futura, a resolver al integrar la Jetson
        // con el gabinete real (este branch "gpio"). Hasta entonces, NO
        // asumir que el flash simulado de arriba reemplaza al MMU: un nodo
        // colgado hoy deja las luces en el último estado aplicado.
        //
        // TODO: Implement actual GPIO controller for Jetson hardware
        spdlog::warn("GPIO controller not yet implemented. Falling back to simulation.");
        return std::make_unique<SimulationSignalController>();
    } else {
        throw std::invalid_argument("Unknown controller type: " + type);
    }
}

} // namespace semaforo
