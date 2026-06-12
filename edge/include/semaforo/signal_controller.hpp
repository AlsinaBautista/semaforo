// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "semaforo/policy_engine.hpp"

namespace semaforo {

/// @brief Physical signal head state for a single direction.
struct SignalHead {
    bool red    = false;  ///< Red lamp state
    bool yellow = false;  ///< Yellow/amber lamp state
    bool green  = false;  ///< Green lamp state
    bool left_arrow = false; ///< Left-turn arrow state
    bool flashing   = false; ///< Lamps alternate at ~1 Hz (terminal flash mode)
};

/// @brief Complete signal state for all directions at the intersection.
struct SignalState {
    SignalHead north_south;  ///< North-South signal head
    SignalHead east_west;    ///< East-West signal head
    Phase current_phase;     ///< Active phase enum
    std::chrono::steady_clock::time_point phase_start; ///< When current phase began
};

/// @brief Abstract interface for controlling traffic signal hardware or simulation.
///
/// Concrete implementations drive either real GPIO pins (for deployment on
/// embedded hardware) or a simulation backend (for testing and development).
/// All implementations must guarantee that conflicting greens cannot be set.
class SignalController {
public:
    virtual ~SignalController() = default;

    /// @brief Set the intersection to the specified phase.
    ///
    /// Implementations must translate the Phase enum into the appropriate
    /// signal head states and activate the corresponding hardware outputs
    /// or simulation state.
    ///
    /// @param phase  The Phase to activate.
    /// @return true if the phase was successfully applied.
    virtual bool set_phase(Phase phase) = 0;

    /// @brief Get the currently active phase.
    /// @return The current Phase.
    virtual Phase get_current_phase() const = 0;

    /// @brief Get the full signal state including individual lamp states.
    /// @return Current SignalState for all directions.
    virtual SignalState get_signal_state() const = 0;

    /// @brief Set all signals to flashing red (emergency/fault mode).
    ///
    /// This is the safe fallback state when the system encounters an error.
    /// Implementations must ensure this always succeeds.
    virtual void set_emergency_flash() = 0;

    /// @brief Check if the controller hardware is healthy and responsive.
    /// @return true if the controller is operational.
    virtual bool is_healthy() const = 0;

    /// @brief Get the elapsed time since the current phase began.
    /// @return Duration since phase start.
    virtual std::chrono::duration<float> elapsed_since_phase_start() const = 0;
};

/// @brief Simulation-mode signal controller for development and testing.
///
/// Maintains signal state in-memory without any hardware interaction.
/// Useful for desktop development and unit/integration testing.
class SimulationSignalController : public SignalController {
public:
    SimulationSignalController();
    ~SimulationSignalController() override;

    bool set_phase(Phase phase) override;
    Phase get_current_phase() const override;
    SignalState get_signal_state() const override;
    void set_emergency_flash() override;
    bool is_healthy() const override;
    std::chrono::duration<float> elapsed_since_phase_start() const override;

    /// @brief Half-period of the sustained flash (lamp on 500 ms, off 500 ms
    /// → 60 flashes/min, inside the 50–60 flashes/min field standard).
    static constexpr std::chrono::milliseconds FLASH_HALF_PERIOD{500};

    /// @brief Pure 1 Hz square wave for flashing heads: whether a flashing
    /// lamp is lit @p since_phase_start into the flash phase. Exposed (and
    /// time-parametrised) so the blink cadence is deterministically testable.
    static bool flash_lamp_lit(std::chrono::milliseconds since_phase_start) {
        return (since_phase_start / FLASH_HALF_PERIOD) % 2 == 0;
    }

private:
    SignalState state_;
};

/// @brief Factory function to create a SignalController instance.
/// @param type  Controller type: "simulation" or "gpio".
/// @return Unique pointer to the created controller.
std::unique_ptr<SignalController> create_signal_controller(const std::string& type);

} // namespace semaforo
