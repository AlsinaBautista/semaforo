// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>

#include "semaforo/policy_engine.hpp"

namespace semaforo {

/// @brief Which axis is the corridor's principal (major) approach.
///
/// Per-intersection configuration consumed by the SafetyLayer's terminal
/// FLASH mode: the major approach shows flashing yellow (yield), the crossing
/// minor approach shows flashing red (stop-and-go). ``UNKNOWN`` (missing or
/// unparsable configuration) is always safe: FLASH degrades to all-red flash.
enum class MajorApproach {
    NS,       ///< North-South is the corridor's main axis.
    EW,       ///< East-West is the corridor's main axis.
    UNKNOWN,  ///< Not configured / ambiguous → all-red flash fallback.
};

/// @brief Parse a config string ("NS"/"EW", case-insensitive) into a
/// MajorApproach. Anything else — empty, garbage, ambiguous — maps to
/// ``UNKNOWN`` so a bad config can only ever make FLASH *more* restrictive.
MajorApproach major_approach_from_string(const std::string& s);

/// @brief Result of safety validation: the phase to physically apply *now*.
struct ValidatedPhase {
    Phase phase;          ///< The phase to actually drive on the signal heads.
    bool was_modified;    ///< True if the applied phase differs from what was requested.
    std::string reason;   ///< Human-readable explanation of any modification.
    float remaining_s;    ///< Seconds remaining in the current timed sub-phase.
};

/// @brief Deterministic, self-timed safety interlock for a signalised intersection.
///
/// This layer is the **single authoritative source of truth** for what the
/// signal heads show.  Every phase the hardware ever displays passes through
/// here, and the class owns *all* timing internally from a monotonic clock, so
/// no software bug, malicious command, or AI mistake can create an unsafe state.
///
/// It guarantees three safety invariants by construction (see the property-based
/// test suite for the proofs):
///
///  * **C-3 — Conflict clearance.** A transition between any two *different*
///    greens is physically mediated by ``YELLOW_<origin>`` (amber) followed by
///    ``ALL_RED`` (clearance), each held for at least its minimum duration. A
///    green can NEVER change directly into a conflicting/cross green.
///
///  * **C-4 — Anti-starvation emergency ring.** In emergency/fallback mode the
///    layer runs an explicit, data-independent fixed ring
///    (``ALL_RED → GREEN_NS → YELLOW_NS → ALL_RED → GREEN_EW → YELLOW_EW → …``)
///    that visits every approach every cycle. It can never stall on one phase.
///
///  * **A-1 — Max-green watchdog.** If a green is held past ``max_green`` —
///    whether because the AI keeps requesting it or stops talking — the layer
///    *forces* a safe transition to the opposing green, independent of the AI,
///    so the intersection can never freeze.
///
///  * **F-1 — Terminal FLASH on corrupt schedule.** If the emergency ring's
///    own timing schedule is untrustworthy (E-5: a green or amber that can
///    never advance), the layer escalates GREEN/YELLOW → brief ALL_RED
///    clearance → terminal FLASH: the configured major approach flashes
///    yellow, the crossing approach flashes red; with no/invalid major
///    approach, everything flashes red. Two crossing approaches can NEVER
///    both flash yellow — the flash pattern set is closed (three values) and
///    statically proven single-yellow. FLASH is terminal: only ``reset()``
///    (manual intervention) leaves it.
///
/// The clock is injectable (the ``now`` overloads) so the entire state machine
/// is deterministically testable without sleeping.
///
/// Thread-safety: all public methods are synchronised by an internal mutex, so
/// the control loop and the MQTT callback thread may both drive it safely.
class SafetyLayer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// @name Default safety timing constants (seconds).
    /// @{
    static constexpr float MIN_GREEN_S       = 5.0f;   ///< Minimum green duration.
    static constexpr float YELLOW_S          = 3.0f;   ///< Amber interval.
    static constexpr float ALL_RED_S         = 2.0f;   ///< All-red clearance.
    static constexpr float MAX_GREEN_S       = 60.0f;  ///< Max green before forced switch.
    static constexpr float EMERGENCY_GREEN_S = 20.0f;  ///< Green duration in the emergency ring.
    /// @}

    /// @brief Configurable timing budget (defaults to the constants above).
    struct Timings {
        float min_green       = MIN_GREEN_S;
        float yellow          = YELLOW_S;
        float all_red         = ALL_RED_S;
        float max_green       = MAX_GREEN_S;
        float emergency_green = EMERGENCY_GREEN_S;
    };

    /// @brief Internal sub-phase of the interlock state machine.
    enum class State {
        GREEN,            ///< A green is displayed (min/max-green apply).
        YELLOW,           ///< Amber of the outgoing green is displayed.
        ALL_RED,          ///< All-red clearance is displayed.
        EMERGENCY,        ///< Deterministic fixed ring is running.
        FLASH_CLEARANCE,  ///< Brief ALL_RED clearing the box before FLASH.
        FLASH,            ///< Terminal flash (F-1). Only reset() leaves it.
    };

    SafetyLayer();
    explicit SafetyLayer(Timings timings);
    SafetyLayer(Timings timings, MajorApproach major);
    ~SafetyLayer();

    // ── Main entry point ────────────────────────────────────────────────────

    /// @brief Validate the AI's desired phase and return the phase to apply now.
    ///
    /// @param desired_phase The green the policy/AI wants to serve. Non-green
    ///        requests are treated as "no change" (the layer never lets the AI
    ///        directly command yellow/all-red).
    /// @return The phase the caller MUST drive on the heads this tick.
    ValidatedPhase validate(Phase desired_phase);

    /// @brief Deterministic overload with an injected clock (for testing).
    /// @param desired_phase The desired green (see above).
    /// @param now           The reference time for this evaluation.
    ValidatedPhase validate(Phase desired_phase, TimePoint now);

    // ── Emergency / fallback control ────────────────────────────────────────

    /// @brief Enable/disable the deterministic anti-starvation emergency ring.
    void set_emergency_mode(bool on);

    /// @brief Emergency toggle with an injected clock (for testing).
    void set_emergency_mode(bool on, TimePoint now);

    /// @brief Whether the emergency ring is currently engaged.
    bool in_emergency() const;

    // ── Lifecycle / introspection ───────────────────────────────────────────

    /// @brief Reset to a safe steady ALL_RED state (clears any transition).
    void reset();
    void reset(TimePoint now);

    /// @brief Current interlock sub-phase.
    State get_state() const;

    /// @brief Phase currently being displayed on the heads.
    Phase get_current_phase() const;

    // ── Pure phase-classification helpers (exposed for tests) ───────────────

    /// @brief True if @p p is any green movement.
    static bool is_green(Phase p) {
        return p == Phase::GREEN_NS || p == Phase::GREEN_EW ||
               p == Phase::GREEN_NS_LEFT || p == Phase::GREEN_EW_LEFT;
    }
    /// @brief True if @p p is any amber movement.
    static bool is_yellow(Phase p) {
        return p == Phase::YELLOW_NS || p == Phase::YELLOW_EW;
    }
    /// @brief True if @p p belongs to the North-South axis.
    static bool is_ns(Phase p) {
        return p == Phase::GREEN_NS || p == Phase::GREEN_NS_LEFT ||
               p == Phase::YELLOW_NS;
    }
    /// @brief True if @p p belongs to the East-West axis.
    static bool is_ew(Phase p) {
        return p == Phase::GREEN_EW || p == Phase::GREEN_EW_LEFT ||
               p == Phase::YELLOW_EW;
    }
    /// @brief The amber that must follow @p green before clearance.
    static Phase yellow_for(Phase green) {
        return is_ns(green) ? Phase::YELLOW_NS : Phase::YELLOW_EW;
    }
    /// @brief The opposing-axis through green (for the max-green forced switch).
    static Phase opposite_green(Phase green) {
        return is_ns(green) ? Phase::GREEN_EW : Phase::GREEN_NS;
    }
    /// @brief True if @p p is any terminal flash pattern.
    static bool is_flash(Phase p) {
        return p == Phase::FLASH_YELLOW_NS || p == Phase::FLASH_YELLOW_EW ||
               p == Phase::FLASH_ALL_RED;
    }

    /// @brief F-1 guard: the flash pattern for a major approach (pure).
    ///
    /// This is the ONLY producer of flash phases. The codomain is the closed
    /// three-pattern set, each of which yellow-flashes at most one axis, so
    /// two crossing approaches in simultaneous flashing yellow is not a
    /// representable output (statically asserted in safety_layer.cpp). Any
    /// value outside the NS/EW enumerators — UNKNOWN, or a corrupted enum —
    /// falls back to the most restrictive pattern, all-red flash.
    static Phase flash_phase_for(MajorApproach m) {
        switch (m) {
            case MajorApproach::NS: return Phase::FLASH_YELLOW_NS;
            case MajorApproach::EW: return Phase::FLASH_YELLOW_EW;
            case MajorApproach::UNKNOWN: return Phase::FLASH_ALL_RED;
        }
        return Phase::FLASH_ALL_RED; // out-of-range cast → restrictive fallback
    }

private:
    // All private helpers assume the caller already holds mtx_.
    ValidatedPhase step_normal(Phase desired, TimePoint now);
    ValidatedPhase step_emergency(TimePoint now);
    ValidatedPhase step_flash(TimePoint now);
    void enter_flash_clearance(Phase stuck_phase, TimePoint now,
                               const char* context);
    void begin_yellow(Phase target_green, TimePoint now);
    float elapsed_s(TimePoint now) const;
    float ring_duration(Phase p) const;
    float flash_clearance_s() const;
    ValidatedPhase make_result(Phase desired, TimePoint now,
                               std::string reason) const;

    Timings t_;
    MajorApproach major_;  ///< Corridor main axis for the FLASH pattern.
    mutable std::mutex mtx_;

    State state_;
    Phase applied_;        ///< Phase currently shown on the heads.
    Phase target_green_;   ///< Green we are transitioning toward.
    bool target_valid_;    ///< Whether a transition target has been committed.
    TimePoint phase_start_;///< When the current applied phase began.

    bool emergency_;       ///< Emergency ring engaged.
    std::size_t ring_idx_; ///< Index into the emergency ring.
    Phase stuck_phase_;    ///< Cause of FLASH: the phase the watchdog caught frozen.
};

} // namespace semaforo
