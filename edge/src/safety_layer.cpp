// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/safety_layer.hpp"

#include <algorithm>
#include <array>

#include <spdlog/spdlog.h>

namespace semaforo {

namespace {

/// @brief Explicit, data-independent emergency ring (C-4).
///
/// Indexing this fixed table — never a data-dependent branch — is what makes the
/// emergency cycle provably free of starvation: every full lap serves BOTH the
/// NS and EW greens, each preceded by amber and an all-red clearance.
constexpr std::array<Phase, 6> kEmergencyRing = {
    Phase::ALL_RED,    // index 0: clearance on entry (from any prior phase)
    Phase::GREEN_NS,   // index 1
    Phase::YELLOW_NS,  // index 2
    Phase::ALL_RED,    // index 3: clearance before the cross green
    Phase::GREEN_EW,   // index 4
    Phase::YELLOW_EW,  // index 5  → wraps back to ALL_RED
};
constexpr std::size_t kRingSize = kEmergencyRing.size();

} // namespace

// ── Constructor / Destructor ────────────────────────────────────────────────

SafetyLayer::SafetyLayer() : SafetyLayer(Timings{}) {}

SafetyLayer::SafetyLayer(Timings timings)
    : t_(timings),
      state_(State::ALL_RED),
      applied_(Phase::ALL_RED),
      target_green_(Phase::GREEN_NS),
      target_valid_(false),
      phase_start_(Clock::now()),
      emergency_(false),
      ring_idx_(0) {
    // Boot in ALL_RED: the first requested green is served only after a full
    // clearance interval, so startup is fail-safe.
}

SafetyLayer::~SafetyLayer() = default;

// ── Time helper ─────────────────────────────────────────────────────────────

float SafetyLayer::elapsed_s(TimePoint now) const {
    return std::chrono::duration<float>(now - phase_start_).count();
}

float SafetyLayer::ring_duration(Phase p) const {
    if (is_green(p)) return t_.emergency_green;
    if (is_yellow(p)) return t_.yellow;
    return t_.all_red; // ALL_RED (and any unexpected phase) clears for all_red
}

// ── Public API ──────────────────────────────────────────────────────────────

ValidatedPhase SafetyLayer::validate(Phase desired_phase) {
    return validate(desired_phase, Clock::now());
}

ValidatedPhase SafetyLayer::validate(Phase desired_phase, TimePoint now) {
    std::lock_guard<std::mutex> lk(mtx_);
    return emergency_ ? step_emergency(now) : step_normal(desired_phase, now);
}

void SafetyLayer::set_emergency_mode(bool on) {
    set_emergency_mode(on, Clock::now());
}

void SafetyLayer::set_emergency_mode(bool on, TimePoint now) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (on == emergency_) return;

    emergency_ = on;
    if (on) {
        // Enter the ring at its ALL_RED clearance, regardless of the prior
        // phase, so any green-in-progress is safely cleared first.
        state_ = State::EMERGENCY;
        ring_idx_ = 0;
        applied_ = kEmergencyRing[0]; // ALL_RED
        phase_start_ = now;
        spdlog::warn("SafetyLayer: ENTERING EMERGENCY RING (deterministic fixed cycle)");
    } else {
        // Leave via a safe ALL_RED steady state; the next green will pass
        // through a fresh clearance before being served.
        state_ = State::ALL_RED;
        applied_ = Phase::ALL_RED;
        target_valid_ = false;
        phase_start_ = now;
        spdlog::info("SafetyLayer: EXITING EMERGENCY RING → ALL_RED");
    }
}

bool SafetyLayer::in_emergency() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return emergency_;
}

void SafetyLayer::reset() { reset(Clock::now()); }

void SafetyLayer::reset(TimePoint now) {
    std::lock_guard<std::mutex> lk(mtx_);
    state_ = State::ALL_RED;
    applied_ = Phase::ALL_RED;
    target_valid_ = false;
    emergency_ = false;
    ring_idx_ = 0;
    phase_start_ = now;
    spdlog::info("SafetyLayer: reset to ALL_RED steady state");
}

SafetyLayer::State SafetyLayer::get_state() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return state_;
}

Phase SafetyLayer::get_current_phase() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return applied_;
}

// ── Normal (AI-driven) interlock ────────────────────────────────────────────

void SafetyLayer::begin_yellow(Phase target_green, TimePoint now) {
    // Capture the amber of the *outgoing* green before we overwrite applied_.
    const Phase amber = yellow_for(applied_);
    target_green_ = target_green;
    target_valid_ = true;
    applied_ = amber;
    state_ = State::YELLOW;
    phase_start_ = now;
    spdlog::info("SafetyLayer: {} → YELLOW ({}), target green {}",
                 phase_to_string(amber == Phase::YELLOW_NS ? Phase::GREEN_NS
                                                           : Phase::GREEN_EW),
                 phase_to_string(amber), phase_to_string(target_green));
}

ValidatedPhase SafetyLayer::step_normal(Phase desired, TimePoint now) {
    const float el = elapsed_s(now);

    switch (state_) {
        case State::GREEN: {
            // Non-green desires are ignored (the AI may never command amber/red).
            const Phase want = is_green(desired) ? desired : applied_;
            const bool wants_change = (want != applied_);
            const bool min_ok = el >= t_.min_green;
            const bool max_exceeded = el >= t_.max_green;

            // ── A-1: max-green watchdog forces a switch no matter what ──────
            if (max_exceeded) {
                const Phase dest = wants_change ? want : opposite_green(applied_);
                spdlog::warn("SafetyLayer: MAX_GREEN ({:.0f}s) exceeded — forcing "
                             "switch {} → {}", t_.max_green,
                             phase_to_string(applied_), phase_to_string(dest));
                begin_yellow(dest, now);
                return make_result(desired, now, "max-green watchdog forced switch");
            }

            // ── C-3: an authorised change always starts with amber ──────────
            if (wants_change && min_ok) {
                begin_yellow(want, now);
                return make_result(desired, now, "transition requested → YELLOW");
            }
            if (wants_change && !min_ok) {
                return make_result(desired, now, "min-green not yet elapsed");
            }
            return make_result(desired, now, "steady green");
        }

        case State::YELLOW: {
            // The destination is LOCKED once a transition has begun. We must not
            // retarget here: doing so would let a constant request cancel a
            // max-green forced switch (defeating A-1) and could ping-pong the
            // target. The interlock still guarantees ALL_RED before any green.
            if (el >= t_.yellow) {
                applied_ = Phase::ALL_RED;
                state_ = State::ALL_RED;
                phase_start_ = now;
                spdlog::info("SafetyLayer: YELLOW → ALL_RED clearance");
                return make_result(desired, now, "yellow complete → ALL_RED");
            }
            return make_result(desired, now, "amber interval active");
        }

        case State::ALL_RED: {
            // Only accept a target when none is committed yet (boot ALL_RED). A
            // clearance that follows a committed transition keeps its locked
            // target so forced/normal switches always complete.
            if (!target_valid_ && is_green(desired)) {
                target_green_ = desired;
                target_valid_ = true;
            }
            if (el >= t_.all_red && target_valid_) {
                applied_ = target_green_;
                state_ = State::GREEN;
                phase_start_ = now;
                spdlog::info("SafetyLayer: ALL_RED → {} (clearance complete)",
                             phase_to_string(applied_));
                return make_result(desired, now, "clearance complete → green");
            }
            return make_result(desired, now,
                               target_valid_ ? "all-red clearance active"
                                             : "all-red steady (awaiting request)");
        }

        case State::EMERGENCY:
            // Should not happen (validate routes emergency separately); fall back.
            return step_emergency(now);
    }

    // Unreachable, but fail safe.
    applied_ = Phase::ALL_RED;
    state_ = State::ALL_RED;
    return make_result(desired, now, "fail-safe ALL_RED");
}

// ── Emergency ring (C-4) ────────────────────────────────────────────────────

ValidatedPhase SafetyLayer::step_emergency(TimePoint now) {
    const Phase current = kEmergencyRing[ring_idx_];
    const float el = elapsed_s(now);

    if (el >= ring_duration(current)) {
        ring_idx_ = (ring_idx_ + 1) % kRingSize;
        applied_ = kEmergencyRing[ring_idx_];
        phase_start_ = now;
        spdlog::info("SafetyLayer[EMERGENCY]: → {} (ring idx {})",
                     phase_to_string(applied_), ring_idx_);
    }

    ValidatedPhase r;
    r.phase = applied_;
    r.was_modified = true; // emergency always overrides the AI
    r.reason = "emergency fixed ring";
    r.remaining_s = std::max(0.0f, ring_duration(applied_) - elapsed_s(now));
    return r;
}

// ── Result builder ──────────────────────────────────────────────────────────

ValidatedPhase SafetyLayer::make_result(Phase desired, TimePoint now,
                                        std::string reason) const {
    ValidatedPhase r;
    r.phase = applied_;
    r.was_modified = (applied_ != desired);
    r.reason = std::move(reason);

    const float el = elapsed_s(now);
    switch (state_) {
        case State::GREEN:
            r.remaining_s = (el < t_.min_green) ? (t_.min_green - el)
                                                : std::max(0.0f, t_.max_green - el);
            break;
        case State::YELLOW:
            r.remaining_s = std::max(0.0f, t_.yellow - el);
            break;
        case State::ALL_RED:
            r.remaining_s = std::max(0.0f, t_.all_red - el);
            break;
        case State::EMERGENCY:
            r.remaining_s = std::max(0.0f, ring_duration(applied_) - el);
            break;
    }
    return r;
}

} // namespace semaforo
