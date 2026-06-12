// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/safety_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

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

/// @brief E-5: sanity timeout for an emergency-ring green whose *schedule* is
/// untrustworthy (ms). This is NOT the ring's normal green duration.
///
/// The ring's scheduled green time (``Timings::emergency_green``) is runtime
/// configuration; if it is corrupt — NaN, infinity, negative, or above
/// ``MAX_GREEN_S`` — the ring's normal ``elapsed >= schedule`` advance can
/// never fire and a green would freeze indefinitely, in EMERGENCY mode, the
/// one state that must never stall.
///
/// When this fires / when it does NOT:
///  * Fires ONLY while the current green's schedule fails the trust check in
///    ``step_emergency`` (``isfinite && >= 0 && <= MAX_GREEN_S``) AND the
///    green has been held past this timeout. It then forces ALL_RED.
///  * NEVER fires for a sane schedule: a healthy green (e.g. the default
///    20 s ``EMERGENCY_GREEN_S``) runs to completion through the normal
///    advance branch, which is evaluated first and unconditionally.
///
/// INVARIANT (E-5 worst-case green bound): time-in-green inside the
/// emergency ring is bounded by
///     max(MAX_GREEN_S, kEmergencyScheduleSanityTimeoutMs / 1000)
/// (+ one validate() polling tick) — trusted schedules are capped at
/// MAX_GREEN_S by the trust check itself, untrusted ones by this timeout.
/// The static_assert below pins the timeout under MAX_GREEN_S, so the bound
/// collapses to MAX_GREEN_S.
static constexpr std::uint32_t kEmergencyScheduleSanityTimeoutMs = 8000;

static_assert(static_cast<float>(kEmergencyScheduleSanityTimeoutMs) <=
                  SafetyLayer::MAX_GREEN_S * 1000.0f,
              "E-5 invariant: the emergency-ring sanity timeout must not "
              "exceed MAX_GREEN_S, so the worst-case time-in-green in the "
              "ring is bounded by MAX_GREEN_S for both trusted and untrusted "
              "schedules.");

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
    const float sched = ring_duration(current);

    // A schedule is only trusted when it is a sane, finite duration bounded by
    // the system-wide green ceiling. A corrupt value (NaN/inf/absurd) makes the
    // `el >= sched` advance below unreachable, which would freeze the ring.
    const bool sched_trusted =
        std::isfinite(sched) && sched >= 0.0f && sched <= MAX_GREEN_S;

    if (el >= sched) {
        ring_idx_ = (ring_idx_ + 1) % kRingSize;
        applied_ = kEmergencyRing[ring_idx_];
        phase_start_ = now;
        spdlog::info("SafetyLayer[EMERGENCY]: → {} (ring idx {})",
                     phase_to_string(applied_), ring_idx_);
    } else if (is_green(current) && !sched_trusted &&
               el * 1000.0f >=
                   static_cast<float>(kEmergencyScheduleSanityTimeoutMs)) {
        // ── E-5: max-green watchdog for the emergency ring ──────────────────
        // The per-phase green timer (elapsed since phase_start_, which resets
        // on every ring transition) exceeded the hard ceiling while the
        // scheduled transition cannot fire. Force ALL_RED immediately rather
        // than hold a green indefinitely; jumping to the ring's next ALL_RED
        // entry keeps the anti-starvation rotation intact (the cross approach
        // is served next). No dynamic allocation on this path.
        spdlog::critical("[SAFETY] max-green watchdog triggered in emergency "
                         "ring, forcing ALL_RED");
        do {
            ring_idx_ = (ring_idx_ + 1) % kRingSize; // bounded: ≤ kRingSize steps
        } while (kEmergencyRing[ring_idx_] != Phase::ALL_RED);
        applied_ = Phase::ALL_RED;
        phase_start_ = now;
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
