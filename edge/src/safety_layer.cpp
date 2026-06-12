// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/safety_layer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
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

/// @brief E-5 (amber leg): sanity timeout for an emergency-ring amber whose
/// schedule is untrustworthy (ms). Companion of
/// ``kEmergencyScheduleSanityTimeoutMs`` — same trust check, same firing
/// rule, applied to the ring's YELLOW entries: a corrupt ``Timings::yellow``
/// (NaN/inf/negative/absurd) makes ``elapsed >= schedule`` unreachable and
/// would otherwise freeze the ring in amber indefinitely. A frozen amber is
/// not a conflict, but it is a frozen intersection — the one outcome this
/// module exists to prevent. Both watchdog legs now escalate to the F-1
/// terminal FLASH (via a brief all-red clearance) instead of resuming a ring
/// whose schedule has already proven corrupt.
static constexpr std::uint32_t kEmergencyYellowSanityTimeoutMs = 4000;

static_assert(static_cast<float>(kEmergencyYellowSanityTimeoutMs) <=
                  SafetyLayer::MAX_GREEN_S * 1000.0f,
              "E-5 invariant (amber leg): the yellow sanity timeout must not "
              "exceed MAX_GREEN_S, so the worst-case time in any single ring "
              "phase stays bounded by MAX_GREEN_S.");

// ── F-1: structural proof of the single-yellow flash invariant ─────────────
//
// The three FLASH phases are the *entire* codomain of
// SafetyLayer::flash_phase_for — the only producer of flash patterns. This
// table states which axes each pattern yellow-flashes; the static_assert
// proves no pattern yellow-flashes both crossing axes, so "two crossing
// approaches in simultaneous flashing yellow" is not a representable output
// of the FSM. The runtime guard in step_flash only re-checks membership in
// this closed set (defense in depth against memory corruption); safety does
// not depend on a runtime check.
struct FlashLampPattern {
    Phase phase;
    bool yellow_ns;  ///< Pattern yellow-flashes the NS approach.
    bool yellow_ew;  ///< Pattern yellow-flashes the EW approach.
};
constexpr std::array<FlashLampPattern, 3> kFlashPatterns = {{
    {Phase::FLASH_YELLOW_NS, true, false},
    {Phase::FLASH_YELLOW_EW, false, true},
    {Phase::FLASH_ALL_RED, false, false},
}};
constexpr bool flash_patterns_are_single_yellow() {
    for (const auto& p : kFlashPatterns) {
        if (p.yellow_ns && p.yellow_ew) return false;
    }
    return true;
}
static_assert(flash_patterns_are_single_yellow(),
              "F-1 invariant: no terminal flash pattern may put both crossing "
              "approaches in flashing yellow — at least one axis must stay "
              "restrictive (red).");

const char* major_approach_to_string(MajorApproach m) {
    switch (m) {
        case MajorApproach::NS: return "NS";
        case MajorApproach::EW: return "EW";
        case MajorApproach::UNKNOWN: return "UNKNOWN";
    }
    return "UNKNOWN";
}

} // namespace

MajorApproach major_approach_from_string(const std::string& s) {
    std::string u;
    u.reserve(s.size());
    for (char c : s) {
        u.push_back(static_cast<char>(
            std::toupper(static_cast<unsigned char>(c))));
    }
    if (u == "NS") return MajorApproach::NS;
    if (u == "EW") return MajorApproach::EW;
    return MajorApproach::UNKNOWN;
}

// ── Constructor / Destructor ────────────────────────────────────────────────

SafetyLayer::SafetyLayer() : SafetyLayer(Timings{}, MajorApproach::UNKNOWN) {}

SafetyLayer::SafetyLayer(Timings timings)
    : SafetyLayer(timings, MajorApproach::UNKNOWN) {}

SafetyLayer::SafetyLayer(Timings timings, MajorApproach major)
    : t_(timings),
      major_(major),
      state_(State::ALL_RED),
      applied_(Phase::ALL_RED),
      target_green_(Phase::GREEN_NS),
      target_valid_(false),
      phase_start_(Clock::now()),
      emergency_(false),
      ring_idx_(0),
      stuck_phase_(Phase::ALL_RED) {
    // Boot in ALL_RED: the first requested green is served only after a full
    // clearance interval, so startup is fail-safe. An unconfigured major
    // approach is also fail-safe: FLASH degrades to all-red flash.
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

float SafetyLayer::flash_clearance_s() const {
    // The clearance before terminal FLASH runs precisely because some part of
    // the timing config has proven corrupt — so it must never depend on an
    // untrusted duration. Use the configured all-red only when it is sane;
    // otherwise pin to the compile-time default so the path to FLASH cannot
    // itself freeze.
    const bool trusted = std::isfinite(t_.all_red) && t_.all_red > 0.0f &&
                         t_.all_red <= MAX_GREEN_S;
    return trusted ? t_.all_red : ALL_RED_S;
}

// ── Public API ──────────────────────────────────────────────────────────────

ValidatedPhase SafetyLayer::validate(Phase desired_phase) {
    return validate(desired_phase, Clock::now());
}

ValidatedPhase SafetyLayer::validate(Phase desired_phase, TimePoint now) {
    std::lock_guard<std::mutex> lk(mtx_);
    // F-1 dominates everything: once the schedule has proven corrupt, no
    // request — AI, emergency toggle, anything — can pull the FSM out of the
    // clearance→FLASH path. Only reset() (manual intervention) leaves it.
    if (state_ == State::FLASH_CLEARANCE || state_ == State::FLASH) {
        return step_flash(now);
    }
    return emergency_ ? step_emergency(now) : step_normal(desired_phase, now);
}

void SafetyLayer::set_emergency_mode(bool on) {
    set_emergency_mode(on, Clock::now());
}

void SafetyLayer::set_emergency_mode(bool on, TimePoint now) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (state_ == State::FLASH_CLEARANCE || state_ == State::FLASH) {
        // Terminal FLASH outranks the emergency toggle: a recovered camera or
        // Brain does not un-corrupt the schedule that put us here.
        spdlog::warn("SafetyLayer: set_emergency_mode({}) ignored — terminal "
                     "FLASH active, reset() required to leave it", on);
        return;
    }
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
    stuck_phase_ = Phase::ALL_RED;
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

        case State::FLASH_CLEARANCE:
        case State::FLASH:
            // Should not happen (validate routes FLASH first); stay terminal.
            return step_flash(now);
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
    } else if (!sched_trusted && is_green(current) &&
               el * 1000.0f >=
                   static_cast<float>(kEmergencyScheduleSanityTimeoutMs)) {
        // ── E-5 (green leg): stuck-green watchdog for the emergency ring ────
        // The per-phase timer (elapsed since phase_start_, which resets on
        // every ring transition) exceeded the hard ceiling while the
        // scheduled transition cannot fire. The schedule has proven corrupt,
        // so resuming the ring would just re-serve this frozen green every
        // lap: escalate F-1 instead — brief all-red clearance, then terminal
        // FLASH. No dynamic allocation on this path.
        enter_flash_clearance(current, now);
        return step_flash(now);
    } else if (!sched_trusted && is_yellow(current) &&
               el * 1000.0f >=
                   static_cast<float>(kEmergencyYellowSanityTimeoutMs)) {
        // ── E-5 (amber leg): stuck-amber watchdog, same escalation ──────────
        // A corrupt yellow schedule freezes the ring in amber forever; a
        // frozen amber is a frozen intersection. Same F-1 escalation.
        enter_flash_clearance(current, now);
        return step_flash(now);
    } else if (!sched_trusted &&
               el * 1000.0f >=
                   static_cast<float>(kEmergencyScheduleSanityTimeoutMs)) {
        // ── E-5 (red leg): the ring's own ALL_RED entry is stuck ────────────
        // (current is ALL_RED here — greens and ambers matched the legs
        // above.) A corrupt all-red schedule would otherwise hold the ring in
        // solid red forever: a permanently frozen intersection, which is
        // exactly the "ALL_RED as permanent rest state" this escalation
        // exists to forbid. Same F-1 path; the pre-FLASH clearance clamps to
        // the compile-time default, so it cannot freeze again.
        enter_flash_clearance(current, now);
        return step_flash(now);
    }

    ValidatedPhase r;
    r.phase = applied_;
    r.was_modified = true; // emergency always overrides the AI
    r.reason = "emergency fixed ring";
    r.remaining_s = std::max(0.0f, ring_duration(applied_) - elapsed_s(now));
    return r;
}

// ── F-1: terminal FLASH degraded mode ───────────────────────────────────────

void SafetyLayer::enter_flash_clearance(Phase stuck_phase, TimePoint now) {
    stuck_phase_ = stuck_phase;
    state_ = State::FLASH_CLEARANCE;
    applied_ = Phase::ALL_RED;
    phase_start_ = now;
    spdlog::critical("[SAFETY][E-5] corrupt emergency schedule froze {} — "
                     "clearing the box, then terminal FLASH",
                     phase_to_string(stuck_phase_));
}

ValidatedPhase SafetyLayer::step_flash(TimePoint now) {
    if (state_ == State::FLASH_CLEARANCE &&
        elapsed_s(now) >= flash_clearance_s()) {
        Phase pattern = flash_phase_for(major_);
        // Runtime re-check of the structural guard (defense in depth): the
        // applied pattern must be a member of the closed, statically proven
        // single-yellow flash set. Anything else collapses to all-red flash.
        if (!is_flash(pattern)) {
            spdlog::error("SafetyLayer: flash pattern guard rejected {} — "
                          "falling back to FLASH_ALL_RED",
                          phase_to_string(pattern));
            pattern = Phase::FLASH_ALL_RED;
        }
        state_ = State::FLASH;
        applied_ = pattern;
        phase_start_ = now;
        spdlog::critical("[SAFETY][TERMINAL-FLASH] entering FLASH degraded "
                         "mode: corrupt emergency schedule froze {} — "
                         "applying {} (major approach {}). Manual reset() "
                         "required to resume signalised operation.",
                         phase_to_string(stuck_phase_),
                         phase_to_string(pattern),
                         major_approach_to_string(major_));
    }

    ValidatedPhase r;
    r.phase = applied_;
    r.was_modified = true; // FLASH always overrides whatever was requested
    if (state_ == State::FLASH) {
        r.reason = "terminal FLASH (manual reset required)";
        r.remaining_s = 0.0f; // terminal: no scheduled transition out
    } else {
        r.reason = "all-red clearance before terminal FLASH";
        r.remaining_s = std::max(0.0f, flash_clearance_s() - elapsed_s(now));
    }
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
        case State::FLASH_CLEARANCE:
            r.remaining_s = std::max(0.0f, flash_clearance_s() - el);
            break;
        case State::FLASH:
            r.remaining_s = 0.0f; // terminal
            break;
    }
    return r;
}

} // namespace semaforo
