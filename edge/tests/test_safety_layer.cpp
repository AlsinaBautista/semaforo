// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Property-based safety proofs for the deterministic SafetyLayer interlock.
//
// These tests do not merely check examples — they drive the state machine over
// long, randomized command sequences using an *injected* clock and assert that
// the core safety invariants hold for EVERY observed transition:
//
//   C-3  No green ever changes directly into another green: every green→green
//        change is mediated by YELLOW(origin) then ALL_RED, each held >= its
//        minimum.  (Proven via the successor invariants:
//          green   → yellow_for(green)
//          yellow  → ALL_RED
//          all_red → green)
//   C-4  The emergency ring visits every approach each cycle and never stalls.
//   A-1  A green held past max_green is force-switched to the cross green,
//        independent of the AI, so the intersection can never freeze.

#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "semaforo/safety_layer.hpp"
#include "semaforo/signal_controller.hpp"

using namespace semaforo;
using TimePoint = SafetyLayer::TimePoint;

namespace {

// Custom timings with a small max_green so the A-1 watchdog is observable
// within a bounded horizon without huge time jumps.
SafetyLayer::Timings test_timings() {
    SafetyLayer::Timings t;
    t.min_green = 5.0f;
    t.yellow = 3.0f;
    t.all_red = 2.0f;
    t.max_green = 20.0f;
    t.emergency_green = 10.0f;
    return t;
}

TimePoint at(TimePoint base, float seconds) {
    return base + std::chrono::milliseconds(static_cast<long>(seconds * 1000.0f));
}

/// One observed segment of the applied timeline: a phase and how long it held.
struct Segment {
    Phase phase;
    float duration_s;
};

/// Drive the layer over a horizon, returning the compressed sequence of applied
/// phases together with how long each was held.  ``desired_fn(step)`` supplies
/// the AI's requested phase at each tick.
std::vector<Segment> run_segments(SafetyLayer& sl, TimePoint base,
                                  const std::function<Phase(int)>& desired_fn,
                                  float dt, int steps) {
    std::vector<Segment> segs;
    Phase last = Phase::ALL_RED;
    int count = 0;
    bool first = true;
    for (int i = 0; i < steps; ++i) {
        auto r = sl.validate(desired_fn(i), at(base, i * dt));
        if (first || r.phase != last) {
            if (!first) segs.push_back({last, count * dt});
            last = r.phase;
            count = 1;
            first = false;
        } else {
            ++count;
        }
    }
    if (!first) segs.push_back({last, count * dt});
    return segs;
}

std::vector<Phase> phases_only(const std::vector<Segment>& segs) {
    std::vector<Phase> out;
    out.reserve(segs.size());
    for (const auto& s : segs) out.push_back(s.phase);
    return out;
}

/// Assert the C-3 interlock on a compressed timeline of applied phases.
void assert_interlock(const std::vector<Phase>& tl) {
    for (size_t i = 0; i + 1 < tl.size(); ++i) {
        const Phase a = tl[i];
        const Phase b = tl[i + 1];

        // The defining safety property: never green → green directly.
        EXPECT_FALSE(SafetyLayer::is_green(a) && SafetyLayer::is_green(b))
            << "Direct green→green transition at index " << i << ": "
            << phase_to_string(a) << " → " << phase_to_string(b);

        // Strong successor invariants that imply the above and more.
        if (SafetyLayer::is_green(a)) {
            EXPECT_EQ(b, SafetyLayer::yellow_for(a))
                << "Green " << phase_to_string(a)
                << " must be followed by its own amber, got " << phase_to_string(b);
        } else if (SafetyLayer::is_yellow(a)) {
            EXPECT_EQ(b, Phase::ALL_RED)
                << "Amber " << phase_to_string(a)
                << " must be followed by ALL_RED, got " << phase_to_string(b);
        } else if (a == Phase::ALL_RED) {
            EXPECT_TRUE(SafetyLayer::is_green(b))
                << "ALL_RED must be followed by a green, got " << phase_to_string(b);
        }
    }
}

} // namespace

// ── Boot / steady behaviour ─────────────────────────────────────────────────

TEST(SafetyLayer, BootsInAllRed) {
    SafetyLayer sl(test_timings());
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::ALL_RED);
    EXPECT_EQ(sl.get_current_phase(), Phase::ALL_RED);
}

TEST(SafetyLayer, ServesFirstGreenOnlyAfterClearance) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();

    // Before the all-red clearance elapses, it must still be ALL_RED.
    auto early = sl.validate(Phase::GREEN_NS, at(base, 1.0f));
    EXPECT_EQ(early.phase, Phase::ALL_RED);

    // After the clearance, the requested green is served.
    auto served = sl.validate(Phase::GREEN_NS, at(base, 3.0f));
    EXPECT_EQ(served.phase, Phase::GREEN_NS);
}

// ── C-3: minimum green and mediated transition ──────────────────────────────

TEST(SafetyLayer, HoldsGreenUntilMinGreen) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();

    // Reach GREEN_NS.
    sl.validate(Phase::GREEN_NS, at(base, 3.0f));
    ASSERT_EQ(sl.get_current_phase(), Phase::GREEN_NS);

    // Request the cross green only 2s into the green (< min_green = 5s):
    // it must stay GREEN_NS (no premature amber).
    auto held = sl.validate(Phase::GREEN_EW, at(base, 5.0f));
    EXPECT_EQ(held.phase, Phase::GREEN_NS);
    EXPECT_TRUE(held.was_modified);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::GREEN);
}

TEST(SafetyLayer, GreenToCrossGreenIsMediatedByYellowThenAllRed) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();

    // Drive GREEN_NS for a while, then request GREEN_EW, and inspect the
    // physical timeline of applied phases.
    auto desired_fn = [&](int step) {
        return (step < 40) ? Phase::GREEN_NS : Phase::GREEN_EW;
    };
    auto segs = run_segments(sl, base, desired_fn, 0.25f, 400);
    auto tl = phases_only(segs);

    assert_interlock(tl);

    // The timeline must contain the exact mediated subsequence:
    // GREEN_NS → YELLOW_NS → ALL_RED → GREEN_EW.
    bool found = false;
    for (size_t i = 0; i + 3 < tl.size(); ++i) {
        if (tl[i] == Phase::GREEN_NS && tl[i + 1] == Phase::YELLOW_NS &&
            tl[i + 2] == Phase::ALL_RED && tl[i + 3] == Phase::GREEN_EW) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "Expected GREEN_NS→YELLOW_NS→ALL_RED→GREEN_EW sequence";
}

TEST(SafetyLayer, AmberAndClearanceRespectMinimumDurations) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();
    const float dt = 0.25f;
    const auto t = test_timings();

    auto desired_fn = [&](int step) {
        return (step < 40) ? Phase::GREEN_NS : Phase::GREEN_EW;
    };
    auto segs = run_segments(sl, base, desired_fn, dt, 400);

    // Inspect interior segments (skip first/last, which the horizon truncates).
    for (size_t i = 1; i + 1 < segs.size(); ++i) {
        const auto& s = segs[i];
        if (SafetyLayer::is_yellow(s.phase)) {
            EXPECT_GE(s.duration_s, t.yellow - dt);
        } else if (s.phase == Phase::ALL_RED) {
            EXPECT_GE(s.duration_s, t.all_red - dt);
        } else if (SafetyLayer::is_green(s.phase)) {
            EXPECT_GE(s.duration_s, t.min_green - dt);
            EXPECT_LE(s.duration_s, t.max_green + dt + 0.01f);
        }
    }
}

// ── A-1: max-green watchdog forces liveness ─────────────────────────────────

TEST(SafetyLayer, MaxGreenWatchdogForcesSwitchUnderConstantRequest) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();
    const float dt = 0.25f;
    const auto t = test_timings();

    // The AI is stuck: it always demands GREEN_NS. The layer must still serve
    // GREEN_EW because of the max-green watchdog.
    auto desired_fn = [](int) { return Phase::GREEN_NS; };
    auto segs = run_segments(sl, base, desired_fn, dt, 600); // 150 s horizon
    auto tl = phases_only(segs);

    assert_interlock(tl);

    bool saw_ns = false, saw_ew = false;
    for (Phase p : tl) {
        saw_ns |= (p == Phase::GREEN_NS);
        saw_ew |= (p == Phase::GREEN_EW);
    }
    EXPECT_TRUE(saw_ns) << "NS green should be served";
    EXPECT_TRUE(saw_ew) << "Max-green watchdog must force EW green despite the "
                           "AI only ever requesting NS";

    // The NS greens should be held ~max_green (forced), not indefinitely.
    for (size_t i = 1; i + 1 < segs.size(); ++i) {
        if (segs[i].phase == Phase::GREEN_NS) {
            EXPECT_LE(segs[i].duration_s, t.max_green + dt + 0.01f);
        }
    }
}

// ── C-4: emergency ring liveness & no stall ─────────────────────────────────

TEST(SafetyLayer, EmergencyRingVisitsAllApproachesAndNeverStalls) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();
    const float dt = 0.5f;

    sl.set_emergency_mode(true, base);
    ASSERT_TRUE(sl.in_emergency());

    // The desired phase is irrelevant in emergency mode — it must be ignored.
    auto desired_fn = [](int step) {
        return (step % 2) ? Phase::GREEN_NS : Phase::GREEN_EW;
    };
    auto segs = run_segments(sl, base, desired_fn, dt, 400); // 200 s, several laps
    auto tl = phases_only(segs);

    assert_interlock(tl);

    int ns = 0, ew = 0, allred = 0, yellow = 0;
    for (Phase p : tl) {
        ns += (p == Phase::GREEN_NS);
        ew += (p == Phase::GREEN_EW);
        allred += (p == Phase::ALL_RED);
        yellow += SafetyLayer::is_yellow(p) ? 1 : 0;
    }
    // Liveness: both approaches served multiple times — no starvation.
    EXPECT_GE(ns, 2);
    EXPECT_GE(ew, 2);
    EXPECT_GE(allred, 4);
    EXPECT_GE(yellow, 4);

    // No stall: the ring keeps advancing, producing many phase changes.
    EXPECT_GE(static_cast<int>(tl.size()), 12);
}

TEST(SafetyLayer, EmergencyEntryClearsBeforeServingGreen) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();

    // Get a green running first.
    sl.validate(Phase::GREEN_NS, at(base, 3.0f));
    ASSERT_EQ(sl.get_current_phase(), Phase::GREEN_NS);

    // Engage emergency: it must drop to ALL_RED immediately (no green→green).
    sl.set_emergency_mode(true, at(base, 4.0f));
    auto r = sl.validate(Phase::GREEN_EW, at(base, 4.0f));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
}

TEST(SafetyLayer, ExitEmergencyReturnsThroughAllRed) {
    SafetyLayer sl(test_timings());
    const TimePoint base = SafetyLayer::Clock::now();

    sl.set_emergency_mode(true, base);
    sl.set_emergency_mode(false, at(base, 30.0f));
    EXPECT_FALSE(sl.in_emergency());
    EXPECT_EQ(sl.get_current_phase(), Phase::ALL_RED);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::ALL_RED);
}

// ── E-5: emergency-ring schedule watchdog escalates to terminal FLASH ───────
// A corrupt emergency_green (NaN / infinity / absurdly large) makes the ring's
// normal self-timed transition unreachable — without a watchdog the green
// would hold forever, an indefinite freeze in the one mode that must never
// stall. The watchdog must force ALL_RED before 8001 ms of stuck green; and
// because the schedule has now *proven* corrupt, resuming the ring would just
// re-serve the same frozen green every lap — so after a brief clearance the
// layer drops to the terminal FLASH pattern instead (F-1).
TEST(SafetyLayer, EmergencyWatchdogForcesAllRed) {
    SafetyLayer::Timings t = test_timings();
    t.emergency_green = std::numeric_limits<float>::infinity(); // stuck green
    SafetyLayer sl(t); // no major approach configured → all-red flash fallback
    const TimePoint base = SafetyLayer::Clock::now();

    sl.set_emergency_mode(true, base);
    ASSERT_EQ(sl.get_current_phase(), Phase::ALL_RED); // ring entry clearance

    // The 2 s entry clearance completes → the first green is served and,
    // with an infinite schedule, can never transition on its own.
    const TimePoint green_start = at(base, 2.0f);
    auto r = sl.validate(Phase::GREEN_NS, green_start);
    ASSERT_EQ(r.phase, Phase::GREEN_NS);

    // Still green just under the watchdog ceiling…
    r = sl.validate(Phase::GREEN_NS,
                    green_start + std::chrono::milliseconds(7999));
    EXPECT_EQ(r.phase, Phase::GREEN_NS);

    // …but by 8000 ms (< 8001 ms) the watchdog must have forced ALL_RED,
    // without leaving emergency mode: this is the brief clearance that
    // precedes terminal FLASH, never a permanent rest state.
    r = sl.validate(Phase::GREEN_NS,
                    green_start + std::chrono::milliseconds(8000));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
    EXPECT_TRUE(sl.in_emergency());

    // After the clearance the layer must NOT resume a ring whose schedule is
    // corrupt — it lands in the terminal FLASH pattern (all-red flash, since
    // no major approach is configured) and stays there.
    r = sl.validate(Phase::GREEN_NS,
                    green_start + std::chrono::milliseconds(10000));
    EXPECT_EQ(r.phase, Phase::FLASH_ALL_RED);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);
}

// ── Randomized stress: invariants hold for arbitrary command storms ─────────

TEST(SafetyLayer, RandomizedCommandStormPreservesInterlock) {
    const Phase greens[] = {Phase::GREEN_NS, Phase::GREEN_EW,
                            Phase::GREEN_NS_LEFT, Phase::GREEN_EW_LEFT};

    for (unsigned seed = 0; seed < 8; ++seed) {
        SafetyLayer sl(test_timings());
        const TimePoint base = SafetyLayer::Clock::now();
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> pick(0, 5); // 0..3 greens, 4..5 garbage
        std::uniform_real_distribution<float> jitter(0.2f, 1.4f);

        auto desired_fn = [&]() -> Phase {
            const int k = pick(rng);
            if (k < 4) return greens[k];
            // Occasionally feed an illegal desire (amber / all-red) — the layer
            // must treat it as "no change" and never break the interlock.
            return (k == 4) ? Phase::YELLOW_NS : Phase::ALL_RED;
        };

        std::vector<Phase> tl;
        Phase last = Phase::ALL_RED;
        bool first = true;
        float clock_s = 0.0f;
        for (int i = 0; i < 4000; ++i) {
            clock_s += jitter(rng); // variable dt → call-frequency independence
            auto r = sl.validate(desired_fn(), at(base, clock_s));
            if (first || r.phase != last) {
                tl.push_back(r.phase);
                last = r.phase;
                first = false;
            }
        }
        assert_interlock(tl);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// F-1: terminal FLASH degraded mode
//
// When the emergency ring's own schedule proves corrupt (E-5, any leg), the
// layer must escalate stuck phase → brief ALL_RED clearance → terminal FLASH:
// the configured major approach flashes yellow, the crossing approach flashes
// red; with no/invalid major approach everything flashes red. The properties
// proven here:
//   * the terminal state is FLASH — never solid all-red, never all-yellow;
//   * ALL_RED appears only as bounded, transient clearance;
//   * no reachable state yellow-flashes two crossing approaches (checked
//     against the REAL phase→lamp mapping in SimulationSignalController);
//   * FLASH is terminal: requests and emergency toggles cannot leave it,
//     only an explicit reset() can.
// ════════════════════════════════════════════════════════════════════════════

namespace {

SafetyLayer::Timings corrupt_green_timings() {
    SafetyLayer::Timings t = test_timings();
    t.emergency_green = std::numeric_limits<float>::infinity();
    return t;
}

/// Lamp view of a phase through the REAL output mapping. A fresh controller
/// is used so the 1 Hz modulation (driven by the controller's wall clock) is
/// still in its initial lit half-cycle: the returned lamps are the commanded
/// pattern, not a dark blink phase.
SignalState lamps_for(Phase p) {
    SimulationSignalController ctrl;
    ctrl.set_phase(p);
    return ctrl.get_signal_state();
}

} // namespace

// Success criterion 1: corrupt green schedule → terminal FLASH with the major
// approach in flashing yellow and the cross approach in flashing red — NOT
// solid all-red, NOT all-yellow.
TEST(SafetyLayerFlash, CorruptGreenScheduleEndsInTerminalFlash) {
    SafetyLayer sl(corrupt_green_timings(), MajorApproach::NS);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);

    auto desired_fn = [](int) { return Phase::GREEN_NS; };
    auto segs = run_segments(sl, base, desired_fn, 0.25f, 200); // 50 s horizon
    auto tl = phases_only(segs);

    // Exact escalation: ring-entry clearance → stuck green → brief clearance
    // → terminal FLASH. Nothing after FLASH.
    const std::vector<Phase> expected = {Phase::ALL_RED, Phase::GREEN_NS,
                                         Phase::ALL_RED,
                                         Phase::FLASH_YELLOW_NS};
    EXPECT_EQ(tl, expected);

    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);
    EXPECT_EQ(sl.get_current_phase(), Phase::FLASH_YELLOW_NS);

    // The lamp pattern through the real mapping: NS flashing yellow, EW
    // flashing red — the cross approach is restrictive.
    const SignalState s = lamps_for(Phase::FLASH_YELLOW_NS);
    EXPECT_TRUE(s.north_south.yellow);
    EXPECT_TRUE(s.north_south.flashing);
    EXPECT_FALSE(s.north_south.red);
    EXPECT_TRUE(s.east_west.red);
    EXPECT_TRUE(s.east_west.flashing);
    EXPECT_FALSE(s.east_west.yellow);
}

// E-5 amber leg: a corrupt yellow schedule freezes the ring in amber; the
// kEmergencyYellowSanityTimeoutMs watchdog must escalate it through the same
// clearance → FLASH path (here with EW as the major approach).
TEST(SafetyLayerFlash, CorruptYellowScheduleEndsInTerminalFlash) {
    SafetyLayer::Timings t = test_timings();
    t.yellow = std::numeric_limits<float>::quiet_NaN(); // stuck amber
    SafetyLayer sl(t, MajorApproach::EW);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);

    auto desired_fn = [](int) { return Phase::GREEN_NS; };
    auto segs = run_segments(sl, base, desired_fn, 0.25f, 120); // 30 s horizon
    auto tl = phases_only(segs);

    // Entry clearance → green (sane 10 s schedule) → stuck amber → brief
    // clearance → terminal FLASH (EW major → EW flashing yellow).
    const std::vector<Phase> expected = {Phase::ALL_RED, Phase::GREEN_NS,
                                         Phase::YELLOW_NS, Phase::ALL_RED,
                                         Phase::FLASH_YELLOW_EW};
    EXPECT_EQ(tl, expected);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);
}

// E-5 red leg: a corrupt all-red schedule would freeze the ring solid-red at
// its entry forever — ALL_RED as a permanent rest state. The watchdog must
// escalate it too, and the pre-FLASH clearance must use the clamped
// compile-time duration (the configured all_red is the corrupt value).
TEST(SafetyLayerFlash, CorruptAllRedStillReachesFlash) {
    SafetyLayer::Timings t = corrupt_green_timings();
    t.all_red = std::numeric_limits<float>::quiet_NaN(); // stuck ring entry
    SafetyLayer sl(t, MajorApproach::NS);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);

    // Just under the watchdog ceiling the ring entry is still ALL_RED…
    auto r = sl.validate(Phase::GREEN_NS, at(base, 7.9f));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::EMERGENCY);

    // …at 8 s the red leg fires (clearance state, still showing ALL_RED)…
    r = sl.validate(Phase::GREEN_NS, at(base, 8.0f));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH_CLEARANCE);

    // …and 2 s later (the CLAMPED default, not the NaN config) it is FLASH.
    r = sl.validate(Phase::GREEN_NS, at(base, 10.0f));
    EXPECT_EQ(r.phase, Phase::FLASH_YELLOW_NS);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);
}

// Success criterion 2: no major_approach configured → FLASH falls back to
// all-red flash (never all-yellow).
TEST(SafetyLayerFlash, MissingMajorApproachFallsBackToAllRedFlash) {
    SafetyLayer sl(corrupt_green_timings()); // ctor without a major approach
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);

    auto desired_fn = [](int) { return Phase::GREEN_EW; };
    auto segs = run_segments(sl, base, desired_fn, 0.25f, 200);
    auto tl = phases_only(segs);

    ASSERT_FALSE(tl.empty());
    EXPECT_EQ(tl.back(), Phase::FLASH_ALL_RED);

    const SignalState s = lamps_for(Phase::FLASH_ALL_RED);
    EXPECT_TRUE(s.north_south.red && s.north_south.flashing);
    EXPECT_TRUE(s.east_west.red && s.east_west.flashing);
    EXPECT_FALSE(s.north_south.yellow);
    EXPECT_FALSE(s.east_west.yellow);
}

// Success criterion 3 (guard): pattern selection is a closed total function —
// parsing garbage yields UNKNOWN, and ANY value outside the NS/EW enumerators
// (including a corrupted enum) selects the restrictive all-red flash.
TEST(SafetyLayerFlash, InvalidMajorApproachHitsGuardFallback) {
    EXPECT_EQ(major_approach_from_string("NS"), MajorApproach::NS);
    EXPECT_EQ(major_approach_from_string("ns"), MajorApproach::NS);
    EXPECT_EQ(major_approach_from_string("Ew"), MajorApproach::EW);
    EXPECT_EQ(major_approach_from_string("EW"), MajorApproach::EW);
    EXPECT_EQ(major_approach_from_string(""), MajorApproach::UNKNOWN);
    EXPECT_EQ(major_approach_from_string("NSEW"), MajorApproach::UNKNOWN);
    EXPECT_EQ(major_approach_from_string("both"), MajorApproach::UNKNOWN);

    EXPECT_EQ(SafetyLayer::flash_phase_for(MajorApproach::NS),
              Phase::FLASH_YELLOW_NS);
    EXPECT_EQ(SafetyLayer::flash_phase_for(MajorApproach::EW),
              Phase::FLASH_YELLOW_EW);
    EXPECT_EQ(SafetyLayer::flash_phase_for(MajorApproach::UNKNOWN),
              Phase::FLASH_ALL_RED);
    EXPECT_EQ(SafetyLayer::flash_phase_for(static_cast<MajorApproach>(99)),
              Phase::FLASH_ALL_RED);

    // End to end: a corrupted major approach still ends in all-red flash.
    SafetyLayer sl(corrupt_green_timings(), static_cast<MajorApproach>(99));
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);
    sl.validate(Phase::GREEN_NS, at(base, 2.0f));   // green served
    sl.validate(Phase::GREEN_NS, at(base, 10.0f));  // watchdog → clearance
    auto r = sl.validate(Phase::GREEN_NS, at(base, 12.0f));
    EXPECT_EQ(r.phase, Phase::FLASH_ALL_RED);
}

// Success criterion 3 (property): over randomized command storms across sane
// and corrupt configurations, every applied phase — mapped through the REAL
// SimulationSignalController lamp table — yellow-lights at most ONE of the two
// crossing approaches, and any yellow always faces a red.
TEST(SafetyLayerFlash, NoReachableStateYellowFlashesBothCrossingApproaches) {
    const Phase greens[] = {Phase::GREEN_NS, Phase::GREEN_EW,
                            Phase::GREEN_NS_LEFT, Phase::GREEN_EW_LEFT};
    const MajorApproach majors[] = {MajorApproach::NS, MajorApproach::EW,
                                    MajorApproach::UNKNOWN,
                                    static_cast<MajorApproach>(7)};

    std::vector<SafetyLayer::Timings> configs;
    configs.push_back(test_timings());
    configs.push_back(corrupt_green_timings());
    {
        SafetyLayer::Timings t = test_timings();
        t.yellow = std::numeric_limits<float>::quiet_NaN();
        configs.push_back(t);
    }
    {
        SafetyLayer::Timings t = corrupt_green_timings();
        t.all_red = std::numeric_limits<float>::quiet_NaN();
        configs.push_back(t);
    }

    for (const auto& cfg : configs) {
        for (MajorApproach major : majors) {
            for (bool emergency : {false, true}) {
                SafetyLayer sl(cfg, major);
                const TimePoint base = SafetyLayer::Clock::now();
                if (emergency) sl.set_emergency_mode(true, base);

                std::mt19937 rng(42u + static_cast<unsigned>(major) * 7u +
                                 (emergency ? 1u : 0u));
                std::uniform_int_distribution<int> pick(0, 3);
                std::uniform_real_distribution<float> jitter(0.2f, 1.4f);

                std::set<Phase> applied;
                float clock_s = 0.0f;
                for (int i = 0; i < 1500; ++i) {
                    clock_s += jitter(rng);
                    auto r = sl.validate(greens[pick(rng)], at(base, clock_s));
                    applied.insert(r.phase);
                }

                for (Phase p : applied) {
                    const SignalState s = lamps_for(p);
                    EXPECT_FALSE(s.north_south.yellow && s.east_west.yellow)
                        << "Crossing approaches simultaneously yellow in "
                        << phase_to_string(p);
                    if (s.north_south.yellow) {
                        EXPECT_TRUE(s.east_west.red)
                            << "NS yellow without restrictive EW red in "
                            << phase_to_string(p);
                    }
                    if (s.east_west.yellow) {
                        EXPECT_TRUE(s.north_south.red)
                            << "EW yellow without restrictive NS red in "
                            << phase_to_string(p);
                    }
                }
            }
        }
    }
}

// Success criterion 4: in the degraded path ALL_RED is only ever a bounded,
// transient clearance — the rest state is FLASH, never solid all-red.
TEST(SafetyLayerFlash, AllRedIsOnlyTransientClearanceInDegradedPath) {
    SafetyLayer sl(corrupt_green_timings(), MajorApproach::NS);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);

    const float dt = 0.25f;
    const auto t = test_timings();
    auto desired_fn = [](int) { return Phase::GREEN_NS; };
    auto segs = run_segments(sl, base, desired_fn, dt, 400); // 100 s horizon

    ASSERT_GE(segs.size(), 2u);
    // Every ALL_RED hold is a bounded clearance, never a rest state.
    for (size_t i = 0; i + 1 < segs.size(); ++i) {
        if (segs[i].phase == Phase::ALL_RED) {
            EXPECT_LE(segs[i].duration_s, t.all_red + dt + 0.01f)
                << "ALL_RED held as a rest state at segment " << i;
        }
    }
    // The terminal segment is FLASH and it consumes the rest of the horizon.
    EXPECT_TRUE(SafetyLayer::is_flash(segs.back().phase));
    EXPECT_NE(segs.back().phase, Phase::ALL_RED);
    EXPECT_GE(segs.back().duration_s, 80.0f);
}

// FLASH is terminal: no request and no emergency toggle leaves it; only an
// explicit manual reset() restores normal fail-safe boot behaviour.
TEST(SafetyLayerFlash, FlashIsTerminalUntilManualReset) {
    SafetyLayer sl(corrupt_green_timings(), MajorApproach::NS);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.set_emergency_mode(true, base);

    sl.validate(Phase::GREEN_NS, at(base, 2.0f));   // green served
    sl.validate(Phase::GREEN_NS, at(base, 10.0f));  // watchdog → clearance
    auto r = sl.validate(Phase::GREEN_EW, at(base, 12.0f));
    ASSERT_EQ(r.phase, Phase::FLASH_YELLOW_NS);

    // Hours later, under any request, still FLASH.
    r = sl.validate(Phase::GREEN_EW, at(base, 3600.0f));
    EXPECT_EQ(r.phase, Phase::FLASH_YELLOW_NS);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);

    // Emergency toggles are ignored: a recovered camera/Brain does not
    // un-corrupt the schedule that put us here.
    sl.set_emergency_mode(false, at(base, 3601.0f));
    r = sl.validate(Phase::GREEN_EW, at(base, 3602.0f));
    EXPECT_EQ(r.phase, Phase::FLASH_YELLOW_NS);
    sl.set_emergency_mode(true, at(base, 3603.0f));
    r = sl.validate(Phase::GREEN_EW, at(base, 3604.0f));
    EXPECT_EQ(r.phase, Phase::FLASH_YELLOW_NS);

    // Manual reset is the only escape, back to the fail-safe boot ALL_RED.
    sl.reset(at(base, 4000.0f));
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::ALL_RED);
    EXPECT_EQ(sl.get_current_phase(), Phase::ALL_RED);
    EXPECT_FALSE(sl.in_emergency());

    // And normal operation resumes through a fresh clearance.
    r = sl.validate(Phase::GREEN_NS, at(base, 4003.0f));
    EXPECT_EQ(r.phase, Phase::GREEN_NS);
}

// The sustained blink: a 1 Hz square wave (500 ms lit / 500 ms dark), i.e.
// 60 flashes per minute — inside the standard 50–60 flashes/min band — and
// it keeps alternating indefinitely (sustained, not a one-shot).
TEST(SafetyLayerFlash, FlashBlinkIsSustainedOneHertz) {
    using std::chrono::milliseconds;
    auto lit = [](long t_ms) {
        return SimulationSignalController::flash_lamp_lit(milliseconds(t_ms));
    };

    EXPECT_TRUE(lit(0));
    EXPECT_TRUE(lit(499));
    EXPECT_FALSE(lit(500));
    EXPECT_FALSE(lit(999));
    EXPECT_TRUE(lit(1000)); // full 1 s period

    // Sustained: exactly 120 edges over one minute → 60 flashes/min.
    int edges = 0;
    bool last = lit(0);
    for (long t_ms = 100; t_ms <= 60'000; t_ms += 100) {
        const bool now = lit(t_ms);
        if (now != last) ++edges;
        last = now;
    }
    EXPECT_EQ(edges, 120);

    // And it does not decay: still alternating after an hour.
    EXPECT_NE(lit(3'600'000), lit(3'600'500));
}

// ════════════════════════════════════════════════════════════════════════════
// Normal-mode corruption guards (F-1 legs in step_normal)
//
// The normal interlock self-times from the same Timings struct as the
// emergency ring, so the same corruption class (NaN/inf/absurd) can make its
// `elapsed >= schedule` transitions unreachable. Every such freeze vector
// must escalate through the SAME F-1 pipeline (brief clamped clearance →
// terminal FLASH), never hold a frozen phase, and never fire on sane timings.
// ════════════════════════════════════════════════════════════════════════════

// A corrupt yellow in NORMAL mode (no emergency engaged) must not freeze the
// amber: the guard escalates it through clearance to terminal FLASH.
TEST(SafetyLayerNormalCorruption, CorruptYellowEndsInTerminalFlash) {
    SafetyLayer::Timings t = test_timings();
    t.yellow = std::numeric_limits<float>::quiet_NaN(); // stuck normal amber
    SafetyLayer sl(t, MajorApproach::NS);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.reset(base);

    // Serve NS, then request the cross green so a (stuck) amber begins.
    auto desired_fn = [](int step) {
        return (step < 40) ? Phase::GREEN_NS : Phase::GREEN_EW;
    };
    auto segs = run_segments(sl, base, desired_fn, 0.25f, 200); // 50 s horizon
    auto tl = phases_only(segs);

    const std::vector<Phase> expected = {Phase::ALL_RED, Phase::GREEN_NS,
                                         Phase::YELLOW_NS, Phase::ALL_RED,
                                         Phase::FLASH_YELLOW_NS};
    EXPECT_EQ(tl, expected);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);
}

// A corrupt all-red in NORMAL mode can never complete a clearance, so no
// green could ever be served again — the guard must refuse to hold ALL_RED
// as a rest state, and the pre-FLASH clearance must use the CLAMPED default
// (the configured all_red is the corrupt value).
TEST(SafetyLayerNormalCorruption, CorruptAllRedEndsInTerminalFlash) {
    SafetyLayer::Timings t = test_timings();
    t.all_red = std::numeric_limits<float>::quiet_NaN(); // stuck clearance
    SafetyLayer sl(t, MajorApproach::EW);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.reset(base);

    // Just under the guard ceiling the boot clearance is still ALL_RED…
    auto r = sl.validate(Phase::GREEN_NS, at(base, 7.9f));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::ALL_RED);

    // …at 8 s the guard fires into the pre-FLASH clearance…
    r = sl.validate(Phase::GREEN_NS, at(base, 8.0f));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH_CLEARANCE);

    // …and 2 s later (clamped compile-time default) it is terminal FLASH.
    r = sl.validate(Phase::GREEN_NS, at(base, 10.0f));
    EXPECT_EQ(r.phase, Phase::FLASH_YELLOW_EW);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);
}

// A corrupt max_green disables the A-1 watchdog, so a constant request would
// hold its green forever. The guard bounds any green by the system-wide
// MAX_GREEN_S ceiling and escalates to terminal FLASH.
TEST(SafetyLayerNormalCorruption, CorruptMaxGreenEndsInTerminalFlash) {
    SafetyLayer::Timings t = test_timings();
    t.max_green = std::numeric_limits<float>::quiet_NaN(); // A-1 disabled
    SafetyLayer sl(t, MajorApproach::NS);
    const TimePoint base = SafetyLayer::Clock::now();
    sl.reset(base);

    const float dt = 0.5f;
    auto desired_fn = [](int) { return Phase::GREEN_NS; }; // AI stuck on NS
    auto segs = run_segments(sl, base, desired_fn, dt, 140); // 70 s horizon
    auto tl = phases_only(segs);

    const std::vector<Phase> expected = {Phase::ALL_RED, Phase::GREEN_NS,
                                         Phase::ALL_RED,
                                         Phase::FLASH_YELLOW_NS};
    EXPECT_EQ(tl, expected);
    EXPECT_EQ(sl.get_state(), SafetyLayer::State::FLASH);

    // The stuck green was bounded by the hard MAX_GREEN_S ceiling.
    ASSERT_GE(segs.size(), 2u);
    EXPECT_LE(segs[1].duration_s, SafetyLayer::MAX_GREEN_S + dt + 0.01f);
}

// Negative control: with sane timings the corruption guards must NEVER fire —
// a long randomized storm in normal mode produces no FLASH and keeps the
// C-3 interlock intact (the guards change nothing on the healthy path).
TEST(SafetyLayerNormalCorruption, SaneSchedulesNeverFlash) {
    const Phase greens[] = {Phase::GREEN_NS, Phase::GREEN_EW,
                            Phase::GREEN_NS_LEFT, Phase::GREEN_EW_LEFT};

    for (unsigned seed = 0; seed < 4; ++seed) {
        SafetyLayer sl(test_timings(), MajorApproach::NS);
        const TimePoint base = SafetyLayer::Clock::now();
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> pick(0, 3);
        std::uniform_real_distribution<float> jitter(0.2f, 1.4f);

        std::vector<Phase> tl;
        Phase last = Phase::ALL_RED;
        bool first = true;
        float clock_s = 0.0f;
        for (int i = 0; i < 2000; ++i) {
            clock_s += jitter(rng);
            auto r = sl.validate(greens[pick(rng)], at(base, clock_s));
            EXPECT_FALSE(SafetyLayer::is_flash(r.phase))
                << "FLASH reached with sane timings (seed " << seed << ")";
            if (first || r.phase != last) {
                tl.push_back(r.phase);
                last = r.phase;
                first = false;
            }
        }
        assert_interlock(tl);
        EXPECT_NE(sl.get_state(), SafetyLayer::State::FLASH);
        EXPECT_NE(sl.get_state(), SafetyLayer::State::FLASH_CLEARANCE);
    }
}

// ── Shutdown emergency flash (output layer) ─────────────────────────────────

// set_emergency_flash() — the shutdown path — must produce a true flashing
// all-red (FLASH_ALL_RED + both heads red&flashing), wiping every other lamp
// including a protected left arrow lit by the previous phase.
TEST(SimulationController, EmergencyFlashIsFlashingRedAllApproaches) {
    SimulationSignalController ctrl;
    ctrl.set_phase(Phase::GREEN_NS_LEFT);
    ASSERT_TRUE(ctrl.get_signal_state().north_south.left_arrow);

    ctrl.set_emergency_flash();
    EXPECT_EQ(ctrl.get_current_phase(), Phase::FLASH_ALL_RED);

    const SignalState s = ctrl.get_signal_state();
    EXPECT_TRUE(s.north_south.red && s.north_south.flashing);
    EXPECT_TRUE(s.east_west.red && s.east_west.flashing);
    EXPECT_FALSE(s.north_south.yellow || s.north_south.green ||
                 s.north_south.left_arrow);
    EXPECT_FALSE(s.east_west.yellow || s.east_west.green ||
                 s.east_west.left_arrow);
    // The 1 Hz cadence itself is proven in FlashBlinkIsSustainedOneHertz.
}

// A protected left arrow must not survive into the next phase: an NS arrow
// still lit against an EW green is a conflicting indication.
TEST(SimulationController, LeftArrowClearedOnPhaseChange) {
    SimulationSignalController ctrl;
    ctrl.set_phase(Phase::GREEN_NS_LEFT);
    ASSERT_TRUE(ctrl.get_signal_state().north_south.left_arrow);

    ctrl.set_phase(Phase::GREEN_EW);
    const SignalState s = ctrl.get_signal_state();
    EXPECT_FALSE(s.north_south.left_arrow);
    EXPECT_TRUE(s.north_south.red);
    EXPECT_TRUE(s.east_west.green);
    EXPECT_FALSE(s.east_west.left_arrow);
}

TEST(SafetyLayer, DeterministicForFixedSeed) {
    auto build_timeline = []() {
        SafetyLayer sl(test_timings());
        const TimePoint base = SafetyLayer::Clock::now();
        sl.reset(base); // align the FSM clock with our injected timeline
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> pick(0, 3);
        const Phase greens[] = {Phase::GREEN_NS, Phase::GREEN_EW,
                                Phase::GREEN_NS_LEFT, Phase::GREEN_EW_LEFT};
        std::vector<Phase> tl;
        for (int i = 0; i < 1000; ++i) {
            auto r = sl.validate(greens[pick(rng)], at(base, i * 0.5f));
            tl.push_back(r.phase);
        }
        return tl;
    };
    EXPECT_EQ(build_timeline(), build_timeline());
}
