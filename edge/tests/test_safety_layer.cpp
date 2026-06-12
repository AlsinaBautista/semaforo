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
#include <vector>

#include "semaforo/safety_layer.hpp"

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

// ── E-5: emergency-ring max-green watchdog ──────────────────────────────────
// A corrupt emergency_green (NaN / infinity / absurdly large) makes the ring's
// normal self-timed transition unreachable — without a watchdog the green
// would hold forever, an indefinite freeze in the one mode that must never
// stall. The watchdog must force ALL_RED before 8001 ms of stuck green, and
// the ring must keep rotating (anti-starvation) afterwards.
TEST(SafetyLayer, EmergencyWatchdogForcesAllRed) {
    SafetyLayer::Timings t = test_timings();
    t.emergency_green = std::numeric_limits<float>::infinity(); // stuck green
    SafetyLayer sl(t);
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
    // without leaving emergency mode.
    r = sl.validate(Phase::GREEN_NS,
                    green_start + std::chrono::milliseconds(8000));
    EXPECT_EQ(r.phase, Phase::ALL_RED);
    EXPECT_TRUE(sl.in_emergency());

    // Liveness: the forced ALL_RED is the clearance BEFORE the cross green,
    // so the opposing approach is served next — no starvation in degradation.
    r = sl.validate(Phase::GREEN_NS,
                    green_start + std::chrono::milliseconds(10000));
    EXPECT_EQ(r.phase, Phase::GREEN_EW);
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
