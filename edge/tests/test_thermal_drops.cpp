// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Thermal-throttling resilience proofs for the SORT tracker. The hazard: under
// sustained load the edge SoC overheats and the perception pipeline drops from
// its nominal 30 FPS to ~10 FPS. With a fixed one-frame motion step the Kalman
// filter then under-extrapolates by 3× across each (now ~100 ms) interval, so a
// vehicle's predicted box lags far behind where it actually moved, the IoU gate
// fails, and the track switches ID — corrupting the queue/velocity estimates
// that drive phase decisions. The fix is a dynamic Δt: ``predict(dt)`` coasts
// the box by velocity × dt, so the prediction tracks the real motion even when
// frames arrive late. These tests inject the throttled inter-frame delta
// (100 ms = 10 FPS) and assert the tracker keeps its IDs.

#include <gtest/gtest.h>

#include <cmath>
#include <set>
#include <vector>

#include "semaforo/tracker.hpp"

using namespace semaforo;

namespace {

Detection make_det(float x, float y, float w, float h,
                   int class_id = 0, float conf = 0.9f) {
    return Detection{BBox{x, y, w, h}, class_id, conf};
}

// Nominal cadence the velocity state is expressed against, and the throttled
// cadence we simulate. A 100 ms inter-frame gap against a 30 FPS design rate is
// exactly three nominal frame intervals, so the injected Δt is 3.0.
constexpr double kNominalFps = 30.0;
constexpr double kNominalIntervalMs = 1000.0 / kNominalFps;   // ≈ 33.33 ms
constexpr double kThrottledIntervalMs = 100.0;                // 10 FPS
const float kDtDrop =
    static_cast<float>(kThrottledIntervalMs / kNominalIntervalMs);  // == 3.0

// Vehicle kinematics. A speed of 20 px per nominal frame against a 60 px-wide
// box is the key adversarial choice: under a fixed Δt = 1 the throttled frame
// jumps 3× this (60 px) while the prediction advances only 20 px, leaving a
// 40 px lag whose IoU (0.2) falls below the 0.3 gate — a guaranteed ID switch.
// The dynamic Δt advances the full 60 px and stays matched.
constexpr float kVx = 20.0f;   // px per nominal frame
constexpr float kW = 60.0f;
constexpr float kH = 40.0f;
constexpr float kY = 200.0f;
constexpr float kX0 = 100.0f;

VehicleTracker::Config make_cfg() {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 30;
    cfg.min_hits = 3;
    return cfg;
}

// Drive a track to a confirmed, velocity-locked state at the nominal 30 FPS.
// Returns the (single) confirmed track id.
uint64_t warm_up_at_nominal_rate(VehicleTracker& tracker, int frames = 12) {
    uint64_t id = 0;
    for (int f = 0; f < frames; ++f) {
        auto tracks = tracker.update({make_det(kX0 + kVx * f, kY, kW, kH)}, 1.0f);
        if (!tracks.empty()) {
            id = tracks.front().track_id;
        }
    }
    return id;
}

} // namespace

// ── The required proof: ID survives a 100 ms (10 FPS) thermal drop ───────────
//
// A vehicle is tracked at the nominal 30 FPS until its velocity locks, then the
// pipeline throttles to 10 FPS. Each subsequent frame the vehicle has really
// moved 3× the per-nominal-frame step (because 3× the time elapsed), and we
// inject the matching Δt = 3.0. The dynamic motion model must coast the box the
// full distance so the prediction still overlaps the next detection and the
// same ID re-attaches — frame after frame.
TEST(ThermalDrops, IdPersistsAcrossTenFpsThrottle) {
    ASSERT_FLOAT_EQ(kDtDrop, 3.0f) << "100 ms @ 30 FPS nominal must be 3 frame steps";

    VehicleTracker tracker(make_cfg());
    const uint64_t original_id = warm_up_at_nominal_rate(tracker);
    ASSERT_NE(original_id, 0u);
    EXPECT_NEAR(tracker.get_all_tracks().at(original_id).obj.velocity_x, kVx, 3.0f)
        << "filter must lock onto the true velocity before the drop";

    // Thermal drop: sustained 10 FPS. The vehicle advances kVx * kDtDrop per
    // processed frame; we inject the throttled Δt each time.
    float x = kX0 + kVx * 11;  // position after the 12-frame warm-up (f = 0..11)
    for (int step = 1; step <= 5; ++step) {
        x += kVx * kDtDrop;
        auto tracks = tracker.update({make_det(x, kY, kW, kH)}, kDtDrop);

        // INVARIANT: the identity is preserved — same id, matched this frame
        // (frames_since_seen back to 0), and no spurious track ever spawned.
        ASSERT_EQ(tracks.size(), 1u) << "drop step " << step;
        EXPECT_EQ(tracks.front().track_id, original_id) << "drop step " << step;
        EXPECT_EQ(tracks.front().frames_since_seen, 0) << "drop step " << step;
        EXPECT_EQ(tracker.total_tracks_created(), 1u) << "drop step " << step;
    }
}

// ── Control: a fixed Δt loses the ID under the identical throttle ────────────
//
// Same warm-up and same real motion (the vehicle jumps 60 px over the 100 ms
// gap), but here we feed the tracker the *naive* Δt = 1.0 — modelling the old
// fixed-one-frame behaviour. The prediction advances only 20 px, the 40 px lag
// drops IoU to 0.2 (below the 0.3 gate), the detection goes unmatched, and a
// fresh ID is spawned. This is exactly the failure the dynamic Δt prevents;
// pinning it here proves the fix is load-bearing, not incidental.
TEST(ThermalDrops, FixedDtSwitchesIdUnderThrottle) {
    VehicleTracker tracker(make_cfg());
    const uint64_t original_id = warm_up_at_nominal_rate(tracker);
    ASSERT_NE(original_id, 0u);

    // The vehicle really moved 3 nominal frames' worth, but we (wrongly) tell
    // the filter only one frame elapsed.
    float x = kX0 + kVx * 11;
    x += kVx * kDtDrop;
    tracker.update({make_det(x, kY, kW, kH)}, /*dt=*/1.0f);

    // INVARIANT: the under-extrapolated prediction missed the detection, so the
    // original track went unmatched and a second identity was created.
    EXPECT_GT(tracker.get_all_tracks().at(original_id).obj.frames_since_seen, 0)
        << "fixed Δt should have failed to match the throttled detection";
    EXPECT_EQ(tracker.total_tracks_created(), 2u)
        << "fixed Δt should have spawned a switched ID";
}

// ── Thermal flapping: alternating 30/10 FPS must not churn IDs ───────────────
//
// A throttling SoC does not settle at one rate — it oscillates as it heats and
// cools. With the per-frame Δt tracking each interval, a single vehicle driven
// through a long alternating 30/10 FPS sequence must keep one stable ID and
// never let the filter diverge to NaN/Inf.
TEST(ThermalDrops, OscillatingFrameRateKeepsStableId) {
    VehicleTracker tracker(make_cfg());

    // Lock velocity at the nominal rate first — a brand-new track has no motion
    // estimate, so no Δt can predict its first throttled jump (inherent to
    // SORT). The throttling only begins once the vehicle is being tracked.
    const uint64_t id = warm_up_at_nominal_rate(tracker);
    ASSERT_NE(id, 0u);

    float x = kX0 + kVx * 11;  // position after the warm-up (f = 0..11)
    for (int frame = 0; frame < 120; ++frame) {
        const float dt = (frame % 2 == 0) ? kDtDrop : 1.0f;  // flap 10 ↔ 30 FPS
        x += kVx * dt;
        auto tracks = tracker.update({make_det(x, kY, kW, kH)}, dt);

        ASSERT_EQ(tracks.size(), 1u) << "frame " << frame;
        const TrackedObject& t = tracks.front();

        // INVARIANT: one identity throughout, matched every frame, finite state.
        EXPECT_EQ(t.track_id, id) << "ID churned at frame " << frame;
        EXPECT_EQ(t.frames_since_seen, 0) << "frame " << frame;
        EXPECT_TRUE(std::isfinite(t.detection.bbox.x));
        EXPECT_TRUE(std::isfinite(t.velocity_x));
    }

    EXPECT_EQ(tracker.total_tracks_created(), 1u)
        << "frame-rate flapping must not create phantom tracks";
}
