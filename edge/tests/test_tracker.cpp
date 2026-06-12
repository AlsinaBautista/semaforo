// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// SORT tracker proofs (audit M-3). The hazard these tests pin down: a vehicle
// that becomes occluded (behind the signal mast, behind a bus) must NOT change
// identity when it reappears on its trajectory — an ID-switch corrupts the
// queue/velocity estimates that feed phase decisions. The Kalman motion model
// must coast occluded tracks along their predicted path, and the Hungarian
// assignment must be globally optimal (no greedy ID steals between close
// vehicles).

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "semaforo/tracker.hpp"

using namespace semaforo;

namespace {

Detection make_det(float x, float y, float w, float h,
                   int class_id = 0, float conf = 0.9f) {
    return Detection{BBox{x, y, w, h}, class_id, conf};
}

} // namespace

// ── M-3: the required proof — ID persists across a 5-frame occlusion ─────────
//
// A vehicle drives at a constant 12 px/frame, is detected for 10 frames,
// disappears for 5 frames (occluded behind the signal mast), then reappears
// exactly on its extrapolated trajectory. The Kalman prediction must have
// coasted the track to the reappearance point so the same ID re-attaches.
TEST(SortTracker, OcclusionIdPersistence) {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 30;   // tolerates up to 30 unseen frames — well above the gap
    cfg.min_hits = 3;
    VehicleTracker tracker(cfg);

    constexpr float kVx = 12.0f;  // px/frame
    constexpr float kW = 60.0f, kH = 40.0f, kY = 200.0f;
    auto bbox_x_at = [&](int frame) { return 100.0f + kVx * static_cast<float>(frame); };

    // Frames 0..9: vehicle visible.
    uint64_t original_id = 0;
    for (int f = 0; f < 10; ++f) {
        auto tracks = tracker.update({make_det(bbox_x_at(f), kY, kW, kH)});
        if (f >= cfg.min_hits - 1) {
            ASSERT_EQ(tracks.size(), 1u) << "frame " << f;
            original_id = tracks[0].track_id;
        }
    }
    ASSERT_NE(original_id, 0u);
    // The filter has locked onto the true velocity before the occlusion.
    EXPECT_NEAR(tracker.get_all_tracks().at(original_id).obj.velocity_x, kVx, 2.0f);

    // Frames 10..14: occluded — no detections at all. The track must survive,
    // coasting on its prediction.
    for (int f = 10; f < 15; ++f) {
        tracker.update({});
        ASSERT_EQ(tracker.get_all_tracks().count(original_id), 1u)
            << "track died during occlusion at frame " << f;
    }

    // Frame 15: the vehicle reappears exactly where the trajectory says.
    auto tracks = tracker.update({make_det(bbox_x_at(15), kY, kW, kH)});
    ASSERT_EQ(tracks.size(), 1u);

    // INVARIANT (M-3): same identity before and after the occlusion, and no
    // spurious track was ever created.
    EXPECT_EQ(tracks[0].track_id, original_id);
    EXPECT_EQ(tracker.total_tracks_created(), 1u);
    EXPECT_EQ(tracks[0].frames_since_seen, 0);
}

// ── Hungarian optimality: greedy would steal an ID, optimal must not ─────────
//
// Two stationary vehicles A (x=0) and B (x=50). The next frame brings
// det1 (x=40) and det2 (x=55):
//   IoU(A,det1)=0.43  IoU(B,det1)=0.82
//   IoU(A,det2)=0.29 (below gate)  IoU(B,det2)=0.90
// Greedy (detection order) hands det1 to B; det2 then has no feasible track
// and spawns a third ID while A is orphaned. The optimal assignment is
// det1→A, det2→B: both tracks matched, no new ID.
TEST(SortTracker, HungarianBeatsGreedy) {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 30;
    cfg.min_hits = 1;
    VehicleTracker tracker(cfg);

    // Establish both tracks as stationary over a few frames.
    for (int f = 0; f < 3; ++f) {
        auto tracks = tracker.update({make_det(0.0f, 0.0f, 100.0f, 100.0f),
                                      make_det(50.0f, 0.0f, 100.0f, 100.0f)});
        ASSERT_EQ(tracks.size(), 2u);
    }
    ASSERT_EQ(tracker.total_tracks_created(), 2u);

    // Adversarial frame: det1 overlaps B more than A, but taking it greedily
    // leaves det2 unassignable.
    auto tracks = tracker.update({make_det(40.0f, 0.0f, 100.0f, 100.0f),
                                  make_det(55.0f, 0.0f, 100.0f, 100.0f)});

    // INVARIANT: globally optimal assignment — both existing tracks matched
    // this frame and no third ID was created.
    ASSERT_EQ(tracks.size(), 2u);
    EXPECT_EQ(tracker.total_tracks_created(), 2u);
    for (const auto& t : tracks) {
        EXPECT_EQ(t.frames_since_seen, 0)
            << "track " << t.track_id << " was orphaned by a greedy-style steal";
    }
}

// ── Class gate: near-identical boxes of different classes never associate ────
TEST(SortTracker, ClassGateNeverCrossMatches) {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 30;
    cfg.min_hits = 1;
    VehicleTracker tracker(cfg);

    auto tracks = tracker.update({make_det(100.0f, 100.0f, 60.0f, 40.0f, /*class=*/0)});
    ASSERT_EQ(tracks.size(), 1u);
    const uint64_t car_id = tracks[0].track_id;

    // Same place, different class: must spawn a new track, not re-label the car.
    tracker.update({make_det(100.0f, 100.0f, 60.0f, 40.0f, /*class=*/4)});
    EXPECT_EQ(tracker.total_tracks_created(), 2u);
    // The original track was NOT matched by the cross-class detection.
    EXPECT_EQ(tracker.get_all_tracks().at(car_id).obj.frames_since_seen, 1);
}

// ── Lifecycle: min_hits confirmation and max_age death ───────────────────────
TEST(SortTracker, LifecycleMinHitsAndMaxAge) {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 5;
    cfg.min_hits = 3;
    VehicleTracker tracker(cfg);

    const Detection det = make_det(300.0f, 300.0f, 80.0f, 50.0f);

    // Not emitted until min_hits consecutive frames of existence.
    EXPECT_TRUE(tracker.update({det}).empty());   // age 1
    EXPECT_TRUE(tracker.update({det}).empty());   // age 2
    EXPECT_EQ(tracker.update({det}).size(), 1u);  // age 3 — confirmed

    // Unseen for more than max_age frames ⇒ the track dies.
    for (int f = 0; f < cfg.max_age + 1; ++f) {
        tracker.update({});
    }
    EXPECT_TRUE(tracker.get_all_tracks().empty());

    // A later reappearance is a NEW identity (the old one expired).
    tracker.update({det});
    EXPECT_EQ(tracker.total_tracks_created(), 2u);
}

// ── Stopped vehicle: the filter must not invent phantom velocity ─────────────
//
// The queue estimator classifies "stopped" by velocity magnitude; a Kalman
// filter fed a jittering-but-stationary box must converge to ~zero velocity.
TEST(SortTracker, StationaryVehicleVelocityNearZero) {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 30;
    cfg.min_hits = 1;
    VehicleTracker tracker(cfg);

    std::vector<TrackedObject> tracks;
    for (int f = 0; f < 40; ++f) {
        const float jitter = (f % 2 == 0) ? 1.0f : -1.0f;  // ±1 px sensor noise
        tracks = tracker.update({make_det(500.0f + jitter, 400.0f, 60.0f, 40.0f)});
    }

    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracker.total_tracks_created(), 1u);  // jitter never broke the ID
    EXPECT_LT(std::abs(tracks[0].velocity_x), 0.5f);
    EXPECT_LT(std::abs(tracks[0].velocity_y), 0.5f);
}

// ── Property-based storm: random detections, invariants always hold ──────────
TEST(SortTracker, RandomStormInvariants) {
    VehicleTracker::Config cfg;
    cfg.iou_threshold = 0.3f;
    cfg.max_age = 30;
    cfg.min_hits = 3;
    VehicleTracker tracker(cfg);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> n_dets(0, 15);
    std::uniform_real_distribution<float> px(0.0f, 1280.0f);
    std::uniform_real_distribution<float> py(0.0f, 720.0f);
    std::uniform_real_distribution<float> dim(20.0f, 120.0f);
    std::uniform_int_distribution<int> cls(0, 4);

    ObjectPool<TrackedObject> pool(64);

    for (int frame = 0; frame < 300; ++frame) {
        std::vector<Detection> dets;
        const int count = n_dets(rng);
        for (int i = 0; i < count; ++i) {
            dets.push_back(make_det(px(rng), py(rng), dim(rng), dim(rng), cls(rng)));
        }

        tracker.update(dets, pool);

        // INVARIANT: emission is hard-bounded by the pool capacity.
        ASSERT_LE(pool.size(), pool.capacity());

        // INVARIANT: every emitted ID is unique within a frame, and every
        // emitted field is finite (the filter never diverges to NaN/Inf).
        std::set<uint64_t> ids;
        for (const auto& t : pool) {
            EXPECT_TRUE(ids.insert(t.track_id).second) << "duplicate ID " << t.track_id;
            EXPECT_TRUE(std::isfinite(t.detection.bbox.x));
            EXPECT_TRUE(std::isfinite(t.detection.bbox.y));
            EXPECT_TRUE(std::isfinite(t.detection.bbox.width));
            EXPECT_TRUE(std::isfinite(t.detection.bbox.height));
            EXPECT_TRUE(std::isfinite(t.velocity_x));
            EXPECT_TRUE(std::isfinite(t.velocity_y));
        }

        // INVARIANT: no zombie tracks — nothing survives past max_age unseen.
        for (const auto& pair : tracker.get_all_tracks()) {
            ASSERT_LE(pair.second.obj.frames_since_seen, cfg.max_age);
        }
    }
}
