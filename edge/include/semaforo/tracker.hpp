// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <cstdint>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

#include "semaforo/detector.hpp"

namespace semaforo {

/// @brief A tracked object with a persistent identity across frames.
struct TrackedObject {
    uint64_t track_id;        ///< Unique track identifier
    Detection detection;      ///< Most recent detection associated with this track
    int age;                  ///< Number of frames this track has existed
    int frames_since_seen;    ///< Number of consecutive frames without a matching detection
    float velocity_x;         ///< Estimated horizontal velocity (px/frame)
    float velocity_y;         ///< Estimated vertical velocity (px/frame)
    std::chrono::steady_clock::time_point first_seen;  ///< Timestamp when track was created
    std::chrono::steady_clock::time_point last_seen;   ///< Timestamp of last matching detection
};

/// @brief Constant-velocity linear Kalman filter over a bounding box (SORT).
///
/// State vector ``x = [u, v, s, r, u̇, v̇, ṡ]`` where ``(u, v)`` is the box
/// centre, ``s`` its area (scale) and ``r`` its aspect ratio (held constant,
/// which keeps predicted boxes from degenerating). Measurement is
/// ``z = [u, v, s, r]`` derived from a detection. Fixed-size ``cv::Matx``
/// algebra: all predict/update work happens on the stack (M-5: no per-frame
/// heap allocation).
struct KalmanBoxFilter {
    using Vec7 = cv::Matx<float, 7, 1>;
    using Mat7 = cv::Matx<float, 7, 7>;

    /// @brief Initialise the state from a first detection (velocities zero,
    ///        large initial velocity uncertainty).
    void init(const BBox& box);

    /// @brief Advance the state under the constant-velocity model for a
    ///        time step of @p dt frames (``x ← F(dt) x``, ``P ← F P Fᵀ + Q``).
    ///        Guards against the predicted scale going non-positive by zeroing
    ///        ``ṡ`` first.
    ///
    /// @p dt is the elapsed time since the previous frame, **expressed in
    /// nominal frame intervals** (so ``dt = 1`` is the design cadence and the
    /// velocity state stays in px/frame). Making it dynamic is what lets the
    /// tracker survive thermal throttling: when the pipeline drops from the
    /// nominal 30 FPS to e.g. 10 FPS, a real ~100 ms gap is ``dt = 3`` and the
    /// motion model coasts the box the full distance the vehicle actually
    /// travelled, so its predicted position still overlaps the next detection
    /// and the IoU gate re-matches the same ID. A fixed ``dt = 1`` would
    /// under-extrapolate by 3× under the same drop and switch IDs.
    void predict(float dt = 1.0f);

    /// @brief Fold a matched detection into the state (standard KF update).
    void update(const BBox& box);

    /// @brief Current state as a pixel-space bounding box.
    BBox bbox() const;

    /// @brief Estimated centre velocity (px/frame).
    float vx() const { return x(4); }
    float vy() const { return x(5); }

    Vec7 x;  ///< State mean ``[u, v, s, r, u̇, v̇, ṡ]``.
    Mat7 P;  ///< State covariance.
};

/// @brief SORT multi-object tracker for vehicles at intersections.
///
/// Implements SORT (Bewley et al.): each track carries a constant-velocity
/// Kalman filter that predicts its bounding box every frame, and detections
/// are associated to predicted boxes by solving the optimal assignment
/// (Hungarian algorithm) over an IoU cost matrix, gated by
/// ``Config::iou_threshold`` and class equality.
///
/// The motion model is what gives ID persistence under occlusion (audit M-3):
/// an unmatched track coasts on its prediction for up to ``Config::max_age``
/// frames, so a vehicle that disappears behind the signal mast and reappears
/// on its extrapolated trajectory recovers its original ID instead of
/// spawning a new one.
///
/// Lightweight by design (fixed-size stack algebra, reused scratch buffers)
/// and suitable for edge deployment where computational resources are limited.
class VehicleTracker {
public:
    /// @brief Configuration parameters for the tracker.
    struct Config {
        float iou_threshold = 0.3f;    ///< Minimum IoU to associate detection with track
        int max_age = 30;              ///< Max frames a track survives without detections
        int min_hits = 3;              ///< Min detections before a track is considered confirmed
    };

    /// @brief Internal per-track record: the public output object plus the
    ///        Kalman filter that drives its motion model.
    struct Track {
        TrackedObject obj;   ///< Public view emitted to consumers.
        KalmanBoxFilter kf;  ///< SORT motion model for this track.
    };

    /// @brief Construct a tracker with default configuration.
    VehicleTracker();

    /// @brief Construct a tracker with the given configuration.
    /// @param config  Tracker parameters (IoU threshold, max age, min hits).
    explicit VehicleTracker(const Config& config);
    ~VehicleTracker();

    /// @brief Update tracks with a new set of detections (convenience form).
    ///
    /// Runs one SORT cycle: Kalman-predict every track, associate detections
    /// to predicted boxes via the Hungarian algorithm, update matched filters,
    /// spawn tracks for unmatched detections, and prune tracks unseen for more
    /// than ``Config::max_age`` frames. Allocates a fresh result vector;
    /// prefer the pooled overload below on the hot path.
    ///
    /// @param detections  Vector of detections from the current frame.
    /// @param dt          Elapsed time since the previous frame, in nominal
    ///                    frame intervals (default 1.0 = design cadence). Pass
    ///                    the real inter-frame ratio when the pipeline throttles
    ///                    (e.g. 3.0 for a 100 ms gap against a 30 FPS nominal).
    /// @return Vector of currently active tracked objects (confirmed tracks only).
    std::vector<TrackedObject> update(const std::vector<Detection>& detections,
                                      float dt = 1.0f);

    /// @brief Allocation-free update: fill a caller-owned ObjectPool (M-5).
    ///
    /// Identical association logic to ``update(detections)`` but emits confirmed
    /// tracks into a recycled ``ObjectPool<TrackedObject>`` and reuses internal
    /// scratch (cost matrix, Hungarian solver state) across frames. Steady-state
    /// operation (a stable set of tracks) performs **no heap allocation**. The
    /// pool is cleared first.
    ///
    /// @param detections  Detections from the current frame.
    /// @param out         Pool to clear and fill with confirmed tracks.
    /// @param dt          Elapsed time since the previous frame, in nominal
    ///                    frame intervals (default 1.0 = design cadence).
    void update(const std::vector<Detection>& detections,
                ObjectPool<TrackedObject>& out,
                float dt = 1.0f);

    /// @brief Get all currently active tracks, including unconfirmed ones.
    ///
    /// Introspection/test hook: exposes the internal records (public object +
    /// Kalman filter) keyed by track id.
    ///
    /// @return Reference to the internal map of track_id → Track.
    const std::unordered_map<uint64_t, Track>& get_all_tracks() const;

    /// @brief Reset the tracker, clearing all existing tracks.
    void reset();

    /// @brief Get the total number of unique tracks created since construction or last reset.
    /// @return Cumulative track count.
    uint64_t total_tracks_created() const;

private:
    /// @brief Compute IoU between two bounding boxes.
    /// @param a  First bounding box.
    /// @param b  Second bounding box.
    /// @return Intersection-over-Union value in [0, 1].
    static float compute_iou(const BBox& a, const BBox& b);

    /// @brief Solve the ``n×n`` assignment problem over ``cost_`` (minimisation,
    ///        Kuhn–Munkres with potentials, O(n³)), filling ``row_of_col_[j]``
    ///        with the row matched to column ``j``.
    void solve_assignment(int n);

    Config config_;
    std::unordered_map<uint64_t, Track> tracks_;
    uint64_t next_id_ = 0;

    // Preallocated per-frame scratch (M-5): capacity reused across frames so the
    // association step performs no heap allocation in steady state.
    std::vector<uint64_t> track_ids_;  ///< Snapshot of current track ids (rows).
    std::vector<double> cost_;         ///< Row-major n×n padded IoU cost matrix.
    std::vector<int> row_of_col_;      ///< Assignment result: column → row.
    std::vector<double> hung_u_, hung_v_, hung_minv_;  ///< Hungarian potentials / slack.
    std::vector<int> hung_p_, hung_way_;               ///< Hungarian matching / parent links.
    std::vector<char> hung_used_;                      ///< Hungarian visited flags.
};

} // namespace semaforo
