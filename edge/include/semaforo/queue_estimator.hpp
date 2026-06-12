// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "semaforo/tracker.hpp"

namespace semaforo {

/// @brief Defines a lane region in the camera's field of view.
struct LaneRegion {
    std::string lane_id;             ///< Unique lane identifier (e.g., "NS_lane_1")
    std::vector<cv::Point2f> polygon; ///< Polygon vertices defining the lane region in pixel coords
    std::string direction;            ///< Traffic direction (e.g., "NS", "EW", "NS_left", "EW_left")
};

/// @brief Configuration describing all lane regions at an intersection.
struct LaneConfig {
    std::vector<LaneRegion> lanes;  ///< All monitored lane regions
};

/// @brief Queue state for a single lane.
struct LaneQueueState {
    std::string lane_id;    ///< Lane identifier
    int count;              ///< Number of vehicles in the lane
    float density;          ///< Vehicle density (vehicles per unit length) in [0, 1]
    float avg_wait_s;       ///< Average wait time in seconds for vehicles in this lane
};

/// @brief Aggregated queue state across all lanes at the intersection.
struct QueueState {
    std::vector<LaneQueueState> lanes;  ///< Per-lane queue states
    int total_count;                    ///< Total vehicle count across all lanes
    float max_density;                  ///< Maximum density across all lanes
    float max_wait_s;                   ///< Maximum average wait time across all lanes
    std::chrono::steady_clock::time_point timestamp; ///< When this estimate was produced
};

/// @brief Estimates queue length and density per lane from tracked objects.
///
/// Uses tracked object positions and lane region polygons to count vehicles
/// in each lane, compute density metrics, and estimate average wait times
/// based on how long each tracked vehicle has been stationary.
class QueueEstimator {
public:
    QueueEstimator();
    ~QueueEstimator();

    /// @brief Estimate the current queue state from tracked objects and lane configuration.
    ///
    /// For each lane region, determines which tracked objects fall within the
    /// polygon, counts them, computes density as count/max_capacity, and
    /// averages the wait time of stationary vehicles.
    ///
    /// Writes into a caller-owned ``out_state`` instead of returning by value,
    /// so the main loop can hoist the buffer and reuse it every frame with no
    /// per-frame copy/allocation (audit M-5).
    ///
    /// @param tracked_objects  Currently active tracked objects from VehicleTracker.
    /// @param lane_config      Lane region definitions for the intersection.
    /// @param out_state        Aggregated state; overwritten in place each call.
    void estimate(const std::vector<TrackedObject>& tracked_objects,
                  const LaneConfig& lane_config,
                  QueueState& out_state);

    /// @brief Set the velocity threshold below which a vehicle is considered stopped.
    /// @param threshold  Velocity magnitude threshold in pixels/frame (default: 2.0).
    void set_stopped_threshold(float threshold);

private:
    /// @brief Check if a point lies inside a polygon using ray-casting.
    /// @param point    The 2D point to test.
    /// @param polygon  Polygon vertices.
    /// @return true if the point is inside the polygon.
    static bool point_in_polygon(const cv::Point2f& point,
                                 const std::vector<cv::Point2f>& polygon);

    /// @brief Get the center point of a bounding box.
    /// @param bbox  The bounding box.
    /// @return Center point as cv::Point2f.
    static cv::Point2f bbox_center(const BBox& bbox);

    float stopped_threshold_ = 2.0f; ///< Velocity threshold for "stopped" classification
};

} // namespace semaforo
