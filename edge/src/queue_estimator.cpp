// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/queue_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include <spdlog/spdlog.h>

namespace semaforo {

// ── Constructor / Destructor ────────────────────────────────────────────────
QueueEstimator::QueueEstimator() = default;
QueueEstimator::~QueueEstimator() = default;

// ── Public methods ──────────────────────────────────────────────────────────

void QueueEstimator::estimate(
    const std::vector<TrackedObject>& tracked_objects,
    const LaneConfig& lane_config,
    QueueState& state) {

    state.lanes.clear();  // reuse the caller's buffer (keeps capacity, no copy)
    state.timestamp = std::chrono::steady_clock::now();
    state.total_count = 0;
    state.max_density = 0.0f;
    state.max_wait_s = 0.0f;

    for (const auto& lane : lane_config.lanes) {
        LaneQueueState lane_state;
        lane_state.lane_id = lane.lane_id;
        lane_state.count = 0;
        lane_state.density = 0.0f;
        lane_state.avg_wait_s = 0.0f;

        std::vector<float> wait_times;

        for (const auto& obj : tracked_objects) {
            cv::Point2f center = bbox_center(obj.detection.bbox);

            if (point_in_polygon(center, lane.polygon)) {
                lane_state.count++;

                // Compute velocity magnitude
                float velocity = std::sqrt(
                    obj.velocity_x * obj.velocity_x +
                    obj.velocity_y * obj.velocity_y);

                // If vehicle is approximately stopped, compute wait time
                if (velocity < stopped_threshold_) {
                    auto now = std::chrono::steady_clock::now();
                    float wait = std::chrono::duration<float>(
                        now - obj.first_seen).count();
                    wait_times.push_back(wait);
                }
            }
        }

        // Compute density as fraction of max expected capacity per lane
        // Max capacity is estimated based on lane polygon area / typical vehicle size
        constexpr int MAX_VEHICLES_PER_LANE = 15;
        lane_state.density = static_cast<float>(lane_state.count) /
                             static_cast<float>(MAX_VEHICLES_PER_LANE);
        lane_state.density = std::clamp(lane_state.density, 0.0f, 1.0f);

        // Compute average wait time of stopped vehicles
        if (!wait_times.empty()) {
            lane_state.avg_wait_s = std::accumulate(
                wait_times.begin(), wait_times.end(), 0.0f) /
                static_cast<float>(wait_times.size());
        }

        // Update aggregated state
        state.total_count += lane_state.count;
        state.max_density = std::max(state.max_density, lane_state.density);
        state.max_wait_s = std::max(state.max_wait_s, lane_state.avg_wait_s);

        state.lanes.push_back(std::move(lane_state));
    }

    spdlog::debug("QueueEstimator: total={}, max_density={:.2f}, max_wait={:.1f}s",
                  state.total_count, state.max_density, state.max_wait_s);
}

void QueueEstimator::set_stopped_threshold(float threshold) {
    stopped_threshold_ = threshold;
}

// ── Private methods ─────────────────────────────────────────────────────────

bool QueueEstimator::point_in_polygon(const cv::Point2f& point,
                                       const std::vector<cv::Point2f>& polygon) {
    // Ray-casting algorithm
    bool inside = false;
    size_t n = polygon.size();

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((polygon[i].y > point.y) != (polygon[j].y > point.y)) &&
            (point.x < (polygon[j].x - polygon[i].x) *
                            (point.y - polygon[i].y) /
                            (polygon[j].y - polygon[i].y) +
                        polygon[i].x)) {
            inside = !inside;
        }
    }

    return inside;
}

cv::Point2f QueueEstimator::bbox_center(const BBox& bbox) {
    return cv::Point2f(
        bbox.x + bbox.width / 2.0f,
        bbox.y + bbox.height / 2.0f);
}

} // namespace semaforo
