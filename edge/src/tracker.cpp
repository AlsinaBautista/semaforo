// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace semaforo {

namespace {

/// Hard bound on confirmed tracks emitted per frame, bounding the result pool
/// (an intersection holds a finite number of vehicles; excess is dropped).
constexpr std::size_t kMaxTracks = 256;

/// Cost assigned to gated-out (low-IoU or cross-class) pairs and to padding
/// cells of the square assignment matrix. Any real pair costs ≤ 1, so an
/// assignment is a true match iff its cost is below this.
constexpr double kInfeasibleCost = 1e6;

using Vec4 = cv::Matx<float, 4, 1>;
using Mat4 = cv::Matx<float, 4, 4>;
using Mat47 = cv::Matx<float, 4, 7>;
using Mat74 = cv::Matx<float, 7, 4>;

/// Constant-velocity transition: u += u̇, v += v̇, s += ṡ (one frame step).
KalmanBoxFilter::Mat7 make_transition() {
    auto f = KalmanBoxFilter::Mat7::eye();
    f(0, 4) = 1.0f;
    f(1, 5) = 1.0f;
    f(2, 6) = 1.0f;
    return f;
}

/// Observation model: the measurement is the first four state components.
Mat47 make_observation() {
    Mat47 h = Mat47::zeros();
    h(0, 0) = 1.0f;
    h(1, 1) = 1.0f;
    h(2, 2) = 1.0f;
    h(3, 3) = 1.0f;
    return h;
}

// Noise parameters follow the SORT reference implementation (Bewley et al.):
// high initial uncertainty on the unobservable velocities, low process noise
// on the scale rate, and a measurement model that trusts position more than
// scale/aspect.
KalmanBoxFilter::Mat7 make_process_noise() {
    auto q = KalmanBoxFilter::Mat7::eye();
    q(4, 4) = 0.01f;
    q(5, 5) = 0.01f;
    q(6, 6) = 1e-4f;
    return q;
}

Mat4 make_measurement_noise() {
    auto r = Mat4::eye();
    r(2, 2) = 10.0f;
    r(3, 3) = 10.0f;
    return r;
}

const KalmanBoxFilter::Mat7 kF = make_transition();
const Mat47 kH = make_observation();
const KalmanBoxFilter::Mat7 kQ = make_process_noise();
const Mat4 kR = make_measurement_noise();

/// Convert a pixel-space box to the measurement vector [u, v, s, r].
Vec4 bbox_to_z(const BBox& b) {
    const float u = b.x + b.width * 0.5f;
    const float v = b.y + b.height * 0.5f;
    const float s = b.width * b.height;
    const float r = b.height > 0.0f ? b.width / b.height : 1.0f;
    return {u, v, s, r};
}

} // namespace

// ── KalmanBoxFilter ──────────────────────────────────────────────────────────

void KalmanBoxFilter::init(const BBox& box) {
    const Vec4 z = bbox_to_z(box);
    x = Vec7::zeros();
    x(0) = z(0);
    x(1) = z(1);
    x(2) = z(2);
    x(3) = z(3);

    P = Mat7::eye() * 10.0f;
    P(4, 4) = 1e4f;
    P(5, 5) = 1e4f;
    P(6, 6) = 1e4f;
}

void KalmanBoxFilter::predict(float dt) {
    // Scale guard: never let the constant-velocity extrapolation drive the
    // area non-positive (a degenerate box would poison the IoU gate).
    if (x(2) + x(6) * dt <= 0.0f) {
        x(6) = 0.0f;
    }

    // At the design cadence (dt == 1) use the cached unit-step transition kF
    // verbatim — the hot path is byte-identical to the original SORT filter.
    // When the pipeline has throttled and a frame interval spans dt > 1 nominal
    // frames, build the scaled transition so the centre/scale advance by
    // velocity × dt: the predicted box coasts the full distance the vehicle
    // actually travelled, keeping it inside the IoU gate of the next detection.
    if (dt == 1.0f) {
        x = kF * x;
        P = kF * P * kF.t() + kQ;
    } else {
        Mat7 f = Mat7::eye();
        f(0, 4) = dt;
        f(1, 5) = dt;
        f(2, 6) = dt;
        x = f * x;
        P = f * P * f.t() + kQ;
    }
}

void KalmanBoxFilter::update(const BBox& box) {
    const Vec4 y = bbox_to_z(box) - kH * x;          // innovation
    const Mat4 s = kH * P * kH.t() + kR;             // innovation covariance
    const Mat74 k = P * kH.t() * s.inv();            // Kalman gain
    x += k * y;
    P = (Mat7::eye() - k * kH) * P;
}

BBox KalmanBoxFilter::bbox() const {
    const float s = std::max(x(2), 1e-3f);
    const float r = std::max(x(3), 1e-3f);
    const float w = std::sqrt(s * r);
    const float h = s / w;
    return {x(0) - w * 0.5f, x(1) - h * 0.5f, w, h};
}

// ── VehicleTracker ───────────────────────────────────────────────────────────

VehicleTracker::VehicleTracker() : config_(Config{}) {}

VehicleTracker::VehicleTracker(const Config& config) : config_(config) {}

VehicleTracker::~VehicleTracker() = default;

float VehicleTracker::compute_iou(const BBox& a, const BBox& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);

    float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float union_area = a.width * a.height + b.width * b.height - inter_area;

    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

void VehicleTracker::solve_assignment(int n) {
    // Kuhn–Munkres with potentials (O(n³) minimisation) over the square
    // row-major matrix ``cost_``. 1-indexed internally; column 0 is the
    // virtual source. Scratch vectors only ever grow, so steady-state frames
    // reuse their capacity (M-5).
    constexpr double kInf = std::numeric_limits<double>::max() / 4.0;

    hung_u_.assign(n + 1, 0.0);
    hung_v_.assign(n + 1, 0.0);
    hung_p_.assign(n + 1, 0);
    hung_way_.assign(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        hung_p_[0] = i;
        int j0 = 0;
        hung_minv_.assign(n + 1, kInf);
        hung_used_.assign(n + 1, 0);
        do {
            hung_used_[j0] = 1;
            const int i0 = hung_p_[j0];
            double delta = kInf;
            int j1 = 0;
            for (int j = 1; j <= n; ++j) {
                if (hung_used_[j]) continue;
                const double cur =
                    cost_[static_cast<std::size_t>(i0 - 1) * n + (j - 1)] -
                    hung_u_[i0] - hung_v_[j];
                if (cur < hung_minv_[j]) {
                    hung_minv_[j] = cur;
                    hung_way_[j] = j0;
                }
                if (hung_minv_[j] < delta) {
                    delta = hung_minv_[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (hung_used_[j]) {
                    hung_u_[hung_p_[j]] += delta;
                    hung_v_[j] -= delta;
                } else {
                    hung_minv_[j] -= delta;
                }
            }
            j0 = j1;
        } while (hung_p_[j0] != 0);
        do {
            const int j1 = hung_way_[j0];
            hung_p_[j0] = hung_p_[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    row_of_col_.assign(n, -1);
    for (int j = 1; j <= n; ++j) {
        row_of_col_[j - 1] = hung_p_[j] - 1;
    }
}

std::vector<TrackedObject> VehicleTracker::update(const std::vector<Detection>& detections,
                                                  float dt) {
    // Convenience form: delegate to the pooled path, then copy out. Allocates a
    // result vector, so it is intended for off-hot-path callers and tests only.
    ObjectPool<TrackedObject> pool(kMaxTracks);
    update(detections, pool, dt);
    return {pool.view().begin(), pool.view().end()};
}

void VehicleTracker::update(const std::vector<Detection>& detections,
                            ObjectPool<TrackedObject>& out,
                            float dt) {
    out.clear();
    auto now = std::chrono::steady_clock::now();

    // 1. Predict: advance every track's Kalman filter one frame and surface
    //    the predicted box as its current position. Occluded tracks therefore
    //    coast along their trajectory instead of freezing in place — this is
    //    what lets a reappearing vehicle re-match its original ID (M-3).
    track_ids_.clear();
    for (auto& pair : tracks_) {
        Track& track = pair.second;
        track.kf.predict(dt);
        track.obj.detection.bbox = track.kf.bbox();
        track.obj.age++;
        track.obj.frames_since_seen++;
        track_ids_.push_back(pair.first);
    }

    const std::size_t n_tracks = track_ids_.size();
    const std::size_t n_dets = detections.size();

    // 2. Associate detections (columns) to predicted tracks (rows) by optimal
    //    assignment over an IoU cost matrix, padded to square. Pairs failing
    //    the IoU gate or the class check are infeasible; padding cells are too.
    if (n_tracks > 0 && n_dets > 0) {
        const std::size_t n = std::max(n_tracks, n_dets);
        cost_.assign(n * n, kInfeasibleCost);
        for (std::size_t k = 0; k < n_tracks; ++k) {
            const Track& track = tracks_[track_ids_[k]];
            for (std::size_t d = 0; d < n_dets; ++d) {
                if (detections[d].class_id != track.obj.detection.class_id) continue;
                const float iou = compute_iou(detections[d].bbox, track.obj.detection.bbox);
                if (iou > config_.iou_threshold) {
                    cost_[k * n + d] = 1.0 - static_cast<double>(iou);
                }
            }
        }
        solve_assignment(static_cast<int>(n));
    } else {
        row_of_col_.assign(n_dets, -1);
    }

    // 3./4. Update matched tracks; spawn a new track per unmatched detection.
    for (std::size_t d = 0; d < n_dets; ++d) {
        const int row = row_of_col_[d];
        const bool matched =
            row >= 0 && static_cast<std::size_t>(row) < n_tracks &&
            cost_[static_cast<std::size_t>(row) * std::max(n_tracks, n_dets) + d] <
                kInfeasibleCost;

        if (matched) {
            Track& track = tracks_[track_ids_[static_cast<std::size_t>(row)]];
            track.kf.update(detections[d].bbox);

            track.obj.detection = detections[d];
            track.obj.detection.bbox = track.kf.bbox();  // posterior (smoothed) box
            track.obj.velocity_x = track.kf.vx();
            track.obj.velocity_y = track.kf.vy();
            track.obj.frames_since_seen = 0;
            track.obj.last_seen = now;
        } else {
            uint64_t new_id = ++next_id_;
            Track new_track;
            new_track.kf.init(detections[d].bbox);
            new_track.obj.track_id = new_id;
            new_track.obj.detection = detections[d];
            new_track.obj.age = 1;
            new_track.obj.frames_since_seen = 0;
            new_track.obj.velocity_x = 0.0f;
            new_track.obj.velocity_y = 0.0f;
            new_track.obj.first_seen = now;
            new_track.obj.last_seen = now;

            tracks_[new_id] = new_track;
        }
    }

    // 5. Remove dead tracks (occlusion tolerance: a track survives up to
    //    max_age frames unseen, coasting on its prediction).
    for (auto it = tracks_.begin(); it != tracks_.end();) {
        if (it->second.obj.frames_since_seen > config_.max_age) {
            it = tracks_.erase(it);
        } else {
            ++it;
        }
    }

    // 6. Emit confirmed tracks into the caller's pool.
    for (const auto& pair : tracks_) {
        if (pair.second.obj.age >= config_.min_hits) {
            if (TrackedObject* slot = out.acquire()) {
                *slot = pair.second.obj;
            } else {
                break;  // pool full — deterministic bound
            }
        }
    }
}

const std::unordered_map<uint64_t, VehicleTracker::Track>&
VehicleTracker::get_all_tracks() const {
    return tracks_;
}

void VehicleTracker::reset() {
    tracks_.clear();
}

uint64_t VehicleTracker::total_tracks_created() const {
    return next_id_;
}

} // namespace semaforo
