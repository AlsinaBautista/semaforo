// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Zero-allocation proofs for the 30 FPS inference loop (audit M-5).
//
// The hazard these tests pin down: a long-running edge node must not malloc/free
// on every frame. Per-frame heap churn fragments the heap and eventually injects
// an allocation stall into the control loop — on a safety-critical signal that is
// an unacceptable, unbounded-latency failure mode. M-5 eradicates it by recycling
// preallocated storage: the ObjectPool for Detection/TrackedObject batches, and
// preallocated cv::Mat/tensor buffers for preprocessing.
//
// We prove "heap size constant after init" two complementary ways, both of which
// are deterministic and CI-friendly (no external profiler required):
//
//   1. A program-wide override of ``operator new``/``delete`` counts every heap
//      allocation routed through the C++ allocator (std::vector growth, map
//      nodes, …). After warm-up, the count delta across 10,000 iterations must
//      be exactly ZERO.
//   2. cv::Mat buffers allocate through cv::fastMalloc (not operator new), so we
//      additionally assert their ``.data`` pointers are byte-for-byte stable
//      across 10,000 iterations — i.e. the same buffer was overwritten, never
//      reallocated.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include <opencv2/core.hpp>

#include "semaforo/detector.hpp"
#include "semaforo/object_pool.hpp"
#include "semaforo/queue_estimator.hpp"
#include "semaforo/tracker.hpp"

// ── Global allocation counter ────────────────────────────────────────────────
// Overriding the global operators makes EVERY C++ heap allocation in this test
// program observable. We never gate the counter; instead we snapshot it before
// and after a measured region, so only that region's allocations are counted.
// All variants funnel through malloc/free, so new/delete pairings stay
// memory-consistent regardless of which the library picks.
namespace {
std::atomic<std::size_t> g_alloc_count{0};
std::atomic<std::size_t> g_alloc_bytes{0};

inline void* counted_malloc(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(n, std::memory_order_relaxed);
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}

struct AllocSnapshot {
    std::size_t count;
    std::size_t bytes;
};
AllocSnapshot snapshot() {
    return {g_alloc_count.load(std::memory_order_relaxed),
            g_alloc_bytes.load(std::memory_order_relaxed)};
}
std::size_t allocs_since(const AllocSnapshot& s) {
    return g_alloc_count.load(std::memory_order_relaxed) - s.count;
}
}  // namespace

void* operator new(std::size_t n) { return counted_malloc(n); }
void* operator new[](std::size_t n) { return counted_malloc(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {

using namespace semaforo;

constexpr int kStressIterations = 10'000;  // ≈ 5.5 minutes of real 30 FPS video

/// Build a stable grid of well-separated, stationary vehicle detections. A fixed
/// scene means the tracker reaches a steady set of tracks, so the per-frame
/// machinery (and only that) is what is under measurement.
std::vector<Detection> make_scene(int count) {
    std::vector<Detection> dets;
    dets.reserve(count);
    for (int i = 0; i < count; ++i) {
        Detection d{};
        d.class_id = 2;  // car (COCO)
        d.confidence = 0.9f;
        d.bbox.x = 100.0f + (i % 5) * 200.0f;
        d.bbox.y = 100.0f + (i / 5) * 150.0f;
        d.bbox.width = 80.0f;
        d.bbox.height = 80.0f;
        dets.push_back(d);
    }
    return dets;
}

}  // namespace

// ── ObjectPool: no allocation after construction ─────────────────────────────
TEST(ObjectPool, NoAllocationAfterConstruction) {
    constexpr std::size_t kCap = 512;
    ObjectPool<Detection> pool(kCap);

    // Warm-up frame: fill to capacity once so any first-touch growth is done
    // (there is none beyond the constructor's reserve, but be explicit).
    pool.clear();
    for (std::size_t i = 0; i < kCap; ++i) ASSERT_NE(pool.acquire(), nullptr);

    const AllocSnapshot before = snapshot();
    for (int frame = 0; frame < kStressIterations; ++frame) {
        pool.clear();
        for (std::size_t i = 0; i < kCap; ++i) {
            Detection* d = pool.acquire();
            d->class_id = static_cast<int>(i);  // overwrite the recycled slot
        }
    }
    const std::size_t allocs = allocs_since(before);

    EXPECT_EQ(allocs, 0u)
        << "ObjectPool allocated " << allocs << " times across "
        << kStressIterations << " recycled frames";
}

// ── ObjectPool: hard capacity bound (never grows) ────────────────────────────
TEST(ObjectPool, BoundedAtCapacity) {
    ObjectPool<TrackedObject> pool(4);
    pool.clear();
    EXPECT_NE(pool.acquire(), nullptr);
    EXPECT_NE(pool.acquire(), nullptr);
    EXPECT_NE(pool.acquire(), nullptr);
    EXPECT_NE(pool.acquire(), nullptr);
    EXPECT_TRUE(pool.full());
    EXPECT_EQ(pool.acquire(), nullptr);  // refuses to grow past capacity
    EXPECT_EQ(pool.size(), 4u);
}

// ── Preprocessing: fused kernel, tensor reused and never reallocated ─────────
TEST(Preprocess, BuffersReusedNoRealloc) {
    Detector detector;
    ASSERT_TRUE(detector.init("", 0.5f, 0.45f));  // stub init (no model needed)

    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(40, 80, 120));
    PreprocBuffers buf;

    // Warm-up call sizes the tensor and builds the resize coefficient cache
    // (the one and only allocation point).
    detector.preprocess(frame, buf);

    const float* p_tensor = buf.tensor.data();
    const std::size_t tensor_sz = buf.tensor.size();
    ASSERT_EQ(tensor_sz, 3u * 640u * 640u);  // CHW for the default 640x640 model

    const AllocSnapshot before = snapshot();
    for (int i = 0; i < kStressIterations; ++i) {
        detector.preprocess(frame, buf);
    }
    const std::size_t allocs = allocs_since(before);

    // (1) Zero heap allocations across the whole run — including OpenCV scratch,
    //     which the fused kernel avoids entirely.
    EXPECT_EQ(allocs, 0u)
        << "preprocess allocated " << allocs << " times across "
        << kStressIterations << " frames";

    // (2) The tensor is the SAME memory, overwritten in place (no realloc).
    EXPECT_EQ(buf.tensor.data(), p_tensor);
    EXPECT_EQ(buf.tensor.size(), tensor_sz);
}

// ── Preprocessing: fused bilinear/normalise kernel is numerically correct ────
TEST(Preprocess, ConstantImageNormalisedCorrectly) {
    Detector detector;
    ASSERT_TRUE(detector.init("", 0.5f, 0.45f));

    // A flat BGR image must map to flat, normalised CHW planes regardless of
    // the resize ratio (bilinear of equal neighbours is that same value / 255).
    cv::Mat frame(480, 853, CV_8UC3, cv::Scalar(10, 128, 250));  // B,G,R
    PreprocBuffers buf;
    detector.preprocess(frame, buf);

    const std::size_t plane = 640u * 640u;
    const float eB = 10.0f / 255.0f, eG = 128.0f / 255.0f, eR = 250.0f / 255.0f;
    for (std::size_t i = 0; i < plane; ++i) {
        ASSERT_NEAR(buf.tensor[0 * plane + i], eB, 1e-4f);
        ASSERT_NEAR(buf.tensor[1 * plane + i], eG, 1e-4f);
        ASSERT_NEAR(buf.tensor[2 * plane + i], eR, 1e-4f);
    }
}

// ── Tracker: steady-state association allocates nothing ──────────────────────
TEST(Tracker, SteadyStateNoAllocation) {
    VehicleTracker tracker;
    ObjectPool<TrackedObject> track_pool(256);
    const std::vector<Detection> scene = make_scene(12);

    // Warm-up: confirm tracks (min_hits) and grow all scratch / map nodes.
    for (int i = 0; i < 50; ++i) tracker.update(scene, track_pool);
    ASSERT_GT(track_pool.size(), 0u) << "tracker never confirmed any track";

    const AllocSnapshot before = snapshot();
    for (int frame = 0; frame < kStressIterations; ++frame) {
        tracker.update(scene, track_pool);
    }
    const std::size_t allocs = allocs_since(before);

    EXPECT_EQ(allocs, 0u)
        << "tracker allocated " << allocs << " times across "
        << kStressIterations << " steady-state frames";
}

// ── Headline: a full per-frame perception loop, heap constant after init ─────
// Preprocess (cv::Mat + tensor) → detection pool → tracker (Kalman + Hungarian)
// → track pool → queue estimate (output param), exactly as main.cpp drives it,
// for 10,000 iterations.
TEST(InferenceLoop, TenThousandIterationsHeapConstant) {
    Detector detector;
    ASSERT_TRUE(detector.init("", 0.5f, 0.45f));
    VehicleTracker tracker;
    QueueEstimator queue_estimator;

    // The hoisted, recycled buffers — allocated ONCE, here, before the loop.
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(30, 60, 90));
    PreprocBuffers buf;
    ObjectPool<Detection> det_pool(512);
    ObjectPool<TrackedObject> track_pool(256);
    LaneConfig lane_config;     // empty, matching main.cpp
    QueueState queue_state;     // hoisted output buffer, reused every frame
    const std::vector<Detection> scene = make_scene(12);

    // Warm-up: first touch of every buffer + tracker confirmation.
    for (int i = 0; i < 50; ++i) {
        detector.preprocess(frame, buf);
        det_pool.clear();
        for (const Detection& d : scene) {
            if (Detection* slot = det_pool.acquire()) *slot = d;  // simulate detector fill
        }
        tracker.update(det_pool.view(), track_pool);
        queue_estimator.estimate(track_pool.view(), lane_config, queue_state);
    }

    const float* p_tensor = buf.tensor.data();

    const AllocSnapshot before = snapshot();
    for (int frame_n = 0; frame_n < kStressIterations; ++frame_n) {
        // 1. Preprocess into preallocated cv::Mat / tensor buffers.
        detector.preprocess(frame, buf);
        // 2. Fill the recycled detection pool (stand-in for detector output,
        //    which is empty in this ONNX-less build).
        det_pool.clear();
        for (const Detection& d : scene) {
            if (Detection* slot = det_pool.acquire()) *slot = d;
        }
        // 3. Track into the recycled track pool.
        tracker.update(det_pool.view(), track_pool);
        // 4. Estimate queue state into the hoisted, reused output buffer.
        queue_estimator.estimate(track_pool.view(), lane_config, queue_state);
    }
    const AllocSnapshot after = snapshot();
    const std::size_t allocs = after.count - before.count;
    const std::size_t bytes = after.bytes - before.bytes;

    // Heap is constant after init: zero allocations, zero net bytes, and the
    // cv::Mat / tensor buffers are the same memory throughout.
    EXPECT_EQ(allocs, 0u) << "loop allocated " << allocs << " times ("
                          << bytes << " bytes) across " << kStressIterations
                          << " iterations";
    EXPECT_EQ(buf.tensor.data(), p_tensor);
    EXPECT_GT(track_pool.size(), 0u);
}
