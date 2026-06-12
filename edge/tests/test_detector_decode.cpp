// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Blindaje de la decodificación YOLO (audit E-1 / E-2).
//
// The hazard: a corrupt, truncated, or unexpected ONNX model can hand back an
// output tensor whose advertised shape does NOT match the buffer it actually
// returned. The per-anchor decode indexes ``(4 + class) * num_anchors + i``;
// with an inconsistent shape that index runs past the buffer and the process
// segfaults — on a safety-critical signal, an unacceptable crash.
//
// ``Detector::decode_yolo_output`` is the single place raw model output is
// indexed, and it validates the tensor geometry against the real element count
// BEFORE dereferencing anything. These tests inject synthetic tensors with
// invalid shapes (simulating a corrupt model) and assert the decoder REJECTS
// them safely — it returns false, leaves the output empty, and never reads out
// of bounds — and that the node's fault contract drives the SafetyLayer into
// the deterministic emergency ring (fail-safe) rather than aborting.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "semaforo/detector.hpp"
#include "semaforo/safety_layer.hpp"

using semaforo::Detector;
using semaforo::Detection;
using semaforo::Phase;
using semaforo::SafetyLayer;

namespace {

// Build a well-formed YOLOv8 buffer [1, 4+num_classes, num_anchors], row-major,
// with a single car (COCO class 2) at the given anchor and a high score.
std::vector<float> make_valid_yolo(int num_classes, int num_anchors,
                                   int car_anchor) {
    const int channels = 4 + num_classes;
    std::vector<float> buf(static_cast<std::size_t>(channels) * num_anchors, 0.0f);
    auto at = [&](int ch, int anchor) -> float& {
        return buf[static_cast<std::size_t>(ch) * num_anchors + anchor];
    };
    // Box for the car anchor (cx, cy, w, h) in model-input pixels.
    at(0, car_anchor) = 320.0f;  // cx
    at(1, car_anchor) = 320.0f;  // cy
    at(2, car_anchor) = 64.0f;   // w
    at(3, car_anchor) = 48.0f;   // h
    // Class 2 (car) channel is 4 + 2 = 6; give it a confident score.
    at(4 + 2, car_anchor) = 0.9f;
    return buf;
}

} // namespace

// ── E-2: a well-formed tensor decodes correctly ────────────────────────────
TEST(DecodeYolo, ValidTensorDecodesVehicle) {
    const int num_classes = 80, num_anchors = 8, car_anchor = 3;
    std::vector<float> buf = make_valid_yolo(num_classes, num_anchors, car_anchor);
    std::array<std::int64_t, 3> shape{1, 4 + num_classes, num_anchors};

    std::vector<Detection> out;
    const bool ok = Detector::decode_yolo_output(
        buf.data(), buf.size(), shape.data(), shape.size(),
        /*frame_w=*/1280, /*frame_h=*/640, /*input_w=*/640, /*input_h=*/640,
        /*conf=*/0.5f, out);

    ASSERT_TRUE(ok);
    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].class_id, 2);
    EXPECT_FLOAT_EQ(out[0].confidence, 0.9f);
    // cx=320,w=64 in a 640-wide input rescaled to a 1280-wide frame (scale 2.0):
    //   x = (320 - 32) * 2 = 576 ; width = 64 * 2 = 128.
    EXPECT_FLOAT_EQ(out[0].bbox.x, 576.0f);
    EXPECT_FLOAT_EQ(out[0].bbox.width, 128.0f);
}

// ── E-2: the corrupt-model case — shape inconsistent with the buffer ────────
// This is the crash the blindaje exists to prevent: an 8-float buffer paired
// with a shape claiming [1, 84, 8400] (705,600 elements). Without the bound
// check the decode would index element ~705k of an 8-element buffer → segfault.
TEST(DecodeYolo, CorruptShapeExceedsBufferIsRejectedSafely) {
    std::vector<float> tiny(8, 0.0f);
    std::array<std::int64_t, 3> lying_shape{1, 84, 8400};

    std::vector<Detection> out;
    const bool ok = Detector::decode_yolo_output(
        tiny.data(), tiny.size(), lying_shape.data(), lying_shape.size(),
        1920, 1080, 640, 640, 0.5f, out);

    EXPECT_FALSE(ok);          // rejected — the test reaching here proves survival
    EXPECT_TRUE(out.empty());
}

// ── E-2: a battery of malformed geometries, none may crash or decode ────────
TEST(DecodeYolo, MalformedGeometriesAllRejected) {
    std::vector<float> buf(1024, 0.1f);
    std::vector<Detection> out;
    auto rejected = [&](const std::vector<std::int64_t>& shape) {
        const bool ok = Detector::decode_yolo_output(
            buf.data(), buf.size(), shape.data(), shape.size(),
            640, 640, 640, 640, 0.5f, out);
        return !ok && out.empty();
    };

    EXPECT_TRUE(rejected({1, 84}));            // wrong rank (2)
    EXPECT_TRUE(rejected({1, 84, 8, 1}));      // wrong rank (4)
    EXPECT_TRUE(rejected({2, 84, 8}));         // batch != 1
    EXPECT_TRUE(rejected({1, 4, 8}));          // no class channels (<=4)
    EXPECT_TRUE(rejected({1, 3, 8}));          // fewer than 4 box channels
    EXPECT_TRUE(rejected({1, 84, 0}));         // zero anchors
    EXPECT_TRUE(rejected({1, -84, 8}));        // negative channel dim (dynamic axis)
    EXPECT_TRUE(rejected({1, 84, -8}));        // negative anchor dim
    EXPECT_TRUE(rejected({1, 1 << 25, 1 << 25}));  // absurd dims (overflow guard)
}

// ── E-2: null pointers are rejected, not dereferenced ───────────────────────
TEST(DecodeYolo, NullInputsRejected) {
    std::array<std::int64_t, 3> shape{1, 84, 8};
    std::vector<float> buf(static_cast<std::size_t>(84) * 8, 0.0f);
    std::vector<Detection> out;

    EXPECT_FALSE(Detector::decode_yolo_output(
        nullptr, buf.size(), shape.data(), shape.size(),
        640, 640, 640, 640, 0.5f, out));
    EXPECT_FALSE(Detector::decode_yolo_output(
        buf.data(), buf.size(), nullptr, shape.size(),
        640, 640, 640, 640, 0.5f, out));
}

// ── E-1: an inference fault forces the deterministic emergency ring ─────────
// Mirrors the main loop's contract: when detection reports a fault (false), the
// single writer forces SafetyLayer emergency mode. The signal then runs the
// data-independent fixed ring (starting at ALL_RED clearance) — it can never be
// driven onto the requested green off garbage perception, and never freezes.
TEST(InferenceFault, CorruptOutputDrivesEmergencyRing) {
    // A corrupt model output (shape inconsistent with buffer) → fault.
    std::vector<float> tiny(4, 0.0f);
    std::array<std::int64_t, 3> lying_shape{1, 84, 8400};
    std::vector<Detection> out;
    const bool inference_ok = Detector::decode_yolo_output(
        tiny.data(), tiny.size(), lying_shape.data(), lying_shape.size(),
        1920, 1080, 640, 640, 0.5f, out);
    ASSERT_FALSE(inference_ok);

    // The node's fail-safe reaction to an inference fault.
    SafetyLayer safety;
    if (!inference_ok) {
        safety.set_emergency_mode(true);
    }
    ASSERT_TRUE(safety.in_emergency());

    // Even when the AI/policy requests a green, the emergency ring overrides it
    // with the safe clearance phase — never the requested conflicting green.
    auto validated = safety.validate(Phase::GREEN_NS);
    EXPECT_EQ(validated.phase, Phase::ALL_RED);
    EXPECT_TRUE(validated.was_modified);
}
