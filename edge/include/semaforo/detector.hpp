// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "semaforo/object_pool.hpp"

namespace semaforo {

/// @brief Bounding box in pixel coordinates (top-left origin).
struct BBox {
    float x;      ///< Top-left x coordinate
    float y;      ///< Top-left y coordinate
    float width;  ///< Width of the bounding box
    float height; ///< Height of the bounding box
};

/// @brief A single object detection result produced by the Detector.
struct Detection {
    BBox bbox;         ///< Bounding box of the detected object
    int class_id;      ///< Class identifier (e.g., 0=car, 1=truck, 2=bus, 3=motorcycle, 4=pedestrian)
    float confidence;  ///< Detection confidence score in [0, 1]
};

/// @brief Caller-owned, preallocated output buffer for preprocessing (M-5).
///
/// Holds the flattened CHW input tensor produced by ``Detector::preprocess``.
/// The owner (the main loop) keeps one of these for the whole process lifetime
/// and passes it back in every tick; it is sized **once** (first call) and
/// overwritten in place thereafter, so preprocessing performs no per-frame
/// heap allocation. The resize coefficients are cached inside the Detector
/// (they depend only on the input/output dimensions, not on the caller), so the
/// fused resize+normalize+HWC→CHW pass touches no intermediate ``cv::Mat`` and
/// makes no OpenCV-internal scratch allocation either.
struct PreprocBuffers {
    std::vector<float> tensor;  ///< Flattened CHW input tensor fed to the model.
};

/// @brief ONNX Runtime-backed object detector for vehicles and pedestrians.
///
/// This class wraps an ONNX Runtime inference session to perform real-time
/// object detection on video frames captured from intersection cameras.
/// It supports YOLOv8/v9 style models exported to ONNX format.
class Detector {
public:
    Detector();
    ~Detector();

    // Non-copyable, movable
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;
    Detector(Detector&&) noexcept;
    Detector& operator=(Detector&&) noexcept;

    /// @brief Initialize the detector by loading an ONNX model.
    ///
    /// Creates an ONNX Runtime inference session with the specified model file.
    /// Configures execution providers (CUDA if available, CPU fallback).
    ///
    /// @param model_path  Filesystem path to the .onnx model file.
    /// @param conf_threshold  Minimum confidence threshold for detections (default: 0.5).
    /// @param nms_threshold   IoU threshold for non-maximum suppression (default: 0.45).
    /// @return true if initialization succeeded, false otherwise.
    bool init(const std::string& model_path,
              float conf_threshold = 0.5f,
              float nms_threshold = 0.45f);

    /// @brief Run object detection on a single video frame (convenience form).
    ///
    /// Preprocesses the input frame (resize, normalize, HWC→CHW), runs
    /// inference through the ONNX model, and applies NMS to filter results.
    /// Allocates a fresh result vector; prefer the pooled overload below on the
    /// hot path.
    ///
    /// @param frame  BGR image (cv::Mat) from the video pipeline.
    /// @return Vector of Detection results above the confidence threshold.
    std::vector<Detection> detect(const cv::Mat& frame);

    /// @brief Allocation-free detection: fill a caller-owned ObjectPool (M-5).
    ///
    /// Identical inference to ``detect(frame)`` but writes survivors into a
    /// recycled ``ObjectPool<Detection>`` instead of returning a freshly
    /// allocated vector. Combined with a hoisted ``PreprocBuffers`` (see
    /// ``preprocess``) and reusable internal scratch, this path performs **no
    /// heap allocation** in steady state. The pool is cleared first; detections
    /// beyond its capacity are dropped (deterministic bound).
    ///
    /// **Inference health contract (audit E-1).** The ONNX session call is
    /// wrapped so it can never throw out of this function and never abort the
    /// process: any ``Ort::Exception``/``std::exception`` from ``Run()``, or a
    /// model whose output tensor fails the geometry checks in
    /// ``decode_yolo_output``, is caught here and reported as a *fault*. On a
    /// fault the pool is cleared and the method returns ``false`` — the single
    /// writer (main loop) MUST react by forcing the SafetyLayer's emergency ring
    /// (fail-safe), rather than driving the signal on stale/garbage perception.
    ///
    /// @param frame  BGR image (cv::Mat) from the video pipeline.
    /// @param out    Pool to clear and fill with above-threshold detections.
    /// @return true if inference completed cleanly; false on an inference fault
    ///         (exception or malformed model output) — caller must fail safe.
    bool detect(const cv::Mat& frame, ObjectPool<Detection>& out);

    /// @brief Preprocess a frame into the model's CHW input tensor (M-5).
    ///
    /// Runs a single fused pass — bilinear resize to the model input size,
    /// normalise to [0,1], and HWC→CHW de-interleave — writing the result
    /// straight into ``buf.tensor``. Resize coefficients are precomputed and
    /// cached on first use (and only rebuilt if the input frame size changes),
    /// and the tensor is sized once, so after warm-up the call performs **no
    /// heap allocation** (not even OpenCV-internal scratch). Exposed publicly so
    /// the zero-allocation guarantee can be verified independently of ONNX
    /// Runtime availability.
    ///
    /// @param frame  BGR image (cv::Mat, CV_8UC3) from the video pipeline.
    /// @param buf    Preallocated output; ``buf.tensor`` overwritten each call.
    void preprocess(const cv::Mat& frame, PreprocBuffers& buf);

    /// @brief Decode a raw YOLOv8 output tensor into detections — safely (E-2).
    ///
    /// Pure, ONNX-independent, and self-validating: this is the **only** place
    /// raw model output is indexed, and it verifies the tensor's geometry against
    /// the actual buffer length *before* dereferencing anything. A corrupt or
    /// unexpected model can advertise a shape that does not match the data it
    /// returned; without these checks the per-anchor index
    /// ``(4 + class) * num_anchors + i`` would read out of bounds and segfault.
    /// Here every dimension is bounds-checked and the maximum flat index this
    /// routine will ever compute is proven ``<= data_len`` first, so a malformed
    /// tensor is rejected (returns false, ``out`` left empty) instead of crashing.
    ///
    /// Expected layout: row-major ``[1, 4 + num_classes, num_anchors]`` (YOLOv8).
    /// Vehicular COCO classes (2,3,5,7) above @p conf_threshold are emitted, with
    /// boxes rescaled from the model input size back to the source frame size.
    ///
    /// @param data          Pointer to the flat float output buffer (may be null).
    /// @param data_len      Number of float elements actually backing @p data.
    /// @param shape          Pointer to the tensor shape dims (may be null).
    /// @param shape_len      Number of entries in @p shape (expected 3).
    /// @param frame_w        Source frame width  (for rescaling boxes).
    /// @param frame_h        Source frame height (for rescaling boxes).
    /// @param input_w        Model input width.
    /// @param input_h        Model input height.
    /// @param conf_threshold Minimum class confidence to keep a detection.
    /// @param out            Cleared, then filled with decoded detections.
    /// @return true if the tensor was well-formed and decoded; false if it failed
    ///         validation (rejected without any out-of-bounds access).
    static bool decode_yolo_output(const float* data, std::size_t data_len,
                                   const std::int64_t* shape, std::size_t shape_len,
                                   int frame_w, int frame_h,
                                   int input_w, int input_h,
                                   float conf_threshold,
                                   std::vector<Detection>& out);

    /// @brief Check whether the detector has been successfully initialized.
    /// @return true if a model is loaded and ready for inference.
    bool is_initialized() const;

    /// @brief Get the expected input size of the loaded model.
    /// @return Input dimensions as cv::Size (width, height).
    cv::Size get_input_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_; ///< PIMPL for ONNX Runtime internals
};

} // namespace semaforo
