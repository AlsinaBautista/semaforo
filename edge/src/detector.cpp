// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/detector.hpp"

#include <spdlog/spdlog.h>

// ONNX Runtime headers (conditionally included)
#ifdef ONNXRUNTIME_AVAILABLE
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <opencv2/imgproc.hpp>

namespace semaforo {

namespace {
/// Hard bound on detections retained per frame. Bounds the per-frame ObjectPool
/// so the result store stays O(1) in memory (an intersection's field of view
/// can only hold so many vehicles; anything past this is dropped, never grown).
constexpr std::size_t kMaxDetections = 512;
} // namespace

// ── PIMPL implementation ────────────────────────────────────────────────────
struct Detector::Impl {
#ifdef ONNXRUNTIME_AVAILABLE
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "semaforo_detector"};
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
#endif

    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    cv::Size input_size{640, 640};
    bool initialized = false;
    bool stub_mode = false;      ///< No model: detect() emits nothing, allocates nothing.

    // ── Preallocated per-frame scratch (M-5) ────────────────────────────────
    // Sized once, overwritten in place every frame: no per-frame heap churn.
    PreprocBuffers buffers_;             ///< Input tensor (owned by the detector).
    std::vector<Detection> raw_;         ///< Pre-NMS detections (capacity reused).
    std::vector<char> suppressed_;       ///< NMS suppression flags (capacity reused).

    // ── Cached bilinear resize coefficients (M-5) ───────────────────────────
    // Built once for a given (source size → input_size) mapping and reused for
    // every frame. Rebuilt only if the incoming frame size changes (rare). This
    // is what lets ``preprocess`` resize without cv::resize's per-call scratch.
    int src_w_ = 0, src_h_ = 0;          ///< Frame size the coefficients are built for.
    std::vector<int> x_idx_, y_idx_;     ///< Left/top source index per output column/row.
    std::vector<float> x_w_, y_w_;       ///< Bilinear fraction per output column/row.

    /// @brief (Re)build the resize coefficient tables for a frame size.
    void build_resize_coeffs(int src_w, int src_h) {
        const int dst_w = input_size.width;
        const int dst_h = input_size.height;
        const float scale_x = static_cast<float>(src_w) / dst_w;
        const float scale_y = static_cast<float>(src_h) / dst_h;

        x_idx_.resize(dst_w);
        x_w_.resize(dst_w);
        for (int ox = 0; ox < dst_w; ++ox) {
            float fx = (ox + 0.5f) * scale_x - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            float wx = fx - x0;
            if (x0 < 0) { x0 = 0; wx = 0.0f; }
            if (x0 > src_w - 2) { x0 = src_w - 2; wx = 1.0f; }
            x_idx_[ox] = x0;
            x_w_[ox] = wx;
        }

        y_idx_.resize(dst_h);
        y_w_.resize(dst_h);
        for (int oy = 0; oy < dst_h; ++oy) {
            float fy = (oy + 0.5f) * scale_y - 0.5f;
            int y0 = static_cast<int>(std::floor(fy));
            float wy = fy - y0;
            if (y0 < 0) { y0 = 0; wy = 0.0f; }
            if (y0 > src_h - 2) { y0 = src_h - 2; wy = 1.0f; }
            y_idx_[oy] = y0;
            y_w_[oy] = wy;
        }

        src_w_ = src_w;
        src_h_ = src_h;
    }

    /// @brief Apply non-maximum suppression, emitting survivors into a pool.
    ///
    /// Sorts ``detections`` in place and pushes the kept boxes into ``out``
    /// (a recycled ObjectPool). Uses the reusable ``suppressed_`` flag buffer
    /// rather than allocating a fresh ``std::vector<bool>`` per frame.
    void apply_nms(std::vector<Detection>& detections, ObjectPool<Detection>& out) {
        if (detections.empty()) return;

        // Sort by confidence descending
        std::sort(detections.begin(), detections.end(),
                  [](const Detection& a, const Detection& b) {
                      return a.confidence > b.confidence;
                  });

        suppressed_.assign(detections.size(), 0);  // reuses capacity

        for (size_t i = 0; i < detections.size(); ++i) {
            if (suppressed_[i]) continue;
            if (Detection* slot = out.acquire()) {
                *slot = detections[i];
            } else {
                break;  // pool full — deterministic bound, drop the rest
            }

            for (size_t j = i + 1; j < detections.size(); ++j) {
                if (suppressed_[j]) continue;
                if (detections[i].class_id != detections[j].class_id) continue;

                float iou = compute_iou(detections[i].bbox, detections[j].bbox);
                if (iou > nms_threshold) {
                    suppressed_[j] = 1;
                }
            }
        }
    }

    /// @brief Compute IoU between two bounding boxes.
    static float compute_iou(const BBox& a, const BBox& b) {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.width, b.x + b.width);
        float y2 = std::min(a.y + a.height, b.y + b.height);

        float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float union_area = a.width * a.height + b.width * b.height - inter_area;

        return union_area > 0.0f ? inter_area / union_area : 0.0f;
    }
};

// ── Constructor / Destructor ────────────────────────────────────────────────
Detector::Detector() : impl_(std::make_unique<Impl>()) {}
Detector::~Detector() = default;
Detector::Detector(Detector&&) noexcept = default;
Detector& Detector::operator=(Detector&&) noexcept = default;

// ── Public methods ──────────────────────────────────────────────────────────

bool Detector::init(const std::string& model_path,
                    float conf_threshold,
                    float nms_threshold) {
    impl_->conf_threshold = conf_threshold;
    impl_->nms_threshold = nms_threshold;

    // Stub mode: an empty or nonexistent model path means "run without a
    // model" — regardless of whether ONNX Runtime is linked into the build.
    // No session is created and detect() emits no detections and performs no
    // heap allocation, which is what the zero-allocation loop proofs rely on.
    std::error_code fs_ec;
    if (model_path.empty() || !std::filesystem::exists(model_path, fs_ec)) {
        impl_->stub_mode = true;
        impl_->initialized = true;
        spdlog::warn("Detector: stub mode — model path '{}' is empty or does "
                     "not exist; no detections will be emitted.", model_path);
        return true;
    }

#ifdef ONNXRUNTIME_AVAILABLE
    try {
        impl_->session_options.SetIntraOpNumThreads(2);
        impl_->session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Thermal-throttling mitigation: prefer the TensorRT execution provider
        // in INT8 precision. INT8 kernels draw far less power and produce far
        // less heat than FP32/FP16, which keeps the edge SoC inside its
        // sustained-clock thermal envelope and avoids the throttling that
        // collapses the inference frame rate. The strict batch size of 1 (the
        // input tensor is always shaped {1,3,H,W} in ``detect``) minimises the
        // per-inference latency spike that drives transient junction-temperature
        // peaks. TensorRT is only present on the NVIDIA edge hardware (e.g.
        // Jetson); on any other build (this macOS dev box, plain-CPU targets)
        // registering the provider throws — we catch it and fall back to the
        // default CPU provider rather than failing the node. Fail-safe over
        // cleverness: a missing accelerator must degrade frame rate, never take
        // the intersection down.
        try {
            OrtTensorRTProviderOptions trt_options{};
            trt_options.device_id = 0;
            trt_options.trt_int8_enable = 1;   // force INT8 precision
            trt_options.trt_fp16_enable = 0;
            trt_options.trt_max_workspace_size = 1ULL << 30;  // 1 GiB
            impl_->session_options.AppendExecutionProvider_TensorRT(trt_options);
            spdlog::info("Detector: TensorRT EP enabled (INT8, batch=1).");
        } catch (const Ort::Exception& trt_ex) {
            spdlog::warn("Detector: TensorRT EP unavailable ({}); "
                         "falling back to CPU provider.", trt_ex.what());
        }

        impl_->session = std::make_unique<Ort::Session>(
            impl_->env, model_path.c_str(), impl_->session_options);

        // Query input/output names and shapes
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = impl_->session->GetInputNameAllocated(0, allocator);
        impl_->input_names.push_back(input_name.get());

        auto output_name = impl_->session->GetOutputNameAllocated(0, allocator);
        impl_->output_names.push_back(output_name.get());

        // Infer input size from model
        auto input_shape = impl_->session->GetInputTypeInfo(0)
                               .GetTensorTypeAndShapeInfo()
                               .GetShape();
        if (input_shape.size() >= 4) {
            impl_->input_size = cv::Size(
                static_cast<int>(input_shape[3]),
                static_cast<int>(input_shape[2]));
        }

        impl_->initialized = true;
        spdlog::info("Detector: ONNX model loaded from {}", model_path);
        spdlog::info("Detector: input size = {}x{}", impl_->input_size.width,
                     impl_->input_size.height);
        return true;

    } catch (const Ort::Exception& ex) {
        spdlog::error("Detector: ONNX Runtime error: {}", ex.what());
        return false;
    }
#else
    spdlog::warn("Detector: ONNX Runtime not available. Model not loaded: {}", model_path);
    // Stub: mark as initialized for development/testing without model
    impl_->initialized = true;
    return true;
#endif
}

void Detector::preprocess(const cv::Mat& frame, PreprocBuffers& buf) {
    // Fused bilinear-resize + normalise + HWC→CHW, writing straight into the
    // preallocated tensor. After warm-up (coefficients cached, tensor sized)
    // this touches no heap at all — not even OpenCV-internal scratch (M-5).
    const int dst_w = impl_->input_size.width;
    const int dst_h = impl_->input_size.height;
    const std::size_t plane = static_cast<std::size_t>(dst_w) * dst_h;
    buf.tensor.resize(3 * plane);  // keeps capacity after the first call

    // Defensive: only CV_8UC3 frames large enough to bilinear-sample are valid.
    // Callers already gate inference on frame.empty(); here we simply no-op.
    if (frame.empty() || frame.type() != CV_8UC3 ||
        frame.cols < 2 || frame.rows < 2) {
        return;
    }

    if (frame.cols != impl_->src_w_ || frame.rows != impl_->src_h_) {
        impl_->build_resize_coeffs(frame.cols, frame.rows);
    }

    constexpr float inv255 = 1.0f / 255.0f;
    float* tensor = buf.tensor.data();

    for (int oy = 0; oy < dst_h; ++oy) {
        const int y0 = impl_->y_idx_[oy];
        const float wy = impl_->y_w_[oy];
        const uchar* row0 = frame.ptr<uchar>(y0);
        const uchar* row1 = frame.ptr<uchar>(y0 + 1);
        const std::size_t orow = static_cast<std::size_t>(oy) * dst_w;

        for (int ox = 0; ox < dst_w; ++ox) {
            const int x0 = impl_->x_idx_[ox];
            const float wx = impl_->x_w_[ox];
            const int c0 = x0 * 3;        // BGR triplet at (y0, x0)
            const int c1 = c0 + 3;        // BGR triplet at (y0, x0+1)

            // Bilinear-blend the four neighbours per BGR channel, then scatter
            // to the channel planes (CHW layout) — preserves the original
            // BGR channel order the inference path was written against.
            for (int c = 0; c < 3; ++c) {
                const float p00 = row0[c0 + c];
                const float p01 = row0[c1 + c];
                const float p10 = row1[c0 + c];
                const float p11 = row1[c1 + c];
                const float top = p00 + wx * (p01 - p00);
                const float bot = p10 + wx * (p11 - p10);
                tensor[static_cast<std::size_t>(c) * plane + orow + ox] =
                    (top + wy * (bot - top)) * inv255;
            }
        }
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    // Convenience form: delegate to the pooled path, then copy out. Allocates a
    // result vector, so it is intended for off-hot-path callers and tests only.
    ObjectPool<Detection> pool(kMaxDetections);
    detect(frame, pool);
    return {pool.view().begin(), pool.view().end()};
}

// ── E-2: standalone, self-validating YOLOv8 output decoder ──────────────────
// Kept ONNX-independent (raw pointer + shape) so the geometry checks can be
// proven against synthetic/corrupt tensors without a live inference session.
bool Detector::decode_yolo_output(const float* data, std::size_t data_len,
                                  const std::int64_t* shape, std::size_t shape_len,
                                  int frame_w, int frame_h,
                                  int input_w, int input_h,
                                  float conf_threshold,
                                  std::vector<Detection>& out) {
    out.clear();

    // Validate the tensor geometry BEFORE indexing. A corrupt or unexpected
    // model can advertise a shape inconsistent with the buffer it returned;
    // every check below exists to make a malformed tensor a clean rejection
    // instead of an out-of-bounds read.
    if (data == nullptr || shape == nullptr) {
        return false;
    }
    if (shape_len != 3) {                          // expect [1, 4+classes, anchors]
        return false;
    }
    if (shape[0] != 1) {                           // single-image batch only
        return false;
    }
    if (shape[1] <= 4 || shape[2] <= 0) {          // need >=1 class channel, >=1 anchor
        return false;
    }
    if (input_w <= 0 || input_h <= 0) {
        return false;
    }

    // Bound the dims so the products below cannot overflow and so we never spin
    // over an absurd anchor count from a garbage header.
    constexpr std::int64_t kMaxDim = 1 << 24;      // 16M — far beyond any real model
    if (shape[1] > kMaxDim || shape[2] > kMaxDim) {
        return false;
    }

    const std::int64_t channels = shape[1];
    const std::int64_t num_anchors = shape[2];
    const std::int64_t num_classes = channels - 4;

    // The largest flat index this routine ever dereferences is
    //   (channels-1)*num_anchors + (num_anchors-1) == channels*num_anchors - 1.
    // Prove the buffer actually holds that many elements before touching it.
    const std::uint64_t required = static_cast<std::uint64_t>(channels) *
                                   static_cast<std::uint64_t>(num_anchors);
    if (required > data_len) {
        return false;
    }

    const float scale_x = static_cast<float>(frame_w) / static_cast<float>(input_w);
    const float scale_y = static_cast<float>(frame_h) / static_cast<float>(input_h);

    const std::size_t anchors = static_cast<std::size_t>(num_anchors);
    const std::size_t classes = static_cast<std::size_t>(num_classes);

    // Output is row-major [1, channels, anchors]. For anchor i, channel k lives
    // at index k*anchors + i. All indices are <= required-1 (verified above).
    for (std::size_t i = 0; i < anchors; ++i) {
        float max_score = 0.0f;
        int class_id = -1;

        for (std::size_t c = 0; c < classes; ++c) {
            const float score = data[(4 + c) * anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = static_cast<int>(c);
            }
        }

        // Filter by confidence and vehicular classes (COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
        if (max_score >= conf_threshold &&
            (class_id == 2 || class_id == 3 || class_id == 5 || class_id == 7)) {

            const float cx = data[0 * anchors + i];
            const float cy = data[1 * anchors + i];
            const float w  = data[2 * anchors + i];
            const float h  = data[3 * anchors + i];

            Detection det;
            det.class_id = class_id;
            det.confidence = max_score;
            det.bbox.x = (cx - w / 2.0f) * scale_x;
            det.bbox.y = (cy - h / 2.0f) * scale_y;
            det.bbox.width = w * scale_x;
            det.bbox.height = h * scale_y;

            out.push_back(det);
        }
    }
    return true;
}

bool Detector::detect(const cv::Mat& frame, ObjectPool<Detection>& out) {
    out.clear();
    if (!impl_->initialized || frame.empty()) {
        return true;  // nothing to infer; not an inference fault
    }

#ifdef ONNXRUNTIME_AVAILABLE
    // Stub mode: no session exists. Emit nothing, allocate nothing, never throw.
    if (impl_->stub_mode || !impl_->session) {
        return true;
    }

    // Preprocess into the preallocated buffers (no per-frame allocation).
    preprocess(frame, impl_->buffers_);
    std::vector<float>& input_data = impl_->buffers_.tensor;

    std::vector<Detection>& detections = impl_->raw_;
    detections.clear();

    // E-1: the inference call is the single point that can throw or hand back a
    // malformed tensor. Contain it hermetically — on ANY failure we clear the
    // output and report a fault so the main loop fails safe to the emergency
    // ring; we never propagate an exception or abort the process.
    try {
        // Create input tensor
        std::array<int64_t, 4> input_shape = {
            1, 3, impl_->input_size.height, impl_->input_size.width};
        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        auto output_tensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            impl_->input_names.data(), &input_tensor, 1,
            impl_->output_names.data(), impl_->output_names.size());

        if (output_tensors.empty()) {
            spdlog::error("Detector: model returned no output tensors — fault.");
            out.clear();
            return false;
        }

        auto output_tensor = std::move(output_tensors.front());
        auto type_info = output_tensor.GetTensorTypeAndShapeInfo();

        // Reject non-float outputs before reinterpreting the buffer as float.
        if (type_info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            spdlog::error("Detector: model output is not float32 — fault.");
            out.clear();
            return false;
        }

        const std::vector<int64_t> shape = type_info.GetShape();
        const std::size_t elem_count = type_info.GetElementCount();
        const float* out_data = output_tensor.GetTensorData<float>();

        // Decode with full geometry validation (E-2). A shape inconsistent with
        // the buffer is rejected here rather than read out of bounds.
        if (!decode_yolo_output(out_data, elem_count, shape.data(), shape.size(),
                                frame.cols, frame.rows,
                                impl_->input_size.width, impl_->input_size.height,
                                impl_->conf_threshold, detections)) {
            spdlog::error("Detector: malformed model output rejected "
                          "(shape/element-count mismatch) — fault.");
            out.clear();
            return false;
        }

        // Apply NMS, emitting survivors into the caller's pool.
        impl_->apply_nms(detections, out);
        return true;

    } catch (const Ort::Exception& ex) {
        spdlog::critical("Detector: ONNX inference threw ({}). Failing safe.",
                         ex.what());
        out.clear();
        return false;
    } catch (const std::exception& ex) {
        spdlog::critical("Detector: inference exception ({}). Failing safe.",
                         ex.what());
        out.clear();
        return false;
    }
#else
    // Stub: emit no detections when ONNX Runtime is unavailable.
    (void)frame;
    spdlog::debug("Detector: stub mode — no detections emitted");
    return true;
#endif
}

bool Detector::is_initialized() const {
    return impl_->initialized;
}

cv::Size Detector::get_input_size() const {
    return impl_->input_size;
}

} // namespace semaforo
