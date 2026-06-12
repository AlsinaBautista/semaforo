// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#include "semaforo/pipeline.hpp"

#include <spdlog/spdlog.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <atomic>
#include <cctype>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

namespace semaforo {

// ── PIMPL implementation ────────────────────────────────────────────────────
//
// Thread-safety model (A-5):
//   * The live ``cv::VideoCapture`` is held behind a ``shared_ptr`` and is read
//     ONLY by the main thread (read_frame). cv::VideoCapture is not thread-safe,
//     so no other thread ever touches the live object.
//   * The reconnect thread NEVER calls release()/open() on the live object.
//     Instead it opens a brand-new capture on a throwaway local object (slow,
//     no lock) and atomically *swaps* it in under a short lock — a double-buffer
//     hand-off. The old object is destroyed only when the last shared_ptr ref
//     drops, so a swap can never pull the buffer out from under an in-flight
//     read() (the reader holds its own ref for the duration of the read).
//   * ``stop`` is a stop-token: the destructor sets it and joins the reconnect
//     thread before any teardown, guaranteeing no background access to a
//     half-destroyed object.

struct VideoPipeline::Impl {
    // Guarded by capture_mtx: the live capture pointer + its metadata.
    mutable std::mutex capture_mtx;
    std::shared_ptr<cv::VideoCapture> capture;
    cv::Size frame_size;
    double fps = 0.0;

    // Set once at init() before any worker thread exists; read-only thereafter.
    std::string source_uri;
    int desired_width = 0;
    int desired_height = 0;

    std::atomic<bool> opened{false};
    std::atomic<bool> is_reconnecting{false};
    std::atomic<bool> stop{false};  // stop-token for clean shutdown
    std::thread reconnect_thread;

    /// @brief Build a GStreamer pipeline string for RTSP sources.
    static std::string build_gst_pipeline(const std::string& uri,
                                          int width, int height) {
        std::string pipeline =
            "rtspsrc location=" + uri + " latency=100 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! video/x-raw,format=BGR";

        if (width > 0 && height > 0) {
            pipeline += " ! videoscale ! video/x-raw,width="
                        + std::to_string(width) + ",height="
                        + std::to_string(height);
        }

        pipeline += " ! appsink drop=1";
        return pipeline;
    }

    /// @brief Open a brand-new VideoCapture (slow; caller holds NO lock).
    /// @return An opened capture, or nullptr on failure.
    static std::shared_ptr<cv::VideoCapture> open_capture(
        const std::string& uri, int width, int height,
        cv::Size& out_size, double& out_fps) {
        auto cap = std::make_shared<cv::VideoCapture>();
        bool success = false;

        if (uri.rfind("rtsp://", 0) == 0) {
            const std::string gst = build_gst_pipeline(uri, width, height);
            spdlog::info("Pipeline: opening GStreamer: {}", gst);
            success = cap->open(gst, cv::CAP_GSTREAMER);
            if (!success) {
                spdlog::warn("Pipeline: GStreamer failed, trying FFMPEG backend");
                success = cap->open(uri, cv::CAP_FFMPEG);
            }
        } else if (uri.rfind("test://", 0) == 0) {
            const std::string test_pipeline =
                "videotestsrc pattern=ball ! "
                "video/x-raw,width=640,height=480,framerate=30/1 ! "
                "videoconvert ! video/x-raw,format=BGR ! appsink";
            spdlog::info("Pipeline: opening test source");
            success = cap->open(test_pipeline, cv::CAP_GSTREAMER);
        } else if (uri.size() == 1 &&
                   std::isdigit(static_cast<unsigned char>(uri[0]))) {
            const int dev_index = uri[0] - '0';
            spdlog::info("Pipeline: opening USB camera {}", dev_index);
            success = cap->open(dev_index, cv::CAP_V4L2);
            if (!success) success = cap->open(dev_index);
        } else {
            spdlog::info("Pipeline: opening file/device: {}", uri);
            success = cap->open(uri);
        }

        if (success && cap->isOpened()) {
            out_fps = cap->get(cv::CAP_PROP_FPS);
            out_size = cv::Size(
                static_cast<int>(cap->get(cv::CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap->get(cv::CAP_PROP_FRAME_HEIGHT)));
            if (width > 0 && height > 0) {
                cap->set(cv::CAP_PROP_FRAME_WIDTH, width);
                cap->set(cv::CAP_PROP_FRAME_HEIGHT, height);
                out_size = cv::Size(width, height);
            }
            return cap;
        }
        return nullptr;
    }

    /// @brief Atomically install a freshly-opened capture as the live one.
    void install(std::shared_ptr<cv::VideoCapture> cap,
                 cv::Size size, double new_fps) {
        std::lock_guard<std::mutex> lk(capture_mtx);
        capture = std::move(cap);  // old capture destructs when its refs drop
        frame_size = size;
        fps = new_fps;
    }
};

// ── Constructor / Destructor ────────────────────────────────────────────────

VideoPipeline::VideoPipeline() : impl_(std::make_unique<Impl>()) {}

VideoPipeline::~VideoPipeline() {
    // Stop-token first: signal, then join, then release — so the worker can
    // never touch a half-destroyed object.
    impl_->stop.store(true);
    if (impl_->reconnect_thread.joinable()) {
        impl_->reconnect_thread.join();
    }
    release();
}

// ── Public methods ──────────────────────────────────────────────────────────

bool VideoPipeline::init(const std::string& source_uri, int width, int height) {
    impl_->source_uri = source_uri;
    impl_->desired_width = width;
    impl_->desired_height = height;

    cv::Size size;
    double new_fps = 0.0;
    auto cap = Impl::open_capture(source_uri, width, height, size, new_fps);

    if (cap && cap->isOpened()) {
        impl_->install(std::move(cap), size, new_fps);
        impl_->opened.store(true);
        spdlog::info("Pipeline: opened successfully ({}x{} @ {:.1f} FPS)",
                     size.width, size.height, new_fps);
        return true;
    }

    spdlog::error("Pipeline: failed to open source: {}", source_uri);
    impl_->opened.store(false);
    return false;
}

cv::Mat VideoPipeline::read_frame() {
    cv::Mat frame;
    read_frame(frame);
    return frame;
}

bool VideoPipeline::read_frame(cv::Mat& out) {
    // Grab a ref to the live capture under a short lock, then read WITHOUT the
    // lock held. Holding the shared_ptr keeps the object alive even if the
    // reconnect thread swaps in a new one mid-read.
    std::shared_ptr<cv::VideoCapture> cap;
    {
        std::lock_guard<std::mutex> lk(impl_->capture_mtx);
        cap = impl_->capture;
    }

    if (!impl_->opened.load() || !cap || !cap->isOpened()) {
        // Leave ``out``'s buffer allocated for reuse; signal "no frame".
        return false;
    }

    // cv::VideoCapture::read reuses out's existing allocation when the next
    // frame matches its size/type — no per-frame Mat allocation (M-5).
    if (!cap->read(out)) {
        spdlog::warn("Pipeline: failed to read frame");
        return false;
    }

    if (impl_->desired_width > 0 && impl_->desired_height > 0 &&
        (out.cols != impl_->desired_width ||
         out.rows != impl_->desired_height)) {
        cv::resize(out, out,
                   cv::Size(impl_->desired_width, impl_->desired_height));
    }

    return true;
}

bool VideoPipeline::is_open() const {
    std::shared_ptr<cv::VideoCapture> cap;
    {
        std::lock_guard<std::mutex> lk(impl_->capture_mtx);
        cap = impl_->capture;
    }
    return impl_->opened.load() && cap && cap->isOpened();
}

double VideoPipeline::get_fps() const {
    std::lock_guard<std::mutex> lk(impl_->capture_mtx);
    return impl_->fps;
}

cv::Size VideoPipeline::get_frame_size() const {
    std::lock_guard<std::mutex> lk(impl_->capture_mtx);
    return impl_->frame_size;
}

void VideoPipeline::release() {
    // Detach the live capture under the lock, destroy it outside the lock.
    std::shared_ptr<cv::VideoCapture> old;
    {
        std::lock_guard<std::mutex> lk(impl_->capture_mtx);
        old = std::move(impl_->capture);
        impl_->capture.reset();
    }
    impl_->opened.store(false);
    if (old) {
        old->release();
        spdlog::info("Pipeline: released");
    }
}

bool VideoPipeline::is_reconnecting() const {
    return impl_->is_reconnecting.load();
}

void VideoPipeline::trigger_reconnect_async() {
    // Only one reconnect at a time.
    bool expected = false;
    if (!impl_->is_reconnecting.compare_exchange_strong(expected, true)) {
        return;
    }

    // Reap the previous (already-finished) worker before reusing the slot.
    if (impl_->reconnect_thread.joinable()) {
        impl_->reconnect_thread.join();
    }

    impl_->reconnect_thread = std::thread([this]() {
        spdlog::info("Pipeline: async reconnect started...");

        // Back off ~2s before retrying, but bail out promptly on shutdown.
        for (int i = 0; i < 20 && !impl_->stop.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (impl_->stop.load()) {
            impl_->is_reconnecting.store(false);
            return;
        }

        // Open a NEW capture on a throwaway object (the live one is untouched),
        // then swap it in atomically.
        cv::Size size;
        double new_fps = 0.0;
        auto cap = Impl::open_capture(impl_->source_uri, impl_->desired_width,
                                      impl_->desired_height, size, new_fps);

        if (cap && cap->isOpened()) {
            impl_->install(std::move(cap), size, new_fps);
            impl_->opened.store(true);
            spdlog::info("Pipeline: async reconnect SUCCESS ({}x{})",
                         size.width, size.height);
        } else {
            spdlog::warn("Pipeline: async reconnect FAILED");
        }

        impl_->is_reconnecting.store(false);
    });
}

} // namespace semaforo
