// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace semaforo {

/// @brief Video pipeline wrapping GStreamer/OpenCV for frame acquisition.
///
/// Provides a simple interface to capture frames from various video sources
/// including RTSP cameras, local video files, and USB cameras. Uses OpenCV's
/// VideoCapture with GStreamer backend for robust streaming support.
class VideoPipeline {
public:
    VideoPipeline();
    ~VideoPipeline();

    // Non-copyable AND non-movable. A background reconnect thread captures
    // ``this``; moving the object would leave that thread with a dangling
    // pointer (A-5). The class is therefore pinned for its whole lifetime.
    VideoPipeline(const VideoPipeline&) = delete;
    VideoPipeline& operator=(const VideoPipeline&) = delete;
    VideoPipeline(VideoPipeline&&) = delete;
    VideoPipeline& operator=(VideoPipeline&&) = delete;

    /// @brief Initialize the video pipeline with a source URI.
    ///
    /// Supports the following URI formats:
    /// - RTSP:  "rtsp://user:pass@host:554/stream"
    /// - File:  "/path/to/video.mp4"
    /// - USB:   "v4l2:///dev/video0"  or device index "0"
    /// - Test:  "test://" for a synthetic test pattern
    ///
    /// @param source_uri  URI of the video source.
    /// @param width       Desired frame width (0 = source native, default: 0).
    /// @param height      Desired frame height (0 = source native, default: 0).
    /// @return true if the pipeline was opened successfully.
    bool init(const std::string& source_uri, int width = 0, int height = 0);

    /// @brief Read the next frame from the video source.
    ///
    /// Blocks until a frame is available or an error occurs.
    /// Returns an empty cv::Mat if the source has ended or an error occurred.
    ///
    /// @return BGR frame as cv::Mat, or empty Mat on failure/end-of-stream.
    cv::Mat read_frame();

    /// @brief Read the next frame into a caller-owned, reused buffer (M-5).
    ///
    /// Reads directly into ``out`` so the main loop can hoist a single
    /// ``cv::Mat`` outside its hot loop and recycle it every frame: OpenCV's
    /// ``VideoCapture::read`` reuses ``out``'s existing allocation when the
    /// frame size and type match, eliminating the per-frame Mat allocation of
    /// the value-returning overload. On failure ``out``'s buffer is left intact
    /// (still preallocated for the next attempt) and ``false`` is returned.
    ///
    /// @param out  Destination frame buffer (reused across calls).
    /// @return true if a frame was read; false on failure/end-of-stream.
    bool read_frame(cv::Mat& out);

    /// @brief Check whether the pipeline is open and ready to provide frames.
    /// @return true if the pipeline is actively capturing.
    bool is_open() const;

    /// @brief Get the current frames per second of the source.
    /// @return FPS value reported by the source, or 0 if unknown.
    double get_fps() const;

    /// @brief Get the frame dimensions.
    /// @return Frame size as cv::Size (width, height).
    cv::Size get_frame_size() const;

    /// @brief Check if a background reconnection attempt is in progress.
    /// @return true if reconnecting.
    bool is_reconnecting() const;

    /// @brief Trigger a reconnection attempt in a background thread.
    /// Does nothing if already reconnecting.
    void trigger_reconnect_async();

    /// @brief Release the video source and free resources.
    void release();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_; ///< PIMPL for OpenCV/GStreamer internals
};

} // namespace semaforo
