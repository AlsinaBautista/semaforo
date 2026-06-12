// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <chrono>
#include <string>

#include <spdlog/spdlog.h>

namespace semaforo::utils {

/// @brief High-resolution timer for measuring elapsed time.
///
/// Useful for profiling inference latency, frame processing time,
/// and other performance-critical sections of the pipeline.
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    /// @brief Construct and start the timer.
    Timer() : start_(Clock::now()) {}

    /// @brief Reset the timer to the current time.
    void reset() {
        start_ = Clock::now();
    }

    /// @brief Get elapsed time in seconds since construction or last reset.
    /// @return Elapsed time in seconds as a double.
    double elapsed_s() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

    /// @brief Get elapsed time in milliseconds since construction or last reset.
    /// @return Elapsed time in milliseconds as a double.
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

    /// @brief Get elapsed time in microseconds since construction or last reset.
    /// @return Elapsed time in microseconds as a double.
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }

    /// @brief Get the start time point.
    /// @return TimePoint when the timer was started or last reset.
    TimePoint start_time() const {
        return start_;
    }

    /// @brief Get elapsed duration as std::chrono::duration<float>.
    /// @return Duration since start.
    std::chrono::duration<float> elapsed_duration() const {
        return std::chrono::duration<float>(Clock::now() - start_);
    }

private:
    TimePoint start_;
};

/// @brief RAII scoped timer that logs elapsed time on destruction.
///
/// Use this to automatically measure and log the duration of a scope.
/// Example:
/// @code
///   {
///       ScopedTimer t("inference");
///       // ... run inference ...
///   } // logs: "[perf] inference: 12.345 ms"
/// @endcode
class ScopedTimer {
public:
    /// @brief Construct a scoped timer with a label.
    /// @param label  Name for the timed section (logged on destruction).
    explicit ScopedTimer(std::string label)
        : label_(std::move(label)), timer_() {}

    /// @brief Destructor logs the elapsed time.
    ~ScopedTimer() {
        spdlog::debug("[perf] {}: {:.3f} ms", label_, timer_.elapsed_ms());
    }

    // Non-copyable, non-movable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string label_;
    Timer timer_;
};

} // namespace semaforo::utils
