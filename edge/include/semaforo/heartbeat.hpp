// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

// POSIX I/O for the sysfs GPIO backend. These are available on Linux (the
// deployment target) and on macOS/BSD (the dev host); on the dev host the
// /sys/class/gpio paths simply do not exist, so SysfsGpioLine reports
// !ok() and every write is a harmless no-op. No heavy dependency is pulled
// in — this is raw open()/write(), nothing else.
#include <fcntl.h>
#include <unistd.h>

namespace semaforo {

/// @brief A single, level-driveable digital output line.
///
/// Abstracted so the dead-man Heartbeat can be exercised against a mock in
/// unit tests (no real hardware) while driving an actual GPIO pin in the
/// field. Implementations MUST be non-throwing: a write failure is reported
/// via the return value / ok(), never an exception — the hot loop can never
/// be allowed to unwind on an I/O hiccup.
class GpioLine {
public:
    virtual ~GpioLine() = default;

    /// @brief Drive the line to @p value (true = logic high).
    /// @return true on success, false on I/O error (never throws).
    virtual bool write(bool value) = 0;

    /// @brief Whether the line was successfully acquired and is usable.
    virtual bool ok() const = 0;
};

/// @brief Real GPIO line over the standard Linux sysfs interface.
///
/// Uses the legacy, dependency-free `/sys/class/gpio` ABI: export the pin,
/// set its direction to "out", then hold the value file descriptor open for
/// fast repeated writes (no per-tick open/close on the 500 ms hot path).
///
/// Construction is best-effort and never throws: if export or fd-open fails
/// (e.g. running on the dev host, or the pin is already owned), ok() returns
/// false and every write() is a silent no-op. The caller decides whether a
/// missing GPIO is fatal; in the field it should log loudly, in the lab it is
/// expected.
class SysfsGpioLine : public GpioLine {
public:
    explicit SysfsGpioLine(int pin, const std::string& sysfs_root = "/sys/class/gpio")
        : pin_(pin) {
        const std::string base = sysfs_root + "/gpio" + std::to_string(pin_);

        // Export the pin (idempotent: EBUSY/already-exported is fine).
        write_string(sysfs_root + "/export", std::to_string(pin_));
        // Drive it as an output.
        write_string(base + "/direction", "out");

        // Hold the value fd open for the lifetime of the line.
        value_fd_ = ::open((base + "/value").c_str(), O_WRONLY | O_CLOEXEC);
        ok_ = (value_fd_ >= 0);
    }

    ~SysfsGpioLine() override {
        if (value_fd_ >= 0) {
            // Best-effort de-assert before releasing the line.
            (void)write(false);
            ::close(value_fd_);
        }
    }

    SysfsGpioLine(const SysfsGpioLine&) = delete;
    SysfsGpioLine& operator=(const SysfsGpioLine&) = delete;

    bool write(bool value) override {
        if (value_fd_ < 0) {
            return false;
        }
        const char byte = value ? '1' : '0';
        // Rewind: the sysfs value file is a 1-byte register, not a stream.
        if (::lseek(value_fd_, 0, SEEK_SET) < 0) {
            return false;
        }
        const ssize_t n = ::write(value_fd_, &byte, 1);
        return n == 1;
    }

    bool ok() const override { return ok_; }

private:
    /// @brief Best-effort write of a short string to a sysfs control file.
    static void write_string(const std::string& path, const std::string& value) {
        const int fd = ::open(path.c_str(), O_WRONLY | O_CLOEXEC);
        if (fd < 0) {
            return;
        }
        (void)!::write(fd, value.c_str(), value.size());
        ::close(fd);
    }

    int pin_;
    int value_fd_ = -1;
    bool ok_ = false;
};

/// @brief Construct a real sysfs-backed GPIO line for @p pin.
inline std::unique_ptr<GpioLine> make_gpio_line(int pin) {
    return std::make_unique<SysfsGpioLine>(pin);
}

/// @brief Physical Heartbeat / Dead-Man's Switch driving a cabinet MMU.
///
/// This is the *hardware* fail-safe, independent of every software interlock.
/// It emits a square wave on a GPIO pin that the cabinet Malfunction
/// Management Unit (MMU) watches: while edges keep arriving the MMU keeps the
/// control relay closed and lets the Edge drive the heads; the instant the
/// pulse train stops the MMU opens the relay and forces the intersection into
/// flashing red — in hardware, with no further software involvement.
///
/// The dead-man property is achieved *by construction*, not by a timer thread:
/// tick() is called inline from the main control loop and is the only thing
/// that toggles the pin. Therefore if the main thread blocks, dead-locks,
/// segfaults, or falls into fatal latency, no further toggle is issued and the
/// pulse train mathematically ceases — the MMU trips. A watchdog living on its
/// own thread would defeat this (it would keep pulsing over a dead main loop),
/// so the toggle is deliberately kept on the hot path.
///
/// Emission is additionally gated on a liveness predicate passed each tick (in
/// the Edge this is `inference_online`): while the predicate is false the line
/// is de-asserted and the train stops, so an inference fault also drops the
/// hardware relay — defense in depth on top of the SafetyLayer's emergency
/// ring. The clock is injectable for deterministic, sleep-free tests.
class Heartbeat {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /// @brief Default toggle half-period: one edge every 500 ms.
    static constexpr std::chrono::milliseconds DEFAULT_PERIOD{500};

    explicit Heartbeat(std::unique_ptr<GpioLine> line,
                       std::chrono::milliseconds period = DEFAULT_PERIOD)
        : line_(std::move(line)), period_(period) {}

    /// @brief Inline dead-man kick. Call once per hot-loop iteration.
    /// @param alive Liveness predicate (Edge: inference_online). When false the
    ///        line is de-asserted and the pulse train stops, tripping the MMU.
    void tick(bool alive) { tick(alive, Clock::now()); }

    /// @brief Deterministic overload with an injected clock (for testing).
    void tick(bool alive, TimePoint now) {
        if (!alive) {
            // Edge-triggered de-assert: drop the line once, then stay silent so
            // the MMU loses its edges and forces flashing red.
            if (armed_) {
                level_ = false;
                if (line_) {
                    (void)line_->write(false);
                }
                armed_ = false;
            }
            return;
        }

        if (!armed_) {
            // (Re)start the square wave: assert immediately so the MMU sees a
            // fresh edge the moment liveness is (re)established.
            armed_ = true;
            last_toggle_ = now;
            level_ = true;
            if (line_) {
                (void)line_->write(true);
            }
            ++pulse_count_;
            return;
        }

        // Hold the cadence: one edge per period, regardless of how often the
        // (faster, ~30 FPS) loop calls in.
        if (now - last_toggle_ >= period_) {
            level_ = !level_;
            if (line_) {
                (void)line_->write(level_);
            }
            last_toggle_ = now;
            ++pulse_count_;
        }
    }

    /// @brief Force the line low and disarm (e.g. on graceful shutdown).
    void stop() {
        if (line_) {
            (void)line_->write(false);
        }
        level_ = false;
        armed_ = false;
    }

    /// @brief Current logic level being driven on the pin.
    bool level() const { return level_; }

    /// @brief Whether the pulse train is currently being emitted.
    bool armed() const { return armed_; }

    /// @brief Total number of edges emitted (monotonic; for tests/telemetry).
    std::uint64_t pulse_count() const { return pulse_count_; }

    /// @brief Whether the underlying GPIO line was successfully acquired.
    bool line_ok() const { return line_ && line_->ok(); }

    std::chrono::milliseconds period() const { return period_; }

private:
    std::unique_ptr<GpioLine> line_;
    std::chrono::milliseconds period_;
    TimePoint last_toggle_{};
    bool level_ = false;            ///< Current driven level.
    bool armed_ = false;            ///< Whether the train is running.
    std::uint64_t pulse_count_ = 0; ///< Edges emitted while alive.
};

} // namespace semaforo
