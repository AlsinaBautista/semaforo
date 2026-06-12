// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Unit tests for the hardware dead-man Heartbeat driving the cabinet MMU.
// The GPIO line is mocked and the clock is injected, so the 500 ms pulse
// train and its gating are validated deterministically without real hardware
// and without sleeping.
//
// The load-bearing invariant (matching the cabinet MMU contract):
//   the heartbeat emits a pulse ONLY while the liveness predicate
//   (inference_online) is true — when it is false, or when the hot loop stops
//   calling tick() (freeze/segfault), the pulse train ceases and the MMU
//   independently forces flashing red.

#include <chrono>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "semaforo/heartbeat.hpp"

using namespace semaforo;
using namespace std::chrono_literals;

namespace {

/// @brief Recording GPIO line: counts writes and remembers every level driven.
class MockGpioLine : public GpioLine {
public:
    bool write(bool value) override {
        ++writes;
        last = value;
        history.push_back(value);
        return ok_;
    }
    bool ok() const override { return ok_; }

    bool ok_ = true;
    int writes = 0;
    bool last = false;
    std::vector<bool> history;
};

/// @brief Build a Heartbeat over a MockGpioLine, returning the raw mock too.
struct Rig {
    MockGpioLine* gpio;
    Heartbeat hb;
};

Rig make_rig(std::chrono::milliseconds period = Heartbeat::DEFAULT_PERIOD) {
    auto line = std::make_unique<MockGpioLine>();
    MockGpioLine* raw = line.get();
    return Rig{raw, Heartbeat(std::move(line), period)};
}

Heartbeat::TimePoint base() { return Heartbeat::Clock::now(); }

} // namespace

// ── Core contract: emission is gated on the liveness predicate ──────────────

// The signal is emitted ONLY when alive (inference_online) is true. While it
// stays false, no pulse is ever produced no matter how much time passes.
TEST(Heartbeat, EmitsNothingWhileNotAlive) {
    auto rig = make_rig();
    const auto t0 = base();

    for (int s = 0; s <= 10; ++s) {
        rig.hb.tick(/*alive=*/false, t0 + std::chrono::seconds(s));
    }

    EXPECT_EQ(rig.hb.pulse_count(), 0u);
    EXPECT_FALSE(rig.hb.armed());
    EXPECT_EQ(rig.gpio->writes, 0)  // never armed → never even de-asserted
        << "no GPIO activity may occur while inference is offline";
}

// As soon as alive becomes true the train starts: an immediate asserting edge.
TEST(Heartbeat, StartsTrainWhenAliveBecomesTrue) {
    auto rig = make_rig();
    const auto t0 = base();

    rig.hb.tick(/*alive=*/true, t0);

    EXPECT_TRUE(rig.hb.armed());
    EXPECT_TRUE(rig.hb.level());
    EXPECT_EQ(rig.hb.pulse_count(), 1u);
    ASSERT_EQ(rig.gpio->writes, 1);
    EXPECT_TRUE(rig.gpio->last);
}

// ── Cadence: one edge per 500 ms half-period, independent of call rate ───────

TEST(Heartbeat, TogglesOncePerPeriodRegardlessOfCallRate) {
    auto rig = make_rig();  // 500 ms
    const auto t0 = base();

    rig.hb.tick(true, t0);              // edge 1 (HIGH)
    EXPECT_EQ(rig.hb.pulse_count(), 1u);

    // Hammer the loop ~30 FPS for the whole half-period: still just one edge.
    for (int ms = 0; ms < 500; ms += 33) {
        rig.hb.tick(true, t0 + std::chrono::milliseconds(ms));
    }
    EXPECT_EQ(rig.hb.pulse_count(), 1u) << "no edge before the half-period elapses";
    EXPECT_TRUE(rig.hb.level());

    rig.hb.tick(true, t0 + 500ms);     // edge 2 (LOW)
    EXPECT_EQ(rig.hb.pulse_count(), 2u);
    EXPECT_FALSE(rig.hb.level());

    rig.hb.tick(true, t0 + 999ms);     // not yet (only 499 ms since edge 2)
    EXPECT_EQ(rig.hb.pulse_count(), 2u);

    rig.hb.tick(true, t0 + 1000ms);    // edge 3 (HIGH)
    EXPECT_EQ(rig.hb.pulse_count(), 3u);
    EXPECT_TRUE(rig.hb.level());

    // The driven levels form a proper square wave.
    const std::vector<bool> expected = {true, false, true};
    EXPECT_EQ(rig.gpio->history, expected);
}

// ── Dead-man: a frozen loop (no tick() calls) stops the pulse ────────────────

// Once armed, the ONLY source of edges is tick(). If the hot loop freezes /
// segfaults and stops calling in, no further edge is produced — the MMU loses
// the train and trips. We model the freeze as the absence of further calls.
TEST(Heartbeat, FrozenLoopProducesNoFurtherEdges) {
    auto rig = make_rig();
    const auto t0 = base();

    rig.hb.tick(true, t0);
    rig.hb.tick(true, t0 + 500ms);
    const auto pulses_before_freeze = rig.hb.pulse_count();
    const int writes_before_freeze = rig.gpio->writes;

    // ── main loop "freezes here" — no tick() for a long wall-clock time ──
    // Nothing toggles the pin; the MMU sees a static level and opens the relay.

    EXPECT_EQ(rig.hb.pulse_count(), pulses_before_freeze);
    EXPECT_EQ(rig.gpio->writes, writes_before_freeze)
        << "a frozen loop must emit zero further edges";
}

// ── Inference fault: de-assert exactly once, then silence ────────────────────

TEST(Heartbeat, DeassertsOnceWhenInferenceFaults) {
    auto rig = make_rig();
    const auto t0 = base();

    rig.hb.tick(true, t0);             // armed, HIGH, 1 write
    ASSERT_EQ(rig.gpio->writes, 1);

    // Inference faults → alive=false. The line is dropped LOW exactly once,
    // the train disarms, and no pulse count is added.
    rig.hb.tick(false, t0 + 10ms);
    EXPECT_FALSE(rig.hb.armed());
    EXPECT_FALSE(rig.hb.level());
    EXPECT_EQ(rig.hb.pulse_count(), 1u);
    ASSERT_EQ(rig.gpio->writes, 2);
    EXPECT_FALSE(rig.gpio->last);

    // Subsequent dead time produces no further GPIO activity.
    rig.hb.tick(false, t0 + 600ms);
    rig.hb.tick(false, t0 + 5s);
    EXPECT_EQ(rig.gpio->writes, 2) << "must stay silent while inference is down";
    EXPECT_EQ(rig.hb.pulse_count(), 1u);
}

// Recovery re-arms the train immediately with a fresh asserting edge.
TEST(Heartbeat, ReArmsImmediatelyOnRecovery) {
    auto rig = make_rig();
    const auto t0 = base();

    rig.hb.tick(true, t0);             // HIGH
    rig.hb.tick(false, t0 + 10ms);     // fault → LOW, disarmed
    ASSERT_FALSE(rig.hb.armed());

    rig.hb.tick(true, t0 + 3s);        // inference recovers
    EXPECT_TRUE(rig.hb.armed());
    EXPECT_TRUE(rig.hb.level());
    EXPECT_EQ(rig.hb.pulse_count(), 2u);
    EXPECT_TRUE(rig.gpio->last);
}

// ── stop(): graceful shutdown drops the line and the train ───────────────────

TEST(Heartbeat, StopDropsLineAndDisarms) {
    auto rig = make_rig();
    const auto t0 = base();

    rig.hb.tick(true, t0);
    rig.hb.stop();

    EXPECT_FALSE(rig.hb.armed());
    EXPECT_FALSE(rig.hb.level());
    EXPECT_FALSE(rig.gpio->last) << "shutdown must leave the line LOW so the MMU trips";
}

// A write failure on the GPIO must never throw or stall the train logic.
TEST(Heartbeat, ToleratesGpioWriteFailure) {
    auto line = std::make_unique<MockGpioLine>();
    line->ok_ = false;  // every write reports failure
    MockGpioLine* raw = line.get();
    Heartbeat hb(std::move(line));
    const auto t0 = base();

    EXPECT_NO_THROW(hb.tick(true, t0));
    EXPECT_NO_THROW(hb.tick(true, t0 + 500ms));
    EXPECT_EQ(hb.pulse_count(), 2u);  // logic advances regardless of I/O result
    EXPECT_GT(raw->writes, 0);
}

// ── Sysfs backend portability: a missing GPIO root is non-fatal ──────────────

// On the dev host (and any box without the pin exported) the real sysfs line
// reports !ok() and is a safe no-op — construction never throws.
TEST(SysfsGpioLine, MissingRootIsNotOkAndNeverThrows) {
    EXPECT_NO_THROW({
        SysfsGpioLine line(17, "/nonexistent/semaforo/gpio/root");
        EXPECT_FALSE(line.ok());
        EXPECT_FALSE(line.write(true));  // no-op, no crash
    });
}
