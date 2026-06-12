// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Backpressure proofs for TelemetryOutbox (audit M-4). The single hazard these
// tests pin down: a downed broker link must NEVER let the outbound backlog grow
// without bound. We saturate the outbox far past its capacity (simulating an
// outage where nothing is ever drained) and assert that (a) its footprint stays
// hard-capped — O(1) memory — and (b) overflow evicts the OLDEST messages while
// retaining the freshest.

#include <gtest/gtest.h>

#include <string>

#include "semaforo/telemetry_outbox.hpp"

using namespace semaforo;

namespace {
OutboundMessage sample(int n) {
    return OutboundMessage{"semaforo/telemetry/edge-001",
                           "{\"seq\":" + std::to_string(n) + "}", 1, false};
}
}  // namespace

// ── Memory stays hard-capped under a never-draining (downed-network) flood ───
TEST(TelemetryOutbox, MemoryBoundedUnderNetworkOutage) {
    constexpr std::size_t kCapacity = 8;
    constexpr int kFlood = 100'000;  // network is "down": nothing is ever drained
    TelemetryOutbox outbox(kCapacity);

    for (int i = 0; i < kFlood; ++i) {
        outbox.push(sample(i));
        // INVARIANT: the buffer can never exceed its hard bound, no matter how
        // long the outage lasts — this is the O(1)-space guarantee.
        ASSERT_LE(outbox.size(), kCapacity);
    }

    EXPECT_EQ(outbox.size(), kCapacity);
    // Everything beyond the retained window was dropped, not accumulated.
    EXPECT_EQ(outbox.dropped(),
              static_cast<std::uint64_t>(kFlood) - kCapacity);
}

// ── Drop-oldest: overflow discards the stalest, keeps the freshest ───────────
TEST(TelemetryOutbox, OverflowDropsOldestRetainsNewest) {
    constexpr std::size_t kCapacity = 4;
    TelemetryOutbox outbox(kCapacity);

    for (int i = 0; i < 10; ++i) outbox.push(sample(i));  // push 0..9 into cap-4

    ASSERT_EQ(outbox.size(), kCapacity);

    // FIFO drain must yield exactly the last 4 pushed (6,7,8,9) — the older
    // samples 0..5 were evicted by drop-oldest.
    OutboundMessage msg;
    for (int expected = 6; expected <= 9; ++expected) {
        ASSERT_TRUE(outbox.try_pop(msg));
        EXPECT_EQ(msg.payload, "{\"seq\":" + std::to_string(expected) + "}");
    }
    EXPECT_FALSE(outbox.try_pop(msg));  // fully drained
    EXPECT_EQ(outbox.size(), 0u);
}

// ── Below capacity nothing is dropped (no spurious eviction) ─────────────────
TEST(TelemetryOutbox, NoDropBelowCapacity) {
    TelemetryOutbox outbox(16);
    for (int i = 0; i < 16; ++i) outbox.push(sample(i));
    EXPECT_EQ(outbox.size(), 16u);
    EXPECT_EQ(outbox.dropped(), 0u);
}
