// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Unit tests for the TTL-latched remote directive. The clock is injected, so
// the 5-second time-to-live is validated deterministically without sleeping.

#include <gtest/gtest.h>

#include "semaforo/command_latch.hpp"

using namespace semaforo;

// A latched directive is the primary control input as soon as it is received.
TEST(DirectiveLatch, LatchesRemoteDirectiveAsPrimary) {
    DirectiveLatch latch;  // default 4000 ms TTL
    EXPECT_FALSE(latch.get(0).has_value());  // nothing latched yet
    EXPECT_FALSE(latch.has_directive());

    latch.update(Phase::GREEN_EW, 1000);
    auto d = latch.get(1000);
    ASSERT_TRUE(d.has_value());
    EXPECT_EQ(*d, Phase::GREEN_EW);
    EXPECT_TRUE(latch.has_directive());
}

// The directive persists across ticks WITHOUT being re-sent (latching).
TEST(DirectiveLatch, PersistsAcrossTicksWithoutResend) {
    DirectiveLatch latch;
    latch.update(Phase::GREEN_NS, 0);
    EXPECT_EQ(latch.get(1000).value(), Phase::GREEN_NS);
    EXPECT_EQ(latch.get(3000).value(), Phase::GREEN_NS);
    EXPECT_EQ(latch.get(3999).value(), Phase::GREEN_NS);  // still fresh just under TTL
}

// Core requirement: after 4.1 s with no Brain update the directive is stale,
// so control degrades to the local policy (latch yields nothing).
TEST(DirectiveLatch, DegradesToLocalAfter4_1s) {
    DirectiveLatch latch;  // 4000 ms TTL
    latch.update(Phase::GREEN_EW, 0);
    ASSERT_TRUE(latch.get(0).has_value());

    const std::int64_t t = 4100;  // 4.1 s, no further updates
    EXPECT_FALSE(latch.get(t).has_value())
        << "directive must be stale at 4.1 s → degrade to local policy";
    EXPECT_TRUE(latch.is_stale(t));
    EXPECT_EQ(latch.age_ms(t), 4100);
}

// The TTL boundary is exact: stale at precisely 4000 ms, fresh just before.
TEST(DirectiveLatch, BoundaryIsExactAtTtl) {
    DirectiveLatch latch;
    latch.update(Phase::GREEN_NS, 0);
    EXPECT_TRUE(latch.get(3999).has_value());   // fresh at 3.999 s
    EXPECT_FALSE(latch.get(4000).has_value());  // stale at exactly 4.000 s
    EXPECT_TRUE(latch.is_stale(4000));
}

// A fresh update restarts the TTL window (rolling, not absolute).
TEST(DirectiveLatch, FreshUpdateRestartsTtl) {
    DirectiveLatch latch;
    latch.update(Phase::GREEN_NS, 0);
    EXPECT_TRUE(latch.get(3000).has_value());

    latch.update(Phase::GREEN_EW, 3000);              // refresh before expiry
    EXPECT_EQ(latch.get(6000).value(), Phase::GREEN_EW);  // 3 s after refresh → fresh
    EXPECT_FALSE(latch.get(7001).has_value());        // 4.001 s after refresh → stale
}

// A custom TTL is honoured.
TEST(DirectiveLatch, RespectsCustomTtl) {
    DirectiveLatch latch(2000);  // 2 s TTL
    latch.update(Phase::GREEN_EW, 0);
    EXPECT_TRUE(latch.get(1999).has_value());
    EXPECT_FALSE(latch.get(2000).has_value());
    EXPECT_EQ(latch.ttl_ms(), 2000);
}
