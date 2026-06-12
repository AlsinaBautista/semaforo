// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Chaos-resilience tests for CommandParser and CommandGuard. These exercise the
// exact attack/failure vectors injected by scripts/chaos_tester.py and assert
// that NONE of them can ever produce an actionable command or crash the parser.

#include <gtest/gtest.h>

#include <string>

#include "semaforo/command_parser.hpp"

using namespace semaforo;

namespace {
constexpr std::int64_t kNow = 1'781'000'000'000;  // fixed reference epoch ms

std::string good_payload(std::int64_t ts = kNow,
                         const std::string& phase = "GREEN_EW") {
    // sequence_number is mandatory under the strict-mode contract, so a
    // "good" payload must always carry one.
    return R"({"action":"CHANGE_PHASE","phase":")" + phase +
           R"(","priority":"HIGH","sequence_number":1,"timestamp":)" +
           std::to_string(ts) + "}";
}

std::string seq_payload(std::int64_t seq, std::int64_t ts = kNow,
                        const std::string& phase = "GREEN_EW") {
    return R"({"action":"CHANGE_PHASE","phase":")" + phase +
           R"(","priority":"HIGH","sequence_number":)" + std::to_string(seq) +
           R"(,"timestamp":)" + std::to_string(ts) + "}";
}
}  // namespace

// ── Happy path ──────────────────────────────────────────────────────────────

TEST(CommandParser, AcceptsWellFormedFreshCommand) {
    CommandParser p;
    auto r = p.parse(good_payload(kNow - 100), kNow);
    EXPECT_EQ(r.status, CommandStatus::OK);
    EXPECT_TRUE(r.is_actionable());
    EXPECT_EQ(r.phase, Phase::GREEN_EW);
    EXPECT_EQ(r.action, "CHANGE_PHASE");
    EXPECT_NEAR(static_cast<double>(r.age_ms), 100.0, 1.0);
}

// ── Malformed / hostile JSON never crashes and never actions ────────────────

TEST(CommandParser, RejectsMalformedJsonWithoutThrowing) {
    CommandParser p;
    const std::string vectors[] = {
        "",                                  // empty
        "   ",                               // whitespace
        "{not valid json",                   // truncated
        "{\"action\":}",                     // dangling value
        "}{",                                // garbage
        "\xff\xfe\x00\x01 binary",           // binary noise
        "[1,2,3]",                           // wrong shape (array)
        "42",                                // wrong shape (scalar)
        "null",                              // null
        "{\"action\":123}",                  // action wrong type
        R"({"action":"CHANGE_PHASE"})",      // missing phase + timestamp
        R"({"action":"REBOOT","phase":"GREEN_EW","timestamp":1})",  // unknown verb
        R"({"action":"CHANGE_PHASE","phase":"DROP TABLE","timestamp":1})", // bad phase
        R"({"action":"CHANGE_PHASE","phase":"GREEN_EW"})",          // missing ts
        R"({"action":"CHANGE_PHASE","phase":"GREEN_EW","timestamp":"soon"})", // ts wrong type
    };
    for (const auto& v : vectors) {
        auto r = p.parse(v, kNow);
        EXPECT_FALSE(r.is_actionable()) << "should reject: " << v;
    }
}

TEST(CommandParser, RejectsOversizedPayload) {
    CommandParser p;
    std::string big(CommandParser::MAX_PAYLOAD_BYTES + 1, 'A');
    auto r = p.parse(big, kNow);
    EXPECT_EQ(r.status, CommandStatus::OVERSIZED);
}

// ── Freshness gating (delayed / replayed / spoofed) ─────────────────────────

TEST(CommandParser, RejectsStaleCommand) {
    CommandParser p(/*max_age_ms=*/2000);
    auto r = p.parse(good_payload(kNow - 5000), kNow);  // 5s old
    EXPECT_EQ(r.status, CommandStatus::STALE);
    EXPECT_FALSE(r.is_actionable());
}

TEST(CommandParser, RejectsFutureTimestamp) {
    CommandParser p(/*max_age_ms=*/2000, /*max_future_skew_ms=*/1000);
    auto r = p.parse(good_payload(kNow + 10000), kNow);  // 10s in the future
    EXPECT_EQ(r.status, CommandStatus::FUTURE_TIMESTAMP);
    EXPECT_FALSE(r.is_actionable());
}

// ── Guard: hysteresis into and out of fixed-cycle fallback ──────────────────

TEST(CommandGuard, EntersFallbackAfterConsecutiveFailures) {
    CommandParser p;
    CommandGuard g(CommandGuard::Config{/*failure_threshold=*/3,
                                        /*recovery_streak=*/3,
                                        /*silence_timeout_ms=*/3000});
    auto bad = p.parse("{garbage", kNow);
    EXPECT_EQ(g.on_command(bad, kNow), GuardDecision::NONE);
    EXPECT_EQ(g.on_command(bad, kNow), GuardDecision::NONE);
    EXPECT_EQ(g.on_command(bad, kNow), GuardDecision::ENTER_FALLBACK);
    EXPECT_TRUE(g.in_fallback());
}

TEST(CommandGuard, RecoversOnlyAfterRecoveryStreak) {
    CommandParser p;
    CommandGuard g(CommandGuard::Config{3, 3, 3000});
    auto bad = p.parse("{garbage", kNow);
    for (int i = 0; i < 3; ++i) g.on_command(bad, kNow);
    ASSERT_TRUE(g.in_fallback());

    auto good = p.parse(good_payload(kNow), kNow);
    EXPECT_EQ(g.on_command(good, kNow), GuardDecision::NONE);   // 1
    EXPECT_EQ(g.on_command(good, kNow), GuardDecision::NONE);   // 2
    EXPECT_EQ(g.on_command(good, kNow), GuardDecision::EXIT_FALLBACK);  // 3
    EXPECT_FALSE(g.in_fallback());
}

TEST(CommandGuard, EntersFallbackOnSilenceTimeout) {
    CommandGuard g(CommandGuard::Config{3, 3, /*silence_timeout_ms=*/3000});
    g.arm(kNow);
    EXPECT_EQ(g.tick(kNow + 1000), GuardDecision::NONE);
    EXPECT_EQ(g.tick(kNow + 4000), GuardDecision::ENTER_FALLBACK);
    EXPECT_TRUE(g.in_fallback());
}

// ── Idempotency: monotonic sequence (audit A-2) ─────────────────────────────

TEST(CommandParser, ParsesSequenceNumber) {
    CommandParser p;
    auto r = p.parse(seq_payload(/*seq=*/42, kNow - 100), kNow);
    EXPECT_EQ(r.status, CommandStatus::OK);
    EXPECT_EQ(r.sequence_number, 42);
}

TEST(CommandGuard, RejectsStaleSequenceNumber) {
    CommandParser p;
    CommandGuard g;

    // First command (seq=5) establishes the high-water mark.
    auto fresh = p.parse(seq_payload(/*seq=*/5, kNow), kNow);
    ASSERT_TRUE(fresh.is_actionable());
    EXPECT_TRUE(g.accept_sequence(fresh));
    EXPECT_EQ(g.last_applied_seq(), 5);

    // A well-formed, fresh-timestamped command carrying an OBSOLETE sequence
    // number must be deterministically rejected (replay / out-of-order).
    auto stale_seq = p.parse(seq_payload(/*seq=*/3, kNow), kNow);
    ASSERT_TRUE(stale_seq.is_actionable());  // parses fine; only the seq is old
    EXPECT_FALSE(g.accept_sequence(stale_seq)) << "obsolete sequence must be discarded";
    EXPECT_EQ(g.last_applied_seq(), 5) << "high-water mark must not regress";

    // An exact duplicate of the last applied sequence is also rejected.
    auto dup = p.parse(seq_payload(/*seq=*/5, kNow), kNow);
    EXPECT_FALSE(g.accept_sequence(dup));

    // The next strictly-greater sequence is accepted and advances the mark.
    auto next = p.parse(seq_payload(/*seq=*/6, kNow), kNow);
    EXPECT_TRUE(g.accept_sequence(next));
    EXPECT_EQ(g.last_applied_seq(), 6);
}

TEST(CommandGuard, SingleGlitchDoesNotFlap) {
    CommandParser p;
    CommandGuard g(CommandGuard::Config{3, 3, 3000});
    g.arm(kNow);
    auto bad = p.parse("{garbage", kNow);
    auto good = p.parse(good_payload(kNow), kNow);
    g.on_command(bad, kNow);   // 1 failure
    g.on_command(good, kNow);  // resets failure streak
    EXPECT_FALSE(g.in_fallback());
}

// ── Strict mode: sequence_number is mandatory; decode never aborts ──────────

// A payload that is otherwise perfectly valid (correct action/phase/timestamp)
// but omits sequence_number must be rejected with MISSING_FIELD — there is no
// backward compatibility for unsequenced commands. The parse itself must never
// throw or abort the process.
TEST(CommandParser, RejectsPayloadMissingSequenceNumber) {
    CommandParser p;
    const std::string no_seq =
        R"({"action":"CHANGE_PHASE","phase":"GREEN_EW","priority":"HIGH","timestamp":)" +
        std::to_string(kNow) + "}";
    ParsedCommand r;
    ASSERT_NO_THROW({ r = p.parse(no_seq, kNow); });  // node survives the injection
    EXPECT_EQ(r.status, CommandStatus::MISSING_FIELD);
    EXPECT_FALSE(r.is_actionable());
}

// Malformed JSON with unclosed braces must be discarded as MALFORMED_JSON, and
// — critically — the hardened decode must survive it without throwing/aborting.
TEST(CommandParser, SurvivesUnclosedBracesWithoutAborting) {
    CommandParser p;
    const std::string unclosed =
        R"({"action":"CHANGE_PHASE","phase":"GREEN_EW","sequence_number":7,"timestamp":)" +
        std::to_string(kNow);  // deliberately no closing brace
    ParsedCommand r;
    ASSERT_NO_THROW({ r = p.parse(unclosed, kNow); });  // node survives the injection
    EXPECT_EQ(r.status, CommandStatus::MALFORMED_JSON);
    EXPECT_FALSE(r.is_actionable());

    // The minimal unclosed-brace case must also be survived and rejected.
    ASSERT_NO_THROW({ r = p.parse("{", kNow); });
    EXPECT_EQ(r.status, CommandStatus::MALFORMED_JSON);
    EXPECT_FALSE(r.is_actionable());
}
