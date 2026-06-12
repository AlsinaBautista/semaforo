// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Hardened command ingestion for the Edge node.
//
// This header isolates ALL parsing of untrusted MQTT payloads coming from the
// Brain (MAPPO) / Coordinator.  The two classes here are the only place where
// a raw network string is turned into an actionable phase, and they are
// designed so that NO malformed, oversized, truncated, type-confused, stale or
// spoofed payload can ever crash the process or reach the hardware:
//
//   * CommandParser  — pure, exception-free JSON validation. Returns a tagged
//                      result describing exactly why a payload was rejected.
//   * CommandGuard   — thread-safe freshness / fallback state machine. Decides
//                      when the AI must be blocked and the SafetyLayer dropped
//                      into a deterministic fixed cycle, and when the network
//                      has stabilised enough to hand control back.
//
// Both are header-only and dependency-light (nlohmann/json) so they can be
// unit-tested in isolation without the full perception pipeline.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

#include <nlohmann/json.hpp>

#include "semaforo/policy_engine.hpp"  // semaforo::Phase, phase_to_string
#include "utils/logger.hpp"            // LOG_WARN — warn-and-discard on JSON faults

namespace semaforo {

// ── Parse outcome taxonomy ──────────────────────────────────────────────────

/// @brief Why a raw command payload was accepted or rejected.
///
/// Every untrusted payload maps to exactly one of these. Anything that is not
/// ::OK must never be applied to the signal hardware.
enum class CommandStatus {
    OK,                ///< Well-formed, fresh, actionable command.
    EMPTY,             ///< Payload was empty / whitespace only.
    OVERSIZED,         ///< Payload exceeded the hard byte limit (DoS guard).
    MALFORMED_JSON,    ///< nlohmann threw while parsing — not valid JSON.
    NOT_AN_OBJECT,     ///< Valid JSON but not a top-level object (e.g. array/number).
    MISSING_FIELD,     ///< A required key was absent.
    WRONG_TYPE,        ///< A field was present but of the wrong JSON type.
    UNKNOWN_ACTION,    ///< "action" was not a recognised verb.
    INVALID_PHASE,     ///< "phase" string did not map to a known Phase.
    STALE,             ///< Command timestamp older than the staleness budget.
    FUTURE_TIMESTAMP,  ///< Timestamp implausibly in the future (clock skew/spoof).
};

/// @brief Human-readable label for a CommandStatus (for logs / telemetry).
inline const char* command_status_to_string(CommandStatus s) {
    switch (s) {
        case CommandStatus::OK:               return "OK";
        case CommandStatus::EMPTY:            return "EMPTY";
        case CommandStatus::OVERSIZED:        return "OVERSIZED";
        case CommandStatus::MALFORMED_JSON:   return "MALFORMED_JSON";
        case CommandStatus::NOT_AN_OBJECT:    return "NOT_AN_OBJECT";
        case CommandStatus::MISSING_FIELD:    return "MISSING_FIELD";
        case CommandStatus::WRONG_TYPE:       return "WRONG_TYPE";
        case CommandStatus::UNKNOWN_ACTION:   return "UNKNOWN_ACTION";
        case CommandStatus::INVALID_PHASE:    return "INVALID_PHASE";
        case CommandStatus::STALE:            return "STALE";
        case CommandStatus::FUTURE_TIMESTAMP: return "FUTURE_TIMESTAMP";
    }
    return "UNKNOWN";
}

/// @brief Result of validating a single raw command payload.
struct ParsedCommand {
    CommandStatus status = CommandStatus::EMPTY;
    std::string   action;                 ///< Recognised verb (e.g. "CHANGE_PHASE").
    Phase         phase = Phase::ALL_RED; ///< Only meaningful when status == OK.
    std::string   priority;               ///< Optional priority hint ("HIGH"...).
    std::int64_t  timestamp_ms = 0;       ///< Brain-supplied epoch millis.
    std::int64_t  sequence_number = -1;   ///< Monotonic seq for idempotency; <0 if absent.
    std::int64_t  age_ms = 0;             ///< now - timestamp_ms (signed).
    std::string   detail;                 ///< Human-readable rejection reason.

    /// @brief True only for a fully-validated, fresh, actionable command.
    bool is_actionable() const { return status == CommandStatus::OK; }
};

// ── Pure, exception-free parser ─────────────────────────────────────────────

/// @brief Validates untrusted Brain/Coordinator JSON command payloads.
///
/// The parser is deliberately strict and total: it never throws and never
/// returns an actionable result for anything it cannot fully verify. It guards
/// against the chaos vectors exercised by ``chaos_tester.py``:
///   * truncated / malformed JSON                → MALFORMED_JSON
///   * wrong top-level shape (arrays, scalars)   → NOT_AN_OBJECT
///   * missing or mistyped fields                → MISSING_FIELD / WRONG_TYPE
///   * unknown phase strings / injection         → INVALID_PHASE
///   * gigantic payloads                         → OVERSIZED
///   * replayed / delayed (stale) commands       → STALE
///   * clock-skewed or spoofed future timestamps → FUTURE_TIMESTAMP
class CommandParser {
public:
    /// Hard cap on accepted payload size. A legitimate command is ~120 bytes;
    /// anything beyond this is treated as hostile and rejected before parsing.
    static constexpr std::size_t MAX_PAYLOAD_BYTES = 8 * 1024;

    /// @param max_age_ms          Commands older than this are STALE (default 2s).
    /// @param max_future_skew_ms  Timestamps further ahead than this are rejected.
    explicit CommandParser(std::int64_t max_age_ms = 2000,
                           std::int64_t max_future_skew_ms = 1000)
        : max_age_ms_(max_age_ms), max_future_skew_ms_(max_future_skew_ms) {}

    /// @brief Current monotonic-ish wall clock in epoch milliseconds.
    static std::int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    /// @brief Map a phase string to a Phase enum. Returns false if unknown.
    static bool phase_from_string(const std::string& s, Phase& out) {
        if (s == "GREEN_NS")        { out = Phase::GREEN_NS;       return true; }
        if (s == "YELLOW_NS")       { out = Phase::YELLOW_NS;      return true; }
        if (s == "ALL_RED")         { out = Phase::ALL_RED;        return true; }
        if (s == "GREEN_EW")        { out = Phase::GREEN_EW;       return true; }
        if (s == "YELLOW_EW")       { out = Phase::YELLOW_EW;      return true; }
        if (s == "GREEN_NS_LEFT")   { out = Phase::GREEN_NS_LEFT;  return true; }
        if (s == "GREEN_EW_LEFT")   { out = Phase::GREEN_EW_LEFT;  return true; }
        return false;
    }

    /// @brief Validate a raw payload against the current wall clock.
    /// @param raw    Untrusted bytes straight off the MQTT topic.
    /// @param now_ms Reference "now" (injected for deterministic testing).
    ParsedCommand parse(const std::string& raw, std::int64_t now_ms) const {
        ParsedCommand cmd;

        // ── 0. Cheap structural guards before touching the JSON parser ──────
        if (raw.empty()) {
            cmd.status = CommandStatus::EMPTY;
            cmd.detail = "empty payload";
            return cmd;
        }
        if (raw.size() > MAX_PAYLOAD_BYTES) {
            cmd.status = CommandStatus::OVERSIZED;
            cmd.detail = "payload " + std::to_string(raw.size()) +
                         "B exceeds limit " + std::to_string(MAX_PAYLOAD_BYTES) + "B";
            return cmd;
        }

        // ── 1..6. ALL nlohmann::json access lives inside this hermetic try ──
        // Even though every field is type-checked before extraction and parsing
        // uses allow_exceptions=false, the entire decode is still wrapped so
        // that NO nlohmann exception can ever escape into the control loop and
        // abort() the process. parse_error / type_error / out_of_range are each
        // caught explicitly below: the garbage is logged at WARN and discarded
        // as a non-actionable result.
        try {
            // ── 1. Parse JSON without ever throwing ─────────────────────────
            // nlohmann::json::parse(..., nullptr, false) returns a `discarded`
            // sentinel on error instead of throwing — first line of defence.
            nlohmann::json root =
                nlohmann::json::parse(raw, /*cb=*/nullptr, /*allow_exceptions=*/false);

            if (root.is_discarded()) {
                cmd.status = CommandStatus::MALFORMED_JSON;
                cmd.detail = "invalid JSON syntax";
                return cmd;
            }
            if (!root.is_object()) {
                cmd.status = CommandStatus::NOT_AN_OBJECT;
                cmd.detail = "top-level JSON is not an object";
                return cmd;
            }

            // ── 2. action (required string) ─────────────────────────────────
            auto action_it = root.find("action");
            if (action_it == root.end()) {
                cmd.status = CommandStatus::MISSING_FIELD;
                cmd.detail = "missing 'action'";
                return cmd;
            }
            if (!action_it->is_string()) {
                cmd.status = CommandStatus::WRONG_TYPE;
                cmd.detail = "'action' is not a string";
                return cmd;
            }
            cmd.action = action_it->get<std::string>();
            if (cmd.action != "CHANGE_PHASE") {
                cmd.status = CommandStatus::UNKNOWN_ACTION;
                cmd.detail = "unsupported action '" + cmd.action + "'";
                return cmd;
            }

            // ── 3. phase (required string, must map to a known enum) ────────
            auto phase_it = root.find("phase");
            if (phase_it == root.end()) {
                cmd.status = CommandStatus::MISSING_FIELD;
                cmd.detail = "missing 'phase'";
                return cmd;
            }
            if (!phase_it->is_string()) {
                cmd.status = CommandStatus::WRONG_TYPE;
                cmd.detail = "'phase' is not a string";
                return cmd;
            }
            if (!phase_from_string(phase_it->get<std::string>(), cmd.phase)) {
                cmd.status = CommandStatus::INVALID_PHASE;
                cmd.detail = "unknown phase '" + phase_it->get<std::string>() + "'";
                return cmd;
            }

            // ── 4. priority (optional string) ───────────────────────────────
            auto prio_it = root.find("priority");
            if (prio_it != root.end() && prio_it->is_string()) {
                cmd.priority = prio_it->get<std::string>();
            }

            // ── 4b. sequence_number (REQUIRED integer; idempotency / replay) ─
            // Strict-mode contract: a payload without a sequence number is
            // rejected outright with MISSING_FIELD. There is deliberately NO
            // backward compatibility for unsequenced commands — every command
            // must be replay-protectable.
            auto seq_it = root.find("sequence_number");
            if (seq_it == root.end()) {
                cmd.status = CommandStatus::MISSING_FIELD;
                cmd.detail = "missing 'sequence_number'";
                return cmd;
            }
            if (!seq_it->is_number_integer()) {
                cmd.status = CommandStatus::WRONG_TYPE;
                cmd.detail = "'sequence_number' is not an integer";
                return cmd;
            }
            cmd.sequence_number = seq_it->get<std::int64_t>();

            // ── 5. timestamp (required integer; freshness gate) ─────────────
            auto ts_it = root.find("timestamp");
            if (ts_it == root.end()) {
                cmd.status = CommandStatus::MISSING_FIELD;
                cmd.detail = "missing 'timestamp'";
                return cmd;
            }
            if (!ts_it->is_number_integer()) {
                cmd.status = CommandStatus::WRONG_TYPE;
                cmd.detail = "'timestamp' is not an integer";
                return cmd;
            }
            cmd.timestamp_ms = ts_it->get<std::int64_t>();
            cmd.age_ms = now_ms - cmd.timestamp_ms;

            if (cmd.age_ms < -max_future_skew_ms_) {
                cmd.status = CommandStatus::FUTURE_TIMESTAMP;
                cmd.detail = "timestamp " + std::to_string(-cmd.age_ms) +
                             "ms in the future";
                return cmd;
            }
            if (cmd.age_ms > max_age_ms_) {
                cmd.status = CommandStatus::STALE;
                cmd.detail = "command age " + std::to_string(cmd.age_ms) +
                             "ms exceeds budget " + std::to_string(max_age_ms_) + "ms";
                return cmd;
            }

            // ── 6. Fully validated ──────────────────────────────────────────
            cmd.status = CommandStatus::OK;
            cmd.detail = "ok";
            return cmd;
        } catch (const nlohmann::json::parse_error& e) {
            LOG_WARN("CommandParser: discarding payload — json parse_error: {}", e.what());
            cmd.status = CommandStatus::MALFORMED_JSON;
            cmd.detail = std::string("json parse_error: ") + e.what();
            return cmd;
        } catch (const nlohmann::json::type_error& e) {
            LOG_WARN("CommandParser: discarding payload — json type_error: {}", e.what());
            cmd.status = CommandStatus::WRONG_TYPE;
            cmd.detail = std::string("json type_error: ") + e.what();
            return cmd;
        } catch (const nlohmann::json::out_of_range& e) {
            LOG_WARN("CommandParser: discarding payload — json out_of_range: {}", e.what());
            cmd.status = CommandStatus::MISSING_FIELD;
            cmd.detail = std::string("json out_of_range: ") + e.what();
            return cmd;
        }
    }

    std::int64_t max_age_ms() const { return max_age_ms_; }

private:
    std::int64_t max_age_ms_;
    std::int64_t max_future_skew_ms_;
};

// ── Freshness / fallback state machine ──────────────────────────────────────

/// @brief Decision made by the guard after observing a parse result or a tick.
enum class GuardDecision {
    NONE,              ///< No state change; carry on.
    ENTER_FALLBACK,    ///< Block the AI and engage the deterministic fixed cycle.
    EXIT_FALLBACK,     ///< Network stabilised; hand control back to the AI.
};

/// @brief Thread-safe watchdog that decides when to trust the Brain.
///
/// The MQTT callback runs on the Paho client thread while the control loop runs
/// on the main thread, so this class is fully synchronised. It implements
/// hysteresis so transient glitches do not flap the signal:
///
///   * Healthy → Fallback when EITHER
///       - ``failure_threshold`` consecutive bad/stale payloads arrive, OR
///       - no valid command is seen for ``silence_timeout_ms`` (broker down,
///         link saturated, Brain crashed).
///   * Fallback → Healthy only after ``recovery_streak`` consecutive *fresh,
///     valid* commands — proving the link is genuinely back, not just blipping.
///
/// While in fallback the caller drives ``SafetyLayer::set_emergency_mode(true)``
/// so the intersection runs a safe fixed cycle until the network stabilises.
class CommandGuard {
public:
    struct Config {
        int          failure_threshold = 3;     ///< bad payloads before fallback
        int          recovery_streak   = 3;     ///< good payloads before recovery
        std::int64_t silence_timeout_ms = 10000; ///< no-valid-command timeout
    };

    CommandGuard() : cfg_(Config{}) {}
    explicit CommandGuard(Config cfg) : cfg_(cfg) {}

    /// @brief Feed the result of parsing one payload into the state machine.
    /// @return The control decision implied by this observation.
    GuardDecision on_command(const ParsedCommand& parsed, std::int64_t now_ms) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (parsed.is_actionable()) {
            last_valid_ms_.store(now_ms, std::memory_order_relaxed);
            consecutive_failures_ = 0;
            ++consecutive_ok_;
            if (in_fallback_ && consecutive_ok_ >= cfg_.recovery_streak) {
                in_fallback_ = false;
                consecutive_ok_ = 0;
                return GuardDecision::EXIT_FALLBACK;
            }
        } else {
            consecutive_ok_ = 0;
            ++consecutive_failures_;
            ++total_rejected_;
            if (!in_fallback_ && consecutive_failures_ >= cfg_.failure_threshold) {
                in_fallback_ = true;
                return GuardDecision::ENTER_FALLBACK;
            }
        }
        return GuardDecision::NONE;
    }

    /// @brief Periodic tick from the control loop to catch *silence* (no
    /// commands at all), which on_command can never observe.
    /// @return ENTER_FALLBACK if the silence timeout has elapsed.
    GuardDecision tick(std::int64_t now_ms) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (in_fallback_) return GuardDecision::NONE;
        const std::int64_t last = last_valid_ms_.load(std::memory_order_relaxed);
        if (last == 0) return GuardDecision::NONE;  // never armed yet
        if (now_ms - last > cfg_.silence_timeout_ms) {
            in_fallback_ = true;
            consecutive_ok_ = 0;
            return GuardDecision::ENTER_FALLBACK;
        }
        return GuardDecision::NONE;
    }

    /// @brief Idempotency gate: accept a command only if its sequence number
    /// is strictly newer than the last one applied.
    ///
    /// Enforces the monotonic-sequence contract (audit A-2): a replayed or
    /// out-of-order command (``sequence_number <= last_applied_seq``) is
    /// deterministically discarded so it can never be re-applied to the
    /// hardware. The caller only ever feeds *actionable* commands here, and the
    /// strict-mode parser guarantees every such command carries a valid integer
    /// sequence, so no unsequenced-command path exists to handle.
    /// @return true if the command is fresh and should be applied; false if it
    ///         is a stale/duplicate sequence and must be dropped.
    bool accept_sequence(const ParsedCommand& parsed) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (parsed.sequence_number <= last_applied_seq_) {      // replay / out of order
            ++total_rejected_;
            return false;
        }
        last_applied_seq_ = parsed.sequence_number;
        return true;
    }

    /// @brief Highest sequence number applied so far (-1 if none yet).
    std::int64_t last_applied_seq() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return last_applied_seq_;
    }

    /// @brief Arm the watchdog. Call once the node is online and expecting
    /// commands so that the silence timeout starts counting.
    void arm(std::int64_t now_ms) {
        last_valid_ms_.store(now_ms, std::memory_order_relaxed);
    }

    bool in_fallback() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return in_fallback_;
    }

    std::uint64_t total_rejected() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return total_rejected_;
    }

private:
    Config cfg_;
    mutable std::mutex mtx_;
    std::atomic<std::int64_t> last_valid_ms_{0};
    int consecutive_failures_ = 0;
    int consecutive_ok_ = 0;
    bool in_fallback_ = false;
    std::uint64_t total_rejected_ = 0;
    std::int64_t last_applied_seq_ = -1;
};

} // namespace semaforo
