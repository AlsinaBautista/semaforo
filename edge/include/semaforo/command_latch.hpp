// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// TTL-latched remote directive: the Brain's phase command as a *persistent*
// primary control input.
//
// Without latching, a remote command only influenced the single tick on which
// it arrived; the node fell back to the local policy the instant no command was
// drained. With this latch the most recent valid remote phase is *enclaved* and
// remains the primary directive across ticks — the Brain does not have to re-send
// it every frame. A strict TTL bounds how long a directive stays authoritative
// without a fresh update: once ``ttl_ms`` elapses with no ``update()``, the
// directive is ``stale`` and the node degrades control to the local policy.
//
// Single-writer by design: only the main control loop touches this (Actor
// model), so no internal synchronisation is needed. The clock is injected
// (``now_ms``) so the TTL behaviour is deterministically testable.

#pragma once

#include <cstdint>
#include <optional>

#include "semaforo/policy_engine.hpp"  // semaforo::Phase

namespace semaforo {

/// @brief Latches the latest valid remote directive with a strict TTL.
class DirectiveLatch {
public:
    /// Default time-to-live: 4000 ms without a Brain update marks the
    /// directive stale and triggers degradation to the local policy.
    static constexpr std::int64_t DEFAULT_TTL_MS = 4000;

    explicit DirectiveLatch(std::int64_t ttl_ms = DEFAULT_TTL_MS)
        : ttl_ms_(ttl_ms) {}

    /// @brief Latch a fresh remote directive, (re)starting its TTL window.
    /// @param phase  The phase the Brain commands.
    /// @param now_ms Receive time in epoch milliseconds.
    void update(Phase phase, std::int64_t now_ms) {
        phase_ = phase;
        last_update_ms_ = now_ms;
        valid_ = true;
    }

    /// @brief The latched directive, iff one exists and is still within TTL.
    /// @param now_ms Current time in epoch milliseconds.
    /// @return The enclaved phase, or ``std::nullopt`` if none/stale.
    std::optional<Phase> get(std::int64_t now_ms) const {
        if (!valid_) return std::nullopt;
        if (now_ms - last_update_ms_ >= ttl_ms_) return std::nullopt;  // stale
        return phase_;
    }

    /// @brief True iff a directive was latched and its TTL has since lapsed.
    bool is_stale(std::int64_t now_ms) const {
        return valid_ && (now_ms - last_update_ms_ >= ttl_ms_);
    }

    /// @brief True iff any directive has ever been latched.
    bool has_directive() const { return valid_; }

    /// @brief Age of the latched directive in ms (0 if none latched).
    std::int64_t age_ms(std::int64_t now_ms) const {
        return valid_ ? (now_ms - last_update_ms_) : 0;
    }

    /// @brief Forget the latched directive entirely.
    void clear() { valid_ = false; }

    /// @brief The configured TTL in milliseconds.
    std::int64_t ttl_ms() const { return ttl_ms_; }

private:
    std::int64_t ttl_ms_;
    Phase phase_ = Phase::ALL_RED;
    std::int64_t last_update_ms_ = 0;
    bool valid_ = false;
};

} // namespace semaforo
