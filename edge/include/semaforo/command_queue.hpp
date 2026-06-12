// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Single-Producer / Single-Consumer command queue — the hand-off boundary of
// the Actor / Single-Writer design that makes the Edge node thread-safe.
//
//   Producer (one thread): the Paho MQTT callback. Its ONLY job is to parse an
//                          untrusted payload and ``push`` the result here.
//   Consumer (one thread): the main control loop. It is the SINGLE WRITER of all
//                          control state (SafetyLayer, CommandGuard, the signal
//                          actuator) and drains this queue each iteration.
//
// Because the two roles never touch shared control state directly, there is no
// data race on the SafetyLayer or the SignalController: every decision is made
// on the main thread. This queue is the only shared object between them and is
// fully synchronised.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>

#include "semaforo/command_parser.hpp"  // semaforo::ParsedCommand

namespace semaforo {

/// @brief Thread-safe, bounded SPSC queue of parsed commands.
///
/// A small ``std::mutex``-guarded deque is intentionally chosen over a lock-free
/// structure: the consumer polls once per control tick (no blocking wait needed)
/// and the critical sections are trivially short, so a lock is both correct and
/// simpler to reason about for a safety-critical node.
///
/// The queue is **bounded with drop-oldest** semantics: under a command flood it
/// can never exhaust memory, and because the consumer only ever acts on the most
/// recent command, dropping stale backlog is the desired behaviour.
class CommandQueue {
public:
    /// @param capacity Maximum number of queued commands before drop-oldest.
    explicit CommandQueue(std::size_t capacity = 64) : capacity_(capacity) {}

    /// @brief Producer side: enqueue a parsed command (MQTT callback thread).
    void push(const ParsedCommand& cmd) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.size() >= capacity_) {
            q_.pop_front();   // drop the oldest, keep the freshest
            ++dropped_;
        }
        q_.push_back(cmd);
    }

    /// @brief Consumer side: pop the oldest pending command (main loop thread).
    /// @param out Receives the command if one was available.
    /// @return true if a command was popped, false if the queue was empty.
    bool try_pop(ParsedCommand& out) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.empty()) return false;
        out = q_.front();
        q_.pop_front();
        return true;
    }

    /// @brief Current number of queued commands.
    std::size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return q_.size();
    }

    /// @brief Total commands dropped due to the capacity bound (for telemetry).
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return dropped_;
    }

private:
    mutable std::mutex mtx_;
    std::deque<ParsedCommand> q_;
    std::size_t capacity_;
    std::uint64_t dropped_ = 0;
};

} // namespace semaforo
