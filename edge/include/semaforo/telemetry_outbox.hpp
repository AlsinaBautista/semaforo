// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Bounded outbound telemetry buffer — the backpressure boundary for the Edge
// node's MQTT egress (audit M-4).
//
// When the broker link drops, the main loop keeps producing telemetry every
// control tick. Without a bound, that backlog would grow without limit and
// exhaust RAM during a prolonged outage. The Outbox caps the buffer at a fixed
// number of messages and applies **drop-oldest** on overflow: the memory it
// consumes is therefore O(1) in the duration of the outage, and the messages
// retained are always the freshest (stale telemetry has no value once the link
// is back). It mirrors the inbound CommandQueue's drop-oldest contract.

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

namespace semaforo {

/// @brief A single buffered outbound MQTT message.
struct OutboundMessage {
    std::string topic;
    std::string payload;
    int         qos = 1;
    bool        retained = false;
};

/// @brief Thread-safe, bounded **drop-oldest** queue of outbound messages.
///
/// Producer: the main control loop (telemetry publish path). Consumer: the
/// flush path that drains to the broker once connected. A small mutex-guarded
/// deque is used for the same reasons as ``CommandQueue`` — short critical
/// sections, polled drain, trivially correct.
///
/// The hard ``capacity`` bound is the whole point: under a downed network the
/// queue can never grow past it, so the buffer's footprint is O(1) regardless
/// of how long the outage lasts.
class TelemetryOutbox {
public:
    /// @param capacity Maximum number of buffered messages before drop-oldest.
    explicit TelemetryOutbox(std::size_t capacity = 256) : capacity_(capacity) {}

    /// @brief Enqueue a message, evicting the oldest if the bound is reached.
    void push(OutboundMessage msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.size() >= capacity_) {
            q_.pop_front();   // drop the oldest, keep the freshest — O(1) memory
            ++dropped_;
        }
        q_.push_back(std::move(msg));
    }

    /// @brief Pop the oldest buffered message (FIFO drain order).
    /// @param out Receives the message if one was available.
    /// @return true if a message was popped, false if the queue was empty.
    bool try_pop(OutboundMessage& out) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        return true;
    }

    /// @brief Current number of buffered messages (never exceeds capacity).
    std::size_t size() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return q_.size();
    }

    /// @brief Hard capacity bound enforced by drop-oldest.
    std::size_t capacity() const { return capacity_; }

    /// @brief Total messages evicted due to the capacity bound (for telemetry).
    std::uint64_t dropped() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return dropped_;
    }

private:
    mutable std::mutex mtx_;
    std::deque<OutboundMessage> q_;
    std::size_t capacity_;
    std::uint64_t dropped_ = 0;
};

} // namespace semaforo
