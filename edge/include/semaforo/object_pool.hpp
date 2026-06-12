// Semáforo Inteligente - Edge Perception Module
// Copyright (c) 2026 BecasHub. All rights reserved.
//
// Fixed-capacity object pool for per-frame perception results (audit M-5).
//
// The 30 FPS inference loop produces a fresh batch of ``Detection`` and
// ``TrackedObject`` records every tick. Returning those batches by value (the
// original ``std::vector<Detection> detect(...)`` shape) reallocates on the
// heap **every frame** — 30 malloc/free storms per second, forever. On a
// long-running, safety-critical edge node that is the classic recipe for heap
// fragmentation and an eventual latency spike (or, in the worst case, an
// allocation stall that freezes the control loop). M-5 eradicates it.
//
// This pool is the per-frame store for those records. The whole point is that
// it allocates **exactly once** — in the constructor — and then recycles that
// same slab forever:
//
//   * Construction reserves ``capacity`` slots up front (the only allocation).
//   * ``clear()`` is an O(1) reset that keeps the backing storage; the next
//     frame refills the same memory.
//   * ``acquire()`` hands back the next reusable slot and never reallocates
//     (it is hard-bounded at ``capacity`` — a deterministic cap is desirable
//     anyway: an intersection can only hold so many vehicles, and an unbounded
//     result set would itself be a memory hazard).
//
// It is deliberately a thin wrapper over a single contiguous ``std::vector``:
// no free-list, no indirection, no per-slot bookkeeping. Access is a plain
// array index, so it adds zero overhead on the hot path (the M-5 constraint:
// the fix must not slow memory access down). Consumers that expect a
// ``std::vector`` (the tracker, the queue estimator) read it through
// ``view()`` with no copy.
//
// Single-threaded by contract: only the main control loop (the single writer)
// touches the pool, so it needs no locking.

#pragma once

#include <cstddef>
#include <vector>

namespace semaforo {

/// @brief Fixed-capacity, allocation-free-after-construction object pool.
///
/// Backed by one contiguous ``std::vector<T>`` whose capacity is reserved once
/// and never grown again. ``clear()`` + ``acquire()`` recycle that storage
/// across frames, so steady-state operation performs **no heap allocation**.
///
/// @tparam T  Trivially-recyclable record type (e.g. ``Detection``,
///            ``TrackedObject``). Slots are default-constructed on first use
///            and overwritten by the caller.
template <typename T>
class ObjectPool {
public:
    /// @brief Construct a pool and reserve its (only ever) backing allocation.
    /// @param capacity  Hard upper bound on live slots per frame.
    explicit ObjectPool(std::size_t capacity) : capacity_(capacity) {
        slots_.reserve(capacity);  // THE one and only heap allocation
    }

    /// @brief O(1) reset: drop all live slots but keep the backing storage.
    ///
    /// ``std::vector::clear()`` does not release capacity, so the next frame
    /// refills the same memory with zero allocation.
    void clear() noexcept { slots_.clear(); }

    /// @brief Acquire the next reusable slot, or ``nullptr`` if at capacity.
    ///
    /// Never reallocates: the constructor reserved ``capacity_`` slots, and we
    /// refuse to grow past that bound. The returned pointer is valid until the
    /// next ``clear()`` (no reallocation can move it within a frame).
    ///
    /// @return Pointer to a default-constructed slot to overwrite, or
    ///         ``nullptr`` when the pool is full (caller drops the record).
    T* acquire() {
        if (slots_.size() >= capacity_) {
            return nullptr;  // bounded: never grow, never reallocate
        }
        slots_.emplace_back();  // no realloc — within reserved capacity
        return &slots_.back();
    }

    /// @brief Number of live slots this frame (never exceeds capacity).
    std::size_t size() const noexcept { return slots_.size(); }

    /// @brief Hard capacity bound enforced by ``acquire()``.
    std::size_t capacity() const noexcept { return capacity_; }

    /// @brief Whether the pool has reached its capacity bound.
    bool full() const noexcept { return slots_.size() >= capacity_; }

    /// @brief Indexed access to a live slot (no bounds check on the hot path).
    T& operator[](std::size_t i) noexcept { return slots_[i]; }
    const T& operator[](std::size_t i) const noexcept { return slots_[i]; }

    /// @brief Zero-copy view as a ``std::vector`` for downstream consumers.
    ///
    /// Lets components that already take ``const std::vector<T>&`` (the tracker,
    /// the queue estimator) read the pooled batch directly, with no copy and no
    /// allocation.
    const std::vector<T>& view() const noexcept { return slots_; }

    // Range-based-for support over the live slots.
    typename std::vector<T>::iterator begin() noexcept { return slots_.begin(); }
    typename std::vector<T>::iterator end() noexcept { return slots_.end(); }
    typename std::vector<T>::const_iterator begin() const noexcept { return slots_.begin(); }
    typename std::vector<T>::const_iterator end() const noexcept { return slots_.end(); }

private:
    std::vector<T> slots_;   ///< Single reserved-once backing slab.
    std::size_t capacity_;   ///< Hard bound; storage never grows past this.
};

} // namespace semaforo
