// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A coalesced wake flag for one owner thread (D-048,
// docs/async-model.md#bounded-queues-and-progress-under-saturation): any
// number of producers publish their results first, then signal; the owner
// consumes the signal, rechecks everything that could have been published,
// and sleeps only while no signal is pending. Many signals before the owner
// looks cost one wakeup.
//
// Why no wakeup is lost: a producer's publication happens before its
// signal (release on the flag, acquire where the owner consumes it). A
// producer that sets the flag while the owner is checking it either is seen
// by that check, or notifies under the mutex, which it can only take once
// the owner is waiting.

#ifndef JITLLM_BASE_WAKE_H_
#define JITLLM_BASE_WAKE_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

namespace jitllm::base {

class WakeFlag {
 public:
  // Any thread, after publishing: wake the owner.
  void Signal() {
    if (!pending_.exchange(true, std::memory_order_acq_rel)) {
      const std::scoped_lock lock(mutex_);
      wake_.notify_one();
    }
  }

  // The owner: takes a pending signal, if there is one, without waiting.
  bool Consume() { return pending_.exchange(false, std::memory_order_acquire); }

  // The owner: waits until a signal is pending or `timeout` passes, and
  // takes the signal. True if there was one. Spurious returns are possible
  // only as `false`, which the owner treats as a timer tick.
  template <typename Rep, typename Period>
  bool WaitFor(std::chrono::duration<Rep, Period> timeout) {
    std::unique_lock lock(mutex_);
    return wake_.wait_for(lock, timeout, [this] { return Consume(); });
  }

 private:
  std::atomic<bool> pending_{false};
  std::mutex mutex_;
  std::condition_variable wake_;
};

}  // namespace jitllm::base

#endif  // JITLLM_BASE_WAKE_H_
