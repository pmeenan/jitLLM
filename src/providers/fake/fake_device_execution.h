// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The deterministic device-execution fake: each stream is a queue of
// copies, cross-stream waits and fence markers that runs only when a test
// steps it, so completion order is the test's to choose. Copies move bytes
// between the fake device memory's addresses (which are host addresses)
// when they run, so a consumer that reads before its fence completes sees
// the old bytes. Releasing a fence before it was seen complete, or
// destroying a stream with work not yet behind a released fence, is
// refused, as the CUDA provider refuses them. Every call takes one lock, so
// a completion lane may query while the submission lane submits.

#ifndef JITLLM_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_
#define JITLLM_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <mutex>
#include <optional>

#include "base/bytes.h"
#include "base/ids.h"
#include "providers/device_execution.h"

namespace jitllm::providers::fake {

class FakeDeviceExecution final : public DeviceExecution {
 public:
  std::expected<StreamId, Failure> CreateStream() override;
  std::expected<void, Failure> DestroyStream(StreamId stream) override;
  std::expected<void, Failure> Copy(StreamId stream, std::uint64_t destination,
                                    std::uint64_t source, Bytes size) override;
  std::expected<void, Failure> Wait(StreamId stream, FenceId fence) override;
  std::expected<FenceId, Failure> Record(StreamId stream) override;
  std::expected<FenceState, Failure> Query(FenceId fence) override;
  std::expected<void, Failure> Release(FenceId fence) override;

  // Runs the stream's next step, if it can: false if the stream is empty
  // or waiting on a fence that is not yet complete.
  bool Step(StreamId stream);
  // Steps every stream until none can move.
  void Drain();
  // The next query of `fence` reports this failure (a device fault).
  void FailNextQuery(FenceId fence, ProviderError error) {
    const std::scoped_lock lock(mutex_);
    fault_ = {fence, error};
  }

  std::size_t streams() const {
    const std::scoped_lock lock(mutex_);
    return streams_.size();
  }
  std::size_t fences() const {
    const std::scoped_lock lock(mutex_);
    return fences_.size();
  }

 private:
  struct Queued {
    enum class Kind : std::uint8_t { kCopy, kWait, kFence } kind = Kind::kCopy;
    std::uint64_t destination = 0;
    std::uint64_t source = 0;
    Bytes size;
    FenceId fence;
  };
  struct Stream {
    std::deque<Queued> steps;
    std::size_t fences = 0;  // unreleased fences recorded on it
    bool unfenced = false;   // work queued since its last fence
  };
  struct Fence {
    StreamId stream;
    bool complete = false;
    bool seen = false;  // a query reported it complete
  };

  bool StepLocked(StreamId stream);

  mutable std::mutex mutex_;
  base::SlotTable<StreamTag, Stream> streams_;
  base::SlotTable<FenceTag, Fence> fences_;
  std::optional<std::pair<FenceId, ProviderError>> fault_;
};

}  // namespace jitllm::providers::fake

#endif  // JITLLM_PROVIDERS_FAKE_FAKE_DEVICE_EXECUTION_H_
