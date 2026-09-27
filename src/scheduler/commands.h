// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The commands the scheduler thread publishes to its service lanes (D-048,
// docs/async-model.md#submission-and-completion-protocol). Each names the
// operation it belongs to by its generation-tagged identity, whose mailbox
// on the completion board receives what the lane observes. A command is a
// value: it owns no memory, and the backing it names is protected by the
// leases or load the scheduler recorded before publishing it.

#ifndef JITLLM_SCHEDULER_COMMANDS_H_
#define JITLLM_SCHEDULER_COMMANDS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <variant>

#include "base/bytes.h"
#include "providers/device_execution.h"
#include "providers/direct_reader.h"
#include "scheduler/completions.h"

namespace jitllm::scheduler {

using base::Bytes;

// An operation's identity as one number: the storage lane's read key and
// waiter.
constexpr std::uint64_t KeyOf(OperationId operation) {
  return (std::uint64_t{operation.index()} << 32U) | operation.generation();
}
constexpr OperationId OperationOf(std::uint64_t key) {
  return {static_cast<std::uint32_t>(key >> 32U), static_cast<std::uint32_t>(key)};
}

// Storage lane: read a whole range into protected memory, or withdraw
// interest in a read (best-effort cancellation; the read still drains).
struct ReadCommand {
  OperationId operation;
  providers::ReadSpec spec;
};
struct CancelRead {
  OperationId operation;
};
using StorageCommand = std::variant<ReadCommand, CancelRead>;

// Device submission lane: copies queued in order on one of the lane's
// streams, then a fence after them, which the device completion lane
// watches.
struct DeviceCopy {
  std::uint64_t destination = 0;
  std::uint64_t source = 0;
  Bytes size;
};
inline constexpr std::size_t kMaxDeviceCopies = 4;
struct DeviceWork {
  std::uint32_t stream = 0;  // an index into the lane's streams
  std::array<DeviceCopy, kMaxDeviceCopies> copies{};
  std::size_t count = 0;
};
struct DeviceCommand {
  OperationId operation;
  DeviceWork work;
};

// CPU worker lane: bounded host work, such as verification, over memory
// the operation's lease protects. It returns its outcome and the bytes it
// covered; its access ends when it returns.
struct CpuResult {
  Outcome outcome = Outcome::kSucceeded;
  std::uint64_t bytes = 0;
};
using CpuJob = std::move_only_function<CpuResult()>;
struct CpuCommand {
  OperationId operation;
  CpuJob job;
};

}  // namespace jitllm::scheduler

#endif  // JITLLM_SCHEDULER_COMMANDS_H_
