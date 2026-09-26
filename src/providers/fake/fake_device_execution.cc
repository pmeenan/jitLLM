// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/fake/fake_device_execution.h"

#include <cstdint>
#include <cstring>
#include <expected>
#include <mutex>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "base/check.h"
#include "providers/device_execution.h"

namespace jitllm::providers::fake {
namespace {

std::unexpected<Failure> Invalid(std::string detail) {
  return std::unexpected(Failure{.error = ProviderError::kInvalid, .detail = std::move(detail)});
}

void* At(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace

std::expected<StreamId, Failure> FakeDeviceExecution::CreateStream() {
  const std::scoped_lock lock(mutex_);
  const StreamId id = streams_.Insert(Stream{});
  if (!id.valid()) {
    return std::unexpected(
        Failure{.error = ProviderError::kFailed, .detail = "the stream table is full"});
  }
  return id;
}

std::expected<void, Failure> FakeDeviceExecution::DestroyStream(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  const Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  if (found->fences > 0 || found->unfenced || !found->steps.empty()) {
    return Invalid("the stream has work not yet behind a completed fence");
  }
  (void)streams_.Erase(stream);
  return {};
}

std::expected<void, Failure> FakeDeviceExecution::Copy(StreamId stream, std::uint64_t destination,
                                                       std::uint64_t source, Bytes size) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  found->unfenced = true;
  found->steps.push_back(Queued{.kind = Queued::Kind::kCopy,
                                .destination = destination,
                                .source = source,
                                .size = size,
                                .fence = {}});
  return {};
}

std::expected<void, Failure> FakeDeviceExecution::Wait(StreamId stream, FenceId fence) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr || fences_.Find(fence) == nullptr) {
    return Invalid("stale or unknown stream or fence");
  }
  found->unfenced = true;
  found->steps.push_back(Queued{
      .kind = Queued::Kind::kWait, .destination = 0, .source = 0, .size = {}, .fence = fence});
  return {};
}

std::expected<FenceId, Failure> FakeDeviceExecution::Record(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  Stream* found = streams_.Find(stream);
  if (found == nullptr) {
    return Invalid("stale or unknown stream");
  }
  const FenceId fence = fences_.Insert(Fence{.stream = stream, .complete = false, .seen = false});
  if (!fence.valid()) {
    return std::unexpected(
        Failure{.error = ProviderError::kFailed, .detail = "the fence table is full"});
  }
  found = streams_.Find(stream);
  found->steps.push_back(Queued{
      .kind = Queued::Kind::kFence, .destination = 0, .source = 0, .size = {}, .fence = fence});
  ++found->fences;
  found->unfenced = false;
  return fence;
}

std::expected<FenceState, Failure> FakeDeviceExecution::Query(FenceId fence) {
  const std::scoped_lock lock(mutex_);
  Fence* found = fences_.Find(fence);
  if (found == nullptr) {
    return Invalid("stale or unknown fence");
  }
  if (fault_ && fault_->first == fence) {
    const ProviderError error = fault_->second;
    fault_.reset();
    return std::unexpected(Failure{.error = error, .detail = "scripted device fault"});
  }
  if (found->complete) {
    found->seen = true;
    return FenceState::kComplete;
  }
  return FenceState::kPending;
}

std::expected<void, Failure> FakeDeviceExecution::Release(FenceId fence) {
  const std::scoped_lock lock(mutex_);
  const Fence* found = fences_.Find(fence);
  if (found == nullptr) {
    return Invalid("stale or unknown fence");
  }
  if (!found->seen) {
    return Invalid("the fence has not been seen complete");
  }
  Stream* stream = streams_.Find(found->stream);
  base::Check(stream != nullptr && stream->fences > 0, "a fence outlived its stream");
  --stream->fences;
  (void)fences_.Erase(fence);
  return {};
}

bool FakeDeviceExecution::Step(StreamId stream) {
  const std::scoped_lock lock(mutex_);
  return StepLocked(stream);
}

bool FakeDeviceExecution::StepLocked(StreamId stream) {
  Stream* found = streams_.Find(stream);
  if (found == nullptr || found->steps.empty()) {
    return false;
  }
  const Queued& step = found->steps.front();
  switch (step.kind) {
    case Queued::Kind::kCopy:
      std::memmove(At(step.destination), At(step.source), step.size.value());
      break;
    case Queued::Kind::kWait: {
      const Fence* awaited = fences_.Find(step.fence);
      if (awaited != nullptr && !awaited->complete) {
        return false;  // blocked on the other stream
      }
      break;
    }
    case Queued::Kind::kFence:
      fences_.Find(step.fence)->complete = true;
      break;
  }
  found->steps.pop_front();
  return true;
}

void FakeDeviceExecution::Drain() {
  const std::scoped_lock lock(mutex_);
  bool moved = true;
  while (moved) {
    moved = false;
    streams_.ForEach([&](StreamId id, const Stream& /*stream*/) {
      while (StepLocked(id)) {
        moved = true;
      }
    });
  }
}

}  // namespace jitllm::providers::fake
