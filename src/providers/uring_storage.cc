// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/uring_storage.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <system_error>
#include <utility>

#include "base/check.h"
#include "platform/io_uring.h"
#include "providers/storage.h"

namespace jitllm::providers {

std::expected<std::unique_ptr<UringStorage>, std::error_code> UringStorage::Create(
    std::size_t depth) {
  if (depth == 0 || depth > 4096) {
    return std::unexpected(std::make_error_code(std::errc::invalid_argument));
  }
  // Room for every request and a cancellation of each.
  auto ring = platform::IoUring::Create(static_cast<unsigned>(depth * 2));
  if (!ring) {
    return std::unexpected(ring.error());
  }
  return std::make_unique<UringStorage>(*std::move(ring), depth);
}

UringStorage::UringStorage(platform::IoUring ring, std::size_t depth)
    : ring_(std::move(ring)), depth_(depth), scratch_(ring_.completion_entries()) {}

UringStorage::~UringStorage() {
  base::Check(in_flight_.empty() && cancels_in_flight_ == 0 && ring_.unconsumed() == 0 &&
                  ring_.prepared() == 0,
              "a storage ring destroyed with requests in flight");
}

Submission UringStorage::Hand() {
  // Once published, a request reaches the kernel on this or a later
  // Submit(): if the kernel has not consumed it yet, its outcome is not
  // known, but it is in flight either way.
  Enter(0);
  return ring_.unconsumed() == 0 ? Submission::kAccepted : Submission::kUnknown;
}

void UringStorage::Enter(unsigned wait_for) {
  if (auto entered = ring_.Submit(wait_for); !entered) {
    last_error_ = entered.error();
  }
}

Submission UringStorage::Submit(const IoRequest& request) {
  if (in_flight_.size() >= depth_ || (request.token & kCancelBit) != 0 ||
      in_flight_.contains(request.token) || cancelling_.contains(request.token) ||
      request.memory == nullptr || request.length == 0) {
    return Submission::kNotStarted;
  }
  const bool prepared = request.kind == IoKind::kRead
                            ? ring_.PrepareRead(request.fd, request.memory, request.length,
                                                request.offset, request.token)
                            : ring_.PrepareWrite(request.fd, request.memory, request.length,
                                                 request.offset, request.token);
  if (!prepared) {
    return Submission::kNotStarted;
  }
  in_flight_.insert(request.token);
  return Hand();
}

Submission UringStorage::Cancel(std::uint64_t token) {
  if (!in_flight_.contains(token) || cancels_in_flight_ >= depth_ ||
      !ring_.PrepareCancel(token, token | kCancelBit)) {
    return Submission::kNotStarted;
  }
  ++cancels_in_flight_;
  ++cancelling_[token];
  return Hand();
}

std::size_t UringStorage::Harvest(std::span<IoCompletion> out, bool wait) {
  if (wait && (!in_flight_.empty() || cancels_in_flight_ > 0)) {
    Enter(1);
  } else if (ring_.unconsumed() > 0) {
    Enter(0);
  }
  std::size_t produced = 0;
  while (produced < out.size()) {
    const std::size_t room = std::min(out.size() - produced, scratch_.size());
    const std::size_t reaped = ring_.Reap(std::span(scratch_).first(room));
    if (reaped == 0) {
      break;
    }
    for (std::size_t i = 0; i < reaped; ++i) {
      const platform::Completion& completion = scratch_[i];
      if ((completion.user_data & kCancelBit) != 0) {
        --cancels_in_flight_;  // the cancellation's own result: never reported
        const std::uint64_t target = completion.user_data & ~kCancelBit;
        if (--cancelling_[target] == 0) {
          cancelling_.erase(target);
        }
        continue;
      }
      in_flight_.erase(completion.user_data);
      out[produced++] = IoCompletion{.token = completion.user_data, .result = completion.result};
    }
  }
  return produced;
}

}  // namespace jitllm::providers
