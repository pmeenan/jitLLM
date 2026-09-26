// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "providers/fake/fake_storage.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

#include "providers/storage.h"

namespace jitllm::providers::fake {

FakeStorage::FakeStorage(std::size_t depth, std::uint32_t alignment)
    : depth_(depth), alignment_(alignment) {}

int FakeStorage::AddFile(std::vector<std::byte> contents) {
  const int fd = next_fd_++;
  files_.emplace(fd, std::move(contents));
  return fd;
}

std::span<const std::byte> FakeStorage::Contents(int fd) const {
  const auto found = files_.find(fd);
  return found != files_.end() ? std::span<const std::byte>(found->second)
                               : std::span<const std::byte>();
}

bool FakeStorage::Release(std::uint64_t token) {
  const auto found = requests_.find(token);
  if (found == requests_.end() || !found->second.held) {
    return false;
  }
  found->second.held = false;
  return true;
}

Submission FakeStorage::Submit(const IoRequest& request) {
  Script script;
  if (!scripts_.empty()) {
    script = scripts_.front();
    scripts_.pop_front();
  }
  if (script.submission == Submission::kNotStarted || requests_.size() >= depth_ ||
      requests_.contains(request.token)) {
    return Submission::kNotStarted;
  }
  submitted_.push_back(request);
  requests_.emplace(request.token, Pending{.request = request,
                                           .result = script.result,
                                           .held = script.hold,
                                           .cancelled = false,
                                           .sequence = next_sequence_++});
  return script.submission;
}

Submission FakeStorage::Cancel(std::uint64_t token) {
  const auto found = requests_.find(token);
  if (found == requests_.end()) {
    return Submission::kNotStarted;
  }
  if (found->second.held) {
    found->second.cancelled = true;
    found->second.held = false;
  }
  return Submission::kAccepted;
}

std::int64_t FakeStorage::Perform(const Pending& pending) {
  const IoRequest& request = pending.request;
  if (pending.cancelled) {
    return -ECANCELED;
  }
  if (pending.result && *pending.result < 0) {
    return *pending.result;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(request.memory);
  if (address % alignment_ != 0 || request.offset % alignment_ != 0 ||
      request.length % alignment_ != 0) {
    return -EINVAL;  // as the kernel refuses unaligned direct I/O
  }
  const auto file = files_.find(request.fd);
  if (file == files_.end()) {
    return -EBADF;
  }
  std::vector<std::byte>& contents = file->second;
  std::uint64_t count = request.length;
  if (pending.result) {
    count = std::min<std::uint64_t>(count, static_cast<std::uint64_t>(*pending.result));
  }
  if (request.kind == IoKind::kRead) {
    const std::uint64_t available =
        request.offset < contents.size() ? contents.size() - request.offset : 0;
    count = std::min(count, available);
    if (count == 0) {
      return 0;  // EOF may be arbitrarily far past the vector's end.
    }
    std::memcpy(request.memory, contents.data() + request.offset,
                count);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  } else {
    if (contents.size() < request.offset + count) {
      contents.resize(request.offset + count);
    }
    std::memcpy(contents.data() + request.offset, request.memory,
                count);  // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  }
  return static_cast<std::int64_t>(count);
}

std::size_t FakeStorage::Harvest(std::span<IoCompletion> out, bool /*wait*/) {
  // Everything due completes, in submission order.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> due;  // sequence, token
  for (const auto& [token, pending] : requests_) {
    if (!pending.held) {
      due.emplace_back(pending.sequence, token);
    }
  }
  std::ranges::sort(due);
  std::size_t produced = 0;
  for (const auto& [sequence, token] : due) {
    if (produced == out.size()) {
      break;
    }
    const auto found = requests_.find(token);
    out[produced++] = IoCompletion{.token = token, .result = Perform(found->second)};
    requests_.erase(found);
  }
  return produced;
}

}  // namespace jitllm::providers::fake
