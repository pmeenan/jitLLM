// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The storage provider (D-034, D-048; docs/architecture.md#providers):
// direct reads into, and writes from, protected backing ranges, with a
// bounded number in flight. Linux implements it with io_uring
// (providers/uring_storage.h); the fake scripts completions
// (providers/fake/fake_storage.h). The storage lane owns one provider and
// is its only caller.
//
// Every submission resolves as not started, accepted, or unknown. An
// accepted or unknown request produces exactly one completion, whether it
// succeeded, failed, was cancelled or transferred fewer bytes than asked;
// a completion also proves the provider will touch that memory no more.
// Cancellation is a request with its own result: the original still
// completes, and only its completion retires the memory (D-048).
//
// Requests carry file descriptors the caller opened for direct I/O beneath
// a storage role (config/storage_roles.h), and memory that the caller
// keeps leased until the completion arrives.

#ifndef JITLLM_PROVIDERS_STORAGE_H_
#define JITLLM_PROVIDERS_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <span>

namespace jitllm::providers {

enum class IoKind : std::uint8_t { kRead, kWrite };

struct IoRequest {
  std::uint64_t token = 0;  // the caller's identity for it, echoed back
  IoKind kind = IoKind::kRead;
  int fd = -1;
  std::uint64_t offset = 0;  // in the file
  std::byte* memory = nullptr;
  std::uint32_t length = 0;
};

enum class Submission : std::uint8_t {
  kNotStarted,  // proven never to have started: the queue was full, or it was refused
  kAccepted,
  kUnknown,  // it may have started: wait for its completion, and fault if none comes
};

struct IoCompletion {
  std::uint64_t token = 0;
  // Bytes transferred (possibly fewer than asked), or -errno.
  std::int64_t result = 0;
};

class Storage {
 public:
  Storage() = default;
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;
  virtual ~Storage() = default;

  // The most requests in flight at once.
  virtual std::size_t depth() const = 0;
  // Requests, and any cancellations of them, whose completions are still to
  // come; the owner drains it to zero before destroying the provider.
  virtual std::size_t in_flight() const = 0;

  virtual Submission Submit(const IoRequest& request) = 0;
  // Asks for the request with `token` to be cancelled. Not started if
  // there is no such request in flight or no room to ask; the request
  // still completes either way.
  virtual Submission Cancel(std::uint64_t token) = 0;

  // Completions, up to out.size(); with `wait`, blocks until at least one
  // arrives if any request is in flight.
  virtual std::size_t Harvest(std::span<IoCompletion> out, bool wait) = 0;
};

}  // namespace jitllm::providers

#endif  // JITLLM_PROVIDERS_STORAGE_H_
