// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Linux storage provider: Storage over one io_uring ring
// (platform/io_uring.h, D-034). Requests are handed to the kernel as they
// are submitted; a failure to hand them over leaves their outcome unknown,
// since the kernel may have consumed some. Cancellation completions are
// consumed here and never reported: only the original's completion retires
// its memory.

#ifndef JITLLM_PROVIDERS_URING_STORAGE_H_
#define JITLLM_PROVIDERS_URING_STORAGE_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <system_error>
#include <vector>

#include "platform/io_uring.h"
#include "providers/storage.h"

namespace jitllm::providers {

class UringStorage final : public Storage {
 public:
  // A ring for `depth` requests in flight, with room for a cancellation
  // of each. std::errc::function_not_supported where there is no io_uring.
  static std::expected<std::unique_ptr<UringStorage>, std::error_code> Create(std::size_t depth);

  explicit UringStorage(platform::IoUring ring, std::size_t depth);
  // Its owner must have harvested every request first (D-048): closing the
  // ring does not stop reads landing in memory, so anything left is fatal.
  ~UringStorage() override;

  UringStorage(const UringStorage&) = delete;
  UringStorage& operator=(const UringStorage&) = delete;
  UringStorage(UringStorage&&) = delete;
  UringStorage& operator=(UringStorage&&) = delete;

  std::size_t depth() const override { return depth_; }
  // Requests and cancellations whose completions are still to come: the
  // owner drains until this is zero before destroying the provider.
  std::size_t in_flight() const override { return in_flight_.size() + cancels_in_flight_; }

  Submission Submit(const IoRequest& request) override;
  Submission Cancel(std::uint64_t token) override;
  std::size_t Harvest(std::span<IoCompletion> out, bool wait) override;

 private:
  // Tokens with this bit are cancellations, never reported.
  static constexpr std::uint64_t kCancelBit = std::uint64_t{1} << 63;

  Submission Hand();
  // io_uring_enter, remembering a failure for diagnostics: what it consumed
  // is still judged by the ring itself.
  void Enter(unsigned wait_for);

  platform::IoUring ring_;
  std::size_t depth_;
  std::set<std::uint64_t> in_flight_;
  std::size_t cancels_in_flight_ = 0;
  // Cancellations not yet completed, by the token they target: a token is
  // not reused until its cancellation is done, or it could hit the new
  // request.
  std::map<std::uint64_t, std::uint32_t> cancelling_;
  std::vector<platform::Completion> scratch_;
  std::error_code last_error_;
};

}  // namespace jitllm::providers

#endif  // JITLLM_PROVIDERS_URING_STORAGE_H_
