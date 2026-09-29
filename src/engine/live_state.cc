// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/live_state.h"

#include <unistd.h>

#include <format>
#include <system_error>
#include <utility>

#include "engine/runner_resources.h"
#include "engine/support.h"
#include "platform/direct_io.h"
#include "providers/device_runtime.h"
#include "scheduler/commands.h"

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using support::Error;
using support::Pointer;
using support::Round;

constexpr std::uint64_t kExtent = kPagedExtent;

}  // namespace

// ------------------------------------------------------------------ regions

LiveState::Status LiveState::Add(PagedNode& node, std::string name, std::uint64_t bytes,
                                 int owner) {
  Region& region = regions_.emplace_back();
  region.bytes = bytes;
  return node.MapResident(region.mapped, std::move(name), bytes, providers::BackingKind::kDevice,
                          catalog::MemoryClass::kLiveState, catalog::Recovery::kPreserve, owner);
}

std::uint64_t LiveState::base(std::size_t region) const {
  return region < regions_.size() ? regions_[region].mapped.base : 0;
}

std::uint64_t LiveState::bytes(std::size_t region) const {
  return region < regions_.size() ? regions_[region].bytes : 0;
}

std::uint64_t LiveState::total_bytes() const {
  std::uint64_t total = 0;
  for (const Region& r : regions_) {
    total += r.bytes;
  }
  return total;
}

std::vector<ExtentId> LiveState::extents() const {
  std::vector<ExtentId> all;
  for (const Region& r : regions_) {
    all.insert(all.end(), r.mapped.extents.begin(), r.mapped.extents.end());
  }
  return all;
}

LiveState::Status LiveState::RegisterSpill(PagedNode& node,
                                           const std::filesystem::path& directory) {
  std::filesystem::create_directories(directory);
  const auto opened = platform::OpenUnnamedDirectFile(directory);
  if (!opened) {
    return Error(std::format("the spill file in {}: {}", directory.string(),
                             std::generic_category().message(opened.error())));
  }
  spill_fd_ = *opened;
  std::uint64_t slot = 0;
  for (Region& region : regions_) {
    Mapped& mapped = region.mapped;
    region.sources.clear();
    for (std::size_t i = 0; i < mapped.extents.size(); ++i, ++slot) {
      const sc::PageSource source{
          .read = {.fd = spill_fd_, .offset = slot * kExtent, .memory = nullptr, .length = kExtent},
          .landed = true,
          .destination = mapped.base + (i * kExtent),
          .backing = sc::BackingPlace{.reservation = mapped.reservation,
                                      .offset = Bytes(i * kExtent),
                                      .size = Bytes(kExtent),
                                      .allocation_class = node.device_class()},
          .write_back = true};
      auto set = node.scheduler().SetSource(mapped.extents[i], source);
      if (!set) {
        return Error(std::format("the state's write-back place: {}", sc::ToString(set.error())));
      }
      region.sources.push_back(source);
    }
    mapped.backings.clear();  // the VMM lane releases them on eviction (D-033)
  }
  return {};
}

void LiveState::CheckPlaces(const sc::Scheduler& scheduler, PlaceCheck& check) const {
  for (const Region& region : regions_) {
    if (region.sources.size() != region.mapped.extents.size()) {
      check.Missing(std::format("{}'s write-back places", region.mapped.name));
    }
    for (std::size_t i = 0; i < region.mapped.extents.size() && i < region.sources.size(); ++i) {
      check.Check(scheduler, region.mapped.extents[i], region.sources[i]);
    }
  }
}

LiveState::Status LiveState::Clear(PagedNode& node, const catalog::Closure& fence,
                                   std::uint32_t stream, std::string_view what) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> zeroed;
  zeroed.reserve(regions_.size());
  for (const Region& r : regions_) {
    zeroed.emplace_back(r.mapped.base, r.bytes);
  }
  restore_count_ = 0;
  commit_keep_ = 0;
  save_count_ = 0;
  saved_.clear();
  verify_rows_ = 0;
  // Unusable until zeroed; a quarantine lifts once the clear has run.
  quarantined_ = true;
  auto cleared = node.Job(
      fence,
      [zeroed](providers::NativeStream native) {
        for (const auto& [base, bytes] : zeroed) {
          if (!providers::FillAsync(native, Pointer(base), 0, bytes).ok()) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!cleared) {
    return cleared;
  }
  quarantined_ = false;
  return {};
}

void* LiveState::HostCopy() { return HavePinned(host_copy_, total_bytes()) ? host_copy_ : nullptr; }

LiveState::Status LiveState::Read(PagedNode& node, const catalog::Closure& fence,
                                  std::uint32_t stream, std::string_view what,
                                  std::span<std::vector<std::byte>* const> out) {
  auto* host = static_cast<std::byte*>(HostCopy());
  if (host == nullptr) {
    return Error("pinned host memory for the state's copy");
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> reads;  // base, bytes
  reads.reserve(regions_.size());
  for (const Region& r : regions_) {
    reads.emplace_back(r.mapped.base, r.bytes);
  }
  auto posted = node.Job(
      fence,
      [&reads, host](providers::NativeStream native) {
        std::uint64_t at = 0;
        for (const auto& [base, bytes] : reads) {
          if (bytes != 0 && !providers::CopyAsync(native, host + at, Pointer(base), bytes,
                                                  providers::CopyKind::kDeviceToHost)
                                 .ok()) {
            return sc::JobResult::kUnknown;
          }
          at += bytes;
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!posted) {
    return posted;
  }
  std::uint64_t at = 0;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::uint64_t n = bytes(i);
    out[i]->assign(host + at, host + at + n);
    at += n;
  }
  return {};
}

LiveState::Status LiveState::Usable() const {
  if (quarantined_) {
    return Error(
        std::format("the {} state is quarantined: a job failed after it may have written it "
                    "(Clear first)",
                    model_));
  }
  return {};
}

// ------------------------------------------------------------------ speculation

LiveState::Status LiveState::AllocateSnapshot(RunnerResources& resources, std::uint32_t capacity) {
  auto save = resources.Pinned(capacity * sizeof(kg::RangeCopy));
  auto restore = resources.Pinned(capacity * sizeof(kg::RangeCopy));
  if (!save || !restore) {
    return Error(std::format("pinned staging for {}'s verify snapshot", model_));
  }
  capacity_ = capacity;
  save_ = static_cast<kg::RangeCopy*>(*save);
  restore_ = static_cast<kg::RangeCopy*>(*restore);
  return {};
}

void LiveState::BeginSaves() {
  saved_.clear();
  save_count_ = 0;
  save_at_ = 0;
}

LiveState::Status LiveState::Save(std::uint64_t address, std::uint64_t bytes, std::int64_t row) {
  if (save_count_ == capacity_ || save_at_ + bytes > snapshot_bytes_) {
    return Error("a verify writes more than its snapshot holds");
  }
  const std::uint64_t saved = snapshot_base_ + save_at_;
  saved_.push_back({.address = address, .saved = saved, .bytes = bytes, .row = row});
  save_[save_count_++] = {.from = address, .to = saved, .bytes = bytes};
  save_at_ += Round(bytes, 256);
  return {};
}

std::expected<void, kg::KernelFailure> LiveState::QueueSaves(kg::LaunchContext& launch) const {
  return kg::CopyRanges(launch, save_, save_count_);
}

LiveState::Status LiveState::AwaitingAccept() const {
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  return {};
}

LiveState::Status LiveState::Accept(std::uint32_t keep) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ == 0 || keep == 0 || keep > verify_rows_) {
    return Error(std::format("accepting {} rows of a verify of {}", keep, verify_rows_));
  }
  std::uint32_t count = 0;
  for (const Saved& s : saved_) {
    if (s.row < 0 || std::cmp_greater_equal(s.row, keep)) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
  }
  restore_count_ = count;
  commit_keep_ = commit_ ? keep : 0;
  verify_rows_ = 0;
  return {};
}

std::expected<void, kg::KernelFailure> LiveState::QueueOwed(kg::LaunchContext& launch) {
  if (restore_count_ != 0) {
    // Still owed until the copy is queued.
    if (auto r = kg::CopyRanges(launch, restore_, restore_count_); !r) {
      return r;
    }
    restore_count_ = 0;
  }
  if (commit_keep_ != 0) {
    if (auto r = commit_(launch, commit_keep_); !r) {
      return r;
    }
    commit_keep_ = 0;
  }
  return {};
}

LiveState::Status LiveState::Rollback(PagedNode& node, const catalog::Closure& closure,
                                      std::uint32_t stream, kg::LaunchContext& launch,
                                      std::string_view what) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (!owed()) {
    return {};
  }
  std::string failed;
  auto posted = node.Job(
      closure,
      [&](providers::NativeStream) {
        if (auto r = QueueOwed(launch); !r) {
          failed = r.error().detail;
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      what, stream);
  if (!posted || !failed.empty()) {
    // What was not queued stays owed; what was has completed (the job
    // retired); a launch of unknown effect quarantines.
    Settle(false, false, !failed.empty() || launch.faulted());
    return Error(failed.empty() ? posted.error() : failed);
  }
  return {};
}

bool LiveState::Settle(bool saved, bool wrote, bool unknown) {
  verify_rows_ = 0;
  if (unknown) {
    quarantined_ = true;
    return false;
  }
  if (saved) {
    // The whole verify undone: every range it saved, before the next job's
    // own work (its queued work has completed: the job has retired), and
    // nothing committed.
    std::uint32_t count = 0;
    for (const Saved& s : saved_) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
    restore_count_ = count;
    commit_keep_ = 0;
    return true;
  }
  if (wrote) {
    quarantined_ = true;
  }
  return false;
}

void LiveState::Release(providers::VmmProvider& memory, std::vector<std::string>& problems) {
  for (Region& r : regions_) {
    if (!ReleaseMapped(memory, r.mapped)) {
      problems.push_back(std::format("{} could not be released", r.mapped.name));
    }
  }
  if (host_copy_ != nullptr) {
    providers::FreePinned(host_copy_);
    host_copy_ = nullptr;
  }
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
    spill_fd_ = -1;
  }
}

}  // namespace jitllm::engine
