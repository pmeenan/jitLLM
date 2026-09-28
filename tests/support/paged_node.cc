// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_node.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <print>
#include <utility>

#include "base/bounded_queue.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/direct_reader.h"

namespace jitllm::test_support {

namespace {

namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;

constexpr auto kPatience = std::chrono::minutes(10);

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Joined(const std::vector<std::string>& problems) {
  if (problems.empty()) {
    return {};
  }
  std::string all;
  for (const std::string& problem : problems) {
    all += (all.empty() ? "" : "; ") + problem;
  }
  return Error(all);
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

// A fence after everything queued on the stream, seen complete and
// released; false if it could not be, or its outcome is unknown.
bool Fence(providers::DeviceExecution& execution, providers::StreamId stream) {
  const auto fence = execution.Record(stream);
  if (!fence) {
    return false;
  }
  const auto give_up = std::chrono::steady_clock::now() + kPatience;
  while (std::chrono::steady_clock::now() < give_up) {
    const auto state = execution.Query(*fence);
    if (!state) {
      return false;  // unknown: never released or retried
    }
    if (*state == providers::FenceState::kComplete) {
      return execution.Release(*fence).has_value();
    }
    std::this_thread::yield();
  }
  return false;
}

}  // namespace

PagedNode::~PagedNode() {
  if (torn_down_ || scheduler_ == nullptr) {
    return;
  }
  // TearDown never ran: stop, so no thread outlives the node. Nothing is
  // released: a model's memory may still be in use.
  scheduler_->RequestShutdown();
  if (threads_.empty()) {
    const auto give_up = std::chrono::steady_clock::now() + kPatience;
    while (!scheduler_->Stopped() && std::chrono::steady_clock::now() < give_up) {
      Round();
    }
  } else {
    threads_.front().join();
  }
  storage_lane_->Close();
  device_lane_->Close();
  backing_lane_->Close();
  threads_.clear();
}

Status PagedNode::Open() {
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    return Error("CUDA device 0 has no context");
  }
  auto memory = providers::cuda::OpenDeviceMemory(0);
  if (!memory) {
    return Error(std::format("OpenDeviceMemory: {}", memory.error().detail));
  }
  memory_ = std::move(*memory);
  auto execution = providers::cuda::OpenDeviceExecution(0);
  if (!execution) {
    return Error("OpenDeviceExecution failed");
  }
  execution_ = std::move(*execution);
  for (std::size_t i = 0; i <= settings_.compute_streams; ++i) {
    auto created = execution_->CreateStream();
    if (!created) {
      return Error("CreateStream failed");
    }
    streams_.push_back(*created);
  }
  bool host_found = false;
  for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
    if (memory_->Classes()[i].kind == providers::BackingKind::kDevice) {
      device_class_ = i;
    } else {
      host_class_ = i;
      host_found = true;
    }
  }
  if (!host_found || memory_->Granularity() != Bytes(kPagedExtent)) {
    return Error("this device has no host-NUMA VMM at 2 MiB (D-034, D-081)");
  }
  auto storage = providers::UringStorage::Create(kPagedDepth);
  if (!storage) {
    return Error(std::format("io_uring: {}", storage.error().message()));
  }
  storage_ = std::move(*storage);
  domain_ = catalog_.AddDomain("gb10");
  // The zone first: a persistent pool, mapped before anything pages.
  return MapResident(zone_, "the landing zone", settings_.slots * kPagedExtent,
                     providers::BackingKind::kHost, MemoryClass::kStaging,
                     catalog::Recovery::kPinned, kShared);
}

Status PagedNode::MapWorkspace(std::uint64_t activations, std::uint64_t pool) {
  if (auto r =
          MapResident(activations_, "the activations", activations, providers::BackingKind::kDevice,
                      MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared);
      !r) {
    return r;
  }
  return MapResident(pool_, "the GGML pool", pool, providers::BackingKind::kDevice,
                     MemoryClass::kScratch, catalog::Recovery::kDiscardable, kShared);
}

Status PagedNode::MapResident(Mapped& mapped, std::string name, std::uint64_t bytes,
                              providers::BackingKind kind, MemoryClass memory_class,
                              catalog::Recovery recovery, int owner) {
  mapped.name = std::move(name);
  mapped.bytes =
      (std::max<std::uint64_t>(bytes, 1) + kPagedExtent - 1) / kPagedExtent * kPagedExtent;
  auto reservation = memory_->Reserve(Bytes(mapped.bytes));
  if (!reservation) {
    return Error(std::format("reserving {}: {}", mapped.name, reservation.error().detail));
  }
  mapped.reservation = *reservation;
  mapped.base = memory_->RangeOf(*reservation).value().base;
  const std::size_t allocation_class =
      kind == providers::BackingKind::kDevice ? device_class_ : host_class_;
  for (std::uint64_t at = 0; at < mapped.bytes; at += kPagedExtent) {
    auto backing = memory_->Create(allocation_class, Bytes(kPagedExtent));
    if (!backing) {
      return Error(std::format("backing {}: {}", mapped.name, backing.error().detail));
    }
    mapped.backings.push_back(*backing);
    if (auto map = memory_->Map(*reservation, Bytes(at), *backing); !map) {
      return Error(std::format("mapping {}: {}", mapped.name, map.error().detail));
    }
    auto extent = catalog_.AddExtent({.domain = domain_,
                                      .memory_class = memory_class,
                                      .recovery = recovery,
                                      .size = Bytes(kPagedExtent),
                                      .content = {}},
                                     true);
    if (!extent) {
      return Error(std::format("cataloging {}", mapped.name));
    }
    mapped.extents.push_back(*extent);
    AddSpan({.base = mapped.base + at,
             .size = kPagedExtent,
             .extent = *extent,
             .memory_class = memory_class,
             .device = kind == providers::BackingKind::kDevice,
             .owner = owner});
  }
  if (auto access = memory_->SetAccess(*reservation, Bytes(0), Bytes(mapped.bytes),
                                       providers::Access::kReadWrite);
      !access) {
    return Error(std::format("access to {}: {}", mapped.name, access.error().detail));
  }
  return {};
}

std::expected<void*, std::string> PagedNode::Pinned(std::uint64_t bytes, int owner,
                                                    std::vector<ExtentId>& staging) {
  void* pointer = nullptr;
  if (cudaMallocHost(&pointer, std::max<std::uint64_t>(bytes, 256)) != cudaSuccess) {
    return Error("pinned memory");
  }
  pinned_.push_back(pointer);
  auto extent = catalog_.AddExtent({.domain = domain_,
                                    .memory_class = MemoryClass::kStaging,
                                    .recovery = catalog::Recovery::kPinned,
                                    .size = Bytes(bytes),
                                    .content = {}},
                                   true);
  if (!extent) {
    return Error("cataloging the staging");
  }
  staging.push_back(*extent);
  AddSpan({.base = Address(pointer),
           .size = bytes,
           .extent = *extent,
           .memory_class = MemoryClass::kStaging,
           .device = false,
           .owner = owner});
  return pointer;
}

void PagedNode::SortSpans() { std::ranges::sort(spans_, {}, &Span::base); }

std::optional<MemoryClass> PagedNode::Covered(std::uint64_t address, std::uint64_t bytes,
                                              int owner) const {
  if (bytes == 0) {
    return std::nullopt;
  }
  std::optional<MemoryClass> found;
  std::uint64_t at = address;
  const std::uint64_t end = address + bytes;
  auto span = std::ranges::upper_bound(spans_, at, {}, &Span::base);
  if (span == spans_.begin()) {
    return std::nullopt;
  }
  --span;
  while (at < end) {
    if (span == spans_.end() || at < span->base || at >= span->base + span->size || !span->device ||
        (span->owner != owner && span->owner != kShared) ||
        (found && *found != span->memory_class)) {
      return std::nullopt;
    }
    const auto view = catalog_.Describe(span->extent);
    if (!view || view->state != catalog::ExtentState::kResident) {
      return std::nullopt;
    }
    found = span->memory_class;
    at = span->base + span->size;
    ++span;
  }
  return found;
}

Status PagedNode::Start(Bytes budget) {
  budget_ = budget;
  board_ = std::make_unique<sc::CompletionBoard>(1024, wake_);
  storage_lane_ = std::make_unique<sc::StorageService>(
      *storage_,
      providers::ReaderSettings{
          .alignment = 4096, .request_bytes = 2U << 20U, .retries = 3, .reads = 1024, .waiters = 8},
      *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32});
  device_lane_ = std::make_unique<sc::DeviceService>(
      *execution_, streams_, *board_,
      sc::DeviceSettings{.queue = {.capacity = 256, .reserved = 16, .batch = 32},
                         .handoff = 256,
                         .poll_sleep = std::chrono::microseconds(0)},
      nullptr);
  // Managed backing's VMM work on a lane of its own, so the zone's copies
  // never wait behind it (docs/experiments/pagein-perf/); that lane alone
  // calls the device-memory provider (device_memory.h).
  backing_lane_ = std::make_unique<sc::BackingService>(
      memory_.get(), *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32});
  sc::LandingZone landing{.slots = {},
                          .slot_bytes = Bytes(kPagedExtent),
                          .stream = static_cast<std::uint32_t>(settings_.compute_streams)};
  for (std::size_t i = 0; i < settings_.slots; ++i) {
    landing.slots.push_back(zone_.base + (i * kPagedExtent));
  }
  scheduler_ = std::make_unique<sc::Scheduler>(
      catalog_, *board_, wake_,
      sc::Lanes{.storage = storage_lane_.get(),
                .device = device_lane_.get(),
                .cpu = nullptr,
                .backing = backing_lane_.get()},
      sc::SchedulerSettings{.tasks = 16, .budget = budget, .landing = landing});
  return {};
}

void PagedNode::Run() {
  SortSpans();
  if (settings_.inline_lanes) {
    return;
  }
  threads_.emplace_back([this] { stopped_ = scheduler_->Run(); });
  threads_.emplace_back([this] { storage_lane_->Run(); });
  threads_.emplace_back([this] { device_lane_->RunSubmission(); });
  threads_.emplace_back([this] { device_lane_->RunCompletion(); });
  threads_.emplace_back([this] { backing_lane_->Run(); });
}

void PagedNode::Round() {
  (void)storage_lane_->Turn(false);
  (void)backing_lane_->Turn();
  (void)device_lane_->SubmissionTurn();
  (void)device_lane_->CompletionTurn();
  (void)scheduler_->Turn();
}

Status PagedNode::Await(Done& done, std::string_view what, std::uint64_t request) {
  auto give_up = std::chrono::steady_clock::now() + kPatience;
  bool cancelled = false;
  // `gone`, not `retired`: the program is destroyed after Retired(), and
  // its destructor is its last touch of `done`.
  while (!done.gone.load()) {
    if (std::chrono::steady_clock::now() > give_up) {
      if (cancelled) {
        // Returning would leave the program, or a job it queued, pointing
        // into frames that are gone.
        std::println(stderr, "{} did not finish, nor drain once cancelled: aborting", what);
        std::abort();
      }
      cancelled = true;
      give_up = std::chrono::steady_clock::now() + kPatience;
      Cancel(request);
    }
    if (threads_.empty()) {
      Round();  // --lanes inline, or before the lane threads start
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
  if (cancelled) {
    return Error(std::format("{} did not finish: cancelled, and drained", what));
  }
  if (done.outcome.load() != static_cast<int>(sc::TaskOutcome::kSucceeded)) {
    const int error = done.error.load();
    return Error(std::format(
        "{} failed{}", what,
        error >= 0 ? ": " + sc::ToString(static_cast<sc::WorkError>(error)) : std::string()));
  }
  // The scheduler's records are its thread's: read here only when this
  // thread drives it. With threads, a fault shows in the stop's result.
  if (threads_.empty()) {
    if (const auto fault = scheduler_->fault()) {
      return Error(std::format("{}: the node faulted: {}", what, sc::ToString(*fault)));
    }
  }
  return {};
}

std::uint64_t PagedNode::Submit(std::unique_ptr<sc::TaskProgram> program) {
  sc::Control start =
      sc::StartRequest{.request = ++request_, .priority = 1, .program = std::move(program)};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(start)) == base::PushResult::kFull) {
    std::this_thread::yield();
  }
  return request_;
}

void PagedNode::Cancel(std::uint64_t request) {
  if (threads_.empty()) {
    (void)scheduler_->Cancel(request);
    return;
  }
  sc::Control cancel = sc::CancelRequest{.request = request};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(cancel)) == base::PushResult::kFull) {
    std::this_thread::yield();
  }
}

Status PagedNode::Post(std::unique_ptr<sc::TaskProgram> program, Done& done,
                       std::string_view what) {
  if (!threads_.empty()) {
    return Await(done, what, Submit(std::move(program)));
  }
  const std::uint64_t request = ++request_;
  if (!scheduler_->Start(request, std::move(program), 1)) {
    return Error(std::format("{} was not admitted", what));
  }
  return Await(done, what, request);
}

Status PagedNode::Load(std::vector<ExtentId> extents, std::string what,
                       std::vector<LoadStats>& log) {
  LoadStats stats{.what = std::move(what), .extents = 0, .seconds = 0};
  for (const ExtentId extent : extents) {
    if (catalog_.Describe(extent).value().state != catalog::ExtentState::kResident) {
      ++stats.extents;
    }
  }
  auto closure = catalog_.ClosureOfExtents(extents);
  if (!closure) {
    return Error("a load's closure");
  }
  Done done;
  const auto start = std::chrono::steady_clock::now();
  if (auto posted =
          Post(std::make_unique<RunProgram>(done, *closure, sc::DeviceJob{}), done, stats.what);
      !posted) {
    return posted;
  }
  stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  log.push_back(std::move(stats));
  return {};
}

Status PagedNode::Evict(std::vector<ExtentId> extents) {
  Done done;
  return Post(std::make_unique<EvictProgram>(done, std::move(extents)), done, "an eviction");
}

Status PagedNode::Job(const catalog::Closure& closure, sc::DeviceJob job, std::string_view what,
                      std::uint32_t stream) {
  Done done;
  return Post(std::make_unique<RunProgram>(done, closure, std::move(job), stream), done, what);
}

Status PagedNode::Call(std::function<Status()> call, std::string_view what) {
  Done done;
  Status status;
  if (auto r = Post(std::make_unique<CallProgram>(done,
                                                  [&call, &status]() -> Status {
                                                    status = call();
                                                    return status;
                                                  }),
                    done, what);
      !r) {
    return !status ? status : r;
  }
  return {};
}

Status PagedNode::Acquire(const catalog::Closure& closure, AcquireReport& report,
                          std::string_view what) {
  // The shared workspace is never a victim, whether or not the closure
  // names it: it is discardable, so it would be chosen first, and nothing
  // restores it.
  std::vector<ExtentId> workspace = activations_.extents;
  workspace.insert(workspace.end(), pool_.extents.begin(), pool_.extents.end());
  Done done;
  auto program = std::make_unique<AcquireProgram>(done, closure, domain_, budget_, report,
                                                  std::move(workspace));
  return Post(std::move(program), done, what);
}

Status PagedNode::TearDown(std::span<PagedModel* const> models) {
  if (torn_down_) {
    return {};
  }
  torn_down_ = true;
  std::vector<std::string> problems;
  if (scheduler_ != nullptr) {
    // A fence after anything noted on each model's stream (a measuring
    // launch context notes work even when nothing runs), so it can be
    // destroyed; then every managed backing released through its
    // eviction; then the scheduler stops and the lanes drain.
    const bool usable = !threads_.empty() || !scheduler_->fault();
    std::vector<ExtentId> managed;
    for (PagedModel* model : models) {
      if (usable && !Job(
                        model->fence_closure(),
                        [](providers::NativeStream) { return sc::JobResult::kQueued; },
                        "fencing a compute stream", model->stream())) {
        problems.emplace_back("a compute stream could not be fenced");
      }
      const auto extents = model->managed_extents();
      managed.insert(managed.end(), extents.begin(), extents.end());
    }
    if ((!threads_.empty() || !scheduler_->fault()) && !Evict(managed)) {
      problems.emplace_back("the weights could not be evicted at the end");
    }
    scheduler_->RequestShutdown();
    if (threads_.empty()) {
      const auto give_up = std::chrono::steady_clock::now() + kPatience;
      while (!scheduler_->Stopped() && std::chrono::steady_clock::now() < give_up) {
        Round();
      }
      stopped_ = scheduler_->Stopped();
    } else {
      threads_.front().join();
    }
    const bool driven = threads_.empty();
    storage_lane_->Close();
    device_lane_->Close();
    backing_lane_->Close();
    if (driven) {
      for (int i = 0; i < 1000 && (storage_->in_flight() > 0 || i < 10); ++i) {
        (void)storage_lane_->Turn(false);
        (void)backing_lane_->Turn();
        (void)device_lane_->SubmissionTurn();
        (void)device_lane_->CompletionTurn();
      }
    }
    threads_.clear();
    if (!stopped_ || !stopped_->has_value()) {
      problems.emplace_back("the scheduler stopped with a fault: backing is left as it is");
      return Joined(problems);
    }
  } else if (execution_ != nullptr) {
    // Setup ended before the scheduler existed, and a model may already
    // have queued work on its stream (the EXL3 launch context zeroes its
    // lock area): each compute stream is fenced before any memory goes.
    for (std::size_t i = 0; i < settings_.compute_streams && i < streams_.size(); ++i) {
      if (!Fence(*execution_, streams_[i])) {
        problems.emplace_back("a compute stream could not be fenced: backing is left as it is");
        return Joined(problems);
      }
    }
  }
  for (PagedModel* model : models) {
    if (auto released = model->Release(); !released) {
      problems.push_back(released.error());
    }
  }
  if (execution_ != nullptr) {
    for (const auto stream : streams_) {
      if (stream.valid() && !execution_->DestroyStream(stream)) {
        problems.emplace_back("a stream could not be destroyed");
      }
    }
  }
  if (memory_ != nullptr) {
    for (Mapped* mapped : {&zone_, &activations_, &pool_}) {
      if (!mapped->reservation.valid()) {
        continue;
      }
      bool released =
          mapped->backings.empty() ||
          memory_
              ->Unmap(mapped->reservation, Bytes(0), Bytes(mapped->backings.size() * kPagedExtent))
              .has_value();
      for (const auto backing : mapped->backings) {
        released = memory_->Release(backing).has_value() && released;
      }
      if (!released || !memory_->Free(mapped->reservation)) {
        problems.push_back(std::format("{} could not be released", mapped->name));
      }
    }
    if (memory_->backings() != 0) {
      problems.push_back(std::format("{} backings were left", memory_->backings()));
    }
  }
  for (void* pointer : pinned_) {
    (void)cudaFreeHost(pointer);
  }
  pinned_.clear();
  return Joined(problems);
}

}  // namespace jitllm::test_support
