// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "paged_node.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <print>
#include <span>
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

// Each device-type lane's completion handoff; a lane holds at most twice
// that many fences unreleased (queued for its completion lane, and
// watched there) and one more waiting to be handed over.
constexpr std::size_t kLaneHandoff = 256;
// The provider's events made ahead: every fence the device and copy lanes
// can hold at once, and a few for the node's own fences, so neither lane
// ever makes one as it goes (RE-029: cuEventCreate blocks while another
// thread launches into a full stream).
constexpr std::size_t kEventsAhead = (2 * ((2 * kLaneHandoff) + 1)) + 16;

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

providers::Submission CountingStorage::Submit(const providers::IoRequest& request) {
  const auto submitted = inner_.Submit(request);
  if (submitted != providers::Submission::kNotStarted) {
    requests.fetch_add(1, std::memory_order_relaxed);
    pieces.fetch_add(std::max<std::size_t>(request.segments.size(), 1), std::memory_order_relaxed);
  }
  return submitted;
}

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
  if (copy_lane_ != nullptr) {
    copy_lane_->Close();
  }
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
  auto execution = providers::cuda::OpenDeviceExecution(
      0, {.events_ahead = kEventsAhead, .events_kept = std::max<std::size_t>(kEventsAhead, 4096)});
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
  // Two timing events per compute stream (StepTimes), made now: made while
  // another thread launches into a full stream, one would block (RE-029).
  times_.assign(settings_.compute_streams, StepTimes{});
  for (std::size_t i = 0; i < settings_.compute_streams; ++i) {
    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess || cudaEventCreate(&end) != cudaSuccess) {
      return Error("the timing events");
    }
    events_.emplace_back(begin, end);
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
  if (settings_.slot_bytes < kPagedExtent || settings_.slot_bytes % 4096 != 0) {
    return Error("a landing slot is at least 2 MiB, in 4 KiB units");
  }
  auto storage = providers::UringStorage::Create(kPagedDepth);
  if (!storage) {
    return Error(std::format("io_uring: {}", storage.error().message()));
  }
  storage_ = std::move(*storage);
  counting_ = std::make_unique<CountingStorage>(*storage_);
  domain_ = catalog_.AddDomain("gb10");
  // The zone first: a persistent pool, mapped before anything pages.
  return MapResident(zone_, "the landing zone", settings_.slots * settings_.slot_bytes,
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
  // One request per 2 MiB chunk (the reader's default), or with coalesce
  // coalesced chunk reads (BP-P1) in spans up to kSpanBytes.
  storage_lane_ = std::make_unique<sc::StorageService>(
      *counting_,
      providers::ReaderSettings{
          .alignment = 4096,
          .request_bytes = static_cast<std::uint32_t>(settings_.slot_bytes),
          .retries = 3,
          .reads = 1024,
          .waiters = 8,
          .span_bytes = settings_.coalesce ? providers::kSpanBytes : providers::kNoCoalescing,
          .span_segments = providers::kMaxSegments},
      *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32});
  device_lane_ = std::make_unique<sc::DeviceService>(
      *execution_, streams_, *board_,
      sc::DeviceSettings{.queue = {.capacity = 256, .reserved = 16, .batch = 32},
                         .handoff = kLaneHandoff,
                         .poll_sleep = std::chrono::microseconds(0),
                         .poll_window = settings_.poll_window},
      nullptr);
  if (settings_.copy_lane) {
    // The zone's copies on their own submission and completion lanes, on
    // the copy stream alone: a model's job launching into a full stream
    // (RE-029) blocks only the device lane.
    copy_lane_ = std::make_unique<sc::DeviceService>(
        *execution_, std::span<const providers::StreamId>(&streams_.back(), 1), *board_,
        sc::DeviceSettings{.queue = {.capacity = 256, .reserved = 16, .batch = 32},
                           .handoff = kLaneHandoff,
                           .poll_sleep = std::chrono::microseconds(0)},
        nullptr);
  }
  // Managed backing's VMM work on a lane of its own, so the zone's copies
  // never wait behind it (docs/experiments/pagein-perf/); that lane alone
  // calls the device-memory provider (device_memory.h).
  backing_lane_ = std::make_unique<sc::BackingService>(
      memory_.get(), *board_, sc::QueueSettings{.capacity = 256, .reserved = 16, .batch = 32});
  sc::LandingZone landing{
      .slots = {},
      .slot_bytes = Bytes(settings_.slot_bytes),
      .stream = copy_lane_ != nullptr ? 0 : static_cast<std::uint32_t>(settings_.compute_streams)};
  for (std::size_t i = 0; i < settings_.slots; ++i) {
    landing.slots.push_back(zone_.base + (i * settings_.slot_bytes));
  }
  scheduler_ =
      std::make_unique<sc::Scheduler>(catalog_, *board_, wake_,
                                      sc::Lanes{.storage = storage_lane_.get(),
                                                .device = device_lane_.get(),
                                                .cpu = nullptr,
                                                .backing = backing_lane_.get(),
                                                .copy = copy_lane_.get()},
                                      sc::SchedulerSettings{.tasks = 16,
                                                            .budget = budget,
                                                            .poll_window = settings_.poll_window,
                                                            .landing = landing,
                                                            .observer = settings_.observer});
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
  if (copy_lane_ != nullptr) {
    threads_.emplace_back([this] { copy_lane_->RunSubmission(); });
    threads_.emplace_back([this] { copy_lane_->RunCompletion(); });
  }
}

void PagedNode::Round() {
  (void)storage_lane_->Turn(false);
  (void)backing_lane_->Turn();
  (void)device_lane_->SubmissionTurn();
  (void)device_lane_->CompletionTurn();
  if (copy_lane_ != nullptr) {
    (void)copy_lane_->SubmissionTurn();
    (void)copy_lane_->CompletionTurn();
  }
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
  LoadStats stats{.what = std::move(what), .extents = 0, .seconds = 0, .requests = 0, .pieces = 0};
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
  const std::uint64_t requests = counting_->requests.load(std::memory_order_relaxed);
  const std::uint64_t pieces = counting_->pieces.load(std::memory_order_relaxed);
  const auto start = std::chrono::steady_clock::now();
  if (auto posted =
          Post(std::make_unique<RunProgram>(done, *closure, sc::DeviceJob{}), done, stats.what);
      !posted) {
    return posted;
  }
  stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  // The load's reads have all completed, so their counts were made before.
  stats.requests = counting_->requests.load(std::memory_order_relaxed) - requests;
  stats.pieces = counting_->pieces.load(std::memory_order_relaxed) - pieces;
  log.push_back(std::move(stats));
  return {};
}

Status PagedNode::Evict(std::vector<ExtentId> extents, sc::EvictOptions options) {
  if (auto ended = EndRequestsOver(extents); !ended) {
    return std::unexpected(ended.error());
  }
  Done done;
  return Post(std::make_unique<EvictProgram>(done, std::move(extents), options), done,
              "an eviction");
}

Status PagedNode::Swap(std::vector<ExtentId> out, const catalog::Closure& in, bool handoff,
                       SwapReport& report) {
  // Asked for between a request's steps: the requests holding what goes
  // out end first, since the swap would wait for their leases.
  auto ended = EndRequestsOver(out);
  if (!ended) {
    return std::unexpected(ended.error());
  }
  Done done;
  auto swapped = Post(std::make_unique<SwapProgram>(done, std::move(out), in, handoff, report),
                      done, "a swap");
  report.requests_ended = *ended;
  return swapped;
}

std::expected<sc::SchedulerStats, std::string> PagedNode::Stats() {
  sc::SchedulerStats stats;
  if (auto r = Call(
          [&]() -> Status {
            stats = scheduler_->stats();
            return {};
          },
          "reading the scheduler's counters");
      !r) {
    return std::unexpected(r.error());
  }
  return stats;
}

Status PagedNode::Job(const catalog::Closure& closure, sc::DeviceJob job, std::string_view what,
                      std::uint32_t stream) {
  if (const auto open = requests_.find(stream); open != requests_.end()) {
    return Step(stream, *open->second, closure, std::move(job), what);
  }
  Done done;
  Timing timing;
  const auto called = std::chrono::steady_clock::now();
  auto posted = Post(
      std::make_unique<RunProgram>(done, closure, Timed(std::move(job), stream, timing), stream),
      done, what);
  if (posted) {
    Note(stream, called, timing);
  }
  return posted;
}

sc::DeviceJob PagedNode::Timed(sc::DeviceJob job, std::uint32_t stream, Timing& timing) {
  if (stream >= events_.size()) {
    return job;  // not a compute stream: untimed
  }
  cudaEvent_t begin = events_[stream].first;
  cudaEvent_t end = events_[stream].second;
  return [inner = std::move(job), &timing, begin,
          end](providers::NativeStream native) mutable -> sc::JobResult {
    auto* const s = static_cast<cudaStream_t>(native.handle);
    timing.started = std::chrono::steady_clock::now();
    (void)cudaEventRecord(begin, s);  // for timing only: a failure leaves the span unread
    const sc::JobResult result = inner(native);
    (void)cudaEventRecord(end, s);
    (void)cudaGetLastError();
    timing.queued = std::chrono::steady_clock::now();
    timing.ran = true;
    return result;
  };
}

void PagedNode::Note(std::uint32_t stream, std::chrono::steady_clock::time_point called,
                     const Timing& timing) {
  if (stream >= times_.size()) {
    return;
  }
  const auto seconds = [](std::chrono::steady_clock::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  const auto now = std::chrono::steady_clock::now();
  StepTimes& t = times_[stream];
  ++t.steps;
  t.wall += seconds(now - called);
  if (timing.ran) {
    t.dispatch += seconds(timing.started - called);
    t.job += seconds(timing.queued - timing.started);
    t.after += seconds(now - timing.queued);
    float ms = 0;
    // The job's fence has completed, so both events have.
    if (cudaEventElapsedTime(&ms, events_[stream].first, events_[stream].second) == cudaSuccess) {
      t.device += static_cast<double>(ms) / 1e3;
    }
    (void)cudaGetLastError();
  }
}

StepTimes PagedNode::TakeTimes(std::uint32_t stream) {
  if (stream >= times_.size()) {
    return {};
  }
  return std::exchange(times_[stream], StepTimes{});
}

void PagedNode::Signal(std::uint64_t request) {
  sc::Control signal = sc::SignalRequest{.request = request};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(signal)) == base::PushResult::kFull) {
    if (threads_.empty()) {
      Round();
    } else {
      std::this_thread::yield();
    }
  }
}

Status PagedNode::BeginRequest(std::uint32_t stream, const catalog::Closure& closure,
                               std::string_view what) {
  if (requests_.contains(stream)) {
    return Error(std::format("{}: a request is open on stream {} already", what, stream));
  }
  auto open = std::make_unique<OpenRequest>();
  open->what = what;
  open->closure = closure;
  // Sorted (a closure is by construction; a hand-made one is made so), so
  // Step and EndRequestsOver can search it.
  std::ranges::sort(open->closure.extents);
  auto program = std::make_unique<RequestProgram>(open->done, closure, open->channel);
  if (!threads_.empty()) {
    open->request = Submit(std::move(program));
  } else {
    open->request = ++request_;
    if (!scheduler_->Start(open->request, std::move(program), 1)) {
      return Error(std::format("{} was not admitted", what));
    }
  }
  // Until its lease is held, or its task has ended without one.
  auto give_up = std::chrono::steady_clock::now() + kPatience;
  bool cancelled = false;
  while (!open->channel.held.load(std::memory_order_acquire) && !open->done.gone.load()) {
    if (std::chrono::steady_clock::now() > give_up && !cancelled) {
      cancelled = true;
      Cancel(open->request);  // Await below waits for the drain
    }
    if (threads_.empty()) {
      Round();
    } else {
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
  if (cancelled || !open->channel.held.load(std::memory_order_acquire)) {
    auto ended = Await(open->done, what, open->request);
    return Error(std::format("{}: no lease{}", what, ended ? "" : ": " + ended.error()));
  }
  requests_.emplace(stream, std::move(open));
  return {};
}

Status PagedNode::Step(std::uint32_t stream, OpenRequest& open, const catalog::Closure& closure,
                       sc::DeviceJob job, std::string_view what) {
  // Within the request's lease, exactly: the same entries, or each an
  // extent it holds at the contents it recorded (a whole model's closure
  // compares equal at a fraction of the search's cost).
  if (closure.extents != open.closure.extents &&
      !std::ranges::all_of(closure.extents, [&open](const auto& extent) {
        return std::ranges::binary_search(open.closure.extents, extent);
      })) {
    return Error(std::format("{}: its closure is not within {}'s", what, open.what));
  }
  Timing timing;
  open.channel.job = Timed(std::move(job), stream, timing);
  open.channel.stream = stream;
  const std::uint64_t before = open.channel.steps.load(std::memory_order_acquire);
  const auto called = std::chrono::steady_clock::now();
  Signal(open.request);
  // Spinning, not sleeping: the step's result is the next step's input,
  // and a sleeping thread wakes slowly on the Spark (RE-017).
  auto give_up = called + kPatience;
  bool cancelled = false;
  while (open.channel.steps.load(std::memory_order_acquire) == before && !open.done.gone.load()) {
    if (std::chrono::steady_clock::now() > give_up) {
      if (cancelled) {
        std::println(stderr, "{} did not finish, nor drain once cancelled: aborting", what);
        std::abort();
      }
      cancelled = true;
      give_up = std::chrono::steady_clock::now() + kPatience;
      Cancel(open.request);
    }
    if (threads_.empty()) {
      Round();
    } else {
      std::this_thread::yield();
    }
  }
  if (open.channel.steps.load(std::memory_order_acquire) == before) {
    // The request's task ended (cancelled, or failed): nothing holds the
    // job, which never ran, any more.
    open.channel.job = nullptr;
    auto ended = Await(open.done, open.what, open.request);
    requests_.erase(stream);  // `open` is gone from here on
    return Error(std::format("{}: the request ended{}", what, ended ? "" : ": " + ended.error()));
  }
  Note(stream, called, timing);
  if (open.channel.step_failed.load(std::memory_order_relaxed)) {
    const int error = open.channel.step_error.load(std::memory_order_relaxed);
    return Error(std::format(
        "{} failed{}", what,
        error >= 0 ? ": " + sc::ToString(static_cast<sc::WorkError>(error)) : std::string()));
  }
  return {};
}

Status PagedNode::EndRequest(std::uint32_t stream) {
  const auto found = requests_.find(stream);
  if (found == requests_.end()) {
    return Error(std::format("no request is open on stream {}", stream));
  }
  OpenRequest& open = *found->second;
  open.channel.end = true;
  Signal(open.request);
  auto ended = Await(open.done, open.what, open.request);
  requests_.erase(found);
  return ended;
}

std::expected<std::uint64_t, std::string> PagedNode::EndRequestsOver(
    std::span<const ExtentId> extents) {
  std::vector<std::uint32_t> holding;
  for (const auto& [stream, open] : requests_) {
    const auto& held = open->closure.extents;  // sorted by extent
    if (std::ranges::any_of(extents, [&held](ExtentId extent) {
          const auto at = std::ranges::lower_bound(held, extent, {},
                                                   &std::pair<ExtentId, std::uint64_t>::first);
          return at != held.end() && at->first == extent;
        })) {
      holding.push_back(stream);
    }
  }
  for (const std::uint32_t stream : holding) {
    if (auto ended = EndRequest(stream); !ended) {
      return std::unexpected(ended.error());
    }
  }
  return holding.size();
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
  // Requests first: their leases would hold what the teardown evicts.
  std::vector<std::uint32_t> open;
  open.reserve(requests_.size());
  for (const auto& [stream, request] : requests_) {
    open.push_back(stream);
  }
  for (const std::uint32_t stream : open) {
    if (auto ended = EndRequest(stream); !ended) {
      problems.push_back(ended.error());
    }
  }
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
    if (copy_lane_ != nullptr) {
      copy_lane_->Close();
    }
    if (driven) {
      for (int i = 0; i < 1000 && (storage_->in_flight() > 0 || i < 10); ++i) {
        (void)storage_lane_->Turn(false);
        (void)backing_lane_->Turn();
        (void)device_lane_->SubmissionTurn();
        (void)device_lane_->CompletionTurn();
        if (copy_lane_ != nullptr) {
          (void)copy_lane_->SubmissionTurn();
          (void)copy_lane_->CompletionTurn();
        }
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
  for (const auto& [begin, end] : events_) {
    (void)cudaEventDestroy(begin);
    (void)cudaEventDestroy(end);
  }
  events_.clear();
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
