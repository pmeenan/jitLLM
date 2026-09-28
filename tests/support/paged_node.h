// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// One node for the paged harnesses (docs/backend-proof.md, P2-P5 and
// BP-S3; M3's swap runner): CUDA device 0 with its providers, a stream per
// model plus the copy stream, io_uring, one catalog domain, the landing
// zone (D-081), the scheduler and its lanes (the zone's copies on a copy
// lane of their own by default, RE-029), and the workspace the models
// share. Model
// runners (benchmarks/fp16_runner.h, exl3_runner.h) register their memory
// and sources with it and post their work through it, so two models can
// share one catalog, one scheduler and one zone (BP-S3). CUDA builds only;
// harness code, never linked into a production binary.
//
// The order of use:
// 1. Open: the device, providers, streams, storage, domain and zone.
// 2. Each model maps its own memory (MapResident, Pinned) and reserves its
//    weights' places, then MapWorkspace maps the shared workspace at the
//    largest request.
// 3. Start(B) builds the scheduler with the execution budget B; each model
//    registers its sources (scheduler().SetSource).
// 4. Run puts the scheduler and every lane on its own thread (unless the
//    lanes are driven inline); from then on only the scheduler's thread
//    touches the scheduler and the catalog (Call runs a function there).
// 5. TearDown: a fence on each model's stream, the models' weights
//    evicted, the scheduler stopped and the lanes drained (or, if setup
//    ended before Start, a fence on each compute stream); then each
//    model's Release (launch contexts, its memory); then the streams, the
//    zone and the workspace, with every backing checked released.
//
// The shared workspace orders nothing between the models' streams: leases
// are shared, so two models' jobs over it would race. They never overlap
// because Post (and Load, Evict, Job, Call, Acquire) returns only once its
// program is gone, a job's only after its fence completed. A caller that
// Submits without waiting runs no other model's job until that request is
// gone.
//
// Memory is registered as spans for BP-A1's in-process check: each has an
// owner (a model's index, or kShared for the zone and the workspace), and
// Covered accepts a model's own spans and the shared ones.

#ifndef JITLLM_TESTS_SUPPORT_PAGED_NODE_H_
#define JITLLM_TESTS_SUPPORT_PAGED_NODE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "base/bytes.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "paged_programs.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/uring_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace jitllm::test_support {

using Status = std::expected<void, std::string>;

inline constexpr std::uint64_t kPagedExtent = std::uint64_t{2} << 20U;  // D-033, D-056's chunk
inline constexpr std::size_t kPagedDepth = 4;                           // D-034's bulk depth
inline constexpr std::size_t kPagedSlots = 2 * kPagedDepth;             // D-081's zone
inline constexpr int kShared = -1;  // the owner of the zone and the shared workspace

// A cataloged range of memory, for the coverage check.
struct Span {
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  catalog::ExtentId extent;
  catalog::MemoryClass memory_class = catalog::MemoryClass::kUnknown;
  bool device = true;
  int owner = kShared;
};

// Memory mapped at setup and released at teardown: one reservation, one
// 2 MiB backing and one resident extent per 2 MiB.
struct Mapped {
  std::string name;
  providers::ReservationId reservation;
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;  // rounded to extents
  std::vector<providers::BackingId> backings;
  std::vector<catalog::ExtentId> extents;
};

struct LoadStats {
  std::string what;
  std::uint64_t extents = 0;  // nonresident when it was posted
  double seconds = 0;
  std::uint64_t requests = 0;  // direct-I/O requests the load's reads took
  std::uint64_t pieces = 0;    // the chunks (segments) they carried
};

struct NodeSettings {
  std::size_t compute_streams = 1;  // one per model: streams 0..n-1; the copy stream is n
  std::size_t slots = kPagedSlots;
  bool inline_lanes = false;
  // BP-P1's coalesced reads (64 MiB spans): the reader's option, off by
  // default as the reader's own default is.
  bool coalesce = false;
  // The zone's copies on a lane of their own (RE-029; scheduler.h), on
  // the copy stream, so a model's long job never holds them up; otherwise
  // on the device lane, as in M2.
  bool copy_lane = true;
  // Each slot's bytes (and the reader's largest request): 2 MiB, or more
  // for reads longer than their extent (a DeepSeek expert slab's pages,
  // benchmarks/dsv4_runner.h). A multiple of 4 KiB.
  std::uint64_t slot_bytes = kPagedExtent;
  // Told of each page-in's progress, on the scheduler's thread; outlives
  // the node. Optional.
  scheduler::PageInObserver* observer = nullptr;
};

// What the storage lane hands io_uring (BP-P1): requests, and the pieces
// they carry (a segment each; a plain request is one). Counted on the
// lane's thread, read between loads.
class CountingStorage final : public providers::Storage {
 public:
  explicit CountingStorage(providers::Storage& inner) : inner_(inner) {}
  std::size_t depth() const override { return inner_.depth(); }
  std::size_t in_flight() const override { return inner_.in_flight(); }
  providers::Submission Submit(const providers::IoRequest& request) override;
  providers::Submission Cancel(std::uint64_t token) override { return inner_.Cancel(token); }
  std::size_t Harvest(std::span<providers::IoCompletion> out, bool wait) override {
    return inner_.Harvest(out, wait);
  }
  void Wake() override { inner_.Wake(); }

  std::atomic<std::uint64_t> requests{0};
  std::atomic<std::uint64_t> pieces{0};

 private:
  providers::Storage& inner_;
};

// What a model gives the node's teardown.
class PagedModel {
 public:
  PagedModel() = default;
  PagedModel(const PagedModel&) = delete;
  PagedModel& operator=(const PagedModel&) = delete;
  PagedModel(PagedModel&&) = delete;
  PagedModel& operator=(PagedModel&&) = delete;
  virtual ~PagedModel() = default;
  // Its compute stream's index, and a closure a fence job there may lease.
  virtual std::uint32_t stream() const = 0;
  virtual const catalog::Closure& fence_closure() const = 0;
  // What must be evicted before the scheduler stops: every extent whose
  // backing the VMM lane manages.
  virtual std::vector<catalog::ExtentId> managed_extents() const = 0;
  // After the scheduler stopped cleanly: its launch contexts, memory and
  // files.
  virtual Status Release() = 0;
};

class PagedNode {
 public:
  explicit PagedNode(NodeSettings settings) : settings_(settings) {}
  PagedNode(const PagedNode&) = delete;
  PagedNode& operator=(const PagedNode&) = delete;
  PagedNode(PagedNode&&) = delete;
  PagedNode& operator=(PagedNode&&) = delete;
  // Stops the scheduler and joins the threads if TearDown did not.
  ~PagedNode();

  Status Open();
  // The shared workspace (scratch, discardable): the activations and the
  // GGML pool, each at the largest size a model asked for (at least one
  // extent). Once, before Start.
  Status MapWorkspace(std::uint64_t activations, std::uint64_t pool);
  Status Start(base::Bytes budget);
  void Run();
  // Every problem in one error. Once.
  Status TearDown(std::span<PagedModel* const> models);

  providers::VmmProvider& memory() { return *memory_; }
  providers::DeviceExecution& execution() { return *execution_; }
  providers::StreamId stream(std::uint32_t index) const { return streams_.at(index); }
  std::size_t device_class() const { return device_class_; }
  std::size_t host_class() const { return host_class_; }
  catalog::Catalog& catalog() { return catalog_; }
  const catalog::Catalog& catalog() const { return catalog_; }
  catalog::DomainId domain() const { return domain_; }
  scheduler::Scheduler& scheduler() { return *scheduler_; }
  base::Bytes budget() const { return budget_; }
  bool threaded() const { return !threads_.empty(); }
  bool inline_lanes() const { return settings_.inline_lanes; }
  bool coalesce() const { return settings_.coalesce; }
  const Mapped& zone() const { return zone_; }
  const Mapped& activations() const { return activations_; }
  const Mapped& pool() const { return pool_; }
  std::size_t slots() const { return settings_.slots; }

  // Maps `bytes` (rounded up to extents) of device or host VMM with
  // access, one resident extent of `memory_class` per 2 MiB. Before Run.
  Status MapResident(Mapped& mapped, std::string name, std::uint64_t bytes,
                     providers::BackingKind kind, catalog::MemoryClass memory_class,
                     catalog::Recovery recovery, int owner);
  // Pinned host memory, cataloged as staging. Before Run; freed at Close.
  std::expected<void*, std::string> Pinned(std::uint64_t bytes, int owner,
                                           std::vector<catalog::ExtentId>& staging);

  void AddSpan(const Span& span) { spans_.push_back(span); }
  void EraseSpans(const std::function<bool(const Span&)>& which) { std::erase_if(spans_, which); }
  void SortSpans();
  // Every byte of the range lies in cataloged, resident spans of device
  // memory of one class, each `owner`'s or shared; returns that class.
  // Read while no page-in or eviction runs.
  std::optional<catalog::MemoryClass> Covered(std::uint64_t address, std::uint64_t bytes,
                                              int owner) const;

  // Posts a program and waits for it to be destroyed. It refers to `done`,
  // and a job it queues may refer to the caller's frame, so this never
  // returns while either may still run: past the patience it cancels the
  // request and waits for the drain, and aborts the process if even that
  // does not come.
  Status Post(std::unique_ptr<scheduler::TaskProgram> program, Done& done, std::string_view what);
  Status Await(Done& done, std::string_view what, std::uint64_t request);
  // Posts without waiting (with threads): the request, to cancel or await.
  std::uint64_t Submit(std::unique_ptr<scheduler::TaskProgram> program);
  void Cancel(std::uint64_t request);

  Status Load(std::vector<catalog::ExtentId> extents, std::string what,
              std::vector<LoadStats>& log);
  Status Evict(std::vector<catalog::ExtentId> extents, scheduler::EvictOptions options = {});
  // A full swap (SwapProgram): `out` evicted, with their backing handed
  // to `in`'s loads if `handoff`, then `in` materialized.
  Status Swap(std::vector<catalog::ExtentId> out, const catalog::Closure& in, bool handoff,
              SwapReport& report);
  // The scheduler's counters, read on its thread.
  std::expected<scheduler::SchedulerStats, std::string> Stats();
  // Requests and pieces the storage lane has handed io_uring so far.
  std::uint64_t requests() const { return counting_->requests.load(); }
  Status Job(const catalog::Closure& closure, scheduler::DeviceJob job, std::string_view what,
             std::uint32_t stream);
  // Runs `call` on the scheduler's thread.
  Status Call(std::function<Status()> call, std::string_view what);
  // Makes room for `closure` under the budget and materializes it
  // (AcquireProgram), never evicting the shared workspace.
  Status Acquire(const catalog::Closure& closure, AcquireReport& report, std::string_view what);

 private:
  void Round();

  NodeSettings settings_;
  std::unique_ptr<providers::VmmProvider> memory_;
  std::unique_ptr<providers::DeviceExecution> execution_;
  std::unique_ptr<providers::UringStorage> storage_;
  std::unique_ptr<CountingStorage> counting_;  // over storage_: what the storage lane calls
  std::vector<providers::StreamId> streams_;   // compute streams, then the copy stream
  std::size_t device_class_ = 0;
  std::size_t host_class_ = 0;

  catalog::Catalog catalog_;
  catalog::DomainId domain_;
  std::vector<Span> spans_;  // sorted by base once setup ends
  base::Bytes budget_;

  Mapped zone_;
  Mapped activations_;
  Mapped pool_;
  std::vector<void*> pinned_;

  base::WakeFlag wake_;
  std::unique_ptr<scheduler::CompletionBoard> board_;
  std::unique_ptr<scheduler::StorageService> storage_lane_;
  std::unique_ptr<scheduler::DeviceService> device_lane_;
  std::unique_ptr<scheduler::BackingService> backing_lane_;
  std::unique_ptr<scheduler::DeviceService> copy_lane_;  // settings_.copy_lane
  std::unique_ptr<scheduler::Scheduler> scheduler_;
  std::vector<std::jthread> threads_;  // the scheduler first
  std::optional<std::expected<void, scheduler::Fault>> stopped_;
  std::uint64_t request_ = 0;
  bool torn_down_ = false;
};

}  // namespace jitllm::test_support

#endif  // JITLLM_TESTS_SUPPORT_PAGED_NODE_H_
