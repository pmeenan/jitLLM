// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P2, oracle rungs 4 and 5 (docs/backend-proof.md): the
// Qwen2.5-0.5B FP16 fixture from its v0 prepared artifact, paged into
// jitLLM's device VMM through the host-VMM landing zone (D-081) by the
// scheduler and its lanes, and executed as device jobs that hold leases on
// everything they touch (docs/experiments/backend-proof-p2/README.md).
//
//   jitllm_fp16_paged --artifact DIR --trajectory control|heldout --tokens FILE
//                     --fusion on|off --out DIR [--restores N] [--relocate]
//                     [--partial] [--spill premapped|managed]
//                     [--embeddings duplicated|shared]
//                     [--lanes threads|inline] [--record]
//   jitllm_fp16_paged --artifact DIR ... --out DIR --load-only N
//                     [--weights device|host] [--backing managed|premapped]
//
// - Memory (all registered in one catalog domain):
//   - every chunk of every group is an extent of device VMM with managed
//     backing (D-033), landed from its shard (D-081): read with O_DIRECT
//     into a landing slot, copied into its place by the copy engine, and
//     published once the copy's fence completes. Group g has a 2 MiB-
//     aligned region; its chunk k maps at the region's base + k x 2 MiB;
//   - the token table's chunks are also extents of host VMM, read there
//     directly: the embedding lookup runs on the CPU, as the bridge's does
//     from its CUDA_Host copy (the tied output head reads the device copy);
//   - the landing zone: 2 x depth (4) slots of 2 MiB of host VMM, a
//     declared persistent pool;
//   - the cache, activations, GGML pool scratch and cuBLAS workspace:
//     device VMM mapped at setup in 2 MiB extents; the input staging and
//     logits: pinned host memory, cataloged.
// - Rung 4: every weight is loaded before the first evaluation, which runs
//   the trajectory; a second repeats it from a cleared cache.
// - Rung 5 (--restores N): N more evaluations, each of which, at the
//   profile's restore point (after 32 tokens for control, 33 for heldout),
//   evicts every weight extent (device and host), releasing its backing,
//   and pages them all back in before continuing. With --relocate, every
//   second restore registers the weights at a second reservation, so they
//   come back at other addresses and every chunk's tensors are bound anew
//   (BP-P5). The cache stays resident.
// - BP-P2 (--partial): one more evaluation, which at the restore point
//   runs each of paging_cases.h's partial evictions in turn (one layer,
//   side vectors and biases, a shared small-tensor chunk, padded tails, a
//   tensor crossing a chunk boundary): it evicts exactly those extents,
//   checks that a chunk's job submitted without materializing is refused
//   before anything runs (invariants 1-2), and pages only them back in.
// - BP-P4 (--spill): one more evaluation, which after the first chunk (a
//   prefill) and again mid-decode (8 tokens after the restore point)
//   writes the cache back through the zone to an unnamed direct-I/O spill
//   file in --out and evicts it, then restores it. `premapped` keeps the
//   cache's backing mapped (its eviction is the catalog's alone) and
//   poisons it with 0xff before the restore; `managed` releases the
//   backing (D-033) and the restore maps fresh backing. Either way the
//   cache is copied back to the host before and after, and must match.
// - BP-P3 (--embeddings shared): the token table is held once, in device
//   VMM: each chunk's embedding rows are copied from it to pinned staging
//   (a job leasing both) and widened on the host, instead of read from
//   the host copy. The logits must equal the duplicated run's.
// - Explainable plans: for each chunk shape, the guaranteed bound (the
//   placement's activation extent and the plan's scratch bound) against
//   the observed peak (the highest activation byte a bound tensor reaches,
//   and the pool's peak in that chunk).
// - Each chunk is one device job (scheduler::LaunchWork) on the compute
//   stream, holding a lease on the whole closure until the fence after it
//   completes: it builds the embedding rows from the host table, copies the
//   inputs in the bridge's order, runs the plan bound through the registry
//   under the K-C launch context, and copies the logits back. Before each
//   chunk every tensor the plan binds is checked to lie in cataloged extents
//   of device memory, of the class it belongs to (BP-A1's in-process check).
// - The cuBLAS handle and its 32 MiB workspace are made current where the
//   bridge creates them: before the first chunk whose plan calls cuBLAS.
// - --lanes inline drives the scheduler and every lane from this thread,
//   in turns; --record needs it, since the launch recorder sees only its
//   own thread's calls (plan_compare.py). --lanes threads (the default) runs
//   each on its own thread, as a program wires them.
//
// Every evaluation's logits must equal the first's bit for bit; the first's
// are written, with a summary of the page-ins (bytes, times) and the
// coverage check.
//
// --load-only N measures page-in alone, through the same scheduler and
// lanes: the device weights loaded and evicted N times, nothing evaluated.
// --weights host reads them in place into host VMM instead (D-034's direct
// path, which D-081 replaced for execution); --backing premapped maps every
// extent's backing at setup instead of on each load, so an eviction is the
// catalog's alone and a load only reads (and copies). Together they separate
// the zone's copy and D-033's per-load backing from the reads themselves.
// --slots N sizes the zone other than D-081's 2 x depth, to see what the
// copy's hand-off costs.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/layout.h"
#include "base/bounded_queue.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "base/wake.h"
#include "catalog/catalog.h"
#include "execution/registry.h"
#include "fp16_common.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "launch_recorder.h"
#include "model/qwen2.h"
#include "paging_cases.h"
#include "plan_record.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/cuda/cuda_device_memory.h"
#include "providers/device_execution.h"
#include "providers/device_memory.h"
#include "providers/direct_reader.h"
#include "providers/uring_storage.h"
#include "scheduler/commands.h"
#include "scheduler/completions.h"
#include "scheduler/scheduler.h"
#include "scheduler/services.h"

namespace {

namespace kg = jitllm::kernels::ggml;
namespace sc = jitllm::scheduler;
using jitllm::base::Bytes;
using jitllm::catalog::ExtentId;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Recovery;
using jitllm::providers::BackingKind;
using jitllm::providers::ReservationId;
using Status = std::expected<void, std::string>;

constexpr std::uint64_t kExtent = std::uint64_t{2} << 20U;  // D-033, D-056's chunk
constexpr std::size_t kDepth = 4;                           // D-034's bulk depth
constexpr std::size_t kSlots = 2 * kDepth;                  // D-081's zone (--slots changes it)
constexpr auto kPatience = std::chrono::minutes(10);

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

// Nothing, or every problem in one error.
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

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

// ------------------------------------------------------------------ options

struct Options {
  std::filesystem::path artifact;
  std::string trajectory;
  std::filesystem::path tokens;
  bool fusion = true;
  std::filesystem::path out;
  int restores = 0;
  bool relocate = false;
  bool inline_lanes = false;
  bool record = false;
  int load_only = 0;
  bool weights_host = false;
  bool premapped = false;
  std::size_t slots = kSlots;
  bool partial = false;
  std::string spill;  // empty, premapped or managed
  bool shared_embeddings = false;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  bool fusion_set = false;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    if (a == "--record") {
      o.record = true;
      continue;
    }
    if (a == "--relocate") {
      o.relocate = true;
      continue;
    }
    if (a == "--partial") {
      o.partial = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    if (a == "--spill") {
      if (v != "premapped" && v != "managed") {
        return Error("--spill is premapped or managed");
      }
      o.spill = v;
    } else if (a == "--embeddings") {
      if (v != "duplicated" && v != "shared") {
        return Error("--embeddings is duplicated or shared");
      }
      o.shared_embeddings = v == "shared";
    } else if (a == "--artifact") {
      o.artifact = v;
    } else if (a == "--trajectory") {
      o.trajectory = v;
    } else if (a == "--tokens") {
      o.tokens = v;
    } else if (a == "--fusion") {
      o.fusion = v == "on";
      fusion_set = v == "on" || v == "off";
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--restores") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.restores).ec != std::errc{} ||
          o.restores < 0) {
        return Error("--restores takes a count");
      }
    } else if (a == "--load-only") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.load_only).ec != std::errc{} ||
          o.load_only < 1) {
        return Error("--load-only takes a count");
      }
    } else if (a == "--slots") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.slots).ec != std::errc{} ||
          o.slots < 1 || o.slots > 64) {
        return Error("--slots takes a count from 1 to 64");
      }
    } else if (a == "--weights") {
      if (v != "device" && v != "host") {
        return Error("--weights is device or host");
      }
      o.weights_host = v == "host";
    } else if (a == "--backing") {
      if (v != "managed" && v != "premapped") {
        return Error("--backing is managed or premapped");
      }
      o.premapped = v == "premapped";
    } else if (a == "--lanes") {
      if (v != "inline" && v != "threads") {
        return Error("--lanes is inline or threads");
      }
      o.inline_lanes = v == "inline";
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.artifact.empty() || o.trajectory.empty() || o.tokens.empty() || o.out.empty() ||
      !fusion_set || (o.record && !o.inline_lanes) ||
      ((o.weights_host || o.premapped || o.slots != kSlots) && o.load_only == 0) ||
      ((o.partial || !o.spill.empty() || o.shared_embeddings) && o.load_only > 0)) {
    return Error(
        "usage: jitllm_fp16_paged --artifact DIR --trajectory control|heldout --tokens FILE "
        "--fusion on|off --out DIR [--restores N] [--relocate] [--partial] "
        "[--spill premapped|managed] [--embeddings duplicated|shared] [--lanes threads|inline] "
        "[--record (with --lanes inline)] | --load-only N [--weights device|host] "
        "[--backing managed|premapped] [--slots N]");
  }
  return o;
}

// ------------------------------------------------------------------ memory

// A cataloged range of memory, for the coverage check: its extent, class
// and kind.
struct Span {
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  ExtentId extent;
  MemoryClass memory_class = MemoryClass::kUnknown;
  bool device = true;
};

// Memory the harness maps itself at setup and unmaps after the lanes have
// drained: one reservation, one 2 MiB backing per extent.
struct Mapped {
  std::string name;
  ReservationId reservation;
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;  // rounded to extents
  std::vector<jitllm::providers::BackingId> backings;
  std::vector<ExtentId> extents;
};

struct Coverage {
  std::uint64_t tensors = 0;
  std::array<std::uint64_t, jitllm::catalog::kMemoryClassCount> by_class{};
  std::uint64_t violations = 0;
  std::string first;
};

struct LoadStats {
  std::string what;
  std::uint64_t extents = 0;
  std::uint64_t bytes = 0;
  double seconds = 0;
};

// ------------------------------------------------------------------ programs

// What a program tells the thread that posted it.
struct Done {
  std::atomic<int> outcome{-1};
  std::atomic<bool> retired{false};
  std::atomic<int> error{-1};  // a WorkError that ended it
  // The program is destroyed: retired, or never admitted (a refused start
  // goes with its control). Its last touch of `done`: a job it submitted
  // retired first (its lease held until its fence), and one never
  // submitted went with it.
  std::atomic<bool> gone{false};
};

// A program that tells `done` when it is destroyed.
class HarnessProgram : public sc::TaskProgram {
 public:
  explicit HarnessProgram(Done& done) : done_(done) {}
  HarnessProgram(const HarnessProgram&) = delete;
  HarnessProgram& operator=(const HarnessProgram&) = delete;
  HarnessProgram(HarnessProgram&&) = delete;
  HarnessProgram& operator=(HarnessProgram&&) = delete;
  ~HarnessProgram() override { done_.gone.store(true); }

 protected:
  Done& done_;
};

// Materializes a closure, then optionally runs one device job over it.
class RunProgram final : public HarnessProgram {
 public:
  RunProgram(Done& done, jitllm::catalog::Closure closure, sc::DeviceJob job)
      : HarnessProgram(done), closure_(std::move(closure)), job_(std::move(job)) {}

  sc::Step Advance(sc::TaskContext& context) override {
    if (context.TakeFailure()) {
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    if (submitted_) {
      return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
    }
    const auto ready = context.Materialize(closure_);
    if (!ready) {
      if (ready.error() == sc::WorkError::kBusy) {
        return sc::Step::Yield();  // no mailbox now: again next turn
      }
      done_.error.store(static_cast<int>(ready.error()));
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    if (*ready == sc::Readiness::kWaiting) {
      return sc::Step::Wait();
    }
    if (!job_) {
      return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
    }
    const auto submitted =
        context.SubmitLaunch(closure_, sc::LaunchWork{.stream = 0, .job = std::move(job_)});
    if (!submitted) {
      done_.error.store(static_cast<int>(submitted.error()));
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    submitted_ = true;
    return sc::Step::Wait();
  }
  void Finished(sc::TaskOutcome outcome) override {
    done_.outcome.store(static_cast<int>(outcome));
  }
  void Retired() override { done_.retired.store(true); }

 private:
  jitllm::catalog::Closure closure_;
  sc::DeviceJob job_;
  bool submitted_ = false;
};

// Runs a call on the scheduler thread, which alone may change the
// scheduler's records once it runs (here: registering sources anew).
class CallProgram final : public HarnessProgram {
 public:
  CallProgram(Done& done, std::function<Status()> call)
      : HarnessProgram(done), call_(std::move(call)) {}
  sc::Step Advance(sc::TaskContext& /*context*/) override {
    status_ = call_();
    return sc::Step::Finish(status_ ? sc::TaskOutcome::kSucceeded : sc::TaskOutcome::kFailed);
  }
  void Finished(sc::TaskOutcome outcome) override {
    done_.outcome.store(static_cast<int>(outcome));
  }
  void Retired() override { done_.retired.store(true); }
  const Status& status() const { return status_; }

 private:
  std::function<Status()> call_;
  Status status_;
};

// Evicts extents one by one, each unmap waited for.
class EvictProgram final : public HarnessProgram {
 public:
  EvictProgram(Done& done, std::vector<ExtentId> extents)
      : HarnessProgram(done), extents_(std::move(extents)) {}
  sc::Step Advance(sc::TaskContext& context) override {
    if (context.TakeFailure()) {
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    while (next_ < extents_.size()) {
      const ExtentId extent = extents_[next_];
      if (context.catalog().Describe(extent).value().state !=
          jitllm::catalog::ExtentState::kResident) {
        ++next_;
        continue;
      }
      const auto evicted = context.Evict(extent);
      if (!evicted) {
        if (evicted.error() == sc::WorkError::kBusy) {
          return sc::Step::Yield();  // no mailbox now
        }
        done_.error.store(static_cast<int>(evicted.error()));
        return sc::Step::Finish(sc::TaskOutcome::kFailed);
      }
      ++next_;
      if (*evicted == sc::Readiness::kWaiting) {
        return sc::Step::Wait();
      }
    }
    return sc::Step::Finish(sc::TaskOutcome::kSucceeded);
  }
  void Finished(sc::TaskOutcome outcome) override {
    done_.outcome.store(static_cast<int>(outcome));
  }
  void Retired() override { done_.retired.store(true); }

 private:
  std::vector<ExtentId> extents_;
  std::size_t next_ = 0;
};

// Submits a job over a closure without materializing it first: with any
// extent nonresident the lease, and so the submission, must be refused
// before the job can run (BP-P2's incomplete closure).
class LaunchOnlyProgram final : public HarnessProgram {
 public:
  LaunchOnlyProgram(Done& done, jitllm::catalog::Closure closure, std::atomic<bool>& ran)
      : HarnessProgram(done), closure_(std::move(closure)), ran_(ran) {}
  sc::Step Advance(sc::TaskContext& context) override {
    if (submitted_) {
      return sc::Step::Finish(context.TakeFailure() ? sc::TaskOutcome::kFailed
                                                    : sc::TaskOutcome::kSucceeded);
    }
    std::atomic<bool>& ran = ran_;
    const auto submitted = context.SubmitLaunch(
        closure_, sc::LaunchWork{.stream = 0, .job = [&ran](jitllm::providers::NativeStream) {
                                   ran.store(true);
                                   return sc::JobResult::kQueued;
                                 }});
    if (!submitted) {
      done_.error.store(static_cast<int>(submitted.error()));
      return sc::Step::Finish(sc::TaskOutcome::kFailed);
    }
    submitted_ = true;
    return sc::Step::Wait();
  }
  void Finished(sc::TaskOutcome outcome) override {
    done_.outcome.store(static_cast<int>(outcome));
  }
  void Retired() override { done_.retired.store(true); }

 private:
  jitllm::catalog::Closure closure_;
  std::atomic<bool>& ran_;
  bool submitted_ = false;
};

// ------------------------------------------------------------------ the harness

// A chunk shape's guaranteed bound against its observed peak (bytes).
struct Peak {
  std::uint64_t chunks = 0;
  std::uint64_t activations_bound = 0;
  std::uint64_t activations_seen = 0;
  std::uint64_t scratch_bound = 0;
  std::uint64_t scratch_seen = 0;
};

struct PagingEvent {
  std::string what;
  std::uint64_t extents = 0;
  double seconds = 0;
  std::string detail;
};

class Harness {
 public:
  Harness(const Options& options, jitllm::test_support::Recording* recording, std::string& record)
      : o_(options), recording_(recording), record_(record) {}
  Harness(const Harness&) = delete;
  Harness& operator=(const Harness&) = delete;
  Harness(Harness&&) = delete;
  Harness& operator=(Harness&&) = delete;
  ~Harness() {
    if (auto finished = Teardown(); !finished) {
      std::println(stderr, "{}", finished.error());
    }
  }

  Status Run();
  // Evicts every weight, stops the scheduler, drains the lanes and
  // releases every backing; what could not be done, as an error. Once.
  Status Teardown();

 private:
  Status Setup();
  Status MapResident(Mapped& mapped, std::string name, std::uint64_t bytes, BackingKind kind,
                     MemoryClass memory_class, Recovery recovery);
  Status RegisterWeights();
  Status Place(std::size_t which);
  std::uint64_t WeightAddress(std::uint32_t resource) const;
  void AddSpan(std::uint64_t base, std::uint64_t size, ExtentId extent, MemoryClass memory_class,
               bool device);
  // Every byte of the range lies in cataloged spans, all device memory of
  // one of `classes`; returns that class.
  std::optional<MemoryClass> Covered(std::uint64_t address, std::uint64_t bytes) const;
  void Check(const kg::Qwen2Graph& graph);

  void Round();
  // Waits for a posted program to retire (or be refused). It refers to `done`,
  // and a job it queues may refer to its caller's frame, so it never
  // returns while either may still run: past the patience it cancels the
  // request and waits for the drain, and aborts the process if even that
  // does not come.
  Status Await(Done& done, std::string_view what, std::uint64_t request);
  Status Post(std::unique_ptr<sc::TaskProgram> program, Done& done, std::string_view what);
  Status Load(std::vector<ExtentId> extents, std::string what);
  Status Evict(std::vector<ExtentId> extents);
  Status Job(const jitllm::catalog::Closure& closure, sc::DeviceJob job, std::string_view what);
  Status Evaluate(int evaluation, std::vector<float>& result);
  Status Restore(int evaluation);
  // BP-P2, BP-P4 and BP-P3's helpers.
  Status Partial();
  Status Spill(std::string_view where);
  Status Snapshot(std::vector<std::byte>& out);
  Status GatherRows(std::span<const std::int32_t> tokens, std::vector<float>& embd);
  Status RegisterCache();
  Status Write(const std::vector<std::vector<float>>& results);
  Status WriteLoads();

  const Options& o_;
  jitllm::test_support::Recording* recording_;
  std::string& record_;

  jitllm::benchmarks::Trajectory t_;
  std::unique_ptr<jitllm::artifact::Artifact> artifact_;
  const jitllm::model::Qwen2Profile& profile_ = jitllm::model::Qwen25Instruct05B();
  jitllm::model::Qwen2Binding binding_;
  std::vector<jitllm::artifact::FileDescriptor> shards_;

  std::unique_ptr<jitllm::providers::VmmProvider> memory_;
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  std::unique_ptr<jitllm::providers::UringStorage> storage_;
  std::array<jitllm::providers::StreamId, 2> streams_{};  // compute, copy
  std::size_t device_class_ = 0;
  std::size_t host_class_ = 0;

  jitllm::catalog::Catalog catalog_;
  jitllm::catalog::DomainId domain_;
  std::vector<Span> spans_;  // sorted by base once setup ends

  // Weights: two places (the second for relocation), group regions, one
  // extent per chunk; the token table's host copy.
  std::array<ReservationId, 2> weights_{};
  std::array<std::uint64_t, 2> weights_base_{};
  std::uint64_t weights_bytes_ = 0;
  std::size_t place_ = 0;
  std::vector<std::uint64_t> group_region_;
  std::vector<std::vector<ExtentId>> chunk_extents_;  // by group, chunk
  std::vector<ExtentId> device_weights_;
  ReservationId table_;
  std::uint64_t table_base_ = 0;
  std::vector<ExtentId> table_extents_;
  std::uint64_t stored_bytes_ = 0;
  std::vector<jitllm::providers::BackingId> premapped_;  // --backing premapped, place 0

  Mapped zone_;
  Mapped kv_;
  Mapped activations_;
  Mapped scratch_;
  Mapped workspace_;
  void* inputs_ = nullptr;
  void* logits_ = nullptr;
  std::uint64_t input_bytes_ = 0;
  std::uint64_t logits_bytes_ = 0;
  std::uint64_t most_activations_ = 0;
  std::uint64_t most_scratch_ = 0;
  std::uint64_t cublas_bytes_ = 0;

  jitllm::base::WakeFlag wake_;
  std::unique_ptr<sc::CompletionBoard> board_;
  std::unique_ptr<sc::StorageService> storage_lane_;
  std::unique_ptr<sc::DeviceService> device_lane_;
  std::unique_ptr<sc::BackingService> backing_lane_;
  std::unique_ptr<sc::Scheduler> scheduler_;
  std::vector<std::jthread> threads_;  // the scheduler first
  std::optional<std::expected<void, sc::Fault>> stopped_;
  std::uint64_t request_ = 0;

  std::unique_ptr<kg::LaunchContext> launch_;
  std::unique_ptr<kg::CublasHandle> cublas_;
  std::unique_ptr<jitllm::execution::Registry> registry_;
  jitllm::catalog::Closure everything_;  // what a chunk leases
  jitllm::catalog::Closure cache_;       // what a cache clear leases

  Coverage coverage_;
  std::vector<LoadStats> loads_;
  bool torn_down_ = false;

  // --spill: the unnamed spill file, and a pinned copy of the cache.
  int spill_fd_ = -1;
  void* kv_copy_ = nullptr;
  std::uint64_t kv_used_ = 0;  // the cache's bytes (kv_.bytes rounds up)
  // --embeddings shared: pinned staging for a chunk's rows.
  void* rows_ = nullptr;
  std::uint64_t rows_bytes_ = 0;
  jitllm::catalog::Closure gather_;  // the device table and the staging
  std::vector<PagingEvent> events_;
  std::uint64_t kv_mismatches_ = 0;
  std::uint64_t refusals_ = 0;           // incomplete closures refused before launch
  std::map<std::uint32_t, Peak> peaks_;  // by chunk rows
};

void Harness::AddSpan(std::uint64_t base, std::uint64_t size, ExtentId extent,
                      MemoryClass memory_class, bool device) {
  spans_.push_back(Span{.base = base,
                        .size = size,
                        .extent = extent,
                        .memory_class = memory_class,
                        .device = device});
}

Status Harness::MapResident(Mapped& mapped, std::string name, std::uint64_t bytes, BackingKind kind,
                            MemoryClass memory_class, Recovery recovery) {
  mapped.name = std::move(name);
  mapped.bytes = jitllm::benchmarks::RoundUp(std::max<std::uint64_t>(bytes, 1), kExtent);
  auto reservation = memory_->Reserve(Bytes(mapped.bytes));
  if (!reservation) {
    return Error(std::format("reserving {}: {}", mapped.name, reservation.error().detail));
  }
  mapped.reservation = *reservation;
  mapped.base = memory_->RangeOf(*reservation).value().base;
  const std::size_t allocation_class = kind == BackingKind::kDevice ? device_class_ : host_class_;
  for (std::uint64_t at = 0; at < mapped.bytes; at += kExtent) {
    auto backing = memory_->Create(allocation_class, Bytes(kExtent));
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
                                      .size = Bytes(kExtent),
                                      .content = {}},
                                     true);
    if (!extent) {
      return Error(std::format("cataloging {}", mapped.name));
    }
    mapped.extents.push_back(*extent);
    AddSpan(mapped.base + at, kExtent, *extent, memory_class, kind == BackingKind::kDevice);
  }
  if (auto access = memory_->SetAccess(*reservation, Bytes(0), Bytes(mapped.bytes),
                                       jitllm::providers::Access::kReadWrite);
      !access) {
    return Error(std::format("access to {}: {}", mapped.name, access.error().detail));
  }
  return {};
}

std::uint64_t Harness::WeightAddress(std::uint32_t resource) const {
  const auto& r = artifact_->resources()[resource];
  return weights_base_.at(place_) + group_region_[r.group] + r.offset.value();
}

Status Harness::RegisterWeights() {
  const auto groups = artifact_->groups();
  group_region_.resize(groups.size());
  chunk_extents_.resize(groups.size());
  for (std::size_t g = 0; g < groups.size(); ++g) {
    group_region_[g] = weights_bytes_;
    weights_bytes_ += std::uint64_t{groups[g].chunks} * kExtent;
    stored_bytes_ += groups[g].stored.value();
  }
  for (std::size_t which = 0; which < weights_.size(); ++which) {
    if (which > 0 && o_.load_only > 0) {
      break;  // nothing relocates
    }
    auto reservation = memory_->Reserve(Bytes(weights_bytes_));
    if (!reservation) {
      return Error(std::format("reserving the weights: {}", reservation.error().detail));
    }
    weights_.at(which) = *reservation;
    weights_base_.at(which) = memory_->RangeOf(*reservation).value().base;
  }
  // The artifact's identity orders victim ties (catalog.h ContentKey).
  std::array<std::uint8_t, 32> id{};
  for (std::size_t i = 0; i < id.size() && (2 * i) + 1 < artifact_->id().size(); ++i) {
    (void)std::from_chars(artifact_->id().data() + (2 * i), artifact_->id().data() + (2 * i) + 2,
                          id.at(i), 16);
  }
  const auto& table = artifact_->resources()[binding_.token_embd];
  const std::uint32_t table_group = table.group;
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      auto extent = catalog_.AddExtent({.domain = domain_,
                                        .memory_class = MemoryClass::kWeights,
                                        .recovery = Recovery::kFromArtifact,
                                        .size = Bytes(kExtent),
                                        .content = {.artifact = id, .group = g, .chunk = c}});
      if (!extent) {
        return Error("cataloging a weight chunk");
      }
      chunk_extents_[g].push_back(*extent);
      device_weights_.push_back(*extent);
    }
  }
  if (o_.premapped) {
    // Every extent's backing, mapped once for the whole run.
    const std::size_t allocation_class = o_.weights_host ? host_class_ : device_class_;
    for (std::uint64_t at = 0; at < weights_bytes_; at += kExtent) {
      auto backing = memory_->Create(allocation_class, Bytes(kExtent));
      if (!backing || !memory_->Map(weights_[0], Bytes(at), *backing)) {
        return Error("premapping the weights");
      }
      premapped_.push_back(*backing);
    }
    if (!memory_->SetAccess(weights_[0], Bytes(0), Bytes(weights_bytes_),
                            jitllm::providers::Access::kReadWrite)) {
      return Error("access to the premapped weights");
    }
  }
  if (o_.shared_embeddings) {
    return Place(0);  // one copy of the table, in device VMM (BP-P3)
  }
  // The token table's host copy: its group's chunks again, read directly
  // into host VMM the CPU maps.
  auto host = memory_->Reserve(Bytes(std::uint64_t{groups[table_group].chunks} * kExtent));
  if (!host) {
    return Error(std::format("reserving the host table: {}", host.error().detail));
  }
  table_ = *host;
  table_base_ = memory_->RangeOf(*host).value().base;
  for (std::uint32_t c = 0; c < groups[table_group].chunks; ++c) {
    auto extent =
        catalog_.AddExtent({.domain = domain_,
                            .memory_class = MemoryClass::kWeights,
                            .recovery = Recovery::kFromArtifact,
                            .size = Bytes(kExtent),
                            .content = {.artifact = id, .group = table_group, .chunk = c}});
    if (!extent) {
      return Error("cataloging a host table chunk");
    }
    table_extents_.push_back(*extent);
    const auto range =
        jitllm::artifact::ChunkRangeOf(artifact_->layout(), {.group = table_group, .chunk = c});
    if (!range) {
      return Error("the table's chunk range");
    }
    const std::uint64_t address = table_base_ + (std::uint64_t{c} * kExtent);
    auto set = scheduler_->SetSource(
        *extent,
        sc::PageSource{.read = {.fd = shards_.at(range->shard).get(),
                                .offset = range->file_offset.value(),
                                // NOLINTNEXTLINE(performance-no-int-to-ptr): host VMM, CPU-mapped
                                .memory = reinterpret_cast<std::byte*>(address),
                                .length = range->length.value()},
                       .landed = false,
                       .destination = 0,
                       .backing = sc::BackingPlace{.reservation = table_,
                                                   .offset = Bytes(std::uint64_t{c} * kExtent),
                                                   .size = Bytes(kExtent),
                                                   .allocation_class = host_class_}});
    if (!set) {
      return Error(std::format("the table's source: {}", sc::ToString(set.error())));
    }
    AddSpan(address, kExtent, *extent, MemoryClass::kWeights, false);
  }
  return Place(0);
}

// Registers every device weight chunk's source at place `which`.
Status Harness::Place(std::size_t which) {
  const auto groups = artifact_->groups();
  std::erase_if(spans_, [&](const Span& span) {
    return span.memory_class == MemoryClass::kWeights && span.device;
  });
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range =
          jitllm::artifact::ChunkRangeOf(artifact_->layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      const std::uint64_t offset = group_region_[g] + (std::uint64_t{c} * kExtent);
      const std::uint64_t address = weights_base_.at(which) + offset;
      const ExtentId extent = chunk_extents_[g][c];
      std::optional<sc::BackingPlace> backing;
      if (!o_.premapped) {
        backing =
            sc::BackingPlace{.reservation = weights_.at(which),
                             .offset = Bytes(offset),
                             .size = Bytes(kExtent),
                             .allocation_class = o_.weights_host ? host_class_ : device_class_};
      }
      auto set = scheduler_->SetSource(
          extent,
          sc::PageSource{
              .read = {.fd = shards_.at(range->shard).get(),
                       .offset = range->file_offset.value(),
                       // In place, for --weights host (D-034).
                       // NOLINTNEXTLINE(performance-no-int-to-ptr)
                       .memory = o_.weights_host ? reinterpret_cast<std::byte*>(address) : nullptr,
                       .length = range->length.value()},
              .landed = !o_.weights_host,
              .destination = o_.weights_host ? 0 : address,
              .backing = backing});
      if (!set) {
        return Error(std::format("a chunk's source: {}", sc::ToString(set.error())));
      }
      AddSpan(address, kExtent, extent, MemoryClass::kWeights, !o_.weights_host);
    }
  }
  std::ranges::sort(spans_, {}, &Span::base);
  place_ = which;
  return {};
}

std::optional<MemoryClass> Harness::Covered(std::uint64_t address, std::uint64_t bytes) const {
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
        (found && *found != span->memory_class)) {
      return std::nullopt;
    }
    const auto view = catalog_.Describe(span->extent);
    if (!view || view->state != jitllm::catalog::ExtentState::kResident) {
      return std::nullopt;
    }
    found = span->memory_class;
    at = span->base + span->size;
    ++span;
  }
  return found;
}

// BP-A1's in-process check: every tensor the chunk's plan binds lies in
// cataloged, resident extents of device memory of one class, and each has
// the class it should: weights, the cache (live state) or the activations
// (scratch).
void Harness::Check(const kg::Qwen2Graph& graph) {
  const auto expect = [&](const ggml_tensor* t, MemoryClass memory_class) {
    ++coverage_.tensors;
    const std::optional<MemoryClass> covered = Covered(Address(t->data), ggml_nbytes(t));
    if (covered) {
      ++coverage_.by_class.at(static_cast<std::size_t>(*covered));
    }
    if (covered != memory_class) {
      if (coverage_.violations++ == 0) {
        coverage_.first =
            std::format("{} ({} bytes at {:#x})", t->name, ggml_nbytes(t), Address(t->data));
      }
    }
  };
  const auto kind_of = [&](const ggml_tensor* t) {
    const ggml_tensor* base = t->view_src != nullptr ? t->view_src : t;
    for (const auto& layer : graph.layers) {
      if (base == layer.k_cache || base == layer.v_cache) {
        return MemoryClass::kLiveState;
      }
    }
    // The other leaves are the weights; everything computed, and the
    // inputs, live in the activations.
    const auto inputs = graph.inputs();
    return base->op == GGML_OP_NONE && std::ranges::find(inputs, base) == inputs.end()
               ? MemoryClass::kWeights
               : MemoryClass::kScratch;
  };
  for (const ggml_tensor* node : graph.nodes) {
    expect(node, kind_of(node));
    for (const ggml_tensor* src : node->src) {
      if (src != nullptr) {
        expect(src, kind_of(src));
      }
    }
  }
}

// ------------------------------------------------------------------ driving

void Harness::Round() {
  (void)storage_lane_->Turn(false);
  (void)backing_lane_->Turn();
  (void)device_lane_->SubmissionTurn();
  (void)device_lane_->CompletionTurn();
  (void)scheduler_->Turn();
}

Status Harness::Await(Done& done, std::string_view what, std::uint64_t request) {
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
      if (threads_.empty()) {
        (void)scheduler_->Cancel(request);
      } else {
        sc::Control cancel = sc::CancelRequest{.request = request};
        // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
        while (scheduler_->Post(std::move(cancel)) == jitllm::base::PushResult::kFull) {
          std::this_thread::yield();
        }
      }
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

Status Harness::Post(std::unique_ptr<sc::TaskProgram> program, Done& done, std::string_view what) {
  sc::Control start =
      sc::StartRequest{.request = ++request_, .priority = 1, .program = std::move(program)};
  const std::uint64_t request = request_;
  if (threads_.empty()) {
    auto& begin = std::get<sc::StartRequest>(start);
    if (!scheduler_->Start(begin.request, std::move(begin.program), begin.priority)) {
      return Error(std::format("{} was not admitted", what));
    }
  } else {
    // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
    while (scheduler_->Post(std::move(start)) == jitllm::base::PushResult::kFull) {
      std::this_thread::yield();
    }
  }
  return Await(done, what, request);
}

Status Harness::Load(std::vector<ExtentId> extents, std::string what) {
  LoadStats stats{.what = std::move(what), .extents = 0, .bytes = 0, .seconds = 0};
  for (const ExtentId extent : extents) {
    if (catalog_.Describe(extent).value().state != jitllm::catalog::ExtentState::kResident) {
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
  stats.seconds = Seconds(std::chrono::steady_clock::now() - start);
  loads_.push_back(std::move(stats));
  return {};
}

Status Harness::Evict(std::vector<ExtentId> extents) {
  Done done;
  return Post(std::make_unique<EvictProgram>(done, std::move(extents)), done, "an eviction");
}

Status Harness::Job(const jitllm::catalog::Closure& closure, sc::DeviceJob job,
                    std::string_view what) {
  Done done;
  return Post(std::make_unique<RunProgram>(done, closure, std::move(job)), done, what);
}

// ------------------------------------------------------------------ setup

Status Harness::Setup() {
  if (auto set = Cuda(cudaSetDevice(0), "cudaSetDevice"); !set) {
    return set;
  }
  if (auto context = Cuda(cudaFree(nullptr), "the CUDA context"); !context) {
    return context;
  }
  auto memory = jitllm::providers::cuda::OpenDeviceMemory(0);
  if (!memory) {
    return Error(std::format("OpenDeviceMemory: {}", memory.error().detail));
  }
  memory_ = std::move(*memory);
  auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
  if (!execution) {
    return Error("OpenDeviceExecution failed");
  }
  execution_ = std::move(*execution);
  for (auto& stream : streams_) {
    auto created = execution_->CreateStream();
    if (!created) {
      return Error("CreateStream failed");
    }
    stream = *created;
  }
  bool host_found = false;
  for (std::size_t i = 0; i < memory_->Classes().size(); ++i) {
    if (memory_->Classes()[i].kind == BackingKind::kDevice) {
      device_class_ = i;
    } else {
      host_class_ = i;
      host_found = true;
    }
  }
  if (!host_found || memory_->Granularity() != Bytes(kExtent)) {
    return Error("this device has no host-NUMA VMM at 2 MiB (D-034, D-081)");
  }
  auto storage = jitllm::providers::UringStorage::Create(kDepth);
  if (!storage) {
    return Error(std::format("io_uring: {}", storage.error().message()));
  }
  storage_ = std::move(*storage);

  auto trajectory = jitllm::benchmarks::LoadTrajectory(o_.trajectory, o_.tokens);
  if (!trajectory) {
    return std::unexpected(trajectory.error());
  }
  t_ = std::move(*trajectory);
  auto artifact = jitllm::artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<jitllm::artifact::Artifact>(std::move(*artifact));
  auto binding = jitllm::model::BindQwen2(profile_, *artifact_);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  for (std::uint32_t s = 0; s < artifact_->shards().size(); ++s) {
    auto fd = artifact_->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards_.push_back(std::move(*fd));
  }

  domain_ = catalog_.AddDomain("gb10");
  // The zone first: a persistent pool, mapped before anything pages.
  if (auto zone = MapResident(zone_, "the landing zone", o_.slots * kExtent, BackingKind::kHost,
                              MemoryClass::kStaging, Recovery::kPinned);
      !zone) {
    return zone;
  }

  // Measure every chunk's activations, scratch and inputs on a context with
  // no workspace, as rung 3 does, then size the regions for the largest.
  auto measure =
      kg::LaunchContext::Create(0, *execution_, streams_[0], {.base = 0, .size = Bytes(0)});
  if (!measure) {
    return Error(measure.error().detail);
  }
  const std::uint64_t layer_cache = std::uint64_t{profile_.kv_width()} * t_.cells * 2;
  const std::uint64_t kv_bytes = layer_cache * 2 * profile_.layers;
  kv_used_ = kv_bytes;
  std::uint32_t most_rows = 0;
  {
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    const jitllm::benchmarks::ChunkMemory placeless{
        .weight = [](std::uint32_t) { return std::uint64_t{1} << 44U; },
        .kv = std::uint64_t{1} << 45U,
        .activations = 0,
        .activation_bytes = 0};
    std::uint32_t n_past = 0;
    for (const std::uint32_t rows : t_.chunks) {
      const std::uint32_t n_kv = jitllm::model::PaddedKv(n_past + rows, t_.cells);
      auto planned = jitllm::benchmarks::PlanChunk(profile_, binding_, placeless, t_.cells, rows,
                                                   n_kv, o_.fusion, choices);
      if (!planned) {
        return Error(std::format("chunk at {}: {}", n_past, planned.error()));
      }
      most_activations_ = std::max(most_activations_, planned->placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch_ = std::max(most_scratch_, *scratch);
      std::uint64_t inputs = 0;
      for (const ggml_tensor* input : planned->graph.inputs()) {
        inputs += jitllm::benchmarks::RoundUp(ggml_nbytes(input), 128);
      }
      input_bytes_ = std::max(input_bytes_, inputs);
      most_rows = std::max(most_rows, rows);
      n_past += rows;
    }
  }
  measure->reset();
  int major = 0;
  int minor = 0;
  (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  cublas_bytes_ = kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor)).value();
  if (auto r = MapResident(kv_, "the cache", kv_bytes, BackingKind::kDevice,
                           MemoryClass::kLiveState, Recovery::kPreserve);
      !r) {
    return r;
  }
  if (auto r = MapResident(activations_, "the activations", most_activations_, BackingKind::kDevice,
                           MemoryClass::kScratch, Recovery::kDiscardable);
      !r) {
    return r;
  }
  if (auto r = MapResident(scratch_, "the pool scratch", most_scratch_, BackingKind::kDevice,
                           MemoryClass::kScratch, Recovery::kDiscardable);
      !r) {
    return r;
  }
  if (auto r = MapResident(workspace_, "the cuBLAS workspace", cublas_bytes_, BackingKind::kDevice,
                           MemoryClass::kRuntime, Recovery::kPinned);
      !r) {
    return r;
  }
  logits_bytes_ = std::uint64_t{most_rows} * profile_.vocab * sizeof(float);
  if (auto r = Cuda(cudaMallocHost(&inputs_, input_bytes_), "the input staging"); !r) {
    return r;
  }
  if (auto r = Cuda(cudaMallocHost(&logits_, logits_bytes_), "the logits"); !r) {
    return r;
  }
  std::vector<ExtentId> staging;
  std::vector<std::pair<void*, std::uint64_t>> pinned = {{inputs_, input_bytes_},
                                                         {logits_, logits_bytes_}};
  if (o_.shared_embeddings) {
    rows_bytes_ = std::uint64_t{most_rows} * profile_.width * 2;
    if (auto r = Cuda(cudaMallocHost(&rows_, rows_bytes_), "the row staging"); !r) {
      return r;
    }
    pinned.emplace_back(rows_, rows_bytes_);
  }
  for (const auto& [pointer, bytes] : pinned) {
    auto extent = catalog_.AddExtent({.domain = domain_,
                                      .memory_class = MemoryClass::kStaging,
                                      .recovery = Recovery::kPinned,
                                      .size = Bytes(bytes),
                                      .content = {}},
                                     true);
    if (!extent) {
      return Error("cataloging the staging");
    }
    staging.push_back(*extent);
    AddSpan(Address(pointer), bytes, *extent, MemoryClass::kStaging, false);
  }

  // The scheduler and its lanes.
  board_ = std::make_unique<sc::CompletionBoard>(1024, wake_);
  storage_lane_ = std::make_unique<sc::StorageService>(
      *storage_,
      jitllm::providers::ReaderSettings{
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
  sc::LandingZone landing{.slots = {}, .slot_bytes = Bytes(kExtent), .stream = 1};
  for (std::size_t i = 0; i < o_.slots; ++i) {
    landing.slots.push_back(zone_.base + (i * kExtent));
  }
  scheduler_ = std::make_unique<sc::Scheduler>(
      catalog_, *board_, wake_,
      sc::Lanes{.storage = storage_lane_.get(),
                .device = device_lane_.get(),
                .cpu = nullptr,
                .backing = backing_lane_.get()},
      sc::SchedulerSettings{
          .tasks = 16, .budget = Bytes(std::uint64_t{64} << 30U), .landing = landing});
  if (auto r = RegisterWeights(); !r) {
    return r;
  }
  if (!o_.spill.empty()) {
    if (auto r = RegisterCache(); !r) {
      return r;
    }
  }
  if (o_.shared_embeddings) {
    // What a gather leases: the device table's chunks and the row staging.
    std::vector<ExtentId> gather =
        chunk_extents_.at(artifact_->resources()[binding_.token_embd].group);
    gather.push_back(staging.back());
    gather_ = catalog_.ClosureOfExtents(gather).value();
  }
  // What a chunk leases: every weight (both copies of the table), the
  // cache, the activations, the scratch, the workspace and the staging.
  std::vector<ExtentId> all = device_weights_;
  all.insert(all.end(), table_extents_.begin(), table_extents_.end());
  for (const Mapped* mapped : {&kv_, &activations_, &scratch_, &workspace_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging.begin(), staging.end());
  everything_ = catalog_.ClosureOfExtents(all).value();
  cache_ = catalog_.ClosureOfExtents(kv_.extents).value();

  auto launch = kg::LaunchContext::Create(0, *execution_, streams_[0],
                                          {.base = scratch_.base, .size = Bytes(most_scratch_)});
  if (!launch) {
    return Error(launch.error().detail);
  }
  launch_ = std::move(*launch);
  auto registry = jitllm::execution::Registry::Create(kg::Implementations());
  if (!registry) {
    return Error(registry.error().detail);
  }
  registry_ = std::make_unique<jitllm::execution::Registry>(std::move(*registry));
  if (!o_.inline_lanes) {
    threads_.emplace_back([this] { stopped_ = scheduler_->Run(); });
    threads_.emplace_back([this] { storage_lane_->Run(); });
    threads_.emplace_back([this] { device_lane_->RunSubmission(); });
    threads_.emplace_back([this] { device_lane_->RunCompletion(); });
    threads_.emplace_back([this] { backing_lane_->Run(); });
  }
  return {};
}

// ------------------------------------------------------------------ evaluating

Status Harness::Evaluate(int evaluation, std::vector<float>& result) {
  // A fresh cache, as the bridge's new context has: a job on the compute
  // stream, leasing the cache.
  const std::uint64_t kv_bytes =
      std::uint64_t{profile_.kv_width()} * t_.cells * 2 * 2 * profile_.layers;
  const std::uint64_t kv = kv_.base;
  if (auto cleared = Job(
          cache_,
          [kv, kv_bytes](jitllm::providers::NativeStream stream) {
            return cudaMemsetAsync(Pointer(kv), 0, kv_bytes,
                                   static_cast<cudaStream_t>(stream.handle)) == cudaSuccess
                       ? sc::JobResult::kQueued
                       : sc::JobResult::kUnknown;
          },
          "clearing the cache");
      !cleared) {
    return cleared;
  }
  result.clear();
  result.reserve(t_.tokens.size() * profile_.vocab);
  std::uint32_t n_past = 0;
  for (std::size_t k = 0; k < t_.chunks.size(); ++k) {
    const std::uint32_t rows = t_.chunks[k];
    auto inputs = jitllm::model::Qwen2ChunkInputs(profile_, t_.cells, n_past, rows);
    if (!inputs) {
      return std::unexpected(inputs.error());
    }
    const jitllm::benchmarks::ChunkMemory memory{
        .weight = [this](std::uint32_t resource) { return WeightAddress(resource); },
        .kv = kv_.base,
        .activations = activations_.base,
        .activation_bytes = most_activations_};
    auto planned =
        jitllm::benchmarks::PlanChunk(profile_, binding_, memory, t_.cells, rows, inputs->n_kv,
                                      o_.fusion, kg::DeviceChoicesOf(*launch_));
    if (!planned) {
      return Error(std::format("chunk {}: {}", k, planned.error()));
    }
    const bool calls_cublas = std::ranges::any_of(
        planned->plan.steps, [](const auto& s) { return s.implementation == kg::kMulMatCublas; });
    if (calls_cublas && !cublas_) {
      // The bridge creates its handle here, lazily: a job on the compute
      // stream, leasing the workspace and scratch it lends.
      std::vector<ExtentId> lent = workspace_.extents;
      lent.insert(lent.end(), scratch_.extents.begin(), scratch_.extents.end());
      Status made;
      if (auto created = Job(
              catalog_.ClosureOfExtents(lent).value(),
              [this, &made](jitllm::providers::NativeStream) {
                launch_.reset();
                auto handle = kg::CublasHandle::Create(
                    0, *execution_, streams_[0],
                    {.base = workspace_.base, .size = Bytes(cublas_bytes_)});
                if (!handle) {
                  made = Error(handle.error().detail);
                  return sc::JobResult::kNotStarted;
                }
                cublas_ = std::move(*handle);
                auto relaunch = kg::LaunchContext::Create(
                    0, *execution_, streams_[0],
                    {.base = scratch_.base, .size = Bytes(most_scratch_)}, cublas_.get());
                if (!relaunch) {
                  made = Error(relaunch.error().detail);
                  return sc::JobResult::kNotStarted;
                }
                launch_ = std::move(*relaunch);
                return sc::JobResult::kQueued;
              },
              "creating the cuBLAS handle");
          !created || !made) {
        return !made ? made : created;
      }
      // The plan's device choices follow the new context.
      planned =
          jitllm::benchmarks::PlanChunk(profile_, binding_, memory, t_.cells, rows, inputs->n_kv,
                                        o_.fusion, kg::DeviceChoicesOf(*launch_));
      if (!planned) {
        return Error(std::format("chunk {}: {}", k, planned.error()));
      }
    }
    auto bound = kg::BoundGraph::Bind(*registry_, planned->plan);
    if (!bound) {
      return Error(std::format("chunk {}: {}", k, bound.error().detail));
    }
    auto step_scratch = kg::PlanScratch(*launch_, planned->plan);
    if (!step_scratch) {
      return Error(step_scratch.error().detail);
    }
    Check(planned->graph);
    const std::uint64_t chunk_logits = std::uint64_t{rows} * profile_.vocab * sizeof(float);
    Status ran;
    const std::span<const std::int32_t> tokens = std::span(t_.tokens).subspan(n_past, rows);
    std::span<const std::uint16_t> host_table;
    std::vector<float> gathered;
    if (o_.shared_embeddings) {
      // BP-P3's shared arm: the rows come from the device table.
      if (auto r = GatherRows(tokens, gathered); !r) {
        return r;
      }
    } else {
      const auto& table = artifact_->resources()[binding_.token_embd];
      host_table =
          std::span(reinterpret_cast<const std::uint16_t*>(  // NOLINT(performance-no-int-to-ptr)
                        table_base_ + table.offset.value()),
                    std::size_t{profile_.vocab} * profile_.width);
    }
    launch_->ResetScratchPeak();
    auto job = [&, rows, chunk = static_cast<int>(k),
                n_past](jitllm::providers::NativeStream native) -> sc::JobResult {
      if (recording_ != nullptr) {
        for (const auto& event : recording_->Take()) {
          record_ += jitllm::test_support::EventLine(event);
        }
        record_ += jitllm::test_support::ChunkLine(
            {.evaluation = evaluation, .chunk = chunk, .rows = rows, .n_past = n_past});
      }
      // The embedding rows, from the host table this job's lease holds (or
      // gathered from the device table before it).
      std::vector<float> embd = gathered;
      if (!o_.shared_embeddings) {
        embd.assign(std::size_t{rows} * profile_.width, 0.0F);
        if (auto r =
                jitllm::model::EmbedRows(host_table, profile_.width, profile_.vocab, tokens, embd);
            !r) {
          ran = std::unexpected(r.error());
          return sc::JobResult::kNotStarted;
        }
      }
      auto* const stream = static_cast<cudaStream_t>(native.handle);
      const kg::Qwen2Graph& g = planned->graph;
      const std::array<std::pair<ggml_tensor*, const void*>, 6> sources = {
          {{g.embd, embd.data()},
           {g.positions, inputs->positions.data()},
           {g.k_idxs, inputs->k_idxs.data()},
           {g.v_idxs, inputs->v_idxs.data()},
           {g.mask, inputs->mask.data()},
           {g.out_ids, inputs->out_ids.data()}}};
      std::uint64_t staged = 0;
      bool queued = false;
      for (const auto& [tensor, source] : sources) {
        auto* at = static_cast<std::byte*>(inputs_) + staged;
        std::memcpy(at, source, ggml_nbytes(tensor));
        if (auto r = Cuda(cudaMemcpyAsync(tensor->data, at, ggml_nbytes(tensor),
                                          cudaMemcpyHostToDevice, stream),
                          "an input copy");
            !r) {
          ran = r;
          return queued ? sc::JobResult::kUnknown : sc::JobResult::kNotStarted;
        }
        queued = true;
        staged += jitllm::benchmarks::RoundUp(ggml_nbytes(tensor), 128);
      }
      if (auto r = bound->Run(*launch_); !r) {
        ran = Error(std::format("chunk {}: {}", chunk, r.error().detail));
        return r.error().error == kg::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                            : sc::JobResult::kFailed;
      }
      if (auto r = Cuda(cudaMemcpyAsync(logits_, g.logits->data, chunk_logits,
                                        cudaMemcpyDeviceToHost, stream),
                        "the logits copy");
          !r) {
        ran = r;
        return sc::JobResult::kUnknown;
      }
      if (recording_ != nullptr) {
        for (const auto& event : recording_->Take()) {
          record_ += jitllm::test_support::EventLine(event);
        }
        record_ += jitllm::test_support::EndChunkLine();
      }
      return sc::JobResult::kQueued;
    };
    if (auto r = Job(everything_, std::move(job), "a chunk"); !r || !ran) {
      return !ran ? ran : Error(std::format("chunk {}: {}", k, r.error()));
    }
    if (launch_->faulted()) {
      return Error(std::format("chunk {}: the launch context faulted", k));
    }
    // The job's fence has completed: the logits are in.
    const auto* values = static_cast<const float*>(logits_);
    result.insert(result.end(), values, values + (chunk_logits / sizeof(float)));
    // The chunk shape's guaranteed bound against what it reached.
    Peak& peak = peaks_[rows];
    ++peak.chunks;
    peak.activations_bound = std::max(peak.activations_bound, planned->placement.extent);
    peak.scratch_bound = std::max(peak.scratch_bound, *step_scratch);
    peak.scratch_seen = std::max(peak.scratch_seen, launch_->scratch_peak().value());
    const auto reach = [&](const ggml_tensor* t) {
      const std::uint64_t at = t != nullptr ? Address(t->data) : 0;
      if (at >= activations_.base && at < activations_.base + activations_.bytes) {
        peak.activations_seen =
            std::max(peak.activations_seen, at + ggml_nbytes(t) - activations_.base);
      }
    };
    for (const ggml_tensor* node : planned->graph.nodes) {
      reach(node);
      for (const ggml_tensor* src : node->src) {
        reach(src);
      }
    }
    n_past += rows;
    const int partial_at = o_.partial ? 3 + o_.restores : -1;
    const int spill_at = o_.spill.empty() ? -1 : 3 + o_.restores + (o_.partial ? 1 : 0);
    if (evaluation > 2 && evaluation <= 2 + o_.restores && n_past == t_.restore_after) {
      if (auto r = Restore(evaluation); !r) {
        return r;
      }
    }
    if (evaluation == partial_at && n_past == t_.restore_after) {
      if (auto r = Partial(); !r) {
        return r;
      }
    }
    if (evaluation == spill_at && (k == 0 || n_past == t_.restore_after + 8)) {
      if (auto r = Spill(k == 0 ? "after the first prefill chunk" : "mid-decode"); !r) {
        return r;
      }
    }
  }
  return {};
}

// BP-P2: each partial eviction in turn, at the restore point.
Status Harness::Partial() {
  const std::string layer =
      std::format("blk.{}.", profile_.layers / 2);  // a middle layer's resources
  const auto cases = jitllm::benchmarks::PartialCases(*artifact_, layer, false);
  for (std::size_t c = 0; c < cases.size(); ++c) {
    const auto& partial = cases[c];
    std::vector<ExtentId> extents;
    for (const auto& [group, chunk] : partial.chunks) {
      extents.push_back(chunk_extents_.at(group).at(chunk));
    }
    const std::size_t backings = memory_->backings();
    const auto start = std::chrono::steady_clock::now();
    if (auto r = Evict(extents); !r) {
      return r;
    }
    const double evicted = Seconds(std::chrono::steady_clock::now() - start);
    std::size_t resident = 0;
    for (const ExtentId extent : device_weights_) {
      resident += catalog_.Describe(extent).value().state == jitllm::catalog::ExtentState::kResident
                      ? 1
                      : 0;
    }
    if (resident + extents.size() != device_weights_.size() ||
        memory_->backings() + extents.size() != backings) {
      return Error(std::format("partial eviction '{}' did not evict exactly its {} extents",
                               partial.name, extents.size()));
    }
    // An incomplete closure refuses the launch: nothing may run.
    std::atomic<bool> ran{false};
    Done refused;
    const Status submitted = Post(std::make_unique<LaunchOnlyProgram>(refused, everything_, ran),
                                  refused, "a launch over an incomplete closure");
    if (submitted || ran.load() ||
        refused.error.load() != static_cast<int>(sc::WorkError::kNotResident)) {
      return Error(
          std::format("partial eviction '{}': a launch over an incomplete closure was "
                      "not refused before it ran",
                      partial.name));
    }
    ++refusals_;
    events_.push_back(PagingEvent{.what = "partial eviction",
                                  .extents = extents.size(),
                                  .seconds = evicted,
                                  .detail = partial.name});
    // Only the missing extents page in; the next chunk binds against them.
    if (auto r = Load(extents, std::format("partial restore: {}", partial.name)); !r) {
      return r;
    }
  }
  return {};
}

// BP-P4: the cache written back through the zone and evicted, then
// restored; its bytes before and after must match.
Status Harness::Spill(std::string_view where) {
  std::vector<std::byte> before;
  if (auto r = Snapshot(before); !r) {
    return r;
  }
  const std::size_t backings = memory_->backings();
  auto start = std::chrono::steady_clock::now();
  if (auto r = Evict(kv_.extents); !r) {
    return r;
  }
  events_.push_back(PagingEvent{.what = "state write-back",
                                .extents = kv_.extents.size(),
                                .seconds = Seconds(std::chrono::steady_clock::now() - start),
                                .detail = std::string(where)});
  for (const ExtentId extent : kv_.extents) {
    const auto view = catalog_.Describe(extent).value();
    if (view.state != jitllm::catalog::ExtentState::kNonresident || !view.preserved) {
      return Error("the cache is not nonresident and preserved after its write-back");
    }
  }
  const bool managed = o_.spill == "managed";
  if (memory_->backings() + (managed ? kv_.extents.size() : 0) != backings) {
    return Error("the write-back did not release (managed) or keep (premapped) the backing");
  }
  if (!managed) {
    // Poisoned while nonresident: the restore must bring back every byte.
    if (auto r = Cuda(cudaMemset(Pointer(kv_.base), 0xff, kv_.bytes), "poisoning the cache"); !r) {
      return r;
    }
    if (auto r = Cuda(cudaDeviceSynchronize(), "poisoning the cache"); !r) {
      return r;
    }
  }
  start = std::chrono::steady_clock::now();
  if (auto r = Load(kv_.extents, std::format("state restore ({})", where)); !r) {
    return r;
  }
  events_.push_back(PagingEvent{.what = "state restore",
                                .extents = kv_.extents.size(),
                                .seconds = Seconds(std::chrono::steady_clock::now() - start),
                                .detail = std::string(where)});
  std::vector<std::byte> after;
  if (auto r = Snapshot(after); !r) {
    return r;
  }
  for (std::size_t i = 0; i < before.size(); ++i) {
    kv_mismatches_ += before[i] != after.at(i) ? 1 : 0;
  }
  return {};
}

// The cache's bytes, copied to the host by a job that leases it.
Status Harness::Snapshot(std::vector<std::byte>& out) {
  const std::uint64_t kv = kv_.base;
  const std::uint64_t bytes = kv_used_;
  void* copy = kv_copy_;
  if (auto r = Job(
          cache_,
          [kv, bytes, copy](jitllm::providers::NativeStream stream) {
            return cudaMemcpyAsync(copy, Pointer(kv), bytes, cudaMemcpyDeviceToHost,
                                   static_cast<cudaStream_t>(stream.handle)) == cudaSuccess
                       ? sc::JobResult::kQueued
                       : sc::JobResult::kUnknown;
          },
          "copying the cache out");
      !r) {
    return r;
  }
  const auto* bytes_in = static_cast<const std::byte*>(kv_copy_);
  out.assign(bytes_in, bytes_in + bytes);
  return {};
}

// BP-P3's shared arm: a chunk's embedding rows copied from the device
// table to pinned staging by a job leasing both, then widened on the host
// as the bridge's CPU lookup widens them.
Status Harness::GatherRows(std::span<const std::int32_t> tokens, std::vector<float>& embd) {
  const std::uint64_t row_bytes = std::uint64_t{profile_.width} * 2;
  const std::uint64_t table = WeightAddress(binding_.token_embd);
  for (const std::int32_t token : tokens) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= profile_.vocab) {
      return Error(std::format("token {} is outside the vocabulary", token));
    }
  }
  void* rows = rows_;
  if (auto r = Job(
          gather_,
          [&, rows](jitllm::providers::NativeStream stream) {
            for (std::size_t i = 0; i < tokens.size(); ++i) {
              if (cudaMemcpyAsync(
                      static_cast<std::byte*>(rows) + (i * row_bytes),
                      Pointer(table + (static_cast<std::uint64_t>(tokens[i]) * row_bytes)),
                      row_bytes, cudaMemcpyDeviceToHost,
                      static_cast<cudaStream_t>(stream.handle)) != cudaSuccess) {
                return i == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kUnknown;
              }
            }
            return sc::JobResult::kQueued;
          },
          "gathering the embedding rows");
      !r) {
    return r;
  }
  std::vector<std::int32_t> local(tokens.size());
  for (std::size_t i = 0; i < local.size(); ++i) {
    local[i] = static_cast<std::int32_t>(i);
  }
  embd.assign(tokens.size() * profile_.width, 0.0F);
  return jitllm::model::EmbedRows(
      std::span(static_cast<const std::uint16_t*>(rows_), tokens.size() * profile_.width),
      profile_.width, static_cast<std::uint32_t>(tokens.size()), local, embd);
}

// --spill: the cache's write-back places, one 2 MiB range of an unnamed
// direct-I/O file per extent, with its backing managed (released on
// eviction) or kept mapped (premapped).
Status Harness::RegisterCache() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(), std::strerror(errno)));
  }
  if (auto r = Cuda(cudaMallocHost(&kv_copy_, kv_used_), "the cache's host copy"); !r) {
    return r;
  }
  const bool managed = o_.spill == "managed";
  for (std::size_t i = 0; i < kv_.extents.size(); ++i) {
    std::optional<sc::BackingPlace> backing;
    if (managed) {
      backing = sc::BackingPlace{.reservation = kv_.reservation,
                                 .offset = Bytes(i * kExtent),
                                 .size = Bytes(kExtent),
                                 .allocation_class = device_class_};
    }
    auto set = scheduler_->SetSource(
        kv_.extents[i],
        sc::PageSource{
            .read = {.fd = spill_fd_, .offset = i * kExtent, .memory = nullptr, .length = kExtent},
            .landed = true,
            .destination = kv_.base + (i * kExtent),
            .backing = backing,
            .write_back = true});
    if (!set) {
      return Error(std::format("the cache's write-back place: {}", sc::ToString(set.error())));
    }
  }
  if (managed) {
    kv_.backings.clear();  // the VMM lane releases them on eviction (D-033)
  }
  return {};
}

// Rung 5: every weight extent evicted, its backing released, then paged
// back in through the zone (and, with --relocate, every second time at the
// other place).
Status Harness::Restore(int evaluation) {
  std::vector<ExtentId> weights = device_weights_;
  weights.insert(weights.end(), table_extents_.begin(), table_extents_.end());
  const std::size_t backings = memory_->backings();
  if (auto r = Evict(weights); !r) {
    return r;
  }
  for (const ExtentId extent : weights) {
    if (catalog_.Describe(extent).value().state != jitllm::catalog::ExtentState::kNonresident) {
      return Error("a weight extent is still resident after the eviction");
    }
  }
  if (memory_->backings() + weights.size() != backings) {
    return Error("the eviction did not release every weight's backing");
  }
  if (o_.relocate && evaluation % 2 == 0) {
    // On the scheduler thread: it owns the sources.
    Done done;
    Status placed;
    if (auto r = Post(std::make_unique<CallProgram>(done,
                                                    [this, &placed] {
                                                      placed = Place(1 - place_);
                                                      return placed;
                                                    }),
                      done, "relocating the weights");
        !r) {
      return !placed ? placed : r;
    }
  }
  if (auto r = Load(device_weights_, std::format("restore {} (device)", evaluation - 2)); !r) {
    return r;
  }
  return Load(table_extents_, std::format("restore {} (host table)", evaluation - 2));
}

Status Harness::Run() {
  if (auto r = Setup(); !r) {
    return r;
  }
  if (o_.load_only > 0) {
    for (int i = 1; i <= o_.load_only; ++i) {
      if (auto r = Load(device_weights_, std::format("load {}", i)); !r) {
        return r;
      }
      if (auto r = Evict(device_weights_); !r) {
        return r;
      }
    }
    return WriteLoads();
  }
  if (auto r = Load(device_weights_, "initial (device)"); !r) {
    return r;
  }
  if (auto r = Load(table_extents_, "initial (host table)"); !r) {
    return r;
  }
  const int evaluations = 2 + o_.restores + (o_.partial ? 1 : 0) + (o_.spill.empty() ? 0 : 1);
  std::vector<std::vector<float>> results(static_cast<std::size_t>(evaluations));
  for (int e = 1; e <= evaluations; ++e) {
    if (auto r = Evaluate(e, results[static_cast<std::size_t>(e - 1)]); !r) {
      return r;
    }
  }
  return Write(results);
}

Status Harness::WriteLoads() {
  std::string loads;
  for (const LoadStats& load : loads_) {
    loads += std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f}}})",
                         loads.empty() ? "" : ",", load.what, load.extents, load.seconds);
  }
  std::uint64_t bytes = 0;
  for (const auto& group : artifact_->groups()) {
    bytes += group.stored.value();
  }
  std::filesystem::create_directories(o_.out);
  std::ofstream file(o_.out / "loads.json");
  file << std::format(
              R"({{"weights":"{}","backing":"{}","lanes":"{}","extents":{},"read_bytes":{},)"
              R"("zone_slots":{},"depth":{},"loads":[{}]}})",
              o_.weights_host ? "host" : "device", o_.premapped ? "premapped" : "managed",
              threads_.empty() ? "inline" : "threads", device_weights_.size(), bytes, o_.slots,
              kDepth, loads)
       << "\n";
  std::println("wrote {}", o_.out.string());
  return {};
}

Status Harness::Write(const std::vector<std::vector<float>>& results) {
  const std::vector<float>& first = results.front();
  std::string differences;
  std::size_t differing_total = 0;
  for (std::size_t e = 1; e < results.size(); ++e) {
    std::size_t differing = results[e].size() == first.size() ? 0 : first.size();
    for (std::size_t i = 0; i < std::min(first.size(), results[e].size()); ++i) {
      differing +=
          std::bit_cast<std::uint32_t>(first[i]) != std::bit_cast<std::uint32_t>(results[e][i]);
    }
    differences += std::format("{}{}", differences.empty() ? "" : ",", differing);
    differing_total += differing;
  }
  jitllm::base::Sha256 hash;
  hash.Update(std::as_bytes(std::span(first)));
  const std::string digest = jitllm::base::ToHex(hash.Finish());
  std::filesystem::create_directories(o_.out);
  {
    std::ofstream raw(o_.out / "logits.f32le", std::ios::binary);
    raw.write(reinterpret_cast<const char*>(first.data()),
              static_cast<std::streamsize>(first.size() * sizeof(float)));
    if (!raw) {
      return Error("the logits could not be written");
    }
  }
  std::string loads;
  for (const LoadStats& load : loads_) {
    loads += std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f}}})",
                         loads.empty() ? "" : ",", load.what, load.extents, load.seconds);
  }
  std::string by_class;
  for (std::size_t c = 0; c < coverage_.by_class.size(); ++c) {
    by_class += std::format("{}{}", c == 0 ? "" : ",", coverage_.by_class.at(c));
  }
  std::string chunks;
  for (const std::uint32_t c : t_.chunks) {
    chunks += std::format("{}{}", chunks.empty() ? "" : ",", c);
  }
  std::uint64_t device_bytes = 0;
  std::uint64_t table_bytes = 0;
  const auto groups = artifact_->groups();
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    device_bytes += groups[g].stored.value();
    if (g == artifact_->resources()[binding_.token_embd].group) {
      table_bytes = groups[g].stored.value();
    }
  }
  const std::string summary = std::format(
      R"({{"trajectory":"{}","fusion":{},"lanes":"{}","tokens":{},"chunks":[{}],"cells":{},)"
      R"("evaluations":{},"restores":{},"restore_after":{},"relocate":{},"logits_sha256":"{}",)"
      R"("bit_differences_from_first":[{}],"artifact":"{}","device_weight_extents":{},)"
      R"("device_weight_read_bytes":{},"host_table_extents":{},"host_table_read_bytes":{},)"
      R"("weights_backing_bytes":{},"stored_bytes":{},"zone_slots":{},"zone_bytes":{},)"
      R"("kv_bytes":{},"activations":{},"scratch":{},"cublas_workspace":{},"loads":[{}],)"
      R"("coverage":{{"tensors":{},"violations":{},"first":"{}","by_class":[{}]}}}})",
      t_.name, o_.fusion ? "true" : "false", o_.inline_lanes ? "inline" : "threads",
      t_.tokens.size(), chunks, t_.cells, results.size(), o_.restores, t_.restore_after,
      o_.relocate ? "true" : "false", digest, differences, artifact_->id(), device_weights_.size(),
      device_bytes, table_extents_.size(), table_bytes, device_weights_.size() * kExtent,
      stored_bytes_, o_.slots, zone_.bytes, kv_.bytes, activations_.bytes, scratch_.bytes,
      workspace_.bytes, loads, coverage_.tensors, coverage_.violations, coverage_.first, by_class);
  // BP-P2, BP-P4 and BP-P3's runs, and each chunk shape's bound against
  // its peak.
  std::string events;
  for (const PagingEvent& event : events_) {
    events += std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f},"detail":"{}"}})",
                          events.empty() ? "" : ",", event.what, event.extents, event.seconds,
                          event.detail);
  }
  std::string peaks;
  bool within = true;
  for (const auto& [rows, peak] : peaks_) {
    within = within && peak.activations_seen <= peak.activations_bound &&
             peak.scratch_seen <= peak.scratch_bound;
    peaks +=
        std::format(R"({}{{"rows":{},"chunks":{},"activations_bound":{},"activations_seen":{},)"
                    R"("scratch_bound":{},"scratch_seen":{}}})",
                    peaks.empty() ? "" : ",", rows, peak.chunks, peak.activations_bound,
                    peak.activations_seen, peak.scratch_bound, peak.scratch_seen);
  }
  {
    std::ofstream file(o_.out / "paging.json");
    file << std::format(R"({{"partial":{},"spill":"{}","embeddings":"{}","refused_launches":{},)"
                        R"("kv_bytes_differing":{},"events":[{}],"peaks":[{}]}})",
                        o_.partial ? "true" : "false", o_.spill,
                        o_.shared_embeddings ? "shared" : "duplicated", refusals_, kv_mismatches_,
                        events, peaks)
         << "\n";
  }
  {
    std::ofstream file(o_.out / "summary.json");
    file << summary << "\n";
  }
  std::println("wrote {}", o_.out.string());
  if (kv_mismatches_ != 0) {
    return Error(std::format("{} bytes of the cache differ after its restore", kv_mismatches_));
  }
  if (!within) {
    return Error("a chunk reached past its guaranteed bound (paging.json)");
  }
  if (coverage_.violations != 0) {
    return Error(
        std::format("{} bound tensors lie outside cataloged extents of their class; "
                    "first: {}",
                    coverage_.violations, coverage_.first));
  }
  if (differing_total != 0) {
    return Error(std::format("a later evaluation differs from the first: [{}]", differences));
  }
  return {};
}

Status Harness::Teardown() {
  if (torn_down_) {
    return {};
  }
  std::vector<std::string> problems;
  torn_down_ = true;
  if (scheduler_ != nullptr) {
    // Every page-in drained and every weight's backing released, then the
    // scheduler stops and the lanes drain (the program's order).
    std::vector<ExtentId> weights = device_weights_;
    weights.insert(weights.end(), table_extents_.begin(), table_extents_.end());
    if (o_.spill == "managed") {
      // Its backing is the VMM lane's to release: written back and evicted.
      weights.insert(weights.end(), kv_.extents.begin(), kv_.extents.end());
    }
    // A fence after anything noted on the compute stream (the measuring
    // launch context notes work even when no chunk runs), so it can be
    // destroyed.
    if ((!threads_.empty() || !scheduler_->fault()) &&
        !Job(
            cache_, [](jitllm::providers::NativeStream) { return sc::JobResult::kQueued; },
            "fencing the compute stream")) {
      problems.emplace_back("the compute stream could not be fenced");
    }
    if ((!threads_.empty() || !scheduler_->fault()) && !Evict(weights)) {
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
  }
  launch_.reset();
  cublas_.reset();
  if (execution_ != nullptr) {
    for (const auto stream : streams_) {
      if (stream.valid() && !execution_->DestroyStream(stream)) {
        problems.emplace_back("a stream could not be destroyed");
      }
    }
  }
  if (memory_ != nullptr) {
    for (Mapped* mapped : {&zone_, &kv_, &activations_, &scratch_, &workspace_}) {
      if (!mapped->reservation.valid()) {
        continue;
      }
      bool released =
          mapped->backings.empty() ||
          memory_->Unmap(mapped->reservation, Bytes(0), Bytes(mapped->backings.size() * kExtent))
              .has_value();
      for (const auto backing : mapped->backings) {
        released = memory_->Release(backing).has_value() && released;
      }
      if (!released || !memory_->Free(mapped->reservation)) {
        problems.push_back(std::format("{} could not be released", mapped->name));
      }
    }
    if (!premapped_.empty()) {
      bool released =
          memory_->Unmap(weights_[0], Bytes(0), Bytes(premapped_.size() * kExtent)).has_value();
      for (const auto backing : premapped_) {
        released = memory_->Release(backing).has_value() && released;
      }
      if (!released) {
        problems.emplace_back("the premapped weights could not be released");
      }
    }
    for (const ReservationId reservation : {weights_[0], weights_[1], table_}) {
      if (reservation.valid() && !memory_->Free(reservation)) {
        problems.emplace_back("a weights reservation still has mappings");
      }
    }
    if (memory_->backings() != 0) {
      problems.push_back(std::format("{} backings were left", memory_->backings()));
    }
  }
  (void)cudaFreeHost(inputs_);
  (void)cudaFreeHost(logits_);
  (void)cudaFreeHost(kv_copy_);
  (void)cudaFreeHost(rows_);
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
  }
  return Joined(problems);
}

}  // namespace

int main(int argc, char** argv) {
  const std::span<char*> args(argv, static_cast<std::size_t>(argc));
  // The recording starts before anything touches CUDA (--record).
  const bool record =
      std::ranges::any_of(args, [](const char* a) { return std::string_view(a) == "--record"; });
  std::unique_ptr<jitllm::test_support::Recording> recording;
  if (record) {
    recording = std::make_unique<jitllm::test_support::Recording>();
  }
  const auto options = Parse(args);
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  std::string lines;
  Status ran;
  {
    Harness harness(*options, recording.get(), lines);
    ran = harness.Run();
    if (auto finished = harness.Teardown(); !finished && ran) {
      ran = finished;
    }
  }
  if (recording) {
    for (const auto& event : recording->Take()) {
      lines += jitllm::test_support::EventLine(event);
    }
    std::filesystem::create_directories(options->out);
    std::ofstream file(options->out / "recording.jsonl");
    file << jitllm::test_support::HeaderLine(
                std::format("jitllm_fp16_paged {} fusion {}", options->trajectory,
                            options->fusion ? "on" : "off"),
                jitllm::test_support::LoadedCublas())
         << lines;
  }
  if (!ran) {
    std::println(stderr, "{}", ran.error());
    return 1;
  }
  return 0;
}
