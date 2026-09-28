// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Backend-proof P3, oracle rungs 4 and 5 for EXL3 (docs/backend-proof.md):
// an EXL3 fixture from its v0 prepared artifact, paged into jitLLM's device
// VMM through the host-VMM landing zone (D-081) by the scheduler and its
// lanes, and run through the native operation plan (kernels/exl3/qwen2.h)
// as device jobs that hold leases on everything they touch
// (docs/experiments/backend-proof-p3/README.md). fp16_paged.cc's harness,
// for the EXL3 plan.
//
//   jitllm_exl3_paged --artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O
//                     --plan PLAN.txt --ids FILE --out DIR
//                     [--prefixes 32,144,145,1023,1024] [--restores N]
//                     [--relocate] [--partial] [--spill premapped|managed]
//                     [--cancel-in-flight]
//                     [--lanes threads|inline] [--record]
//
// - Memory (all registered in one catalog domain):
//   - every chunk of every group is an extent of device VMM with managed
//     backing (D-033), landed from its shard (D-081): read with O_DIRECT
//     into a landing slot, copied into its place by the copy engine, and
//     published once the copy's fence completes. Group g has a 2 MiB-
//     aligned region; its chunk k maps at the region's base + k x 2 MiB;
//   - the landing zone: 2 x depth (4) slots of 2 MiB of host VMM;
//   - derived at load, device VMM mapped at setup: the norms widened
//     exactly to F32 (weights, pinned), and each layer's multi-GEMM tables
//     (runtime objects), rewritten whenever the weights move (BP-P5);
//   - the cache (live state), the activation region and the GGML pool
//     (scratch) and the EXL3 lock area (runtime): device VMM mapped at
//     setup; the host inputs, the logits' host copy and the derivation's
//     staging: pinned host memory, cataloged.
// - Every phase is planned first, as rung 3's harness plans it
//   (exl3_exec.cc); each is bound (Qwen2Program::Bind) just before it runs,
//   against the weights' current addresses, and every address the bound
//   program reads or writes is checked to lie in cataloged, resident
//   extents of device memory of its class (BP-A1's in-process check).
// - Rung 4: every weight is loaded before the first evaluation, which runs
//   every prefix's trajectory; a second repeats it.
// - Rung 5 (--restores N): N more evaluations, each of which, after every
//   prefix's prefill, evicts every weight extent, releasing its backing,
//   and pages them all back in before the 16 steps. With --relocate, every
//   second restore registers the weights at a second reservation, so they
//   come back at other addresses: the tables are rewritten and every phase
//   is bound anew. The cache stays resident.
// - Each phase is one device job on the compute stream, holding a lease on
//   the whole closure until the fence after it completes.
// - --record (with --lanes inline): the first evaluation, as exl3_exec.cc
//   records it (op_plan_compare.py).
// - BP-P2 (--partial): one more evaluation, which after the first prefix's
//   prefill runs each of paging_cases.h's partial evictions (one layer,
//   side vectors and biases, the trellis only, a shared small-tensor
//   chunk, padded tails, a tensor crossing a chunk boundary): it evicts
//   exactly those extents, checks that a phase's job submitted without
//   materializing is refused before anything runs (invariants 1-2), and
//   pages only them back in.
// - BP-P4 (--spill premapped|managed): one more evaluation, which after
//   every prefix's prefill and again after its eighth step writes the
//   cache back through the zone to an unnamed direct-I/O spill file in
//   --out and evicts it, then restores it; `premapped` keeps its backing
//   mapped and poisons it with 0xff first, `managed` releases the backing
//   (D-033). The cache's bytes before and after must match.
// - BP-P3: the head is a representation of its own, never the embedding:
//   checked at setup (distinct resources whose chunks do not overlap).
// - BP-L1 and BP-L3 (--cancel-in-flight, with threads): after the
//   evaluations, the largest reconstruction phase (the 1,023-row prefill,
//   GGML and EXL3 work, each reconstruction slice followed by its GEMM)
//   is submitted behind a gate its job queues first (a stream wait on a
//   host flag), and its request is cancelled once the job has queued
//   everything. While the gate holds, the phase's lease must still hold
//   every extent it touches, the activations and reconstruction scratch
//   among them; the task retires, cancelled, only after the gate opens and
//   the fence completes, and only then are the leases gone.
// - Explainable plans: for each phase kind, the guaranteed bound (the
//   plan's activation region and the GGML pool's bound) against the
//   observed peak (the highest region byte an operation reaches, and the
//   pool's peak in that phase).
//
// Every evaluation's logits must equal the first's bit for bit; the first's
// are written as exl3_exec.cc writes them (.npy per prefix), with a summary
// of the page-ins and the coverage check. Rung 4 against rung 3 is the
// comparison of those files with exl3_exec.cc's.

#include <cuda.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
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
#include <sstream>
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
#include "exl3_common.h"
#include "kernels/exl3/implementations.h"
#include "kernels/exl3/launch.h"
#include "kernels/exl3/qwen2.h"
#include "kernels/exl3/recon_gemm.h"
#include "kernels/exl3/validate.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "launch_recorder.h"
#include "model/qwen2.h"
#include "model/qwen2_exl3.h"
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

namespace exl3 = jitllm::kernels::exl3;
namespace kg = jitllm::kernels::ggml;
namespace model = jitllm::model;
namespace sc = jitllm::scheduler;
using jitllm::base::Bytes;
using jitllm::benchmarks::Bf16ToFloat;
using jitllm::benchmarks::Hex;
using jitllm::benchmarks::HexFile;
using jitllm::benchmarks::kCells;
using jitllm::benchmarks::kSuffix;
using jitllm::catalog::ExtentId;
using jitllm::catalog::MemoryClass;
using jitllm::catalog::Recovery;
using jitllm::providers::BackingKind;
using jitllm::providers::ReservationId;
using Status = std::expected<void, std::string>;

constexpr std::uint64_t kExtent = std::uint64_t{2} << 20U;  // D-033, D-056's chunk
constexpr std::size_t kDepth = 4;                           // D-034's bulk depth
constexpr std::size_t kSlots = 2 * kDepth;                  // D-081's zone
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

std::uint64_t RoundUp(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

struct Options {
  std::filesystem::path artifact;
  std::string fixture;
  model::Exl3Arm arm = model::Exl3Arm::kG;
  std::filesystem::path plan;
  std::filesystem::path ids;
  std::filesystem::path out;
  std::vector<int> prefixes = {32, 144, 145, 1023, 1024};
  int restores = 0;
  bool relocate = false;
  bool inline_lanes = false;
  bool record = false;
  bool partial = false;
  std::string spill;  // empty, premapped or managed
  bool cancel = false;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  bool arm = false;
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
    if (a == "--cancel-in-flight") {
      o.cancel = true;
      continue;
    }
    if (i + 1 >= args.size()) {
      return Error(std::format("{} needs a value", a));
    }
    const std::string_view v = args[++i];
    if (a == "--artifact") {
      o.artifact = v;
    } else if (a == "--spill") {
      if (v != "premapped" && v != "managed") {
        return Error("--spill is premapped or managed");
      }
      o.spill = v;
    } else if (a == "--fixture") {
      o.fixture = v;
    } else if (a == "--arm") {
      if (v != "G" && v != "O") {
        return Error("--arm is G or O");
      }
      o.arm = v == "G" ? model::Exl3Arm::kG : model::Exl3Arm::kO;
      arm = true;
    } else if (a == "--plan") {
      o.plan = v;
    } else if (a == "--ids") {
      o.ids = v;
    } else if (a == "--out") {
      o.out = v;
    } else if (a == "--prefixes") {
      o.prefixes.clear();
      std::istringstream list{std::string(v)};
      for (std::string item; std::getline(list, item, ',');) {
        int prefix = 0;
        const auto [end, error] = std::from_chars(item.data(), item.data() + item.size(), prefix);
        if (error != std::errc() || end != item.data() + item.size() || prefix <= 0) {
          return Error("--prefixes takes positive integers");
        }
        o.prefixes.push_back(prefix);
      }
    } else if (a == "--restores") {
      if (std::from_chars(v.data(), v.data() + v.size(), o.restores).ec != std::errc{} ||
          o.restores < 0 || o.restores > 8) {
        return Error("--restores takes a count up to 8");
      }
    } else if (a == "--lanes") {
      if (v != "inline" && v != "threads") {
        return Error("--lanes is inline or threads");
      }
      o.inline_lanes = v == "inline";
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.artifact.empty() || (o.fixture != "4.0bpw" && o.fixture != "4.5bpw") || !arm ||
      o.plan.empty() || o.ids.empty() || o.out.empty() || o.prefixes.empty() ||
      (o.record && !o.inline_lanes) || (o.cancel && o.inline_lanes)) {
    return Error(
        "usage: jitllm_exl3_paged --artifact DIR --fixture 4.0bpw|4.5bpw --arm G|O --plan PLAN.txt "
        "--ids FILE --out DIR [--prefixes LIST] [--restores N] [--relocate] [--partial] "
        "[--spill premapped|managed] [--cancel-in-flight] [--lanes threads|inline] "
        "[--record (with --lanes inline)]");
  }
  return o;
}

// ------------------------------------------------------------------ memory

struct Span {
  std::uint64_t base = 0;
  std::uint64_t size = 0;
  ExtentId extent;
  MemoryClass memory_class = MemoryClass::kUnknown;
  bool device = true;
};

struct Mapped {
  std::string name;
  ReservationId reservation;
  std::uint64_t base = 0;
  std::uint64_t bytes = 0;  // rounded to extents
  std::vector<jitllm::providers::BackingId> backings;
  std::vector<ExtentId> extents;
};

struct Coverage {
  std::uint64_t ranges = 0;
  std::array<std::uint64_t, jitllm::catalog::kMemoryClassCount> by_class{};
  std::uint64_t violations = 0;
  std::string first;
};

struct LoadStats {
  std::string what;
  std::uint64_t extents = 0;
  double seconds = 0;
};

// ------------------------------------------------------------------ programs

struct Done {
  std::atomic<int> outcome{-1};
  std::atomic<bool> retired{false};
  std::atomic<int> error{-1};
  std::atomic<bool> gone{false};
};

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
        return sc::Step::Yield();
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

// Runs a call on the scheduler thread (registering sources anew).
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
          return sc::Step::Yield();
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

// A phase kind's guaranteed bound against its observed peak (bytes).
struct Peak {
  std::uint64_t phases = 0;
  std::uint64_t region_bound = 0;
  std::uint64_t region_seen = 0;
  std::uint64_t pool_bound = 0;
  std::uint64_t pool_seen = 0;
};

struct PagingEvent {
  std::string what;
  std::uint64_t extents = 0;
  double seconds = 0;
  std::string detail;
};

struct PhaseRun {
  int prefix = 0;
  int index = 0;
  model::Exl3PhasePlan plan;
};

class Harness {
 public:
  explicit Harness(const Options& options) : o_(options) {}
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
  Status Teardown();

 private:
  Status Setup();
  Status MapResident(Mapped& mapped, std::string name, std::uint64_t bytes, BackingKind kind,
                     MemoryClass memory_class, Recovery recovery);
  std::expected<void*, std::string> Pinned(std::uint64_t bytes, std::vector<ExtentId>& staging);
  Status RegisterWeights();
  Status Place(std::size_t which);
  std::uint64_t WeightAddress(std::uint32_t resource) const;
  exl3::Qwen2Linear Linear(const model::Exl3LinearBinding& l) const;
  void AddSpan(std::uint64_t base, std::uint64_t size, ExtentId extent, MemoryClass memory_class,
               bool device);
  std::optional<MemoryClass> Covered(std::uint64_t address, std::uint64_t bytes) const;
  void Expect(std::string_view what, std::uint64_t address, std::uint64_t bytes,
              MemoryClass memory_class);
  void Check(const exl3::Qwen2Program& program, const PhaseRun& phase);
  Status Derive();

  void Round();
  Status Await(Done& done, std::string_view what, std::uint64_t request);
  Status Post(std::unique_ptr<sc::TaskProgram> program, Done& done, std::string_view what);
  Status Load(std::vector<ExtentId> extents, std::string what);
  Status Evict(std::vector<ExtentId> extents);
  Status Job(const jitllm::catalog::Closure& closure, sc::DeviceJob job, std::string_view what);
  Status Evaluate(int evaluation, std::map<int, std::vector<float>>& result);
  Status RunPhase(const PhaseRun& phase, int evaluation, std::vector<float>& out);
  Status Restore(int evaluation);
  Status Write(const std::vector<std::map<int, std::vector<float>>>& results);
  // BP-P2, BP-P4 and BP-P3's helpers.
  Status Partial();
  Status Spill(std::string_view where);
  Status Snapshot(std::vector<std::byte>& out);
  Status RegisterCache();
  Status HeadIsItsOwn() const;
  Status CancelInFlight();

  const Options& o_;
  const model::Qwen2Profile& profile_ = model::Qwen25Instruct05BExl3();
  std::unique_ptr<jitllm::artifact::Artifact> artifact_;
  model::Exl3Binding binding_;
  std::optional<model::Exl3LaunchTable> table_;
  std::vector<std::int32_t> ids_;
  std::vector<jitllm::artifact::FileDescriptor> shards_;
  std::vector<PhaseRun> phases_;

  std::unique_ptr<jitllm::providers::VmmProvider> memory_;
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  std::unique_ptr<jitllm::providers::UringStorage> storage_;
  std::array<jitllm::providers::StreamId, 2> streams_{};  // compute, copy
  std::size_t device_class_ = 0;
  std::size_t host_class_ = 0;

  jitllm::catalog::Catalog catalog_;
  jitllm::catalog::DomainId domain_;
  std::vector<Span> spans_;

  std::array<ReservationId, 2> weights_{};
  std::array<std::uint64_t, 2> weights_base_{};
  std::uint64_t weights_bytes_ = 0;
  std::size_t place_ = 0;
  std::vector<std::uint64_t> group_region_;
  std::vector<std::vector<ExtentId>> chunk_extents_;
  std::vector<ExtentId> device_weights_;
  std::uint64_t stored_bytes_ = 0;

  Mapped zone_;
  Mapped kv_;
  Mapped region_;
  Mapped pool_;
  Mapped locks_;
  Mapped norms_;
  Mapped tables_;
  void* inputs_ = nullptr;
  void* logits_ = nullptr;
  void* derive_ = nullptr;  // the derivation's staging
  std::vector<void*> pinned_;
  std::uint64_t inputs_bytes_ = 0;
  std::uint64_t logits_bytes_ = 0;
  std::uint64_t region_bytes_ = 0;
  std::uint64_t pool_bytes_ = 0;
  std::uint64_t kv_bytes_ = 0;

  jitllm::base::WakeFlag wake_;
  std::unique_ptr<sc::CompletionBoard> board_;
  std::unique_ptr<sc::StorageService> storage_lane_;
  std::unique_ptr<sc::DeviceService> device_lane_;
  std::unique_ptr<sc::Scheduler> scheduler_;
  std::vector<std::jthread> threads_;
  std::optional<std::expected<void, sc::Fault>> stopped_;
  std::uint64_t request_ = 0;

  std::unique_ptr<kg::LaunchContext> ggml_;
  std::unique_ptr<exl3::LaunchContext> launch_;
  std::unique_ptr<exl3::ReconGemm> gemm_;
  std::unique_ptr<jitllm::execution::Registry> registry_;
  exl3::Qwen2Memory memory_map_;
  jitllm::catalog::Closure everything_;
  jitllm::catalog::Closure cache_;

  std::unique_ptr<jitllm::test_support::Recording> recording_;
  std::string record_;
  Coverage coverage_;
  std::vector<LoadStats> loads_;
  std::map<std::string, std::string> identities_;  // "prefix/phase" -> plan identity
  bool torn_down_ = false;

  // --spill: the unnamed spill file, and a pinned copy of the cache.
  int spill_fd_ = -1;
  void* kv_copy_ = nullptr;
  std::vector<PagingEvent> events_;
  std::uint64_t kv_mismatches_ = 0;
  std::uint64_t refusals_ = 0;                 // incomplete closures refused before launch
  std::map<std::pair<int, int>, Peak> peaks_;  // by (rows, padded K)
  std::string cancel_result_;                  // --cancel-in-flight's, as JSON
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
  mapped.bytes = RoundUp(std::max<std::uint64_t>(bytes, 1), kExtent);
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

std::expected<void*, std::string> Harness::Pinned(std::uint64_t bytes,
                                                  std::vector<ExtentId>& staging) {
  void* pointer = nullptr;
  if (auto r = Cuda(cudaMallocHost(&pointer, std::max<std::uint64_t>(bytes, 256)), "pinned memory");
      !r) {
    return std::unexpected(r.error());
  }
  pinned_.push_back(pointer);
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
  return pointer;
}

std::uint64_t Harness::WeightAddress(std::uint32_t resource) const {
  const auto& r = artifact_->resources()[resource];
  return weights_base_.at(place_) + group_region_[r.group] + r.offset.value();
}

exl3::Qwen2Linear Harness::Linear(const model::Exl3LinearBinding& l) const {
  return {.weights = {.trellis = WeightAddress(l.trellis),
                      .suh = WeightAddress(l.suh),
                      .svh = WeightAddress(l.svh),
                      .k = l.k,
                      .n = l.n,
                      .bits = l.bits},
          .bias = l.bias ? WeightAddress(*l.bias) : 0};
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
    auto reservation = memory_->Reserve(Bytes(weights_bytes_));
    if (!reservation) {
      return Error(std::format("reserving the weights: {}", reservation.error().detail));
    }
    weights_.at(which) = *reservation;
    weights_base_.at(which) = memory_->RangeOf(*reservation).value().base;
  }
  std::array<std::uint8_t, 32> id{};
  for (std::size_t i = 0; i < id.size() && (2 * i) + 1 < artifact_->id().size(); ++i) {
    (void)std::from_chars(artifact_->id().data() + (2 * i), artifact_->id().data() + (2 * i) + 2,
                          id.at(i), 16);
  }
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
  return Place(0);
}

// Registers every weight chunk's source at place `which`.
Status Harness::Place(std::size_t which) {
  const auto groups = artifact_->groups();
  std::erase_if(spans_, [&](const Span& span) {
    return span.memory_class == MemoryClass::kWeights && span.device &&
           std::ranges::find(device_weights_, span.extent) != device_weights_.end();
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
      auto set = scheduler_->SetSource(
          chunk_extents_[g][c],
          sc::PageSource{.read = {.fd = shards_.at(range->shard).get(),
                                  .offset = range->file_offset.value(),
                                  .memory = nullptr,
                                  .length = range->length.value()},
                         .landed = true,
                         .destination = address,
                         .backing = sc::BackingPlace{.reservation = weights_.at(which),
                                                     .offset = Bytes(offset),
                                                     .size = Bytes(kExtent),
                                                     .allocation_class = device_class_}});
      if (!set) {
        return Error(std::format("a chunk's source: {}", sc::ToString(set.error())));
      }
      AddSpan(address, kExtent, chunk_extents_[g][c], MemoryClass::kWeights, true);
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

void Harness::Expect(std::string_view what, std::uint64_t address, std::uint64_t bytes,
                     MemoryClass memory_class) {
  ++coverage_.ranges;
  const std::optional<MemoryClass> covered = Covered(address, bytes);
  if (covered) {
    ++coverage_.by_class.at(static_cast<std::size_t>(*covered));
  }
  if (covered != memory_class && coverage_.violations++ == 0) {
    coverage_.first = std::format("{} ({} bytes at {:#x})", what, bytes, address);
  }
}

// BP-A1's in-process check: every range the bound program reads or writes
// lies in cataloged, resident extents of device memory of its class.
void Harness::Check(const exl3::Qwen2Program& program, const PhaseRun& phase) {
  const model::Exl3PhasePlan& plan = phase.plan;
  for (std::size_t op = 0; op < plan.ops.size(); ++op) {
    const model::Exl3Op& o = plan.ops[op];
    for (const auto& names : {o.inputs, o.outputs}) {
      for (const std::string& name : names) {
        if (name == "k_cache" || name == "v_cache") {
          const std::uint64_t cells = o.name.starts_with("kv_write")
                                          ? static_cast<std::uint64_t>(plan.phase.rows)
                                          : static_cast<std::uint64_t>(plan.padded);
          Expect(name, program.Address(op, name), cells * profile_.kv_width() * 2,
                 MemoryClass::kLiveState);
        } else if (const auto t = plan.tensors.find(name); t != plan.tensors.end()) {
          Expect(name, program.Address(op, name), t->second.bytes(), MemoryClass::kScratch);
        }
      }
    }
  }
  const auto weights = [&](const exl3::Qwen2Linear& l) {
    Expect("trellis", l.weights.trellis, exl3::TrellisBytes(l.weights), MemoryClass::kWeights);
    Expect("suh", l.weights.suh, static_cast<std::uint64_t>(l.weights.k) * 2,
           MemoryClass::kWeights);
    Expect("svh", l.weights.svh, static_cast<std::uint64_t>(l.weights.n) * 2,
           MemoryClass::kWeights);
    if (l.bias != 0) {
      Expect("bias", l.bias, static_cast<std::uint64_t>(l.weights.n) * 2, MemoryClass::kWeights);
    }
  };
  Expect("embed", memory_map_.embed, std::uint64_t{profile_.vocab} * profile_.width * 2,
         MemoryClass::kWeights);
  Expect("final norm", memory_map_.final_norm, std::uint64_t{profile_.width} * 4,
         MemoryClass::kWeights);
  weights(memory_map_.lm_head);
  for (const exl3::Qwen2Layer& layer : memory_map_.layers) {
    for (const exl3::Qwen2Linear* l :
         {&layer.q, &layer.k, &layer.v, &layer.o, &layer.gate, &layer.up, &layer.down}) {
      weights(*l);
    }
    Expect("norm", layer.attn_norm, std::uint64_t{profile_.width} * 4, MemoryClass::kWeights);
    Expect("norm", layer.mlp_norm, std::uint64_t{profile_.width} * 4, MemoryClass::kWeights);
    Expect("tables", layer.trellis_table, 48, MemoryClass::kRuntime);
  }
}

// ------------------------------------------------------------------ driving

void Harness::Round() {
  (void)storage_lane_->Turn(false);
  (void)device_lane_->SubmissionTurn();
  (void)device_lane_->CompletionTurn();
  (void)scheduler_->Turn();
}

Status Harness::Await(Done& done, std::string_view what, std::uint64_t request) {
  auto give_up = std::chrono::steady_clock::now() + kPatience;
  bool cancelled = false;
  while (!done.gone.load()) {
    if (std::chrono::steady_clock::now() > give_up) {
      if (cancelled) {
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
      Round();
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
  LoadStats stats{.what = std::move(what), .extents = 0, .seconds = 0};
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

  auto ids = jitllm::benchmarks::LoadIds(o_.ids);
  if (!ids) {
    return std::unexpected(ids.error());
  }
  ids_ = std::move(*ids);
  auto artifact = jitllm::artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<jitllm::artifact::Artifact>(std::move(*artifact));
  auto binding = model::BindQwen2Exl3(profile_, *artifact_);
  if (!binding) {
    return Error("the artifact does not bind: " + binding.error());
  }
  binding_ = std::move(*binding);
  auto table = jitllm::benchmarks::LoadTable(o_.plan, binding_);
  if (!table) {
    return std::unexpected(table.error());
  }
  table_.emplace(std::move(*table));
  for (std::uint32_t s = 0; s < artifact_->shards().size(); ++s) {
    auto fd = artifact_->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards_.push_back(std::move(*fd));
  }

  // Every phase, planned as rung 3 plans it.
  for (const int prefix : o_.prefixes) {
    if (prefix + kSuffix > static_cast<int>(ids_.size())) {
      return Error(std::format("prefix {} and its steps exceed the held-out IDs", prefix));
    }
    for (int index = 0; index <= kSuffix; ++index) {
      const model::Exl3Phase phase = index == 0
                                         ? model::Exl3Phase{.rows = prefix, .past = 0}
                                         : model::Exl3Phase{.rows = 1, .past = prefix + index - 1};
      auto plan = model::PlanPhase(profile_, binding_, *table_, o_.arm, phase);
      if (!plan) {
        return Error(std::format("prefix {} phase {}: {}", prefix, index, plan.error()));
      }
      region_bytes_ = std::max(region_bytes_, plan->region);
      inputs_bytes_ = std::max(inputs_bytes_, exl3::HostInputsLayout(*plan).bytes);
      logits_bytes_ = std::max(logits_bytes_, plan->tensors.at("logits").bytes());
      phases_.push_back({.prefix = prefix, .index = index, .plan = std::move(*plan)});
    }
  }

  domain_ = catalog_.AddDomain("gb10");
  if (auto zone = MapResident(zone_, "the landing zone", kSlots * kExtent, BackingKind::kHost,
                              MemoryClass::kStaging, Recovery::kPinned);
      !zone) {
    return zone;
  }
  kv_bytes_ = std::uint64_t{profile_.layers} * 2 * kCells * profile_.kv_width() * 2;
  const std::uint64_t norm_bytes = std::uint64_t{profile_.width} * 4;
  const std::uint64_t norms_bytes = norm_bytes * (2 * std::uint64_t{profile_.layers} + 1);
  for (const auto& [mapped, name, bytes, cls, recovery] :
       {std::tuple{&kv_, "the cache", kv_bytes_, MemoryClass::kLiveState, Recovery::kPreserve},
        std::tuple{&region_, "the activations", region_bytes_, MemoryClass::kScratch,
                   Recovery::kDiscardable},
        std::tuple{&locks_, "the lock area", exl3::kLockBytes, MemoryClass::kRuntime,
                   Recovery::kPinned},
        std::tuple{&norms_, "the F32 norms", norms_bytes, MemoryClass::kWeights, Recovery::kPinned},
        std::tuple{&tables_, "the multi-GEMM tables", std::uint64_t{48} * profile_.layers,
                   MemoryClass::kRuntime, Recovery::kPinned}}) {
    if (auto r = MapResident(*mapped, name, bytes, BackingKind::kDevice, cls, recovery); !r) {
      return r;
    }
  }
  std::vector<ExtentId> staging;
  auto inputs = Pinned(inputs_bytes_, staging);
  auto logits = Pinned(logits_bytes_, staging);
  auto derive = Pinned(norms_bytes + std::uint64_t{48} * profile_.layers, staging);
  if (!inputs || !logits || !derive) {
    return std::unexpected(!inputs ? inputs.error() : !logits ? logits.error() : derive.error());
  }
  inputs_ = *inputs;
  logits_ = *logits;
  derive_ = *derive;

  // The registry, the reconstruction GEMM, and the GGML pool's size: every
  // phase bound on a planning context (no workspace), for the largest
  // attention scratch.
  std::vector<jitllm::execution::Implementation> implementations = kg::Implementations();
  for (auto& implementation : exl3::Implementations()) {
    implementations.push_back(std::move(implementation));
  }
  auto registry = jitllm::execution::Registry::Create(std::move(implementations));
  if (!registry) {
    return Error("the registry was refused: " + registry.error().detail);
  }
  registry_ = std::make_unique<jitllm::execution::Registry>(std::move(*registry));
  auto gemm = exl3::ReconGemm::Create();
  if (!gemm) {
    return Error("the reconstruction GEMM was refused: " + gemm.error().detail);
  }
  gemm_ = std::move(*gemm);

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
      memory_.get());
  sc::LandingZone landing{.slots = {}, .slot_bytes = Bytes(kExtent), .stream = 1};
  for (std::size_t i = 0; i < kSlots; ++i) {
    landing.slots.push_back(zone_.base + (i * kExtent));
  }
  scheduler_ = std::make_unique<sc::Scheduler>(
      catalog_, *board_, wake_,
      sc::Lanes{.storage = storage_lane_.get(), .device = device_lane_.get(), .cpu = nullptr},
      sc::SchedulerSettings{
          .tasks = 16, .budget = Bytes(std::uint64_t{64} << 30U), .landing = landing});
  if (auto r = RegisterWeights(); !r) {
    return r;
  }
  if (auto r = HeadIsItsOwn(); !r) {
    return r;
  }
  if (!o_.spill.empty()) {
    if (auto r = RegisterCache(); !r) {
      return r;
    }
  }
  if (!o_.inline_lanes) {
    threads_.emplace_back([this] { stopped_ = scheduler_->Run(); });
    threads_.emplace_back([this] { storage_lane_->Run(); });
    threads_.emplace_back([this] { device_lane_->RunSubmission(); });
    threads_.emplace_back([this] { device_lane_->RunCompletion(); });
  }
  if (auto r = Load(device_weights_, "initial"); !r) {
    return r;
  }

  // The memory map at place 0 (weights' addresses), then the pool: bind
  // every phase on a planning context for the largest attention scratch.
  memory_map_.embed = WeightAddress(binding_.embed);
  memory_map_.cells = kCells;
  memory_map_.region = region_.base;
  memory_map_.region_bytes = region_bytes_;
  memory_map_.final_norm = norms_.base;
  memory_map_.lm_head = Linear(binding_.lm_head);
  const std::uint64_t per_layer = std::uint64_t{2} * kCells * profile_.kv_width() * 2;
  for (std::uint32_t l = 0; l < profile_.layers; ++l) {
    const model::Exl3LayerBinding& b = binding_.layers[l];
    exl3::Qwen2Layer layer;
    layer.attn_norm = norms_.base + (norm_bytes * (1 + (2 * std::uint64_t{l})));
    layer.mlp_norm = layer.attn_norm + norm_bytes;
    layer.q = Linear(b.q);
    layer.k = Linear(b.k);
    layer.v = Linear(b.v);
    layer.o = Linear(b.o);
    layer.gate = Linear(b.gate);
    layer.up = Linear(b.up);
    layer.down = Linear(b.down);
    layer.trellis_table = tables_.base + (std::uint64_t{48} * l);
    layer.suh_table = layer.trellis_table + 16;
    layer.svh_table = layer.trellis_table + 32;
    layer.k_cache = kv_.base + (per_layer * l);
    layer.v_cache = layer.k_cache + (per_layer / 2);
    memory_map_.layers.push_back(layer);
  }
  // The EXL3 launch context, made in a job: it queues its lock area's
  // zeroing on the compute stream.
  {
    const auto locks = catalog_.ClosureOfExtents(locks_.extents).value();
    Status made;
    if (auto created = Job(
            locks,
            [this, &made](jitllm::providers::NativeStream) {
              auto launch = exl3::LaunchContext::Create(0, *execution_, streams_[0], locks_.base);
              if (!launch) {
                made = Error("the EXL3 launch context was refused: " + launch.error().detail);
                return sc::JobResult::kNotStarted;
              }
              launch_ = std::move(*launch);
              return sc::JobResult::kQueued;
            },
            "creating the EXL3 launch context");
        !created || !made) {
      return !made ? made : created;
    }
  }
  // The GGML pool's size: every phase bound on a planning context (no
  // workspace; it queues nothing), for the largest attention scratch.
  {
    auto planning =
        kg::LaunchContext::Create(0, *execution_, streams_[0], {.base = 0, .size = Bytes(0)});
    if (!planning) {
      return Error(planning.error().detail);
    }
    // The tables are written only by Derive, below: planning binds a copy
    // that records them as they will be written, and runs nothing.
    exl3::Qwen2Memory planned = memory_map_;
    for (exl3::Qwen2Layer& layer : planned.layers) {
      layer.tables_written = exl3::MultiGemmTables(layer.gate.weights, layer.up.weights);
    }
    for (const PhaseRun& phase : phases_) {
      auto program =
          exl3::Qwen2Program::Bind(*registry_, profile_, phase.plan, planned, **planning, *launch_);
      if (!program) {
        return Error(std::format("prefix {} phase {} did not bind: {}", phase.prefix, phase.index,
                                 program.error().detail));
      }
      pool_bytes_ = std::max(pool_bytes_, (*program)->ggml_scratch());
    }
  }
  if (auto r = MapResident(pool_, "the GGML pool", pool_bytes_, BackingKind::kDevice,
                           MemoryClass::kScratch, Recovery::kDiscardable);
      !r) {
    return r;
  }
  std::vector<ExtentId> all = device_weights_;
  for (const Mapped* mapped : {&kv_, &region_, &pool_, &locks_, &norms_, &tables_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging.begin(), staging.end());
  everything_ = catalog_.ClosureOfExtents(all).value();
  cache_ = catalog_.ClosureOfExtents(kv_.extents).value();
  auto ggml = kg::LaunchContext::Create(0, *execution_, streams_[0],
                                        {.base = pool_.base, .size = Bytes(pool_bytes_)});
  if (!ggml) {
    return Error("the GGML launch context was refused: " + ggml.error().detail);
  }
  ggml_ = std::move(*ggml);
  return Derive();
}

// The norms widened to F32 from the device weights, and the tables of the
// weights' current place: copied down to staging, widened on the host, and
// copied up, in two jobs.
Status Harness::Derive() {
  const std::uint64_t norm_bytes = std::uint64_t{profile_.width} * 4;
  const std::uint64_t half = std::uint64_t{profile_.width} * 2;
  std::vector<std::uint32_t> sources = {binding_.final_norm};
  for (const model::Exl3LayerBinding& b : binding_.layers) {
    sources.push_back(b.attn_norm);
    sources.push_back(b.mlp_norm);
  }
  auto* staged = static_cast<std::byte*>(derive_);
  Status copied;
  if (auto r = Job(
          everything_,
          [&](jitllm::providers::NativeStream) {
            for (std::size_t i = 0; i < sources.size(); ++i) {
              if (auto c = execution_->Copy(streams_[0], Address(staged) + (i * half),
                                            WeightAddress(sources[i]), Bytes(half));
                  !c) {
                copied = Error(c.error().detail);
                return i == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kFailed;
              }
            }
            return sc::JobResult::kQueued;
          },
          "copying the norms down");
      !r || !copied) {
    return !copied ? copied : r;
  }
  // Widen in place, from the back (F32 is twice F16's size).
  const std::size_t count = sources.size() * profile_.width;
  std::vector<std::uint16_t> bf16(count);
  std::memcpy(bf16.data(), staged, count * 2);
  for (std::size_t i = 0; i < count; ++i) {
    const float value = Bf16ToFloat(bf16[i]);
    std::memcpy(staged + (i * 4), &value, 4);
  }
  if (auto r = Job(
          everything_,
          [&](jitllm::providers::NativeStream) {
            if (auto c = execution_->Copy(streams_[0], norms_.base, Address(staged),
                                          Bytes(sources.size() * norm_bytes));
                !c) {
              copied = Error(c.error().detail);
              return sc::JobResult::kNotStarted;
            }
            // Each layer's record is set only once its tables' copies are
            // queued: after a failure here, Bind refuses a stale table.
            if (auto c =
                    exl3::WriteMultiGemmTables(*execution_, streams_[0],
                                               std::span(staged + (sources.size() * norm_bytes),
                                                         std::uint64_t{48} * profile_.layers),
                                               memory_map_);
                !c) {
              copied = Error(c.error().detail);
              return sc::JobResult::kFailed;
            }
            return sc::JobResult::kQueued;
          },
          "uploading the norms and tables");
      !r || !copied) {
    return !copied ? copied : r;
  }
  return {};
}

// ------------------------------------------------------------------ evaluating

Status Harness::RunPhase(const PhaseRun& phase, int evaluation, std::vector<float>& out) {
  auto program =
      exl3::Qwen2Program::Bind(*registry_, profile_, phase.plan, memory_map_, *ggml_, *launch_);
  if (!program) {
    return Error(std::format("prefix {} phase {} did not bind: {}", phase.prefix, phase.index,
                             program.error().detail));
  }
  identities_[std::format("{}/{}/{}", evaluation, phase.prefix, phase.index)] =
      jitllm::base::ToHex((*program)->identity());
  Check(**program, phase);
  const model::Exl3PhasePlan& plan = phase.plan;
  // The phase kind's guaranteed bound, and the highest region byte its
  // bound operations reach.
  Peak& peak = peaks_[{plan.phase.rows, plan.padded}];
  ++peak.phases;
  peak.region_bound = std::max(peak.region_bound, plan.region);
  peak.pool_bound = std::max(peak.pool_bound, (*program)->ggml_scratch());
  for (std::size_t op = 0; op < plan.ops.size(); ++op) {
    for (const auto& names : {plan.ops[op].inputs, plan.ops[op].outputs}) {
      for (const std::string& name : names) {
        const auto t = plan.tensors.find(name);
        const std::uint64_t at = (*program)->Address(op, name);
        if (t != plan.tensors.end() && at >= region_.base && at < region_.base + region_.bytes) {
          peak.region_seen = std::max(peak.region_seen, at + t->second.bytes() - region_.base);
        }
      }
    }
  }
  ggml_->ResetScratchPeak();
  const std::span<const std::int32_t> tokens(ids_.data() + plan.phase.past,
                                             static_cast<std::size_t>(plan.phase.rows));
  if (auto written = exl3::WriteHostInputs(
          profile_, plan, tokens,
          std::span(static_cast<std::byte*>(inputs_), exl3::HostInputsLayout(plan).bytes));
      !written) {
    return Error(written.error().detail);
  }
  const std::uint64_t logits_bytes = plan.tensors.at("logits").bytes();
  const std::uint64_t logits_at = region_.base + plan.slots.at("logits").offset;
  Status ran;
  const bool record = recording_ != nullptr && evaluation == 1;
  auto job = [&](jitllm::providers::NativeStream native) -> sc::JobResult {
    if (record) {
      for (const auto& event : recording_->Take()) {
        record_ += jitllm::test_support::EventLine(event);
      }
      record_ += jitllm::test_support::ChunkLine({.evaluation = evaluation,
                                                  .chunk = phase.index,
                                                  .rows = plan.phase.rows,
                                                  .n_past = plan.phase.past});
    }
    exl3::Qwen2Hooks hooks;
    if (record) {
      hooks.before = [&](std::size_t op) -> std::expected<void, exl3::KernelFailure> {
        for (const auto& event : recording_->Take()) {
          record_ += jitllm::test_support::EventLine(event);
        }
        record_ += jitllm::test_support::OpLine(plan.ops[op].name, plan.ops[op].layer);
        return {};
      };
    }
    if (auto r = (*program)->Run(*ggml_, *launch_, *gemm_, *execution_, streams_[0],
                                 Address(inputs_), hooks);
        !r) {
      ran =
          Error(std::format("prefix {} phase {}: {}", phase.prefix, phase.index, r.error().detail));
      return r.error().error == exl3::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                            : sc::JobResult::kFailed;
    }
    if (record) {
      for (const auto& event : recording_->Take()) {
        record_ += jitllm::test_support::EventLine(event);
      }
      record_ += jitllm::test_support::OpLine("outputs", -1);
    }
    if (auto r =
            Cuda(cudaMemcpyAsync(logits_, Pointer(logits_at), logits_bytes, cudaMemcpyDeviceToHost,
                                 static_cast<cudaStream_t>(native.handle)),
                 "the logits copy");
        !r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    if (record) {
      for (const auto& event : recording_->Take()) {
        record_ += jitllm::test_support::EventLine(event);
      }
      record_ += jitllm::test_support::EndChunkLine();
    }
    return sc::JobResult::kQueued;
  };
  if (auto r = Job(everything_, std::move(job), "a phase"); !r || !ran) {
    return !ran
               ? ran
               : Error(std::format("prefix {} phase {}: {}", phase.prefix, phase.index, r.error()));
  }
  if (launch_->faulted() || ggml_->faulted()) {
    return Error(
        std::format("prefix {} phase {}: a launch context faulted", phase.prefix, phase.index));
  }
  peak.pool_seen = std::max(peak.pool_seen, ggml_->scratch_peak().value());
  const auto* half = static_cast<const std::uint16_t*>(logits_);
  for (std::size_t i = 0; i < logits_bytes / 2; ++i) {
    out.push_back(model::HalfToFloat(half[i]));
  }
  return {};
}

Status Harness::Evaluate(int evaluation, std::map<int, std::vector<float>>& result) {
  for (const PhaseRun& phase : phases_) {
    if (phase.index == 0) {
      const std::uint64_t kv = kv_.base;
      const std::uint64_t bytes = kv_bytes_;
      if (auto cleared = Job(
              cache_,
              [kv, bytes](jitllm::providers::NativeStream stream) {
                return cudaMemsetAsync(Pointer(kv), 0, bytes,
                                       static_cast<cudaStream_t>(stream.handle)) == cudaSuccess
                           ? sc::JobResult::kQueued
                           : sc::JobResult::kUnknown;
              },
              "clearing the cache");
          !cleared) {
        return cleared;
      }
      result[phase.prefix].clear();
    }
    if (auto r = RunPhase(phase, evaluation, result[phase.prefix]); !r) {
      return r;
    }
    const int partial_at = o_.partial ? 3 + o_.restores : -1;
    const int spill_at = o_.spill.empty() ? -1 : 3 + o_.restores + (o_.partial ? 1 : 0);
    if (evaluation > 2 && evaluation <= 2 + o_.restores && phase.index == 0) {
      if (auto r = Restore(evaluation); !r) {
        return r;
      }
    }
    if (evaluation == partial_at && phase.index == 0 && phase.prefix == phases_.front().prefix) {
      if (auto r = Partial(); !r) {
        return r;
      }
    }
    if (evaluation == spill_at && (phase.index == 0 || phase.index == 8)) {
      if (auto r = Spill(std::format("prefix {} {}", phase.prefix,
                                     phase.index == 0 ? "after the prefill" : "mid-decode"));
          !r) {
        return r;
      }
    }
  }
  return {};
}

// BP-P2: each partial eviction in turn.
Status Harness::Partial() {
  const std::string layer = std::format("model.layers.{}.", profile_.layers / 2);
  const auto cases = jitllm::benchmarks::PartialCases(*artifact_, layer, true);
  for (const auto& partial : cases) {
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

Status Harness::Snapshot(std::vector<std::byte>& out) {
  const std::uint64_t kv = kv_.base;
  const std::uint64_t bytes = kv_bytes_;
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

// --spill: the cache's write-back places (fp16_paged.cc's).
Status Harness::RegisterCache() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(), std::strerror(errno)));
  }
  if (auto r = Cuda(cudaMallocHost(&kv_copy_, kv_bytes_), "the cache's host copy"); !r) {
    return r;
  }
  pinned_.push_back(kv_copy_);
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
    kv_.backings.clear();  // the device lane releases them on eviction (D-033)
  }
  return {};
}

// BP-P3's EXL3 half: the head is its own representation, never the
// embedding's storage.
Status Harness::HeadIsItsOwn() const {
  const auto embed = artifact_->ResourcePlacement(binding_.embed);
  const auto head = artifact_->ResourcePlacement(binding_.lm_head.trellis);
  if (!embed || !head) {
    return Error("the embedding or the head has no placement");
  }
  const auto& resources = artifact_->resources();
  const bool shared_name =
      std::ranges::find(resources[binding_.embed].roles, binding_.lm_head.name + ".trellis") !=
      resources[binding_.embed].roles.end();
  const bool overlap = embed->group == head->group &&
                       embed->offset.value() < head->offset.value() + head->readable.value() &&
                       head->offset.value() < embed->offset.value() + embed->readable.value();
  if (binding_.embed == binding_.lm_head.trellis || shared_name || overlap) {
    return Error("the EXL3 head shares the embedding's storage");
  }
  return {};
}

// BP-L1 and BP-L3: a phase cancelled with its GGML and EXL3 work queued
// behind a gate keeps its lease until the fence after that work completes.
Status Harness::CancelInFlight() {
  const PhaseRun* chosen = nullptr;
  for (const PhaseRun& phase : phases_) {
    if (phase.index == 0 && (chosen == nullptr || phase.plan.region > chosen->plan.region)) {
      chosen = &phase;  // the largest reconstruction phase
    }
  }
  if (chosen == nullptr) {
    return Error("no prefill to cancel");
  }
  const model::Exl3PhasePlan& plan = chosen->plan;
  auto program =
      exl3::Qwen2Program::Bind(*registry_, profile_, plan, memory_map_, *ggml_, *launch_);
  if (!program) {
    return Error(std::format("the cancelled phase did not bind: {}", program.error().detail));
  }
  const std::span<const std::int32_t> tokens(ids_.data() + plan.phase.past,
                                             static_cast<std::size_t>(plan.phase.rows));
  if (auto written = exl3::WriteHostInputs(
          profile_, plan, tokens,
          std::span(static_cast<std::byte*>(inputs_), exl3::HostInputsLayout(plan).bytes));
      !written) {
    return Error(written.error().detail);
  }
  void* gate = nullptr;
  if (auto r = Cuda(cudaHostAlloc(&gate, sizeof(std::uint32_t), cudaHostAllocMapped), "the gate");
      !r) {
    return r;
  }
  pinned_.push_back(gate);
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(gate)).store(0);
  void* device_gate = nullptr;
  if (auto r = Cuda(cudaHostGetDevicePointer(&device_gate, gate, 0), "the gate's device address");
      !r) {
    return r;
  }
  std::atomic<bool> started{false};  // the gate is queued: what follows waits behind it
  std::atomic<bool> queued{false};   // the job has queued everything (more launches than
                                     // the stream's pending queue holds wait in the driver)
  Status ran;
  auto job = [&, device_gate](jitllm::providers::NativeStream native) -> sc::JobResult {
    if (cuStreamWaitValue32(static_cast<CUstream>(native.handle),
                            reinterpret_cast<CUdeviceptr>(device_gate), 1,
                            CU_STREAM_WAIT_VALUE_GEQ) != CUDA_SUCCESS) {
      ran = Error("cuStreamWaitValue32 was refused");
      return sc::JobResult::kNotStarted;
    }
    started.store(true);
    if (auto r = (*program)->Run(*ggml_, *launch_, *gemm_, *execution_, streams_[0],
                                 Address(inputs_), exl3::Qwen2Hooks{});
        !r) {
      ran = Error(r.error().detail);
      return r.error().error == exl3::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                            : sc::JobResult::kFailed;
    }
    queued.store(true);
    return sc::JobResult::kQueued;
  };
  Done done;
  const std::uint64_t request = ++request_;
  sc::Control start =
      sc::StartRequest{.request = request,
                       .priority = 1,
                       .program = std::make_unique<RunProgram>(done, everything_, std::move(job))};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(start)) == jitllm::base::PushResult::kFull) {
    std::this_thread::yield();
  }
  // Until the job has queued what it can; the lane may still be inside it,
  // its later launches waiting in the driver behind the gate.
  const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (!started.load() && !done.gone.load() && std::chrono::steady_clock::now() < give_up) {
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const bool was_started = started.load();
  const bool was_queued = queued.load();
  sc::Control cancel = sc::CancelRequest{.request = request};
  // NOLINTNEXTLINE(bugprone-use-after-move): Post moves only what it takes
  while (scheduler_->Post(std::move(cancel)) == jitllm::base::PushResult::kFull) {
    std::this_thread::yield();
  }
  // On the scheduler thread, with the gate still shut: what the lease holds.
  const auto held = [&] {
    std::vector<ExtentId> touched = region_.extents;
    touched.insert(touched.end(), kv_.extents.begin(), kv_.extents.end());
    touched.insert(touched.end(), device_weights_.begin(), device_weights_.end());
    std::size_t holding = 0;
    for (const ExtentId extent : touched) {
      const auto view = catalog_.Describe(extent).value();
      holding += view.leases > 0 && !jitllm::catalog::Catalog::Evictable(view) ? 1 : 0;
    }
    return holding == touched.size() ? touched.size() : 0;
  };
  std::size_t holding = 0;
  {
    // Not returned early on failure: the gate must open before this frame,
    // which the job refers to, can end (a failed probe leaves `holding` 0).
    Done probe;
    (void)Post(std::make_unique<CallProgram>(probe,
                                             [&]() -> Status {
                                               holding = held();
                                               return {};
                                             }),
               probe, "probing the cancelled phase's lease");
  }
  const bool retired_early = done.gone.load();
  std::atomic_ref<std::uint32_t>(*static_cast<std::uint32_t*>(gate)).store(1);
  (void)Await(done, "the cancelled phase", request);  // its outcome is checked below
  const bool cancelled = done.outcome.load() == static_cast<int>(sc::TaskOutcome::kCancelled);
  std::size_t after = 1;
  {
    Done probe;
    if (auto r = Post(
            std::make_unique<CallProgram>(probe,
                                          [&]() -> Status {
                                            after = 0;
                                            for (const ExtentId extent : region_.extents) {
                                              after += catalog_.Describe(extent).value().leases;
                                            }
                                            return {};
                                          }),
            probe, "probing after the fence");
        !r) {
      return r;
    }
  }
  cancel_result_ = std::format(
      R"({{"phase_rows":{},"gate_queued_before_cancel":{},"all_queued_before_cancel":{},)"
      R"("extents_held_while_gated":{},"retired_before_the_gate_opened":{},)"
      R"("outcome_cancelled":{},"leases_after_fence":{}}})",
      plan.phase.rows, was_started ? "true" : "false", was_queued ? "true" : "false", holding,
      retired_early ? "true" : "false", cancelled ? "true" : "false", after);
  if (!ran || !was_started || holding == 0 || retired_early || !cancelled || after != 0) {
    return Error(
        std::format("cancelling in flight: {}{}", cancel_result_, ran ? "" : ": " + ran.error()));
  }
  return {};
}

// Rung 5: every weight extent evicted, its backing released, then paged
// back in through the zone (with --relocate, every second time at the
// other place, and the tables rewritten for it).
Status Harness::Restore(int evaluation) {
  const std::size_t backings = memory_->backings();
  if (auto r = Evict(device_weights_); !r) {
    return r;
  }
  for (const ExtentId extent : device_weights_) {
    if (catalog_.Describe(extent).value().state != jitllm::catalog::ExtentState::kNonresident) {
      return Error("a weight extent is still resident after the eviction");
    }
  }
  if (memory_->backings() + device_weights_.size() != backings) {
    return Error("the eviction did not release every weight's backing");
  }
  const bool relocate = o_.relocate && evaluation % 2 == 0;
  if (relocate) {
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
  if (auto r = Load(device_weights_, std::format("restore {}", evaluation - 2)); !r) {
    return r;
  }
  // The weights' addresses: every linear and the embedding, and the tables.
  memory_map_.embed = WeightAddress(binding_.embed);
  memory_map_.lm_head = Linear(binding_.lm_head);
  for (std::uint32_t l = 0; l < profile_.layers; ++l) {
    const model::Exl3LayerBinding& b = binding_.layers[l];
    exl3::Qwen2Layer& layer = memory_map_.layers[l];
    layer.q = Linear(b.q);
    layer.k = Linear(b.k);
    layer.v = Linear(b.v);
    layer.o = Linear(b.o);
    layer.gate = Linear(b.gate);
    layer.up = Linear(b.up);
    layer.down = Linear(b.down);
  }
  return Derive();
}

Status Harness::Run() {
  if (auto r = Setup(); !r) {
    return r;
  }
  if (o_.record) {
    recording_ = std::make_unique<jitllm::test_support::Recording>();
    (void)recording_->Take();
  }
  const int evaluations = 2 + o_.restores + (o_.partial ? 1 : 0) + (o_.spill.empty() ? 0 : 1);
  std::vector<std::map<int, std::vector<float>>> results(static_cast<std::size_t>(evaluations));
  for (int e = 1; e <= evaluations; ++e) {
    if (auto r = Evaluate(e, results[static_cast<std::size_t>(e - 1)]); !r) {
      return r;
    }
    if (e == 1 && recording_) {
      for (const auto& event : recording_->Take()) {
        record_ += jitllm::test_support::EventLine(event);
      }
      recording_.reset();
    }
  }
  if (o_.cancel) {
    if (auto r = CancelInFlight(); !r) {
      return r;
    }
  }
  return Write(results);
}

Status Harness::Write(const std::vector<std::map<int, std::vector<float>>>& results) {
  std::filesystem::create_directories(o_.out);
  const auto& first = results.front();
  std::string differences;
  std::size_t differing_total = 0;
  for (std::size_t e = 1; e < results.size(); ++e) {
    std::size_t differing = 0;
    for (const auto& [prefix, values] : first) {
      const auto& other = results[e].at(prefix);
      differing += values.size() == other.size() ? 0 : values.size();
      for (std::size_t i = 0; i < std::min(values.size(), other.size()); ++i) {
        differing +=
            std::bit_cast<std::uint32_t>(values[i]) != std::bit_cast<std::uint32_t>(other[i]);
      }
    }
    differences += std::format("{}{}", differences.empty() ? "" : ",", differing);
    differing_total += differing;
  }
  std::string logits;
  const auto vocab = static_cast<std::int64_t>(profile_.vocab);
  for (const auto& [prefix, values] : first) {
    const std::size_t prefill = static_cast<std::size_t>(prefix) * profile_.vocab;
    const auto bytes = std::as_bytes(std::span(values));
    if (auto r = jitllm::benchmarks::WriteNpy(o_.out / std::format("logits-{}.prefill.npy", prefix),
                                              "<f4", {prefix, vocab}, bytes.first(prefill * 4));
        !r) {
      return r;
    }
    if (auto r = jitllm::benchmarks::WriteNpy(o_.out / std::format("logits-{}.suffix.npy", prefix),
                                              "<f4", {kSuffix, vocab}, bytes.subspan(prefill * 4));
        !r) {
      return r;
    }
    logits += std::format(R"({}"{}": {{"prefill_sha256": "{}", "suffix_sha256": "{}"}})",
                          logits.empty() ? "" : ", ", prefix, Hex(bytes.first(prefill * 4)),
                          Hex(bytes.subspan(prefill * 4)));
  }
  std::string loads;
  for (const LoadStats& load : loads_) {
    loads += std::format(R"({}{{"what": "{}", "extents": {}, "seconds": {:.6f}}})",
                         loads.empty() ? "" : ", ", load.what, load.extents, load.seconds);
  }
  std::string by_class;
  for (std::size_t c = 0; c < coverage_.by_class.size(); ++c) {
    by_class += std::format("{}{}", c == 0 ? "" : ", ", coverage_.by_class.at(c));
  }
  std::string identities;
  for (const auto& [key, identity] : identities_) {
    identities += std::format(R"({}"{}": "{}")", identities.empty() ? "" : ", ", key, identity);
  }
  std::ofstream summary(o_.out / "summary.json");
  summary << std::format(
      "{{\"harness\": \"jitllm_exl3_paged\", \"fixture\": \"{}\", \"arm\": \"{}\", \"artifact\": "
      "\"{}\", \"plan_file_sha256\": \"{}\", \"memory\": \"device VMM through the landing zone\", "
      "\"lanes\": \"{}\", \"evaluations\": {}, \"restores\": {}, \"relocate\": {},\n "
      "\"bit_differences_from_first\": [{}],\n \"logits\": {{{}}},\n \"weight_extents\": {}, "
      "\"stored_bytes\": {}, \"zone_bytes\": {}, \"kv_bytes\": {}, \"region_bytes\": {}, "
      "\"pool_bytes\": {}, \"loads\": [{}],\n \"coverage\": {{\"ranges\": {}, \"violations\": {}, "
      "\"first\": \"{}\", \"by_class\": [{}]}},\n \"plan_identities\": {{{}}}}}\n",
      o_.fixture, o_.arm == model::Exl3Arm::kG ? "G" : "O", artifact_->id(), HexFile(o_.plan),
      threads_.empty() ? "inline" : "threads", results.size(), o_.restores,
      o_.relocate ? "true" : "false", differences, logits, device_weights_.size(), stored_bytes_,
      zone_.bytes, kv_.bytes, region_.bytes, pool_.bytes, loads, coverage_.ranges,
      coverage_.violations, coverage_.first, by_class, identities);
  if (!record_.empty()) {
    std::ofstream file(o_.out / "record.jsonl");
    file << jitllm::test_support::HeaderLine(
                std::format("jitllm_exl3_paged {} {}", o_.fixture,
                            o_.arm == model::Exl3Arm::kG ? "EXL3-G" : "EXL3-O"),
                jitllm::test_support::LoadedCublas())
         << record_;
  }
  // BP-P2 and BP-P4's runs, and each phase kind's bound against its peak.
  std::string events;
  for (const PagingEvent& event : events_) {
    events += std::format(R"({}{{"what":"{}","extents":{},"seconds":{:.6f},"detail":"{}"}})",
                          events.empty() ? "" : ",", event.what, event.extents, event.seconds,
                          event.detail);
  }
  std::string peaks;
  bool within = true;
  for (const auto& [kind, peak] : peaks_) {
    within = within && peak.region_seen <= peak.region_bound && peak.pool_seen <= peak.pool_bound;
    peaks +=
        std::format(R"({}{{"rows":{},"padded":{},"phases":{},"region_bound":{},"region_seen":{},)"
                    R"("pool_bound":{},"pool_seen":{}}})",
                    peaks.empty() ? "" : ",", kind.first, kind.second, peak.phases,
                    peak.region_bound, peak.region_seen, peak.pool_bound, peak.pool_seen);
  }
  {
    std::ofstream file(o_.out / "paging.json");
    file << std::format(
                R"({{"partial":{},"spill":"{}","head_is_its_own":true,"refused_launches":{},)"
                R"("kv_bytes_differing":{},"cancel_in_flight":{},"events":[{}],"peaks":[{}]}})",
                o_.partial ? "true" : "false", o_.spill, refusals_, kv_mismatches_,
                cancel_result_.empty() ? std::string("null") : cancel_result_, events, peaks)
         << "\n";
  }
  std::println("wrote {}", o_.out.string());
  if (kv_mismatches_ != 0) {
    return Error(std::format("{} bytes of the cache differ after its restore", kv_mismatches_));
  }
  if (!within) {
    return Error("a phase reached past its guaranteed bound (paging.json)");
  }
  if (coverage_.violations != 0) {
    return Error(std::format("{} ranges lie outside cataloged extents of their class; first: {}",
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
    if ((!threads_.empty() || !scheduler_->fault()) &&
        !Job(
            cache_, [](jitllm::providers::NativeStream) { return sc::JobResult::kQueued; },
            "fencing the compute stream")) {
      problems.emplace_back("the compute stream could not be fenced");
    }
    std::vector<ExtentId> evicted = device_weights_;
    if (o_.spill == "managed") {
      // Its backing is the device lane's to release: written back and evicted.
      evicted.insert(evicted.end(), kv_.extents.begin(), kv_.extents.end());
    }
    if ((!threads_.empty() || !scheduler_->fault()) && !Evict(evicted)) {
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
    if (driven) {
      for (int i = 0; i < 1000 && (storage_->in_flight() > 0 || i < 10); ++i) {
        (void)storage_lane_->Turn(false);
        (void)device_lane_->SubmissionTurn();
        (void)device_lane_->CompletionTurn();
      }
    }
    threads_.clear();
    if (!stopped_ || !stopped_->has_value()) {
      problems.emplace_back("the scheduler stopped with a fault: backing is left as it is");
      return std::unexpected(problems.front());
    }
  }
  launch_.reset();
  ggml_.reset();
  if (execution_ != nullptr) {
    for (const auto stream : streams_) {
      if (stream.valid() && !execution_->DestroyStream(stream)) {
        problems.emplace_back("a stream could not be destroyed");
      }
    }
  }
  if (memory_ != nullptr) {
    for (Mapped* mapped : {&zone_, &kv_, &region_, &pool_, &locks_, &norms_, &tables_}) {
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
    for (const ReservationId reservation : weights_) {
      if (reservation.valid() && !memory_->Free(reservation)) {
        problems.emplace_back("a weights reservation still has mappings");
      }
    }
    if (memory_->backings() != 0) {
      problems.push_back(std::format("{} backings were left", memory_->backings()));
    }
  }
  for (void* pointer : pinned_) {
    (void)cudaFreeHost(pointer);
  }
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
  }
  if (!problems.empty()) {
    std::string joined;
    for (const std::string& problem : problems) {
      joined += (joined.empty() ? "" : "; ") + problem;
    }
    return Error(joined);
  }
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  const auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  Status ran;
  {
    Harness harness(*options);
    ran = harness.Run();
    if (auto finished = harness.Teardown(); !finished && ran) {
      ran = finished;
    }
  }
  if (!ran) {
    std::println(stderr, "FAILED: {}", ran.error());
    return 1;
  }
  std::println("DONE {}", options->out.string());
  return 0;
}
