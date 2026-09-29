// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 Flash Next as a model on a paged node (paged_node.h; M3's swap
// path, docs/experiments/fast-swap/swap.md): its v0 prepared
// artifact paged into device VMM through the node's landing zone, each
// chunk run as one device job on the model's own stream under a lease on
// its whole closure (D-086): the request's, held from its start to its end
// when the driver opens one on the model's stream (PagedNode::BeginRequest,
// M3's lease per request), else the job's own. With the graph, plan and
// kernels of the resident harness (benchmarks/qwen38_exec.cc, via
// qwen38_plan.h).
// The layout of DeepSeek's runner (dsv4_runner.h), through paged_weights.h:
//
// - Weights: every dense group but the n-gram table's in a 2 MiB-aligned
//   region, a chunk an extent; each layer's routed experts a slab at the
//   resident layout's stride, an extent a 2 MiB page of it landed in
//   pieces: 2,764,800 bytes for an artifact in the CUTLASS layout, whose
//   slots the grouped GEMM and vector products read as they land (no
//   rewrite), or 2,768,976 for GGML's (mul_mat_id). The slab's offset in
//   its first page is a multiple of 16, the stride's own alignment (the
//   resident harness's expert e is 16-aligned for odd e too): the 80-byte
//   gap between GGML-layout groups cannot hold DeepSeek's 256 where a
//   layer's experts change shard.
// - The n-gram (PLE) table is not paged whole: before each chunk's job the
//   rows its tokens name are read on demand into a pinned landing and the
//   job gathers them into row slots (ple_rows.h); the graph is built over a
//   copy of the binding whose table has the slots' rows, and the chunk's
//   row indices are the slots'. Only the table's group is left unpaged: a
//   resource sharing it is refused.
// - The state (model/qwen38.h: the QSA layers' K, V and indexer caches, the
//   linear-attention layers' recurrent and convolution state, the n-gram
//   layer's convolution history) is kPreserve live state with a write-back
//   place in an unnamed direct-I/O spill file, as DeepSeek's.
// - Every weight and state extent's place is pinned in the scheduler when
//   registered (D-090, as DeepSeek's), so a swap maps them back where every
//   captured graph names them.
// - The n-gram hash's constants are read back and checked
//   (CheckQwen38PleHash) after every full load (ReadPleHash), as DeepSeek's
//   hash-routing tables are.
// - A chunk: its host-built inputs (Qwen38Chunk, over the whole history),
//   its rows read, the graph planned for its shape (kept per shape), and
//   one job that copies the inputs, gathers the rows, runs the bound plan
//   and copies the last row's logits out. The first chunk of each shape
//   checks every tensor the plan binds against the catalog (BP-A1).
// - Decode graphs (D-090), as DeepSeek's (dsv4_runner.h): a one-row chunk
//   (and a verify, and a draft) whose shape has run once launch by launch
//   is captured (the input copies from staging offsets fixed per shape, the
//   rows' gather, the plan's steps and the outputs' copies) and later runs
//   of that shape replay it as one launch. The gather's row count is data
//   the device reads (ple_rows.h), never a launch parameter. At most
//   kMaxGraphs are kept.
// - Speculation (Qwen38Options::drafter; docs/experiments/qwen38-mtp/): the
//   MTP drafter's own v0 artifact paged beside the target's (its dense
//   groups as regions, its experts as a slab), binding the target's token
//   table and head (model/qwen38.h Qwen38MtpBinding); its state (its KV and
//   indexer caches and the target's streams it reads) a second kPreserve
//   region spilled and restored with the target's. A prefill chunk with
//   `inject` also exports its rows' streams and runs the drafter's pass over
//   the positions whose next token it knows, in the same job. Draft runs the
//   drafter's catch-up over the rows the last verify kept (or the prefill
//   left) and its further passes, one job, returning the drafts. Verify runs
//   the target over the anchor and the drafts in the verify form (every
//   row's logits; the recurrent, convolution and n-gram state read but not
//   written, each row's inputs saved; the KV and indexer cells it writes
//   saved first), and Accept then commits the kept rows (qwen38_commit.h)
//   and restores the rejected rows' cells, before the next job's own work
//   (or at once, with Rollback): the state a verify of the kept rows alone
//   would have left. A failed verify is undone whole; any other failure
//   after a job may have written the state quarantines it until Clear.

#ifndef JITLLM_ENGINE_QWEN38_RUNNER_H_
#define JITLLM_ENGINE_QWEN38_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "engine/dsv4_runner.h"  // Dsv4Path, Dsv4GraphStats: how a model's chunks ran
#include "engine/paged_node.h"
#include "engine/paged_weights.h"
#include "engine/ple_rows.h"
#include "engine/qwen38_plan.h"
#include "execution/registry.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/qwen38_commit.h"
#include "kernels/ggml/qwen38_graph.h"
#include "model/qwen38.h"
#include "providers/device_execution.h"
#include "providers/storage.h"

namespace jitllm::engine {

struct Qwen38Options {
  std::filesystem::path artifact;
  std::filesystem::path out;  // the spill file's directory
  std::uint32_t context = 8704;
  std::uint32_t max_rows = 512;
  bool graphs = true;  // decode graphs (D-090); set_graphs changes it between chunks
  // The MTP drafter's artifact (model/qwen38.h Qwen38MtpBinding); empty: no
  // speculation.
  std::filesystem::path drafter;
  // Drafts a step (the drafter's passes) and the draft head's rows (the
  // lowest token IDs; 0: the whole vocabulary). The defaults were the
  // fastest of depths 2–3 and 32,768 rows to the whole vocabulary on
  // `prose` and `code` (docs/experiments/qwen38-mtp/).
  std::uint32_t draft_rows = 2;
  std::uint32_t draft_vocab = 65536;
};

// What the n-gram rows cost, summed over chunks.
struct PleStats {
  std::uint64_t chunks = 0;
  std::uint64_t lookups = 0;
  std::uint64_t rows = 0;          // distinct within each chunk, summed
  std::uint64_t reads = 0;         // direct reads
  std::uint64_t read_bytes = 0;    // their bytes
  std::uint64_t useful_bytes = 0;  // rows x 90
  std::uint64_t extent_bytes = 0;  // what whole 2 MiB chunks would have read
  double seconds = 0;              // planning and reading, on the caller's thread
};

class Qwen38Runner final : public PagedModel {
 public:
  using Status = engine::Status;
  // The most graphs kept (D-090): with speculation a context window holds
  // a decode step's, a verify's two (with and without its logits' copy) and
  // a draft's four (its catch-up of 1 to 4 rows); two windows' worth, so a
  // step past a 256-cell boundary does not recapture the steps before it.
  static constexpr std::size_t kMaxGraphs = 16;

  Qwen38Runner(PagedNode& node, const Qwen38Options& options, int owner, std::uint32_t stream)
      : node_(node), o_(options), owner_(owner), stream_(stream), graphs_(options.graphs) {}
  ~Qwen38Runner() override;
  Qwen38Runner(const Qwen38Runner&) = delete;
  Qwen38Runner& operator=(const Qwen38Runner&) = delete;
  Qwen38Runner(Qwen38Runner&&) = delete;
  Qwen38Runner& operator=(Qwen38Runner&&) = delete;

  // Before the scheduler exists: the artifact, binding and state layout,
  // the largest chunk shapes measured, the model's own memory mapped (state,
  // row slots, cuBLAS workspace, staging) and its weights' places reserved
  // and cataloged.
  Status Setup();
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  // After Start, before Run: every weight's and the state's source.
  Status Register();
  // After the node's workspace, before Run: the closures, the cuBLAS
  // handle, the launch context and the registry.
  Status Bind();

  // After a full load: the n-gram hash's constants read back and checked.
  Status ReadPleHash();
  // Fills the weight extents' unwritten bytes (PagedWeights::Unwritten):
  // the slab pages', the dense chunks' tails, or both.
  Status Scrub(std::uint8_t value, bool slabs, bool dense);
  // The state zeroed (a job leasing it), the drafter's too, and any pending
  // commit dropped.
  Status Clear();
  // One chunk: history[n_past, end) after n_past (history holds every token
  // from position 0); the last row's logits in `logits`. With `inject`
  // (speculating), the drafter's streams and its pass over the chunk too.
  Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
               std::vector<float>& logits, bool inject = false);

  // Speculation (Qwen38Options::drafter).
  bool speculative() const { return dartifact_ != nullptr; }
  std::uint32_t draft_rows() const { return o_.draft_rows; }
  // The drafter's catch-up and its passes: `history` every token through
  // the anchor (at position history.size() - 1, not yet in the target's
  // cache); the drafts of the next positions, draft_rows of them, and with
  // `probabilities` each draft's softmax probability over the draft head's
  // rows (the drafter's confidence; an adaptive window's input).
  Status Draft(std::span<const std::int32_t> history, std::vector<std::int32_t>& drafts,
               std::vector<float>* probabilities = nullptr);
  // A verify: history[n_past, end) the anchor and the drafts (history as
  // Chunk's), at most draft_rows + 1 rows; every row's argmax (the lowest
  // index among equals, on the device) and, with `logits`, every row's
  // logits (rows × vocab; a sampler's or a check's). The next job after one
  // must be preceded by its Accept.
  Status Verify(std::span<const std::int32_t> history, std::uint32_t n_past,
                std::vector<std::int32_t>& argmax, std::vector<float>* logits);
  // After a verify: its first `keep` rows (1 to its rows) stay; they are
  // committed and the rest's cells restored before the next job's own work
  // (Rollback runs that now).
  Status Accept(std::uint32_t keep);
  Status Rollback();
  // The target's state and the drafter's, read to the host (a job; any
  // pending commit runs first).
  Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter);
  const Dsv4GraphStats& draft_stats() const { return draft_stats_; }
  const model::Qwen38MtpState& mtp_state() const { return mtp_layout_; }
  std::uint64_t drafter_read_bytes() const { return dweights_.read_bytes(); }

  // Forgets every planned shape and its graph.
  void DropPlans() {
    plans_.clear();
    mplans_.clear();
  }
  std::size_t plans() const { return plans_.size(); }
  std::size_t graphs() const;
  double plan_seconds() const { return plan_seconds_; }
  const PleStats& ple() const { return ple_; }
  // Decode graphs on or off for the next chunks; captured graphs are kept.
  void set_graphs(bool on) { graphs_ = on; }
  const Dsv4GraphStats& graph_stats() const { return graph_stats_; }

  const catalog::Closure& everything() const { return everything_; }
  // The weight extents (the target's, then the drafter's) and the state's
  // (the target's, then the drafter's).
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  std::uint64_t weight_read_bytes() const { return weights_.read_bytes() + dweights_.read_bytes(); }
  std::uint64_t state_bytes() const { return layout_.bytes; }
  std::uint64_t state_base() const { return state_.base; }
  // The drafter's state (0 bytes without speculation), and the streams rows
  // its next draft catches up on (host-side): with the target's state, the
  // conversation state a check saves and puts back (fence_closure() leases
  // both regions). set_pending_rows only after Rollback, restoring a value
  // pending_rows() gave for the same state.
  std::uint64_t drafter_state_base() const { return mstate_.base; }
  std::uint64_t drafter_state_bytes() const { return speculative() ? mtp_layout_.bytes : 0; }
  std::uint32_t pending_rows() const { return pending_rows_; }
  void set_pending_rows(std::uint32_t rows) { pending_rows_ = rows; }
  std::uint64_t slab_padding() const { return weights_.slab_padding(); }
  std::uint64_t table_bytes() const { return table_.rows * table_.row_bytes; }
  std::uint64_t coverage_tensors() const { return coverage_tensors_; }
  std::uint64_t coverage_violations() const { return coverage_violations_; }
  const std::string& first_violation() const { return first_violation_; }
  std::uint32_t vocab() const { return profile_.vocab; }
  const artifact::Artifact& artifact() const { return *artifact_; }
  const model::Qwen38StateLayout& state_layout() const { return layout_; }

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  // Input copies: each input's device address, its bytes and its offset in
  // the staging; output copies: each output's host address, device
  // address and bytes.
  using Copies = std::vector<std::array<std::uint64_t, 3>>;

  // A plan's runs: launched, or captured as a graph (D-090) and replayed.
  struct Runs {
    std::uint32_t eager_runs = 0;
    bool uncapturable = false;  // a capture was refused
    std::optional<kernels::ggml::CapturedGraph> graph;
    Copies copies;  // the input copies the graph holds
  };

  // One chunk shape's plan, and its decode graph once captured; a verify's
  // `lean` runs copy its argmaxes out without its logits.
  struct ShapePlan : Runs {
    kernels::ggml::Qwen38ChunkShape shape;
    Qwen38ChunkKind kind;
    std::unique_ptr<Qwen38Planned> planned;
    Runs lean;
  };

  // One drafter shape's plan, and its graph once captured.
  struct MtpPlan : Runs {
    kernels::ggml::Qwen38MtpShape shape;
    std::unique_ptr<Qwen38MtpPlanned> planned;
  };

  // What queueing a plan's run did.
  struct Queued {
    Dsv4Path path = Dsv4Path::kEager;
    bool before = false;  // work queued before any failure
    std::expected<void, kernels::ggml::KernelFailure> result;
  };

  // A saved range of a verify's cells: the state bytes at `address`, saved
  // at `saved`, written by verify row `row`.
  struct Saved {
    std::uint64_t address = 0;
    std::uint64_t saved = 0;
    std::uint64_t bytes = 0;
    std::uint32_t row = 0;
  };

  Status ReserveWeights();
  Status RegisterState();
  std::expected<ShapePlan*, std::string> Planned(const kernels::ggml::Qwen38ChunkShape& shape,
                                                 Qwen38ChunkKind kind);
  std::expected<MtpPlan*, std::string> PlannedMtp(const kernels::ggml::Qwen38MtpShape& shape);
  void Check(const kernels::ggml::Qwen38Graph& graph);
  void CheckMtp(const kernels::ggml::Qwen38MtpGraph& graph);
  // Copies `sources`' inputs into the staging from `base`.
  std::expected<Copies, std::string> Stage(
      std::span<const std::pair<ggml_tensor*, const void*>> sources, std::uint64_t base);
  // Queues a plan's input copies, `between` (the rows' gather), its steps
  // and its outputs' copies: its graph replayed if it has one, captured if
  // `capture`, else launch by launch.
  Queued QueueRuns(Runs& runs, const Copies& copies,
                   const std::function<bool(void* stream)>& between,
                   kernels::ggml::BoundGraph& bound, const Copies& outputs, bool capture,
                   Dsv4GraphStats& stats, providers::NativeStream native);
  // Makes room for one more graph (the oldest destroyed past kMaxGraphs).
  void RoomForGraph();
  // The chunk's n-gram rows planned, read and their slots' sources set.
  std::expected<std::vector<std::int32_t>, std::string> ReadRows(
      const model::Qwen38ChunkInputs& in);
  // Queues the pending commit and restore (a verify's kept and rejected
  // rows), if any; they stay owed unless queued.
  std::expected<void, kernels::ggml::KernelFailure> QueueCommit();
  // After a failed job, so the state is never left half-written (as
  // DeepSeek's runner, dsv4_runner.h Settle).
  void Settle(bool verified, bool wrote, bool unknown);
  Status Usable() const;
  // The drafter's shape and pass inputs for `rows` rows from `first` and
  // `passes` - 1 single rows after them.
  std::expected<std::pair<kernels::ggml::Qwen38MtpShape, std::vector<model::Qwen38ChunkInputs>>,
                std::string>
  MtpInputs(std::uint32_t first, std::uint32_t rows, std::uint32_t passes, bool head,
            std::int64_t hidden_row) const;

  PagedNode& node_;
  const Qwen38Options& o_;
  int owner_;
  std::uint32_t stream_;

  std::unique_ptr<artifact::Artifact> artifact_;
  const model::Qwen38Profile& profile_ = model::Qwen38Flash();
  model::Qwen38Binding binding_;        // the artifact's
  model::Qwen38Binding graph_binding_;  // the table's rows the slots'
  model::Qwen38StateLayout layout_;
  model::Qwen38PleHash hash_;
  bool hash_checked_ = false;
  Qwen38Model model_;
  std::vector<artifact::FileDescriptor> shards_;
  std::array<std::uint8_t, 32> id_{};
  PagedWeights weights_;

  // The n-gram rows: the table in its shard, the slots (device), the
  // landing and the slots' sources (pinned), the runner's own ring.
  PleTable table_;
  std::uint64_t slots_ = 0;  // row slots: max_rows x ple_heads
  Mapped slot_memory_;
  std::byte* landing_ = nullptr;
  std::uint64_t landing_bytes_ = 0;
  std::uint32_t* sources_ = nullptr;
  std::uint32_t* ple_count_ = nullptr;  // pinned: the rows the next gather takes
  std::unique_ptr<providers::Storage> ring_;
  bool rows_stalled_ = false;  // reads left in flight: no chunk runs again
  PleStats ple_;

  Mapped state_;
  std::vector<scheduler::PageSource> state_sources_;
  Mapped cublas_workspace_;
  std::vector<catalog::ExtentId> staging_;
  void* inputs_ = nullptr;
  std::uint64_t input_bytes_ = 0;
  void* logits_ = nullptr;
  void* hash_host_ = nullptr;  // pinned: the hash constants, read back
  std::vector<PagedWeights::Range> unwritten_;
  std::uint64_t* scrub_ = nullptr;  // pinned: Scrub's ranges
  std::uint64_t activation_bytes_ = 0;
  std::uint64_t scratch_bytes_ = 0;
  std::uint64_t cublas_bytes_ = 0;
  int spill_fd_ = -1;

  std::unique_ptr<kernels::ggml::CublasHandle> cublas_;
  std::unique_ptr<kernels::ggml::LaunchContext> launch_;
  std::unique_ptr<execution::Registry> registry_;
  catalog::Closure everything_;
  catalog::Closure fence_;  // the state: what a clear or a fence leases

  std::vector<ShapePlan> plans_;  // destroyed before the launch context (Release)
  std::vector<MtpPlan> mplans_;
  bool graphs_ = true;
  Dsv4GraphStats graph_stats_;
  Dsv4GraphStats draft_stats_;
  double plan_seconds_ = 0;
  std::uint64_t coverage_tensors_ = 0;
  std::uint64_t coverage_violations_ = 0;
  std::string first_violation_;
  bool released_ = false;

  // The MTP drafter (Qwen38Options::drafter).
  std::unique_ptr<artifact::Artifact> dartifact_;
  model::Qwen38MtpBinding dbinding_;
  model::Qwen38MtpState mtp_layout_;
  model::Qwen38CommitLayout commit_layout_;
  std::vector<artifact::FileDescriptor> dshards_;
  std::array<std::uint8_t, 32> did_{};
  PagedWeights dweights_;
  Mapped mstate_;  // its state (spilled with the target's)
  std::vector<scheduler::PageSource> mstate_sources_;
  // A verify's saves and its cells' snapshot (D-068 working state, charged
  // with the model, never spilled: mapped for the model's life, it stays
  // across a swap, and a commit still pending then runs at the next job).
  Mapped commit_;
  std::uint64_t snapshot_offset_ = 0;            // in commit_, after the saves
  kernels::ggml::Qwen38CommitArgs commit_args_;  // every place but `keep`, from Bind
  std::uint64_t mtp_base_ = 0;                   // a drafter pass's inputs are staged from here
  kernels::ggml::RangeCopy* save_ = nullptr;
  kernels::ggml::RangeCopy* restore_ = nullptr;
  kernels::ggml::RangeCopy* carry_ = nullptr;  // a prefill's pending streams row
  std::uint32_t save_count_ = 0;               // the next verify's
  std::uint32_t restore_count_ = 0;            // pending
  std::uint32_t commit_keep_ = 0;              // pending: rows to commit (0: none)
  std::vector<Saved> saved_;                   // the last verify's
  std::uint32_t verify_rows_ = 0;              // its rows; 0 once accepted
  std::uint32_t pending_rows_ = 0;             // streams rows the next draft catches up on
  void* drafts_ = nullptr;                     // pinned: a draft's, then a verify's argmaxes
  // A job failed after it may have written the state (Settle): every chunk,
  // draft, verify, rollback and read is refused until Clear has run.
  bool quarantined_ = false;
  void* state_host_ = nullptr;  // ReadState's pinned host copy (harness only)
};

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_QWEN38_RUNNER_H_
