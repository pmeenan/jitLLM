// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash as a model on a paged node (tests/support/paged_node.h;
// M3's swap path, docs/experiments/fast-swap/swap.md): its v0 prepared
// artifact paged into device VMM through the node's landing zone, each
// chunk run as one device job on the model's own stream under a lease on
// its whole closure (D-086): the request's, held from its start to its end
// when the harness opens one on the model's stream (PagedNode::BeginRequest,
// M3's lease per request), else the job's own. With the graph, plan and
// kernels of the resident harness (dsv4_exec.cc, via dsv4_common.h). CUDA
// builds only.
//
// - Memory, registered in the node's one catalog domain, every extent
//   2 MiB with managed backing (D-033):
//   - each dense group (a layer's non-expert tensors, the head, the
//     globals) has a 2 MiB-aligned region; its chunk k is an extent at the
//     region's base + k x 2 MiB, landed from its shard (D-081);
//   - each layer's routed experts form a slab (the resident expert layout,
//     docs/artifact-format.md#executable-views): expert e's group at
//     slab + e x S, S the group's stored bytes rounded up to its blocks and
//     16 bytes. S is not a multiple of 2 MiB, so an extent is a 2 MiB page
//     of the slab's address range, not an artifact chunk: its contents are
//     the stored bytes of the (at most two) groups it overlaps, which are
//     consecutive in the file, read as one 4 KiB-aligned range of up to
//     2 MiB + 8 KiB into a landing slot and copied into the page in two
//     pieces (PageSource::pieces); the S - stored bytes between groups are
//     never read by the kernels and are not written. The slab starts δ
//     bytes into its first page (δ a multiple of 256) so that where the
//     layer's groups change shard, a page boundary falls in the gap between
//     two groups: no page needs two files. Refused if a layer's groups are
//     not consecutive in file order within a shard, or change shard twice;
//   - the token table's chunks are extents of host VMM, read there
//     directly: the embedding rows are dequantized on the CPU, as
//     llama.cpp looks them up;
//   - the state (model/dsv4.h: the window cache, the compressed and
//     indexer caches and the compressor rings, three D-068
//     representations) is kPreserve live state with a write-back place in
//     an unnamed direct-I/O spill file in the output directory: evicting it
//     writes it back through the zone (D-081's reverse path), and loading
//     it restores it;
//   - the cuBLAS workspace: device VMM mapped at setup; the activations and
//     the GGML pool: the node's shared workspace; the input staging and the
//     logits row: pinned host memory, cataloged.
// - Extents are registered in file order, so a closure (sorted by
//   identity) loads in file order (RE-026).
// - A chunk: its host-built inputs, the graph planned for its shape (kept
//   per shape; DropPlans forgets them, for first-use measurements), and one
//   job that copies the inputs, runs the bound plan under the K-C launch
//   context and copies the last row's logits out. The first chunk of each
//   shape checks every tensor the plan binds against the catalog (BP-A1).
// - Decode graphs (D-090), with graphs on: a one-row chunk whose shape has
//   run once launch by launch is captured, the input copies, the plan's
//   4,972 steps and the logits copy together, as one CUDA graph on the
//   model's stream, and every later chunk of that shape replays it: one
//   launch. What varies between steps of a shape (the token, its position,
//   the cache cells, the masks and the compressors' indices) is the inputs'
//   data, which the host stages at the same offsets of the pinned staging
//   every time (checked at each replay), never a launch parameter: the
//   plan's launch parameters follow from the shape alone. Every address a
//   graph holds stays put across swaps: the weights' and the state's
//   places are pinned in the scheduler at registration (SetSource refuses
//   to move them), so a load maps whatever backing it takes at the same
//   place; the workspace, pool, cuBLAS workspace and staging are mapped for
//   the model's life; the landing zone is never in a graph. Graphs live
//   with their plans: DropPlans and Release destroy them first; at most
//   kMaxGraphs are kept (a capture past it destroys the oldest graph).
//   The places stay pinned for the runner's life (never unpinned: the
//   pins go with the scheduler, after Release has destroyed every graph).
//   Prefill chunks run launch by launch. A capture the runtime refuses
//   leaves that shape launch by launch.
// - The hash-routed layers' token-to-expert tables, which the kernels index
//   with unchecked, are checked after every full load (CheckHashRouting).
// - Speculation (Dsv4Options::drafter; docs/experiments/dspark/): the DSpark
//   drafter's own v0 artifact paged beside the target's, the same way (its
//   dense groups as regions, its experts as slabs), binding the target's
//   head and token table (model/dspark.h), its KV ring a second kPreserve
//   state region spilled and restored with the target's. A chunk of kind
//   kInject (prefill) or kVerify also computes the target's features and
//   injects its last rows into the ring, in the same graph. A verify runs
//   the row-invariant plan (D-092), returns every row's logits, and first
//   saves every state byte it will write (the target's cells, ring rows and
//   compressed rows, the drafter's ring cells: model/dsv4.h
//   Dsv4ChunkWrites, model/dspark.h DsparkWrites) into a snapshot region;
//   Accept then restores the rejected rows' bytes and the chunk's scratch
//   rows, before the next job's own work (or at once, with Rollback), which
//   leaves exactly what a verify of the accepted rows alone would have left
//   (D-068 truncation). Draft runs the drafter's block (a job of its own,
//   captured as a graph from its second run) and returns its drafts.
//   DraftVerify chains both in one job, saving a round trip a step: the
//   draft's graph, its drafts copied into the verify's staged tokens and
//   their embedding rows looked up on the device into its staged rows
//   (equal to the host's lookup, CheckDeviceEmbedding), then the verify's
//   graph.

#ifndef JITLLM_BENCHMARKS_DSV4_RUNNER_H_
#define JITLLM_BENCHMARKS_DSV4_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "dsv4_common.h"
#include "execution/registry.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "model/dspark.h"
#include "model/dsv4.h"
#include "paged_node.h"

namespace jitllm::benchmarks {

struct Dsv4Options {
  std::filesystem::path artifact;
  std::filesystem::path out;  // the spill file's directory
  std::uint32_t context = 8704;
  std::uint32_t max_rows = 512;
  bool graphs = true;  // decode graphs (D-090); set_graphs changes it between chunks
  // The DSpark drafter's artifact (model/dspark.h); empty: no speculation.
  std::filesystem::path drafter;
  // The most rows a verify takes: the anchor and its drafts.
  std::uint32_t max_verify = 4;
  // The rows of a draft block (the drafts it proposes).
  std::uint32_t draft_rows = 3;
  // The reference mode (dsv4_common.h Dsv4Model::exact): llama.cpp's
  // unfused graph and D-092's row-invariant verify; off, the fast plan.
  bool exact = false;
};

// What a chunk computes beside its target rows' own work.
enum class Dsv4ChunkKind : std::uint8_t {
  kPlain,   // the target alone
  kInject,  // and the DSpark drafter's features and injection (a prefill beside it)
  kVerify,  // a speculative verify: the row-invariant plan, the injection,
            // every row's logits, and a snapshot of what it writes
};

// How the chunks ran (decode graphs, D-090).
enum class Dsv4Path : std::uint8_t {
  kEager,     // launch by launch
  kCaptured,  // captured, then replayed once
  kReplayed,  // one launch of a graph captured earlier
};

struct Dsv4GraphStats {
  std::uint64_t eager = 0;
  std::uint64_t captured = 0;
  std::uint64_t replayed = 0;
  std::uint64_t refused = 0;       // captures refused: those shapes stay launch by launch
  std::uint64_t dropped = 0;       // graphs destroyed to keep at most kMaxGraphs
  double capture_seconds = 0;      // capturing
  double instantiate_seconds = 0;  // instantiating and uploading
  std::uint64_t nodes = 0;         // in every graph captured
  std::int64_t memory_bytes = 0;   // the drop in the device's free memory across captures
  std::string first_refusal;
};

// A page of a layer's expert slab: its read and the (at most two) pieces
// copied from the slot into the page. Exposed for tests.
inline constexpr std::size_t kSlabPieces = 4;  // at most kMaxDeviceCopies
struct SlabPage {
  std::uint32_t shard = 0;
  std::uint64_t file_offset = 0;  // 4 KiB-aligned
  std::uint64_t length = 0;       // 4 KiB-aligned
  std::array<std::uint64_t, kSlabPieces> slot_offset{};
  std::array<std::uint64_t, kSlabPieces> page_offset{};  // within the page
  std::array<std::uint64_t, kSlabPieces> bytes{};
  std::size_t pieces = 0;
};
// The most a page's read can take: a page's bytes, and 4 KiB before and
// after for alignment. The node's slot size for this model.
inline constexpr std::uint64_t kSlabSlotBytes =
    test_support::kPagedExtent + (std::uint64_t{8} * 1024);

// One layer's slab: where its experts' groups are in the file (group e:
// `shard[e]`, `file[e]`, each `stored` bytes), its stride and the slab's
// offset δ in its first page, a multiple of `alignment` (a power of two,
// 16 to 4,096). Computes δ and the pages; refused if the groups are not
// consecutive in each shard, change shard more than once, or a stride is
// below the stored bytes.
struct SlabLayout {
  std::uint64_t stride = 0;
  std::uint64_t delta = 0;
  std::vector<SlabPage> pages;
  std::uint64_t bytes() const { return pages.size() * test_support::kPagedExtent; }
};
std::expected<SlabLayout, std::string> LayOutSlab(std::span<const std::uint32_t> shard,
                                                  std::span<const std::uint64_t> file,
                                                  std::uint64_t stored, std::uint64_t stride,
                                                  std::uint64_t alignment = 256);

class Dsv4Runner final : public test_support::PagedModel {
 public:
  using Status = test_support::Status;
  // The most decode graphs kept (D-090): driver memory outside the
  // catalog, tens of MiB each; capturing another destroys the oldest.
  static constexpr std::size_t kMaxGraphs = 8;

  Dsv4Runner(test_support::PagedNode& node, const Dsv4Options& options, int owner,
             std::uint32_t stream)
      : node_(node), o_(options), owner_(owner), stream_(stream), graphs_(options.graphs) {}

  // Before the scheduler exists: the artifact, binding and state layout,
  // the largest chunk shapes measured, the model's own memory mapped
  // (state, cuBLAS workspace, staging) and its weights' places reserved
  // and cataloged.
  Status Setup();
  std::uint64_t activations_needed() const { return activation_bytes_; }
  std::uint64_t pool_needed() const { return scratch_bytes_; }
  // After Start, before Run: every weight's and the state's source, their
  // places pinned (D-090).
  Status Register();
  // After the node's workspace, before Run: the closures, the cuBLAS
  // handle, the launch context and the registry.
  Status Bind();

  // After Run. The state zeroed (a job leasing it), the drafter's ring
  // too, and any pending restore dropped.
  Status Clear();
  // One chunk of `tokens` after n_past: the last row's logits in `logits`
  // (every row's, rows × vocab, for kVerify).
  // With `meanwhile`, the chunk's job is submitted without waiting, and
  // `meanwhile` runs on this thread while it is in flight (the RE-029
  // probe: a page-in beside a long job); its failure fails the chunk.
  // kInject and kVerify need the drafter; a verify's rows must run at the
  // mask widths their one-row steps would (model/dsv4.h Dsv4SameWidths)
  // and number at most max_verify, and the next chunk after one must be
  // preceded by its Accept. A failed verify is undone before the next job;
  // any other failure after the chunk may have written the state
  // quarantines it until Clear.
  Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, const std::function<Status()>& meanwhile = {},
               Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain);

  // Speculation (Dsv4Options::drafter).
  bool speculative() const { return dartifact_ != nullptr; }
  // After a verify: its first `keep` rows (1 to its rows) stay; the rest,
  // and its scratch rows, are restored from its snapshot at the start of
  // the next job (Rollback runs that now).
  Status Accept(std::uint32_t keep);
  Status Rollback();
  // The drafter's block of draft_rows rows from pos0 (the anchor's position,
  // the target's state holding every position before it), anchored on
  // `anchor`: each slot's draft.
  Status Draft(std::uint32_t pos0, std::int32_t anchor, std::vector<std::int32_t>& drafts);
  // Draft and a verify of its first `rows` - 1 drafts in one job: the
  // drafts go from the draft's output into the verify's staged tokens, and
  // their embedding rows are looked up on the device from the host table
  // into its staged rows (which CheckDeviceEmbedding proves equal to the
  // host's), so the host sees the drafts only with the verify's logits.
  Status DraftVerify(std::uint32_t pos, std::int32_t anchor, std::uint32_t rows,
                     std::vector<std::int32_t>& drafts, std::vector<float>& logits);
  // Every token's embedding row looked up on the device equals the host's
  // lookup, bit for bit (a job per 2,048 tokens); the tokens checked.
  std::expected<std::uint64_t, std::string> CheckDeviceEmbedding();
  // The target's state and the drafter's ring, read to the host (a job).
  Status ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter);
  // The last verify's writes (model/dsv4.h Dsv4ChunkWrites, model/dspark.h
  // DsparkWrites): each range's offset in the target's state, or with
  // `ring` in the drafter's ring (as ReadState reads them), its bytes and
  // its row (-1: the chunk's scratch rows). Nothing else of either is
  // written by a verify.
  struct VerifyWrite {
    bool ring = false;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
    std::int64_t row = 0;
  };
  std::vector<VerifyWrite> last_verify_writes() const;
  // The reference mode on or off for the next chunks (Dsv4Options::exact):
  // every plan and graph dropped.
  void set_exact(bool on) {
    model_.exact = on;
    dmodel_.exact = on;
    DropPlans();
  }
  // Runs of the drafter's block: how they ran, and its last job's host time.
  const Dsv4GraphStats& draft_stats() const { return draft_stats_; }
  const model::DsparkProfile& dspark_profile() const { return dprofile_; }
  const model::Dsv4StateLayout& state_layout() const { return layout_; }
  const model::Dsv4Profile& profile() const { return profile_; }
  std::uint64_t drafter_read_bytes() const { return dread_bytes_; }
  // The device's time for `count` replays of the decode graph of the step
  // at n_past, queued back to back in one job with one input (the same
  // token and position each time): the decode step's GPU time without the
  // host's part. Changes the state (clear it after); refused if that shape
  // has no graph yet.
  std::expected<double, std::string> TimeReplays(std::uint32_t n_past, std::int32_t token,
                                                 std::uint32_t count);
  // The hash-routed layers' tables name only experts.
  Status CheckHashRouting();
  // Every weight and state extent is still pinned at the place registered
  // for it, which every captured graph names (D-090); refused, dropping
  // every graph, if one has moved. Between chunks, with the scheduler
  // running.
  Status CheckPlaces();
  // Forgets every planned shape and its graph: the next chunk of each
  // plans it again.
  void DropPlans() {
    plans_.clear();
    dplans_.clear();
  }
  std::size_t plans() const { return plans_.size(); }
  std::size_t graphs() const;
  double plan_seconds() const { return plan_seconds_; }  // spent planning, in all
  // Decode graphs on or off for the next chunks; captured graphs are kept.
  void set_graphs(bool on) { graphs_ = on; }
  const Dsv4GraphStats& graph_stats() const { return graph_stats_; }
  // The last chunk: how it ran, and the job's host time (the inputs built
  // and staged, and everything queued, waits for room in the stream
  // included, RE-029).
  Dsv4Path last_path() const { return last_path_; }
  double last_submit_seconds() const { return last_submit_seconds_; }

  // Every extent a chunk leases: weights, state, workspace and staging.
  const catalog::Closure& everything() const { return everything_; }
  // The weight extents (device, then the host table; then the drafter's),
  // and the state's (the target's, then the drafter's ring).
  std::vector<catalog::ExtentId> weights() const;
  std::vector<catalog::ExtentId> state() const;
  std::uint64_t weight_read_bytes() const { return read_bytes_; }
  std::uint64_t state_bytes() const { return layout_.bytes; }
  std::uint64_t state_base() const { return state_.base; }
  std::uint64_t slab_padding() const { return slab_padding_; }
  std::uint64_t coverage_tensors() const { return coverage_tensors_; }
  std::uint64_t coverage_violations() const { return coverage_violations_; }
  const std::string& first_violation() const { return first_violation_; }
  std::uint32_t vocab() const { return profile_.vocab; }
  const artifact::Artifact& artifact() const { return *artifact_; }

  std::uint32_t stream() const override { return stream_; }
  const catalog::Closure& fence_closure() const override { return fence_; }
  std::vector<catalog::ExtentId> managed_extents() const override;
  Status Release() override;

 private:
  // Input copies: each input's device address, its bytes and its offset in
  // the staging.
  using Copies = std::vector<std::array<std::uint64_t, 3>>;

  // A plan's runs: launched, or captured as a graph (D-090) and replayed.
  struct Runs {
    std::uint32_t eager_runs = 0;
    bool uncapturable = false;  // a capture was refused
    std::optional<kernels::ggml::CapturedGraph> graph;
    Copies copies;  // the input copies the graph holds
  };

  // One chunk shape's plan, and its decode graph once captured.
  struct ShapePlan : Runs {
    kernels::ggml::Dsv4ChunkShape shape;
    Dsv4ChunkKind kind = Dsv4ChunkKind::kPlain;
    std::int64_t inject_rows = 0;
    std::unique_ptr<Dsv4Planned> planned;
  };

  // A draft block's plan and its graph once captured.
  struct DraftPlan : Runs {
    std::unique_ptr<DsparkPlanned> planned;
  };

  // What queueing a plan's run did.
  struct Queued {
    Dsv4Path path = Dsv4Path::kEager;
    bool before = false;  // work queued before any failure
    std::expected<void, kernels::ggml::KernelFailure> result;
  };

  // One artifact's weights: one device reservation (dense regions, then
  // slabs), each group's device address, each layer's expert stride.
  struct Part {
    const artifact::Artifact* artifact = nullptr;
    const model::Dsv4Binding* binding = nullptr;
    std::uint32_t layers = 0;
    std::vector<artifact::FileDescriptor> shards;
    std::array<std::uint8_t, 32> id{};
    providers::ReservationId weights;
    std::uint64_t base = 0;
    std::uint64_t bytes = 0;
    std::vector<std::uint64_t> group_address;
    std::vector<std::uint64_t> stride;
  };

  // A saved range of a verify's snapshot: the state bytes at `address`,
  // saved at `saved`, and the verify row whose write it is (-1: the chunk's
  // scratch).
  struct Saved {
    std::uint64_t address = 0;
    std::uint64_t saved = 0;
    std::uint64_t bytes = 0;
    std::int64_t row = 0;
  };

  Status ReserveWeights();
  // `part`'s places (reserved and cataloged), with the host token table if
  // `table`.
  Status ReservePart(Part& part, bool table, std::uint64_t& read_bytes);
  static Status OpenPart(Part& part, const std::filesystem::path& path,
                         std::unique_ptr<artifact::Artifact>& opened);
  Status RegisterState();
  std::expected<ShapePlan*, std::string> Planned(const kernels::ggml::Dsv4ChunkShape& shape,
                                                 Dsv4ChunkKind kind, std::int64_t inject_rows);
  std::expected<DraftPlan*, std::string> PlannedDraft();
  void Check(const kernels::ggml::Dsv4Graph& graph);
  // Queues the pending restore (a verify's rejected rows), if any; it stays
  // owed unless queued.
  std::expected<void, kernels::ggml::KernelFailure> QueueRestore();
  // After a failed job, so the state is never left half-written: a verify
  // whose snapshot was queued (`saved`) is undone whole before the next
  // job's own work; anything else that may have written the state
  // (`wrote`), or a launch of unknown effect, quarantines it until Clear.
  void Settle(bool saved, bool wrote, bool unknown);
  // Refused while the state is quarantined.
  Status Usable() const;
  // Copies `host`'s inputs into the staging from `base`.
  std::expected<Copies, std::string> Stage(const Dsv4HostInputs& host, std::uint64_t base);
  // Queues a plan's input copies, its steps and its output's copy to
  // `out`: its graph replayed if it has one, captured if `capture`, else
  // launch by launch.
  Queued QueueRuns(Runs& runs, const Copies& copies, kernels::ggml::BoundGraph& bound, void* out,
                   const void* from, std::uint64_t out_bytes, bool capture, Dsv4GraphStats& stats,
                   void* native);
  // The verify's embedding rows of `n` drafts, looked up on the device
  // from the host table into the verify's staging (DraftVerify).
  std::expected<ggml_tensor*, std::string> DraftRowsNode(std::uint32_t n, const void* drafts);
  // A verify's snapshot: the ranges it writes, saved.
  Status PlanSnapshot(const model::Dsv4ChunkInputs& in);

  test_support::PagedNode& node_;
  const Dsv4Options& o_;
  int owner_;
  std::uint32_t stream_;

  std::unique_ptr<artifact::Artifact> artifact_;
  const model::Dsv4Profile& profile_ = model::Dsv4Flash();
  model::Dsv4Binding binding_;
  model::Dsv4StateLayout layout_;
  Dsv4Model model_;

  // Weights: the target's part and its host table (the table group's
  // address in the part is 0); the drafter's part.
  Part target_;
  std::uint32_t table_group_ = 0;
  providers::ReservationId table_;
  std::uint64_t table_base_ = 0;
  std::uint64_t table_bytes_ = 0;
  // Registration, in file order: each extent's source (its reservation's
  // offset, and its place in the landing).
  struct WeightExtent {
    catalog::ExtentId extent;
    bool host = false;
    std::uint64_t address = 0;  // its place's address
    scheduler::PageSource source;
    bool drafter = false;     // the drafter's part
    std::uint32_t group = 0;  // its artifact group (a slab page: its layer's first)
  };
  std::vector<WeightExtent> extents_;
  std::uint64_t read_bytes_ = 0;
  std::uint64_t slab_padding_ = 0;

  test_support::Mapped state_;
  std::vector<scheduler::PageSource> state_sources_;  // registered, by state extent
  test_support::Mapped cublas_workspace_;
  std::vector<catalog::ExtentId> staging_;
  void* inputs_ = nullptr;
  std::uint64_t input_bytes_ = 0;
  void* logits_ = nullptr;
  std::uint64_t activation_bytes_ = 0;
  std::uint64_t scratch_bytes_ = 0;
  std::uint64_t cublas_bytes_ = 0;
  int spill_fd_ = -1;

  std::unique_ptr<kernels::ggml::CublasHandle> cublas_;
  std::unique_ptr<kernels::ggml::LaunchContext> launch_;
  std::unique_ptr<execution::Registry> registry_;
  catalog::Closure everything_;
  catalog::Closure fence_;  // the state: what a clear or a fence leases
  // A draft's closure: the drafter's weights, the target's head and token
  // table, both states and the snapshot, the workspace and the staging (a
  // lease of a tenth of `everything_`'s extents).
  catalog::Closure draft_closure_;
  void* hash_tables_ = nullptr;  // pinned: the hash-routed layers' tables, read back

  std::vector<ShapePlan> plans_;  // destroyed before the launch context (Release)
  bool graphs_ = true;
  Dsv4GraphStats graph_stats_;
  Dsv4Path last_path_ = Dsv4Path::kEager;
  double last_submit_seconds_ = 0;
  double plan_seconds_ = 0;
  std::uint64_t coverage_tensors_ = 0;
  std::uint64_t coverage_violations_ = 0;
  std::string first_violation_;
  bool released_ = false;

  // The DSpark drafter (Dsv4Options::drafter).
  std::unique_ptr<artifact::Artifact> dartifact_;
  const model::DsparkProfile& dprofile_ = model::DsparkDeepSeekV4Flash();
  model::DsparkBinding dbinding_;
  model::DsparkStateLayout dlayout_;
  DsparkModel dmodel_;
  Part drafter_;
  std::uint64_t dread_bytes_ = 0;
  test_support::Mapped dstate_;  // its ring
  std::vector<scheduler::PageSource> dstate_sources_;
  // A verify's snapshot (working state), its save and restore descriptors
  // (pinned, read by the copy kernel), and the drafts' pinned row.
  test_support::Mapped snapshot_;
  kernels::ggml::RangeCopy* save_ = nullptr;
  kernels::ggml::RangeCopy* restore_ = nullptr;
  std::uint32_t range_capacity_ = 0;
  std::uint32_t restore_count_ = 0;  // pending, queued before the next job's work
  std::uint32_t save_count_ = 0;     // the next verify's
  std::vector<Saved> saved_;         // the last verify's snapshot
  std::uint32_t verify_rows_ = 0;    // its rows; 0 once accepted
  // A job failed after it may have written the state (Settle): every chunk,
  // draft, verify, rollback and read is refused until Clear has run.
  bool quarantined_ = false;
  void* drafts_ = nullptr;
  std::vector<DraftPlan> dplans_;  // destroyed before the launch context
  Dsv4GraphStats draft_stats_;
  // ReadState's pinned host copy (harness only, allocated on first use).
  void* state_host_ = nullptr;
  // A verify's inputs are staged from here, a draft's from 0, so one job
  // can stage both.
  std::uint64_t verify_base_ = 0;
  // DraftRowsNode's nodes (and CheckDeviceEmbedding's), in their own arena.
  std::optional<kernels::ggml::TensorArena> rows_arena_;
  std::vector<ggml_tensor*> draft_rows_;  // by draft count - 1
};

}  // namespace jitllm::benchmarks

#endif  // JITLLM_BENCHMARKS_DSV4_RUNNER_H_
