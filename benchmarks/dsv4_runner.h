// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash as a model on a paged node (tests/support/paged_node.h;
// M3's swap path, docs/experiments/fast-swap/swap.md): its v0 prepared
// artifact paged into device VMM through the node's landing zone, each
// chunk run as one device job on the model's own stream under a lease on
// its whole closure (D-086), with the graph, plan and kernels of the
// resident harness (dsv4_exec.cc, via dsv4_common.h). CUDA builds only.
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
#include "kernels/ggml/launch.h"
#include "model/dsv4.h"
#include "paged_node.h"

namespace jitllm::benchmarks {

struct Dsv4Options {
  std::filesystem::path artifact;
  std::filesystem::path out;  // the spill file's directory
  std::uint32_t context = 8704;
  std::uint32_t max_rows = 512;
  bool graphs = true;  // decode graphs (D-090); set_graphs changes it between chunks
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
// offset δ in its first page. Computes δ and the pages; refused if the
// groups are not consecutive in each shard, change shard more than once,
// or a stride is below the stored bytes.
struct SlabLayout {
  std::uint64_t stride = 0;
  std::uint64_t delta = 0;
  std::vector<SlabPage> pages;
  std::uint64_t bytes() const { return pages.size() * test_support::kPagedExtent; }
};
std::expected<SlabLayout, std::string> LayOutSlab(std::span<const std::uint32_t> shard,
                                                  std::span<const std::uint64_t> file,
                                                  std::uint64_t stored, std::uint64_t stride);

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

  // After Run. The state zeroed (a job leasing it).
  Status Clear();
  // One chunk of `tokens` after n_past: the last row's logits in `logits`.
  // With `meanwhile`, the chunk's job is submitted without waiting, and
  // `meanwhile` runs on this thread while it is in flight (the RE-029
  // probe: a page-in beside a long job); its failure fails the chunk.
  Status Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
               std::vector<float>& logits, const std::function<Status()>& meanwhile = {});
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
  void DropPlans() { plans_.clear(); }
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
  // The weight extents (device, then the host table), and the state's.
  std::vector<catalog::ExtentId> weights() const;
  const std::vector<catalog::ExtentId>& state() const { return state_.extents; }
  std::uint64_t weight_read_bytes() const { return read_bytes_; }
  std::uint64_t state_bytes() const { return layout_.bytes; }
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
  // One chunk shape's plan, and its decode graph once captured.
  struct ShapePlan {
    kernels::ggml::Dsv4ChunkShape shape;
    std::unique_ptr<Dsv4Planned> planned;
    std::uint32_t eager_runs = 0;
    bool uncapturable = false;  // a capture was refused
    std::optional<kernels::ggml::CapturedGraph> graph;
    // The input copies the graph holds: each input's device address, its
    // bytes and its offset in the staging.
    std::vector<std::array<std::uint64_t, 3>> copies;
  };

  Status ReserveWeights();
  Status RegisterState();
  std::expected<ShapePlan*, std::string> Planned(const kernels::ggml::Dsv4ChunkShape& shape);
  void Check(const kernels::ggml::Dsv4Graph& graph);

  test_support::PagedNode& node_;
  const Dsv4Options& o_;
  int owner_;
  std::uint32_t stream_;

  std::unique_ptr<artifact::Artifact> artifact_;
  const model::Dsv4Profile& profile_ = model::Dsv4Flash();
  model::Dsv4Binding binding_;
  model::Dsv4StateLayout layout_;
  Dsv4Model model_;
  std::vector<artifact::FileDescriptor> shards_;
  std::array<std::uint8_t, 32> id_{};

  // Weights: one device reservation (dense regions, then slabs), the host
  // table's; each group's device address (0 for the table).
  providers::ReservationId weights_;
  std::uint64_t weights_base_ = 0;
  std::uint64_t weights_bytes_ = 0;
  std::vector<std::uint64_t> group_address_;
  std::uint32_t table_group_ = 0;
  providers::ReservationId table_;
  std::uint64_t table_base_ = 0;
  std::uint64_t table_bytes_ = 0;
  // Registration, in file order: each extent's source (its reservation's
  // offset, and its place in the landing).
  struct WeightExtent {
    catalog::ExtentId extent;
    bool host = false;
    std::uint64_t offset = 0;  // in its reservation
    scheduler::PageSource source;
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
  catalog::Closure fence_;       // the state: what a clear or a fence leases
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
};

}  // namespace jitllm::benchmarks

#endif  // JITLLM_BENCHMARKS_DSV4_RUNNER_H_
