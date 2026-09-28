// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 Flash Next as a model on a paged node (tests/support/paged_node.h;
// M3's swap path, docs/experiments/fast-swap/swap.md): its v0 prepared
// artifact paged into device VMM through the node's landing zone, each
// chunk run as one device job on the model's own stream under a lease on
// its whole closure (D-086): the request's, held from its start to its end
// when the harness opens one on the model's stream (PagedNode::BeginRequest,
// M3's lease per request), else the job's own. With the graph, plan and
// kernels of the resident harness (qwen38_exec.cc, via qwen38_common.h).
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
//   registered (D-090, as DeepSeek's; no graphs yet, but a swap already
//   maps them back where they were).
// - The n-gram hash's constants are read back and checked
//   (CheckQwen38PleHash) after every full load (ReadPleHash), as DeepSeek's
//   hash-routing tables are.
// - A chunk: its host-built inputs (Qwen38Chunk, over the whole history),
//   its rows read, the graph planned for its shape (kept per shape), and
//   one job that copies the inputs, gathers the rows, runs the bound plan
//   and copies the last row's logits out. The first chunk of each shape
//   checks every tensor the plan binds against the catalog (BP-A1).

#ifndef JITLLM_BENCHMARKS_QWEN38_RUNNER_H_
#define JITLLM_BENCHMARKS_QWEN38_RUNNER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "execution/registry.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/qwen38_graph.h"
#include "model/qwen38.h"
#include "paged_node.h"
#include "paged_weights.h"
#include "ple_rows.h"
#include "providers/uring_storage.h"
#include "qwen38_common.h"

namespace jitllm::benchmarks {

struct Qwen38Options {
  std::filesystem::path artifact;
  std::filesystem::path out;  // the spill file's directory
  std::uint32_t context = 8704;
  std::uint32_t max_rows = 512;
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

class Qwen38Runner final : public test_support::PagedModel {
 public:
  using Status = test_support::Status;

  Qwen38Runner(test_support::PagedNode& node, const Qwen38Options& options, int owner,
               std::uint32_t stream)
      : node_(node), o_(options), owner_(owner), stream_(stream) {}
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
  // The state zeroed (a job leasing it).
  Status Clear();
  // One chunk: history[n_past, end) after n_past (history holds every token
  // from position 0); the last row's logits in `logits`.
  Status Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
               std::vector<float>& logits);
  void DropPlans() { plans_.clear(); }
  std::size_t plans() const { return plans_.size(); }
  double plan_seconds() const { return plan_seconds_; }
  const PleStats& ple() const { return ple_; }

  const catalog::Closure& everything() const { return everything_; }
  std::vector<catalog::ExtentId> weights() const { return weights_.extents(); }
  const std::vector<catalog::ExtentId>& state() const { return state_.extents; }
  std::uint64_t weight_read_bytes() const { return weights_.read_bytes(); }
  std::uint64_t state_bytes() const { return layout_.bytes; }
  std::uint64_t state_base() const { return state_.base; }
  std::uint64_t slab_padding() const { return weights_.slab_padding(); }
  std::uint64_t table_bytes() const { return table_.rows * table_.row_bytes; }
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
  Status ReserveWeights();
  Status RegisterState();
  std::expected<Qwen38Planned*, std::string> Planned(const kernels::ggml::Qwen38ChunkShape& shape);
  void Check(const kernels::ggml::Qwen38Graph& graph);

  test_support::PagedNode& node_;
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
  test_support::Mapped slot_memory_;
  std::byte* landing_ = nullptr;
  std::uint64_t landing_bytes_ = 0;
  std::uint32_t* sources_ = nullptr;
  std::unique_ptr<providers::UringStorage> ring_;
  bool rows_stalled_ = false;                          // reads left in flight: no chunk runs again
  providers::UringStorage* abandoned_ring_ = nullptr;  // such a ring, never destroyed
  PleStats ple_;

  test_support::Mapped state_;
  test_support::Mapped cublas_workspace_;
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

  std::vector<std::pair<kernels::ggml::Qwen38ChunkShape, std::unique_ptr<Qwen38Planned>>> plans_;
  double plan_seconds_ = 0;
  std::uint64_t coverage_tensors_ = 0;
  std::uint64_t coverage_violations_ = 0;
  std::string first_violation_;
  bool released_ = false;
};

}  // namespace jitllm::benchmarks

#endif  // JITLLM_BENCHMARKS_QWEN38_RUNNER_H_
