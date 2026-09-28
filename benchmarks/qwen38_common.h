// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What the Qwen3.8 harnesses share (docs/experiments/qwen38-native/,
// docs/experiments/fast-swap/): each chunk's graph built, bound, planned and
// placed, with the weights' and state's addresses given by the caller, so
// the paged runner (qwen38_runner.h) plans over device VMM exactly as the
// resident harness (qwen38_exec.cc) plans over cudaMalloc, both through
// this one path; and each chunk's host-built inputs in the graph's copy
// order. The paged runner's logits are checked equal to the resident
// harness's outputs bit for bit. CUDA builds only.

#ifndef JITLLM_BENCHMARKS_QWEN38_COMMON_H_
#define JITLLM_BENCHMARKS_QWEN38_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen38.h"

namespace jitllm::benchmarks {

// Where a Qwen3.8 model's weights and state live.
struct Qwen38Places {
  // A resource's device address.
  std::function<std::uint64_t(std::uint32_t resource)> resource;
  // An expert array's view: expert 0's slice, the others at the layer's
  // stride.
  std::function<std::uint64_t(std::uint32_t array)> array;
  std::vector<std::uint64_t> stride;  // each layer's expert stride
  std::uint64_t state = 0;            // the state region (model/qwen38.h)
  // Where the n-gram table's rows are read: the table's own address, or
  // (row paging) the chunk's row slots, which `binding`'s table then
  // describes (its rows the slots').
  std::uint64_t ple_table = 0;
};

struct Qwen38Model {
  const artifact::Artifact* artifact = nullptr;
  const model::Qwen38Profile* profile = nullptr;
  // The binding the graph is built from: the artifact's, or with row
  // paging a copy whose n-gram table has the row slots' rows.
  const model::Qwen38Binding* binding = nullptr;
  const model::Qwen38StateLayout* state = nullptr;
  Qwen38Places places;
  // The graph (qwen38_graph.h Qwen38GraphOptions): jitLLM's fusions, and
  // whether each layer's expert slots hold the CUTLASS layout (the
  // artifact's, binding->cutlass(), or converted at load by the resident
  // harness) rather than GGML's.
  bool fused = true;
  // The fused graph's reference form (Qwen38GraphOptions::exact) rather
  // than the fast one.
  bool exact = false;
  bool cutlass = false;
};

// One chunk shape's graph, plan, placement and bound implementations.
struct Qwen38Planned {
  std::optional<kernels::ggml::TensorArena> arena;
  kernels::ggml::Qwen38Graph graph;
  kernels::ggml::GraphPlan plan;
  kernels::ggml::Placement placement;
  std::optional<kernels::ggml::BoundGraph> bound;
  std::uint64_t scratch = 0;
  std::uint64_t inputs_bytes = 0;
};

// Binds the graph's weights and state at the model's places.
void BindQwen38Weights(const Qwen38Model& m, kernels::ggml::Qwen38Graph& g);

// Builds, binds, plans and places one chunk shape's graph: every computed
// tensor first at its own address, then placed in `activations` (0 to
// measure only), planned again, which must give the same plan; the named
// intermediates `keep` stay live to the end (the resident harness's
// dumps). Not bound to the registry.
std::expected<std::unique_ptr<Qwen38Planned>, std::string> PlanQwen38Chunk(
    const Qwen38Model& m, const kernels::ggml::Qwen38ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::uint64_t activations,
    std::uint64_t activation_bytes, std::span<const std::string> keep = {});

// A chunk's host-built inputs in the graph's copy order (the resident
// harness's): each input tensor and its bytes, which `in`, `out_ids` and
// `zeros` own. `ple_rows` replaces in.ple_rows when not empty (the row
// slots' indices).
struct Qwen38HostInputs {
  std::vector<std::int32_t> out_ids;
  std::int64_t zero_row = 0;
  std::int32_t zero_index = 0;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};
void Qwen38Sources(const kernels::ggml::Qwen38Graph& g, const model::Qwen38ChunkInputs& in,
                   std::uint32_t outputs, std::span<const std::int32_t> ple_rows,
                   Qwen38HostInputs& out);

}  // namespace jitllm::benchmarks

#endif  // JITLLM_BENCHMARKS_QWEN38_COMMON_H_
