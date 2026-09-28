// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What the DeepSeek V4 harnesses share (docs/experiments/dsv4-native/,
// docs/experiments/fast-swap/): each chunk's graph built, bound, planned
// and placed as the resident harness (dsv4_exec.cc) first did it, with the
// weights' and state's addresses given by the caller, so the paged runner
// (dsv4_runner.h) plans over device VMM exactly as the resident one plans
// over cudaMalloc; each chunk's host-built inputs; and the token lines and
// logits helpers. CUDA builds only.

#ifndef JITLLM_BENCHMARKS_DSV4_COMMON_H_
#define JITLLM_BENCHMARKS_DSV4_COMMON_H_

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
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/tensors.h"
#include "model/dsv4.h"

namespace jitllm::benchmarks {

// Where a DeepSeek model's weights and state live.
struct Dsv4Places {
  // A resource's device address (the token table's group has none: it is
  // read on the host).
  std::function<std::uint64_t(std::uint32_t resource)> resource;
  // An expert array's view: expert 0's slice, the others at the layer's
  // stride.
  std::function<std::uint64_t(std::uint32_t array)> array;
  std::vector<std::uint64_t> stride;  // each layer's expert stride
  std::uint64_t state = 0;            // the state region (model/dsv4.h)
};

struct Dsv4Model {
  const artifact::Artifact* artifact = nullptr;
  const model::Dsv4Profile* profile = nullptr;
  const model::Dsv4Binding* binding = nullptr;
  const model::Dsv4StateLayout* state = nullptr;
  Dsv4Places places;
  std::vector<float> rot;  // the indexer's Hadamard matrix
};

// One chunk shape's graph, plan, placement and bound implementations.
struct Dsv4Planned {
  std::optional<kernels::ggml::TensorArena> arena;
  kernels::ggml::Dsv4Graph graph;
  kernels::ggml::GraphPlan plan;
  kernels::ggml::Placement placement;
  std::optional<kernels::ggml::BoundGraph> bound;
  std::uint64_t scratch = 0;
  std::uint64_t inputs_bytes = 0;
};

// Binds the graph's weights and state at the model's places.
void BindDsv4Weights(const Dsv4Model& m, kernels::ggml::Dsv4Graph& g);

// Builds, binds, plans and places one chunk shape's graph: every computed
// tensor first at its own address, then placed in `activations` (0 to
// measure only), planned again, which must give the same plan. `keep`
// names llama.cpp callback tensors to keep alive. Not bound to the registry.
std::expected<std::unique_ptr<Dsv4Planned>, std::string> PlanDsv4Chunk(
    const Dsv4Model& m, const kernels::ggml::Dsv4ChunkShape& shape,
    const kernels::ggml::DeviceChoices& choices, std::span<const std::string> keep,
    std::uint64_t activations, std::uint64_t activation_bytes);

// A chunk's host-built inputs, in the graph's copy order: each input
// tensor and the bytes it takes (owned here, so they live as long as this).
struct Dsv4HostInputs {
  std::vector<float> embd;
  std::vector<std::int32_t> tokens;
  std::vector<std::int32_t> out_ids;
  std::vector<std::uint16_t> zeros;
  std::vector<std::pair<ggml_tensor*, const void*>> sources;
};

// The inputs of `tokens` at `in`'s positions: the embedding rows
// dequantized on the host from `table` (the token table group's bytes, in
// host memory, as llama.cpp's CPU backend looks them up), and the chunk
// plan's indices and masks. Refused for a token outside the vocabulary.
std::expected<void, std::string> BuildDsv4Inputs(
    const Dsv4Model& m, const kernels::ggml::Dsv4Graph& g, const model::Dsv4ChunkInputs& in,
    std::span<const std::int32_t> tokens, std::span<const std::byte> table, Dsv4HostInputs& out);

// A file of token lines: `name<TAB>ids...`, or ids alone.
struct TokenLine {
  std::string name;
  std::vector<std::int32_t> ids;
};
std::expected<std::vector<TokenLine>, std::string> ReadTokenLines(const std::filesystem::path& p);

std::int32_t Argmax(std::span<const float> row);
// -log softmax(row)[target], in double.
double Nll(std::span<const float> row, std::int32_t target);
std::expected<void, std::string> WriteFloats(const std::filesystem::path& p,
                                             std::span<const float> v);

}  // namespace jitllm::benchmarks

#endif  // JITLLM_BENCHMARKS_DSV4_COMMON_H_
