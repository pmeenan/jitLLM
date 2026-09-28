// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML graph of one DeepSeek V4 chunk (model/dsv4.h), built with GGML's
// graph functions over tensor descriptors (tensors.h) as llama.cpp b29c606e2
// builds it (src/models/deepseek4.cpp, MIT) for these settings: flash
// attention on, F16 caches, one sequence and stream, no speculative rollback
// planes, every row an output, no LoRA or control vector, and upstream's
// defaults for the model-level fused operations (dsv4_hc_pre, dsv4_hc_comb,
// dsv4_hc_post and the lightning indexer). The same operations, parameters,
// views and ggml_build_forward_expand calls give the same nodes in the same
// order (fusion.h GraphOrder), with four differences, none of which changes
// a computed value:
//   - the compressor state is read straight from the ring (llama.cpp's
//     restore copies run only with rollback planes, n_rs_seq > 0);
//   - a cache is viewed once at the width attention reads, where llama.cpp
//     views it at the cache's width and then narrows the view;
//   - the token embedding rows are an input, looked up on the host as
//     llama.cpp looks them up on the CPU;
//   - the indexer's Hadamard matrix is an input the transform never reads.
//
// Routed experts are 3D weights [k, n, experts] whose expert stride (nb[2])
// is the caller's: the GGUF's packed stride, or the stride of the repacked
// expert groups laid out at a uniform stride (the resident expert layout,
// docs/artifact-format.md#executable-views). GGML's mul_mat_id kernels
// address expert e at base + e·nb[2] and count the stride in whole blocks,
// so it must be a multiple of every expert projection's block size.
//
// Every tensor is created unbound. Bind the leaves (inputs, weights and the
// state), give every computed node that is no view memory, then bind the
// views (graph_plan.h BindViews). Every profile builds this; nothing here
// launches.

#ifndef JITLLM_KERNELS_GGML_DSV4_GRAPH_H_
#define JITLLM_KERNELS_GGML_DSV4_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "model/dsv4.h"

namespace jitllm::kernels::ggml {

// A chunk's shape, from its host-built inputs.
struct Dsv4ChunkShape {
  std::int64_t rows = 0;
  std::int64_t raw_n_kv = 0;
  std::int64_t raw_cells = 0;
  std::int64_t csa_n_kv = 0;  // also the indexer's
  std::int64_t hca_n_kv = 0;
  std::int64_t csa_cells = 0;
  std::int64_t hca_cells = 0;
  std::int64_t csa_blocks = 0;
  std::int64_t hca_blocks = 0;
  std::int64_t csa_persist = 0;
  std::int64_t hca_persist = 0;
  std::int64_t csa_state_rows = 0;
  std::int64_t hca_state_rows = 0;

  bool operator==(const Dsv4ChunkShape&) const = default;
};

Dsv4ChunkShape Dsv4ShapeOf(const model::Dsv4StateLayout& state,
                           const model::Dsv4ChunkInputs& chunk);

// One compressor's host-built inputs.
struct Dsv4CompInputs {
  ggml_tensor* state_pos = nullptr;    // I32 [rows]
  ggml_tensor* persist_src = nullptr;  // I32
  ggml_tensor* persist_dst = nullptr;  // I32
  ggml_tensor* read_idxs = nullptr;    // I32
  ggml_tensor* write_idxs = nullptr;   // I64 [blocks]
  ggml_tensor* write_pos = nullptr;    // I32 [blocks]
  ggml_tensor* mask = nullptr;         // F16 [n_kv, rows, 1, 1]
};

// A layer's weights (unbound leaves: bind each at its resource's address)
// and state (bind at the state layout's offsets).
struct Dsv4LayerTensors {
  ggml_tensor* attn_norm = nullptr;
  ggml_tensor* attn_sinks = nullptr;
  ggml_tensor* q_a = nullptr;
  ggml_tensor* q_a_norm = nullptr;
  ggml_tensor* q_b = nullptr;
  ggml_tensor* kv = nullptr;
  ggml_tensor* kv_norm = nullptr;
  ggml_tensor* out_a = nullptr;  // [heads·head / groups, o_lora, groups]
  ggml_tensor* out_b = nullptr;
  ggml_tensor* hc_attn_fn = nullptr;
  ggml_tensor* hc_attn_base = nullptr;
  ggml_tensor* hc_attn_scale = nullptr;
  ggml_tensor* hc_ffn_fn = nullptr;
  ggml_tensor* hc_ffn_base = nullptr;
  ggml_tensor* hc_ffn_scale = nullptr;
  ggml_tensor* comp_kv = nullptr;
  ggml_tensor* comp_gate = nullptr;
  ggml_tensor* comp_ape = nullptr;
  ggml_tensor* comp_norm = nullptr;
  ggml_tensor* idx_q_b = nullptr;
  ggml_tensor* idx_proj = nullptr;
  ggml_tensor* idx_comp_kv = nullptr;
  ggml_tensor* idx_comp_gate = nullptr;
  ggml_tensor* idx_comp_ape = nullptr;
  ggml_tensor* idx_comp_norm = nullptr;
  ggml_tensor* ffn_norm = nullptr;
  ggml_tensor* router = nullptr;
  ggml_tensor* router_bias = nullptr;  // routed layers
  ggml_tensor* tid2eid = nullptr;      // hash layers
  ggml_tensor* up_exps = nullptr;      // [width, ffn, experts] at the caller's stride
  ggml_tensor* gate_exps = nullptr;
  ggml_tensor* down_exps = nullptr;
  ggml_tensor* up_shexp = nullptr;
  ggml_tensor* gate_shexp = nullptr;
  ggml_tensor* down_shexp = nullptr;
  // State.
  ggml_tensor* raw_k = nullptr;         // F16 [head, raw_cells, 1]
  ggml_tensor* csa_k = nullptr;         // F16 [head, csa_cells, 1]
  ggml_tensor* csa_state_kv = nullptr;  // F32 [2·head, 8]
  ggml_tensor* csa_state_score = nullptr;
  ggml_tensor* lid_k = nullptr;  // F16 [indexer head, csa_cells, 1]
  ggml_tensor* lid_state_kv = nullptr;
  ggml_tensor* lid_state_score = nullptr;
  ggml_tensor* hca_k = nullptr;         // F16 [head, hca_cells, 1]
  ggml_tensor* hca_state_kv = nullptr;  // F32 [head, 128]
  ggml_tensor* hca_state_score = nullptr;
};

struct Dsv4GraphOptions {
  // Each layer's routed-expert stride in bytes (nb[2] of the three expert
  // weights); 0 for the packed stride of one [k, n] slice.
  std::vector<std::uint64_t> expert_stride;
};

struct Dsv4Graph {
  ggml_tensor* embd = nullptr;        // F32 [width, rows]: the embedding rows
  ggml_tensor* tokens = nullptr;      // I32 [rows]: the hash layers' routing
  ggml_tensor* positions = nullptr;   // I32 [rows]
  ggml_tensor* raw_k_idxs = nullptr;  // I64 [rows]: each token's ring cell
  ggml_tensor* raw_mask = nullptr;    // F16 [raw_n_kv, rows, 1, 1]
  ggml_tensor* out_ids = nullptr;     // I32 [rows]: 0 .. rows - 1, every row an output
  Dsv4CompInputs csa, hca, lid;
  ggml_tensor* lid_rot = nullptr;      // F32 [indexer head, indexer head]: never read
  ggml_tensor* top_k_zeros = nullptr;  // F16: the zero fill's source, never read
  std::vector<Dsv4LayerTensors> layers;
  ggml_tensor* output_norm = nullptr;
  ggml_tensor* output = nullptr;
  ggml_tensor* hc_head_fn = nullptr;
  ggml_tensor* hc_head_base = nullptr;
  ggml_tensor* hc_head_scale = nullptr;
  ggml_tensor* logits = nullptr;    // F32 [vocab, rows]: the last node
  std::vector<ggml_tensor*> nodes;  // GGML's order, views included
  // Intermediate tensors under llama.cpp's callback names ("l_last-7",
  // "attn_out-7", "ffn_moe_out-7", "hc_head-1", ...), for comparisons.
  std::vector<std::pair<std::string, ggml_tensor*>> named;

  // The host-built inputs, in the order they are copied.
  std::vector<ggml_tensor*> inputs() const;
  ggml_tensor* Named(std::string_view name) const;
};

// How many tensors a chunk's graph creates at most, for TensorArena.
std::size_t Dsv4GraphTensors(const model::Dsv4Profile& profile);

// Builds the chunk's graph on `arena` with the binding's weight types and
// shapes. Refused if the shape is not one the state holds, a weight type is
// not a GGML type, an expert stride is not a whole number of the expert
// projections' blocks, or the arena lacks room.
std::expected<Dsv4Graph, KernelFailure> BuildDsv4Graph(TensorArena& arena,
                                                       const model::Dsv4Profile& profile,
                                                       const model::Dsv4Binding& binding,
                                                       const Dsv4ChunkShape& shape,
                                                       const Dsv4GraphOptions& options);

// The GGML type of a type name, if GGML has one.
std::expected<ggml_type, KernelFailure> GgmlTypeOf(std::string_view name);

// The Hadamard matrix llama.cpp gives the indexer (ggml_gen_hadamard), n x n
// F32, row-major.
std::vector<float> HadamardMatrix(std::int64_t n);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_DSV4_GRAPH_H_
