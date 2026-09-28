// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML graph of one Qwen3.8 Flash Next chunk (model/qwen38.h), built
// with GGML's graph functions over tensor descriptors (tensors.h) after
// llama.cpp b29c606e2's qwen4exp graph (src/models/qwen4exp.cpp and the
// llm_graph_context parts it calls, MIT): hyper-connections, the n-gram
// embedding layer, Gated DeltaNet with the fused gated delta rule, QSA with
// its indexer, the MoE with the shared expert, and the head. Settings: one
// sequence and stream, F16 caches, flash attention, no speculative rollback
// planes, no LoRA. The oracle is Mia's vLLM on the same checkpoint, not
// llama.cpp, so the port keeps upstream's operation plan but not its node
// order; where they differ:
//   - the checkpoint's formats: MXFP8 products run jitLLM's own operations
//     (jitllm_ops.h), the vector product up to 8 rows and otherwise the
//     weights dequantized to BF16 for GGML's float product; the n-gram
//     table's NVFP4 rows are gathered by jitllm.nvfp4.get_rows; routed
//     experts are GGML NVFP4 with each expert's global scale applied after
//     its product, as llama.cpp's build_lora_mm_id applies a `_s` tensor;
//   - the indexer's fused q/k projection is one product whose rows are
//     viewed (llama.cpp's converter splits it);
//   - state is written back with set_rows at row 0 of each state tensor
//     (llama.cpp copies into a view of its recurrent cache);
//   - the F32 copy of the causal mask the indexer adds is an input, not a
//     cast of the F16 one, and the indexer's key cache is F32 (GGML's row
//     gather takes F32 or BF16 rows);
//   - the shared expert's gate (one value a token) is the gate row, read as
//     F32, times each token, summed: GGML's products refuse a one-row
//     output;
//   - the chunk's rows are transposed into packed rows before the
//     convolution histories are concatenated to them;
//   - when attention reads no more cells than the indexer's budget keeps,
//     the selection keeps every cell, so its scoring is not built (the
//     indexer's keys are still cached);
//   - the rows the head computes are gathered before the final mixer
//     (llama.cpp gathers them in the last layer).
//
// Routed experts are 3D weights [k, n, experts] at the caller's expert
// stride (the resident expert layout, docs/artifact-format.md#executable-views);
// the down projection's rows (640 elements) are not whole 512-element steps,
// so its weights are marked as padded (validate_ext.h
// MarkRowPaddingReadable): the caller's slab must hold the artifact's
// readable bytes past each slice, as the artifact reserves them.
//
// Every tensor is created unbound; bind the leaves, place the computed
// nodes, then bind the views (graph_plan.h BindViews). Nothing here
// launches.

#ifndef JITLLM_KERNELS_GGML_QWEN38_GRAPH_H_
#define JITLLM_KERNELS_GGML_QWEN38_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/tensors.h"
#include "model/qwen38.h"

namespace jitllm::kernels::ggml {

struct Qwen38ChunkShape {
  std::int64_t rows = 0;
  std::int64_t n_kv = 0;
  std::int64_t cells = 0;
  std::int64_t outputs = 0;  // rows whose logits are computed: the last `outputs`
  bool qsa_select = false;
  std::int64_t qsa_blocks = 0;

  bool operator==(const Qwen38ChunkShape&) const = default;
};

Qwen38ChunkShape Qwen38ShapeOf(const model::Qwen38StateLayout& state,
                               const model::Qwen38ChunkInputs& chunk, std::int64_t outputs);

struct Qwen38Mxfp8Tensors {
  ggml_tensor* codes = nullptr;   // I8 [k, n]
  ggml_tensor* scales = nullptr;  // I8 [k / 32, n]
};

struct Qwen38LayerTensors {
  ggml_tensor* hc_attn_norm = nullptr;
  ggml_tensor* hc_attn_down = nullptr;
  ggml_tensor* hc_attn_up = nullptr;
  ggml_tensor* hc_attn_inject = nullptr;
  ggml_tensor* hc_ffn_norm = nullptr;
  ggml_tensor* hc_ffn_down = nullptr;
  ggml_tensor* hc_ffn_up = nullptr;
  ggml_tensor* hc_ffn_inject = nullptr;
  Qwen38Mxfp8Tensors qkv, z, beta, alpha, ssm_out;
  ggml_tensor* dt_bias = nullptr;
  ggml_tensor* ssm_a = nullptr;
  ggml_tensor* conv1d = nullptr;
  ggml_tensor* ssm_norm = nullptr;
  Qwen38Mxfp8Tensors q, k, v, o, idx_qk;
  ggml_tensor* q_norm = nullptr;
  ggml_tensor* k_norm = nullptr;
  ggml_tensor* idx_q_norm = nullptr;
  ggml_tensor* idx_k_norm = nullptr;
  ggml_tensor* ple_key = nullptr;
  ggml_tensor* ple_value = nullptr;
  ggml_tensor* ple_norm_key = nullptr;
  ggml_tensor* ple_norm_query = nullptr;
  ggml_tensor* ple_norm_conv = nullptr;
  ggml_tensor* ple_conv1d = nullptr;
  ggml_tensor* router = nullptr;
  ggml_tensor* shared_gate = nullptr;
  Qwen38Mxfp8Tensors gate_shexp, up_shexp, down_shexp;
  ggml_tensor* gate_exps = nullptr;  // [k, n, experts] at the caller's stride
  ggml_tensor* up_exps = nullptr;
  ggml_tensor* down_exps = nullptr;
  ggml_tensor* gate_exps_scale = nullptr;  // F32 [experts]
  ggml_tensor* up_exps_scale = nullptr;
  ggml_tensor* down_exps_scale = nullptr;
  // State (bind at the state layout's offsets).
  ggml_tensor* cache_k = nullptr;     // F16 [head_dim · kv_heads, cells]
  ggml_tensor* cache_v = nullptr;     // F16 [head_dim · kv_heads, cells]
  ggml_tensor* cache_idx = nullptr;   // F32 [indexer_head_dim, cells]
  ggml_tensor* conv_state = nullptr;  // F32 [(conv - 1) · channels, 1]
  ggml_tensor* recurrent = nullptr;   // F32 [head² · v_heads, 1]
  ggml_tensor* ple_state = nullptr;   // F32 [ple_history · hc_width, 1]
};

struct Qwen38GraphOptions {
  // Each layer's routed-expert stride in bytes (nb[2] of the three expert
  // weights); 0 for the packed stride of one slice.
  std::vector<std::uint64_t> expert_stride;
};

struct Qwen38Graph {
  // Inputs, host-built (model/qwen38.h Qwen38ChunkInputs).
  ggml_tensor* tokens = nullptr;       // I32 [rows]
  ggml_tensor* positions = nullptr;    // I32 [4 · rows]
  ggml_tensor* cells = nullptr;        // I64 [rows]
  ggml_tensor* mask = nullptr;         // F16 [n_kv, rows, 1, 1]
  ggml_tensor* mask_f32 = nullptr;     // F32 [n_kv, rows] (QSA selection only)
  ggml_tensor* ple_rows = nullptr;     // I32 [ple_heads · rows]
  ggml_tensor* state_row = nullptr;    // I64 [1]: 0, the state tensors' only row
  ggml_tensor* row_zero = nullptr;     // I32 [1]: 0, for reading a weight row as F32
  ggml_tensor* out_ids = nullptr;      // I32 [outputs]: the rows the head computes
  ggml_tensor* cell_block = nullptr;   // I32 [n_kv] (QSA selection only)
  ggml_tensor* block_cells = nullptr;  // I32 [ratio · blocks]
  ggml_tensor* block_pos = nullptr;    // I32 [4 · blocks]
  ggml_tensor* block_bias = nullptr;   // F32 [blocks, rows]
  // Weights.
  ggml_tensor* token_embd = nullptr;
  ggml_tensor* ple_table = nullptr;        // I8 [row bytes, rows]
  ggml_tensor* ple_table_scale = nullptr;  // F32 [1]
  ggml_tensor* output = nullptr;
  ggml_tensor* output_hc_norm = nullptr;
  ggml_tensor* output_hc_down = nullptr;
  ggml_tensor* output_hc_up = nullptr;
  std::vector<Qwen38LayerTensors> layers;
  ggml_tensor* logits = nullptr;  // F32 [vocab, outputs]: the last node
  std::vector<ggml_tensor*> nodes;
  // Intermediates under llama.cpp's callback names ("l_last-7", ...).
  std::vector<std::pair<std::string, ggml_tensor*>> named;

  // The host-built inputs, in the order they are copied (those the shape
  // does not use left out).
  std::vector<ggml_tensor*> inputs() const;
  ggml_tensor* Named(std::string_view name) const;
};

// How many tensors a chunk's graph creates at most, for TensorArena.
std::size_t Qwen38GraphTensors(const model::Qwen38Profile& profile);

// Builds the chunk's graph on `arena`. Refused if the shape is not one the
// state holds, a type is not one the operations take, an expert stride is
// not a whole number of the expert slices' blocks, or the arena lacks room.
std::expected<Qwen38Graph, KernelFailure> BuildQwen38Graph(TensorArena& arena,
                                                           const model::Qwen38Profile& profile,
                                                           const model::Qwen38Binding& binding,
                                                           const Qwen38ChunkShape& shape,
                                                           const Qwen38GraphOptions& options);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_QWEN38_GRAPH_H_
