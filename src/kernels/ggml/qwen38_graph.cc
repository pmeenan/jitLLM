// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// A port of llama.cpp b29c606e2's src/models/qwen4exp.cpp graph (with the
// llm_graph_context and delta-net parts it calls), which is MIT: its
// structure, helpers and parameters follow upstream's closely, so the file
// carries GGML's notice (docs/licensing.md). qwen38_graph.h lists where it
// departs from upstream.

#include "kernels/ggml/qwen38_graph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"
#include "model/qwen38.h"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// A non-negative extent as a byte multiplier.
std::size_t U(std::int64_t n) { return static_cast<std::size_t>(n); }

class Builder {
 public:
  Builder(ggml_context* c, const model::Qwen38Profile& p, const model::Qwen38Binding& b,
          const Qwen38ChunkShape& s, Qwen38Graph& g)
      : c_(c), p_(p), b_(b), s_(s), g_(g) {}

  std::expected<void, KernelFailure> Leaves(const Qwen38GraphOptions& options);
  void Build();

 private:
  void Name(ggml_tensor* t, std::string_view name, int il) {
    g_.named.emplace_back(il >= 0 ? std::format("{}-{}", name, il) : std::string(name), t);
  }
  void Expand(ggml_tensor* t) { expanded_.push_back(t); }

  // GGML's row norms take packed input: a view of heads is made packed first
  // (every stride a dense tensor's, even over extents of 1, which
  // ggml_is_contiguous overlooks).
  ggml_tensor* Packed(ggml_tensor* x) {
    bool packed = x->nb[0] == ggml_type_size(x->type);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
      packed = packed && x->nb[i] == x->nb[i - 1] * static_cast<std::size_t>(x->ne[i - 1]);
    }
    return packed ? x : ggml_cont(c_, x);
  }
  // build_norm with LLM_NORM_RMS: rows scaled to unit RMS, then the weight.
  ggml_tensor* Norm(ggml_tensor* x, ggml_tensor* weight) {
    return ggml_mul(c_, ggml_rms_norm(c_, Packed(x), p_.rms_eps), weight);
  }
  // build_gdn_l2_norm (models.h): rms_norm(x, eps / n) / sqrt(n).
  ggml_tensor* L2Norm(ggml_tensor* x) {
    const auto n = static_cast<float>(x->ne[0]);
    return ggml_scale(c_, ggml_rms_norm(c_, Packed(x), p_.rms_eps / n), 1.0f / std::sqrt(n));
  }
  ggml_tensor* Rope(ggml_tensor* x, ggml_tensor* pos) {
    std::array<int, 4> sections = p_.rope_sections;
    return ggml_rope_multi(c_, x, pos, nullptr, static_cast<int>(p_.rope_dims), sections.data(),
                           GGML_ROPE_TYPE_IMROPE, 262144, p_.rope_base, 1.0f, 0.0f, 1.0f, 32.0f,
                           1.0f);
  }
  // An MXFP8 product y[n, t] = W x: jitLLM's vector product up to its
  // column bound, else the weights dequantized to BF16 for GGML's product.
  ggml_tensor* Linear(const Qwen38Mxfp8Tensors& w, ggml_tensor* x) {
    if (!ggml_is_contiguous(x)) {
      x = ggml_cont(c_, x);
    }
    if (x->ne[2] != 1 || x->ne[3] != 1) {
      x = ggml_reshape_2d(c_, x, x->ne[0], ggml_nelements(x) / x->ne[0]);
    }
    if (x->ne[1] <= kMxfp8VecColumns) {
      return Mxfp8MulMatVec(c_, w.codes, w.scales, x);
    }
    return ggml_mul_mat(c_, Mxfp8Dequant(c_, w.codes, w.scales), x);
  }
  // build_lora_mm_id with a per-expert scale (llama-graph.cpp:1545-1581).
  ggml_tensor* MulMatId(ggml_tensor* w, ggml_tensor* x, ggml_tensor* ids, ggml_tensor* scale) {
    ggml_tensor* res = ggml_mul_mat_id(c_, w, x, ids);
    const std::int64_t n_expert = scale->ne[0];
    const std::int64_t nt = x->ne[2];
    ggml_tensor* s = ggml_reshape_3d(c_, scale, 1, n_expert, 1);
    s = ggml_repeat_4d(c_, s, 1, n_expert, nt, 1);
    s = ggml_get_rows(c_, s, ids);
    return ggml_mul(c_, res, s);
  }
  // A state tensor's single row rewritten from `src` (packed, the row's size).
  ggml_tensor* StoreState(ggml_tensor* state, ggml_tensor* src) {
    ggml_tensor* rows = ggml_reshape_2d(c_, src, state->ne[0], 1);
    return ggml_set_rows(c_, state, rows, g_.state_row);
  }

  ggml_tensor* HcMix(ggml_tensor* x, ggml_tensor* w_norm, ggml_tensor* w_down, ggml_tensor* w_up,
                     ggml_tensor* w_inject, ggml_tensor** inject, int il);
  ggml_tensor* HcCombine(ggml_tensor* residual, ggml_tensor* block_out, ggml_tensor* inject);
  ggml_tensor* ConvStateAt(ggml_tensor* state, ggml_tensor* x, std::int64_t cols,
                           std::int64_t channels);
  ggml_tensor* Ple(const Qwen38LayerTensors& l, ggml_tensor* emb, ggml_tensor* hidden, int il);
  ggml_tensor* LinearAttention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il);
  ggml_tensor* QsaTopK(const Qwen38LayerTensors& l, ggml_tensor* cur, int il);
  ggml_tensor* Attention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il);
  ggml_tensor* Moe(const Qwen38LayerTensors& l, ggml_tensor* cur, int il);

  ggml_context* c_;
  const model::Qwen38Profile& p_;
  const model::Qwen38Binding& b_;
  const Qwen38ChunkShape& s_;
  Qwen38Graph& g_;
  std::vector<ggml_tensor*> expanded_;
};

std::expected<ggml_tensor*, KernelFailure> Leaf(ggml_context* c, const model::Qwen38Tensor& t,
                                                std::string_view role) {
  ggml_type type = GGML_TYPE_COUNT;
  if (t.plain) {
    // Checkpoint-layout bytes: I8 to GGML; F32 and I64 keep their type.
    if (t.type == "F8_E4M3" || t.type == "U8") {
      type = GGML_TYPE_I8;
    } else if (t.type == "F32") {
      type = GGML_TYPE_F32;
    } else if (t.type == "I64") {
      type = GGML_TYPE_I64;
    } else {
      return Rejected(std::format("{}: plain {} is not a type the graph reads", role, t.type));
    }
  } else {
    auto parsed = GgmlTypeOf(t.type);
    if (!parsed) {
      return std::unexpected(parsed.error());
    }
    type = *parsed;
  }
  std::array<std::int64_t, 4> ne = {1, 1, 1, 1};
  if (t.ne.empty() || t.ne.size() > 4) {
    return Rejected(std::format("{}: not a GGML shape", role));
  }
  for (std::size_t i = 0; i < t.ne.size(); ++i) {
    ne[i] = static_cast<std::int64_t>(t.ne[i]);
  }
  if (ne[0] % ggml_blck_size(type) != 0) {
    return Rejected(std::format("{}: rows are not whole blocks", role));
  }
  return ggml_new_tensor(c, type, static_cast<int>(t.ne.size()), ne.data());
}

std::expected<void, KernelFailure> Builder::Leaves(const Qwen38GraphOptions& options) {
  const std::int64_t n = s_.rows;
  g_.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, 4 * n);
  g_.cells = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
  g_.mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, s_.n_kv, n, 1, 1);
  g_.ple_rows = ggml_new_tensor_1d(c_, GGML_TYPE_I32, std::int64_t{p_.ple_heads()} * n);
  g_.state_row = ggml_new_tensor_1d(c_, GGML_TYPE_I64, 1);
  g_.row_zero = ggml_new_tensor_1d(c_, GGML_TYPE_I32, 1);
  g_.out_ids = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s_.outputs);
  if (s_.qsa_select) {
    const std::int64_t blocks = s_.qsa_blocks;
    g_.mask_f32 = ggml_new_tensor_2d(c_, GGML_TYPE_F32, s_.n_kv, n);
    g_.cell_block = ggml_new_tensor_1d(c_, GGML_TYPE_I32, s_.n_kv);
    g_.block_cells = ggml_new_tensor_1d(c_, GGML_TYPE_I32, std::int64_t{p_.indexer_ratio} * blocks);
    g_.block_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, 4 * blocks);
    g_.block_bias = ggml_new_tensor_2d(c_, GGML_TYPE_F32, blocks, n);
  }

  const auto leaf = [&](ggml_tensor*& into, const model::Qwen38Tensor& t,
                        std::string_view role) -> std::expected<void, KernelFailure> {
    auto made = Leaf(c_, t, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    into = *made;
    return {};
  };
  const auto mx = [&](Qwen38Mxfp8Tensors& into, const model::Qwen38Mxfp8& t,
                      std::string_view role) -> std::expected<void, KernelFailure> {
    if (auto a = leaf(into.codes, t.codes, role); !a) {
      return a;
    }
    return leaf(into.scales, t.scales, role);
  };
#define JITLLM_LEAF(into, tensor)                       \
  if (auto made = leaf(into, tensor, #tensor); !made) { \
    return made;                                        \
  }
#define JITLLM_MX(into, tensor)                       \
  if (auto made = mx(into, tensor, #tensor); !made) { \
    return made;                                      \
  }
  JITLLM_LEAF(g_.token_embd, b_.token_embd)
  JITLLM_LEAF(g_.ple_table, b_.ple_table)
  JITLLM_LEAF(g_.ple_table_scale, b_.ple_table_scale)
  JITLLM_LEAF(g_.output, b_.output)
  JITLLM_LEAF(g_.output_hc_norm, b_.output_hc_norm)
  JITLLM_LEAF(g_.output_hc_down, b_.output_hc_down)
  JITLLM_LEAF(g_.output_hc_up, b_.output_hc_up)
  if (!options.expert_stride.empty() && options.expert_stride.size() != p_.layers) {
    return Rejected("an expert stride per layer, or none");
  }
  g_.layers.resize(p_.layers);
  for (std::uint32_t il = 0; il < p_.layers; ++il) {
    const model::Qwen38Layer& w = b_.layers[il];
    Qwen38LayerTensors& l = g_.layers[il];
    JITLLM_LEAF(l.hc_attn_norm, w.hc_attn_norm)
    JITLLM_LEAF(l.hc_attn_down, w.hc_attn_down)
    JITLLM_LEAF(l.hc_attn_up, w.hc_attn_up)
    JITLLM_LEAF(l.hc_attn_inject, w.hc_attn_inject)
    JITLLM_LEAF(l.hc_ffn_norm, w.hc_ffn_norm)
    JITLLM_LEAF(l.hc_ffn_down, w.hc_ffn_down)
    JITLLM_LEAF(l.hc_ffn_up, w.hc_ffn_up)
    JITLLM_LEAF(l.hc_ffn_inject, w.hc_ffn_inject)
    if (w.linear) {
      JITLLM_MX(l.qkv, w.qkv)
      JITLLM_MX(l.z, w.z)
      JITLLM_MX(l.beta, w.beta)
      JITLLM_MX(l.alpha, w.alpha)
      JITLLM_MX(l.ssm_out, w.ssm_out)
      JITLLM_LEAF(l.dt_bias, w.dt_bias)
      JITLLM_LEAF(l.ssm_a, w.ssm_a)
      JITLLM_LEAF(l.conv1d, w.conv1d)
      JITLLM_LEAF(l.ssm_norm, w.ssm_norm)
      const std::int64_t d = p_.lin_head_dim;
      l.conv_state =
          ggml_new_tensor_2d(c_, GGML_TYPE_F32, std::int64_t{p_.conv - 1} * p_.conv_channels(), 1);
      l.recurrent = ggml_new_tensor_2d(c_, GGML_TYPE_F32, d * d * p_.lin_v_heads, 1);
    } else {
      JITLLM_MX(l.q, w.q)
      JITLLM_MX(l.k, w.k)
      JITLLM_MX(l.v, w.v)
      JITLLM_MX(l.o, w.o)
      JITLLM_MX(l.idx_qk, w.idx_qk)
      JITLLM_LEAF(l.q_norm, w.q_norm)
      JITLLM_LEAF(l.k_norm, w.k_norm)
      JITLLM_LEAF(l.idx_q_norm, w.idx_q_norm)
      JITLLM_LEAF(l.idx_k_norm, w.idx_k_norm)
      const std::int64_t kv = std::int64_t{p_.head_dim} * p_.kv_heads;
      l.cache_k = ggml_new_tensor_2d(c_, GGML_TYPE_F16, kv, s_.cells);
      l.cache_v = ggml_new_tensor_2d(c_, GGML_TYPE_F16, kv, s_.cells);
      l.cache_idx = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.indexer_head_dim, s_.cells);
    }
    if (il == p_.ple_layer) {
      JITLLM_LEAF(l.ple_key, w.ple_key)
      JITLLM_LEAF(l.ple_value, w.ple_value)
      JITLLM_LEAF(l.ple_norm_key, w.ple_norm_key)
      JITLLM_LEAF(l.ple_norm_query, w.ple_norm_query)
      JITLLM_LEAF(l.ple_norm_conv, w.ple_norm_conv)
      JITLLM_LEAF(l.ple_conv1d, w.ple_conv1d)
      l.ple_state =
          ggml_new_tensor_2d(c_, GGML_TYPE_F32, std::int64_t{p_.ple_history()} * p_.hc_width(), 1);
    }
    JITLLM_LEAF(l.router, w.router)
    JITLLM_LEAF(l.shared_gate, w.shared_gate)
    JITLLM_MX(l.gate_shexp, w.gate_shexp)
    JITLLM_MX(l.up_shexp, w.up_shexp)
    JITLLM_MX(l.down_shexp, w.down_shexp)
    JITLLM_LEAF(l.gate_exps_scale, w.gate_exps_scale)
    JITLLM_LEAF(l.up_exps_scale, w.up_exps_scale)
    JITLLM_LEAF(l.down_exps_scale, w.down_exps_scale)
    const std::uint64_t stride = options.expert_stride.empty() ? 0 : options.expert_stride[il];
    const auto experts = [&](ggml_tensor*& into,
                             const model::Qwen38Tensor& t) -> std::expected<void, KernelFailure> {
      auto type = GgmlTypeOf(t.type);
      if (!type) {
        return std::unexpected(type.error());
      }
      into = ggml_new_tensor_3d(c_, *type, static_cast<std::int64_t>(t.ne[0]),
                                static_cast<std::int64_t>(t.ne[1]), p_.experts);
      const std::size_t slice = into->nb[2];
      if (stride != 0) {
        if (stride < slice || stride % ggml_type_size(*type) != 0 || stride % 16 != 0) {
          return Rejected(std::format(
              "layer {}: an expert stride of {} bytes does not hold whole, 16-byte aligned {} "
              "slices",
              il, stride, t.type));
        }
        into->nb[2] = stride;
        into->nb[3] = stride * static_cast<std::size_t>(p_.experts);
      }
      // Rows short of a 512-element step read past the slice: into the
      // next slice of the group or, after the last row, into the readable
      // bytes the artifact reserves (and zeroes) for it. Marked only where
      // the stride provably holds them for every expert, the last one's
      // included; otherwise the quantized products refuse the short rows.
      const std::int64_t k = into->ne[0];
      if (k % 512 != 0) {
        const std::uint64_t over = ggml_row_size(*type, 512 - (k % 512));
        if (stride != 0 && t.readable >= slice + over && t.group_offset <= stride &&
            t.readable <= stride - t.group_offset) {
          MarkRowPaddingReadable(into);
        }
      }
      return {};
    };
    if (auto e = experts(l.gate_exps, w.gate_exps); !e) {
      return e;
    }
    if (auto e = experts(l.up_exps, w.up_exps); !e) {
      return e;
    }
    if (auto e = experts(l.down_exps, w.down_exps); !e) {
      return e;
    }
  }
#undef JITLLM_LEAF
#undef JITLLM_MX
  return {};
}

// build_hc_mix (qwen4exp.cpp:266-312).
ggml_tensor* Builder::HcMix(ggml_tensor* x, ggml_tensor* w_norm, ggml_tensor* w_down,
                            ggml_tensor* w_up, ggml_tensor* w_inject, ggml_tensor** inject,
                            int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t hc_dim = p_.hc_width();
  const std::int64_t nt = x->ne[2];
  const std::int64_t n_embd = p_.width;
  ggml_tensor* xn = ggml_rms_norm(c_, x, p_.rms_eps);
  xn = ggml_reshape_2d(c_, xn, hc_dim, nt);
  xn = ggml_mul(c_, xn, w_norm);
  ggml_tensor* lo = ggml_mul_mat(c_, w_down, xn);
  lo = ggml_silu(c_, ggml_scale(c_, lo, 1.0f / static_cast<float>(hc)));
  ggml_tensor* gate = ggml_sigmoid(c_, ggml_mul_mat(c_, w_up, lo));
  ggml_tensor* gated = ggml_mul(c_, xn, gate);
  gated = ggml_reshape_3d(c_, gated, n_embd, hc, nt);
  const std::size_t row = ggml_row_size(gated->type, n_embd);
  const std::size_t stride = row * U(hc);
  ggml_tensor* mixed = ggml_cont(c_, ggml_view_2d(c_, gated, n_embd, nt, stride, 0));
  for (std::int64_t k = 1; k < hc; ++k) {
    ggml_tensor* s = ggml_view_2d(c_, gated, n_embd, nt, stride, row * U(k));
    mixed = ggml_add(c_, mixed, s);
  }
  mixed = ggml_scale(c_, mixed, 1.0f / static_cast<float>(hc));
  Name(mixed, "hc_mixed", il);
  if (inject != nullptr) {
    *inject = ggml_mul_mat(c_, w_inject, xn);
  }
  return mixed;
}

// build_hc_combine (qwen4exp.cpp:314-334).
ggml_tensor* Builder::HcCombine(ggml_tensor* residual, ggml_tensor* block_out,
                                ggml_tensor* inject) {
  const std::int64_t hc = p_.hc;
  const std::int64_t nt = residual->ne[2];
  ggml_tensor* w = ggml_sigmoid(c_, ggml_scale(c_, inject, 1.0f / static_cast<float>(hc)));
  w = ggml_scale(c_, w, 2.0f);
  w = ggml_reshape_3d(c_, w, 1, hc, nt);
  ggml_tensor* b = ggml_reshape_3d(c_, block_out, p_.width, 1, nt);
  b = ggml_repeat_4d(c_, b, p_.width, hc, nt, 1);
  return ggml_add(c_, residual, ggml_mul(c_, b, w));
}

// build_conv_state_at (qwen4exp.cpp:1117-1169), one sequence, no rollback
// slots: the history [cols, channels] then the chunk's rows, time fastest;
// the last `cols` columns become the new history.
ggml_tensor* Builder::ConvStateAt(ggml_tensor* state, ggml_tensor* x, std::int64_t cols,
                                  std::int64_t channels) {
  ggml_tensor* history = ggml_reshape_3d(c_, state, cols, channels, 1);
  // The chunk's rows transposed into packed rows first: the concatenation
  // takes contiguous rows only.
  ggml_tensor* input = ggml_concat(c_, history, ggml_cont(c_, ggml_transpose(c_, x)), 0);
  const std::int64_t start = input->ne[0] - cols;
  ggml_tensor* tail = ggml_view_3d(c_, input, cols, channels, 1, input->nb[1], input->nb[2],
                                   ggml_row_size(input->type, start));
  Expand(StoreState(state, ggml_cont(c_, tail)));
  return input;
}

// build_ple (qwen4exp.cpp:1192-1283).
ggml_tensor* Builder::Ple(const Qwen38LayerTensors& l, ggml_tensor* emb, ggml_tensor* hidden,
                          int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t hc_dim = p_.hc_width();
  const std::int64_t n_embd = p_.width;
  const std::int64_t nt = hidden->ne[2];
  ggml_tensor* key = ggml_mul_mat(c_, l.ple_key, emb);
  ggml_tensor* value = ggml_mul_mat(c_, l.ple_value, emb);
  const auto grouped_norm = [&](ggml_tensor* x, ggml_tensor* w) {
    ggml_tensor* t = ggml_reshape_3d(c_, x, n_embd, hc, nt);
    t = ggml_rms_norm(c_, t, p_.rms_eps);
    t = ggml_reshape_2d(c_, t, hc_dim, nt);
    t = ggml_mul(c_, t, w);
    return ggml_reshape_3d(c_, t, n_embd, hc, nt);
  };
  key = grouped_norm(key, l.ple_norm_key);
  ggml_tensor* query = grouped_norm(hidden, l.ple_norm_query);
  ggml_tensor* s = ggml_sum_rows(c_, ggml_mul(c_, key, query));
  s = ggml_scale(c_, s, 1.0f / std::sqrt(static_cast<float>(n_embd)));
  ggml_tensor* mag = ggml_sqrt(c_, ggml_clamp(c_, ggml_abs(c_, s), 1e-6f, INFINITY));
  ggml_tensor* gate = ggml_sigmoid(c_, ggml_mul(c_, ggml_sgn(c_, s), mag));
  Name(gate, "ple_gate", il);
  ggml_tensor* v3 = ggml_reshape_3d(c_, value, n_embd, 1, nt);
  v3 = ggml_repeat_4d(c_, v3, n_embd, hc, nt, 1);
  ggml_tensor* gated = ggml_mul(c_, v3, gate);
  ggml_tensor* normalized = grouped_norm(ggml_reshape_2d(c_, gated, hc_dim, nt), l.ple_norm_conv);
  normalized = ggml_reshape_2d(c_, normalized, hc_dim, nt);
  const std::int64_t kern = p_.ple_conv;
  const std::int64_t dil = p_.ngram;
  const std::int64_t hist = (kern - 1) * dil;
  ggml_tensor* padded =
      ConvStateAt(l.ple_state, ggml_reshape_3d(c_, normalized, hc_dim, nt, 1), hist, hc_dim);
  ggml_tensor* conv_out = nullptr;
  for (std::int64_t k = 0; k < kern; ++k) {
    const std::int64_t start = hist - ((kern - 1 - k) * dil);
    ggml_tensor* shifted = ggml_cont(
        c_, ggml_transpose(c_, ggml_view_3d(c_, padded, nt, hc_dim, 1, padded->nb[1], padded->nb[2],
                                            ggml_row_size(padded->type, start))));
    ggml_tensor* wk =
        ggml_cont(c_, ggml_view_2d(c_, l.ple_conv1d, 1, hc_dim, l.ple_conv1d->nb[1],
                                   static_cast<std::size_t>(k) * l.ple_conv1d->nb[0]));
    wk = ggml_reshape_1d(c_, wk, hc_dim);
    ggml_tensor* term = ggml_mul(c_, shifted, wk);
    conv_out = conv_out != nullptr ? ggml_add(c_, conv_out, term) : term;
  }
  conv_out = ggml_silu(c_, conv_out);
  conv_out = ggml_reshape_3d(c_, ggml_cont(c_, conv_out), n_embd, hc, nt);
  Name(conv_out, "ple_conv_out", il);
  return ggml_add(c_, hidden, ggml_add(c_, gated, conv_out));
}

// build_layer_attn_linear (qwen4exp.cpp:847-972) with the fused gated delta
// rule (delta-net-base.cpp build_delta_net_fused, K = 1).
ggml_tensor* Builder::LinearAttention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il) {
  const std::int64_t nt = cur->ne[1];
  const std::int64_t d = p_.lin_head_dim;
  const std::int64_t hk = p_.lin_k_heads;
  const std::int64_t hv = p_.lin_v_heads;
  const std::int64_t channels = p_.conv_channels();
  ggml_tensor* qkv = Linear(l.qkv, cur);
  qkv = ggml_reshape_3d(c_, qkv, qkv->ne[0], nt, 1);
  ggml_tensor* z = Linear(l.z, cur);
  ggml_tensor* beta = Linear(l.beta, cur);
  beta = ggml_reshape_4d(c_, beta, 1, hv, nt, 1);
  beta = ggml_sigmoid(c_, beta);
  ggml_tensor* alpha = Linear(l.alpha, cur);
  alpha = ggml_reshape_3d(c_, alpha, hv, nt, 1);
  ggml_tensor* alpha_sp = ggml_softplus(c_, ggml_add(c_, alpha, l.dt_bias));
  ggml_tensor* gate = ggml_mul(c_, alpha_sp, l.ssm_a);
  gate = ggml_reshape_4d(c_, gate, 1, hv, nt, 1);
  ggml_tensor* conv_input = ConvStateAt(l.conv_state, qkv, p_.conv - 1, channels);
  ggml_tensor* state = ggml_reshape_4d(c_, l.recurrent, d, d, hv, 1);
  ggml_tensor* conv = ggml_silu(c_, ggml_ssm_conv(c_, conv_input, l.conv1d));
  // A head's stride (views' nb1), a token's (nb2) and the chunk's (nb3).
  const std::size_t head_stride = ggml_row_size(conv->type, d);
  const std::size_t token_stride = ggml_row_size(conv->type, channels);
  const std::size_t chunk_stride = token_stride * U(nt);
  ggml_tensor* q = ggml_view_4d(c_, conv, d, hk, nt, 1, head_stride, token_stride, chunk_stride, 0);
  ggml_tensor* k = ggml_view_4d(c_, conv, d, hk, nt, 1, head_stride, token_stride, chunk_stride,
                                ggml_row_size(conv->type, d * hk));
  ggml_tensor* v = ggml_view_4d(c_, conv, d, hv, nt, 1, head_stride, token_stride, chunk_stride,
                                ggml_row_size(conv->type, 2 * d * hk));
  q = L2Norm(q);
  k = L2Norm(k);
  ggml_tensor* result = ggml_gated_delta_net(c_, q, k, v, gate, beta, state, 1);
  ggml_tensor* output = ggml_view_4d(c_, result, d, hv, nt, 1, ggml_row_size(result->type, d),
                                     ggml_row_size(result->type, d * hv),
                                     ggml_row_size(result->type, d * hv * nt), 0);
  ggml_tensor* new_state = ggml_view_4d(
      c_, result, d, d, hv, 1, ggml_row_size(result->type, d), ggml_row_size(result->type, d * d),
      ggml_row_size(result->type, d * d * hv), ggml_row_size(result->type, d * hv * nt));
  Expand(StoreState(l.recurrent, new_state));
  ggml_tensor* z4 = ggml_reshape_4d(c_, z, d, hv, nt, 1);
  // build_norm_gated: the sigmoid output gate.
  ggml_tensor* normed = ggml_mul(c_, Norm(output, l.ssm_norm), ggml_sigmoid(c_, z4));
  ggml_tensor* flat = ggml_reshape_2d(c_, normed, d * hv, nt);
  ggml_tensor* out = Linear(l.ssm_out, flat);
  Name(out, "linear_attn_out", il);
  return out;
}

// build_qsa_top_k (qwen4exp.cpp:525-674), one stream, the per-block bias.
ggml_tensor* Builder::QsaTopK(const Qwen38LayerTensors& l, ggml_tensor* cur, int il) {
  const std::int64_t idx_dim = p_.indexer_head_dim;
  const std::int64_t n_idx_h = p_.indexer_heads;
  const std::int64_t r = p_.indexer_ratio;
  const std::int64_t n_kv = s_.n_kv;
  const std::int64_t n_blocks = s_.qsa_blocks;
  const std::int64_t nt = cur->ne[1];
  ggml_tensor* qk = Linear(l.idx_qk, cur);  // [q heads · dim | dim, nt]
  ggml_tensor* k_raw =
      ggml_view_2d(c_, qk, idx_dim, nt, qk->nb[1], ggml_row_size(qk->type, n_idx_h * idx_dim));
  // The cached keys are raw: pooling precedes their norm and rotation.
  Expand(ggml_set_rows(c_, l.cache_idx, Packed(k_raw), g_.cells));
  if (!s_.qsa_select) {
    return nullptr;
  }
  ggml_tensor* k_all = ggml_view_2d(c_, l.cache_idx, idx_dim, n_kv, l.cache_idx->nb[1], 0);
  ggml_tensor* members = ggml_get_rows(c_, k_all, g_.block_cells);
  members = ggml_reshape_4d(c_, members, idx_dim, r, n_blocks, 1);
  ggml_tensor* pooled = nullptr;
  for (std::int64_t i = 0; i < r; ++i) {
    ggml_tensor* slice =
        ggml_cont(c_, ggml_view_3d(c_, members, idx_dim, n_blocks, 1, members->nb[2],
                                   members->nb[3], static_cast<std::size_t>(i) * members->nb[1]));
    pooled = pooled != nullptr ? ggml_add(c_, pooled, slice) : slice;
  }
  pooled = ggml_scale(c_, pooled, 1.0f / static_cast<float>(r));
  pooled = ggml_reshape_3d(c_, pooled, idx_dim, n_blocks, 1);
  pooled = Norm(pooled, l.idx_k_norm);
  pooled = ggml_reshape_3d(c_, pooled, idx_dim, 1, n_blocks);
  pooled = Rope(pooled, g_.block_pos);
  pooled = ggml_reshape_3d(c_, pooled, idx_dim, n_blocks, 1);
  ggml_tensor* q =
      ggml_view_3d(c_, qk, idx_dim, n_idx_h, nt, ggml_row_size(qk->type, idx_dim), qk->nb[1], 0);
  q = Norm(q, l.idx_q_norm);
  q = Rope(q, g_.positions);
  ggml_tensor* score = ggml_mul_mat(c_, pooled, ggml_reshape_3d(c_, q, idx_dim, n_idx_h * nt, 1));
  score = ggml_reshape_4d(c_, score, n_blocks, n_idx_h, nt, 1);
  score = ggml_relu(c_, score);
  ggml_tensor* summed = nullptr;
  for (std::int64_t h = 0; h < n_idx_h; ++h) {
    ggml_tensor* slice = ggml_view_3d(c_, score, n_blocks, nt, 1, score->nb[2], score->nb[3],
                                      static_cast<std::size_t>(h) * score->nb[1]);
    summed = summed != nullptr ? ggml_add(c_, summed, slice) : ggml_cont(c_, slice);
  }
  score = ggml_add(c_, summed, ggml_reshape_3d(c_, g_.block_bias, n_blocks, nt, 1));
  Name(score, "indexer_score", il);
  ggml_tensor* expanded =
      ggml_get_rows(c_, ggml_cont(c_, ggml_permute(c_, score, 1, 0, 2, 3)), g_.cell_block);
  expanded = ggml_cont(c_, ggml_permute(c_, expanded, 1, 0, 2, 3));
  expanded = ggml_add(c_, expanded, ggml_reshape_3d(c_, g_.mask_f32, n_kv, nt, 1));
  const std::int64_t width = std::min<std::int64_t>(n_kv, std::int64_t{p_.indexer_budget} + r - 1);
  ggml_tensor* top_k = ggml_cont(c_, ggml_top_k(c_, expanded, static_cast<int>(width)));
  return ggml_reshape_4d(c_, top_k, width, nt, 1, 1);
}

// build_layer_attn with build_attn_qsa (qwen4exp.cpp:676-845).
ggml_tensor* Builder::Attention(const Qwen38LayerTensors& l, ggml_tensor* cur, int il) {
  const std::int64_t d = p_.head_dim;
  const std::int64_t heads = p_.heads;
  const std::int64_t kvh = p_.kv_heads;
  const std::int64_t nt = cur->ne[1];
  const std::int64_t n_kv = s_.n_kv;
  ggml_tensor* top_k = QsaTopK(l, cur, il);
  ggml_tensor* q_full = Linear(l.q, cur);  // [(d · 2) · heads, nt]: per head, q then its gate
  const std::size_t f = ggml_element_size(q_full);
  const std::size_t per_head = f * U(d) * 2;  // q then its gate
  ggml_tensor* q = ggml_view_3d(c_, q_full, d, heads, nt, per_head, per_head * U(heads), 0);
  q = Norm(q, l.q_norm);
  ggml_tensor* k = Linear(l.k, cur);
  ggml_tensor* v = Linear(l.v, cur);
  k = ggml_reshape_3d(c_, k, d, kvh, nt);
  k = Norm(k, l.k_norm);
  ggml_tensor* gate =
      ggml_view_3d(c_, q_full, d, heads, nt, per_head, per_head * U(heads), f * U(d));
  gate = ggml_cont_2d(c_, gate, d * heads, nt);
  v = ggml_reshape_3d(c_, v, d, kvh, nt);
  q = Rope(q, g_.positions);
  k = Rope(k, g_.positions);
  Expand(q);
  Expand(v);
  Expand(k);
  // llama_kv_cache::cpy_k and cpy_v: merge the heads, store at the cells.
  Expand(ggml_set_rows(c_, l.cache_k, ggml_reshape_2d(c_, k, d * kvh, nt), g_.cells));
  Expand(ggml_set_rows(c_, l.cache_v, ggml_reshape_2d(c_, v, d * kvh, nt), g_.cells));
  ggml_tensor* kq_mask = g_.mask;
  if (top_k != nullptr) {
    // build_attn_qsa's mask: -inf everywhere but the selected cells, plus
    // the causal mask.
    ggml_tensor* all = ggml_fill(c_, kq_mask, -INFINITY);
    all = ggml_view_4d(c_, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1],
                       all->nb[2], 0);
    ggml_tensor* top_k_3d =
        ggml_view_4d(c_, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1],
                     top_k->nb[2], static_cast<std::size_t>(top_k->ne[3]) * top_k->nb[3], 0);
    ggml_tensor* zeros =
        ggml_new_tensor_4d(c_, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(c_, zeros, 0.0f);
    ggml_tensor* masked = ggml_set_rows(c_, all, zeros, top_k_3d);
    masked = ggml_view_4d(c_, masked, masked->ne[1], masked->ne[2], 1, masked->ne[3], masked->nb[2],
                          masked->nb[3], masked->nb[3], 0);
    kq_mask = ggml_add(c_, masked, kq_mask);
  }
  // get_k / get_v over the first n_kv cells: [d, n_kv, kv heads].
  const std::size_t hrow = ggml_row_size(l.cache_k->type, d);
  const std::size_t crow = l.cache_k->nb[1];
  ggml_tensor* kc = ggml_view_3d(c_, l.cache_k, d, n_kv, kvh, crow, hrow, 0);
  ggml_tensor* vc = ggml_view_3d(c_, l.cache_v, d, n_kv, kvh, crow, hrow, 0);
  ggml_tensor* qp = ggml_permute(c_, q, 0, 2, 1, 3);
  const float scale = 1.0f / std::sqrt(static_cast<float>(d));
  ggml_tensor* attn = ggml_flash_attn_ext(c_, qp, kc, vc, kq_mask, scale, 0.0f, 0.0f);
  ggml_prec_set_acc(attn, GGML_PREC_F32);
  attn = ggml_reshape_2d(c_, attn, attn->ne[0] * attn->ne[1], attn->ne[2] * attn->ne[3]);
  Name(attn, "attn_pregate", il);
  attn = ggml_mul(c_, attn, ggml_sigmoid(c_, gate));
  ggml_tensor* out = Linear(l.o, attn);
  Name(out, "attn_output", il);
  return out;
}

// build_layer_ffn with build_moe_ffn and build_ffn (qwen4exp.cpp:974-1022,
// llama-graph.cpp:1748-2380): softmax routing, the top experts' weights
// renormalized, SwiGLU experts, and the shared expert behind its sigmoid
// gate.
ggml_tensor* Builder::Moe(const Qwen38LayerTensors& l, ggml_tensor* cur, int il) {
  const std::int64_t n_embd = cur->ne[0];
  const std::int64_t nt = cur->ne[1];
  const std::int64_t n_expert = p_.experts;
  const std::int64_t used = p_.experts_used;
  ggml_tensor* logits = ggml_mul_mat(c_, l.router, cur);
  ggml_tensor* probs = ggml_soft_max(c_, logits);
  ggml_tensor* selected = ggml_argsort_top_k(c_, probs, static_cast<int>(used));
  Name(selected, "ffn_moe_topk", il);
  probs = ggml_reshape_3d(c_, probs, 1, n_expert, nt);
  ggml_tensor* weights = ggml_get_rows(c_, probs, selected);
  weights = ggml_reshape_2d(c_, weights, used, nt);
  ggml_tensor* sum = ggml_sum_rows(c_, weights);
  sum = ggml_clamp(c_, sum, 6.103515625e-5f, INFINITY);
  weights = ggml_div(c_, weights, sum);
  weights = ggml_reshape_3d(c_, weights, 1, used, nt);
  Expand(weights);
  ggml_tensor* x = ggml_reshape_3d(c_, cur, n_embd, 1, nt);
  ggml_tensor* up = MulMatId(l.up_exps, x, selected, l.up_exps_scale);
  ggml_tensor* gate = MulMatId(l.gate_exps, x, selected, l.gate_exps_scale);
  ggml_tensor* act = ggml_swiglu_split(c_, gate, up);
  ggml_tensor* experts = MulMatId(l.down_exps, act, selected, l.down_exps_scale);
  experts = ggml_mul(c_, experts, weights);
  Expand(experts);
  std::vector<ggml_tensor*> views;
  for (std::int64_t i = 0; i < used; ++i) {
    views.push_back(ggml_view_2d(c_, experts, n_embd, nt, experts->nb[2],
                                 static_cast<std::size_t>(i) * experts->nb[1]));
    Expand(views.back());
  }
  ggml_tensor* moe_out = views[0];
  for (std::size_t i = 1; i < views.size(); ++i) {
    moe_out = ggml_add(c_, moe_out, views[i]);
    Expand(moe_out);
  }
  if (used == 1) {
    moe_out = ggml_cont(c_, moe_out);
  }
  Name(moe_out, "ffn_moe_out", il);
  ggml_tensor* sh_up = Linear(l.up_shexp, cur);
  ggml_tensor* sh_gate = Linear(l.gate_shexp, cur);
  ggml_tensor* sh = Linear(l.down_shexp, ggml_swiglu_split(c_, sh_gate, sh_up));
  // The shared expert's gate, one value a token: GGML's products refuse a
  // one-row output (its rows are not 8-byte aligned), so the dot product is
  // the gate row (read as F32) times each token, summed.
  ggml_tensor* gate_row =
      ggml_get_rows(c_, ggml_reshape_2d(c_, l.shared_gate, n_embd, 1), g_.row_zero);
  ggml_tensor* shared_gate = ggml_sigmoid(c_, ggml_sum_rows(c_, ggml_mul(c_, cur, gate_row)));
  sh = ggml_mul(c_, sh, shared_gate);
  Name(sh, "ffn_shexp_gated", il);
  return ggml_add(c_, moe_out, sh);
}

void Builder::Build() {
  const std::int64_t nt = s_.rows;
  const std::int64_t hc = p_.hc;
  ggml_tensor* inpl = ggml_get_rows(c_, g_.token_embd, g_.tokens);
  Expand(inpl);
  ggml_tensor* ple = Nvfp4Rows(c_, g_.ple_table, g_.ple_rows, g_.ple_table_scale, p_.ple_row);
  ple = ggml_reshape_2d(c_, ple, p_.ple_width(), nt);
  Name(ple, "ple_embd", -1);
  Expand(ple);
  ggml_tensor* res =
      ggml_repeat_4d(c_, ggml_reshape_3d(c_, inpl, p_.width, 1, nt), p_.width, hc, nt, 1);
  Name(res, "hc_init", -1);
  for (std::uint32_t il_u = 0; il_u < p_.layers; ++il_u) {
    const int il = static_cast<int>(il_u);
    const Qwen38LayerTensors& l = g_.layers[il_u];
    if (il_u == p_.ple_layer) {
      res = Ple(l, ple, res, il);
      Name(res, "ple_out", il);
    }
    ggml_tensor* inject = nullptr;
    ggml_tensor* cur =
        HcMix(res, l.hc_attn_norm, l.hc_attn_down, l.hc_attn_up, l.hc_attn_inject, &inject, il);
    Expand(cur);
    cur = b_.layers[il_u].linear ? LinearAttention(l, cur, il) : Attention(l, cur, il);
    res = HcCombine(res, cur, inject);
    cur = HcMix(res, l.hc_ffn_norm, l.hc_ffn_down, l.hc_ffn_up, l.hc_ffn_inject, &inject, il);
    cur = Moe(l, cur, il);
    Name(cur, "ffn_out", il);
    res = HcCombine(res, cur, inject);
    Name(res, "l_last", il);
  }
  // The rows the head computes.
  ggml_tensor* flat = ggml_reshape_2d(c_, res, p_.hc_width(), nt);
  flat = ggml_get_rows(c_, flat, g_.out_ids);
  res = ggml_reshape_3d(c_, flat, p_.width, hc, s_.outputs);
  ggml_tensor* cur =
      HcMix(res, g_.output_hc_norm, g_.output_hc_down, g_.output_hc_up, nullptr, nullptr, -1);
  Name(cur, "result_norm", -1);
  g_.logits = ggml_mul_mat(c_, g_.output, cur);
  Name(g_.logits, "result_output", -1);
  Expand(g_.logits);
  g_.nodes = GraphOrder(expanded_);
}

}  // namespace

Qwen38ChunkShape Qwen38ShapeOf(const model::Qwen38StateLayout& state,
                               const model::Qwen38ChunkInputs& chunk, std::int64_t outputs) {
  return {.rows = chunk.rows,
          .n_kv = chunk.n_kv,
          .cells = state.cells,
          .outputs = outputs,
          .qsa_select = chunk.qsa_select,
          .qsa_blocks = chunk.qsa_select ? chunk.qsa.blocks : 0};
}

std::vector<ggml_tensor*> Qwen38Graph::inputs() const {
  std::vector<ggml_tensor*> all = {tokens,   positions, cells,    mask,
                                   ple_rows, state_row, row_zero, out_ids};
  for (ggml_tensor* t : {mask_f32, cell_block, block_cells, block_pos, block_bias}) {
    if (t != nullptr) {
      all.push_back(t);
    }
  }
  return all;
}

ggml_tensor* Qwen38Graph::Named(std::string_view name) const {
  for (const auto& [n, t] : named) {
    if (n == name) {
      return t;
    }
  }
  return nullptr;
}

std::size_t Qwen38GraphTensors(const model::Qwen38Profile& profile) {
  // Leaves: about 40 per layer; nodes: at most about 200 per layer (a QSA
  // layer with its selection, MoE included). Rounded up generously.
  return 512 + (std::size_t{profile.layers} * 384);
}

std::expected<Qwen38Graph, KernelFailure> BuildQwen38Graph(TensorArena& arena,
                                                           const model::Qwen38Profile& profile,
                                                           const model::Qwen38Binding& binding,
                                                           const Qwen38ChunkShape& shape,
                                                           const Qwen38GraphOptions& options) {
  const Qwen38ChunkShape& s = shape;
  if (s.rows <= 0 || s.cells <= 0 || s.n_kv < 256 || s.n_kv > s.cells || s.n_kv % 256 != 0 ||
      s.outputs <= 0 || s.outputs > s.rows || (s.qsa_select && s.qsa_blocks <= 0)) {
    return Rejected("not a Qwen3.8 chunk shape the state holds");
  }
  // QSA selects exactly when the cells pass the budget (plus the tail block
  // model/qwen38.cc's Qwen38Chunk keeps), over every cell's pooled block.
  const std::int64_t ratio = profile.indexer_ratio;
  const bool past_budget = ratio > 0 && s.n_kv > std::int64_t{profile.indexer_budget} + ratio - 1;
  if (ratio <= 0 || s.qsa_select != past_budget ||
      s.qsa_blocks != (s.qsa_select ? (s.n_kv + ratio - 1) / ratio : s.qsa_blocks)) {
    return Rejected("a QSA selection that is not the indexer budget's");
  }
  if (binding.layers.size() != profile.layers) {
    return Rejected("the binding is not the profile's");
  }
  if (auto room = arena.Reserve(Qwen38GraphTensors(profile)); !room) {
    return std::unexpected(room.error());
  }
  Qwen38Graph g;
  Builder builder(arena.context(), profile, binding, shape, g);
  if (auto leaves = builder.Leaves(options); !leaves) {
    return std::unexpected(leaves.error());
  }
  builder.Build();
  return g;
}

}  // namespace jitllm::kernels::ggml
