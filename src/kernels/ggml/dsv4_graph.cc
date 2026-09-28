// SPDX-FileCopyrightText: 2023-2026 The ggml authors
// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: MIT AND Apache-2.0

// A port of llama.cpp b29c606e2's src/models/deepseek4.cpp graph (and the
// llm_graph_context and llama_kv_cache parts it calls, the Hadamard matrix
// among them), which is MIT: its structure, helpers and parameters follow
// upstream's closely, so the file carries GGML's notice (docs/licensing.md).

#include "kernels/ggml/dsv4_graph.h"

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
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/tensors.h"
#include "model/dsv4.h"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// dsv4_rope_attn_factor (deepseek4.cpp:11-17).
float RopeAttnFactor(float freq_scale, float ext_factor) {
  if (ext_factor == 0.0f) {
    return 1.0f;
  }
  return 1.0f / (1.0f + (0.1f * logf(1.0f / freq_scale)));
}

// The RoPE parameters of llm_graph_context and deepseek4.cpp for this
// profile: YaRN (ext_factor 1, freq_scale 1 / factor) on the compressed
// layers, the indexer and the compressed rows; none on the window-only
// layers.
struct Rope {
  int n_ctx_orig = 0;
  float base = 0;
  float scale = 1;
  float ext = 0;
  float attn = 1;
  float beta_fast = 0;
  float beta_slow = 0;
};

Rope CompressedRope(const model::Dsv4Profile& p) {
  const float scale = 1.0f / p.rope_scale;
  return {.n_ctx_orig = static_cast<int>(p.yarn_original_context),
          .base = p.compress_rope_base,
          .scale = scale,
          .ext = 1.0f,
          .attn = RopeAttnFactor(scale, 1.0f),
          .beta_fast = p.yarn_beta_fast,
          .beta_slow = p.yarn_beta_slow};
}

Rope LayerRope(const model::Dsv4Profile& p, std::uint32_t ratio) {
  if (ratio != 0) {
    return CompressedRope(p);
  }
  return {.n_ctx_orig = 0,
          .base = p.rope_base,
          .scale = 1.0f,
          .ext = 0.0f,
          .attn = RopeAttnFactor(1.0f, 0.0f),
          .beta_fast = 0.0f,
          .beta_slow = 0.0f};
}

// LLAMA_ROPE_TYPE_NORM: consecutive pairs.
constexpr int kRopeMode = 0;

class Builder {
 public:
  Builder(ggml_context* c, const model::Dsv4Profile& p, const model::Dsv4Binding& b,
          const Dsv4ChunkShape& s, Dsv4Graph& g)
      : c_(c), p_(p), b_(b), s_(s), g_(g) {}

  std::expected<void, KernelFailure> Leaves(const Dsv4GraphOptions& options);
  void Build();

 private:
  // A named intermediate, as llama.cpp's callback names it.
  void Name(ggml_tensor* t, std::string_view name, int il) {
    g_.named.emplace_back(il >= 0 ? std::format("{}-{}", name, il) : std::string(name), t);
  }
  void Expand(ggml_tensor* t) { expanded_.push_back(t); }

  ggml_tensor* Norm(ggml_tensor* x, ggml_tensor* weight) {
    return ggml_mul(c_, ggml_rms_norm(c_, x, p_.rms_eps), weight);
  }
  ggml_tensor* RopeExt(ggml_tensor* x, ggml_tensor* pos, const Rope& r, int n_ctx_orig) {
    return ggml_rope_ext(c_, x, pos, nullptr, static_cast<int>(p_.rope_dims), kRopeMode, n_ctx_orig,
                         r.base, r.scale, r.ext, r.attn, r.beta_fast, r.beta_slow);
  }
  // llama_mul_mat_hadamard (llama-impl.h).
  ggml_tensor* Hadamard(ggml_tensor* cur, ggml_tensor* rot) {
    const std::int64_t n = rot->ne[0];
    ggml_tensor* res = ggml_is_contiguous(cur)
                           ? ggml_reshape_2d(c_, cur, n, ggml_nelements(cur) / n)
                           : ggml_cont_2d(c_, cur, n, ggml_nelements(cur) / n);
    res = ggml_mul_mat(c_, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    return ggml_reshape_4d(c_, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
  }

  ggml_tensor* HcPre(ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                     ggml_tensor** post, ggml_tensor** comb, int il);
  ggml_tensor* HcHead(ggml_tensor* x);
  ggml_tensor* OverlapCompress(ggml_tensor* kv_state, ggml_tensor* score_state,
                               ggml_tensor* read_idxs, ggml_tensor* comp_pos, ggml_tensor* norm,
                               std::int64_t ratio, std::int64_t head);
  ggml_tensor* HcaCompress(ggml_tensor* kv_state, ggml_tensor* score_state, ggml_tensor* read_idxs,
                           ggml_tensor* comp_pos, ggml_tensor* norm, std::int64_t head);
  ggml_tensor* AppendZeroRow(ggml_tensor* t, bool neg_inf);
  ggml_tensor* LidTopK(const Dsv4LayerTensors& l, ggml_tensor* qr, ggml_tensor* cur, int il);
  ggml_tensor* TopKMask(ggml_tensor* kq_mask, ggml_tensor* top_k);
  ggml_tensor* AttnMha(ggml_tensor* q, ggml_tensor* k, ggml_tensor* kq_mask, ggml_tensor* sinks,
                       std::int64_t n_kv_max);
  ggml_tensor* CpyK(ggml_tensor* cache, ggml_tensor* k_cur, ggml_tensor* idxs);
  ggml_tensor* GetK(ggml_tensor* cache, std::int64_t n_kv);
  ggml_tensor* Attention(std::uint32_t il, ggml_tensor* cur);
  ggml_tensor* Moe(std::uint32_t il, ggml_tensor* cur);

  ggml_context* c_;
  const model::Dsv4Profile& p_;
  const model::Dsv4Binding& b_;
  const Dsv4ChunkShape& s_;
  Dsv4Graph& g_;
  std::vector<ggml_tensor*> expanded_;
};

std::expected<ggml_tensor*, KernelFailure> Leaf(ggml_context* c, const model::Dsv4Tensor& t,
                                                std::string_view role) {
  auto type = GgmlTypeOf(t.type);
  if (!type) {
    return std::unexpected(type.error());
  }
  std::array<std::int64_t, 4> ne = {1, 1, 1, 1};
  if (t.ne.empty() || t.ne.size() > 4) {
    return Rejected(std::format("{}: not a GGML shape", role));
  }
  for (std::size_t i = 0; i < t.ne.size(); ++i) {
    ne[i] = static_cast<std::int64_t>(t.ne[i]);
  }
  if (ne[0] % ggml_blck_size(*type) != 0) {
    return Rejected(std::format("{}: rows are not whole blocks", role));
  }
  return ggml_new_tensor(c, *type, static_cast<int>(t.ne.size()), ne.data());
}

std::expected<void, KernelFailure> Builder::Leaves(const Dsv4GraphOptions& options) {
  const std::int64_t n = s_.rows;
  g_.embd = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.width, n);
  g_.tokens = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.positions = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  g_.raw_k_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, n);
  g_.raw_mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, s_.raw_n_kv, n, 1, 1);
  g_.out_ids = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
  const auto comp = [&](Dsv4CompInputs& in, std::int64_t blocks, std::int64_t persist,
                        std::int64_t reads, std::int64_t n_kv) {
    in.state_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, n);
    in.persist_src = ggml_new_tensor_1d(c_, GGML_TYPE_I32, persist);
    in.persist_dst = ggml_new_tensor_1d(c_, GGML_TYPE_I32, persist);
    in.read_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I32, reads);
    in.write_idxs = ggml_new_tensor_1d(c_, GGML_TYPE_I64, blocks);
    in.write_pos = ggml_new_tensor_1d(c_, GGML_TYPE_I32, blocks);
    in.mask = ggml_new_tensor_4d(c_, GGML_TYPE_F16, n_kv, n, 1, 1);
  };
  const std::int64_t csa_reads = 2 * std::int64_t{model::kDsv4CsaRatio} * s_.csa_blocks;
  comp(g_.csa, s_.csa_blocks, s_.csa_persist, csa_reads, s_.csa_n_kv);
  comp(g_.hca, s_.hca_blocks, s_.hca_persist, model::kDsv4HcaRatio * s_.hca_blocks, s_.hca_n_kv);
  comp(g_.lid, s_.csa_blocks, s_.csa_persist, csa_reads, s_.csa_n_kv);
  g_.lid_rot = ggml_new_tensor_2d(c_, GGML_TYPE_F32, p_.indexer_head_dim, p_.indexer_head_dim);
  const std::int64_t top_k = std::min<std::int64_t>(s_.csa_n_kv, p_.indexer_top_k);
  g_.top_k_zeros = ggml_new_tensor_4d(c_, GGML_TYPE_F16, 1, top_k, n, 1);

  const auto leaf = [&](ggml_tensor*& into, const model::Dsv4Tensor& t,
                        std::string_view role) -> std::expected<void, KernelFailure> {
    auto made = Leaf(c_, t, role);
    if (!made) {
      return std::unexpected(made.error());
    }
    into = *made;
    return {};
  };
#define JITLLM_LEAF(into, tensor)                       \
  if (auto made = leaf(into, tensor, #tensor); !made) { \
    return made;                                        \
  }
  JITLLM_LEAF(g_.output_norm, b_.output_norm)
  JITLLM_LEAF(g_.output, b_.output)
  JITLLM_LEAF(g_.hc_head_fn, b_.hc_head_fn)
  JITLLM_LEAF(g_.hc_head_base, b_.hc_head_base)
  JITLLM_LEAF(g_.hc_head_scale, b_.hc_head_scale)
  g_.layers.resize(p_.layers);
  if (!options.expert_stride.empty() && options.expert_stride.size() != p_.layers) {
    return Rejected("an expert stride per layer, or none");
  }
  for (std::uint32_t il = 0; il < p_.layers; ++il) {
    const model::Dsv4Layer& w = b_.layers[il];
    Dsv4LayerTensors& l = g_.layers[il];
    JITLLM_LEAF(l.attn_norm, w.attn_norm)
    JITLLM_LEAF(l.attn_sinks, w.attn_sinks)
    JITLLM_LEAF(l.q_a, w.q_a)
    JITLLM_LEAF(l.q_a_norm, w.q_a_norm)
    JITLLM_LEAF(l.q_b, w.q_b)
    JITLLM_LEAF(l.kv, w.kv)
    JITLLM_LEAF(l.kv_norm, w.kv_norm) {
      // llama.cpp loads wo_a as [heads·head / groups, o_lora, groups].
      auto type = GgmlTypeOf(w.out_a.type);
      if (!type) {
        return std::unexpected(type.error());
      }
      l.out_a = ggml_new_tensor_3d(c_, *type, static_cast<std::int64_t>(w.out_a.ne[0]), p_.o_lora,
                                   p_.o_groups);
    }
    JITLLM_LEAF(l.out_b, w.out_b)
    JITLLM_LEAF(l.hc_attn_fn, w.hc_attn_fn)
    JITLLM_LEAF(l.hc_attn_base, w.hc_attn_base)
    JITLLM_LEAF(l.hc_attn_scale, w.hc_attn_scale)
    JITLLM_LEAF(l.hc_ffn_fn, w.hc_ffn_fn)
    JITLLM_LEAF(l.hc_ffn_base, w.hc_ffn_base)
    JITLLM_LEAF(l.hc_ffn_scale, w.hc_ffn_scale)
    if (w.ratio != 0) {
      JITLLM_LEAF(l.comp_kv, w.comp_kv)
      JITLLM_LEAF(l.comp_gate, w.comp_gate)
      JITLLM_LEAF(l.comp_ape, w.comp_ape)
      JITLLM_LEAF(l.comp_norm, w.comp_norm)
    }
    if (w.ratio == model::kDsv4CsaRatio) {
      JITLLM_LEAF(l.idx_q_b, w.idx_q_b)
      JITLLM_LEAF(l.idx_proj, w.idx_proj)
      JITLLM_LEAF(l.idx_comp_kv, w.idx_comp_kv)
      JITLLM_LEAF(l.idx_comp_gate, w.idx_comp_gate)
      JITLLM_LEAF(l.idx_comp_ape, w.idx_comp_ape)
      JITLLM_LEAF(l.idx_comp_norm, w.idx_comp_norm)
    }
    JITLLM_LEAF(l.ffn_norm, w.ffn_norm)
    JITLLM_LEAF(l.router, w.router)
    if (w.hash) {
      JITLLM_LEAF(l.tid2eid, w.tid2eid)
    } else {
      JITLLM_LEAF(l.router_bias, w.router_bias)
    }
    JITLLM_LEAF(l.up_shexp, w.up_shexp)
    JITLLM_LEAF(l.gate_shexp, w.gate_shexp)
    JITLLM_LEAF(l.down_shexp, w.down_shexp)
    // Routed experts: 3D at the layer's stride.
    const std::uint64_t stride = options.expert_stride.empty() ? 0 : options.expert_stride[il];
    const auto experts = [&](ggml_tensor*& into,
                             const model::Dsv4Tensor& t) -> std::expected<void, KernelFailure> {
      auto type = GgmlTypeOf(t.type);
      if (!type) {
        return std::unexpected(type.error());
      }
      into = ggml_new_tensor_3d(c_, *type, static_cast<std::int64_t>(t.ne[0]),
                                static_cast<std::int64_t>(t.ne[1]), p_.experts);
      if (stride != 0) {
        const std::size_t slice = into->nb[2];
        if (stride < slice || stride % ggml_type_size(*type) != 0) {
          return Rejected(
              std::format("layer {}: an expert stride of {} bytes does not hold "
                          "whole {} slices",
                          il, stride, t.type));
        }
        into->nb[2] = stride;
        into->nb[3] = stride * static_cast<std::size_t>(p_.experts);
      }
      return {};
    };
    if (auto e = experts(l.up_exps, w.up_exps); !e) {
      return e;
    }
    if (auto e = experts(l.gate_exps, w.gate_exps); !e) {
      return e;
    }
    if (auto e = experts(l.down_exps, w.down_exps); !e) {
      return e;
    }
    // State.
    const std::int64_t head = p_.head_dim;
    l.raw_k = ggml_new_tensor_3d(c_, GGML_TYPE_F16, head, s_.raw_cells, 1);
    if (w.ratio == model::kDsv4CsaRatio) {
      l.csa_k = ggml_new_tensor_3d(c_, GGML_TYPE_F16, head, s_.csa_cells, 1);
      l.csa_state_kv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, 2 * head, s_.csa_state_rows);
      l.csa_state_score = ggml_new_tensor_2d(c_, GGML_TYPE_F32, 2 * head, s_.csa_state_rows);
      l.lid_k = ggml_new_tensor_3d(c_, GGML_TYPE_F16, p_.indexer_head_dim, s_.csa_cells, 1);
      const std::int64_t lid_ring = 2 * std::int64_t{p_.indexer_head_dim};
      l.lid_state_kv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, lid_ring, s_.csa_state_rows);
      l.lid_state_score = ggml_new_tensor_2d(c_, GGML_TYPE_F32, lid_ring, s_.csa_state_rows);
    } else if (w.ratio == model::kDsv4HcaRatio) {
      l.hca_k = ggml_new_tensor_3d(c_, GGML_TYPE_F16, head, s_.hca_cells, 1);
      l.hca_state_kv = ggml_new_tensor_2d(c_, GGML_TYPE_F32, head, s_.hca_state_rows);
      l.hca_state_score = ggml_new_tensor_2d(c_, GGML_TYPE_F32, head, s_.hca_state_rows);
    }
  }
#undef JITLLM_LEAF
  return {};
}

// build_hc_pre with the fused comb and pre (deepseek4.cpp:354-410).
ggml_tensor* Builder::HcPre(ggml_tensor* x, ggml_tensor* fn, ggml_tensor* scale, ggml_tensor* base,
                            ggml_tensor** post, ggml_tensor** comb, int il) {
  const std::int64_t hc = p_.hc;
  const std::int64_t nt = x->ne[2];
  ggml_tensor* flat = ggml_reshape_2d(c_, x, p_.hc_width(), nt);
  ggml_tensor* flat_norm = ggml_rms_norm(c_, flat, p_.rms_eps);
  ggml_tensor* mixes = ggml_mul_mat(c_, fn, flat_norm);
  ggml_tensor* scale_pre = ggml_view_1d(c_, scale, 1, ggml_row_size(scale->type, 0));
  ggml_tensor* scale_post = ggml_view_1d(c_, scale, 1, ggml_row_size(scale->type, 1));
  ggml_tensor* base_pre = ggml_view_1d(c_, base, hc, ggml_row_size(base->type, 0));
  ggml_tensor* base_post = ggml_view_1d(c_, base, hc, ggml_row_size(base->type, hc));
  ggml_tensor* pre = ggml_view_2d(c_, mixes, hc, nt, mixes->nb[1], ggml_row_size(mixes->type, 0));
  pre = ggml_add(c_, ggml_mul(c_, pre, scale_pre), base_pre);
  pre = ggml_sigmoid(c_, pre);
  pre = ggml_scale_bias(c_, pre, 1.0f, p_.hc_eps);
  *post = ggml_view_2d(c_, mixes, hc, nt, mixes->nb[1], ggml_row_size(mixes->type, hc));
  *post = ggml_add(c_, ggml_mul(c_, *post, scale_post), base_post);
  *post = ggml_sigmoid(c_, *post);
  *post = ggml_scale(c_, *post, 2.0f);
  *comb = ggml_dsv4_hc_comb(c_, mixes, scale, base, p_.hc_eps,
                            static_cast<std::int32_t>(p_.sinkhorn_iterations));
  (void)il;
  return ggml_dsv4_hc_pre(c_, x, pre);
}

// build_hc_head (deepseek4.cpp:449-469), its pre fused (il -1 is not fused
// upstream: build_hc_pre's weighted sum runs as views, muls and adds).
ggml_tensor* Builder::HcHead(ggml_tensor* x) {
  const std::int64_t hc = p_.hc;
  const std::int64_t nt = x->ne[2];
  ggml_tensor* flat = ggml_reshape_2d(c_, x, p_.hc_width(), nt);
  ggml_tensor* flat_norm = ggml_rms_norm(c_, flat, p_.rms_eps);
  ggml_tensor* mixes = ggml_mul_mat(c_, g_.hc_head_fn, flat_norm);
  ggml_tensor* pre = ggml_add(c_, ggml_mul(c_, mixes, g_.hc_head_scale), g_.hc_head_base);
  pre = ggml_sigmoid(c_, pre);
  pre = ggml_scale_bias(c_, pre, 1.0f, p_.hc_eps);
  // build_hc_pre(x, pre, -1): cparams.fused_dsv4_hc_pre && il >= 0 is false.
  ggml_tensor* result = nullptr;
  for (std::int64_t ih = 0; ih < hc; ++ih) {
    const auto h = static_cast<std::size_t>(ih);
    ggml_tensor* xh = ggml_view_2d(c_, x, p_.width, nt, x->nb[2], h * x->nb[1]);
    ggml_tensor* wh = ggml_view_2d(c_, pre, 1, nt, pre->nb[1], h * pre->nb[0]);
    ggml_tensor* cur = ggml_mul(c_, xh, wh);
    result = result != nullptr ? ggml_add(c_, result, cur) : cur;
  }
  return result;
}

// dsv4_append_zero_row.
ggml_tensor* Builder::AppendZeroRow(ggml_tensor* t, bool neg_inf) {
  ggml_tensor* row = ggml_view_1d(c_, t, t->ne[0], 0);
  row = neg_inf ? ggml_scale_bias(c_, row, 0.0f, -INFINITY) : ggml_scale(c_, row, 0.0f);
  row = ggml_reshape_2d(c_, row, t->ne[0], 1);
  return ggml_concat(c_, t, row, 1);
}

// build_overlap_compressed_kv_from_state (deepseek4.cpp:518-589).
ggml_tensor* Builder::OverlapCompress(ggml_tensor* kv_state, ggml_tensor* score_state,
                                      ggml_tensor* read_idxs, ggml_tensor* comp_pos,
                                      ggml_tensor* norm, std::int64_t ratio, std::int64_t head) {
  const std::int64_t n_blocks = comp_pos->ne[0];
  kv_state = AppendZeroRow(kv_state, false);
  score_state = AppendZeroRow(score_state, true);
  const std::int64_t n_read = ratio * n_blocks;
  const auto reads = static_cast<std::size_t>(n_read);
  ggml_tensor* kv_rows = ggml_get_rows(c_, kv_state, read_idxs);
  ggml_tensor* score_rows = ggml_get_rows(c_, score_state, read_idxs);
  ggml_tensor* kv_prev = ggml_cont(c_, ggml_view_2d(c_, kv_rows, head, n_read, kv_rows->nb[1], 0));
  kv_prev = ggml_reshape_3d(c_, kv_prev, head, ratio, n_blocks);
  ggml_tensor* score_prev =
      ggml_cont(c_, ggml_view_2d(c_, score_rows, head, n_read, score_rows->nb[1], 0));
  score_prev = ggml_reshape_3d(c_, score_prev, head, ratio, n_blocks);
  ggml_tensor* kv_cur =
      ggml_cont(c_, ggml_view_2d(c_, kv_rows, head, n_read, kv_rows->nb[1],
                                 (reads * kv_rows->nb[1]) + ggml_row_size(kv_rows->type, head)));
  kv_cur = ggml_reshape_3d(c_, kv_cur, head, ratio, n_blocks);
  ggml_tensor* score_cur = ggml_cont(
      c_, ggml_view_2d(c_, score_rows, head, n_read, score_rows->nb[1],
                       (reads * score_rows->nb[1]) + ggml_row_size(score_rows->type, head)));
  score_cur = ggml_reshape_3d(c_, score_cur, head, ratio, n_blocks);
  ggml_tensor* values = ggml_concat(c_, kv_prev, kv_cur, 1);
  ggml_tensor* scores = ggml_concat(c_, score_prev, score_cur, 1);
  values = ggml_cont(c_, ggml_permute(c_, values, 1, 0, 2, 3));
  scores = ggml_cont(c_, ggml_permute(c_, scores, 1, 0, 2, 3));
  ggml_tensor* weights = ggml_soft_max(c_, scores);
  ggml_tensor* comp = ggml_mul(c_, values, weights);
  comp = ggml_sum_rows(c_, comp);
  comp = ggml_cont(c_, ggml_permute(c_, comp, 1, 0, 2, 3));
  comp = Norm(comp, norm);
  const Rope r = CompressedRope(p_);
  comp = RopeExt(comp, comp_pos, r, r.n_ctx_orig);
  return ggml_rope_set_offset(comp, static_cast<int>(head - p_.rope_dims));
}

// build_hca_compressed_kv_from_state (deepseek4.cpp:471-516).
ggml_tensor* Builder::HcaCompress(ggml_tensor* kv_state, ggml_tensor* score_state,
                                  ggml_tensor* read_idxs, ggml_tensor* comp_pos, ggml_tensor* norm,
                                  std::int64_t head) {
  const std::int64_t n_blocks = comp_pos->ne[0];
  const std::int64_t ratio = model::kDsv4HcaRatio;
  ggml_tensor* kv = ggml_get_rows(c_, kv_state, read_idxs);
  kv = ggml_reshape_3d(c_, kv, head, ratio, n_blocks);
  ggml_tensor* score = ggml_get_rows(c_, score_state, read_idxs);
  score = ggml_reshape_3d(c_, score, head, ratio, n_blocks);
  ggml_tensor* values = ggml_cont(c_, ggml_permute(c_, kv, 1, 0, 2, 3));
  ggml_tensor* scores = ggml_cont(c_, ggml_permute(c_, score, 1, 0, 2, 3));
  ggml_tensor* weights = ggml_soft_max(c_, scores);
  ggml_tensor* comp = ggml_mul(c_, values, weights);
  comp = ggml_sum_rows(c_, comp);
  comp = ggml_cont(c_, ggml_permute(c_, comp, 1, 0, 2, 3));
  comp = Norm(comp, norm);
  const Rope r = CompressedRope(p_);
  comp = RopeExt(comp, comp_pos, r, r.n_ctx_orig);
  return ggml_rope_set_offset(comp, static_cast<int>(head - p_.rope_dims));
}

// llama_kv_cache::cpy_k: merge the heads, store the rows at the cells.
ggml_tensor* Builder::CpyK(ggml_tensor* cache, ggml_tensor* k_cur, ggml_tensor* idxs) {
  const std::int64_t n_embd_gqa = k_cur->ne[0] * k_cur->ne[1];
  k_cur = ggml_view_2d(c_, k_cur, n_embd_gqa, k_cur->ne[2], k_cur->nb[2], 0);
  return ggml_set_rows(c_, cache, k_cur, idxs);
}

// llama_kv_cache::get_k over the first n_kv cells (one KV head, one stream).
ggml_tensor* Builder::GetK(ggml_tensor* cache, std::int64_t n_kv) {
  const std::int64_t head = cache->ne[0];
  return ggml_view_4d(c_, cache, head, 1, n_kv, 1, ggml_row_size(cache->type, head),
                      ggml_row_size(cache->type, head),
                      ggml_row_size(cache->type, head * cache->ne[1]), 0);
}

// build_attn_mha with flash attention, one stream (llama-graph.cpp:2591-2650).
ggml_tensor* Builder::AttnMha(ggml_tensor* q, ggml_tensor* k, ggml_tensor* kq_mask,
                              ggml_tensor* sinks, std::int64_t n_kv_max) {
  ggml_tensor* v = k;
  q = ggml_view_4d(c_, q, q->ne[0], q->ne[1], q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], 0);
  q = ggml_permute(c_, q, 0, 2, 1, 3);
  k = ggml_permute(c_, k, 0, 2, 1, 3);
  v = ggml_permute(c_, v, 0, 2, 1, 3);
  const float scale = 1.0f / std::sqrt(static_cast<float>(p_.head_dim));
  ggml_tensor* cur = ggml_flash_attn_ext(c_, q, k, v, kq_mask, scale, 0.0f, 0.0f);
  ggml_flash_attn_ext_add_sinks(cur, sinks);
  ggml_flash_attn_ext_set_n_kv_max(cur, static_cast<std::int32_t>(n_kv_max));
  ggml_prec_set_acc(cur, GGML_PREC_F32);
  return ggml_reshape_2d(c_, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
}

// build_lid_top_k (deepseek4.cpp:591-677), the indexer fused.
ggml_tensor* Builder::LidTopK(const Dsv4LayerTensors& l, ggml_tensor* qr, ggml_tensor* cur,
                              int il) {
  const std::int64_t ih = p_.indexer_head_dim;
  const std::int64_t heads = p_.indexer_heads;
  const std::int64_t nt = cur->ne[1];
  ggml_tensor* q = ggml_mul_mat(c_, l.idx_q_b, qr);
  q = ggml_reshape_3d(c_, q, ih, heads, nt);
  const Rope r = CompressedRope(p_);
  q = RopeExt(q, g_.positions, r, r.n_ctx_orig);
  q = ggml_rope_set_offset(q, static_cast<int>(ih - p_.rope_dims));
  q = Hadamard(q, g_.lid_rot);
  ggml_tensor* weights = ggml_mul_mat(c_, l.idx_proj, cur);
  weights = ggml_scale(c_, weights, 1.0f / sqrtf(static_cast<float>(ih * heads)));
  ggml_tensor* k = GetK(l.lid_k, s_.csa_n_kv);
  q = ggml_view_4d(c_, q, q->ne[0], q->ne[1], q->ne[2], 1, q->nb[1], q->nb[2], q->nb[3], 0);
  weights = ggml_view_4d(c_, weights, weights->ne[0], weights->ne[1], weights->ne[2], 1,
                         weights->nb[1], weights->nb[2], weights->nb[3], 0);
  ggml_tensor* score = ggml_lightning_indexer(c_, q, k, weights, g_.lid.mask);
  Name(score, "lid_score_masked", il);
  const std::int64_t top = std::min<std::int64_t>(score->ne[0], p_.indexer_top_k);
  return ggml_cont(c_, ggml_top_k(c_, score, static_cast<int>(top)));
}

// build_top_k_mask (deepseek4.cpp:679-706).
ggml_tensor* Builder::TopKMask(ggml_tensor* kq_mask, ggml_tensor* top_k) {
  ggml_tensor* all = ggml_fill(c_, kq_mask, -INFINITY);
  all = ggml_view_4d(c_, all, 1, all->ne[0], all->ne[1], all->ne[3], all->nb[0], all->nb[1],
                     all->nb[2], 0);
  ggml_tensor* top_k_3d =
      ggml_view_4d(c_, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1],
                   top_k->nb[2], static_cast<std::size_t>(top_k->ne[3]) * top_k->nb[3], 0);
  ggml_tensor* zeros = ggml_fill(c_, g_.top_k_zeros, 0.0f);
  ggml_tensor* masked = ggml_set_rows(c_, all, zeros, top_k_3d);
  masked = ggml_view_4d(c_, masked, masked->ne[1], masked->ne[2], 1, masked->ne[3], masked->nb[2],
                        masked->nb[3], masked->nb[3], 0);
  return ggml_add(c_, masked, kq_mask);
}

// build_attention_impl (deepseek4.cpp:879-1221) for one layer.
ggml_tensor* Builder::Attention(std::uint32_t il_u, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::uint32_t ratio = p_.compress_ratios[il_u];
  const std::int64_t head = p_.head_dim;
  const std::int64_t nope = head - p_.rope_dims;
  const std::int64_t heads = p_.heads;
  const std::int64_t groups = p_.o_groups;
  const std::int64_t nt = cur->ne[1];
  const Rope rl = LayerRope(p_, ratio);

  ggml_tensor* qr = ggml_mul_mat(c_, l.q_a, cur);
  qr = Norm(qr, l.q_a_norm);
  ggml_tensor* q = ggml_mul_mat(c_, l.q_b, qr);
  q = ggml_reshape_3d(c_, q, head, heads, nt);
  q = ggml_rms_norm(c_, q, p_.rms_eps);
  q = RopeExt(q, g_.positions, rl, rl.n_ctx_orig);
  q = ggml_rope_set_offset(q, static_cast<int>(nope));
  Name(q, "q", il);

  ggml_tensor* kv = ggml_mul_mat(c_, l.kv, cur);
  kv = Norm(kv, l.kv_norm);
  kv = ggml_reshape_3d(c_, kv, head, 1, nt);
  kv = RopeExt(kv, g_.positions, rl, rl.n_ctx_orig);
  kv = ggml_rope_set_offset(kv, static_cast<int>(nope));
  Name(kv, "kv", il);

  ggml_tensor* hca_state_kv = nullptr;
  ggml_tensor* hca_state_score = nullptr;
  if (ratio == model::kDsv4HcaRatio) {
    hca_state_kv = ggml_mul_mat(c_, l.comp_kv, cur);
    hca_state_score = ggml_mul_mat(c_, l.comp_gate, cur);
    ggml_tensor* ape_rows = ggml_get_rows(c_, l.comp_ape, g_.hca.state_pos);
    hca_state_score = ggml_add(c_, hca_state_score, ape_rows);
  }
  if (ratio == model::kDsv4CsaRatio) {
    ggml_tensor* csa_kv = ggml_mul_mat(c_, l.comp_kv, cur);
    ggml_tensor* csa_score = ggml_mul_mat(c_, l.comp_gate, cur);
    ggml_tensor* ape_rows = ggml_get_rows(c_, l.comp_ape, g_.csa.state_pos);
    csa_score = ggml_add(c_, csa_score, ape_rows);
    // The ring state, read in place (no rollback planes to restore from).
    ggml_tensor* base_kv = ggml_view_2d(c_, l.csa_state_kv, l.csa_state_kv->ne[0],
                                        s_.csa_state_rows, l.csa_state_kv->nb[1], 0);
    ggml_tensor* base_score = ggml_view_2d(c_, l.csa_state_score, l.csa_state_score->ne[0],
                                           s_.csa_state_rows, l.csa_state_score->nb[1], 0);
    ggml_tensor* source_kv = ggml_concat(c_, base_kv, csa_kv, 1);
    ggml_tensor* source_score = ggml_concat(c_, base_score, csa_score, 1);
    ggml_tensor* comp = OverlapCompress(source_kv, source_score, g_.csa.read_idxs, g_.csa.write_pos,
                                        l.comp_norm, model::kDsv4CsaRatio, head);
    Name(comp, "csa_state_compress", il);
    Expand(CpyK(l.csa_k, comp, g_.csa.write_idxs));
    ggml_tensor* persist_kv = ggml_get_rows(c_, csa_kv, g_.csa.persist_src);
    ggml_tensor* persist_score = ggml_get_rows(c_, csa_score, g_.csa.persist_src);
    Expand(ggml_set_rows(c_, l.csa_state_kv, persist_kv, g_.csa.persist_dst));
    Expand(ggml_set_rows(c_, l.csa_state_score, persist_score, g_.csa.persist_dst));

    const std::int64_t ih = p_.indexer_head_dim;
    ggml_tensor* lid_kv = ggml_mul_mat(c_, l.idx_comp_kv, cur);
    ggml_tensor* lid_score = ggml_mul_mat(c_, l.idx_comp_gate, cur);
    ggml_tensor* lid_ape_rows = ggml_get_rows(c_, l.idx_comp_ape, g_.lid.state_pos);
    lid_score = ggml_add(c_, lid_score, lid_ape_rows);
    ggml_tensor* lid_base_kv = ggml_view_2d(c_, l.lid_state_kv, l.lid_state_kv->ne[0],
                                            s_.csa_state_rows, l.lid_state_kv->nb[1], 0);
    ggml_tensor* lid_base_score = ggml_view_2d(c_, l.lid_state_score, l.lid_state_score->ne[0],
                                               s_.csa_state_rows, l.lid_state_score->nb[1], 0);
    ggml_tensor* lid_source_kv = ggml_concat(c_, lid_base_kv, lid_kv, 1);
    ggml_tensor* lid_source_score = ggml_concat(c_, lid_base_score, lid_score, 1);
    ggml_tensor* lid_comp =
        OverlapCompress(lid_source_kv, lid_source_score, g_.lid.read_idxs, g_.lid.write_pos,
                        l.idx_comp_norm, model::kDsv4CsaRatio, ih);
    lid_comp = Hadamard(lid_comp, g_.lid_rot);
    Name(lid_comp, "lid_state_compress_rot", il);
    Expand(CpyK(l.lid_k, lid_comp, g_.lid.write_idxs));
    ggml_tensor* lid_persist_kv = ggml_get_rows(c_, lid_kv, g_.lid.persist_src);
    ggml_tensor* lid_persist_score = ggml_get_rows(c_, lid_score, g_.lid.persist_src);
    Expand(ggml_set_rows(c_, l.lid_state_kv, lid_persist_kv, g_.lid.persist_dst));
    Expand(ggml_set_rows(c_, l.lid_state_score, lid_persist_score, g_.lid.persist_dst));
  }
  if (ratio == model::kDsv4HcaRatio) {
    ggml_tensor* base_kv = ggml_view_2d(c_, l.hca_state_kv, l.hca_state_kv->ne[0],
                                        s_.hca_state_rows, l.hca_state_kv->nb[1], 0);
    ggml_tensor* base_score = ggml_view_2d(c_, l.hca_state_score, l.hca_state_score->ne[0],
                                           s_.hca_state_rows, l.hca_state_score->nb[1], 0);
    ggml_tensor* source_kv = ggml_concat(c_, base_kv, hca_state_kv, 1);
    ggml_tensor* source_score = ggml_concat(c_, base_score, hca_state_score, 1);
    ggml_tensor* comp =
        HcaCompress(source_kv, source_score, g_.hca.read_idxs, g_.hca.write_pos, l.comp_norm, head);
    Name(comp, "hca_state_compress", il);
    Expand(CpyK(l.hca_k, comp, g_.hca.write_idxs));
    ggml_tensor* persist_kv = ggml_get_rows(c_, hca_state_kv, g_.hca.persist_src);
    ggml_tensor* persist_score = ggml_get_rows(c_, hca_state_score, g_.hca.persist_src);
    Expand(ggml_set_rows(c_, l.hca_state_kv, persist_kv, g_.hca.persist_dst));
    Expand(ggml_set_rows(c_, l.hca_state_score, persist_score, g_.hca.persist_dst));
  }

  ggml_tensor* out = nullptr;
  if (ratio == model::kDsv4CsaRatio) {
    // build_csa_lid_attention.
    ggml_tensor* top_k = LidTopK(l, qr, cur, il);
    Expand(q);
    Expand(kv);
    Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
    ggml_tensor* raw_k = GetK(l.raw_k, s_.raw_n_kv);
    ggml_tensor* csa_k = GetK(l.csa_k, s_.csa_n_kv);
    ggml_tensor* k_all = ggml_concat(c_, raw_k, csa_k, 2);
    ggml_tensor* csa_mask = TopKMask(g_.csa.mask, top_k);
    ggml_tensor* kq_mask = ggml_concat(c_, g_.raw_mask, csa_mask, 0);
    const std::int64_t n_kv_max =
        std::min<std::int64_t>(g_.raw_mask->ne[0], p_.window) + top_k->ne[0];
    out = AttnMha(q, k_all, kq_mask, l.attn_sinks, n_kv_max);
    Name(out, "attn_csa_lid", il);
  } else if (ratio == model::kDsv4HcaRatio) {
    Expand(q);
    Expand(kv);
    Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
    ggml_tensor* raw_k = GetK(l.raw_k, s_.raw_n_kv);
    ggml_tensor* hca_k = GetK(l.hca_k, s_.hca_n_kv);
    ggml_tensor* k_all = ggml_concat(c_, raw_k, hca_k, 2);
    ggml_tensor* kq_mask = ggml_concat(c_, g_.raw_mask, g_.hca.mask, 0);
    out = AttnMha(q, k_all, kq_mask, l.attn_sinks, 0);
    Name(out, "attn_hca", il);
  } else {
    Expand(q);
    Expand(kv);
    Expand(CpyK(l.raw_k, kv, g_.raw_k_idxs));
    ggml_tensor* k = GetK(l.raw_k, s_.raw_n_kv);
    out = AttnMha(q, k, g_.raw_mask, l.attn_sinks, 0);
    Name(out, "attn_raw", il);
  }

  out = ggml_reshape_3d(c_, out, head, heads, nt);
  out = ggml_rope_ext_back(c_, out, g_.positions, nullptr, static_cast<int>(p_.rope_dims),
                           kRopeMode, rl.n_ctx_orig, rl.base, rl.scale, rl.ext, rl.attn,
                           rl.beta_fast, rl.beta_slow);
  out = ggml_rope_set_offset(out, static_cast<int>(nope));
  out = ggml_reshape_3d(c_, out, (heads / groups) * head, groups, nt);
  out = ggml_permute(c_, out, 0, 2, 1, 3);
  ggml_tensor* oa = ggml_mul_mat(c_, l.out_a, out);
  oa = ggml_permute(c_, oa, 0, 2, 1, 3);
  oa = ggml_cont_2d(c_, oa, std::int64_t{p_.o_lora} * groups, nt);
  out = ggml_mul_mat(c_, l.out_b, oa);
  Name(out, "attn_out", il);
  return out;
}

// build_moe_ffn and build_ffn for one layer (llama-graph.cpp:1748-2380).
ggml_tensor* Builder::Moe(std::uint32_t il_u, ggml_tensor* cur) {
  const int il = static_cast<int>(il_u);
  const Dsv4LayerTensors& l = g_.layers[il_u];
  const std::int64_t n_embd = cur->ne[0];
  const std::int64_t nt = cur->ne[1];
  const std::int64_t n_expert = p_.experts;
  const std::int64_t used = p_.experts_used;

  ggml_tensor* selected = nullptr;
  ggml_tensor* bias = l.router_bias;
  if (l.tid2eid != nullptr) {
    selected = ggml_get_rows(c_, l.tid2eid, g_.tokens);
    bias = nullptr;
  }
  ggml_tensor* logits = ggml_mul_mat(c_, l.router, cur);
  ggml_prec_set_acc(logits, GGML_PREC_F32);
  ggml_tensor* probs = ggml_sqrt(c_, ggml_softplus(c_, logits));
  Name(probs, "ffn_moe_probs", il);
  ggml_tensor* selection = probs;
  if (bias != nullptr) {
    selection = ggml_add(c_, probs, bias);
  }
  if (selected == nullptr) {
    selected = ggml_argsort_top_k(c_, selection, static_cast<int>(used));
  }
  Name(selected, "ffn_moe_topk", il);
  probs = ggml_reshape_3d(c_, probs, 1, n_expert, nt);
  ggml_tensor* weights = ggml_get_rows(c_, probs, selected);
  if (p_.expert_weights_norm) {
    weights = ggml_reshape_2d(c_, weights, used, nt);
    ggml_tensor* sum = ggml_sum_rows(c_, weights);
    sum = ggml_clamp(c_, sum, 6.103515625e-5f, INFINITY);
    weights = ggml_div(c_, weights, sum);
    weights = ggml_reshape_3d(c_, weights, 1, used, nt);
  }
  if (p_.expert_weights_scale != 0.0f && p_.expert_weights_scale != 1.0f) {
    weights = ggml_scale(c_, weights, p_.expert_weights_scale);
  }
  Name(weights, "ffn_moe_weights_scaled", il);
  Expand(weights);
  ggml_tensor* x = ggml_reshape_3d(c_, cur, n_embd, 1, nt);
  ggml_tensor* up = ggml_mul_mat_id(c_, l.up_exps, x, selected);
  ggml_tensor* gate = ggml_mul_mat_id(c_, l.gate_exps, x, selected);
  ggml_tensor* act = nullptr;
  if (p_.swiglu_limit > 1e-6f) {
    act = ggml_swiglu_clamp(c_, gate, up, p_.swiglu_limit);
  } else {
    act = ggml_swiglu_split(c_, gate, up);
  }
  ggml_tensor* experts = ggml_mul_mat_id(c_, l.down_exps, act, selected);
  experts = ggml_mul(c_, experts, weights);
  Expand(experts);
  std::vector<ggml_tensor*> views;
  const auto n_used = static_cast<std::size_t>(used);
  for (std::size_t i = 0; i < n_used; ++i) {
    views.push_back(ggml_view_2d(c_, experts, n_embd, nt, experts->nb[2], i * experts->nb[1]));
    Expand(views.back());
  }
  ggml_tensor* moe_out = views[0];
  for (std::size_t i = 1; i < n_used; ++i) {
    moe_out = ggml_add(c_, moe_out, views[i]);
    Expand(moe_out);
  }
  if (used == 1) {
    moe_out = ggml_cont(c_, moe_out);
  }
  Name(moe_out, "ffn_moe_out", il);

  // build_ffn: up, then gate (parallel), the clamped SwiGLU, down.
  ggml_tensor* sh_up = ggml_mul_mat(c_, l.up_shexp, cur);
  ggml_tensor* sh_gate = ggml_mul_mat(c_, l.gate_shexp, cur);
  ggml_tensor* sh = p_.swiglu_limit_shared > 1e-6f
                        ? ggml_swiglu_clamp(c_, sh_gate, sh_up, p_.swiglu_limit_shared)
                        : ggml_swiglu_split(c_, sh_gate, sh_up);
  sh = ggml_mul_mat(c_, l.down_shexp, sh);
  Name(sh, "ffn_shexp", il);
  ggml_tensor* out = ggml_add(c_, moe_out, sh);
  Name(out, "ffn_out", il);
  return out;
}

void Builder::Build() {
  const std::int64_t nt = s_.rows;
  const std::int64_t hc = p_.hc;
  ggml_tensor* inp = ggml_reshape_3d(c_, g_.embd, p_.width, 1, nt);
  ggml_tensor* inpl = ggml_repeat_4d(c_, inp, p_.width, hc, nt, 1);
  Name(inpl, "hc_init", -1);
  for (std::uint32_t il_u = 0; il_u < p_.layers; ++il_u) {
    const int il = static_cast<int>(il_u);
    const Dsv4LayerTensors& l = g_.layers[il_u];
    ggml_tensor* residual = inpl;
    ggml_tensor* post = nullptr;
    ggml_tensor* comb = nullptr;
    ggml_tensor* cur = HcPre(inpl, l.hc_attn_fn, l.hc_attn_scale, l.hc_attn_base, &post, &comb, il);
    Name(cur, "hc_attn_pre", il);
    cur = Norm(cur, l.attn_norm);
    Name(cur, "attn_norm", il);
    cur = Attention(il_u, cur);
    inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
    Name(inpl, "hc_attn_post", il);
    residual = inpl;
    cur = HcPre(inpl, l.hc_ffn_fn, l.hc_ffn_scale, l.hc_ffn_base, &post, &comb, il);
    Name(cur, "hc_ffn_pre", il);
    Expand(residual);
    Expand(post);
    Expand(comb);
    cur = Norm(cur, l.ffn_norm);
    Name(cur, "ffn_norm", il);
    cur = Moe(il_u, cur);
    inpl = ggml_dsv4_hc_post(c_, cur, residual, post, comb);
    Name(inpl, "l_last", il);
  }
  // Every row is an output, gathered as llama.cpp gathers them.
  ggml_tensor* flat = ggml_reshape_2d(c_, inpl, p_.hc_width(), nt);
  ggml_tensor* flat_out = ggml_get_rows(c_, flat, g_.out_ids);
  inpl = ggml_reshape_3d(c_, flat_out, p_.width, hc, nt);
  ggml_tensor* cur = HcHead(inpl);
  Name(cur, "hc_head", -1);
  cur = Norm(cur, g_.output_norm);
  Name(cur, "result_norm", -1);
  g_.logits = ggml_mul_mat(c_, g_.output, cur);
  Name(g_.logits, "result_output", -1);
  Expand(g_.logits);
  g_.nodes = GraphOrder(expanded_);
}

}  // namespace

Dsv4ChunkShape Dsv4ShapeOf(const model::Dsv4StateLayout& state,
                           const model::Dsv4ChunkInputs& chunk) {
  return {.rows = chunk.rows,
          .raw_n_kv = chunk.raw_n_kv,
          .raw_cells = state.raw_cells,
          .csa_n_kv = chunk.csa.n_kv,
          .hca_n_kv = chunk.hca.n_kv,
          .csa_cells = state.csa_cells,
          .hca_cells = state.hca_cells,
          .csa_blocks = chunk.csa.blocks(),
          .hca_blocks = chunk.hca.blocks(),
          .csa_persist = static_cast<std::int64_t>(chunk.csa.persist_src.size()),
          .hca_persist = static_cast<std::int64_t>(chunk.hca.persist_src.size()),
          .csa_state_rows = state.csa_state_rows,
          .hca_state_rows = state.hca_state_rows};
}

std::vector<ggml_tensor*> Dsv4Graph::inputs() const {
  std::vector<ggml_tensor*> all = {embd, tokens, positions, raw_k_idxs, raw_mask, out_ids};
  for (const Dsv4CompInputs* in : {&csa, &hca, &lid}) {
    all.insert(all.end(), {in->state_pos, in->persist_src, in->persist_dst, in->read_idxs,
                           in->write_idxs, in->write_pos, in->mask});
  }
  all.push_back(lid_rot);
  all.push_back(top_k_zeros);
  return all;
}

ggml_tensor* Dsv4Graph::Named(std::string_view name) const {
  for (const auto& [n, t] : named) {
    if (n == name) {
      return t;
    }
  }
  return nullptr;
}

std::size_t Dsv4GraphTensors(const model::Dsv4Profile& profile) {
  // Leaves: about 60 per layer; nodes: at most about 330 per layer (a CSA
  // layer with its indexer), 40 for the head. Rounded up generously.
  return 256 + (std::size_t{profile.layers} * 512);
}

std::expected<ggml_type, KernelFailure> GgmlTypeOf(std::string_view name) {
  // GGML names its types in lower case ("q8_0"); the artifact's index in
  // upper case ("Q8_0"), as GGUF tools print them.
  const auto same = [](std::string_view a, std::string_view b) {
    return a.size() == b.size() && std::ranges::equal(a, b, [](char x, char y) {
             const auto lower = [](char c) {
               return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
             };
             return lower(x) == lower(y);
           });
  };
  for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
    const auto type = static_cast<ggml_type>(t);
    const char* type_name = ggml_type_name(type);
    if (ggml_blck_size(type) > 0 && ggml_type_size(type) > 0 && type_name != nullptr &&
        same(name, type_name)) {
      return type;
    }
  }
  return Rejected(std::format("{} is not a GGML type", name));
}

std::vector<float> HadamardMatrix(std::int64_t n) {
  std::vector<float> data(static_cast<std::size_t>(n * n), 0.0f);
  if (n <= 0 || (n & (n - 1)) != 0) {
    return data;
  }
  data[0] = 1.0f / sqrtf(static_cast<float>(n));
  for (std::int64_t s = 1; s < n; s *= 2) {
    for (std::int64_t i = 0; i < s; ++i) {
      for (std::int64_t j = 0; j < s; ++j) {
        const float v = data[static_cast<std::size_t>((i * n) + j)];
        data[static_cast<std::size_t>(((i + s) * n) + j)] = v;
        data[static_cast<std::size_t>((i * n) + j + s)] = v;
        data[static_cast<std::size_t>(((i + s) * n) + j + s)] = -v;
      }
    }
  }
  return data;
}

std::expected<Dsv4Graph, KernelFailure> BuildDsv4Graph(TensorArena& arena,
                                                       const model::Dsv4Profile& profile,
                                                       const model::Dsv4Binding& binding,
                                                       const Dsv4ChunkShape& shape,
                                                       const Dsv4GraphOptions& options) {
  const Dsv4ChunkShape& s = shape;
  if (s.rows <= 0 || s.raw_cells <= 0 || s.raw_n_kv < 256 || s.raw_n_kv > s.raw_cells ||
      s.raw_n_kv % 256 != 0 || s.csa_n_kv < 256 || s.csa_n_kv > s.csa_cells ||
      s.csa_n_kv % 256 != 0 || s.hca_n_kv < 256 || s.hca_n_kv > s.hca_cells ||
      s.hca_n_kv % 256 != 0 || s.csa_blocks <= 0 || s.hca_blocks <= 0 || s.csa_persist <= 0 ||
      s.hca_persist <= 0 || s.csa_state_rows != 2 * std::int64_t{model::kDsv4CsaRatio} ||
      s.hca_state_rows != std::int64_t{model::kDsv4HcaRatio}) {
    return Rejected("not a DeepSeek V4 chunk shape the state holds");
  }
  if (binding.layers.size() != profile.layers || profile.compress_ratios.size() != profile.layers ||
      profile.hc != 4 || profile.heads % profile.o_groups != 0) {
    return Rejected("the binding is not the profile's");
  }
  if (auto room = arena.Reserve(Dsv4GraphTensors(profile)); !room) {
    return std::unexpected(room.error());
  }
  Dsv4Graph g;
  Builder builder(arena.context(), profile, binding, shape, g);
  if (auto leaves = builder.Leaves(options); !leaves) {
    return std::unexpected(leaves.error());
  }
  builder.Build();
  return g;
}

}  // namespace jitllm::kernels::ggml
