// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_common.h"

#include <algorithm>
#include <cstdint>
#include <format>
#include <numeric>
#include <string_view>
#include <utility>

namespace jitllm::benchmarks {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

}  // namespace

void BindQwen38Weights(const Qwen38Model& m, kg::Qwen38Graph& g) {
  const auto bind = [&](ggml_tensor* t, const md::Qwen38Tensor& r) {
    if (t != nullptr) {
      kg::TensorArena::Bind(t, m.places.resource(r.index));
    }
  };
  const auto mx = [&](const kg::Qwen38Mxfp8Tensors& t, const md::Qwen38Mxfp8& r) {
    bind(t.codes, r.codes);
    bind(t.scales, r.scales);
  };
  const auto& b = *m.binding;
  bind(g.token_embd, b.token_embd);
  kg::TensorArena::Bind(g.ple_table, m.places.ple_table);
  bind(g.ple_table_scale, b.ple_table_scale);
  bind(g.output, b.output);
  bind(g.output_hc_norm, b.output_hc_norm);
  bind(g.output_hc_down, b.output_hc_down);
  bind(g.output_hc_up, b.output_hc_up);
  using K = md::Qwen38StateTensor::Kind;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Qwen38Layer& r = b.layers[il];
    kg::Qwen38LayerTensors& l = g.layers[il];
    bind(l.hc_attn_norm, r.hc_attn_norm);
    bind(l.hc_attn_down, r.hc_attn_down);
    bind(l.hc_attn_up, r.hc_attn_up);
    bind(l.hc_attn_inject, r.hc_attn_inject);
    bind(l.hc_ffn_norm, r.hc_ffn_norm);
    bind(l.hc_ffn_down, r.hc_ffn_down);
    bind(l.hc_ffn_up, r.hc_ffn_up);
    bind(l.hc_ffn_inject, r.hc_ffn_inject);
    if (r.linear) {
      mx(l.qkv, r.qkv);
      mx(l.z, r.z);
      mx(l.beta, r.beta);
      mx(l.alpha, r.alpha);
      mx(l.ssm_out, r.ssm_out);
      bind(l.dt_bias, r.dt_bias);
      bind(l.ssm_a, r.ssm_a);
      bind(l.conv1d, r.conv1d);
      bind(l.ssm_norm, r.ssm_norm);
    } else {
      mx(l.q, r.q);
      mx(l.k, r.k);
      mx(l.v, r.v);
      mx(l.o, r.o);
      mx(l.idx_qk, r.idx_qk);
      bind(l.q_norm, r.q_norm);
      bind(l.k_norm, r.k_norm);
      bind(l.idx_q_norm, r.idx_q_norm);
      bind(l.idx_k_norm, r.idx_k_norm);
    }
    if (il == m.profile->ple_layer) {
      bind(l.ple_key, r.ple_key);
      bind(l.ple_value, r.ple_value);
      bind(l.ple_norm_key, r.ple_norm_key);
      bind(l.ple_norm_query, r.ple_norm_query);
      bind(l.ple_norm_conv, r.ple_norm_conv);
      bind(l.ple_conv1d, r.ple_conv1d);
    }
    bind(l.router, r.router);
    bind(l.shared_gate, r.shared_gate);
    mx(l.gate_shexp, r.gate_shexp);
    mx(l.up_shexp, r.up_shexp);
    mx(l.down_shexp, r.down_shexp);
    bind(l.gate_exps_scale, r.gate_exps_scale);
    bind(l.up_exps_scale, r.up_exps_scale);
    bind(l.down_exps_scale, r.down_exps_scale);
    if (l.experts != nullptr) {
      // The CUTLASS layout fills each slot from its start: the layer's first
      // array's address less its offset in the expert group.
      const md::Qwen38Tensor& first = b.cutlass() ? r.gate_up_codes : r.gate_exps;
      kg::TensorArena::Bind(l.experts, m.places.array(first.index) - first.group_offset);
    } else {
      kg::TensorArena::Bind(l.gate_exps, m.places.array(r.gate_exps.index));
      kg::TensorArena::Bind(l.up_exps, m.places.array(r.up_exps.index));
      kg::TensorArena::Bind(l.down_exps, m.places.array(r.down_exps.index));
    }
    const auto state = [&](ggml_tensor* t, K kind) {
      if (t == nullptr) {
        return;
      }
      const std::int64_t i = m.state->Find(il, kind);
      kg::TensorArena::Bind(t,
                            m.places.state + m.state->tensors[static_cast<std::size_t>(i)].offset);
    };
    state(l.cache_k, K::kK);
    state(l.cache_v, K::kV);
    state(l.cache_idx, K::kIndexerK);
    state(l.conv_state, K::kConv);
    state(l.recurrent, K::kRecurrent);
    state(l.ple_state, K::kPleConv);
  }
}

std::expected<std::unique_ptr<Qwen38Planned>, std::string> PlanQwen38Chunk(
    const Qwen38Model& m, const kg::Qwen38ChunkShape& shape, const kg::DeviceChoices& choices,
    std::uint64_t activations, std::uint64_t activation_bytes, std::span<const std::string> keep) {
  auto out = std::make_unique<Qwen38Planned>();
  auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(*m.profile));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out->arena.emplace(std::move(*arena));
  auto graph =
      kg::BuildQwen38Graph(*out->arena, *m.profile, *m.binding, shape,
                           {.expert_stride = m.places.stride,
                            .fused = m.fused,
                            .experts = m.cutlass ? kg::Qwen38GraphOptions::Experts::kCutlass
                                                 : kg::Qwen38GraphOptions::Experts::kGgml});
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Qwen38Graph& g = out->graph;
  BindQwen38Weights(m, g);
  std::vector<ggml_tensor*> kept;
  for (const std::string& name : keep) {
    ggml_tensor* t = g.Named(name);
    if (t == nullptr) {
      return Error(std::format("the graph names no {}", name));
    }
    kept.push_back(t);
  }
  constexpr std::uint64_t kDistinct = std::uint64_t{1} << 46U;
  std::uint64_t leaf = kDistinct - (std::uint64_t{1} << 40U);
  const auto inputs = g.inputs();
  for (ggml_tensor* input : inputs) {
    kg::TensorArena::Bind(input, leaf);
    leaf += Round(ggml_nbytes(input), 256) + 256;
  }
  kg::BindDistinct(g.nodes, kDistinct);
  auto first = kg::PlanGraph(g.nodes, /*fusion=*/false, choices);
  if (!first) {
    return Error(first.error().detail);
  }
  auto placement = kg::PlaceActivations(g.nodes, *first, inputs, 256, kept);
  if (!placement) {
    return Error(placement.error().detail);
  }
  out->placement = std::move(*placement);
  for (ggml_tensor* input : inputs) {
    out->inputs_bytes += Round(ggml_nbytes(input), 256);
  }
  if (activations == 0) {
    out->plan = std::move(*first);
    return out;
  }
  if (out->placement.extent > activation_bytes) {
    return Error(std::format("the activations ({} bytes) exceed their region ({} bytes)",
                             out->placement.extent, activation_bytes));
  }
  for (const auto& [tensor, offset] : out->placement.offsets) {
    kg::TensorArena::Bind(tensor, activations + offset);
  }
  kg::BindViews(g.nodes);
  auto second = kg::PlanGraph(g.nodes, false, choices);
  if (!second) {
    return Error(second.error().detail);
  }
  if (!kg::SamePlan(*first, *second)) {
    return Error("the plan changed once the activations were placed");
  }
  out->plan = std::move(*second);
  return out;
}

void Qwen38Sources(const kg::Qwen38Graph& g, const md::Qwen38ChunkInputs& in, std::uint32_t outputs,
                   std::span<const std::int32_t> ple_rows, Qwen38HostInputs& out) {
  out.out_ids.resize(outputs);
  std::ranges::iota(out.out_ids, static_cast<std::int32_t>(in.rows - outputs));
  out.zero_row = 0;
  out.zero_index = 0;
  out.sources = {{g.tokens, in.tokens.data()},
                 {g.positions, in.positions.data()},
                 {g.cells, in.cells.data()},
                 {g.mask, in.mask.data()},
                 {g.ple_rows, ple_rows.empty() ? in.ple_rows.data() : ple_rows.data()},
                 {g.state_row, &out.zero_row},
                 {g.row_zero, &out.zero_index},
                 {g.out_ids, out.out_ids.data()}};
  if (in.qsa_select) {
    out.sources.emplace_back(g.mask_f32, in.mask_f32.data());
    out.sources.emplace_back(g.cell_block, in.qsa.cell_block.data());
    out.sources.emplace_back(g.block_cells, in.qsa.block_cells.data());
    out.sources.emplace_back(g.block_pos, in.qsa.block_pos.data());
    out.sources.emplace_back(g.block_bias, in.qsa.bias.data());
  }
}

}  // namespace jitllm::benchmarks
