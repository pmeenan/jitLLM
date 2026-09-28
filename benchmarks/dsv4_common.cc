// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "dsv4_common.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <format>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string_view>
#include <utility>

namespace jitllm::benchmarks {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

}  // namespace

void BindDsv4Weights(const Dsv4Model& m, kg::Dsv4Graph& g) {
  const auto bind = [&](ggml_tensor* t, const md::Dsv4Tensor& r) {
    if (t != nullptr) {
      kg::TensorArena::Bind(t, m.places.resource(r.index));
    }
  };
  const auto& b = *m.binding;
  bind(g.output_norm, b.output_norm);
  bind(g.output, b.output);
  bind(g.hc_head_fn, b.hc_head_fn);
  bind(g.hc_head_base, b.hc_head_base);
  bind(g.hc_head_scale, b.hc_head_scale);
  using K = md::Dsv4StateTensor::Kind;
  for (std::uint32_t il = 0; il < m.profile->layers; ++il) {
    const md::Dsv4Layer& r = b.layers[il];
    kg::Dsv4LayerTensors& l = g.layers[il];
    bind(l.attn_norm, r.attn_norm);
    bind(l.attn_sinks, r.attn_sinks);
    bind(l.q_a, r.q_a);
    bind(l.q_a_norm, r.q_a_norm);
    bind(l.q_b, r.q_b);
    bind(l.kv, r.kv);
    bind(l.kv_norm, r.kv_norm);
    bind(l.out_a, r.out_a);
    bind(l.out_b, r.out_b);
    bind(l.hc_attn_fn, r.hc_attn_fn);
    bind(l.hc_attn_base, r.hc_attn_base);
    bind(l.hc_attn_scale, r.hc_attn_scale);
    bind(l.hc_ffn_fn, r.hc_ffn_fn);
    bind(l.hc_ffn_base, r.hc_ffn_base);
    bind(l.hc_ffn_scale, r.hc_ffn_scale);
    bind(l.comp_kv, r.comp_kv);
    bind(l.comp_gate, r.comp_gate);
    bind(l.comp_ape, r.comp_ape);
    bind(l.comp_norm, r.comp_norm);
    bind(l.idx_q_b, r.idx_q_b);
    bind(l.idx_proj, r.idx_proj);
    bind(l.idx_comp_kv, r.idx_comp_kv);
    bind(l.idx_comp_gate, r.idx_comp_gate);
    bind(l.idx_comp_ape, r.idx_comp_ape);
    bind(l.idx_comp_norm, r.idx_comp_norm);
    bind(l.ffn_norm, r.ffn_norm);
    bind(l.router, r.router);
    bind(l.router_bias, r.router_bias);
    bind(l.tid2eid, r.tid2eid);
    bind(l.up_shexp, r.up_shexp);
    bind(l.gate_shexp, r.gate_shexp);
    bind(l.down_shexp, r.down_shexp);
    kg::TensorArena::Bind(l.up_exps, m.places.array(r.up_exps.index));
    kg::TensorArena::Bind(l.gate_exps, m.places.array(r.gate_exps.index));
    kg::TensorArena::Bind(l.down_exps, m.places.array(r.down_exps.index));
    const auto state = [&](ggml_tensor* t, K kind) {
      if (t == nullptr) {
        return;
      }
      const std::int64_t i = m.state->Find(il, kind);
      kg::TensorArena::Bind(t,
                            m.places.state + m.state->tensors[static_cast<std::size_t>(i)].offset);
    };
    state(l.raw_k, K::kRawK);
    state(l.csa_k, K::kCsaK);
    state(l.csa_state_kv, K::kCsaStateKv);
    state(l.csa_state_score, K::kCsaStateScore);
    state(l.lid_k, K::kLidK);
    state(l.lid_state_kv, K::kLidStateKv);
    state(l.lid_state_score, K::kLidStateScore);
    state(l.hca_k, K::kHcaK);
    state(l.hca_state_kv, K::kHcaStateKv);
    state(l.hca_state_score, K::kHcaStateScore);
  }
}

std::expected<std::unique_ptr<Dsv4Planned>, std::string> PlanDsv4Chunk(
    const Dsv4Model& m, const kg::Dsv4ChunkShape& shape, const kg::DeviceChoices& choices,
    std::span<const std::string> keep_names, std::uint64_t activations,
    std::uint64_t activation_bytes) {
  auto out = std::make_unique<Dsv4Planned>();
  auto arena = kg::TensorArena::Create(kg::Dsv4GraphTensors(*m.profile));
  if (!arena) {
    return Error(arena.error().detail);
  }
  out->arena.emplace(std::move(*arena));
  auto graph = kg::BuildDsv4Graph(*out->arena, *m.profile, *m.binding, shape,
                                  {.expert_stride = m.places.stride});
  if (!graph) {
    return Error(graph.error().detail);
  }
  out->graph = std::move(*graph);
  kg::Dsv4Graph& g = out->graph;
  BindDsv4Weights(m, g);
  std::vector<ggml_tensor*> keep;
  for (const std::string& name : keep_names) {
    if (ggml_tensor* t = g.Named(name); t != nullptr) {
      keep.push_back(t);
    } else {
      return Error(std::format("the graph names no {}", name));
    }
  }
  // First pass: every computed tensor at its own address.
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
  auto placement = kg::PlaceActivations(g.nodes, *first, inputs, 256, keep);
  if (!placement) {
    return Error(placement.error().detail);
  }
  out->placement = std::move(*placement);
  for (ggml_tensor* input : inputs) {
    out->inputs_bytes += ggml_nbytes(input);
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

std::expected<void, std::string> BuildDsv4Inputs(const Dsv4Model& m, const kg::Dsv4Graph& g,
                                                 const md::Dsv4ChunkInputs& in,
                                                 std::span<const std::int32_t> tokens,
                                                 std::span<const std::byte> table,
                                                 Dsv4HostInputs& out) {
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  // The embedding rows, dequantized on the host.
  const md::Dsv4Tensor& embedding = m.binding->token_embd;
  auto type = kg::GgmlTypeOf(embedding.type);
  if (!type) {
    return Error(type.error().detail);
  }
  const auto* traits = ggml_get_type_traits(*type);
  const std::uint64_t row_bytes = ggml_row_size(*type, m.profile->width);
  const std::uint64_t table_offset = m.artifact->resources()[embedding.index].offset.value();
  out.embd.assign(std::size_t{rows} * m.profile->width, 0.0F);
  for (std::uint32_t i = 0; i < rows; ++i) {
    if (tokens[i] < 0 || std::cmp_greater_equal(tokens[i], m.profile->vocab)) {
      return Error(std::format("token {} is outside the vocabulary", tokens[i]));
    }
    const std::uint64_t at = table_offset + (static_cast<std::uint64_t>(tokens[i]) * row_bytes);
    if (at + row_bytes > table.size()) {
      return Error("the token table is shorter than its rows");
    }
    traits->to_float(table.data() + at, out.embd.data() + (std::size_t{i} * m.profile->width),
                     m.profile->width);
  }
  out.tokens.assign(tokens.begin(), tokens.end());
  out.out_ids.resize(rows);
  std::ranges::iota(out.out_ids, 0);
  out.zeros.assign(static_cast<std::size_t>(ggml_nelements(g.top_k_zeros)), 0);
  const auto comp = [](const kg::Dsv4CompInputs& t, const md::Dsv4CompPlan& plan,
                       const std::vector<std::uint16_t>& mask) {
    return std::array<std::pair<ggml_tensor*, const void*>, 7>{
        {{t.state_pos, plan.state_pos.data()},
         {t.persist_src, plan.persist_src.data()},
         {t.persist_dst, plan.persist_dst.data()},
         {t.read_idxs, plan.read_idxs.data()},
         {t.write_idxs, plan.write_idxs.data()},
         {t.write_pos, plan.write_pos.data()},
         {t.mask, mask.data()}}};
  };
  out.sources = {{g.embd, out.embd.data()},          {g.tokens, out.tokens.data()},
                 {g.positions, in.positions.data()}, {g.raw_k_idxs, in.raw_cells.data()},
                 {g.raw_mask, in.raw_mask.data()},   {g.out_ids, out.out_ids.data()}};
  for (const auto& s : comp(g.csa, in.csa, in.csa_mask)) {
    out.sources.push_back(s);
  }
  for (const auto& s : comp(g.hca, in.hca, in.hca_mask)) {
    out.sources.push_back(s);
  }
  for (const auto& s : comp(g.lid, in.lid, in.lid_mask)) {
    out.sources.push_back(s);
  }
  out.sources.emplace_back(g.lid_rot, m.rot.data());
  out.sources.emplace_back(g.top_k_zeros, out.zeros.data());
  return {};
}

std::expected<std::vector<TokenLine>, std::string> ReadTokenLines(const std::filesystem::path& p) {
  std::ifstream file(p);
  if (!file) {
    return Error(std::format("cannot read {}", p.string()));
  }
  std::vector<TokenLine> out;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    TokenLine t;
    const auto tab = line.find('\t');
    std::string_view rest = line;
    if (tab != std::string::npos) {
      t.name = line.substr(0, tab);
      rest = std::string_view(line).substr(tab + 1);
    }
    std::istringstream ids{std::string(rest)};
    std::int64_t id = 0;
    while (ids >> id) {
      t.ids.push_back(static_cast<std::int32_t>(id));
    }
    out.push_back(std::move(t));
  }
  return out;
}

std::int32_t Argmax(std::span<const float> row) {
  return static_cast<std::int32_t>(std::ranges::max_element(row) - row.begin());
}

double Nll(std::span<const float> row, std::int32_t target) {
  const double most = *std::ranges::max_element(row);
  double sum = 0;
  for (const float v : row) {
    sum += std::exp(static_cast<double>(v) - most);
  }
  return (most + std::log(sum)) - static_cast<double>(row[static_cast<std::size_t>(target)]);
}

std::expected<void, std::string> WriteFloats(const std::filesystem::path& p,
                                             std::span<const float> v) {
  std::ofstream out(p, std::ios::binary);
  out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size_bytes()));
  return out ? std::expected<void, std::string>{}
             : Error(std::format("cannot write {}", p.string()));
}

}  // namespace jitllm::benchmarks
