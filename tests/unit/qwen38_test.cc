// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The Qwen3.8 Flash Next adapter (model/qwen38.h) and its chunk graph
// (kernels/ggml/qwen38_graph.h), in every profile:
// - the binding of a synthetic resource list shaped and typed as the
//   artifact import_m3.py writes from Mia's checkpoint, and its refusals;
// - the n-gram hash's constants checked against the table, and its rows
//   worked by hand (llm_graph_input_ple::set_input at b29c606e2);
// - the state layout's sizes, and a chunk's masks, positions and QSA block
//   tables against llama.cpp's rules (set_input_qsa), worked by hand;
// - the graph at prefill, decode and past-the-budget shapes: every node
//   planned by an implementation of this module (a model of the device's
//   choices), MXFP8 products by jitLLM's vector product or the BF16
//   dequantization, the activations placed.

#include "model/qwen38.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "expected_error.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/qwen38_graph.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate_ext.h"

namespace {

namespace md = jitllm::model;
namespace kg = jitllm::kernels::ggml;
using jitllm::test_support::Failed;

template <typename T>
std::string Why(const std::expected<T, std::string>& result) {
  return Failed(result).value_or(std::string());
}
template <typename T>
std::string Why(const std::expected<T, kg::KernelFailure>& result) {
  return Failed(result, &kg::KernelFailure::detail).value_or(std::string());
}

constexpr std::uint64_t kTableRows = 320001536;
// One NVFP4 expert slice (640 rows of 40 blocks, or 2,560 of 10, at 36
// bytes), down's over-read (384 elements), the stored group (2,768,896
// bytes) and the slab's stride over it (whole 36-byte blocks and 16 bytes).
constexpr std::uint64_t kExpertSlice = 921600;
constexpr std::uint64_t kDownOverRead = 216;
constexpr std::uint64_t kExpertStride = 2768976;

// The artifact's resources (docs/experiments/artifact-layout/modelopt_qwen38.py):
// GGML BF16 and F32, plain MXFP8 and I64 and the n-gram table, then the
// routed experts' NVFP4 arrays.
std::vector<md::Qwen38Resource> ArtifactLike(const md::Qwen38Profile& p) {
  std::vector<md::Qwen38Resource> r;
  std::vector<md::Qwen38Resource> arrays;
  const auto ggml = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = false, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto plain = [&](std::string name, std::string type, std::vector<std::uint64_t> ne) {
    r.push_back(
        {.roles = {std::move(name)}, .plain = true, .type = std::move(type), .ne = std::move(ne)});
  };
  const auto mx = [&](const std::string& name, std::uint64_t k, std::uint64_t n) {
    plain(name + ".weight", "F8_E4M3", {k, n});
    plain(name + ".weight_scale", "U8", {k / 32, n});
  };
  ggml("token_embd.weight", "BF16", {2560, 248320});
  ggml("output.weight", "BF16", {2560, 248320});
  ggml("output_hc_norm.weight", "F32", {10240});
  ggml("output_hc_down.weight", "BF16", {10240, 320});
  ggml("output_hc_up.weight", "BF16", {320, 10240});
  plain("per_layer_token_embd.weight", "U8", {90, kTableRows});
  plain("per_layer_token_embd.weight_scale_2", "F32", {1});
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    const std::string n = std::format("blk.{}.", il);
    for (const char* kind : {"attn", "ffn"}) {
      ggml(std::format("{}hc_{}_norm.weight", n, kind), "F32", {10240});
      ggml(std::format("{}hc_{}_down.weight", n, kind), "BF16", {10240, 320});
      ggml(std::format("{}hc_{}_up.weight", n, kind), "BF16", {320, 10240});
      ggml(std::format("{}hc_{}_inject.weight", n, kind), "BF16", {10240, 4});
    }
    if (p.linear(il)) {
      mx(n + "attn_qkv", 2560, 10240);
      mx(n + "attn_gate", 2560, 6144);
      mx(n + "ssm_beta", 2560, 48);
      mx(n + "ssm_alpha", 2560, 48);
      ggml(n + "ssm_dt.bias", "F32", {48});
      ggml(n + "ssm_a", "F32", {48});
      ggml(n + "ssm_conv1d.weight", "F32", {4, 10240});
      ggml(n + "ssm_norm.weight", "F32", {128});
      mx(n + "ssm_out", 6144, 2560);
    } else {
      mx(n + "attn_q", 2560, 12288);
      mx(n + "attn_k", 2560, 512);
      mx(n + "attn_v", 2560, 512);
      mx(n + "attn_output", 6144, 2560);
      ggml(n + "attn_q_norm.weight", "F32", {256});
      ggml(n + "attn_k_norm.weight", "F32", {256});
      mx(n + "indexer.qk_proj", 2560, 640);
      ggml(n + "indexer.q_norm.weight", "F32", {128});
      ggml(n + "indexer.k_norm.weight", "F32", {128});
    }
    if (il == 1) {
      ggml(n + "ple_key.weight", "BF16", {2560, 10240});
      ggml(n + "ple_value.weight", "BF16", {2560, 2560});
      for (const char* part : {"key", "query", "conv"}) {
        ggml(std::format("{}ple_norm_{}.weight", n, part), "F32", {10240});
      }
      ggml(n + "ple_conv1d.weight", "F32", {4, 10240});
      plain(n + "ple_multipliers", "I64", {3});
      plain(n + "ple_head_offsets", "I64", {16});
      plain(n + "ple_head_vocab", "I64", {16});
    }
    ggml(n + "ffn_gate_inp.weight", "BF16", {2560, 512});
    ggml(n + "ffn_gate_inp_shexp.weight", "BF16", {2560});
    mx(n + "ffn_gate_shexp", 2560, 640);
    mx(n + "ffn_up_shexp", 2560, 640);
    mx(n + "ffn_down_shexp", 640, 2560);
    // Each expert group as the importer writes it: gate, up and down slices
    // of 921,600 bytes, down's rows padded to 1,024 elements (216 bytes of
    // over-read after its last row).
    std::uint64_t group_offset = 0;
    for (const char* proj : {"gate", "up", "down"}) {
      const bool down = std::string_view(proj) == "down";
      ggml(std::format("{}ffn_{}_exps.weight_scale_2", n, proj), "F32", {512});
      arrays.push_back({.roles = {std::format("{}ffn_{}_exps.weight", n, proj)},
                        .plain = false,
                        .type = "NVFP4",
                        .ne = down ? std::vector<std::uint64_t>{640, 2560}
                                   : std::vector<std::uint64_t>{2560, 640},
                        .expert_array = true,
                        .count = 512,
                        .group_offset = group_offset,
                        .readable = kExpertSlice + (down ? kDownOverRead : 0)});
      group_offset += kExpertSlice;
    }
  }
  r.insert(r.end(), arrays.begin(), arrays.end());
  return r;
}

// Hash constants of the checkpoint's shape: 16 heads of 20,000,008 rows.
md::Qwen38PleHash Hash() {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  std::vector<std::int64_t> m = {3, 5, 7};
  std::vector<std::int64_t> offsets;
  std::vector<std::int64_t> vocab;
  for (std::int64_t h = 0; h < 16; ++h) {
    offsets.push_back(h * 20000096);
    vocab.push_back(20000096 - (h * 8));
  }
  return md::CheckQwen38PleHash(p, m, offsets, vocab, kTableRows).value();
}

TEST(Qwen38Test, TheProfileIsQwen38FlashNext) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  EXPECT_EQ(p.layers, 48U);
  EXPECT_EQ(p.hc_width(), 10240U);
  EXPECT_EQ(p.conv_channels(), 10240U);
  EXPECT_EQ(p.ple_heads(), 16U);
  EXPECT_EQ(p.ple_width(), 2560U);
  EXPECT_EQ(p.ple_history(), 9U);
  // 12 x (3 linear, then QSA).
  int qsa = 0;
  for (std::uint32_t il = 0; il < p.layers; ++il) {
    qsa += p.linear(il) ? 0 : 1;
    EXPECT_EQ(p.linear(il), il % 4 != 3) << il;
  }
  EXPECT_EQ(qsa, 12);
  EXPECT_TRUE(p.linear(p.ple_layer));
}

TEST(Qwen38Test, BindsTheArtifactsTensorsAndRefusesWhatDiffers) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto bound = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(bound.has_value()) << Why(bound);
  EXPECT_TRUE(bound->layers[0].linear);
  EXPECT_FALSE(bound->layers[3].linear);
  EXPECT_TRUE(bound->layers[3].q.codes.plain);
  EXPECT_EQ(bound->layers[3].q.codes.ne, (std::vector<std::uint64_t>{2560, 12288}));
  EXPECT_EQ(bound->ple_table.ne[1], kTableRows);
  EXPECT_EQ(bound->layers[47].down_exps.type, "NVFP4");

  EXPECT_NE(Why(md::BindQwen38(p, "qwen3next", resources)).find("qwen4exp"), std::string::npos);
  auto missing = resources;
  missing.erase(missing.begin() + 3);
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", missing)).find("has no"), std::string::npos);
  auto extra = resources;
  extra.push_back({.roles = {"blk.0.surprise"}, .plain = false, .type = "F32", .ne = {4}});
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", extra)).find("does not read"), std::string::npos);
  auto reshaped = resources;
  for (md::Qwen38Resource& r : reshaped) {
    if (r.roles[0] == "blk.3.attn_k.weight") {
      r.ne = {2560, 256};
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", reshaped)).find("blk.3.attn_k.weight"),
            std::string::npos);
  auto retyped = resources;
  for (md::Qwen38Resource& r : retyped) {
    if (r.roles[0] == "blk.3.attn_k.weight") {
      r.plain = false;  // the same bytes, but claimed as a GGML tensor
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", retyped)).find("plain"), std::string::npos);
  auto few = resources;
  for (md::Qwen38Resource& r : few) {
    if (r.expert_array) {
      r.count = 256;
    }
  }
  EXPECT_NE(Why(md::BindQwen38(p, "qwen4exp", few)).find("experts"), std::string::npos);
}

TEST(Qwen38Test, TheNgramHashMustStayInsideTheTable) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<std::int64_t> m = {3, 5, 7};
  std::vector<std::int64_t> offsets(16, 0);
  std::vector<std::int64_t> vocab(16, 100);
  EXPECT_TRUE(md::CheckQwen38PleHash(p, m, offsets, vocab, 100).has_value());
  vocab[5] = 101;
  EXPECT_NE(Why(md::CheckQwen38PleHash(p, m, offsets, vocab, 100)).find("head 5"),
            std::string::npos);
  vocab[5] = 0;
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, 100).has_value());
  vocab[5] = 100;
  offsets[2] = -1;
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, 100).has_value());
  offsets[2] = 0;
  // Past the I32 row index, whatever the table holds.
  offsets[0] = std::numeric_limits<std::int32_t>::max();
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, std::uint64_t{1} << 40).has_value());
  offsets[0] = 0;
  EXPECT_FALSE(
      md::CheckQwen38PleHash(p, std::vector<std::int64_t>{3, 5}, offsets, vocab, 100).has_value());
  EXPECT_FALSE(md::CheckQwen38PleHash(p, m, offsets, vocab, 0).has_value());
  // A hash not made by the check is checked again before any row is hashed.
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  const std::vector<std::int32_t> history(4, 1000);
  md::Qwen38PleHash built = md::CheckQwen38PleHash(p, m, offsets, vocab, 100).value();
  ASSERT_TRUE(md::Qwen38Chunk(p, *state, built, history, 0, 4).has_value());
  built.vocab[7] = 0;
  EXPECT_NE(Why(md::Qwen38Chunk(p, *state, built, history, 0, 4)).find("head 7"),
            std::string::npos);
  built.vocab[7] = 100;
  built.offsets[3] = 1;
  EXPECT_NE(Why(md::Qwen38Chunk(p, *state, built, history, 0, 4)).find("head 3"),
            std::string::npos);
  built.offsets[3] = 0;
  built.table_rows = 99;
  EXPECT_FALSE(md::Qwen38Chunk(p, *state, built, history, 0, 4).has_value());
}

TEST(Qwen38Test, NgramRowsFollowLlamaCppsHash) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const md::Qwen38PleHash h = Hash();
  const std::vector<std::int32_t> history = {11, 22, 33, p.ple_eos, 44, 55};
  const auto rows_of = [&](std::uint64_t mixed2, std::uint64_t mixed3) {
    std::vector<std::int32_t> want;
    for (std::uint32_t head = 0; head < 16; ++head) {
      const std::uint64_t mixed = head < 8 ? mixed2 : mixed3;
      want.push_back(static_cast<std::int32_t>((mixed % h.vocab[head]) + h.offsets[head]));
    }
    return want;
  };
  const auto eos = static_cast<std::uint64_t>(p.ple_eos);
  // Position 0: both predecessors are missing and read as EOS.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 0),
            rows_of((11UL * 3UL) ^ (eos * 5UL), (11UL * 3UL) ^ (eos * 5UL) ^ (eos * 7UL)));
  // Position 2: a full window.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 2),
            rows_of((33UL * 3UL) ^ (22UL * 5UL), (33UL * 3UL) ^ (22UL * 5UL) ^ (11UL * 7UL)));
  // Position 3: the token is EOS; its own EOS does not cut its context.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 3),
            rows_of((eos * 3UL) ^ (33UL * 5UL), (eos * 3UL) ^ (33UL * 5UL) ^ (22UL * 7UL)));
  // Position 5: the EOS two back cuts the window there.
  EXPECT_EQ(md::Qwen38PleRows(p, h, history, 5),
            rows_of((55 * 3) ^ (44 * 5), (55 * 3) ^ (44 * 5) ^ (eos * 7)));
}

TEST(Qwen38Test, TheStateIsBoundedAndSized) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4000, 512);
  ASSERT_TRUE(s.has_value()) << Why(s);
  EXPECT_EQ(s->cells, 4096U);
  using K = md::Qwen38StateTensor::Kind;
  const auto& k = s->tensors[static_cast<std::size_t>(s->Find(3, K::kK))];
  EXPECT_EQ(k.bytes, 512ULL * 4096 * 2);
  const auto& rec = s->tensors[static_cast<std::size_t>(s->Find(0, K::kRecurrent))];
  EXPECT_EQ(rec.bytes, 128ULL * 128 * 48 * 4);
  const auto& conv = s->tensors[static_cast<std::size_t>(s->Find(0, K::kConv))];
  EXPECT_EQ(conv.bytes, 3ULL * 10240 * 4);
  const auto& ple = s->tensors[static_cast<std::size_t>(s->Find(1, K::kPleConv))];
  EXPECT_EQ(ple.bytes, 9ULL * 10240 * 4);
  EXPECT_EQ(s->Find(0, K::kPleConv), -1);
  EXPECT_EQ(s->Find(0, K::kK), -1);
  std::uint64_t kv = 0;
  for (const auto& t : s->tensors) {
    EXPECT_EQ(t.offset % 256, 0U);
    EXPECT_LE(t.offset + t.bytes, s->bytes);
    kv += t.kind == K::kK || t.kind == K::kV ? t.bytes : 0;
  }
  const auto reps = s->Representations();
  ASSERT_EQ(reps.size(), 3U);
  EXPECT_EQ(reps[0].block_bytes.value(), kv);
  EXPECT_FALSE(md::Qwen38State(p, 0, 1).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 16, 0).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 16, 32).has_value());
  // Chunks wider than the bound, or whose masks would pass I32 elements.
  EXPECT_FALSE(md::Qwen38State(p, 65536, md::kQwen38MaxRows + 1).has_value());
  EXPECT_TRUE(md::Qwen38State(p, 262144, 8191).has_value());
  EXPECT_FALSE(md::Qwen38State(p, 262400, 8192).has_value());
  EXPECT_FALSE(
      md::Qwen38State(p, static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()), 1)
          .has_value());
}

TEST(Qwen38Test, AChunksMaskAndPositionsAreCausal) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  const md::Qwen38PleHash h = Hash();
  std::vector<std::int32_t> history(40, 7);
  auto c = md::Qwen38Chunk(p, *s, h, history, 37, 3);
  ASSERT_TRUE(c.has_value()) << Why(c);
  EXPECT_EQ(c->n_kv, 256U);
  EXPECT_FALSE(c->qsa_select);
  EXPECT_EQ(c->cells, (std::vector<std::int64_t>{37, 38, 39}));
  EXPECT_EQ(c->positions,
            (std::vector<std::int32_t>{37, 38, 39, 37, 38, 39, 37, 38, 39, 37, 38, 39}));
  for (std::uint32_t i = 0; i < 3; ++i) {
    for (std::uint32_t j = 0; j < 256; ++j) {
      const bool visible = j <= 37 + i;
      EXPECT_EQ(c->mask[(i * 256) + j], visible ? md::kQwen38HalfZero : md::kQwen38HalfNegInf);
      EXPECT_EQ(c->mask_f32[(i * 256) + j] == 0.0f, visible);
    }
  }
  EXPECT_EQ(c->ple_rows.size(), 48U);
  // Refusals: history of the wrong length, a chunk past the context, a
  // token outside the vocabulary.
  EXPECT_FALSE(md::Qwen38Chunk(p, *s, h, history, 36, 3).has_value());
  std::vector<std::int32_t> long_history(4097, 7);
  EXPECT_FALSE(md::Qwen38Chunk(p, *s, h, long_history, 4096, 1).has_value());
  history[39] = static_cast<std::int32_t>(p.vocab);
  EXPECT_FALSE(md::Qwen38Chunk(p, *s, h, history, 37, 3).has_value());
}

TEST(Qwen38Test, PastTheBudgetQsaSelectsWholeBlocksAndTheTail) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  auto s = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(s.has_value());
  const md::Qwen38PleHash h = Hash();
  // 2,302 positions: n_kv 2,304 > 2,051, so the indexer selects.
  std::vector<std::int32_t> history(2302, 9);
  auto c = md::Qwen38Chunk(p, *s, h, history, 2300, 2);
  ASSERT_TRUE(c.has_value()) << Why(c);
  EXPECT_EQ(c->n_kv, 2304U);
  ASSERT_TRUE(c->qsa_select);
  const md::Qwen38QsaInputs& q = c->qsa;
  EXPECT_EQ(q.blocks, 576U);
  // 575 full blocks (positions 0..2299); cells 2300 and 2301 and the empty
  // cells point at the spare block 575.
  EXPECT_EQ(q.cell_block[0], 0);
  EXPECT_EQ(q.cell_block[2299], 574);
  EXPECT_EQ(q.cell_block[2300], 575);
  EXPECT_EQ(q.cell_block[2303], 575);
  EXPECT_EQ(q.block_cells[(574 * 4) + 3], 2299);
  EXPECT_EQ(q.block_cells[575UL * 4UL], 0);
  EXPECT_EQ(q.block_pos[574], 2296);
  EXPECT_EQ(q.block_pos[(3 * 576) + 574], 2296);
  // Token at 2300: its tail starts at 2300, so every full block is scored
  // (0) and the spare one, the tail, kept (1e9).
  for (std::uint32_t row = 0; row < 2; ++row) {
    const float* bias = q.bias.data() + (std::size_t{row} * 576);
    EXPECT_EQ(bias[0], 0.0f);
    EXPECT_EQ(bias[574], 0.0f);
    EXPECT_EQ(bias[575], 1e9f);
  }
  // Mid-block, in a prefill chunk: block 574 (positions 2296-2299) is the
  // tail of a token at 2298 (kept, 1e9) but a scored block for one at 2299,
  // whose tail starts at 2300.
  std::vector<std::int32_t> prefill(2302, 9);
  auto d = md::Qwen38Chunk(p, *s, h, prefill, 2296, 6);
  ASSERT_TRUE(d.has_value());
  EXPECT_EQ(d->qsa.bias[(std::size_t{2} * 576) + 574], 1e9f);  // token 2298
  EXPECT_EQ(d->qsa.bias[(std::size_t{3} * 576) + 574], 0.0f);  // token 2299
  EXPECT_EQ(d->qsa.bias[(std::size_t{3} * 576) + 573], 0.0f);
}

kg::DeviceChoices ModelDevice() {
  return {
      .mul_mat = [](const ggml_tensor* node) -> std::expected<kg::MulMatPath, kg::KernelFailure> {
        const std::int64_t columns = node->src[1]->ne[1] * node->src[1]->ne[2];
        if (columns == 1) {
          return kg::MulMatPath::kVector;
        }
        return columns <= 16 ? kg::MulMatPath::kTensorCore : kg::MulMatPath::kCublas;
      },
      .vector_fusible = [](const ggml_tensor*) { return false; },
      .quant =
          [](const ggml_tensor* node) -> std::expected<kg::QuantMulMatPath, kg::KernelFailure> {
        const std::int64_t columns =
            node->op == GGML_OP_MUL_MAT_ID ? node->src[2]->ne[1] : node->src[1]->ne[1];
        return columns <= 8 ? kg::QuantMulMatPath::kVector : kg::QuantMulMatPath::kTile;
      }};
}

TEST(Qwen38Test, TheChunkGraphIsPlannedByThisModulesImplementations) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value()) << Why(binding);
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  const md::Qwen38PleHash h = Hash();
  const std::vector<std::uint64_t> strides(p.layers, kExpertStride);
  for (const auto& [n_past, rows] : {std::pair{0U, 37U}, {37U, 1U}, {2800U, 512U}, {4095U, 1U}}) {
    std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
    auto chunk = md::Qwen38Chunk(p, *state, h, history, n_past, rows);
    ASSERT_TRUE(chunk.has_value()) << Why(chunk);
    const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, rows);
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    auto graph = kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    std::uint64_t next = std::uint64_t{1} << 40U;
    const auto bind_leaf = [&](ggml_tensor* t) {
      if (t != nullptr && t->data == nullptr) {
        kg::TensorArena::Bind(t, next);
        next += ((ggml_nbytes(t) + 255) / 256 * 256) + 256;
      }
    };
    for (ggml_tensor* t : graph->inputs()) {
      bind_leaf(t);
    }
    for (ggml_tensor* node : graph->nodes) {
      for (ggml_tensor* src : node->src) {
        if (src != nullptr && src->op == GGML_OP_NONE && src->view_src == nullptr) {
          bind_leaf(src);
        }
      }
    }
    kg::BindDistinct(graph->nodes, std::uint64_t{1} << 46U);
    auto plan = kg::PlanGraph(graph->nodes, false, ModelDevice());
    ASSERT_TRUE(plan.has_value()) << rows << " at " << n_past << ": " << Why(plan);
    std::set<std::string_view> used;
    for (const auto& step : plan->steps) {
      used.insert(step.implementation);
    }
    for (const std::string_view name :
         {kg::kNvfp4RowsName, kg::kSsmConvName, kg::kGatedDeltaNetName, kg::kFlashAttnMmaName,
          kg::kArgsortName, kg::kRopeExtName, kg::kSetRowsExtName, kg::kConcatName,
          kg::kSumRowsName, kg::kRepeatName}) {
      EXPECT_TRUE(used.contains(name)) << name << " at " << rows << " rows";
    }
    EXPECT_TRUE(used.contains(rows <= 8 ? kg::kMulMatIdVecQ : kg::kMulMatIdQ));
    EXPECT_TRUE(used.contains(rows <= 8 ? kg::kMxfp8MulMatVecName : kg::kMxfp8DequantName));
    EXPECT_EQ(used.contains(kg::kTopKName), chunk->qsa_select) << rows << " at " << n_past;
    auto placed = kg::PlaceActivations(graph->nodes, *plan, graph->inputs(), 256);
    ASSERT_TRUE(placed.has_value()) << Why(placed);
    EXPECT_NE(graph->Named("l_last-47"), nullptr);
    EXPECT_NE(graph->Named("result_output"), nullptr);
    EXPECT_EQ(graph->logits->ne[0], 248320);
    EXPECT_EQ(graph->logits->ne[1], rows);
  }
}

TEST(Qwen38Test, ExpertStridesAreWholeBlocks) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value());
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(1, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 0, 1);
  ASSERT_TRUE(chunk.has_value());
  const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, 1);
  auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(arena.has_value());
  std::vector<std::uint64_t> strides(p.layers, 2768905);  // not whole 36-byte blocks
  const auto torn = kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(torn).find("expert stride"), std::string::npos) << Why(torn);
  auto unaligned = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(unaligned.has_value());
  strides.assign(p.layers, 2768904);  // whole blocks, but not 16-byte aligned
  const auto skew =
      kg::BuildQwen38Graph(*unaligned, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(skew).find("16-byte aligned"), std::string::npos) << Why(skew);
  auto again = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
  ASSERT_TRUE(again.has_value());
  strides.assign(3, kExpertStride);
  const auto few = kg::BuildQwen38Graph(*again, p, *binding, shape, {.expert_stride = strides});
  EXPECT_NE(Why(few).find("per layer"), std::string::npos) << Why(few);
}

// A shape whose QSA selection is not the indexer budget's is refused: past
// the budget without selection, selection within it, or blocks that do not
// cover the cells.
TEST(Qwen38Test, TheGraphRefusesASelectionThatIsNotTheBudgets) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value());
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(2800, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 2799, 1);
  ASSERT_TRUE(chunk.has_value());
  ASSERT_TRUE(chunk->qsa_select);
  const kg::Qwen38ChunkShape past = kg::Qwen38ShapeOf(*state, *chunk, 1);
  const std::vector<std::uint64_t> strides(p.layers, kExpertStride);
  const auto build = [&](const kg::Qwen38ChunkShape& shape) {
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    EXPECT_TRUE(arena.has_value());
    return Why(kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides}));
  };
  EXPECT_EQ(build(past), "");
  kg::Qwen38ChunkShape s = past;
  s.qsa_select = false;
  s.qsa_blocks = 0;
  EXPECT_NE(build(s).find("indexer budget"), std::string::npos);
  s = past;
  s.qsa_blocks -= 1;
  EXPECT_NE(build(s).find("indexer budget"), std::string::npos);
  s = past;
  s.n_kv = 2048;
  EXPECT_NE(build(s).find("indexer budget"), std::string::npos);
}

// The down projection's 640-element rows read past each slice; only a
// stride that holds the artifact's readable bytes for every expert, the
// last one's included, marks them readable (else the products refuse them).
TEST(Qwen38Test, ShortExpertRowsAreReadableOnlyInsideTheStride) {
  const md::Qwen38Profile& p = md::Qwen38Flash();
  const std::vector<md::Qwen38Resource> resources = ArtifactLike(p);
  auto binding = md::BindQwen38(p, "qwen4exp", resources);
  ASSERT_TRUE(binding.has_value());
  EXPECT_EQ(binding->layers[0].down_exps.group_offset, 2 * kExpertSlice);
  EXPECT_EQ(binding->layers[0].down_exps.readable, kExpertSlice + kDownOverRead);
  auto state = md::Qwen38State(p, 4096, 512);
  ASSERT_TRUE(state.has_value());
  std::vector<std::int32_t> history(1, 1000);
  auto chunk = md::Qwen38Chunk(p, *state, Hash(), history, 0, 1);
  ASSERT_TRUE(chunk.has_value());
  const kg::Qwen38ChunkShape shape = kg::Qwen38ShapeOf(*state, *chunk, 1);
  // Whether every NVFP4 weight the graph reads with short rows is marked,
  // and how many there are.
  const auto marked = [](const kg::Qwen38Graph& g) {
    std::set<const ggml_tensor*> short_rows;
    bool all = true;
    for (const ggml_tensor* node : g.nodes) {
      for (const ggml_tensor* src : node->src) {
        if (src != nullptr && src->type == GGML_TYPE_NVFP4) {
          if (src->ne[0] % 512 != 0) {
            short_rows.insert(src);
            all = all && kg::RowPaddingReadable(src);
          } else {
            EXPECT_FALSE(kg::RowPaddingReadable(src));
          }
        }
      }
    }
    return std::pair{all, short_rows.size()};
  };
  // 2,765,016 bytes are used by the group; each stride is a multiple of 144
  // (2,764,944 the largest short of it).
  for (const auto& [stride, readable] : {std::pair{kExpertStride, true},
                                         {std::uint64_t{2764944}, false},
                                         {std::uint64_t{0}, false}}) {
    auto arena = kg::TensorArena::Create(kg::Qwen38GraphTensors(p));
    ASSERT_TRUE(arena.has_value());
    std::vector<std::uint64_t> strides;
    if (stride != 0) {
      strides.assign(p.layers, stride);
    }
    auto graph = kg::BuildQwen38Graph(*arena, p, *binding, shape, {.expert_stride = strides});
    ASSERT_TRUE(graph.has_value()) << Why(graph);
    const auto [all, count] = marked(*graph);
    EXPECT_EQ(count, p.layers) << stride;
    EXPECT_EQ(all, readable) << stride;
  }
}

}  // namespace
