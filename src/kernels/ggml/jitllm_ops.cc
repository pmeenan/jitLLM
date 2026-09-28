// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/jitllm_ops.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <utility>

#include "cutlass/version.h"
#include "ggml.h"
#include "kernels/ggml/moe_layout.h"
#include "kernels/ggml/validate_ext.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {

// The CUTLASS layout's scale swizzle (moe_cutlass.h SfOffset), which the
// checks here, the conversions and the importer's artifacts (modelopt_qwen38.py)
// all write out by hand, is CUTLASS 4.7.1's block-scaled layout, the source
// lock's `cutlass`: another CUTLASS is reviewed against it before the lock
// moves. (Every profile compiles this, the CPU-only one too, whose receipt
// lists the component.)
static_assert(CUTLASS_MAJOR == 4 && CUTLASS_MINOR == 7 && CUTLASS_PATCH == 1,
              "the CUTLASS expert layout was written against CUTLASS 4.7.1");

namespace {

// GGML's layout of a custom node's op_params (ggml-impl.h
// ggml_custom_op_params), which ggml_custom_4d fills.
struct JitllmCustomParams {
  ggml_custom_op_t fun;
  int n_tasks;
  void* userdata;
};

using detail::Aligned;
using detail::AlignedEverywhere;
using detail::AllCurrent;
using detail::AllSane;
using detail::AnyEmpty;
using detail::Bound;
using detail::Disjoint;
using detail::IsF32;
using detail::kInt32Max;
using detail::Packed;
using detail::Rejected;

// A jitLLM node's function, never called, and its operation's name: the
// userdata points at one of these distinct objects (distinct addresses,
// unlike functions a linker may fold).
void JitllmCustomTag(ggml_tensor* /*dst*/, int /*ith*/, int /*nth*/, void* /*userdata*/) {}
constinit std::array kTagMxfp8MulMatVec = std::to_array("jitllm.mxfp8.mul_mat_vec");
constinit std::array kTagMxfp8Dequant = std::to_array("jitllm.mxfp8.dequant");
constinit std::array kTagNvfp4Rows = std::to_array("jitllm.nvfp4.get_rows");
constinit std::array kTagArgmax = std::to_array("jitllm.argmax");
constinit std::array kTagHcCombine = std::to_array("jitllm.hc.combine");
constinit std::array kTagHcNorm = std::to_array("jitllm.hc.norm");
constinit std::array kTagHcMix = std::to_array("jitllm.hc.mix");
constinit std::array kTagMoeGlu = std::to_array("jitllm.moe.glu");
constinit std::array kTagMoeCombine = std::to_array("jitllm.moe.combine");
constinit std::array kTagBf16 = std::to_array("jitllm.bf16");
constinit std::array kTagGemmBf16 = std::to_array("jitllm.gemm.bf16");
constinit std::array kTagMoeRoute = std::to_array("jitllm.moe.route");
constinit std::array kTagMoeQuantize = std::to_array("jitllm.moe.quantize");
constinit std::array kTagMoeGemm = std::to_array("jitllm.moe.gemm");
constinit std::array kTagMoeGluQuantize = std::to_array("jitllm.moe.glu_quantize");
constinit std::array kTagMoeCombineSorted = std::to_array("jitllm.moe.combine_sorted");
constinit std::array kTagMoeGemv = std::to_array("jitllm.moe.gemv");
constinit std::array kTagGdnConv = std::to_array("jitllm.gdn.conv");
constinit std::array kTagGdnNormGate = std::to_array("jitllm.gdn.norm_gate");

// Where a norm's epsilon sits in op_params: after GGML's custom parameters.
constexpr std::size_t kEpsOffset = 32;
static_assert(sizeof(JitllmCustomParams) <= kEpsOffset);
static_assert(kEpsOffset + (8 * sizeof(std::int32_t)) <= GGML_MAX_OP_PARAMS);

bool IsBytes(const ggml_tensor* t) { return t != nullptr && t->type == GGML_TYPE_I8; }

bool Matrix2d(const ggml_tensor* t) { return t->ne[2] == 1 && t->ne[3] == 1; }

// MXFP8 weights: codes I8 [k, n] and scales I8 [k / 32, n], both packed,
// codes 16-byte aligned for the kernels' vector loads.
std::expected<void, KernelFailure> CheckMxfp8Weights(const ggml_tensor* codes,
                                                     const ggml_tensor* scales) {
  if (!IsBytes(codes) || !IsBytes(scales) || !Bound(codes) || !Bound(scales)) {
    return Rejected("MXFP8 weights are bound I8 codes and scales");
  }
  if (AnyEmpty({codes, scales}) || !AllSane({codes, scales}) || !Matrix2d(codes) ||
      !Matrix2d(scales)) {
    return Rejected("MXFP8 weights are non-empty matrices");
  }
  const std::int64_t k = codes->ne[0];
  const std::int64_t n = codes->ne[1];
  if (k % 32 != 0 || scales->ne[0] != k / 32 || scales->ne[1] != n) {
    return Rejected("MXFP8 weights: k a multiple of 32, one scale per 32 codes a row");
  }
  if (!Packed(codes) || !Packed(scales) || !Aligned(codes, 16)) {
    return Rejected("MXFP8 weights packed, the codes 16-byte aligned");
  }
  if (std::cmp_greater(n, kInt32Max) || std::cmp_greater(k, kInt32Max)) {
    return Rejected("MXFP8 weights beyond the kernels' 32-bit extents");
  }
  return {};
}

std::expected<void, KernelFailure> CheckCustom(const ggml_tensor* node, JitllmOp op,
                                               int arguments) {
  if (node == nullptr || JitllmOpOf(node) != op || !Bound(node)) {
    return Rejected("not a bound node of this operation");
  }
  for (int i = 0; i < arguments; ++i) {
    if (!Bound(node->src[i])) {
      return Rejected("an unbound operand");
    }
  }
  if (arguments < GGML_MAX_SRC && node->src[arguments] != nullptr) {
    return Rejected("more operands than the operation takes");
  }
  return {};
}

ggml_tensor* Custom(ggml_context* context, ggml_type type, std::array<std::int64_t, 4> ne,
                    std::initializer_list<ggml_tensor*> args, char* name) {
  std::array<ggml_tensor*, GGML_MAX_SRC> list{};
  std::size_t n = 0;
  for (ggml_tensor* a : args) {
    list[n++] = a;
  }
  return ggml_custom_4d(context, type, ne[0], ne[1], ne[2], ne[3], list.data(), static_cast<int>(n),
                        &JitllmCustomTag, 1, name);
}

}  // namespace

JitllmOp JitllmOpOf(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_CUSTOM) {
    return JitllmOp::kNone;
  }
  JitllmCustomParams params{};
  static_assert(sizeof(params) <= sizeof(node->op_params));
  std::memcpy(&params, node->op_params, sizeof(params));
  if (params.fun != &JitllmCustomTag) {
    return JitllmOp::kNone;
  }
  if (params.userdata == kTagMxfp8MulMatVec.data()) {
    return JitllmOp::kMxfp8MulMatVec;
  }
  if (params.userdata == kTagMxfp8Dequant.data()) {
    return JitllmOp::kMxfp8Dequant;
  }
  if (params.userdata == kTagNvfp4Rows.data()) {
    return JitllmOp::kNvfp4Rows;
  }
  if (params.userdata == kTagArgmax.data()) {
    return JitllmOp::kArgmax;
  }
  const std::array<std::pair<const char*, JitllmOp>, 15> fused = {{
      {kTagGdnConv.data(), JitllmOp::kGdnConv},
      {kTagGdnNormGate.data(), JitllmOp::kGdnNormGate},
      {kTagHcCombine.data(), JitllmOp::kHcCombine},
      {kTagHcNorm.data(), JitllmOp::kHcNorm},
      {kTagHcMix.data(), JitllmOp::kHcMix},
      {kTagMoeGlu.data(), JitllmOp::kMoeGlu},
      {kTagMoeCombine.data(), JitllmOp::kMoeCombine},
      {kTagBf16.data(), JitllmOp::kBf16},
      {kTagGemmBf16.data(), JitllmOp::kGemmBf16},
      {kTagMoeRoute.data(), JitllmOp::kMoeRoute},
      {kTagMoeQuantize.data(), JitllmOp::kMoeQuantize},
      {kTagMoeGemm.data(), JitllmOp::kMoeGemm},
      {kTagMoeGluQuantize.data(), JitllmOp::kMoeGluQuantize},
      {kTagMoeCombineSorted.data(), JitllmOp::kMoeCombineSorted},
      {kTagMoeGemv.data(), JitllmOp::kMoeGemv},
  }};
  for (const auto& [tag, op] : fused) {
    if (params.userdata == tag) {
      return op;
    }
  }
  return JitllmOp::kNone;
}

float JitllmOpEps(const ggml_tensor* node) {
  float eps = 0.0f;
  std::memcpy(&eps, reinterpret_cast<const char*>(node->op_params) + kEpsOffset, sizeof(eps));
  return eps;
}

float JitllmOpFloat(const ggml_tensor* node, int index) {
  float value = 0.0f;
  if (index < 0 || index >= 8) {
    return 0.0f;
  }
  std::memcpy(&value,
              reinterpret_cast<const char*>(node->op_params) + kEpsOffset +
                  (static_cast<std::size_t>(index) * sizeof(value)),
              sizeof(value));
  return value;
}

std::int32_t JitllmOpInt(const ggml_tensor* node, int index) {
  std::int32_t value = 0;
  if (index < 0 || index >= 8) {
    return 0;
  }
  std::memcpy(&value,
              reinterpret_cast<const char*>(node->op_params) + kEpsOffset +
                  (static_cast<std::size_t>(index) * sizeof(value)),
              sizeof(value));
  return value;
}

ggml_tensor* Argmax(ggml_context* context, ggml_tensor* x) {
  return Custom(context, GGML_TYPE_I32, {x->ne[1], 1, 1, 1}, {x}, kTagArgmax.data());
}

std::expected<void, KernelFailure> CheckArgmax(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kArgmax, 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(x) || node->type != GGML_TYPE_I32 || AnyEmpty({x, node}) || !AllSane({x, node}) ||
      !Matrix2d(x) || node->ne[0] != x->ne[1] || ggml_nrows(node) != 1) {
    return Rejected("F32 rows into one I32 index a row");
  }
  if (!Packed(x) || !Packed(node) || !Aligned(x, 4) || !Aligned(node, 4) ||
      std::cmp_greater(x->ne[0], kInt32Max) || x->ne[1] > 65535) {
    return Rejected("packed operands within the kernel's grid");
  }
  if (!AllCurrent({node, x}) || !Disjoint(node, x, false)) {
    return Rejected("a stale view, or an output overlapping its operand");
  }
  return {};
}

ggml_tensor* Mxfp8MulMatVec(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales,
                            ggml_tensor* x) {
  return Custom(context, GGML_TYPE_F32, {codes->ne[1], x->ne[1], 1, 1}, {codes, scales, x},
                kTagMxfp8MulMatVec.data());
}

ggml_tensor* Mxfp8Dequant(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales) {
  return Custom(context, GGML_TYPE_BF16, {codes->ne[0], codes->ne[1], 1, 1}, {codes, scales},
                kTagMxfp8Dequant.data());
}

ggml_tensor* Nvfp4Rows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids,
                       ggml_tensor* scale, std::int64_t values) {
  return Custom(context, GGML_TYPE_F32, {values, ids->ne[0], 1, 1}, {table, ids, scale},
                kTagNvfp4Rows.data());
}

ggml_tensor* HcCombine(ggml_context* context, ggml_tensor* res, ggml_tensor* out,
                       ggml_tensor* inject) {
  return Custom(context, GGML_TYPE_F32, {res->ne[0], res->ne[1], res->ne[2], 1}, {res, out, inject},
                kTagHcCombine.data());
}

namespace {

ggml_tensor* WithEps(ggml_tensor* node, float eps) {
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset, &eps, sizeof(eps));
  return node;
}

// Integer parameters 0.. (at most 8), each within int32 (the builders'
// callers pass the model's extents; a value that does not fit is stored as
// -1, which every check refuses).
ggml_tensor* WithInts(ggml_tensor* node, std::initializer_list<std::int64_t> values) {
  std::size_t at = kEpsOffset;
  for (const std::int64_t v : values) {
    const std::int32_t stored =
        v >= 0 && std::cmp_less_equal(v, kInt32Max) ? static_cast<std::int32_t>(v) : -1;
    std::memcpy(reinterpret_cast<char*>(node->op_params) + at, &stored, sizeof(stored));
    at += sizeof(stored);
  }
  return node;
}

}  // namespace

ggml_tensor* HcNorm(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, float eps,
                    ggml_type type) {
  return WithEps(
      Custom(context, type, {x->ne[0] * x->ne[1], x->ne[2], 1, 1}, {x, weight}, kTagHcNorm.data()),
      eps);
}

ggml_tensor* HcMix(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* gate,
                   float eps) {
  return WithEps(Custom(context, GGML_TYPE_F32, {x->ne[0], x->ne[2], 1, 1}, {x, weight, gate},
                        kTagHcMix.data()),
                 eps);
}

ggml_tensor* MoeGlu(ggml_context* context, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* ids,
                    ggml_tensor* gate_scale, ggml_tensor* up_scale) {
  return Custom(context, GGML_TYPE_F32, {gate->ne[0], gate->ne[1], gate->ne[2], 1},
                {gate, up, ids, gate_scale, up_scale}, kTagMoeGlu.data());
}

ggml_tensor* MoeCombine(ggml_context* context, ggml_tensor* down, ggml_tensor* ids,
                        ggml_tensor* down_scale, ggml_tensor* weights, ggml_tensor* shared,
                        ggml_tensor* shared_gate) {
  return Custom(context, GGML_TYPE_F32, {down->ne[0], down->ne[2], 1, 1},
                {down, ids, down_scale, weights, shared, shared_gate}, kTagMoeCombine.data());
}

ggml_tensor* ToBf16(ggml_context* context, ggml_tensor* x) {
  return Custom(context, GGML_TYPE_BF16, {x->ne[0], x->ne[1], x->ne[2], x->ne[3]}, {x},
                kTagBf16.data());
}

ggml_tensor* GemmBf16(ggml_context* context, ggml_tensor* weights, ggml_tensor* x) {
  return Custom(context, GGML_TYPE_F32, {weights->ne[1], x->ne[1], 1, 1}, {weights, x},
                kTagGemmBf16.data());
}

ggml_tensor* GdnConv(ggml_context* context, ggml_tensor* x, ggml_tensor* history,
                     ggml_tensor* weight, std::int64_t qk_channels, std::int64_t head, float eps,
                     float scale) {
  ggml_tensor* node = WithInts(Custom(context, GGML_TYPE_F32, {x->ne[0], x->ne[1], 1, 1},
                                      {x, history, weight}, kTagGdnConv.data()),
                               {qk_channels, head});
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 8, &eps, sizeof(eps));
  std::memcpy(reinterpret_cast<char*>(node->op_params) + kEpsOffset + 12, &scale, sizeof(scale));
  return node;
}

ggml_tensor* GdnNormGate(ggml_context* context, ggml_tensor* o, ggml_tensor* weight, ggml_tensor* z,
                         float eps, ggml_type type) {
  return WithEps(Custom(context, type, {o->ne[0] * o->ne[1], o->ne[2], 1, 1}, {o, weight, z},
                        kTagGdnNormGate.data()),
                 eps);
}

namespace {

// A route node's extents: experts, experts used, tokens (its parameters).
struct RouteExtents {
  std::int64_t experts = 0;
  std::int64_t used = 0;
  std::int64_t tokens = 0;
  std::int64_t slots() const { return used * tokens; }
  moe::RouteLayout layout() const {
    return {.experts = experts,
            .slots = slots(),
            .chunks = (tokens + moe::kRouteChunk - 1) / moe::kRouteChunk};
  }
  moe::QuantLayout quant(std::int64_t k) const {
    return {.k = static_cast<std::uint64_t>(k),
            .slots = static_cast<std::uint64_t>(slots()),
            .experts = static_cast<std::uint64_t>(experts)};
  }
};

RouteExtents ExtentsOf(const ggml_tensor* route) {
  return {.experts = JitllmOpInt(route, 0),
          .used = JitllmOpInt(route, 1),
          .tokens = JitllmOpInt(route, 2)};
}

std::int64_t QuantBytes(const RouteExtents& r, std::int64_t k) {
  return static_cast<std::int64_t>(r.quant(k).bytes());
}

}  // namespace

ggml_tensor* MoeRoute(ggml_context* context, ggml_tensor* ids, std::int64_t experts) {
  const RouteExtents r{.experts = experts, .used = ids->ne[0], .tokens = ids->ne[1]};
  return WithInts(
      Custom(context, GGML_TYPE_I32, {r.layout().ints(), 1, 1, 1}, {ids}, kTagMoeRoute.data()),
      {r.experts, r.used, r.tokens});
}

ggml_tensor* MoeQuantize(ggml_context* context, ggml_tensor* x, ggml_tensor* route) {
  const RouteExtents r = ExtentsOf(route);
  const std::int64_t k = x->ne[0];
  return WithInts(Custom(context, GGML_TYPE_I8, {QuantBytes(r, k), 1, 1, 1}, {x, route},
                         kTagMoeQuantize.data()),
                  {r.experts, r.used, r.tokens, k});
}

ggml_tensor* MoeGemm(ggml_context* context, ggml_tensor* a, ggml_tensor* route,
                     ggml_tensor* weights, std::int64_t n, std::uint64_t codes_offset,
                     std::uint64_t scales_offset) {
  const RouteExtents r = ExtentsOf(route);
  const std::int64_t k = JitllmOpInt(a, 3);
  return WithInts(Custom(context, GGML_TYPE_BF16, {n, r.slots(), 1, 1}, {a, route, weights},
                         kTagMoeGemm.data()),
                  {r.experts, r.used, r.tokens, k, n, static_cast<std::int64_t>(codes_offset),
                   static_cast<std::int64_t>(scales_offset)});
}

ggml_tensor* MoeGluQuantize(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                            ggml_tensor* route, ggml_tensor* gate_scale, ggml_tensor* up_scale) {
  const RouteExtents r = ExtentsOf(route);
  const std::int64_t f = d->ne[0] / 2;
  return WithInts(Custom(context, GGML_TYPE_I8, {QuantBytes(r, f), 1, 1, 1},
                         {d, a, route, gate_scale, up_scale}, kTagMoeGluQuantize.data()),
                  {r.experts, r.used, r.tokens, f});
}

ggml_tensor* MoeCombineSorted(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                              ggml_tensor* route, ggml_tensor* down_scale, ggml_tensor* weights,
                              ggml_tensor* shared, ggml_tensor* shared_gate) {
  const RouteExtents r = ExtentsOf(route);
  return WithInts(
      Custom(context, GGML_TYPE_F32, {d->ne[0], r.tokens, 1, 1},
             {d, a, route, down_scale, weights, shared, shared_gate}, kTagMoeCombineSorted.data()),
      {r.experts, r.used, r.tokens});
}

ggml_tensor* MoeGemv(ggml_context* context, ggml_tensor* weights, ggml_tensor* x, ggml_tensor* ids,
                     std::int64_t n, std::int64_t row0, std::int64_t rows,
                     std::uint64_t codes_offset, std::uint64_t scales_offset) {
  return WithInts(Custom(context, GGML_TYPE_F32, {n, ids->ne[0], ids->ne[1], 1}, {weights, x, ids},
                         kTagMoeGemv.data()),
                  {row0, rows, static_cast<std::int64_t>(codes_offset),
                   static_cast<std::int64_t>(scales_offset)});
}

ggml_tensor* MoeGemvSwiglu(ggml_context* context, ggml_tensor* weights, ggml_tensor* x,
                           ggml_tensor* ids, std::int64_t f, ggml_tensor* gate_scale,
                           ggml_tensor* up_scale, std::uint64_t codes_offset,
                           std::uint64_t scales_offset) {
  return WithInts(Custom(context, GGML_TYPE_F32, {f, ids->ne[0], ids->ne[1], 1},
                         {weights, x, ids, gate_scale, up_scale}, kTagMoeGemv.data()),
                  {0, 2 * f, static_cast<std::int64_t>(codes_offset),
                   static_cast<std::int64_t>(scales_offset)});
}

bool IsMoeGemvSwiglu(const ggml_tensor* node) { return node != nullptr && node->src[3] != nullptr; }

std::expected<void, KernelFailure> CheckMxfp8MulMatVec(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8MulMatVec, 3); !checked) {
    return checked;
  }
  const ggml_tensor* codes = node->src[0];
  const ggml_tensor* scales = node->src[1];
  const ggml_tensor* x = node->src[2];
  if (auto checked = CheckMxfp8Weights(codes, scales); !checked) {
    return checked;
  }
  if (!IsF32(x) || !IsF32(node) || AnyEmpty({x, node}) || !AllSane({x, node}) || !Matrix2d(x) ||
      !Matrix2d(node)) {
    return Rejected("F32 activations and output, matrices");
  }
  const std::int64_t t = x->ne[1];
  if (x->ne[0] != codes->ne[0] || node->ne[0] != codes->ne[1] || node->ne[1] != t ||
      t > kMxfp8VecColumns) {
    return Rejected("y[n, t] = W[n, k] x[k, t] for at most 8 columns");
  }
  // The kernel reads x in 16-byte vectors along each column.
  // The kernel's grid counts rows in int, 8 a block.
  if (x->nb[0] != sizeof(float) || x->nb[1] % 16 != 0 || !Aligned(x, 16) || !Packed(node) ||
      !Aligned(node, sizeof(float)) || std::cmp_greater(x->nb[1] / sizeof(float), kInt32Max) ||
      std::cmp_greater(codes->ne[1], kInt32Max - 8)) {
    return Rejected("x columns 16-byte aligned at 16-byte strides, and a packed, aligned output");
  }
  if (!AllCurrent({node, codes, scales, x}) || !Disjoint(node, codes, false) ||
      !Disjoint(node, scales, false) || !Disjoint(node, x, false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMxfp8Dequant(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMxfp8Dequant, 2); !checked) {
    return checked;
  }
  const ggml_tensor* codes = node->src[0];
  const ggml_tensor* scales = node->src[1];
  if (auto checked = CheckMxfp8Weights(codes, scales); !checked) {
    return checked;
  }
  if (node->type != GGML_TYPE_BF16 || !ggml_are_same_shape(node, codes) || !Packed(node) ||
      !Aligned(node, 16)) {
    return Rejected("a packed, 16-byte aligned BF16 matrix of the codes' shape");
  }
  if (std::cmp_greater(ggml_nelements(codes) / 16, kInt32Max) ||
      !AllCurrent({node, codes, scales}) || !Disjoint(node, codes, false) ||
      !Disjoint(node, scales, false)) {
    return Rejected("beyond the kernel's grid, a stale view, or an overlapping output");
  }
  return {};
}

std::expected<void, KernelFailure> CheckNvfp4Rows(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kNvfp4Rows, 3); !checked) {
    return checked;
  }
  const ggml_tensor* table = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const ggml_tensor* scale = node->src[2];
  const std::int64_t values = node->ne[0];
  if (!IsBytes(table) || ids->type != GGML_TYPE_I32 || !IsF32(scale) || !IsF32(node)) {
    return Rejected("an I8 table, I32 ids and an F32 scale into F32");
  }
  if (AnyEmpty({table, ids, scale, node}) || !AllSane({table, ids, scale, node}) ||
      !Matrix2d(table) || !Matrix2d(node) || ggml_nelements(scale) != 1 || ggml_nrows(ids) != 1) {
    return Rejected("a table of rows, one row of ids, one scale");
  }
  if (values % 16 != 0 || values > 1024 || table->ne[0] != (values / 2) + (values / 16) ||
      node->ne[1] != ids->ne[0]) {
    return Rejected("rows of a multiple of 16 values (at most 1,024): codes then scales");
  }
  if (!Packed(table) || !Packed(ids) || !Packed(node) || !Aligned(ids, 4) || !Aligned(scale, 4) ||
      !Aligned(node, 4) || std::cmp_greater(table->ne[1], kInt32Max) ||
      std::cmp_greater(ids->ne[0], 65535 * 1024LL)) {
    return Rejected("packed operands within the kernel's grid");
  }
  if (!AllCurrent({node, table, ids, scale}) || !Disjoint(node, table, false) ||
      !Disjoint(node, ids, false) || !Disjoint(node, scale, false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

namespace {

bool Shaped(const ggml_tensor* t, std::int64_t n0, std::int64_t n1, std::int64_t n2) {
  return t->ne[0] == n0 && t->ne[1] == n1 && t->ne[2] == n2 && t->ne[3] == 1;
}

// Bound, sane, non-empty, packed operands and an output disjoint from each,
// every view current.
std::expected<void, KernelFailure> CheckDense(const ggml_tensor* node,
                                              std::initializer_list<const ggml_tensor*> packed,
                                              const ggml_tensor* strided = nullptr) {
  if (!Packed(node) || AnyEmpty({node}) || !AllSane({node}) || !AllCurrent({node})) {
    return Rejected("a packed, non-empty output");
  }
  for (const ggml_tensor* t : packed) {
    if (!Packed(t) || AnyEmpty({t}) || !AllSane({t}) || !AllCurrent({t}) ||
        !Disjoint(node, t, false)) {
      return Rejected("packed, non-empty operands, current views, disjoint from the output");
    }
  }
  if (strided != nullptr && (AnyEmpty({strided}) || !AllSane({strided}) || !AllCurrent({strided}) ||
                             !Disjoint(node, strided, false))) {
    return Rejected("the expert ids are measurable, current and disjoint from the output");
  }
  return {};
}

// Expert ids: I32 [used, t] with packed elements and a row stride of whole
// ids at least a row long.
bool ExpertIds(const ggml_tensor* ids, std::int64_t used, std::int64_t t) {
  return ids->type == GGML_TYPE_I32 && Shaped(ids, used, t, 1) &&
         ids->nb[0] == sizeof(std::int32_t) && ids->nb[1] % sizeof(std::int32_t) == 0 &&
         ids->nb[1] >= static_cast<std::size_t>(used) * sizeof(std::int32_t) &&
         ids->nb[1] / sizeof(std::int32_t) <= kInt32Max && Aligned(ids, sizeof(std::int32_t));
}

bool Vector(const ggml_tensor* t, std::int64_t n) { return IsF32(t) && Shaped(t, n, 1, 1); }

}  // namespace

std::expected<void, KernelFailure> CheckHcCombine(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcCombine, 3); !checked) {
    return checked;
  }
  const ggml_tensor* res = node->src[0];
  const ggml_tensor* out = node->src[1];
  const ggml_tensor* inject = node->src[2];
  const std::int64_t width = res->ne[0];
  const std::int64_t hc = res->ne[1];
  const std::int64_t t = res->ne[2];
  if (!IsF32(node) || !IsF32(res) || !IsF32(out) || !IsF32(inject) || res->ne[3] != 1 ||
      !Shaped(node, width, hc, t) || !Shaped(out, width, t, 1) || !Shaped(inject, hc, t, 1)) {
    return Rejected("F32 streams [width, hc, t], an output [width, t] and weights [hc, t]");
  }
  if (width % 4 != 0 || !Aligned(node, 16) || !Aligned(res, 16) || !Aligned(out, 16) ||
      !Aligned(inject, sizeof(float)) || std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected("rows of whole float4s, 16-byte aligned, within the kernel's grid");
  }
  return CheckDense(node, {res, out, inject});
}

namespace {

// The norms' streams: x F32 [width, hc, t] and its weight F32 [width · hc],
// width at least GGML's 1,024-thread rms_norm's.
std::expected<void, KernelFailure> CheckNormOperands(const ggml_tensor* node, const ggml_tensor* x,
                                                     const ggml_tensor* weight) {
  const std::int64_t width = x->ne[0];
  const std::int64_t hc = x->ne[1];
  const float eps = JitllmOpEps(node);
  if (!IsF32(x) || x->ne[3] != 1 || !Vector(weight, width * hc)) {
    return Rejected("F32 streams [width, hc, t] and an F32 weight [width · hc]");
  }
  if (width < 1024 || std::cmp_greater(width, kInt32Max) || hc > 8 ||
      std::cmp_greater(x->ne[2], kInt32Max) || !std::isfinite(eps) || eps < 0.0f) {
    return Rejected("streams of 1,024 to 2^31 elements, at most 8 of them, a finite epsilon");
  }
  if (!Aligned(x, sizeof(float)) || !Aligned(weight, sizeof(float))) {
    return Rejected("aligned F32 operands");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckHcNorm(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcNorm, 2); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* weight = node->src[1];
  if (auto checked = CheckNormOperands(node, x, weight); !checked) {
    return checked;
  }
  if ((node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) ||
      !Shaped(node, x->ne[0] * x->ne[1], x->ne[2], 1) ||
      !Aligned(node, ggml_type_size(node->type))) {
    return Rejected("an aligned F32 or BF16 output [width · hc, t]");
  }
  return CheckDense(node, {x, weight});
}

std::expected<void, KernelFailure> CheckHcMix(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kHcMix, 3); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* weight = node->src[1];
  const ggml_tensor* gate = node->src[2];
  if (auto checked = CheckNormOperands(node, x, weight); !checked) {
    return checked;
  }
  if (!IsF32(gate) || !Shaped(gate, x->ne[0] * x->ne[1], x->ne[2], 1) || !IsF32(node) ||
      !Shaped(node, x->ne[0], x->ne[2], 1) || !Aligned(gate, sizeof(float)) ||
      !Aligned(node, sizeof(float))) {
    return Rejected("an F32 gate [width · hc, t] and an aligned F32 output [width, t]");
  }
  return CheckDense(node, {x, weight, gate});
}

std::expected<void, KernelFailure> CheckMoeGlu(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGlu, 5); !checked) {
    return checked;
  }
  const ggml_tensor* gate = node->src[0];
  const ggml_tensor* up = node->src[1];
  const ggml_tensor* ids = node->src[2];
  const ggml_tensor* gate_scale = node->src[3];
  const ggml_tensor* up_scale = node->src[4];
  const std::int64_t n = gate->ne[0];
  const std::int64_t used = gate->ne[1];
  const std::int64_t t = gate->ne[2];
  const std::int64_t experts = gate_scale->ne[0];
  if (!IsF32(node) || !IsF32(gate) || !IsF32(up) || gate->ne[3] != 1 || !Shaped(up, n, used, t) ||
      !Shaped(node, n, used, t) || !ExpertIds(ids, used, t) || !Vector(gate_scale, experts) ||
      !Vector(up_scale, experts) || std::cmp_greater(experts, kInt32Max)) {
    return Rejected("F32 products [n, used, t], I32 ids [used, t] and F32 scales [experts]");
  }
  if (n % 4 != 0 || !Aligned(node, 16) || !Aligned(gate, 16) || !Aligned(up, 16) ||
      !Aligned(gate_scale, sizeof(float)) || !Aligned(up_scale, sizeof(float)) ||
      std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected("rows of whole float4s, 16-byte aligned, within the kernel's grid");
  }
  return CheckDense(node, {gate, up, gate_scale, up_scale}, ids);
}

std::expected<void, KernelFailure> CheckMoeCombine(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeCombine, 6); !checked) {
    return checked;
  }
  const ggml_tensor* down = node->src[0];
  const ggml_tensor* ids = node->src[1];
  const ggml_tensor* down_scale = node->src[2];
  const ggml_tensor* weights = node->src[3];
  const ggml_tensor* shared = node->src[4];
  const ggml_tensor* shared_gate = node->src[5];
  const std::int64_t width = down->ne[0];
  const std::int64_t used = down->ne[1];
  const std::int64_t t = down->ne[2];
  const std::int64_t experts = down_scale->ne[0];
  if (!IsF32(node) || !IsF32(down) || down->ne[3] != 1 || !ExpertIds(ids, used, t) ||
      !Vector(down_scale, experts) || std::cmp_greater(experts, kInt32Max) || !IsF32(weights) ||
      !Shaped(weights, 1, used, t) || !IsF32(shared) || !Shaped(shared, width, t, 1) ||
      !IsF32(shared_gate) || !Shaped(shared_gate, 1, t, 1) || !Shaped(node, width, t, 1)) {
    return Rejected(
        "F32 products [width, used, t], ids [used, t], scales [experts], weights [1, used, t], "
        "a shared product [width, t] and its gate [1, t]");
  }
  if (width % 4 != 0 || !Aligned(node, 16) || !Aligned(down, 16) || !Aligned(shared, 16) ||
      !Aligned(down_scale, sizeof(float)) || !Aligned(weights, sizeof(float)) ||
      !Aligned(shared_gate, sizeof(float)) ||
      std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected("rows of whole float4s, 16-byte aligned, within the kernel's grid");
  }
  return CheckDense(node, {down, down_scale, weights, shared, shared_gate}, ids);
}

std::expected<void, KernelFailure> CheckBf16(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kBf16, 1); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  if (!IsF32(x) || node->type != GGML_TYPE_BF16 || !ggml_are_same_shape(node, x) ||
      !Aligned(x, sizeof(float)) || !Aligned(node, 2) ||
      std::cmp_greater(ggml_nelements(node), kInt32Max)) {
    return Rejected("F32 into an aligned BF16 tensor of its shape, within the kernel's grid");
  }
  return CheckDense(node, {x});
}

namespace {

// A route node's extents, each positive and within the kernels' grids.
bool SaneExtents(const RouteExtents& r) {
  return r.experts > 0 && r.experts <= 4096 && r.used > 0 && r.used <= 64 && r.tokens > 0 &&
         std::cmp_less_equal(r.layout().ints(), kInt32Max) &&
         std::cmp_less_equal(r.slots(), kInt32Max / 4);
}

bool IsRoute(const ggml_tensor* route, const RouteExtents& want) {
  if (JitllmOpOf(route) != JitllmOp::kMoeRoute || !Bound(route) || !Packed(route) ||
      !Aligned(route, sizeof(std::int32_t)) || !AllCurrent({route})) {
    return false;
  }
  const RouteExtents r = ExtentsOf(route);
  return r.experts == want.experts && r.used == want.used && r.tokens == want.tokens &&
         route->type == GGML_TYPE_I32 && Shaped(route, r.layout().ints(), 1, 1);
}

// A quantization of k-wide rows over the route's slots, laid out by that
// route node itself (its rows and scale blocks are where that route put
// them; another route of the same extents may sort differently).
bool IsQuantized(const ggml_tensor* a, const ggml_tensor* route, const RouteExtents& r,
                 std::int64_t k) {
  const JitllmOp op = JitllmOpOf(a);
  const ggml_tensor* laid_out_by = op == JitllmOp::kMoeQuantize      ? a->src[1]
                                   : op == JitllmOp::kMoeGluQuantize ? a->src[2]
                                                                     : nullptr;
  return laid_out_by != nullptr && laid_out_by == route && Bound(a) && a->type == GGML_TYPE_I8 &&
         Packed(a) && Aligned(a, 16) && AllCurrent({a}) && JitllmOpInt(a, 0) == r.experts &&
         JitllmOpInt(a, 1) == r.used && JitllmOpInt(a, 2) == r.tokens && JitllmOpInt(a, 3) == k &&
         Shaped(a, QuantBytes(r, k), 1, 1);
}

// Expert weights in the CUTLASS layout: I8 [stride, experts], the stride
// whole 16 bytes; `rows` rows of k at the offsets fit each slot.
bool ExpertSlots(const ggml_tensor* w, std::int64_t experts, std::int64_t rows, std::int64_t k,
                 std::int64_t codes, std::int64_t scales) {
  if (w->type != GGML_TYPE_I8 || !Bound(w) || !Shaped(w, w->ne[0], experts, 1) || !Packed(w) ||
      !Aligned(w, 16) || w->ne[0] % 16 != 0 || AnyEmpty({w}) || !AllSane({w}) || !AllCurrent({w})) {
    return false;
  }
  const std::int64_t stride = w->ne[0];
  return rows > 0 && rows % moe::kScaleRows == 0 && k > 0 && k % 64 == 0 && codes >= 0 &&
         scales >= 0 && codes % 16 == 0 && scales % 16 == 0 && codes + (rows * (k / 2)) <= stride &&
         scales + (rows * (k / 16)) <= stride;
}

}  // namespace

std::expected<void, KernelFailure> CheckMoeRoute(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeRoute, 1); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* ids = node->src[0];
  if (!SaneExtents(r) || !ExpertIds(ids, r.used, r.tokens) || node->type != GGML_TYPE_I32 ||
      !Shaped(node, r.layout().ints(), 1, 1) || !Aligned(node, sizeof(std::int32_t))) {
    return Rejected("I32 expert ids [used, t] into the routing's layout of their extents");
  }
  return CheckDense(node, {}, ids);
}

std::expected<void, KernelFailure> CheckMoeQuantize(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeQuantize, 2); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* x = node->src[0];
  const std::int64_t k = JitllmOpInt(node, 3);
  if (!SaneExtents(r) || !IsRoute(node->src[1], r) || !IsF32(x) || !Shaped(x, k, r.tokens, 1) ||
      k % 64 != 0 || k <= 0 || k > 16384 || !Aligned(x, 16) || node->type != GGML_TYPE_I8 ||
      !Shaped(node, QuantBytes(r, k), 1, 1) || !Aligned(node, 16)) {
    return Rejected("F32 activations [k, t], k a multiple of 64, into the route's quantized rows");
  }
  return CheckDense(node, {x, node->src[1]});
}

std::expected<void, KernelFailure> CheckMoeGemm(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGemm, 3); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const std::int64_t k = JitllmOpInt(node, 3);
  const std::int64_t n = JitllmOpInt(node, 4);
  const std::int64_t codes = JitllmOpInt(node, 5);
  const std::int64_t scales = JitllmOpInt(node, 6);
  if (!SaneExtents(r) || !IsRoute(node->src[1], r) ||
      !IsQuantized(node->src[0], node->src[1], r, k) ||
      !ExpertSlots(node->src[2], r.experts, n, k, codes, scales) || node->type != GGML_TYPE_BF16 ||
      !Shaped(node, n, r.slots(), 1) || !Aligned(node, 16)) {
    return Rejected(
        "quantized rows of the route's slots, expert weights in the CUTLASS layout, and a BF16 "
        "output [n, slots]");
  }
  return CheckDense(node, {node->src[0], node->src[1], node->src[2]});
}

std::expected<void, KernelFailure> CheckMoeGluQuantize(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGluQuantize, 5); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* d = node->src[0];
  const ggml_tensor* a = node->src[1];
  const std::int64_t f = JitllmOpInt(node, 3);
  if (!SaneExtents(r) || !IsRoute(node->src[2], r) || f <= 0 || f % 64 != 0 || f > 16384 ||
      d->type != GGML_TYPE_BF16 || !Shaped(d, 2 * f, r.slots(), 1) || !Aligned(d, 16) ||
      JitllmOpOf(a) != JitllmOp::kMoeQuantize ||
      !IsQuantized(a, node->src[2], r, JitllmOpInt(a, 3)) || !Vector(node->src[3], r.experts) ||
      !Vector(node->src[4], r.experts) || !Aligned(node->src[3], sizeof(float)) ||
      !Aligned(node->src[4], sizeof(float)) || node->type != GGML_TYPE_I8 ||
      !Shaped(node, QuantBytes(r, f), 1, 1) || !Aligned(node, 16)) {
    return Rejected(
        "BF16 gate and up products [2f, slots], their input's quantization, F32 scales "
        "[experts], into the route's quantized rows of f");
  }
  return CheckDense(node, {d, a, node->src[2], node->src[3], node->src[4]});
}

std::expected<void, KernelFailure> CheckMoeCombineSorted(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kMoeCombineSorted, 7); !checked) {
    return checked;
  }
  const RouteExtents r = ExtentsOf(node);
  const ggml_tensor* d = node->src[0];
  const ggml_tensor* a = node->src[1];
  const std::int64_t width = d->ne[0];
  if (!SaneExtents(r) || !IsRoute(node->src[2], r) || width <= 0 || width % 4 != 0 ||
      d->type != GGML_TYPE_BF16 || !Shaped(d, width, r.slots(), 1) || !Aligned(d, 16) ||
      JitllmOpOf(a) != JitllmOp::kMoeGluQuantize ||
      !IsQuantized(a, node->src[2], r, JitllmOpInt(a, 3)) || !Vector(node->src[3], r.experts) ||
      !Aligned(node->src[3], sizeof(float)) || !IsF32(node->src[4]) ||
      !Shaped(node->src[4], 1, r.used, r.tokens) || !Aligned(node->src[4], sizeof(float)) ||
      !IsF32(node->src[5]) || !Shaped(node->src[5], width, r.tokens, 1) ||
      !Aligned(node->src[5], 16) || !IsF32(node->src[6]) || !Shaped(node->src[6], 1, r.tokens, 1) ||
      !Aligned(node->src[6], sizeof(float)) || !IsF32(node) || !Shaped(node, width, r.tokens, 1) ||
      !Aligned(node, 16) || std::cmp_greater(ggml_nelements(node) / 4, kInt32Max)) {
    return Rejected(
        "BF16 down products [w, slots], their input's quantization, F32 scales [experts], "
        "weights [1, used, t], a shared product [w, t] and its gate [1, t]");
  }
  return CheckDense(node,
                    {d, a, node->src[2], node->src[3], node->src[4], node->src[5], node->src[6]});
}

std::expected<void, KernelFailure> CheckMoeGemv(const ggml_tensor* node) {
  const bool swiglu = IsMoeGemvSwiglu(node);
  if (auto checked = CheckCustom(node, JitllmOp::kMoeGemv, swiglu ? 5 : 3); !checked) {
    return checked;
  }
  const ggml_tensor* w = node->src[0];
  const ggml_tensor* x = node->src[1];
  const ggml_tensor* ids = node->src[2];
  const std::int64_t n = node->ne[0];
  const std::int64_t used = ids->ne[0];
  const std::int64_t t = ids->ne[1];
  const std::int64_t k = x->ne[0];
  const std::int64_t row0 = JitllmOpInt(node, 0);
  const std::int64_t rows = JitllmOpInt(node, 1);
  if (!ExpertSlots(w, w->ne[1], rows, k, JitllmOpInt(node, 2), JitllmOpInt(node, 3)) || row0 < 0 ||
      n <= 0 || row0 + n > rows || !ExpertIds(ids, used, t) || t > kMoeGemvTokens || used > 64 ||
      !IsF32(x) || (x->ne[1] != 1 && x->ne[1] != used) || x->ne[2] != t || x->ne[3] != 1 ||
      !Aligned(x, 16) || !IsF32(node) || !Shaped(node, n, used, t) ||
      !Aligned(node, sizeof(float)) || std::cmp_greater(w->ne[1], kInt32Max)) {
    return Rejected(
        "expert weights in the CUTLASS layout, F32 activations [k, 1 or used, t] for at most 8 "
        "tokens and their ids, into F32 [n, used, t]");
  }
  if (!swiglu) {
    return CheckDense(node, {w, x}, ids);
  }
  const ggml_tensor* gate_scale = node->src[3];
  const ggml_tensor* up_scale = node->src[4];
  if (row0 != 0 || rows != 2 * n || !Vector(gate_scale, w->ne[1]) || !Vector(up_scale, w->ne[1]) ||
      !Aligned(gate_scale, sizeof(float)) || !Aligned(up_scale, sizeof(float))) {
    return Rejected("the SwiGLU form: gate rows then as many up rows, F32 scales [experts]");
  }
  return CheckDense(node, {w, x, gate_scale, up_scale}, ids);
}

std::expected<void, KernelFailure> CheckGdnConv(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGdnConv, 3); !checked) {
    return checked;
  }
  const ggml_tensor* x = node->src[0];
  const ggml_tensor* history = node->src[1];
  const ggml_tensor* weight = node->src[2];
  const std::int64_t channels = x->ne[0];
  const std::int64_t t = x->ne[1];
  const std::int64_t qk = JitllmOpInt(node, 0);
  const std::int64_t head = JitllmOpInt(node, 1);
  const float eps = JitllmOpFloat(node, 2);
  const float scale = JitllmOpFloat(node, 3);
  if (!IsF32(x) || !Shaped(x, channels, t, 1) || !IsF32(weight) ||
      !Shaped(weight, 4, channels, 1) || !IsF32(history) ||
      ggml_nelements(history) != 3 * channels || !IsF32(node) || !Shaped(node, channels, t, 1)) {
    return Rejected("F32 rows [channels, t], a 4-tap weight [4, channels] and a history of 3");
  }
  if (head != 128 || qk < 0 || qk % head != 0 || qk > channels || channels % head != 0 ||
      t > 65535 || std::cmp_greater(ggml_nelements(node), kInt32Max) || !std::isfinite(eps) ||
      eps < 0.0f || !std::isfinite(scale)) {
    return Rejected("heads of 128 channels, normalized heads leading, within the kernel's grid");
  }
  for (const ggml_tensor* tensor : {x, history, weight, node}) {
    if (!Aligned(tensor, sizeof(float))) {
      return Rejected("aligned F32 operands");
    }
  }
  return CheckDense(node, {x, history, weight});
}

std::expected<void, KernelFailure> CheckGdnNormGate(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGdnNormGate, 3); !checked) {
    return checked;
  }
  const ggml_tensor* o = node->src[0];
  const ggml_tensor* weight = node->src[1];
  const ggml_tensor* z = node->src[2];
  const std::int64_t d = o->ne[0];
  const std::int64_t heads = o->ne[1];
  const std::int64_t t = o->ne[2];
  const float eps = JitllmOpEps(node);
  if (!IsF32(o) || d != 128 || o->ne[3] != 1 || !Vector(weight, d) || !IsF32(z) ||
      !Shaped(z, d * heads, t, 1) ||
      (node->type != GGML_TYPE_F32 && node->type != GGML_TYPE_BF16) ||
      !Shaped(node, d * heads, t, 1) || !std::isfinite(eps) || eps < 0.0f ||
      std::cmp_greater(heads * t, kInt32Max)) {
    return Rejected("F32 heads [128, heads, t], a weight [128] and a gate [128 · heads, t]");
  }
  for (const ggml_tensor* tensor : {o, weight, z}) {
    if (!Aligned(tensor, sizeof(float))) {
      return Rejected("aligned F32 operands");
    }
  }
  if (!Aligned(node, ggml_type_size(node->type))) {
    return Rejected("an aligned output");
  }
  return CheckDense(node, {o, weight, z});
}

bool GatedDeltaNetColumnsFits(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_GATED_DELTA_NET) {
    return false;
  }
  const ggml_tensor* q = node->src[0];
  const ggml_tensor* v = node->src[2];
  const ggml_tensor* g = node->src[3];
  return q != nullptr && v != nullptr && g != nullptr && v->ne[0] == 128 && v->ne[3] == 1 &&
         q->ne[3] == 1 && g->ne[0] == 1 && node->op_params[0] == 1;
}

std::expected<void, KernelFailure> CheckGatedDeltaNetColumns(const ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNet(node); !checked) {
    return checked;
  }
  if (!GatedDeltaNetColumnsFits(node) || std::cmp_greater(node->src[2]->ne[2], kInt32Max) ||
      std::cmp_greater(node->src[2]->ne[1], 65535)) {
    return Rejected(
        "jitllm.gated_delta_net.columns takes one sequence of 128-wide heads, a scalar gate "
        "and no snapshots");
  }
  return {};
}

bool GatedDeltaNetLanesFits(const ggml_tensor* node) {
  return GatedDeltaNetColumnsFits(node) && node->src[2]->ne[2] > kGatedDeltaNetLanesTokens;
}

std::expected<void, KernelFailure> CheckGatedDeltaNetLanes(const ggml_tensor* node) {
  if (auto checked = CheckGatedDeltaNetColumns(node); !checked) {
    return checked;
  }
  for (const ggml_tensor* t : std::initializer_list<const ggml_tensor*>{
           node->src[0], node->src[1], node->src[2], node->src[5], node}) {
    if (!AlignedEverywhere(t, 16)) {
      return Rejected(
          "jitllm.gated_delta_net.lanes loads q, k, v and the state and stores the new state as "
          "float4s");
    }
  }
  return {};
}

std::expected<void, KernelFailure> CheckGemmBf16(const ggml_tensor* node) {
  if (auto checked = CheckCustom(node, JitllmOp::kGemmBf16, 2); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* x = node->src[1];
  const std::int64_t k = weights->ne[0];
  const std::int64_t n = weights->ne[1];
  const std::int64_t t = x->ne[1];
  if (weights->type != GGML_TYPE_BF16 || x->type != GGML_TYPE_BF16 || !IsF32(node) ||
      !Shaped(weights, k, n, 1) || !Shaped(x, k, t, 1) || !Shaped(node, n, t, 1)) {
    return Rejected("BF16 weights [k, n] and activations [k, t] into F32 [n, t]");
  }
  if (std::cmp_greater(k, kInt32Max) || std::cmp_greater(n, kInt32Max) ||
      std::cmp_greater(t, kInt32Max) || !Aligned(weights, 16) || !Aligned(x, 16) ||
      !Aligned(node, 16)) {
    return Rejected("cuBLAS's int extents and 16-byte aligned operands");
  }
  return CheckDense(node, {weights, x});
}

}  // namespace jitllm::kernels::ggml
