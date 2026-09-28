// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/jitllm_ops.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <expected>
#include <utility>

#include "ggml.h"
#include "kernels/ggml/validate_util.h"

namespace jitllm::kernels::ggml {
namespace {

// GGML's layout of a custom node's op_params (ggml-impl.h
// ggml_custom_op_params), which ggml_custom_4d fills.
struct JitllmCustomParams {
  ggml_custom_op_t fun;
  int n_tasks;
  void* userdata;
};

using detail::Aligned;
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
  return JitllmOp::kNone;
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

}  // namespace jitllm::kernels::ggml
