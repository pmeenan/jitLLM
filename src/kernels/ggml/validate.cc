// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/validate.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "ggml.h"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

bool IsF32(const ggml_tensor* tensor) { return tensor != nullptr && tensor->type == GGML_TYPE_F32; }

bool Bound(const ggml_tensor* tensor) { return tensor != nullptr && tensor->data != nullptr; }

// The kernels load whole elements, or pairs of them, from the data
// pointer; a misaligned load is a sticky fault that ends every CUDA call in
// the process.
bool Aligned(const ggml_tensor* tensor, std::uint64_t alignment) {
  return reinterpret_cast<std::uintptr_t>(tensor->data) % alignment == 0;
}

// The base and every stride are multiples of `alignment`: the kernels
// offset the base by any combination of strides before a paired load.
bool AlignedEverywhere(const ggml_tensor* tensor, std::uint64_t alignment) {
  if (!Aligned(tensor, alignment)) {
    return false;
  }
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (tensor->nb[i] % alignment != 0) {
      return false;
    }
  }
  return true;
}

// The bytes from a bound tensor's first element to one past its last, by
// checked arithmetic: nothing for an empty tensor, a stride GGML would
// treat as negative, or an extent or end address that overflows
// (ggml_nbytes wraps). Unblocked types only.
std::optional<std::uint64_t> Extent(const ggml_tensor* tensor) {
  std::uint64_t extent = ggml_type_size(tensor->type);
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (tensor->ne[i] <= 0 || tensor->nb[i] > std::numeric_limits<std::int64_t>::max()) {
      return std::nullopt;
    }
    std::uint64_t step = 0;
    if (__builtin_mul_overflow(static_cast<std::uint64_t>(tensor->ne[i] - 1), tensor->nb[i],
                               &step) ||
        __builtin_add_overflow(extent, step, &extent)) {
      return std::nullopt;
    }
  }
  std::uint64_t end = 0;
  if (__builtin_add_overflow(
          static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data)), extent,
          &end)) {
    return std::nullopt;
  }
  return extent;
}

bool AllSane(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::all_of(tensors,
                             [](const ggml_tensor* tensor) { return Extent(tensor).has_value(); });
}

// Strides exactly those of a dense tensor of its shape: what the dense
// operations below accept, since GGML's contiguity test skips extents of
// one and its broadcast launcher merges dimensions assuming packed strides.
bool Packed(const ggml_tensor* tensor) {
  if (tensor->nb[0] != ggml_type_size(tensor->type)) {
    return false;
  }
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    if (tensor->nb[i] != tensor->nb[i - 1] * static_cast<std::uint64_t>(tensor->ne[i - 1])) {
      return false;
    }
  }
  return true;
}

// Whether a's bytes and b's overlap (both bound); an unmeasurable tensor
// counts as overlapping.
bool Overlap(const ggml_tensor* a, const ggml_tensor* b) {
  const auto a_extent = Extent(a);
  const auto b_extent = Extent(b);
  if (!a_extent || !b_extent) {
    return true;
  }
  const auto a_begin = reinterpret_cast<std::uintptr_t>(a->data);
  const auto b_begin = reinterpret_cast<std::uintptr_t>(b->data);
  return a_begin < b_begin + *b_extent && b_begin < a_begin + *a_extent;
}

// An output may share no byte with an input, except, where the operation
// works element by element or row by row, by being exactly that input:
// same address, shape and strides (in place). Upstream's allocator never
// places an output partly over an input.
bool Disjoint(const ggml_tensor* output, const ggml_tensor* input, bool in_place) {
  if (!Overlap(output, input)) {
    return true;
  }
  return in_place && output->data == input->data && ggml_are_same_shape(output, input) &&
         ggml_are_same_stride(output, input);
}

// Every view in the chain still points where its source does: a view made
// before its source was bound, or kept after the source was bound again,
// keeps the old address (tensors.h).
bool Current(const ggml_tensor* tensor) {
  for (; tensor != nullptr && tensor->view_src != nullptr; tensor = tensor->view_src) {
    const auto* expected = static_cast<const char*>(tensor->view_src->data);
    if (expected == nullptr ||
        static_cast<const char*>(tensor->data) != expected + tensor->view_offs) {
      return false;
    }
  }
  return true;
}

bool AllCurrent(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::all_of(tensors, Current);
}

// The tensor that owns a view's storage.
const ggml_tensor* Root(const ggml_tensor* tensor) {
  while (tensor->view_src != nullptr) {
    tensor = tensor->view_src;
  }
  return tensor;
}

// Every stride is a whole number of elements, as the launchers divide
// strides by the element size (the broadcast launcher asserts it).
bool ElementStrides(const ggml_tensor* tensor) {
  const std::uint64_t size = ggml_type_size(tensor->type);
  return std::ranges::all_of(tensor->nb, [size](std::size_t stride) { return stride % size == 0; });
}

// The elements from the first to the last one the tensor addresses: the
// kernels index within it with 32-bit arithmetic. Only for sane tensors.
std::uint64_t Span(const ggml_tensor* tensor) {
  return Extent(tensor).value_or(std::numeric_limits<std::uint64_t>::max()) /
         ggml_type_size(tensor->type);
}

// Upstream skips empty nodes; its launchers divide by extents and build
// fast divisors from them, so an empty operand would abort the process.
bool AnyEmpty(std::initializer_list<const ggml_tensor*> tensors) {
  return std::ranges::any_of(tensors, ggml_is_empty);
}

// Every extent and element stride fits 32 bits, as the broadcast launcher
// asserts.
bool Fits32(const ggml_tensor* tensor) {
  const std::uint64_t size = ggml_type_size(tensor->type);
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    if (std::cmp_greater(tensor->ne[i], std::numeric_limits<std::uint32_t>::max()) ||
        tensor->nb[i] / size > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
  }
  return true;
}

// The row kernels launch one block per row, channel and sample.
bool FitsRowGrid(const ggml_tensor* tensor) {
  return tensor->ne[1] <= std::numeric_limits<std::int32_t>::max() && tensor->ne[2] <= 65535 &&
         tensor->ne[3] <= 65535;
}

float Epsilon(const ggml_tensor* norm) {
  float eps = 0.0f;
  std::memcpy(&eps, norm->op_params, sizeof(eps));
  return eps;
}

std::expected<void, KernelFailure> CheckNormSource(const ggml_tensor* norm) {
  if (norm == nullptr || norm->op != GGML_OP_RMS_NORM || !IsF32(norm) || !IsF32(norm->src[0]) ||
      !Bound(norm->src[0])) {
    return Rejected("not a bound F32 rms_norm node");
  }
  if (AnyEmpty({norm, norm->src[0]}) || !Extent(norm->src[0])) {
    return Rejected("rms_norm over an empty or unmeasurable tensor");
  }
  if (!ggml_are_same_shape(norm, norm->src[0]) || !Packed(norm->src[0])) {
    return Rejected("rms_norm over a packed F32 tensor of its own shape");
  }
  // The kernels index with 32-bit arithmetic, and step a row's column
  // index by up to 1,024 past its last element (norm.cu).
  if (!(Epsilon(norm) >= 0.0f) || norm->src[0]->nb[0] != sizeof(float) ||
      !ElementStrides(norm->src[0]) || !FitsRowGrid(norm->src[0]) ||
      ggml_nelements(norm) > std::numeric_limits<std::int32_t>::max() - 1024 ||
      !Aligned(norm->src[0], sizeof(float))) {
    return Rejected(
        "rms_norm needs a non-negative epsilon and aligned, contiguous rows within the grid");
  }
  return {};
}

}  // namespace

std::expected<void, KernelFailure> CheckBinary(const ggml_tensor* node, ggml_op op) {
  if (node == nullptr || node->op != op || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound node of the operation");
  }
  if (!IsF32(node) || !IsF32(node->src[0]) || !IsF32(node->src[1])) {
    return Rejected("this implementation is F32 only");
  }
  if (AnyEmpty({node, node->src[0], node->src[1]}) ||
      !AllSane({node, node->src[0], node->src[1]})) {
    return Rejected("a binary operation on an empty or unmeasurable tensor");
  }
  if (!ggml_are_same_shape(node, node->src[0]) || !Packed(node) || !Packed(node->src[0]) ||
      !Packed(node->src[1])) {
    return Rejected("a binary operation over packed operands, the output of src0's shape");
  }
  // The launcher collapses contiguous dimensions and may launch one thread
  // per element, asserting that each collapsed extent, stride and the
  // thread count (a multiple of its 128-thread blocks) fit 32 bits.
  if (!ggml_can_repeat(node->src[1], node->src[0]) || !Fits32(node) || !Fits32(node->src[0]) ||
      !Fits32(node->src[1]) || !ElementStrides(node) || !ElementStrides(node->src[0]) ||
      !ElementStrides(node->src[1]) ||
      std::cmp_greater(ggml_nelements(node), std::numeric_limits<std::uint32_t>::max() - 127)) {
    return Rejected("operands that do not broadcast or exceed 32-bit extents");
  }
  if (!Aligned(node, sizeof(float)) || !Aligned(node->src[0], sizeof(float)) ||
      !Aligned(node->src[1], sizeof(float))) {
    return Rejected("F32 operands at misaligned addresses");
  }
  // The kernel steps through rows one element at a time, whatever the
  // first stride says, and writes the output densely.
  if (!ggml_is_contiguous(node) || node->src[0]->nb[0] != sizeof(float) ||
      node->src[1]->nb[0] != sizeof(float)) {
    return Rejected("a contiguous output over operands with contiguous rows");
  }
  if (!AllCurrent({node, node->src[0], node->src[1]}) ||
      !Disjoint(node, node->src[0], /*in_place=*/true) ||
      !Disjoint(node, node->src[1], /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping an operand other than in place");
  }
  return {};
}

// What GGML's matrix-multiply entry points assert, apart from the kernel
// family's own selection.
std::expected<void, KernelFailure> CheckMulMat(const ggml_tensor* node) {
  if (node == nullptr || node->op != GGML_OP_MUL_MAT || !Bound(node) || !Bound(node->src[0]) ||
      !Bound(node->src[1])) {
    return Rejected("not a bound mul_mat node");
  }
  const ggml_tensor* weights = node->src[0];
  const ggml_tensor* input = node->src[1];
  if (!IsF32(input) || !IsF32(node) ||
      (weights->type != GGML_TYPE_F16 && weights->type != GGML_TYPE_F32 &&
       weights->type != GGML_TYPE_BF16)) {
    return Rejected("F16, BF16 or F32 weights with F32 activations and output");
  }
  // A hint (op_params[1]) lets upstream route the node to another
  // operation, such as a Hadamard transform.
  std::int32_t hint = 0;
  std::memcpy(&hint, &node->op_params[1], sizeof(hint));
  if (hint != GGML_HINT_NONE) {
    return Rejected("a mul_mat node with a routing hint");
  }
  if (input->ne[3] != node->ne[3] || weights->nb[0] != ggml_type_size(weights->type) ||
      input->nb[0] != sizeof(float) || node->nb[0] != sizeof(float) || !ElementStrides(weights) ||
      !ElementStrides(input) || !ElementStrides(node)) {
    return Rejected("mul_mat needs contiguous rows and matching samples");
  }
  if (AnyEmpty({node, weights, input}) || !AllSane({node, weights, input})) {
    return Rejected("mul_mat on an empty or unmeasurable tensor");
  }
  // ggml_can_mul_mat (static in ggml.c): input channels and samples are
  // whole multiples of the weights'.
  if (weights->ne[0] != input->ne[0] || input->ne[2] % weights->ne[2] != 0 ||
      input->ne[3] % weights->ne[3] != 0 || node->ne[0] != weights->ne[1] ||
      node->ne[1] != input->ne[1] || node->ne[2] != input->ne[2] || node->ne[3] != input->ne[3]) {
    return Rejected("mul_mat whose shape does not follow from its operands");
  }
  // The kernels take strides and offsets as 32-bit integers, and launch a
  // block per output row, channel and sample.
  constexpr std::uint64_t kInt32 = std::numeric_limits<std::int32_t>::max();
  const auto element_strides_fit = [](const ggml_tensor* tensor) {
    const std::uint64_t size = ggml_type_size(tensor->type);
    return std::ranges::all_of(tensor->nb,
                               [size](std::size_t stride) { return stride / size <= kInt32; });
  };
  if (Span(weights) > kInt32 || Span(input) > kInt32 || Span(node) > kInt32 ||
      !element_strides_fit(weights) || !element_strides_fit(input) || !element_strides_fit(node) ||
      node->ne[2] > 65535 || node->ne[3] > 65535) {
    return Rejected("mul_mat operands beyond the kernels' 32-bit indexing or grid");
  }
  // Weights load as element pairs (half2, nv_bfloat162, float2), and
  // activations and output as float2.
  if (!AlignedEverywhere(weights, 2 * ggml_type_size(weights->type)) ||
      !AlignedEverywhere(input, 8) || !AlignedEverywhere(node, 8)) {
    return Rejected("mul_mat operands at misaligned addresses or strides");
  }
  if (!AllCurrent({node, weights, input}) || !Disjoint(node, weights, /*in_place=*/false) ||
      !Disjoint(node, input, /*in_place=*/false)) {
    return Rejected("a stale view, or an output overlapping an operand");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRmsNorm(const ggml_tensor* norm) {
  if (auto checked = CheckNormSource(norm); !checked) {
    return checked;
  }
  // The kernel writes its output densely, whatever the node's strides.
  if (!Bound(norm) || !ggml_is_contiguous(norm) || !Aligned(norm, sizeof(float))) {
    return Rejected("rms_norm writes an aligned, contiguous F32 tensor");
  }
  if (!AllCurrent({norm, norm->src[0]}) || !Disjoint(norm, norm->src[0], /*in_place=*/true)) {
    return Rejected("a stale view, or an output overlapping the input other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckRmsNormMul(const ggml_tensor* norm,
                                                   const ggml_tensor* mul) {
  if (auto checked = CheckNormSource(norm); !checked) {
    return checked;
  }
  if (mul == nullptr || mul->op != GGML_OP_MUL || !Bound(mul) || !IsF32(mul) ||
      !IsF32(mul->src[0]) || !IsF32(mul->src[1]) || !Fits32(mul)) {
    return Rejected("not a bound F32 mul node");
  }
  // GGML's fusion gate (ggml_cuda_can_fuse) and the fused launcher's checks.
  const ggml_tensor* weight = nullptr;
  if (mul->src[0] == norm) {
    weight = mul->src[1];
  } else if (mul->src[1] == norm) {
    weight = mul->src[0];
    if (!ggml_are_same_shape(weight, norm)) {
      return Rejected("with the norm as the second operand, fusion does not broadcast");
    }
  } else {
    return Rejected("the mul does not scale this norm");
  }
  // The kernel writes the product densely, whatever the node's strides.
  if (!Bound(weight) || ggml_is_empty(weight) || !ggml_is_contiguous_rows(mul->src[0]) ||
      !ggml_is_contiguous_rows(mul->src[1]) || weight->nb[0] != sizeof(float) ||
      !ElementStrides(weight) || !Fits32(weight) || !ggml_is_contiguous(mul)) {
    return Rejected("fusion needs contiguous rows and a contiguous product");
  }
  if (!Aligned(weight, sizeof(float)) || !Aligned(mul, sizeof(float))) {
    return Rejected("fusion operands at misaligned addresses");
  }
  if (!AllSane({weight, mul}) || !Packed(weight) || !Packed(mul) ||
      !ggml_are_same_shape(mul, norm) || !ggml_can_repeat(weight, norm)) {
    return Rejected("fusion over packed operands, the product of the norm's shape");
  }
  // The fused kernel never writes the norm, so the weight must not be the
  // norm, a view of it or memory that overlaps it: upstream fuses only a
  // norm with no other use (ggml_can_fuse_ext).
  // Comparing storage roots covers a weight that is the norm, a view of it,
  // or the storage an in-place norm is a view of.
  if (Root(weight) == Root(norm) || (Bound(norm) && Overlap(weight, norm))) {
    return Rejected("the weight shares the norm's storage, which the fused kernel never writes");
  }
  if (!AllCurrent({norm, norm->src[0], mul, weight}) ||
      !Disjoint(mul, norm->src[0], /*in_place=*/true) ||
      !Disjoint(mul, weight, /*in_place=*/true)) {
    return Rejected("a stale view, or a product overlapping an operand other than in place");
  }
  return {};
}

std::expected<void, KernelFailure> CheckMulMatF(const ggml_tensor* node) {
  if (auto checked = CheckMulMat(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  if (node->src[1]->ne[1] > 16) {
    return Rejected("MMF takes at most 16 activation columns");
  }
  // The launcher counts F16 and BF16 strides in pairs and asserts them even,
  // which upstream's selection does not fully check.
  const std::uint64_t pair = weights->type == GGML_TYPE_F32 ? 2 : 4;
  if ((weights->nb[1] / ggml_type_size(weights->type)) % pair != 0 ||
      (node->src[1]->nb[1] / sizeof(float)) % pair != 0) {
    return Rejected("MMF needs even weight row and activation column strides");
  }
  return {};
}

std::expected<void, KernelFailure> CheckClearOf(const ggml_tensor* node, std::uint64_t base,
                                                std::uint64_t size) {
  if (size == 0) {
    return {};
  }
  for (const ggml_tensor* tensor :
       std::initializer_list<const ggml_tensor*>{node, node->src[0], node->src[1]}) {
    if (tensor == nullptr) {
      continue;
    }
    const auto extent = Extent(tensor);
    if (!extent) {
      return Rejected("an operand that cannot be measured");
    }
    const auto begin = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(tensor->data));
    if (begin < base + size && base < begin + *extent) {
      return Rejected("an operand overlaps a workspace the launch writes");
    }
  }
  return {};
}

std::expected<CublasMulMat, KernelFailure> CheckMulMatCublas(const ggml_tensor* node,
                                                             ggml_type compute, bool f32_output) {
  if (auto checked = CheckMulMat(node); !checked) {
    return std::unexpected(checked.error());
  }
  if (compute != GGML_TYPE_F32 && compute != GGML_TYPE_F16 && compute != GGML_TYPE_BF16) {
    return Rejected("cuBLAS computes in F32, F16 or BF16");
  }
  const ggml_tensor* src0 = node->src[0];
  const ggml_tensor* src1 = node->src[1];
  // The launcher asserts a contiguous output and indexes it as packed.
  if (!Packed(node)) {
    return Rejected("cuBLAS writes a packed output");
  }
  // CheckMulMat bounds every extent and element stride to 32 bits, so the
  // products below cannot overflow 64.
  const std::uint64_t compute_size = ggml_type_size(compute);
  const std::uint64_t ts0 = ggml_type_size(src0->type);
  const std::uint64_t ts1 = ggml_type_size(src1->type);
  CublasMulMat plan{.compute = compute, .f32_output = f32_output};
  plan.s01 = static_cast<std::int64_t>(src0->nb[1] / ts0);
  plan.s02 = static_cast<std::int64_t>(src0->nb[2] / ts0);
  plan.s03 = static_cast<std::int64_t>(src0->nb[3] / ts0);
  plan.s11 = static_cast<std::int64_t>(src1->nb[1] / ts1);
  plan.s12 = static_cast<std::int64_t>(src1->nb[2] / ts1);
  plan.s13 = static_cast<std::int64_t>(src1->nb[3] / ts1);
  bool src0_cont_2 = ggml_is_contiguous_2(src0);
  bool src1_cont_2 = ggml_is_contiguous_2(src1);

  // The pool hands out blocks from 256-byte boundaries (launch.cu), in the
  // launcher's order: weights, input, output, then the pointer arrays.
  const auto draw = [&plan](std::uint64_t bytes) {
    plan.scratch = ((plan.scratch + 255) / 256 * 256) + bytes;
  };
  // Each operand that is not already the compute type is converted into
  // scratch: element by element if its bytes are exactly its elements
  // (strides kept, blocks of one element), else gathered into packed rows.
  if (src0->type != compute) {
    draw(static_cast<std::uint64_t>(ggml_nelements(src0)) * compute_size);
    if (ggml_is_contiguously_allocated(src0)) {
      plan.weights = CublasOperand::kConverted;
    } else {
      plan.weights = CublasOperand::kPacked;
      plan.s01 = src0->ne[0];
      plan.s02 = src0->ne[1] * plan.s01;
      plan.s03 = src0->ne[2] * plan.s02;
      src0_cont_2 = true;
    }
  }
  if (src1->type != compute) {
    draw(static_cast<std::uint64_t>(ggml_nelements(src1)) * compute_size);
    if (ggml_is_contiguously_allocated(src1)) {
      plan.input = CublasOperand::kConverted;
    } else {
      plan.input = CublasOperand::kPacked;
      plan.s11 = src1->ne[0];
      plan.s12 = src1->ne[1] * plan.s11;
      plan.s13 = src1->ne[2] * plan.s12;
      src1_cont_2 = true;
    }
  }
  if (!f32_output && compute != GGML_TYPE_F32) {
    draw(static_cast<std::uint64_t>(ggml_nelements(node)) * compute_size);
  }

  // What cuBLAS is given: each operand's base (a scratch block's is 256-
  // aligned) and strides in the type it reads, and the output's.
  const auto align = [&plan](std::uint64_t value) {
    if (value != 0) {
      plan.alignment = std::min(plan.alignment, value & (~value + 1));
    }
  };
  const auto operand = [&align](const ggml_tensor* src, CublasOperand how, std::uint64_t size,
                                std::initializer_list<std::int64_t> strides) {
    align(how == CublasOperand::kDirect
              ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(src->data))
              : 256);
    for (const std::int64_t stride : strides) {
      align(static_cast<std::uint64_t>(stride) * size);
    }
  };
  operand(src0, plan.weights, plan.weights == CublasOperand::kDirect ? ts0 : compute_size,
          {plan.s01, plan.s02, plan.s03});
  operand(src1, plan.input, plan.input == CublasOperand::kDirect ? ts1 : compute_size,
          {plan.s11, plan.s12, plan.s13});
  const bool output_direct = f32_output || compute == GGML_TYPE_F32;
  const std::uint64_t output_size = output_direct ? sizeof(float) : compute_size;
  align(output_direct ? static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(node->data))
                      : 256);
  for (int i = 1; i < GGML_MAX_DIMS; ++i) {
    align(node->nb[i] / sizeof(float) * output_size);
  }

  // cuBLAS reads the weights transposed: each of its columns is a weight
  // row, so both operands' leading dimensions must reach k (it refuses
  // less), and it takes the batch count as an int.
  const std::int64_t k = src0->ne[0];
  const std::int64_t batches = src1->ne[2] * src1->ne[3];
  if (plan.s01 < k || plan.s11 < k) {
    return Rejected("cuBLAS needs rows at least k elements apart in both operands");
  }
  if (batches > std::numeric_limits<std::int32_t>::max()) {
    return Rejected("more matrices than cuBLAS takes in one call");
  }
  const bool broadcast = src1->ne[2] != src0->ne[2] || src1->ne[3] != src0->ne[3];
  if (src1->ne[2] == 1 && src1->ne[3] == 1) {
    plan.gemm = compute == GGML_TYPE_F32 ? CublasGemm::kSgemm : CublasGemm::kGemmEx;
  } else if (!broadcast && src0_cont_2 && src1_cont_2) {
    plan.gemm = CublasGemm::kGemmStridedBatchedEx;
  } else {
    plan.gemm = CublasGemm::kGemmBatchedEx;
    // Two input and one output pointer per matrix.
    draw(2 * static_cast<std::uint64_t>(batches) * sizeof(void*));
    draw(static_cast<std::uint64_t>(batches) * sizeof(void*));
  }
  return plan;
}

}  // namespace jitllm::kernels::ggml
