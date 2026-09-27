// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/implementations.h"

#include <array>
#include <cstddef>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(JITLLM_GGML_SOURCE_TREE) || !defined(JITLLM_GGML_SDK) ||           \
    !defined(JITLLM_GGML_TARGET) || !defined(JITLLM_GGML_CUDA_ARCHITECTURES) || \
    !defined(JITLLM_GGML_BUILD_TYPE) || !defined(JITLLM_GGML_SANITIZE)
#error "implementations.cc needs the GGML source tree, SDK, target, architectures and build type"
#endif

namespace jitllm::kernels::ggml {

struct RmsNormMulKernel::Entry {
  std::string_view name;
  std::string_view variant;
  std::expected<void, KernelFailure> (*check)(const ggml_tensor* norm, const ggml_tensor* mul);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch, ggml_tensor* norm,
                                            ggml_tensor* mul);
};

struct Kernel::Entry {
  std::string_view name;
  execution::Operation operation;
  std::string_view variant;
  std::size_t arity;
  // Called with exactly `arity` nodes.
  std::expected<void, KernelFailure> (*check)(std::span<const ggml_tensor* const> nodes);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch,
                                            std::span<ggml_tensor* const> nodes);
};

namespace {

// Whether this build keeps asserts, GGML's device asserts among them.
#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif

constexpr std::array<RmsNormMulKernel::Entry, 2> kRmsNormMul = {{
    {.name = "ggml.rms_norm_mul.fused",
     .variant = "ggml_cuda_op_rms_norm_fused; upstream launch configuration",
     .check = &CheckRmsNormMul,
     .run = &RmsNormMul},
    {.name = "ggml.rms_norm_mul.unfused",
     .variant = "ggml_cuda_op_rms_norm, then ggml_cuda_op_mul; upstream launch configuration",
     .check = &CheckRmsNormThenMul,
     .run = &RmsNormThenMul},
}};

// The other implementations (implementations.h), each checking and running
// its nodes through ops.h.
using Nodes = std::span<ggml_tensor* const>;
using ConstNodes = std::span<const ggml_tensor* const>;

constexpr std::array<Kernel::Entry, 15> kKernels = {{
    {.name = "ggml.rms_norm",
     .operation = execution::Operation::kRmsNorm,
     .variant = "ggml_cuda_op_rms_norm: rms_norm_f32<block, false, false>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRmsNorm(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RmsNorm(launch, n[0]); }},
    {.name = "ggml.add",
     .operation = execution::Operation::kAdd,
     .variant = "ggml_cuda_op_add: k_bin_bcast<op_add, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_ADD); },
     .run = [](LaunchContext& launch, Nodes n) { return Add(launch, n[0]); }},
    {.name = "ggml.mul",
     .operation = execution::Operation::kMul,
     .variant = "ggml_cuda_op_mul: k_bin_bcast<op_mul, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_MUL); },
     .run = [](LaunchContext& launch, Nodes n) { return Mul(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmvf",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_vec_f: mul_mat_vec_f<T, type_acc, ncols, block, false, false> "
                "as upstream selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecF(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmf",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_f: mul_mat_f<T, warp, cols, nwarps, false> as upstream "
                "selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatF(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatF(launch, n[0]); }},
    {.name = "ggml.mul_mat.cublas",
     .operation = execution::Operation::kMatMul,
     .variant = "GGML's cuBLAS path (mul_mat_cublas.cu): conversions, GemmEx, strided or "
                "pointer-array batched GEMM on the lent handle, as upstream plans them",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatCublas(launch, n[0]); }},
    {.name = "ggml.get_rows",
     .operation = execution::Operation::kGetRows,
     .variant = "ggml_cuda_op_get_rows: k_get_rows_float_vec on 16-byte vectors, aligned rows "
                "and at least 128 blocks, else k_get_rows_float; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGetRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GetRows(launch, n[0]); }},
    {.name = "ggml.set_rows",
     .operation = execution::Operation::kSetRows,
     .variant = "ggml_cuda_op_set_rows: k_set_rows<float, int64_t, half>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSetRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SetRows(launch, n[0]); }},
    {.name = "ggml.rope.neox",
     .operation = execution::Operation::kRope,
     .variant = "ggml_cuda_op_rope: rope_neox<true, false, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRope(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Rope(launch, n[0]); }},
    {.name = "ggml.rope_set_rows.fused",
     .operation = execution::Operation::kRopeSetRows,
     .variant = "ggml_cuda_op_rope_fused: rope_neox<true, false, float, half> writing the KV "
                "destination; upstream launch configuration",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckRopeSetRows(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return RopeSetRows(launch, n[0], n[1]); }},
    {.name = "ggml.soft_max",
     .operation = execution::Operation::kSoftMax,
     .variant = "ggml_cuda_op_soft_max: soft_max_f32<true, ncols, block, float> for 32 to 4,096 "
                "columns in powers of two, else soft_max_f32<true, 0, 0, float>, rows in shared "
                "memory only; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSoftMax(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SoftMax(launch, n[0]); }},
    {.name = "ggml.cont",
     .operation = execution::Operation::kCont,
     .variant = "ggml_cuda_dup: cudaMemcpyAsync if contiguous, cudaMemcpy2DAsync for a pitched "
                "block, else cpy_scalar<cpy_1_scalar<float, float>>; no tiled transpose; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) -> std::expected<void, KernelFailure> {
       if (auto checked = CheckCont(n[0]); !checked) {
         return std::unexpected(checked.error());
       }
       return {};
     },
     .run = [](LaunchContext& launch, Nodes n) { return Cont(launch, n[0]); }},
    {.name = "ggml.swiglu",
     .operation = execution::Operation::kSwiGlu,
     .variant = "ggml_cuda_op_swiglu: unary_gated_op_kernel<op_silu, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSwiGlu(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SwiGlu(launch, n[0]); }},
    {.name = "ggml.mul_mat_add.mmvf_fused",
     .operation = execution::Operation::kMulMatAdd,
     .variant = "ggml_cuda_mul_mat_vec_f with x_bias, writing the add: mul_mat_vec_f<T, "
                "type_acc, 1, block, true, false>, the add's precision; upstream launch "
                "configuration",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckMulMatVecBias(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecBias(launch, n[0], n[1]); }},
    {.name = "ggml.mul_mat_glu.mmvf_fused",
     .operation = execution::Operation::kMulMatGlu,
     .variant = "ggml_cuda_mul_mat_vec_f with gate and SwiGLU, writing the GLU: "
                "mul_mat_vec_f<T, type_acc, 1, block, true, false>, the GLU's parameters as "
                "precision; upstream launch configuration",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckMulMatVecGlu(n[0], n[1], n[2]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecGlu(launch, n[0], n[1], n[2]); }},
}};

execution::Implementation Declare(std::string_view name, execution::Operation operation,
                                  std::string_view variant) {
  return {.name = std::string(name),
          .operation = operation,
          .source = "ggml",
          .revision = std::format("ggml tree {}; jitllm module {}", JITLLM_GGML_SOURCE_TREE,
                                  ModuleSourcesDigest()),
          .build = std::format("sdk {}; target {}; cuda {}; build type {}; {}; sanitizers {}",
                               JITLLM_GGML_SDK, JITLLM_GGML_TARGET, JITLLM_GGML_CUDA_ARCHITECTURES,
                               JITLLM_GGML_BUILD_TYPE, kAsserts, JITLLM_GGML_SANITIZE),
          .variant = std::string(variant)};
}

execution::Implementation Declare(const RmsNormMulKernel::Entry& entry) {
  return Declare(entry.name, execution::Operation::kRmsNormMul, entry.variant);
}

execution::Implementation Declare(const Kernel::Entry& entry) {
  return Declare(entry.name, entry.operation, entry.variant);
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kRmsNormMul.size() + kKernels.size());
  for (const RmsNormMulKernel::Entry& entry : kRmsNormMul) {
    declared.push_back(Declare(entry));
  }
  for (const Kernel::Entry& entry : kKernels) {
    declared.push_back(Declare(entry));
  }
  return declared;
}

std::expected<RmsNormMulKernel, KernelFailure> RmsNormMulKernel::Bind(
    const execution::Implementation& implementation) {
  for (const Entry& entry : kRmsNormMul) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return RmsNormMulKernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML rms_norm_mul implementation of this build",
                            implementation.name)});
}

std::expected<void, KernelFailure> RmsNormMulKernel::Check(const ggml_tensor* norm,
                                                           const ggml_tensor* mul) const {
  return entry_->check(norm, mul);
}

std::expected<void, KernelFailure> RmsNormMulKernel::Run(LaunchContext& launch, ggml_tensor* norm,
                                                         ggml_tensor* mul) const {
  return entry_->run(launch, norm, mul);
}

std::string_view RmsNormMulKernel::name() const { return entry_->name; }

std::expected<Kernel, KernelFailure> Kernel::Bind(const execution::Implementation& implementation) {
  for (const Entry& entry : kKernels) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return Kernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML implementation of this build", implementation.name)});
}

std::expected<void, KernelFailure> Kernel::Check(std::span<const ggml_tensor* const> nodes) const {
  if (nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->check(nodes);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               std::span<ggml_tensor* const> nodes) const {
  if (nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->run(launch, nodes);
}

std::string_view Kernel::name() const { return entry_->name; }

execution::Operation Kernel::operation() const { return entry_->operation; }

std::size_t Kernel::arity() const { return entry_->arity; }

}  // namespace jitllm::kernels::ggml
