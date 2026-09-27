// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/implementations.h"

#include <array>
#include <expected>
#include <format>
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

execution::Implementation Declare(const RmsNormMulKernel::Entry& entry) {
  return {.name = std::string(entry.name),
          .operation = execution::Operation::kRmsNormMul,
          .source = "ggml",
          .revision = std::format("ggml tree {}; jitllm module {}", JITLLM_GGML_SOURCE_TREE,
                                  ModuleSourcesDigest()),
          .build = std::format("sdk {}; target {}; cuda {}; build type {}; {}; sanitizers {}",
                               JITLLM_GGML_SDK, JITLLM_GGML_TARGET, JITLLM_GGML_CUDA_ARCHITECTURES,
                               JITLLM_GGML_BUILD_TYPE, kAsserts, JITLLM_GGML_SANITIZE),
          .variant = std::string(entry.variant)};
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kRmsNormMul.size());
  for (const RmsNormMulKernel::Entry& entry : kRmsNormMul) {
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

}  // namespace jitllm::kernels::ggml
