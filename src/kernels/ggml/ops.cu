// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include <expected>
#include <string>
#include <utility>

#include "base/bytes.h"
#include "binbcast.cuh"
#include "common.cuh"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/validate.h"
#include "mmf.cuh"
#include "mmvf.cuh"
#include "norm.cuh"

namespace jitllm::kernels::ggml {
namespace {

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

// The launch context's device, read without a CUDA call: its table was
// read when the context was created.
const ggml_cuda_device_info::cuda_device_info& Device(const LaunchContext& launch) {
  return ggml_cuda_info().devices[launch.device()];
}

}  // namespace

std::expected<void, KernelFailure> RmsNorm(LaunchContext& launch, ggml_tensor* norm) {
  if (auto checked = CheckRmsNorm(norm); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm(context, norm);
  });
}

std::expected<void, KernelFailure> RmsNormMul(LaunchContext& launch, ggml_tensor* norm,
                                              ggml_tensor* mul) {
  if (auto checked = CheckRmsNormMul(norm, mul); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [norm, mul](ggml_backend_cuda_context& context) {
    ggml_cuda_op_rms_norm_fused(context, norm, mul);
  });
}

std::expected<void, KernelFailure> Add(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_ADD); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_add(context, node);
  });
}

std::expected<void, KernelFailure> Mul(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckBinary(node, GGML_OP_MUL); !checked) {
    return checked;
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_op_mul(context, node);
  });
}

std::expected<void, KernelFailure> MulMatVecF(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMat(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  if (!ggml_cuda_should_use_mmvf(weights->type, Device(launch).cc, weights->ne, weights->nb,
                                 node->src[1]->ne[1])) {
    return Rejected("upstream does not select MMVF for these operands");
  }
  // The launcher's even column-stride assertion holds: CheckMulMat requires
  // every activation stride to be a multiple of 8 bytes.
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_vec_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

std::expected<void, KernelFailure> MulMatF(LaunchContext& launch, ggml_tensor* node) {
  if (auto checked = CheckMulMatF(node); !checked) {
    return checked;
  }
  const ggml_tensor* weights = node->src[0];
  const auto& device = Device(launch);
  if (!ggml_cuda_should_use_mmf(weights->type, device.cc, device.warp_size, weights->ne,
                                weights->nb, static_cast<int>(node->src[1]->ne[1]),
                                /*mul_mat_id=*/false)) {
    return Rejected("upstream does not select MMF for these operands");
  }
  return launch.Run(base::Bytes(0), [node](ggml_backend_cuda_context& context) {
    ggml_cuda_mul_mat_f(context, node->src[0], node->src[1], nullptr, node);
  });
}

}  // namespace jitllm::kernels::ggml
