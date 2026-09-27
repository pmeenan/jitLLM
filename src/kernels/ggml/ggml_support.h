// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// What the launch context needs from ggml_support.cu, jitLLM's definitions
// of the ggml-cuda.cu symbols GGML's launchers use. Internal to this module.

#ifndef JITLLM_KERNELS_GGML_GGML_SUPPORT_H_
#define JITLLM_KERNELS_GGML_GGML_SUPPORT_H_

#include <optional>
#include <string>

namespace jitllm::kernels::ggml::internal {

// The first CUDA failure GGML recorded on this thread since the last call,
// if any; taking it clears it.
std::optional<std::string> TakeCudaError();

}  // namespace jitllm::kernels::ggml::internal

#endif  // JITLLM_KERNELS_GGML_GGML_SUPPORT_H_
