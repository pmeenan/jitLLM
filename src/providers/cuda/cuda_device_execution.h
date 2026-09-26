// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The CUDA device-execution provider: DeviceExecution over the driver API.
// Streams are non-blocking (they never synchronize with the legacy default
// stream); copies are asynchronous copies between VMM addresses; fences
// are timing-free events, queried with cuEventQuery. Like the device-memory
// provider it retains the device's primary context and makes it current on
// each call. Errors map as the device-memory provider's do (cuda_errors.h). This header holds no
// CUDA types.

#ifndef JITLLM_PROVIDERS_CUDA_CUDA_DEVICE_EXECUTION_H_
#define JITLLM_PROVIDERS_CUDA_CUDA_DEVICE_EXECUTION_H_

#include <expected>
#include <memory>

#include "providers/device_execution.h"

namespace jitllm::providers::cuda {

std::expected<std::unique_ptr<DeviceExecution>, Failure> OpenDeviceExecution(int ordinal);

}  // namespace jitllm::providers::cuda

#endif  // JITLLM_PROVIDERS_CUDA_CUDA_DEVICE_EXECUTION_H_
