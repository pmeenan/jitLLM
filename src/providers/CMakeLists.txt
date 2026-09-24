# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The provider interfaces and their implementations
# (docs/architecture.md#providers, D-026). So far, the device probe:
# jitllm_device_probe is the CUDA provider's in a CUDA build, and in the
# CPU-only profile one that reports no device provider.
if(JITLLM_CUDA)
  add_subdirectory(cuda)
  add_library(jitllm_device_probe ALIAS jitllm_cuda_probe)
else()
  add_library(jitllm_device_probe STATIC cpu_only_device_probe.cc)
  target_include_directories(jitllm_device_probe PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(jitllm_device_probe PUBLIC jitllm_report PRIVATE jitllm_warnings)
endif()
