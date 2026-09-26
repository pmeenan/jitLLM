# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The provider interfaces and their implementations
# (docs/architecture.md#providers, D-026). The device probe:
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

# The device-memory interface and the rules every implementation shares
# (device_memory.h).
add_library(jitllm_device_memory STATIC device_memory.cc)
target_include_directories(jitllm_device_memory PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_device_memory PUBLIC jitllm_base PRIVATE jitllm_warnings)

# The storage interface (storage.h), its io_uring implementation and whole
# reads over any implementation (direct_reader.h).
add_library(jitllm_storage STATIC uring_storage.cc direct_reader.cc)
target_include_directories(jitllm_storage PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_storage PUBLIC jitllm_base jitllm_platform PRIVATE jitllm_warnings)

# The deterministic fakes every build has (fake/).
add_library(jitllm_fake_providers STATIC fake/fake_device_memory.cc fake/fake_device_execution.cc
                                           fake/fake_storage.cc)
target_include_directories(jitllm_fake_providers PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_fake_providers PUBLIC jitllm_device_memory jitllm_storage PRIVATE jitllm_warnings)
