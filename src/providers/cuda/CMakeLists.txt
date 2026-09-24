# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The CUDA provider (D-006, D-026), so far its device probe. Only this
# module includes CUDA's headers. It uses the driver API from ordinary C++
# units and links the NVIDIA driver's libcuda.so.1, which every host running
# a CUDA build must have (D-072).

# The driver, linked through NVIDIA's stub in the SDK: binaries record
# libcuda.so.1 as needed and the driver supplies it at run time.
set(stub "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/lib/stubs/libcuda.so")
if(NOT EXISTS "${stub}")
  message(FATAL_ERROR "The SDK lacks the CUDA driver stub ${stub}; run `mise run setup`.")
endif()
add_library(jitllm_cuda_driver SHARED IMPORTED)
set_target_properties(jitllm_cuda_driver PROPERTIES IMPORTED_LOCATION "${stub}"
                                                    IMPORTED_SONAME libcuda.so.1)
# Tests that need no GPU load the stub on hosts without the driver (qemu-user,
# the reference container); the test presets put this directory on the
# library path. The stub's cuInit() fails, so doctor reports it.
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cuda-stub")
file(COPY_FILE "${stub}" "${PROJECT_BINARY_DIR}/cuda-stub/libcuda.so.1" ONLY_IF_DIFFERENT)

# What the probe judges, apart from the driver (cuda_facts.h).
add_library(jitllm_cuda_facts STATIC cuda_report.cc)
target_include_directories(jitllm_cuda_facts PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_cuda_facts PUBLIC jitllm_platform jitllm_report PRIVATE jitllm_warnings)

add_library(jitllm_cuda_probe STATIC cuda_probe.cc)
target_include_directories(jitllm_cuda_probe SYSTEM PRIVATE
                           "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
list(JOIN CMAKE_CUDA_ARCHITECTURES "," architectures)
target_compile_definitions(jitllm_cuda_probe PRIVATE "JITLLM_CUDA_ARCHITECTURES=\"${architectures}\"")
target_link_libraries(jitllm_cuda_probe PUBLIC jitllm_cuda_facts jitllm_cuda_driver
                      PRIVATE jitllm_platform jitllm_warnings)
