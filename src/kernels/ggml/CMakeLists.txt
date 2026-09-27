# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# GGML-derived implementations (D-053; docs/backend-proof.md), over the
# locked GGML subset (third_party/sources.lock.json, component `ggml`).
#
#   jitllm_ggml_tensors  GGML tensor descriptors over jitLLM memory
#                        (tensors.h) and the implementations' operand
#                        checks (validate.h). Every profile.
#   jitllm_ggml_kernels  the K-C launch context (launch.h), the operation
#                        implementations (ops.h) and jitLLM's definitions of
#                        the ggml-cuda.cu symbols GGML's launchers use
#                        (ggml_support.cu). CUDA builds only.
add_library(jitllm_ggml_tensors STATIC tensors.cc validate.cc)
target_include_directories(jitllm_ggml_tensors PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_ggml_tensors PUBLIC jitllm_base jitllm_ggml_base PRIVATE jitllm_warnings)

if(JITLLM_CUDA)
  add_library(jitllm_ggml_kernels STATIC ggml_support.cu launch.cu ops.cu)
  target_include_directories(jitllm_ggml_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(jitllm_ggml_kernels
                        PUBLIC jitllm_ggml_tensors jitllm_ggml_cuda jitllm_device_memory
                        # the NVIDIA driver, for ggml_support.cu's device queries
                        PRIVATE jitllm_cuda_device_memory jitllm_warnings)
  set_target_properties(jitllm_ggml_kernels PROPERTIES CUDA_ARCHITECTURES 121a-real)
endif()
