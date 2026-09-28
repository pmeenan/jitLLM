# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# jitLLM's own kernels for the Qwen-Image-2.1 pipeline's BF16 operations
# (ops.h) and its BF16 products on cuBLAS (gemm.h). CUDA builds only.
if(JITLLM_CUDA)
  add_library(jitllm_image_kernels STATIC ops.cu flash_attention.cu gemm.cc)
  target_include_directories(jitllm_image_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(jitllm_image_kernels SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_image_kernels PUBLIC jitllm_cublas jitllm_cublaslt
                                             PRIVATE jitllm_warnings)
  # sm_121a, as the GGML module's units build, and the profile's discrete
  # GPUs (D-082).
  set_target_properties(jitllm_image_kernels PROPERTIES
                        CUDA_ARCHITECTURES "121a-real;${JITLLM_CUDA_DISCRETE_SASS}")
endif()
