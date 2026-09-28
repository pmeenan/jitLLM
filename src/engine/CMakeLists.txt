# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The engine (docs/runtime-serving.md): the paged node that composes the
# providers, the resource core and the lanes on one GPU (paged_node.h), and
# each M3 model's runner on it: its weights' places and sources, its state,
# its chunk plans and decode graphs, and its device jobs (dsv4_runner.h,
# qwen38_runner.h, qwen_image_runner.h). Above the kernels in the layers;
# like the kernel units, it may use the CUDA runtime, so CUDA builds only.
# jitllm-runtime links it, and so do the paged harnesses and their tests.
if(JITLLM_CUDA)
  add_library(jitllm_engine STATIC paged_node.cc paged_weights.cc paged_weights.cu ple_rows.cc
                                   ple_rows.cu dsv4_plan.cc dsv4_runner.cc qwen38_plan.cc
                                   qwen38_runner.cc qwen_image_runner.cc)
  target_include_directories(jitllm_engine PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(jitllm_engine SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_engine PUBLIC jitllm_scheduler jitllm_memory jitllm_storage
                                             jitllm_cuda_device_memory jitllm_ggml_kernels
                                             jitllm_image_kernels jitllm_artifact jitllm_model
                                             jitllm_tokenizer jitllm_chat
                                      PRIVATE jitllm_base jitllm_platform jitllm_warnings)
  # The fill and n-gram gather kernels: sm_121a, as the GGML module's units
  # build, and the profile's discrete GPUs (D-082).
  set_target_properties(jitllm_engine PROPERTIES
                        CUDA_ARCHITECTURES "121a-real;${JITLLM_CUDA_DISCRETE_SASS}")
endif()
