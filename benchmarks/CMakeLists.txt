# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Measurement harnesses: built with the rest, never run by CTest. Each
# prints aggregate tables that a report under docs/experiments/ records.

# D-048's lanes: wakeup latency against polling, lane throughput, and the
# completion board's round trip (docs/experiments/task-lanes/).
add_executable(jitllm_lanes_bench lanes_bench.cc)
target_link_libraries(jitllm_lanes_bench PRIVATE jitllm_scheduler jitllm_warnings)

# The retained-backing comparison's deterministic replay of the swap trace
# on the fake provider (docs/experiments/retained-backing/).
add_subdirectory(retained_backing)

# Storage queue depth and request size, direct into host VMM through the
# providers (docs/experiments/storage-queue/). Needs a CUDA device.
if(JITLLM_CUDA)
  add_executable(jitllm_storage_bench storage_bench.cc)
  target_link_libraries(jitllm_storage_bench PRIVATE jitllm_cuda_device_memory jitllm_storage jitllm_warnings)

  # Which OS counters include VMM backing (docs/experiments/vmm-counters/).
  add_executable(jitllm_vmm_counters vmm_counters.cc)
  target_include_directories(jitllm_vmm_counters SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_vmm_counters PRIVATE jitllm_cuda_device_memory jitllm_warnings)

  # Host time per launch of RMSNorm-mul, for each implementation a plan can
  # select, through the launch context and around it (BP-F4 in backend-proof
  # P1; docs/experiments/launch-overhead/). Needs a GB10.
  add_executable(jitllm_launch_bench launch_bench.cc)
  target_include_directories(jitllm_launch_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_launch_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                    jitllm_warnings)
  # It links cuBLAS dynamically (D-076), found as the GPU tests find it.
  file(RELATIVE_PATH runtime "${CMAKE_CURRENT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/cublas")
  target_link_options(jitllm_launch_bench PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/${runtime}")

  # BP-F1: GGML kernels with their operands in host VMM or cudaMalloc
  # memory, one timing block per process (docs/experiments/backend-proof-p1/).
  # Finds the SDK's cuBLAS in build/<preset>/cublas, as the GPU tests do.
  add_executable(jitllm_ggml_vmm_bench ggml_vmm_bench.cc)
  target_include_directories(jitllm_ggml_vmm_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_ggml_vmm_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                      jitllm_warnings)
  target_link_options(jitllm_ggml_vmm_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Backend-proof P2, rung 3: the FP16 fixture from its v0 artifact on
  # cudaMalloc memory, recorded for the FP16 gate's plan comparison or read
  # for the allocation census (docs/experiments/backend-proof-p2/). Links the
  # launch recorder; finds the SDK's cuBLAS in build/<preset>/cublas.
  add_executable(jitllm_fp16_exec fp16_exec.cc)
  target_include_directories(jitllm_fp16_exec SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_fp16_exec PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                 jitllm_artifact jitllm_model jitllm_launch_recorder
                                                 jitllm_warnings)
  target_link_options(jitllm_fp16_exec PRIVATE "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")

  # Why host VMM slowed BP-F1's products, and what the alternatives cost:
  # microkernels over each kind of memory, copies between them, and GGML
  # products with each buffer group placed separately
  # (docs/experiments/host-vmm-diagnosis/). Diagnostic, not a gate.
  add_executable(jitllm_vmm_diag_bench vmm_diag_bench.cc vmm_diag_kernels.cu)
  target_include_directories(jitllm_vmm_diag_bench SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_vmm_diag_bench PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                                      jitllm_storage jitllm_warnings)
  set_target_properties(jitllm_vmm_diag_bench PROPERTIES CUDA_ARCHITECTURES 121a-real)
  target_link_options(jitllm_vmm_diag_bench PRIVATE
                      "LINKER:--enable-new-dtags,-rpath,$ORIGIN/../cublas")
endif()
