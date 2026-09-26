# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Measurement harnesses: built with the rest, never run by CTest. Each
# prints aggregate tables that a report under docs/experiments/ records.

# D-048's lanes: wakeup latency against polling, lane throughput, and the
# completion board's round trip (docs/experiments/task-lanes/).
add_executable(jitllm_lanes_bench lanes_bench.cc)
target_link_libraries(jitllm_lanes_bench PRIVATE jitllm_scheduler jitllm_warnings)

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
endif()
