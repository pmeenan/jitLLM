# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Unit tests of jitLLM's modules, one GoogleTest executable per module
# (docs/architecture.md#testing). Every test carries the `unit` label.

add_executable(cli_test cli_test.cc)
target_link_libraries(cli_test PRIVATE jitllm_cli jitllm_surface_versions)
target_compile_definitions(cli_test PRIVATE "JITLLM_TEST_PROJECT_VERSION=\"${PROJECT_VERSION}\"")
jitllm_gtest(cli_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(base_test base_test.cc)
target_link_libraries(base_test PRIVATE jitllm_report)
jitllm_gtest(base_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(platform_test platform_test.cc)
target_link_libraries(platform_test PRIVATE jitllm_platform)
jitllm_gtest(platform_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The CUDA provider, apart from its device queries, which need a GPU
# (smoke.doctor in tests/smoke).
if(JITLLM_CUDA)
  add_executable(cuda_test cuda_test.cc)
  target_link_libraries(cuda_test PRIVATE jitllm_cuda_probe)
  jitllm_gtest(cuda_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The node configuration. Tests that need scratch trees (here and in
# platform_test) put them in the build tree, whose parents no other user
# shares, and remote runs map it (JITLLM_TEST_SCRATCH).
add_executable(config_test config_test.cc)
target_link_libraries(config_test PRIVATE jitllm_config jitllm_platform)
jitllm_gtest(config_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The runtime's startup steps and the process services it relies on.
add_executable(runtime_test runtime_test.cc)
target_link_libraries(runtime_test PRIVATE jitllm_runtime jitllm_platform)
jitllm_gtest(runtime_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The resource core: the catalog and the memory manager's policy, on no
# vendor SDK (D-026).
add_executable(catalog_test catalog_test.cc)
target_link_libraries(catalog_test PRIVATE jitllm_catalog jitllm_base)
jitllm_gtest(catalog_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(memory_test memory_test.cc)
target_link_libraries(memory_test PRIVATE jitllm_memory jitllm_catalog)
jitllm_gtest(memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(admission_test admission_test.cc)
target_link_libraries(admission_test PRIVATE jitllm_scheduler)
jitllm_gtest(admission_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-048's lanes, completion board and task trees, with real threads.
add_executable(lanes_test lanes_test.cc)
target_link_libraries(lanes_test PRIVATE jitllm_scheduler)
jitllm_gtest(lanes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Storage and whole reads, on the fake and on io_uring with a real file.
add_executable(storage_test storage_test.cc)
target_link_libraries(storage_test PRIVATE jitllm_fake_providers jitllm_storage jitllm_platform)
jitllm_gtest(storage_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The providers' shared rules, on the deterministic fakes.
add_executable(device_memory_test device_memory_test.cc)
target_link_libraries(device_memory_test PRIVATE jitllm_fake_providers)
jitllm_gtest(device_memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(device_execution_test device_execution_test.cc)
target_link_libraries(device_execution_test PRIVATE jitllm_fake_providers)
jitllm_gtest(device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The CUDA device-memory provider against a real device: label `gpu`, so
# only on a Spark.
if(JITLLM_CUDA)
  add_executable(cuda_device_execution_test cuda_device_execution_test.cc)
  target_include_directories(cuda_device_execution_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_device_execution_test PRIVATE jitllm_cuda_device_memory)
  jitllm_gtest(cuda_device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  add_executable(cuda_device_memory_test cuda_device_memory_test.cc)
  target_link_libraries(cuda_device_memory_test PRIVATE jitllm_cuda_device_memory jitllm_storage)
  jitllm_gtest(cuda_device_memory_test TEST_PREFIX unit. PROPERTIES LABELS gpu
               ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
endif()
