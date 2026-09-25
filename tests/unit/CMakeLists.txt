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
