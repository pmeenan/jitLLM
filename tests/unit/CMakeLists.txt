# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Unit tests of jitLLM's modules, one GoogleTest executable per module
# (docs/architecture.md#testing). Every test carries the `unit` label.

add_executable(cli_test cli_test.cc)
target_link_libraries(cli_test PRIVATE jitllm_cli jitllm_surface_versions)
target_compile_definitions(cli_test PRIVATE "JITLLM_TEST_PROJECT_VERSION=\"${PROJECT_VERSION}\"")
jitllm_gtest(cli_test TEST_PREFIX unit. PROPERTIES LABELS unit)
