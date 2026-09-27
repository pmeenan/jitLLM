# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The execution layer (docs/architecture.md#layers-and-dependency-rules): so
# far the implementation registry and plans (registry.h, D-053), which
# kernel modules declare into and never the reverse. Model layer: no vendor
# or kernel-module types.
add_library(jitllm_execution STATIC registry.cc)
target_include_directories(jitllm_execution PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_execution PUBLIC jitllm_base PRIVATE jitllm_warnings)
