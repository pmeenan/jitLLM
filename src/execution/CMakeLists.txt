# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The execution layer (docs/architecture.md#layers-and-dependency-rules): so
# far the implementation registry and plans (registry.h, D-053), which
# kernel modules declare into and never the reverse, and phase kinds,
# decoding modes and request programs with their envelopes (program.h,
# D-050, D-068). Model layer: no vendor or kernel-module types.
add_library(jitllm_execution STATIC program.cc registry.cc)
target_include_directories(jitllm_execution PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_execution PUBLIC jitllm_base jitllm_catalog jitllm_memory jitllm_model
                                       PRIVATE jitllm_warnings)

# Greedy and seeded sampling on the host (sampling.h), for M3's decode
# loop and speculation checks: base only.
add_library(jitllm_sampling STATIC sampling.cc)
target_include_directories(jitllm_sampling PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_sampling PRIVATE jitllm_warnings)
