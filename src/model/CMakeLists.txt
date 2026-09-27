# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The model layer's adapters (docs/architecture.md#layers-and-dependency-rules):
# so far state representations and their capabilities, a request's live
# state (state.h), and model contexts composed of components (context.h),
# as D-068 needs them expressed. Model layer: no vendor or kernel-module
# types, and the resource core never includes it.
add_library(jitllm_model STATIC context.cc state.cc)
target_include_directories(jitllm_model PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_model PUBLIC jitllm_base jitllm_catalog PRIVATE jitllm_warnings)
