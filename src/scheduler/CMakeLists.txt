# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The scheduler (D-048, D-050, D-069): admission and the execution slot with
# the switching policy; the task table and ready queue; the completion
# board; and lanes (lane.h, a template). Resource core: no vendor types.
add_library(jitllm_scheduler STATIC admission.cc completions.cc tasks.cc)
target_include_directories(jitllm_scheduler PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_scheduler PUBLIC jitllm_base jitllm_memory PRIVATE jitllm_warnings)
