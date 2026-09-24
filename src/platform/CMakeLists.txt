# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Linux services (docs/architecture.md#layers-and-dependency-rules): so far,
# small reads of /proc and /sys, and the host half of the capability probe.
add_library(jitllm_platform STATIC files.cc host_probe.cc)
target_include_directories(jitllm_platform PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_platform PUBLIC jitllm_report PRIVATE jitllm_warnings)
