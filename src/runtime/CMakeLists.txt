# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The node runtime process (/usr/libexec/jitllm/jitllm-runtime once
# packaged, D-063): startup, readiness and stop (runtime.h).
add_library(jitllm_runtime STATIC runtime.cc)
target_include_directories(jitllm_runtime PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_runtime PUBLIC jitllm_config jitllm_platform
                      PRIVATE jitllm_build_info jitllm_report jitllm_device_probe jitllm_warnings)

add_executable(jitllm-runtime main.cc)
target_link_libraries(jitllm-runtime PRIVATE jitllm_runtime jitllm_platform jitllm_warnings)
