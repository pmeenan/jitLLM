# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The `jitllm` command (/usr/bin/jitllm once packaged, D-063).
add_library(jitllm_cli STATIC cli.cc doctor.cc)
target_include_directories(jitllm_cli PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_cli PUBLIC jitllm_build_info jitllm_report
                      PRIVATE jitllm_platform jitllm_device_probe jitllm_warnings)

add_executable(jitllm main.cc)
target_link_libraries(jitllm PRIVATE jitllm_cli jitllm_warnings)
