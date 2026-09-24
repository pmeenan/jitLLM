# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# What this build is (build_info.h): its definition is generated on every
# build from Git and the build configuration (D-062, cmake/JitllmVersion.cmake).
jitllm_build_info_library(jitllm_build_info)

# The versions of jitLLM's own public surfaces (surface_versions.h).
add_library(jitllm_surface_versions INTERFACE)
target_include_directories(jitllm_surface_versions INTERFACE "${PROJECT_SOURCE_DIR}/src")
