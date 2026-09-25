# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The node's configuration (D-063, D-073): the strict TOML document and its
# storage roles. toml++ stays inside node_config.cc, in its TOML 1.0.0 mode
# without exceptions (D-066) or formatters.
add_library(jitllm_config STATIC node_config.cc storage_roles.cc)
target_include_directories(jitllm_config PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(jitllm_config PRIVATE TOML_EXCEPTIONS=0 TOML_ENABLE_FORMATTERS=0
                                                 TOML_ENABLE_UNRELEASED_FEATURES=0 TOML_HEADER_ONLY=1)
target_link_libraries(jitllm_config PUBLIC jitllm_report jitllm_surface_versions PRIVATE jitllm_platform tomlplusplus::tomlplusplus jitllm_warnings)
