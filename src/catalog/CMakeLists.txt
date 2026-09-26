# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The resource catalog (D-006, D-007): extents, resources, leases,
# registrations, generations and occupancy. Resource core: no vendor types.
add_library(jitllm_catalog STATIC catalog.cc)
target_include_directories(jitllm_catalog PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_catalog PUBLIC jitllm_base PRIVATE jitllm_warnings)
