# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Linux services (docs/architecture.md#layers-and-dependency-rules): small
# reads of /proc and /sys, the host half of the capability probe, the trust
# walk that checks who could replace a path, and the storage roles'
# filesystem and direct-I/O probe, and process services: the crash policy,
# lock files, the service manager's readiness protocol, and job processes
# in delegated cgroups with their confinement; and a minimal io_uring ring.
add_library(jitllm_platform STATIC files.cc host_probe.cc path_trust.cc direct_io.cc crash_policy.cc
                                   lock_file.cc sd_notify.cc job.cc
                                   confine.cc io_uring.cc)
target_include_directories(jitllm_platform PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_platform PUBLIC jitllm_report PRIVATE jitllm_warnings)
