# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Run by cpack: the Debian version, architecture and dependencies of the
# build being packaged, which tools/jitllm_package.py wrote (control.json).
if(NOT EXISTS "${CPACK_JITLLM_CONTROL}")
  message(FATAL_ERROR "${CPACK_JITLLM_CONTROL} is missing: package with `mise run package`")
endif()
file(READ "${CPACK_JITLLM_CONTROL}" _jitllm_control)
string(JSON _jitllm_version GET "${_jitllm_control}" version)
string(JSON CPACK_DEBIAN_PACKAGE_ARCHITECTURE GET "${_jitllm_control}" architecture)
string(JSON CPACK_DEBIAN_PACKAGE_DEPENDS GET "${_jitllm_control}" depends)
# "<upstream>-<revision>"
string(REGEX MATCH "^(.+)-([^-]+)$" _jitllm_match "${_jitllm_version}")
set(CPACK_DEBIAN_PACKAGE_VERSION "${CMAKE_MATCH_1}")
set(CPACK_DEBIAN_PACKAGE_RELEASE "${CMAKE_MATCH_2}")
set(CPACK_PACKAGE_VERSION "${CMAKE_MATCH_1}")
