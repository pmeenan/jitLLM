# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The product version (D-062), from project(VERSION) and Git
# (cmake/version/derive.cmake has the rules).
#
# Including this derives it once at configure, which stops on a version the
# rules refuse and sets JITLLM_BUILD_VERSION, JITLLM_BUILD_DEBIAN,
# JITLLM_BUILD_COMMIT, JITLLM_BUILD_DIRTY, JITLLM_BUILD_ORIGIN and
# JITLLM_BUILD_GIT, and JITLLM_BUILD_JSON for the build receipt.
#
#   jitllm_build_info_library(<target>)  after jitllm_sources_add(): a static
#       library defining base::GetBuildInfo() (src/base/build_info.h). Every
#       build derives the version again (cmake/version/stamp.cmake) and
#       updates the library's generated source and the receipt, so neither
#       goes stale when a commit or an edit needs no configure.

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/version/derive.cmake")

set(_JITLLM_VERSION_SCRIPTS "${CMAKE_CURRENT_LIST_DIR}/version")

jitllm_version_derive(JITLLM_BUILD SOURCE_DIR "${PROJECT_SOURCE_DIR}" PROJECT_VERSION "${PROJECT_VERSION}")
jitllm_version_json(JITLLM_BUILD_JSON JITLLM_BUILD)
message(STATUS "jitLLM version ${JITLLM_BUILD_VERSION}")

function(jitllm_build_info_library target)
  get_property(license_profile GLOBAL PROPERTY JITLLM_SOURCES_PROFILE)
  if(NOT license_profile)
    message(FATAL_ERROR "call jitllm_sources_add() before jitllm_build_info_library()")
  endif()
  set(generated "${CMAKE_CURRENT_BINARY_DIR}/${target}.cc")
  add_custom_target(${target}_stamp ALL
    COMMAND "${CMAKE_COMMAND}" "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DPROJECT_VERSION=${PROJECT_VERSION}"
            "-DGIT=${JITLLM_BUILD_GIT}" "-DOUTPUT=${generated}"
            "-DRECEIPT=${PROJECT_BINARY_DIR}/jitllm-receipt.json" "-DLICENSE_PROFILE=${license_profile}"
            "-DSDK=${JITLLM_SDK_IDENTITY}" "-DTARGET=${JITLLM_TARGET_TRIPLE}"
            -P "${_JITLLM_VERSION_SCRIPTS}/stamp.cmake"
    BYPRODUCTS "${generated}"
    COMMENT "Deriving the jitLLM version (D-062)"
    VERBATIM)
  add_library(${target} STATIC "${generated}")
  add_dependencies(${target} ${target}_stamp)
  target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_link_libraries(${target} PRIVATE jitllm_warnings)
endfunction()
