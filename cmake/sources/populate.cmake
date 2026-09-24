# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Unpacks one verified source archive with FetchContent's script-mode
# population (D-057, D-058). tools/prepare-sources runs it with the SDK's
# CMake, after fetching the archive into its cache and checking its SHA-256;
# FetchContent checks the hash again before extracting. Nothing from the
# archive is configured or built here.
#
#   cmake -DNAME=<id> -DARCHIVE=<file> -DSHA256=<hex> -DSOURCE_DIR=<new dir>
#         -DWORK_DIR=<scratch dir> -P populate.cmake
cmake_minimum_required(VERSION 4.4.3)
if(NOT CMAKE_VERSION VERSION_EQUAL "4.4.3")
  message(FATAL_ERROR "source preparation runs with the SDK's CMake 4.4.3 (D-058), not ${CMAKE_VERSION}")
endif()

foreach(var IN ITEMS NAME ARCHIVE SHA256 SOURCE_DIR WORK_DIR)
  if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
    message(FATAL_ERROR "set ${var}")
  endif()
endforeach()
foreach(var IN ITEMS ARCHIVE SOURCE_DIR WORK_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    message(FATAL_ERROR "${var} must be an absolute path")
  endif()
endforeach()
if(EXISTS "${SOURCE_DIR}")
  message(FATAL_ERROR "${SOURCE_DIR} exists; population writes only into a new directory")
endif()

# FetchContent keeps its bookkeeping under its base directory and the binary
# directory, which default to the caller's working directory (the M0
# dev-tools finding); keep them in the scratch directory.
set(FETCHCONTENT_BASE_DIR "${WORK_DIR}")
include(FetchContent)
FetchContent_Populate("${NAME}"
  URL "${ARCHIVE}"
  URL_HASH "SHA256=${SHA256}"
  TLS_VERIFY ON
  SOURCE_DIR "${SOURCE_DIR}"
  BINARY_DIR "${WORK_DIR}/build"
  QUIET)
