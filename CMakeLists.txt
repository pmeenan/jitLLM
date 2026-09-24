# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# jitLLM (D-010, D-011, D-058, D-059, D-060, D-066). Configure through a
# preset in CMakePresets.json, which selects a toolchain file in
# cmake/toolchains/; `mise run build` does that with the pinned SDK.
cmake_minimum_required(VERSION 4.4.3)
if(NOT CMAKE_VERSION VERSION_EQUAL "4.4.3")
  message(FATAL_ERROR "jitLLM builds with the SDK's CMake 4.4.3 (D-058), not ${CMAKE_VERSION}. "
                      "Use `mise run build`, or run $JITLLM_SDK/bin/cmake.")
endif()

project(jitllm LANGUAGES CXX)

if(NOT JITLLM_PROFILE)
  message(FATAL_ERROR "Configure with a preset from CMakePresets.json (`mise run build` does); "
                      "its toolchain file in cmake/toolchains/ selects the SDK's compilers.")
endif()
# A change to the SDK's inputs reruns configure, which then checks the SDK again.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${JITLLM_SDK_INPUT_FILES})
message(STATUS "jitLLM profile ${JITLLM_PROFILE}, SDK ${JITLLM_SDK_IDENTITY}, CUDA ${JITLLM_CUDA}")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# jitLLM uses no C++ modules; skip the dependency scan CMake adds for C++20
# and later (D-059).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Executables link their runtimes statically and need no RPATH (D-060), and
# nothing builds a shared library (D-064).
set(CMAKE_SKIP_RPATH ON)
set(BUILD_SHARED_LIBS OFF)

# No exceptions anywhere in the build, tests and third-party code included
# (D-066).
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>")

if(JITLLM_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 23)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-exceptions>")
endif()

# A build directory keeps the compilers and initial flags of its first
# configure, whatever the toolchain file now says. If JITLLM_SDK names another
# SDK than that one, the two would mix: require a fresh configure.
set(_jitllm_compilers CXX "${JITLLM_SDK}/bin/clang++")
if(JITLLM_CUDA)
  list(APPEND _jitllm_compilers CUDA "${JITLLM_SDK}/cuda/bin/nvcc")
endif()
while(_jitllm_compilers)
  list(POP_FRONT _jitllm_compilers _jitllm_lang _jitllm_want)
  cmake_path(NORMAL_PATH _jitllm_want)
  cmake_path(SET _jitllm_have NORMALIZE "${CMAKE_${_jitllm_lang}_COMPILER}")
  if(NOT _jitllm_have STREQUAL _jitllm_want)
    message(FATAL_ERROR
      "${PROJECT_BINARY_DIR} was first configured with ${_jitllm_have}, not the SDK at ${JITLLM_SDK}. "
      "Configure it afresh: `mise run build -- --fresh <preset>`, or add --fresh to `cmake --preset`.")
  endif()
endwhile()

# D-059's warnings, for jitLLM's own targets only (link this to each one);
# third-party code keeps its own warning settings. NVCC's host pass sees its
# generated line markers and CUDA's own headers as ordinary source, so it
# goes without -Wold-style-cast and allows GNU line markers.
add_library(jitllm_warnings INTERFACE)
target_compile_options(jitllm_warnings INTERFACE
  "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Wshadow;-Wconversion;-Wsign-conversion;-Wnon-virtual-dtor;-Wold-style-cast;-Wimplicit-fallthrough;-Werror>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings;-Xcompiler=-Wall,-Wextra,-Wpedantic,-Wno-gnu-line-marker,-Wshadow,-Wconversion,-Wsign-conversion,-Wnon-virtual-dtor,-Wimplicit-fallthrough,-Werror>")

# Marks a test executable: in a cross build, CTest runs it under qemu-user,
# or on a Spark over SSH once deployed (tools/run-target, `mise run test`).
# Only tests get this runner; a build-time tool built for the target must
# never run under emulation (D-012).
function(jitllm_test_executable target)
  if(CMAKE_CROSSCOMPILING)
    set_property(TARGET ${target} PROPERTY CROSSCOMPILING_EMULATOR
      "${PROJECT_SOURCE_DIR}/tools/run-target;--build-dir;${PROJECT_BINARY_DIR};--sysroot;${CMAKE_SYSROOT};--qemu;qemu-${CMAKE_SYSTEM_PROCESSOR}-static;--")
  endif()
endfunction()

enable_testing()
add_subdirectory(tests/toolchain)
