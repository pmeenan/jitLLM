# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The shared part of the profile toolchain files beside it (D-011, D-032,
# D-049, D-060, D-070). A profile sets these, then includes this file:
#
#   JITLLM_PROFILE        the profile's name, for messages and the project
#   JITLLM_BUILD_ARCH     the build host architecture the profile runs on
#   JITLLM_TARGET_TRIPLE  the target triple Clang compiles for
#   JITLLM_TARGET_MARCH   the explicit target CPU baseline (never `native`)
#   JITLLM_LINK_FLAGS     how the SDK's Clang links for this target
#   JITLLM_SYSROOT        cross builds only: the sysroot, relative to the SDK
#
# Compilers and build tools come from the SDK that `mise run setup` provisions;
# Spark-native also uses the host's declared GNU linker. JITLLM_SDK names the
# SDK root, as a cache entry or environment variable (`mise run build` passes
# it), and the SDK's receipt must match this checkout's toolchain inputs. JITLLM_CUDA
# (default ON) selects CUDA; with it OFF, nothing here refers to the SDK's
# CUDA components (the CPU-only profile, D-026).

cmake_path(GET CMAKE_CURRENT_LIST_DIR PARENT_PATH _jitllm_cmake_dir)
cmake_path(GET _jitllm_cmake_dir PARENT_PATH JITLLM_SOURCE_ROOT)

# CMake re-reads the toolchain file in every try_compile project; these carry
# the settings there.
list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES JITLLM_SDK JITLLM_CUDA JITLLM_TOOLCHAIN_DIR)

# The SDK -----------------------------------------------------------------------

if(NOT JITLLM_SDK)
  set(JITLLM_SDK "$ENV{JITLLM_SDK}")
endif()
if(NOT JITLLM_SDK)
  message(FATAL_ERROR
    "JITLLM_SDK is not set. `mise run build` sets it; to configure by hand, run\n"
    "  export JITLLM_SDK=$(tools/setup-toolchain --print-root)\n"
    "  $JITLLM_SDK/bin/cmake --preset <name>")
endif()
set(JITLLM_SDK "${JITLLM_SDK}" CACHE PATH "Root of the pinned jitLLM SDK (tools/setup-toolchain --print-root)")

set(_jitllm_receipt_file "${JITLLM_SDK}/sdk.json")
if(NOT EXISTS "${_jitllm_receipt_file}")
  message(FATAL_ERROR "No SDK at ${JITLLM_SDK} (no sdk.json). Run `mise run setup`.")
endif()
file(READ "${_jitllm_receipt_file}" _jitllm_receipt)
string(JSON _jitllm_schema ERROR_VARIABLE _jitllm_error GET "${_jitllm_receipt}" schema)
string(JSON JITLLM_SDK_IDENTITY ERROR_VARIABLE _jitllm_error GET "${_jitllm_receipt}" identity)
string(JSON _jitllm_sdk_arch ERROR_VARIABLE _jitllm_error GET "${_jitllm_receipt}" host_arch)
string(JSON _jitllm_inputs ERROR_VARIABLE _jitllm_error LENGTH "${_jitllm_receipt}" inputs)
if(_jitllm_error OR NOT _jitllm_schema EQUAL 1 OR NOT _jitllm_inputs GREATER 0)
  message(FATAL_ERROR "${_jitllm_receipt_file} is not a valid SDK receipt; remove the SDK and run `mise run setup`.")
endif()
if(NOT CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL JITLLM_BUILD_ARCH)
  message(FATAL_ERROR
    "The ${JITLLM_PROFILE} profile builds on ${JITLLM_BUILD_ARCH} hosts; this host is "
    "${CMAKE_HOST_SYSTEM_PROCESSOR}. Choose another preset (CMakePresets.json).")
endif()
if(NOT _jitllm_sdk_arch STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
  message(FATAL_ERROR "${JITLLM_SDK} is an SDK for ${_jitllm_sdk_arch} hosts, not ${CMAKE_HOST_SYSTEM_PROCESSOR}.")
endif()

# The receipt lists the files whose bytes determined the SDK. Each must match
# this checkout, or the SDK was built from other pins.
set(JITLLM_SDK_INPUT_FILES "")
math(EXPR _jitllm_last "${_jitllm_inputs} - 1")
foreach(_jitllm_i RANGE ${_jitllm_last})
  string(JSON _jitllm_name MEMBER "${_jitllm_receipt}" inputs ${_jitllm_i})
  string(JSON _jitllm_want GET "${_jitllm_receipt}" inputs "${_jitllm_name}")
  set(_jitllm_file "${JITLLM_SOURCE_ROOT}/${_jitllm_name}")
  if(EXISTS "${_jitllm_file}")
    file(SHA256 "${_jitllm_file}" _jitllm_have)
  else()
    set(_jitllm_have "missing")
  endif()
  if(NOT _jitllm_have STREQUAL _jitllm_want)
    message(FATAL_ERROR
      "The SDK at ${JITLLM_SDK} was set up from a different ${_jitllm_name} than this "
      "checkout has. Run `mise run setup`, then configure afresh with the new SDK: "
      "`mise run build -- --fresh <preset>`.")
  endif()
  list(APPEND JITLLM_SDK_INPUT_FILES "${_jitllm_file}")
endforeach()

# Compilers and tools -----------------------------------------------------------

set(CMAKE_C_COMPILER "${JITLLM_SDK}/bin/clang")
set(CMAKE_CXX_COMPILER "${JITLLM_SDK}/bin/clang++")
set(CMAKE_C_COMPILER_TARGET "${JITLLM_TARGET_TRIPLE}")
set(CMAKE_CXX_COMPILER_TARGET "${JITLLM_TARGET_TRIPLE}")
set(CMAKE_MAKE_PROGRAM "${JITLLM_SDK}/bin/ninja" CACHE FILEPATH "The SDK's Ninja (D-059)" FORCE)
foreach(_jitllm_tool IN ITEMS AR RANLIB NM OBJCOPY OBJDUMP READELF STRIP)
  string(TOLOWER "${_jitllm_tool}" _jitllm_name)
  set(CMAKE_${_jitllm_tool} "${JITLLM_SDK}/bin/llvm-${_jitllm_name}" CACHE FILEPATH "" FORCE)
endforeach()

# The GCC 16.2 C++ runtime (D-060): its headers for every compile, and its
# static libstdc++ and libgcc for every link. A cross build finds it inside the
# sysroot, the only place Clang searches a GCC install's libraries.
if(DEFINED JITLLM_SYSROOT)
  set(CMAKE_SYSROOT "${JITLLM_SDK}/${JITLLM_SYSROOT}")
  set(_jitllm_gcc_root "${CMAKE_SYSROOT}/opt/gcc")
  set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
  set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
else()
  set(_jitllm_gcc_root "${JITLLM_SDK}/gcc/${JITLLM_TARGET_TRIPLE}")
endif()
set(JITLLM_GCC_INSTALL_DIR "${_jitllm_gcc_root}/lib/gcc/${JITLLM_TARGET_TRIPLE}/16")
foreach(_jitllm_path IN ITEMS "${CMAKE_CXX_COMPILER}" "${CMAKE_MAKE_PROGRAM}" "${JITLLM_GCC_INSTALL_DIR}/libgcc.a")
  if(NOT EXISTS "${_jitllm_path}")
    message(FATAL_ERROR "${_jitllm_path} is missing from the SDK; remove ${JITLLM_SDK} and run `mise run setup`.")
  endif()
endforeach()

# Quotes a value for a POSIX shell, including paths in compiler flag strings.
function(_jitllm_sh_quote out value)
  string(REPLACE "'" "'\\''" value "${value}")
  set(${out} "'${value}'" PARENT_SCOPE)
endfunction()

_jitllm_sh_quote(_jitllm_gcc_flag "--gcc-install-dir=${JITLLM_GCC_INSTALL_DIR}")
set(_jitllm_target_flags "-march=${JITLLM_TARGET_MARCH} ${_jitllm_gcc_flag}")
set(CMAKE_C_FLAGS_INIT "${_jitllm_target_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_jitllm_target_flags}")
# The SDK's GCC runtime has only static archives, so every link, CMake's own
# checks included, takes the runtime statically (D-060).
list(APPEND JITLLM_LINK_FLAGS -static-libstdc++ -static-libgcc)
list(JOIN JITLLM_LINK_FLAGS " " CMAKE_EXE_LINKER_FLAGS_INIT)

# CUDA ----------------------------------------------------------------------------

if(NOT DEFINED JITLLM_CUDA)
  set(JITLLM_CUDA ON CACHE BOOL "Build CUDA code for sm_121. OFF is the CPU-only profile, which uses no CUDA SDK (D-026).")
endif()
if(NOT JITLLM_TOOLCHAIN_DIR)
  set(JITLLM_TOOLCHAIN_DIR "${CMAKE_BINARY_DIR}/toolchain")
endif()

if(JITLLM_CUDA)
  if(JITLLM_TARGET_TRIPLE STREQUAL "x86_64-linux-gnu")
    set(JITLLM_CUDA_TARGET_DIR x86_64-linux)
  else()
    set(JITLLM_CUDA_TARGET_DIR sbsa-linux)
  endif()
  set(CMAKE_CUDA_COMPILER "${JITLLM_SDK}/cuda/bin/nvcc")
  if(NOT EXISTS "${CMAKE_CUDA_COMPILER}" OR NOT EXISTS "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}")
    message(FATAL_ERROR "The SDK lacks CUDA for ${JITLLM_CUDA_TARGET_DIR}; remove ${JITLLM_SDK} and run `mise run setup`.")
  endif()
  set(CMAKE_CUDA_FLAGS_INIT "--target-directory ${JITLLM_CUDA_TARGET_DIR}")
  # GB10 only, as SASS: no PTX to JIT (D-011, D-032).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 121-real)
  endif()

  # NVCC takes a host compiler path but no arguments for it, and runs it for
  # preprocessing, compiling and CMake's link checks. This wrapper adds the
  # target flags, and the link flags only when linking, so that compile-only
  # passes see no unused arguments (the validated D-032/D-060 arrangement).
  set(_jitllm_compile "")
  foreach(_jitllm_arg IN ITEMS "${CMAKE_CXX_COMPILER}" "--target=${JITLLM_TARGET_TRIPLE}"
                                "-march=${JITLLM_TARGET_MARCH}" "--gcc-install-dir=${JITLLM_GCC_INSTALL_DIR}")
    _jitllm_sh_quote(_jitllm_quoted "${_jitllm_arg}")
    string(APPEND _jitllm_compile " ${_jitllm_quoted}")
  endforeach()
  if(CMAKE_SYSROOT)
    _jitllm_sh_quote(_jitllm_quoted "--sysroot=${CMAKE_SYSROOT}")
    string(APPEND _jitllm_compile " ${_jitllm_quoted}")
  endif()
  set(_jitllm_link "")
  foreach(_jitllm_arg IN LISTS JITLLM_LINK_FLAGS)
    _jitllm_sh_quote(_jitllm_quoted "${_jitllm_arg}")
    string(APPEND _jitllm_link " ${_jitllm_quoted}")
  endforeach()
  set(CMAKE_CUDA_HOST_COMPILER "${JITLLM_TOOLCHAIN_DIR}/cuda-host-clang++")
  file(CONFIGURE OUTPUT "${CMAKE_CUDA_HOST_COMPILER}" @ONLY CONTENT [=[
#!/bin/sh
# Generated by cmake/toolchains/sdk.cmake for the @JITLLM_PROFILE@ profile:
# NVCC's host compiler, the SDK's Clang with the target flags.
link=yes
for arg in "$@"; do
  case "$arg" in -c|-E|-S|-M|-MM|-fsyntax-only) link= ;; esac
done
if [ -n "$link" ]; then
  set --@_jitllm_link@ "$@"
fi
exec@_jitllm_compile@ "$@"
]=])
  file(CHMOD "${CMAKE_CUDA_HOST_COMPILER}" FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
       GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endif()
