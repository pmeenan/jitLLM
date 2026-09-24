# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The toolchain contract, tested in what each profile produces (D-011, D-026,
# D-032, D-059, D-060, D-066). Every test carries the `toolchain` label; `gpu`
# marks those that need a GB10, which run only on a Spark.

string(REGEX MATCH "^[^-]+" target_arch "${JITLLM_TARGET_TRIPLE}")
# Spark binaries must run on the Sparks' glibc, which the cross build's
# sysroot pins (D-070). Native x86-64 binaries link the build host's own
# glibc, so they get no ceiling.
if(target_arch STREQUAL "aarch64")
  set(glibc_max 2.39)
else()
  set(glibc_max "")
endif()

function(jitllm_toolchain_test target)
  target_link_libraries(${target} PRIVATE jitllm_warnings)
  jitllm_test_executable(${target})
  add_test(NAME toolchain.${target} COMMAND ${target})
  add_test(NAME toolchain.${target}.binary
           COMMAND ${CMAKE_COMMAND} -DREADELF=${CMAKE_READELF} -DARCH=${target_arch}
                   -DGLIBC_MAX=${glibc_max} -DBINARY=$<TARGET_FILE:${target}>
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/check_binary.cmake)
  set_tests_properties(toolchain.${target} toolchain.${target}.binary PROPERTIES LABELS toolchain)
endfunction()

add_executable(toolchain_contract_test toolchain_contract_test.cc)
target_compile_definitions(toolchain_contract_test PRIVATE JITLLM_TEST_TARGET_${target_arch})
jitllm_toolchain_test(toolchain_contract_test)

# Exercise compiler discovery with a legal SDK path that needs shell quoting.
# This configures another CPU-only tree but never runs its nested tests.
add_test(NAME toolchain.sdk_path
         COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR} -DSDK=${JITLLM_SDK}
                 -DTOOLCHAIN=${CMAKE_TOOLCHAIN_FILE} -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/sdk-path
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/check_sdk_path.cmake)
set_tests_properties(toolchain.sdk_path PROPERTIES LABELS toolchain)

if(JITLLM_CUDA)
  add_executable(cuda_contract_test cuda_contract_test.cu)
  jitllm_toolchain_test(cuda_contract_test)
  set_property(TEST toolchain.cuda_contract_test APPEND PROPERTY LABELS gpu)
else()
  add_test(NAME toolchain.cpu_only
           COMMAND ${CMAKE_COMMAND} -DBUILD_DIR=${PROJECT_BINARY_DIR}
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/check_cpu_only.cmake)
  set_tests_properties(toolchain.cpu_only PROPERTIES LABELS toolchain)
endif()
