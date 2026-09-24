# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The source mechanism (D-017, D-057), checked on this build and, in the CPU
# profile, end to end on a synthetic lock. Every test carries the `sources`
# label.

# JITLLM_PYTHON comes from cmake/sources/TreeDigest.cmake.

# The whole lock is valid, every patch file included, not only the closure
# configure read (D-017, D-057).
add_test(NAME sources.lock
         COMMAND ${JITLLM_PYTHON} ${PROJECT_SOURCE_DIR}/tools/prepare-sources --check)

# The receipt matches the lock and names exactly the profile's closure.
add_test(NAME sources.receipt
         COMMAND ${CMAKE_COMMAND} -DRECEIPT=${PROJECT_BINARY_DIR}/jitllm-receipt.json
                 -DLOCK=${PROJECT_SOURCE_DIR}/third_party/sources.lock.json
                 -DSDK_IDENTITY=${JITLLM_SDK_IDENTITY} -P ${CMAKE_CURRENT_SOURCE_DIR}/check_receipt.cmake)

# What the build actually compiled and linked is the receipt's closure, from
# allowed places only, and without exceptions.
set(cross_flag "")
if(CMAKE_CROSSCOMPILING)
  set(cross_flag --cross)
endif()
add_test(NAME sources.closure
         COMMAND ${JITLLM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/check_closure.py
                 --build-dir ${PROJECT_BINARY_DIR} --source-dir ${PROJECT_SOURCE_DIR}
                 --sdk ${JITLLM_SDK} --ninja ${CMAKE_MAKE_PROGRAM} ${cross_flag})
set_tests_properties(sources.lock sources.receipt sources.closure PROPERTIES LABELS sources)

# The mechanism's gates on fixtures. They build natively for the host, so
# they run once, in the CPU-only profile.
if(NOT CMAKE_CROSSCOMPILING AND NOT JITLLM_CUDA)
  add_test(NAME sources.mechanism
           COMMAND ${JITLLM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/mechanism_test.py
                   --repo ${PROJECT_SOURCE_DIR} --work ${CMAKE_CURRENT_BINARY_DIR}/mechanism
                   --cmake ${CMAKE_COMMAND} --ninja ${CMAKE_MAKE_PROGRAM} --sdk ${JITLLM_SDK}
                   --toolchain ${CMAKE_TOOLCHAIN_FILE})
  set_tests_properties(sources.mechanism PROPERTIES LABELS sources TIMEOUT 600)
endif()
