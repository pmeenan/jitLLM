# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Native x86-64 build on the workstation (D-011, D-032): the SDK's Clang
# and LLD, its GCC 16.2 runtime and the host's glibc (>= 2.39, D-070). CUDA
# code compiles for sm_121 but cannot run here. The `native` and `cpu`
# presets use this file.
set(JITLLM_PROFILE native)
set(JITLLM_BUILD_ARCH x86_64)
set(JITLLM_TARGET_TRIPLE x86_64-linux-gnu)
set(JITLLM_TARGET_MARCH x86-64)
set(JITLLM_LINK_FLAGS -fuse-ld=lld)
include("${CMAKE_CURRENT_LIST_DIR}/sdk.cmake")
