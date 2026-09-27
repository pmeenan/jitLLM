# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# ExLlamaV3-derived implementations (D-052, D-053; docs/backend-proof.md),
# over the locked ExLlamaV3 subset (third_party/sources.lock.json, component
# `exllamav3`, MIT).
#
#   jitllm_exl3          the locked headers, for the host constants that
#                        size the kernels' device context
#                        (quant/exl3_devctx.cuh), and launch_contract.h.
#                        Every profile.
#   jitllm_exl3_kernels  the EXL3 GEMM kernels for the M2 fixtures' rates
#                        (jitllm_exl3_cuda: upstream's compilation units,
#                        built as the P0 reference was, with the kernel
#                        tables they define) and the CUDA headers they need.
#                        CUDA builds only. jitLLM's launchers, which will
#                        replace upstream's ATen host wrappers, are not
#                        here yet. They include launch_contract.h and keep
#                        it: every kernel launched cooperatively with all
#                        blocks co-resident (grid syncs trap otherwise, spin
#                        barriers hang), and lock slots no concurrently
#                        running launch shares.
#
# Nothing here includes ExLlamaV3's GEMV kernel or a file that does yet:
# it is core-eligible (D-080) and enters with its own per-file record.
add_library(jitllm_exl3 INTERFACE)
target_include_directories(jitllm_exl3 INTERFACE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_exl3 INTERFACE jitllm_exl3_headers)

if(JITLLM_CUDA)
  add_library(jitllm_exl3_kernels INTERFACE)
  target_include_directories(jitllm_exl3_kernels SYSTEM INTERFACE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_exl3_kernels INTERFACE jitllm_exl3 jitllm_exl3_cuda)
endif()
