# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# GGML-derived implementations (D-053; docs/backend-proof.md), over the
# locked GGML subset (third_party/sources.lock.json, component `ggml`).
#
#   jitllm_ggml_tensors  GGML tensor descriptors over jitLLM memory
#                        (tensors.h) and the implementations' operand
#                        checks (validate.h). Every profile.
#   jitllm_ggml_kernels  the K-C launch context (launch.h), jitLLM's cuBLAS
#                        handle (cublas.h), the operation implementations
#                        (ops.h, mul_mat_cublas.cu) and jitLLM's definitions
#                        of the ggml-cuda.cu symbols GGML's launchers use
#                        (ggml_support.cu). CUDA builds only.
add_library(jitllm_ggml_tensors STATIC tensors.cc validate.cc)
target_include_directories(jitllm_ggml_tensors PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_ggml_tensors PUBLIC jitllm_base jitllm_ggml_base PRIVATE jitllm_warnings)

if(JITLLM_CUDA)
  # cuBLAS, the SDK's pinned shared library, linked dynamically (D-076):
  # binaries record libcublas.so.13 and libcublasLt.so.13, which
  # libcublas.so.13 finds beside itself (GNU ld on the Spark looks for it
  # when linking). The package ships
  # neither until a packaged binary links them; tests find them in
  # build/<preset>/cublas (jitllm_gpu_gtest), hard links to the SDK's.
  set(cublas_dir "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/lib")
  add_library(jitllm_cublas SHARED IMPORTED GLOBAL)
  set_target_properties(jitllm_cublas PROPERTIES IMPORTED_LOCATION "${cublas_dir}/libcublas.so"
                                                 IMPORTED_SONAME libcublas.so.13)
  target_link_options(jitllm_cublas INTERFACE "LINKER:-rpath-link,${cublas_dir}")
  # cuBLASLt too, whose version the handle checks (cublas.h).
  add_library(jitllm_cublaslt SHARED IMPORTED GLOBAL)
  set_target_properties(jitllm_cublaslt PROPERTIES IMPORTED_LOCATION "${cublas_dir}/libcublasLt.so"
                                                   IMPORTED_SONAME libcublasLt.so.13)
  file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cublas")
  foreach(library IN ITEMS libcublas.so.13 libcublasLt.so.13)
    file(REAL_PATH "${cublas_dir}/${library}" resolved)
    file(CREATE_LINK "${resolved}" "${PROJECT_BINARY_DIR}/cublas/${library}" COPY_ON_ERROR)
  endforeach()

  add_library(jitllm_ggml_kernels STATIC ggml_support.cu launch.cu ops.cu mul_mat_cublas.cu
                                         cublas.cc)
  target_include_directories(jitllm_ggml_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
  target_include_directories(jitllm_ggml_kernels SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_ggml_kernels
                        PUBLIC jitllm_ggml_tensors jitllm_ggml_cuda jitllm_device_memory
                               jitllm_cublas jitllm_cublaslt
                        # the NVIDIA driver, for ggml_support.cu's device queries
                        PRIVATE jitllm_cuda_device_memory jitllm_warnings)
  set_target_properties(jitllm_ggml_kernels PROPERTIES CUDA_ARCHITECTURES 121a-real)
  # The copy of upstream's cuBLAS path builds with the FP16 bridge's device
  # flag, as GGML's own files do (third_party/patches/ggml/0002), so that its
  # kernel's SASS compares with the bridge's. (NVCC's -O sets only the host
  # optimization level; device code is always -O3.)
  set_source_files_properties(mul_mat_cublas.cu PROPERTIES COMPILE_OPTIONS "-use_fast_math")
endif()
