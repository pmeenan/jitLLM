# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# ExLlamaV3-derived implementations (D-052, D-053; docs/backend-proof.md),
# over the locked ExLlamaV3 subset (third_party/sources.lock.json, component
# `exllamav3`, MIT).
#
#   jitllm_exl3          the host checks of every launch (validate.h),
#                        upstream's GEMV choice (upstream_gemv.h), the
#                        launch contract (launch_contract.h) and the locked
#                        headers, for the host constants that size the
#                        kernels' device context (quant/exl3_devctx.cuh).
#                        Every profile.
#   jitllm_exl3_kernels  jitLLM's launchers (launch.h), which replace
#                        upstream's ATen host wrappers and keep
#                        launch_contract.h; the reconstruction GEMM on
#                        cuBLASLt (recon_gemm.h); the linear's paths
#                        (linear.h); and the module's declarations in the
#                        implementation registry (implementations.h). The
#                        kernels themselves are the component's
#                        (jitllm_exl3_cuda: upstream's GEMM units and
#                        jitLLM's instances of the rest,
#                        jitllm_exl3_kernels.h), built as the P0 reference
#                        was. CUDA builds only.
add_library(jitllm_exl3 STATIC upstream_gemv.cc validate.cc)
target_include_directories(jitllm_exl3 PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_exl3 PUBLIC jitllm_exl3_headers PRIVATE jitllm_warnings)

if(JITLLM_CUDA)
  add_library(jitllm_exl3_kernels STATIC implementations.cc launch.cc linear.cc recon_gemm.cc)
  target_include_directories(jitllm_exl3_kernels SYSTEM PUBLIC
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(jitllm_exl3_kernels
                        PUBLIC jitllm_exl3 jitllm_exl3_cuda jitllm_device_memory jitllm_cublaslt
                               jitllm_execution
                        PRIVATE jitllm_warnings)
  # What each implementation's identity records of the build (D-053): the
  # prepared ExLlamaV3 tree the build used, as the source receipt records
  # it (D-057), the SDK, the target, the kernels' device architecture, the
  # build type and any sanitizers. A missing receipt, entry or digest fails
  # configure: never an identity without the tree.
  get_property(sources GLOBAL PROPERTY JITLLM_SOURCES_ENTRIES)
  set(exl3_tree "")
  string(JSON count ERROR_VARIABLE error LENGTH "${sources}")
  if(NOT error AND count GREATER 0)
    math(EXPR last "${count} - 1")
    foreach(i RANGE ${last})
      string(JSON id ERROR_VARIABLE error GET "${sources}" ${i} id)
      if(NOT error AND id STREQUAL "exllamav3")
        string(JSON exl3_tree ERROR_VARIABLE error GET "${sources}" ${i} source_tree)
      endif()
    endforeach()
  endif()
  string(LENGTH "${exl3_tree}" length)
  if(NOT length EQUAL 64 OR NOT exl3_tree MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "The source receipt has no ExLlamaV3 tree digest for the registry's identities")
  endif()
  string(REPLACE ";" "+" sanitize "${JITLLM_SANITIZE}")
  set_source_files_properties(implementations.cc PROPERTIES COMPILE_DEFINITIONS
    "JITLLM_EXL3_SOURCE_TREE=\"${exl3_tree}\";JITLLM_EXL3_SDK=\"${JITLLM_SDK_IDENTITY}\";JITLLM_EXL3_TARGET=\"${JITLLM_TARGET_TRIPLE}\";JITLLM_EXL3_CUDA_ARCHITECTURES=\"121-real\";JITLLM_EXL3_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\";JITLLM_EXL3_SANITIZE=\"${sanitize}\"")

  # And jitLLM's own code in the module: every file in this directory,
  # hashed at build time into a generated source (module_digest.cmake), as
  # the GGML module does. A file added here without being listed fails the
  # next configure.
  set(module_files CMakeLists.txt implementations.cc implementations.h launch.cc launch.h
                   launch_contract.h linear.cc linear.h module_digest.cmake recon_gemm.cc
                   recon_gemm.h upstream_gemv.cc upstream_gemv.h validate.cc validate.h)
  file(GLOB present CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
       "${CMAKE_CURRENT_SOURCE_DIR}/[!.]*")
  foreach(name IN LISTS present)
    if(NOT name IN_LIST module_files AND NOT name MATCHES "(^#|~$|\\.orig$|\\.rej$)")
      message(FATAL_ERROR "src/kernels/exl3/${name} is not in module_files, so the "
                          "implementations' identity would not cover it")
    endif()
  endforeach()
  string(JOIN ":" module_names ${module_files})
  set_property(GLOBAL PROPERTY JITLLM_EXL3_MODULE_NAMES "${module_names}")
  list(TRANSFORM module_files PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE module_paths)
  set(module_digest "${CMAKE_CURRENT_BINARY_DIR}/module_digest.cc")
  add_custom_command(
    OUTPUT "${module_digest}"
    COMMAND "${CMAKE_COMMAND}" "-DBASE=${CMAKE_CURRENT_SOURCE_DIR}" "-DNAMES=${module_names}"
            "-DOUTPUT=${module_digest}" -P "${CMAKE_CURRENT_SOURCE_DIR}/module_digest.cmake"
    DEPENDS ${module_paths}
    COMMENT "Hashing the EXL3 module's sources for its implementation identities (D-053)"
    VERBATIM)
  target_sources(jitllm_exl3_kernels PRIVATE "${module_digest}")
endif()
