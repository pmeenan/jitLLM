# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Unit tests of jitLLM's modules, one GoogleTest executable per module
# (docs/architecture.md#testing). Every test carries the `unit` label.

add_executable(cli_test cli_test.cc)
target_link_libraries(cli_test PRIVATE jitllm_cli jitllm_surface_versions)
target_compile_definitions(cli_test PRIVATE "JITLLM_TEST_PROJECT_VERSION=\"${PROJECT_VERSION}\"")
jitllm_gtest(cli_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

add_executable(base_test base_test.cc)
target_link_libraries(base_test PRIVATE jitllm_report jitllm_base)
jitllm_gtest(base_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(platform_test platform_test.cc)
target_link_libraries(platform_test PRIVATE jitllm_platform)
jitllm_gtest(platform_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The CUDA provider, apart from its device queries, which need a GPU
# (smoke.doctor in tests/smoke).
if(JITLLM_CUDA)
  add_executable(cuda_test cuda_test.cc)
  target_link_libraries(cuda_test PRIVATE jitllm_cuda_probe)
  jitllm_gtest(cuda_test TEST_PREFIX unit. PROPERTIES LABELS unit)
endif()

# The node configuration. Tests that need scratch trees (here and in
# platform_test) put them in the build tree, whose parents no other user
# shares, and remote runs map it (JITLLM_TEST_SCRATCH).
add_executable(config_test config_test.cc)
target_link_libraries(config_test PRIVATE jitllm_config jitllm_platform)
jitllm_gtest(config_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The runtime's startup steps and the process services it relies on.
add_executable(runtime_test runtime_test.cc)
target_link_libraries(runtime_test PRIVATE jitllm_runtime jitllm_platform)
jitllm_gtest(runtime_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The resource core: the catalog and the memory manager's policy, on no
# vendor SDK (D-026).
add_executable(catalog_test catalog_test.cc)
target_link_libraries(catalog_test PRIVATE jitllm_catalog jitllm_base)
jitllm_gtest(catalog_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(memory_test memory_test.cc)
target_link_libraries(memory_test PRIVATE jitllm_memory jitllm_catalog)
jitllm_gtest(memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(admission_test admission_test.cc)
target_link_libraries(admission_test PRIVATE jitllm_scheduler)
jitllm_gtest(admission_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-048's lanes, completion board and task trees, with real threads.
add_executable(lanes_test lanes_test.cc)
target_link_libraries(lanes_test PRIVATE jitllm_scheduler)
jitllm_gtest(lanes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The scheduler thread's turn loop and its lanes over the fake providers:
# deterministic turns, and threaded lost-wakeup and memory-ordering stress.
add_executable(scheduler_test scheduler_test.cc)
target_link_libraries(scheduler_test PRIVATE jitllm_scheduler jitllm_fake_providers)
jitllm_gtest(scheduler_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# Storage and whole reads, on the fake and on io_uring with a real file.
add_executable(storage_test storage_test.cc)
target_link_libraries(storage_test PRIVATE jitllm_fake_providers jitllm_storage jitllm_platform)
jitllm_gtest(storage_test TEST_PREFIX unit. PROPERTIES LABELS unit
             ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

# The providers' shared rules, on the deterministic fakes.
add_executable(device_memory_test device_memory_test.cc)
target_link_libraries(device_memory_test PRIVATE jitllm_fake_providers)
jitllm_gtest(device_memory_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(device_execution_test device_execution_test.cc)
target_link_libraries(device_execution_test PRIVATE jitllm_fake_providers)
jitllm_gtest(device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The CUDA device-memory provider against a real device: label `gpu`, so
# only on a Spark.
if(JITLLM_CUDA)
  add_executable(cuda_device_execution_test cuda_device_execution_test.cc)
  target_include_directories(cuda_device_execution_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(cuda_device_execution_test PRIVATE jitllm_cuda_device_memory)
  jitllm_gtest(cuda_device_execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

  # The scheduler and its lanes over io_uring, CUDA host VMM and CUDA
  # streams (label `gpu`).
  add_executable(cuda_lanes_test cuda_lanes_test.cc)
  target_link_libraries(cuda_lanes_test PRIVATE jitllm_scheduler jitllm_cuda_device_memory)
  jitllm_gtest(cuda_lanes_test TEST_PREFIX unit. PROPERTIES LABELS gpu
               ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")

  add_executable(cuda_device_memory_test cuda_device_memory_test.cc)
  target_link_libraries(cuda_device_memory_test PRIVATE jitllm_cuda_device_memory jitllm_storage)
  jitllm_gtest(cuda_device_memory_test TEST_PREFIX unit. PROPERTIES LABELS gpu
               ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
endif()

# The executed-plan recording's JSON lines (tests/support/plan_record.h),
# which docs/experiments/backend-proof-p2/plan_compare.py reads.
add_executable(plan_record_test plan_record_test.cc)
target_link_libraries(plan_record_test PRIVATE jitllm_plan_record jitllm_plan_record_sample)
jitllm_gtest(plan_record_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The implementation registry and plans (D-053), on fake declarations.
add_executable(execution_test execution_test.cc)
target_link_libraries(execution_test PRIVATE jitllm_execution)
jitllm_gtest(execution_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# D-068's shape contracts: state representations and composed contexts
# (model_test), and request programs planned from them and driven through
# admission, the ledgers and the catalog over the fake device-memory
# provider (shapes_test), in every profile.
add_executable(model_test model_test.cc)
target_link_libraries(model_test PRIVATE jitllm_model)
jitllm_gtest(model_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(shapes_test shapes_test.cc)
target_link_libraries(shapes_test PRIVATE jitllm_execution jitllm_scheduler jitllm_fake_providers)
jitllm_gtest(shapes_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# GGML tensor descriptors over jitLLM memory, in every profile, and GGML's
# CUDA launchers and the cuBLAS path under jitLLM's launch context on a
# real device (label `gpu`, so only on a Spark).
add_executable(ggml_tensors_test ggml_tensors_test.cc)
target_link_libraries(ggml_tensors_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(ggml_tensors_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen2 adapter (model/qwen2.h): binding a synthetic index to the
# compiled-in profile, and a chunk's host-built inputs.
add_executable(qwen2_test qwen2_test.cc)
target_link_libraries(qwen2_test PRIVATE jitllm_model)
jitllm_gtest(qwen2_test TEST_PREFIX unit. PROPERTIES LABELS unit)

# The Qwen2 chunk graph, its plan with and without fusion, and activation
# placement (kernels/ggml/qwen2_graph.h, graph_plan.h).
add_executable(ggml_graph_test ggml_graph_test.cc)
target_link_libraries(ggml_graph_test PRIVATE jitllm_ggml_tensors jitllm_model)
jitllm_gtest(ggml_graph_test TEST_PREFIX unit. PROPERTIES LABELS unit)

add_executable(ggml_validate_test ggml_validate_test.cc)
target_link_libraries(ggml_validate_test PRIVATE jitllm_ggml_tensors)
jitllm_gtest(ggml_validate_test TEST_PREFIX unit. PROPERTIES LABELS unit)

if(JITLLM_CUDA)
  add_executable(ggml_kernels_test ggml_kernels_test.cc)
  target_include_directories(ggml_kernels_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_kernels_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(ggml_kernels_test)

  # The module's implementation identities follow every one of its source
  # files (src/kernels/ggml/module_digest.cmake). A script on the build
  # host, so it runs in every CUDA profile without a GPU.
  get_property(module_names GLOBAL PROPERTY JITLLM_GGML_MODULE_NAMES)
  add_test(NAME unit.GgmlModuleDigest
           COMMAND ${CMAKE_COMMAND} "-DBASE=${PROJECT_SOURCE_DIR}/src/kernels/ggml"
                   "-DNAMES=${module_names}"
                   "-DGENERATED=${PROJECT_BINARY_DIR}/src/kernels/ggml/module_digest.cc"
                   "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/module_digest"
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/ggml_module_digest_test.cmake)
  set_tests_properties(unit.GgmlModuleDigest PROPERTIES LABELS unit)

  # The remaining operations of the FP16 bridge's recorded plan. The test
  # records each launch through tests/support's launch recorder, which wraps
  # the CUDA runtime's entry points at link time, and compares it with the
  # recorded plan.
  add_executable(ggml_ops_test ggml_ops_test.cc)
  target_include_directories(ggml_ops_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_ops_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory
                                              jitllm_launch_recorder jitllm_plan_record_sample)
  jitllm_gpu_gtest(ggml_ops_test)

  add_executable(ggml_cublas_test ggml_cublas_test.cc)
  target_include_directories(ggml_cublas_test SYSTEM PRIVATE
                             "${JITLLM_SDK}/cuda/targets/${JITLLM_CUDA_TARGET_DIR}/include")
  target_link_libraries(ggml_cublas_test PRIVATE jitllm_ggml_kernels jitllm_cuda_device_memory)
  jitllm_gpu_gtest(ggml_cublas_test)
endif()

# The v0 artifact reader, in every profile, against a corpus the M0
# prototype builds and judges at build time (artifact_corpus.py): its views
# of valid artifacts and its verdicts on about 190 mutated ones and 800
# seeded random mutations of their documents (with 8 divergence probes). The real
# v0 fixtures are opened where they are installed (a Spark) and skipped
# elsewhere.
set(artifact_corpus "${CMAKE_CURRENT_BINARY_DIR}/artifact-corpus")
set(artifact_prototype "${PROJECT_SOURCE_DIR}/docs/experiments/artifact-layout")
add_custom_command(
  OUTPUT "${artifact_corpus}/stamp"
  COMMAND "${JITLLM_PYTHON}" -B "${CMAKE_CURRENT_SOURCE_DIR}/artifact_corpus.py" "${artifact_corpus}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/artifact_corpus.py" "${artifact_prototype}/layout.py"
          "${artifact_prototype}/test_layout.py"
  COMMENT "Building the artifact reader's corpus with the M0 prototype"
  VERBATIM)
add_custom_target(artifact_corpus ALL DEPENDS "${artifact_corpus}/stamp")
add_executable(artifact_test artifact_test.cc)
target_link_libraries(artifact_test PRIVATE jitllm_artifact)
add_dependencies(artifact_test artifact_corpus)
# CTest runs it here, beside the corpus (remote runs map this directory).
jitllm_gtest(artifact_test TEST_PREFIX unit. WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
             PROPERTIES LABELS unit ENVIRONMENT "JITLLM_TEST_SCRATCH=${CMAKE_CURRENT_BINARY_DIR}/scratch")
