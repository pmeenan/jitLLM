# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# The retained-backing comparison's deterministic replay (D-035, D-079;
# docs/experiments/retained-backing/): the candidate backing designs and the
# replay engine, on the fake device-memory provider. The designs are
# candidates for a measurement, not the memory manager; tests/unit's
# rb_replay_test checks their invariants.
add_library(jitllm_retained_backing STATIC json.cc trace.cc designs.cc replay.cc)
target_include_directories(jitllm_retained_backing PUBLIC "${PROJECT_SOURCE_DIR}/benchmarks")
target_link_libraries(jitllm_retained_backing PUBLIC jitllm_base jitllm_fake_providers PRIVATE jitllm_warnings)

add_executable(jitllm_rb_replay main.cc)
target_link_libraries(jitllm_rb_replay PRIVATE jitllm_retained_backing jitllm_warnings)
