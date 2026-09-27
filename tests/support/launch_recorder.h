// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Records the kernel launches, copies, memsets and cuBLAS calls a thread
// makes, for tests and benchmarks that compare an executed plan with a
// recorded one (plan_record.h). Linking jitllm_launch_recorder wraps the
// CUDA runtime's launch, copy and memset entry points, the driver's copy
// and cuBLAS's matrix products at link time (--wrap); each wrapper notes
// the call while a Recording is open on its thread, then makes it. What
// cuBLAS launches inside itself is not seen here (an nsys trace of the run
// supplies it; plan_compare.py --nsys). Never linked into a production
// binary (tests/support/CMakeLists.txt checks).

#ifndef JITLLM_TESTS_SUPPORT_LAUNCH_RECORDER_H_
#define JITLLM_TESTS_SUPPORT_LAUNCH_RECORDER_H_

#include <string>
#include <utility>
#include <vector>

#include "plan_record.h"

namespace jitllm::test_support {

// Records this thread's calls while it lives. One at a time per thread.
class Recording {
 public:
  Recording();
  ~Recording();
  Recording(const Recording&) = delete;
  Recording& operator=(const Recording&) = delete;
  Recording(Recording&&) = delete;
  Recording& operator=(Recording&&) = delete;

  // What was recorded since the last Take, in call order.
  std::vector<Event> Take();

 private:
  std::vector<Event> events_;
};

// The cuBLAS and cuBLASLt libraries this process loaded, each as its file
// name and path, for the recording's header.
std::vector<std::pair<std::string, std::string>> LoadedCublas();

}  // namespace jitllm::test_support

#endif  // JITLLM_TESTS_SUPPORT_LAUNCH_RECORDER_H_
