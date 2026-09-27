// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The K-C launch context (D-053; docs/backend-proof.md#dispatch-and-implementations-d-053):
// GGML's CUDA operation launchers run with a ggml_backend_cuda_context that
// jitLLM fills with the stream to launch on and a scratch pool over
// workspace the caller declared and charged. GGML creates nothing of its
// own: no stream, pool, cuBLAS handle, workspace or process-wide setting
// (ggml_support.cu); it keeps only a host-side cache of which kernels may
// use programmatic dependent launch. Run checks an operation's scratch bound against the
// workspace, calls its launchers, and reports the first CUDA error they
// recorded instead of aborting.
//
// A launch only queues work; completion is the caller's, through a fence
// after the last launch. The pool reuses workspace offsets in stream order
// as GGML frees them when a host launcher returns, so the workspace must
// stay mapped and charged until that fence has completed
// (docs/async-model.md). After an error, what was queued is undetermined:
// the context refuses further runs, and its stream and workspace wait for
// recovery. One context per stream, used on the device submission lane
// only. Each run takes the stream again through DeviceExecution::Submission,
// which counts it as queued work, so the provider refuses to destroy the
// stream until a fence after the run has completed and been released; the
// context must not outlive the provider or the stream.

#ifndef JITLLM_KERNELS_GGML_LAUNCH_H_
#define JITLLM_KERNELS_GGML_LAUNCH_H_

#include <cstdint>
#include <expected>
#include <memory>
#include <utility>

#include "base/bytes.h"
#include "kernels/ggml/tensors.h"
#include "providers/device_execution.h"

struct ggml_backend_cuda_context;

namespace jitllm::kernels::ggml {

class WorkspacePool;

class LaunchContext {
 public:
  // A device range the pool hands out: mapped with access and charged by
  // the caller for as long as the context's launches can use it.
  struct Workspace {
    std::uint64_t base = 0;
    base::Bytes size;
  };

  // Launches on `stream`, a CUDA provider's stream on `device`.
  static std::expected<std::unique_ptr<LaunchContext>, KernelFailure> Create(
      int device, providers::DeviceExecution& execution, providers::StreamId stream,
      Workspace workspace);

  LaunchContext(const LaunchContext&) = delete;
  LaunchContext& operator=(const LaunchContext&) = delete;
  LaunchContext(LaunchContext&&) = delete;
  LaunchContext& operator=(LaunchContext&&) = delete;
  ~LaunchContext();

  // Calls launch(context) for GGML launchers that together draw at most
  // `scratch` from the pool at once, counting each block the pool hands out
  // from a 256-byte boundary (so the bound includes the rounding). Refused, with nothing queued, if
  // the workspace is smaller, the provider refuses the stream or the context is faulted; kUnknown
  // if a launcher recorded a CUDA error, which faults the context.
  template <typename Launch>
  std::expected<void, KernelFailure> Run(base::Bytes scratch, Launch&& launch) {
    if (auto begun = Begin(scratch); !begun) {
      return begun;
    }
    std::forward<Launch>(launch)(*context_);
    return End();
  }

  // The most scratch any run has held at once.
  base::Bytes scratch_peak() const;
  bool faulted() const { return faulted_; }
  int device() const { return device_; }

 private:
  LaunchContext(int device, providers::DeviceExecution& execution, providers::StreamId stream,
                providers::NativeStream native, std::unique_ptr<ggml_backend_cuda_context> context,
                std::unique_ptr<WorkspacePool> pool, Workspace workspace);

  std::expected<void, KernelFailure> Begin(base::Bytes scratch);
  std::expected<void, KernelFailure> End();

  int device_;
  providers::DeviceExecution& execution_;
  providers::StreamId stream_;
  providers::NativeStream native_;
  std::unique_ptr<WorkspacePool> pool_;
  // Declared after the pool: destroyed first, once it has handed the pool
  // and stream back.
  std::unique_ptr<ggml_backend_cuda_context> context_;
  Workspace workspace_;
  bool faulted_ = false;
};

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_LAUNCH_H_
