// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Running a planned GGML graph (graph_plan.h) under the K-C launch context
// (D-053): the plan is resolved against the implementation registry once
// (execution/registry.h), every step bound to the module's implementation
// it names, identity and all, and then each run walks the steps in order on
// the context's stream. Nothing is substituted: a plan naming an
// implementation this build lacks, or one whose identity changed, binds
// nothing. CUDA builds only.

#ifndef JITLLM_KERNELS_GGML_EXECUTOR_H_
#define JITLLM_KERNELS_GGML_EXECUTOR_H_

#include <cstdint>
#include <expected>
#include <utility>
#include <variant>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// The device's part of upstream's choices on the context's device
// (ops.h SelectMulMat and MulMatVecFusible), for PlanGraph. The context
// must outlive the result.
DeviceChoices DeviceChoicesOf(const LaunchContext& launch);

// The most GGML pool scratch any step of `plan` draws at once on the
// context's device: its cuBLAS products' plans (ops.h PlanMulMatCublas),
// and the quantized products', top-k's and tensor-core attention's
// (ops_ext.h). The other implementations draw none.
std::expected<std::uint64_t, KernelFailure> PlanScratch(const LaunchContext& launch,
                                                        const GraphPlan& plan);

// A plan whose every step is bound to an implementation of this build.
class BoundGraph {
 public:
  // Resolves `plan` against `registry` and binds each step; refused with
  // the registry's rejection, or if a step's nodes fail its host checks.
  static std::expected<BoundGraph, KernelFailure> Bind(const execution::Registry& registry,
                                                       const GraphPlan& plan);

  // Runs every step in order on the context's stream; stops at the first
  // step refused or faulted (launch.h), which is returned.
  std::expected<void, KernelFailure> Run(LaunchContext& launch) const;

  const execution::BoundPlan& bound() const { return bound_; }

 private:
  struct Step {
    std::variant<Kernel, RmsNormMulKernel> kernel;
    std::vector<ggml_tensor*> nodes;
  };
  BoundGraph(execution::BoundPlan bound, std::vector<Step> steps)
      : bound_(std::move(bound)), steps_(std::move(steps)) {}

  execution::BoundPlan bound_;
  std::vector<Step> steps_;
};

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_EXECUTOR_H_
