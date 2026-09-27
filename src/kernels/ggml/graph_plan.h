// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Planning a GGML graph's execution as upstream's CUDA backend would run
// it (D-053; docs/backend-proof.md, Tier E): which of the module's
// implementations (implementations.h) runs each node, fused or not, and
// where each computed tensor lives.
//
// PlanGraph walks the nodes in order as ggml_cuda_graph_evaluate_and_capture
// does (ggml-cuda.cu:4185-4360). Views and no-ops launch nothing. With
// fusion on (FP16-F), each node is first offered to upstream's fusion
// patterns in ggml_cuda_try_fuse's order, through fusion.h's gates; a graph
// where a pattern jitLLM does not implement might apply is refused, never
// run differently. With fusion off (FP16-U, GGML_CUDA_DISABLE_FUSION),
// upstream consults no pattern: an RMSNorm whose mul follows it runs as the
// unfused RMSNorm-mul implementation (rms_norm's launcher, then mul's), and
// everything else node by node. A matrix product's kernel family is
// upstream's selection on the device (ggml_cuda_mul_mat), which the caller
// supplies (ops.h on a CUDA build; a model of it in tests).
//
// Upstream's fusion gates compare the fused output's memory with its
// inputs', so the plan depends on addresses, and the addresses on the plan
// (what is live together). PlanActivations breaks the circle: plan with
// every computed tensor at distinct addresses, place the tensors by that
// plan's steps (a step's outputs never share memory with anything live
// during it), bind them, plan again with the real addresses and require the
// same plan (SamePlan). Every profile builds this; nothing here launches.

#ifndef JITLLM_KERNELS_GGML_GRAPH_PLAN_H_
#define JITLLM_KERNELS_GGML_GRAPH_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"

namespace jitllm::kernels::ggml {

// The device's part of upstream's choices.
struct DeviceChoices {
  // The family ggml_cuda_mul_mat selects for a product, or a refusal where
  // it would take a path no implementation here has.
  std::function<std::expected<MulMatPath, KernelFailure>(const ggml_tensor*)> mul_mat;
  // ops.h MulMatVecFusible: the MMVF fusion gates' device condition.
  std::function<bool(const ggml_tensor*)> vector_fusible;
};

// One implementation's run over its nodes, in the order implementations.h
// lists for it.
struct PlanStep {
  execution::Operation operation = execution::Operation::kAdd;
  std::string_view implementation;
  std::vector<ggml_tensor*> nodes;
};

struct GraphPlan {
  std::vector<PlanStep> steps;

  // The plan as the registry takes it (execution/registry.h).
  std::vector<execution::Choice> Choices() const;
};

// The module's implementation names (implementations.h).
inline constexpr std::string_view kRmsNormMulFused = "ggml.rms_norm_mul.fused";
inline constexpr std::string_view kRmsNormMulUnfused = "ggml.rms_norm_mul.unfused";
inline constexpr std::string_view kRmsNormName = "ggml.rms_norm";
inline constexpr std::string_view kAddName = "ggml.add";
inline constexpr std::string_view kMulName = "ggml.mul";
inline constexpr std::string_view kMulMatVector = "ggml.mul_mat.mmvf";
inline constexpr std::string_view kMulMatTensorCore = "ggml.mul_mat.mmf";
inline constexpr std::string_view kMulMatCublas = "ggml.mul_mat.cublas";
inline constexpr std::string_view kGetRowsName = "ggml.get_rows";
inline constexpr std::string_view kSetRowsName = "ggml.set_rows";
inline constexpr std::string_view kRopeName = "ggml.rope.neox";
inline constexpr std::string_view kRopeSetRowsFused = "ggml.rope_set_rows.fused";
inline constexpr std::string_view kSoftMaxName = "ggml.soft_max";
inline constexpr std::string_view kContName = "ggml.cont";
inline constexpr std::string_view kSwiGluName = "ggml.swiglu";
inline constexpr std::string_view kMulMatAddFused = "ggml.mul_mat_add.mmvf_fused";
inline constexpr std::string_view kMulMatGluFused = "ggml.mul_mat_glu.mmvf_fused";

// Upstream's no-op nodes (ggml_cuda_is_view_or_noop).
bool LaunchesNothing(const ggml_tensor* node);

// The plan of `graph` (fusion.h's node order) with fusion on or off.
std::expected<GraphPlan, KernelFailure> PlanGraph(GraphNodes graph, bool fusion,
                                                  const DeviceChoices& device);

// Whether two plans run the same implementations over the same nodes.
bool SamePlan(const GraphPlan& a, const GraphPlan& b);

// Where each computed tensor of a plan lives in one region.
struct Placement {
  std::vector<std::pair<ggml_tensor*, std::uint64_t>> offsets;  // tensor, offset
  std::uint64_t extent = 0;  // the region's bytes the placement uses
};

// Places every computed node of `graph` that is no view, and each of
// `inputs` (leaves the region also holds), in one region at offsets
// aligned to `alignment`, each rounded up to it (ggml-alloc's rule). A
// tensor lives from the step that computes it (inputs from the start) to
// the last step reading it or a view of it; the graph's last node lives to
// the end. Tensors whose lifetimes meet never share bytes, so a step's
// outputs share none with its inputs. Placed largest first, each at the
// lowest offset free for its whole lifetime.
std::expected<Placement, KernelFailure> PlaceActivations(GraphNodes graph, const GraphPlan& plan,
                                                         std::span<ggml_tensor* const> inputs,
                                                         std::uint64_t alignment);

// Points every view among `nodes` into its source's memory (the source's
// address plus the view's offset), once the sources are bound.
void BindViews(std::span<ggml_tensor* const> nodes);

// Gives every computed node that is no view a distinct address from
// `base` on (PlaceActivations's first pass), then binds the views.
void BindDistinct(GraphNodes graph, std::uint64_t base);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_GRAPH_PLAN_H_
