// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/graph_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/fusion.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {
namespace {

using execution::Operation;

std::unexpected<KernelFailure> Rejected(std::string detail) {
  return std::unexpected(
      KernelFailure{.error = KernelError::kRejected, .detail = std::move(detail)});
}

std::string Where(GraphNodes graph, std::size_t i) {
  return std::format("node {} ({} {})", i, ggml_op_desc(graph[i]), graph[i]->name);
}

// A computed tensor with memory of its own.
bool Computed(const ggml_tensor* t) { return t->op != GGML_OP_NONE && t->view_src == nullptr; }

// The tensor whose memory `t` is: itself, or the end of its view chain
// (GGML points a view of a view at the first source, but that is
// upstream's choice, not a contract).
const ggml_tensor* Storage(const ggml_tensor* t) {
  while (t->view_src != nullptr) {
    t = t->view_src;
  }
  return t;
}

std::string_view MulMatName(MulMatPath path) {
  switch (path) {
    case MulMatPath::kVector:
      return kMulMatVector;
    case MulMatPath::kTensorCore:
      return kMulMatTensorCore;
    case MulMatPath::kCublas:
      return kMulMatCublas;
  }
  return kMulMatCublas;
}

std::uint64_t Rounded(std::uint64_t bytes, std::uint64_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

}  // namespace

std::vector<execution::Choice> GraphPlan::Choices() const {
  std::vector<execution::Choice> choices;
  choices.reserve(steps.size());
  for (const PlanStep& step : steps) {
    choices.push_back(
        {.operation = step.operation, .implementation = std::string(step.implementation)});
  }
  return choices;
}

bool LaunchesNothing(const ggml_tensor* node) {
  return ggml_is_empty(node) || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE ||
         node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_NONE;
}

std::expected<GraphPlan, KernelFailure> PlanGraph(GraphNodes graph, bool fusion,
                                                  const DeviceChoices& device) {
  GraphPlan plan;
  std::vector<bool> taken(graph.size(), false);
  const auto add = [&](Operation operation, std::string_view name, std::size_t first,
                       std::vector<ggml_tensor*> nodes, std::size_t span) {
    for (std::size_t k = first; k < first + span; ++k) {
      taken[k] = true;
    }
    plan.steps.push_back(
        {.operation = operation, .implementation = name, .nodes = std::move(nodes)});
  };
  for (std::size_t i = 0; i < graph.size(); ++i) {
    ggml_tensor* node = graph[i];
    if (taken[i] || LaunchesNothing(node)) {
      continue;
    }
    if (fusion) {
      // ggml_cuda_try_fuse's order: the patterns jitLLM lacks must not
      // apply, then RoPE and its store, the gate/up products, the product
      // and its add, and the RMSNorm and its mul.
      if (const auto pattern = UnimplementedFusionAt(graph, i)) {
        return Rejected(
            std::format("{}: upstream's {} fusion might apply there, and no "
                        "implementation here reproduces it",
                        Where(graph, i), *pattern));
      }
      if (node->op == GGML_OP_ROPE) {
        if (const auto f = RopeSetRowsFusionAt(graph, i)) {
          add(Operation::kRopeSetRows, kRopeSetRowsFused, i, {f->rope, f->set_rows}, 3);
          continue;
        }
      }
      if (node->op == GGML_OP_MUL_MAT) {
        if (const auto f = MulMatGluFusionAt(graph, i); f && device.vector_fusible(f->up)) {
          add(Operation::kMulMatGlu, kMulMatGluFused, i, {f->gate, f->up, f->glu}, 3);
          continue;
        }
        if (const auto f = MulMatAddFusionAt(graph, i); f && device.vector_fusible(f->mul_mat)) {
          add(Operation::kMulMatAdd, kMulMatAddFused, i, {f->mul_mat, f->add}, 2);
          continue;
        }
      }
      if (node->op == GGML_OP_RMS_NORM) {
        if (const auto f = RmsNormMulFusionAt(graph, i)) {
          add(Operation::kRmsNormMul, kRmsNormMulFused, i, {f->norm, f->mul}, 2);
          continue;
        }
      }
    }
    switch (node->op) {
      case GGML_OP_RMS_NORM: {
        ggml_tensor* next = i + 1 < graph.size() ? graph[i + 1] : nullptr;
        if (!fusion && next != nullptr && next->op == GGML_OP_MUL && next->src[0] == node) {
          add(Operation::kRmsNormMul, kRmsNormMulUnfused, i, {node, next}, 2);
        } else {
          add(Operation::kRmsNorm, kRmsNormName, i, {node}, 1);
        }
        break;
      }
      case GGML_OP_MUL:
        add(Operation::kMul, kMulName, i, {node}, 1);
        break;
      case GGML_OP_ADD:
        add(Operation::kAdd, kAddName, i, {node}, 1);
        break;
      case GGML_OP_MUL_MAT: {
        const auto path = device.mul_mat(node);
        if (!path) {
          return Rejected(std::format("{}: {}", Where(graph, i), path.error().detail));
        }
        add(Operation::kMatMul, MulMatName(*path), i, {node}, 1);
        break;
      }
      case GGML_OP_ROPE:
        if (node->op_params[2] != GGML_ROPE_TYPE_NEOX) {
          return Rejected(std::format("{}: only NEOX RoPE is implemented", Where(graph, i)));
        }
        add(Operation::kRope, kRopeName, i, {node}, 1);
        break;
      case GGML_OP_SET_ROWS:
        add(Operation::kSetRows, kSetRowsName, i, {node}, 1);
        break;
      case GGML_OP_GET_ROWS:
        add(Operation::kGetRows, kGetRowsName, i, {node}, 1);
        break;
      case GGML_OP_SOFT_MAX:
        add(Operation::kSoftMax, kSoftMaxName, i, {node}, 1);
        break;
      case GGML_OP_CONT:
        add(Operation::kCont, kContName, i, {node}, 1);
        break;
      case GGML_OP_GLU:
        if (ggml_get_glu_op(node) != GGML_GLU_OP_SWIGLU || node->src[1] == nullptr) {
          return Rejected(std::format("{}: only a split SwiGLU is implemented", Where(graph, i)));
        }
        add(Operation::kSwiGlu, kSwiGluName, i, {node}, 1);
        break;
      default:
        return Rejected(std::format("{}: no implementation of this operation", Where(graph, i)));
    }
  }
  return plan;
}

bool SamePlan(const GraphPlan& a, const GraphPlan& b) {
  return std::ranges::equal(a.steps, b.steps, [](const PlanStep& x, const PlanStep& y) {
    return x.operation == y.operation && x.implementation == y.implementation && x.nodes == y.nodes;
  });
}

std::expected<Placement, KernelFailure> PlaceActivations(GraphNodes graph, const GraphPlan& plan,
                                                         std::span<ggml_tensor* const> inputs,
                                                         std::uint64_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return Rejected("the alignment is not a power of two");
  }
  struct Life {
    ggml_tensor* tensor = nullptr;
    std::int64_t first = 0;  // the step computing it; -1 for an input
    std::int64_t last = 0;   // the last step reading it
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;
    bool placed = false;
  };
  std::vector<Life> lives;
  std::unordered_map<const ggml_tensor*, std::size_t> index;
  const auto track = [&](ggml_tensor* t, std::int64_t first) {
    if (index.contains(t)) {
      return;
    }
    index.emplace(t, lives.size());
    lives.push_back(
        {.tensor = t, .first = first, .last = first, .bytes = Rounded(ggml_nbytes(t), alignment)});
  };
  for (ggml_tensor* input : inputs) {
    if (input->op != GGML_OP_NONE || input->view_src != nullptr) {
      return Rejected("an input to place is not a leaf");
    }
    track(input, -1);
  }
  for (std::size_t s = 0; s < plan.steps.size(); ++s) {
    for (ggml_tensor* node : plan.steps[s].nodes) {
      if (Computed(node)) {
        track(node, static_cast<std::int64_t>(s));
      }
    }
  }
  for (const ggml_tensor* node : graph) {
    if (Computed(node) && !index.contains(node)) {
      return Rejected(std::format("{} is computed by no step", node->name));
    }
  }
  for (std::size_t s = 0; s < plan.steps.size(); ++s) {
    for (const ggml_tensor* node : plan.steps[s].nodes) {
      for (const ggml_tensor* src : node->src) {
        if (src == nullptr) {
          continue;
        }
        if (const auto found = index.find(Storage(src)); found != index.end()) {
          Life& life = lives[found->second];
          life.last = std::max(life.last, static_cast<std::int64_t>(s));
        }
      }
    }
  }
  // The graph's output lives to the end, in whatever tensor it views.
  if (!graph.empty()) {
    if (const auto found = index.find(Storage(graph.back())); found != index.end()) {
      lives[found->second].last = static_cast<std::int64_t>(plan.steps.size());
    }
  }
  std::vector<std::size_t> order(lives.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::ranges::stable_sort(
      order, [&](std::size_t a, std::size_t b) { return lives[a].bytes > lives[b].bytes; });
  Placement placement;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> busy;  // offset, end
  for (const std::size_t i : order) {
    Life& life = lives[i];
    busy.clear();
    for (const Life& other : lives) {
      if (other.placed && other.first <= life.last && life.first <= other.last) {
        busy.emplace_back(other.offset, other.offset + other.bytes);
      }
    }
    std::ranges::sort(busy);
    std::uint64_t offset = 0;
    for (const auto& [start, end] : busy) {
      if (start >= offset + life.bytes) {
        break;
      }
      offset = std::max(offset, end);
    }
    life.offset = offset;
    life.placed = true;
    placement.extent = std::max(placement.extent, offset + life.bytes);
  }
  placement.offsets.reserve(lives.size());
  for (const Life& life : lives) {
    placement.offsets.emplace_back(life.tensor, life.offset);
  }
  return placement;
}

void BindViews(std::span<ggml_tensor* const> nodes) {
  for (ggml_tensor* node : nodes) {
    if (node->view_src != nullptr) {
      TensorArena::Bind(node,
                        reinterpret_cast<std::uintptr_t>(node->view_src->data) + node->view_offs);
    }
  }
}

void BindDistinct(GraphNodes graph, std::uint64_t base) {
  std::uint64_t next = base;
  for (ggml_tensor* node : graph) {
    if (Computed(node)) {
      TensorArena::Bind(node, next);
      next += Rounded(ggml_nbytes(node), 256) + 256;
    }
  }
  BindViews(graph);
}

}  // namespace jitllm::kernels::ggml
