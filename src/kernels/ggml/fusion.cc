// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

#include "ggml.h"

namespace jitllm::kernels::ggml {
namespace {

// MATRIX_ROW_PADDING (ggml-cuda/common.cuh:186): the CUDA buffer type
// pads a quantized tensor's last row to a multiple of this many elements.
constexpr std::int64_t kMatrixRowPadding = 512;

// ggml_backend_cuda_buffer_type_get_alloc_size (ggml-cuda.cu:908-925) for
// the operations here: the tensor's bytes, and a quantized tensor's row
// padding. (It measures flash attention's output differently; no gate here
// sees one.)
std::int64_t AllocSize(const ggml_tensor* tensor) {
  auto size = static_cast<std::int64_t>(ggml_nbytes(tensor));
  const std::int64_t ne0 = tensor->ne[0];
  if (ggml_is_quantized(tensor->type) && ne0 % kMatrixRowPadding != 0) {
    size += static_cast<std::int64_t>(
        ggml_row_size(tensor->type, kMatrixRowPadding - (ne0 % kMatrixRowPadding)));
  }
  return size;
}

// nodes_overlap in ggml_cuda_check_fusion_memory_ranges, in upstream's
// signed arithmetic.
bool Overlaps(const ggml_tensor* a, const ggml_tensor* b) {
  const auto a_start = static_cast<std::int64_t>(reinterpret_cast<std::intptr_t>(a->data));
  const std::int64_t a_end = a_start + AllocSize(a);
  const auto b_start = static_cast<std::int64_t>(reinterpret_cast<std::intptr_t>(b->data));
  const std::int64_t b_end = b_start + AllocSize(b);
  return (b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end);
}

bool InGraph(GraphNodes graph, std::size_t index, std::size_t count) {
  return count > 0 && index <= graph.size() && count <= graph.size() - index;
}

// How many inputs of the graph's nodes are `tensor`: GGML's use count
// (ggml_visit_parents_graph counts one per input edge, ggml.c:7243-7255).
int Uses(GraphNodes graph, const ggml_tensor* tensor) {
  int uses = 0;
  for (const ggml_tensor* node : graph) {
    for (const ggml_tensor* src : node->src) {
      uses += src == tensor ? 1 : 0;
    }
  }
  return uses;
}

bool Among(std::span<ggml_tensor* const> nodes, const ggml_tensor* tensor) {
  return std::ranges::find(nodes, tensor) != nodes.end();
}

// ggml_can_fuse_ext for consecutive nodes (ggml-impl.h:653-709): every node
// but the last has exactly one use, is no view and no output, and is an
// input of the next, which has its shape. (Every node listed is computed.)
bool CanFuse(GraphNodes graph, std::size_t index, std::initializer_list<ggml_op> ops) {
  if (!InGraph(graph, index, ops.size())) {
    return false;
  }
  std::size_t i = 0;
  for (const ggml_op op : ops) {
    const ggml_tensor* node = graph[index + i];
    if (node->op != op) {
      return false;
    }
    if (i + 1 < ops.size() && (Uses(graph, node) != 1 || node->view_src != nullptr ||
                               (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0)) {
      return false;
    }
    if (i > 0) {
      const ggml_tensor* prev = graph[index + i - 1];
      if ((node->src[0] != prev && node->src[1] != prev) || !ggml_are_same_shape(node, prev)) {
        return false;
      }
    }
    ++i;
  }
  return true;
}

// ggml_can_fuse_subgraph_ext for consecutive nodes with one output
// (ggml.c:7737-7794): each other node is no output, all its uses are by
// later nodes of the subgraph, and any view's sources lie in it (upstream
// also allows constant weights, which no tensor without a buffer is).
bool CanFuseSubgraph(GraphNodes graph, std::size_t index, std::initializer_list<ggml_op> ops,
                     std::size_t output) {
  if (!InGraph(graph, index, ops.size())) {
    return false;
  }
  const auto subgraph = graph.subspan(index, ops.size());
  std::size_t i = 0;
  for (const ggml_op op : ops) {
    const ggml_tensor* node = subgraph[i];
    if (node->op != op) {
      return false;
    }
    if (index + i != output) {
      if ((node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
          Uses(subgraph.subspan(i + 1), node) != Uses(graph, node)) {
        return false;
      }
      for (const ggml_tensor* source = node->view_src; source != nullptr;
           source = source->view_src) {
        if (!Among(subgraph, source)) {
          return false;
        }
      }
    }
    ++i;
  }
  return true;
}

// ggml_cuda_should_fuse_mul_mat (ggml-cuda.cu:1673-1765) for a gate and an
// up product with no bias and no scale.
bool ShouldFuseMulMat(const ggml_tensor* up, const ggml_tensor* gate, const ggml_tensor* glu) {
  if (up->op != GGML_OP_MUL_MAT || gate->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU) {
    return false;
  }
  if (glu->src[0] != gate || glu->src[1] != up) {
    return false;
  }
  if (up->src[0]->type != gate->src[0]->type || !ggml_are_same_shape(up->src[0], gate->src[0]) ||
      !ggml_are_same_stride(up->src[0], gate->src[0])) {
    return false;
  }
  if (up->src[1] != gate->src[1]) {
    return false;
  }
  const ggml_glu_op op = ggml_get_glu_op(glu);
  if (op != GGML_GLU_OP_SWIGLU && op != GGML_GLU_OP_GEGLU && op != GGML_GLU_OP_SWIGLU_OAI &&
      op != GGML_GLU_OP_SWIGLU_CLAMP) {
    return false;
  }
  // Swapped (op_params[1]).
  return glu->op_params[1] == 0;
}

// ggml_cuda_should_fuse_rope_set_rows (ggml-cuda.cu:2666-2698).
bool ShouldFuseRopeSetRows(const ggml_tensor* rope, const ggml_tensor* view,
                           const ggml_tensor* set_rows) {
  if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
    return false;
  }
  if (rope->src[0]->ne[3] != 1) {
    return false;
  }
  if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
    return false;
  }
  if (set_rows->src[1]->type != GGML_TYPE_I64) {
    return false;
  }
  if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
    return false;
  }
  const std::int32_t mode = rope->op_params[2];
  return mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX;
}

void Visit(ggml_tensor* tensor, std::vector<ggml_tensor*>& seen, std::vector<ggml_tensor*>& nodes) {
  if (Among(seen, tensor)) {
    return;
  }
  seen.push_back(tensor);
  for (ggml_tensor* src : tensor->src) {
    if (src != nullptr) {
      Visit(src, seen, nodes);
    }
  }
  if (tensor->op != GGML_OP_NONE || (tensor->flags & GGML_TENSOR_FLAG_PARAM) != 0) {
    nodes.push_back(tensor);
  }
}

}  // namespace

std::vector<ggml_tensor*> GraphOrder(std::span<ggml_tensor* const> outputs) {
  std::vector<ggml_tensor*> seen;
  std::vector<ggml_tensor*> nodes;
  for (ggml_tensor* output : outputs) {
    Visit(output, seen, nodes);
  }
  return nodes;
}

bool FusionMemoryClear(GraphNodes graph, std::size_t index, std::size_t count, std::size_t output) {
  if (!InGraph(graph, index, count) || output >= graph.size()) {
    return false;
  }
  const ggml_tensor* dst = graph[output];
  for (std::size_t j = index; j < index + count; ++j) {
    for (const ggml_tensor* src : graph[j]->src) {
      if (src == nullptr || src->op == GGML_OP_NONE || !Overlaps(dst, src)) {
        continue;
      }
      // An overlapping input must be one of the fused nodes before this one.
      if (!Among(graph.subspan(index, j - index), src)) {
        return false;
      }
    }
  }
  return true;
}

std::optional<MulMatGluNodes> MulMatGluFusionAt(GraphNodes graph, std::size_t index) {
  // ggml_cuda_can_fuse: the gate product at index, the up product next.
  if (!CanFuseSubgraph(graph, index, {GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU}, index + 2)) {
    return std::nullopt;
  }
  ggml_tensor* gate = graph[index];
  ggml_tensor* up = graph[index + 1];
  ggml_tensor* glu = graph[index + 2];
  if (!ShouldFuseMulMat(up, gate, glu) || !FusionMemoryClear(graph, index, 3, index + 2)) {
    return std::nullopt;
  }
  // ggml_cuda_try_fuse then takes the gate and up products from the GLU,
  // which the gate above has already tied to these nodes.
  return MulMatGluNodes{.gate = gate, .up = up, .glu = glu};
}

std::optional<MulMatAddNodes> MulMatAddFusionAt(GraphNodes graph, std::size_t index) {
  if (!CanFuse(graph, index, {GGML_OP_MUL_MAT, GGML_OP_ADD})) {
    return std::nullopt;
  }
  ggml_tensor* mul_mat = graph[index];
  ggml_tensor* add = graph[index + 1];
  // Upstream fuses no broadcast add.
  if (!ggml_are_same_shape(add->src[0], add->src[1])) {
    return std::nullopt;
  }
  return MulMatAddNodes{.mul_mat = mul_mat, .add = add};
}

std::optional<RopeSetRowsNodes> RopeSetRowsFusionAt(GraphNodes graph, std::size_t index) {
  if (!CanFuseSubgraph(graph, index, {GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS}, index + 2)) {
    return std::nullopt;
  }
  ggml_tensor* rope = graph[index];
  ggml_tensor* view = graph[index + 1];
  ggml_tensor* set_rows = graph[index + 2];
  if (!ShouldFuseRopeSetRows(rope, view, set_rows) ||
      !FusionMemoryClear(graph, index, 3, index + 2)) {
    return std::nullopt;
  }
  return RopeSetRowsNodes{.rope = rope, .view = view, .set_rows = set_rows};
}

}  // namespace jitllm::kernels::ggml
