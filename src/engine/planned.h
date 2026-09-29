// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A model's planned shapes (docs/engine.md): the graph of one chunk (or
// drafter pass) shape built in its own tensor arena, planned, its computed
// tensors placed in the node's activations and planned again (which must
// give the same plan), its scratch checked against the GGML pool and its
// implementations bound against the registry (D-053); the cache of them a
// runner keeps per shape with each plan's runs (graph_runs.h); the cap on
// a model's graphs (D-090: graph memory is the driver's, outside the
// catalog); and BP-A1's in-process check that every tensor a plan binds
// lies in cataloged, resident memory of the class it should.
//
// The model's graph builder, binding and inputs stay the model's own
// (dsv4_plan.h, qwen38_plan.h); only the mechanics are here.

#ifndef JITLLM_ENGINE_PLANNED_H_
#define JITLLM_ENGINE_PLANNED_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine/graph_runs.h"
#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::engine {

class PagedNode;

// What every planned shape holds besides its graph: its arena (which owns
// the graph's tensors, so it outlives them), plan, placement, bound
// implementations, and the pool scratch and staged input bytes it needs.
struct PlannedBase {
  std::optional<kernels::ggml::TensorArena> arena;
  kernels::ggml::GraphPlan plan;
  kernels::ggml::Placement placement;
  std::optional<kernels::ggml::BoundGraph> bound;
  std::uint64_t scratch = 0;
  std::uint64_t inputs_bytes = 0;  // as GraphRuns::Stage stages them
};

// A planned shape of a model's graph type (kernels/ggml/*_graph.h).
template <typename Graph>
struct PlannedGraph : PlannedBase {
  Graph graph;
};

// Places a graph's computed tensors: `inputs` bound at distinct placeless
// addresses and every computed node at its own, planned; the activations
// placed (`keep` live to the end); then, unless `activations` is 0 (measure
// only), bound in [activations, + activation_bytes), planned again, which
// must give the same plan.
std::expected<void, std::string> PlaceAndPlan(PlannedBase& out, std::span<ggml_tensor* const> nodes,
                                              std::span<ggml_tensor* const> inputs,
                                              std::span<ggml_tensor* const> keep,
                                              const kernels::ggml::DeviceChoices& choices,
                                              std::uint64_t activations,
                                              std::uint64_t activation_bytes);

// The scratch `planned`'s plan needs, checked against the launch context's
// pool, and its implementations bound against `registry` (D-053). `what`
// names the plan in a refusal ("the plan", "the draft").
std::expected<void, std::string> BindPlanned(PlannedBase& planned,
                                             kernels::ggml::LaunchContext& launch,
                                             const execution::Registry& registry,
                                             std::string_view what);

// A model's planned shapes, by key (a shape and what the model computes
// beside it), each with its runs: `Variants` of them where one plan runs
// with different outputs (a Qwen3.8 verify with and without its logits'
// copy). At most `capacity` shapes are kept, the oldest dropped with its
// graphs. Entries stay put until the next Add or Clear.
template <typename Key, typename Planned, std::size_t Variants = 1>
class PlanCache {
 public:
  struct Entry {
    Key key{};
    std::unique_ptr<Planned> planned;
    std::array<PlanRuns, Variants> runs;
  };

  explicit PlanCache(std::size_t capacity) : capacity_(capacity) {}

  Entry* Find(const Key& key) {
    for (Entry& e : entries_) {
      if (e.key == key) {
        return &e;
      }
    }
    return nullptr;
  }
  Entry& Add(Key key, std::unique_ptr<Planned> planned) {
    if (entries_.size() >= capacity_) {
      entries_.erase(entries_.begin());  // its graphs with it
    }
    Entry& entry = entries_.emplace_back();
    entry.key = std::move(key);
    entry.planned = std::move(planned);
    return entry;
  }
  // Destroys every plan and graph (before the launch context).
  void Clear() { entries_.clear(); }
  std::size_t size() const { return entries_.size(); }
  // Graphs kept, every variant's.
  std::size_t graphs() const {
    std::size_t n = 0;
    for (const Entry& e : entries_) {
      for (const PlanRuns& r : e.runs) {
        n += r.graph.has_value() ? 1 : 0;
      }
    }
    return n;
  }
  // The oldest entry's first variant with a graph, or null.
  PlanRuns* OldestGraph() {
    for (Entry& e : entries_) {
      for (PlanRuns& r : e.runs) {
        if (r.graph.has_value()) {
          return &r;
        }
      }
    }
    return nullptr;
  }

 private:
  std::size_t capacity_;
  std::vector<Entry> entries_;
};

// Before a capture: with `most` graphs kept across `caches`, the oldest
// graph of the first cache that has one goes (between jobs: nothing in
// flight replays it), counted in `stats`.
template <typename... Caches>
void RoomForGraph(std::size_t most, GraphStats& stats, Caches&... caches) {
  if ((caches.graphs() + ...) < most) {
    return;
  }
  PlanRuns* victim = nullptr;
  ((victim = victim != nullptr ? victim : caches.OldestGraph()), ...);
  if (victim != nullptr) {
    victim->DropGraph();
    ++stats.dropped;
  }
}

// BP-A1's in-process check, once per planned shape: each tensor a plan
// binds (every node and its sources) should lie in cataloged, resident
// device memory of one class, the owner's or shared (PagedNode::Covered):
// live state for `state`, runtime for `runtime`, scratch for `scratch`,
// the inputs and every computed tensor; any other leaf is a weight.
// Initialized so callers may designate only the fields they use.
// NOLINTBEGIN(readability-redundant-member-init)
struct TensorClasses {
  std::span<const ggml_tensor* const> state = {};
  std::span<const ggml_tensor* const> runtime = {};
  std::span<const ggml_tensor* const> scratch = {};
  std::span<ggml_tensor* const> inputs = {};
  // A fill's sources only shape it: not checked (Qwen3.8's QSA zeros).
  bool fill_reads_nothing = false;
  std::string_view what = {};  // the plan, in the first violation ("", "the drafter's ")
};
// NOLINTEND(readability-redundant-member-init)
struct Coverage {
  std::uint64_t tensors = 0;
  std::uint64_t violations = 0;
  std::string first_violation;
};
void CheckCoverage(const PagedNode& node, int owner, std::span<ggml_tensor* const> nodes,
                   const TensorClasses& classes, Coverage& coverage);

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_PLANNED_H_
