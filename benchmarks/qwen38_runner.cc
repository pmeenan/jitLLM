// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen38_runner.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "artifact/layout.h"
#include "ggml.h"
#include "kernels/ggml/dsv4_graph.h"  // GgmlTypeOf
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"

namespace jitllm::benchmarks {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace sc = jitllm::scheduler;
namespace ts = jitllm::test_support;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using catalog::Recovery;
using providers::BackingKind;
using Status = test_support::Status;

constexpr std::uint64_t kExtent = test_support::kPagedExtent;
constexpr std::size_t kRingDepth = 32;
// The slabs' offset in their first page: the stride's own alignment (16),
// since the 80-byte gap between Qwen3.8's expert groups cannot hold 256.
constexpr std::uint64_t kSlabAlignment = 16;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

}  // namespace

Qwen38Runner::~Qwen38Runner() = default;

std::vector<ExtentId> Qwen38Runner::managed_extents() const {
  std::vector<ExtentId> all = weights_.extents();
  all.insert(all.end(), state_.extents.begin(), state_.extents.end());
  return all;
}

Status Qwen38Runner::Setup() {
  auto artifact = artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<artifact::Artifact>(std::move(*artifact));
  auto binding = md::BindQwen38(profile_, *artifact_);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  auto layout = md::Qwen38State(profile_, o_.context, o_.max_rows);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  layout_ = std::move(*layout);
  for (std::uint32_t s = 0; s < artifact_->shards().size(); ++s) {
    auto fd = artifact_->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards_.push_back(std::move(*fd));
  }
  id_ = ArtifactKey(*artifact_);

  // The n-gram table: its group alone, stored contiguously in one shard.
  const auto groups = artifact_->groups();
  const artifact::Resource& table = artifact_->resources()[binding_.ple_table.index];
  const std::uint32_t table_group = table.group;
  for (std::uint32_t r = 0; r < artifact_->resources().size(); ++r) {
    if (r != binding_.ple_table.index && artifact_->resources()[r].group == table_group) {
      return Error(
          std::format("{} shares the n-gram table's group", artifact_->resources()[r].name));
    }
  }
  const auto first =
      artifact::ChunkRangeOf(artifact_->layout(), {.group = table_group, .chunk = 0});
  if (!first) {
    return Error("the n-gram table's file range");
  }
  for (std::uint32_t c = 1; c < groups[table_group].chunks; ++c) {
    const auto range =
        artifact::ChunkRangeOf(artifact_->layout(), {.group = table_group, .chunk = c});
    if (!range || range->shard != first->shard ||
        range->file_offset.value() != first->file_offset.value() + (std::uint64_t{c} * kExtent)) {
      return Error("the n-gram table is not stored contiguously in one shard");
    }
  }
  if (binding_.ple_table.ne.size() != 2) {
    return Error("the n-gram table is not a table of rows");
  }
  table_ = PleTable{.fd = shards_.at(first->shard).get(),
                    .file_offset = first->file_offset.value() + table.offset.value(),
                    .rows = binding_.ple_table.ne[1],
                    .row_bytes = binding_.ple_table.ne[0],
                    .chunk_file_offset = first->file_offset.value(),
                    .file_bytes = first->file_offset.value() + groups[table_group].stored.value()};
  slots_ = std::uint64_t{o_.max_rows} * profile_.ple_heads();
  graph_binding_ = binding_;
  graph_binding_.ple_table.ne[1] = slots_;

  // The state first: its extents come before the weights' in a closure, so
  // a swap back restores it before paging the weights in.
  if (auto r = node_.MapResident(state_, "the Qwen3.8 state", layout_.bytes, BackingKind::kDevice,
                                 MemoryClass::kLiveState, Recovery::kPreserve, owner_);
      !r) {
    return r;
  }
  if (auto r =
          node_.MapResident(slot_memory_, "the Qwen3.8 n-gram row slots", slots_ * table_.row_bytes,
                            BackingKind::kDevice, MemoryClass::kScratch, Recovery::kPinned, owner_);
      !r) {
    return r;
  }
  if (auto r = ReserveWeights(); !r) {
    return r;
  }

  int major = 0;
  int minor = 0;
  (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  cublas_bytes_ = kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor)).value();
  if (auto r =
          node_.MapResident(cublas_workspace_, "the Qwen3.8 cuBLAS workspace", cublas_bytes_,
                            BackingKind::kDevice, MemoryClass::kRuntime, Recovery::kPinned, owner_);
      !r) {
    return r;
  }
  auto cublas =
      kg::CublasHandle::Create(0, node_.execution(), node_.stream(stream_),
                               {.base = cublas_workspace_.base, .size = Bytes(cublas_bytes_)});
  if (!cublas) {
    return Error(cublas.error().detail);
  }
  cublas_ = std::move(*cublas);

  // The largest shapes, as the resident harness sizes them (one output
  // row: the runner reads only the last row's logits), planned over
  // placeless addresses with a stand-in hash (the rows do not shape a
  // chunk).
  std::vector<std::uint64_t> stride = std::move(model_.places.stride);  // ReserveWeights's
  model_ = Qwen38Model{.artifact = artifact_.get(),
                       .profile = &profile_,
                       .binding = &graph_binding_,
                       .state = &layout_,
                       .places = {.resource = [](std::uint32_t) { return std::uint64_t{1} << 44U; },
                                  .array = [](std::uint32_t) { return std::uint64_t{1} << 44U; },
                                  .stride = std::move(stride),
                                  .state = std::uint64_t{1} << 45U,
                                  .ple_table = std::uint64_t{1} << 44U},
                       .cutlass = binding_.cutlass()};
  md::Qwen38PleHash stand_in;
  stand_in.multipliers.assign(profile_.ngram, 1);
  stand_in.offsets.assign(profile_.ple_heads(), 0);
  stand_in.vocab.assign(profile_.ple_heads(), 1);
  stand_in.table_rows = 1;
  std::uint64_t most_activations = 0;
  std::uint64_t most_scratch = 0;
  std::uint64_t most_inputs = 0;
  {
    auto measure = kg::LaunchContext::Create(0, node_.execution(), node_.stream(stream_),
                                             {.base = 0, .size = Bytes(0)}, cublas_.get());
    if (!measure) {
      return Error(measure.error().detail);
    }
    const kg::DeviceChoices choices = kg::DeviceChoicesOf(**measure);
    const std::array<std::pair<std::uint32_t, std::uint32_t>, 4> probes = {
        {{0, o_.max_rows}, {o_.context - o_.max_rows, o_.max_rows}, {o_.context - 1, 1}, {0, 1}}};
    for (const auto& [n_past, rows] : probes) {
      std::vector<std::int32_t> history(std::size_t{n_past} + rows, 1000);
      auto in = md::Qwen38Chunk(profile_, layout_, stand_in, history, n_past, rows);
      if (!in) {
        return std::unexpected(in.error());
      }
      auto planned = PlanQwen38Chunk(model_, kg::Qwen38ShapeOf(layout_, *in, 1), choices, 0, 0);
      if (!planned) {
        return Error(
            std::format("measuring a chunk of {} at {}: {}", rows, n_past, planned.error()));
      }
      most_activations = std::max(most_activations, (*planned)->placement.extent);
      auto scratch = kg::PlanScratch(**measure, (*planned)->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, (*planned)->inputs_bytes);
    }
  }
  activation_bytes_ = Round(most_activations + (most_activations / 4), kExtent);
  scratch_bytes_ = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  input_bytes_ = Round((most_inputs * 2) + (1U << 20U), kExtent);

  landing_bytes_ = PleLandingBound(slots_);
  auto inputs = node_.Pinned(input_bytes_, owner_, staging_);
  auto logits = node_.Pinned(std::uint64_t{profile_.vocab} * sizeof(float), owner_, staging_);
  auto hash =
      node_.Pinned((std::uint64_t{profile_.ngram} + (2 * std::uint64_t{profile_.ple_heads()})) * 8,
                   owner_, staging_);
  auto landing = node_.Pinned(landing_bytes_, owner_, staging_);
  auto sources = node_.Pinned(slots_ * sizeof(std::uint32_t), owner_, staging_);
  unwritten_ = weights_.Unwritten();
  auto scrub = node_.Pinned(unwritten_.size() * 2 * sizeof(std::uint64_t), owner_, staging_);
  if (!inputs || !logits || !hash || !landing || !sources || !scrub) {
    return Error("pinned staging for Qwen3.8");
  }
  scrub_ = static_cast<std::uint64_t*>(*scrub);
  inputs_ = *inputs;
  logits_ = *logits;
  hash_host_ = *hash;
  landing_ = static_cast<std::byte*>(*landing);
  sources_ = static_cast<std::uint32_t*>(*sources);
  if (Address(landing_) % kPleBlock != 0) {
    return Error("the n-gram rows' landing is not 4 KiB-aligned for direct reads");
  }
  auto ring = providers::UringStorage::Create(kRingDepth);
  if (!ring) {
    return Error(std::format("the n-gram rows' ring: {}", ring.error().message()));
  }
  ring_ = std::move(*ring);
  return {};
}

// Every dense group but the n-gram table's, and a slab per layer.
Status Qwen38Runner::ReserveWeights() {
  const auto groups = artifact_->groups();
  std::vector<bool> place(groups.size(), false);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    place[g] = groups[g].kind != artifact::GroupKind::kExpert;
  }
  place[artifact_->resources()[binding_.ple_table.index].group] = false;
  std::vector<SlabSpec> slabs;
  model_.places.stride.assign(profile_.layers, 0);
  for (std::uint32_t il = 0; il < profile_.layers; ++il) {
    const md::Qwen38Layer& l = binding_.layers[il];
    std::uint64_t unit = 16;
    std::optional<std::uint32_t> first_group;
    for (const md::Qwen38Tensor* t : l.expert_arrays(binding_.cutlass())) {
      const auto& a = artifact_->expert_arrays()[t->index];
      auto type = kg::GgmlTypeOf(t->type);
      if (!type) {
        return Error(type.error().detail);
      }
      unit = std::lcm(unit, static_cast<std::uint64_t>(ggml_type_size(*type)));
      if (first_group && *first_group != a.first_group) {
        return Error(std::format("layer {}'s expert arrays do not share their groups", il));
      }
      first_group = a.first_group;
    }
    const std::uint32_t g = first_group.value_or(0);
    const std::uint64_t stride = Round(groups[g].stored.value(), unit);
    model_.places.stride[il] = stride;
    slabs.push_back({.first_group = g,
                     .count = profile_.experts,
                     .stride = stride,
                     .alignment = kSlabAlignment});
  }
  return weights_.Reserve(node_, *artifact_, shards_, id_, place, slabs);
}

Status Qwen38Runner::Register() {
  if (auto r = weights_.Register(node_, owner_); !r) {
    return r;
  }
  if (auto r = RegisterState(); !r) {
    return r;
  }
  // D-090, for every model: the places stay put for the model's life
  // (never unpinned; the pins go with the scheduler).
  if (auto pinned = node_.scheduler().PinPlaces(managed_extents()); !pinned) {
    return Error(std::format("pinning Qwen3.8's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

// The state's write-back places: one 2 MiB range of an unnamed direct-I/O
// spill file per extent, landed through the zone, its backing managed.
Status Qwen38Runner::RegisterState() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(),
                             std::generic_category().message(errno)));
  }
  for (std::size_t i = 0; i < state_.extents.size(); ++i) {
    auto set = node_.scheduler().SetSource(
        state_.extents[i],
        sc::PageSource{
            .read = {.fd = spill_fd_, .offset = i * kExtent, .memory = nullptr, .length = kExtent},
            .landed = true,
            .destination = state_.base + (i * kExtent),
            .backing = sc::BackingPlace{.reservation = state_.reservation,
                                        .offset = Bytes(i * kExtent),
                                        .size = Bytes(kExtent),
                                        .allocation_class = node_.device_class()},
            .write_back = true});
    if (!set) {
      return Error(std::format("the state's write-back place: {}", sc::ToString(set.error())));
    }
  }
  state_.backings.clear();  // the VMM lane releases them on eviction (D-033)
  return {};
}

Status Qwen38Runner::Bind() {
  auto& catalog = node_.catalog();
  std::vector<ExtentId> all = weights_.extents();
  for (const ts::Mapped* mapped : std::initializer_list<const ts::Mapped*>{
           &state_, &slot_memory_, &node_.activations(), &node_.pool(), &cublas_workspace_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  everything_ = catalog.ClosureOfExtents(all).value();
  fence_ = catalog.ClosureOfExtents(state_.extents).value();
  model_.places.resource = [this](std::uint32_t resource) {
    const auto& r = artifact_->resources()[resource];
    return weights_.group_address(r.group) + r.offset.value();
  };
  model_.places.array = [this](std::uint32_t array) {
    const auto& a = artifact_->expert_arrays()[array];
    return weights_.group_address(a.first_group) + a.group_offset.value();
  };
  model_.places.state = state_.base;
  model_.places.ple_table = slot_memory_.base;
  auto launch = kg::LaunchContext::Create(
      0, node_.execution(), node_.stream(stream_),
      {.base = node_.pool().base, .size = Bytes(scratch_bytes_)}, cublas_.get());
  if (!launch) {
    return Error(launch.error().detail);
  }
  launch_ = std::move(*launch);
  auto registry = execution::Registry::Create(kg::Implementations());
  if (!registry) {
    return Error(registry.error().detail);
  }
  registry_ = std::make_unique<execution::Registry>(std::move(*registry));
  return {};
}

// ------------------------------------------------------------------ work

Status Qwen38Runner::ReadPleHash() {
  const md::Qwen38Layer& l = binding_.layers[profile_.ple_layer];
  const std::array<const md::Qwen38Tensor*, 3> parts = {&l.ple_multipliers, &l.ple_head_offsets,
                                                        &l.ple_head_vocab};
  std::vector<std::pair<std::uint64_t, std::uint64_t>> copies;  // address, bytes
  copies.reserve(parts.size());
  for (const md::Qwen38Tensor* t : parts) {
    copies.emplace_back(model_.places.resource(t->index), t->ne[0] * sizeof(std::int64_t));
  }
  void* host = hash_host_;
  if (auto r = node_.Job(
          everything_,
          [&copies, host](providers::NativeStream stream) {
            std::uint64_t at = 0;
            for (const auto& [address, bytes] : copies) {
              if (cudaMemcpyAsync(static_cast<std::byte*>(host) + at, Pointer(address), bytes,
                                  cudaMemcpyDeviceToHost,
                                  static_cast<cudaStream_t>(stream.handle)) != cudaSuccess) {
                return sc::JobResult::kUnknown;
              }
              at += bytes;
            }
            return sc::JobResult::kQueued;
          },
          "reading the n-gram hash", stream_);
      !r) {
    return r;
  }
  const auto* values = static_cast<const std::int64_t*>(host);
  const std::span<const std::int64_t> m(values, profile_.ngram);
  const std::span<const std::int64_t> o(values + profile_.ngram, profile_.ple_heads());
  const std::span<const std::int64_t> v(values + profile_.ngram + profile_.ple_heads(),
                                        profile_.ple_heads());
  auto hash = md::CheckQwen38PleHash(profile_, m, o, v, table_.rows);
  if (!hash) {
    return std::unexpected(hash.error());
  }
  hash_ = std::move(*hash);
  hash_checked_ = true;
  return {};
}

Status Qwen38Runner::Scrub(std::uint8_t value, bool slabs, bool dense) {
  std::uint32_t count = 0;
  for (const PagedWeights::Range& r : unwritten_) {
    if (r.slab ? slabs : dense) {
      scrub_[2 * std::size_t{count}] = r.address;
      scrub_[(2 * std::size_t{count}) + 1] = r.bytes;
      ++count;
    }
  }
  const std::uint64_t* ranges = scrub_;
  return node_.Job(
      everything_,
      [ranges, count, value](providers::NativeStream stream) {
        return FillRanges(ranges, count, value, stream.handle) ? sc::JobResult::kQueued
                                                               : sc::JobResult::kUnknown;
      },
      "filling the weights' unwritten bytes", stream_);
}

Status Qwen38Runner::Clear() {
  const std::uint64_t base = state_.base;
  const std::uint64_t bytes = layout_.bytes;
  return node_.Job(
      fence_,
      [base, bytes](providers::NativeStream stream) {
        return cudaMemsetAsync(Pointer(base), 0, bytes, static_cast<cudaStream_t>(stream.handle)) ==
                       cudaSuccess
                   ? sc::JobResult::kQueued
                   : sc::JobResult::kUnknown;
      },
      "clearing the Qwen3.8 state", stream_);
}

std::expected<Qwen38Planned*, std::string> Qwen38Runner::Planned(
    const kg::Qwen38ChunkShape& shape) {
  const auto found = std::ranges::find_if(plans_, [&](const auto& e) { return e.first == shape; });
  if (found != plans_.end()) {
    return found->second.get();
  }
  const auto start = std::chrono::steady_clock::now();
  auto planned = PlanQwen38Chunk(model_, shape, kg::DeviceChoicesOf(*launch_),
                                 node_.activations().base, node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  auto scratch = kg::PlanScratch(*launch_, (*planned)->plan);
  if (!scratch) {
    return Error(scratch.error().detail);
  }
  if (*scratch > launch_->workspace().size.value()) {
    return Error(std::format("the plan's scratch ({} bytes) exceeds the pool ({} bytes)", *scratch,
                             launch_->workspace().size.value()));
  }
  (*planned)->scratch = *scratch;
  auto bound = kg::BoundGraph::Bind(*registry_, (*planned)->plan);
  if (!bound) {
    return Error(bound.error().detail);
  }
  (*planned)->bound.emplace(std::move(*bound));
  Check((*planned)->graph);
  plan_seconds_ += Seconds(std::chrono::steady_clock::now() - start);
  if (plans_.size() >= 32) {
    plans_.erase(plans_.begin());
  }
  plans_.emplace_back(shape, std::move(*planned));
  return plans_.back().second.get();
}

// BP-A1's in-process check, once per planned shape: every tensor the plan
// binds lies in cataloged, resident extents of device memory of one class,
// and each has the class it should: weights, the state (live state), the
// row slots and the activations (scratch).
void Qwen38Runner::Check(const kg::Qwen38Graph& graph) {
  std::vector<const ggml_tensor*> state;
  for (const kg::Qwen38LayerTensors& l : graph.layers) {
    for (const ggml_tensor* t :
         {l.cache_k, l.cache_v, l.cache_idx, l.conv_state, l.recurrent, l.ple_state}) {
      if (t != nullptr) {
        state.push_back(t);
      }
    }
  }
  const auto inputs = graph.inputs();
  const auto kind_of = [&](const ggml_tensor* t) {
    const ggml_tensor* base = t->view_src != nullptr ? t->view_src : t;
    if (std::ranges::find(state, base) != state.end()) {
      return MemoryClass::kLiveState;
    }
    if (base == graph.ple_table) {
      return MemoryClass::kScratch;  // the row slots
    }
    return base->op == GGML_OP_NONE && std::ranges::find(inputs, base) == inputs.end()
               ? MemoryClass::kWeights
               : MemoryClass::kScratch;
  };
  const auto expect = [&](const ggml_tensor* t, const ggml_tensor* consumer) {
    ++coverage_tensors_;
    const std::optional<MemoryClass> covered =
        node_.Covered(Address(t->data), ggml_nbytes(t), owner_);
    if (covered != kind_of(t) && coverage_violations_++ == 0) {
      first_violation_ = std::format(
          "{} ({} {} [{}, {}, {}, {}], {} bytes at {:#x}, read by {} {})", t->name, ggml_op_desc(t),
          ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_nbytes(t),
          Address(t->data), consumer != nullptr ? ggml_op_desc(consumer) : "-",
          consumer != nullptr ? consumer->name : "");
    }
  };
  for (const ggml_tensor* node : graph.nodes) {
    expect(node, nullptr);
    if (node->op == GGML_OP_FILL) {
      continue;  // its source only shapes it: a fill reads nothing (QSA's zeros)
    }
    for (const ggml_tensor* src : node->src) {
      if (src != nullptr) {
        expect(src, node);
      }
    }
  }
}

Status Qwen38Runner::Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
                           std::vector<float>& logits) {
  if (!hash_checked_) {
    return Error("the n-gram hash is not checked since the last load");
  }
  if (rows_stalled_) {
    return Error("the n-gram rows' reads stalled earlier; their landing may still be written");
  }
  if (history.size() <= n_past) {
    return Error("an empty chunk");
  }
  const auto rows = static_cast<std::uint32_t>(history.size() - n_past);
  auto in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows);
  if (!in) {
    return std::unexpected(in.error());
  }
  // The chunk's n-gram rows: planned, read and their slots' sources set,
  // while no job of this model holds the landing (the last one's fence
  // has completed: Job returns only after it).
  const auto reading = std::chrono::steady_clock::now();
  auto rows_plan = PlanPleRows(table_, in->ple_rows, landing_bytes_, slots_);
  if (!rows_plan) {
    return std::unexpected(rows_plan.error());
  }
  if (auto r = ReadPleRows(*ring_, table_.fd, *rows_plan, landing_); !r) {
    rows_stalled_ = ring_->in_flight() != 0;  // reads that may still land
    return r;
  }
  std::ranges::copy(rows_plan->sources, sources_);
  ple_.chunks += 1;
  ple_.lookups += in->ple_rows.size();
  ple_.rows += rows_plan->sources.size();
  ple_.reads += rows_plan->reads.size();
  ple_.read_bytes += rows_plan->landing_bytes;
  ple_.useful_bytes += rows_plan->useful_bytes;
  ple_.extent_bytes += rows_plan->extents * kExtent;
  ple_.seconds += Seconds(std::chrono::steady_clock::now() - reading);

  auto planned = Planned(kg::Qwen38ShapeOf(layout_, *in, 1));
  if (!planned) {
    return std::unexpected(planned.error());
  }
  Qwen38Planned* p = *planned;
  const kg::Qwen38Graph& g = p->graph;
  Qwen38HostInputs host;
  Qwen38Sources(g, *in, 1, rows_plan->slots, host);
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  const auto count = static_cast<std::uint32_t>(rows_plan->sources.size());
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    auto* const stream = static_cast<cudaStream_t>(native.handle);
    std::uint64_t staged = 0;
    for (const auto& [tensor, source] : host.sources) {
      const std::uint64_t bytes = ggml_nbytes(tensor);
      if (staged + bytes > input_bytes_) {
        ran = Error("the inputs exceed their staging");
        return staged == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kFailed;
      }
      auto* at = static_cast<std::byte*>(inputs_) + staged;
      std::memcpy(at, source, bytes);
      if (auto r = Cuda(cudaMemcpyAsync(tensor->data, at, bytes, cudaMemcpyHostToDevice, stream),
                        "an input copy");
          !r) {
        ran = r;
        return sc::JobResult::kUnknown;
      }
      staged += Round(bytes, 256);
    }
    if (!GatherPleRows(landing_, sources_, count, static_cast<std::uint32_t>(table_.row_bytes),
                       static_cast<std::byte*>(Pointer(slot_memory_.base)), stream)) {
      ran = Error("the n-gram rows' gather");
      return sc::JobResult::kUnknown;
    }
    if (auto r = p->bound->Run(*launch_); !r) {
      ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
      return r.error().error == kg::KernelError::kUnknown ? sc::JobResult::kUnknown
                                                          : sc::JobResult::kFailed;
    }
    if (auto r = Cuda(
            cudaMemcpyAsync(logits_, g.logits->data, row_bytes, cudaMemcpyDeviceToHost, stream),
            "the logits copy");
        !r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 chunk", stream_);
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return Error(std::format("chunk at {}: {}", n_past, posted.error()));
  }
  if (launch_->faulted()) {
    return Error(std::format("chunk at {}: the launch context faulted", n_past));
  }
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + profile_.vocab);
  return {};
}

Status Qwen38Runner::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  std::vector<std::string> problems;
  plans_.clear();
  launch_.reset();
  cublas_.reset();
  auto& memory = node_.memory();
  for (ts::Mapped* mapped : {&state_, &slot_memory_, &cublas_workspace_}) {
    if (!mapped->reservation.valid()) {
      continue;
    }
    bool released =
        mapped->backings.empty() ||
        memory.Unmap(mapped->reservation, Bytes(0), Bytes(mapped->backings.size() * kExtent))
            .has_value();
    for (const auto backing : mapped->backings) {
      released = memory.Release(backing).has_value() && released;
    }
    if (!released || !memory.Free(mapped->reservation)) {
      problems.push_back(std::format("{} could not be released", mapped->name));
    }
  }
  if (auto r = weights_.Release(memory); !r) {
    problems.push_back(r.error());
  }
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
    spill_fd_ = -1;
  }
  if (ring_ != nullptr && ring_->in_flight() != 0) {
    // Reads that stalled may still land: the ring (and the pages the
    // kernel holds for them) is left to the process's end, never freed
    // under them.
    abandoned_ring_ = ring_.release();
    problems.emplace_back("n-gram row reads were still in flight; their ring was not destroyed");
  }
  ring_.reset();  // otherwise every read was harvested (ReadPleRows drains)
  if (problems.empty()) {
    return {};
  }
  std::string all;
  for (const std::string& problem : problems) {
    all += (all.empty() ? "" : "; ") + problem;
  }
  return Error(all);
}

}  // namespace jitllm::benchmarks
