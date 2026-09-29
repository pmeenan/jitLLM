// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/qwen38_runner.h"

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

namespace jitllm::engine {

namespace {

namespace kg = jitllm::kernels::ggml;
namespace md = jitllm::model;
namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using catalog::Recovery;
using providers::BackingKind;

constexpr std::uint64_t kExtent = kPagedExtent;
constexpr std::size_t kRingDepth = 32;
// The slabs' offset in their first page: the stride's own alignment (16),
// since the 80-byte gap between Qwen3.8's expert groups cannot hold 256.
constexpr std::uint64_t kSlabAlignment = 16;
// The most cell ranges a verify saves: 3 a QSA layer a row, at most 8 rows.
constexpr std::uint32_t kRangeCapacity = 512;
// Where a verify's argmaxes land in the drafts' pinned buffer (I32s).
constexpr std::size_t kArgmaxAt = 64;
// And a draft's probabilities (F32 bits), after its drafts (at most 8).
constexpr std::size_t kProbabilityAt = 32;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

kg::KernelFailure Unknown(std::string what) {
  return {.error = kg::KernelError::kUnknown, .detail = std::move(what)};
}

void Count(Dsv4GraphStats& stats, Dsv4Path path) {
  switch (path) {
    case Dsv4Path::kEager:
      ++stats.eager;
      break;
    case Dsv4Path::kCaptured:
      ++stats.captured;
      break;
    case Dsv4Path::kReplayed:
      ++stats.replayed;
      break;
  }
}

// Every layer's expert arrays of `binding` share their groups: the slab
// of the layer's first group at the stride over its stored bytes.
std::expected<std::pair<std::uint32_t, std::uint64_t>, std::string> SlabOf(
    const artifact::Artifact& artifact, const md::Qwen38Layer& l, bool cutlass, std::uint32_t il) {
  std::uint64_t unit = 16;
  std::optional<std::uint32_t> first_group;
  for (const md::Qwen38Tensor* t : l.expert_arrays(cutlass)) {
    const auto& a = artifact.expert_arrays()[t->index];
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
  return std::pair{g, Round(artifact.groups()[g].stored.value(), unit)};
}

}  // namespace

Qwen38Runner::~Qwen38Runner() = default;

std::vector<ExtentId> Qwen38Runner::weights() const {
  std::vector<ExtentId> all = weights_.extents();
  all.insert(all.end(), dweights_.extents().begin(), dweights_.extents().end());
  return all;
}

std::vector<ExtentId> Qwen38Runner::state() const {
  std::vector<ExtentId> all = state_.extents;
  all.insert(all.end(), mstate_.extents.begin(), mstate_.extents.end());
  return all;
}

std::vector<ExtentId> Qwen38Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  const std::vector<ExtentId> live = state();
  all.insert(all.end(), live.begin(), live.end());
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
  if (!o_.drafter.empty()) {
    const std::uint32_t verify = o_.draft_rows + 1;
    if (o_.draft_rows == 0 || verify > kg::kMxfp8VecColumns || verify > o_.max_rows ||
        o_.draft_vocab > profile_.vocab || !binding_.cutlass()) {
      return Error(
          std::format("a draft of 1 to {} rows (its verify the vector products' rows) over at "
                      "most the vocabulary, beside a CUTLASS-layout target",
                      kg::kMxfp8VecColumns - 1));
    }
    auto drafter = artifact::Artifact::Open(o_.drafter);
    if (!drafter) {
      return Error(std::format("the drafter was refused: {}", drafter.error().reason));
    }
    dartifact_ = std::make_unique<artifact::Artifact>(std::move(*drafter));
    auto dbinding = md::BindQwen38Mtp(profile_, *dartifact_);
    if (!dbinding) {
      return Error(std::format("the drafter: {}", dbinding.error()));
    }
    dbinding_ = std::move(*dbinding);
    auto mtp = md::Qwen38MtpStateOf(profile_, layout_);
    auto commit = md::Qwen38Commit(profile_, verify);
    if (!mtp || !commit) {
      return Error(!mtp ? mtp.error() : commit.error());
    }
    mtp_layout_ = *mtp;
    commit_layout_ = std::move(*commit);
    for (std::uint32_t s = 0; s < dartifact_->shards().size(); ++s) {
      auto fd = dartifact_->OpenShardForDirectRead(s);
      if (!fd) {
        return Error(std::format("the drafter's shard {} cannot be opened for direct reads", s));
      }
      dshards_.push_back(std::move(*fd));
    }
    did_ = ArtifactKey(*dartifact_);
  }

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
  if (speculative()) {
    if (auto r = node_.MapResident(mstate_, "the Qwen3.8 MTP drafter's state", mtp_layout_.bytes,
                                   BackingKind::kDevice, MemoryClass::kLiveState,
                                   Recovery::kPreserve, owner_);
        !r) {
      return r;
    }
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
  // row: the runner reads only the last row's logits; a verify every
  // row's), planned over placeless addresses with a stand-in hash (the rows
  // do not shape a chunk); beside a drafter, its prefill pass and its draft
  // too.
  const auto placeless = [](std::uint32_t) { return std::uint64_t{1} << 44U; };
  std::vector<std::uint64_t> stride = std::move(model_.places.stride);  // ReserveWeights's
  const std::uint64_t mtp_stride = model_.mtp_stride;
  model_ = Qwen38Model{.artifact = artifact_.get(),
                       .profile = &profile_,
                       .binding = &graph_binding_,
                       .state = &layout_,
                       .places = {.resource = placeless,
                                  .array = placeless,
                                  .stride = std::move(stride),
                                  .state = std::uint64_t{1} << 45U,
                                  .ple_table = std::uint64_t{1} << 44U,
                                  .mtp_resource = placeless,
                                  .mtp_array = placeless,
                                  .mtp_state = std::uint64_t{1} << 45U,
                                  .commit = std::uint64_t{1} << 45U},
                       .cutlass = binding_.cutlass(),
                       .drafter = speculative() ? &dbinding_ : nullptr,
                       .mtp_state = speculative() ? &mtp_layout_ : nullptr,
                       .mtp_stride = mtp_stride,
                       .commit = speculative() ? &commit_layout_ : nullptr};
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
    struct Probe {
      std::uint32_t n_past;
      std::uint32_t rows;
      Qwen38ChunkKind kind;
    };
    const std::uint32_t verify = o_.draft_rows + 1;
    std::vector<Probe> probes = {{0, o_.max_rows, {}},
                                 {o_.context - o_.max_rows, o_.max_rows, {}},
                                 {o_.context - 1, 1, {}},
                                 {0, 1, {}}};
    if (speculative()) {
      for (const std::uint32_t at : {0U, o_.context - o_.max_rows}) {
        probes.push_back({at, o_.max_rows, {.verify = false, .export_streams = true}});
      }
      for (const std::uint32_t at : {0U, o_.context - verify}) {
        probes.push_back({at, verify, {.verify = true, .export_streams = true}});
      }
    }
    const auto account = [&](auto& planned) -> Status {
      most_activations = std::max(most_activations, planned->placement.extent);
      auto scratch = kg::PlanScratch(**measure, planned->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, planned->inputs_bytes);
      return {};
    };
    for (const Probe& probe : probes) {
      std::vector<std::int32_t> history(std::size_t{probe.n_past} + probe.rows, 1000);
      auto in = md::Qwen38Chunk(profile_, layout_, stand_in, history, probe.n_past, probe.rows);
      if (!in) {
        return std::unexpected(in.error());
      }
      auto planned = PlanQwen38Chunk(
          model_, kg::Qwen38ShapeOf(layout_, *in, probe.kind.verify ? probe.rows : 1), choices, 0,
          0, {}, probe.kind);
      if (!planned) {
        return Error(std::format("measuring a chunk of {} at {}: {}", probe.rows, probe.n_past,
                                 planned.error()));
      }
      if (auto r = account(*planned); !r) {
        return r;
      }
    }
    if (speculative()) {
      // A prefill pass of a whole chunk at the end, and a draft there.
      for (const auto& [from, rows, passes, head] :
           {std::tuple{o_.context - o_.max_rows - 1, o_.max_rows, 1U, false},
            std::tuple{o_.context - verify - o_.draft_rows, verify, o_.draft_rows, true}}) {
        auto shaped = MtpInputs(from, rows, passes, head, head ? 1 : 0);
        if (!shaped) {
          return std::unexpected(shaped.error());
        }
        auto planned = PlanQwen38Mtp(model_, shaped->first, choices, 0, 0);
        if (!planned) {
          return Error(std::format("measuring the drafter: {}", planned.error()));
        }
        if (auto r = account(*planned); !r) {
          return r;
        }
      }
    }
  }
  activation_bytes_ = Round(most_activations + (most_activations / 4), kExtent);
  scratch_bytes_ = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  input_bytes_ = Round((most_inputs * 2) + (1U << 20U), kExtent);
  // A drafter pass's inputs from the staging's second half, which the
  // largest inputs fit, so one job stages a chunk's and its pass's.
  mtp_base_ = Round(input_bytes_ / 2, 256);

  landing_bytes_ = PleLandingBound(slots_);
  const std::uint64_t logit_rows = speculative() ? o_.draft_rows + 1 : 1;
  auto inputs = node_.Pinned(input_bytes_, owner_, staging_);
  auto logits =
      node_.Pinned(logit_rows * std::uint64_t{profile_.vocab} * sizeof(float), owner_, staging_);
  auto hash =
      node_.Pinned((std::uint64_t{profile_.ngram} + (2 * std::uint64_t{profile_.ple_heads()})) * 8,
                   owner_, staging_);
  auto landing = node_.Pinned(landing_bytes_, owner_, staging_);
  // The slots' sources, then the count of rows the next gather takes.
  auto sources = node_.Pinned((slots_ + 1) * sizeof(std::uint32_t), owner_, staging_);
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
  ple_count_ = sources_ + slots_;
  *ple_count_ = 0;
  if (Address(landing_) % kPleBlock != 0) {
    return Error("the n-gram rows' landing is not 4 KiB-aligned for direct reads");
  }
  if (speculative()) {
    // A verify's saves, then the snapshot of the cells it writes (D-068
    // working state), mapped for the model's life.
    const std::uint32_t qsa = profile_.layers / 4;
    const std::uint64_t row_cells = (2 * std::uint64_t{profile_.head_dim} * profile_.kv_heads * 2) +
                                    (std::uint64_t{profile_.indexer_head_dim} * 4);
    snapshot_offset_ = Round(commit_layout_.bytes, 256);
    const std::uint64_t commit_bytes =
        snapshot_offset_ + Round((std::uint64_t{o_.draft_rows + 1} * qsa * row_cells) + 4096, 256);
    if (auto r = node_.MapResident(commit_, "the Qwen3.8 verify's saves", commit_bytes,
                                   BackingKind::kDevice, MemoryClass::kRuntime, Recovery::kPinned,
                                   owner_);
        !r) {
      return r;
    }
    auto save = node_.Pinned(kRangeCapacity * sizeof(kg::RangeCopy), owner_, staging_);
    auto restore = node_.Pinned(kRangeCapacity * sizeof(kg::RangeCopy), owner_, staging_);
    auto carry = node_.Pinned(4 * sizeof(kg::RangeCopy), owner_, staging_);
    // A draft's drafts, their probabilities (from kProbabilityAt), then
    // (from kArgmaxAt) a verify's argmaxes.
    auto drafts = node_.Pinned(512, owner_, staging_);
    if (!save || !restore || !carry || !drafts) {
      return Error("pinned staging for Qwen3.8's speculation");
    }
    save_ = static_cast<kg::RangeCopy*>(*save);
    restore_ = static_cast<kg::RangeCopy*>(*restore);
    carry_ = static_cast<kg::RangeCopy*>(*carry);
    drafts_ = *drafts;
  }
  auto ring = providers::UringStorage::Create(kRingDepth);
  if (!ring) {
    return Error(std::format("the n-gram rows' ring: {}", ring.error().message()));
  }
  ring_ = std::move(*ring);
  return {};
}

// Every dense group but the n-gram table's, and a slab per layer; the
// drafter's dense groups and its slab.
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
    auto slab = SlabOf(*artifact_, binding_.layers[il], binding_.cutlass(), il);
    if (!slab) {
      return std::unexpected(slab.error());
    }
    model_.places.stride[il] = slab->second;
    slabs.push_back({.first_group = slab->first,
                     .count = profile_.experts,
                     .stride = slab->second,
                     .alignment = kSlabAlignment});
  }
  if (auto r = weights_.Reserve(node_, *artifact_, shards_, id_, place, slabs); !r) {
    return r;
  }
  if (!speculative()) {
    return {};
  }
  const auto dgroups = dartifact_->groups();
  std::vector<bool> dplace(dgroups.size(), false);
  for (std::size_t g = 0; g < dgroups.size(); ++g) {
    dplace[g] = dgroups[g].kind != artifact::GroupKind::kExpert;
  }
  auto slab = SlabOf(*dartifact_, dbinding_.layer, true, 0);
  if (!slab) {
    return std::unexpected(slab.error());
  }
  model_.mtp_stride = slab->second;
  const std::array<SlabSpec, 1> dslabs = {SlabSpec{.first_group = slab->first,
                                                   .count = profile_.experts,
                                                   .stride = slab->second,
                                                   .alignment = kSlabAlignment}};
  return dweights_.Reserve(node_, *dartifact_, dshards_, did_, dplace, dslabs);
}

Status Qwen38Runner::Register() {
  if (auto r = weights_.Register(node_, owner_); !r) {
    return r;
  }
  if (speculative()) {
    if (auto r = dweights_.Register(node_, owner_); !r) {
      return r;
    }
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
// spill file per extent (the target's, then the drafter's), landed through
// the zone, its backing managed.
Status Qwen38Runner::RegisterState() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(),
                             std::generic_category().message(errno)));
  }
  std::uint64_t slot = 0;
  const auto spill = [&](Mapped& mapped, std::vector<sc::PageSource>& sources) -> Status {
    sources.clear();
    for (std::size_t i = 0; i < mapped.extents.size(); ++i, ++slot) {
      const sc::PageSource source{
          .read = {.fd = spill_fd_, .offset = slot * kExtent, .memory = nullptr, .length = kExtent},
          .landed = true,
          .destination = mapped.base + (i * kExtent),
          .backing = sc::BackingPlace{.reservation = mapped.reservation,
                                      .offset = Bytes(i * kExtent),
                                      .size = Bytes(kExtent),
                                      .allocation_class = node_.device_class()},
          .write_back = true};
      auto set = node_.scheduler().SetSource(mapped.extents[i], source);
      if (!set) {
        return Error(std::format("the state's write-back place: {}", sc::ToString(set.error())));
      }
      sources.push_back(source);
    }
    mapped.backings.clear();  // the VMM lane releases them on eviction (D-033)
    return {};
  };
  if (auto r = spill(state_, state_sources_); !r) {
    return r;
  }
  return spill(mstate_, mstate_sources_);
}

Status Qwen38Runner::Bind() {
  auto& catalog = node_.catalog();
  std::vector<ExtentId> all = weights();
  for (const Mapped* mapped :
       std::initializer_list<const Mapped*>{&state_, &mstate_, &slot_memory_, &node_.activations(),
                                            &node_.pool(), &cublas_workspace_, &commit_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  everything_ = catalog.ClosureOfExtents(all).value();
  fence_ = catalog.ClosureOfExtents(state()).value();
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
  if (speculative()) {
    model_.places.mtp_resource = [this](std::uint32_t resource) {
      const auto& r = dartifact_->resources()[resource];
      return dweights_.group_address(r.group) + r.offset.value();
    };
    model_.places.mtp_array = [this](std::uint32_t array) {
      const auto& a = dartifact_->expert_arrays()[array];
      return dweights_.group_address(a.first_group) + a.group_offset.value();
    };
    model_.places.mtp_state = mstate_.base;
    model_.places.commit = commit_.base;
    // The commit's places: every linear-attention layer's state and saves.
    using K = md::Qwen38StateTensor::Kind;
    const auto at = [&](std::uint32_t il, K kind) {
      return state_.base + layout_.tensors[static_cast<std::size_t>(layout_.Find(il, kind))].offset;
    };
    kg::Qwen38CommitArgs& a = commit_args_;
    a.layers = static_cast<int>(commit_layout_.layers.size());
    a.channels = static_cast<int>(profile_.conv_channels());
    a.qk_heads = static_cast<int>(profile_.lin_k_heads);
    a.v_heads = static_cast<int>(profile_.lin_v_heads);
    a.taps = static_cast<int>(profile_.conv - 1);
    for (std::size_t i = 0; i < commit_layout_.layers.size(); ++i) {
      const std::uint32_t il = commit_layout_.layers[i];
      const std::uint64_t base = commit_.base;
      a.layer[i] = {.state = static_cast<float*>(Pointer(at(il, K::kRecurrent))),
                    .history = static_cast<float*>(Pointer(at(il, K::kConv))),
                    .conv = static_cast<const float*>(Pointer(base + commit_layout_.conv_out(i))),
                    .qkv = static_cast<const float*>(Pointer(base + commit_layout_.qkv(i))),
                    .gate = static_cast<const float*>(Pointer(base + commit_layout_.gate(i))),
                    .beta = static_cast<const float*>(Pointer(base + commit_layout_.beta(i)))};
    }
    a.ple_history = static_cast<float*>(Pointer(at(profile_.ple_layer, K::kPleConv)));
    a.ple_rows = static_cast<const float*>(Pointer(commit_.base + commit_layout_.ple()));
    a.ple_width = static_cast<int>(profile_.hc_width());
    a.ple_taps = static_cast<int>(profile_.ple_history());
  }
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
  std::vector<std::pair<std::uint64_t, std::uint64_t>> regions = {{state_.base, layout_.bytes}};
  if (speculative()) {
    regions.emplace_back(mstate_.base, mtp_layout_.bytes);
  }
  restore_count_ = 0;
  commit_keep_ = 0;
  save_count_ = 0;
  saved_.clear();
  verify_rows_ = 0;
  pending_rows_ = 0;
  // Unusable until zeroed; a quarantine lifts once the clear has run.
  quarantined_ = true;
  auto cleared = node_.Job(
      fence_,
      [regions](providers::NativeStream stream) {
        for (const auto& [base, bytes] : regions) {
          if (cudaMemsetAsync(Pointer(base), 0, bytes, static_cast<cudaStream_t>(stream.handle)) !=
              cudaSuccess) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      "clearing the Qwen3.8 state", stream_);
  if (!cleared) {
    return cleared;
  }
  quarantined_ = false;
  return {};
}

std::size_t Qwen38Runner::graphs() const {
  const auto count = [](const auto& plans) {
    return static_cast<std::size_t>(
        std::ranges::count_if(plans, [](const Runs& p) { return p.graph.has_value(); }));
  };
  const auto lean = static_cast<std::size_t>(
      std::ranges::count_if(plans_, [](const ShapePlan& p) { return p.lean.graph.has_value(); }));
  return count(plans_) + count(mplans_) + lean;
}

void Qwen38Runner::RoomForGraph() {
  if (graphs() < kMaxGraphs) {
    return;
  }
  // Graph memory is the driver's, outside the catalog: the oldest decode
  // graph goes, else the oldest draft's (no job is in flight between
  // chunks, so nothing replays it).
  const auto oldest = std::ranges::find_if(
      plans_, [](const ShapePlan& e) { return e.graph.has_value() || e.lean.graph.has_value(); });
  Runs* victim = nullptr;
  if (oldest != plans_.end()) {
    victim = oldest->graph.has_value() ? static_cast<Runs*>(&*oldest) : &oldest->lean;
  }
  if (victim == nullptr) {
    const auto draft =
        std::ranges::find_if(mplans_, [](const MtpPlan& e) { return e.graph.has_value(); });
    victim = draft != mplans_.end() ? static_cast<Runs*>(&*draft) : nullptr;
  }
  if (victim != nullptr) {
    victim->graph.reset();
    victim->copies.clear();
    ++graph_stats_.dropped;
  }
}

std::expected<Qwen38Runner::ShapePlan*, std::string> Qwen38Runner::Planned(
    const kg::Qwen38ChunkShape& shape, Qwen38ChunkKind kind) {
  const auto found = std::ranges::find_if(
      plans_, [&](const ShapePlan& e) { return e.shape == shape && e.kind == kind; });
  if (found != plans_.end()) {
    return &*found;
  }
  const auto start = std::chrono::steady_clock::now();
  auto planned = PlanQwen38Chunk(model_, shape, kg::DeviceChoicesOf(*launch_),
                                 node_.activations().base, node_.activations().bytes, {}, kind);
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
    plans_.erase(plans_.begin());  // its graph with it
  }
  ShapePlan& entry = plans_.emplace_back();
  entry.shape = shape;
  entry.kind = kind;
  entry.planned = std::move(*planned);
  return &entry;
}

std::expected<Qwen38Runner::MtpPlan*, std::string> Qwen38Runner::PlannedMtp(
    const kg::Qwen38MtpShape& shape) {
  const auto found =
      std::ranges::find_if(mplans_, [&](const MtpPlan& e) { return e.shape == shape; });
  if (found != mplans_.end()) {
    return &*found;
  }
  const auto start = std::chrono::steady_clock::now();
  auto planned = PlanQwen38Mtp(model_, shape, kg::DeviceChoicesOf(*launch_),
                               node_.activations().base, node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  auto scratch = kg::PlanScratch(*launch_, (*planned)->plan);
  if (!scratch) {
    return Error(scratch.error().detail);
  }
  if (*scratch > launch_->workspace().size.value()) {
    return Error(std::format("the drafter's scratch ({} bytes) exceeds the pool ({} bytes)",
                             *scratch, launch_->workspace().size.value()));
  }
  (*planned)->scratch = *scratch;
  auto bound = kg::BoundGraph::Bind(*registry_, (*planned)->plan);
  if (!bound) {
    return Error(bound.error().detail);
  }
  (*planned)->bound.emplace(std::move(*bound));
  CheckMtp((*planned)->graph);
  plan_seconds_ += Seconds(std::chrono::steady_clock::now() - start);
  if (mplans_.size() >= 32) {
    mplans_.erase(mplans_.begin());
  }
  MtpPlan& entry = mplans_.emplace_back();
  entry.shape = shape;
  entry.planned = std::move(*planned);
  return &entry;
}

// BP-A1's in-process check, once per planned shape: every tensor the plan
// binds lies in cataloged, resident extents of device memory of one class,
// and each has the class it should: weights, the state (live state: the
// target's, the drafter's caches and streams), the verify's saves
// (runtime) and the row slots and activations (scratch).
void Qwen38Runner::Check(const kg::Qwen38Graph& graph) {
  std::vector<const ggml_tensor*> state;
  std::vector<const ggml_tensor*> saves;
  for (const kg::Qwen38LayerTensors& l : graph.layers) {
    for (const ggml_tensor* t :
         {l.cache_k, l.cache_v, l.cache_idx, l.conv_state, l.recurrent, l.ple_state}) {
      if (t != nullptr) {
        state.push_back(t);
      }
    }
    for (const ggml_tensor* t :
         {l.commit_conv, l.commit_qkv, l.commit_gate, l.commit_beta, l.commit_ple}) {
      if (t != nullptr) {
        saves.push_back(t);
      }
    }
  }
  if (graph.streams != nullptr) {
    state.push_back(graph.streams);
  }
  const auto inputs = graph.inputs();
  const auto kind_of = [&](const ggml_tensor* t) {
    const ggml_tensor* base = t->view_src != nullptr ? t->view_src : t;
    if (std::ranges::find(state, base) != state.end()) {
      return MemoryClass::kLiveState;
    }
    if (std::ranges::find(saves, base) != saves.end()) {
      return MemoryClass::kRuntime;
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

void Qwen38Runner::CheckMtp(const kg::Qwen38MtpGraph& graph) {
  const std::vector<const ggml_tensor*> state = {graph.layer.cache_k, graph.layer.cache_v,
                                                 graph.layer.cache_idx, graph.streams};
  const auto inputs = graph.inputs();
  const auto kind_of = [&](const ggml_tensor* t) {
    const ggml_tensor* base = t->view_src != nullptr ? t->view_src : t;
    if (std::ranges::find(state, base) != state.end()) {
      return MemoryClass::kLiveState;
    }
    return base->op == GGML_OP_NONE && std::ranges::find(inputs, base) == inputs.end()
               ? MemoryClass::kWeights
               : MemoryClass::kScratch;
  };
  const auto expect = [&](const ggml_tensor* t) {
    ++coverage_tensors_;
    const std::optional<MemoryClass> covered =
        node_.Covered(Address(t->data), ggml_nbytes(t), owner_);
    if (covered != kind_of(t) && coverage_violations_++ == 0) {
      first_violation_ = std::format("the drafter's {} ({} bytes at {:#x})", t->name,
                                     ggml_nbytes(t), Address(t->data));
    }
  };
  for (const ggml_tensor* node : graph.nodes) {
    expect(node);
    if (node->op == GGML_OP_FILL) {
      continue;
    }
    for (const ggml_tensor* src : node->src) {
      if (src != nullptr) {
        expect(src);
      }
    }
  }
}

std::expected<Qwen38Runner::Copies, std::string> Qwen38Runner::Stage(
    std::span<const std::pair<ggml_tensor*, const void*>> sources, std::uint64_t base) {
  Copies copies;
  copies.reserve(sources.size());
  std::uint64_t staged = base;
  for (const auto& [tensor, source] : sources) {
    const std::uint64_t bytes = ggml_nbytes(tensor);
    if (staged + bytes > input_bytes_) {
      return Error("the inputs exceed their staging");
    }
    std::memcpy(static_cast<std::byte*>(inputs_) + staged, source, bytes);
    copies.push_back({Address(tensor->data), bytes, staged});
    staged += Round(bytes, 256);
  }
  return copies;
}

Qwen38Runner::Queued Qwen38Runner::QueueRuns(Runs& runs, const Copies& copies,
                                             const std::function<bool(void* stream)>& between,
                                             kg::BoundGraph& bound, const Copies& outputs,
                                             bool capture, Dsv4GraphStats& stats, void* native) {
  auto* const stream = static_cast<cudaStream_t>(native);
  // The input copies, the gather, the plan and the outputs' copies, as one
  // run queues them and a capture records them.
  const auto queue = [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
    for (const auto& [to, bytes, at] : copies) {
      if (const cudaError_t copied =
              cudaMemcpyAsync(Pointer(to), static_cast<const std::byte*>(inputs_) + at, bytes,
                              cudaMemcpyHostToDevice, stream);
          copied != cudaSuccess) {
        return std::unexpected(
            Unknown(std::format("an input copy: {}", cudaGetErrorString(copied))));
      }
    }
    if (between && !between(stream)) {
      return std::unexpected(Unknown("the n-gram rows' gather"));
    }
    if (auto r = bound.Run(launch); !r) {
      return r;
    }
    for (const auto& [to, from, bytes] : outputs) {
      if (const cudaError_t copied =
              cudaMemcpyAsync(Pointer(to), Pointer(from), bytes, cudaMemcpyDeviceToHost, stream);
          copied != cudaSuccess) {
        return std::unexpected(
            Unknown(std::format("an output's copy: {}", cudaGetErrorString(copied))));
      }
    }
    return {};
  };
  Queued q;
  if (graphs_ && runs.graph.has_value()) {
    // What the graph copies must be where the host staged it.
    if (copies != runs.copies) {
      q.result = std::unexpected(
          kg::KernelFailure{.error = kg::KernelError::kRejected,
                            .detail = "the inputs' staging differs from the captured graph's"});
      return q;
    }
    q.path = Dsv4Path::kReplayed;
    q.result = launch_->Launch(*runs.graph);
    return q;
  }
  if (capture) {
    std::size_t free_before = 0;
    std::size_t free_after = 0;
    std::size_t total = 0;
    (void)cudaMemGetInfo(&free_before, &total);
    auto captured = launch_->Capture(queue);
    (void)cudaMemGetInfo(&free_after, &total);
    (void)cudaGetLastError();
    if (captured) {
      stats.capture_seconds += captured->capture_seconds();
      stats.instantiate_seconds += captured->instantiate_seconds();
      stats.nodes += captured->nodes();
      stats.memory_bytes +=
          static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
      runs.graph.emplace(std::move(*captured));
      runs.copies = copies;
      q.path = Dsv4Path::kCaptured;
      q.before = true;  // the upload
      q.result = launch_->Launch(*runs.graph);
      return q;
    }
    if (captured.error().error == kg::KernelError::kUnknown) {
      q.result = std::unexpected(captured.error());
      return q;
    }
    // Refused, with nothing queued: this plan runs launch by launch.
    runs.uncapturable = true;
    if (stats.refused++ == 0) {
      stats.first_refusal = captured.error().detail;
    }
  }
  ++runs.eager_runs;
  q.before = true;  // any input copy before a refusal
  q.result = queue(*launch_);
  return q;
}

std::expected<std::vector<std::int32_t>, std::string> Qwen38Runner::ReadRows(
    const md::Qwen38ChunkInputs& in) {
  // The chunk's n-gram rows: planned, read and their slots' sources set,
  // while no job of this model holds the landing (the last one's fence
  // has completed: Job returns only after it).
  const auto reading = std::chrono::steady_clock::now();
  auto rows_plan = PlanPleRows(table_, in.ple_rows, landing_bytes_, slots_);
  if (!rows_plan) {
    return std::unexpected(rows_plan.error());
  }
  // The gather's grid is the chunk's lookups (rows × heads) and the device
  // reads the count: it must never exceed them, nor the slots.
  if (rows_plan->sources.size() > in.ple_rows.size() || rows_plan->sources.size() > slots_) {
    return std::unexpected(std::format("{} n-gram rows for {} lookups and {} slots",
                                       rows_plan->sources.size(), in.ple_rows.size(), slots_));
  }
  if (auto r = ReadPleRows(*ring_, table_.fd, *rows_plan, landing_); !r) {
    rows_stalled_ = ring_->in_flight() != 0;  // reads that may still land
    return std::unexpected(r.error());
  }
  std::ranges::copy(rows_plan->sources, sources_);
  *ple_count_ = static_cast<std::uint32_t>(rows_plan->sources.size());
  ple_.chunks += 1;
  ple_.lookups += in.ple_rows.size();
  ple_.rows += rows_plan->sources.size();
  ple_.reads += rows_plan->reads.size();
  ple_.read_bytes += rows_plan->landing_bytes;
  ple_.useful_bytes += rows_plan->useful_bytes;
  ple_.extent_bytes += rows_plan->extents * kExtent;
  ple_.seconds += Seconds(std::chrono::steady_clock::now() - reading);
  return std::move(rows_plan->slots);
}

std::expected<void, kg::KernelFailure> Qwen38Runner::QueueCommit() {
  if (restore_count_ != 0) {
    // Still owed until the copy is queued.
    if (auto r = kg::CopyRanges(*launch_, restore_, restore_count_); !r) {
      return r;
    }
    restore_count_ = 0;
  }
  if (commit_keep_ != 0) {
    kg::Qwen38CommitArgs args = commit_args_;
    args.keep = static_cast<int>(commit_keep_);
    if (auto r = kg::Qwen38Commit(*launch_, args); !r) {
      return r;
    }
    commit_keep_ = 0;
  }
  return {};
}

void Qwen38Runner::Settle(bool verified, bool wrote, bool unknown) {
  verify_rows_ = 0;
  if (unknown || launch_->faulted()) {
    quarantined_ = true;
    return;
  }
  if (verified) {
    // The whole verify undone: every cell it saved restored before the
    // next job's own work, nothing committed. It wrote no other target
    // state, but its streams rows may have overwritten the ones the next
    // draft would catch up on (rows 1 ..): none is pending until a chunk
    // with the injection or an accepted verify writes them again.
    std::uint32_t count = 0;
    for (const Saved& s : saved_) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
    restore_count_ = count;
    commit_keep_ = 0;
    pending_rows_ = 0;
    return;
  }
  if (wrote) {
    quarantined_ = true;
  }
}

Status Qwen38Runner::Usable() const {
  if (!hash_checked_) {
    return Error("the n-gram hash is not checked since the last load");
  }
  if (rows_stalled_) {
    return Error("the n-gram rows' reads stalled earlier; their landing may still be written");
  }
  if (quarantined_) {
    return Error(
        "the Qwen3.8 state is quarantined: a job failed after it may have written it "
        "(Clear first)");
  }
  return {};
}

std::expected<std::pair<kg::Qwen38MtpShape, std::vector<md::Qwen38ChunkInputs>>, std::string>
Qwen38Runner::MtpInputs(std::uint32_t first, std::uint32_t rows, std::uint32_t passes, bool head,
                        std::int64_t hidden_row) const {
  const std::uint64_t end = std::uint64_t{first} + rows + passes - 1;
  if (rows == 0 || passes == 0 || end > mtp_layout_.context) {
    return Error(std::format("a draft of {} passes after {} rows at {} passes the context", passes,
                             rows, first));
  }
  const auto n_kv =
      static_cast<std::uint32_t>(std::min<std::uint64_t>(Round(end, 256), mtp_layout_.cells));
  std::vector<md::Qwen38ChunkInputs> ins;
  for (std::uint32_t p = 0; p < passes; ++p) {
    auto in = md::Qwen38Rows(profile_, mtp_layout_.cells, p == 0 ? first : first + rows + p - 1,
                             p == 0 ? rows : 1, n_kv, false);
    if (!in) {
      return std::unexpected(in.error());
    }
    ins.push_back(std::move(*in));
  }
  const kg::Qwen38MtpShape shape{.rows = rows,
                                 .passes = passes,
                                 .n_kv = n_kv,
                                 .cells = mtp_layout_.cells,
                                 .qsa_select = ins.front().qsa_select,
                                 .qsa_blocks = ins.front().qsa_select ? ins.front().qsa.blocks : 0,
                                 .head = head,
                                 .head_rows = o_.draft_vocab,
                                 .hidden_row = hidden_row,
                                 .hidden_rows = mtp_layout_.hidden_rows};
  return std::pair{shape, std::move(ins)};
}

Status Qwen38Runner::Chunk(std::span<const std::int32_t> history, std::uint32_t n_past,
                           std::vector<float>& logits, bool inject) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  if (inject && !speculative()) {
    return Error("an injection needs the drafter");
  }
  if (history.size() <= n_past) {
    return Error("an empty chunk");
  }
  const auto rows = static_cast<std::uint32_t>(history.size() - n_past);
  // Without the selection's host masks, which the fast graph makes on the
  // device; built below if the planned graph reads them.
  auto in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, false);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto slots = ReadRows(*in);
  if (!slots) {
    return std::unexpected(slots.error());
  }
  const Qwen38ChunkKind kind{.verify = false, .export_streams = inject};
  auto planned = Planned(kg::Qwen38ShapeOf(layout_, *in, 1), kind);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ShapePlan& entry = **planned;
  Qwen38Planned* p = entry.planned.get();
  const kg::Qwen38Graph& g = p->graph;
  if (in->qsa_select && (g.mask != nullptr || g.mask_f32 != nullptr)) {
    // This graph selects over the host's masks (GGML's top-k).
    in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, true);
    if (!in) {
      return std::unexpected(in.error());
    }
  }
  Qwen38HostInputs host;
  Qwen38Sources(g, *in, 1, *slots, host, 1);
  auto copies = Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  // The drafter's pass over the positions whose next token the chunk
  // holds: the pending row of the chunk before (its streams in row 0) and
  // the chunk's rows but its last (in rows 1 ..), or at the start of the
  // sequence the chunk's rows but its last; then the last row's streams
  // become the pending row (rows 0 and 1).
  MtpPlan* mentry = nullptr;
  Copies mcopies;
  Qwen38MtpHostInputs mhost;
  std::vector<md::Qwen38ChunkInputs> mins;
  std::uint32_t carries = 0;
  if (inject) {
    const std::uint32_t mrows = n_past > 0 ? rows : rows - 1;
    const std::uint32_t first = n_past > 0 ? n_past - 1 : 0;
    if (mrows > 0) {
      auto shaped = MtpInputs(first, mrows, 1, false, n_past > 0 ? 0 : 1);
      if (!shaped) {
        return std::unexpected(shaped.error());
      }
      mins = std::move(shaped->second);
      auto mplanned = PlannedMtp(shaped->first);
      if (!mplanned) {
        return std::unexpected(mplanned.error());
      }
      mentry = *mplanned;
      Qwen38MtpSources(mentry->planned->graph, mins, history.subspan(first + 1, mrows), mhost);
      auto staged = Stage(mhost.sources, mtp_base_);
      if (!staged) {
        return std::unexpected(staged.error());
      }
      mcopies = std::move(*staged);
    }
    const std::uint64_t row = std::uint64_t{profile_.hc_width()} * sizeof(float);
    const std::uint64_t streams = mstate_.base + mtp_layout_.hidden;
    carry_[carries++] = {.from = streams + (rows * row), .to = streams, .bytes = row};
    if (rows != 1) {
      carry_[carries++] = {.from = streams + (rows * row), .to = streams + row, .bytes = row};
    }
  }
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  // The gather's grid: the shape's lookups, a slot each at most.
  const auto max_count = static_cast<std::uint32_t>(std::uint64_t{rows} * profile_.ple_heads());
  const auto gather = [this, max_count](void* stream) {
    return GatherPleRows(landing_, sources_, ple_count_, max_count,
                         static_cast<std::uint32_t>(table_.row_bytes),
                         static_cast<std::byte*>(Pointer(slot_memory_.base)), stream);
  };
  // Decode graphs (D-090): replay a shape's graph; capture a one-row shape
  // that has run once launch by launch; otherwise launch by launch.
  const bool replay = graphs_ && entry.graph.has_value();
  const bool capture =
      graphs_ && !replay && rows == 1 && !inject && !entry.uncapturable && entry.eager_runs > 0;
  if (capture) {
    RoomForGraph();
  }
  const Copies outputs = {
      {Address(logits_), Address(static_cast<const std::byte*>(g.logits->data)), row_bytes}};
  Dsv4Path path = Dsv4Path::kEager;
  Status ran;
  bool wrote = false;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    // A pending commit or restore queued here is work this job must fence,
    // even if its own run is then refused before anything else.
    const bool committing = restore_count_ != 0 || commit_keep_ != 0;
    if (auto r = QueueCommit(); !r) {
      ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    const Queued queued =
        QueueRuns(entry, *copies, gather, *p->bound, outputs, capture, graph_stats_, native.handle);
    path = queued.path;
    wrote = queued.result.has_value() || queued.before;
    if (!queued.result) {
      ran = Error(std::format("chunk at {}: {}", n_past, queued.result.error().detail));
      if (queued.result.error().error == kg::KernelError::kUnknown) {
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      return committing || queued.before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    if (mentry != nullptr) {
      const Queued m = QueueRuns(*mentry, mcopies, {}, *mentry->planned->bound, {}, false,
                                 draft_stats_, native.handle);
      if (!m.result) {
        ran = Error(std::format("the drafter's pass at {}: {}", n_past, m.result.error().detail));
        unknown = m.result.error().error == kg::KernelError::kUnknown;
        return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
      }
      Count(draft_stats_, m.path);
    }
    if (carries != 0) {
      if (auto r = kg::CopyRanges(*launch_, carry_, carries); !r) {
        ran = Error(std::format("the drafter's pending row at {}: {}", n_past, r.error().detail));
        unknown = true;
        return sc::JobResult::kUnknown;
      }
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 chunk", stream_);
  if (!posted || !ran || launch_->faulted()) {
    Settle(false, wrote, unknown);
    if (launch_->faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    return !ran ? ran : Error(std::format("chunk at {}: {}", n_past, posted.error()));
  }
  Count(graph_stats_, path);
  // With the injection the chunk's last row is the pending one; without,
  // the streams rows no longer precede the anchor, so no draft may read
  // them until a chunk with the injection or an accepted verify.
  pending_rows_ = inject ? 1 : 0;
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + profile_.vocab);
  return {};
}

Status Qwen38Runner::Draft(std::span<const std::int32_t> history, std::vector<std::int32_t>& drafts,
                           std::vector<float>* probabilities) {
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  const std::uint32_t rows = pending_rows_;
  if (rows == 0 || history.size() < std::size_t{rows} + 1) {
    return Error("no streams to draft from (a prefill with the injection, or a verify, first)");
  }
  // The catch-up: the rows the last verify kept (or the prefill left), at
  // the positions before the anchor, each with the token after it. Those
  // tokens index the target's table on the device (the anchor no chunk has
  // checked yet among them).
  const auto n = static_cast<std::uint32_t>(history.size() - 1);
  for (const std::int32_t t : history.subspan(n - rows + 1, rows)) {
    if (t < 0 || std::cmp_greater_equal(t, profile_.vocab)) {
      return Error(std::format("token {} is outside the vocabulary", t));
    }
  }
  auto shaped = MtpInputs(n - rows, rows, o_.draft_rows, true, 1);
  if (!shaped) {
    return std::unexpected(shaped.error());
  }
  auto planned = PlannedMtp(shaped->first);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  MtpPlan& entry = **planned;
  const kg::Qwen38MtpGraph& g = entry.planned->graph;
  Qwen38MtpHostInputs host;
  Qwen38MtpSources(g, shaped->second, history.subspan(n - rows + 1, rows), host);
  auto copies = Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  Copies outputs;
  for (std::size_t j = 0; j < g.drafts.size(); ++j) {
    outputs.push_back({Address(static_cast<std::int32_t*>(drafts_) + j), Address(g.drafts[j]->data),
                       sizeof(std::int32_t)});
  }
  for (std::size_t j = 0; j < g.probabilities.size(); ++j) {
    outputs.push_back({Address(static_cast<std::int32_t*>(drafts_) + kProbabilityAt + j),
                       Address(g.probabilities[j]->data), sizeof(std::int32_t)});
  }
  const bool capture =
      graphs_ && !entry.graph.has_value() && !entry.uncapturable && entry.eager_runs > 0;
  if (capture) {
    RoomForGraph();
  }
  Dsv4Path path = Dsv4Path::kEager;
  Status ran;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (auto r = QueueCommit(); !r) {
      ran = Error(std::format("draft at {}: {}", n, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    const Queued queued = QueueRuns(entry, *copies, {}, *entry.planned->bound, outputs, capture,
                                    draft_stats_, native.handle);
    path = queued.path;
    if (!queued.result) {
      ran = Error(std::format("draft at {}: {}", n, queued.result.error().detail));
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 MTP draft", stream_);
  if (!posted || !ran || launch_->faulted()) {
    // A draft writes the drafter's cells alone (the committed ones as the
    // next draft rewrites them); a launch of unknown effect quarantines.
    Settle(false, false, unknown);
    if (launch_->faulted()) {
      return Error(std::format("draft at {}: the launch context faulted", n));
    }
    return !ran ? ran : Error(std::format("draft at {}: {}", n, posted.error()));
  }
  Count(draft_stats_, path);
  const auto* values = static_cast<const std::int32_t*>(drafts_);
  drafts.assign(values, values + g.drafts.size());
  if (probabilities != nullptr) {
    probabilities->resize(g.probabilities.size());
    std::memcpy(probabilities->data(), values + kProbabilityAt,
                g.probabilities.size() * sizeof(float));
  }
  return {};
}

Status Qwen38Runner::Verify(std::span<const std::int32_t> history, std::uint32_t n_past,
                            std::vector<std::int32_t>& argmax, std::vector<float>* logits) {
  if (!speculative()) {
    return Error("a verify needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  if (history.size() <= n_past || history.size() - n_past > std::size_t{o_.draft_rows} + 1) {
    return Error(std::format("a verify of 1 to {} rows", o_.draft_rows + 1));
  }
  const auto rows = static_cast<std::uint32_t>(history.size() - n_past);
  auto in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, false);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto slots = ReadRows(*in);
  if (!slots) {
    return std::unexpected(slots.error());
  }
  const Qwen38ChunkKind kind{.verify = true, .export_streams = true};
  auto planned = Planned(kg::Qwen38ShapeOf(layout_, *in, rows), kind);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ShapePlan& entry = **planned;
  Qwen38Planned* p = entry.planned.get();
  const kg::Qwen38Graph& g = p->graph;
  if (in->qsa_select && (g.mask != nullptr || g.mask_f32 != nullptr)) {
    in = md::Qwen38Chunk(profile_, layout_, hash_, history, n_past, rows, true);
    if (!in) {
      return std::unexpected(in.error());
    }
  }
  Qwen38HostInputs host;
  Qwen38Sources(g, *in, rows, *slots, host, 1);
  auto copies = Stage(host.sources, 0);
  if (!copies) {
    return std::unexpected(copies.error());
  }
  // The cells it writes (each QSA layer's K, V and indexer rows), saved.
  using K = md::Qwen38StateTensor::Kind;
  saved_.clear();
  save_count_ = 0;
  std::uint64_t at = commit_.base + snapshot_offset_;
  for (std::uint32_t i = 0; i < rows; ++i) {
    const std::uint64_t cell = std::uint64_t{n_past} + i;
    for (const md::Qwen38StateTensor& t : layout_.tensors) {
      if (t.kind != K::kK && t.kind != K::kV && t.kind != K::kIndexerK) {
        continue;
      }
      const std::uint64_t bytes = t.ne0 * (t.f16 ? 2 : 4);
      if (save_count_ == kRangeCapacity || at + bytes > commit_.base + commit_.bytes) {
        return Error("a verify writes more than its snapshot holds");
      }
      const std::uint64_t address = state_.base + t.offset + (cell * bytes);
      saved_.push_back({.address = address, .saved = at, .bytes = bytes, .row = i});
      save_[save_count_++] = {.from = address, .to = at, .bytes = bytes};
      at += Round(bytes, 256);
    }
  }
  const auto max_count = static_cast<std::uint32_t>(std::uint64_t{rows} * profile_.ple_heads());
  const auto gather = [this, max_count](void* stream) {
    return GatherPleRows(landing_, sources_, ple_count_, max_count,
                         static_cast<std::uint32_t>(table_.row_bytes),
                         static_cast<std::byte*>(Pointer(slot_memory_.base)), stream);
  };
  // The argmaxes always; the logits (its own runs, `entry`'s) when asked.
  Runs& runs = logits != nullptr ? static_cast<Runs&>(entry) : entry.lean;
  const bool capture =
      graphs_ && !runs.graph.has_value() && !runs.uncapturable && runs.eager_runs > 0;
  if (capture) {
    RoomForGraph();
  }
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  auto* const argmax_host = static_cast<std::int32_t*>(drafts_) + kArgmaxAt;
  Copies outputs = {{Address(argmax_host), Address(g.argmax->data), rows * sizeof(std::int32_t)}};
  if (logits != nullptr) {
    outputs.push_back({Address(logits_), Address(g.logits->data), rows * row_bytes});
  }
  Dsv4Path path = Dsv4Path::kEager;
  Status ran;
  bool saved = false;
  bool unknown = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    if (auto r = QueueCommit(); !r) {
      ran = Error(std::format("verify at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    if (auto r = kg::CopyRanges(*launch_, save_, save_count_); !r) {
      ran = Error(std::format("verify at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    saved = true;
    const Queued queued =
        QueueRuns(runs, *copies, gather, *p->bound, outputs, capture, graph_stats_, native.handle);
    path = queued.path;
    if (!queued.result) {
      ran = Error(std::format("verify at {}: {}", n_past, queued.result.error().detail));
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(everything_, std::move(job), "a Qwen3.8 verify", stream_);
  if (!posted || !ran || launch_->faulted()) {
    // Never left half-written: the verify is undone before the next job's
    // work, or the state quarantined.
    Settle(saved, false, unknown);
    if (launch_->faulted()) {
      return Error(std::format("verify at {}: the launch context faulted", n_past));
    }
    return !ran ? ran : Error(std::format("verify at {}: {}", n_past, posted.error()));
  }
  Count(graph_stats_, path);
  verify_rows_ = rows;
  argmax.assign(argmax_host, argmax_host + rows);
  if (logits != nullptr) {
    const auto* values = static_cast<const float*>(logits_);
    logits->assign(values, values + (std::size_t{rows} * profile_.vocab));
  }
  return {};
}

Status Qwen38Runner::Accept(std::uint32_t keep) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ == 0 || keep == 0 || keep > verify_rows_) {
    return Error(std::format("accepting {} rows of a verify of {}", keep, verify_rows_));
  }
  std::uint32_t count = 0;
  for (const Saved& s : saved_) {
    if (s.row >= keep) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
  }
  restore_count_ = count;
  commit_keep_ = keep;
  pending_rows_ = keep;
  verify_rows_ = 0;
  return {};
}

Status Qwen38Runner::Rollback() {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (restore_count_ == 0 && commit_keep_ == 0) {
    return {};
  }
  std::string failed;
  auto posted = node_.Job(
      everything_,
      [&](providers::NativeStream) {
        if (auto r = QueueCommit(); !r) {
          failed = r.error().detail;
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      "committing a verify's kept rows", stream_);
  if (!posted || !failed.empty()) {
    // A commit not queued stays owed; one queued has completed (the job
    // retired); a launch of unknown effect quarantines.
    Settle(false, false, !failed.empty());
    return Error(failed.empty() ? posted.error() : failed);
  }
  return {};
}

Status Qwen38Runner::ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
  if (auto r = Rollback(); !r) {
    return r;
  }
  if (verify_rows_ != 0) {
    return Error("reading the state with a verify awaiting its Accept");
  }
  const std::uint64_t mtp = speculative() ? mtp_layout_.bytes : 0;
  if (state_host_ == nullptr && cudaMallocHost(&state_host_, layout_.bytes + mtp) != cudaSuccess) {
    state_host_ = nullptr;
    return Error("pinned host memory for the state's copy");
  }
  const std::uint64_t target_base = state_.base;
  const std::uint64_t target_bytes = layout_.bytes;
  const std::uint64_t mtp_base = mstate_.base;
  void* host = state_host_;
  auto posted = node_.Job(
      fence_,
      [=](providers::NativeStream stream) {
        auto* s = static_cast<cudaStream_t>(stream.handle);
        if (cudaMemcpyAsync(host, Pointer(target_base), target_bytes, cudaMemcpyDeviceToHost, s) !=
            cudaSuccess) {
          return sc::JobResult::kUnknown;
        }
        if (mtp != 0 &&
            cudaMemcpyAsync(static_cast<std::byte*>(host) + target_bytes, Pointer(mtp_base), mtp,
                            cudaMemcpyDeviceToHost, s) != cudaSuccess) {
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      "reading the Qwen3.8 state", stream_);
  if (!posted) {
    return posted;
  }
  const auto* bytes = static_cast<const std::byte*>(state_host_);
  target.assign(bytes, bytes + target_bytes);
  drafter.assign(bytes + target_bytes, bytes + target_bytes + mtp);
  return {};
}

Status Qwen38Runner::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  std::vector<std::string> problems;
  plans_.clear();
  mplans_.clear();
  launch_.reset();
  cublas_.reset();
  auto& memory = node_.memory();
  for (Mapped* mapped : {&state_, &mstate_, &slot_memory_, &cublas_workspace_, &commit_}) {
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
  if (speculative()) {
    if (auto r = dweights_.Release(memory); !r) {
      problems.push_back(r.error());
    }
  }
  if (state_host_ != nullptr) {
    (void)cudaFreeHost(state_host_);
    state_host_ = nullptr;
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

}  // namespace jitllm::engine
