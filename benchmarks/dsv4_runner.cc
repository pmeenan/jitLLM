// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "dsv4_runner.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstring>
#include <expected>
#include <format>
#include <initializer_list>
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "artifact/layout.h"
#include "ggml.h"
#include "kernels/ggml/executor.h"
#include "kernels/ggml/graph_plan.h"
#include "kernels/ggml/implementations.h"
#include "providers/direct_reader.h"
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
constexpr std::uint64_t kFileAlignment = 4096;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

void* Pointer(std::uint64_t address) {
  return reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t Address(const void* pointer) { return reinterpret_cast<std::uintptr_t>(pointer); }

std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

std::uint64_t Lcm(std::uint64_t a, std::uint64_t b) { return a / std::gcd(a, b) * b; }

double Seconds(std::chrono::steady_clock::duration d) {
  return std::chrono::duration<double>(d).count();
}

}  // namespace

// ------------------------------------------------------------------ the slab

std::expected<SlabLayout, std::string> LayOutSlab(std::span<const std::uint32_t> shard,
                                                  std::span<const std::uint64_t> file,
                                                  std::uint64_t stored, std::uint64_t stride,
                                                  std::uint64_t alignment) {
  const std::size_t n = file.size();
  if (n == 0 || shard.size() != n || stored == 0 || stride < stored ||
      stored % kFileAlignment != 0) {
    return Error("not a slab: no groups, or a stride below the groups' stored bytes");
  }
  if (alignment < 16 || alignment > kFileAlignment || (alignment & (alignment - 1)) != 0) {
    return Error("a slab's alignment is a power of two from 16 to 4,096");
  }
  // Consecutive in the file within a shard, changing shard at most once.
  std::optional<std::size_t> change;
  for (std::size_t e = 1; e < n; ++e) {
    if (shard[e] == shard[e - 1]) {
      if (file[e] != file[e - 1] + stored) {
        return Error(
            std::format("expert {}'s group does not follow expert {}'s in the file", e, e - 1));
      }
    } else if (change) {
      return Error("the slab's groups change shard more than once");
    } else {
      change = e;
    }
  }
  SlabLayout out;
  out.stride = stride;
  const std::uint64_t gap = stride - stored;
  if (change) {
    // A page boundary in the gap before the group that starts the second
    // shard: δ ≡ -e·S (mod 2 MiB), rounded up to the alignment within the
    // gap.
    const std::uint64_t at = (*change * stride) % kExtent;
    const std::uint64_t r = (kExtent - at) % kExtent;
    const std::uint64_t delta = Round(r, alignment);
    if (delta - r > gap) {
      return Error("the gap between groups is too small to align a shard change with a page");
    }
    out.delta = delta % kExtent;
  }
  const std::uint64_t end = out.delta + ((n - 1) * stride) + stored;
  const std::uint64_t pages = (end + kExtent - 1) / kExtent;
  out.pages.reserve(pages);
  for (std::uint64_t p = 0; p < pages; ++p) {
    const std::uint64_t lo = p * kExtent;
    const std::uint64_t hi = lo + kExtent;
    SlabPage page;
    std::uint64_t first = 0;  // the first piece's file position
    std::uint64_t last = 0;   // the last piece's file end
    // Experts whose groups overlap [lo, hi): from the one containing lo.
    std::size_t e = lo > out.delta ? static_cast<std::size_t>((lo - out.delta) / stride) : 0;
    for (; e < n && out.delta + (e * stride) < hi; ++e) {
      const std::uint64_t start = out.delta + (e * stride);
      const std::uint64_t piece_lo = std::max(lo, start);
      const std::uint64_t piece_hi = std::min(hi, start + stored);
      if (piece_lo >= piece_hi) {
        continue;  // the page starts in the gap after this group
      }
      const std::uint64_t position = file[e] + (piece_lo - start);
      if (page.pieces == 0) {
        page.shard = shard[e];
        first = position;
      } else if (shard[e] != page.shard || position != last) {
        return Error(std::format("page {} needs bytes that are not one range of one file", p));
      }
      if (page.pieces == kSlabPieces) {
        return Error(std::format("page {} needs more than {} pieces", p, kSlabPieces));
      }
      page.page_offset.at(page.pieces) = piece_lo - lo;
      page.bytes.at(page.pieces) = piece_hi - piece_lo;
      page.slot_offset.at(page.pieces) = position;  // made relative below
      ++page.pieces;
      last = position + (piece_hi - piece_lo);
    }
    if (page.pieces == 0) {
      return Error(std::format("page {} holds no group's bytes", p));
    }
    page.file_offset = first / kFileAlignment * kFileAlignment;
    page.length = Round(last, kFileAlignment) - page.file_offset;
    if (page.length > kSlabSlotBytes) {
      return Error(std::format("page {}'s read is longer than a slot", p));
    }
    for (std::size_t i = 0; i < page.pieces; ++i) {
      page.slot_offset.at(i) -= page.file_offset;
    }
    out.pages.push_back(page);
  }
  return out;
}

// ------------------------------------------------------------------ setup

std::vector<ExtentId> Dsv4Runner::weights() const {
  std::vector<ExtentId> all;
  all.reserve(extents_.size());
  for (const WeightExtent& w : extents_) {
    all.push_back(w.extent);
  }
  return all;
}

std::vector<ExtentId> Dsv4Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  all.insert(all.end(), state_.extents.begin(), state_.extents.end());
  return all;
}

Status Dsv4Runner::Setup() {
  auto artifact = artifact::Artifact::Open(o_.artifact);
  if (!artifact) {
    return Error(std::format("the artifact was refused: {}", artifact.error().reason));
  }
  artifact_ = std::make_unique<artifact::Artifact>(std::move(*artifact));
  auto binding = md::BindDsv4(profile_, *artifact_);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  auto layout = md::Dsv4State(profile_, o_.context, o_.max_rows);
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
  // The artifact's identity orders victim ties (catalog.h ContentKey).
  for (std::size_t i = 0; i < id_.size() && (2 * i) + 1 < artifact_->id().size(); ++i) {
    (void)std::from_chars(artifact_->id().data() + (2 * i), artifact_->id().data() + (2 * i) + 2,
                          id_.at(i), 16);
  }
  // The state first: its extents come before the weights' in a closure, so
  // a swap back restores it before paging the weights in.
  if (auto r = node_.MapResident(state_, "the DeepSeek state", layout_.bytes, BackingKind::kDevice,
                                 MemoryClass::kLiveState, Recovery::kPreserve, owner_);
      !r) {
    return r;
  }
  if (auto r = ReserveWeights(); !r) {
    return r;
  }

  // cuBLAS, with upstream's workspace for the device, as the resident
  // harness has it: the router's BF16 products run there at prefill widths.
  int major = 0;
  int minor = 0;
  (void)cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
  (void)cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
  cublas_bytes_ = kg::CublasHandle::UpstreamWorkspace((100 * major) + (10 * minor)).value();
  if (auto r =
          node_.MapResident(cublas_workspace_, "the DeepSeek cuBLAS workspace", cublas_bytes_,
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

  // The largest shapes, as the resident harness sizes them: a full chunk
  // at the start and at the end of the context, and a decode step at the
  // end, planned over placeless addresses.
  std::vector<std::uint64_t> stride = model_.places.stride;  // ReserveWeights's
  model_ = Dsv4Model{.artifact = artifact_.get(),
                     .profile = &profile_,
                     .binding = &binding_,
                     .state = &layout_,
                     .places = {.resource = [](std::uint32_t) { return std::uint64_t{1} << 44U; },
                                .array = [](std::uint32_t) { return std::uint64_t{1} << 44U; },
                                .stride = std::move(stride),
                                .state = std::uint64_t{1} << 45U},
                     .rot = kg::HadamardMatrix(profile_.indexer_head_dim)};
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
        {{0, o_.max_rows},
         {o_.context - o_.max_rows, o_.max_rows},
         {o_.context - 1, 1},
         {o_.max_rows, 1}}};
    for (const auto& [n_past, rows] : probes) {
      auto in = md::Dsv4Chunk(profile_, layout_, n_past, rows);
      if (!in) {
        return std::unexpected(in.error());
      }
      auto planned = PlanDsv4Chunk(model_, kg::Dsv4ShapeOf(layout_, *in), choices, {}, 0, 0);
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
  // Margins: other shapes of these widths place a little differently.
  activation_bytes_ = Round(most_activations + (most_activations / 4), kExtent);
  scratch_bytes_ = Round(most_scratch + (most_scratch / 4) + (1U << 20U), kExtent);
  input_bytes_ = Round((most_inputs * 2) + (1U << 20U), kExtent);

  auto inputs = node_.Pinned(input_bytes_, owner_, staging_);
  auto logits = node_.Pinned(std::uint64_t{profile_.vocab} * sizeof(float), owner_, staging_);
  std::uint64_t table_bytes = 0;
  for (const md::Dsv4Layer& l : binding_.layers) {
    if (l.hash) {
      table_bytes += artifact_->resources()[l.tid2eid.index].bytes.value();
    }
  }
  auto tables = node_.Pinned(std::max<std::uint64_t>(table_bytes, 256), owner_, staging_);
  if (!inputs || !logits || !tables) {
    return Error("pinned staging for DeepSeek");
  }
  inputs_ = *inputs;
  logits_ = *logits;
  hash_tables_ = *tables;
  return {};
}

// The weights' places, one extent per dense chunk, per slab page and per
// host table chunk, cataloged in file order.
Status Dsv4Runner::ReserveWeights() {
  auto& memory = node_.memory();
  const auto groups = artifact_->groups();
  table_group_ = artifact_->resources()[binding_.token_embd.index].group;
  for (std::uint32_t r = 0; r < artifact_->resources().size(); ++r) {
    if (r != binding_.token_embd.index && artifact_->resources()[r].group == table_group_) {
      return Error(
          std::format("{} shares the token table's group", artifact_->resources()[r].name));
    }
  }
  // Each layer's expert groups and stride (the resident layout's).
  const std::uint32_t layers = profile_.layers;
  std::vector<std::uint32_t> first(layers, 0);
  std::vector<SlabLayout> slabs(layers);
  std::vector<std::int64_t> layer_of(groups.size(), -1);
  model_.places.stride.assign(layers, 0);
  for (std::uint32_t il = 0; il < layers; ++il) {
    const md::Dsv4Layer& l = binding_.layers[il];
    std::uint64_t unit = 16;
    std::optional<std::uint32_t> first_group;
    for (const md::Dsv4Tensor* t : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto& a = artifact_->expert_arrays()[t->index];
      auto type = kg::GgmlTypeOf(t->type);
      if (!type) {
        return Error(type.error().detail);
      }
      unit = Lcm(unit, ggml_type_size(*type));
      if (first_group && *first_group != a.first_group) {
        return Error(std::format("layer {}'s expert arrays do not share their groups", il));
      }
      first_group = a.first_group;
    }
    first[il] = first_group.value_or(0);
    const std::uint64_t stored = groups[first[il]].stored.value();
    std::vector<std::uint32_t> shard(profile_.experts);
    std::vector<std::uint64_t> file(profile_.experts);
    for (std::uint32_t e = 0; e < profile_.experts; ++e) {
      const std::uint32_t g = first[il] + e;
      if (g >= groups.size() || groups[g].kind != artifact::GroupKind::kExpert ||
          groups[g].stored.value() != stored) {
        return Error(std::format("layer {}'s expert groups are not uniform", il));
      }
      const auto range = artifact::ChunkRangeOf(artifact_->layout(), {.group = g, .chunk = 0});
      if (!range) {
        return Error("an expert group's file range");
      }
      shard[e] = range->shard;
      file[e] = range->file_offset.value();
      layer_of[g] = il;
    }
    const std::uint64_t stride = Round(stored, unit);
    auto slab = LayOutSlab(shard, file, stored, stride);
    if (!slab) {
      return Error(std::format("layer {}: {}", il, slab.error()));
    }
    slabs[il] = std::move(*slab);
    model_.places.stride[il] = stride;
    slab_padding_ += (stride - stored) * profile_.experts;
  }
  // Places, in file order: dense regions and slabs in one device
  // reservation, the table in host memory.
  std::vector<std::uint32_t> order(groups.size());
  std::ranges::iota(order, 0U);
  std::ranges::sort(order, [&](std::uint32_t a, std::uint32_t b) {
    return std::pair(groups[a].shard, groups[a].offset.value()) <
           std::pair(groups[b].shard, groups[b].offset.value());
  });
  std::vector<std::uint64_t> region(groups.size(), 0);  // dense: its region's offset
  std::vector<std::uint64_t> slab_region(layers, 0);
  std::vector<bool> slab_placed(layers, false);
  for (const std::uint32_t g : order) {
    if (g == table_group_) {
      continue;
    }
    if (layer_of[g] >= 0) {
      const auto il = static_cast<std::uint32_t>(layer_of[g]);
      if (!slab_placed[il]) {
        slab_placed[il] = true;
        slab_region[il] = weights_bytes_;
        weights_bytes_ += slabs[il].bytes();
      }
      continue;
    }
    if (groups[g].kind == artifact::GroupKind::kExpert) {
      return Error(std::format("expert group {} belongs to no bound array", g));
    }
    region[g] = weights_bytes_;
    weights_bytes_ += std::uint64_t{groups[g].chunks} * kExtent;
  }
  auto reservation = memory.Reserve(Bytes(weights_bytes_));
  if (!reservation) {
    return Error(std::format("reserving the weights: {}", reservation.error().detail));
  }
  weights_ = *reservation;
  weights_base_ = memory.RangeOf(weights_).value().base;
  table_bytes_ = std::uint64_t{groups[table_group_].chunks} * kExtent;
  auto host = memory.Reserve(Bytes(table_bytes_));
  if (!host) {
    return Error(std::format("reserving the host table: {}", host.error().detail));
  }
  table_ = *host;
  table_base_ = memory.RangeOf(table_).value().base;
  group_address_.assign(groups.size(), 0);
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    if (g == table_group_) {
      continue;
    }
    if (layer_of[g] >= 0) {
      const auto il = static_cast<std::uint32_t>(layer_of[g]);
      group_address_[g] = weights_base_ + slab_region[il] + slabs[il].delta +
                          (std::uint64_t{g - first[il]} * slabs[il].stride);
    } else {
      group_address_[g] = weights_base_ + region[g];
    }
  }

  const auto add = [&](catalog::ContentKey content,
                       bool host_memory) -> std::expected<ExtentId, std::string> {
    auto extent = node_.catalog().AddExtent({.domain = node_.domain(),
                                             .memory_class = MemoryClass::kWeights,
                                             .recovery = Recovery::kFromArtifact,
                                             .size = Bytes(kExtent),
                                             .content = content});
    if (!extent) {
      return Error(host_memory ? "cataloging a host table chunk" : "cataloging a weight extent");
    }
    return *extent;
  };
  std::ranges::fill(slab_placed, false);
  for (const std::uint32_t g : order) {
    if (layer_of[g] >= 0) {
      const auto il = static_cast<std::uint32_t>(layer_of[g]);
      if (slab_placed[il]) {
        continue;
      }
      slab_placed[il] = true;
      const SlabLayout& slab = slabs[il];
      for (std::uint32_t p = 0; p < slab.pages.size(); ++p) {
        const SlabPage& page = slab.pages[p];
        auto extent = add({.artifact = id_, .group = first[il], .chunk = p}, false);
        if (!extent) {
          return std::unexpected(extent.error());
        }
        const std::uint64_t offset = slab_region[il] + (std::uint64_t{p} * kExtent);
        sc::PageSource source{
            .read = {.fd = shards_.at(page.shard).get(),
                     .offset = page.file_offset,
                     .memory = nullptr,
                     .length = page.length},
            .landed = true,
            .destination = 0,
            .pieces = {},
            .piece_count = page.pieces,
            .backing = sc::BackingPlace{.reservation = weights_,
                                        .offset = Bytes(offset),
                                        .size = Bytes(kExtent),
                                        .allocation_class = node_.device_class()}};
        for (std::size_t i = 0; i < page.pieces; ++i) {
          source.pieces.at(i) =
              sc::LandedPiece{.slot_offset = page.slot_offset.at(i),
                              .destination = weights_base_ + offset + page.page_offset.at(i),
                              .length = Bytes(page.bytes.at(i))};
        }
        read_bytes_ += page.length;
        extents_.push_back({.extent = *extent, .host = false, .offset = offset, .source = source});
      }
      continue;
    }
    const bool host_memory = g == table_group_;
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range = artifact::ChunkRangeOf(artifact_->layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      auto extent = add({.artifact = id_, .group = g, .chunk = c}, host_memory);
      if (!extent) {
        return std::unexpected(extent.error());
      }
      const std::uint64_t offset = (host_memory ? 0 : region[g]) + (std::uint64_t{c} * kExtent);
      const std::uint64_t address = (host_memory ? table_base_ : weights_base_) + offset;
      sc::PageSource source{
          .read = {.fd = shards_.at(range->shard).get(),
                   .offset = range->file_offset.value(),
                   // In place, into host VMM the CPU maps (D-034).
                   // NOLINTNEXTLINE(performance-no-int-to-ptr)
                   .memory = host_memory ? reinterpret_cast<std::byte*>(address) : nullptr,
                   .length = range->length.value()},
          .landed = !host_memory,
          .destination = host_memory ? 0 : address,
          .backing = sc::BackingPlace{
              .reservation = host_memory ? table_ : weights_,
              .offset = Bytes(offset),
              .size = Bytes(kExtent),
              .allocation_class = host_memory ? node_.host_class() : node_.device_class()}};
      read_bytes_ += range->length.value();
      extents_.push_back(
          {.extent = *extent, .host = host_memory, .offset = offset, .source = source});
    }
  }
  return {};
}

Status Dsv4Runner::Register() {
  for (const WeightExtent& w : extents_) {
    auto set = node_.scheduler().SetSource(w.extent, w.source);
    if (!set) {
      return Error(std::format("a DeepSeek weight's source: {}", sc::ToString(set.error())));
    }
    node_.AddSpan({.base = (w.host ? table_base_ : weights_base_) + w.offset,
                   .size = kExtent,
                   .extent = w.extent,
                   .memory_class = MemoryClass::kWeights,
                   .device = !w.host,
                   .owner = owner_});
  }
  if (auto r = RegisterState(); !r) {
    return r;
  }
  // D-090: the places every graph will name stay put for the model's life.
  const std::vector<ExtentId> managed = managed_extents();
  if (auto pinned = node_.scheduler().PinPlaces(managed); !pinned) {
    return Error(std::format("pinning DeepSeek's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

Status Dsv4Runner::CheckPlaces() {
  std::size_t moved = 0;
  std::string first;
  auto checked = node_.Call(
      [&]() -> Status {
        const auto check = [&](ExtentId extent, const sc::PageSource& registered) {
          const sc::PageSource* now = node_.scheduler().SourceOf(extent);
          if (now == nullptr || !sc::SamePlace(*now, registered) ||
              !node_.scheduler().PlacePinned(extent)) {
            if (moved++ == 0) {
              first = std::format("extent {}", extent.index());
            }
          }
        };
        for (const WeightExtent& w : extents_) {
          check(w.extent, w.source);
        }
        for (std::size_t i = 0; i < state_.extents.size() && i < state_sources_.size(); ++i) {
          check(state_.extents[i], state_sources_[i]);
        }
        return {};
      },
      "checking DeepSeek's places");
  if (!checked) {
    return checked;
  }
  if (moved != 0 || state_sources_.size() != state_.extents.size()) {
    DropPlans();
    return Error(
        std::format("{} extents are no longer pinned at their places (first: {}); every "
                    "graph was dropped",
                    moved, first));
  }
  return {};
}

std::size_t Dsv4Runner::graphs() const {
  return static_cast<std::size_t>(
      std::ranges::count_if(plans_, [](const ShapePlan& p) { return p.graph.has_value(); }));
}

// The state's write-back places: one 2 MiB range of an unnamed direct-I/O
// spill file per extent, landed through the zone, its backing managed.
Status Dsv4Runner::RegisterState() {
  std::filesystem::create_directories(o_.out);
  spill_fd_ = ::open(o_.out.c_str(), O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
  if (spill_fd_ < 0) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(),
                             std::generic_category().message(errno)));
  }
  state_sources_.clear();
  for (std::size_t i = 0; i < state_.extents.size(); ++i) {
    const sc::PageSource source{
        .read = {.fd = spill_fd_, .offset = i * kExtent, .memory = nullptr, .length = kExtent},
        .landed = true,
        .destination = state_.base + (i * kExtent),
        .backing = sc::BackingPlace{.reservation = state_.reservation,
                                    .offset = Bytes(i * kExtent),
                                    .size = Bytes(kExtent),
                                    .allocation_class = node_.device_class()},
        .write_back = true};
    auto set = node_.scheduler().SetSource(state_.extents[i], source);
    if (!set) {
      return Error(std::format("the state's write-back place: {}", sc::ToString(set.error())));
    }
    state_sources_.push_back(source);
  }
  state_.backings.clear();  // the VMM lane releases them on eviction (D-033)
  return {};
}

Status Dsv4Runner::Bind() {
  auto& catalog = node_.catalog();
  std::vector<ExtentId> all = weights();
  for (const ts::Mapped* mapped : std::initializer_list<const ts::Mapped*>{
           &state_, &node_.activations(), &node_.pool(), &cublas_workspace_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  everything_ = catalog.ClosureOfExtents(all).value();
  fence_ = catalog.ClosureOfExtents(state_.extents).value();
  model_.places.resource = [this](std::uint32_t resource) {
    const auto& r = artifact_->resources()[resource];
    return group_address_[r.group] + r.offset.value();
  };
  model_.places.array = [this](std::uint32_t array) {
    const auto& a = artifact_->expert_arrays()[array];
    return group_address_[a.first_group] + a.group_offset.value();
  };
  model_.places.state = state_.base;
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

Status Dsv4Runner::Clear() {
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
      "clearing the DeepSeek state", stream_);
}

Status Dsv4Runner::CheckHashRouting() {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> tables;  // address, bytes
  for (const md::Dsv4Layer& l : binding_.layers) {
    if (l.hash) {
      tables.emplace_back(model_.places.resource(l.tid2eid.index),
                          artifact_->resources()[l.tid2eid.index].bytes.value());
    }
  }
  void* host = hash_tables_;
  if (auto r = node_.Job(
          everything_,
          [&tables, host](providers::NativeStream stream) {
            std::uint64_t at = 0;
            for (const auto& [address, bytes] : tables) {
              if (cudaMemcpyAsync(static_cast<std::byte*>(host) + at, Pointer(address), bytes,
                                  cudaMemcpyDeviceToHost,
                                  static_cast<cudaStream_t>(stream.handle)) != cudaSuccess) {
                return sc::JobResult::kUnknown;
              }
              at += bytes;
            }
            return sc::JobResult::kQueued;
          },
          "reading the hash-routing tables", stream_);
      !r) {
    return r;
  }
  std::uint64_t at = 0;
  std::uint32_t il = 0;
  for (const auto& [address, bytes] : tables) {
    const std::span<const std::int32_t> table(
        reinterpret_cast<const std::int32_t*>(static_cast<const std::byte*>(host) + at),
        bytes / sizeof(std::int32_t));
    if (auto checked = md::CheckDsv4HashRouting(profile_, table); !checked) {
      return Error(std::format("hash table {}: {}", il, checked.error()));
    }
    at += bytes;
    ++il;
  }
  return {};
}

std::expected<Dsv4Runner::ShapePlan*, std::string> Dsv4Runner::Planned(
    const kg::Dsv4ChunkShape& shape) {
  const auto found =
      std::ranges::find_if(plans_, [&](const ShapePlan& e) { return e.shape == shape; });
  if (found != plans_.end()) {
    return &*found;
  }
  const auto start = std::chrono::steady_clock::now();
  auto planned = PlanDsv4Chunk(model_, shape, kg::DeviceChoicesOf(*launch_), {},
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
    plans_.erase(plans_.begin());  // its graph with it
  }
  plans_.push_back(ShapePlan{.shape = shape,
                             .planned = std::move(*planned),
                             .eager_runs = 0,
                             .uncapturable = false,
                             .graph = std::nullopt,
                             .copies = {}});
  return &plans_.back();
}

// BP-A1's in-process check, once per planned shape: every tensor the plan
// binds lies in cataloged, resident extents of device memory of one class,
// and each has the class it should: weights, the state (live state) or the
// activations (scratch).
void Dsv4Runner::Check(const kg::Dsv4Graph& graph) {
  std::vector<const ggml_tensor*> state;
  for (const kg::Dsv4LayerTensors& l : graph.layers) {
    for (const ggml_tensor* t :
         {l.raw_k, l.csa_k, l.csa_state_kv, l.csa_state_score, l.lid_k, l.lid_state_kv,
          l.lid_state_score, l.hca_k, l.hca_state_kv, l.hca_state_score}) {
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
    return base->op == GGML_OP_NONE && std::ranges::find(inputs, base) == inputs.end()
               ? MemoryClass::kWeights
               : MemoryClass::kScratch;
  };
  const auto expect = [&](const ggml_tensor* t) {
    ++coverage_tensors_;
    const std::optional<MemoryClass> covered =
        node_.Covered(Address(t->data), ggml_nbytes(t), owner_);
    if (covered != kind_of(t) && coverage_violations_++ == 0) {
      first_violation_ =
          std::format("{} ({} bytes at {:#x})", t->name, ggml_nbytes(t), Address(t->data));
    }
  };
  for (const ggml_tensor* node : graph.nodes) {
    expect(node);
    for (const ggml_tensor* src : node->src) {
      if (src != nullptr) {
        expect(src);
      }
    }
  }
}

Status Dsv4Runner::Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
                         std::vector<float>& logits, const std::function<Status()>& meanwhile) {
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, rows);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto planned = Planned(kg::Dsv4ShapeOf(layout_, *in));
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ShapePlan& entry = **planned;
  Dsv4Planned* p = entry.planned.get();
  const kg::Dsv4Graph& g = p->graph;
  // The host table, which the job's lease holds resident.
  const std::span<const std::byte> table(
      reinterpret_cast<const std::byte*>(table_base_),  // NOLINT(performance-no-int-to-ptr)
      table_bytes_);
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  // Decode graphs (D-090): replay a shape's graph; capture a one-row
  // shape that has run once launch by launch; otherwise launch by launch.
  const bool replay = graphs_ && entry.graph.has_value();
  const bool capture =
      graphs_ && !replay && rows == 1 && !entry.uncapturable && entry.eager_runs > 0;
  if (capture && graphs() >= kMaxGraphs) {
    // Graph memory is the driver's, outside the catalog: at most
    // kMaxGraphs are kept. Decode moves on to later shapes, so the oldest
    // goes (no job is in flight between chunks: nothing replays it).
    const auto oldest =
        std::ranges::find_if(plans_, [](const ShapePlan& p) { return p.graph.has_value(); });
    oldest->graph.reset();
    oldest->copies.clear();
    ++graph_stats_.dropped;
  }
  Dsv4Path path = Dsv4Path::kEager;
  Status ran;
  Dsv4HostInputs host;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    // The embedding rows and the chunk plan's inputs, staged in order.
    if (auto r = BuildDsv4Inputs(model_, g, *in, tokens, table, host); !r) {
      ran = std::unexpected(r.error());
      return sc::JobResult::kNotStarted;
    }
    auto* const stream = static_cast<cudaStream_t>(native.handle);
    std::vector<std::array<std::uint64_t, 3>> copies;  // to, bytes, staging offset
    copies.reserve(host.sources.size());
    std::uint64_t staged = 0;
    for (const auto& [tensor, source] : host.sources) {
      const std::uint64_t bytes = ggml_nbytes(tensor);
      if (staged + bytes > input_bytes_) {
        ran = Error("the inputs exceed their staging");
        return sc::JobResult::kNotStarted;
      }
      std::memcpy(static_cast<std::byte*>(inputs_) + staged, source, bytes);
      copies.push_back({Address(tensor->data), bytes, staged});
      staged += Round(bytes, 256);
    }
    // The input copies, the plan and the logits copy, as one run queues
    // them and a capture records them.
    const auto queue = [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
      const auto unknown = [](std::string what) {
        return std::unexpected(
            kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(what)});
      };
      for (const auto& [to, bytes, at] : copies) {
        if (const cudaError_t copied =
                cudaMemcpyAsync(Pointer(to), static_cast<const std::byte*>(inputs_) + at, bytes,
                                cudaMemcpyHostToDevice, stream);
            copied != cudaSuccess) {
          return unknown(std::format("an input copy: {}", cudaGetErrorString(copied)));
        }
      }
      if (auto r = p->bound->Run(launch); !r) {
        return r;
      }
      // The last row's logits: the next token's.
      if (const cudaError_t copied = cudaMemcpyAsync(
              logits_,
              static_cast<const std::byte*>(g.logits->data) + (std::uint64_t{rows - 1} * row_bytes),
              row_bytes, cudaMemcpyDeviceToHost, stream);
          copied != cudaSuccess) {
        return unknown(std::format("the logits copy: {}", cudaGetErrorString(copied)));
      }
      return {};
    };
    std::expected<void, kg::KernelFailure> queued;
    bool before = false;  // work queued before `queued`'s outcome
    if (replay) {
      // What the graph copies must be where the host staged it.
      if (copies != entry.copies) {
        ran = Error("the inputs' staging differs from the captured graph's");
        return sc::JobResult::kNotStarted;
      }
      path = Dsv4Path::kReplayed;
      queued = launch_->Launch(*entry.graph);
    } else {
      if (capture) {
        std::size_t free_before = 0;
        std::size_t free_after = 0;
        std::size_t total = 0;
        (void)cudaMemGetInfo(&free_before, &total);
        auto captured = launch_->Capture(queue);
        (void)cudaMemGetInfo(&free_after, &total);
        (void)cudaGetLastError();
        if (captured) {
          graph_stats_.capture_seconds += captured->capture_seconds();
          graph_stats_.instantiate_seconds += captured->instantiate_seconds();
          graph_stats_.nodes += captured->nodes();
          graph_stats_.memory_bytes +=
              static_cast<std::int64_t>(free_before) - static_cast<std::int64_t>(free_after);
          entry.graph.emplace(std::move(*captured));
          entry.copies = copies;
          path = Dsv4Path::kCaptured;
          before = true;  // the upload
          queued = launch_->Launch(*entry.graph);
        } else if (captured.error().error == kg::KernelError::kUnknown) {
          ran = Error(std::format("chunk at {}: {}", n_past, captured.error().detail));
          return sc::JobResult::kUnknown;
        } else {
          // Refused, with nothing queued: this shape runs launch by launch.
          entry.uncapturable = true;
          if (graph_stats_.refused++ == 0) {
            graph_stats_.first_refusal = captured.error().detail;
          }
        }
      }
      if (path == Dsv4Path::kEager) {
        ++entry.eager_runs;
        queued = queue(*launch_);
        before = true;  // any input copy before a refusal
      }
    }
    last_submit_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!queued) {
      ran = Error(std::format("chunk at {}: {}", n_past, queued.error().detail));
      if (queued.error().error == kg::KernelError::kUnknown) {
        return sc::JobResult::kUnknown;
      }
      return before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    return sc::JobResult::kQueued;
  };
  Status posted;
  Status alongside;
  if (meanwhile) {
    // Submitted without waiting; the frame (and `job`'s references) lives
    // until Await has seen the program gone.
    ts::Done done;
    const std::uint64_t request =
        node_.Submit(std::make_unique<ts::RunProgram>(done, everything_, std::move(job), stream_));
    alongside = meanwhile();
    posted = node_.Await(done, "a DeepSeek chunk", request);
  } else {
    posted = node_.Job(everything_, std::move(job), "a DeepSeek chunk", stream_);
  }
  if (!posted || !ran || !alongside) {
    if (!ran) {
      return ran;
    }
    return Error(
        std::format("chunk at {}: {}", n_past, !posted ? posted.error() : alongside.error()));
  }
  if (launch_->faulted()) {
    return Error(std::format("chunk at {}: the launch context faulted", n_past));
  }
  last_path_ = path;
  switch (path) {
    case Dsv4Path::kEager:
      ++graph_stats_.eager;
      break;
    case Dsv4Path::kCaptured:
      ++graph_stats_.captured;
      break;
    case Dsv4Path::kReplayed:
      ++graph_stats_.replayed;
      break;
  }
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + profile_.vocab);
  return {};
}

std::expected<double, std::string> Dsv4Runner::TimeReplays(std::uint32_t n_past, std::int32_t token,
                                                           std::uint32_t count) {
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, 1);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto planned = Planned(kg::Dsv4ShapeOf(layout_, *in));
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ShapePlan& entry = **planned;
  if (!entry.graph) {
    return Error("the step's shape has no graph yet");
  }
  const std::span<const std::byte> table(
      reinterpret_cast<const std::byte*>(table_base_),  // NOLINT(performance-no-int-to-ptr)
      table_bytes_);
  Dsv4HostInputs host;
  if (auto r =
          BuildDsv4Inputs(model_, entry.planned->graph, *in, std::span(&token, 1), table, host);
      !r) {
    return std::unexpected(r.error());
  }
  std::vector<std::array<std::uint64_t, 3>> copies;
  std::uint64_t staged = 0;
  for (const auto& [tensor, source] : host.sources) {
    const std::uint64_t bytes = ggml_nbytes(tensor);
    if (staged + bytes > input_bytes_) {
      return Error("the inputs exceed their staging");
    }
    std::memcpy(static_cast<std::byte*>(inputs_) + staged, source, bytes);
    copies.push_back({Address(tensor->data), bytes, staged});
    staged += Round(bytes, 256);
  }
  if (copies != entry.copies) {
    return Error("the inputs' staging differs from the captured graph's");
  }
  std::string failed;
  const auto start = std::chrono::steady_clock::now();
  auto posted = node_.Job(
      everything_,
      [&](providers::NativeStream) {
        for (std::uint32_t i = 0; i < count; ++i) {
          if (auto r = launch_->Launch(*entry.graph); !r) {
            failed = r.error().detail;
            if (r.error().error == kg::KernelError::kUnknown) {
              return sc::JobResult::kUnknown;
            }
            return i == 0 ? sc::JobResult::kNotStarted : sc::JobResult::kFailed;
          }
        }
        return sc::JobResult::kQueued;
      },
      "back-to-back replays", stream_);
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (!posted || !failed.empty()) {
    return Error(failed.empty() ? posted.error() : failed);
  }
  return seconds / count;
}

Status Dsv4Runner::Release() {
  if (released_) {
    return {};
  }
  released_ = true;
  std::vector<std::string> problems;
  plans_.clear();
  launch_.reset();
  cublas_.reset();
  auto& memory = node_.memory();
  for (ts::Mapped* mapped : {&state_, &cublas_workspace_}) {
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
  for (const providers::ReservationId reservation : {weights_, table_}) {
    if (reservation.valid() && !memory.Free(reservation)) {
      problems.emplace_back("a DeepSeek weights reservation still has mappings");
    }
  }
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
  }
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
