// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/dsv4_runner.h"

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
#include "kernels/ggml/ops_ext.h"
#include "platform/direct_io.h"
#include "providers/device_runtime.h"
#include "providers/direct_reader.h"
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
constexpr std::uint64_t kFileAlignment = 4096;
// The most snapshot ranges a verify saves: at most 8 rows of about 230
// ranges each (a cell a layer, a CSA layer's six ring and compressed rows,
// an HCA layer's three, the drafter's three cells), and the scratch rows.
constexpr std::uint32_t kRangeCapacity = 4096;

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

std::vector<ExtentId> Dsv4Runner::state() const {
  std::vector<ExtentId> all = state_.extents;
  all.insert(all.end(), dstate_.extents.begin(), dstate_.extents.end());
  return all;
}

std::vector<ExtentId> Dsv4Runner::managed_extents() const {
  std::vector<ExtentId> all = weights();
  const std::vector<ExtentId> live = state();
  all.insert(all.end(), live.begin(), live.end());
  return all;
}

Status Dsv4Runner::OpenPart(Part& part, const std::filesystem::path& path,
                            std::unique_ptr<artifact::Artifact>& opened) {
  auto artifact = artifact::Artifact::Open(path);
  if (!artifact) {
    return Error(
        std::format("the artifact {} was refused: {}", path.string(), artifact.error().reason));
  }
  opened = std::make_unique<artifact::Artifact>(std::move(*artifact));
  part.artifact = opened.get();
  for (std::uint32_t s = 0; s < opened->shards().size(); ++s) {
    auto fd = opened->OpenShardForDirectRead(s);
    if (!fd) {
      return Error(
          std::format("shard {} of {} cannot be opened for direct reads", s, path.string()));
    }
    part.shards.push_back(std::move(*fd));
  }
  // The artifact's identity orders victim ties (catalog.h ContentKey).
  for (std::size_t i = 0; i < part.id.size() && (2 * i) + 1 < opened->id().size(); ++i) {
    (void)std::from_chars(opened->id().data() + (2 * i), opened->id().data() + (2 * i) + 2,
                          part.id.at(i), 16);
  }
  return {};
}

Status Dsv4Runner::Setup() {
  if (auto r = OpenPart(target_, o_.artifact, artifact_); !r) {
    return r;
  }
  auto binding = md::BindDsv4(profile_, *artifact_);
  if (!binding) {
    return std::unexpected(binding.error());
  }
  binding_ = std::move(*binding);
  target_.binding = &binding_;
  target_.layers = profile_.layers;
  auto layout = md::Dsv4State(profile_, o_.context, o_.max_rows);
  if (!layout) {
    return std::unexpected(layout.error());
  }
  layout_ = std::move(*layout);
  // The largest shapes are planned at the context's end and a decode step
  // after a whole chunk (below): checked here, so those positions cannot
  // wrap.
  if (o_.max_rows >= o_.context || (!o_.drafter.empty() && o_.max_verify > o_.context)) {
    return Error(std::format("a context of {} leaves no room for chunks of {} rows", o_.context,
                             o_.max_rows));
  }
  if (!o_.drafter.empty()) {
    if (o_.draft_rows == 0 || o_.draft_rows > dprofile_.block_size || o_.max_verify == 0 ||
        o_.max_verify > o_.draft_rows + 1 || std::cmp_greater(o_.max_verify, kg::kRowsMaxColumns)) {
      return Error(
          std::format("a draft block of 1 to {} rows, a verify of 1 to the drafts + 1 "
                      "(at most {}) rows",
                      dprofile_.block_size, kg::kRowsMaxColumns));
    }
    if (auto r = OpenPart(drafter_, o_.drafter, dartifact_); !r) {
      return r;
    }
    auto dbinding = md::BindDspark(dprofile_, *dartifact_, profile_, binding_);
    if (!dbinding) {
      return Error(std::format("the drafter: {}", dbinding.error()));
    }
    dbinding_ = std::move(*dbinding);
    drafter_.binding = &dbinding_.blocks;
    drafter_.layers = dprofile_.blocks.layers;
    auto dlayout = md::DsparkState(dprofile_, o_.draft_rows);
    if (!dlayout) {
      return std::unexpected(dlayout.error());
    }
    dlayout_ = std::move(*dlayout);
  }
  // The state first: its extents come before the weights' in a closure, so
  // a swap back restores it before paging the weights in.
  if (auto r = node_.MapResident(state_, "the DeepSeek state", layout_.bytes, BackingKind::kDevice,
                                 MemoryClass::kLiveState, Recovery::kPreserve, owner_);
      !r) {
    return r;
  }
  if (speculative()) {
    if (auto r = node_.MapResident(dstate_, "the DSpark ring", dlayout_.bytes, BackingKind::kDevice,
                                   MemoryClass::kLiveState, Recovery::kPreserve, owner_);
        !r) {
      return r;
    }
  }
  if (auto r = ReserveWeights(); !r) {
    return r;
  }

  // cuBLAS, with upstream's workspace for the device, as the resident
  // harness has it: the router's BF16 products run there at prefill widths.
  const providers::DeviceFacts facts =
      providers::QueryDeviceFacts(0).value_or(providers::DeviceFacts{});
  cublas_bytes_ = kg::CublasHandle::UpstreamWorkspace(static_cast<int>(facts.architecture)).value();
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
  // end, planned over placeless addresses; beside a drafter, the same
  // chunks with its injection, verifies and its draft block too.
  const auto placeless = [](std::uint32_t) { return std::uint64_t{1} << 44U; };
  model_ = Dsv4Model{.artifact = artifact_.get(),
                     .profile = &profile_,
                     .binding = &binding_,
                     .state = &layout_,
                     .places = {.resource = placeless,
                                .array = placeless,
                                .stride = target_.stride,
                                .state = std::uint64_t{1} << 45U},
                     .rot = kg::HadamardMatrix(profile_.indexer_head_dim),
                     .exact = o_.exact};
  if (speculative()) {
    dmodel_ = DsparkModel{.artifact = dartifact_.get(),
                          .profile = &dprofile_,
                          .binding = &dbinding_,
                          .state = &dlayout_,
                          .places = {.resource = placeless,
                                     .array = placeless,
                                     .stride = drafter_.stride,
                                     .state = std::uint64_t{1} << 45U},
                          .target_resource = placeless,
                          .exact = o_.exact};
  }
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
      Dsv4ChunkKind kind;
    };
    std::vector<Probe> probes = {{0, o_.max_rows, Dsv4ChunkKind::kPlain},
                                 {o_.context - o_.max_rows, o_.max_rows, Dsv4ChunkKind::kPlain},
                                 {o_.context - 1, 1, Dsv4ChunkKind::kPlain},
                                 {o_.max_rows, 1, Dsv4ChunkKind::kPlain}};
    if (speculative()) {
      probes.push_back({0, o_.max_rows, Dsv4ChunkKind::kInject});
      probes.push_back({o_.context - o_.max_rows, o_.max_rows, Dsv4ChunkKind::kInject});
      probes.push_back({0, o_.max_verify, Dsv4ChunkKind::kVerify});
      probes.push_back({o_.context - o_.max_verify, o_.max_verify, Dsv4ChunkKind::kVerify});
    }
    for (const Probe& probe : probes) {
      auto in = md::Dsv4Chunk(profile_, layout_, probe.n_past, probe.rows);
      if (!in) {
        return std::unexpected(in.error());
      }
      Dsv4Speculation speculation;
      if (probe.kind != Dsv4ChunkKind::kPlain) {
        speculation = {.verify = probe.kind == Dsv4ChunkKind::kVerify,
                       .drafter = &dmodel_,
                       .inject_rows = static_cast<std::int64_t>(
                           md::DsparkInject(dlayout_, probe.n_past, probe.rows).cells.size())};
      }
      auto planned =
          PlanDsv4Chunk(model_, kg::Dsv4ShapeOf(layout_, *in), choices, {}, 0, 0, speculation);
      if (!planned) {
        return Error(std::format("measuring a chunk of {} at {}: {}", probe.rows, probe.n_past,
                                 planned.error()));
      }
      most_activations = std::max(most_activations, (*planned)->placement.extent);
      auto scratch = kg::PlanScratch(**measure, (*planned)->plan);
      if (!scratch) {
        return Error(scratch.error().detail);
      }
      most_scratch = std::max(most_scratch, *scratch);
      most_inputs = std::max(most_inputs, (*planned)->inputs_bytes);
    }
    if (speculative()) {
      auto planned = PlanDsparkDraft(dmodel_, o_.draft_rows, choices, 0, 0);
      if (!planned) {
        return Error(std::format("measuring the draft block: {}", planned.error()));
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
  // A verify's inputs from the staging's second half, which the largest
  // inputs fit.
  verify_base_ = Round(input_bytes_ / 2, 256);

  const std::uint64_t logit_rows = speculative() ? o_.max_verify : 1;
  auto inputs = node_.Pinned(input_bytes_, owner_, staging_);
  auto logits =
      node_.Pinned(logit_rows * std::uint64_t{profile_.vocab} * sizeof(float), owner_, staging_);
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
  if (speculative()) {
    // A verify's snapshot: every byte it may write, the target's and the
    // ring's (D-068 working state).
    const std::uint64_t snapshot =
        md::Dsv4VerifySnapshotBytes(profile_, layout_, o_.max_verify) +
        (std::uint64_t{o_.max_verify} * dprofile_.blocks.layers * dprofile_.blocks.head_dim * 2);
    if (auto r = node_.MapResident(snapshot_, "the DeepSeek verify snapshot", snapshot,
                                   BackingKind::kDevice, MemoryClass::kRuntime, Recovery::kPinned,
                                   owner_);
        !r) {
      return r;
    }
    range_capacity_ = kRangeCapacity;
    auto save = node_.Pinned(range_capacity_ * sizeof(kg::RangeCopy), owner_, staging_);
    auto restore = node_.Pinned(range_capacity_ * sizeof(kg::RangeCopy), owner_, staging_);
    auto drafts = node_.Pinned(std::max<std::uint64_t>(o_.draft_rows * sizeof(std::int32_t), 256),
                               owner_, staging_);
    if (!save || !restore || !drafts) {
      return Error("pinned staging for DeepSeek's speculation");
    }
    save_ = static_cast<kg::RangeCopy*>(*save);
    restore_ = static_cast<kg::RangeCopy*>(*restore);
    drafts_ = *drafts;
  }
  return {};
}

Status Dsv4Runner::ReserveWeights() {
  if (auto r = ReservePart(target_, true, read_bytes_); !r) {
    return r;
  }
  if (speculative()) {
    if (auto r = ReservePart(drafter_, false, dread_bytes_); !r) {
      return Error(std::format("the drafter: {}", r.error()));
    }
    read_bytes_ += dread_bytes_;
  }
  return {};
}

// A part's places, one extent per dense chunk, per slab page and per host
// table chunk, cataloged in file order.
Status Dsv4Runner::ReservePart(Part& part, bool table, std::uint64_t& read_bytes) {
  auto& memory = node_.memory();
  const artifact::Artifact& a = *part.artifact;
  const auto groups = a.groups();
  std::optional<std::uint32_t> table_group;
  if (table) {
    table_group_ = a.resources()[part.binding->token_embd.index].group;
    table_group = table_group_;
    for (std::uint32_t r = 0; r < a.resources().size(); ++r) {
      if (r != part.binding->token_embd.index && a.resources()[r].group == table_group_) {
        return Error(std::format("{} shares the token table's group", a.resources()[r].name));
      }
    }
  }
  // Each layer's expert groups and stride (the resident layout's).
  const std::uint32_t layers = part.layers;
  std::vector<std::uint32_t> first(layers, 0);
  std::vector<SlabLayout> slabs(layers);
  std::vector<std::int64_t> layer_of(groups.size(), -1);
  part.stride.assign(layers, 0);
  for (std::uint32_t il = 0; il < layers; ++il) {
    const md::Dsv4Layer& l = part.binding->layers[il];
    std::uint64_t unit = 16;
    std::optional<std::uint32_t> first_group;
    for (const md::Dsv4Tensor* t : {&l.gate_exps, &l.up_exps, &l.down_exps}) {
      const auto& e = a.expert_arrays()[t->index];
      auto type = kg::GgmlTypeOf(t->type);
      if (!type) {
        return Error(type.error().detail);
      }
      unit = Lcm(unit, ggml_type_size(*type));
      if (first_group && *first_group != e.first_group) {
        return Error(std::format("layer {}'s expert arrays do not share their groups", il));
      }
      first_group = e.first_group;
    }
    first[il] = first_group.value_or(0);
    const std::uint64_t stored = groups[first[il]].stored.value();
    const std::uint32_t experts =
        part.binding == &binding_ ? profile_.experts : dprofile_.blocks.experts;
    std::vector<std::uint32_t> shard(experts);
    std::vector<std::uint64_t> file(experts);
    for (std::uint32_t e = 0; e < experts; ++e) {
      const std::uint32_t g = first[il] + e;
      if (g >= groups.size() || groups[g].kind != artifact::GroupKind::kExpert ||
          groups[g].stored.value() != stored) {
        return Error(std::format("layer {}'s expert groups are not uniform", il));
      }
      const auto range = artifact::ChunkRangeOf(a.layout(), {.group = g, .chunk = 0});
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
    part.stride[il] = stride;
    slab_padding_ += (stride - stored) * experts;
  }
  // Places, in file order: dense regions and slabs in one device
  // reservation, the table in host memory.
  std::vector<std::uint32_t> order(groups.size());
  std::ranges::iota(order, 0U);
  std::ranges::sort(order, [&](std::uint32_t x, std::uint32_t y) {
    return std::pair(groups[x].shard, groups[x].offset.value()) <
           std::pair(groups[y].shard, groups[y].offset.value());
  });
  std::vector<std::uint64_t> region(groups.size(), 0);  // dense: its region's offset
  std::vector<std::uint64_t> slab_region(layers, 0);
  std::vector<bool> slab_placed(layers, false);
  for (const std::uint32_t g : order) {
    if (g == table_group) {
      continue;
    }
    if (layer_of[g] >= 0) {
      const auto il = static_cast<std::uint32_t>(layer_of[g]);
      if (!slab_placed[il]) {
        slab_placed[il] = true;
        slab_region[il] = part.bytes;
        part.bytes += slabs[il].bytes();
      }
      continue;
    }
    if (groups[g].kind == artifact::GroupKind::kExpert) {
      return Error(std::format("expert group {} belongs to no bound array", g));
    }
    region[g] = part.bytes;
    part.bytes += std::uint64_t{groups[g].chunks} * kExtent;
  }
  auto reservation = memory.Reserve(Bytes(part.bytes));
  if (!reservation) {
    return Error(std::format("reserving the weights: {}", reservation.error().detail));
  }
  part.weights = *reservation;
  part.base = memory.RangeOf(part.weights).value().base;
  if (table_group) {
    table_bytes_ = std::uint64_t{groups[*table_group].chunks} * kExtent;
    auto host = memory.Reserve(Bytes(table_bytes_));
    if (!host) {
      return Error(std::format("reserving the host table: {}", host.error().detail));
    }
    table_ = *host;
    table_base_ = memory.RangeOf(table_).value().base;
  }
  part.group_address.assign(groups.size(), 0);
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    if (g == table_group) {
      continue;
    }
    if (layer_of[g] >= 0) {
      const auto il = static_cast<std::uint32_t>(layer_of[g]);
      part.group_address[g] = part.base + slab_region[il] + slabs[il].delta +
                              (std::uint64_t{g - first[il]} * slabs[il].stride);
    } else {
      part.group_address[g] = part.base + region[g];
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
        auto extent = add({.artifact = part.id, .group = first[il], .chunk = p}, false);
        if (!extent) {
          return std::unexpected(extent.error());
        }
        const std::uint64_t offset = slab_region[il] + (std::uint64_t{p} * kExtent);
        sc::PageSource source{
            .read = {.fd = part.shards.at(page.shard).get(),
                     .offset = page.file_offset,
                     .memory = nullptr,
                     .length = page.length},
            .landed = true,
            .destination = 0,
            .pieces = {},
            .piece_count = page.pieces,
            .backing = sc::BackingPlace{.reservation = part.weights,
                                        .offset = Bytes(offset),
                                        .size = Bytes(kExtent),
                                        .allocation_class = node_.device_class()}};
        for (std::size_t i = 0; i < page.pieces; ++i) {
          source.pieces.at(i) =
              sc::LandedPiece{.slot_offset = page.slot_offset.at(i),
                              .destination = part.base + offset + page.page_offset.at(i),
                              .length = Bytes(page.bytes.at(i))};
        }
        read_bytes += page.length;
        extents_.push_back({.extent = *extent,
                            .host = false,
                            .address = part.base + offset,
                            .source = source,
                            .drafter = &part == &drafter_,
                            .group = first[il]});
      }
      continue;
    }
    const bool host_memory = g == table_group;
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range = artifact::ChunkRangeOf(a.layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      auto extent = add({.artifact = part.id, .group = g, .chunk = c}, host_memory);
      if (!extent) {
        return std::unexpected(extent.error());
      }
      const std::uint64_t offset = (host_memory ? 0 : region[g]) + (std::uint64_t{c} * kExtent);
      const std::uint64_t address = (host_memory ? table_base_ : part.base) + offset;
      sc::PageSource source{
          .read = {.fd = part.shards.at(range->shard).get(),
                   .offset = range->file_offset.value(),
                   // In place, into host VMM the CPU maps (D-034).
                   // NOLINTNEXTLINE(performance-no-int-to-ptr)
                   .memory = host_memory ? reinterpret_cast<std::byte*>(address) : nullptr,
                   .length = range->length.value()},
          .landed = !host_memory,
          .destination = host_memory ? 0 : address,
          .backing = sc::BackingPlace{
              .reservation = host_memory ? table_ : part.weights,
              .offset = Bytes(offset),
              .size = Bytes(kExtent),
              .allocation_class = host_memory ? node_.host_class() : node_.device_class()}};
      read_bytes += range->length.value();
      extents_.push_back({.extent = *extent,
                          .host = host_memory,
                          .address = address,
                          .source = source,
                          .drafter = &part == &drafter_,
                          .group = g});
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
    node_.AddSpan({.base = w.address,
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
        for (std::size_t i = 0; i < dstate_.extents.size() && i < dstate_sources_.size(); ++i) {
          check(dstate_.extents[i], dstate_sources_[i]);
        }
        return {};
      },
      "checking DeepSeek's places");
  if (!checked) {
    return checked;
  }
  if (moved != 0 || state_sources_.size() != state_.extents.size() ||
      dstate_sources_.size() != dstate_.extents.size()) {
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
// spill file per extent (the target's, then the drafter's ring), landed
// through the zone, its backing managed.
Status Dsv4Runner::RegisterState() {
  std::filesystem::create_directories(o_.out);
  const auto opened = platform::OpenUnnamedDirectFile(o_.out);
  if (!opened) {
    return Error(std::format("the spill file in {}: {}", o_.out.string(),
                             std::generic_category().message(opened.error())));
  }
  spill_fd_ = *opened;
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
  return spill(dstate_, dstate_sources_);
}

Status Dsv4Runner::Bind() {
  auto& catalog = node_.catalog();
  std::vector<ExtentId> all = weights();
  for (const Mapped* mapped :
       std::initializer_list<const Mapped*>{&state_, &dstate_, &node_.activations(), &node_.pool(),
                                            &cublas_workspace_, &snapshot_}) {
    all.insert(all.end(), mapped->extents.begin(), mapped->extents.end());
  }
  all.insert(all.end(), staging_.begin(), staging_.end());
  everything_ = catalog.ClosureOfExtents(all).value();
  fence_ = catalog.ClosureOfExtents(state()).value();
  if (speculative()) {
    // A draft reads the drafter's weights, the target's head (its group)
    // and token table; its job restores a verify's rows first (both states
    // and the snapshot).
    const std::uint32_t head = artifact_->resources()[binding_.output.index].group;
    std::vector<ExtentId> draft;
    for (const WeightExtent& w : extents_) {
      if (w.drafter || w.host || w.group == head) {
        draft.push_back(w.extent);
      }
    }
    for (const Mapped* mapped :
         std::initializer_list<const Mapped*>{&state_, &dstate_, &node_.activations(),
                                              &node_.pool(), &cublas_workspace_, &snapshot_}) {
      draft.insert(draft.end(), mapped->extents.begin(), mapped->extents.end());
    }
    draft.insert(draft.end(), staging_.begin(), staging_.end());
    draft_closure_ = catalog.ClosureOfExtents(draft).value();
  }
  model_.places.resource = [this](std::uint32_t resource) {
    const auto& r = artifact_->resources()[resource];
    return target_.group_address[r.group] + r.offset.value();
  };
  model_.places.array = [this](std::uint32_t array) {
    const auto& a = artifact_->expert_arrays()[array];
    return target_.group_address[a.first_group] + a.group_offset.value();
  };
  model_.places.state = state_.base;
  if (speculative()) {
    dmodel_.places.resource = [this](std::uint32_t resource) {
      const auto& r = dartifact_->resources()[resource];
      return drafter_.group_address[r.group] + r.offset.value();
    };
    dmodel_.places.array = [this](std::uint32_t array) {
      const auto& a = dartifact_->expert_arrays()[array];
      return drafter_.group_address[a.first_group] + a.group_offset.value();
    };
    dmodel_.places.state = dstate_.base;
    dmodel_.target_resource = model_.places.resource;
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

Status Dsv4Runner::Clear() {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> regions = {{state_.base, layout_.bytes}};
  if (speculative()) {
    regions.emplace_back(dstate_.base, dlayout_.bytes);
  }
  restore_count_ = 0;
  save_count_ = 0;
  saved_.clear();
  verify_rows_ = 0;
  // Unusable until zeroed (a pending restore is dropped); a quarantine
  // lifts once the clear has run.
  quarantined_ = true;
  auto cleared = node_.Job(
      fence_,
      [regions](providers::NativeStream stream) {
        for (const auto& [base, bytes] : regions) {
          if (!providers::FillAsync(stream, Pointer(base), 0, bytes).ok()) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      "clearing the DeepSeek state", stream_);
  if (!cleared) {
    return cleared;
  }
  quarantined_ = false;
  return {};
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
              if (!providers::CopyAsync(stream, static_cast<std::byte*>(host) + at,
                                        Pointer(address), bytes, providers::CopyKind::kDeviceToHost)
                       .ok()) {
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
    const kg::Dsv4ChunkShape& shape, Dsv4ChunkKind kind, std::int64_t inject_rows) {
  const auto found = std::ranges::find_if(plans_, [&](const ShapePlan& e) {
    return e.shape == shape && e.kind == kind && e.inject_rows == inject_rows;
  });
  if (found != plans_.end()) {
    return &*found;
  }
  const auto start = std::chrono::steady_clock::now();
  Dsv4Speculation speculation;
  if (kind != Dsv4ChunkKind::kPlain) {
    speculation = {
        .verify = kind == Dsv4ChunkKind::kVerify, .drafter = &dmodel_, .inject_rows = inject_rows};
  }
  auto planned = PlanDsv4Chunk(model_, shape, kg::DeviceChoicesOf(*launch_), dump_,
                               node_.activations().base, node_.activations().bytes, speculation);
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
  entry.inject_rows = inject_rows;
  entry.planned = std::move(*planned);
  return &entry;
}

std::expected<Dsv4Runner::DraftPlan*, std::string> Dsv4Runner::PlannedDraft() {
  if (!dplans_.empty()) {
    return &dplans_.front();
  }
  const auto start = std::chrono::steady_clock::now();
  auto planned = PlanDsparkDraft(dmodel_, o_.draft_rows, kg::DeviceChoicesOf(*launch_),
                                 node_.activations().base, node_.activations().bytes);
  if (!planned) {
    return std::unexpected(planned.error());
  }
  auto scratch = kg::PlanScratch(*launch_, (*planned)->plan);
  if (!scratch) {
    return Error(scratch.error().detail);
  }
  if (*scratch > launch_->workspace().size.value()) {
    return Error(std::format("the draft's scratch ({} bytes) exceeds the pool ({} bytes)", *scratch,
                             launch_->workspace().size.value()));
  }
  (*planned)->scratch = *scratch;
  auto bound = kg::BoundGraph::Bind(*registry_, (*planned)->plan);
  if (!bound) {
    return Error(bound.error().detail);
  }
  (*planned)->bound.emplace(std::move(*bound));
  plan_seconds_ += Seconds(std::chrono::steady_clock::now() - start);
  dplans_.emplace_back().planned = std::move(*planned);
  return &dplans_.front();
}

// BP-A1's in-process check, once per planned shape: every tensor the plan
// binds lies in cataloged, resident extents of device memory of one class,
// and each has the class it should: weights, the state (live state: the
// target's and the drafter's ring) or the activations (scratch).
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
  if (graph.inject) {
    state.insert(state.end(), graph.inject->ring.begin(), graph.inject->ring.end());
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

std::expected<void, kg::KernelFailure> Dsv4Runner::QueueRestore() {
  if (restore_count_ == 0) {
    return {};
  }
  // Still owed until the copy is queued.
  if (auto r = kg::CopyRanges(*launch_, restore_, restore_count_); !r) {
    return r;
  }
  restore_count_ = 0;
  return {};
}

void Dsv4Runner::Settle(bool saved, bool wrote, bool unknown) {
  verify_rows_ = 0;
  if (unknown || launch_->faulted()) {
    quarantined_ = true;
    return;
  }
  if (saved) {
    // The whole verify undone: every range it saved, before the next
    // job's own work (its queued work has completed: the job has retired).
    std::uint32_t count = 0;
    for (const Saved& s : saved_) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
    restore_count_ = count;
    return;
  }
  if (wrote) {
    quarantined_ = true;
  }
}

Status Dsv4Runner::Usable() const {
  if (quarantined_) {
    return Error(
        "the DeepSeek state is quarantined: a job failed after it may have written it "
        "(Clear first)");
  }
  return {};
}

Status Dsv4Runner::PlanSnapshot(const md::Dsv4ChunkInputs& in) {
  const md::Dsv4Writes writes = md::Dsv4ChunkWrites(profile_, layout_, in);
  const std::vector<std::vector<md::StateRange>> ring =
      md::DsparkWrites(dprofile_, dlayout_, in.n_past, in.rows);
  saved_.clear();
  save_count_ = 0;
  std::uint64_t at = 0;
  const auto add = [&](const md::StateRange& r, std::uint64_t base, std::int64_t row) -> Status {
    if (save_count_ == range_capacity_ || at + r.bytes > snapshot_.bytes) {
      return Error("a verify writes more than its snapshot holds");
    }
    saved_.push_back(
        {.address = base + r.offset, .saved = snapshot_.base + at, .bytes = r.bytes, .row = row});
    save_[save_count_++] = {.from = base + r.offset, .to = snapshot_.base + at, .bytes = r.bytes};
    at += Round(r.bytes, 256);
    return {};
  };
  for (std::uint32_t i = 0; i < in.rows; ++i) {
    for (const md::StateRange& r : writes.rows[i]) {
      if (auto added = add(r, state_.base, i); !added) {
        return added;
      }
    }
    for (const md::StateRange& r : ring[i]) {
      if (auto added = add(r, dstate_.base, i); !added) {
        return added;
      }
    }
  }
  for (const md::StateRange& r : writes.scratch) {
    if (auto added = add(r, state_.base, -1); !added) {
      return added;
    }
  }
  return {};
}

std::vector<Dsv4Runner::VerifyWrite> Dsv4Runner::last_verify_writes() const {
  std::vector<VerifyWrite> out;
  out.reserve(saved_.size());
  for (const Saved& s : saved_) {
    const bool ring = s.address >= dstate_.base && s.address < dstate_.base + dstate_.bytes;
    out.push_back({.ring = ring,
                   .offset = s.address - (ring ? dstate_.base : state_.base),
                   .bytes = s.bytes,
                   .row = s.row});
  }
  return out;
}

Status Dsv4Runner::Accept(std::uint32_t keep) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ == 0 || keep == 0 || keep > verify_rows_) {
    return Error(std::format("accepting {} rows of a verify of {}", keep, verify_rows_));
  }
  std::uint32_t count = 0;
  for (const Saved& s : saved_) {
    if (s.row < 0 || std::cmp_greater_equal(s.row, keep)) {
      restore_[count++] = {.from = s.saved, .to = s.address, .bytes = s.bytes};
    }
  }
  restore_count_ = count;
  verify_rows_ = 0;
  return {};
}

Status Dsv4Runner::Rollback() {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (restore_count_ == 0) {
    return {};
  }
  std::string failed;
  auto posted = node_.Job(
      everything_,
      [&](providers::NativeStream) {
        if (auto r = QueueRestore(); !r) {
          failed = r.error().detail;
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      "restoring a verify's rejected rows", stream_);
  if (!posted || !failed.empty()) {
    // A restore not queued stays owed; one queued has completed (the job
    // retired); a launch of unknown effect quarantines.
    Settle(false, false, !failed.empty());
    return Error(failed.empty() ? posted.error() : failed);
  }
  return {};
}

Status Dsv4Runner::Chunk(std::uint32_t n_past, std::span<const std::int32_t> tokens,
                         std::vector<float>& logits, const std::function<Status()>& meanwhile,
                         Dsv4ChunkKind kind) {
  const auto rows = static_cast<std::uint32_t>(tokens.size());
  if (kind != Dsv4ChunkKind::kPlain && !speculative()) {
    return Error("an injection or a verify needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, rows);
  if (!in) {
    return std::unexpected(in.error());
  }
  const bool verify = kind == Dsv4ChunkKind::kVerify;
  md::DsparkInjection inject;
  if (kind != Dsv4ChunkKind::kPlain) {
    inject = md::DsparkInject(dlayout_, n_past, rows);
  }
  if (verify) {
    if (rows > o_.max_verify || !md::Dsv4SameWidths(layout_, n_past, rows)) {
      return Error(std::format("a verify of {} rows at {}: at most {}, at its steps' mask widths",
                               rows, n_past, o_.max_verify));
    }
    if (auto r = PlanSnapshot(*in); !r) {
      return r;
    }
  }
  auto planned =
      Planned(kg::Dsv4ShapeOf(layout_, *in), kind, static_cast<std::int64_t>(inject.cells.size()));
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
  const std::uint32_t out_rows = verify ? rows : 1;
  // Decode graphs (D-090): replay a shape's graph; capture a one-row
  // shape (or a verify's) that has run once launch by launch; otherwise
  // launch by launch.
  const bool replay = graphs_ && entry.graph.has_value();
  const bool capture = graphs_ && !replay &&
                       ((rows == 1 && kind == Dsv4ChunkKind::kPlain) || verify) &&
                       !entry.uncapturable && entry.eager_runs > 0;
  if (capture && graphs() >= kMaxGraphs) {
    // Graph memory is the driver's, outside the catalog: at most
    // kMaxGraphs are kept. Decode moves on to later shapes, so the oldest
    // goes (no job is in flight between chunks: nothing replays it).
    const auto oldest =
        std::ranges::find_if(plans_, [](const ShapePlan& e) { return e.graph.has_value(); });
    oldest->graph.reset();
    oldest->copies.clear();
    ++graph_stats_.dropped;
  }
  Dsv4Path path = Dsv4Path::kEager;
  Status ran;
  Dsv4HostInputs host;
  // What the job queued, for a failure's settling (Settle).
  bool saved = false;    // this verify's snapshot
  bool wrote = false;    // anything that may write the state
  bool unknown = false;  // a launch of unknown effect
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    // A rejected draft's rows first, then this verify's snapshot.
    const bool restoring = restore_count_ != 0;
    if (auto r = QueueRestore(); !r) {
      ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    bool before = restoring;  // work queued before a failure
    if (verify) {
      if (auto r = kg::CopyRanges(*launch_, save_, save_count_); !r) {
        ran = Error(std::format("chunk at {}: {}", n_past, r.error().detail));
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      before = true;
      saved = true;
    }
    // The embedding rows and the chunk plan's inputs, staged in order.
    if (auto r = BuildDsv4Inputs(model_, g, *in, tokens, table, host, inject.cells); !r) {
      ran = std::unexpected(r.error());
      return before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    auto copies = Stage(host, verify ? verify_base_ : 0);
    if (!copies) {
      ran = std::unexpected(copies.error());
      return before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    // The last row's logits (the next token's), or a verify's every row's.
    const Queued queued = QueueRuns(entry, *copies, *p->bound, logits_,
                                    static_cast<const std::byte*>(g.logits->data) +
                                        (std::uint64_t{rows - out_rows} * row_bytes),
                                    out_rows * row_bytes, capture, graph_stats_, native);
    path = queued.path;
    wrote = queued.result.has_value() || queued.before;  // the chunk's own writes
    last_submit_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!queued.result) {
      ran = Error(std::format("chunk at {}: {}", n_past, queued.result.error().detail));
      if (queued.result.error().error == kg::KernelError::kUnknown) {
        unknown = true;
        return sc::JobResult::kUnknown;
      }
      return before || queued.before ? sc::JobResult::kFailed : sc::JobResult::kNotStarted;
    }
    return sc::JobResult::kQueued;
  };
  Status posted;
  Status alongside;
  if (meanwhile) {
    // Submitted without waiting; the frame (and `job`'s references) lives
    // until Await has seen the program gone.
    sc::ProgramDone done;
    const std::uint64_t request =
        node_.Submit(std::make_unique<sc::RunProgram>(done, everything_, std::move(job), stream_));
    alongside = meanwhile();
    posted = node_.Await(done, "a DeepSeek chunk", request);
  } else {
    posted = node_.Job(everything_, std::move(job), "a DeepSeek chunk", stream_);
  }
  if (!posted || !ran || !alongside || launch_->faulted()) {
    // Never left half-written: a verify is undone, anything else that may
    // have written the state quarantines it.
    Settle(saved, wrote, unknown);
    if (launch_->faulted()) {
      return Error(std::format("chunk at {}: the launch context faulted", n_past));
    }
    if (!ran) {
      return ran;
    }
    return Error(
        std::format("chunk at {}: {}", n_past, !posted ? posted.error() : alongside.error()));
  }
  last_path_ = path;
  last_planned_ = p;
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
  if (verify) {
    verify_rows_ = rows;
  }
  const auto* values = static_cast<const float*>(logits_);
  logits.assign(values, values + (std::size_t{out_rows} * profile_.vocab));
  return {};
}

Status Dsv4Runner::Draft(std::uint32_t pos0, std::int32_t anchor,
                         std::vector<std::int32_t>& drafts) {
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  auto in = md::DsparkBlock(dprofile_, dlayout_, pos0, anchor, o_.draft_rows);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto planned = PlannedDraft();
  if (!planned) {
    return std::unexpected(planned.error());
  }
  DraftPlan& entry = **planned;
  DsparkPlanned* p = entry.planned.get();
  const kg::DsparkGraph& g = p->graph;
  const std::span<const std::byte> table(
      reinterpret_cast<const std::byte*>(table_base_),  // NOLINT(performance-no-int-to-ptr)
      table_bytes_);
  const std::uint64_t draft_bytes = std::uint64_t{o_.draft_rows} * sizeof(std::int32_t);
  const bool capture =
      graphs_ && !entry.graph.has_value() && !entry.uncapturable && entry.eager_runs > 0;
  Dsv4Path path = Dsv4Path::kEager;
  Status ran;
  Dsv4HostInputs host;
  bool unknown = false;  // a launch of unknown effect
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    if (auto r = QueueRestore(); !r) {
      ran = Error(std::format("draft at {}: {}", pos0, r.error().detail));
      unknown = true;
      return sc::JobResult::kUnknown;
    }
    if (auto r = BuildDsparkInputs(model_, g, *in, table, host); !r) {
      ran = std::unexpected(r.error());
      return sc::JobResult::kFailed;
    }
    auto copies = Stage(host, 0);
    if (!copies) {
      ran = std::unexpected(copies.error());
      return sc::JobResult::kFailed;
    }
    const Queued queued = QueueRuns(entry, *copies, *p->bound, drafts_, g.drafts->data, draft_bytes,
                                    capture, draft_stats_, native);
    path = queued.path;
    last_submit_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!queued.result) {
      ran = Error(std::format("draft at {}: {}", pos0, queued.result.error().detail));
      // A restore may have been queued before a refusal.
      unknown = queued.result.error().error == kg::KernelError::kUnknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    }
    return sc::JobResult::kQueued;
  };
  auto posted = node_.Job(draft_closure_, std::move(job), "a DSpark draft", stream_);
  if (!posted || !ran || launch_->faulted()) {
    // A draft writes only its own block's ring cells, past the committed
    // positions; a launch of unknown effect quarantines the state.
    Settle(false, false, unknown);
    if (launch_->faulted()) {
      return Error(std::format("draft at {}: the launch context faulted", pos0));
    }
    return !ran ? ran : Error(std::format("draft at {}: {}", pos0, posted.error()));
  }
  switch (path) {
    case Dsv4Path::kEager:
      ++draft_stats_.eager;
      break;
    case Dsv4Path::kCaptured:
      ++draft_stats_.captured;
      break;
    case Dsv4Path::kReplayed:
      ++draft_stats_.replayed;
      break;
  }
  const auto* values = static_cast<const std::int32_t*>(drafts_);
  drafts.assign(values, values + o_.draft_rows);
  return {};
}

std::expected<Dsv4Runner::Copies, std::string> Dsv4Runner::Stage(const Dsv4HostInputs& host,
                                                                 std::uint64_t base) {
  Copies copies;
  copies.reserve(host.sources.size());
  std::uint64_t staged = base;
  for (const auto& [tensor, source] : host.sources) {
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

Dsv4Runner::Queued Dsv4Runner::QueueRuns(Runs& runs, const Copies& copies, kg::BoundGraph& bound,
                                         void* out, const void* from, std::uint64_t out_bytes,
                                         bool capture, Dsv4GraphStats& stats,
                                         providers::NativeStream native) {
  // The input copies, the plan and the output's copy, as one run queues
  // them and a capture records them.
  const auto queue = [&](kg::LaunchContext& launch) -> std::expected<void, kg::KernelFailure> {
    const auto unknown = [](std::string what) {
      return std::unexpected(
          kg::KernelFailure{.error = kg::KernelError::kUnknown, .detail = std::move(what)});
    };
    for (const auto& [to, bytes, at] : copies) {
      if (const providers::DeviceStatus copied =
              providers::CopyAsync(native, Pointer(to), static_cast<const std::byte*>(inputs_) + at,
                                   bytes, providers::CopyKind::kHostToDevice);
          !copied.ok()) {
        return unknown(std::format("an input copy: {}", copied.text()));
      }
    }
    if (auto r = bound.Run(launch); !r) {
      return r;
    }
    if (const providers::DeviceStatus copied =
            providers::CopyAsync(native, out, from, out_bytes, providers::CopyKind::kDeviceToHost);
        !copied.ok()) {
      return unknown(std::format("the output's copy: {}", copied.text()));
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
    const std::size_t free_before =
        providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
    auto captured = launch_->Capture(queue);
    const std::size_t free_after =
        providers::QueryDeviceMemory().value_or(providers::DeviceMemoryInfo{}).free;
    (void)providers::TakeLastError();
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

std::expected<ggml_tensor*, std::string> Dsv4Runner::DraftRowsNode(std::uint32_t n,
                                                                   const void* drafts) {
  if (n == 0 || n > o_.draft_rows) {
    return Error("no drafts' rows to look up");
  }
  if (!rows_arena_) {
    auto arena = kg::TensorArena::Create(64);
    if (!arena) {
      return Error(arena.error().detail);
    }
    rows_arena_.emplace(std::move(*arena));
  }
  if (draft_rows_.empty()) {
    ggml_context* c = rows_arena_->context();
    const md::Dsv4Tensor& embedding = binding_.token_embd;
    auto type = kg::GgmlTypeOf(embedding.type);
    if (!type) {
      return Error(type.error().detail);
    }
    ggml_tensor* table = ggml_new_tensor_2d(c, *type, profile_.width, profile_.vocab);
    kg::TensorArena::Bind(table,
                          table_base_ + artifact_->resources()[embedding.index].offset.value());
    for (std::uint32_t k = 1; k <= o_.draft_rows; ++k) {
      ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, k);
      kg::TensorArena::Bind(ids, Address(drafts));
      ggml_tensor* rows = ggml_get_rows(c, table, ids);
      // The verify's embedding input is staged first: row 0 the anchor's,
      // the drafts' after it.
      kg::TensorArena::Bind(
          rows, Address(inputs_) + verify_base_ + (std::uint64_t{profile_.width} * sizeof(float)));
      draft_rows_.push_back(rows);
    }
  }
  // A draft planned again (DropPlans) may place its drafts elsewhere.
  kg::TensorArena::Bind(draft_rows_[n - 1]->src[1], Address(drafts));
  return draft_rows_[n - 1];
}

Status Dsv4Runner::DraftVerify(std::uint32_t pos, std::int32_t anchor, std::uint32_t rows,
                               std::vector<std::int32_t>& drafts, std::vector<float>& logits) {
  if (!speculative()) {
    return Error("drafting needs the drafter");
  }
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (verify_rows_ != 0) {
    return Error("the last verify awaits its Accept");
  }
  if (rows == 0 || rows > o_.max_verify || rows > o_.draft_rows + 1 ||
      !md::Dsv4SameWidths(layout_, pos, rows)) {
    return Error(
        std::format("a verify of {} rows at {}: at most {} and the drafts, at its steps' "
                    "mask widths",
                    rows, pos, o_.max_verify));
  }
  // The draft.
  auto block = md::DsparkBlock(dprofile_, dlayout_, pos, anchor, o_.draft_rows);
  if (!block) {
    return std::unexpected(block.error());
  }
  auto dplanned = PlannedDraft();
  if (!dplanned) {
    return std::unexpected(dplanned.error());
  }
  DraftPlan& dentry = **dplanned;
  const kg::DsparkGraph& dg = dentry.planned->graph;
  const bool dcapture =
      graphs_ && !dentry.graph.has_value() && !dentry.uncapturable && dentry.eager_runs > 0;
  // The verify: its tokens the anchor and placeholders the drafts replace
  // on the device.
  auto in = md::Dsv4Chunk(profile_, layout_, pos, rows);
  if (!in) {
    return std::unexpected(in.error());
  }
  const md::DsparkInjection inject = md::DsparkInject(dlayout_, pos, rows);
  if (auto r = PlanSnapshot(*in); !r) {
    return r;
  }
  auto planned = Planned(kg::Dsv4ShapeOf(layout_, *in), Dsv4ChunkKind::kVerify,
                         static_cast<std::int64_t>(inject.cells.size()));
  if (!planned) {
    return std::unexpected(planned.error());
  }
  ShapePlan& ventry = **planned;
  const kg::Dsv4Graph& vg = ventry.planned->graph;
  const bool vcapture =
      graphs_ && !ventry.graph.has_value() && !ventry.uncapturable && ventry.eager_runs > 0;
  if (vcapture && graphs() >= kMaxGraphs) {
    const auto oldest =
        std::ranges::find_if(plans_, [](const ShapePlan& e) { return e.graph.has_value(); });
    oldest->graph.reset();
    oldest->copies.clear();
    ++graph_stats_.dropped;
  }
  ggml_tensor* lookup = nullptr;
  if (rows > 1) {
    auto node = DraftRowsNode(rows - 1, dg.drafts->data);
    if (!node) {
      return std::unexpected(node.error());
    }
    lookup = *node;
  }
  const std::vector<std::int32_t> placeholders(rows, anchor);
  const std::span<const std::byte> table(
      reinterpret_cast<const std::byte*>(table_base_),  // NOLINT(performance-no-int-to-ptr)
      table_bytes_);
  const std::uint64_t draft_bytes = std::uint64_t{o_.draft_rows} * sizeof(std::int32_t);
  const std::uint64_t row_bytes = std::uint64_t{profile_.vocab} * sizeof(float);
  Queued dq;
  Queued vq;
  Status ran;
  Dsv4HostInputs dhost;
  Dsv4HostInputs vhost;
  // What the job queued, for a failure's settling (Settle): the verify's
  // snapshot, and a launch of unknown effect. Before the snapshot only the
  // draft's own ring cells, past the committed positions, are written.
  bool saved = false;
  bool unknown_effect = false;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    const auto started = std::chrono::steady_clock::now();
    const auto failed = [&](std::string what, bool unknown) {
      ran = Error(std::format("draft and verify at {}: {}", pos, what));
      unknown_effect = unknown_effect || unknown;
      return unknown ? sc::JobResult::kUnknown : sc::JobResult::kFailed;
    };
    if (auto r = QueueRestore(); !r) {
      return failed(r.error().detail, true);
    }
    // Both stagings first: the host writes them before anything it queues
    // reads them.
    if (auto r = BuildDsparkInputs(model_, dg, *block, table, dhost); !r) {
      return failed(r.error(), false);
    }
    auto dcopies = Stage(dhost, 0);
    if (!dcopies) {
      return failed(dcopies.error(), false);
    }
    // The draft's staging must end before the verify's begins.
    for (const auto& copy : *dcopies) {
      if (copy[2] + copy[1] > verify_base_) {
        return failed("the draft's inputs reach the verify's staging", false);
      }
    }
    if (auto r = BuildDsv4Inputs(model_, vg, *in, placeholders, table, vhost, inject.cells); !r) {
      return failed(r.error(), false);
    }
    auto vcopies = Stage(vhost, verify_base_);
    // Staged first the embedding rows, then the tokens: the drafts' rows
    // and ids are written over them below.
    if (!vcopies || vcopies->size() < 2 || (*vcopies)[0][2] != verify_base_ ||
        vhost.sources[0].second != vhost.embd.data() ||
        vhost.sources[1].second != vhost.tokens.data()) {
      return failed(vcopies ? "the verify's staging" : vcopies.error(), false);
    }
    dq = QueueRuns(dentry, *dcopies, *dentry.planned->bound, drafts_, dg.drafts->data, draft_bytes,
                   dcapture, draft_stats_, native);
    if (!dq.result) {
      return failed(dq.result.error().detail, dq.result.error().error == kg::KernelError::kUnknown);
    }
    if (rows > 1) {
      // The drafts into the staged tokens after the anchor, and their
      // embedding rows into the staged rows after the anchor's.
      const std::uint64_t tokens_at = (*vcopies)[1][2];
      if (!providers::CopyAsync(native,
                                static_cast<std::byte*>(inputs_) + tokens_at + sizeof(std::int32_t),
                                dg.drafts->data, std::uint64_t{rows - 1} * sizeof(std::int32_t),
                                providers::CopyKind::kDeviceToHost)
               .ok()) {
        return failed("the drafts' copy into the verify's tokens", true);
      }
      if (auto r = kg::GetRowsExt(*launch_, lookup); !r) {
        return failed(r.error().detail, r.error().error == kg::KernelError::kUnknown);
      }
    }
    if (auto r = kg::CopyRanges(*launch_, save_, save_count_); !r) {
      return failed(r.error().detail, true);
    }
    saved = true;
    vq = QueueRuns(ventry, *vcopies, *ventry.planned->bound, logits_, vg.logits->data,
                   rows * row_bytes, vcapture, graph_stats_, native);
    last_submit_seconds_ =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (!vq.result) {
      return failed(vq.result.error().detail, vq.result.error().error == kg::KernelError::kUnknown);
    }
    return sc::JobResult::kQueued;
  };
  auto posted = node_.Job(everything_, std::move(job), "a DSpark draft and its verify", stream_);
  if (!posted || !ran || launch_->faulted()) {
    // Never left half-written: the verify is undone before the next job's
    // work, or the state quarantined.
    Settle(saved, false, unknown_effect);
    if (launch_->faulted()) {
      return Error(std::format("draft and verify at {}: the launch context faulted", pos));
    }
    return !ran ? ran : Error(std::format("draft and verify at {}: {}", pos, posted.error()));
  }
  const auto count = [](Dsv4GraphStats& stats, Dsv4Path path) {
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
  };
  count(draft_stats_, dq.path);
  count(graph_stats_, vq.path);
  last_path_ = vq.path;
  verify_rows_ = rows;
  const auto* values = static_cast<const std::int32_t*>(drafts_);
  drafts.assign(values, values + o_.draft_rows);
  const auto* rows_out = static_cast<const float*>(logits_);
  logits.assign(rows_out, rows_out + (std::size_t{rows} * profile_.vocab));
  return {};
}

std::expected<std::uint64_t, std::string> Dsv4Runner::CheckDeviceEmbedding() {
  constexpr std::uint32_t kBatch = 2048;
  const md::Dsv4Tensor& embedding = binding_.token_embd;
  auto type = kg::GgmlTypeOf(embedding.type);
  if (!type) {
    return Error(type.error().detail);
  }
  const std::uint64_t width = profile_.width;
  const std::uint64_t out_bytes = kBatch * width * sizeof(float);
  const std::uint64_t ids_at = Round(out_bytes, 256);
  if (ids_at + (kBatch * sizeof(std::int32_t)) > node_.activations().bytes ||
      kBatch * sizeof(std::int32_t) > input_bytes_) {
    return Error("the lookups' batch does not fit the activations or the staging");
  }
  const std::uint64_t ring = speculative() ? dlayout_.bytes : 0;
  if (!HavePinned(state_host_, std::max(layout_.bytes + ring, out_bytes))) {
    return Error("pinned host memory for the lookups' rows");
  }
  if (layout_.bytes + ring < out_bytes) {
    return Error("the state's host copy is shorter than a batch of rows");
  }
  auto arena = kg::TensorArena::Create(8);
  if (!arena) {
    return Error(arena.error().detail);
  }
  ggml_context* c = arena->context();
  ggml_tensor* table = ggml_new_tensor_2d(c, *type, profile_.width, profile_.vocab);
  kg::TensorArena::Bind(table,
                        table_base_ + artifact_->resources()[embedding.index].offset.value());
  ggml_tensor* ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, kBatch);
  kg::TensorArena::Bind(ids, node_.activations().base + ids_at);
  ggml_tensor* rows = ggml_get_rows(c, table, ids);
  kg::TensorArena::Bind(rows, node_.activations().base);
  const std::span<const std::byte> host_table(
      reinterpret_cast<const std::byte*>(table_base_),  // NOLINT(performance-no-int-to-ptr)
      table_bytes_);
  std::vector<std::int32_t> batch(kBatch);
  std::vector<float> want;
  std::uint64_t checked = 0;
  for (std::uint32_t first = 0; first < profile_.vocab; first += kBatch) {
    for (std::uint32_t i = 0; i < kBatch; ++i) {
      batch[i] = static_cast<std::int32_t>(std::min(first + i, profile_.vocab - 1));
    }
    std::memcpy(inputs_, batch.data(), kBatch * sizeof(std::int32_t));
    std::string failed;
    void* host = state_host_;
    auto posted = node_.Job(
        everything_,
        [&](providers::NativeStream native) {
          if (!providers::CopyAsync(native, ids->data, inputs_, kBatch * sizeof(std::int32_t),
                                    providers::CopyKind::kHostToDevice)
                   .ok()) {
            return sc::JobResult::kUnknown;
          }
          if (auto r = kg::GetRowsExt(*launch_, rows); !r) {
            failed = r.error().detail;
            return sc::JobResult::kFailed;
          }
          return providers::CopyAsync(native, host, rows->data, out_bytes,
                                      providers::CopyKind::kDeviceToHost)
                         .ok()
                     ? sc::JobResult::kQueued
                     : sc::JobResult::kUnknown;
        },
        "looking up embedding rows", stream_);
    if (!posted || !failed.empty()) {
      return Error(failed.empty() ? posted.error() : failed);
    }
    // The host's lookup, as every chunk makes it.
    if (auto r = Dsv4EmbeddingRows(model_, batch, host_table, want); !r) {
      return std::unexpected(r.error());
    }
    if (std::memcmp(state_host_, want.data(), out_bytes) != 0) {
      return Error(
          std::format("an embedding row of tokens {} to {} differs between the device's "
                      "lookup and the host's",
                      first, first + kBatch - 1));
    }
    checked += std::min<std::uint64_t>(kBatch, profile_.vocab - first);
  }
  return checked;
}

Status Dsv4Runner::ReadState(std::vector<std::byte>& target, std::vector<std::byte>& drafter) {
  if (auto usable = Usable(); !usable) {
    return usable;
  }
  if (restore_count_ != 0 || verify_rows_ != 0) {
    return Error("reading the state with a verify's rollback pending");
  }
  const std::uint64_t ring = speculative() ? dlayout_.bytes : 0;
  if (!HavePinned(state_host_, layout_.bytes + ring)) {
    return Error("pinned host memory for the state's copy");
  }
  const std::uint64_t target_base = state_.base;
  const std::uint64_t target_bytes = layout_.bytes;
  const std::uint64_t ring_base = dstate_.base;
  void* host = state_host_;
  auto posted = node_.Job(
      fence_,
      [=](providers::NativeStream stream) {
        if (!providers::CopyAsync(stream, host, Pointer(target_base), target_bytes,
                                  providers::CopyKind::kDeviceToHost)
                 .ok()) {
          return sc::JobResult::kUnknown;
        }
        if (ring != 0 &&
            !providers::CopyAsync(stream, static_cast<std::byte*>(host) + target_bytes,
                                  Pointer(ring_base), ring, providers::CopyKind::kDeviceToHost)
                 .ok()) {
          return sc::JobResult::kUnknown;
        }
        return sc::JobResult::kQueued;
      },
      "reading the DeepSeek state", stream_);
  if (!posted) {
    return posted;
  }
  const auto* bytes = static_cast<const std::byte*>(state_host_);
  target.assign(bytes, bytes + target_bytes);
  drafter.assign(bytes + target_bytes, bytes + target_bytes + ring);
  return {};
}

Status Dsv4Runner::DumpLast(std::vector<Dumped>& out) {
  out.clear();
  if (last_planned_ == nullptr || dump_.empty()) {
    return Error("no chunk has run with a dump since the plans were dropped");
  }
  struct Read {
    const void* from = nullptr;
    std::uint64_t at = 0;
    std::uint64_t bytes = 0;
  };
  std::vector<Read> reads;
  std::uint64_t total = 0;
  for (const auto& [name, t] : last_planned_->graph.named) {
    if (t->data == nullptr || !ggml_is_contiguous(t)) {
      continue;
    }
    Dumped& d = out.emplace_back();
    d.name = name;
    d.type = t->type;
    d.ne = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    reads.push_back({.from = t->data, .at = total, .bytes = ggml_nbytes(t)});
    total += Round(ggml_nbytes(t), 256);
  }
  auto pinned = providers::AllocatePinned(std::max<std::uint64_t>(total, 256));
  if (!pinned) {
    return Error("pinned host memory for a dump");
  }
  void* const host = *pinned;
  auto posted = node_.Job(
      everything_,
      [&](providers::NativeStream stream) {
        for (const Read& r : reads) {
          if (!providers::CopyAsync(stream, static_cast<std::byte*>(host) + r.at, r.from, r.bytes,
                                    providers::CopyKind::kDeviceToHost)
                   .ok()) {
            return sc::JobResult::kUnknown;
          }
        }
        return sc::JobResult::kQueued;
      },
      "reading a DeepSeek chunk's dump", stream_);
  if (posted) {
    for (std::size_t i = 0; i < reads.size(); ++i) {
      const auto* bytes = static_cast<const std::byte*>(host) + reads[i].at;
      out[i].bytes.assign(bytes, bytes + reads[i].bytes);
    }
  }
  providers::FreePinned(host);
  return posted;
}

std::expected<double, std::string> Dsv4Runner::TimeReplays(std::uint32_t n_past, std::int32_t token,
                                                           std::uint32_t count) {
  auto in = md::Dsv4Chunk(profile_, layout_, n_past, 1);
  if (!in) {
    return std::unexpected(in.error());
  }
  auto planned = Planned(kg::Dsv4ShapeOf(layout_, *in), Dsv4ChunkKind::kPlain, 0);
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
  dplans_.clear();
  launch_.reset();
  cublas_.reset();
  auto& memory = node_.memory();
  for (Mapped* mapped : {&state_, &dstate_, &cublas_workspace_, &snapshot_}) {
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
  for (const providers::ReservationId reservation : {target_.weights, table_, drafter_.weights}) {
    if (reservation.valid() && !memory.Free(reservation)) {
      problems.emplace_back("a DeepSeek weights reservation still has mappings");
    }
  }
  if (state_host_ != nullptr) {
    providers::FreePinned(state_host_);
    state_host_ = nullptr;
  }
  if (spill_fd_ >= 0) {
    (void)::close(spill_fd_);  // unnamed: nothing outlives the process
    spill_fd_ = -1;
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

}  // namespace jitllm::engine
