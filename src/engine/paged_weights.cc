// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "engine/paged_weights.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <format>
#include <numeric>
#include <optional>
#include <utility>

#include "artifact/layout.h"
#include "engine/dsv4_runner.h"

namespace jitllm::engine {

namespace {

namespace sc = jitllm::scheduler;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using catalog::Recovery;

constexpr std::uint64_t kExtent = kPagedExtent;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

}  // namespace

std::array<std::uint8_t, 32> ArtifactKey(const artifact::Artifact& artifact) {
  std::array<std::uint8_t, 32> id{};
  const std::string& hex = artifact.id();
  for (std::size_t i = 0; i < id.size() && (2 * i) + 1 < hex.size(); ++i) {
    (void)std::from_chars(hex.data() + (2 * i), hex.data() + (2 * i) + 2, id.at(i), 16);
  }
  return id;
}

PagedWeights::Status PagedWeights::Reserve(PagedNode& node, const artifact::Artifact& artifact,
                                           std::span<const artifact::FileDescriptor> shards,
                                           const std::array<std::uint8_t, 32>& id,
                                           const std::vector<bool>& place,
                                           std::span<const SlabSpec> slabs) {
  const auto groups = artifact.groups();
  if (place.size() != groups.size()) {
    return Error("a placement flag per group");
  }
  // Each slab's layout, and which slab each of its groups is in.
  std::vector<SlabLayout> layouts(slabs.size());
  std::vector<std::int64_t> slab_of(groups.size(), -1);
  for (std::size_t s = 0; s < slabs.size(); ++s) {
    const SlabSpec& spec = slabs[s];
    if (spec.count == 0 || std::uint64_t{spec.first_group} + spec.count > groups.size()) {
      return Error(std::format("slab {} names groups the artifact does not have", s));
    }
    const std::uint64_t stored = groups[spec.first_group].stored.value();
    std::vector<std::uint32_t> shard(spec.count);
    std::vector<std::uint64_t> file(spec.count);
    for (std::uint32_t e = 0; e < spec.count; ++e) {
      const std::uint32_t g = spec.first_group + e;
      if (groups[g].kind != artifact::GroupKind::kExpert || groups[g].stored.value() != stored ||
          place[g] || slab_of[g] >= 0) {
        return Error(std::format("slab {}'s expert groups are not uniform", s));
      }
      const auto range = artifact::ChunkRangeOf(artifact.layout(), {.group = g, .chunk = 0});
      if (!range) {
        return Error("an expert group's file range");
      }
      shard[e] = range->shard;
      file[e] = range->file_offset.value();
      slab_of[g] = static_cast<std::int64_t>(s);
    }
    auto layout = LayOutSlab(shard, file, stored, spec.stride, spec.alignment);
    if (!layout) {
      return Error(std::format("slab {}: {}", s, layout.error()));
    }
    layouts[s] = std::move(*layout);
    slab_padding_ += (spec.stride - stored) * spec.count;
  }
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (place[g] && groups[g].kind == artifact::GroupKind::kExpert) {
      return Error(std::format("expert group {} is placed as a dense group", g));
    }
  }
  // Places, in file order: dense regions and slabs in one reservation.
  std::vector<std::uint32_t> order(groups.size());
  std::ranges::iota(order, 0U);
  std::ranges::sort(order, [&](std::uint32_t a, std::uint32_t b) {
    return std::pair(groups[a].shard, groups[a].offset.value()) <
           std::pair(groups[b].shard, groups[b].offset.value());
  });
  std::vector<std::uint64_t> region(groups.size(), 0);
  std::vector<std::uint64_t> slab_region(slabs.size(), 0);
  std::vector<bool> slab_placed(slabs.size(), false);
  for (const std::uint32_t g : order) {
    if (slab_of[g] >= 0) {
      const auto s = static_cast<std::size_t>(slab_of[g]);
      if (!slab_placed[s]) {
        slab_placed[s] = true;
        slab_region[s] = bytes_;
        bytes_ += layouts[s].bytes();
      }
    } else if (place[g]) {
      region[g] = bytes_;
      bytes_ += std::uint64_t{groups[g].chunks} * kExtent;
    }
  }
  auto& memory = node.memory();
  auto reservation = memory.Reserve(Bytes(std::max<std::uint64_t>(bytes_, kExtent)));
  if (!reservation) {
    return Error(std::format("reserving the weights: {}", reservation.error().detail));
  }
  reservation_ = *reservation;
  base_ = memory.RangeOf(reservation_).value().base;
  group_address_.assign(groups.size(), 0);
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    if (slab_of[g] >= 0) {
      const auto s = static_cast<std::size_t>(slab_of[g]);
      group_address_[g] = base_ + slab_region[s] + layouts[s].delta +
                          (std::uint64_t{g - slabs[s].first_group} * slabs[s].stride);
    } else if (place[g]) {
      group_address_[g] = base_ + region[g];
    }
  }

  const auto add = [&](catalog::ContentKey content) -> std::expected<ExtentId, std::string> {
    auto extent = node.catalog().AddExtent({.domain = node.domain(),
                                            .memory_class = MemoryClass::kWeights,
                                            .recovery = Recovery::kFromArtifact,
                                            .size = Bytes(kExtent),
                                            .content = content});
    if (!extent) {
      return Error("cataloging a weight extent");
    }
    return *extent;
  };
  const auto place_of = [&](std::uint64_t offset) {
    return sc::BackingPlace{.reservation = reservation_,
                            .offset = Bytes(offset),
                            .size = Bytes(kExtent),
                            .allocation_class = node.device_class()};
  };
  std::ranges::fill(slab_placed, false);
  for (const std::uint32_t g : order) {
    if (slab_of[g] >= 0) {
      const auto s = static_cast<std::size_t>(slab_of[g]);
      if (slab_placed[s]) {
        continue;
      }
      slab_placed[s] = true;
      const SlabLayout& slab = layouts[s];
      for (std::uint32_t p = 0; p < slab.pages.size(); ++p) {
        const SlabPage& page = slab.pages[p];
        auto extent = add({.artifact = id, .group = slabs[s].first_group, .chunk = p});
        if (!extent) {
          return std::unexpected(extent.error());
        }
        const std::uint64_t offset = slab_region[s] + (std::uint64_t{p} * kExtent);
        sc::PageSource source{.read = {.fd = shards[page.shard].get(),
                                       .offset = page.file_offset,
                                       .memory = nullptr,
                                       .length = page.length},
                              .landed = true,
                              .destination = 0,
                              .pieces = {},
                              .piece_count = page.pieces,
                              .backing = place_of(offset)};
        for (std::size_t i = 0; i < page.pieces; ++i) {
          source.pieces.at(i) =
              sc::LandedPiece{.slot_offset = page.slot_offset.at(i),
                              .destination = base_ + offset + page.page_offset.at(i),
                              .length = Bytes(page.bytes.at(i))};
        }
        read_bytes_ += page.length;
        extents_.push_back(*extent);
        sources_.push_back({.offset = offset, .source = source});
      }
      continue;
    }
    if (!place[g]) {
      continue;
    }
    for (std::uint32_t c = 0; c < groups[g].chunks; ++c) {
      const auto range = artifact::ChunkRangeOf(artifact.layout(), {.group = g, .chunk = c});
      if (!range) {
        return Error("a chunk's range");
      }
      auto extent = add({.artifact = id, .group = g, .chunk = c});
      if (!extent) {
        return std::unexpected(extent.error());
      }
      const std::uint64_t offset = region[g] + (std::uint64_t{c} * kExtent);
      sc::PageSource source{.read = {.fd = shards[range->shard].get(),
                                     .offset = range->file_offset.value(),
                                     .memory = nullptr,
                                     .length = range->length.value()},
                            .landed = true,
                            .destination = base_ + offset,
                            .backing = place_of(offset)};
      read_bytes_ += range->length.value();
      extents_.push_back(*extent);
      sources_.push_back({.offset = offset, .source = source});
    }
  }
  return {};
}

PagedWeights::Status PagedWeights::Register(PagedNode& node, int owner) {
  for (std::size_t i = 0; i < extents_.size(); ++i) {
    auto set = node.scheduler().SetSource(extents_[i], sources_[i].source);
    if (!set) {
      return Error(std::format("a weight's source: {}", sc::ToString(set.error())));
    }
    node.AddSpan({.base = base_ + sources_[i].offset,
                  .size = kExtent,
                  .extent = extents_[i],
                  .memory_class = MemoryClass::kWeights,
                  .device = true,
                  .owner = owner});
  }
  return {};
}

std::vector<PagedWeights::Range> PagedWeights::Unwritten() const {
  std::vector<Range> out;
  for (const Source& s : sources_) {
    const std::uint64_t start = base_ + s.offset;
    const std::uint64_t end = start + kExtent;
    if (s.source.piece_count == 0) {
      const std::uint64_t written = s.source.destination + s.source.read.length;
      if (written < end) {
        out.push_back({.address = written, .bytes = end - written, .slab = false});
      }
      continue;
    }
    // Pieces ascend within the page (LayOutSlab).
    std::uint64_t at = start;
    for (std::size_t i = 0; i < s.source.piece_count; ++i) {
      const sc::LandedPiece& piece = s.source.pieces.at(i);
      if (piece.destination > at) {
        out.push_back({.address = at, .bytes = piece.destination - at, .slab = true});
      }
      at = std::max(at, piece.destination + piece.length.value());
    }
    if (at < end) {
      out.push_back({.address = at, .bytes = end - at, .slab = true});
    }
  }
  return out;
}

PagedWeights::Status PagedWeights::Release(providers::VmmProvider& memory) {
  if (reservation_.valid() && !memory.Free(reservation_)) {
    return Error("a weights reservation still has mappings");
  }
  reservation_ = {};
  return {};
}

}  // namespace jitllm::engine
