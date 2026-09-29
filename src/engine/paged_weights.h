// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A v0 artifact's weights as extents on a paged node (paged_node.h), for
// the M3 runners that page Qwen3.8 and Qwen-Image
// (qwen38_runner.h, qwen_image_runner.h; docs/experiments/fast-swap/):
// the layout DeepSeek's runner (dsv4_runner.h) introduced, as one helper.
//
// - Each placed dense group gets a 2 MiB-aligned region of one device
//   reservation; its chunk k is an extent at the region's base + k x 2 MiB,
//   landed from its shard through the zone (D-081).
// - Each slab (a layer's routed experts at a uniform stride S, the resident
//   expert layout) is laid out by LayOutSlab (dsv4_runner.h): an extent is
//   a 2 MiB page of the slab's address range whose contents are the stored
//   bytes of the groups it overlaps, landed in pieces.
// - Extents are cataloged in file order (RE-026), all of one memory class,
//   restorable from the artifact, with managed backing (D-033).
// - Groups neither placed nor in a slab are not paged at all (the n-gram
//   table's rows are read on demand, qwen38_runner.h; a component's parts
//   its phases never read).
//
// CUDA builds only.

#ifndef JITLLM_ENGINE_PAGED_WEIGHTS_H_
#define JITLLM_ENGINE_PAGED_WEIGHTS_H_

#include <array>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include "artifact/artifact.h"
#include "catalog/catalog.h"
#include "engine/paged_node.h"
#include "providers/device_memory.h"
#include "scheduler/scheduler.h"

namespace jitllm::engine {

// One layer's routed experts: `count` groups from `first_group`, each at
// `stride` in the slab, the slab's offset in its first page a multiple of
// `alignment` (LayOutSlab).
struct SlabSpec {
  std::uint32_t first_group = 0;
  std::uint32_t count = 0;
  std::uint64_t stride = 0;
  std::uint64_t alignment = 256;
};

class PagedWeights {
 public:
  using Status = engine::Status;

  // Reserves and catalogs the places of the groups `place` marks (one flag
  // per group) and of the slabs, in file order. `shards` are the artifact's
  // direct-read descriptors, which must outlive this. Refused if a placed
  // group is an expert group, a slab's groups are not uniform expert
  // groups, or LayOutSlab refuses one.
  Status Reserve(PagedNode& node, const artifact::Artifact& artifact,
                 std::span<const artifact::FileDescriptor> shards,
                 const std::array<std::uint8_t, 32>& id, const std::vector<bool>& place,
                 std::span<const SlabSpec> slabs);
  // After the node's Start: every extent's source, and its span for the
  // coverage check.
  Status Register(PagedNode& node, int owner);
  // Frees the reservation (its extents must have been evicted).
  Status Release(providers::VmmProvider& memory);

  // The bytes of the extents a load does not write: each dense chunk's
  // tail past its stored length, and a slab page's bytes outside its
  // pieces (the gaps between groups, before the first and after the last).
  // They hold whatever the backing held (a handoff's is the outgoing
  // model's), so a kernel that reads them reads stale bytes.
  struct Range {
    std::uint64_t address = 0;
    std::uint64_t bytes = 0;
    bool slab = false;
  };
  std::vector<Range> Unwritten() const;

  // A placed group's (or a slab expert's) device address; 0 if not paged.
  std::uint64_t group_address(std::uint32_t group) const { return group_address_.at(group); }
  const std::vector<catalog::ExtentId>& extents() const { return extents_; }
  std::uint64_t read_bytes() const { return read_bytes_; }  // one full load reads
  std::uint64_t slab_padding() const { return slab_padding_; }
  std::uint64_t bytes() const { return bytes_; }  // the reservation

 private:
  struct Source {
    std::uint64_t offset = 0;  // in the reservation
    scheduler::PageSource source;
  };
  providers::ReservationId reservation_;
  std::uint64_t base_ = 0;
  std::uint64_t bytes_ = 0;
  std::vector<std::uint64_t> group_address_;
  std::vector<catalog::ExtentId> extents_;
  std::vector<Source> sources_;  // by extent, in extents_' order
  std::uint64_t read_bytes_ = 0;
  std::uint64_t slab_padding_ = 0;
};

// The artifact's identity as the catalog's content key takes it.
std::array<std::uint8_t, 32> ArtifactKey(const artifact::Artifact& artifact);

}  // namespace jitllm::engine

#endif  // JITLLM_ENGINE_PAGED_WEIGHTS_H_
