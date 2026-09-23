// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0
//
// Probe how much GPU virtual address space one process can reserve with the
// CUDA VMM driver API, and whether a 2 MiB physical extent mapped at the far
// end of a large reservation is usable. Answers whether uniform-stride expert
// views (stride = lcm(extent, quant block sizes)) are affordable in VA.
// Reservations are released before exit; no physical memory beyond one
// granule is ever created.
#include <cuda.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    std::fprintf(stderr, "%s failed: %d %s\n", what, static_cast<int>(r), s ? s : "?");
    std::exit(1);
  }
}

constexpr std::uint64_t kGiB = 1ull << 30;

}  // namespace

int main() {
  check(cuInit(0), "cuInit");
  CUdevice dev;
  check(cuDeviceGet(&dev, 0), "cuDeviceGet");
  CUcontext ctx;
  check(cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
  check(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");

  // Same allocation properties as the D-033/D-034 host-VMM path.
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
  prop.location.id = 0;
  std::size_t gran = 0;
  check(cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "granularity");
  std::printf("{\"granularity\": %zu,\n", gran);

  // 1. Largest single reservation, doubling from 1 GiB.
  std::uint64_t largest = 0;
  for (std::uint64_t size = kGiB; size <= (1ull << 50); size <<= 1) {
    CUdeviceptr p = 0;
    if (cuMemAddressReserve(&p, size, gran, 0, 0) != CUDA_SUCCESS) break;
    largest = size;
    check(cuMemAddressFree(p, size), "cuMemAddressFree");
  }
  std::printf(" \"largest_single_reservation_bytes\": %" PRIu64 ",\n", largest);

  // 2. Aggregate: keep reserving 1 TiB ranges until failure (bounded).
  std::vector<CUdeviceptr> held;
  constexpr std::uint64_t kChunk = 1024 * kGiB;
  for (int i = 0; i < 1024; ++i) {
    CUdeviceptr p = 0;
    if (cuMemAddressReserve(&p, kChunk, gran, 0, 0) != CUDA_SUCCESS) break;
    held.push_back(p);
  }
  std::printf(" \"aggregate_1tib_reservations\": %zu,\n", held.size());
  for (CUdeviceptr p : held) check(cuMemAddressFree(p, kChunk), "cuMemAddressFree");

  // 3. Map one granule at the last granule of a 4 TiB reservation and use it.
  const std::uint64_t big = 4096 * kGiB;
  CUdeviceptr base = 0;
  const bool reserved = cuMemAddressReserve(&base, big, gran, 0, 0) == CUDA_SUCCESS;
  bool mapped_ok = false;
  if (reserved) {
    CUmemGenericAllocationHandle h;
    check(cuMemCreate(&h, gran, &prop, 0), "cuMemCreate");
    const CUdeviceptr at = base + big - gran;
    check(cuMemMap(at, gran, 0, h, 0), "cuMemMap");
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check(cuMemSetAccess(at, gran, &access, 1), "cuMemSetAccess");
    check(cuMemsetD8(at, 0x5a, gran), "cuMemsetD8");
    unsigned char probe[16] = {};
    check(cuMemcpyDtoH(probe, at + gran - sizeof probe, sizeof probe), "cuMemcpyDtoH");
    mapped_ok = true;
    for (unsigned char b : probe) mapped_ok = mapped_ok && b == 0x5a;
    check(cuMemUnmap(at, gran), "cuMemUnmap");
    check(cuMemRelease(h), "cuMemRelease");
    check(cuMemAddressFree(base, big), "cuMemAddressFree");
  }
  std::printf(" \"far_end_map_4tib_reserved\": %s,\n \"far_end_map_4tib_ok\": %s}\n",
              reserved ? "true" : "false", mapped_ok ? "true" : "false");
  check(cuDevicePrimaryCtxRelease(dev), "cuDevicePrimaryCtxRelease");
  return 0;
}
