<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Rough edges — findings log

CUDA driver, DGX Spark platform, toolchain, and library bugs, quirks,
surprising limits, performance cliffs, and missing capabilities encountered
while building jitLLM. Log the ones that burned real debugging time and will
bite again — this is a save-future-you log, not a compliance artifact.

**Before adding:** grep for the API/library involved to avoid duplicates.
**Before debugging weirdness:** check here first — it may be known.

A good entry says what environment it happened in (workstation or Spark,
driver, toolkit, compiler versions) and what was observed vs. expected;
include a reproduction when it's cheap to capture. Platform constraints that
were known from documentation before any code existed (Spark's
compatibility-mode-only GDS, no GPUDirect RDMA) are recorded in
[environment.md](environment.md) and D-004/D-034, not here; this log is for
what the documentation did not tell us.

Format:

```
## RE-NNN: Title  (YYYY-MM-DD, status: open | fixed-upstream | worked-around | wontfix)
Environment / Repro or measurement / Observed / Expected / Impact / Links
```

Newest first. RE-numbers are never reused.

---

## RE-027: The Spark's SSD reads recently written data ~11% faster than data at rest  (2026-09-27, status: open)

On `spark-b` (Samsung `MZALC4T0HBL1-00B07`, ext4 root, kernel
7.0.0-1019-nvidia), 2 MiB `O_DIRECT` reads at four in flight ran at
14.7–14.9 GB/s from files written minutes to an hour before, and at
13.2–13.4 GB/s from files at rest, whatever read them (in place into host
VMM or through the landing zone, the runtime or the standalone probe).
In one run, a 9 GiB pattern file written seconds before read at
14.65–14.80, and a 16 GiB one that had read at 14.7–14.9 for the hour
after it was written read at 13.20–13.41, 90 minutes after; a GGUF shard
written six days before read at 13.17–13.41, and the FP16 artifact's shard (installed four
days before) at 12.2–13.3 against 14.3–14.9 for a `dd` copy of it.
Not fragmentation: `filefrag` on `spark-b` found the six-day-old GGUF
shards (32–49 GB) in 18–33 physical runs averaging 1.5–1.9 GB, the 16 GiB
pattern file after it slowed in 22 runs (~745 MiB on average), and the
FP16 artifact's shard in 15 runs for its 988 MB against 5 for the fresh
`dd` copy: all effectively contiguous for 2 MiB reads. The drive's write
cache (SLC) serving recent writes, displaced by later writes or by time,
fits the timing but is inferred, not proven. So the device's bandwidth for artifacts
at rest is ~13.3 GB/s, not the ~14.9 of earlier measurements, which all
read files just written (io-path, storage-queue, dmabuf-direct). Compare
paths on the same file in the same session, and state the file's age.
Evidence: [pagein-perf](experiments/pagein-perf/README.md).

## RE-026: The Spark's SSD reads 4 KiB-offset 2 MiB requests ~18% slower out of order  (2026-09-27, status: worked-around)

On `spark-b` (Samsung `MZALC4T0HBL1-00B07`, ext4 root, kernel
7.0.0-1019-nvidia, io_uring `O_DIRECT` reads of 2 MiB, four in flight),
reads of a pattern file whose offsets were 2 MiB multiples ran at
14.5–14.9 GB/s whether or not consecutive requests were in file order. With
every offset 4 KiB past a 2 MiB multiple, as an artifact's chunks are
(D-056 aligns groups to 4 KiB), the same reads ran at 14.4–14.8 GB/s in
order but 11.9–12.4 GB/s when about 60% of consecutive submissions were
locally swapped (a window of about eight). The runtime's direct reader
started queued reads in the order of their keys, which are reused mailbox
indices, so page-in through the zone hit this and in-place reads (fresh
keys, in order) did not. Worked around: the reader starts reads in arrival
order (`providers/direct_reader.h`). Keep reads of one load in file order.
Evidence: [pagein-perf](experiments/pagein-perf/README.md).

## RE-025: GB10 device memory cannot be exported as a dma-buf, and NVIDIA dma-buf mappings refuse direct I/O  (2026-09-27, status: open)

On `spark` (GB10, driver 580.178.04, CUDA 13.0 toolkit, kernel
7.0.0-1019-nvidia), `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` is 0.
`cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...)`
returns `CUDA_ERROR_INVALID_VALUE` (not `NOT_SUPPORTED`) for device VMM and
host VMM, with or without the PCIe flag, and for `cuMemAlloc`. The CUDA 13.4
header's `CU_DEVICE_ATTRIBUTE_DMA_BUF_MMAP_SUPPORTED` (152) is unknown to
this driver (`INVALID_VALUE`), and its "cached mapping on coherent ARM" note
does not apply. Only `cuMemAllocHost` exports. Its `mmap` is CPU-cached but
is a PFN map (`VM_PFNMAP | VM_IO`, smaps `pf io`), so `O_DIRECT`, io_uring
reads and `IORING_REGISTER_BUFFERS` into it fail with `EFAULT`.
`pin_user_pages` refuses such VMAs, and the open module's `nv_dma_buf_mmap`
says so; 610.57.04's source still maps by PFN (read, not tested). The
reverse direction works outside the documentation: a `udmabuf` of a shmem
`memfd` imports through `cuImportExternalMemory(..._DMABUF_FD)`
(documented for Jetson Thor only), and a hugetlb one is refused. The GPU treats it as host-located memory: no
L2 reuse, and 2.6 GB/s random reads with 4 KiB pages. Net: no route lands a
direct read in L2-cacheable memory on the Spark. Evidence:
[dmabuf-direct](experiments/dmabuf-direct/README.md).

## RE-024: MemAvailable misses host memory held on the per-CPU page lists  (2026-09-27, status: worked-around)

Environment: `spark-b` (kernel 7.0.0-1019-nvidia, 4 KiB pages, 20 CPUs,
driver 580.178.04), an otherwise idle host.

Observed: 64 MiB from `malloc`, written, then freed, moved `RssAnon` by
exactly 64 MiB every time, but `MemAvailable` (and `MemFree`) by anything
from 0 to 62 MiB, and by −15.8 to +70.6 MiB across the census runs' 144
control moves. Waiting 1 to 2.5 s did not help. Freed order-0 pages go to
the freeing CPU's page list and allocations are served from it, without
moving `NR_FREE_PAGES`, which `MemAvailable` reads. This kernel's adaptive
list sizes allow up to 198,125 pages (774 MiB) per CPU per zone
(`/proc/zoneinfo`, `high_max`). `cudaMalloc` and VMM backing, taken in
large blocks, moved `MemAvailable` within about 1 MiB of their size.

Expected: `MemAvailable` to follow an allocation of tens of MiB.

Impact: the M2 census rule (backend-proof.md) voided every run through its
host control. Adding the lists' pages (each zone's pagesets `count:` in
`/proc/zoneinfo`, readable by any user) to `MemAvailable` brings the host
control within 0.4 MiB of 64 MiB; the census reads that sum. Root can
drain the lists instead (`vm.percpu_pagelist_high_fraction`, compaction),
which an unprivileged harness cannot.

## RE-023: ext4 reuses inode numbers at once, and coarse timestamps hide the swap  (2026-09-27, status: worked-around)

Environment: the workstation (kernel 7.0.0-31-generic) with a loop-mounted
ext4 made with `mkfs.ext4 -I 128`, and `spark-b` (kernel 7.0.0-1019-nvidia,
ext4 with 256-byte inodes).

Observed: a file deleted and created again gets the same inode number
straight away on both. With 128-byte inodes ext4 has no room for
sub-second timestamps, so the new file's `st_ctim` was also identical to the
old one's (whole seconds, nanoseconds zero) and, at the same size, `fstat`
could not tell them apart. `FS_IOC_GETVERSION` did: the inode generation
differed (`dedc637f` against `7142ca29`); ext4 draws a new one for every
inode it creates. Overlayfs and tmpfs answer the ioctl with `ENOTTY`.

Expected: device, inode and status-change time to identify a file between
two opens.

Impact: the artifact reader's shard identity (`OpenShardForDirectRead`,
`src/artifact/artifact.h`) includes the generation where the file system
reports one; the replacement test fails on the 128-byte-inode ext4 without
it. Where there is none (overlayfs, tmpfs, NFS), a same-size replacement
within one timestamp tick is not detected. The request's declared argument
is a `long` even though ext4, btrfs and xfs write an `int`: FUSE copies back
up to the declared 8 bytes from its server, so the buffer must be a `long`.

## RE-022: The GB10's L2 does not keep host-located CUDA memory, so re-reads go to DRAM  (2026-09-27, status: worked-around)

On `spark` (GB10, driver 580.178.04), GPU reads of memory that CUDA
allocates at a host location miss L2 every time they re-read it. That
covers `cuMemCreate` at `HOST_NUMA` or `HOST` (jitLLM's host VMM, whether
mapped for the CPU or not) and `cudaMallocHost`. A kernel re-reading a
4 MiB buffer gets 0 of 8,388,608 L2 sector hits and 243 GB/s (DRAM rate).
The same kernel over `cudaMalloc` or device VMM gets 98.4% hits and
1,952 GB/s. Streaming reads run at ~240 GB/s from every kind, so a
bandwidth scan cannot show the difference; that is how D-034's evidence
missed it. Blocks that write to such memory also finish more slowly
(151,936 one-write blocks: 246 vs 96 µs). GEMMs, grouped attention and
matrix-vector products re-read through L2, and ran 1.1–4.9× slower with
all their buffers there (BP-F1). Ordinary memory read through ATS
(pageable, registered, managed) is cached in L2, but it streams at only
~165 GB/s. The CUDA 13.4 headers offer no allocation or access flag for
caching, and a persisting access-policy window does not change it.
`cuMemSetAccess` refuses to map device VMM for the CPU
(`CUDA_ERROR_NOT_SUPPORTED`), and it cannot be exported as a dma-buf
either (RE-025). Measurements, options and reproduction:
[host-vmm-diagnosis](experiments/host-vmm-diagnosis/README.md).
Before placing any buffer the GPU re-reads in host-located memory on the
Spark, measure it with a re-reading kernel, not a scan. D-081 keeps
weights and state in device VMM and uses host VMM only as a landing zone.

## RE-021: GGML's graph object trips UBSan on creation  (2026-09-27, status: worked-around)

In the `cross-asan` build on `spark-b` (address and undefined sanitizers,
GGML's `ggml.c` instrumented too), `ggml_new_graph_custom` and
`ggml_graph_overhead_custom` end the process: `ggml_graph_nbytes` sizes the
graph by advancing a null pointer (`ggml.c:7424`, "applying non-zero offset
96 to null pointer"). The plain builds never notice. So jitLLM builds no
`ggml_cgraph`: the fusion gates (`src/kernels/ggml/fusion.h`) take a node
list in GGML's order (`GraphOrder`) and count uses as GGML's graph does.
Anything else that needs a `ggml_cgraph` (the graph allocator, CPU
diagnostics) meets this first under the sanitizers.

## RE-020: The reference container denies io_uring setup  (2026-09-26, status: worked-around)

On the workstation, the three `unit.UringTest.*` tests pass, but the
default reference container used by `check:full` returns `EPERM` from
`io_uring_setup`, before any I/O. The tests now explicitly skip when setup
is unavailable (`ENOSYS`, as under qemu-user) or denied (`EPERM`), while
other setup errors still fail. Real I/O remains covered on the workstation
and by `check:spark`; an offline-container pass does not claim that coverage.

## RE-019: CUDA VMM backing escapes cgroup memory accounting on the Spark  (2026-09-25, status: open)

Environment: `spark-c4e2`, GB10, driver 580.178.04, kernel
7.0.0-1019-nvidia, cgroup v2; backing created with `cuMemCreate` through
jitLLM's CUDA provider.

Observed: 8 GiB of device-local or host-NUMA backing moved the process's
cgroup `memory.current` by at most 40 MiB, while `MemAvailable` fell by the
full 8 GiB at creation. Device backing never appears in the process's RSS;
host backing appears there (as `RssFile`) only while mapped with access.
The driver's per-extent bookkeeping (about 34 KiB of unreclaimable slab per
2 MiB extent) is not charged to the cgroup either.

Expected: memory a process pins to be charged to its cgroup.

Impact: `MemoryMax=` on `jitllm.service` would not bound the runtime's
backing, and an OOM decision based on the cgroup would not see it. The
runtime's own budget `B`, checked on every materialization, is the bound;
the memory breakdown reconciles against `MemAvailable`. Measurement:
[vmm-counters](experiments/vmm-counters/README.md).

## RE-018: Btrfs quietly serves misaligned or compressed direct I/O through the page cache  (2026-09-25, status: open)

Environment: the workstation's build tree, btrfs mounted with
`compress=zstd:1`, kernel 7.0.0-31-generic. A file opened with `O_DIRECT`
and read at file offset 1, through io_uring or `preadv`, returned the data
(4096 bytes) instead of `EINVAL`.

Expected: a refusal, as ext4 and XFS give. Btrfs falls back to buffered I/O
for direct I/O it cannot do in place (misaligned requests, and compressed
extents), so the page cache fills and the caller cannot tell.

Impact: a storage role on btrfs can pass D-034's direct-I/O probe
(`platform/direct_io.h`, which accepts btrfs) yet page through the cache,
against D-034's no-page-cache intent. Tests that expect the kernel to refuse
misaligned direct I/O accept either outcome on btrfs
(`unit.UringTest.*`). The Spark roles are ext4.

## RE-017: A sleeping thread takes hundreds of microseconds to wake on the Spark  (2026-09-24, status: open)

Environment: `spark-c4e2`, GB10 (Cortex-X925/A725), DGX OS 7.6.0, kernel
7.0.0-1019-nvidia, cpuidle `acpi_idle` with the `menu` governor (LPI-0 to
LPI-3, exit latencies 0/42/231/433 µs), cpufreq `performance`; jitLLM's
`WakeFlag` (a mutex and condition variable) built with the pinned SDK.

Observed: after a 100–400 µs idle gap, a thread sleeping on a condition
variable took 207–283 µs at p50 and 451–485 µs at p99 to run after it was
signalled (three runs of 5,000 samples; earlier runs gave p50 from 88 to
370 µs). The workstation (i9-11900KF, `intel_idle`) took 2.7–72 µs at p50
across all runs. Polling with `yield` woke in under
1 µs on both, at a whole core's CPU.

Expected: tens of microseconds, as on the workstation.

Likely cause, not isolated: the governor choosing LPI-2 or LPI-3 for the
idle gap. Disabling idle states to confirm it is a system change that needs
the owner's approval.

Impact: a scheduler that sleeps between a launch and its completion can add
up to about half a millisecond per step at the tail. It should poll while a
critical-path completion is imminent, and sleep only when idle. Measurement
and harness: [task-lanes](experiments/task-lanes/README.md). The storage and
device submission lanes poll the same way: asleep between a load's reads
and copies, they took ~100–200 µs, and at the tail ~380 µs, to
wake for the next one, which kept the landing zone below depth
([pagein-perf](experiments/pagein-perf/README.md)).

## RE-016: Ubuntu's snapshot service has no ports archive, so arm64 packages cannot be pinned by date  (2026-09-24, status: worked-around)

`https://snapshot.ubuntu.com/ubuntu-ports/<timestamp>/` answers HTTP 401,
while `https://snapshot.ubuntu.com/ubuntu/<timestamp>/` serves the amd64
archive. With `APT::Snapshot` set in an arm64 Ubuntu 24.04 container, `apt-get
update` still fetches the live `ports.ubuntu.com` indexes, and `apt-get
install systemd` then fails with "Unable to locate package". So the arm64
install-test image (`packaging/install-test/`) takes systemd from the live
ports archive and the test prints the version it got. The SDK is unaffected:
it pins each arm64 `.deb` by URL and SHA-256 (D-070).

## RE-015: A multi-arch image digest can run the wrong architecture from the local image store  (2026-09-23, status: worked-around)

Workstation, Docker 29.8.1 with the containerd image store and qemu binfmt
registered. `docker run ubuntu:24.04@sha256:008173c2…` (the multi-platform
index digest) ran the **arm64** image under qemu-user, with only the warning
"The requested image's platform (linux/arm64/v8) does not match the detected
host platform". An arm64 `ubuntu:24.04` pulled for the D-061 qemu tests was
the local content for that digest. With binfmt registered, nothing fails, so
builds and tests silently run emulated for the wrong target. Workaround: the
reference container's `FROM` and every `docker run` or `docker build` of it
name `--platform linux/amd64` (the Dockerfile skips BuildKit's
`FromPlatformFlagConstDisallowed` check on purpose), and `doctor` inside the
container reports the architecture.

## RE-014: LeakSanitizer aborts AArch64 tests under qemu-user  (2026-09-23, status: worked-around)

Workstation (x86-64), Ubuntu `qemu-user-static` 1:8.2.2+ds-0ubuntu1.18 via
binfmt, running the D-059 GoogleTest suite cross-built with Clang 22.1.8,
`-fsanitize=address,undefined`, the D-060 static GCC 16.2 runtime and the
arm64 compiler-rt from the D-059 SDK, with `QEMU_LD_PREFIX` set to the
sysroot. All 10 tests pass. At exit, LeakSanitizer reports "LeakSanitizer has
encountered a fatal error" and the process exits 1, so a passing suite looks
failed. With `ASAN_OPTIONS=detect_leaks=0` it exits 0, and ASan still catches
the heap-overflow probe. Workaround (D-061): leak detection is off for
emulated AArch64 runs; LSan runs natively on x86-64 and on the Sparks.
ThreadSanitizer under qemu-user was not tried.

## RE-013: Ubuntu 24.04 blocks unprivileged network sandboxes (`unshare -rn`, `bwrap`)  (2026-09-23, status: worked-around)

The workstation, `spark` and `spark-b` (Ubuntu 24.04.5) all set
`kernel.apparmor_restrict_unprivileged_userns=1`. Two unprivileged ways to
deny a build its network both fail:

- `unshare -rn` stops at `write failed /proc/self/uid_map: Operation not permitted`.
- `bwrap --unshare-net` stops at `loopback: Failed RTM_NEWADDR: Operation not permitted`.

Both need an AppArmor profile or root. Workaround (D-061): the offline
build gate runs in the reference container with `docker run --network none`,
which works on the workstation. The owner's `spark` account is not in the
`docker` group; arm64 containers run on the workstation through qemu binfmt
(installed 2026-09-23, D-061). Changing the sysctl or adding an AppArmor
profile is a system change the owner has not made.

## RE-012: Pinned SGLang MiMo-V2 startup fails with a misleading processor error without torchcodec  (2026-09-22, status: worked-around)

`lmsysorg/sglang` nightly `0f6761b5` (arm64 digest `9e1fb4c3…`) on both
Sparks, MiMo-V2.6-Flash-RL. Startup, even for text-only use and with
`--enable-multimodal`, aborted in the tokenizer manager with
`No processor registered for architecture: ['MiMoV2ForCausalLM']`. The real
cause is earlier and swallowed: `multimodal/processors/mimo_v2.py` imports
`torchcodec`, which the image lacks, and SGLang's processor discovery logs and
skips modules that fail to import. Two boots were spent before the import
was tested directly. Workaround: a derived image adding hash-pinned
`torchcodec` 0.16.0 ([MiMo reference](experiments/mimo-reference/README.md));
the MiaAI recipe also installs it. The boot without `--enable-multimodal`
failed identically, so the engine builds this architecture's multimodal
processor regardless of that flag; dropping the flag with `torchcodec`
present was not tried. When a registry lookup reports "not
registered", import the module directly before debugging arguments.

## RE-011: Upstream GGML CUDA broadcast ops abort on strides above 2^32 elements  (2026-09-22, status: open)

stable-diffusion.cpp `c92d73c` built against upstream GGML `8846b79` (CUDA
13.0.3, `sm_121a`), GB10 `spark-b`, Qwen-Image-2.1 Q4_K_M at 2048²/40. All 40
denoising steps completed; the Wan VAE decode then failed
`GGML_ASSERT(s02 <= std::numeric_limits<uint32_t>::max())` in
`src/ggml-cuda/binbcast.cu:270` (exit 133). The same build matched the
patched fork's pixels exactly at 512² and 1024². The assert is unchanged in the
fork and in upstream master `179b60f` (2026-09-22), and upstream master still has
no CUDA `conv3d.cu`, which the fork adds with implicit-GEMM conv2d/conv3d.
Plausibly the fork's conv path never materializes the oversized intermediate
that upstream's im2col path passes to a broadcast op; not verified, and
upstream master was not executed. Impact: GGML CUDA elementwise/broadcast
kernels carry 32-bit stride limits, so very large activation or workspace
tensors (here a 38.3 GiB decode buffer) need chunking or different kernels.
Check tensor strides against these limits before planning GGML execution of
high-resolution image/video decoders. [Report](experiments/image-gguf/README.md).

## RE-010: Adding graph outputs changes logits with CUDA optimizations  (2026-09-22, status: open)

Pinned llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, GB10, driver
580.178.04, Gemma 4 26B A4B. A libllama patch that only appended copies of
each layer's routing IDs as graph outputs changed every logit row and 44 of
3,072 teacher-forced predictions with CUDA fusion and graphs enabled. It
changed nothing with both disabled, and the patched library matched the
official one bit-for-bit with the feature off. An eval callback that never requests data
also changed nothing. Extending a tensor's lifetime or adding outputs
changes the compute-buffer layout. A change in which fusions qualify is a
plausible explanation, but these controls disabled fusion and CUDA graphs
together: neither their individual roles nor the specific affected operation
was isolated.

Treat any graph-shape or lifetime change (instrumentation, extra outputs,
debug copies) as a potential numerical-plan change under CUDA optimizations.
Reference controls and jitLLM's own GGML integration must compare optimized
logits exactly before assuming an observation point is transparent. See the
[fused-routes experiment](experiments/fused-routes/README.md#rejected-design-routes-as-graph-outputs).

## RE-009: Pinned ExLlamaV3 compiles x86-only CPU helpers on Spark  (2026-09-22, status: worked-around)

ExLlamaV3 `6b84a21b6f1e5da3f291b9e1019061f0de788279`, Spark AArch64,
GCC 13.3.0, CUDA 13.0.88 and PyTorch 2.14.0+cu130. Import builds every
extension translation unit, including x86 CPU feature probes, CPU MoE and
CPU collectives. The unmodified build fails on `__builtin_cpu_supports` in
`avx512_target.cpp`; host spin waits also use `__builtin_ia32_pause`.

The [external reference patch](experiments/exl3-reference/arm-reference.patch)
returns false for x86 capability probes on ARM, makes unsupported CPU MoE
and CPU-reduction entry points fail explicitly, and uses the ARM `yield`
instruction for host spin waits. EXL3 GPU kernel bodies are unchanged.
This is a bounded **single-GPU reference build**, not a portable CPU backend
or validation of upstream CPU offload or tensor-parallel collectives.
The [baseline report](experiments/exl3-reference/README.md) records the build
and execution identities. Native jitLLM integration still adopts only its
audited operation closure; it does not inherit this whole external extension.

## RE-008: Extended reference runs do not always preserve exact top-1 predictions  (2026-09-21, status: open)

Pinned llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, GB10,
driver 580.178.04, CUDA fusion and graphs disabled. Two distinct limits surfaced
in the [full paging study](experiments/paging-feasibility/full-study.md):

- DeepSeek V4 Flash sequence snapshots reserialize identically after restore,
  but append-only continuation differs from live state on 16/1,536 top-1
  predictions (0/6/10 by turn). A same-host repeat reproduces those differences;
  restored predictions also match across the two Sparks. Raw/compressed KV and
  compressor state are present in the serializer. Cache compaction changes
  attention shapes, but neither a numerical explanation nor missing semantic
  state has been established. This is not the Gemma rollback failure in RE-007.
- Qwen3.8 Flash Next's identical untraced binary repeated on one host differs
  on 6/1,536 predictions (3/2/1). A restore probe differs on 10, including five
  before any restore, so restoration is not an isolated cause. Qwen also prunes
  the last prefill layer to output rows (`models/qwen4exp.cpp:400`); assuming
  every layer routes the whole input batch aborts capture. Record the actual
  final-layer output-only dependency, without reducing consumed input tokens.

No numerical tolerance is inferred from these counts. Capture defaults remain
strict; Qwen's explicit drift-recording mode labels failed equivalence, and
replay/locality analysis require a separate opt-in. Both large-model spill
returns use conservative recomputation in the study. Raw evidence remains in
external `paging/large-capture-2`, `deepseek-restore-probe-2`, `qwen-capture-1`,
`qwen-capture-3`, and `large-restore-probe-1`; their identities and comparisons
are retained in the study's aggregate evidence. Do not promote these reference
observations into a jitLLM numerical or restore-compatibility guarantee.

The [image-reference follow-up](experiments/image-reference/README.md)
(2026-09-22) also finds a short Gemma configuration difference with this
llama.cpp revision and its default CUDA fusion/graph settings: a 639-token
continuation restored from 627 saved tokens matches resident execution, but
fresh-context full prefill differs at 15 of 32 generated positions, starting
at zero-based position 17. Full-prefill controls before and after image
generation match each other, so image switching is not required to reproduce
the difference. This is a free-running token comparison, not a teacher-forced
logit diagnosis, and does not establish a common cause with the cases above.
Preserve the failed cross-mode comparison rather than silently declaring
cached and recomputed trajectories equivalent.

## RE-007: Gemma sequence snapshots lose SWA history needed after prompt rollback  (2026-09-21, status: worked-around)

On Spark GB10/driver 580.178.04 and pinned llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, the extended four-turn
Gemma restore probe changed **58/3,072** teacher-forced next-token argmax
predictions versus live full-SWA state (per turn: 0, 12, 19, 27). Ornith
changed **0/3,072**. CUDA fusion and graphs were disabled in both arms.
Each snapshot reserialized byte-for-byte identically after restoration into
a fresh context; that checks the stored bytes, not sufficient context coverage.
Raw negative evidence is `paging/restore-probe-1` on Spark, with the matched
live controls in `paging/small-capture-2`.

Pinned `src/llama-kv-cache.cpp:2080` drops cells outside the final SWA window
when serializing an individual sequence, even with full-SWA allocation.
Gemma uses standard SWA of 1,024 tokens. Its first snapshot ends at position
8,105 and retains SWA positions 7,082–8,105. The canonical next prompt shares
only 7,335 tokens: its first resumed query needs positions 6,312–7,335,
including **770 positions absent from the snapshot**. The next two returns
have the same missing-window count. Successful tail removal does not detect
this gap. Physical cache compaction also changes attention shapes
(`llama-kv-cache.cpp:1250`), but the missing dependencies alone invalidate
assuming equivalent restored execution; the prediction changes are not
classified as harmless numerical noise.

The native session and restore harnesses now check the earliest retained
position against the model's actual SWA window before reuse, including after
full-SWA restoration. They reset and recompute when coverage is insufficient,
following the conservative checkpoint-coverage principle in pinned
`tools/server/server-context.cpp:3297` and `:3349`. They have no older
checkpoint to restore. Replay uses captured recompute routes for Gemma
sequence-spill returns with prefix rollback, and rejects apparent reuse from
legacy normal-SWA captures by selecting the conservative recompute scenario.
Serialized byte identity and a successful short response do not establish
that a checkpoint supports arbitrary template rewrites. The earlier A→B→A
six-token rollback's matching output remains a narrow observation, not a
proof of complete retained-window coverage. Full-history snapshots would be
a different, larger spill contract and are not established by this probe.

## RE-006: Reading MoE routes through the llama.cpp callback changes the CUDA path  (2026-09-21, status: worked-around)

On the pinned reference `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`,
GB10/driver 580.178.04, a callback at `ffn_moe_topk-N` changed 51 of
3,072 teacher-forced next-token argmax predictions over four Gemma turns.
A repeated untraced control matched all 3,072 original control predictions;
the first traced difference occurred before any conversation reuse. The
short first-cut 118-prediction probe had matched, so it did not expose this.

The scheduler splits and synchronizes the graph at requested callback nodes.
The pinned CUDA implementation has a fused routing operation spanning the
selected top-k view, so this observation point changes the execution path.
Disabling both CUDA fusion and CUDA graphs in **both** traced and control
runs (`GGML_CUDA_DISABLE_FUSION=1`, `GGML_CUDA_DISABLE_GRAPHS=1`) restored
exact prediction equality in the extended retained and recomputed Gemma
captures. Those two settings were changed together; this does not isolate
their individual effects or prove bitwise-logit equivalence.

Keep instrumented-route experiments explicitly matched to their control
configuration, and keep their timing separate from the reference's normal
optimized path. See the
[paging-feasibility experiment](experiments/paging-feasibility/README.md).

**Fusion-preserving capture (2026-09-22).** Reading each layer's IDs at the
end of its gated-activation fusion group, before the down projection
consumes them, keeps upstream fusion and CUDA graphs: logits were
bit-identical to untraced runs on all 3,072 Gemma and 3,072 Ornith outputs,
using the unmodified image
([fused-routes experiment](experiments/fused-routes/README.md)). With fusion
and graphs off, the same read point reproduces the study's recorded routes
exactly. The optimized plan and the plan with both disabled select different
expert sets in 40–43% of token-layer rows, rising with depth. The study's
routes describe the plan with fusion and graphs disabled. New captures should
use the boundary read and re-check untraced equality per model/revision; the
recorded captures were not redone.

## RE-005: Spark perftest warmup option stalled an RDMA-CM sweep  (2026-09-21, status: worked-around)

Environment: both Sparks, kernel `7.0.0-1019-nvidia`, ConnectX firmware
`28.45.4028`, `rdma-core 50.0-2ubuntu0.2`, Ubuntu perftest
`24.01.0+0.38-1build2` (binary reports 6.20), RoCE v2 / MTU 1024.
An `ib_write_bw -d rocep1s0f1 -R -p 18700 -a -n 1000 --report_gbits
--perform_warm_up` server/client pair failed to finish within 90 seconds;
both remote timeouts returned 124. Removing the optional warmup flag
completed the sweep, and the subsequent repeated baseline passed. Duration
tests discard a one-second start/end margin instead.

This is an observed option-combination timeout, not an isolated root cause
or evidence that the DAC requires a reboot. The excluded pilot receipt and
logs remain in workstation `/tmp/jitllm-interconnect/host-sweep/`; the
[baseline report](experiments/interconnect/README.md) records the working
protocol. Also, `ib_write_bw --version` prints `Version: 6.20` but exits 1;
do not treat that informational exit as a failed transfer.

## RE-004: llama.cpp Gemma slot restore reports success but default SWA re-prefills  (2026-09-21, status: worked-around)

Environment: `spark-c4e2`, GB10, driver 580.178.04; pinned llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` in the CUDA 13.3 ARM64 container;
Gemma 4 26B A4B Unsloth UD-Q4_K_M. Full identities and the retained
[smoke harness](experiments/reference-setup/README.md) accompany the result.

With an 8192-token context, f16 K/V, one slot, batch/microbatch 512, and
default SWA retention, the slot save and fresh-process restore endpoints
both reported **627 tokens / 141,268,896 bytes**. A 639-token continuation
then re-evaluated **all 639 tokens**; the server logged a full prompt
re-processing fallback due to missing cache data. The equivalent resident
continuation evaluated 18 tokens after an internal checkpoint rollback.
The prompt was shorter than the 1024-token sliding window. Successful
serialization counters therefore do not establish useful continuation reuse.

The pinned [server source](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/server-context.cpp)
uses SWA coverage thresholds and context checkpoints when deciding whether
the common prefix can be reused. `--swa-full` disables that SWA checkpoint
path. With it, two runs restored the 627-token prefix, evaluated only the
12-token extension, and matched all 32 resident-continuation output token
IDs. This is a measured workaround for this pinned configuration, not a
general fix for all hybrid models or proof of long-context coverage.

Cost: logged KV allocation grows from **460 MiB** (160 global + 300 SWA)
to **1760 MiB** (160 global + 1600 full-SWA) at this context. Keep the
normal/windowed and full-SWA reference configurations separate in the
A→B→A experiment and include their actual memory costs. To reproduce the
negative control, remove only `--swa-full` from an external copy of
`smoke.py`; its restore assertions must fail rather than report a pass from
the matching API counters. The two differing continuation token streams in
the negative control alone are not evidence of corrupted KV: they used
different prefill paths. No upstream fix is claimed.

Long-context follow-up: the [A→B→A report](experiments/reference-aba/README.md)
reproduces this behavior at context 32,768. Three default-SWA restore trials
reported 18,303 restored tokens but processed all 18,339 continuation input
tokens. Full-SWA restore reused 18,297 and processed 42, matching every
118-token output. The six-token difference between saved and reusable
counts is legitimate: Gemma's canonical chat template removes the empty
generation-only thinking marker from completed turns. Compare the actual
longest common token prefix, not just the save API's count. A separate
early/late notebook recall check passed. Also retain the measured decode
speed distinction (about 27–28 tokens/s live/recomputed full-SWA versus
46–47 after restore/default SWA); its cause was not isolated here.

## RE-003: nvme-cli 2.8 feature control requires --value, and zero has a different printed form  (2026-09-21, status: worked-around)

Environment: Spark, installed nvme-cli 2.8-1ubuntu0.1. During the bounded
interrupt-coalescing comparison, a command using `set-feature -f 8 -v 263`
set zero: `-v` is **verbosity**, not value, in this version; the value option
is `-V` or `--value`. Also, `get-feature` prints nonzero as
`Current value:0x00000107` but zero as `Current value:00000000`. A parser
requiring `0x` rejected zero, including during the attempted cleanup.

The hardware readback exposed the mismatch before any A/B measurement was
retained. The original value was restored with
`nvme set-feature /dev/nvme0 --feature-id=8 --value=263`, then independently
read back as `0x107`. The comparison was rerun in full with long options,
both output forms accepted, and verified restoration. Do not trust mocks
based on remembered short flags for a device-control command: check the
installed tool's help and read back the actual state. The retained
[coalescing harness](experiments/io-path/coalescing.py) records the original
value before changes and bounds/restores its temporary setting.

## RE-002: cuFile compatibility mode rejects a descriptor opened with O_NOFOLLOW  (2026-09-21, status: worked-around)

Environment: `spark`, GB10, driver 580.178.04, installed libcufile package
1.15.1.6-1 (reported API version 2.12), Linux 7.0.0-1019-nvidia, ext4.
The I/O spike opened its private regular file with
`O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW`; `cuFileHandleRegister`
failed with **5019 / CU_FILE_INVALID_FILE_OPEN_FLAG**. The library log
reported unsupported open flags `229376`. Native direct reads on the same
descriptor worked.

The comparison harness keeps its validated original descriptor open,
reopens `/proc/self/fd/<fd>` with `O_RDONLY | O_DIRECT | O_CLOEXEC`, and
checks device/inode identity before registering that new descriptor with
cuFile. This preserves file identity without following the original user
pathname again. Registration and subsequent GPU-verified reads then passed.
Do not respond by removing path protections from the original file open.
See the [I/O experiment](experiments/io-path/README.md) and its retained
`main.cc`. This workaround is confined to the comparison backend; the
native direct-file candidate does not need it.

## RE-001: CUDA 13.0 NVCC rejects C++23 even with a C++23-capable Clang host  (2026-09-21, status: worked-around)

Environment: x86-64 Ubuntu 24.04, NVCC V13.0.88, Ubuntu Clang 18.1.3;
AArch64 cross target using the Spark DGX OS 7.6.0 snapshot, GB10 `sm_121`.
Passing `-std=c++23` to NVCC fails before compilation with
`Value 'c++23' is not defined for option 'std'`. Clang's support for C++23
does not establish NVCC front-end support. This corrects the expectation
in D-010's original context; its allowance for a separate CUDA dialect
already covers the fix.

The older-toolkit comparison used C++23 `.cc` files and C++20 `.cu`
files and passed cross-built and native Spark runs. The selected D-032
pin instead uses Toolkit 13.4.2 / NVCC 13.4.92, which accepts C++23;
both execution paths passed with an explicit C++23 host/device feature
probe. M1 should use C++23 throughout for that selected SDK. Do not
silently fall back to the installed 13.0 compiler and assume the same
dialect support. See the [reproduction and pins](experiments/toolchain-smoke/README.md)
and [NVCC 13.4 options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
