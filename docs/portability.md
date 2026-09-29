<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Portability

How jitLLM stays ready for other GPU platforms (AMD, Apple silicon later)
and operating systems (macOS, Windows) without rewriting large parts of the
tree, while costing nothing on the GB10 (D-026, D-082). The owner's goal
(2026-09-29): every platform a compile target of the same tree, with shared
code maximized. Nothing here is built for another platform yet; this is
where the seams are, what checks them, and what a port would still have to
do.

## Where vendor and OS code may live

| Code | Only in | Everything else |
| --- | --- | --- |
| CUDA: headers, runtime and driver API names and types, kernels and launches, `.cu` files | `src/providers/cuda/` and `src/kernels/` | reaches the device through the provider interfaces, the device runtime and the kernels' own vendor-free headers |
| Linux-specific headers and calls: `<linux/...>`, epoll, eventfd, signalfd, io_uring, memfd, `O_DIRECT`, `O_TMPFILE`, `getrandom`, `accept4`, `SOCK_CLOEXEC`, `MSG_NOSIGNAL`, prctl, raw system calls, `/proc` and `/sys` paths | `src/platform/` and the Linux providers (`src/providers/uring_storage.*`) | calls the platform module |
| POSIX (open, poll, sockets, mmap, pthreads) | anywhere | macOS has it; a Windows port gives the platform module its own implementations (below) |

`tools/jitllm_boundaries.py` enforces the first two rows over `src/`, in
the light check tier (`tools/check`'s `boundaries` step) and in the tools
tests (`tools/tests/test_boundaries.py` checks this tree). It reads code
with comments removed, matches names outside string literals and paths
inside them, and fails on an include of `providers/cuda/`,
`platform/io_uring.h` or `providers/uring_storage.h` elsewhere. An
exception needs an entry in its `EXCEPTIONS` table with the reason, and a
stale exception is itself a failure. **There are no exceptions today.**
Tests and benchmarks drive the CUDA build directly and are not checked.

## The GPU seam

A backend is four things, all chosen at build time (D-028, D-053: no
runtime plugin ABI):

1. **The device-memory and device-execution providers**
   (`providers/device_memory.h`, `device_execution.h`): VMM reservations,
   backing and mapping; streams, fences and copies. Virtual interfaces,
   because the resource core also runs them over the fakes.
2. **The device runtime** (`providers/device_runtime.h`): opening the
   device, and what the engine's device jobs queue besides kernels: copies
   and fills on the job's stream, timing marks, recorded work (a captured
   and instantiated sequence replayed as one, CUDA graphs today), the
   thread's error state, pinned host memory, the device's architecture
   number and free memory. Plain functions the backend's provider module
   defines (`providers/cuda/cuda_device_runtime.cc` over the CUDA runtime
   API), one direct call around the backend's own: no virtual dispatch,
   no table lookup, no allocation on success.
3. **Kernel modules** (`src/kernels/<source>/`), which unwrap the
   job's `NativeStream` handle themselves and register their
   implementations (D-053).
4. **The registry's plans**, which a backend can only run if every
   operation has an implementation it provides ([below](#the-registry-rule)).

The engine (`src/engine/`) is above all four and holds no vendor code: it
is built without CUDA's include path, so a CUDA header reaching it fails
the build, and the boundary check fails a CUDA name. It stays a CUDA-build
module only because it links the CUDA kernel modules and GGML graphs
(D-096).

### What moved out of the engine (2026-09-29)

The engine made 69 direct CUDA runtime calls. They now go through the
device runtime, one for one: 22 `cudaMemcpyAsync` → `CopyAsync`, 3
`cudaMemsetAsync` → `FillAsync`, `cudaMallocHost`/`cudaFreeHost` (9) →
`AllocatePinned`/`FreePinned` (and the node's `HavePinned` helper for a
runner's state copy), the node's per-stream timing events (create,
record, elapsed, destroy) → timing marks, the image runner's graph
capture, instantiate, launch and destroy → `BeginRecording`,
`EndRecording`, `Replay` and `RecordedWork`, `cudaMemGetInfo` →
`QueryDeviceMemory`, the compute-capability queries and
`cudaGetDeviceProperties` → `QueryDeviceFacts`, `cudaSetDevice` +
`cudaFree(nullptr)` and the CUDA providers' constructors → `OpenDevice`,
and `cudaGetLastError`/`cudaGetErrorString` → `TakeLastError` and
`DeviceStatus::text`. The two `.cu` kernels the engine held (the weights'
range fill and Qwen3.8's n-gram row gather) moved to `src/kernels/paging/`
with a vendor-free header. The engine also opens its storage rings through
`providers::OpenStorage` rather than naming the io_uring provider.

**Cost on the GB10:** none measurable. Plain decode through `jitllm-runtime
chat --plain` (256 tokens, second turn of each model, two alternating
rounds; `spark`, 2026-09-29): DeepSeek V4 Flash 22.13 and 22.21 tok/s on
main, 22.10 and 22.16 on the slice; Qwen3.8 Flash Next 27.60 and 27.67 on
main, 27.64 and 27.72 on the slice.

**Left as it is:** the decode graphs of DeepSeek and Qwen3.8 are captured
by the GGML kernel module's launch context (`kernels/ggml/launch.h`,
`CapturedGraph`), which is kernel-layer code and may use CUDA; a backend
without graphs replays launch by launch, the path those runners already
take when graphs are off. `RecordedWork` is the same concept at the
provider seam, which the kernel layer could adopt later.

### What a new GPU backend supplies

`providers/<vendor>/` with the two providers and the device runtime (for
HIP, a near copy of the CUDA module over `hip*` calls; for Metal, MTLHeap
placement for backing, command queues for streams, MTLSharedEvent for
fences, and MTLIndirectCommandBuffer or plain re-encoding for recorded
work), kernel modules for its operations (GGML's HIP and Metal backends
are kernel sources to adapt under jitLLM's dispatch, D-053), and a CMake
profile that builds them instead of CUDA's. The device-memory design
assumes explicit virtual reservation and mapping (D-006); where a platform
has no equivalent, the provider maps the rules D-033 needs onto what it has,
which is that port's decision to make.

## The OS seam

| Service | Interface | Linux | macOS | Windows |
| --- | --- | --- | --- | --- |
| Event loop for the chat route | `platform/event_loop.h`: `EventLoop` (level-triggered readiness: readable, writable, peer closed, hang-up, error, with a tag) and `Waker` | epoll, eventfd | kqueue (`EVFILT_READ`/`WRITE`, `EV_EOF`), `EVFILT_USER` | an I/O completion port with AFD poll requests for socket readiness (as wepoll and mio do), a posted completion as the waker |
| Signals as readiness | `SignalWatch` | signalfd over blocked signals | `EVFILT_SIGNAL` | a console control handler posting to a waker |
| Available memory | `platform::AvailableMemoryBytes` | `/proc/meminfo` MemAvailable | `host_statistics64` (free, inactive, purgeable) | `GlobalMemoryStatusEx` |
| Spill file | `OpenUnnamedDirectFile` | `O_TMPFILE` + `O_DIRECT` | `mkstemp`, unlinked at once, `F_NOCACHE` | `FILE_FLAG_DELETE_ON_CLOSE` + `FILE_FLAG_NO_BUFFERING` |
| Shard for direct reads | `OpenForDirectRead` (with an optional buffered fallback) | `O_DIRECT` | `F_NOCACHE` (a hint) | `FILE_FLAG_NO_BUFFERING` |
| File identity | `FileGeneration` | `FS_IOC_GETVERSION` | `st_gen` (root only) or none | the volume serial and file ID |
| Anonymous shared memory (the fake device memory) | `OpenAnonymousMemoryFile` | memfd | `shm_open` + `shm_unlink` | a pagefile-backed section |
| Sockets | `platform/sockets.h` | `SOCK_NONBLOCK`/`SOCK_CLOEXEC`, `accept4`, `MSG_NOSIGNAL` | `fcntl` after the call, `SO_NOSIGPIPE` | Winsock |
| Random bytes | `FillRandom` | `getrandom` | `getentropy`/`arc4random_buf` | `BCryptGenRandom` |

Behaviour and performance on Linux are unchanged: each is the call the
runtime made before, moved.

### Storage and direct I/O

The storage provider (`providers/storage.h`) is where a port's I/O goes:
`OpenStorage(depth)` is the system's implementation, io_uring's on Linux
(`providers/uring_storage.*`). Every request resolves as not started,
accepted or unknown, and only its completion retires its memory (D-048),
which any asynchronous mechanism can honour:

- **macOS:** a small thread pool issuing `pread`/`pwrite` on descriptors
  opened with `F_NOCACHE`, completions queued back to the lane, `Wake` by
  a waker; or `dispatch_io` with the same queue. There is no true direct
  I/O: `F_NOCACHE` asks the unified buffer cache not to keep the pages,
  which matters because the page cache comes out of the same unified
  memory budget, as on the GB10 (D-034).
- **Windows:** overlapped `ReadFile`/`WriteFile` on handles opened with
  `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` (sector-aligned, like
  `O_DIRECT`), completions from an I/O completion port, `Wake` by
  `PostQueuedCompletionStatus`.

jitLLM probes direct I/O per storage role at startup
(`platform::ProbeDirectIo`) and refuses a role without it, because on the
GB10 reads through the cache would spend the one memory budget.
`OpenForDirectRead`'s `buffered_fallback` exists for a platform where
direct I/O is only a preference (as TensorFold's reader falls back to
buffered reads when `O_DIRECT` is refused); jitLLM never asks for it on
Linux, and a port that does decides what the role check accepts.

### What a new OS still needs

A Windows port gives the platform module a descriptor type of its own
(it is `int` throughout today), replaces the POSIX calls that the rest of
the tree makes directly (`open`/`openat` beneath a directory for the trust
walk and the artifact reader, `poll` in the chat route's driver, BSD
sockets), and the process services (`platform/job.*`, `confine.*`,
`crash_policy.*`, `lock_file.*`, `sd_notify.*`), which are Linux's by
design. A macOS port needs only the platform module and the storage
provider: the rest is POSIX. Neither is started.

## The registry rule

(D-053's note of 2026-09-29.) Every fused or fast operation has a fallback
composed of primitive operations in the registry, so a new backend can run
a model with primitives alone, and fused kernels are optional speedups.

**How plans choose today.** The registry's `Operation` enum
(`execution/registry.h`) lists 57 operations, primitives and fusions
alike, unmarked; an `Implementation` records nothing about what it is
equivalent to, and `Resolve` binds exactly the names a plan records. Only
one operation declares a fused and an unfused implementation
(`ggml.rms_norm_mul.fused` and `.unfused`). Most fusion choices are not the
registry's but the graph builders': which nodes they emit
(`Dsv4GraphOptions::fused`, `Qwen38GraphOptions::fused`, `exact`,
`experts`), plus `PlanGraph`'s `DeviceChoices` and upstream's four fusion
gates (`kernels/ggml/fusion.h`). So "the fallback" is, per model, a graph
built another way, not a registry entry.

**Audit (2026-09-29):**

| Model | Primitive path today | Where the rule fails |
| --- | --- | --- |
| Qwen2 FP16 fixture | Yes: the FP16-U plan, fusion off (`mul_mat`, `soft_max` attention, NEOX RoPE, `rms_norm`, SwiGLU, `get_rows`, `set_rows`) | none |
| EXL3 fixture | No: its plan table admits `ggml.rms_norm_mul.fused` and `ggml.flash_attn_ext.vec` but not `.unfused`, and the EXL3 linear has only ExLlamaV3's kernels | the EXL3 trellis dequant and GEMM (any backend needs its own) |
| DeepSeek V4 Flash | Mostly: the exact plan (`--exact on`) is llama.cpp's graph node for node, the unfused form of every fast-plan operation (`jitllm.vecq`, `.dsv4.route`, `.combine`, `.hc_mix`, `.compress`, the fused norm, `mmvf_rows`) | `dsv4_hc_comb` (Sinkhorn), `dsv4_hc_post` and `lightning_indexer` have no composition of primitives; `mul_mat.fwht` is forced by its hint though the dense rotation matrix is at hand; attention is `flash_attn_ext` with sinks at D 512 only; the DSpark drafter's `jitllm.argmax` has no alternative |
| Qwen3.8 Flash Next | In the harness only (`jitllm_qwen38_exec --unfused` on the GGML-layout artifact): MXFP8 dequant + `mul_mat`, `mul_mat_id` experts, GGML nodes for hyper-connections, routing and QSA | serving is always fused and fast from the CUTLASS-layout artifact, and the MTP drafter and batched verify exist only in the fast form; `ssm_conv`, `gated_delta_net`, `jitllm.nvfp4.get_rows` and `jitllm.argmax` have no alternative; `jitllm.mxfp8.mul_mat_vec` is always chosen at 8 rows or fewer though dequant + `mul_mat` would do |
| Qwen-Image-2.1 | No: the legacy plan is jitLLM's own CUDA too, differing from the fast plan in five roles (cuBLAS instead of cuBLASLt, unfused gated residual, im2col convolution, unfused attention norm, plain conversion); no role is GGML's | every one of its 30 roles is a jitLLM kernel or a cuBLAS product; 25 are declared for one implementation only |

**Minimum primitive set a new backend needs**, taking each model's most
unfused path that exists:

- **Qwen2 FP16:** F16 `mul_mat`, add, mul, `rms_norm`, NEOX RoPE, masked
  scaled `soft_max`, cont, SwiGLU, `get_rows` (F32), `set_rows` (F32 to
  F16).
- **DeepSeek V4 Flash (exact plan):** quantized `mul_mat` and
  `mul_mat_id` for Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS and MXFP4;
  F32, F16 and BF16 products; the Hadamard rotation (or dense
  `mul_mat`); `rms_norm`; NEOX RoPE with offset, and its inverse;
  attention with sinks at D 512; add, mul, div, scale with bias, sigmoid,
  softplus, sqrt, clamp, fill, SwiGLU and clamped SwiGLU; cont, repeat,
  concat, `sum_rows`, `soft_max`; `get_rows`, `set_rows`; argsort top-k and
  `top_k`; and, with no fallback yet, `dsv4_hc_comb`, `dsv4_hc_pre`,
  `dsv4_hc_post` and `lightning_indexer`. DSpark adds argmax.
- **Qwen3.8 Flash Next (unfused, GGML expert layout):** NVFP4
  `mul_mat_id` and NVFP4-row `get_rows`; BF16 and F32 products; MXFP8
  GEMV or dequant to BF16; `rms_norm`; multi-section RoPE (IMROPE);
  attention at D 256 and 512; add, mul, div, scale, sigmoid, SiLU,
  softplus, sqrt, ReLU, abs, sign, clamp, fill; repeat, concat, transpose,
  cont, `sum_rows`, `soft_max`; `get_rows`, `set_rows`; argsort top-k and
  `top_k`; `ssm_conv` and `gated_delta_net`. Verify and MTP add argmax.
- **Qwen-Image-2.1:** BF16 GEMM; 3×3 and 1×1 convolution (im2col and
  GEMM); non-causal attention at D 128 and a short causal one; SiLU,
  GELU-tanh, SwiGLU, add, gated residual, scale-shift, all rounding to
  BF16 where diffusers does; RMS, zero-centred RMS, layer norm with
  modulation and channel norms; 2D complex and NEOX RoPE; row embedding,
  transpose, nearest 2× upsampling, the Euler step and F32-to-BF16
  conversion. All jitLLM kernels today.
- **EXL3:** the FP16 set without `soft_max`, plus F16 and F32
  conversion, attention at D 64, the EXL3 dequant and GEMM, and a half
  bias add.

**What closing the gaps takes**, as the rule applies to new work from
now on: an unfused composition for the model-specific operations
(hyper-connection mixing and Sinkhorn, the lightning indexer, the gated
delta net and its convolution, argmax, NVFP4 rows) expressed in GGML
primitives, even slow, so a backend with GGML's primitives runs every
model; a dense fallback for the Hadamard product; a builder switch that
reaches the unfused Qwen3.8 graph from the served artifact (or an
artifact-independent expert layout); and for the image family a
primitive plan over GGML's operations or an equivalent set, with the
BF16 rounding points the reference needs. None of this changes the
GB10's plans, which keep their fastest implementations.

## The runners' shared skeleton

The M3 runners (`src/engine/`: `dsv4_runner.cc` 2,037 lines,
`qwen38_runner.cc` 1,708, `qwen_image_runner.cc` 732) grew one model at a
time, and a port touches every copy of what they share. Audit of
2026-09-29 (line numbers at this revision):

| Concern | DeepSeek V4 | Qwen3.8 | Qwen-Image | Shared today |
| --- | --- | --- | --- | --- |
| Weights as extents | its own `ReservePart`/`Register` (dsv4 457–688), a near copy of `PagedWeights::Reserve`/`Register` with a host-memory token table, per-layer slabs and drafter tags | `PagedWeights` | `PagedWeights` | `paged_weights.h` (not by DeepSeek) |
| Setup | cuBLAS workspace and handle (280–296), launch context and registry bind (835–847), pinned staging (398–438), shape measuring (328–396) | the same blocks (267–282, 622–634, 397–451, 318–394) | cuBLAS block (338–353) | nothing |
| Chunk plans | `Planned`, `PlannedDraft` (925–998): plan, scratch, pool check, bind, coverage, drop past 32 | `Planned`, `PlannedMtp` (768–843), the same | its own phases | `PlaceAndPlan` / `PlanPlaced` (`dsv4_plan.cc` 106–152, `qwen38_plan.cc` 204–250), identical but for names (DeepSeek summed its inputs unrounded while `Stage` rounds each to 256 bytes, as Qwen3.8 did; fixed in this slice) |
| Graph capture and replay | `QueueRuns` (1452–1528), `Stage`, graph eviction inline twice, path counting inline three times | `QueueRuns` (963–1041, a superset: the gather hook, several outputs), `Stage`, `RoomForGraph`, `Count` | one recorded step (`RecordedWork`) | the `Runs`/`Queued` types, copied |
| Speculation | `Settle`, `Usable`, `Accept`, `Rollback`, `QueueRestore`, `PlanSnapshot`, saved ranges | the same mechanics (1095–1134, 1558–1603), `QueueCommit` adding the commit kernel | none | nothing; which ranges a verify writes is model-specific |
| State spill and restore | `RegisterState` (748–782), `Clear` (852–880), `ReadState` (1835–1874), `CheckPlaces` | byte-identical apart from names (528–562, 702–731, 1605–1644) | none | nothing |
| Teardown | the unmap-release-free loop (2000–2014) | the same (1657–1671) | the same (701–716) | copied six times with the node and two harness runners |
| Serving adapters | `serving.cc` 150–348 | 352–585: the same constructor, forwarding overrides, tokenizer tail and accept loop | its own | `Llm::Prefill`/`Generate` are shared |

About 1,000–1,300 of the LLM runners' 3,750 lines are near-duplicates.
Recommended skeleton, as helpers a runner holds rather than one deep base
class (the image runner needs only the resource helpers), in order of
value:

1. **`GraphRuns`**: `Runs`, `Queued`, Qwen3.8's `QueueRuns`, `Stage`,
   `RoomForGraph`, `Count`, with model-neutral names for `Dsv4Path` and
   `Dsv4GraphStats`. About 170 lines; low risk. It is also where a
   backend without recorded work falls back to launch by launch.
2. **`LiveState`**: the state's mapped regions and spill sources,
   `RegisterState`, `Clear`, `ReadState` and the state half of
   `CheckPlaces`. About 110 lines; keeps the extent order (state before
   weights) and the slot numbering across target and drafter.
3. **`PlanCache<Planned>`** and one `PlaceAndPlan`. About 150 lines.
4. **`RunnerResources`**: the cuBLAS workspace, launch context, registry,
   pinned staging, `ReleaseMapped` with the node's `Joined`, place pinning,
   and the `Error`/`Pointer`/`Address`/`Round` helpers each file repeats.
   About 150 lines over seven files.
5. **DeepSeek onto `PagedWeights`**, once it takes a host-memory group and
   per-extent tags. About 210 lines; medium risk.
6. **`VerifySnapshot`**: saved ranges, `Accept`, `Rollback`, `Settle` and
   the quarantine, with hooks for Qwen3.8's commit kernel and pending
   rows. About 110 lines; medium risk (subtle semantics).
7. **`LlmServed<Runner>`** for the serving adapters. About 80 lines.

Each is a refactor with the harnesses' greedy, forced-rejection and swap
checks as its guard; none is done here. Of the small defects the audit
found on the way, the two that differed from Qwen3.8 are fixed (DeepSeek
sized its input staging without `Stage`'s rounding, and its `Release`
closed `spill_fd_` without resetting it); left: `qwen38_runner.h`
includes `dsv4_runner.h` only for the path and statistics types, and
`paged_weights.cc` includes it for `LayOutSlab`.

## Distribution

D-098: the `.deb` is primary (systemd, cgroup-delegated jobs, io_uring and
`O_DIRECT`, unlimited memlock); an OCI image built from the same `.deb` is
secondary, run with a seccomp profile that allows `io_uring_setup`,
`io_uring_enter` and `io_uring_register` (Docker's default blocks them since
25.0), `--ulimit memlock=-1:-1`, `--gpus all` through NVIDIA's container
toolkit, and the storage roles bind-mounted from the host's filesystem.
Homebrew or `.pkg` and winget or MSI come with their ports.
