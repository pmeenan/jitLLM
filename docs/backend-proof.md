<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Early backend integration proof

D-028, D-051 and D-052 require the Qwen2.5-0.5B-Instruct FP16 control and
both small EXL3 fixtures to run natively on jitLLM-owned memory before the
internal operation contract and executable layout are settled. Under D-053,
jitLLM owns dispatch: GGML- and ExLlamaV3-derived kernels, and later other
sources or our own, are build-time implementations of operations, selected
per plan. This is the M0 scope for that proof, which M2 executes alongside the resource core. It
fixes stages, numerical oracles, cases and evidence, and records what the
pinned sources imply for them. It is not an implementation, a measurement or
a support claim. The artifact encoding is D-056's [v0 format](artifact-format.md);
sources are acquired under D-057's [source-dependency mechanism](source-dependencies.md),
and the [retained-backing comparison](#retained-backing-comparison) stays a
separate plan item; the proof
consumes or hosts them.

## What the proof settles

| Question | Settled by | Feeds |
| --- | --- | --- |
| How GGML-derived operations run under jitLLM dispatch, per operation: GGML's launchers behind a jitLLM-supplied context, or lifted kernels behind owned launchers | P1–P2, BP-A/BP-L cases | D-053 integration record and build-time patch set |
| Coexistence and swapping: several implementations of one operation, and several kernel sources, in one build and process, selected per plan | P1, P3, BP-S cases | Operation contract and implementation registry |
| Dispatch overhead against upstream's captured decode | BP-F4 | Whether M3 parity needs graph capture with a relocation proof |
| The operation contract: dependencies, workspace, streams/fences, captured pointers, backend allocations, errors (ideation §10) | All stages | Decision entry at M2 close, before M3 builds on it |
| Executable-layout constraints: alignment, padding, kernel-readable ranges, tile rules | P2–P4 against D-056's v0 encoding | Validates or amends the experimental artifact |
| Phase envelopes and fixed runtime overhead `F` for the declared profiles | P6 | D-050 admission numbers for M2/M3 |
| Whether actual kernels regress on host VMM | BP-F1 | D-034 reopen check |
| EXL3 per-kernel time and workspace parity with upstream | BP-F2 | D-052 M2 gate |

The proof is not complete with a loader, one matrix multiply, an external
reference process or a fake backend. It exercises the M2 catalog, lease and
D-048 completion code on real providers. It does not implement a pager only
for the proof.

## Entry conditions and ordering

- **M1 delivered:** SDK and toolchain (D-032/D-049), CMake presets, the
  CPU-only guardrail build, and the question-7 mechanism supplying the pinned
  GGML subset and the selected ExLlamaV3 files. That mechanism also supplies
  any patch as a reviewed file, with hashes and notices.
- **P0/P1 need no artifacts.** They start once M1 builds. P2 onward runs from
  prepared artifacts in D-056's experimental v0 encoding. The M0 layout
  study built and verified all three fixtures in it. A proof-only file format does
  not satisfy "from prepared artifacts".
- **P4/P5 run on the M2 resource core:** catalog, leases, storage and device
  services. They validate the code M3 builds on.
- **Before evaluating any native result**, run the held-out trajectories on
  the references, record numerical profiles and cross-implementation bounds
  from reference controls ([first-slice.md](first-slice.md),
  [exl3-bringup.md](exl3-bringup.md)), and freeze the performance protocol.
  The owner approves thresholds before native output is seen. A bound set
  after a failure is not acceptance.
- **The GEMV provenance gate below closes before that kernel is ported.** The
  rest of the EXL3 closure does not wait for it.

## Pinned inputs

| Item | Identity |
| --- | --- |
| FP16 fixture | D-051 official GGUF; identities in [first-slice.md](first-slice.md) |
| EXL3 fixtures | D-052 4.0 bpw and mixed 4.5 bpw; identities in [exl3-bringup.md](exl3-bringup.md) |
| GGML source | `ggml/` subtree of llama.cpp [`b29c606e2`](https://github.com/ggml-org/llama.cpp/tree/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml), root MIT; compiled into jitLLM's build under owned dispatch (D-053) |
| EXL3 kernels | ExLlamaV3 [`6b84a21b`](https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279), root MIT; the selected closure below |
| llama.cpp reference | Digest-pinned image: GNU 14.2.0, cudart 13.3.29, cuBLAS 13.5.1.27 ([pins](experiments/first-slice/pins.json)) |
| ExLlamaV3 reference | PyTorch 2.14.0+cu130, NVCC 13.0.88 `-O3 --use_fast_math`, ARM host-helper patch ([report](experiments/exl3-reference/README.md)) |
| Native toolchain | D-032: LLVM 22.1.8, NVCC 13.4.92, `sm_121`, C++23 |
| Hosts | Workstation: CPU-only, fake-backend and guardrail builds. `spark`: all CUDA, VMM, I/O and timing stages. `spark-b`: repeat evidence |

## Source findings at the pins

Read-only source inspection on 2026-09-22. These are facts about the code,
not measurements. The proof confirms each on Spark before relying on it.
The GGML backend-runtime findings are why D-053 moves dispatch into jitLLM.
They also list what adapted launchers must not inherit.

### GGML (llama.cpp `b29c606e2`)

- **Tensors over jitLLM memory.** Launchers dereference `tensor->data`, but
  some compute paths query `src->buffer` (usage and allocation size), so
  tensor descriptors still need a buffer object. The CUDA backend has no
  buffer-from-pointer entry point ([`buffer_from_host_ptr = NULL`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L5590)).
  The workaround is a wrapper buffer: `ggml_backend_buffer_init` with the
  CUDA *buffer type*, a jitLLM interface and a jitLLM address range. That
  buffer passes the always-on asynchronous-transfer asserts, which compare
  buffer-type pointers ([example](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L2449)),
  and it passes `supports_buft`. GGML casts a buffer's context only inside
  CUDA's own buffer functions. Place tensors with `ggml_backend_tensor_alloc`.
  Nothing in `ggml-cuda` queries pointer attributes or needs `cudaMalloc`
  memory. `ggml_backend_buffer_init` and the event layout are declared only in
  the uninstalled `ggml-backend-impl.h`: a pinned internal interface that may
  change on upgrade. GGML's own CUDA buffer type allocates with
  [`cudaMalloc`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L883-L900)
  and must never hold model data.
- **Activations.** If jitLLM reuses GGML's graph allocator to plan activation
  offsets: `ggml_gallocr` obtains memory only through its buffer type's
  `alloc_buffer` and does not check the returned buffer's type. A jitLLM
  buffer type can return wrapper buffers, putting the compute buffer in
  charged workspace. `ggml_gallocr_reserve_n_size` sizes it without
  allocating; `ggml_gallocr_reserve` allocates. Owned dispatch uses neither
  `ggml_backend_sched` nor GGML's CUDA graph-compute loop.
- **Streams.** GGML's backend creates and owns non-blocking streams and
  exposes no public way to supply or read one. Stock buffer set/get/clear
  functions use `cudaStreamPerThread` plus a synchronization and are
  unordered with compute. Owned dispatch does not use them; it supplies its
  stream through the launcher context below.
- **Operation launchers.** `ggml_cuda_op_*` functions take a
  `ggml_backend_cuda_context` and draw their stream, scratch and cuBLAS
  handle from it. Scratch goes through `ggml_cuda_pool`, an
  [abstract allocate/free interface](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/common.cuh#L1207-L1212).
  The stream, handle and pool members are public and are created only when
  absent, so jitLLM can supply its own. Pre-setting the cuBLAS handle also
  skips GGML's workspace allocation and handle setup (stream binding,
  `CUBLAS_TF32_TENSOR_OP_MATH`, 32 MiB workspace); a supplied handle
  reproduces that setup or records its own in the numerical plan.
  Matrix-multiply routing among MMVF, MMF, MMVQ, MMQ and cuBLAS
  ([`ggml_cuda_mul_mat`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L1823-L1877))
  and the cuBLAS path are `static` in `ggml-cuda.cu`; only per-kernel entry
  points are external. That file also defines symbols every launcher needs
  (`ggml_cuda_info`, `ggml_cuda_error`, the pool factory, the context
  destructor), and `ggml_cuda_info()` runs device initialization lazily on
  the first launcher call. Environment switches read at launch
  (`GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_PDL`) change numerics or launch
  attributes; the build fixes them.

  The context adapter needs build-time changes for:
  - the context [destructor](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L700-L721), which
    destroys whatever streams, handles, workspaces and pools it holds;
  - GGML's device initialization, which carries the GB10 device-flag side
    effect below;
  - recoverable failures: `CUDA_CHECK` and `CUBLAS_CHECK` still call
    [`ggml_cuda_error`, which aborts](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L100-L108).
    Pool consumers also use allocation results without checking for null.
    Supplying a bounded pool alone is insufficient: preflight its complete
    scratch requirement before launch and adapt error propagation through
    the selected launchers, or use an owned launcher. Failed submissions
    retain already-submitted work's resources until D-048 retirement; no
    exception may cross a C ABI.

  `ggml_cuda_pool_alloc` releases its pool allocation as the host launcher
  returns, while device work can still be in flight. The adapter may reuse
  those offsets in stream order, but must retain the workspace's backing
  and charge until all consumers retire. Pool `free` is not catalog release.

  Fusion decisions live in GGML's graph-compute loop (`ggml_cuda_can_fuse`),
  which owned dispatch replaces. Fused launchers such as
  `ggml_cuda_op_rms_norm_fused` are separate implementations that the plan
  may choose. Runtime-API launchers bind to the current runtime context; P1
  checks that against jitLLM's context.
- **Hidden allocations in GGML's own backend.** None of these can be capped,
  pre-sized or queried through an API. Under owned dispatch the pool becomes
  a jitLLM `ggml_cuda_pool` over declared workspace, and the cuBLAS handle and
  workspace are jitLLM's. The usage analysis after this list sizes that
  workspace.
  - **Scratch pool.** One per device and stream. By default it reserves
    [32 GiB of virtual address space](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L536)
    and grows with device-location `cuMemCreate`. It never shrinks, and
    [growth failure aborts the process](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L580-L597).
    `GGML_CUDA_NO_VMM` selects a `cudaMalloc` pool instead.
  - **cuBLAS.** A lazy handle per stream plus a `cudaMalloc` workspace:
    [32 MiB at compute capability 9.0 and above](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/common.cuh#L1540-L1556),
    which includes GB10. cuBLAS's own internal memory comes on top.

  Among the dense Qwen2 operations, only matrix multiplications routed to
  cuBLAS use the pool. With F16 weights these are the ones with more than
  16 activation columns: the pool holds an F16 copy of the activations and
  an F16 output temporary. Smaller batches use MMVF/MMF kernels. With Flash
  Attention off, as in the reference, the KQ and KQV products follow the
  same routing. llama.cpp's graph marks KQ `GGML_PREC_F32`, so on the cuBLAS
  path the pool also holds an F32 copy of the K view, and KQV an F16 copy
  of the scores. Pool peaks therefore grow with context and prefill chunk.
- **Process side effect on GB10.** Device initialization on compute
  capability 12.1 calls
  [`cudaSetDeviceFlags(cudaDeviceScheduleSpin)`](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/ggml-cuda.cu#L366-L369),
  a process-wide synchronization policy that also affects jitLLM's own waits.
  D-053 patches it out.
- **Duplication hazards.** Two settings convert whole F16 weights to F32 in
  the pool, which never shrinks: `GGML_PREC_F32` on a weight multiply, and
  `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32`. CPU extra buffer types (repack, AMX)
  duplicate weights. KleidiAI makes transient per-call copies. Standalone
  defaults keep CUDA graphs and llamafile SGEMM off, whereas llama.cpp's own
  build enables both. A standalone build must still set the NCCL option off.
- **CUDA graphs**, if ever enabled, capture virtual addresses and recapture
  when node addresses or shapes change. Remapping physical backing behind
  the same address is invisible to GGML. Fusion is on by default and is part
  of the numerical plan (first-slice.md, RE-010). GGML's CUDA backend has
  **no custom operation**, so an EXL3 linear cannot be a node inside a GGML
  CUDA graph. Owned dispatch sidesteps this by sequencing both kinds of
  launch on one stream.
- **CPU backend.** `ggml_graph_plan`/`ggml_graph_compute` accept a
  caller-supplied work buffer and thread pool and compute on `tensor->data`
  only. Host VMM is mapped with host access
  ([I/O report](experiments/io-path/README.md)), so CPU diagnostics can run on
  the same backing. `ggml_init` accepts a caller-supplied metadata buffer.
  Spark's CPU heap draws on the same physical budget. This graph-compute
  path is a diagnostic, not a serving implementation under D-053.
- **Tied weights.** One tensor can feed both the embedding lookup and the
  output multiply. No placement restriction applies.

### ExLlamaV3 (`6b84a21b`)

- **Kernel closure for the two fixtures** (`mcg`, K=4/5/6/8, full-length
  side vectors):
  - `exl3_gemm_kernel` for each K, with FP16 and FP32 outputs and tile
    shapes 1–3. Shape 4 never fits these widths.
  - The K=4 GEMV kernel, the only rate these fixtures use in the GEMV range.
  - The fused gate/up `exl3_mgemm_kernel`, which upstream uses at up to
    32 rows.
  - `reconstruct` and `reconstruct_had` for each K, the 128-point Hadamard
    kernels, cuBLAS `GemmEx` with FP32 compute for reconstructed prefill,
    and an FP16 bias add.

  Upstream's Qwen2 writes [FP32 outputs for `o_proj`, gate, up and down](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/architecture/llama.py#L93-L112).
  Output dtypes are part of the numerical plan. Every Qwen2.5-0.5B dimension
  is a multiple of 128, so no padding fallback applies. Other codebooks,
  fractional rates, MoE, tensor-parallel and packed-sign paths are outside
  this closure.
- **Separation.** Kernels are templates over raw pointers and integers, in
  headers with no ATen types. The host wrappers use ATen tensors, the current
  PyTorch stream and blas handle, `TORCH_CHECK` and `at::empty`, so they are
  rewritten, not ported. The reconstruct and Hadamard kernels share `.cu`
  files with their wrappers and must be split. Kernel headers rely on the
  including file for some macros. Upstream's
  [`cuda_check` calls `exit`](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/util.cuh#L92-L100);
  native wrappers return errors.
- **Extension state.** A per-device context holds 4,202,760 B of lock slots
  and a 16 MiB workspace that serves as the cuBLAS workspace. Kernels require
  zeroed lock slots and reset them on normal exit. One lock area is not safe
  for concurrent GEMMs on different streams: serialize, or give each admitted
  stream its own zeroed area.
- **Autotuning.** It times candidate tile shapes and grids with live buffers,
  allocates an L2-thrash buffer of up to twice the L2 size through the
  PyTorch allocator, synchronizes, and appends results to a
  cache file under `~/.cache/exllamav3`. The cache key omits the driver,
  toolkit and build identities. The chosen grid changes the split-K
  partition and thus FP16 rounding. The native plan therefore records a
  fixed shape and grid per case, obtained by an explicit tuning job outside
  the catalog lock. Forced shape/grid arguments give upstream the same plan
  for comparison. The reference's tuned choices were not recorded; P0
  records them.
- **Graphs.** Upstream decode captures attention and gated-MLP blocks.
  Capture forces a device-wide synchronization and fixes trellis, side
  vector, scratch, lock and pointer-table addresses at capture. Nothing
  invalidates a captured graph when weights move. The native path rejects
  capture until a relocation proof exists (exl3-bringup.md).
- **Alignment.** Source-derived minimums are 16 B for the trellis, the
  transformed input and FP32 outputs, and 8 B for side vectors, activations
  and FP16 outputs. Upstream packs small tensors at 256 B. Use at least
  256 B inside shared extents unless the proof shows cuBLAS and kernel
  selection are unchanged at smaller alignment.
- **Numerics.** Accumulation is FP32 on GB10, and split-K partials are
  combined in a fixed lock-ordered sequence. The GEMV kernel accumulates in
  FP16 and folds into FP32. There are no floating-point atomics. For a fixed
  plan, grid and build, the dense path is repeatable. Across builds, results
  can differ with compile flags (`--use_fast_math`), tuning choices and cuBLAS
  heuristics.
- **GEMV provenance gate.** The K=4 GEMV kernel header describes a
  ["QTIP-style structure"](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh#L3-L4)
  and cites QTIP's `qtip-kernels/src/inference.cu`. QTIP's repository is
  GPL-3.0 ([licensing.md](licensing.md#early-exl3-companion-d-052)). Whether
  any of its code was incorporated is unresolved. Resolve it with the owner
  before porting that kernel, or a file that includes it, into a
  core-eligible module. Until then, the native plan runs the EXL3 GEMM kernel
  wherever upstream would select GEMV (m ≤ 8). Upstream supports that
  configuration (`EXL3_GEMV=0`), and all other dispatch paths are unchanged.
  Measure the gap to upstream's GEMV-enabled normal
  configuration. Under D-052 a regression there needs a fix or an explicit
  owner-approved tradeoff; it is not a pass.

## Dispatch and implementations (D-053)

jitLLM's dispatcher launches every operation on a jitLLM stream, with
jitLLM workspace and library handles. GGML's backend runtime does not execute
model work.

**GGML-derived operations.** For each operation, the proof chooses between
two approaches and records the choice:

- **K-C (context adapter).** Call GGML's CUDA operation launchers with a
  jitLLM-populated context: jitLLM's stream, its cuBLAS handle and workspace,
  and a `ggml_cuda_pool` over declared, charged workspace. Build-time patches
  cover context ownership on destruction, the GB10 device flag and error
  propagation through the selected launchers. Scratch sizing is checked
  before submission, and pool reuse preserves completion-owned lifetimes.
  Tensor descriptors use wrapper buffers over jitLLM memory. Prefer this
  where it holds: it reuses upstream launch selection with the least new code.
  For matrix multiplication that selection is `static` in `ggml-cuda.cu`, so
  reuse needs a linkage patch there or a recorded jitLLM copy.
- **K-L (lifted kernel).** Call the device kernel from a jitLLM launcher.
  Use this when a launcher needs more than the adapter or small patches
  provide. The launcher keeps upstream's launch-parameter selection (D-013)
  or records the difference in implementation identity.

Either way, operation scratch must fit declared workspace (BP-A1/A2).
Library handles and unavoidable driver/library allocations are separately
bounded and charged; supplying a cuBLAS workspace does not account for all
of cuBLAS's internal memory. Workspace exhaustion must return an error,
never abort (BP-V2). Fusion is an explicit plan choice among fused and
unfused implementations. GGML's graph-time fusion checks do not run.

**EXL3-derived operations.** Lifted kernels behind jitLLM launchers
(exl3-bringup.md). The dispatcher sequences them and GGML-derived operations
on the same stream. No segment boundaries or cross-stream events are needed.

**Coexistence and swapping** are proof obligations, not later features.

- One build holds at least two implementations of one operation. The plan
  selects between them, and each passes its own reference comparison and
  envelope.
  - Natural first case: EXL3 GEMM versus GEMV at m ≤ 8, once the GEMV
    provenance gate clears.
  - Until then: GGML's fused versus unfused RMSNorm, inside the fused and
    unfused plans that have bridge fusion arms as oracles, or its MMF versus
    cuBLAS matrix-multiply paths at a shared shape.
- Implementation identity is part of plan identity. The implementation
  registry resolves every operation or rejects the plan; nothing is silently
  substituted.
- The FP16 and EXL3 models, using different kernel sources, are resident and
  run alternately in one process.

**Dispatch overhead.** Upstream decodes these fixtures at about 3.4–3.7 ms per
token with captured blocks ([report](experiments/exl3-reference/README.md)),
and a decode token runs 169 linear layers plus norm, RoPE and attention
launches. Host cost per launch can
therefore threaten M3 parity even when M2 kernel parity passes. BP-F4
measures it early. jitLLM-owned graph capture remains possible only under
the captured-pointer relocation rules.

## Numerical oracles

Each rung isolates one source of difference. A difference is localized, not
absorbed into a tolerance.

1. **Reference.** The pinned external runs, as already recorded.
2. **Toolchain bridge.** The pinned llama.cpp source and the same reference
   harness, built with the jitLLM SDK. This separates compiler, CUDA and
   cuBLAS effects from integration. For EXL3, the equivalent bridge compiles
   the upstream kernel sources with jitLLM's NVCC and upstream's flags, and
   runs them on captured inputs with forced shape and grid.
3. **Native dispatch, conventional memory.** jitLLM's dispatcher on
   `cudaMalloc` memory. Expect bit-identical logits to the bridge when the
   plan reproduces the bridge's kernel selection, launch parameters, fusion,
   batch splits, cuBLAS paths and handle setup. Compare an unfused plan with
   a fusion-disabled bridge arm, the counterpart of the recorded reference
   arm (first-slice.md).
4. **Native, jitLLM host VMM.** Expect results identical to rung 3.
5. **Native after eviction, restoration or relocation.** Must be identical to
   rung 4.

The cross-implementation bounds are declared before native evaluation:
native versus the image reference, and full-model EXL3 versus ExLlamaV3,
whose norm, RoPE and attention run in PyTorch or Triton. Localize EXL3
error with per-layer teacher-forced comparisons at the declared dtypes.
Individual packed linears must be byte-identical to upstream at the same
plan and inputs, or the bridge must explain the difference. Instrumented
observation points can change CUDA plans (RE-010). Every instrumented run
has an uninstrumented control.

**Inputs.**

- **FP16:** the existing 76-token trajectory, plus rows either side of the
  16/17-column MMF/cuBLAS boundary and a larger prefill chunk.
- **EXL3:** the existing prefixes and the declared held-out trajectory. Rows
  1, 8, 9, 16, 32, 33, 144, 145, 1,023 and 1,024 cover GEMV, the fused
  gate/up range, packed/reconstruct and fused-reconstruct boundaries.

P0 declares the finite context and prefill-chunk profile for each fixture.
First-slice context and envelope limits apply.

## Stages

| Stage | Needs | Exit evidence |
| --- | --- | --- |
| **P0** Bridges and controls | M1 build | Toolchain bridges run, FP16 with fusion on and off. Held-out trajectories run on both references. The reference EXL3 tuned shapes and grids are decoded. Numerical profiles, bounds and the performance protocol are frozen and owner-approved |
| **P1** Substrate probes | M1; no artifacts | GGML launchers under a jitLLM context (K-C): stream, handle and pool injection, runtime-context binding, patched destructor and device flag. Values and kernel times on host VMM versus `cudaMalloc`. Allocation census. One native EXL3 linear byte-equal to upstream at a forced plan, including alignment probes. Two implementations of one operation selected by plan. Per-launch host cost |
| **P2** Resident FP16 | Question-5 encoding and importer; M2 catalog | Prepared-artifact execution on host VMM; oracle rungs 3–4; BP-A cases |
| **P3** Resident EXL3 | P2 infrastructure; GEMV gate or GEMM-only plan | Both fixtures; per-linear and full-model oracles; BP-F2 kernel parity |
| **P4** Paging | M2 leases and storage service | BP-P cases on both representations |
| **P5** Lifetime and failure | D-048 completion services | BP-L and BP-V cases on real providers |
| **P6** Envelopes and contract | P2–P5 | Complete accounting; phase envelopes and `F` per profile; contract draft; per-operation K-C/K-L choices, registry and D-034/D-052/D-053 status recorded |

The [retained-backing comparison](#retained-backing-comparison) runs on the
P4 harness, but its acceptance stays a separate plan item.

## Case matrix

Invariant numbers refer to architecture.md's pager invariants. Fake-backend
and CPU-only cases run on the workstation; everything else runs on `spark`.

**Backing and accounting**

- **BP-A1:** Every pointer a launch dereferences lies in a cataloged jitLLM
  range or a cataloged backend allocation. Reconcile a complete run's CUDA
  allocation census with the catalog, distinguishing virtual reservations,
  physical backing and aliases. Use CUPTI callbacks or an equivalent whose
  coverage is verified with controls for direct driver allocations, runtime
  allocations and VMM backing. CUDA callbacks do not observe ordinary host
  `malloc`/`new`/`mmap`: instrument those separately, including allocator-held
  capacity, and validate that coverage with a host-allocation control. Charge
  opaque driver/library overhead conservatively and reconcile remaining
  physical-memory differences in BP-A5; a CUDA-only trace is not a complete
  Spark memory census (invariant 5, D-006/D-050).
- **BP-A2:** Kernel scratch comes only from declared workspace. That
  covers GGML launchers' pool requests, cuBLAS workspace, EXL3 locks and
  workspace, and tuning allocations. Handles and unavoidable driver/library
  allocations are separately bounded under BP-A1. Each charge belongs in
  `F` or in the phase's peak working set `E`; no allocation escapes the
  envelope, including lazy library growth after warm-up.
- **BP-A3:** No hidden weight duplication: no F32 weight conversion, no CPU
  extra buffer types and no permanent FP16 shadow. Transient reconstruction
  is charged to the phase peak.
- **BP-A4:** After retirement and unmap, no backend object references
  jitLLM backing. Submitting a stale graph or pointer table is rejected
  before launch. A negative control faults deterministically instead of
  reading stale memory (invariant 1).
- **BP-A5:** Node-level observations (`/proc/meminfo`, cgroup, CUDA memory
  queries) are reconciled with the catalog. Record what each one does and
  does not cover on unified memory.

**Numerics**

- **BP-N1:** The toolchain bridges versus the references; differences
  recorded.
- **BP-N2:** Native dispatch on conventional memory versus the bridge.
- **BP-N3:** Host VMM versus conventional memory.
- **BP-N4:** Native versus the reference within the declared bound, on
  held-out inputs.
- **BP-N5:** Per-linear EXL3 byte equality at a forced plan across the row
  sweep, together with a decode check of the reconstructed weights.
- **BP-N6:** Full-model EXL3 within the declared bound, with per-layer
  localization.
- **BP-N7:** CPU diagnostics on the same host-VMM backing versus the
  reference's CPU path.

**Paging (weights and state)**

- **BP-P1:** Evict all weights and restore them with coalesced chunk-closure
  direct reads. Logits are bit-identical to resident.
- **BP-P2:** Partial eviction of one layer, of the trellis only, of side
  vectors or biases only, of a shared small-tensor chunk, of a padded tail
  and of a tensor crossing a chunk boundary. An incomplete closure refuses
  launch (invariants 1–2).
- **BP-P3:** Views of the FP16 tied embedding and output compute
  identically whether storage is duplicated or shared. The EXL3 head is a
  different representation and is never deduplicated with the embedding.
- **BP-P4:** Evict and restore KV state after a prefill chunk and in
  mid-decode. The continuation is bit-identical. Test with poisoned live
  state (invariant 4).
- **BP-P5:** Relocate to different addresses or backing. Tensor descriptors,
  EXL3 pointer tables and any captured graphs are rebuilt or revalidated, and
  stale ones are rejected.
- **BP-P6:** Cold and warm restore times and bytes read are reported, not
  gated.

**Lifetime and cancellation**

- **BP-L1:** Cancel with GGML or EXL3 work in flight. Submission stops, and
  leases hold until the recorded completion (invariant 2).
- **BP-L2:** Cancel during a page-in, including the `io_uring` cancellation
  race. A late DMA cannot corrupt a reassigned extent (invariant 3).
- **BP-L3:** Cancel between reconstruction and the GEMM. The scratch
  allowance persists until retirement.
- **BP-L4:** D-048 event permutations with real providers: completion before
  acceptance, duplicate completion, and unknown outcome leading to
  quarantine (invariant 8).
- **BP-L5:** Shared EXL3 locks and workspace across plans or streams
  serialize or reject.
- **BP-L6:** Registered I/O buffers over host-VMM extents are unregistered
  before unmap or reassignment.

**Validation and failure**

- **BP-V1:** Importer negative cases, both FP16 and the EXL3 list in
  exl3-bringup.md: shapes, types, ranges, rate/codebook mismatches,
  truncation and hashes.
- **BP-V2:** Inject failures: VMM create or map failure, cuBLAS handle
  failure and workspace exhaustion. Each returns an error and gets a complete
  unwind with no leaked charge. No path aborts the process (invariant 7).
- **BP-V3:** A tight budget admits the largest reconstruction phase or
  rejects the plan as impossible.

**Performance** (under the frozen protocol, with the reference repeated
beside the candidate)

- **BP-F1:** GGML kernel times on host VMM versus `cudaMalloc` memory (the
  D-034 check).
- **BP-F2:** EXL3 kernel parity on the 176 declared cases (the D-052 M2
  gate).
- **BP-F3:** Resident full-model timings for all three fixtures, reported
  for M3's gates.
- **BP-F4:** Dispatch overhead: host submission per launch and per token,
  against upstream's captured decode.

**Coexistence and swapping**

- **BP-S1:** Two implementations of one operation in one build, selected by
  plan. Each passes its own oracle comparison and envelope.
- **BP-S2:** Changing an operation's implementation changes plan identity.
  A plan built for the old identity is rejected before launch. Retained state
  from the old plan is reused only with validated compatibility; otherwise it
  is recomputed.
- **BP-S3:** Models using different kernel sources are resident and run
  alternately in one process, with correct accounting of shared workspace.
- **BP-S4:** In the fake backend, a build profile lacking an eligible
  implementation reports the model as unsupported. Nothing is substituted.

D-050's M2 adversarial rows that need real allocation or retirement map to
BP-A1–A2, BP-P2–P3, BP-L1–L6, BP-V2 and BP-V3. The rest stay fake-backend
tests.

## Retained-backing comparison

This is the owner's D-035 follow-up. M0 deferred it to M2 because it needs
the M2 resource core and the P4 harness. It is not a BP case: the proof can
pass while it is still open. The internal contract is not settled, and M2
does not close, until D-033 is explicitly retained or amended. D-033 stays
the baseline unless the predeclared criteria below are met. The comparison
has two independent parts.

**(a) Physical backing.** Compare D-033's independent 2 MiB handles with
larger persistently mapped slabs, including 1 GiB, using software
suballocation and real executable tensor views. Both designs must keep
ordinary paging free of avoidable create/release cycles. Measure:

- warm reuse, plus remapping and registration costs where a design needs
  them;
- fragmentation;
- growing and shrinking the shared pool;
- concurrent compute;
- end-to-end restore latency.

Any design that changes addresses must pass BP-P5, BP-L2 and BP-L6, plus
checks for aliases and captured pointers. Both designs keep useful contents
within budget. Disk transfer size is independent of either design, and
D-056's layout serves both.

Evaluate these slab-hole policies:

- contiguous-run eviction;
- size classes;
- a hybrid;
- the owner's activity-sorted compaction. This is a completion-safe
  relocation that competes with decode for memory bandwidth
  ([artifact-format.md](artifact-format.md)).

**(b) Checkpoint-batch submission.** For batches that mix small and bulk
transfers, compare serial with bounded asynchronous submission. Measure the
effect of scheduling order and depth, the time to the last required
completion, and consumer stalls. The M0 I/O spike did not measure these
mixes. M4 extends this to simultaneous demand reads and state write-back,
with dependency safety and bounded queues.

**Inputs and open prerequisites.**

- The measured expert-closure sizes (about 1.8–10.9 MB) and the per-model
  memory padding of per-chunk handles (3.49–10.87%) come from D-056's
  [worked examples](artifact-format.md#worked-examples-measured-plans-of-real-files).
- **A cross-model swap trace that leaves holes in slabs does not exist
  yet.** Build it before (a) runs, as an eviction/restore sequence across
  several library models. Its extents are synthetic, drawn from D-056's
  measured group and closure sizes. The reference A→B→A frozen trace and
  the paging-feasibility routing captures are raw material for it. Like
  other replay inputs, the trace stays outside Git; its generator and
  verified identity are recorded.
- **The proof fixtures cannot exercise realistic fragmentation.** The FP16
  artifact's groups total 988,208,640 bytes, under one 1 GiB slab, the EXL3
  fixtures are smaller, and none has experts. Compare hole policies by
  replaying the trace on the fake backend. Measure costs on `spark` with the
  M2 memory manager and synthetic extents. Report end-to-end restore on the
  real fixtures. This uses synthetic extents only,
  not MoE execution, which stays in M5.
- **Criteria before measurement.** As in P0, the owner approves the metrics,
  budgets, trace identity and the margins at which a slab or hybrid design
  would amend D-033. This happens before any candidate result is seen.
- **Expert compaction is partial at M2.** Relocation without VA remapping
  assumes pointer-table dispatch. For experts, the M5 GGML proof decides
  that. M2 evaluates compaction for dense groups, and the expert
  case is completed in M5.

## Evidence and limits

Record aggregates in an `experiments/backend-proof/` report:

- identities and the per-operation implementation choices;
- the allocation census with each allocation's category;
- per-profile envelopes and `F`;
- oracle results per rung;
- timing statistics under the frozen rule.

Raw logits, traces and logs stay outside Git. The operation contract,
implementation registry and patch set become a decision entry when M2
closes.

Out of scope: MoE and routed closures (M5), sharding (M6), multimodal
components, tokenizer/sampling/API (M3), spill format and retention (M4),
other EXL3 variants, other GPUs, and CUDA graph capture beyond explicit
rejection or a completed relocation proof. Passing this proof establishes
the small dense fixtures only, not model support.
