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
  CPU-only guardrail build, and the question-7 mechanism, proven on
  GoogleTest. P0 uses it to admit the pinned GGML subset and the selected
  ExLlamaV3 files, supplying any patch as a reviewed file with hashes and
  notices (plan.md).
- **P0/P1 need no artifacts.** They start once M1 builds. P2 onward runs from
  prepared artifacts in D-056's experimental v0 encoding. The M0 layout
  study built and verified all three fixtures in it. A proof-only file format does
  not satisfy "from prepared artifacts".
- **P4/P5 run on the M2 resource core:** catalog, leases, storage and device
  services. They validate the code M3 builds on.
- **Before evaluating a native result**, run the held-out trajectories on
  the references, record numerical profiles and cross-implementation bounds
  from reference controls ([first-slice.md](first-slice.md),
  [exl3-bringup.md](exl3-bringup.md)), and freeze the performance protocol.
  The owner approves each threshold before any native output it governs
  is seen: approval may come in parts ([P0 declarations](#p0-declarations)),
  but never after. A bound set after a failure is not acceptance.
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
| Native toolchain | D-032: LLVM 22.1.8, NVCC 13.4.92, `sm_121`, C++23; cuBLAS 13.8.0.4, linked dynamically (D-076) |
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

## P0 declarations

**Status.** Approval is by part, each before the native output it governs.

- **Approved by the owner on 2026-09-26, in force:**
  - the numerical profiles;
  - the Tier E items marked *approved* below: the FP16 exactness gate with
    its recorded-plan match, rungs 4 and 5, and the EXL3 packed linears
    with their reconstructed weights.

  These are all that native FP16 work (P1, P2) needs.
- **Approved by the owner on 2026-09-26, in force (EXL3):**
  - Tier C: the rule, the bounds in `tierc.json` and their calibration;
  - the reconstruction-path Tier E item, scoped to the recorded chunk
    sizes;
  - the kernel-timing rule's statistic and decision procedure
    (`timing_protocol.py` at `c05fd2dd…`);
  - the native EXL3 operation plan and its record, `exl3-op-plan.json`;
  - the operation-level Tier E item;
  - the persistent-workspace limit.
- **Deferred to P3 entry.** These are approved once the ExLlamaV3 port
  exists, and before any native EXL3 timing or memory result is seen:
  - *BP-F2's reference arm.* Four things are open:
    - add the fused gate/up kernel (`exl3_mgemm_kernel`) to the timed
      cases;
    - time ExLlamaV3's bias add in the reference, not PyTorch's;
    - have one frozen tuning cache govern both the model plan and the
      timing cases;
    - check that the port's build reproduces the reference's SASS.

    The changed case set needs a new calibration and holdout under the
    approved rule. The NVCC 13.4.92 calibration and holdout below validate
    the rule's mechanics on the current cases.
  - *The EXL3 phase memory limits,* tightened against native's itemized
    buffer plan. The allowance below adds per-layer intermediates to a
    peak set when the logits are allocated, when those intermediates are
    already dead, so it is loose.
- **Still proposed, not yet ready:** the declared-departure contingency,
  the FP16 memory limits, BP-F1's calibration, and the retained-backing
  criteria.

Each part is approved before any native result it would judge is seen.
Five rounds of review and challenge shaped the EXL3 parts. Their basis is
in the report: the
[third pass](experiments/backend-proof-p0/README.md#exl3-third-pass-calibrating-the-full-model-bound),
the [GGML operation study](experiments/backend-proof-p0/README.md#ggml-operations-for-native-exl3),
the [pinning probe](experiments/backend-proof-p0/README.md#reconstruction-gemm-pinning)
and the [timing controls](experiments/backend-proof-p0/README.md#timing-controls).

The measured basis is the [P0 report](experiments/backend-proof-p0/README.md),
all of it reference runs on `spark`:

- both toolchain bridges;
- the EXL3 reference arms (three passes), an FP64 oracle and the Tier C
  calibration;
- the FP16 bridge's recorded executed plan;
- the GGML operation study, the cuBLASLt pinning probe and the kernel
  launch record;
- 20 timing sessions, plus one stopped unevaluated.

### Numerical profiles (approved)

| Profile | Fixture | Settings | Trajectories |
| --- | --- | --- | --- |
| FP16-F | FP16 GGUF | first-slice.md's settings: F16 K/V, one sequence, no flash attention, no CUDA graphs, fusion on | `control`: 76 tokens, context 512, batch 64, chunks 32 then 44 × 1, restore after 32. `heldout`: 577 IDs, context 1,024, batch 512, chunks 16, 17, 16 × 1, 512, 16 × 1, restore after 33 |
| FP16-U | FP16 GGUF | FP16-F with fusion off | as FP16-F |
| EXL3-G | 4.0 and 4.5 bpw | Upstream's optimized profile with GEMV off (`EXL3_GEMV=0`); the reconstruction GEMM pinned to cuBLAS (`EXL3_HGEMM_F16ACC=0`, what upstream's timing probe chooses on GB10); F16 cache of 4,096 tokens; the frozen GEMM-only tuning caches `tune-40-gemvoff` and `tune-45-gemvoff` | Prefixes of 32, 144, 145, 1,023 and 1,024 held-out IDs, every prefill row's logits, then 16 single-token steps |
| EXL3-O (reported only) | 4.0 and 4.5 bpw | EXL3-G with GEMV on, caches `tune-40` and `tune-45` | as EXL3-G |

The trajectories, the chunking and the harnesses are the P0 report's. The
held-out IDs have SHA-256 `6dd8da89…`. The caches' bytes and decoded choices
are in its `results.json`.

- **Context.** The proof covers FP16 trajectories of up to 577 tokens in a
  1,024-token context, and EXL3 up to 1,040 tokens in a 4,096-token cache.
  It covers nothing beyond, timing included.
- **GEMV.** EXL3-G is the native plan until the GEMV provenance gate
  closes. EXL3-O measures the gap.
- **Per-linear sweep (BP-N5).** Rows 1, 8, 9, 16, 32, 33, 144, 145, 1,023
  and 1,024, on every real projection of both fixtures. It runs at a forced
  plan, the same tile shape, block, SMs and concurrency on both sides,
  recorded per case.

### Tier E: exact

A difference in any of these is a defect, to be localized and fixed. It is
never bounded. Logits and restored storage must be bit-identical.

- **FP16, native against the bridge (approved)**, on both FP16 profiles and
  both trajectories (rung 3), and therefore against the image reference as
  well.
  - The native GGML kernels are built as the bridge's are, for `sm_121a`
    (GGML's CMake maps `121-real` to it), since the SASS hashes must match.
  - The native executed plan must first match the bridge's recorded plan
    (`fp16-plan.json`), per chunk shape. That means the ordered kernel
    sequence with each kernel's SASS hash, grid, block and shared memory;
    every cuBLAS call's parameters and resolved algorithm; and the handle's
    state: a 32 MiB workspace, TF32 math mode and the SM count.
  - The plan also has to reproduce the conditions the record lists:
    - the `n_kv` padding to 256 and the KV cell layout;
    - the MMVF/MMF and softmax variants chosen by `n_kv`;
    - the fusion gates that compare data ranges;
    - `get_rows`' alignment choice and 128-byte buffer alignment;
    - embedding lookup on the host;
    - the precision rules that fusion changes.
  - Only stream identity, addresses (subject to those alignment conditions)
    and the PDL launch attribute may differ.
  - Logits are compared only after the plans match.
- **A declared departure (proposed).** A native plan could depart from the
  recorded plan only with a bridge arm, built from a patch recorded in the
  report, that makes the same departure. Tier E would then hold against
  that arm. The challenge found this needs its own accuracy bound. Until
  it is approved, a native FP16 plan may not depart from the recorded plan.
- **Rung 4 against rung 3, and rung 5 against rung 4 (approved)**, for every profile
  and fixture, including every eviction, restore and relocation arm and
  every repeat.
- **EXL3 packed linears (up to 144 rows) (approved)** against upstream's kernel at the
  same forced plan and inputs, and reconstructed FP16 weights against
  upstream's reconstruction.
- **EXL3 reconstruction-path linears (145 rows and more) (approved
  2026-09-26).** These are compared against upstream running cuBLAS
  13.8.0.4 (the report's substitution arm), at the same forced plan and
  inputs.
  - *Scope:* the recorded chunk sizes of 145, 1,023 and 1,024 rows. Before
    native runs the reconstruction path at any other size (BP-F3's
    512-row prefill, or BP-F2's synthetic shapes), that size's GEMMs are
    recorded and pinned in the same way, from a reference-only run.
  - Upstream's executed plan is recorded in
    [`exl3-recon-plan.json`](experiments/backend-proof-p0/exl3-recon-plan.json):
    21 distinct cuBLASLt GEMMs per fixture, with their layouts, compute
    descriptor (`COMPUTE_32F`, 48 SMs targeted), heuristic preferences
    (16 MiB workspace limit, 16-byte alignment, the logged `implMask`) and
    resolved algorithms.
  - The native plan pins each GEMM's complete algorithm configuration, all
    nine cuBLASLt attributes, recorded in
    [`exl3-recon-pin.json`](experiments/backend-proof-p0/exl3-recon-pin.json).
    None uses split-K or a workspace.
  - The report's pinning probe replayed all 21 GEMMs on the SDK's cuBLAS
    13.8.0.4 and showed:
    - the heuristic resolves the algorithms the arm logged;
    - an algorithm rebuilt from its configuration alone is bit-identical to
      ExLlamaV3's legacy `cublasGemmEx` call;
    - so is the heuristic's own choice.
  - Reconstructed FP16 weights must equal upstream's, and each GEMM's
    output must equal the arm's bit for bit. The per-linear harness maps
    only cuBLAS 13.8.0.4.
  - Native's cuBLAS kernel names and grids must equal the arm's.
  - Under PyTorch's cuBLAS 13.1.1 the resolved algorithms differ for 19 of
    the 21 GEMMs, so the cuBLAS version is part of the plan.
  - No contingency is needed: pinning is shown to work. A future cuBLAS
    change requires a new record and approval.
- **GGML-derived operations inside the EXL3 plan (approved
  2026-09-26).**
  - *Coverage.* Every native kernel in an EXL3 plan is either ExLlamaV3's
    (a linear or its bias add, gated above with the linear) or a
    GGML-derived operation gated here: norms (including the final norm),
    RoPE, attention (with its mask pre-pass and combine), SwiGLU gating,
    residual adds, embedding lookup and casts. The KV write is a byte copy,
    checked by the wiring rule below. The logits leave the plan as
    `lm_head`'s F16 output; widening them is exact and not a plan
    operation.
  - *Plan first.* Native's recorded executed plan must equal
    [`exl3-op-plan.json`](experiments/backend-proof-p0/exl3-op-plan.json)
    per phase kind before any operation or model comparison, as the FP16
    gate requires of `fp16-plan.json`. That means:
    - the same operations in the same order;
    - each launching the same kernels in the same order, with the same
      grid, block and shared memory. GGML kernels are identified by mangled
      name (NVCC's per-file `_INTERNAL_` hash normalized) and SASS hash from
      the NVCC 13.4.92 build; ExLlamaV3's and cuBLAS's by name;
    - the same linear paths and output dtypes.

    Only stream identity, addresses, the launch API and the PDL attribute
    may differ. A native that let GGML pick its MMA kernel for prefill fails
    here, before any accuracy check.
    - A phase kind is keyed by its row count, starting position and padded
      K length (n, P, Npad). Grids, the attention's parallel blocks, its
      mask pre-pass and the linear paths depend on all three.
    - An unrecorded kind needs a reference-only record, approved, before
      native runs it.
    - Besides the recorded kernels, native may only upload its host-built
      inputs (ids, positions, mask) and zero-fill K/V padding. It may launch
      no other kernel.
    - Kernel names are compared after normalizing NVCC's whole
      `_INTERNAL_…` token.
  - *Wiring.* Each operation's recorded input must equal, byte for byte,
    the recorded output of the operation that produced it; the record names
    the producer of every tensor. This catches a stale or mis-wired buffer
    that each operation's own check would pass. Attention's K/V input over
    cells `[0, Npad)` is wired too:
    - cells below `P + n` are the `kv_write` outputs of that layer, from
      every earlier phase of the trajectory and this one;
    - padded cells are the declared zeros.
  - *The check.* Native records each such operation's inputs and output,
    for every invocation in its own EXL3-G run. That recording must
    reproduce the uninstrumented output (RE-010). The bridge's GGML kernel
    (same source, flags and `sm_121a` build) recomputes the output from
    those inputs, and the two must be bit-identical.
  - *Where the bridge's inputs come from:*
    - its semantic parameters, from its own code, using the model's
      configuration and the trajectory: eps, RoPE base and positions,
      softmax scale `head_dim^-½`, `n_kv`, mask and KV slot. This harness
      is reference-only code in the P0 experiment directory. It shares no
      code with native's planner, and it is reviewed before any native
      output is seen;
    - its weights (norm scales, embedding, biases), from the artifact;
    - only implementation parameters (the kernel variant and launch
      configuration), from native's recorded plan, which the plan-first
      step has already matched to the record.
  - *Dtype chain.* Every tensor between operations must have the dtype and
    element format the record declares (its `tensors`). Casts are
    operations too, so a rounding to a narrower format and back fails the
    gate.
  - *Bias add.* The q/k/v bias add has one owner on every path:
    ExLlamaV3's `add_kernel_hhh`, in F16, gated with the linear. Its output
    must equal upstream's bit for bit, which on the reconstruction path is
    PyTorch's add (the native EXL3 operation plan below).
  - This gate is what catches the subtle faults the Tier C bound cannot
    see: a wrong scale, or a lower-precision intermediate in one layer.
- **BP-S1 (approved, part of the FP16 gate).** Each of the two
  implementations is exact against the bridge arm that uses it: fused
  RMSNorm against the fused arm, unfused against the unfused arm.

Basis:
- both bridges reproduce their references bit for bit, the EXL3 bridge now
  also over every 1,023- and 1,024-row prefill;
- every reference arm repeats and restores exactly;
- the executed-plan record reproduces its logits under profiling.

### Native EXL3 operation plan (approved 2026-09-26)

Native EXL3-G runs ExLlamaV3's kernels for the quantized linears (the
packed and reconstruction paths above) and their bias add. Its other
operations run GGML's CUDA kernels from the pinned `b29c606e2`, built as
the FP16 plan's are (NVCC 13.4.92, `-use_fast_math`, `sm_121a`).

**The plan is a record,**
[`exl3-op-plan.json`](experiments/backend-proof-p0/exl3-op-plan.json),
generated by [`exl3_op_plan.py`](experiments/backend-proof-p0/exl3_op_plan.py)
from real runs (the report's
[plan record](experiments/backend-proof-p0/README.md#the-operation-plan-record)).
It covers both fixtures and seven phase kinds: prefills of 32, 144, 145,
1,023 and 1,024 rows, and single-token steps with K padded to 256 and to
1,280 positions, the only two padded lengths the P0 trajectories reach.
Before native runs any other phase kind (another chunk size, another padded
length such as BP-F3's 768, or another parallel-block count), that kind is
recorded from a reference-only run in the same way, and approved. For
the embedding, one decoder layer (the same in all 24) and the output, it
lists in order:
- every operation, with its owner and GGML operation type;
- every kernel it launches, with grid, block and shared memory. GGML
  kernels also carry their mangled name and SASS hash;
- every tensor between operations, with its dtype and shape, and the
  operation that produces it;
- each linear's path and output dtype;
- the attention's parameters, the padding and mask rules, and the KV write.

The Tier E item above requires native's executed plan to equal the record
per phase kind before any comparison.

In summary (the record has the kernels and launches):

| Operation | Plan | Against FP64 (the operation study) |
| --- | --- | --- |
| Embedding | GGML `get_rows` on the BF16 table, to F32 | bit-identical to upstream |
| RMSNorm, final norm | GGML `rms_norm` fused with `mul`, F32 on the F32 residual stream; cast to F16 for the linears | equal to upstream: bit-identical in 211 of 330 cases, one F16 rounding apart in the rest |
| q/k/v bias add | ExLlamaV3's `add_kernel_hhh`, F16, on every path | bit-identical to upstream (below) |
| RoPE (NEOX) | the F16 projections widened to F32, GGML `rope` in F32. Q stays F32 into attention; K is cast to F16 and copied into the cache with V | 0.003–0.16 times upstream's relative error |
| Attention | GGML's vector flash-attention kernel for every phase: F32 Q, F16 K/V and mask; output cast to F16 for `o_proj` | 2.3–4.1e-6 relative error against upstream's 2.2–3.6e-4. Its output rounded to F16: 0.82–0.99 times upstream's; with Q from F32 RoPE, as the plan chains them, 0.24–0.49 times |
| Residual adds | GGML `add` in F32 | bit-identical to upstream's |
| SwiGLU gating | GGML `swiglu` in F32 on F32 gate and up; cast to F16 | bit-identical to upstream |

The linears give F16 (q, k, v, `lm_head`) or F32 (`o_proj`, gate, up,
down) outputs, as upstream's do. Up to 144 rows they run packed; gate and
up run as one multi-linear up to 32 rows. From 145 rows they reconstruct,
and from 1,024 rows they use the fused reconstruction. The plan's output is
`lm_head`'s F16 logits.

**Bias add.** Upstream adds the bias inside its packed linear with
ExLlamaV3's `add_kernel_hhh`. On the reconstruction path it uses PyTorch's
elementwise add: F16 operands, F32 arithmetic, one rounding. The plan uses
ExLlamaV3's kernel on every path, so the bias add has one owner and one
precision and stays with its linear. The two adds are the same function:
- they agree on all 2^32 pairs of F16 inputs (every non-NaN result
  bit-identical, NaN in the same places);
- they agree on every reconstruction-path bias add of both probes.

A GGML F32 add would need F32 linear outputs, a dtype chain unlike
upstream's; it was not adopted.

**Attention** is declared explicitly:
- **Padding, in every phase.** Prefill and single-token steps alike attend
  K and V padded to a multiple of 256, as llama.cpp pads its cache. Padded
  cells must be finite: their scores are masked to −∞, so finite values
  get weight exactly 0, but Inf or NaN would poison the row. They are zero
  in the reference and declared zero.
- **Mask.** F16, one row per query: 0 up to the row's own position, −∞
  after it, including every padded column.
- **Decode.** With the padding, GGML would pick the vector kernel itself.
- **Prefill.** GGML would pick its MMA kernel. That kernel accumulates V·P
  in F16 and is up to 2.85 times *less* accurate than upstream at 1,023
  and 1,024 rows. Its precision flag is not read on CUDA. The plan
  therefore calls the vector kernel's launcher directly
  (`ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>`) for every phase. From
  1,024 rows the launcher also runs a mask pre-pass that skips fully
  masked KV tiles.
- **Also rejected:** the tile kernel (13–24 times less accurate) and the
  non-flash path. On GB10 the non-flash path runs K·Q in TF32 through
  llama.cpp's handle, and it materializes the scores.

The vector kernel never materializes the scores. Its scratch is about
15 KB and 48 KB at decode (K padded to 256 and 1,280), and 0.36, 1.6 and
11.4 MB at 32, 144 and 1,024 prefill rows. It is counted in `E`, against
the memory limits below.

**Evidence.** The `ggml_ops` arm runs this plan inside upstream, with two
kinds of exception:
- operations left to upstream: the embedding and the MLP residual add
  (GGML's are bit-identical to them), and the reconstruction-path bias add
  (PyTorch's);
- its GGML library is built with GCC 13.3 as host compiler, as shared
  libraries, with `GGML_CUDA_GRAPHS=OFF`.

With and without cuBLAS 13.8.0.4 it passes every repeat, restore and
capture check. Its overall accuracy is within 1.5% of frozen EXL3-G's: the
full model is dominated by the F16 rounding at operation boundaries that
both share.

The record's probe runs the whole plan, those three operations included:
- its GGML library was built with NVCC 13.4.92;
- ExLlamaV3's extension was the container's NVCC 13.0.88 build
  (`7c9d383f…`), which is where the probe's ExLlamaV3 launches come from.

The probe's logits equal the arm's (cuBLAS 13.8.0.4) bit for bit in every
phase kind on both fixtures. Two more results:
- the four GGML kernels the FP16 bridge also launches have the same SASS
  in both builds, despite the different host compilers;
- the GGML library built with the container's NVCC 13.0.88 and with the
  SDK's 13.4.92 gave identical outputs for all 3,670 operation cases.

### Tier C: EXL3 full model against an FP64 oracle (approved 2026-09-26)

For BP-N6 the native plan cannot be exact. Upstream runs norm, RoPE,
attention, embedding and residuals in PyTorch or Triton, so the full model
has no common kernel to match. The bound is therefore accuracy against a
common oracle: a teacher-forced FP64 forward pass over the same decoded
EXL3 weights (`oracle.py`). Its calibration is in the report's
[third pass](experiments/backend-proof-p0/README.md#exl3-third-pass-calibrating-the-full-model-bound).

- **The rule.** For each fixture, the P0 report computes 750 statistics
  against the oracle, over every prefix of EXL3-G:
  - *averaged*: the logits' RMS error, prefill and single-token steps
    separately, and each block's row-averaged relative RMS error;
  - *extreme*: the worst row's logit RMS error, the largest absolute logit
    error, each block's worst-row relative error, and K's and V's worst
    position (RMS error relative to the oracle's magnitude at that
    position), over the prefix and over the positions the single-token
    steps wrote.

  Native EXL3-G fails if any statistic exceeds its bound: 2 (averaged) or
  8 (extreme) times the median of the legitimate arms' distinct values.
  There are 15 legitimate arms:
  - frozen EXL3-G and its four retunings;
  - GEMV on;
  - cuBLAS 13.8.0.4;
  - five variants using upstream's own alternative norm, RoPE and attention
    code, and all of them together with cuBLAS 13.8.0.4;
  - the native-like `ggml_ops` arm, with and without cuBLAS 13.8.0.4.

  The bounds are listed per statistic in
  [`tierc.json`](experiments/backend-proof-p0/tierc.json).
- **The run.** Native EXL3-G runs the trajectories with the report's
  captures. Its logits must be finite. A captured run must reproduce the
  uninstrumented logits exactly (RE-010). The first layer whose statistic
  exceeds its bound is the finding. Top-1 agreement with the oracle is
  reported, not gated.
- **Calibration** (from `tierc.json`):
  - Leave-one-out, the largest ratio any legitimate arm reaches is 1.25
    (averaged) and 2.14 (extreme) at 4.0 bpw, and 1.12 and 1.51 at
    4.5 bpw. The native-like arms reach 1.07–1.15 and 1.34–1.78.
  - Five gross faults fail, and a more accurate native plan passes easily:
    - a wrong norm epsilon;
    - Q alone shifted by one RoPE position;
    - Q and K shifted together, which RoPE's relative form nearly cancels,
      so only the absolute K statistic catches it;
    - the same shift in single-token steps only;
    - one corrupted key position.
- **What this bound cannot see.** Two subtle faults pass on both fixtures,
  as a legitimate implementation change would: a softmax scale 1% high in
  one layer, and one layer's MLP output rounded to BF16. The
  operation-level exactness gate and its dtype-chain check catch them
  (Tier E).

### Memory and workspace (the M2 gate in exl3-bringup.md)

**FP16: proposed; still needs per-phase limits.** The bridge's compute
buffer is sized for the largest batch.

**EXL3: the persistent-workspace limit is approved (2026-09-26); the phase
limits are deferred to P3 entry.** The per-phase measurements are in the
report's
[per-phase memory](experiments/backend-proof-p0/README.md#per-phase-memory).
They are identical under cuBLAS 13.1.1 and 13.8.0.4.

- **Envelopes.**
  - Every native plan declares, per phase kind and profile, its envelope
    `E` before the run. `E` covers everything the phase allocates or holds
    as scratch: activations and intermediates, operation scratch (the
    attention pool), transient reconstruction and the logits output.
  - Its observed peak, from the BP-A1 census (jitLLM's catalog plus the
    driver and library census), must stay within `E`.
  - Weights must equal the artifact's bytes. Padding is reported
    separately.
  - KV must equal the declared layout.
  - Persistent library workspaces are limited separately (below).
  - `F` holds only handles and module state. It is reported, never used to
    hide a workspace.
- **EXL3 phase limits.** Native's `E` per phase may not exceed upstream's
  PyTorch peak allocation above the phase's start (FP16 logits,
  activations and reconstruction included) plus an allowance for the
  declared plan's own needs. The allowance has three parts, all taken from
  the plan and its measurements, not chosen:
  - *The vector attention kernel's pool scratch,* measured in the operation
    study. Upstream's Triton prefill needs none.
  - *The plan's F32 intermediates of one layer,* counted with no buffer
    reuse: `attn_norm.f32`, `rope_q.in`, `q_rope`, `rope_k.in`,
    `k_rope.f32`, `attn.f32`, `mlp_norm.f32` and `swiglu.f32` in
    `exl3-op-plan.json`. That is `n × 38,400` bytes. Upstream keeps these
    tensors in F16 or never materializes them.
  - *The attention mask,* F16 `[n, Npad]` in the record: `n × Npad × 2`
    bytes. Upstream's causal attention builds none.

  | Phase | Upstream peak | Attention scratch | F32 intermediates | Mask | Limit (bytes) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | prefill, 32 rows (Npad 256) | 9,953,792 | 354,816 | 1,228,800 | 16,384 | 11,553,792 |
  | prefill, 144 rows (Npad 256) | 44,790,272 | 1,596,672 | 5,529,600 | 73,728 | 51,990,272 |
  | prefill, 145 rows (Npad 256) | 103,822,336 | 1,607,760 | 5,568,000 | 74,240 | 111,072,336 |
  | prefill, 1,023 rows (Npad 1,024) | 376,915,456 | 11,343,024 | 39,283,200 | 2,095,104 | 429,636,784 |
  | prefill, 1,024 rows (Npad 1,024) | 375,390,720 | 11,356,160 | 39,321,600 | 2,097,152 | 428,165,632 |
  | single-token step (Npad up to 1,280) | 310,272 | 48,048 | 38,400 | 2,560 | 399,280 |

  - The allowance is 7–16% over upstream's peak in prefill and 29% in a
    single-token step. It is the approval this limit asks for: the
    declared plan's F32 chain and vector attention cost that much memory
    in exchange for their accuracy.
  - Native EXL3-G's logits are F16, as upstream's are.
  - Chunks between measured sizes use the next larger measured phase, up
    to 1,024 rows. Chunks above 1,024 rows are out of scope.
  - Upstream's peak is its caching allocator's allocated bytes. It does
    not include reserved-but-unused cache, which native has no counterpart
    for.
- **Persistent library workspaces.** Native's total may not exceed
  upstream's device context: ExLlamaV3's 16 MiB workspace and 4,202,760
  lock bytes, 20,979,976 bytes in all.
  - Native's pinned cuBLASLt GEMMs need no workspace.
  - PyTorch's 32 MiB cuBLAS workspace is not counted. ExLlamaV3 replaces
    it with its own 16 MiB before every call, so the recorded plan never
    uses it.
- **No permanent FP16 shadow** (BP-A3).

### Performance protocol (rule approved 2026-09-26; BP-F2's reference deferred to P3 entry)

The owner approved the kernel rule below (the statistic, thresholds,
aggregate test and confirmation procedure) for BP-F2, the EXL3 kernels.
BP-F2's reference arm, deferred to P3 entry (see the status above), is
currently:
- upstream's extension built with the SDK's NVCC 13.4.92, as native's port
  is (`exllamav3_ext.so` `aa8b9f16…`);
- cuBLAS 13.8.0.4.

The NVCC 13.0.88 build shares identical SASS with the 13.4.92 build for
only 16 of its 1,506 kernels. So the rule was recalibrated and its holdout
rerun on the 13.4.92 build (below).
BP-F1 uses the same rule, but needs its own calibration sessions before its
approval. The model cases (BP-F3) and BP-F4 are reported, not gated.

This protocol extends the frozen rule of the
[EXL3 reference](experiments/exl3-reference/README.md#memory-and-acceptance-limits).
Its departures from M0 come from the report's
[timing controls](experiments/backend-proof-p0/README.md#timing-controls):
the sessions that timed upstream against itself (A/A) and against its GEMV
configuration. [`timing_protocol.py`](experiments/backend-proof-p0/timing_protocol.py)
at `c05fd2dd…`, fixed before its calibration and holdout sessions ran,
is the measured implementation of the kernel rule. The current script also
rejects incomplete or nonfinite sessions and treats a constant positive
aggregate shift as a failure; these checks leave the recorded calibration
and holdout outcomes unchanged. Every BP-F2 session runs with the SDK's cuBLAS
13.8.0.4, which the native plan pins, bind-mounted over PyTorch's in the
reference container.

- **Session.** A session is one uninterrupted sequence on `spark`:
  - no other GPU workload runs, and the clock policy and idle states stay
    unchanged;
  - SM clock and temperature are recorded at every block boundary;
  - each arm tunes (or warms up) in a discarded process, and one more
    discarded process runs immediately before the first timed block;
  - eight timed blocks follow, each a fresh process that runs every case,
    in the order A1 B1 B2 A2 B3 A3 A4 B4 for a primary session and B1 A1
    A2 B2 A3 B3 B4 A4 for a confirmation. A is the reference, B the
    candidate.

  `spark-b` sessions are stability evidence only.
  - *Why not two blocks per arm, as M0:* separate processes of one plan
    differed by more than one pair of reference processes captured. M0's
    rule failed 26 of 176 A/A cases.
  - *The extra warm-up process* keeps the first timed block from starting
    on an idle GPU. It does not remove the first-block effect. With it, the
    first block still ran 0.1–0.45% fast in some sessions (`c1`, `h1`). The
    primary order therefore leans slightly against the candidate, and the
    mirrored confirmation leans slightly towards it. The calibrated `σ`
    includes this effect.
- **Kernel cases (BP-F1, BP-F2).**
  - Five warm calls, then 31 samples per block.
  - Each sample replays one CUDA graph of ten invocations, bracketed by
    events, and records the interval divided by ten.
  - The graph is captured only in the benchmark, identically on both sides.
  - Every launched kernel is verified per case before timing. The native
    case's ordered launches must equal the recorded ones: kernel name,
    grid, block, shared memory (static plus dynamic, as the profiler
    reports it) and registers per thread. For BP-F2 the record is
    [`exl3-launch.json`](experiments/backend-proof-p0/exl3-launch.json),
    re-recorded on the NVCC 13.4.92 reference with each ExLlamaV3 kernel's
    SASS hash, which native's must equal. The tuner's decoded choice is
    forced to match. The bias add follows the native operation plan's
    owner, not upstream's PyTorch kernel.
  - A stream-launched arm (no graph) is reported beside it.
- **Plans.**
  - Upstream runs on frozen tuning caches, recorded in the reference's
    session. The 96 synthetic shapes use those of the report's A/A
    sessions.
  - Native forces the same choices, and the caches are verified unchanged
    after every block.
  - Separately tuned upstream plans differed by up to 12% in speed, so the
    frozen caches fix the performance reference as well as the numerics.
- **Model cases (BP-F3).**
  - 15 warm-ups, then 31 trials per block, in the same eight-block order.
  - Prefill at 32, 144, 145, 512, 1,023 and 1,024 tokens, for every fixture.
  - Decode of 64 tokens after a 512-token prefix, 31 requests per block.
  - Reported, not gated, at M2.
- **Statistic.**
  - Each case's process-to-process noise `σ` comes from calibration A/A
    sessions, upstream against itself on one frozen plan. It is the
    relative standard deviation of block medians within a session, pooled
    over the sessions.
  - For BP-F2, four sessions on the reference arm set it (`c5`–`c8`, two in
    each order), recorded in
    [`timing-calibration.json`](experiments/backend-proof-p0/timing-calibration.json).
    - The median `σ` is 0.62%.
    - The per-case thresholds (`z · σ · √½`) are below 2% for 96 of the
      176 cases and below 5% for 146.
    - The noisiest, synthetic 14,336 × 4,096 K4 at 1 row, reaches 18.5%.
  - An earlier calibration on the NVCC 13.0.88 build (`c1`–`c4`, median
    `σ` 0.69%) is kept in `timing.json`.
  - Native's ExLlamaV3 port is compiled as the reference is: NVCC 13.4.92,
    `-O3 --use_fast_math`, SASS for `sm_121` (not `sm_121a`), with the same
    sources. That is what makes the launch record's SASS hashes a
    requirement native can meet.
  - Per case and session:
    `d = (candidate median / reference median − 1) / (σ · √½)`. Here each
    arm's median is taken over all 124 of its samples, and √½ reflects the
    four processes behind each arm.
- **Pass.**
  - A case fails a session when `d` exceeds `z = 3.86`. That is the
    one-sided normal quantile for a family-wise false-failure rate of 1%
    over the 176 cases.
  - Aggregate: cases are not independent, because a whole process can run
    fast or slow. Each block's session-wide shift is the median, over the
    cases, of the block's median relative to that case's mean. A one-sided
    two-sample t-test compares the candidate's four shifts with the
    reference's (6 degrees of freedom). The aggregate fails above 3.143,
    the 1% quantile. This catches a systematic regression spread thinly
    over many cases.
  - If any case fails, or the aggregate fails, a confirmation session in
    the mirrored order repeats every case. A case fails the stage only
    when it fails both sessions, and the aggregate must also pass in the
    confirmation. The outcome is mechanical, and every session is
    reported.
  - There is no floor, no averaging away of a case, and no dropped sample.
    A stage failure stands until its cause is fixed.
- **Validation.** Earlier designs are recorded in the report:
  - M0's two blocks per arm;
  - in-session allowances;
  - spread checks that single outlier processes tripped;
  - a case-independent aggregate test that a common process shift broke.

  The rule was fixed at `c05fd2dd…` before any cuBLAS 13.8.0.4 session
  ran. It was declared in advance that a holdout pair failing the stage
  would reject the rule, and any redesign would need a fresh holdout. Each
  holdout was timed A/A and not used for calibration:
  - *NVCC 13.0.88 build (`h5`, primary; `h6`, mirrored):* the stage passes.
    - In `h5`, the 4.0 bpw `up_proj` at 145 rows (d = 4.23) and the
      4.5 bpw `down_proj` at 16 rows (d = 3.98) exceeded `z`, which
      triggered the confirmation.
    - In `h6`, neither failed.
    - The aggregate `t` was 0.89 and −0.04.
  - *The reference arm, NVCC 13.4.92 (`h7`, primary; `h8`, mirrored):* the
    stage passes in `h7` alone. No case exceeded `z`, the aggregate `t` was
    0.77, and no confirmation was needed. `h8` also passes on its own: no
    case over `z`, aggregate `t` −2.31.
  - Every in-sample pairing of either calibration passes, needing at most
    a confirmation.
- **Power** on the reference arm, from `timing.json`:
  - *One slowed case,* slowed in both sessions of a pair (the holdout pair
    and three calibration pairs): detected 48–54% of the time at 2%,
    63–68% at 3%, 79–80% at 5% and 87–90% at 10%. Most misses are the
    noisier cases.
  - *A slowed subset* (`timing_power.py`), on the holdout pair:
    - the stage fails from 0.5% when all cases or the 66 reconstruction
      cases are slowed;
    - it fails from 1% for the 22 fused-reconstruction cases;
    - it did not fail up to 3% when only the noisiest quarter was slowed.
- **Harness equivalence (before BP-F2 runs).** The native benchmark harness
  times upstream's own kernel as its candidate: the extension binary of the
  reference arm, identified by its SHA-256. `measure.py` on the same binary
  is the reference, under this rule. The session must pass before any
  native kernel is timed.
- **BP-F1: host VMM against `cudaMalloc`.**
  - Compares the same GGML kernels, at the held-out trajectory's chunk
    shapes (1, 16, 17 and 512 rows).
  - Each sample rotates through weight buffers whose total exceeds four
    times the queried L2 size, so the kernels read from memory, not L2.
  - Both memory kinds run in separate processes, as blocks, under the
    kernel rule above.
  - Before the first comparison, four A/A calibration sessions on
    `cudaMalloc` set BP-F1's `σ`, and two holdout sessions validate the
    rule for it. Both are brought to the owner before BP-F1 is gated.
  - A regression reopens D-034 for the owner; it does not block other
    stages.
- **BP-F2: EXL3 kernels.**
  - All 176 cases, against upstream EXL3-G with cuBLAS 13.8.0.4, the
    matched plan.
  - Upstream's GEMV gap at 1 to 8 rows is measured in the report, as
    EXL3-O against EXL3-G. It applies only to the 4.0 bpw fixture: its
    GEMM-only kernels are 1.14–1.55× slower on q, k and down at 1 and 8
    rows, and 0.90–0.94× on the fused gate/up. The 4.5 bpw fixture
    launches the same kernels either way. Accepting the gap at M2 is an
    explicit tradeoff for the owner, not a pass (above). The tradeoff expires when the GEMV
    provenance gate closes, and at M3's entry at the latest: from then on,
    BP-F2 is gated against EXL3-O.
- **BP-F4: host submission time.**
  - Reported per launch and per token, against both upstreams' decode:
    llama.cpp with CUDA graphs on and off, and ExLlamaV3's graph-captured
    decode.
  - Graph capture becomes an M3 prerequisite if native host time per token
    exceeds upstream's measured time per token minus native device time per
    token.

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

**Proposed criteria** (2026-09-25, second draft, pending the owner's
approval). The trace's identity is added and approved once the trace exists,
before any design runs on it. Margins and budgets are proposals, not
measurements.

Still needed:
- define "the allowance" for (a) and (b), and an exact rule for
  deterministic counts;
- replace "end-to-end minus transfer", which is undefined where the two
  overlap;
- say which of the cold and warm restores each criterion applies to;
- an interleaving design;
- extent alignment of at least 4 KiB where direct reads land
  (artifact-format.md), not 256 B.

- **Designs.**
  - The baseline is D-033: one 2 MiB handle per extent, no free pool.
  - The alternatives are persistently mapped slabs of 32 MiB, 256 MiB and
    1 GiB, each with software suballocation.
  - Each slab size is tried with the four hole policies above.
  - Suballocation keeps the validated alignment classes: the larger of the
    kernels' (at least 256 B inside shared extents, "Source findings") and
    the direct reads' (4 KiB). Rung 5 covers relocation within a slab.
- **Trace.**
  - A seeded generator, kept in Git, builds a library of synthetic models.
    Their dense groups and expert closures take D-056's measured sizes.
  - The switching sequence derives from the reference A→B→A frozen trace
    and the paging-feasibility routing captures.
  - Its identity is the generator's hash, its parameters, its seed and the
    output's SHA-256.
  - A second seed, never looked at while choosing, confirms the winner in a
    fresh session.
- **Budgets.** The physical budget is set so that the trace's unique bytes
  are 1.25, 1.5 and 2 times the budget.
- **Metrics, per design and budget.** Cold restores (backing created) and
  warm restores (backing handed off) are reported separately.
  - End-to-end restore latency per closure, from request to ready for its
    consumer, at p50, p95 and p99. Its 95% interval comes from a bootstrap
    over switch events.
  - The part of that latency the design causes: the end-to-end time minus
    the same run's measured transfer time for the closure's bytes.
  - Driver calls and time per restored byte, and io_uring buffer
    registration and unregistration.
  - Stranded bytes (free inside slabs but unusable), as a time-weighted
    mean and a peak, as a share of the budget.
  - Admissions delayed or refused because of fragmentation, and the total
    delay.
  - The time to return 1 GiB to the OS on demand, and what it evicts.
  - Decode inflation during restore or compaction, measured as EXL3-G
    decode token time against a quiet control.
  - Bytes moved by compaction.
- **Amending D-033.** A slab or hybrid design replaces the baseline only if
  it meets all of the following at every budget and on both seeds:
  - its design-attributable p95 restore time is better by more than the
    larger of the measurement allowance and 10%;
  - its end-to-end p95 is no worse, beyond the allowance;
  - its total waste, stranded plus padding, is no more than the baseline's
    padding, both as a time-weighted mean and at the peak;
  - it has no more fragmentation-induced refusals than the baseline, and no
    more total admission delay beyond the allowance;
  - when shrinking, it evicts no more useful content than the baseline, and
    its p95 shrink time is no worse, beyond the allowance;
  - it inflates decode by no more than the baseline, beyond the allowance;
  - if it moves addresses, it passes BP-P5, BP-L2 and BP-L6.

  Otherwise D-033 stays. A design that wins at only some budgets is reported
  for the owner.
- **Part (b).** The declared mixes are:
  - all small transfers (64 KiB or less);
  - all bulk transfers (2 MiB runs);
  - small and bulk, one to one by count;
  - the size distribution of D-056's measured closures.

  Bounded asynchronous submission replaces serial submission if both hold
  for every mix:
  - the time to the last required completion improves by more than the
    allowance;
  - consumer-stall p95 is no worse, beyond the allowance.

  The chosen depth is the smallest within the allowance of the best.

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
