<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof P3, part 1: native EXL3 linears — 2026-09-27

This is the first part of P3 of the [backend proof](../../backend-proof.md#stages):
jitLLM's own launchers of ExLlamaV3's kernels, judged per linear by the
approved Tier E items for EXL3 packed linears (up to 144 rows) and
reconstruction-path linears (145 rows and more). It is BP-N5, the
per-linear sweep: every real projection of both EXL3 fixtures at rows 1,
8, 9, 16, 32, 33, 144, 145, 1,023 and 1,024, native against upstream at
the same forced plan and inputs, in both EXL3-G (GEMV off) and EXL3-O
(GEMV on, BP-F2's gated reference since D-080). The full-model run
(Tier C) and BP-F2's timing are part 2.

**Results in brief** (2026-09-27, `spark-b`; `cudaMalloc`, rung 3, and device VMM):

| Fixture | Arm | Cases exact | Stages exact | Weights and reconstructed weights | Placements |
| --- | --- | ---: | ---: | --- | --- |
| 4.0 bpw | EXL3-G | 1,570 of 1,570 | 5,398 | 169 of 169 linears | all four |
| 4.0 bpw | EXL3-O | 1,570 of 1,570 (240 through the GEMV) | 5,398 | 169 of 169 | all four |
| 4.5 bpw | EXL3-G | 1,570 of 1,570 | 5,398 | 169 of 169 | all four |
| 4.5 bpw | EXL3-O | 1,570 of 1,570 (28 through the GEMV) | 5,398 | 169 of 169 | all four |

Every buffer each step of each path writes (the transformed input, the
product, each reconstruction slice, the reconstruction GEMM's output, the
output transform, the bias) is bit-identical to upstream's, and so is the
final output: packed linears byte-equal to upstream's kernel output,
reconstructed F16 weights equal to upstream's reconstruction, and the
reconstruction path exact under the approved pinned cuBLASLt rule.
Native's composed implementations (the registry's) reproduce native's
steps in every case. The same holds with every operand at the smallest
alignment the host checks accept, and with every operand, weights
included, flush against an unmapped VMM granule on either side (the
over-read probe: **no ExLlamaV3 kernel the sweep launches reads or writes
outside its operands**, so v0's `readable_bytes` = `bytes` for these
fixtures' EXL3 resources stands).
Every kernel native launches, ExLlamaV3's and cuBLAS's, is the one upstream
launched, with the same grid and block ([below](#launches)), and the SASS
of every ExLlamaV3 kernel native launches equals the reference build's
([below](#sass)).

## What runs

**Kernels.** The source lock's `exllamav3` component now holds, besides
upstream's GEMM compilation units, the K = 4 GEMV kernel's header and the
three upstream sources that held the reconstruction, Hadamard and bias-add
kernels together with their ATen wrappers
([licensing.md](../../licensing.md#exllamav3-gemm-kernels-in-the-core-m2)):
- patch 0003 reduces `quant/reconstruct.cu`, `quant/hadamard.cu` and
  `add.cu` to their kernels, removing the ATen, c10 and PyTorch-API
  includes, the host wrappers and `reconstruct.cu`'s instance tables (and
  two no-op `register` specifiers). The kernels are unchanged;
- patch 0002 adds `jitllm/jitllm_exl3_kernels.cu`, jitLLM's instance unit,
  built with upstream's flags beside the GEMM units: it includes those
  sources and the GEMV header and instantiates only what the linear
  launches (the eight K = 4 mcg GEMV instances of upstream's
  `exl3_gemv_select_kernel`, `reconstruct_kernel` and
  `reconstruct_had_kernel` at K = 4, 5, 6 and 8, three Hadamard variants,
  `add_kernel_hhh`, which `add.cu` defines with its other adds and two MoE
  bias adds, compiled but never launched), with host lookups of every
  kernel (`jitllm_exl3_kernels.h`, no CUDA types).

**Launchers** (`src/kernels/exl3/`, K-L under D-053). They replace
upstream's host wrappers:
- `validate.h`, in every profile: each launch's operand and plan checks,
  run on the host before anything is queued: shapes, rates, alignment
  (the source-derived minimums), overlaps, the kernels' int bounds, and a
  cooperative grid no larger than the device holds at once;
- `upstream_gemv.h`: a recorded copy of upstream's GEMV choice
  (`exl3_gemv_cfg` and `exl3_gemv_try_launch`'s rules), MIT AND
  Apache-2.0;
- `launch.h`: the launch context. It launches on a provider stream
  (`DeviceExecution::Submission`), zeroes its lock area (a declared
  workspace of upstream's 4,202,760 bytes) on that stream before its first
  launch, refuses a lock area another live context holds, computes each
  cooperative kernel's co-resident limit from the occupancy calculator,
  launches cooperatively where `launch_contract.h` requires it, and returns
  a launch error as a fault (after which it refuses every launch). The
  multi-GEMM never passes an expert range, so its selection state is never
  touched;
- `recon_gemm.h`: the reconstruction GEMM on cuBLASLt with the algorithm
  rebuilt from its nine pinned attributes (`exl3-recon-pin.json`), checked
  by `cublasLtMatmulAlgoCheck` before any launch;
- `linear.h`: each path of the linear in upstream's order (packed through
  the GEMM or the GEMV, the fused gate/up multi-GEMM, the reconstruction
  path in slices of at most 32,768 columns, the fused reconstruction),
  with the bias on every path through `add_kernel_hhh` as the approved plan
  requires;
- `implementations.h`: five registry declarations (`exl3.linear.gemm`,
  `.gemv`, `.reconstruct`, `.reconstruct_fused`, `exl3.multi_linear.mgemm`)
  whose identities cover the prepared tree, a build-time digest of every
  file of the module, the SDK, target, architecture, build type and, for
  the reconstruction paths, cuBLASLt.

**The reference** ([`linear_reference.py`](linear_reference.py),
[`run_reference.sh`](run_reference.sh)) runs in P0's reference container
on upstream's extension built by the SDK's NVCC 13.4.92
(`exllamav3_ext.so` `aa8b9f16…`, BP-F2's reference arm) with the SDK's
cuBLAS 13.8.0.4 bind-mounted (the only cuBLASLt mapped, version 130800),
`EXL3_HGEMM_F16ACC=0`, and `EXL3_GEMV=0` for EXL3-G. It loads each
fixture through upstream's Model API and calls upstream's extension as
upstream's modules do, one step at a time (`exl3_gemm` as `BC_LinearEXL3`
calls it, `exl3_mgemm` as the gated MLP does, `reconstruct_hgemm`'s steps),
hashing every buffer a step writes. Each case also runs through upstream's
own module call, whose output the steps reproduce in every case, and runs
its steps a second time, which repeat bit for bit. A second process
(`repeat-*`) repeats every stage of every case.

**Inputs** ([`cases.py`](cases.py)): row-major F16 activations from a
counter-based generator both sides implement, each element (top 11 bits of
`splitmix64`) − 1,024 over 1,024, which F16 holds exactly; the seed is
SHA-256 of the fixture, the linear and the row count. Native's inputs hash
equal to the reference's in every case.

**Plans.** Upstream's tuner chooses each packed launch (tile shape, grid,
concurrency), keyed by the problem with the rows bucketed to the next power
of two up to 16. Each reference arm starts from P0's frozen model cache
(`tune-40-gemvoff`, `tune-45-gemvoff` for EXL3-G; `tune-40`, `tune-45` for
EXL3-O). They lack only the 8-row bucket, which no P0 trajectory reached:
a first, discarded pass let upstream's tuner add those records (6, 11, 2
and 10 of them; no frozen record changed), and the measured pass and its
repeat ran on the result unchanged (cache SHA-256 before and after equal).
[`native_plan.py`](native_plan.py) decodes each case's record from the
final cache and checks it against the launches the reference profiled
(kernel, grid, block): no case disagrees. Where EXL3-O launched the GEMV,
the plan is the configuration and grid it launched, and jitLLM's copy of
upstream's choice (`upstream_gemv.h`, on the device's own occupancy) picks
exactly those, and the GEMM wherever upstream kept the GEMM. The
reconstruction paths take each slice's pinned algorithm; every
reconstruction GEMM of the sweep is one of the 21 recorded.

**Native** ([`benchmarks/exl3_linear_sweep.cc`](../../../benchmarks/exl3_linear_sweep.cc))
opens each v0 artifact with the native reader, reads each linear's
tensors, and runs every case through the launch context step by step,
hashing what the reference hashes, then again through the implementation
the registry binds for its path. [`compare.py`](compare.py) holds every
stage, every weight hash and the full reconstructed weights, rotated and
fused, to the reference's.

## Placements: alignment and over-read

The sweep ran four times per fixture and arm, with every operand placed:
- **malloc**: `cudaMalloc`, 256-byte aligned (rung 3);
- **minimal**: at exactly the smallest alignment the host checks accept,
  not more: the trellis, the transformed input and F32 outputs at 16
  bytes, side vectors, activations and F16 outputs at 8, biases at 2, the
  reconstruction GEMM's operands at the 16 its pinned algorithms assume;
- **flush-end** and **flush-start**: in device VMM (D-081), each operand
  (every weight tensor, input, scratch and output) ending exactly at the
  end of its mapping with an unmapped granule after it, or starting
  exactly at its start with one before it.

All four are exact in every case of every arm. An access past either end
of any operand would have faulted; none did. That settles the question
[artifact-format.md](../../artifact-format.md) left open: for the kernels
of these paths, at these shapes, rates and plans, EXL3 resources need no
over-read reservation (`readable_bytes` = `bytes`). The GEMV's prefetch
ring and every tile loop stop at their operands.

The sweep's plans launched the GEMV only in its narrow configuration at
one and eight rows, the GEMM at tile shapes 1 and 2 and the multi-GEMM at
2 and 3, each at its tuned grid. A GPU unit test
(`unit.Exl3LinearTest.NoPackedKernelReadsOutsideItsOperandsAtTheFixturesShapes`,
added in the challenge round) runs the rest on random weights at every
linear shape and rate of both fixtures (the eleven (k, n, K) of
`results.json`, F16 and F32 outputs), in `cudaMalloc` memory and flush
against an unmapped granule at either end: the GEMV in both
configurations at one to eight rows (140 launches in its row-guarded
mode at two to eight rows), the GEMM at every tile shape each shape takes
at 1, 3, 8 and 16 rows, the multi-GEMM at every tile shape of the
gate/up shape with concurrency 1 and 2, each at the co-resident grid and
at 7 blocks. No fixture shape takes tile shape 4 (its n must be a
multiple of 512, and none of 128, 896, 4,864 and 151,936 is), so it ran
on a synthetic 896 × 1,024. All 680 launches per placement ran without a
fault and gave the `cudaMalloc` run's bits (`spark-b`, 2026-09-27). The
verdict covers those kernels at these rates and shapes, not other
shapes, rates, codebooks or kernel variants.

## Launches

An nsys trace (`-t cuda`) of each sweep in `malloc` placement records
every kernel native launched. [`launches_compare.py`](launches_compare.py)
holds them, in order, to the launches the reference profiled for each
case (each case runs twice natively: its steps, then the registry's
implementation): the same kernel, grid and block, for all 6,994 launches
per arm (5,956 ExLlamaV3, 1,038 cuBLAS). The one declared difference is
the approved operation plan's: the reconstruction path's bias add is
ExLlamaV3's `add_kernel_hhh` (as `add_gr` launches it) where upstream uses
PyTorch's element-wise add. So native's cuBLAS kernel names and grids equal
the arm's, as the reconstruction-path Tier E item requires (13 distinct
cuBLAS kernels: nvjet and three CUTLASS kernels), and the GEMM, multi-GEMM
and GEMV grids are the decoded plans'.

## SASS

[`sass_compare.py`](sass_compare.py) hashes each function's SASS as
P0's `fp16_plan.py` does (instruction text with addresses stripped, and
the raw encodings), ending each function at the next or at the end of its
cubin, so no hash takes in another member's header (the defect of P0's
`sass_hashes.py`). With cuobjdump 13.0.85 on `spark-b`:
- all 27 ExLlamaV3 kernels the reference runs launched (the bias add; the
  GEMM at K = 4, 5, 6 and 8 in shapes 1 and 2 with both outputs as the
  plans use them; the multi-GEMM at K = 4 and 5; four GEMV instances; the
  three Hadamard variants; both reconstruction kernels at every rate) have
  text and encoding hashes identical to the reference extension's
  (`aa8b9f16…`) in native's sweep binary;
- so do all 93 ExLlamaV3 functions the binary holds (the 64 GEMM and
  multi-GEMM instances and the 29 of the instance unit), each against its
  namesake in the extension.

The port therefore runs the reference's own machine code, which is what
the bit-exact outputs above rely on, and what BP-F2's P3-entry item asks
("check that the port's build reproduces the reference's SASS").

## GPU unit tests

`unit.Exl3LinearTest.*` (label `gpu`, `spark-b`) runs every path on a
synthetic q_proj-shaped linear: bit-identical in `cudaMalloc` memory and
device VMM, with the GEMV within 5.7e-4 relative RMS of the GEMM and the
fused reconstruction within 9.9e-4 of the unfused one (reported, not
gated); the same over-read probe; a grid larger than the device holds
refused before launch; no two live contexts sharing lock slots; a fault
returned as a fault, after which the context refuses; and the registry
binding each implementation to its own calls only, a stale or foreign
declaration to none. Since the challenge round it also runs the over-read
probe at the fixtures' shapes ([above](#placements-alignment-and-over-read));
two contexts on two streams launching cooperative grids at the device's
co-resident limit back to back, 100 rounds of GEMM, multi-GEMM and GEMV
each, which both complete (5.0 ms on one stream, 8.3 ms on two: the grids
partly overlap and neither starves the other); a reconstruction GEMM
refusing a pin recorded for another GEMM (another row count, output or
row stride); and each identity recording libstdc++'s assertions (D-083).
`unit.Exl3ValidateTest.*` covers the host checks and the GEMV choice in
every profile, the multi-GEMM's refusal of tables written for tensors
that have since moved (BP-P5) among them.

## Not covered here

- The full-model run of both fixtures (Tier C) and the operation-level
  gate's recorded plan (`exl3-op-plan.json`), and BP-F2's reference arm
  and timing, which P3-entry approvals govern: part 2.
- The launch recorder (`tests/support/`) does not wrap
  `cudaLaunchCooperativeKernel` or `cudaLaunchKernel`'s C entry point;
  native launches are checked here through nsys instead. Part 2's plan
  comparison against `exl3-op-plan.json` needs one or the other.
- The over-read verdict holds for the kernels and shapes swept and
  probed ([above](#placements-alignment-and-over-read)); the artifact
  format keeps it as a measured fact of these fixtures, not a rule for
  other rates, codebooks or kernels.

## Reproduction

On `spark-b`, with P0's reference assets (`exl3-reference-20260922`,
`p0-20260925/build-cache-nvcc134` and `cuda-bridge-torch`) and the
reference image copied from `spark`:
1. `native_plan.py caches DIR/frozen` writes P0's four frozen caches.
2. Per fixture and arm, copy the frozen cache and run `run_reference.sh`
   three times: a tuning pass (`--no-weights`, discarded), the measured
   pass (`--profile`) and its repeat.
3. `native_plan.py plan ref-F-A.json --out plan-F-A.txt`. Since the
   challenge round each reconstruction slice's plan names the GEMM its
   pin was recorded for (kind, m, k, n, ldc), which the native side
   requires; the recorded runs used the earlier format, and a re-run of
   all four arms with the new plans, in `malloc` and `flush-end`
   placement, was again exact in every case (`spark-b`, 2026-09-27).
4. `jitllm_exl3_linear_sweep --artifact ART --fixture F --plan plan-F-A.txt
   --out native.jsonl --placement P` for each placement, then
   `compare.py ref-F-A.json native.jsonl --repeat repeat-F-A.json`.
5. `nsys profile -t cuda` of one sweep per arm, `nsys export --type
   sqlite`, then `launches_compare.py ref-F-A.json trace.sqlite`.
6. `sass_compare.py --reference exllamav3_ext.so --port
   jitllm_exl3_linear_sweep --launches ref-*.json`, and `record.py DIR
   --out results.json`.

[`results.json`](results.json) holds per fixture every linear's weight and
reconstructed-weight SHA-256s; per arm its environment, libraries, final
tuning cache (bytes and SHA-256), every case's plan and final-output
SHA-256, the comparisons and the launch verdict; and the SASS comparison. Raw runs (every
stage's hash, the launch profiles) stay on `spark-b` under
`~/.local/share/jitllm/p3a-20260927`.
