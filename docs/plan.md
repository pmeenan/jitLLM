<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note. When a milestone exits, its section shrinks
to a short summary and its task-by-task history moves to a record beside this
file ([M0](m0-record.md), [M1](m1-record.md), [M2](m2-record.md)).

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `done`

Ran 2026-09-20 to 2026-09-23 and exited on the owner's approval of the plan.
M0 turned the [design brief](ideation.md) into the vision, the triaged feature
matrix, the approved architecture and decisions D-001–D-069, and gathered the
evidence they rest on: toolchain, VMM, I/O and interconnect spikes on the
Sparks; the llama.cpp, EXL3, image and MiMo reference runs; the measured A→B→A
reference cycle; the paging-feasibility study; and the v0 artifact layout
study. The [M0 record](m0-record.md) keeps each task's outcome, evidence links
and caveats. Everything M0 left open is owned by the milestone exits below.

## Milestone ladder

Rewritten from the provisional ladder at the end of M0 (2026-09-23; see the
[M0 record](m0-record.md)), then re-sequenced after M2 by the owner on
2026-09-27 (D-087): the fast full model swap, the owner's first real work,
comes first, on one Spark at M3 and on two at M4, and everything after moved
back two places (the old M3 is M5, M4 is M6, M4a is M6a, and so on to M8,
now M10). After the swap work the order is by risk: one resident model end
to end, with the importer and the full front door, at M5; the first useful
product at M6 (A→B→A with retention); configured placement at M6a;
demand-paged MoE with the first daily-driver models at M7; sharding under
pressure and failure at M8; performance and the remaining decoding modes at
M9; and the remaining product scope with the first tagged release at M10.
Each milestone leaves a usable, testable result. None has a promised date.

| Milestone | Result | Needs |
| --- | --- | --- |
| M1 Bootstrap | Pinned SDK, builds, local check gate, package skeleton, confined-job proof | M0 (done) |
| M2 Resource core | Catalog, admission and leases on a fake backend and a Spark; the backend proof settles the operation contract | M1 (done) |
| M3 Single-Spark fast swap | DeepSeek V4 Flash, Qwen3.8 Flash Next and Qwen-Image-2.1 swap A→B→A on one Spark, aiming at ~10 s to first token, as correct, fast and lean as their references | M2 |
| M4 Two-Spark fast swap | GLM-5.3 Flash, then DeepSeek v4.1 Flash, sharded over both Sparks in the same cycle | M3 |
| M5 One resident model | Importer, verifier, the three client protocols, TLS and management on the small fixtures | M4 |
| M6 First useful product | A→B→A with partial retention; switching policy chosen from measurement | M5 |
| M6a Configured placement | Conductor, enrolled nodes, whole-model placement and routing | M6 |
| M7 Demand-paged MoE | Exact expert paging; Gemma 4 and Ornith as daily drivers with reasoning and constrained output | M6 |
| M8 Sharding under pressure | Sharded execution correct under asymmetric pressure, cancellation and failure, with coordinated admission | M6a, M7 |
| M9 Performance | D-036's benefit target on a library larger than memory; the remaining speculative and diffusion decoding | M7; M8 before exit |
| M10 Product and release | Dashboard, remaining API scope, signed apt repository, first tagged 0.x release | M9, for the release |

How to read the milestones below:

- **Exit criteria are the contract.** Scope lists say what a milestone
  builds; its entry work refines them into tasks. An exit criterion changes
  here, visibly and with the owner, never by drift.
- **Detail lives in the linked designs.** Where a gate cites a matrix or an
  acceptance section, every row that section assigns to the milestone is
  part of the gate.
- **Evidence rules apply to every gate.** Thresholds are declared before the
  measurement they judge, and inconclusive comparisons do not pass (D-036).
  Every result names the check tiers and hosts that produced it (D-061,
  [workflow.md](workflow.md)). No performance gate is met by giving up
  correctness.
- **Heavy path.** Work in a [blast-radius area](workflow.md#blast-radius-changes-get-the-heavy-path-by-default)
  passes its adversarial challenge before the milestone exits.
- **Support is earned per checkpoint.** From M3 the
  [support matrix](model-support.md) records
  what each exit validated; a milestone exit names its configurations.
- **Release.** The first tagged 0.x release (an owner-signed tag, D-062) is
  cut at M10's exit, after every milestone has exited (owner, 2026-09-23). The
  repository is already public (D-061); until then it carries only `-dev`
  builds.

## M1 — Bootstrap  `done`

Ran 2026-09-23 to 2026-09-24 and exited on the owner's word. M1 built the
pinned SDK and its reference container (D-070), the CMake presets and build
tasks, the source lock with GoogleTest and toml++ (D-057, D-073), the local
check gate (`check`, `check:full`, `check:spark`; D-061), license and
provenance records (D-071), versioning (D-062), `jitllm doctor` (D-072), the
node configuration and storage-role checks (D-073), and the arm64 package
with `jitllm-runtime`, `jitllm.service`, crash handling and confined jobs in
delegated cgroups (D-074), validated on `spark`. The
[M1 record](m1-record.md) keeps each item's outcome, verification and
hand-offs. What it handed on: the GPU, VMM, I/O and ARM stress suites in
`check:spark`, a system-call filter with io_uring, and a single reaper for jobs
started from other threads (M2; the last two moved to M5); front-door, TLS and switching-policy keys
(M5, M6); the importer on the confined-job mechanism and spill file names
(M5); cluster-member checks of credential files and enrollment records
(M6a); drain-before-restart upgrades and the apt repository (M10).
Milestone numbers here follow D-087's renumbering.

## M2 — Resource core and backend proof  `done`

Ran 2026-09-24 to 2026-09-27 and exited on the owner's word. M2 built the
resource core: the catalog, commitment ledger, LRU victim baseline and
materialization planning (D-006, D-007), admission with D-069's pauses
(D-050), D-048's task lanes and scheduler loop, and the providers with
their fakes, CUDA VMM and io_uring (D-026). Its backend proof (P0–P6) ran
GGML's and ExLlamaV3's kernels under jitLLM's dispatch (D-077, D-080),
selected by plan (D-053), with cuBLAS on a jitLLM handle (D-076): native
Qwen2.5-0.5B FP16 matches the bridge bit for bit and both EXL3 fixtures
match their approved record and bounds, paged at disk speed through a
host-VMM landing zone into device VMM (D-081), evicted, written back,
restored and relocated bit-identically, FP16 and EXL3 alternating on one
node. D-086 records the operation contract; D-033 is retained; D-068's
shapes are expressible; the `native` build also targets discrete `sm_86`
GPUs (D-082); and D-085 made performance and memory coarse end-to-end
checks, so BP-F2 did not run. The [M2 record](m2-record.md) keeps each
item's outcome, evidence and caveats, and the gate. What it handed on,
in D-087's numbering: BP-F4's per-token host cost, the D-033 handoff of an
evicted model's backing, and ReconGemm's descriptors prepared once per
bound GEMM (the swap work, M3 and M4); end-to-end parity of each engine
with its reference on the small fixtures (BP-F3), the importer,
suballocation with state blocks, D-050's moved rows (suballocation holes,
stalled-client termination, every queue full at once, capacity-loss
injection), and M1's io_uring system-call filter and single job reaper
(M5); victim selection on a miss, spill as retention with D-055's spill
format, the cohort pause with several paused peers, fork and
copy-on-write, cached-state promotion and fast-swap validation on the
discrete GPU (M6); the runtime closure-excess check (M7); repeated
speculation with prefetch (M9); and, with no milestone yet, `spark-native`
sanitizer presets and the EXL3 memory outside the catalog in the paged
harness.

## M3 — Single-Spark fast full swap  `in progress`

Goal: one user swaps among three large models on one Spark, A→B→A, and each
swap reaches its first token in about 10 s: as close to that as we can
get, and at most about 20 s at exit. A's conversation state comes back
without a re-prefill, and each model is as correct, as fast and as lean as
its reference. A full swap: the outgoing model leaves, and no expert is
demand-paged. The owner's first real work after M2 (D-087).

**Entry:** M2 exit. Before a model's first native evaluation, its
checkpoint, reference engines and their configurations are pinned and their
licenses recorded ([licensing.md](licensing.md); a model's weight license
is recorded for information and gates nothing, D-087), and its prompt set,
perplexity text, or image prompts and seeds, are fixed with the bounds
below. The provenance of llama.cpp's generated Unicode tables is cleared
under D-017 before the native tokenizer is adopted
([first-slice.md](first-slice.md)): traced to UCD 15.1.0 on 2026-09-28;
jitLLM generates its own tables, and the owner accepted D-088 on
2026-09-28.

**Models, in this order.** Each runs its reference's quantization.

1. **DeepSeek V4 Flash 0731**, GGUF UD-Q2_K_XL
   (`unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93`, MIT; 96,832,508,352 B
   in 3 shards). It replaces the `e3aa0d6a` revision on the Sparks, which
   the M0 baselines and artifact-format.md's example used: same shard sizes,
   new hashes, a download of about 97 GB per node. 0731 has no usable MTP
   (llama.cpp PR #25784), so its drafter is DSpark.
2. **Qwen3.8 Flash Next** as Mia's single-Spark build: NVFP4 routed
   experts, MXFP8 attention and shared expert
   (`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`, a mirror of
   `local-inference-lab`'s; 105,935,742,983 B, 26.8 GiB of it the packed
   PLE table).
3. **Qwen-Image-2.1** in BF16, like diffusers: text encoder (Qwen3-VL-8B),
   single-stream DiT and VAE (Qwen Research License, non-commercial,
   recorded for information). GGUF quantizations may follow later if
   memory matters (owner, 2026-09-28).

**Oracles and comparators.** Each model's correctness is judged only
against its same-format oracle. A cross-quantization comparison reports
speed and memory only, never correctness, and is labeled as such wherever
it appears.

| Model | Correctness oracle (same format) | Performance comparators | Cross-quantization (speed and memory only) |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | llama.cpp on the same GGUF | llama.cpp | vLLM or SGLang where they support it, unless on the same GGUF |
| Qwen3.8 Flash Next | Mia's vLLM on the same NVFP4 checkpoint, in the recipe's deterministic mode (its default launch is not repeatable) | Mia's vLLM | TensorFold (MLX 4-bit); llama.cpp on a GGUF |
| Qwen-Image-2.1 | diffusers, BF16 | diffusers, BF16 | stable-diffusion.cpp's GGUFs, also for image quality |

**Scope:**

- [x] **Provenance and licenses** (D-017, D-080): pin and audit
      `MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark` and decide whether
      its AGPL scripts may be run as a baseline recipe; record each
      checkpoint's pins and license hashes, the 0731 GGUF's among them,
      and the Qwen NVFP4 card's Apache-2.0 beside the base model's Qwen
      Community License 1.0 (weight licenses are recorded for information
      and gate nothing, D-087); pin TensorFold and its checkpoint.
      *Done 2026-09-28* ([licensing.md](licensing.md#fast-swap-models-and-baselines-m3-m4),
      [pins](experiments/fast-swap/pins.json)): the recipe pinned at
      `b8439110` with all 70 tracked files classified, and run unmodified
      as a baseline (owner, 2026-09-28); TensorFold at `beddbb7b` and its
      checkpoint cleared (the pin moved to `71377a53`, 0.3.6.2, the same
      day, with the same MIT terms); the four checkpoints, the vLLM images
      and the llama.cpp pin recorded; the vLLM, FlashInfer and CUTLASS parts of
      Mia's NVFP4 and MXFP8 path identified with their licenses, a
      CuTe-DSL kernel's runtime among them under NVIDIA's proprietary
      terms. The checkpoints are on both Sparks' NVMe
      ([environment.md](environment.md#m3-model-store-2026-09-28)).
      The default vLLM image's commit (`8e685d198`) and the kernels its
      engine log selects were read in the baselines item.
- [ ] **Baselines,** installed and run on the Sparks by us: MiaAI's
      configurations, TensorFold, llama.cpp for the GGUF, and vLLM or SGLang
      where they support these models; for the image, diffusers in BF16 as
      the speed and format reference (the fastest measured), and
      stable-diffusion.cpp's GGUFs as an additional quality and format
      comparison. Each reference's load or swap, prefill, decode
      and peak memory are measured on the same prompts as jitLLM's. Mia's
      Qwen3.8 cold start (creator-reported 10 min 51 s to `/health`) runs
      once, stated as needed under D-085, and is recorded as a measurement;
      its prefill and decode are then measured on the same warm server.
      *Measured 2026-09-28 on `spark`*
      ([baselines](experiments/fast-swap/baselines.md), on the fixed
      [prompt set](experiments/fast-swap/prompts.json)): llama.cpp on
      DeepSeek 0731 (load 92–104 s, prefill ~350 tok/s at 8K, decode 19.9
      tok/s, 30.8–31.9 with DSpark, peak 93–105 GiB) and one llama.cpp swap
      cycle with Qwen3.8's GGUF; Mia's vLLM (cold start 13 min 11 s, prefill
      2,066 tok/s at 8K, decode 37.9 tok/s with MTP 3 and 25.1–25.3 without,
      peak ~103 GiB); TensorFold and llama.cpp on Qwen3.8 as
      cross-quantization comparators; diffusers on Qwen-Image (1.26 s per
      step, 52.6 s per 1024², 40-step generation). Greedy and top-5 logprob
      references for both LLMs are saved beside the report, and the
      reference image on both Sparks by hash. Mia's default launch is not repeatable under greedy decoding,
      so Qwen3.8's oracle is the recipe's deterministic mode (MTP off,
      `VLLM_QSA_DET_TOPK=1`, `VLLM_MOE_DET_FINALIZE=1`), which took a second
      launch (10 min 52 s). vLLM or SGLang on DeepSeek 0731 is not
      available on one Spark (no GGUF path for this quantization; the
      native checkpoint does not fit). Open: stable-diffusion.cpp's GGUFs.
- [ ] **Import:** M0's Python prototype importer writes the D-056 artifacts
      for the three models, including NVFP4 and MXFP8 tensors and the image
      pipeline's BF16 components. The C++ importer and verifier stay in M5.
      *DeepSeek V4 Flash 0731:* `import_m3.py` (the pinned prototype, its
      sources checked against the M3 pins) wrote artifact `8a355bfb…`,
      96.84 GB, 11,053 groups, the plan of the D-056 worked example, in
      11 min 16 s on `spark-b`
      ([dsv4-native](experiments/dsv4-native/README.md)).
      *Qwen3.8 Flash Next (Mia's NVFP4):* `import_m3.py` with
      `modelopt_qwen38.py`, which repacks losslessly as it writes (experts
      into GGML NVFP4 blocks, the n-gram table into 90-byte rows, linear
      attention's value heads into tiled order; MXFP8 kept as is), wrote
      artifact `67617f87…`, 103.9 GB, 24,627 groups, in 9 min 16 s on
      `spark-b` ([qwen38-native](experiments/qwen38-native/README.md),
      [artifact-format.md](artifact-format.md#qwen38-flash-next-modelopt-nvfp4-and-mxfp8));
      since the prefill work the experts go into the CUTLASS layout the
      grouped GEMM reads, so nothing is rewritten at load or on a swap:
      artifact `c4fb47a9…`, 103.8 GB, in 12 min 49 s with its verification.
      *Qwen-Image-2.1:* one artifact per component and a composition naming
      them (D-089; [artifact-format.md](artifact-format.md#compositions)):
      `import_m3.py component` wrote the text encoder (17.53 GB), denoiser
      (14.23 GB) and VAE (1.35 GB, F32) and `compose` their composition
      `eca21baa…` on `spark` in under three minutes; `artifact/composition.h`
      reads it natively ([qwen-image-native](experiments/qwen-image-native/README.md)).
- [ ] **Kernels and the source lock** (D-053, D-057, D-077): the pinned
      llama.cpp has much of what the models need (quantized matmul and
      `mul_mat_id`, MoE routing, the lightning indexer, `dsv4-hc`, gated
      delta net and `ssm-conv`, prefill flash attention at head dimensions
      256 and 512, and the image's 3D convolution and VAE operations), but
      the narrowed GGML build compiles none of it. Widening it is a
      source-lock change on the heavy path, with D-057's gates and the
      workstation tier D-084 requires. Qwen3.8's NVFP4 and MXFP8 matrix
      products take the fastest correct implementation from any source,
      chosen per operation by a quick A/B (D-085): GGML's NVFP4 MMQ (MXFP8
      in GGML is not verified), vLLM, FlashInfer or CUTLASS kernels, or our
      own, with licenses handled per D-080.
      *GGML widened for the two LLMs* (`kernels/ggml/ops_ext.h`): quantized
      MMVQ and MMQ products and `mul_mat_id` for DeepSeek's GGUF types (Q8_0,
      Q4_K, Q5_K, Q6_K, IQ2_XS, IQ3_XXS, MXFP4); tensor-core flash attention
      at head dimensions 256 and 512, 8 query heads per KV head, with sinks
      and DeepSeek's sparse gather; the lightning indexer, hyper-connections,
      gated delta net and `ssm_conv`; RoPE with offsets, YaRN and IMROPE,
      forward and back; argsort, top-k and the elementwise and row
      operations. Each is checked on the host in every profile and matches
      an FP64 reference on a GB10 within upstream's test-backend-ops bounds.
      *Qwen3.8's formats, by a quick A/B*
      ([qwen38-native](experiments/qwen38-native/README.md#kernel-ab-d-085)):
      NVFP4 experts on GGML's MMVQ and MMQ (the NVFP4 MMQ instance unit added
      to the lock); MXFP8 products on jitLLM's own vector product up to 8
      rows and otherwise dequantized to BF16 for cuBLAS (since the second
      prefill pass, the reference form's; the default takes CUTLASS's
      MXFP8 GEMM, 1.9–2.5× cuBLAS's at 4,096 rows); the n-gram table's
      NVFP4 rows on jitLLM's own lookup (`kernels/ggml/jitllm_ops.h`). Each
      matches an FP64 reference built from the format's dequantization on a
      GB10. For prefill, CUTLASS 4.7.1's NVFP4 grouped GEMM (BSD-3, a new
      lock component, headers only) now takes the routed experts over a
      CUTLASS layout the importer writes, with jitLLM's own vector products
      over it for decode; 1.16–2.84× GGML's MoE block at 512 to 8,192
      tokens
      ([qwen38-native](experiments/qwen38-native/README.md#prefill-d-085)).
      *The image pipeline's* (Qwen-Image-2.1, BF16, chosen per operation by
      speed): cuBLAS BF16 products, jitLLM's own FlashAttention-2 kernel
      (3.37 ms per denoiser block, as PyTorch's flash kernel; GGML's
      tensor-core kernel, built for D = 128 without head grouping, took
      19.9 ms) and jitLLM's fused BF16 kernels for the norms, modulation,
      rotary embeddings, residuals and the VAE, each rounding where
      diffusers rounds (`kernels/image`,
      [qwen-image-native](experiments/qwen-image-native/README.md)); the
      VAE's convolutions (its causal 3D convolutions are 2-D at one frame)
      were im2col and cuBLAS and are, since the speed slice, jitLLM's own
      implicit GEMM, so GGML's were not needed; the products are pinned
      cuBLASLt algorithms. The image's operations are declared in the
      registry and run through a bound plan. No source-lock change.
      Open: the vector attention at D = 256, which
      upstream picks for Qwen3.8's decode below 8,192 cells (the MMA kernel
      runs it meanwhile). CUB stays out although its licenses are
      cleared (D-091): upstream's CUB top-k is 2–3× faster for one row but
      under 1% of a decode step below 1M positions, and 3.6–71× slower for
      prefill's many rows (measured 2026-09-28,
      [licensing.md](licensing.md)).
- [ ] **Model graphs and state** (pulled from M7 and M9): DeepSeek V4's
      compressed sparse attention with its indexer (CSA/HCA) and mHC;
      Qwen3.8's QSA, hyper-connections and Gated DeltaNet layers; the
      resident MoE execution both need (pulled from M7; no demand-paged
      experts); Qwen-Image's text encoder, DiT and VAE, releasing each
      component outside its phases. Each model's KV, indexer and recurrent
      state has a state adapter with spill and restore coverage (RE-004,
      RE-007).
      *DeepSeek V4 Flash, native and resident* (`model/dsv4.h`,
      `kernels/ggml/dsv4_graph.h`, `jitllm_dsv4_exec`): llama.cpp's
      `deepseek4.cpp` graph (CSA with the lightning indexer and top-k, HCA,
      the window, sinks, q/o LoRA and output groups, mHC with its Sinkhorn
      comb, 256 experts top-6 plus the shared one with sqrtsoftplus and
      noaux_tc routing, the hash-routed layers, and the head) planned
      unfused through the registry, from the artifact on `spark-b`. Against
      llama.cpp on the same 0731 GGUF with its fusion off, every logit of
      the 8 prompts' 256 greedy steps and every token's perplexity NLL is
      bit-identical, and the free-running continuations are identical
      ([dsv4-native](experiments/dsv4-native/README.md)). The KV, indexer
      and compressor state is explicit and bounded (`Dsv4StateLayout`, three
      D-068 representations); its spill and restore are the swap path's.
      *On the paged node* (`engine/dsv4_runner.h`): each chunk a device
      job over its leased closure (D-086), 46,232 extents paged through the
      landing zone, each layer's expert slab as 2 MiB pages whose contents
      land in pieces; all 8 prompts' 32 steps bit-identical to the resident
      harness's logits ([swap](experiments/fast-swap/swap.md)).
      Open: the executed-plan record against llama.cpp's, and the other
      models.
      *Qwen3.8 Flash Next, native and resident* (`model/qwen38.h`,
      `kernels/ggml/qwen38_graph.h`, `jitllm_qwen38_exec`): llama.cpp's
      `qwen4exp.cpp` operation plan (hyper-connections, the n-gram
      embedding layer, Gated DeltaNet, QSA with its indexer and budget, 512
      experts top-10 plus the gated shared one, the head) over the artifact
      on `spark-b`. Against Mia's vLLM (deterministic, MTP off) on the same
      checkpoint: 180 of 192 teacher-forced greedy steps agree, the other 12
      at oracle margins of at most 1.0 nats, within the 95th percentile of
      jitLLM's own kernel-to-kernel margin noise (a bound set after the
      first comparison, so not pre-registered; it fails at the 90th);
      perplexity 14.43 against 14.66 (−1.5%), top-1 accuracy equal. The
      KV, indexer, recurrent and convolution state is explicit and bounded
      (`Qwen38StateLayout`, three D-068 representations), and a spill and
      restore of it is bit-identical. Decode 0.99× the oracle's; prefill,
      after fused hyper-connection, MoE-output and Gated DeltaNet kernels
      and CUTLASS's grouped GEMM for the routed experts, 0.91× at 8,192
      tokens in 8,192-row chunks, within D-085's 10% gate (0.89× in
      4,096-row chunks misses it; 1.40× at 2,048, 1.10× at 512); peak
      memory 0.97× at a 4,096-token context against vLLM's 262,144, 1.07×
      while prefilling 8,192 tokens in one chunk
      ([qwen38-native](experiments/qwen38-native/README.md#prefill-d-085)).
      A second prefill pass under D-085's speed before bit exactness made
      the graph's default a fast form (the first pass's fusions stay as
      its reference form): the MXFP8 products on CUTLASS's MXFP8 GEMM over
      activations quantized to MXFP8, as the oracle runs them, and fused
      hyper-connection, routing, Gated DeltaNet and QSA kernels, QSA's
      selection making its mask on the device. Prefill 1.41× the oracle's
      at 8,192 tokens in 8,192-row chunks, 1.38× in 4,096-row chunks, 2.02×
      at 2,048, 1.56× at 512; decode unchanged; perplexity −0.8% to −1.2%; one of
      the 192 greedy steps now misses the near-tie bound (jitLLM's own
      margin there is 0.18 nats in the reference form; accepted by the owner,
      2026-09-28, as a known divergence)
      ([qwen38-native](experiments/qwen38-native/README.md#prefill-second-pass-speed-before-bit-exactness)).
      *On the paged node* (`engine/qwen38_runner.h`): chunks as device
      jobs over leased closures, the expert slabs as DeepSeek's pages, the
      28.8 GB n-gram table never resident but read by rows before each
      chunk (4 KiB-aligned direct reads, D-035's evidence recorded: 486×
      fewer bytes than whole chunks), the state spilled and restored
      through the swap path; the six prompts' 32 steps bit-identical to the
      resident harness's logits ([swap](experiments/fast-swap/swap.md)),
      again since the prefill work with the fused graph and CUTLASS's
      grouped GEMM over the CUTLASS-layout artifact, which it pages in as
      is (swaps into Qwen3.8 6.9–7.9 s, against 7.4–8.8 s before).
      Past 2,051 attended cells the QSA indexer's top-k is not repeatable
      (RE-031: GGML's radix select picks among ties by timing).
      Open: a deterministic top-k.
      *Qwen-Image-2.1, native* (`model/qwen_image.h`,
      `jitllm_qwen_image_exec`): the text encoder (Qwen3-VL's text path, the
      system turn dropped), the block-causal DiT with its text K/V prefix
      cache and the flow-matching Euler scheduler, and the VAE decoder, from
      the composition on `spark`, each phase bounded and, with
      `--phases released`, each component loaded for its phase and freed
      after it. Against diffusers BF16 at the fixed teapot prompt (1024², 40
      steps, seed 42, diffusers' initial latents), every pre-registered
      bound passes: the image at PSNR 41.8 dB and SSIM 0.996 (bounds 32 dB,
      0.98; an FP32 DiT gives 38.8 dB); full generation 36.9–37.0 s against
      diffusers' 52.6 s, 0.89 s per step against 1.26 s, peak memory 31.2
      GiB against 43.4 GiB resident and 15.9 GiB released
      ([qwen-image-native](experiments/qwen-image-native/README.md)).
      *On the paged node* (`engine/qwen_image_runner.h`): each component
      a set of extents, each phase a device job over its own component's
      closure, a generation one request leasing all three (D-093); the
      image pixel for pixel the resident
      harness's, generated before and after swaps
      ([swap](experiments/fast-swap/swap.md)).
      *Speed* (the same BF16 numerics): both harnesses run one pipeline
      dispatched through a plan bound against the registry (D-053,
      `kernels/image/pipeline.h`); pinned cuBLASLt products (bit for bit
      `cublasGemmEx`'s), the gated residual fused with the next norm, the
      prefix cache read in place and the query norm fused into the
      attention (all bit for bit the M3 slice's denoiser), implicit-GEMM
      VAE convolutions and a CUDA graph per step: full generation 33.4 s
      against 36.2 s before on the same host (0.64× diffusers' 52.6 s), a
      step 0.82 s, decode 0.36 s; image 41.77 dB, SSIM 0.996, repeatable
      ([qwen-image-native](experiments/qwen-image-native/README.md#speed)).
- [ ] **Resident expert layout** (the initial choice pulled from M7):
      repacked expert groups get executable views that GGML's `mul_mat_id`
      and the NVFP4 path's grouped GEMM accept with every expert resident.
      Pointer-table or uniform-stride dispatch is chosen per format, by a
      quick A/B where both are viable (D-085), and proven by one MoE layer
      whose output is bit-identical to the reference layout's. Compaction
      and demand-paged dispatch stay in M7.
      *GGML, DeepSeek V4:* uniform stride over a per-layer slab with stock
      kernels ([artifact-format.md](artifact-format.md#executable-views));
      no A/B, since the pointer table needs a kernel patch and gives a
      resident model nothing. All 43 layers' routed products (387 cases,
      MMVQ and MMQ) equal the reference layout's bit for bit.
      *GGML NVFP4, Qwen3.8:* the same layout, S = 2,768,976 bytes; the down
      projection's short rows read their padding inside the slab. All 48
      layers' routed products (432 cases) equal the packed reference's bit
      for bit ([qwen38-native](experiments/qwen38-native/README.md)). A
      grouped-GEMM path brings its own view: the harness rewrites each slot
      in place into CUTLASS's layout (the same bytes, permuted; every layer
      converts back to the loaded bytes exactly), which the grouped GEMM
      and jitLLM's decode products read
      ([artifact-format.md](artifact-format.md#executable-views)).
- [x] **Tokenizer and chat templates** (pulled from M5; D-067): the native
      tokenizer, renderers for each model's pinned template (DeepSeek's
      upstream ships Python encoding scripts, not a template), stop rules,
      and greedy and seeded sampling. Landed in tests
      ([tokenizer.md](tokenizer.md)): token for token with llama.cpp and
      Hugging Face on a 92-item corpus for all three models, and both
      templates byte for byte; shipped binaries may link them (D-088,
      accepted 2026-09-28). The output
      parsers and the request mapping are the chat route's.
- [ ] **Speculative decoding in the core** (pulled from M9; D-068): Qwen3.8's
      MTP layer and DeepSeek's DSpark drafter, with verify and rollback that
      keep only the accepted prefix, and forced draft rejection for the
      exit's speculation checks (moved from M9). Every published Mia and
      TensorFold decode number uses speculation.
      *DeepSeek's DSpark landed* ([dspark](experiments/dspark/README.md),
      D-092): the drafter its own artifact, binding the target's token
      table and head; its KV ring a D-068 representation; a verify of the
      anchor and 3 drafts on a row-invariant plan, so greedy speculation is
      bit-identical to greedy decoding; rollback by restoring the bytes a
      verify saved; draft and verify one job, both replayed as graphs.
      All four speculation checks pass on `spark-b`: greedy bit-identical
      (8 prompts), forced rejections equal to their control state by state
      (91 steps), rollback across a swap, and sampled speculation (total
      variation 0.0005–0.027, bound 0.1; plain sampling's run took 10.6
      min at a host load of 16–20, over D-085's 10). Decode 27.9 / 29.3
      tok/s on `prose` / `code` against llama.cpp's 30.8 / 31.9 with the
      same drafter (0.91× / 0.92× on the medians, a narrow pass: one
      `prose` repeat of three was 0.87×; acceptance 0.55 / 0.58), peak
      memory equal (104.7 GiB). Since then, with each generation a
      request (D-093) and the runtime wake (D-094): 29.7 / 30.8 tok/s,
      0.96× / 0.97× ([dspark](experiments/dspark/README.md#performance-and-memory)).
      Open: the verify's device time.
      *Qwen3.8's MTP layer landed* ([qwen38-mtp](experiments/qwen38-mtp/README.md);
      D-089's and D-092's notes). The MTP block is a drafter artifact of its
      own (1.6 GB, imported in 6.7 s; the target is unchanged) that binds
      the target's token table and head. Its caches and the target's
      streams are a D-068 state.
      The verify is batched (the owner's speed before bit exactness). It
      never writes the recurrent, convolution or n-gram state: a commit
      kernel replays the kept rows into them, and the rejected rows' KV
      and indexer cells are restored from a snapshot.
      On `spark`, at depth 2 with a 65,536-row draft head, it decodes at
      42.46 / 39.09 tok/s (`prose` / `code`, medians of three) against Mia's
      MTP 3 at 37.85 / 37.85: 1.12× / 1.03×. Acceptance is 0.631 / 0.539
      against Mia's 0.42, and the peak `MemAvailable` drop 75.8 GiB against
      103.4, not like for like: the 28.8 GB n-gram table stays on the SSD
      here (read by rows, D-035), and vLLM's 16.2 GiB KV pool is sized for
      its concurrency.
      The checks:
      - greedy on 8 prompts: every token is the plain argmax, or a near-tie
        within 1.0 logit (9 near-ties, 0 violations);
      - forced rejections at depths 2 and 3: every state equal to the
        control's;
      - sampled speculation: total variation 0.009–0.040 (bound 0.1);
      - rollback across a swap (Qwen3.8 out for the FP16 fixture and back,
        a commit owed across it): all 97 steps' states, 160 tokens and
        their logits equal the unswapped control's.
      The 1.0 bound is qwen38-native's, not dsv4-decode's later rule (the
      verify's own noise, p99 0.23–2.39 per prompt, measured afterwards);
      it is kept as the stricter test.
- [ ] **The swap path:** evict the outgoing model and hand its backing to
      the incoming one (D-033's handoff, pulled from M6; D-081), with
      page-in through the landing zone overlapping the rest. Creating and
      mapping backing costs about 70–76 µs (`cuMemCreate`) and 40–43 µs
      (`cuMemSetAccess`) per 2 MiB extent (measured medians), about 5 s of
      serial work for DeepSeek's ~46.2k extents (computed), so reusing the
      evicted model's backing matters; unmapping the outgoing extents and
      one access call per contiguous range are not yet measured. Large
      sparse lookup tables (Qwen3.8's PLE) may stay on the SSD with rows
      paged on demand (D-035); everything else is resident before the first
      token. A's KV and recurrent state are spilled on swap-out through M2's
      write-back path and restored on return, with no re-prefill; that cost
      counts in the swap time. D-055's named spill format and the retention
      policy stay in M6.
      *Core landed, with DeepSeek* ([swap](experiments/fast-swap/swap.md)):
      the handoff (an eviction parks its unmapped backing, charged; a load
      of the same class and size takes it; the rest is released when the
      evicting task ends), state spill and restore through the zone, and
      the zone's copies on a copy lane of their own with fences from a
      pooled set of events (RE-029: `cuEventCreate` blocks too). DeepSeek
      (8,192 context tokens) ↔ the FP16 fixture on `spark-b`: B→A 7.36 s
      prepared, 7.44 s first use, A→B 1.79–1.86 s, DeepSeek evict-all
      reload 9.07 s, all from the request to the first token; page-in at
      13.4 GB/s; A resumed after B bit-identical to A never swapped. The
      handoff saves 1.3–1.4 s of a 46k-extent eviction.
      Graph restore: DeepSeek's decode graphs survive swaps (D-090).
      *M3's pairs* ([swap](experiments/fast-swap/swap.md#results-m3s-swap-pairs-spark-b-2026-09-28)):
      every ordered pair of DeepSeek, Qwen3.8 (its n-gram table paged by
      rows, D-035) and Qwen-Image (each phase over its own component),
      A→B→A with 8K and 0 context, first use and prepared, on `spark-b`:
      all 32 swaps under the ~10 s goal, the worst LLM↔LLM swap 9.38 s
      (8.76 s prepared and 9.04 s first use at 8K context; an earlier run
      of the same path reached 9.66 s; the margin is the SSD's at-rest
      rate for 75–97 GB), the image's first step 5.0–6.3 s after a swap
      from an LLM; an LLM A's restored state byte-identical
      and its continuation bit-identical to the same state's, the image
      regenerated pixel for pixel. Open: overlapping eviction with page-in,
      Qwen3.8's and the image's graphs, and a deterministic top-k for
      Qwen3.8 (RE-031).
      *Through `jitllm-runtime`* (D-096, [swap](experiments/fast-swap/swap.md#through-jitllm-runtime-d-096)):
      all three models registered in one process, every fast path on
      (speculation, so the drafters page in with their targets: DeepSeek
      108.4 GB, Qwen3.8 77.0), every ordered pair A→B→A, 8K and 0 context,
      first use and prepared on `spark-b`: all 32 swaps under the ~10 s goal
      and exact (A's state digest and its 16 continued tokens and logits,
      B's output, the image's pixels, graphs replayed after prepared
      returns); **the worst LLM↔LLM swap 9.72 s** (Qwen3.8 → DeepSeek, first
      use, 8K; 9.64 s prepared), 0.8 s of it paging DSpark's 10.9 GB; the
      image's first step 5.2–6.0 s after a swap; peak 108.7 GiB. With
      speculation off the LLM pairs take 7.5–8.9 s, within 0.6 s of the
      harness's. That is M3's swap table in a running process, against
      llama.cpp's 77 / 104 s, Mia's vLLM's 13 min 13 s, TensorFold's 141 s
      and diffusers' 212 s.
- [ ] **CUDA graphs for decode** (pulled from M9): captured per model and
      plan and replayed after swaps that restore every extent at the same
      virtual addresses, with setup and tuning state restored the same way.
      This reopens D-086 and needs the relocation proof first. BP-F4's
      per-token host cost is measured on these models (about 1.9 µs per
      launch measured on the fixtures; several milliseconds per MoE token
      is an estimate).
      *DeepSeek V4 Flash, on the paged node* (D-090,
      [graphs](experiments/fast-swap/graphs.md)): each one-row chunk shape
      is captured on its second step through the launch context (input
      copies, 4,972 steps, logits copy: one 5,920-node graph, about 25 ms)
      and replayed as one launch; the relocation proof pins the weights'
      and state's places in the scheduler, so a swap maps backing back at
      the addresses every graph names. Replayed steps are bit-identical to
      launch by launch and to the resident harness (so to llama.cpp with
      fusion off), and A resumed after B replays graphs captured before the
      swap without capturing again. Decode 19.05–19.53 tok/s against
      18.13–18.82 launch by launch; llama-bench's tg64 is 20.62 with fusion
      and graphs on, 20.04 with fusion off; the step's device time alone is
      48.3–48.6 ms, and the rest is the paged node's per-step round trip.
      The job's host time per token fell from ~41.5 ms (launches waiting on
      a full stream) to 0.13–0.16 ms. The owner (2026-09-28) questioned
      whether graphs are worth their complexity at 1.04–1.05×; re-measured
      once a request leases its closure once (D-093, below), they are still
      worth 1.046–1.055× and stay (D-090's note).
      *A lease per request* (D-093, [swap](experiments/fast-swap/swap.md#a-lease-per-request)):
      a request (a turn; for the image, one generation) leases its model's
      closure once and runs every chunk under it; with the lanes polling
      through a step (harness-polled) the per-step round trip fell from
      1.3–2.6 ms to 0.01 ms. DeepSeek decodes at 20.34–20.46 tok/s, 1.016–1.022×
      llama.cpp's fusion-off, graphs-on tg64 (20.02) and 0.990–0.996× its
      default (20.54), the same session (two runs); Qwen3.8 from the
      CUTLASS-layout artifact at 23.71–23.81 tok/s paged, 0.94× Mia's vLLM
      with speculation off: its job's device span alone is 1.03× vLLM's
      step, and ~1.3 ms a step is host work outside the job (the n-gram
      rows and inputs), not the lease. Everything stays bit for bit, and
      the DeepSeek ↔ Qwen3.8 swaps (first artifact) did not move. A holder
      of a request's lease is refused a wait for another's (no hold and
      wait).
      *Qwen3.8's decode graphs* ([qwen38-mtp](experiments/qwen38-mtp/README.md#performance-and-memory)):
      the n-gram row gather now reads its row count from pinned memory, so
      one graph serves every step. Verify and draft shapes are captured the
      same way. Plain decode on `spark-b` (a request lease, the runtime
      wake) runs at 25.65–25.88 tok/s against 23.84–24.05 launch by launch,
      and every step's logits are identical between the two. That is
      1.01–1.03× Mia's vLLM with speculation off (25.12 / 25.33). A step is
      37.4 ms on the device, and its host time is 0.07 ms. Across a swap
      (`jitllm_swap_pairs --a qwen38 --b image`), a prepared return
      replays the graph captured before it for every continued step,
      bit-identical to the unswapped continuation.
- [x] **RE-029's lead:** read `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`
      on the GB10, one `cuDeviceGetAttribute` call. Mia's
      `patch_ple_offload.py` reports it as 0, with `cuStreamWaitValue32`
      then blocking the host's next launch (creator-reported). The answer
      decides how page-in copies and phases share the submission lane
      during a swap. *Done 2026-09-28* ([RE-029](rough-edges.md#re-029-a-jobs-kernel-launches-can-block-its-lane-while-the-stream-is-busy--2026-09-27-status-worked-around)):
      the deprecated `_V1` attribute reads 0 and the current memory
      operations are supported; the wait does not block the next launch.
      A stream holds about 1,020 pending operations and a launch into a
      full one blocks the thread, so the swap path keeps the zone's copies
      off a thread that can launch into a full stream, or keeps each
      stream under the limit.
- [x] **The runtime wake** (D-094, [runtime-wake](experiments/runtime-wake/README.md)):
      benchmarks measure what the runtime does, not the harness's 100 ms
      of polling. On the GB10 a blocking-sync event or a host function
      woke 1.0–1.7 ms after a step's end; every sleeping thread on a
      step's path cost 0.1–0.6 ms (RE-017). The device completion lane
      now sleeps through most of a step, spins only around its likely
      ends and wakes the scheduler and the submission lane ahead of the
      completion; the scheduler polls after a step for about as long as
      its client takes to ask for the next. A synthetic step's round trip
      is 26–28 µs at 0.11–0.12 of a core while stepping and none idle
      (harness-polled: 9 µs, four cores; the old defaults: 0.56–0.67 ms,
      two). With it DeepSeek decodes at 20.42–20.50 tok/s (graphs),
      Qwen3.8 (after its fast prefill, `f9a4e0f`) at 23.85–24.09, and
      DSpark speculates at 29.67 / 30.85 tok/s (`prose` / `code`, 0.96× /
      0.97× llama.cpp's), each generation now a request; DeepSeek is
      1.020–1.024× llama.cpp's fusion-off tg64 and 0.999–1.003× its
      default in the same session, Qwen3.8 0.94–0.96× Mia's vLLM. Against
      the old window the wake costs 0.01–0.03 ms a step, inside the runs'
      spread. Prefill (8,192 tokens: DeepSeek 27.37–27.45 s, Qwen3.8
      6.84 s) and the DeepSeek ↔ Qwen3.8 swaps (7.1–8.8 s, page-in
      13.3–14.1 GB/s) did not move with the wake; every check stays exact.
- [x] **DeepSeek decode past llama.cpp** (the owner, 2026-09-28: speed
      before bit exactness, D-085's and D-092's notes;
      [dsv4-decode](experiments/dsv4-decode/README.md)): DeepSeek's fast
      plan is the default for decode, verify and draft chunks. One
      quantized vector kernel (`jitllm.vecq`) serves every product and
      reads each routed expert once for all the rows that select it (the
      batched verify), with fused routing, combine, hyper-connection
      pre-mix and compressor kernels, and PDL. A decode step drops from
      5,575 kernels to 2,558. With the runtime wake it decodes at
      21.90–22.21 tok/s against llama.cpp's 20.41 in the same session
      (1.07–1.09×; 1.10–1.11× its fusion-off arm), and DSpark at 31.6–31.7
      / 34.3–34.4 on `prose` / `code` (1.03× / 1.07–1.08× the recorded
      30.80 / 31.94). Correctness is coarse against llama.cpp: greedy
      244/256 and 250/256 equal, the rest near-ties (oracle margins under
      1.0); perplexity within 0.5%; speculation and forced rejections
      with no stale state byte; sampled speculation's total variation
      0.004–0.038 (bound 0.1). Swap, restore and repeat stay
      bit-identical. The exact plan and D-092's verify remain as
      `--exact on`. The recorded near-tie bound (6.11, twice the largest
      fast-against-reference move) is too loose to be a test; later
      slices take the 99th percentile of noise between two of jitLLM's
      own paths ([dsv4-decode](experiments/dsv4-decode/README.md#the-bound-going-forward)),
      under which one forced-run speculative token (a 3.62-nat
      disagreement) is flagged. Open: that token's diagnosis; "decisive"
      (about 1.1×) is not reached on any target; the products run at
      about 200–210 GB/s in the model against 230–245 alone, unexplained.
- [x] **TensorFold's techniques, Qwen3.8's decode gap first** (the owner,
      2026-09-28; [tensorfold-techniques](experiments/tensorfold-techniques/README.md)):
      the gap to TensorFold is format, not engine: a Qwen3.8 decode token
      reads 7.3–7.6 GB in Mia's NVFP4, MXFP8 and BF16 against 4.5 GB in
      MLX 4-bit, and jitLLM already reads a byte faster (about 202 GB/s
      against 165–178). Adopted within the format: the Gated DeltaNet state
      updated in place (`jitllm.gdn.step`; TensorFold double-buffers it),
      the hyper-connection prep across a cluster of blocks, a BF16 vector
      kernel for the mixes' one-row products, the one-row convolution
      fused, and PDL with L2 prefetch for the vector products. Plain decode
      +5.4–8.1% against main in the same session (25.83 → 27.36–27.39
      tok/s, 37.4 → 35.2 ms on the device); a speculative step 2.4%
      shorter, its rate moving with acceptance; the adaptive speculation
      window (the drafter's probability, TensorFold's 0.3 cut) is measured
      and left off; DeepSeek unchanged (its path already had these).
      Greedy agreement (bound recorded first), perplexity, forced
      rejections and the swap check pass; loads, evictions and decode
      migrated no pages unless another process pressed memory. Open: the
      remaining gap is the MXFP8 layers and BF16 head (the
      quality/performance-modes item); TensorFold's kernel-level profile
      (its container's CUPTI recorded nothing).
- [x] **Swap runner:** a native CLI harness in `jitllm-runtime` that drives
      A→B→A in a running process (tokenize, prefill, decode, detokenize) and
      reports each part of the swap time.
      *First as harness binaries* (`jitllm_swap_runner`, `jitllm_swap_pairs`).
      *Done 2026-09-28 in the runtime* (D-096,
      [runtime-serving.md](runtime-serving.md)): the paged node and the three
      models' runners moved into an `engine` module and the task programs
      into the scheduler; the configuration names the models
      (`[models.<name>]`: artifact or composition, drafter, context,
      tokenizer and template); `jitllm-runtime chat --turn MODEL TEXT...`
      serves turns with the native tokenizer and renderers, speculative by
      default (DSpark, MTP), swapping as needed, and `jitllm-runtime
      swap-table` runs every ordered pair in one process. Through the
      runtime, on `spark-b`: greedy tokens equal the speculation
      harnesses' on the fixed prompts, speculative and plain, for both LLMs
      (6 chat prompts at 32 tokens and 2 decode prompts at 256; prompt IDs
      equal too); the image's pixels the reference's; decode at the
      harnesses' speeds (DeepSeek 30.6 / 34.6 tok/s speculative on `prose`
      / `code`, 21.9 / 22.2 plain; Qwen3.8 40.1 / 38.2 and 25.7 / 26.0); the
      harnesses' greedy, forced-rejection and swap speculation checks pass
      over the moved engine. The runtime's swap table is below (the swap
      path). The package now ships cuBLAS and GGML's and CUTLASS's notices.
      The package step of `check:full` (the arm64 package, its inventory and
      its install test), with REUSE, passed on the workstation on 2026-09-28;
      the header check ran on `spark-b`.
      Open: the image takes one prompt a process from a latents file (its
      runner fixes both at setup).
- [x] Start the model support matrix (moved from M5), recording template
      hashes. *Done 2026-09-28* ([model-support.md](model-support.md)):
      each model and drafter jitLLM runs, the M2 fixtures among them, with
      its checkpoint pin, artifact, template hash as `src/chat` keys it,
      tokenizer, decoding modes, context exercised, evidence, known
      divergences, status and level.
- [x] Last, once the swap floor is proven: a minimal OpenAI-compatible
      `/v1/chat/completions` on loopback (D-014), with numeric intake
      bounds fixed before it accepts input
      ([client-api-baseline.md](client-api-baseline.md#shared-correctness-and-limits)).
      The full front door stays in M5.
      *Done 2026-09-28* (D-097, [runtime-serving.md](runtime-serving.md#the-chat-route)):
      with models configured, the service listens on `[client] bind`
      (loopback only, default `127.0.0.1:8114`) for `POST
      /v1/chat/completions` (JSON and SSE) and `GET /v1/models`, over its
      own bounded HTTP/1.1 reader (no new dependency), one request at a
      time behind a queue of four; every bound (head, target, body, JSON,
      messages, message bytes, parts, `max_tokens` against the model's
      usable context, temperature, top_p, seed, stop strings, timeouts,
      queue) is tabulated there with its status; unknown fields are
      refused. Unit tests cover parsing, every bound at its edge and the
      server over a loopback socket (routes, browser guards, HTTP bounds,
      timeouts, the queue's 429, streaming, errors after the headers,
      disconnect and stop). On `spark-b` with DeepSeek and Qwen3.8
      configured, through curl: DeepSeek, a swap to Qwen3.8 streamed and
      a swap back (10.0, 9.1 and 11.0 s from request to response, the
      swaps 8.1, 7.8 and 9.5 s); each greedy reply equals `jitllm-runtime
      chat`'s on the same prompt (text, 52 and 37 tokens, prompt counts,
      finish), the stream ends in `[DONE]`, a seeded sampled request
      repeats exactly, a stop string ends the answer, and a request over
      each bound was refused (413, 400 with `context_length_exceeded`
      for the prompt and for `max_tokens`, 404, 403, 415, 405). The
      runtime now samples too (seeded, speculative sampling where there
      is a drafter); DeepSeek, like Qwen3.8, is refused at registration
      when its template has no renderer. Open: M5's front door
      (an optional API key, loopback-origin CORS, tools, reasoning
      controls, the other routes, client validation).
      *Amended by the owner 2026-09-28* (D-097 accepted as amended, and
      an owner note on D-014): it listens on loopback and the tailnet by
      default and on any configured address, authentication optional on
      each (unauthenticated listeners named at startup),
      ignores unknown fields by name with a table at
      `/jitllm/v1/ignored-fields`, honors `top_k` and `min_p`, and serves
      persistent connections from an epoll I/O thread (1,024 connections,
      64 queued, pipelining refused, slow clients isolated) with SSE
      keepalives through the queue, swaps and prefill. Unit tests cover
      the bind resolution over fake interface lists, the Host and Origin
      guard, unknown fields and their table, keep-alive, pipelining,
      idle connections and eviction, the queue, keepalive comments,
      early starts and slow clients. On `spark-b` (2026-09-28) with
      DeepSeek and Qwen3.8: it listened on 127.0.0.1, ::1 and both
      tailnet addresses, naming `spark-b.coati-puffin.ts.net`; two
      requests shared one connection; a request with unknown fields was a
      200 and both names appeared in the table (logged once, values
      never); through the tailnet address the MagicDNS name and short
      name passed the Host check while `evil.example` and a trailing-dot
      name got 403 and the table route 404; 8 concurrent streams all
      ended in `[DONE]` in 17.1 s (one swap); 6 streams alternating the
      two models (a swap each) started at 15.0 s in the queue, heard
      keepalive comments and all finished (the last in 52.8 s); greedy
      replies still equal `jitllm-runtime chat`'s (DeepSeek 52 tokens,
      Qwen3.8 37, streamed).

**Exit criteria:**

- **Swap time,** in a running process, with page-in, setup, graph and
  tuning restore and A's state spill and restore counted. Defaults are
  owner-adjustable (D-087):

  | Item | Rule |
  | --- | --- |
  | Pairs | Every ordered pair of the three models, each run as A→B→A: DeepSeek↔Qwen3.8 both ways, and each LLM↔Qwen-Image both ways. Each swap is reported separately; the headline is the worst LLM↔LLM swap |
  | Saved context | An LLM A holds 8K tokens of conversation; its KV, recurrent and indexer state are spilled on swap-out and restored on return. The bound applies here; a 0-context swap is reported too |
  | Graphs | Previously prepared: B ran earlier in this process, and its graphs and tuning are restored. First use: nothing is prepared for B in this process. The ~10 s goal and the ~20 s bound apply to previously prepared swaps; first use has its own target, no worse than about 2× the bound (~40 s) |
  | LLM endpoint | From the swap request to B's first generated token for a short prompt |
  | Image endpoint | From the swap request to the pipeline ready: the first denoising step's output produced. Full generation for a fixed prompt, size, step count and seed is timed separately under the Image criterion |
  | Also per swap | Bytes read, read throughput, peak memory, and each part of the swap time |

  The report gives each swap against the ~10 s goal and the ~20 s bound,
  and beside the baselines ([measured](experiments/fast-swap/baselines.md),
  cold page cache, one run each): Mia's vLLM Qwen3.8 at 13 min 13 s from
  start to first token (11–14 min creator-reported), TensorFold at 141 s
  at `beddbb7b` and 143 s at `71377a53` (about 90 s creator-reported),
  the pinned llama.cpp at 77 s from DeepSeek 0731 to Qwen3.8's GGUF and
  104 s back with A's 8K state restored (75–93 s per switch in M0's run
  of the older revision), and diffusers at 212 s from process start to
  Qwen-Image's first denoising step.
- **LLM correctness:** on a short prompt set, greedy tokens match the
  model's same-format oracle (table above), with small logit differences
  allowed, and perplexity on a fixed text is within a few percent of the
  oracle's. Greedy decoding with speculation gives the same tokens as
  without, except near-ties (the bound: the 99th percentile of the
  engine's own kernel noise between two of its paths, neither the
  oracle, recorded before the comparison).
  *Amended 2026-09-28 (the owner: speed before bit exactness, D-085's
  note): was "the same tokens as without"; the bit-for-bit form is the
  optional reference mode's check.*
- **Speculation correctness** (moved from M9, D-068), per drafter:
  - forced draft rejections at varied positions, all-reject and partial
    accept among them, leave no stale KV, recurrent (Gated DeltaNet or
    linear-attention), drafter or MTP, or indexer state: rollback stays
    exact in effect, checked against a control that drafted only the
    accepted tokens (the reference mode, bit for bit) or, on the default
    fast path, by every state byte outside the accepted rows' writes
    being as it was before the verify;
  - after each rejection, the output equals non-speculative greedy
    decoding's except near-ties (the same bound, from the verify's rows
    against one-row decoding); bit for bit, in the same engine with the same kernels, is an
    optional reference-mode check. *Amended 2026-09-28 (the owner, D-085's
    note): was "bit for bit" on the default path.*
  - rollback composes with swap: after rejected drafts, A is swapped out
    mid-conversation and restored, and continues exactly as the unswapped
    control;
  - where sampled speculation is enabled, it preserves the token
    distribution: on 4 fixed prompts, seeds 0–255 and the first 8
    generated tokens (8,192 sampled tokens per mode, sized to stay within
    D-085's 10 minutes), each prompt's pooled histogram over its 16 most
    frequent tokens plus an "other" bin is within a total-variation
    distance of 0.1 of plain sampling's (about twice the sampling noise
    expected at this size, estimated).
- **Swap correctness:** our own swap and restore cycles are bit-identical:
  A resumed after B gives the same logits and tokens as A never swapped out.
- **Image:** for fixed prompts and seeds, output is within a simple
  image-similarity bound of diffusers' BF16 pipeline's; the pipeline swaps in
  and out with the text models; and full generation for a fixed prompt,
  size, step count and seed is not more than 10% slower than diffusers
  (D-085).
- **Performance and memory** (D-085): prefill and decode are not more than
  about 10% slower than the same-format performance comparator, and peak
  memory is at most about 1.1× the comparator's. Cross-quantization
  comparators are reported beside them for speed and memory, not gated.
  Each comparison names the comparator, its format and whether both sides
  speculated.
- A standard OpenAI-compatible client completes a chat with each LLM
  through the minimal route, swapping between them.
- The source-lock widening, the swap path and the chat route's request
  parser pass their adversarial challenge (heavy path).

**Open questions** (for the owner):

- The image pipeline, and a drafter that shares its target's tables, need
  manifest references to another artifact, with shared resources counted
  once ([artifact-format.md](artifact-format.md#deliberately-open)).
  *Settled by D-089 (2026-09-28, accepted under the owner's overnight
  delegation; the owner may amend):* one
  artifact per component and a composition naming them by ID; the
  pipeline is imported that way, and a drafter is to be a composition with
  its target. DSpark's binding is settled: its own artifact, binding the
  target artifact's token table and head at load (D-089's note).

## M4 — Two-Spark fast full swap  `pending`

Goal: the same cycle for models too big for one Spark, sharded across both.
Each node holds its shard on disk, and the conductor loads both shards at
once.

**Entry:** M3 exit. By entry, model-parallel artifact partitioning is
decided ([artifact-format.md](artifact-format.md#deliberately-open); moved
from M8's entry), and each model's checkpoint, recipe and baselines are
pinned and audited as in M3.

**Models, in this order:**

1. **GLM-5.3 Flash** (`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@c1b7d4c`;
   EXL3 at about 4 bpw, TP2, about 80 GiB of weights per node,
   creator-reported): KDA linear attention, DSA sparse MLA with an indexer,
   mHC, and 288 routed experts.
2. **DeepSeek v4.1 Flash** (Mia's EXL3 at 2.9 bpw, TP2, 99.5 GiB of weights
   per node, creator-reported): a 552B encoder-decoder with CSA2 attention
   and a hierarchical indexer, which no GGML code covers. Its ~190 GiB of
   Engram lookup tables may stay row-paged from each node's SSD (D-035).

**Oracles and comparators,** under M3's rule (cross-quantization is speed
and memory only):

| Model | Correctness oracle (same format) | Performance comparators | Cross-quantization (speed and memory only) |
| --- | --- | --- | --- |
| GLM-5.3 Flash | Mia's EXL3 configuration | Mia's configuration | TensorFold (MLX 4-bit) |
| DeepSeek v4.1 Flash | Mia's configuration | Mia's configuration | none |

**Scope:**

- [ ] **Provenance and licenses:** re-audit the GLM recipe if its baseline
      moves past the pin (HEAD is 65 commits ahead); record the GLM
      checkpoint mirror's license and the DFlash2 drafter's (CC BY-NC-ND
      4.0) for information (D-087); Mia's E3 and cooperative MoE kernels
      (AGPL or mixed, D-080); and whether Mia's ExLlamaV3 revisions'
      formats match our `6b84a21`.
- [ ] **Baselines:** Mia's two-Spark configurations and TensorFold, run by
      us, measured as in M3.
- [ ] **Conductor, minimal** (pulled from M6a): one configured two-node
      topology that loads both shards at once and runs each phase on both
      ranks. Discovery, enrollment, cluster trust and general placement
      stay in M6a. It is development-only (owner, 2026-09-28; D-087): it
      runs over the direct Spark-to-Spark link, trusted like loopback, is
      off by default, and is documented as development-only, not a
      supported deployment. Mutual TLS (D-038) arrives with M6a (D-014).
- [ ] **Sharded execution** (pulled from M8): TP2 over NCCL with
      conductor-issued distributed phase IDs, a separately budgeted
      communication-buffer pool that honors NCCL's registration and
      threading contracts, ordered collective submission and completion
      fences, and per-step collective latency measured before bandwidth.
- [ ] **Coordinated readiness, minimal** (the start of M8's coordinated
      admission): no collective or execution of B begins until both ranks
      report ready, their shard loaded and its memory obtained. A
      preparation failure on either rank, or no report within the stated
      timeout (default 30 s, owner-adjustable), aborts the swap: the other
      rank releases or rolls back what it prepared within that timeout, no
      rank enters a collective alone, and the swap reports failure cleanly.
      Memory a rank has not confirmed released stays counted as held; no
      timeout proves reclaim. General recovery, prepare/commit and the
      broader pressure and failure matrix stay in M8.
- [ ] **Kernels:** EXL3 MoE, the codebooks and rates these checkpoints use,
      sparse MLA decode and prefill on `sm_121`, KDA, the DSA indexer, CSA2
      and Engram row gathers, each the fastest correct implementation under
      D-053, chosen by a quick A/B.
- [ ] **Per model,** as in M3: graphs, state adapters, templates, the
      resident expert layout for EXL3 MoE, and speculation with M3's
      forced-rejection checks where the model supports it (for GLM, MTP or the
      DFlash2 drafter that Mia's GLM configuration uses, whichever is faster
      and correct; the
      nextn layers or DSpark head for v4.1). ReconGemm prepares its
      cuBLASLt descriptors once per bound GEMM (M2's hand-off).
- [ ] The minimal `/v1/chat/completions` serves the sharded models.

**Exit criteria:**

- **Swap time,** as in M3 across both nodes, with its owner-adjustable
  defaults:

  | Item | Rule |
  | --- | --- |
  | Pairs | GLM→v4.1→GLM and v4.1→GLM→v4.1, each swap reported separately; the headline is the worse |
  | Saved context | A holds 8K tokens; its KV, recurrent and indexer state on both ranks are spilled and restored, and count. The bound applies here; a 0-context swap is reported too |
  | Graphs | As in M3: the ~10 s goal and ~20 s bound apply to previously prepared swaps; first use no worse than about 2× the bound |
  | Endpoint | From the swap request to B's first generated token for a short prompt |
  | Also per swap | Per node: bytes read, read throughput and peak memory; the time between the two ranks' ready reports |

  The report sets each swap beside the baselines, Mia's GLM among them at
  65–70 s cold (creator-reported).
- M3's correctness, speculation, performance and memory criteria against
  each model's oracle and comparators above, with memory judged per node.
  Swap and restore are bit-identical on both ranks.
- **Readiness:** a load failure and a memory failure injected on one rank,
  each on either rank in turn, start no collective, the other rank rolls
  back within the timeout, the swap reports failure cleanly, neither
  catalog keeps a lease, grant or backing from B, and the next swap
  succeeds.

## M5 — One resident model, end to end  `pending`

Goal: import the small fixtures, serve them resident through native GGML and
EXL3 execution, and complete chats from named clients over the three
baseline protocols, with bounded, explainable memory.

**Entry:** M4 exit. The native tokenizer, stop rules and sampling come from
M3.

**Scope:**

- [ ] **Importer and verifier** (D-009, D-056): the C++ importer and the
      standalone verifier, with M0's Python prototype as the exact
      accept/reject oracle, running on the workstation and on a Spark.
      Confined import jobs take sources only from the configured stores
      (the `checkpoints` role or the long-term store; D-054, D-063) or the
      Hugging Face Hub (resumable and verified; the token comes from a
      `.env` or config file and is never logged). Hard-linked and symlinked
      sources are rejected, and the user docs say to copy them in or
      download with `--local-dir` (D-063). Removal waits for
      quiescence. Jobs keep their records, locks and grants and report
      progress, status, cancellation and retry (D-041). An install that
      lacks space fails before transferring, reporting the space it needs
      and the removal candidates.
- [ ] **Registration and first use** ([model lifecycle](architecture.md#model-lifecycle)):
      the compact installed-model index cache, shallow checks on every
      startup, and detail and plans built on first use inside an `F`
      reservation. Startup rejects an installed store that fails the
      direct-I/O probe (D-054).
- [ ] **Resident execution:** the FP16 and both EXL3 fixtures under jitLLM
      dispatch, finite context and chunk profiles within the 8K context,
      and state block sizes and KV layouts for each adapter. Decode graphs
      and their relocation proof come from M3.
- [ ] **Tokenization and output:** D-067 renderers for both fixture
      template hashes on M3's native tokenizer, stop rules and sampling
      (moved to M3), and the tool-call parser; host versus device sampling
      is settled against D-052's decode gates.
- [ ] **Front door** ([M5 surface](client-api-baseline.md#m5-surface)),
      growing from M3's minimal Chat Completions:
      Chat Completions, stateless Responses, Messages with token counting,
      `/v1/models` in both shapes with D-046's metadata, and the discovery
      document (D-041); D-045's listener, authentication, CORS, `Host`,
      status and keepalive rules; D-047's non-streaming and storage rules;
      the Claude Code profile; and explicit rejection of `transforms` and
      `plugins`. Numeric intake bounds are fixed before any external input
      is accepted, with the total input limit reconciled with named-client
      tests ([cluster design](cluster-design.md)).
- [ ] **Local management API and CLI** (D-064): versioned routes for import,
      listing, representation inspection, removal, jobs and node health.
- [ ] **Service hardening** (D-074; M1's hand-offs, moved from M2 by the
      owner on 2026-09-27): a system-call filter for `jitllm.service`
      that admits io_uring, and a single reaper for jobs started from
      other threads.
- [ ] **TLS** (D-065): per-name certificate files selected by SNI, reload on
      change, key-match and expiry checks, the name-constrained local CA,
      the certbot deploy hook, and the Tailscale certificate timer. The
      root-run hook and timer touch only `/etc/jitllm/tls/`, their units are
      sandboxed to it, and they take the heavy path.
- [ ] **Surface definitions:** front-door, alias and TLS configuration keys;
      the individual `jitllm-` header and body-field names (D-062); the
      HTTP, TLS and JSON libraries, chosen under D-017, D-057 and D-066.
- [ ] Move the fixtures' rows in the model support matrix (started in M3,
      where they are listed as fixtures) to served, with their templates.

**Exit criteria:**

- Teacher-forced logits and declared intermediates for the three fixtures
  match their pinned references within bounds declared before evaluation
  ([first-slice.md](first-slice.md), [exl3-bringup.md](exl3-bringup.md)).
  Rendered bytes and token IDs match the golden fixtures.
- EXL3 resident prefill, time to first token and decode inter-token
  p50/p95/p99 meet the predeclared upstream parity bounds against upstream
  serving controls in matched and normal views, with memory inside the
  declared bounds; a regression needs a fix or an explicit owner-approved
  tradeoff (D-052).
- Each engine, served resident, is at least as fast as its reference end
  to end (GGML against llama.cpp, EXL3 against ExLlamaV3; D-085's coarse
  comparison), with BP-F3's resident timings reported in it (moved from M2
  by the owner, 2026-09-27; BP-F4's per-token host cost is measured in
  M3).
- The [M5 acceptance cases](client-api-baseline.md#acceptance-owed-in-m5)
  pass as scoped there. At least one named client completes a chat
  unmodified with each representation, and Cursor stays an explicit gap
  unless resolved. The context-compacted release moves to M6, which
  delivers D-041's close, and the Ollama-native checks move to M10 with the
  Ollama profile.
- Finite default context and output bounds bound every admitted request,
  and the M5 rows of D-050's matrix pass, with the parts moved to M5
  (suballocation holes, stalled-client termination, every queue full at
  once, capacity-loss injection). Memory use is bounded and
  explained by the memory breakdown.
- The importer, verifier and front-door parsers pass their adversarial
  challenge; an interrupted import never appears valid.
- Measured and recorded: startup time and metadata memory as the installed
  library grows, the cold-switch cost of on-demand detail, and both
  contexts' state bytes, from which D-055's
  [capacity values](retention-policy.md#bounds-and-defaults) are pinned.

## M6 — First useful product: A→B→A with partial retention  `pending`

Goal: two small model contexts share one local budget. Switching to B
displaces only what B needs, retained conversation state lets A resume
without a full re-prefill, and the switching policy is chosen from
measurement. Useful without MoE or sharding. M3 already restores one
model's state across a full swap; M6 adds partial retention and the
retention policy around it.

**Entry:** M5 exit with its capacity values pinned, plus everything
[M6 entry pins](retention-policy.md#what-m6-entry-pins): the frozen
transcript, budgets and reference paths among them, and the spill write
budget from the drive's rated endurance. EXL3 switching budgets come from
new matched controls ([exl3-bringup.md](exl3-bringup.md)).

**Scope:**

- [ ] **Retention** ([policy](retention-policy.md); D-024, D-031, D-055):
      prefix and continuation entries with their identity, restore
      boundaries, branches, sharing and refresh; capacity-driven expiry with
      24-hour idle caps; the victim-order baseline; spill with its protected
      directory, preallocation, direct I/O, digest check on restore and
      deletion at startup; miss reasons and fallback reporting.
- [ ] **Partial eviction** of a quiescent model, reloading only missing
      dependencies (D-008).
- [ ] **Concurrency when it fits:** all-resident cohorts under
      full-envelope checks, reporting whether requests ran concurrently or
      time-sliced.
- [ ] **Switching policies** (D-069): the priority-aware default,
      run-to-completion and time-slicing with their guards, their
      configuration keys and per-alias overrides.
- [ ] **Request control** (D-042): interactive and background classes,
      maximum queue waits, cancellation and bounded progress events.
- [ ] **Release** (D-041, D-045): the final-turn flag, idempotent
      continuation close and the context-compacted release, with their wire
      names fixed here.
- [ ] **Warm jobs** (D-041): capacity-constrained, never an implicit
      download.
- [ ] **Diagnostics:** status, the admission what-if query, Perfetto trace
      export and eviction explanations; management controls for priorities,
      residency policies and trace capture; Prometheus `/metrics` with
      compatible health and load queries (D-044).
- [ ] **Storage scheduling:** demand reads mixed with spill write-back,
      inside the pinned write budget.
- [ ] Add the A→B→A workload and its regression thresholds to `check:spark`.
- [ ] **Discrete GPU, secondary** (D-082; after the two-Spark swap, D-087):
      the A→B→A fast swap, one model
      active and partial retention within device memory, on the
      workstation's discrete GPU (`mise run test -- native --gpu`), with
      its PCIe restore rate reported, within a configured device budget and
      with the landing zone reported apart from it. Whole-model swaps and
      paging within device memory only; no on-demand expert paging from the
      SSD there. Not an exit criterion.

**Exit criteria:**

- D-055's [timed workload](retention-policy.md#m6-acceptance-workload)
  passes its pass rule: Qwen2.5-0.5B FP16 and EXL3 4.0 bpw in both
  orientations, six jitLLM arms against fresh interleaved references, 72
  accepted repetitions per arm, orientation and cache condition. For each
  floor arm (J-partial, J-spill), orientation, direction and cache
  condition, at the median and at p95, jitLLM's one-sided 97.5% upper bound
  is at most the smallest one-sided 97.5% lower bound among the valid
  reference arms; a comparison with no valid reference arm does not pass
  (D-036). The report covers latency distributions, bytes read and written,
  peak memory and spill, prompt tokens reused versus recomputed, and deltas
  against jitLLM's whole-model control (M9's comparator).
- The [correctness gates](retention-policy.md#correctness-gates) pass: exact
  outputs and bit-identical teacher-forced logits against
  provenance-matched controls, and catalog state and events show that only
  selected extents were displaced.
- The [functional and adversarial cases](retention-policy.md#functional-and-adversarial-cases)
  and the M6 rows of D-050's matrix, with the parts moved to M6 (fork and
  copy-on-write, cached-state promotion), pass
  with an EXL3 context in the
  matrix; cache expiry never destroys admitted suspended work.
- Through at least one unmodified named client: a long conversation on A,
  B under pressure, then A resumed, over both resident reuse and forced
  spill/restore. The context-compacted release case deferred from M5
  passes.
- B arriving while A is still generating is measured under each D-069
  policy, with queue delay reported apart from paging and switch time and
  from first-token compute, together with pauses and bytes reloaded. The
  owner keeps or changes the default on these results.
- An all-resident control shows concurrent progress when both complete
  envelopes fit.

## M6a — Configured placement across nodes  `pending`

Goal: one configured conductor places whole models on enrolled nodes and
routes requests with retained-state affinity, so a subagent's model runs on
the other Spark while the main model stays resident. It follows M6,
independently of M7, and grows M4's minimal two-node conductor into the
cluster's placement layer; sharding under pressure and failure is M8.

**Entry:** M6 exit.

**Scope:**

- [ ] **Setup and discovery** (D-038, D-039, [cluster design](cluster-design.md)):
      inventory, trusted SSH, the mDNS window, layout classification,
      bounded dedicated-QSFP subnet scans, one-time enrollment and path
      re-detection for enrolled members.
- [ ] **Configuration and trust:** the shared and node-local v2 documents,
      the private CA with TLS 1.3 mutual authentication, epochs, session
      fencing and restart reconciliation, and protocol v1 with its bounded
      state, schema tests and exact per-message field catalog.
- [ ] **Conductor** (D-037): placement, affinity routing, single-attempt
      dispatch, credit-based streaming, health states and authoritative
      per-node admission; stale or aggregate reports never admit.
- [ ] **Availability** (D-041, D-046): the per-model `endpoints` shape.
- [ ] **One import per cluster** (D-054): peer replication of verified
      prepared artifacts over the cluster link, archive to and install from
      the long-term store, and the explicit archive-or-delete choice when a
      node lacks space.
- [ ] The package gains its rdma-core dependencies when jitLLM first links
      them (D-063).
- [ ] Decide whether a worker node serves its own loopback management
      listener.

**Exit criteria:**

- The [required validation](cluster-design.md#required-validation-and-handoff)
  challenge conditions and the conductor's fake-transport and Spark
  scenarios ([architecture](architecture.md#conductor-ownership-and-admission))
  pass.
- An unmodified standard client completes A→B→A through one endpoint, with B
  placed on the other node while A stays resident. Each node enforces its
  full local budget and compatible state reuse, and models run concurrently
  when placement permits.
- Stale capacity reports, node loss and cancellation end in bounded failure
  or unwind, never in unsafe admission or silent replay of a started
  stream.
- Remote management and cluster access require authentication and
  transport protection (D-014, D-065); inference authentication stays
  optional (D-014's owner note). A replicated or archived artifact is published only after
  verification against an identity held outside its source.

## M7 — Demand-paged MoE and the first daily drivers  `pending`

Goal: exact demand-paged routed-expert execution on the named Gemma 4
26B-A4B and Ornith 1.5 35B-A3B pair, and those two models usable day to day
through the named clients, reasoning and constrained output included (owner,
2026-09-23). Resident MoE execution arrives earlier, with M3's full swaps.

**Entry:** M6 exit; M6a is independent. The pair's checkpoints and
representations are pinned, and their GGML source closures, tokenizers and
chat templates are selected and audited (D-013, D-057, D-067). Approved
before measurement: the performance protocols; a resident-performance bound
against the pinned llama.cpp reference (prefill, time to first token and
decode inter-token p50/p95/p99, matched and normal views); the named
budgets, each with its loading policy, including at least one per model at
which its routed experts do not all fit beside its other extents, state and
headroom, so selected experts miss during both prefill and decode; and the
sustained-use schedule and duration.

**Scope:**

- [ ] **Routing boundary** (D-008, [architecture](architecture.md#routing-boundary-for-moe-7)):
      selected-expert leases, asynchronous misses and resumable tasks,
      brought up on a synthetic or tiny MoE before the named pair
      (features.md). Envelopes cover the worst-case union of every allowed
      closure and assume no fixed number of positions per phase (D-068).
- [ ] **Loading policies:** eager active-model loading that keeps inactive
      extents (the feasibility study's recommended first policy) and routed
      demand paging, both selectable, compared at the named budgets.
- [ ] **Expert layout:** expert compaction and demand-paged dispatch from
      the M7 GGML proof, building on M3's initial pointer-table or
      uniform-stride choice per format, and the MoE mapping in v0 artifacts.
- [ ] **Shapes that come with these models:** hybrid sliding-window and
      global attention (Gemma 4) and recurrent or linear-attention layers
      (Ornith), with their state adapters and explicit restore coverage
      (RE-004, RE-007), building on M3's Gated DeltaNet layers and state
      adapters. M6's retention matrix extends to them.
- [ ] **Traces:** native routing-trace capture and policy replay, checked
      against the M0 reference experiment; captured traces stay outside Git
      and replay by verified hash in the gate.
- [ ] **Reasoning** (D-043, D-046, D-047): protocol-specific reasoning,
      final and tool fields and streaming, model-supported thinking
      controls, signed blocks, `reasoning` and `reasoning_details`, and
      cached-token usage from real prefix reuse.
- [ ] **Constrained output** (D-043): `response_format` JSON object and
      schema, strict tool arguments and vLLM's `structured_outputs.json`,
      over a documented schema subset with explicit rejection of the rest.
- [ ] **Sustained use:** a bounded multi-hour agent/subagent session on the
      pair with repeated switches, branches, cancellations (during I/O and
      while paused included), expiry by capacity and by test-shortened idle
      caps, and spill and restore.
- [ ] Demand-paged EXL3 MoE only if claimed, with its own routed-expert
      closure, kernel, quality and performance baselines (D-052); resident
      EXL3 MoE arrives with M4's models.
- [ ] Assess artifact compatibility guarantees with M6's dense and M7's
      MoE evidence, in a separate decision (D-018).

**Exit criteria:**

- Acceptance exercises real demand misses. At each miss-forcing budget,
  catalog state and events record nonzero selected-expert misses in prefill
  and in decode, with their count, bytes and wait time and the phase and
  layer at which each suspended. There, a measured window without misses is
  inconclusive for the paging gates, and the next two criteria are judged
  on runs with misses.
- No unselected expert loads beyond declared metadata and read-ahead, and
  no expert is substituted or its contribution dropped, verified from
  catalog state and events.
- Numerics stay correct against the pinned references after eviction and
  restoration and across within-step misses, and the M7 rows of D-050's
  matrix, with the runtime closure-excess check moved to M7, pass with
  real routes.
- D-036's generation limits hold on the pair: at most 10% added generation
  time, continuation time to first token included, and at most 20 ms p95 /
  100 ms p99 added token gaps against a resident control with matched state
  provenance, at every named budget, the miss-forcing ones included. Pause
  gaps are reported separately (D-069). M6's switching floor still holds.
- Resident prefill, time to first token and decode inter-token p50/p95/p99
  on both models meet the approved bound against the pinned llama.cpp
  reference, or the owner approves an explicit tradeoff, as D-052 requires
  for EXL3. Paging limits compare jitLLM with itself, so they cannot stand
  in for this.
- The sustained-use run ends with every completed, cancelled and paused
  request retired and no lease, grant, task or I/O outstanding. Memory, the
  catalog and retained-entry metadata, task and job records and queue
  depths stay within their bounds and level off instead of growing with
  elapsed time; task and job records and queue depths return to their idle
  levels after each switch cycle. Spill stays within `S_spill` and the write
  budget, and the memory breakdown reconciles with OS counters at the end.
- Measured and reported: the routing boundary's resident-hit overhead, each
  routed phase kind's bound against its observed peak, and whether idle
  retained state starves weight residency.
- Named clients complete reasoning and tool round trips on both models,
  including acceptance case 7's reasoning checks, and constrained-output
  requests pass their pinned compatibility fixtures. Any OpenRouter-mode
  claim rests on D-046's pinned-client run. The support matrix records the
  client versions.

## M8 — Sharded execution under pressure and failure (two Sparks)  `pending`

Goal: a flagship model sharded across both Sparks, correct under asymmetric
pressure, cancellation and controlled failure. M4 already runs TP2 sharded
full swaps (its communication-buffer pool, collective ordering and
collective latency moved there, with a minimal ready-before-collective rule
and bounded abort), and placement-only use works at M6a.

**Entry:** M6a and M7 exits. By entry: the flagship checkpoint and its
validated parallelism recipe are named (M4's models, and the
MiMo-V2.6-Flash-RL TP=2/EP=2 reference, are the candidates); and the
two-node admission design chooses between prepare/commit and the deferred
mirrored-ledger shortcut (features.md). Model-parallel artifact
partitioning is decided at M4's entry.

**Scope:**

- [ ] Parallelism beyond M4's TP2, porting the recipe's first (TP, PP and
      EP are different plans), with conductor-issued distributed phase IDs.
- [ ] Coordinated admission, generalizing M4's minimal readiness rule:
      node-issued reservations and every rank ready before commit, with
      prepare failures unwound; an unknown completion never frees another
      rank's buffers.
- [ ] Sharded models under partial eviction and paging, not only full
      swaps.

**Exit criteria:**

- Both ranks stay correct against the pinned reference under asymmetric
  pressure, cancellation and controlled failure, including partial
  preparation, a lost commit, mismatched rank generations and node loss
  during a collective. No timeout is treated as proof of reclaimed memory.
- Sharded performance is reported against the recipe's reference deployment
  in both views.

## M9 — Performance and new decoding modes  `pending`

Goal: meet D-036's benefit target on a library larger than memory, and
execute the speculative and block-diffusion shapes designed since M0
(D-068) that M3 and M4 did not.

**Entry:** M7 exit. M9 may start before M8 exits, but MiMo's stored MTP
layers run only on M8's sharded execution, so that work waits for M8 and M9
exits after it. Before M9 planning, the bounded DiffusionGemma reference
study measures the per-step expert closures of wide phases, and untriggered
deferrals are reviewed ([features.md](features.md) and the table below).
Before the new modes execute, numerical and statistical bounds and trace
protocols for speculative sampling and diffusion decoding are declared;
manifest references to another artifact by ID are settled in M3. The named
configurations, including the over-memory library, and the
partial-retention benefit workload are pinned before acceptance runs
(D-036).

**Scope:**

- [ ] **Larger-than-memory library:** DeepSeek V4 Flash with Qwen3.8 Flash
      Next on one node is the canonical pair (D-036). M3 runs both, with
      their compressed attention and indexers, Qwen3.8's sparse n-gram rows
      and its linear-attention layers, as full swaps; M9 adds partial
      retention and paging on them. If the pair cannot be validated, name
      another whose prepared weights exceed physical memory rather than
      pass M9 on the small pair alone.
- [ ] **Speculative decoding** (D-068), beyond M3's and M4's: stored MTP
      layers for Ornith (and MiMo's once it runs sharded under M8) and
      Gemma 4 companion drafters, with speculative sampling at every
      supported setting, and draft-length and acceptance tuning.
- [ ] **Block diffusion** (D-068): DiffusionGemma-26B-A4B.
- [ ] **Execution speed:** CUDA graphs beyond M3's decode graphs, further
      kernels and plans (D-053), and target-assisted import tuning as an
      explicit, separately keyed mode.
- [ ] **TensorFold's format** (owner, 2026-09-28; D-087): import and run
      MLX-style affine 4-bit weights (TensorFold's checkpoints; reconcile
      the group size, 32 or 64) for Qwen3.8 Flash and GLM-5.3 Flash, and
      optimize them. TensorFold is then a same-format oracle and a gated
      comparator for those models (D-085's ~10% and ~1.1× bounds). Until
      then, in M3 and M4, it is a cross-quantization comparator: reported,
      not gated.
- [ ] **Victim policy:** compare global LRU, frequency/recency and the
      cost-aware heuristic on identical recorded traces, with the confirmed
      hysteresis, minimum-residency and reload-cost refinements
      ([baseline](architecture.md#victim-selection-initial-baseline)) and
      per-model statistics so no model monopolizes reclaimable bytes. A
      candidate replaces the baseline, and the cost-aware heuristic becomes
      the default, only if replay shows fewer miss bytes and reloads.
- [ ] Deferred optimizations whose triggers have fired (prefetch, residency
      warm-start and the others in features.md), each against its
      demand-only control.

**Exit criteria:**

- D-036's benefit target: at least 25% lower median return-switch latency
  than jitLLM's own whole-model control, with identical state handling at
  the same budget, on the agreed partial-retention workload, with at least
  one named library exceeding physical memory, while the switching floor
  and generation limits still hold.
- Measured and reported on the over-memory library: retention under
  physical pressure and whether idle retained state starves weight
  residency (D-055).
- The speculative verify path's teacher-forced logits match plain decoding
  within declared bounds, with top-1 agreement reported and free-running
  divergence reported at its first position (RE-008). These gates cover
  M3's and M4's drafters too. The targeted checks those drafters need
  (forced rejection, rollback across a swap and a coarse sampled
  distribution) moved to M3's exit and apply to M9's drafters as well.
- Speculative sampling preserves the target distribution. On recorded target
  and draft distributions, acceptance, rejection and residual resampling
  match a reference implementation of the rule exactly under fixed seeds;
  sampled outputs match plain sampling's distribution within declared
  statistical bounds, at the supported sampling settings.
- Rollback leaves exactly the accepted prefix. After rejected drafts,
  teacher-forced continuation matches a control that drafted only the
  accepted tokens, bit-identical where the plan is the same at both verify
  widths and otherwise within the declared bounds. No rejected position's
  KV, recurrent update or drafter state survives, none enters a retained
  entry (D-055, D-068), cancelling between draft and verify retires all
  draft work, and a pause there commits none of it before verify.
- Diffusion stays within declared bounds of its reference, compared step by
  step under a fixed seed. A cancelled canvas never commits, and a paused one
  commits nothing until it resumes and finishes.
- Matched and normal reference comparisons are reported, with no numerical
  or lifetime regression in any supported configuration.
- If prefetch is built, the part of D-050's
  [matrix](reservation-policy.md#worked-cases-and-implementation-gates)
  moved to M9 passes: repeated speculation keeps its full peak in `J` or
  the owning phase.

## M10 — Product and first release  `pending`

Goal: the remaining confirmed product scope, and a first tagged 0.x release
that a stranger can install from the project's package repository and run
(owner, 2026-09-23).

**Entry:** M9 exit. The owner may pull an item forward once its dependencies
exist (for example the Ollama profile or tokenization endpoints after M5, or
the dashboard after M6's management controls); the release itself waits for
every earlier exit. Each model added for embeddings, reranking or file
inputs is named at entry with its pinned reference engine and numerical
bounds, declared before native evaluation.

**Scope:**

- [ ] **Dashboard** (D-064): a separate service that calls the management
      API from its own server side, including setting the Hugging Face
      token.
- [ ] **MCP management adapter** (D-042): a separate process exposing
      discovery, status and explicitly authorized actions.
- [ ] **Application permissions** (D-042): per-application credentials and
      scopes for inference, read-only status and model administration.
- [ ] **Session and hint extensions** (D-022, D-046): the optional session
      ID and release, client warm hints, and D-046's `session_id`, `user`
      and `metadata` hints. M5's front door already accepts them as
      advisory preferences, and M6 records `session_id` for release lookup.
- [ ] **Ollama subset** (D-041, D-045): listing, details, chat and
      generation with the `keep_alive` mapping, tested with a named
      Ollama-native client including its load-time bound.
- [ ] **Tokenization and diagnostics** (D-043, D-044): vLLM-compatible
      tokenize, detokenize, tokenizer information and prompt rendering; raw
      Completions with standard log-probabilities and bounded token
      diagnostics.
- [ ] **Pooled outputs** (D-042, D-044): embeddings and reranking on
      validated models named at entry.
- [ ] **File inputs** (D-042): text resources, images (starting with Gemma
      4's vision encoder) and audio files, on models validated for them,
      with bounded preprocessing.
- [ ] **Packaging** (D-027): the signed arm64 apt repository and its signing
      keys, optional copyleft modules in the default install with a
      build-time opt-out (D-080), and drain-before-restart upgrades.
- [ ] **Release readiness** (D-061, D-062): the release checklist, the
      published support matrix, notices and source obligations for every
      shipped profile, and user documentation.

**Exit criteria:**

- Each new route passes its pinned compatibility fixtures, including
  negative and interrupted-stream cases, with the unmodified clients its
  decision names; unsupported features fail explicitly.
- Every new model and output path meets numerical acceptance against its
  pinned reference within the declared bounds before any support claim:
  embedding vectors with the model's pooling and normalization; rerank
  scores and orderings; encoder outputs and the language model's
  teacher-forced logits on image and audio prompts, with preprocessing
  matched to the reference's pinned implementation. Raw Completions
  log-probabilities agree with the validated teacher-forced logits, and
  tokenize and render output matches inference byte for byte and ID for ID.
  Generated text alone is not evidence.
- On a fresh Spark, installing from the apt repository and following only
  the checked-in docs reaches a running supported model (vision.md).
- The default and copyleft-disabled builds meet D-017 with a complete,
  audited closure; builds with optional modules enabled ship matching
  notices and source.
- The release commit passes `check`, `check:full` from a fresh clone and
  `check:spark` (D-061), and the owner tags the first 0.x release.

## Deferred delivery and proposals

Confirmed scope stays confirmed when its implementation is deferred. A
candidate stays unapproved until revisited; reaching a trigger is a reason
to evaluate it. Deferred optimization triggers (predictive prefetch,
dependency-group scoring, optimistic MoE) live in features.md.

| Item | Earliest work / revisit trigger | Scope |
| --- | --- | --- |
| Quality/performance modes: KV-cache and output-head compression (FP8 caches, TurboQuant and other algorithms; a quantized output head) and FP8 linears for Qwen-Image's DiT | After M3's speed work; earliest when long contexts or several resident conversations press on memory, or to shrink swap spill | [Per-model options with long-context quality checks](features.md#compute-backends-and-execution) (owner, 2026-09-28)  Exposed as model-mode aliases and the image endpoint's `quality` (owner, 2026-09-28) |
| Load/temperature-aware GPU operating policy | Earliest M9 evaluation, or earlier diagnosis if reproducible throttling or unexplained shutdowns occur | [Proposed telemetry and optional adaptive clock ceiling](features.md#load--and-temperature-aware-operating-policy-proposal); measure stock/fixed/adaptive policies first, no assumed fault or automatic host changes |
| Ollama registry and other management compatibility | After the basic subset and relevant native management operation, when a named client needs them | Deferred D-041 candidate; lifecycle mapping needs separate proof |
| Regex/grammar constrained output | After validated JSON/schema support, when a concrete client requires it | D-043 deferral; no automatic milestone delivery |
| LoRA adapters | Earliest M9 planning after validated base-model execution, when a concrete adapter workload needs them | D-044 deferral; no automatic delivery |
| Classification/reward/generic pooling APIs | Earliest M9 planning after validated base-model execution, when a concrete model/task workload needs them | D-044 deferral; not implied by embedding/reranking support |
| Live audio/video input | Earliest M9 planning after initial file-input evidence, when a concrete workload establishes streaming/synchronization requirements | Deferred D-042 candidate; not an automatic M9 deliverable |
| Batch/background inference jobs | Earliest M9 planning after validated request scheduling, when a concrete workload justifies scheduling/storage needs | Deferred D-042 candidate; ordinary background request priority is already confirmed |
| Automatic membership changes | After M6a, when configured enrollment and explicit restart cannot reasonably serve membership churn | Candidate mechanism under D-023/D-038; bootstrap discovery and path refresh for enrolled nodes are already M6a scope |
| Conductor election | After M6a, when conductor failover becomes an explicit requirement; first define fencing and in-flight request handling | Candidate mechanism under D-023; one configured conductor initially |
| Automatic replica placement and balancing | After M6a, when measured overlapping demand on a small model causes waiting while another node has sufficient headroom | Confirmed D-023 scope with deferred delivery; preserve affinity and include duplicated weights/state in budgets |
| Discrete-GPU fast swap of large models | After the two-Spark swap (D-087); small models only fit a 12 GB card | M6 validates the A→B→A fast swap on the workstation's RTX 3080 Ti (D-082); a later slice or milestone takes the swap path further there |

Review untriggered items during M9 and M10 planning; they do not
automatically enter either milestone's scope or block earlier milestone exits.
