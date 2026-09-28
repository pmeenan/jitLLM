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
- **Support is earned per checkpoint.** From M3 the support matrix records
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
jitLLM generates its own tables, and D-088 (proposed) awaits the owner.

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
      checkpoint cleared; the four checkpoints, the vLLM images and the
      llama.cpp pin recorded; the vLLM, FlashInfer and CUTLASS parts of
      Mia's NVFP4 and MXFP8 path identified with their licenses, a
      CuTe-DSL kernel's runtime among them under NVIDIA's proprietary
      terms. The checkpoints are on both Sparks' NVMe, TensorFold's on
      `spark` only ([environment.md](environment.md#m3-model-store-2026-09-28)).
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
      Open: the NVFP4/MXFP8 A/B; the vector attention at D = 256, which
      upstream picks for Qwen3.8's decode below 8,192 cells (the MMA kernel
      runs it meanwhile); the image's operations; and whether to admit CUB,
      which the build leaves out (top-k takes GGML's radix select;
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
      Open: running each chunk as a device job over leased closures on the
      paged node (D-086; the harness is resident on `cudaMalloc` memory, like
      the backend proof's rung 3), the executed-plan record against
      llama.cpp's, and the other models.
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
      MMVQ and MMQ) equal the reference layout's bit for bit. Open: the
      NVFP4 path's.
- [x] **Tokenizer and chat templates** (pulled from M5; D-067): the native
      tokenizer, renderers for each model's pinned template (DeepSeek's
      upstream ships Python encoding scripts, not a template), stop rules,
      and greedy and seeded sampling. Landed in tests
      ([tokenizer.md](tokenizer.md)): token for token with llama.cpp and
      Hugging Face on a 92-item corpus for all three models, and both
      templates byte for byte; shipped binaries wait for D-088. The output
      parsers and the request mapping are the chat route's.
- [ ] **Speculative decoding in the core** (pulled from M9; D-068): Qwen3.8's
      MTP layer and DeepSeek's DSpark drafter, with verify and rollback that
      keep only the accepted prefix, and forced draft rejection for the
      exit's speculation checks (moved from M9). Every published Mia and
      TensorFold decode number uses speculation.
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
- [ ] **CUDA graphs for decode** (pulled from M9): captured per model and
      plan and replayed after swaps that restore every extent at the same
      virtual addresses, with setup and tuning state restored the same way.
      This reopens D-086 and needs the relocation proof first. BP-F4's
      per-token host cost is measured on these models (about 1.9 µs per
      launch measured on the fixtures; several milliseconds per MoE token
      is an estimate).
- [x] **RE-029's lead:** read `CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS`
      on the GB10, one `cuDeviceGetAttribute` call. Mia's
      `patch_ple_offload.py` reports it as 0, with `cuStreamWaitValue32`
      then blocking the host's next launch (creator-reported). The answer
      decides how page-in copies and phases share the submission lane
      during a swap. *Done 2026-09-28* ([RE-029](rough-edges.md#re-029-a-jobs-kernel-launches-can-block-its-lane-while-the-stream-is-busy--2026-09-27-status-open)):
      the deprecated `_V1` attribute reads 0 and the current memory
      operations are supported; the wait does not block the next launch.
      A stream holds about 1,020 pending operations and a launch into a
      full one blocks the thread, so the swap path keeps the zone's copies
      off a thread that can launch into a full stream, or keeps each
      stream under the limit.
- [ ] **Swap runner:** a native CLI harness in `jitllm-runtime` that drives
      A→B→A in a running process (tokenize, prefill, decode, detokenize) and
      reports each part of the swap time.
- [ ] Start the model support matrix (moved from M5), recording template
      hashes.
- [ ] Last, once the swap floor is proven: a minimal OpenAI-compatible
      `/v1/chat/completions` on loopback (D-014), with numeric intake
      bounds fixed before it accepts input
      ([client-api-baseline.md](client-api-baseline.md#shared-correctness-and-limits)).
      The full front door stays in M5.

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
  (about 90 s creator-reported), the pinned llama.cpp at 77 s from DeepSeek
  0731 to Qwen3.8's GGUF and 104 s back with A's 8K state restored (75–93 s
  per switch in M0's run of the older revision), and diffusers at 212 s
  from process start to Qwen-Image's first denoising step.
- **LLM correctness:** on a short prompt set, greedy tokens match the
  model's same-format oracle (table above), with small logit differences
  allowed, and perplexity on a fixed text is within a few percent of the
  oracle's. Greedy decoding with speculation gives the same tokens as
  without.
- **Speculation correctness** (moved from M9, D-068), per drafter:
  - forced draft rejections at varied positions, all-reject and partial
    accept among them, leave no stale KV, recurrent (Gated DeltaNet or
    linear-attention), drafter or MTP, or indexer state, checked against a
    control that drafted only the accepted tokens;
  - after each rejection, the output equals non-speculative greedy
    decoding's bit for bit, in the same engine with the same kernels;
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
  once ([artifact-format.md](artifact-format.md#deliberately-open)). Settle
  this format change before the pipeline is imported.

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
- [ ] Add the fixtures to the model support matrix (started in M3).

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
- Remote access requires authentication and transport protection (D-014,
  D-065). A replicated or archived artifact is published only after
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
