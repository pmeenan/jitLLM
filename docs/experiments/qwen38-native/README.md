<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8 Flash Next, native and resident (M3)

M3's second model slice ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)):
Qwen3.8 Flash Next in Mia's NVFP4 and MXFP8 checkpoint, run by jitLLM
natively and resident on one Spark from its D-056 prepared artifact, against
Mia's vLLM on the same checkpoint (the oracle,
[baselines](../fast-swap/baselines.md#qwen38-flash-next-mias-vllm-nvfp4)).

## Inputs and bounds

Inputs, fixed in the repository before jitLLM's first run:

- **Checkpoint:** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6`, its 34 shards
  and `config.json` pinned in [fast-swap/pins.json](../fast-swap/pins.json).
- **Oracle:** Mia's vLLM in its deterministic mode with MTP off
  ([reference-qwen3.8-nvfp4-vllm.json](../fast-swap/reference-qwen3.8-nvfp4-vllm.json),
  [reference-qwen3.8-nvfp4-vllm-ppl.json](../fast-swap/reference-qwen3.8-nvfp4-vllm-ppl.json)):
  FP8 KV, BF16 SSM state, `VLLM_QSA_DET_TOPK=1`, `VLLM_MOE_DET_FINALIZE=1`;
  routed experts on FlashInfer's CUTLASS NVFP4 MoE, MXFP8 linears on
  FlashInfer's CUTLASS MXFP8 kernel. Its logprobs come from BF16 logits, so
  near the top they sit on a grid of 0.125.
- **Tokens:** the oracle's own, [reference_inputs.py](reference_inputs.py)
  writes them: [prompts.tsv](prompts.tsv) (the six chat prompts as the
  checkpoint's template renders them, 60–72 tokens), [forced.tsv](forced.tsv)
  (the oracle's 32 greedy tokens each) and [ppl.tsv](ppl.tsv) (`docs/async-model.md`
  at `e7973e5`, 3,558 tokens; past 2,048, so QSA's indexer selects).

Bounds ([compare.py](compare.py)):

1. **Greedy tokens, teacher-forced:** at each of the 192 steps jitLLM's argmax
   is the oracle's token, except at a near-tie: a step where the oracle's
   top-1 to top-2 margin is at most jitLLM's own kernel noise. That noise is
   measured without the oracle: the same 192 steps run twice, once as they
   run (prompt prefill on MMQ with FP4 activations, MXFP8 through BF16 and
   cuBLAS) and once `--stepwise` (every product on decode's kernels: MMVQ with
   8-bit activations, jitLLM's MXFP8 vector product); the 95th percentile of
   how far the top-1 to top-2 margin moves between them, rounded up to the
   oracle's 0.125 grid. **This bound was written after jitLLM's first
   comparison with the oracle, not before it, so it is not pre-registered**
   (plan.md's M3 entry fixes bounds before a model's first native
   evaluation); the raw margins and the bound's sensitivity are listed
   below so it can be judged.
2. **Logprobs:** reported per prompt, max and RMS of |jitLLM − oracle| over
   the oracle's top-5 tokens, beside the same between jitLLM's two runs.
3. **Perplexity:** within 3% (relative) of the oracle's; per-position NLL
   differences and top-1 agreement reported.
4. **Resident expert layout:** every layer's routed products over the slab
   equal those over the reference layout (the slices packed) bit for bit.
5. **State:** a spill and restore of the whole state (to the host, the device
   copy overwritten, back) leaves every later logit bit-identical.

Performance and memory are not bounds of this slice: M3's exit gates them
(D-085: prefill and decode within about 10% of Mia's vLLM, peak memory at
most about 1.1× its), and the results below say where they stand.

## What runs

- **Import** ([import_m3.py](../artifact-layout/import_m3.py) with
  [modelopt_qwen38.py](../artifact-layout/modelopt_qwen38.py), converter
  `m3-1+layout-a0d1980a9eddd1ad+modelopt_qwen38-bb49d9620fe3d9ca`; the
  module in the tree is `bbbadc444bdc9b50`, which only adds refusals of
  invalid scales, global scales and duplicate keys, and a scan of every
  scale in the checkpoint found none it refuses, so it would write the
  same shards and index under a new converter name and artifact ID; not
  re-imported): on
  `spark-b`, 9 min 16 s (peak RSS 2.47 GB, 8 repack processes), artifact
  `67617f87f2199408faa6a3663ca86f6d7aafb258a687a8d0a11092ffd74741b6` in
  `~/.local/share/jitllm/m3-artifacts/`: 103,921,031,115 bytes in 20 shards;
  24,627 groups (the token table, the n-gram table, 48 layers, 24,576 expert
  groups of 2,765,016 used and 2,768,896 stored bytes, the head), 1,600
  resources, 144 expert arrays, 66,261 chunks. What it repacks, all
  losslessly ([artifact-format.md](../../artifact-format.md#qwen38-flash-next-modelopt-nvfp4-and-mxfp8)):
  routed experts into GGML's `block_nvfp4` (one expert group each, their
  global scales gathered per layer); MXFP8 kept as E4M3 codes and E8M0
  scales; the n-gram table's 128 shards into one table of 90-byte rows (80
  code bytes, 10 scales); linear attention's value heads into tiled order,
  as llama.cpp's converter; norms with (1 + w) folded in, A as −exp(A_log).
  Vision tower, MTP block and the experts' static activation scales are not
  imported. **Since the prefill work's review** the importer writes the
  routed experts in the CUTLASS layout instead ([below](#prefill-d-085)):
  converter `m3-1+layout-a0d1980a9eddd1ad+modelopt_qwen38-031ed8750c9da55d`,
  on `spark-b` in 12 min 49 s wall with its deep verification (peak RSS
  2.52 GB, 8 repack processes), artifact
  `c4fb47a911207c11f935f932d05196dc1701aa0d886eac1b5e91934e554b5a93`:
  103,813,038,080 bytes in 20 shards, the same 24,627 groups, 1,600
  resources and 66,261 chunks, 192 expert arrays (four a layer), each
  expert group 2,764,800 bytes used and stored. Every other resource is the
  first import's.
- **Load** (`jitllm_qwen38_exec`): the artifact opened as untrusted input,
  bound to the compiled-in profile (`model/qwen38.h`), and read with
  coalesced direct reads through pinned staging into `cudaMalloc` memory:
  103,902,969,856 bytes in 7.45–8.14 s (12.8–13.9 GB/s; the
  CUTLASS-layout artifact's 103,802,306,560 in 7.71 s). Everything is
  resident, the 28.8 GB n-gram table included (row paging is the swap
  path's). The n-gram hash's constants are checked against the table before
  any chunk.
- **Resident expert layout:** as loaded, uniform stride over a per-layer slab, stock
  GGML `mul_mat_id` kernels, as DeepSeek V4's: S = 2,768,976 bytes (the
  stored group rounded to 144, NVFP4's 36-byte blocks and 16), 1,966,080
  bytes over the groups in all. The down projection's 640-element rows are
  not whole 512-element steps; the artifact's readable bytes hold the
  padding GGML reads past them, and the graph marks those weights so
  (`validate_ext.h MarkRowPaddingReadable`) only where the stride holds
  every slice's readable bytes, the last expert's included. `--layout-proof`: every layer's
  gate, up and down products at 1, 5 and 64 tokens (432 cases, MMVQ and MMQ)
  over the slab and the packed reference: 129,024,000 outputs, 0 differing.
  The CUTLASS layout (`kernels/ggml/moe_layout.h`,
  [artifact-format.md](../../artifact-format.md#executable-views)): gate and
  up as one block of 1,280 rows, then down, E2M1 codes then the swizzled
  E4M3 scales, 2,764,800 bytes a slot. The prefill work first wrote it over
  each slot of the GGML-layout artifact at load (3.9–4.0 s; now only with
  `--convert-experts`, whose `--layout-proof` converts every layer back and
  compares: 67,947,724,800 bytes, 0 differing); the importer now writes it,
  and the harness and the paged node read it as loaded, `S` = 2,764,800
  (no padding).
- **Graph** (`kernels/ggml/qwen38_graph.h`): llama.cpp's `qwen4exp.cpp`
  operation plan — hyper-connections (4 streams, rank 320), the n-gram
  layer (hash on the host, lookup, gate and dilated convolution), Gated
  DeltaNet with the fused gated delta rule, QSA (24 query and 2 KV heads of
  256, interleaved mrope over 64 dimensions, the indexer's pooled-block
  scores and a budget of 2,048 tokens plus the tail), 512 experts top-10
  plus the gated shared expert, and the head — planned with fusion off:
  5,052 steps a decode step at position 0, 5,436 for a 512-row prefill
  chunk, 5,808 with QSA's selection. The graph header lists where it
  departs from upstream's nodes. Since the prefill work ([below](#prefill-d-085))
  the graph runs jitLLM's fusions of those nodes and the CUTLASS expert
  path by default (`--unfused` builds the nodes): 2,717 steps a decode step
  at position 0, 3,017 for a prefill chunk, 3,389 with QSA's selection.
- **State** (`model/qwen38.h Qwen38StateLayout`): the QSA layers' F16 K and V
  caches and F32 indexer keys (a cell per position), the linear-attention
  layers' F32 recurrent state (128 × 128 × 48) and convolution history, and
  the n-gram layer's convolution history, as three D-068 representations:
  243,867,648 bytes at context 4,096.

## Kernel A/B (D-085)

Layer 0's (and layer 3's) weights on `spark-b`, 50 runs each, mean
(`--ab`). The decode-width rows repeat one layer's weights, so part of them
is served from L2: they rank the candidates, not the model's bandwidth.

| Operation | Candidate | 1 token | 8 tokens | 512 | 2,048 |
| --- | --- | ---: | ---: | ---: | ---: |
| NVFP4 experts, gate 2560→640, 10 of 512 | GGML MMVQ / MMQ (chosen first; now `--ggml-experts`) | 0.010 ms | 0.323 ms | 2.80 ms | 3.79 ms |
| NVFP4 experts, down 640→2560 | GGML MMVQ / MMQ (chosen first; now `--ggml-experts`) | 0.020 ms | 0.315 ms | 3.42 ms | 5.97 ms |
| NVFP4 grouped GEMM, 512 groups | CUTLASS 4.7.1 example 79d (sm_121a), m 16 / 40 per group (chosen for prefill, [below](#prefill-d-085)) | — | — | 2.50 ms (m 16, gate shape) | 2.71 ms (m 40) |
| MXFP8 QKV 2560→10240 | jitLLM vector product (chosen ≤ 8) | 0.080 ms | 0.233 ms | | |
| | BF16 weights, GGML MMVF / MMF | 0.198 ms | 0.220 ms | | |
| | dequantize to BF16, cuBLAS (chosen > 8) | | | 0.745 ms | |
| MXFP8 out 6144→2560 | jitLLM vector product | 0.030 ms | 0.149 ms | 0.517 ms (dequant + cuBLAS) | |
| | BF16 weights, MMVF / MMF | 0.126 ms | 0.138 ms | | |
| MXFP8 Q 2560→12288 | jitLLM vector product | 0.134 ms | 0.281 ms | 1.075 ms (dequant + cuBLAS) | |
| | BF16 weights, MMVF / MMF | 0.243 ms | 0.264 ms | | |

- **NVFP4 experts: GGML at first** (a new MMQ instance unit, `mmq-instance-nvfp4.cu`,
  in the source lock; MMVQ already instantiated NVFP4). CUTLASS's
  block-scaled grouped GEMM (BSD-3, v4.7.1, the version vLLM fetches) exists
  for SM12x and builds for `sm_121a` with the SDK's NVCC. Over 512 groups it
  took 2.50 ms for 16 rows a group (8,192 rows, against MMQ's 5,120 at 512
  tokens in 2.80 ms) and 2.71 ms for 40 (20,480 rows, against MMQ's 3.79 ms
  for gate and 5.97 ms for down at 2,048 tokens), measured on the gate's
  shape only; deferred to the prefill work, which adopted it
  ([below](#prefill-d-085)). The CuTe-DSL kernels were not considered
  (licensing.md).
- **MXFP8: jitLLM's own**, a vector product up to 8 columns (1.8–4.3× the BF16
  candidate at one token) and, wider, the weights dequantized to BF16 in the
  activations and GGML's cuBLAS product. Converting MXFP8 to BF16 at import
  (lossless) would need no kernel but doubles those weights' bytes (+2.8 GB)
  and is slower at decode widths.
- **n-gram table rows:** jitLLM's own lookup of ModelOpt NVFP4 rows (GGML's
  NVFP4 needs 64-value blocks; the rows hold 160 values).

## Prefill (D-085)

The first results left prefill at 0.37× the oracle's. The prefill work
took the levers a profile named, biggest first, on `spark-b` (`nsys`,
kernels; `--bench-prefill 8192`, synthetic tokens from an empty context,
best of three):

| Change (cumulative) | 2,048-row chunks | 4,096 | 8,192 |
| --- | ---: | ---: | ---: |
| Before (`a7a9faa`, GGML's nodes unfused) | 10.6 s (773 tok/s) | | |
| Hyper-connections' mix, combine and norms, the experts' SwiGLU and weighted sum as jitLLM kernels (`jitllm.hc.*`, `jitllm.moe.glu`, `jitllm.moe.combine`); F32 activations to BF16 once per product | 7.14 s (1,148) | | |
| The gated delta rule, 4 columns a warp (`jitllm.gdn.columns`, bit-identical to upstream's) | 6.91 s (1,185) | | |
| Routed experts on CUTLASS's NVFP4 grouped GEMM (below) | 5.56 s | 5.41 s | |
| Gated DeltaNet's convolution with its norms, and its gated norm into BF16 (`jitllm.gdn.conv`, `jitllm.gdn.norm_gate`) | 4.86 s | 4.67 s | 4.56 s |
| The gated delta rule over 16-token chunks staged in shared memory, 8 lanes a column (`jitllm.gdn.lanes`, past 16 rows) | 4.54 s (1,805) | 4.36 s (1,880) | 4.27 s (1,920) |

Each fused kernel is checked against the unfused nodes and an FP64
reference (`qwen38_fused_test`); all but the chunked delta rule are
bit-identical to GGML's nodes (before the chunked delta rule, the fused
graph with GGML's experts gave every logit of the six prompts equal to the
unfused graph's). The chunked
delta rule reorders the state's sums: NMSE about 2e-16 against upstream's
kernel.

**Profile, before and after** (GPU time a pass of 8,192 tokens): before,
in 2,048-row chunks, about 10.4 s: elementwise multiply 2.04 s and add
1.34 s (the hyper-connections' four streams, the n-gram layer, the
experts' weighted sum), NVFP4 MMQ 2.0 s, the gated delta rule 0.74 s,
cuBLAS's F32-to-BF16 conversions 0.65 s, BF16 GEMMs 0.51 s, sigmoid 0.34 s,
RMS norms 0.31 s. After, in 8,192-row chunks, about 4.1 s: the dense BF16
GEMMs (the MXFP8 linears dequantized, the shared expert, the head) 1.04 s,
the routed experts 0.67 s (CUTLASS 0.42 s, the weighted sum 0.12 s,
quantization 0.11 s), the hyper-connection kernels 0.85 s, flash attention
0.30 s, Gated DeltaNet's recurrence, convolution and gated norm 0.37 s, the
remaining elementwise nodes about 0.48 s and QSA's top-k 0.11 s.

**CUTLASS for the routed experts.** CUTLASS 4.7.1's SM120 block-scaled
grouped GEMM (`KernelPtrArrayTmaWarpSpecializedPingpong`, 128×128×256
tiles, BF16 out, `kernels/ggml/moe_cutlass.cu`) enters the source lock
(BSD-3, headers only, `third_party/sources.lock.json`). The prefill path
sorts the routing by expert (`jitllm.moe.route`), quantizes each token's
activations once to NVFP4 as GGML's MMQ does (`jitllm.moe.quantize`), runs
gate and up as one grouped GEMM, SwiGLU and the second quantization in one
kernel, the down GEMM, and the weighted sum in expert order with the shared
expert (`jitllm.moe.combine_sorted`). A/B over one layer's MoE block (the
whole block, routing to weighted sum, layer 0's weights; `--ab`, mean of
10 runs after a warm-up):

| Tokens | GGML MMQ | CUTLASS | Ratio |
| ---: | ---: | ---: | ---: |
| 512 | 9.60 ms | 8.26 ms | 1.16× |
| 2,048 | 15.57 ms | 9.92 ms | 1.57× |
| 8,192 | 45.76 ms | 16.14 ms | 2.84× |

End to end, 6.91 s to 5.56 s at 2,048-row chunks (−20%): adopted. The
grouped GEMM reads the CUTLASS layout, which the harness first wrote over
each slab slot at load (3.9–4.0 s, lossless). That would have added ~4 s
to every swap into Qwen3.8 on the paged node, which pages the model in
each time, so the importer writes the layout instead (a new artifact,
[above](#what-runs)): every logit of the six prompts' 32 teacher-forced
steps is the same, bit for bit, from the new artifact as from the old one
converted at load (`--convert-experts`, same build). So decode reads the
same layout: `jitllm.moe.gemv` computes up to 8 tokens'
routed products from it, the activations quantized to 8 bits a 16-value
block and 8-bit dot products as GGML's MMVQ does, gate and up with their
SwiGLU in one launch (NMSE 2.2e-5 against FP64, bound 5e-4, upstream's
test-backend-ops bound). MMVQ's `q8_1` blocks are 32 values; a block of
16 is each NVFP4 scale's own span, so one lane's dot product needs one
activation scale, and the finer blocks only lower the quantization error.
Decode on
`spark`, one run each in one session, 128 steps: 24.69 tok/s with it,
24.93 with the fused graph on GGML's MMVQ, 24.54 unfused (the graph before
this work).

**MXFP8 on tensor cores** (not adopted). A standalone A/B of CUTLASS's
MXFP8 GEMM (activations quantized to MXFP8, F32 out) against the current
dequantize-to-BF16 and cuBLAS at 4,096 rows: QKV 1.64 against 2.64 ms, z
0.98 against 1.59, out 0.88 against 1.64, Q 2.04 against 3.12; at 8,192
rows F32 out was slower for the wide products. About 0.3 s a pass of
8,192 tokens at best (estimated from those numbers, not measured end to
end), and it would quantize the activations the checkpoint's linears see,
which jitLLM keeps in BF16. Left for later: prefill meets the gate without
it.

**Chunk size.** Larger chunks help (fewer passes over the weights); the
memory they take is activations and the attention's mask and scores:

| Tokens (chunk rows) | jitLLM, 3 runs | Mia's vLLM, MTP off, deterministic | MTP 3 launch | Peak memory (jitLLM, context 8,704) |
| --- | ---: | ---: | ---: | ---: |
| 8,192 (8,192) | 1,918–1,924 tok/s: **0.91×** | 2,101 (8,266 tokens) | 2,066: 0.93× | 110.3–110.5 GiB (1.07× vLLM's 102.7) |
| 8,192 (4,096) | 1,879–1,883: 0.89× | 2,101 | 2,066: 0.91× | 104.7–104.9 GiB (1.02×) |
| 2,048 (2,048) | 1,947–1,972: 1.39–1.41× | 1,398 (~2,130) | 1,625: 1.20–1.21× | |
| 512 (512) | 1,334–1,346: 1.09–1.10× | 1,220 (~590) | 1,198: 1.11–1.12× | |

The baseline binary in the same session: 772–773, 802 and 749–751 tok/s.
From the CUTLASS-layout artifact (the review's build, `spark-b`, two
runs, each best of three): 8,192 tokens in 8,192-row chunks at
1,931.7–1,943.5 tok/s (0.92×), peak 110.4–110.5 GiB (1.08× vLLM's, under
D-085's 1.1×); decode 24.98 tok/s (128 steps, context 4,096; 0.99×).
Prefill at 8,192 tokens is within D-085's 10% of the oracle's in
8,192-row chunks, and just outside it (0.89×) in 4,096-row chunks; at 512
and 2,048 tokens jitLLM is ahead. What remains is the dense BF16 GEMMs
(the MXFP8 lever above), the hyper-connections' four-stream traffic and
the elementwise nodes left unfused (the n-gram layer's, QSA's).

## RE-030

Not in the way: Qwen3.8's QSA has no attention sinks (neither the
checkpoint nor llama.cpp's graph carries them), and RE-030 is an over-read of
the sinks only. At 12 query heads per KV head GGML's tensor-core kernel
groups 8 heads a tile, so a KV head's second tile has 4 padded heads whose Q
loads and output writes are bounded. `Qwen38OpsTest.TensorCoreAttentionAtTwelveQueryHeadsPerKvHead`
runs Qwen3.8's shape (24 and 2 heads of 256, 1, 7 and 64 tokens) against an
FP64 reference (NMSE at most 2.2e-7, bound 5e-4); `CheckFlashAttnMma` still
refuses sinks at that ratio. No patch.

## Results (`spark-b`, 2026-09-28)

GB10, driver 580.178.04. Raw outputs in `~/scratch/m3qwen/` on `spark-b`.

**Correctness:**

| Check | Result |
| --- | --- |
| Greedy, teacher-forced (bound 1) | 180 of 192 argmax equal to the oracle's; the 12 others at oracle margins 0.0 (×5), 0.125, 0.25, 0.625 and 1.0 (×4); bound 1.0 (jitLLM's own margin move, p95 0.941): **passes, with four disagreements at the bound** |
| jitLLM's two runs against each other | argmax equal at 188 of 192 steps; top-5 \|dlogprob\| RMS 0.493 |
| Logprobs vs oracle (bound 2) | \|dlogprob\| over the oracle's top-5: RMS 0.56–1.21 per prompt, max 2.0–6.1 |
| Free-running greedy | identical to the oracle's for the first 27, 3, 21, 29, 7 and 2 tokens (each diverges at its first disagreement above) |
| Perplexity (bound 3), 3,557 positions | jitLLM 14.4326, oracle 14.6582: **−1.5%**; top-1 is the next token at 43.89% of positions in both; top-1 agreement 84.6%; \|dNLL\| RMS 0.462, mean 0.280 before position 2,048 and 0.284 after (the indexer's selection) |
| The n-gram rows matter | the same text with every n-gram row shifted gives 14.8924 (+3.2%) |
| Expert layout (bound 4) | 0 of 129,024,000 outputs differ |
| State spill and restore (bound 5) | 0 of 47,677,440 logits differ (6 prompts, restored at step 16 of 32) |

**Again after the prefill work** (the default graph: fused kernels, the
chunked delta rule, CUTLASS's experts and `jitllm.moe.gemv`; on `spark`,
2026-09-28, raw outputs in `~/scratch/m3qpre/v2_*`): greedy 181 of 192
equal to the oracle's, the 11 others at oracle margins 0.0 (×4), 0.125,
0.25, 0.625, 0.75 and 1.0 (×3), so within both this run's bound (1.125,
jitLLM's own margin move p95 1.024, top-5 \|dlogprob\| RMS 0.465 between
its two runs) and the first results' 1.0: **passes**; \|dlogprob\| against
the oracle RMS 0.50–1.19 per prompt, max 2.5–5.8; free-running identical
for the first 22, 3, 21, 32, 9 and 2 tokens. Perplexity 14.4165 in
512-row chunks (−1.6%) and 14.4747 in 4,096-row chunks (−1.3%), top-1
agreement 84.3% and 84.0%. Expert layout 0 of 129,024,000 outputs differ
(GGML's layout, before the conversion), and the conversion back from
CUTLASS's layout gives all 67,947,724,800 bytes exactly; state spill and
restore 0 of 47,677,440 logits differ. The perplexity before position
2,048 repeats exactly (mean \|dNLL\| 0.26916 in this run and in the
previous build's, whose prefill path is the same); after it QSA's selection is not
repeatable run to run (ties in its top-k), so the whole text's perplexity
moves in the fourth digit (14.4800 and 14.4165 in 512-row chunks).

**Again from the CUTLASS-layout artifact** (`c4fb47a9…`, the review's
build, `spark-b`, 2026-09-28, raw outputs in `~/scratch/m3qpre-review/`):
the six prompts' 32 teacher-forced steps give every logit equal, bit for
bit, to the first artifact's with the experts converted at load
(`--convert-experts`, whose layout proof and conversion proof again found
0 of 129,024,000 outputs and 0 of 67,947,724,800 bytes differing); greedy
181 of 192, the 11 others near-ties at oracle margins up to 1.0 (bound
1.125): **passes**; perplexity 14.4800 in 512-row chunks (−1.2%) and
14.4747 in 4,096-row chunks (−1.3%), top-1 agreement 84.1% and 84.0%;
state spill and restore 0 of 47,677,440 logits differ.

**Bound 1's sensitivity** (from jitLLM's runs only; repeating either run
gives identical logits): the top-1 to top-2 margin moves between the two
runs by p50 0.22, p90 0.75, p95 0.94, p99 1.85 and at most 1.92 nats over
the 192 steps. The result passes with the bound at the 95th percentile or
above (1.0, or 2.0 at the maximum) and fails at the 90th (0.75: the four
disagreements at 1.0 fail). The margin test reads only the oracle's side:
at `haiku` step 3 jitLLM prefers its own token by 1.70 nats where the oracle
prefers the other by 1.0, a swing of 2.70 nats, beyond anything jitLLM's own
two runs produce; the other eleven swing at most 1.47.

The disagreements come with small oracle margins and jitLLM's kernel choice
alone moves logprobs by a comparable amount; the perplexity and the top-1
accuracy match the oracle's. The two engines quantize differently where
the checkpoint leaves it open: vLLM quantizes the experts' activations to
FP4 with the checkpoint's static scales, the MXFP8 linears' to MXFP8, and
keeps KV in FP8 and SSM state in BF16; jitLLM quantizes the experts'
activations to FP4 per row (prefill) or to 8 bits (decode), keeps the MXFP8
linears' in F32 or BF16, and keeps KV in F16 and all other state in F32.

**Performance and memory** (against M3's exit gate, not this slice's
bounds; neither side speculates):

| | jitLLM | Mia's vLLM (MTP off, deterministic mode) |
| --- | --- | --- |
| Prefill, 8,192 tokens | first results: 772 tok/s in 2,048-row chunks, 681 in 512-row (best of 3); after the prefill work: 1,918–1,924 in 8,192-row chunks, 1,879–1,883 in 4,096-row ([above](#prefill-d-085)) | 2,101 tok/s (8,266 tokens) |
| Decode | 24.77 tok/s (128 steps from an empty context, mean of 3 after a warm-up); 23.9–25.1 in the prompt runs; after the prefill work 24.53–24.60 against the earlier binary's 24.47–24.50 in the same session (3 runs each, alternating) | 25.12 / 25.33 tok/s (`prose` / `code`, 256 steps) |
| Load | 7.5–8.1 s (artifact, direct reads); the CUTLASS-layout artifact 7.7 s, as loaded (the prefill work's load-time rewrite added 3.9–4.0 s) | 10 min 52 s to `/health` |
| Peak memory (drop in `MemAvailable`) | 99.3–99.9 GiB at context 4,096; 104.7–104.9 GiB prefilling 8,192 tokens in 4,096-row chunks, 110.3–110.5 in one chunk (context 8,704) | 102.7 GiB |

So decode is 0.97–0.99× the oracle's and within D-085's 10%. Prefill was
0.37× the oracle's at first; after the prefill work it is 0.91× at 8,192
tokens in 8,192-row chunks, within D-085's 10%. Peak memory is 0.97×, but not like for like: jitLLM
holds a 4,096-token context (244 MB of state), vLLM its configured 262,144
(a 19.21 GiB KV pool). At vLLM's context jitLLM's state alone would add
about 7.5 GiB (30,720 bytes a position in the QSA layers' caches,
computed, not measured), about 1.05× before the chunk masks and indexer
scores, which also grow with the context. At first a
profile of the 2,048-row prefill (`nsys`, kernels) put the time in the
elementwise broadcasts the hyper-connections, the n-gram layer and the
experts' weighted sum make at four streams' width (multiply 17%, add 12%),
the NVFP4 MMQ products (16%), the gated delta rule (6%), cuBLAS's F32-to-BF16
activation conversion (5%) and the BF16 and dequantized MXFP8 GEMMs; the
prefill work took those levers ([above](#prefill-d-085)). Decode launches
each of its steps from the host (2,717 at position 0 after the prefill
work, 5,052 before; no CUDA graphs yet, an M3 item).

## Judgement calls

- **Importer:** a module of its own beside `import_m3.py` (which only
  dispatches to it), reusing `layout.py`'s container, index and verifier; its
  own writer, since `layout.build` copies source ranges verbatim. Its
  SHA-256 is in the converter version.
- **Representations:** GGML where the kernels read GGML types (NVFP4
  experts, BF16 matrices, F32 vectors); plain (checkpoint-layout) resources
  where GGML has no type (MXFP8, the n-gram table, the hash's I64
  constants). The indexer's fused q/k projection stays one product.
- **Not imported:** the vision tower, the MTP block (the speculation slice
  re-imports) and the experts' static activation scales (GGML quantizes
  activations per row).
- **Kernels:** as the A/B above; CUTLASS left for the prefill work, which
  adopted it.
- **Prefill work:** the fused kernels are jitLLM's own, planned as named
  implementations (D-053) with the unfused nodes still built by
  `--unfused`; CUTLASS enters the lock as a header-only component rather
  than a vendored copy; the CUTLASS layout replaces GGML's in the artifact
  (the same bytes permuted, so no second copy of 68 GB, and no rewrite at
  load, which every swap into the model would pay), which moves decode
  onto jitLLM's vector product (MMVQ's arithmetic, a scale per 16 values
  rather than 32); its arrays are GGML `I8` bytes, so the artifact format
  and readers are unchanged; MXFP8 on tensor cores is not taken, since the
  gate is met without changing what the linears' activations are.
- **Chunks of 1,024 rows and more** are rounded down to a multiple of 64
  (`ChunkRows`): the tensor-core flash attention's mask pre-pass reads
  whole column tiles past a chunk's last row, which the check refuses; the
  earlier binary fails the same way at 4,096-row chunks.
- **The down projection's short rows:** a flag the binder sets on weights
  whose padding is readable, rather than a relaxed check for every weight.
- **Graph departures** (qwen38_graph.h): an F32 indexer cache, the shared
  gate as a row dot product, transposed rows packed before concatenation,
  state written back with `set_rows`, the budget's selection not built when
  it keeps every cell.
- **Bound 1:** derived from jitLLM's own kernel noise, after the first
  comparison (see above).

## Limits

- One sequence; contexts to 8,704 run; prompts of 60–72 tokens, so QSA's
  selection is exercised only by the perplexity text.
- The oracle's tokens stand in for the native tokenizer's.
- Resident on `cudaMalloc` here. On the paged node, as device jobs over
  leased closures (one lease per request, D-093) with the n-gram table
  read by rows, its logits equal this harness's bit for bit on the six
  prompts' 32 steps, and from the CUTLASS-layout artifact it decodes at
  23.71–23.81 tok/s, 0.94× the oracle's (the job's device span 40.8 ms a
  step, and ~1.3 ms of host work outside the job: the rows read and the
  inputs built; the lease's round trip 0.01 ms;
  [swap](../fast-swap/swap.md#a-lease-per-request)).
- Not repeatable past 2,051 attended cells: the QSA indexer's top-k (GGML's
  radix select) picks among tied scores nondeterministically (RE-031), so
  the perplexity run, whose positions pass 2,048, need not repeat bit for
  bit (at 8,192 tokens, reruns differed from the 11th to the 16th chunk of
  512 on); the
  prompts, below that, do.
- The state's spill and restore is the harness's copy of one region; the
  swap path's spill format is M4's (D-086).
- The grouped GEMM is built for `sm_121a` only (`PlanMoeGemm` requires
  compute capability 12.1): elsewhere a CUTLASS-layout artifact's prefill
  is refused, and the GGML-layout artifact runs GGML's products. The
  harness's `--unfused`, `--layout-proof` and `--ab` read GGML's layout, so
  they take the first artifact.
- The unfused graph (`--unfused`) keeps GGML's `ssm_conv`, which past 32
  tokens loads whole 32-token blocks (RE-032): it pads a chunk that is not
  whole blocks and drops the padding's outputs, and `CheckSsmConv` refuses
  an unpadded one; the fused graph uses `ssm_conv` only for chunks of 1–2
  tokens.
