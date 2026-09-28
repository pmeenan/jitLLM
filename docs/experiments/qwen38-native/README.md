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
  imported.
- **Load** (`jitllm_qwen38_exec`): the artifact opened as untrusted input,
  bound to the compiled-in profile (`model/qwen38.h`), and read with
  coalesced direct reads through pinned staging into `cudaMalloc` memory:
  103,902,969,856 bytes in 7.45–8.14 s (12.8–13.9 GB/s). Everything is
  resident, the 28.8 GB n-gram table included (row paging is the swap
  path's). The n-gram hash's constants are checked against the table before
  any chunk.
- **Resident expert layout:** uniform stride over a per-layer slab, stock
  GGML `mul_mat_id` kernels, as DeepSeek V4's: S = 2,768,976 bytes (the
  stored group rounded to 144, NVFP4's 36-byte blocks and 16), 1,966,080
  bytes over the groups in all. The down projection's 640-element rows are
  not whole 512-element steps; the artifact's readable bytes hold the
  padding GGML reads past them, and the graph marks those weights so
  (`validate_ext.h MarkRowPaddingReadable`) only where the stride holds
  every slice's readable bytes, the last expert's included. `--layout-proof`: every layer's
  gate, up and down products at 1, 5 and 64 tokens (432 cases, MMVQ and MMQ)
  over the slab and the packed reference: 129,024,000 outputs, 0 differing.
- **Graph** (`kernels/ggml/qwen38_graph.h`): llama.cpp's `qwen4exp.cpp`
  operation plan — hyper-connections (4 streams, rank 320), the n-gram
  layer (hash on the host, lookup, gate and dilated convolution), Gated
  DeltaNet with the fused gated delta rule, QSA (24 query and 2 KV heads of
  256, interleaved mrope over 64 dimensions, the indexer's pooled-block
  scores and a budget of 2,048 tokens plus the tail), 512 experts top-10
  plus the gated shared expert, and the head — planned with fusion off:
  5,052 steps a decode step at position 0, 5,436 for a 512-row prefill
  chunk, 5,808 with QSA's selection. The graph header lists where it
  departs from upstream's nodes.
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
| NVFP4 experts, gate 2560→640, 10 of 512 | GGML MMVQ / MMQ (chosen) | 0.010 ms | 0.323 ms | 2.80 ms | 3.79 ms |
| NVFP4 experts, down 640→2560 | GGML MMVQ / MMQ (chosen) | 0.020 ms | 0.315 ms | 3.42 ms | 5.97 ms |
| NVFP4 grouped GEMM, 512 groups | CUTLASS 4.7.1 example 79d (sm_121a), m 16 / 40 per group | — | — | 2.50 ms (m 16, gate shape) | 2.71 ms (m 40) |
| MXFP8 QKV 2560→10240 | jitLLM vector product (chosen ≤ 8) | 0.080 ms | 0.233 ms | | |
| | BF16 weights, GGML MMVF / MMF | 0.198 ms | 0.220 ms | | |
| | dequantize to BF16, cuBLAS (chosen > 8) | | | 0.745 ms | |
| MXFP8 out 6144→2560 | jitLLM vector product | 0.030 ms | 0.149 ms | 0.517 ms (dequant + cuBLAS) | |
| | BF16 weights, MMVF / MMF | 0.126 ms | 0.138 ms | | |
| MXFP8 Q 2560→12288 | jitLLM vector product | 0.134 ms | 0.281 ms | 1.075 ms (dequant + cuBLAS) | |
| | BF16 weights, MMVF / MMF | 0.243 ms | 0.264 ms | | |

- **NVFP4 experts: GGML** (a new MMQ instance unit, `mmq-instance-nvfp4.cu`,
  in the source lock; MMVQ already instantiated NVFP4). CUTLASS's
  block-scaled grouped GEMM (BSD-3, v4.7.1, the version vLLM fetches) exists
  for SM12x and builds for `sm_121a` with the SDK's NVCC (built and run in
  scratch, nothing incorporated). Over 512 groups it took 2.50 ms for 16
  rows a group (8,192 rows, against MMQ's 5,120 at 512 tokens in 2.80 ms) and
  2.71 ms for 40 (20,480 rows, against MMQ's 3.79 ms for gate and 5.97 ms
  for down at 2,048 tokens): about 1.4× MMQ's gate and 2.2× its down at
  prefill widths, measured on the gate's shape only. It would be a new
  source-lock component (large), with the MoE's grouped problem setup, the
  scale-factor swizzle and routing on the device still to build. vLLM's
  Apache-2.0 NVFP4 paths on this GPU are CUTLASS-based too (FlashInfer's
  CUTLASS MoE, which Mia's engine runs; vLLM's own `nvfp4_scaled_mm_sm120`),
  so they cost the same integration. Deferred: the prefill lever below. The
  CuTe-DSL kernels were not considered (licensing.md).
- **MXFP8: jitLLM's own**, a vector product up to 8 columns (1.8–4.3× the BF16
  candidate at one token) and, wider, the weights dequantized to BF16 in the
  activations and GGML's cuBLAS product. Converting MXFP8 to BF16 at import
  (lossless) would need no kernel but doubles those weights' bytes (+2.8 GB)
  and is slower at decode widths.
- **n-gram table rows:** jitLLM's own lookup of ModelOpt NVFP4 rows (GGML's
  NVFP4 needs 64-value blocks; the rows hold 160 values).

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
| Prefill, 8,192 tokens | 772 tok/s in 2,048-row chunks, 681 in 512-row (best of 3) | 2,101 tok/s (8,266 tokens) |
| Decode | 24.77 tok/s (128 steps from an empty context, mean of 3 after a warm-up); 23.9–25.1 in the prompt runs | 25.12 / 25.33 tok/s (`prose` / `code`, 256 steps) |
| Load | 7.5–8.1 s (artifact, direct reads) | 10 min 52 s to `/health` |
| Peak memory (drop in `MemAvailable`) | 99.3–99.9 GiB at context 4,096 | 102.7 GiB |

So decode is 0.99× the oracle's and within D-085's 10%. **Prefill is 0.37×
the oracle's and fails D-085's 10% gate; it stays open** (M3's exit needs
at least about 0.9×). Peak memory is 0.97×, but not like for like: jitLLM
holds a 4,096-token context (244 MB of state), vLLM its configured 262,144
(a 19.21 GiB KV pool). At vLLM's context jitLLM's state alone would add
about 7.5 GiB (30,720 bytes a position in the QSA layers' caches,
computed, not measured), about 1.05× before the chunk masks and indexer
scores, which also grow with the context. A
profile of the 2,048-row prefill (`nsys`, kernels) puts the time in the
elementwise broadcasts the hyper-connections, the n-gram layer and the
experts' weighted sum make at four streams' width (multiply 17%, add 12%),
the NVFP4 MMQ products (16%), the gated delta rule (6%), cuBLAS's F32-to-BF16
activation conversion (5%) and the BF16 and dequantized MXFP8 GEMMs; the
levers are fused hyper-connection and MoE-reduction kernels, the grouped
NVFP4 GEMM above, and an MXFP8 tensor-core GEMM. Decode launches each of its
5,052 steps from the host (no CUDA graphs yet, an M3 item).

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
- **Kernels:** as the A/B above; CUTLASS left for the prefill work.
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
- Resident on `cudaMalloc`, not yet as device jobs over leased closures on
  the paged node (as DeepSeek V4's).
- The state's spill and restore is the harness's copy of one region; the
  swap path's spill format is M4's (D-086).
