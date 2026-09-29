<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Long context: the baseline (2026-09-29)

This is phase 1 of M3's **Long context** item
([plan.md](../../plan.md#m3--single-spark-fast-full-swap--in-progress)).
It measures each LLM's one-Spark maximum context and runs the context
ladder through `jitllm-runtime` and through the same-format comparators on
the same prompts. The gaps it finds set the optimization slices of
phase 2. Coarse by design (D-085): one run per depth. Every number here is
**measured** on `spark` or `spark-b` (GB10, driver 580.178.04) unless
marked **computed**. jitLLM was measured at `6c182c3` plus this change's
working tree (the fixes below).

## Headline

Through `jitllm-runtime`'s chat route against the same-format comparator
on the same prompt, jitLLM first. DeepSeek against llama.cpp b11254
(UD-Q2_K_XL; speculative: DSpark on both sides); Qwen3.8 against Mia's
vLLM (NVFP4; prefill against the faster of its two launches, plain decode
against its deterministic MTP-off launch, speculative against MTP 3 with
jitLLM's MTP depth 2). "—": not run (below).

| Model, depth | Prefill, tok/s (ratio) | Plain decode, tok/s (ratio) | Speculative decode, tok/s (ratio) | Retrieval (jitLLM) |
| --- | --- | --- | --- | --- |
| DeepSeek, 8K (M3's earlier runs) | 463 / 352 (b10964) | 21.9–22.2 / 19.9 (b10964) | 31.6 / 30.8 | — |
| DeepSeek, 32K | 333 / 286 (1.16×) | 14.7 / 18.8 (**0.78×**) | 31.3 / 30.6 (1.02×) | pass |
| DeepSeek, 64K | 230 / 275 (**0.84×**) | 10.8 / 18.0 (**0.60×**) | 21.4 / 29.0 (**0.74×**) | pass |
| DeepSeek, 128K | — / 258 | — / 16.8 | — / 30.6 | — |
| DeepSeek, 256K | — / 229 | — / 14.7 | — / 28.4 | — |
| Qwen3.8, 8K (M3's earlier runs) | 2,320 / 2,101 | 27.4 / 25.1 | 40.1–42.5 / 37.9 | — |
| Qwen3.8, 32K | 2,053 / 1,729 (1.19×) | 21.8 / 24.6 (**0.89×**) | 35.4 / 37.3 (0.95×) | pass |
| Qwen3.8, 64K | 1,367 / 1,947 (**0.70×**) | 17.1 / 24.1 (**0.71×**) | refused / 40.5 | pass |
| Qwen3.8, 128K | 859 / 1,909 (**0.45×**) | 11.9 / 23.8 (**0.50×**) | refused / 48.7 | pass |
| Qwen3.8, 256K (its maximum) | 487 / 1,775 (**0.27×**) | 6.8 / 23.7 (**0.29×**) | refused / 37.7 | pass |

**The finding is the slope.** Both comparators are nearly flat with depth
(llama.cpp's decode falls 22% from 32K to 256K, Mia's 4%); jitLLM's
per-token cost grows linearly: DeepSeek's decode step costs 0.7–0.8 ms
more per 1K tokens of context (llama.cpp's 0.07), Qwen3.8's 0.3–0.4 ms
(Mia's 0.01; computed from the rates and the profile's step times). The
profile ([Where the time goes](#where-the-time-goes))
shows why: jitLLM's attention does dense work over every cached cell,
where both architectures only need a window and a fixed top-k.

**Maximum context on one Spark** (the runtime's memory guard with its 4 GiB
margin, `kUncountedMargin`; one model registered):

| Model | Configured | Plain | Speculative | Bound by |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8 Flash Next | 262,144 | **262,144**, verified with a 258,633-token prompt (2,040-row chunks, RE-037) | **32,768** | plain: its configured maximum; speculative: the MTP drafter's selection works only up to 8,192 blocks, so registration refuses more |
| DeepSeek V4 Flash | 262,144 (trained 1,048,576) | **262,144**, served (32K and 64K prompts at that context); the guard passes by 1.2–1.3 GiB and refused once | **143,360** (the guard; not run with a prompt) | the configuration's bound, then memory: the full-size window cache is 49.7 KiB a token |

## Contents

- [The maximum contexts](#the-maximum-contexts): state per token, the
  guard, and the runs at each maximum.
- [Corpus and harness](#corpus-and-harness): the prompts, the retrieval
  check, the perplexity text, and how to rebuild them.
- [Comparators](#comparators): the llama.cpp pin with sparse
  flash-attention prefill, Mia's vLLM, and how each ran.
- [Results at depth](#results-at-depth): per model and depth; [Where the
  time goes](#where-the-time-goes): the per-kernel profile, and what the
  architectures allow.
- [Correctness at depth](#correctness-at-depth).
- [Swap with a long saved context](#swap-with-a-long-saved-context).
- [Turn-to-turn reuse](#turn-to-turn-reuse).
- [Memory and the guard's margin](#memory-and-the-guards-margin).
- [Fixed to measure](#fixed-to-measure): the blockers this phase fixed.
- [Gap list for phase 2](#gap-list-for-phase-2), ranked, with levers and
  effort.
- [Not run, and why](#not-run-and-why); [Reproduce](#reproduce).

## The maximum contexts

### State per token, from the model layer (computed)

| Model | Per token | Fixed | At 262,144 | At 1,048,576 |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek V4 Flash (`model/dsv4.h` `Dsv4State`) | 50,912 B: the window cache 44,032 (43 layers × 512 F16, a cell per position: `swa_full`), CSA keys 5,376 (21 layers, one row per 4 positions), indexer keys 1,344, HCA keys 160 (20 layers, per 128) | ~12 MiB of compressor rings; DSpark's window ring | 12.4 GiB | 49.7 GiB |
| … with the window cache as a ring of window + chunk rows (not built) | 6,880 B | ~95 MiB at 2,048-row chunks | 1.7 GiB | 6.7 GiB |
| Qwen3.8 Flash Next (`model/qwen38.h` `Qwen38State`) | 30,720 B: 12 QSA layers × (K and V 1,024 B each in F16, indexer keys 512 B in F32) | 118 MB: 36 Gated DeltaNet layers' recurrent (3 MiB) and convolution state | 7.5 GiB | — (configured maximum 262,144) |
| … its MTP drafter's layer | +2,560 B | — | +0.6 GiB | — |

Mia's vLLM keeps Qwen3.8's KV in FP8 (half of jitLLM's F16 per token) and
sizes its pool at 16–19 GiB for four sequences; llama.cpp keeps DeepSeek's
window as a ring.

### What the runtime reserves: the guard

At registration the runtime maps each model's own memory (its state at the
configured context, the workspace for the largest chunk, the chunk-input
staging) for the node's life, and refuses to serve unless the largest
model's weights plus a 4 GiB margin fit the memory then available
(`kUncountedMargin`, runtime-serving.md). Measured at startup on `spark-b`
(one model registered, nothing else running; "fixed" is the node's mapped
memory, the workspace part of it):

| Model, mode | Context | Prefill chunk | Fixed, GiB (workspace) | Available after, GiB | Weights + 4, GiB | Serves |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DeepSeek, DSpark | 65,536 | 2,048 | 5.90 (1.89) | 110.34 | 104.4 | yes |
| DeepSeek, DSpark | 131,072 | 2,048 | 10.66 (2.79) | 105.40 | 104.4 | yes |
| DeepSeek, DSpark | 143,360 | 2,048 | 11.72 (3.12) | 104.58 | 104.4 | yes |
| DeepSeek, DSpark | 147,456 | 2,048 | 12.0 | 104.3 | 104.4 | no |
| DeepSeek, plain | 262,144 | 2,048 | 20.36 (4.78) | 95.53–95.62 (93.8 once) | 94.3 | yes, by 1.2–1.3 GiB (refused once) |
| Qwen3.8, plain | 131,072 | 4,096 | 21.80 (10.38) | 94.2 | 73.9 | yes |
| Qwen3.8, plain | 196,608 | 4,096 | 31.86 (15.06) | 83.76 | 73.9 | yes |
| Qwen3.8, plain | 262,144 | 4,096 | — | — | — | no: RE-037 |
| Qwen3.8, plain | 262,144 | 3,584 | 37.65 (17.29) | 78.01 | 73.9 | yes (then failed at 147K: RE-037) |
| Qwen3.8, plain | 262,144 | 2,040 (the cap since RE-037) | 24.74 (9.84) | 89.71 | 73.9 | yes |
| Qwen3.8, MTP | 32,768 | 4,096 | 3.87 (1.77) | 112.49 | 79.3 | yes |
| Qwen3.8, MTP | 36,864 and up | 4,096 | — | — | — | no: "the drafter selects on the device only" |

So past the state itself, the runtime's fixed memory grows about 77 KiB
per token for DeepSeek (state 49.7, chunk-input staging ~12 for its
double-buffered F16 masks [n_kv, 2,048], workspace ~15) and 161 KiB per
token for Qwen3.8 at 4,096-row chunks (state 30; the rest the
[n_kv, rows] masks and scores in the staging and workspace), 94 KiB at
2,040 rows (computed from the table).

### The maxima, verified

- **Qwen3.8: 262,144 (its configured maximum), plain.** Verified by the
  256K retrieval prompt (258,633 tokens rendered) at `context = 262144`:
  prefilled in 2,040-row chunks, answered with all three codenames, peak
  101.2 GiB. It needed the RE-037 fix (below): before it the service
  refused to start at 4,096 rows and failed at 147K at 3,584.
  **Speculative: 32,768.** Past 8,192 selection blocks the MTP drafter's
  graph has no host-mask fallback (the target falls back to GGML's
  top-k; the drafter is refused at registration), so a context above
  32,768 cannot speculate at all.
- **DeepSeek V4 Flash: 262,144, plain** (the configuration's upper bound,
  and within 1.3 GiB of the guard): the runtime registered and served at
  `context = 262144`, with the 32K and 64K prompts; peak 116.0 GiB, the
  whole Spark. A prompt near the maximum was not run: the 128K prompt
  outran the chat route's 600 s request deadline (the state kept 108,544
  tokens), and then the owner stopped the rungs past 64K until the
  scaling is fixed. It needed the RE-038 fix: before it every prompt past
  about 52K failed. **With DSpark: 143,360** (the guard refuses 147,456),
  from the guard alone. The trained 1,048,576 would need
  49.7 GiB of state as built, beside 90.3 GiB of weights (100.4 with
  DSpark): it does not fit, and the configuration refuses a context above
  262,144 anyway. With the window cache as a ring, 1M is 6.7 GiB of state
  (computed), and fits beside the weights and the drafter.

## Corpus and harness

Built by [build_prompts.py](build_prompts.py) from
[corpus.json](corpus.json), both fixed before the first run, and run on a
Spark in the pinned PyTorch image (its `tokenizers`); the prompts stay
outside Git on both Sparks under `~/.local/share/jitllm/m3lc/prompts/`,
identified by their content hashes (below).

- **Coding context:** source files of llama.cpp at `8019dc563` (b11254,
  MIT, `LICENSE` SHA-256 `94f29bbe…`), in a fixed order (`tools/server`
  first, then `src`, `include`, `common`, `ggml`, the other tools,
  examples, tests), each whole file that still fits the rung's budget
  (files over 256 KiB skipped), each introduced by `=== FILE: path ===`,
  then one question: quote three notes' values, then explain how
  llama-server picks a slot and when it reuses a cached prompt. Rungs are
  sized with each model's own tokenizer so the rendered prompt plus 512
  generated tokens fits the rung: 32K (32,768), 64K, 128K, 256K (262,144)
  and 1M (1,048,576, DeepSeek only).
- **Retrieval check:** three notes (A at about 10% of the files' tokens,
  B at 50%, C at 90%, placed at file boundaries; recorded token offsets)
  in every rung. The first prompts called their values "maintenance
  passphrases", and at 256K Mia's vLLM reasoned they were planted secrets
  and declined to repeat them; so the check has its own prompts (`-r`, the
  same files) whose notes give neutral "release codenames" and whose one
  question asks for them in three lines. A rung passes when all three
  appear in the output (reasoning or answer).
- **Perplexity:** *War and Peace* (Project Gutenberg #2600, public domain
  in the USA; the download's SHA-256 `2d5bb2ad…`, its header and footer
  cut: `ppl.txt`, SHA-256 `c7156148…`, 777,232 DeepSeek tokens). Each
  oracle tokenizes it itself and jitLLM is fed the oracle's window of IDs;
  scored is the second half of one window (llama-perplexity's rule for one
  chunk): 32,768 and 131,072 tokens. The book is heavily memorized (a
  32K-window perplexity of 1.4–1.9), so only ratios mean anything.
- **Turn reuse:** `s64k`, a 64K rung built with 4,096 tokens of room for
  two follow-up questions (write a function; then what changes for
  multimodal chunks).

| Prompt | DeepSeek content tokens (SHA-256 of the content) | Qwen3.8 content tokens (SHA-256) |
| --- | --- | --- |
| `8k` | — | 7,534 (`b2d24aa0…`) |
| `32k` | 31,701 (`1676feba…`) | 31,691 (`c7928ea7…`) |
| `64k` | 64,443 (`92ff03c0…`) | 64,058 (`1aa0c238…`) |
| `128k` | 128,817 (`e94961ec…`) | 128,747 (`b7154f4b…`) |
| `256k` | 258,852 (`d1e6ac95…`) | 258,650 (`14cf7d90…`) |
| `1m` | 1,037,954 (`aad4e576…`) | — |
| `s64k` (turn reuse) | 61,111 (`71a7a214…`) | 60,850 (`607f2bb4…`) |
| `32k-r` … `256k-r` (retrieval) | 31,624 (`677ca618…`), 64,488 (`63da4e81…`), 128,740 (`7ff7d36f…`), 258,775 (`a534146a…`) | 31,622 (`b182da30…`), 63,989 (`82129ba6…`), 128,985 (`7ff7d36f…`), 258,581 (`ab1912e3…`) |

The engines' own counts, rendered with the template, are 30–70 tokens
more (`32k`: 31,705 for DeepSeek, 31,743 for Qwen3.8). The builder
writes a `manifest.json` per model with every prompt's counts, files and
note offsets.

The harness is [longctx.py](longctx.py) (it reuses the M3 baselines'
[baseline.py](../fast-swap/baseline.py) for its client and memory
sampler) and [judge.py](judge.py). Prefill = first streamed piece −
request sent (one decode step included); decode = (tokens − 1) ÷ (last
piece − first piece), 512 generated tokens greedy (fewer when the model
stopped); peak memory = the drop in `MemAvailable` from before the engine
started, sampled every 200 ms. jitLLM runs start with a short warm-up
request so no measured prefill includes paging the model in.

## Comparators

- **llama.cpp for DeepSeek**, UD-Q2_K_XL, same-format oracle and
  comparator: built by us on `spark-b` from `8019dc563` (b11254, which has
  sparse flash-attention prefill for DeepSeek V4, #29298, and for Qwen3.8,
  #28770) with upstream's `.devops/cuda.Dockerfile`, target `full`, CUDA
  13.4.1, `CUDA_DOCKER_ARCH=121a-real` (upstream's default list includes
  121a-real; the GB10's code is the same), local image
  `jitllm-llamacpp:b11254-cuda13`, `sha256:6dd02591…`, copied to `spark`
  ([pins.json](../fast-swap/pins.json)). Upstream publishes no image for
  b11254 yet (the newest is b11243). The b10964 pin stays the 8K oracle.
  Server arguments as the M3 baselines: `-ngl all -fa on -c 262144 -np 1
  --fit off -cram 0`; DSpark adds `-md dspark-…-Q8_0.gguf --spec-type
  draft-dspark --spec-draft-n-max 3 -ngld all`. Requests are the prompt's
  IDs (the server's own template and tokenizer) to `/completion` with
  `cache_prompt: false` and five log-probabilities.
- **Mia's vLLM for Qwen3.8**, NVFP4, oracle and comparator: the recipe
  and image the baselines pinned, on `spark` with its `.env` (262,144
  context, FP8 KV, 2,048-token prefill chunks). Two cold launches: the
  oracle's deterministic mode with MTP off (`MTP_NUM_SPECULATIVE_TOKENS=0
  VLLM_QSA_DET_TOPK=1 VLLM_MOE_DET_FINALIZE=1`; ready in 622 s), and the
  default MTP-3 launch (685 s). Each request a unique `cache_salt` (no
  prefix reuse), prompt IDs from `/tokenize` with the template.
- TensorFold was not run (below).

## Results at depth

One run per depth, 512 greedy tokens (fewer where the model stopped).
Prompt tokens are the engine's count; "peak" is the drop in
`MemAvailable` over the whole run (so it is the deepest rung's).

### DeepSeek V4 Flash (UD-Q2_K_XL)

| Depth | Engine | Prompt tokens | Prefill s (tok/s) | Decode tok/s | Speculative decode tok/s (acceptance) | Peak GiB (context) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32K | jitLLM (plain on `spark-b`, DSpark on `spark`; 2,048-row chunks) | 31,705 | 95.2 (333) | 14.66 | 31.34 (context 65,536; prefill 332 tok/s with DSpark) | 116.0 (262,144) |
| 32K | llama.cpp b11254 (`spark-b`; DSpark on `spark`) | 31,705 | 110.8 (286) | 18.82 | 30.61 (0.62) | 95.8 (262,144); 107.1 with DSpark |
| 64K | jitLLM | 64,447 | 279.8 (230) | 10.79 | 21.40 (prefill 242 tok/s) | 110.8 with DSpark (65,536) |
| 64K | llama.cpp | 64,447 | 234.1 (275) | 18.01 | 28.96 (0.59) | |
| 128K | jitLLM | 128,821 | the route's 600 s deadline stopped it at 108,544 tokens | — | — | |
| 128K | llama.cpp | 128,821 | 498.6 (258) | 16.76 | 30.59 (0.71) | |
| 256K | llama.cpp | 258,856 | 1,128.7 (229) | 14.73 | 28.41 (0.74) | |

The resident harness (`jitllm_dsv4_exec`, the same fast plan) ran the
128K prompt for the correctness check (below): prefill 812.5 s (159
tok/s), decode 7.4 tok/s (forced tokens, launch by launch).
llama.cpp's prefill with DSpark loaded was within 4% of without at every
depth (276, 269, 252, 225 tok/s). llama.cpp's own 8K numbers are the M3
baselines' at the b10964 pin (352 / 19.9); b11254 was not run at 8K.

### Qwen3.8 Flash Next (NVFP4)

| Depth | Engine | Prompt tokens | Prefill s (tok/s) | Decode tok/s | Speculative decode tok/s | Peak GiB (context) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 32K | jitLLM (`spark-b`, 4,096-row chunks; MTP at context 32,768) | 31,743 | 15.5 (2,053) | 21.81 | 35.39 (MTP 2) | 100.6 (131,072); 79.1 with MTP (32,768) |
| 32K | Mia's vLLM (`spark`): deterministic, MTP off / MTP 3 | 31,743 | 19.0 (1,673) / 18.4 (1,729) | 24.62 | 37.25 (0.45) | 102.4 / 100.7 (262,144 pool) |
| 64K | jitLLM | 64,110 | 46.9 (1,367) | 17.10 | refused | |
| 64K | Mia's vLLM | 64,110 | 32.9 (1,947) / 33.9 (1,891) | 24.12 | 40.45 (0.50) | |
| 128K | jitLLM | 128,799 | 149.9 (859) | 11.90 | refused | |
| 128K | Mia's vLLM | 128,799 | 67.5 (1,909) / 69.2 (1,862) | 23.80 | 48.65 (0.69) | |
| 256K | jitLLM (context 262,144, 2,040-row chunks; the `256k-r` prompt) | 258,633 | 531.4 (487) | 6.83 | refused | 101.2 (262,144) |
| 256K | Mia's vLLM | 258,702 | 145.7 (1,775) / 149.6 (1,729) | 23.65 | 37.71 (0.44) | |

jitLLM's 32K row is the warm run (the first attempt's included the
model's 5.6 s page-in: 1,491 tok/s); its 64K and 128K rows followed a
warm request in the same process. Mia's acceptance is vLLM's accepted ÷
drafted tokens; the chat route does not report jitLLM's. Mia's 128K MTP
rate reflects a higher acceptance on that prompt's answer.

## Where the time goes

`nsys` (`--trace=cuda`) over the resident harnesses, which run the same
fast plan and kernels as the runtime, launch by launch: one run per depth
with a prompt of whole chunks and 2 decode steps
([profile.py](profile.py) splits the kernels at each chunk's logits copy).
Device busy time of the prompt's last prefill chunk (DeepSeek 2,048 rows,
Qwen3.8 4,096) and of the last decode step, by operation family, on
`spark` (2026-09-29):

**DeepSeek V4 Flash**, chunks at 6,144 / 28,672 / 61,440 tokens of
context:

| ms | Prefill chunk 8K | 32K | 64K | Decode step 8K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash attention | 1,289 | 4,034 | 7,856 | 2.8 | 7.6 | 15.0 |
| Concatenating the window cells and compressed rows into K (and copies) | 42 | 94 | 171 | 3.2 | 13.0 | 27.3 |
| Lightning indexer scoring (and its Hadamard) | 128 | 439 | 887 | 0.2 | 0.5 | 0.8 |
| Indexer top-k (GGML's radix select) | 9 | 41 | 82 | 0.6 | 0.5 | 0.5 |
| Sparse-index prep, masks and fills | 9 | 24 | 44 | 0.7 | 1.0 | 1.3 |
| Everything else (weights, MoE, norms, hyper-connections, compressors) | 3,228 | 3,300 | 3,291 | 47.8 | 49.8 | 50.7 |
| **Total** | **4,704** | **7,932** | **12,331** | **55.2** | **72.5** | **95.7** |

The lightning indexer's own share, split: scoring
2.7% / 5.5% / 7.2% of the prefill chunk at 8K / 32K / 64K and 0.4% /
0.7% / 0.8% of the decode step; its top-k 0.2% / 0.5% / 0.7% and 1.1% /
0.7% / 0.5%; its data movement nothing separate (its keys are read in
place from the state by the scoring kernel; the gathers and masks after
the top-k belong to attention, the "sparse-index prep" row).
From 8K to 64K the decode step grows 40.5 ms: 60% the K concatenation,
30% flash attention, 1% the indexer. The prefill chunk grows 7.6 s: 86%
flash attention, 10% the indexer. The weights' share is flat. The cause
is structural: jitLLM's DeepSeek state keeps a window-cache cell per
position (`swa_full`, as llama.cpp's library default and the exact-mode
oracle have it), and every layer's attention concatenates all of those
cells (43 layers) with its compressed rows into one K and attends over
them under a mask, so the work is O(context) per token in every layer,
though only the last 128 positions and 512 selected compressed rows
contribute. llama-server runs with `swa_full` off (a ring of window plus
micro-batch cells), and #28770/#29298's sparse flash attention gathers the
selected cells, so its per-token work is flat except the indexer.

**Qwen3.8 Flash Next**, prefill chunks at 4,096 / 28,672 (2,048 rows) /
57,344 tokens, decode at 8,195 / 30,723 / 61,443:

| ms | Prefill chunk 8K | 32K (2,048 rows) | 64K | Decode step 8K | 32K | 64K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flash attention (dense, masked to the 2,051 selected cells) | 157 | 306 | 1,268 | 1.0 | 3.6 | 6.7 |
| QSA selection: jitLLM's device select (to 8,192 blocks) | 16 | 42 | — | 0.4 | 1.5 | — |
| QSA selection: GGML fallback (scores expanded to cells, ReLU, adds, transposes, radix top-k, masks) | — | — | 1,130 | — | — | 7.3 |
| Pooling the indexer's raw keys into blocks (gather) | 0.5 | 1.8 | 73 | 0.3 | 1.3 | 3.0 |
| Gated DeltaNet (lanes, convolution, norm and gate) | 168 | 85 | 170 | 1.2 | 1.3 | 1.4 |
| n-gram rows (the lookup kernel; the reads are host-side) | 4.1 | 2.3 | 4.1 | 0.01 | 0.01 | 0.01 |
| Everything else (weights, MoE, hyper-connections) | 1,194 | 739 | 1,175 | 38.2 | 39.6 | 39.2 |
| **Total** | **1,540** | **1,176** | **3,820** | **41.1** | **47.3** | **57.6** |

The Gated DeltaNet layers and the n-gram rows are flat (O(1) in depth, as
the architecture promises). Growth from 8K to 64K: decode +16.5 ms (flash
attention 5.7, the selection 6.9, block pooling 2.7); prefill +2.3 s per
4,096 rows (flash attention 1.1 s, the selection 1.1 s). Below 32,768
cells jitLLM's device select is cheap and only attention and pooling
grow; past it the GGML fallback also makes the indexer's scores
cell-sized ([n_kv, rows] F32: the RE-037 tensors) and builds both masks
on the host for every chunk (host time, not in these device times).

**Qwen3.8 with MTP, through the runtime** (its decode graphs, timed per
graph launch by `nsys`'s graph trace; the 8K and 32K prompts, context
8,704 and 32,768; a depth-2 draft then a batched verify each step; `spark`):

| Mean device time per launch, ms | 8K | 32K | Growth |
| --- | ---: | ---: | ---: |
| Plain decode step (MTP off) | 37.8 | 45.0 | +19% |
| Verify (the target over the anchor and drafts) | 48.5 | 55.2 | +14% |
| One drafter pass (its one QSA layer) | 5.1–5.4 | 6.2–6.6 | +21% |
| Speculative decode, tok/s (under `nsys`) | 39.1 | 34.8 | −11% |

The drafter grows like the target: its QSA layer has the same dense
attention and per-step pooling, and past 32,768 cells it has no selection
at all (gap 2.1). The verify grows by about as much as a plain step
(6.7 ms against 7.2), so speculation keeps its ratio but not the rate.

### What the architectures allow

Per token, what stays O(1) in the context and what cannot:

| | O(1) in depth | O(n) by design | Its cost at 64K / 128K / 1M (computed) |
| --- | --- | --- | --- |
| DeepSeek V4 Flash | the window (128 positions), the CSA attention over the indexer's top 512 compressed rows, the compressors' rings, MoE, hyper-connections | the lightning indexer's scoring over every compressed row (64 heads × 128 dims, F16 keys, one row per 4 positions, 21 layers: 1,344 B a token of context), and HCA's dense attention over one row per 128 positions (20 layers: 160 B a token) | indexer keys read per decode step 88 MB / 176 MB / 1.41 GB, HCA 10 / 20 / 160 MB, against about 10 GB of weights a token (the decode step's ~48 ms at ~205 GB/s, dsv4-decode): +1% / +2% / +16% |
| Qwen3.8 Flash Next | attention over the 2,051 selected cells, Gated DeltaNet's recurrent state (3 MiB a layer), the n-gram rows, MoE | the QSA indexer's scoring over every block of 4 cells (4 query heads, one 128-dim key a block, 12 layers: 1,536 B a token as F32 pooled keys, 768 in BF16; today jitLLM reads every cell's raw F32 key, 6,144 B, and pools them each step) | 100 / 200 MB per step, 400 MB at its 262,144 maximum (F32 pooled keys), against about 7.5 GB a token: +1.3% / +2.7% / +5.3% |
| Qwen3.8's MTP drafter | the same as one QSA layer | its one QSA layer's scoring | 1/12 of the target's |
| Dense attention (the M2 Qwen2 fixture) | — | reading every cell's K and V each step | exactly linear; "flat" is impossible on the exact path, and the lever is a quality mode (KV compression), not measured here |

In prefill the indexers are compute: DeepSeek's scoring at 64K is 5.8
TFLOP per 2,048-row chunk (computed), which the WMMA kernel runs at about
6.6 TFLOP/s (887 ms); as a batched tensor-core GEMM it would be tens of
milliseconds, about 1% of a flat chunk, and about 16% at 1M. So "flat" in
practice: DeepSeek within about 2% to 128K and about 15% at 1M; Qwen3.8
within about 3% at 128K and 5% at its maximum (half that with BF16 block
keys). Mia's vLLM, 6% slower at 256K than at 8K, is on that line.

## Correctness at depth

With the resident harnesses (the runtime's fast plan and kernels) on
`spark`, [judge.py](judge.py) in the pinned PyTorch image. Greedy: jitLLM
teacher-forced on the oracle's 512 greedy tokens after the oracle's own
prompt IDs, its argmax at each step against the oracle's token, a
difference passing as a near-tie when the oracle's log-probability margin
between the two is under the bound. **The bound was recorded before each
model's first comparison:** the 99th percentile of the top-two margin's
move between jitLLM's default fast plan and its reference form
(`--exact`), forced on the same tokens at 32K (dsv4-decode's rule).
Perplexity: the second half of one window of the book, against
llama-perplexity (DeepSeek) and vLLM's prompt log-probabilities (Qwen3.8),
on the oracle's own token IDs.

| Check | DeepSeek V4 Flash vs llama.cpp b11254 | Qwen3.8 vs Mia's vLLM (deterministic) |
| --- | --- | --- |
| Near-tie bound (p99 fast vs exact, 512 steps at 32K) | 1.24 (p50 0.17, max 1.56; 498/512 argmax equal) | 1.47 (p50 0.19, max 2.96; 495/512) |
| Greedy at 32K | 500/512 equal, 11 near-ties (oracle margin ≤ 1.02), **1 outside: step 249**, margin 2.62 (the same in a second run; the exact form passes, 498 equal + 14 near-ties) | 474/512 equal, 38 near-ties (≤ 1.13), 0 outside: **pass** |
| Greedy at 128K | 508/512 equal, 4 near-ties (≤ 0.46), 0 outside: **pass** | 482/512 equal, 30 near-ties (≤ 1.00), 0 outside: **pass** |
| Perplexity at 32K (16,383 tokens scored) | 1.8516 against 1.8528 (−0.1%): **pass** | 1.4352 against 1.4656 (−2.1%): **pass** |
| Perplexity at 128K (65,535 scored) | not run (stopped past 64K) | 3.9467 against 4.0042 (−1.4%): **pass** |
| Retrieval, through the runtime | passes at 32K and 64K (the first prompts' notes, quoted in the answer); deeper not run | passes at 32K, 64K, 128K and 256K (`-r` prompts) |
| The same run twice, bit for bit | **no**: two runs of the 32K forced prompt differ from the first step (largest logit difference 6.13; the margin moves by p99 0.93 between them): RE-031's radix top-k in the lightning indexer | **yes** at 32K (jitLLM's own device selection, ties by cell); past 32,768 cells the GGML fallback is not repeatable (RE-031; not re-run here) |

Step 249 on DeepSeek is like dsv4-decode's step 93: the fast plan's
margin there moves further than the bound on one token, repeatably, where
the reference form agrees with the oracle; not diagnosed in this phase.
The DeepSeek bound itself includes RE-031's run-to-run noise (the fast
plan does not repeat), so it is looser than a repeatable engine's would
be. The comparators' own retrieval: llama.cpp quoted all three notes at
every depth (32K–256K) without DSpark, and at 64K–256K with it (at 32K
its 512 tokens ended first); Mia's vLLM quoted them at 32K–128K in its
deterministic launch and declined at 256K (the "passphrase" wording).

## Swap with a long saved context

At 64K, not the plan's 128K and maximum (runs past 64K stopped): `jitllm-runtime
swap-table` with both LLMs registered at `context = 65536`, plain, A
holding 61,440 tokens of the book, one first-use cycle each way, every
check of the M3 swap table on (A's restored state must hash as it left;
its 16 continued tokens and logits must equal the unswapped
continuation's). On `spark`, 2026-09-29:

| A (61,440 tokens) | A→B total | Spilled | B→A total | Restore (bytes read back) | Exact |
| --- | ---: | ---: | ---: | ---: | --- |
| DeepSeek → Qwen3.8 | 8.16 s (evict with spill 2.14) | 3.35 GB | 8.57 s | 0.29 s (the state within the 100.35 GB paged in) | yes |
| Qwen3.8 → DeepSeek | 8.68 s (evict with spill 1.47) | 2.13 GB | 7.54 s | 0.18 s (within 77.13 GB) | yes |

Every swap stays under the ~10 s goal at 64K, the continuation is exact,
and restore runs at the page-in rate (about 11.5 GB/s). Two findings: the
spill is the whole state region at the configured context, not the used
part (DeepSeek at 262,144 would spill 13.3 GB for any conversation: gap
5), and the pair peaked at 114 GiB with `MemAvailable` down to 2.97 GiB,
both models' states being mapped at once.

## Turn-to-turn reuse

A 3-turn coding session through each chat route (`s64k`: ~61K tokens of
files and a question, then two short follow-ups), greedy, 512 tokens a
turn, the client sending each answer back. Both models think by default,
and 512 tokens end inside the reasoning, so each reply's `content` is
short or empty. "Drop" sends back only the content (what clients do);
"keep" also sends `reasoning_content`. On `spark`, plain decoding.

| Engine, client | Turn 1: prompt, reused, prefill | Turn 2 | Turn 3 |
| --- | --- | --- | --- |
| jitLLM Qwen3.8, drop | 60,902, 0, 43.1 s | 60,949, **0**, 42.9 s | 61,113, **0**, 43.8 s |
| jitLLM Qwen3.8, keep | 60,902, 0, 43.4 s | 61,462, 61,413, **0.27 s** | 62,007, 61,973, **0.23 s** |
| jitLLM DeepSeek, drop | 61,115, 0, 247.6 s | 61,152, **0**, 246.7 s | 61,173, **0**, 246.9 s |
| llama.cpp DeepSeek, drop (prompt cache on) | 61,115, 0, 231.5 s | 61,152, 61,111, **0.59 s** | 61,173, 61,148, **0.49 s** |

jitLLM reuses a conversation only when the re-rendered prompt extends
everything the state holds. With the reasoning dropped the re-rendered
history leaves out the last turn's reasoning, which the state holds, so
jitLLM clears it and prefills the whole prompt again: about 4 minutes a
turn for DeepSeek at 64K. llama.cpp keeps the longest common prefix
(61,111 of 61,152 tokens) and prefills only the rest. When the client
sends the reasoning back, the rendered prompt extends the state and
jitLLM prefills only the new tokens.

## Memory and the guard's margin

Peak memory (drop in `MemAvailable`) against the comparators at the same
configured context:

| Model, context | jitLLM, GiB | Comparator, GiB | Ratio |
| --- | ---: | ---: | ---: |
| DeepSeek, 262,144, plain | 116.0 | 95.8 (llama.cpp) | **1.21×** |
| DeepSeek, DSpark | 110.8 at 65,536 (the guard refuses above 143,360) | 107.1 at 262,144 | ≥ 1.04× (at a quarter of the context) |
| Qwen3.8, 131,072 (jitLLM) / 262,144 (Mia's pool) | 100.6 | 102.4 | 0.98× |
| Qwen3.8, 262,144 | 101.2 | 102.4 (deterministic) / 100.7 (MTP 3) | 0.99–1.00× |
| Qwen3.8, 32,768 with MTP | 79.1 | 100.7 (MTP 3, its 262,144 pool) | 0.79× (not like for like) |

The guard (`kUncountedMargin`, 4 GiB) holds back room for what the
catalog does not count. Measured here, peak minus the budget the runtime
logged:

| Run | Budget, GiB (fixed) | Peak, GiB | Uncounted, GiB |
| --- | ---: | ---: | ---: |
| Qwen3.8, 32,768, MTP | 75.27 (3.87) | 79.06 | 3.8 |
| DeepSeek, 65,536, DSpark | 106.34 (5.90) | 110.84 | 4.5 |
| DeepSeek, 262,144 (two runs) | 110.66 (20.36) | 115.95–116.01 | 5.3–5.4 |
| Qwen3.8, 262,144, 2,040 rows | 94.64 (24.74) | 101.16 | 6.5 |
| Qwen3.8, 131,072, 4,096 rows (two runs) | 91.70 (21.80) | 99.92–100.65 | 8.2–9.0 |

So the uncounted memory is not a constant: besides the decode graphs and
the driver's and cuBLAS's own memory (4.6–5.1 GiB at 8K, fix B's
review), it grows with the host-side chunk inputs a model builds before
they are staged, which the catalog does not see: Qwen3.8's fallback path
builds F16 and F32 masks of [n_kv, rows] on the host (3 GiB a chunk at
131,072 × 4,096). DeepSeek at 262,144 left `MemAvailable` within 0.2 GiB
of zero at its peak.

**Recommendation:** raise `kUncountedMargin` to 6 GiB now (it covers every
run here except Qwen3.8's fallback at 4,096 rows), and count the host-side
chunk inputs in each model's fixed memory (they are bounded by the same
planning that sizes the staging), so the margin is again a constant.
Phase 2's device-side masks and sparse attention remove most of those
inputs. With a 6 GiB margin DeepSeek's plain maximum would fall from
262,144 to about 250,000 and its DSpark maximum to about 118,000
(computed from the guard table), until the window ring shrinks its state.

## Fixed to measure

Two blockers, each fixed minimally with a unit test, and two aids (the
Spark check set ran on the final tree):

- **RE-037** (Qwen3.8 at 147K–262K): GGML takes the strides of [context,
  rows] tensors as 32-bit ints (flash attention's mask; `ggml_permute`,
  fixed upstream in #29227 after the pin). `Qwen38State` and
  `Qwen38MostRows` now bound a chunk so an F32 [context, rows] tensor stays
  under 2^31 bytes: at 262,144 the runtime caps `prefill_chunk` at 2,040
  rows and says so. `qwen38_test` checks the edges.
- **RE-038** (DeepSeek past ~52K): `CheckConcat` applied the per-row
  concat kernel's 65,535-channel grid to GGML's contiguous kernel too,
  which has no such limit, so the CSA layers' concatenation of window cells
  and compressed rows was refused. The check now follows `concat_cuda`'s
  dispatch; `ggml_ext_validate_test` checks both kernels at 66,560
  channels.
- The executor's refusal now names the refused node and its first
  operand with shapes and strides, which found RE-037's second case.
- The resident harnesses (`jitllm_dsv4_exec`, `jitllm_qwen38_exec`)
  prefill a `--prompts` prompt longer than `--max-rows` in chunks, so the
  teacher-forced checks run at 32K and 128K (benchmark code only; covered
  by those runs).

## Gap list for phase 2

Ranked; the owner's target is per-token cost flat with depth, fixed at
64K first. Each fix is on the exact default path (the same results as
the comparator's default configuration, not an approximation); effort is
a rough estimate of agent days, including tests.

1. **DeepSeek's per-token cost grows with the whole context** (prefill 463
   → 230 tok/s and decode 22 → 10.8 tok/s from 8K to 64K, against
   llama.cpp's nearly flat 286 → 229 and 18.8 → 14.7 from 32K to 256K).
   Cause, from the profile: the full-size window cache, concatenated with
   the compressed rows into one K in every layer and attended under a
   mask. Fix plan:
   1. *The window cache as a ring* of window plus chunk rows (what
      llama-server runs; `swa_full` stays the exact mode's):
      removes the concatenation's O(n) copies (60% of the decode growth)
      and the dense window attention (most of the rest and 86% of the
      prefill growth); state 50,912 → 6,880 B a token, so 1M fits beside
      the weights and DSpark. Touches the state layout, the chunk inputs,
      the graph, DSpark's verify snapshots and rollback, spill and restore.
      **2–3 days.**
   2. *Attention over the gathered selection:* CSA layers attend the ring
      plus the indexer's top 512 rows gathered by index through flash
      attention's sparse path (#28770, #29298), with no concatenation of
      every compressed row and no n/4-wide mask; HCA layers read their
      n/128 rows in place. **1–2 days** (a port, or the pin bump).
   3. *The lightning indexer, exact and fused:* the model's shapes are 64
      heads × 128 dims, a Hadamard-rotated query, F16 keys (as llama.cpp
      caches them), one key per 4 positions in 21 layers, the top 512
      kept. One pass scores q·k over block-contiguous keys (coalesced
      reads), applies the heads' weighted ReLU sum and keeps a streaming
      top 512 with ties broken by index, which is deterministic and closes
      RE-031 for DeepSeek. Prefill scores as a batched tensor-core GEMM
      (5.8 TFLOP a 2,048-row chunk at 64K, run today at 6.6 TFLOP/s:
      887 ms); a DSpark verify's rows share one pass. **2–3 days.**
   4. Expected after 1–3: decode within about 1% of 8K at 64K and about
      2% at 128K, prefill within a few percent; the floor is the
      indexer's scoring and HCA's n/128 rows (+16% of the weights' bytes
      at 1M, computed above).
2. **Qwen3.8's per-token cost grows with the whole context, and MTP stops
   at 32K** (prefill 2,053 → 487 tok/s and decode 21.8 → 6.8 from 32K to
   256K, against Mia's 1,729–1,947 and 23.6–24.6). Fix plan:
   1. *Selection on the device at any depth:* a tiled, deterministic
      select over blocks past 8,192 (TensorFold #93's technique, ties by
      index) that emits the selected cells: removes the GGML fallback
      (cell-sized scores, ReLU, adds, transposes, radix top-k, host-built
      masks: RE-031's and RE-037's tensors, about 40% of the decode growth
      at 64K), and lets the MTP drafter speculate at any depth. **1–2
      days.**
   2. *Sparse flash attention over the 2,051 selected cells* by gather,
      not dense masked attention over every cell (#28770's technique;
      about 35% of the decode growth and half of prefill's). **1–2 days.**
   3. *Pooled block keys cached* when a block completes (the pooling,
      norm and rotation computed once, the same arithmetic), laid out per
      block for coalesced reads: removes the per-step gather of every raw
      indexer key (about 16% of the decode growth). **1 day.**
   4. Prefill scoring as a batched GEMM with the streaming top-k; a
      verify's rows share one pass. Expected: within about 1% at 64K, 3% at
      128K and 5% at the maximum (the floor: block scoring).
3. **The chat route's 600 s request deadline** (D-097's `kDeadlineMs`):
   at DeepSeek's 220–230 tok/s a 128K prompt needs 9–10 minutes (the
   route stopped it at 108,544 of 128,821 tokens), 256K about 19, 1M
   hours even after fix 1; a client that does not resend the request
   fails (a resend resumes, fix B). To evaluate in phase 2: a deadline
   that scales with the prompt (a floor prefill rate), or none for a
   stream that is sending keepalives and making progress. **Hours, and a
   D-097 amendment.**
4. **DeepSeek's memory and maximum:** 1.21× llama.cpp's peak at 262,144;
   DSpark only to 143,360; 1M impossible as built; the configuration
   refuses more than 262,144. Fix 1.1 is the lever; then the
   configurable ceiling (a configuration-semantics change: a decision).
5. **State reserved at the ceiling:** each registered model maps its
   state at its configured context for the node's life, a swap spills
   the whole region (12.4 GiB for DeepSeek at 262,144 whatever the
   conversation), and two long-context models do not register together
   under the guard. Lever: state extents grown with the conversation and
   spilled only as used (the state is already whole extents; the pinned
   places of D-090 must hold). **3+ days.**
6. **Turn reuse:** with a client that drops the reasoning (the common
   case), every turn at 64K re-prefills about 61K tokens: 43 s on Qwen3.8
   and 4 min on DeepSeek, against llama.cpp's 0.5–0.6 s
   ([above](#turn-to-turn-reuse)). Lever: reuse the longest common prefix,
   not only an extension. The attention caches, the indexer keys and the
   compressed rows are append-only, so truncating to a prefix is a
   position change. Qwen3.8's recurrent and convolution state, and
   DeepSeek's compressor rings, cannot be rolled back, so they need
   checkpoints where a later turn may diverge: at the end of each
   rendered user message (before the assistant's reasoning). **2–3 days.**
7. **The guard's margin:** 6 GiB and the host-side inputs counted
   ([above](#memory-and-the-guards-margin)). **Hours.**
8. **Repeatability (RE-031):** DeepSeek at 32K and Qwen3.8 past 32K;
   closed by 1.3 and 2.1.
9. **The prefill chunk at depth:** [n_kv, rows] tensors force narrower
   chunks at depth (Qwen3.8 2,040 rows at 262,144, RE-037); with
   gathered attention and device selection they go away and the chunk can
   stay wide.

**Future quality and performance modes** (never the default; each off,
behind a per-alias flag, and each needing a quality check against the
exact indexer): hierarchical selection (coarse super-blocks, then the
exact top-k within the chosen ones); reusing a step's selection for the
next few decode steps with a periodic full rescoring; approximate
nearest-neighbour search over the index keys; and, for dense-attention
models, KV compression.

## Not run, and why

- **Every jitLLM rung past 64K after 11:00** (the owner, 2026-09-29: we
  know enough at 64K; fix the scaling first). Qwen3.8's 128K and 256K and
  DeepSeek's 128K correctness had run by then; not run: DeepSeek at 128K,
  256K and its maximum through the runtime, DeepSeek's perplexity at 128K,
  the 1M comparator run (llama.cpp at `-c 1048576`), Qwen3.8's repeat at
  128K, and the swaps with 128K and maximum saved context (a 64K one
  instead, [above](#swap-with-a-long-saved-context)).
- **DSpark past 64K and MTP past 32K:** DSpark's maximum is 143,360
  under the guard; the MTP drafter is refused past 32,768 (gap 2.1).
- **TensorFold** (optional cross-quantization information): not run; its
  long-context changes (PR #93) are open upstream, and the comparison that
  matters is same-format.
- **llama.cpp b11254 at 8K:** the 8K reference is b10964's (M3
  baselines).
- **jitLLM's speculation acceptance at depth:** the chat route does not
  report it, and `jitllm-runtime chat` takes its prompt as an argument
  (128 KiB at most), too small for 32K.

## Reproduce

On a Spark, with this directory at `~/.local/share/jitllm/m3lc/lc/long-context`
and `docs/experiments/fast-swap` beside it, the M3 model store, the pinned
PyTorch image, and llama.cpp at `8019dc563` cloned to `~/src/lc/llama.cpp`:

```sh
W=~/.local/share/jitllm/m3lc
IMG=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
# The prompts (the tokenizer files as corpus.json pins them), and the book.
curl -sSLo $W/corpus/pg2600.txt https://www.gutenberg.org/cache/epub/2600/pg2600.txt
sudo docker run --rm --network none --user $(id -u):$(id -g) --entrypoint python3 \
  -v $W/lc/long-context:/tools:ro -v ~/src/lc/llama.cpp:/repo:ro \
  -v ~/.local/share/jitllm/models:/models:ro -v $W/corpus:/corpus -v $W/prompts:/out $IMG \
  -I /tools/build_prompts.py --repo /repo --model qwen3.8 \
  --tokenizer /models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json --out /out \
  --rung 8k=8704 --rung 32k=32768 --rung 64k=65536 --rung 128k=131072 --rung 256k=262144 \
  --session s64k=65536 --retrieval 32k-r=32768 ... --ppl-source /corpus/pg2600.txt
# (DeepSeek: --model deepseek with DeepSeek's tokenizer.json; add --rung 1m=1048576.)

# llama.cpp (the image built from 8019dc563, pins.json), and its perplexity.
python3 longctx.py llama $W/raw/ds-llama-plain $W/prompts/deepseek/{32k,64k,128k,256k}.json \
  --image jitllm-llamacpp:b11254-cuda13 --model .../DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --args "-ngl all -fa on -c 262144 -np 1 --fit off -cram 0"   # DSpark: add -md ... --spec-type draft-dspark ...
python3 longctx.py llama-ppl $W/raw/ds-llama-ppl --image ... --model ... --text $W/prompts/ppl.txt \
  --ctx 32768 --ctx 131072 --args "-ngl all -fa on --fit off"

# Mia's vLLM (as the M3 baselines set it up), the oracle's launch, then MTP 3.
python3 longctx.py vllm $W/raw/qw-mia-det $W/prompts/qwen3.8/{32k,64k,128k,256k}.json --port 8888 \
  --start "cd .../mia && MTP_NUM_SPECULATIVE_TOKENS=0 VLLM_QSA_DET_TOPK=1 VLLM_MOE_DET_FINALIZE=1 exec ./start.sh" \
  --stop "cd .../mia && ./stop.sh" --ppl $W/prompts/ppl.txt --ctx 32768 --ctx 131072

# jitLLM through the runtime (a configuration naming one model, its context).
python3 longctx.py jitllm $W/raw/qw-jit $W/prompts/qwen3.8/32k.json ... --port 18140 \
  --runtime build/spark-native/src/runtime/jitllm-runtime --config qwen-131072.toml --model qwen3.8 --retries 3

# Correctness: the oracle's IDs as harness input, the forced runs, the judge.
judge.py inputs raw/ds-llama-plain/deepseek-32k.json hin/ds d32k
jitllm_dsv4_exec --artifact ... --out raw/hd-32k-fast --context 33280 --max-rows 2048 \
  --prompts hin/ds/d32k.prompt.tsv --force hin/ds/d32k.force.tsv --generate 512   # and --exact on
judge.py noise raw/hd-32k-fast raw/hd-32k-exact d32k --vocab 129280
judge.py greedy raw/ds-llama-plain/deepseek-32k.json raw/hd-32k-fast d32k --vocab 129280 --bound B
jitllm_dsv4_exec ... --ppl raw/ds-llama-ppl/ppl-32768.ids; judge.py ppl 1.8528 raw/hd-ppl-32k/ppl.nll.f64 --ctx 32768

# The profile: one nsys run per depth, then the split.
nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o prof/ds-64k \
  jitllm_dsv4_exec ... --context 65536 --max-rows 2048 --prompts ds-64k.tsv --generate 3
python3 profile.py prof/ds-64k.sqlite 129280
```

Long runs went through `tools/spark-job` (`start --gpu --steps`), after
checking free memory and that no other model process was on the Spark.
Raw outputs (every request's record, server logs, traces) stay on the
Sparks under `~/.local/share/jitllm/m3lc/`.
