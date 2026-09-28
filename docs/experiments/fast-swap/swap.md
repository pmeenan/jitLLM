<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The swap path: the M3 models on the paged node, full swaps A→B→A (M3)

M3's swap path and swap runner ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)):
DeepSeek V4 Flash 0731, Qwen3.8 Flash Next and the Qwen-Image-2.1
pipeline run as device jobs over leased closures on the paged node
(D-086), paged into device VMM through the landing zone (D-081); a full
swap evicts the outgoing model, spilling its conversation state, and hands
its backing to the incoming one (D-033); the zone's copies have a lane of
their own (RE-029). The swap runners drive A→B→A in one process and time
each part: [M3's swap pairs](#m3s-swap-pairs) between the three models
(`jitllm_swap_pairs`), and [DeepSeek with the FP16 stand-in](#what-was-built-deepseek-and-the-fp16-stand-in)
(`jitllm_swap_runner`), where the path was first built.

## M3's swap pairs

### Qwen3.8 Flash Next on the paged node

`benchmarks/qwen38_runner.h`: the resident harness's graph, plan and
kernels (`qwen38_common.h`, one planning path that `qwen38_exec.cc` now
calls too; since the prefill slice, the fused graph and, from the
CUTLASS-layout artifact, CUTLASS's grouped GEMM and jitLLM's vector
products over the slots as they land) over catalog extents, in DeepSeek's
layout, now one helper (`paged_weights.h`):
- **Dense groups:** every group but the n-gram table's, a 2 MiB-aligned
  region each, a chunk an extent, landed from its shard.
- **Expert slabs:** each layer's 512 experts at the resident layout's
  stride (2,768,976 bytes), an extent a 2 MiB page of the slab landed in
  pieces (`LayOutSlab`). The gap between groups is 80 bytes, too small to
  hold DeepSeek's 256-byte alignment of the slab where a layer's experts
  change shard, so the slab's offset in its first page is a multiple of
  16, the stride's own alignment (`LayOutSlab` takes it as a parameter
  now; the resident harness's odd experts are 16-aligned too).
  35,873 extents, 75,235,266,560 bytes read per load (the table's
  28,800,138,240 not among them). The CUTLASS-layout artifact's groups
  are 2,764,800 bytes with no padding, so its stride is that and a load
  reads 75,002,167,296 bytes; nothing is rewritten after page-in.
- **The state** (`Qwen38StateLayout`: the QSA layers' K, V and indexer
  caches, the linear-attention layers' recurrent and convolution state,
  the n-gram layer's convolution history): kPreserve live state with a
  write-back place in an unnamed direct-I/O spill file, first in the
  closure, as DeepSeek's. 184 extents at the runs' 8,704-token context.
- **Setup after each full load:** the n-gram hash's constants read back
  and checked (`CheckQwen38PleHash`), as DeepSeek's hash-routing tables
  are.
- **A chunk:** its host-built inputs over the whole history (the n-gram
  hash reads each token's predecessors), its n-gram rows read (below),
  the graph planned for its shape, and one job that copies the inputs,
  gathers the rows, runs the bound plan and copies the last row's logits
  out; BP-A1's check on the first chunk of each shape.

### The n-gram table by rows (D-035)

`benchmarks/ple_rows.h`. The table (28.8 GB of 90-byte NVFP4 rows) is
never resident. Before each chunk's job the runner computes the chunk's
rows (16 a token, `Qwen38Chunk`'s hash), deduplicates them, reads them from
the artifact on a ring of its own (32 in flight) into a pinned landing,
and the job gathers each into a row slot on the device with a small kernel
(the GPU copies; no CPU payload copy). The graph is built over a copy of
the binding whose table has the slots' rows (512 × 16 = 8,192 slots,
737,280 bytes), and the chunk's row indices are the slots'.
- **Granularity: 4 KiB-aligned direct reads, smaller than a chunk.** A
  row's 90 bytes are covered by the one or two 4 KiB blocks around them;
  rows whose blocks touch or overlap share a read, up to 64 KiB. Every read
  lies within the table group's stored range, inside one chunk or across
  two consecutive chunks of the table's one shard (checked at setup). This
  is the only path that reads below a chunk; artifact-format.md keeps
  row-granular reads out of the runtime reader's scope and now records
  this runner's path and why the format needs no change
  ([page-in contract](../../artifact-format.md#page-in-contract)).
- **Why not whole chunks (D-035's default):** a token's 16 rows are
  hashed across the table, so almost every lookup lands in a different
  2 MiB chunk. Over the six chat prompts (192 chunks, 9,184 lookups, 8,880
  distinct rows a chunk summed), the row reads took 8,880 requests and
  37.4 MB; whole chunks would have read 18.2 GB (486×), and the useful
  bytes were 0.80 MB. An 8,192-token context touches nearly every chunk
  of the table.
- **Validity and accounting:** the rows belong to the chunk that read
  them. The landing (64 MiB, the bound for 8,192 lookups of two blocks)
  and the slots are fixed, cataloged and charged whole (staging and
  scratch); nothing carries from one chunk to the next, so no row has a
  residency to track, evict or restore across a swap. Every row index is
  checked against the table before a read is planned (whatever the hash
  gave), every read against the group's stored range, and the plan is
  refused, never split, past the landing's bytes or the slots. The rows
  are read to completion before the job that gathers them is posted: a
  short or failed read refuses the chunk after the rest drain, an unknown
  submission is in flight and waited for (the storage lane's rules), and
  reads that stall stop the rows for good, their ring and landing never
  reused or freed under them. The ring reads only the table's group, which
  is no extent, into the cataloged landing.

### Qwen-Image-2.1 on the paged node

`benchmarks/qwen_image_runner.h`: the three component artifacts, joined by
their composition (D-089), each a set of extents (`paged_weights.h`,
only the groups its phase reads: the text encoder's table and language
layers, 15.14 GB; the denoiser, 14.23 GB; the VAE's decoder, 1.35 GB of
F32), with the resident harness's kernels in its call order (copied from
`qwen_image_exec.cc`, whose comparison with diffusers then needs no rerun).
- **Phases lease only their component:** encode (one job over the text
  encoder's closure), denoise (a job per step over the denoiser's; the
  first also projects the text rows and fills the prefix K/V cache),
  decode (one job over the VAE's), each with the image's own memory, the
  shared workspace, the cuBLAS workspace and the staging. The VAE's F32
  weights are paged as stored and rounded to BF16 by the decode job into
  its workspace (the resident harness rounds them on the host, as
  diffusers' `torch_dtype=bfloat16` load does; same rounding).
- **The image's own memory** (16.7 MB, mapped at setup): what lives from
  one job to the next within a generation (the prompt embeddings, the text
  rows, the prefix cache, the rotary tables, the latents and the noise
  prediction). Nothing outlives a generation, so a swap has no image state
  to spill; every per-job buffer (3.02 GB at most, the decoder's) is the
  node's shared workspace.
- **Endpoint:** the prompt encoded and the first denoising step's output
  produced (M3's image endpoint). The rest of the generation and the
  decoder follow when the image is A.

### The swap pairs runner

`benchmarks/swap_pairs.cc` (`jitllm_swap_pairs`, a harness binary until
it moves into `jitllm-runtime`): two of the three models on one node, one process per
ordered pair, A→B→A as `jitllm_swap_runner` does it (see its header for
the protocol): a control, a first-use cycle (B never ran in the process;
A's plans dropped before it returns), a prepared cycle, and for an LLM A a
0-context pair. An LLM A holds 8,192 tokens of `docs/decisions.md` at
`4655685` (SHA-256 `6b159ff2…`), each model's own tokenization, context
8,704; an LLM B answers the first prompt of its correctness set from a
cleared state (DeepSeek: `capital`, 6 tokens, BOS first, no template;
Qwen3.8: `capital`, 64 tokens, the chat template rendered); the image is
the teapot prompt at 1,024², 40 steps, from diffusers' seed-42 latents.
The budget is the fixed memory plus the larger model's weights: the two
never fit together.
- **Swap correctness for an LLM A** is checked against the same state,
  not a rerun: after each cycle's prefill the state is saved to the host
  and hashed, the unswapped continuation run from it (the reference) and
  the state put back; after the swap back the restored state must hash
  the same (outside the timed parts) and the continuation's every logit
  equal the reference's. Qwen3.8 past 2,051 attended cells is not
  repeatable (RE-031: GGML's radix top-k picks among tied indexer scores
  nondeterministically), so a rerun of the prefill is no reference for it;
  each cycle's prefill is still compared with the control's, and noted.
  The substitution is sound one way only: the state digest is exact, and
  a continuation equal to the reference's shows the weights came back
  whole (wrong weights cannot give equal logits), but each continued step
  attends past 2,051 cells too, so RE-031 could make a continuation
  differ with nothing wrong in the swap. None did (below); a difference
  would need a rerun of the same state to tell the two apart.
- **An image A** has nothing to spill: its control is one full generation,
  and after the swap back the generation runs again from the prompt, its
  pixels equal to the control's.
- **Places (D-090):** every model pins its weights' and state's places
  when it registers them, graphs or not; after each swap the incoming
  model's are checked still pinned, DeepSeek's also against the sources
  its graphs name.

## Results: M3's swap pairs (`spark-b`, 2026-09-28)

GB10, kernel 7.0.0-1019-nvidia, driver 580.178.04, the `spark-native`
build, `CUDA_DISABLE_PTX_JIT=1`, the node as above (8 landing slots of
2 MiB + 8 KiB, four reads in flight, copy lane, handoff on). DeepSeek
`8a355bfb…` and Qwen3.8 `67617f87…` as imported on `spark-b`; the image's
composition `eca21baa…` and its three components copied from `spark` over
the direct link (10.100.208.x, rsync, 05:02–05:04, 84 s for 33 GB). File
ages at the final runs, from their change times (RE-027): DeepSeek's
shards 4.7–5.2 h, Qwen3.8's 3.2–3.8 h (both at rest), the image's 1.7–2.0 h
(written by the copy, and still reading at 14.2–14.8 GB/s, RE-027's
recent-write rate). Each pair started once `spark-b` had no GPU process,
more than 110 GB `MemAvailable` and a 1-minute load average under 6;
other agents' work shared the host between and during runs (load
averages up to 10 at a run's end). One process per ordered pair, one run
each (`pairs-final`, 06:25–07:02); raw outputs in
`~/scratch/m3pairs/` on `spark-b` (`pairs-final`, the earlier `pairs-try1`
with the same swap path, `q38-*`, `img-paged-1`, the probes).

**Correctness** (every check of the final runs passed; `exact` in every
row below):

| Check | Result |
| --- | --- |
| Paged Qwen3.8 against the resident harness (context 4,096, the six qwen38-native prompts, prefill + 31 greedy steps, same build) | 0 of 6 × 32 × 248,320 logits differ |
| Paged image against the resident harness (teapot, 1,024², 40 steps, seed 42's latents) | pixels `95fbcbc5…`, the resident harness's, in every generation: 2 controls, 4 after swaps, 1 alone |
| An LLM A's state after the swap back against the state it left with (SHA-256 of the whole region) | identical in all 12 returns at 8K context (DeepSeek 462,635,008 bytes, Qwen3.8 385,425,408) |
| An LLM A's 16 continued steps against the same state's unswapped continuation | every logit and token identical, all 12 returns |
| B's first output across its cycles and the 0-context pair (logits, or the image's first noise prediction) | identical, every pair |
| BP-A1: bound tensors in cataloged extents of their class | 0 outside (DeepSeek and Qwen3.8, every shape planned) |
| Qwen3.8's cycle prefill against the process's control (a rerun, not a swap check) | differs in 3 of 4 cycles from the 11th–14th chunk on (RE-031); DeepSeek's always equal |

**Swap times** (seconds, each part from the end of the one before, adding
up to the total from the swap request to the first output; LLM B: its
first token for its short prompt from a cleared state; LLM A: the next
token after its 8,192-token context, or at 0 context a 16-token prompt's;
image: the prompt encoded and the first denoising step's output). Page-in
counts the incoming weights and, returning to an LLM A, its state. Peak:
in use at the swap's lowest `MemAvailable` against the process's start.

| A ↔ B | Swap | Total | Evict and spill | Restore | Page-in (GB at GB/s) | First output (planning) | Peak GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 8.789 | 1.746 | — | 6.539 (75.24 at 11.5) | 0.501 (0.087) | 94.6 |
| | B→A, first use | **9.040** | 1.263 | 0.099 | 7.556 (97.46 at 12.7) | 0.119 (0.053) | 94.8 |
| | A→B, prepared | 7.673 | 1.701 | — | 5.599 (75.24 at 13.4) | 0.370 | 96.0 |
| | B→A, prepared | **8.763** | 1.287 | 0.096 | 7.312 (97.46 at 13.2) | 0.064 | 94.9 |
| | A→B, 0 context | 7.773 | 1.755 | — | 5.643 (75.24 at 13.3) | 0.373 | 94.9 |
| | B→A, 0 context | 8.849 | 1.329 | — | 7.296 (97.00 at 13.3) | 0.221 (0.054) | 94.4 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | 8.729 | 1.275 | — | 7.261 (97.00 at 13.4) | 0.189 (0.079) | 96.3 |
| | B→A, first use | 7.372 | 1.593 | 0.059 | 5.602 (75.62 at 13.4) | 0.115 (0.057) | 96.2 |
| | A→B, prepared | 8.635 | 1.266 | — | 7.262 (97.00 at 13.4) | 0.104 | 95.6 |
| | B→A, prepared | 7.423 | 1.635 | 0.059 | 5.677 (75.62 at 13.2) | 0.049 | 95.6 |
| | A→B, 0 context | **9.378** | 1.295 | — | 7.975 (97.00 at 12.2) | 0.105 | 96.2 |
| | B→A, 0 context | 7.495 | 1.598 | — | 5.603 (75.24 at 13.4) | 0.292 (0.086) | 96.2 |
| DeepSeek ↔ image | A→B, first use | 5.609 | 1.747 | — | 2.083 (30.72 at 14.7) | 1.777 | 98.1 |
| | B→A, first use | 7.938 | 0.532 | 0.092 | 7.195 (97.46 at 13.4) | 0.116 (0.053) | 98.3 |
| | A→B, prepared | 5.512 | 1.709 | — | 2.083 (30.72 at 14.7) | 1.718 | 98.4 |
| | B→A, prepared | 8.278 | 0.514 | 0.086 | 7.610 (97.46 at 12.7) | 0.063 | 100.1 |
| | A→B, 0 context | 5.457 | 1.665 | — | 2.083 (30.72 at 14.7) | 1.708 | 100.0 |
| | B→A, 0 context | 8.148 | 0.529 | — | 7.364 (97.00 at 13.2) | 0.251 (0.084) | 99.2 |
| image ↔ DeepSeek | A→B, first use | 9.094 | 0.536 | — | 8.217 (97.00 at 11.8) | 0.336 (0.096) | 103.9 |
| | B→A, first use | 6.280 | 2.530 | — | 2.132 (30.72 at 14.4) | 1.615 | 104.0 |
| | A→B, prepared | 8.350 | 0.540 | — | 7.699 (97.00 at 12.6) | 0.106 | 100.0 |
| | B→A, prepared | 5.620 | 1.656 | — | 2.255 (30.72 at 13.6) | 1.708 | 100.0 |
| Qwen3.8 ↔ image | A→B, first use | 5.060 | 1.310 | — | 2.078 (30.72 at 14.8) | 1.671 | 77.1 |
| | B→A, first use | 6.277 | 0.524 | 0.044 | 5.593 (75.62 at 13.4) | 0.115 (0.056) | 77.3 |
| | A→B, prepared | 5.025 | 1.303 | — | 2.090 (30.72 at 14.7) | 1.631 | 77.3 |
| | B→A, prepared | 6.234 | 0.538 | 0.044 | 5.600 (75.62 at 13.4) | 0.050 | 77.3 |
| | A→B, 0 context | 5.047 | 1.308 | — | 2.086 (30.72 at 14.7) | 1.652 | 77.3 |
| | B→A, 0 context | 6.425 | 0.546 | — | 5.641 (75.24 at 13.3) | 0.235 (0.082) | 77.2 |
| image ↔ Qwen3.8 | A→B, first use | 6.503 | 0.516 | — | 5.623 (75.24 at 13.4) | 0.363 (0.081) | 76.6 |
| | B→A, first use | 5.002 | 1.261 | — | 2.099 (30.72 at 14.6) | 1.641 | 76.6 |
| | A→B, prepared | 6.591 | 0.535 | — | 5.787 (75.24 at 13.0) | 0.266 | 79.4 |
| | B→A, prepared | 5.045 | 1.265 | — | 2.168 (30.72 at 14.2) | 1.610 | 79.5 |

Setup (DeepSeek's hash-routing check, Qwen3.8's n-gram hash; the image
has none) took 1–5 ms every time and is not shown. Qwen3.8's row reads
within its first output took at most 10 ms. Every swap handed off the
outgoing backing the incoming model could take (35,873 extents between
the LLMs, 14,719 with the image), and released the rest after the swap.
Bytes read per swap are the page-in column's; the n-gram rows added at
most 0.2 MB.

**Against M3's targets** (plan.md's swap table): every swap is under the
~10 s goal and so under the ~20 s bound, first use included (its own
target is ~40 s). **The worst LLM↔LLM swap is 9.38 s** (Qwen3.8 →
DeepSeek, prepared, 0 context); at 8K saved context the worst prepared
one is 8.76 s and the worst first-use one 9.04 s (both DeepSeek's
return). An earlier run of the same swap path (`pairs-try1`, before the
state-digest check was added) measured 7.24–9.66 s for the LLM↔LLM swaps,
its worst 9.66 s a Qwen3.8 → DeepSeek at 12.3 GB/s: the margin to 10 s is
the SSD's rate for DeepSeek's 97 GB. The swaps into an LLM are page-in
bound (75–97 GB at 11.5–13.4 GB/s is 5.6–8.2 s; the eviction before it
0.5 s of the image's 14,719 extents, 1.3–1.8 s of an LLM's 36,057–46,453;
the first token 0.05–0.5 s); into the image they take 5.0–6.3 s, 2.1 s of
it the 30.7 GB page-in and 1.6–1.8 s the encode and first step, whose step
runs at about 1.5 s against 0.90 s alone while the backing no load took is
released beside it (judgement calls). These runs had no CUDA graphs:
they predate the decode-graphs slice (D-090).

**After the rebase onto the decode graphs** (one run, `pairs-review1`,
07:29–07:32, DeepSeek → Qwen3.8 only, the same protocol): DeepSeek's
decode steps ran as graphs (4 captured, 72 replayed, none refused), every
model's places pinned at registration and found pinned after every swap
(DeepSeek's also at their registered sources). Every check exact: A's
state digest in both returns, every continued step, B's output, DeepSeek's
cycle prefills equal to the control's. Totals A→B 7.80, 7.84 and 7.66 s (0
context), B→A 8.65, 8.66 and 8.96 s (0 context), page-in 13.1–13.4 GB/s,
the shards 5.7–5.8 h (DeepSeek) and 4.2 h (Qwen3.8) old; the prepared
return's first token 0.059 s (a replayed graph) against 0.064 s before.
Peak memory 96.1–97.0 GiB in five swaps and 107.6 GiB in the first
(94.4–96.0 GiB before); not investigated (one sample, `spark-b` shared).

**Qwen3.8 from the CUTLASS-layout artifact** (the prefill slice's
review: `c4fb47a9…`, the fused graph with CUTLASS's grouped GEMM on the
paged node, no rewrite at load; `spark-b`, 08:51–08:57, one process per
ordered pair, the same protocol, raw outputs in
`~/scratch/m3qpre-review/`). The paged runner's logits equal the resident
harness's for the six prompts' 32 steps (0 of 6 × 32 × 248,320 differ, the
resident run from the same build and artifact). Every swap row exact: A's
state digest after every return, every continued step, B's output across
cycles.

| A ↔ B | Swap | Total | Evict | Page-in (GB at GB/s) | First output | Before (above) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DeepSeek ↔ Qwen3.8 | A→B, first use | 7.917 | 2.424 | 5.109 (75.00 at 14.7) | 0.380 | 8.789 |
| | B→A, first use | 8.886 | 1.211 | 7.458 (97.46 at 12.9) | 0.119 | 9.040 |
| | A→B, prepared | 7.031 | 1.654 | 5.083 (75.00 at 14.8) | 0.291 | 7.673 |
| | B→A, prepared | 8.638 | 1.262 | 7.216 (97.46 at 13.3) | 0.061 | 8.763 |
| Qwen3.8 ↔ DeepSeek | A→B, first use | 8.767 | 1.330 | 7.263 (97.00 at 13.4) | 0.171 | 8.729 |
| | B→A, first use | 6.863 | 1.679 | 5.038 (75.39 at 14.8) | 0.084 | 7.372 |
| | A→B, prepared | 8.698 | 1.335 | 7.254 (97.00 at 13.4) | 0.106 | 8.635 |
| | B→A, prepared | 6.879 | 1.704 | 5.055 (75.39 at 14.7) | 0.059 | 7.423 |

Swaps into Qwen3.8 took 6.9–7.9 s against 7.4–8.8 s before: nothing is
added at load (the prefill work's load-time rewrite took 3.9–4.0 s), and
its page-in ran at 14.7–14.8 GB/s against 11.5–13.4 (the new artifact's
shards were under an hour old, RE-027's recent-write rate, so part of
that is file age). The swaps out of it are unchanged.
Peak memory 94.2–100.3 GiB (100.3 in the first DeepSeek → Qwen3.8 swap,
95.1–95.5 in the other order), against 94.4–96.3 before; one sample.
Qwen3.8's cycle prefills against the process's control differ from the
11th chunk on, and so its continued steps against the control's (RE-031,
not a swap check).

Beside the baselines ([baselines.md](baselines.md), cold page cache, one
run each): llama.cpp's DeepSeek 0731 → Qwen3.8 (UD-IQ3_XXS) swap took
76.6 s and the return with 8K state restored 104.4 s; jitLLM's
DeepSeek → Qwen3.8 took 7.7–8.8 s and the return 8.8–9.0 s. Mia's vLLM
reaches Qwen3.8's first token 13 min 13 s from start and TensorFold
141 s; diffusers reaches Qwen-Image's first denoising step 212 s from
process start, jitLLM 5.0–6.3 s from the swap request.

## What was built (DeepSeek and the FP16 stand-in)

**DeepSeek on the paged node** (`benchmarks/dsv4_runner.h`). The resident
harness's graph, plan and kernels (`dsv4_common.h`, from `dsv4_exec.cc`),
over catalog extents:
- **Dense groups:** each group's 2 MiB chunks are extents at a 2 MiB-aligned
  region, landed from the artifact as FP16's are.
- **Expert slabs:** the resident expert layout keeps each layer's routed
  experts at a uniform stride S, the group's stored bytes rounded up to its
  blocks (8,064,224 bytes on 41 layers), so that GGML's stock `mul_mat_id`
  addresses expert e at slab + e·S. S is not a multiple of 2 MiB, so an
  artifact chunk cannot be an extent with its own 2 MiB backing. An extent
  is instead a 2 MiB **page** of the slab's address range. Its contents
  are the stored bytes of the at most two groups it overlaps, which are
  consecutive in the file: one 4 KiB-aligned read of up to 2 MiB + 8 KiB
  into a landing slot (slots are that size for this model), then up to two
  copies into the page (`PageSource::pieces`, new). The S − stored bytes
  between groups (3,296 on most layers) are never read by the kernels and
  are not written. Where a layer's groups change shard, the slab starts δ
  bytes into its first page (a multiple of 256) so that a page boundary
  falls in the gap between two groups, and no page needs two files
  (`LayOutSlab`, which refuses anything else). DeepSeek: 46,232 extents,
  97,001,283,584 bytes read per load (0.18% over the stored bytes, the
  pages' alignment).
- **The token table:** host VMM extents, read in place; embedding rows
  dequantized on the host, as llama.cpp's CPU backend does.
- **The state** (`Dsv4StateLayout`: the window cache, the compressed and
  indexer caches, the compressor rings; three D-068 representations):
  kPreserve live state, each 2 MiB extent with a write-back place in an
  unnamed direct-I/O spill file. Evicting it writes it back through the
  zone (D-081's reverse path); loading it restores it. Its extents come
  first in the closure, so a swap back restores the state before paging
  the weights in.
- **A chunk** is one device job on the model's stream, leasing the whole
  closure (weights, state, workspace, staging) until its fence. Plans are
  kept per chunk shape; the first chunk of each shape checks every bound
  tensor against the catalog (BP-A1). The hash-routing tables are checked
  after every full load.

**The handoff** (`scheduler.h`, D-033). An eviction asked for with a
handoff unmaps the backing but keeps it on the VMM lane, and parks: the
extent stays EVICTING and charged, so the catalog counts the kept backing,
and the evictor is told it is done. A page-in whose managed backing has the
same class and size takes a parked eviction's backing: in one step the load
begins, allowed that extent's bytes over B, and the parked eviction
completes, so the charge moves and occupancy never exceeds B. The VMM lane
maps the kept backing at the new place and sets access: no `cuMemCreate`,
no release. A parked extent materialized again takes its own backing back.
Backing no load took is released when the evicting task finishes, a few at
a time (16) so thousands of releases never take every mailbox; never an
idle pool. A load that took kept backing and ends without mapping it
(cancelled, or its map refused) releases it before the extent is
nonresident again.

**Page-in submission and RE-029** (`paged_node.h`, `scheduler.h`,
`cuda_device_execution.h`). A stream holds about 1,020 pending operations
and a launch into a full one blocks the launching thread (RE-029); a
DeepSeek chunk is 4,972 launches. The zone's copies, in and out, now go to
a **copy lane**: a second `DeviceService` with its own submission and
completion threads over the zone's stream alone. And since `cuEventCreate`
turned out to block too while any thread is blocked in such a launch, the
CUDA provider takes fences' events from a **pool** made when it opens.
Together, a job that blocks the device lane's thread no longer holds up a
single copy.

**The full swap** (`tests/support/paged_programs.h` `SwapProgram`): evicts
the outgoing extents, state first, 256 at a time, with or without the
handoff, then materializes the incoming closure, and notes when each ended.

**The swap runner** (`benchmarks/swap_runner.cc`, `jitllm_swap_runner`): a
harness binary, built on the test harness's paged node (the native
tokenizer it links is cleared for production binaries by D-088). See its
header for the
protocol. In short:
- **A** is DeepSeek V4 Flash (artifact `8a355bfb…`), context 8,704. Its
  8,192-token context is the first 8,192 tokens, BOS first, of
  `docs/decisions.md` at `4655685` (SHA-256 `6b159ff2…`), tokenized by the
  native tokenizer from the artifact's GGUF metadata. **B** is the
  Qwen2.5-0.5B FP16 fixture (`b93cdc32…`) running the backend proof's
  `control` trajectory (32-token prompt, then 44 teacher-forced steps),
  whose logits must hash to rung 3's `bb8ae5e7…`.
- **Control:** A prefills the context (chunks of 512) and decodes 16 tokens
  greedily, never swapped. **Each cycle:** A prefills the context again
  (its logits must equal the control's), swaps to B (A's state written
  back and A's weights evicted; B's weights paged in), B runs its prompt,
  swaps back (B evicted; A's state restored, A's weights paged in), and A
  continues: each of the 16 continued steps' logits must equal the
  control's bit for bit. The first cycle is **first use** (B never ran in
  the process, so its cuBLAS handle is made in its first token; A's plans
  are dropped before it returns); the second is **previously prepared**. A
  last pair runs at **0 context**: A's state is not spilled (it stays
  resident) and A's first token is a 16-token prompt's.
- **Reloads:** DeepSeek → evict everything (state and weights) → DeepSeek
  again, twice: the heaviest page-in.
- **Overlap (RE-029):** B's weights paged in while one of A's 512-row
  prefill chunks (4,972 launches) runs on A's stream, against B paged in
  alone.
- **Parts of a swap,** from the swap request: *evict* until every outgoing
  extent is evicted or parked (A's state written back included);
  *restore* until A's state is resident again; *page-in* until the
  incoming weights are; *setup* (A: the hash-routing check); the *first
  token* (B: its cache cleared, its cuBLAS handle on first use, its
  32-token prompt; A: one decode step, planned on first use). Backing no
  load took is released after the swap, off the critical path: *released
  by* is when the last of it was (observed after the first token).

## Results with the FP16 stand-in (`spark-b`, 2026-09-28)

GB10, kernel 7.0.0-1019-nvidia, driver 580.178.04, the `spark-native`
build, `CUDA_DISABLE_PTX_JIT=1`, lanes on their own threads, 8 landing
slots of 2 MiB + 8 KiB, four reads in flight, no coalescing. The DeepSeek
artifact's shards were written at 01:43–01:47 and read from 04:14: about
2.5 hours old, at rest (RE-027; 13.3–13.4 GB/s here). No other GPU
process ran (checked before each run; `spark-b` is shared, see the
handoff table's note on CPU load). Raw outputs in
`~/.local/share/jitllm/m3swap-20260928/` on `spark-b` (`swap-5`, handoff
on; `swap-6-nohandoff`; `swap-7-nocopylane`; `prompts-2`, and
`prompts-1` on an earlier build, the same). A re-run after review's
changes (the handoff kept within a domain, 1,042 events made ahead;
`review-1`, one run with one reload) was exact throughout and within 3%
of `swap-5` on every total: B→A 7.42 and 7.38 s, reload 9.11 s, the
overlap probe 0.135 s alone and 0.194 s beside the chunk.

**Correctness.**

| Check | Result |
| --- | --- |
| Paged DeepSeek against the resident harness (context 4,096, the 8 dsv4-native prompts, prefill + 31 greedy steps) | 0 of 8 × 32 × 129,280 logits differ; the same 256 tokens |
| A resumed after B against A never swapped (8,192 context tokens, 16 continued steps) | every step's logits bit-identical, and the tokens, in both returns (first use, prepared) of every run: 4 runs, handoff on and off |
| A's prefill in each cycle against the control's | bit-identical |
| B after each swap | logits SHA-256 `bb8ae5e7…`, rung 3's, every time |
| A's bound tensors in cataloged extents of their class (BP-A1, once per shape) | 775,732 checked, 0 outside |

**Swap times** (seconds; handoff on; each part from the end of the one
before, so they add up to the total, the swap request to the first
token). A's context is 8,192 tokens unless noted; its state is 462,635,008
bytes (221 extents, 463,470,592 bytes written back and read again).

| Swap | Total | Evict and spill | Restore | Page-in (GB at GB/s) | Setup | First token | Handed off | Released by |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A→B, first use | 1.857 | 1.689 | — | 0.107 (1.26 at 11.8) | — | 0.061 | 620 | 1.55 |
| B→A, first use | **7.439** | 0.027 | 0.087 | 7.179 (97.46 at 13.4) | 0.003 | 0.142 (0.079 planning) | 620 | 1.11 |
| A→B, prepared | 1.790 | 1.655 | — | 0.110 (1.26 at 11.4) | — | 0.025 | 620 | 1.54 |
| B→A, prepared | **7.362** | 0.027 | 0.086 | 7.182 (97.46 at 13.4) | 0.003 | 0.064 | 620 | 0.98 |
| A→B, 0 context | 1.828 | 1.689 | — | 0.114 (1.26 at 11.0) | — | 0.026 | 620 | 1.57 |
| B→A, 0 context | 7.522 | 0.030 | — | 7.244 (97.00 at 13.4) | 0.003 | 0.246 (0.080 planning) | 620 | 0.25 |
| DeepSeek reload 1 (evict all, reload) | 9.071 | 1.672 | 0.091 | 7.193 (97.46 at 13.4) | 0.003 | 0.112 | 46,453 | 0.12 |
| DeepSeek reload 2 | 9.074 | 1.725 | 0.091 | 7.199 (97.46 at 13.4) | 0.003 | 0.056 | 46,453 | 0.06 |

Beside M3's targets (D-087), from the swap request to the first token:
every swap here is under the ~10 s goal (worst 7.52 s among the A↔B swaps,
9.07 s for the DeepSeek reload), and first use is within 0.08 s of
prepared. This is not M3's exit measurement: B is a 0.5B stand-in, so
A→B pages in 1.26 GB where Qwen3.8 pages in 75 GB; M3's pairs are
[above](#results-m3s-swap-pairs-spark-b-2026-09-28). The baselines' numbers stand beside them:
the pinned llama.cpp switched between DeepSeek V4 and Qwen3.8 in 75–93 s
to the first token (measured, one run, M0), TensorFold loads Qwen3.8 in
about 90 s and Mia's vLLM in 11–14 min (both creator-reported). B→A is
page-in bound: 97.46 GB at the SSD's at-rest rate is 7.2 s, and everything
else in the swap took 0.18–0.28 s. A's first load in each run, cold, took
7.26 s (13.37 GB/s). A's prefill of 8,192 tokens took 27.3 s
(300 tokens/s) and 16 decode steps 1.10 s (14.5 tokens/s at 8K context),
on the paged node with no CUDA graphs (decode graphs since: [graphs](graphs.md)). Peak memory by `MemAvailable`:
96.4 GiB over the whole run (A and B both resident at its end).

**The handoff's effect** (the same runs with `--handoff off`: evictions
unmap and release, loads create; seconds):

| Swap | Handoff on (`swap-5`) | Off (`swap-3`; `swap-6`) | What changes |
| --- | ---: | ---: | --- |
| A→B, 0 context: evict | 1.689 | 2.975; 2.995 | 46,232 unmaps kept, not released: 36.5 against 64.8 µs each |
| DeepSeek reload: evict | 1.672, 1.725 | 3.094, 3.020; 3.076, 3.019 | the same, plus the state's write-back |
| DeepSeek reload: total | 9.071, 9.074 | 10.510, 10.361; 10.472, 10.361 | −1.3 to −1.4 s |
| A→B, 8K context: evict | 1.689, 1.655 | 3.052, 3.047; — | |
| B→A: page-in | 7.179–7.244 | 7.189–7.245; 7.200–7.256 | none: page-in is read-bound, and the VMM lane's creates (46k, about 110 µs each) run ahead of the reads |

So the handoff removes the release from every eviction and the create
from every load that takes kept backing; on this path the release is what
shows, 1.3–1.4 s of a 46k-extent eviction, because the creates already
hide behind the reads. Two runs with the handoff off: `swap-3`, before
the event pool below (with the handoff off, an eviction fences only its
221 write-back copies), and `swap-6`, on the measured build but not
clean: another workload shared the host during it (load average 9.2 over
those minutes, and its peak memory in use 110.7 GiB against 95–96 GiB in
every other run; `spark-b` is shared with the Qwen3.8 and image slices).
`swap-6`'s two 8K-context A→B evictions (12.2 and 21.7 s) and its control
prefill (31.9 s against 27.3 s) are therefore not used; its other rows
agree with `swap-3`'s within 0.04 s.

**Page-in beside a busy stream (RE-029).** B's 1.26 GB paged in while one
of A's 512-row prefill chunks (4,972 launches, 1.56–1.72 s) runs on A's
stream:

| | B alone | B beside A's chunk |
| --- | ---: | ---: |
| Copy lane and event pool (`swap-5`) | 0.118 s | 0.188 s |
| Zone copies on the device lane (`swap-7`) | 0.111 s | 1.661 s |
| Copy lane, events made per fence (`swap-2`, before the pool) | 0.119 s | 0.569 s |

On the device lane the page-in waits for nearly the whole chunk: its
copies queue behind the job on the one submission thread, which is blocked
launching into the full stream. The copy lane alone was not enough:
`cuEventCreate`, which fenced each copy, blocks while any thread is
blocked launching into a full stream (RE-029's update; a gated stream held
it for 30 s in the unit test). With fences from a pool made ahead, the
page-in beside the chunk takes 0.07 s more than alone.

## Tests

- `unit.VmmWork/PageInTest.*` (fake providers, VMM work on its own lane
  and on the device lane): a handoff moves evicted backing to the loads
  that follow with no backing created and occupancy never over B; parked
  backing stays charged while its evictor runs and is released when it
  finishes; a parked extent takes its own backing back; live state written
  back and parked comes back whole; a load that took kept backing and is
  cancelled, or whose map is refused, releases it; a landed read lands in
  pieces, which are refused for write-back and beyond the read.
  `unit.VmmWork/CopyLaneTest.*`: page-ins land through the copy lane while
  the device lane never turns, and a job still runs on the device lane.
- `unit.CudaPagedNodeTest.*` (GB10): a full swap with the handoff each way
  between two synthetic models on the real VMM provider, every byte read
  back, the counts exact, no backing left; RE-029's case, a page-in beside
  a job whose stream is full.
- `unit.CudaExecutionPoolTest.FencesReuseEventsMadeAhead` (scripted driver):
  fencing makes and destroys no event once the pool is made.
- `unit.SlabLayoutTest.*`: every stored byte of every group lands once, at
  its place, from its place in one file; a shard change falls on a page
  boundary in a gap; the layouts the runner cannot page are refused; at
  Qwen3.8's 80-byte gap, a shard change the default 256-byte alignment
  cannot place lays out at 16, and alignments that are not a power of two
  from 16 to 4,096 are refused.
- `unit.PleRowsTest.*` (host): each lookup's slot holds its row's bytes
  once the planned reads land, duplicates and rows crossing a block or a
  chunk included; reads are 4 KiB-aligned, packed in the landing,
  ascending, inside the table's stored range and at most 64 KiB, and rows
  whose blocks touch share one; the whole-chunk count is the chunks the
  rows touch; rows outside the table, a landing too small, too many
  distinct rows and a table past its range are refused; on the storage
  fake, an unknown submission is waited for and its row used, and a short
  or failed read refuses the chunk's rows only after every read has
  drained.
  `unit.CudaPleRowsTest.*` (GB10): the reads through io_uring from a real
  file and the gather kernel put every row in its slot.
- Not unit-tested, checked by the runs above instead: the paged Qwen3.8
  and image runners and the pairs runner (the bit-identity and pixel
  checks against the resident harnesses, the state digests and
  continuations), and `PagedWeights`, whose layout is DeepSeek's.

## Judgement calls

- **Pages, not chunks, for the expert slabs.** A chunk-sized extent would
  straddle two 2 MiB backing pages at the slab's uniform stride, which
  needs shared backing refcounted across extents; a stride that is a
  multiple of 2 MiB and of the IQ2_XS and IQ3_XXS blocks would be 74 MiB
  per expert. A page read of up to 2 MiB + 8 KiB into a slightly larger
  slot, copied in up to two pieces, keeps D-033's one handle per extent,
  and the handoff, with 0.18% more bytes read.
- **The copy lane, not bounded streams,** for RE-029, plus the provider's
  event pool once the probe showed `cuEventCreate` blocking. Bounding each
  stream's queue would split every DeepSeek phase (4,972 launches) into
  jobs of under 1,000. By default the pool makes 256 events when the
  provider opens and keeps up to 4,096; released fences return theirs to
  it. Past what it holds, a fence makes its event as it goes (and may
  block as above). The paged node makes 1,042, every fence its device and
  copy lanes can hold at once, so neither lane ever does; the timed runs
  above made 256.
- **Parked backing is released when its evictor finishes,** and only its
  releases are paced (16 at a time): the first run without pacing
  released 45,833 handles at once, took every mailbox, and B's first job
  was refused.
- **A full swap evicts the outgoing model whole,** though B (1.3 GB) would
  fit beside A: the plan's M3 swap is between models that do not fit
  together (DeepSeek and Qwen3.8), and the stand-in B must not hide that.
  A→B's times are therefore mostly A's eviction; B→A's are the
  representative page-in.
- **The resident harness stays as it was;** the paged runner's graph and
  input code is a copy of it (`dsv4_common.cc`), so `dsv4_exec.cc`, the
  validated comparison with llama.cpp, needs no rerun. The prompts check
  above shows the two equal bit for bit.
- **0 context keeps A's state resident** rather than spilling a cleared
  one: a real 0-context model has nothing to save.
- **The n-gram table by 4 KiB-aligned row reads, per chunk, no cache**
  (D-035 asks for evidence before a smaller-read path; above): whole
  chunks would read 486× the bytes on the correctness prompts and nearly
  the whole table at 8K context. A row cache across chunks would save the
  repeated rows of nearby tokens but needs a residency contract; the reads
  cost 0.3 ms a chunk on those prompts (0.056 s over 192 chunks). They run
  on the runner's own ring on the caller's thread, not the scheduler's
  storage lane, whose sources are whole extents.
- **Qwen3.8's slab offset aligned to 16, not 256** (the resident layout's
  odd experts are 16-aligned too); the logits equal the resident
  harness's.
- **The weights' unwritten bytes are left as the backing had them.** A
  page-in writes neither a slab page's bytes between groups nor a dense
  chunk's tail past its stored length, so after a handoff they hold the
  outgoing model's bytes (weights, or spilled state: an LLM's KV), where
  the resident harness has zeros. No kernel reads them, from the code:
  every tensor a plan binds is a resource or an expert slice, and a
  quantized product reads past a row only into the slice's readable
  bytes (`CheckMulMatQ` refuses short rows otherwise, and the Qwen3.8
  graph marks them readable only inside the stride); the reader refuses
  an artifact whose readable range leaves its group's stored bytes
  (`CheckPlacement`: offset + readable ≤ stored, the last ending at
  `used_bytes`), and the artifact writes that padding as zeros inside the
  group; each page-in writes a group's whole stored range (a dense chunk
  its stored length, a slab page its groups' stored pieces). So what a
  kernel reads, the page-in wrote from the file, whatever the backing
  held. Probed too, when a first rerun differed (`--scrub-probe`, 4,096
  tokens): filling them with 0xFF (NaN in every float format, and NVFP4's
  scales, which the padding's zero activations would turn into NaN)
  changed none of Qwen3.8's logits, nor did filling the shared workspace
  so before each chunk (`--poison-probe`, Qwen3.8 and DeepSeek). The
  difference was RE-031. Nothing zeroes them: one process serves one
  user (D-019), and nothing reads or exports those bytes; a multi-tenant
  runtime would zero handed-off backing's unwritten bytes (D-014).
- **BP-A1's check skips a fill's source:** QSA's selection mask fills a
  shape-only tensor that is never bound (and never read), which the
  check first counted as 312 tensors outside the catalog.
- **One process per ordered pair,** so that every pair has a genuine first
  use of B; the image A's return has no plans to drop (its "first use"
  labels B only).
- **Copies, not shared code,** of the resident harnesses' planning
  (Qwen3.8) and phases (the image), as DeepSeek's: the validated resident
  comparisons need no rerun, and equality is checked instead.
- **The swaps' check for an LLM A compares against the same state**
  (above), since Qwen3.8 is not repeatable past 2,051 cells (RE-031); the
  fix belongs to the Qwen3.8 work, not the swap path.
- **Backing no load took is released while B makes its first output,**
  not before: with the image as B, waiting for the release first
  (`--release-first on`, DeepSeek → image, `try3rf`) took 1.00–1.01 s and
  the first output then 0.98–1.07 s, 5.73–5.83 s in all, against
  1.67–1.79 s overlapped and 5.40–5.56 s in all (the release's ~31,000
  `cuMemRelease` calls slow the first denoising step from 0.90 s alone to
  about 1.5–1.6 s, but less than they take).

## Reproduction

On `spark-b`, with the `spark-native` build, the artifacts installed as
dsv4-native's and the backend proof's are, and P2's `control-tokens.txt`:

    jitllm_swap_runner --dsv4-artifact DSV4 --fp16-artifact FP16 \
      --tokens control-tokens.txt --fp16-expect bb8ae5e7e3ac6da7… \
      --text decisions.md --out DIR --reload 2 --overlap \
      [--handoff off] [--copy-lane off --cycles 0 --overlap]
    jitllm_swap_runner ... --cycles 0 --context 4096 \
      --prompts dsv4-native/oracle/unfused/prompts.tokens \
      --expect dsv4-native/jit-free --generate 32

Each run takes 1–4 minutes and needs about 97 GiB free; it exits 1 on any
failed check, and `swap.json` holds every number above. Every run used
above started only once `spark-b` had no GPU process and more than 110 GB
`MemAvailable` (a wrapper that waits for both); the runner itself does not
check, and a first attempt started beside another 99 GB process was
killed by the kernel's OOM killer (no number here comes from it).

M3's pairs, each ordered pair one process (3–4 minutes each), with the
image's component artifacts and the reference's initial latents
installed as qwen-image-native's are (copied to `spark-b` for these runs):

    jitllm_swap_pairs --a dsv4|qwen38|image --b dsv4|qwen38|image --out DIR \
      --dsv4-artifact DSV4 --qwen38-artifact QWEN38 --image-store STORE \
      --image-composition eca21baa… --image-noise ref1/latents_init.bf16 \
      --text decisions.md --qwen38-tokenizer tokenizer.json \
      --dsv4-prompt dsv4-native/oracle/unfused/prompts.tokens \
      --qwen38-prompt qwen38-native/prompts.tsv [--image-expect 95fbcbc5…]
    jitllm_swap_pairs --a qwen38 --b dsv4 ... --cycles 0 --context 4096 \
      --prompts qwen38-native/prompts.tsv --expect RESIDENT --generate 32
    jitllm_swap_pairs --a image --b qwen38 ... --cycles 0 --image-expect 95fbcbc5…

where RESIDENT is `jitllm_qwen38_exec --context 4096 --max-rows 512
--prompts qwen38-native/prompts.tsv --generate 32`'s output from the same
build. The runs used a wrapper that also waits for the 1-minute load
average to fall below 6, and checks twice 20–40 s apart: another agent's
run started beside one of these in the same second once (its logs are
not used).

## Limits

- One process, one run per configuration; timings are single samples on
  `spark-b` (D-085's coarse comparison), not distributions.
- In the FP16 stand-in's runs, B's own page-in is 1.26 GB; M3's pairs
  are measured with `jitllm_swap_pairs` above.
- The pairs' table has no CUDA graphs: "prepared" means A's plans exist
  and B ran before in the process. Since the rebase DeepSeek's decode
  steps run as graphs (`--graphs on`, the default) and one pair was rerun
  with them (above); Qwen3.8 and the image run launch by launch, their
  places pinned all the same.
- The n-gram rows are read on the caller's thread before each chunk's
  job, synchronously: a decode step waits for its 16 reads (0.3 ms). No
  row cache, no overlap with the previous chunk's job.
- The image's phases run their kernels directly, not as registry-bound
  plans (D-053), and are not checked by BP-A1's coverage check; the image
  runner's memory is cataloged but its tensors are not.
- The pairs' pinned state snapshot (the harness's check, up to 0.46 GB)
  is outside the catalog and inside the peak memory figures.
- The spill file is unnamed (`O_TMPFILE`, mode 0600, gone when the
  process exits) and a restore must read every byte back, but nothing
  checks the bytes it reads: silent corruption on the SSD would come back
  as state (D-055's named spill format stays in M6).
- The FP16 fixture's file age at the runs was not recorded (RE-027); its
  page-in is too small for the at-rest rate to show.
- These runs had no CUDA graphs: "prepared" meant A's plans and B's
  cuBLAS handle exist. With decode graphs (D-090, [graphs](graphs.md)) a
  prepared return's first token replays a graph captured before the swap:
  0.060 s against 0.064 s here, within noise, so the table above stands.
- The overlap probe's page-in beside a busy chunk is still 0.07 s slower
  than alone (the GPU and memory are shared); not investigated. Other
  driver calls than those checked (record, query, copy, synchronize on an
  idle stream) may also block behind a full stream's launch.
- Unmapping the outgoing extents one at a time dominates A→B (about 35 µs
  each over 46,453 extents); one unmap per contiguous range is not yet
  measured.
