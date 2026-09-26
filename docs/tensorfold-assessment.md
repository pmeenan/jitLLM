<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorFold assessment

The owner asked, on 2026-09-26, that [TensorFold](https://github.com/ashhart/TensorFold)
be considered as a benchmark target and as a source of optimization ideas.
This is a source review of commit
[`d7470ed`](https://github.com/ashhart/TensorFold/tree/d7470ed8f6365ad3c7c7268f3c32da548eb1c343)
(version 0.3.1). TensorFold itself has not been run. Every number about
TensorFold below is creator-reported and unverified; only the `/proc`
observations in item 6 are ours. Measuring TensorFold on our Sparks is a
separate step ([below](#proposed-use)).

## What it is

- **Engine.** A single-stream local inference engine with an
  OpenAI-compatible `/v1/chat/completions` endpoint, for Apple Silicon
  (MLX) and NVIDIA (CUDA).
- **Maturity.** The repository was created on 2026-09-25, with 14 commits
  and "Development Status :: 3 - Alpha". The CUDA engines for the DGX Spark
  landed on 2026-09-26.
- **Runtime.** On CUDA, each supported model family has its own PyTorch
  forward pass with Triton and CUDA kernels written for that family
  (`src/tensorfold/families/<name>/cuda/`). It runs inside NVIDIA's PyTorch
  container; kernel `.cu` sources build on first use with the container's
  NVCC.
- **Models.** On CUDA: Qwen3.8-27B (dense), Qwen3.8 Flash Next, and
  GLM-5.3-Flash (two Sparks only). Nemotron 3.5 Lightning appears in its
  Mac results. Families are recipes chosen from the checkpoint's
  `config.json`, and a checkpoint without a recipe is refused.
  - Qwen3.8 Flash Next is already one of our candidate models
    ([features](features.md), MoE reference candidates).
- **Formats.** MLX 4-bit checkpoints from Hugging Face only. It refuses
  EXL3, NVFP4, GPTQ and AWQ.
  - The CUDA path reads the *format*, not the MLX library: `mlx` and
    `mlx-lm` are macOS-only dependencies, and no CUDA module imports them.
  - The format is 4-bit affine group quantization: groups of 64, eight
    values packed per 32-bit word, and a bf16 scale and bias per group
    (`families/qwen3_5/cuda/qmm.py`). That is close to GGUF `Q4_1` or
    asymmetric GPTQ.
  - Its Triton matmul computes
    `Σ_g (scale·(x·q) + bias·Σx)`, over the groups in order, after
    regrouping the packed words at load.
- **Hardware.** Per its README, tested on one and two DGX Sparks (GB10),
  with tensor parallelism over NCCL across the direct link. That is our
  exact target.

## License

- **Code.** TensorFold's own code is MIT, which is core-eligible under
  D-017. It adapts code from mlx-lm (MIT), transformers (Apache-2.0) and
  z-lab's DFlash (MIT, vendored unmodified). PyTorch (BSD-3-Clause) and
  Triton (MIT) come from NVIDIA's container, not bundled.
- **Model weights.** It ships none. The Qwen3.8-27B DFlash2 drafter is
  Apache-2.0 per its model card. GLM-5.3-Flash's optional drafter is
  CC BY-NC-ND 4.0 (non-commercial, no derivatives), so it must not enter
  any jitLLM artifact or default. As a benchmark-only component it needs
  the owner's decision.
- **Architecture.** Its runtime is Python and PyTorch. jitLLM's hot path
  stays native (D-010), so reuse would mean porting ideas or individual
  kernels under their licenses (D-013), never adopting its runtime.

## Its claims on DGX Spark

The README and `docs/recipes/cuda.md` compare it with vLLM using MTP=3,
through the same OpenAI client:
- one stream, 64-token replies;
- the median of seeds 1234–1238;
- code and chat prompts, sampled (T 1, top-k 20, top-p 0.95) and greedy.

"Each TensorFold number is byte-identical to its own serial decoding", and
the recipe notes that vLLM's drafted output is not.

| Model | Sparks | vs vLLM with MTP (range over the four columns) |
| --- | --- | --- |
| Qwen3.8-27B + DFlash2 | 1 | 2.70–3.05× (for example 49.6 vs 17.7 tok/s, code sampled) |
| Qwen3.8-27B + DFlash2 | 2 | 1.94–2.49× |
| Qwen3.8 Flash Next | 1 | 1.60–1.79× |
| Qwen3.8 Flash Next | 2 | 1.74–2.24× |
| GLM-5.3-Flash | 2 | 1.78–2.06× |

These are decode rates for short replies to short prompts: the benchmark
prompts were 14 and 31 tokens, per its own notes. They say nothing about:
- prefill;
- long contexts;
- model switching;
- concurrency.

The comparison also mixes engines *and* drafters: DFlash2 draft trees
against vLLM's MTP.

## Ideas worth taking

Mapped to where they would land in jitLLM:

1. **Exact speculative decoding by row-invariant kernels.**
   - By its design notes, every kernel on the verify path gives a row the
     same bits whether it runs alone or in a window of *n* rows. Split-K depends only on the
     weight's shape; slices are added in order; attention is chunked by the
     row's own key range; there is no reduction across rows.
   - Drafts are then accepted exactly when serial decoding would produce
     them.
   - This is stronger than our Tier E exactness and rung-5 restore checks,
     which compare the same plan and trajectory across paging. They do
     not establish equality between serial and batched execution. For
     D-068 (M7), exact speculative verification would need that additional
     guarantee across its draft and verify shapes. The M2 operation
     contract should record row-invariance per kernel, which the P0 study
     partly shows:
     - GGML's vector attention kernel is row-invariant by construction;
     - cuBLAS's algorithm choice is not.
2. **Lay weights out for the kernel's read.** Regrouping MLX's packed
   4-bit words into contiguous per-program blocks took the 27B's matmuls
   from 107–130 to 200–220 GB/s, against 240 it reports measuring for
   GB10 (creator-reported). That bears
   on D-056's import-time repacking: the prepared artifact can carry the
   layout the chosen kernel reads best, not only paging-friendly groups.
3. **Account host time against GPU time before reaching for graphs.** Its
   notes report that the 27B's 918 kernels take 12–14 ms of host time, hidden behind 50–85 ms of
   GPU work, so graphs would buy little. Small MoE forwards are
   launch-bound and need them. This is BP-F4's decision rule, with a worked
   example.
4. **Deterministic tensor parallelism across two Sparks.**
   - Row-parallel partial sums are all-gathered and added in rank order.
     NCCL's all-reduce order is an implementation detail, so it is not
     used.
   - The head is split by vocabulary, with a top-k merge.
   - This is directly relevant to M6 sharding if restored or migrated
     state must reproduce bit for bit.
5. **Choose drafters and window widths by measurement.**
   - The engine picks per request whichever drafter commits more tokens
     per millisecond.
   - The window grows only while committed tokens per round-time rise.
   - Draft heads read a partial vocabulary. Its notes report that 98,304
     ids cover 99.6% of committed tokens.

   These are inputs to M7.
6. **A trap to check on our Sparks.** Its notes report GB10's kernel
   migrating pages under a running model. Some runs went at half speed
   during bursts of 45,000–265,000 migrated pages, and dropping the page
   cache after load moved about 25 GB. Our design leans on unified memory
   and the page cache (D-004, D-034), and this could also explain some of
   P0's between-process timing noise.

   A first look at `spark` on 2026-09-26 (read-only, `/proc`):
   - automatic NUMA balancing is off (`kernel.numa_balancing=0`), and
     `numa_pages_migrated` is 0;
   - memory compaction has been moving pages. Since the 2026-09-21 boot:
     - `pgmigrate_success` reached 58,470,221. That counts every source of
       migration;
     - `compact_isolated` and `compact_success`/`compact_fail` tie most of
       it to compaction;
     - there were 162,635 direct-compaction stalls (`compact_stall`) and
       443,297 wake-ups of the compaction daemon;
     - `compaction_proactiveness` is 20, the kernel default;
   - over one minute during a P0 timing session, `pgmigrate_success` did
     not change.

   So compaction, not NUMA balancing, is the likely mover here. The
   migrations may cluster around large allocations and page-cache churn.
   It is worth one measurement in M2 or M4: the counters sampled across
   loads, evictions and a steady decode.
7. **Benchmark method.**
   - The same client for both engines.
   - Medians over seeds, since, per its notes, single seeds varied up to
     2×.
   - Nothing else on either GPU.
   - A hash check of the drafted output against serial decoding on every
     run.

   This matches our frozen protocol's spirit, and the hash check is a good
   addition wherever we compare decoding with drafts.

## Proposed use

- **As a benchmark target.**
  - It becomes a pinned reference engine for the models it supports that
    overlap ours: Qwen3.8 Flash Next now, others as they enter both
    libraries.
  - Its comparisons would use its normal configuration (drafts on) and a
    matched one (`--no-drafts`, its serial reference), as D-021 and D-068
    ask. They land where drafting does: M3 for resident decode, and M7 for
    speculative decoding.
- **Format.** It reads only MLX 4-bit. A like-for-like comparison needs
  either jitLLM support for that format, which is a new import format and
  so a feature question, or an explicit cross-quantization comparison
  reported as such.
- **Before relying on its numbers.**
  - Pin a commit, and run its own benchmark client on `spark` against its
    serial mode and a pinned vLLM, as a reference-only run.
  - Record the result under `docs/experiments/`, as with the other
    references.
  - This waits until the Spark is free of P0's timing sessions.
