<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# TensorFold

- **Repository:** [ashhart/TensorFold](https://github.com/ashhart/TensorFold),
  MIT. A single-stream engine for MLX-format checkpoints with CUDA engines
  for the DGX Spark. It is an M3 baseline for Qwen3.8 Flash Next and a
  source of techniques ([tensorfold-assessment.md](../tensorfold-assessment.md)).
- **jitLLM's pin:** `71377a53` (0.3.6.2), measured as a baseline
  ([baselines](../experiments/fast-swap/baselines.md#qwen38-flash-next-tensorfold-mlx-4-bit-cross-quantization)).
  None of jitLLM's code comes from TensorFold.
- **The load study** is outside this repository, on the workstation at
  `/home/pmeenan/src/tensorfold-load-study/` (2026-09-28). Its `README.md`
  has the method and phase tables; `tensorfold/PATCHES.md` is the
  self-contained handoff, with diffs against `beddbb7b` and `71377a53` in
  `tensorfold/patches*/` and the tests in `patches/tests-test_load_io.diff`.

## Cold start: buffered reads and an int64 repack take ~130 of ~145 s

- **Status:** PR open,
  [ashhart/TensorFold#82](https://github.com/ashhart/TensorFold/pull/82),
  "perf: Improve cuda startup/loading performance (3-15x faster on DGX
  spark)", by the owner, opened 2026-09-29; no maintainer comment on
  2026-09-29.
- **Found:** 2026-09-28, TensorFold `beddbb7b` (0.3.5.1), `spark`.
- **Problem:** Flash Next's `weights.load` reads 80 GB through small buffered
  reads (the kernel's 128 KiB readahead at a queue depth under 1:
  1.1–1.2 GB/s from an SSD that reads ~13 GB/s), then repacks expert nibbles
  with an int64 loop (~1 s of GPU time a layer). The two do not overlap.
- **The study's patches (measured):** O_DIRECT reads into two pinned 64 MiB
  pieces with asynchronous copies, and the nibble shuffle in int32
  (bit-identical). Start to ready 145.1 → 45.5 s, same greedy output token
  for token.
- **What the PR does**, building on the study: O_DIRECT checkpoint reads
  with read-ahead across neighbouring tensors
  (`tensorfold/cuda/direct_read.py`); a fused CUDA expert-pack kernel
  (`tensorfold/cuda/experts_pack.cu`) replacing about 40 elementwise
  operations; GLM experts stacked on the GPU; fewer device synchronisations.
  Reported in the PR: 2.9–15× across models; Flash Next MLX 146 → 24 s to
  first token. Not re-measured in jitLLM.
- **Proposed action:** follow the PR through review.

## RUNBOOK: persist the kernel caches (study patch 4)

- **Status:** open.
- **Found:** 2026-09-28, `beddbb7b`; still applies at `71377a53`.
- **Problem:** in NVIDIA's container nothing persists `~/.cache` and
  `~/.triton`, so every start in a new container spends ~60 s compiling
  kernels a cache would keep (baseline difference).
- **Proposed action:** a small documentation PR: named volumes for both
  caches in `RUNBOOK.md` (`patches/patch-4-runbook-persist-kernel-caches.diff`
  in the study, untested as written). It is not among PR #82's listed
  changes; check the PR before sending it separately. Minutes.

## JIT kernels built for every architecture fail in NVIDIA's container (fixed upstream)

- **Status:** fixed upstream at `34bae79` (0.3.6.1, "CUDA builds inside
  NVIDIA's containers again", fixes TensorFold #56).
- **Problem:** the first start compiled for `TORCH_CUDA_ARCH_LIST`, which in
  NVIDIA's container starts at sm_80, and `qmm.cu` failed there.
- **Proposed action:** none.

## Upstream techniques to adopt

- **Tiled, bit-exact QSA block selection past 131,072 keys:**
  [ashhart/TensorFold#93](https://github.com/ashhart/TensorFold/pull/93)
  (open, by MovieMaker93, 2026-09-29), "perf: Flash Next lists QSA blocks
  past 131,072 keys in tiles (256k prompts 2.1x faster, same bits)", in
  `families/qwen4_exp/cuda/attention.py`.
  - TensorFold's `_select` kept the block scores in registers up to 32,768
    blocks. Past that they spilled, and a 256-row prompt block's layer took
    26–27 ms instead of 0.8 ms on a DGX Spark.
  - The fix, `_select_tiles`, processes longer rows in tiles (4,096 blocks
    for prompt rows, 8,192 for decode windows) with a radix select for the
    512th-largest order-preserving key, one byte per pass by histogram,
    keeping the block order identical.
  - Reported: 27.2 → 1.5 ms per block at 262K keys; a 250K-token prompt
    311 → 145 s and a 140K one 86 → 74 s; identical state after 140K and
    250K prompts, with BF16 and int8 KV. All creator-reported.
  - **Relevance to jitLLM:** `jitllm.qsa.select`
    (`src/kernels/ggml/jitllm_ops.h`, `kQsaSelectMaxBlocks` = 8,192 blocks,
    32 KiB of shared memory, 32,768 cells at Qwen3.8's ratio of 4) falls
    back to GGML's nondeterministic top-k past that
    ([ggml.md](ggml.md#radix-top-k-breaks-ties-nondeterministically-re-031),
    RE-031). A tiled, deterministic radix select through global memory like
    #93's would lift the limit and keep long-context Qwen3.8 repeatable.
    TensorFold is MIT, so porting the idea is fine (D-091).
  - **Proposed action:** a jitLLM long-context slice; no upstream action.
- Other TensorFold techniques, ranked for jitLLM, are in
  [tensorfold-assessment.md](../tensorfold-assessment.md#upstream-to-0362-2026-09-28).
