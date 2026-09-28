<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 Flash decode past llama.cpp (M3)

The owner, 2026-09-28: "at least match llama.cpp decode speed if not beat
it. Bonus if we can decisively beat it", and "speed matters more than bit
exactness". Targets, against llama.cpp's defaults (fusion and CUDA graphs
on) on the same GGUF: plain decode above 20.54 tok/s (llama-bench tg64),
DSpark decode above 30.80 / 31.94 tok/s on `prose` / `code`
([baselines](../fast-swap/baselines.md)); "decisively" is clearly beyond
run-to-run spread, about 1.1×.

## The policy

- **Default paths optimize for speed:** fused operations, batched kernels,
  other reduction orders. The bit-exact paths stay as an optional
  reference mode (`--exact on` in `jitllm_dsv4_exec` and
  `jitllm_spec_runner`, `Dsv4Options::exact` in the runner): the graph node
  for node as llama.cpp builds it, planned unfused, and D-092's
  row-invariant verify.
- **Correctness is judged coarsely against llama.cpp** on the same GGUF
  (D-085's note of 2026-09-28): greedy agreement except near-ties, and
  perplexity within 3% (relative).
  The near-tie bound comes from jitLLM's own kernel-to-kernel noise and is
  recorded below before any comparison with llama.cpp.
- **The engine's own determinism stays exact:** the same engine and kernels
  repeat themselves bit for bit, across runs, swaps and restores; a
  speculative verify's rollback keeps the accepted rows' writes and
  restores every other state byte (the rejected rows' and the scratch
  rows') to its value before the verify.

## Correctness protocol (fixed before the comparison)

Inputs are dsv4-native's ([README](../dsv4-native/README.md)): the eight
prompts of its `prompts.tsv`, 32 greedy tokens each, llama.cpp's oracle
arms (fusion off and on) at `b29c606e`, and its perplexity text.

1. **The noise bound.** `jitllm_dsv4_exec` forced on the unfused oracle's
   tokens twice, `--exact on` and the fast plan; `judge.py noise` takes,
   at each of the 256 steps, how far the fast plan moves the reference's
   top-two margin (its logit difference between the reference's two best
   tokens, against the reference's), and the bound B is twice the largest
   such move. B is recorded here before step 2 runs.
2. **Greedy agreement.** The fast plan forced on each oracle arm's
   generated tokens (`judge.py greedy ORACLE JITLLM --bound B`): its argmax
   equals the arm's token at every step, except where the arm's own margin
   between its token and jitLLM's argmax is below B. Every exception is
   listed.
3. **Perplexity.** The fast plan's perplexity on the text within 3% of each
   arm's, in chunks of 8 tokens (so that the fast plan's decode kernels, not
   only its prefill, score every token) and of 512.
4. **Speculation** (`jitllm_spec_runner`): speculative greedy decoding's
   every token is the plain engine's argmax on its own prefix (the plain
   engine teacher-forced on the speculative tokens), or within B of it; the
   three speculative runs of a prompt repeat each other bit for bit; the
   forced-rejection run's rollbacks leave every state byte outside the
   accepted rows' writes as it was before the verify (the whole target
   state and the drafter's ring compared at every step); sampled
   speculation's total variation within 0.1, as before. The reference mode
   keeps D-092's bit-for-bit checks, the control run among them.

Recorded values (before step 2 ran):

- The reference mode (`--exact on`) forced on the unfused oracle's tokens:
  256 of 256 argmax the oracle's, every logit bit-identical (dsv4-native's
  `compare.py`: max abs 0.0 on all 8 prompts). The reference is intact.
- The fast plan against it (`judge.py noise`, 256 steps): the top-two
  margin moves by median 0.13, p95 0.85, p99 1.41, at most 3.05; every
  logit by at most 12.8 (RMS at most 0.54 per prompt). **B = 6.11**
  (twice 3.05).

### The bound, going forward

B = 6.11 is too loose to be a test, for two reasons found in review:

- **It is circular on the unfused arm.** The reference mode equals the
  unfused oracle bit for bit, so "the fast plan against the reference"
  on the oracle's tokens is the very comparison step 2 judges on that
  arm; with B at or above the largest move, that check cannot fail.
- **Twice the maximum rewards outliers.** One step (3.05, against a p99
  of 1.41) set it, and doubling it admits disagreements where the oracle
  prefers its token by up to 6 nats (about 450:1), where a real defect
  (a wrong expert, a stale state row) can hide.

Later slices set the near-tie bound as **the 99th percentile of the
top-two margin move between two of jitLLM's own paths on the same
tokens, neither of them the oracle, for the path the verdict judges**,
recorded before the comparison: for plain decode against the oracle,
e.g. the fast decode against another jitLLM plan (as qwen38-native did,
its 95th percentile); for speculation, the batched verify's rows against
plain one-row decode teacher-forced on the same tokens, which
`jitllm_spec_runner` now reports per prompt (`verify_noise`). Not twice
the maximum, and not a noise measured against the arm being judged.

On this slice's data (a review's re-run of the final build, `spark-b`,
2026-09-28; post hoc, so a recommendation, not a re-judgement): the fast
decode against the reference moves the margin by p99 1.18, at most 1.55;
the batched verify against one-row decode by median 0.09–0.42 per prompt,
p99 2.11 (`prose`, 255 steps) and 2.35 (`capital` forced, 255 steps),
at most 3.94. With B at about 2.5 every greedy exception passes (the
largest is 0.96), and every speculative exception but one: `capital`'s
forced run at step 93, where plain decoding prefers the end of sequence
by 3.62 nats and the verify's row prefers its token by 0.32 (a move of
3.94, the largest seen). It repeats bit for bit across runs; whether it
is kernel noise (the verify's attention and routing on four rows) or a
defect is not settled, and it passes only under 6.11.

## The fast plan

`Dsv4GraphOptions::fused` (`kernels/ggml/dsv4_graph.cc`), the default
for decode, verify and draft chunks of up to 8 rows (`kVecQTokens`), and
for the drafter; prefill chunks keep GGML's matrix kernels. The kernels
are in `kernels/ggml/dsv4_fast.cu` (the vector product's body derives
from GGML's MMVQ, MIT), declared as jitLLM operations
(`jitllm_ops.h`, "DeepSeek V4's fast plan") and bound through the
registry like every other implementation (D-053):

- **`jitllm.vecq`**, one quantized vector-product kernel for every product
  of a decode or verify chunk: dense (the attention projections, the
  shared expert, the head), grouped (the attention output's 8 groups in
  one launch) and routed (the experts). Routed, each distinct expert of a
  chunk is read once, by one block that computes every (row, slot) that
  selected it, so a 4-row verify reads the ~19–20 distinct experts of its
  24 selections a layer, not 24 (the batched verify). The gate and up
  products share the launch with SwiGLU (and DeepSeek's clamp) applied on
  the way out. Weight types IQ2_XS, IQ3_XXS, MXFP4, Q8_0, Q4_K, Q5_K and
  Q6_K; launch shapes (rows per block, warps, passes, warp or block
  reduction) chosen per type and shape from `jitllm_vecq_bench`'s
  measurements on the Spark; programmatic dependent launch (PDL) so each
  kernel's weight prefetch overlaps its predecessor's tail, and L2
  prefetch two iterations ahead for the byte-heavy types.
- **`jitllm.q8_1`**: the activations quantized once per chunk and shared
  by every product that reads them (GGML quantizes per product).
- **`jitllm.dsv4.route`**: router scores to experts and weights in one
  kernel (sqrt-softplus, the bias, top-6, ties to the lower index, the
  hash layers' table, normalization and scale); **`jitllm.dsv4.combine`**:
  the weighted sum of the experts plus the shared expert.
- **`jitllm.dsv4.hc_mix`** and **`jitllm.dsv4.hc_pre`**: the
  hyper-connection pre-mix (the 24 mixes, Sinkhorn on the 4×4
  combination, the weighted sum and the RMS norm) in two launches instead
  of about ten.
- **`jitllm.dsv4.compress`**: a CSA or HCA compressor's scoring, softmax
  and weighted sum in one kernel.
- Planning (`graph_plan.h` `DeviceChoices`): RMS norm × weight fused
  where GGML's gate allows it, and small float products on the
  row-vector kernel.

The exact path is untouched: `--exact on` builds the graph node for node
as before and plans it unfused, and the verify is D-092's row-invariant
plan.

## Levers, in the order tried

`spark-b`, one host, each run gated on an idle GPU and memory; plain
decode is `jitllm_swap_pairs --bench 64`'s "request lease, graphs" mean of
three (as the wake baselines), DSpark the median of the three `prose` /
`code` speculative runs. Before the runtime wake merged, the harness
polled (its 100 ms window).

| Step | Change | Plain tok/s | DSpark `prose` / `code` | Kept |
| --- | --- | --- | --- | --- |
| Baseline (main, polled) | exact plan, row-invariant verify | 20.34–20.46 | 28.75 / 29.70 | – |
| 1 | batched verify on GGML's own kernels | – | 29.85 / 34.05 | yes (superseded by 3) |
| 2 | `jitllm.vecq`, first shape (2 rows, 4 warps, 8 passes) | – | 22.15 / 22.93 (verify 120–169 ms) | no: reworked |
| 3 | `jitllm.vecq` with measured launch shapes, dense and routed; route, combine, fused SwiGLU, HC pre-mix | 20.99 | 32.06 / 32.64 | yes |
| 4 | the grouped attention output on `jitllm.vecq`; norms fused in the plan, small float products on the vector kernel | – | 32.77 / 33.41 | yes |
| 5 | the fused compressor | – | 32.30 / 33.89 | yes |
| 6 | PDL and L2 prefetch in the vector kernel | 21.74 | 32.19 / 33.66 | yes |
| – | several products in one launch (grouped multi-matrix) | – | no gain | no: removed |
| – | a 4-row HC mix shared by decode | decode regressed | – | no: one-row and four-row forms |
| Final (main's wake, 00aaa97) | – | **21.90 / 22.21** | **31.64 / 34.33** and **31.71 / 34.38** | – |

Steps 1–6 changed accepted tokens as well as speed (acceptance 0.53–0.56
on `prose`, 0.57–0.60 on `code`), so the DSpark column moves with both.

## Results

**Speed** (the final build, the runtime's own wake, 2026-09-28 13:05–13:17,
no other GPU process; llama-bench in the same session, tg64, three
repetitions):

| | jitLLM | llama.cpp | Ratio |
| --- | --- | --- | --- |
| Plain decode (two runs) | 22.21, 21.90 tok/s | 20.41 (fusion on), 19.98 (off) | 1.07–1.09× on, 1.10–1.11× off |
| DSpark `prose` (two runs, repeats) | 31.64, 31.71 (31.0–31.8) | 30.80 | 1.03× (1.01–1.03×) |
| DSpark `code` (two runs, repeats) | 34.33, 34.38 (34.2–34.5) | 31.94 | 1.07–1.08× |

Harness-polled, before the wake merged (step 6 above): plain 21.74,
DSpark 32.19 / 33.66 (1.06×, 1.05× / 1.05×). The llama.cpp DSpark numbers
are the recorded ones ([baselines](../fast-swap/baselines.md)); only its
plain decode was re-measured in the session. Acceptance 0.531 / 0.597
(2.55 / 2.77 tokens a verify); a speculative step is 79.5–80.2 ms. A
plain decode step's device time is 44.8–45.4 ms (llama.cpp's
1/20.41 s is 49.0 ms).

**Memory:** plain decode's peak by `MemAvailable` 94.9–95.0 GiB; DSpark's
peak drop 106.6–106.7 GiB (106.5–106.8 on the exact plan after the
lease and wake changes); `jitllm_dsv4_exec` 93.1–93.2 GiB against the
exact plan's 92.6 (the shared Q8_1 activations and the fused
intermediates).

**Profile** (`nsys`, graph nodes traced, which inflates each step;
`jitllm_spec_runner --check greedy --only prose --tokens 64`, the same
command before and after):

| Graph | Before (exact) | After (fast) |
| --- | --- | --- |
| Decode step | 52.06 ms, 5,575 kernels | 45.55 ms, 2,558 |
| – of it quantized products | 43.17 ms (`mul_mat_vec_q`, 619) | 39.89 ms (`VecQKernel`, 533) |
| – of it kernels under 10 µs | 4,762, 9.64 ms | 1,896, 5.37 ms |
| Verify (4 rows) | 83.52 ms, 6,041 kernels | 72.44 ms, 2,590 |
| – of it quantized products | 70.91 ms (row-invariant, 623) | 66.00 ms (533) |
| Draft | 7.62 ms, 281 kernels | 7.23 ms, 133 |

The products remain 88% of a decode step and 91% of a verify. Measured
alone (`jitllm_vecq_bench`), the vector kernel streams the decode's
shapes at 230–245 GB/s; inside the model it reaches about 200–210 GB/s.
The gap is not explained and stays open: tracing, the VMM layout, CPU
spinning and clock drift were each ruled out. A 4-row verify costs 1.59× a decode
step because it reads the ~19–20 distinct experts of 24 selections.

**Correctness** (the final build; B = 6.11 as recorded above):

- The reference mode: 256 of 256 argmax equal to llama.cpp's unfused
  arm, every logit bit-identical on all 8 prompts.
- The fast plan's noise re-measured on the final build: margin moves
  median 0.15, p95 0.69, p99 1.18, at most 1.55 (the rule would give
  B = 3.10; the recorded 6.11 was kept). The greedy and perplexity
  verdicts below hold under 3.10; the speculation verdict does not: one
  forced-run token (`capital`, step 93, 3.62) passes only under 6.11 (see
  "The bound, going forward").
- Greedy agreement: 244 of 256 with the unfused arm and 250 of 256 with
  the fused arm; every exception a near-tie (the arm's own margin
  0.005–0.49 unfused, 0.02–0.96 fused), 0 violations.
- Perplexity 20.248 (chunks of 512) and 20.142 (chunks of 8) against
  20.231 unfused and 20.215 fused: within 0.09–0.44%.
- Speculation: every speculative token the plain engine's argmax on its
  prefix or a near-tie (`prose` and `code` 254 of 256, the chat prompts
  30–32 of 32, the forced run 251 of 256; 0 violations under 6.11, the
  largest exception 3.62 in the forced run, every other under 0.78),
  the three speculative runs of each prompt
  bit-identical, both greedy runs identical to each other. Forced
  rejections (fast plan): 153 steps, 141 with rejected rows (all rejected,
  one and two accepted, CSA and HCA blocks rejected), 0 stale bytes in
  70.8 GB compared; the exact mode's control: 138 steps, 0 states
  differing. Swap: A out after step 78 (rows rejected) and back, 156
  steps, 0 states differ, B's logits their recorded hash. Sampled
  (`capital`, `haiku`, `sky`, `fibonacci`, 256 seeds, the first 8
  tokens): total variation 0.0039, 0.0112, 0.0376, 0.0220 (bound 0.1);
  plain sampling took 599 s, speculative 437 s.

## Judgement calls and limits

- **Not decisive.** The "decisive" mark (about 1.1×) is not reached on
  any target: plain decode is 1.07–1.09× llama.cpp's default (1.10–1.11×
  only its fusion-off arm), DSpark `code` 1.07–1.08×, and DSpark `prose`
  1.01–1.03×, a win inside two runs' spread. The owner's first goal
  (match or beat llama.cpp) is met; the bonus is not. What is left is the
  products' bandwidth and the verify's distinct experts; the attention,
  indexer and host levers were not reached, since every profile put them
  under 5% of a step.
- **Open: the products' in-model bandwidth.** The vector kernel streams
  230–245 GB/s alone and about 200–210 GB/s inside the model; the gap is
  not explained (tracing, the VMM layout, CPU spinning and clock drift
  ruled out) and is the largest lever left.
- **The fast forced check has no control:** a verify's rows need not
  equal a shorter verify's on the batched plan, so the control run's
  hashes cannot match; the whole target state and the drafter's ring are
  instead read before each verify and after its rollback and compared
  everywhere outside the accepted rows' writes, so the rejected rows'
  and the scratch rows' writes must be back to their bytes before the
  verify, and nothing the write list does not name may change. What it
  does not show on its own: that the write list attributes each write to
  the right row (the model's `Dsv4ChunkWrites`, the same on both plans,
  which the reference mode's control run checks bit for bit), and that
  the accepted rows' contents are right (only the near-tie rule on the
  tokens after them checks that).
- **B kept at 6.11** as recorded before the comparison, although it is
  too loose to be a test (circular on the unfused arm, twice an outlier):
  every greedy exception is under 1.0, but one speculative exception is
  3.62 and passes only under it. Later slices use the rule in "The
  bound, going forward"; that exception is open until diagnosed.
- **Speed from the default wake** as the final number; the polled
  numbers are reported beside it.

## Reproduction

On `spark-b` with the `spark-native` build, dsv4-native's artifact and
oracle, DSpark's drafter and the FP16 fixture:

    jitllm_dsv4_exec --artifact DSV4 --context 4096 --prompts P --generate 32 \
      --force ORACLE/{unfused,fused}/generated.tokens --out DIR [--exact on] \
      [--ppl ORACLE/unfused/ppl.tokens [--max-rows 8]]
    judge.py noise EXACT FAST
    judge.py greedy ORACLE FAST --bound 6.11
    judge.py ppl ORACLE FAST
    jitllm_spec_runner --dsv4-artifact DSV4 --drafter DRAFTER --prompts prompts.json \
      --out DIR --check greedy|forced|swap|sampled-plain|sampled-spec \
      --margin 6.11 [--exact on]
    jitllm_swap_pairs --a dsv4 --b qwen38 --cycles 0 --bench 64 ...
    jitllm_vecq_bench [--only NAME] [--launches N] [--vmm on]

`judge.py` runs under the pinned llama.cpp image's Python (it needs
NumPy).
