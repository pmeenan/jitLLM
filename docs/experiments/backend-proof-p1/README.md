<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Backend proof P1: BP-F1's timing calibration — 2026-09-27

BP-F1 asks whether jitLLM's GGML kernels run slower when their memory is
host VMM than when it is `cudaMalloc` memory (D-034's reopen condition;
[backend-proof.md](../../backend-proof.md#performance-protocol-rule-approved-2026-09-26-bp-f2s-reference-deferred-to-p3-entry)).
The approved kernel-timing rule needs BP-F1's own noise calibration and
holdout before any comparison. This report records that calibration, which
settles BP-F1's rule under D-079. It holds no host-VMM timing: none has
been run.

**Results in brief.**

- **53 cases**, derived from the FP16 bridge's recorded plan for the
  held-out trajectory at 1, 16, 17 and 512 rows. Each is a kernel jitLLM
  has: RMSNorm, fused RMSNorm-multiply, multiply, three kinds of add, five
  projections, and attention's KQ and KQV products. Every block verified
  every captured launch against the recorded plan.
- **Calibration** from four A/A `cudaMalloc` sessions on `spark` (`c1` to
  `c4`, two in each order): the median `σ` is 1.23%, from 0.10% (the output
  head at one row) to 2.75% (the k/v projection at one row). With
  `z = 3.555` for 53 cases, the per-case thresholds `z · σ · √½` are below
  2% for 13 cases and below 5% for 45; the largest is 6.9%.
- **The holdout passes.** It was declared in advance to reject the rule if
  either holdout session, taken as the primary with the other as its
  confirmation, failed the stage. `h1` (primary order) passes alone:
  no case over `z`, aggregate `t` 1.87. `h2` (mirrored) as the primary has
  one case over `z` (fused RMSNorm-multiply at 16 rows, d = 3.73), which
  `h1` clears as its confirmation; aggregate `t` −2.48 and 1.87. The stage
  passes both ways.
- **Every in-sample pairing** of the calibration sessions passes; those
  with `c3` as the primary need a confirmation (16-row 128-wide bias add,
  d = 3.68).
- **Power** for a small slowdown of one case is lower than BP-F2's,
  because these kernels are small: slowed by 2% in both sessions, a case
  is detected 21–28% of the time (BP-F2: 48–54%); at 5%, 74–83%; at 10%,
  98–100%. A slowdown of every case, or of every matrix
  product, fails the stage from 0.5%.
- **The calibration file** [`bpf1-calibration.json`](bpf1-calibration.json)
  has SHA-256
  `aa1581271357e8c1bfeed2b8da98ae98ee35031b4160df6500cb8b6e991b1369`,
  pre-registered in backend-proof.md with the harness's and the case
  file's. The session driver refuses a host-VMM arm under any other
  calibration, harness or case file.

## Conditions and provenance

- **Host.** `spark` (`spark-c4e2`): GB10, kernel 7.0.0-1019-nvidia, driver
  580.178.04, persistence mode on, application clock 2,418 MHz, CPU
  governor `performance`. The clock policy and idle states were left
  unchanged. Before each session the driver checked that no compute
  process was on the GPU and the load average was below 1.0; from `c3` on,
  a wrapper also waited for it to fall below 0.5 (our own previous
  session's decay). The
  driver checked for other compute processes again before every block; none
  appeared. Between block boundaries the SM clock stayed at 2,405–2,437 MHz
  and the GPU at 49–63 °C, with no active throttle reason.
- **Sessions.** 2026-09-27, 05:27–05:51 UTC: `c1` (primary order), `c2`
  (mirrored), `c3` (primary), `c4` (mirrored), then `h1` (primary) and `h2`
  (mirrored). Each took about two minutes: three discarded warm-up
  processes and eight timed blocks. A first `c3` attempt stopped before
  running anything because the load average (1.04) was above the limit; its
  empty directory was removed.
- **Harness.** `jitllm_ggml_vmm_bench` from the `cross` preset, SHA-256
  `05348df868e83768a441302bc2831df8ebbe4fa609a13c5b5cd874d9048f3b92`,
  copied read-only beside the pinned cuBLAS 13.8.0.4 (`libcublas.so.13`
  `ee7c1657…`, `libcublasLt.so.13` `ba3b942f…`, the libraries the FP16 plan
  was recorded with). That copy, kept with the raw sessions, is the binary
  the comparison runs: a rebuild of the same source reproduces it only at
  the same path, since it embeds its source paths. The case file
  [`bpf1-cases.txt`](bpf1-cases.txt) is `fe78d033…`; the session driver
  `bpf1_session.py` was `846f41b3…` (the current one adds only the
  host-VMM check of the harness and case file). Sources: commit `7a4b7688` plus the
  uncommitted BP-F1 work, SHA-256 of `git diff --binary HEAD` followed by
  `sha256sum` of each untracked file `f89a3b70…`. Each manifest records
  all of these.
- **Rule.** `timing_protocol.py` at `c05fd2dd…` (the approved rule, the
  copy on `spark` from P0) computed the calibration, the outcomes and the
  power. The current script gives the same calibration byte for byte and
  the same outcomes.
- **Aggregates.** [`bpf1-timing.json`](bpf1-timing.json) holds every
  session's manifest (host, GPU settings, clocks and temperatures at block
  boundaries, identities) and each case's block medians for both arms,
  graph and stream-launched; the in-sample pairings, the holdout, and the
  power estimates. Raw samples and logs stay outside Git, on `spark` in
  `~/.local/share/jitllm/bpf1-20260927/`.

## Cases

[`bpf1_cases.py`](bpf1_cases.py) derives the cases from
[`fp16-plan.json`](../backend-proof-p0/fp16-plan.json), the bridge's
recorded executed plan for the held-out trajectory. It flattens each chunk
size's sequence, cuts it into operations (a cuBLAS product is its
conversions, pointer setup, memset and GEMM, and its output conversion when
it computes in F16), and labels them by position in the layer. Operations
of the same shape (q and o, k and v, gate and up, the three norms, the
two residual adds) must have launched identically in every layer; each
becomes one case, whose launches are the recorded ones. At one row a
residual add has the bias add's operands exactly, so they are one case.
The unfused arm gives the plain operations; the fused arm gives the fused
RMSNorm-multiply.

| Operation | Shape | 1 row | 16 rows | 17 rows | 512 rows |
| --- | --- | --- | --- | --- | --- |
| `rms_norm`, `mul`, `rms_norm_mul` | 896 wide | GGML | GGML | GGML | GGML |
| `add.bias_896`, `add.bias_128` | bias broadcast over rows | GGML | GGML | GGML | GGML |
| `add.residual` | 896 × rows, both operands | (= bias add) | GGML | GGML | GGML |
| `linear.q_o` | 896 → 896 | MMVF | MMF | cuBLAS | cuBLAS |
| `linear.k_v` | 896 → 128 | MMVF | MMF | cuBLAS (split-K) | cuBLAS |
| `linear.gate_up` | 896 → 4,864 | MMVF | MMF | cuBLAS | cuBLAS |
| `linear.down` | 4,864 → 896 | MMVF | MMF | cuBLAS | cuBLAS |
| `linear.lm_head` | 896 → 151,936 | MMVF | MMF | cuBLAS | cuBLAS |
| `attn.kq` | 64 × 14 heads over 2 KV heads, cache 1,024 | MMVF (256 cells), MMF (768) | MMF (256) | cuBLAS TF32, pointer array (256) | cuBLAS (768) |
| `attn.kqv` | the same, V transposed | MMVF (256 and 768) | MMF (256) | cuBLAS, pointer array (256) | cuBLAS (768) |

The KV cells in use are the trajectory's (`n_past + rows`, padded to 256)
at each sequence's first occurrence: 256 up to the 512-row chunk and 768
after it, so one row has both.

Not cases, because jitLLM has no implementation of them yet: RoPE, softmax,
`set_rows`, `get_rows`, the copy after attention, the SiLU gate, and the
fused MMVF variants (bias, gate and residual fused into the product) of the
fused arm's single-row steps.

## Method

[`../../../benchmarks/ggml_vmm_bench.cc`](../../../benchmarks/ggml_vmm_bench.cc)
runs one block: every case in one process, all its memory of one kind. The
session driver [`bpf1_session.py`](bpf1_session.py) runs the blocks.

- **Operand placement.** In a block, every buffer the case's kernels are
  given is in the block's memory kind: weights, activations and outputs,
  GGML's scratch pool (sized by the plan) and the cuBLAS workspace (32 MiB,
  upstream's). For `cudaMalloc` each is a `cudaMalloc` allocation; for host
  VMM, a host-backed VMM reservation from jitLLM's provider, mapped
  read-write. Only a staging buffer for setup (host VMM in both kinds) is
  outside; no timed kernel touches it.
- **Rotation.** Each case has a ring of operand sets, each holding all its
  tensors, with more sets than fit four times into the queried L2 (24 MiB
  on GB10), and at least two. Invocation `k` uses set `k mod N`, so no
  invocation reads what the one before it used, and a set is read again
  only after the ring's other sets, which with it total more than four
  times L2. The graph arm uses sets 0 to 359 and the stream arm the next
  360, so the rings of more than 720 sets (the small operations) are only
  partly read. The sets
  hold identical data. N ranges from 2 (the output head) to 65,537
  (the 128-wide bias add at one row).
- **Launch verification.** Each sample's graph is captured separately (36
  graphs, ten invocations each, on consecutive sets) and read back through
  the driver's graph API. Every graph's launches must equal the recorded
  plan's, ten times over, in order: kernel name (with NVCC's per-file
  internal-namespace tag normalized), grid, block, static plus dynamic
  shared memory and registers; or a memset's size and value. A mismatch
  stops the block before anything is timed. All 53 cases matched in every
  block.
- **Samples.** Five warm replays, then 31 samples, each one graph replay
  bracketed by two captured events, the interval divided by ten. The
  stream-launched arm follows: the same through the launch context on the
  stream, on the next 360 sets of the ring, reported beside and not gated.
- **Outputs.** Set 0's output is hashed after an eager invocation and again
  after both arms; the two must match, and `bpf1_stats.py` requires one
  hash per case across all eight blocks of a session.
- **Session.** A discarded warm-up process per arm, one more immediately
  before the first timed block, then eight blocks in the primary order
  A1 B1 B2 A2 B3 A3 A4 B4 or the mirrored B1 A1 A2 B2 A3 B3 B4 A4. Here A
  and B are both `cudaMalloc`.
- **Statistics.** [`bpf1_stats.py`](bpf1_stats.py) summarizes a session in
  the schema `timing_protocol.py` reads: per case the ratio of the arms'
  medians over their 124 samples and each block's median.
  [`bpf1_record.py`](bpf1_record.py) condenses the sessions and computes the
  outcomes.

## Calibration

`σ` per case is the relative standard deviation of the eight block medians
within a session, pooled over `c1`–`c4` (32 process medians per case).

| Kernels | Cases | Median `σ` | Largest `σ` |
| --- | --- | --- | --- |
| Elementwise and norms | 23 | 1.50% | 2.51% (`mul`, 1 row) |
| Projections | 20 | 1.11% | 2.75% (`linear.k_v`, 1 row) |
| Attention products | 10 | 0.75% | 2.15% |
| All | 53 | 1.23% | 2.75% |

The quietest cases are the long ones: the output head (0.10–0.43%) and the
single-row KQV at 256 cells (0.14%). The noisiest are the one- to
two-microsecond kernels at 1, 16 and 17 rows. Across the six sessions the
A/A ratios of arm medians spanned 0.962–1.046.

Whole processes run slightly fast or slow, as P0 found for EXL3: a sign
test over the cases' ratios rejects at 1% in two of the six A/A sessions
(`c1`, 34 slower and 14 faster; `h2`, 13 slower and 38 faster). That is what
the rule's block-level aggregate test is for, and its `t` stayed within
−2.48 to 2.39.

## Holdout

Declared before `h1` and `h2` ran, and recorded in their manifests: the rule
is rejected if either holdout session, taken as the primary with the other
as its confirmation, fails the stage.

| Primary | Confirmation | Over `z` in the primary | Aggregate `t` | Stage |
| --- | --- | --- | --- | --- |
| `h1` | (not needed) | none | 1.87 | passes |
| `h2` | `h1` | `rms_norm_mul` at 16 rows, d = 3.73 | −2.48, 1.87 | passes |

In sample, every ordered pairing of `c1`–`c4` passes; the three with `c3`
as the primary need a confirmation (`add.bias_128` at 16 rows, d = 3.68).

## Power

One case slowed in both sessions of a pair (`timing_protocol.py --power`):

| Slowdown | `h1` + `h2` | `c1` + `c2` | `c3` + `c4` |
| --- | --- | --- | --- |
| 2% | 21% | 28% | 25% |
| 3% | 38% | 43% | 38% |
| 5% | 74% | 83% | 75% |
| 10% | 100% | 98% | 100% |

A subset slowed together (the stage fails at or above):

| Subset | Cases | `h1` + `h2` | `c1` + `c2` | `c3` + `c4` |
| --- | --- | --- | --- | --- |
| All cases | 53 | 0.5% | 0.5% | 0.5% |
| Matrix products | 30 | 0.5% | 0.5% | 0.5% |
| Elementwise and norms | 23 | 2% | 1% | 2% |
| 512 rows | 13 | 0.5% | 1% | 0.5% |
| Noisiest quarter | 14 | not at 3% | 3% | 3% |

A host-VMM penalty, if there is one, would be expected across many cases
at once, where the aggregate and the per-case tests together are
sensitive.

## Limitations

- **One host, one day.** All sessions ran on `spark` within half an hour;
  `spark-b` served only development runs, which are not evidence.
- **Not the whole model.** The cases are the kernels jitLLM has; RoPE,
  softmax, the KV writes, the SiLU gate and the fused single-row MMVF
  variants are not covered until they exist. Attention runs at the
  trajectory's two KV lengths only.
- **Synthetic operand values.** Deterministic pseudo-random values stand in
  for the fixture's weights; the same values fill both memory kinds.
- **Everything in one kind.** The comparison places scratch and the cuBLAS
  workspace with the operands; it does not separate weights on host VMM
  from activations elsewhere.
- **Small kernels are noisy.** Thresholds reach 5–7% for some one-row
  elementwise kernels and the one-row k/v projection; a regression smaller
  than that in one such kernel alone would pass.
- **The rule is BP-F2's.** Its structure (four processes per arm, the
  aggregate test, the confirmation) was tuned on EXL3 kernels; this
  calibration and holdout validate it for these cases, as D-079 delegates.

## Reproduction

On the workstation, build and deploy the `cross` preset
(`mise run deploy -- --host spark cross`), regenerate or check the case
file (`bpf1_cases.py ../backend-proof-p0/fp16-plan.json bpf1-cases.txt`;
`tools/tests/test_bpf1.py` checks it), and compute the source identity:

```bash
git rev-parse HEAD
{ git diff --binary HEAD; git ls-files -o --exclude-standard -z | sort -z | xargs -0 sha256sum; } | sha256sum
```

On `spark`, with the harness binary beside a `cublas/` directory holding
the SDK's `libcublas.so.13` and `libcublasLt.so.13` (the binary's RUNPATH is
`$ORIGIN/../cublas`), run each session:

```bash
bpf1_session.py OUT/c1 --harness benchmarks/jitllm_ggml_vmm_bench --cases bpf1-cases.txt \
  --arm-a cuda-malloc --arm-b cuda-malloc --order primary \
  --source-commit COMMIT --source-diff-sha256 DIFF --note '…'
```

Then, on the workstation:

```bash
bpf1_stats.py OUT/c1 > c1.json          # likewise for each session
timing_protocol.py --calibrate c1.json c2.json c3.json c4.json > bpf1-calibration.json
timing_protocol.py bpf1-calibration.json h1.json h2.json
bpf1_record.py --protocol timing_protocol.py --calibration bpf1-calibration.json \
  --sessions c1.json c2.json c3.json c4.json --holdout h1.json h2.json > bpf1-timing.json
```

A comparison session passes `--arm-b host-vmm --registry
docs/backend-proof.md --calibration bpf1-calibration.json` with the
registered harness and case file, and `bpf1_stats.py` needs `--registry`
for it.
