<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The runtime wake: from a step's fence to the next step's launch — 2026-09-28

The owner, on 2026-09-28: "We should benchmark with the actual runtime
wake mechanism in place so we have representative measurements." Until
then the paged harness polled for 100 ms (`NodeSettings::poll_window`),
so the scheduler thread and the device submission lane spun through
every step, the completion lane spun on its fence and the harness's
driver on the step's result: four cores, and a round trip of 0.01 ms a
step. The runtime's own defaults (200 µs windows) slept between steps
and paid 0.26–0.65 ms a step in wakeups (RE-017). This measures the
candidate ways to wait on the GB10, picks one, makes it the scheduler's
and lanes' default (D-094), and re-measures the models with it.

## The path

A decode step crosses four threads, each of which may be asleep:

1. the **device completion lane** (C) sees the step's fence complete
   (a query) and publishes it on the completion board;
2. the **scheduler thread** (K) harvests it and reports the step to the
   request's client;
3. the **client** (D; the harness's driver, which samples the next token
   on the host) does its host work and hands K the next step;
4. K posts it to the **device submission lane** (S), which launches it.

A sleeping thread on the Spark takes 0.1–0.8 ms to run once woken
(RE-017), and each hop can pay it.

## Method

`benchmarks/wake_bench.cc` (`jitllm_wake_bench`), two modes:

- **chain:** the four threads over raw CUDA, K and S waiting on jitLLM's
  `WakeFlag` as the scheduler and lanes do, the step a one-thread kernel
  that spins on the GPU's global timer and stamps its start and end. Round
  trip = the GPU's idle gap between a step's end and the next step's
  start, less the client's host work. Hops are host times; the GPU's
  timer is related to the host clock by a calibration (the smallest
  host-minus-GPU difference over 30 ms of a kernel publishing its timer,
  per configuration: the two drift apart by ~2 µs a second, so a hop can
  be off by some microseconds, and "detect" can read a few microseconds
  negative). CPU = the process's CPU time over wall, in cores, while
  stepping and while idle (the second after the last step, from 0.3 s
  on). Configurations: how C learns of the fence, and how K, S and D
  wait (below).
- **node:** the runtime's own path on `tests/support`'s paged node: a
  request holding its lease (D-093), 100 steps of the same kernel as
  device jobs under it, the harness's driver waiting for each; round trip
  = a step's wall less its device span (CUDA events around the job, as
  `StepTimes`).

Scenarios (chain): 45 ms steps with 0.2 ms of client host work (DeepSeek's
decode), 40 ms with 1.3 ms (Qwen3.8's), 5 ms with 0.2 ms (a short step),
and 45 ms mixed pseudo-randomly with 5 ms (a third of steps), so a
prediction is often wrong. 100 steps (200 for 5 ms), three repeats each.

Host: `spark-b` (GB10, Cortex-X925 and A725, 20 cores, kernel
7.0.0-1019-nvidia, driver 580.178.04, cpufreq `performance`, cpuidle
`menu`, LPI-0–3), the `spark-native` build of this slice's tree
(SDK `aarch64-e0a0c85c42806fb1`). No system settings changed. Each run
waited until no other GPU process ran and the 1-minute load was under 6
(twice, 20–40 s apart); other agents used the host between runs (load
0.6–2.5 at the runs' starts). Raw outputs: `~/scratch/m3wake/before`
(09:45–10:04, the tree before the runtime changed) and `after`
(10:09–10:26) on `spark-b`.

## The mechanisms (chain, before the change)

How C detected the fence: (a) `cuEventSynchronize` on a
`CU_EVENT_BLOCKING_SYNC` event; (b) a host function after the step
(`cuLaunchHostFunc`) bumping a futex word C sleeps on; (c) a stream
memory operation (`cuStreamWriteValue32`) writing a mapped host flag,
polled with yield, with a 50 µs timed futex wait (a GPU write wakes no
futex, so this is a timed poll), or with WFE; the query, yielding between
(the runtime until now); (d) an adaptive spin: sleep until 1 ms before
the expected end (in these runs the shortest of the stream's last eight
lengths; after the change each of them, below), querying at least every
1 ms, then spin to 1 ms past it, then back off; (e) (d) plus the relay:
as C starts to spin it anticipates K and S (`WakeFlag::Anticipate`), which
poll until then, and K, after a step, polls for about as long as the
client took lately and has S do the same; optionally with (b)'s host
function as the backstop in C's sleep. Whatever woke C, a query proves
the fence (D-048). Except in the first two rows, D waits by the same
adaptive spin as C (asleep through most of the step, woken early by K);
in the harness rows it spins. K and S otherwise poll for their 200 µs
window after their last progress and then sleep.

45 ms steps, 0.2 ms host work (µs; ranges over three runs of 99 hops;
p99 includes the run's first step, which has no history):

| Configuration | Round trip p50 | p99 | C detects, p50 | Cores stepping | Cores idle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harness-polled (100 ms windows; C, D spin) | 8 | 9 | ~0 | 3.99 | 0 |
| Runtime until now (200 µs windows; C, D spin) | 234–676 | 523–809 | ~0 | 2.00 | 0 |
| C queries, spinning; D adaptive | 277–462 | 677–774 | ~0 | 1.03 | 0 |
| (a) blocking-sync event | 1,589–1,781 | 2,163–2,396 | 991–1,415 | 0.05 | 0 |
| (b) host function + futex | 1,860–2,121 | 2,471–3,294 | 1,401–1,719 | 0.05–0.06 | 0 |
| (c) memop flag, yield | 286–711 | 719–1,538 | 0–1 | 1.03 | 0 |
| (c) memop flag, 50 µs futex poll | 304–346 | 392–821 | 17–68 | 0.06–0.07 | 0 |
| (c) memop flag, WFE | 289–732 | 765–1,089 | ~0 | 1.03 | 0 |
| (d) adaptive spin, C only | 271–477 | 497–878 | ~0 | 0.08 | 0 |
| **(e) adaptive spin + relay** | **29–30** | **38–212** | ~0 | **0.12–0.13** | **0** |
| (e) with a host function as backstop | 108–394 | 270–418 | ~0 | 0.13–0.14 | 0 |

40 ms steps, 1.3 ms host work: harness-polled 7–8 µs (3.97 cores); until
now 690–705 (1.97); (a) 698–1,959; (b) 1,589–2,203; (c) 599–1,744;
(d) 625–946 (0.11–0.12); (e) 29 µs, p99 100–315 (0.20–0.21 cores); (e) with the
host function 29–30 (0.20–0.22). 5 ms steps: harness-polled 8 (3.95);
until now 238–481 (2.00); (a) 1,634–1,712; (b) 902–1,900; (c) 224–715;
(d) 230–274 (0.43–0.47); (e) 24–29, p99 32–53, at 0.98–1.10 cores (C, K,
S and D each spin about 1.3 ms of every 5.2 ms); (e) with the host
function 38–95.

What they show:

1. **The GPU's own signal is slow on the GB10.** A blocking-sync event's
   wait returned 1.0–1.4 ms after the step's end at the median, a host
   function's futex 1.4–1.7 ms. Queries and a mapped flag see the end
   within microseconds, so detection wants polling.
2. **A host function holds its stream.** The next step, launched while
   the driver's callback thread has not yet run the previous step's host
   function, waited for it: S to GPU start was 114–398 µs with 0.2 ms of
   host work, and ~35 µs with 1.3 ms (the callback thread had run by
   then). It also adds a queue entry per fence (RE-029).
3. **Detection is not the cost; sleeping hops are.** Every configuration
   that detects in microseconds but lets K, S or D sleep pays 100–600 µs
   at each sleeping hop: (c) and (d) cost what the old defaults did.
4. **Only keeping the next thread awake ahead of time removes it.** With
   the relay, every hop is under 2 µs at the median; the remaining ~20 µs
   over the harness's is S's launch (S to GPU start 31–38 µs against
   12–15 µs when every thread spins; not isolated: possibly the core S
   runs on after sleeping, or its caches).
5. **A single expected length fails mixed steps.** With 45 ms and 5 ms
   steps mixed, (e) predicting the shortest recent length gave 920–1,159 µs
   at the median (C spun around 5 ms, then backed off through the long
   steps). Taking each recent length as a likely end fixed it (below).

## The choice (D-094)

(e), without host functions, is the default of the scheduler and the
device lanes (`src/scheduler/services.h` `DeviceSettings`,
`scheduler.h` `SchedulerSettings`, `base/wake.h`):

- The completion lane takes each of its stream's last eight fence
  lengths as a likely end (`base::Expectation`), sleeps until 1 ms before
  the next one a fence has not outlasted (querying at least every 1 ms),
  spins to 1 ms past it, and past the longest backs off (50 µs doubling to
  1 ms). A stream with no history spins its first 1 ms. With no fence it
  sleeps.
- As it starts to spin around a likely end it anticipates the scheduler
  (`CompletionBoard::Anticipate`) and its submission lane until 1 ms past
  it.
- The scheduler polls while anticipated, and after a step (device work,
  under a request's lease or its own) for 1.5 times the recent gap to the
  next step's publication plus its 200 µs window, at most 10 ms
  (`follow_limit`), and has the device lane poll as long.
- The submission lane sleeps on its own wake flag (signalled by `Submit`
  and `Close`) outside its window and anticipations.

Why not the others: (a) and (b) wake 1–2 ms late; (b) also stalls the
next launch and adds stream entries; (c) needs polling anyway (a GPU
write wakes nothing) and adds an entry per fence, for no faster
detection than a query; (d) alone leaves K, S and D asleep. The margins
(1 ms ahead, 1 ms past, 1 ms backstop) cover a timed sleep's wake on this
host (up to ~0.8 ms, RE-017); they are measured choices, not tuned per
model. Completion semantics are unchanged: only a query that sees a fence
complete proves it; an anticipation only says when to poll.

The harness's 100 ms window stays as a labelled diagnostic
(`NodeSettings::poll_window`, `--poll-us` in `jitllm_swap_pairs` and
`jitllm_spec_runner`); by default the harness runs the runtime's wake,
and its driver waits the same way (asleep through most of a step,
spinning around its likely ends, woken early by the step's report).

## After the change

Chain, (e) as built (each recent length a likely end), same scenarios,
three runs:

| Scenario | Harness-polled p50 (cores) | Until now p50 (cores) | (e) p50 | (e) p99 | (e) cores stepping |
| --- | ---: | ---: | ---: | ---: | ---: |
| 45 ms, 0.2 ms host | 8 (3.99) | 218–487 (2.00) | 26–31 | 50–1,122 | 0.12 |
| 40 ms, 1.3 ms host | 8–9 (3.97) | 700–869 (1.96–1.97) | 30–32 | 240–353 | 0.20–0.21 |
| 5 ms, 0.2 ms host | 8 (3.95) | 228–238 (2.01–2.02) | 28 | 48 | 0.95–0.98 |
| 45 ms and 5 ms mixed | 8 (3.99) | 239–676 (1.99–2.00) | 25–34 | 656–1,655 | 0.33–0.35 |

Every configuration was idle at 0.00 cores. The p99s over 99 steps
include the first step, which no history predicts, and in the mixed
scenario the short steps that follow eight long ones (C asleep: its
backstop sees them within ~0.3 ms, p99).

**The runtime path** (node, 45 ms steps, 0.2 ms host work, three runs of
99 steps each):

| Node | Round trip p50 | p99 | Cores stepping | Cores idle |
| --- | ---: | ---: | ---: | ---: |
| Before, harness-polled (100 ms) | 9.1–9.3 µs | 13.3–18.4 µs | 3.99 | 0 |
| Before, runtime defaults (200 µs) | 561–667 µs | 772–776 µs | 2.00 | 0 |
| **After, runtime wake (defaults)** | **25.6–27.8 µs** | **46–132 µs** | **0.11–0.12** | **0** |
| After, 100 ms windows (diagnostic) | 10.6–10.9 µs | 120–151 µs | 2.06 | 0 |

## Re-benchmark: the models with the runtime wake

`spark-b`, 10:31–11:39, the artifacts of the M3 runs (DeepSeek
`8a355bfb…`, Qwen3.8 `c4fb47a9…`, the DSpark drafter `dd2d3f9c…`), the
memory gate before each process (no GPU process, more than 110 GB
`MemAvailable`, load under 6, twice), coarse (D-085): two or three runs of
the decode benches, one of each other; this change on the
lease-per-request tree (`a84146c`), then again after main's Qwen3.8 fast
prefill (`f9a4e0f`), from which Qwen3.8's figures come. "100 ms windows" is the same binary
with `--poll-us 100000`, the old harness's polling (its completion lane
and driver now wait the new way; on a synthetic step it took 10.6–10.9 µs
at the median against the old harness's 9.1–9.3), as a same-session
reference. Details and raw outputs: [swap](../fast-swap/swap.md#with-the-runtime-wake),
[graphs](../fast-swap/graphs.md), [dspark](../dspark/README.md#performance-and-memory).

| Measure | Runtime wake | 100 ms windows | Before (harness-polled unless noted) | Reference |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek decode, graphs, a request, tok/s | **20.42; 20.50; 20.48** | 20.32 | 20.34–20.46 | llama.cpp tg64 20.01 (fusion off) / 20.43 (on), same session: **1.020–1.024× / 0.999–1.003×** |
| DeepSeek's round trip a step | 0.033–0.046 ms | 0.017 ms | 0.009–0.013 ms | |
| Qwen3.8 decode, a request, tok/s (`f9a4e0f`) | **24.09; 23.85** | 24.03 | 23.71–23.81 | Mia's vLLM, speculation off, 25.12–25.33: **0.94–0.96×** |
| Qwen3.8's round trip a step | 0.029–0.033 ms | 0.015 ms | 0.009–0.012 ms | |
| DeepSeek with DSpark, `prose` / `code`, tok/s (median of three) | **29.67 / 30.85** | 29.63 / 30.30 | 27.94 / 29.33 (lanes at 200 µs, a lease per step) | llama.cpp with the same drafter 30.80 / 31.94: **0.96× / 0.97×** |
| DeepSeek prefill, 8,192 tokens | 27.37–27.45 s | 27.56 s | 27.34 s | |
| Qwen3.8 prefill, 8,192 tokens (`f9a4e0f`) | 6.84 s | | 8.10 s (before the fast prefill; lanes at 200 µs, a lease per step) | |
| DeepSeek ↔ Qwen3.8 swaps | 7.06–8.81 s | 7.40–8.80 s | 6.86–8.89 s (lanes at 200 µs) | |

A step with a lease of its own (the per-step arms, M7's path for routed
experts) first paid more with the wake than polled (Qwen3.8's round trip
1.65–2.04 ms against 1.05): its closure is walked and leased before the
submission, by which time the device lane's anticipation had lapsed. The
follow window was then extended from a request's steps to every step
(device work through a task), measuring the gap to the next step's
publication, walk included; a confirming run (11:19–11:24) gave 1.00 ms
(Qwen3.8) and 1.32 ms (DeepSeek, graphs), against 1.05 and 1.51 with the
100 ms windows.

So with the runtime's own wake the models decode as fast as they did
harness-polled: the 0.01–0.03 ms a step it adds is less than the device's
run-to-run spread (DeepSeek's step 48.5–48.9 ms, Qwen3.8's 40.2–40.9).
DSpark gained 5–6% over its first figures (lanes at 200 µs, a lease per
step), mostly from running each generation as a request (D-093), which
the spec runner now does: per step the lease alone cost DeepSeek
1.3–2.6 ms, against the old windows' 0.26–0.65 ms of wakeups that the
wake removes, and the rest of the 5.4 ms a step is not isolated; against
the 100 ms windows in the same session the wake changed nothing
measurable. Every check stayed
exact: every bench pass equal to its warm-up, every DSpark greedy check,
every swap's state digest and continued steps.

## Tests

- `unit.WakeFlag.*`, `unit.Expectation.*` (`lanes_test`): an
  anticipation only raises the deadline and wakes the owner; `WaitUntil`
  returns for a signal or at its deadline; four publishers that
  anticipate between publications lose no wakeup of an owner that polls
  while anticipated and sleeps otherwise; the likely ends.
- `unit.DeviceWakeTest.*`, `unit.DeviceWakeShutdown.*` (`wake_test`,
  fakes, both device lanes on their own threads): fences that end on
  time, early, late and at once are all seen with their proof; one ending
  at once when 100 ms is expected is seen within 50 ms (the backstop, not
  the expectation), and ones far longer than every recent length within
  20 ms of completing (median); a query whose outcome is unknown, or that
  keeps being refused, while the lane sleeps is published unproven long
  before the expected end; the owner and the submission lane are told to poll
  before an expected end, and not at the launch; a sleeping submission
  lane wakes for each command (median under 50 ms, the tick being 100);
  four submitters lose no command or fence; closing wakes sleeping lanes;
  a close with a fence out waits for it, however late.
- `unit.HeldLeaseTest.AStepsEndHasTheDeviceLanePollForTheNext`: a
  request's step ending has the device lane poll for the next.
- `unit.CudaPagedNodeTest.RequestStepsSleepBetweenAndLoseNoCompletion`
  (GB10): a request's steps gated on a host flag the test opens on time,
  early, late or at once all complete; before a step's expected end the
  scheduler and the device lane are told to poll, after the completion
  lane slept through most of it; stepping costs under a core, and nothing
  spins once the request has ended.
- The death tests of the lanes' and the scheduler's settings cover the
  new bounds.
- Under ThreadSanitizer (`spark-native` with `JITLLM_SANITIZE=thread` on
  `spark-b`): `wake_test` and `held_lease_test` three times each, and the
  scheduler, page-in, lanes and acquisition tests, with no report.

## Limits

- One host, a shared one; three runs per configuration; D-085's coarse
  comparison. Synthetic steps in the chain and node modes.
- The relay needs history: a stream's first fences, and a step shorter
  than any of the last eight, pay a sleeping hop or two (up to ~1 ms).
- Short steps cost more CPU: the spins around each likely end are fixed
  at 1 ms, so 5 ms steps keep about a core busy in all (idle is still
  zero).
- The ~20 µs left over the harness's polling is S's launch after a spin
  that began only 1 ms before; not isolated.
- Not tried: a PM QoS request (`/dev/cpu_dma_latency`, root) that would
  keep cores out of deep idle while a request runs, making sleep cheap;
  it is a system setting, the owner's call.

## Reproduce

On `spark-b`, from the tree's `spark-native` build:

```bash
build/spark-native/benchmarks/jitllm_wake_bench chain --steps 100 --step-us 45000 --host-us 200 --repeats 3
```

```bash
build/spark-native/benchmarks/jitllm_wake_bench node --steps 100 --repeats 3 [--poll-us 100000]
```

`--mix` mixes in 1/9-length steps; `--only PREFIX,...` picks
configurations by name.
