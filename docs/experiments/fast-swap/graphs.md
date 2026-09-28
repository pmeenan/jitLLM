<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Decode graphs: DeepSeek V4 Flash's decode steps as CUDA graphs (M3)

M3's "CUDA graphs for decode" ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)),
first for DeepSeek V4 Flash on the paged node ([swap](swap.md)). The
decision is D-090; the relocation proof it rests on is below.

## The relocation proof

A captured graph holds device addresses in its kernels' parameters and
copies. It may replay after a swap only if every address it names holds
the same thing again. What a DeepSeek decode graph names, and why each
stays put:

| Memory | Where it lives | Why a swap brings it back at the same address |
| --- | --- | --- |
| Weights (dense regions, expert-slab pages) | one device reservation, made at setup and freed only in `Release` | each extent's source names a `BackingPlace` (reservation, offset); a load maps whatever backing it takes, created or handed off (D-033), at that place (`services.cc` `kMap`: `Map(reservation, offset, backing)`), and copies the landed read to the source's destination or pieces, all inside the place |
| State (window, compressed and indexer caches, rings) | its own reservation, the same way, written back and restored through the zone | the same: its write-back source names its place |
| Activations, GGML pool | the node's shared workspace, mapped at setup, unmapped at teardown | no managed backing: no eviction unmaps it |
| cuBLAS workspace, input and logits staging | the model's, mapped (or `cudaMallocHost`) at setup | the same |
| Landing zone | host VMM | never in a graph: only the copy lane touches it |

The one way the scheduler could move an extent is a new source with
another place for a nonresident extent (BP-P5's relocation). D-090 closes
it: a model pins its places when it registers them
(`Scheduler::PinPlaces`), and `SetSource` refuses any source that puts a
pinned extent's contents anywhere else (`SamePlace`), resident or not.
The DeepSeek runner pins its 46,453 weight and state extents at
registration and, after every swap back in the swap runner, checks that
each is still pinned at its registered place (`CheckPlaces`; it drops every
graph and fails if not).

What varies between decode steps of one shape (the token, its position,
the cache cells, the masks and the compressors' indices) is the inputs'
data, which the host stages at the same offsets of the pinned staging and
the graph copies in. GGML's launchers derive every launch parameter from
shapes, strides, addresses and operation parameters, which the plan fixes
per shape, and the pool hands out the same blocks in the same order from
an empty stack each run. So no kernel node needs
`cudaGraphExecKernelNodeSetParams`; each replay checks that the staging
layout is the captured one.

## What was built

- **Capture through the launch context** (`kernels/ggml/launch.h`):
  `LaunchContext::Capture(record)` begins a thread-local capture on the
  context's stream, runs `record` (its runs, and copies it queues on the
  same stream), ends the capture, instantiates and uploads the graph;
  `Launch(graph)` queues one replay. A refused capture queues nothing and
  leaves the context as it was (not faulted, not capturing); only a fault
  the stream or provider reports faults it. A graph replays only on the
  context that captured it (by an identity never reused in the process,
  not the context's address).
- **Pinned places** (`scheduler.h`): `PinPlaces`, `UnpinPlaces`,
  `PlacePinned`, `SourceOf` and `SamePlace`.
- **The runner** (`benchmarks/dsv4_runner.h`): per chunk shape, a one-row
  chunk whose shape ran once launch by launch is captured (the ~35 input
  copies, the plan's 4,972 steps and the logits copy: 5,920 graph nodes),
  and later chunks of the shape replay it. Prefill chunks run launch by
  launch. Graphs live and die with their plans (`DropPlans` for first use).
- **The swap runner** (`--graphs on|off`, on by default; `--bench N`): the
  control runs launch by launch, so every continued step after a swap is a
  graph compared with launch-by-launch logits; each swap reports how A's
  first token ran; `--bench` is llama-bench's tg-N (one-token steps from an
  empty context), launch by launch and replayed, plus the graph replayed
  back to back in one job (the step's device time alone).

## Results (`spark-b`, 2026-09-28)

GB10, driver 580.178.04, `spark-native`, `CUDA_DISABLE_PTX_JIT=1`,
artifact `8a355bfb…`. Each run started once `spark-b` had no GPU process
and more than 110 GB `MemAvailable` (`gate.sh`); the Qwen3.8 slice used
the GPU between runs. Raw outputs in `~/.local/share/jitllm/m3graph-20260928/`
(`pb-1`, `bench-2`, `swap-g1`, `llb-*`).

**Bit-identity.**

| Check | Result |
| --- | --- |
| The 8 dsv4-native prompts, prefill + 31 steps, graphs on, against the resident harness (itself bit-identical to llama.cpp with fusion off) | 0 of 8 × 32 × 129,280 logits differ; 246 of the 248 decode steps replayed, one captured, one launch by launch |
| Bench, 64 steps from BOS, three runs: every replayed pass against the launch-by-launch warm-up | 0 steps differ, in all 12 replayed passes |
| A resumed after B (8,192 context), replayed, against the launch-by-launch control | every continued step bit-identical, first use and prepared; the prepared return's first token replayed a graph captured before the swap |
| Reloads (evict everything, reload) | exact; places pinned and unchanged after every swap back |

After review's changes (the graph cap, the owner's identity), the 8
prompts were run once more with graphs on: again 0 logits differ, 246
decode steps replayed (`~/.local/share/jitllm/m3graph-review/prompts-1`).

**Decode speed** (tok/s; 64 one-token steps from an empty context; means
of three passes per run, three runs):

| | tok/s | per token |
| --- | ---: | ---: |
| jitLLM, launch by launch (paged node) | 18.13–18.82 | 53.1–55.2 ms |
| jitLLM, graphs (paged node) | 19.05–19.53 (1.04–1.05×) | 51.2–52.5 ms |
| jitLLM, one graph replayed back to back (device time; two runs) | 20.58–20.70 | 48.3–48.6 ms |
| jitLLM resident harness, launch by launch ([dsv4-native](../dsv4-native/README.md)) | 19.41 | 51.5 ms |
| llama.cpp, fusion and CUDA graphs on (llama-bench's defaults) | 20.62 ± 0.09 | 48.5 ms |
| llama.cpp, fusion off, CUDA graphs on | 20.04 ± 0.06 | 49.9 ms |
| llama.cpp, fusion off, CUDA graphs off | 19.86 ± 0.01 | 50.4 ms |

llama.cpp is `llama-bench -ngl 99 -fa on -p 512 -n 64 -r 3` in the pinned
image (`run_oracle.sh` `bench`, `bench-unfused`, `bench-unfused-nographs`),
its tg64. jitLLM's plan is unfused, so llama.cpp's like-for-like arm is
fusion off, graphs on: jitLLM with graphs is at 0.95–0.97× of it;
launch by launch, jitLLM was at 0.91–0.95× of llama.cpp launch by launch.
Graphs also help llama.cpp little (0.9%).

**Where the rest goes.** The job's host time per step falls from 41.5–41.9
ms (the ~4,972 launches, mostly waiting for room in a full stream, RE-029)
to 0.13–0.16 ms (the inputs built and staged, one launch). The step's
device time alone is 48.3–48.6 ms, 1.3–1.6 ms below llama.cpp's whole
unfused step. The 3.0–4.2 ms between it and a replayed step's wall time is the
paged node's round trip per chunk, which graphs do not touch: posting the
job, the scheduler materializing and leasing the ~46,500-extent closure,
the fence, and releasing the lease, with the GPU idle meanwhile. That, not
launching, is now the gap to llama.cpp (not decomposed further).

**Capture cost and memory.** One capture per decode shape: 5.5–7.0 ms
capturing and 18.0–18.5 ms instantiating and uploading a 5,920-node graph
(about 25 ms in all, once per shape and process). All 64 bench steps and
the prompts' decode steps below 256 positions share one shape; the swap
run captured four shapes in all (positions below 256, and around 8,192).
Graph memory, by the drop in the device's free memory across a capture
(system memory on the GB10, so anything else allocating counts too): 28.6,
44 (177 MiB for 4) and 67.8 MiB per graph in three runs: tens of MiB per
shape, coarse. Peak memory by `MemAvailable` stayed 91.9–94.9 GiB.

**Across swaps** (`swap-g1`, handoff on; seconds, from the swap request):

| Swap | Total | First token | How A's first token ran |
| --- | ---: | ---: | --- |
| B→A, first use (8K context) | 7.457 | 0.142 | launch by launch (plans dropped: nothing prepared) |
| B→A, prepared (8K context) | 7.377 | 0.060 | replayed, captured before the swap |
| B→A, prepared (0 context) | 7.542 | 0.251 | a 16-token prefill, launch by launch |
| DeepSeek reload 1, 2 | 9.347, 9.100 | 0.112, 0.082 | launch by launch, then captured |

So a previously prepared swap replays A's graphs without capturing again,
and a first-use one pays nothing for graphs on its first token (the shape
runs once launch by launch, then is captured on its second step, about 25
ms). The prepared first token is 0.060 s against 0.064 s launch by launch
in [swap.md](swap.md)'s run: one decode step, within noise, so the swap
table there stands; B→A is page-in bound (7.2 s of 7.4 s). This run
shared `spark-b` with other work between phases (its launch-by-launch
bench was 3–4% slower than `pb-1`'s), so its totals are within 3% of
swap.md's, not a replacement for them.

## Tests

- `unit.VmmWork/PageInTest.AfterASwapEveryPinnedExtentIsBackAtItsAddress/*`
  (fakes): A→B→A with the handoff maps each of A's extents at its own
  place and nowhere else, every byte at the captured address; a pinned
  place cannot move, resident or not (resident, not even to another
  destination within the same backing place; a new file range for the
  same place is no move); pins count and need a source; once unpinned, a relocation
  maps the extent elsewhere and `SamePlace` against the captured source
  fails: the check that detects a changed address.
- `unit.PlaceTest.SamePlaceComparesWhereContentsGoNotWhereTheyComeFrom`.
- `unit.CudaGraphTest.*` (GB10), a small GGML plan on the paged node
  (get_rows, unfused RMSNorm-mul, a Q8_0 MMVQ product with pool scratch, an
  F16 MMVF product, a product over the cache, RoPE and a set_rows into an
  F16 cache), token, position and cell varying as data:
  - a replay equals the step launched launch by launch, bit for bit, and
    the cache it writes;
  - after A→B→A twice (the cache written back and restored; the second
    time without the handoff, so A's weights come back over new backing),
    A is at its pinned places and the graph captured before, never
    captured again, replays bit for bit what a never-swapped control
    computes;
  - refused cleanly: a synchronization inside a capture, a record that
    fails, a nested capture (the outer one still succeeds), a replay on
    another context, a capture on a faulted context; the context is neither
    faulted nor capturing after the first three and runs as before;
  - RE-029: a stream took 1,020 replays of a 1,500-kernel graph behind a
    closed gate before a launch blocked.

## Judgement calls

- **Pin places rather than update graphs.** Re-pointing a graph after a
  relocation would mean decoding each GGML launcher's parameter struct; a
  model whose places never move needs none of it, and the scheduler can
  enforce that they do not (D-090).
- **Data, not parameters.** The input copies are inside the graph, from
  fixed staging offsets that each replay checks, so a decode step is one
  launch.
- **Capture on a shape's second step.** Its first runs launch by launch:
  one-time work (lazy module loading, the kernels' shared-memory limits)
  stays out of the capture, and a first-use swap's first token costs
  nothing extra. Prefill chunks are never captured: few steps share a
  prefill shape, and a 512-row chunk is device-bound.
- **Thread-local capture mode,** so that an uncapturable call on the lane's
  thread fails the capture (refused, shape stays launch by launch) instead
  of running, while the other lanes' threads are unaffected.
- **The control runs launch by launch** in the swap runner, so each cycle's
  continued steps compare graphs with launches.

## Limits

- One process, single samples per configuration (three bench runs), on a
  shared `spark-b`; D-085's coarse comparison.
- Graph memory is read from free memory, not itemized, and is not in the
  catalog's account (it is driver memory, like handles: D-086's `F`). It
  is bounded: the runner keeps at most 8 graphs (`kMaxGraphs`; a capture
  past it destroys the oldest, counted as `graphs_dropped`), so at the
  sizes above at most about 0.5 GiB. A decode shape's KV lengths are
  padded to 256, so a long generation meets a new shape at least every
  256 positions and captures it (about 25 ms each). The cap was added in review, after the runs above, none
  of which kept more than four graphs.
- Pins are never released in the harness: the runner destroys its graphs
  in `Release`, after the scheduler has stopped, and the pins go with the
  scheduler. A runtime that unpins (to relocate or remove a model) must
  destroy every graph naming those extents first; the scheduler cannot
  see graphs (D-090).
- Only DeepSeek so far; Qwen3.8's decode graphs follow its slice.
- The paged node's per-chunk round trip (3.0–4.2 ms) is measured, not
  broken down or reduced.
