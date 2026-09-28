<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# The swap path: DeepSeek V4 Flash on the paged node, full swaps A→B→A (M3)

M3's swap path and swap runner ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)):
DeepSeek V4 Flash 0731 runs as device jobs over leased closures on the
paged node (D-086), paged into device VMM through the landing zone
(D-081); a full swap evicts the outgoing model, spilling its conversation
state, and hands its backing to the incoming one (D-033); the zone's
copies have a lane of their own (RE-029). The swap runner drives A→B→A in
one process and times each part.

## What was built

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

## Results (`spark-b`, 2026-09-28)

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
A→B pages in 1.26 GB where Qwen3.8 would page in its whole weights, and
the Qwen3.8 and Qwen-Image pairs are not yet measured. The baselines' numbers stand beside them:
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
  boundary in a gap; the layouts the runner cannot page are refused.

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

## Limits

- One process, one run per configuration; timings are single samples on
  `spark-b` (D-085's coarse comparison), not distributions.
- B is the 0.5B FP16 fixture, a stand-in until Qwen3.8 runs: B's own
  page-in is 1.26 GB. The Qwen3.8 and Qwen-Image pairs, M3's actual
  swaps, are not measured here.
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
