<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note. When a milestone exits, its section shrinks
to a short summary and its task-by-task history moves to a record beside this
file ([M0](m0-record.md), [M1](m1-record.md)).

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `done`

Ran 2026-09-20 to 2026-09-23 and exited on the owner's approval of the plan.
M0 turned the [design brief](ideation.md) into the vision, the triaged feature
matrix, the approved architecture and decisions D-001–D-069, and gathered the
evidence they rest on: toolchain, VMM, I/O and interconnect spikes on the
Sparks; the llama.cpp, EXL3, image and MiMo reference runs; the measured A→B→A
reference cycle; the paging-feasibility study; and the v0 artifact layout
study. The [M0 record](m0-record.md) keeps each task's outcome, evidence links
and caveats. Everything M0 left open is owned by the milestone exits below.

## Milestone ladder

Rewritten from the provisional ladder at the end of M0 (2026-09-23; see the
[M0 record](m0-record.md)) and ordered by risk: the substrate; the resource core with the backend proof; one
resident model end to end; the first useful product at M4 (A→B→A with
retention); configured placement at M4a; demand-paged MoE with the first
daily-driver models at M5; sharding at M6; performance and the new decoding
modes at M7; and the remaining product scope with the first tagged release at
M8. Each milestone leaves a usable, testable result. None has a promised date.

| Milestone | Result | Needs |
| --- | --- | --- |
| M1 Bootstrap | Pinned SDK, builds, local check gate, package skeleton, confined-job proof | M0 (done) |
| M2 Resource core | Catalog, admission and leases on a fake backend and a Spark; the backend proof settles the operation contract | M1 |
| M3 One resident model | Import, native GGML/EXL3 serving and the three client protocols on the small fixtures | M2 |
| M4 First useful product | A→B→A with partial retention; switching policy chosen from measurement | M3 |
| M4a Configured placement | Conductor, enrolled nodes, whole-model placement and routing | M4 |
| M5 Demand-paged MoE | Exact expert paging; Gemma 4 and Ornith as daily drivers with reasoning and constrained output | M4 |
| M6 Sharding | A flagship model sharded across both Sparks | M4a, M5 |
| M7 Performance | D-036's benefit target on a library larger than memory; speculative and diffusion decoding | M5; M6 before exit |
| M8 Product and release | Dashboard, remaining API scope, signed apt repository, first tagged 0.x release | M7, for the release |

How to read the milestones below:

- **Exit criteria are the contract.** Scope lists say what a milestone
  builds; its entry work refines them into tasks. An exit criterion changes
  here, visibly and with the owner, never by drift.
- **Detail lives in the linked designs.** Where a gate cites a matrix or an
  acceptance section, every row that section assigns to the milestone is
  part of the gate.
- **Evidence rules apply to every gate.** Thresholds are declared before the
  measurement they judge, and inconclusive comparisons do not pass (D-036).
  Every result names the check tiers and hosts that produced it (D-061,
  [workflow.md](workflow.md)). No performance gate is met by giving up
  correctness.
- **Heavy path.** Work in a [blast-radius area](workflow.md#blast-radius-changes-get-the-heavy-path-by-default)
  passes its adversarial challenge before the milestone exits.
- **Support is earned per checkpoint.** From M3 the support matrix records
  what each exit validated; a milestone exit names its configurations.
- **Release.** The first tagged 0.x release (an owner-signed tag, D-062) is
  cut at M8's exit, after every milestone has exited (owner, 2026-09-23). The
  repository is already public (D-061); until then it carries only `-dev`
  builds.

## M1 — Bootstrap  `done`

Ran 2026-09-23 to 2026-09-24 and exited on the owner's word. M1 built the
pinned SDK and its reference container (D-070), the CMake presets and build
tasks, the source lock with GoogleTest and toml++ (D-057, D-073), the local
check gate (`check`, `check:full`, `check:spark`; D-061), license and
provenance records (D-071), versioning (D-062), `jitllm doctor` (D-072), the
node configuration and storage-role checks (D-073), and the arm64 package
with `jitllm-runtime`, `jitllm.service`, crash handling and confined jobs in
delegated cgroups (D-074), validated on `spark`. The
[M1 record](m1-record.md) keeps each item's outcome, verification and
hand-offs. What it handed on: the GPU, VMM, I/O and ARM stress suites in
`check:spark`, a system-call filter with io_uring, and a single reaper for jobs
started from other threads (M2); front-door, TLS and switching-policy keys
(M3, M4); the importer on the confined-job mechanism and spill file names
(M3); cluster-member checks of credential files and enrollment records
(M4a); drain-before-restart upgrades and the apt repository (M8).

## M2 — Resource core and backend proof  `in progress`

Goal: the node-wide catalog, ledgers, reservation and lease state machine and
D-048's task lanes, deterministic on a fake backend and real on a Spark.
Alongside them, the early backend integration proof runs real GGML FP16 and
EXL3 kernels on jitLLM-owned memory and settles the internal operation
contract. M2 stays small (owner, 2026-09-23), so it adds no tokenizer, C++
importer, HTTP or new model shapes. Future shapes appear only as fake-provider scenarios,
and the concrete interfaces come from the real GGML/EXL3 proof.

**Entry:** M1 exit. The prerequisites the provisional ladder listed (paging
feasibility, the measured reference cycle, D-036's criteria and D-050's
reservation policy) were recorded in M0.

**Scope:**

- [x] **Catalog and ledgers** (D-006, D-007): typed IDs and generations,
      dependency closures, shared extents charged once, the memory-domain
      key, separate commitment and occupancy ledgers, unknown allocations
      non-evictable, and the deterministic LRU
      [victim-selection baseline](architecture.md#victim-selection-initial-baseline).
      *Landed:* `base/` gains generation-checked identities (`ids.h`),
      checked byte counts (`bytes.h`) and fatal invariant checks
      (`check.h`). `src/catalog/` holds extents with the architecture's
      state machine: one load or eviction at a time, each named by a
      ticket that alone can finish it (joining a load in progress is the
      resource service's job); failed loads released or quarantined;
      evictions that exclude new leases and can be cancelled or
      quarantined. It also holds resources placed in byte ranges of one
      domain's extents, closures that charge each extent once and record
      its content generation, all-or-none leases refused for changed
      contents, registrations, 64-bit backing and content generations, and
      per-domain occupancy by state bucket and class. Descriptors are
      range-checked and a domain's total size bounds every sum. Unknown
      allocations must be pinned and are never evictable; only their owner
      releases them. Recovery is fixed per extent: preserved state is
      evictable only after it is deliberately invalidated, and only its
      sole lease-holding writer, with no live registrations, replaces it.
      Every load checks occupancy
      against B. `src/memory/` holds the commitment ledger, keyed by domain
      (D-050's inequality over the active cohort and any promised
      resumption, with checked arithmetic, atomic envelope replacement,
      retirement never refused, no budget below outstanding claims), the
      deterministic LRU victim baseline (discarded contents first, then
      least recent actual use, ties by content and identity, protected
      extents never chosen, whole-extent credit), and materialization
      planning (only the missing dependencies, victims only for the
      shortfall against B). Tests: `unit.CatalogTest.*`,
      `unit.CommitmentLedger.*`, `unit.VictimTest.*`,
      `unit.MaterializeTest.*` and the base tests. Retained entries and
      `M_state` join the victim order with the retention cache;
      suballocation within extents comes with state blocks.
- [x] **Admission and leases:** D-050's guarantee for one active request at
      a time (concurrency arrives in M4), with D-069's pause rules and
      guards.
      *Landed:* `src/scheduler/admission` admits, queues or refuses each
      request against one domain's ledger. It holds the execution slot
      from a request's first phase to its retirement and admits joiners
      only through the cohort check. D-069's three policies are there with
      their guards:
      - the minimum run, the pause cap, and one open pause with one
        substitute;
      - resume-next;
      - the promised resumption, held in the ledger so that no change
        breaks it: others' changes are deferred, the substitute's own are
        refused;
      - deadline checks that count the pause point;
      - aging.
      Queued requests that can no longer fit, have waited past the
      configured limit, or would miss their deadline are refused
      explicitly, and every capacity change drains the queue. Leases are
      the catalog's (above). Tests: `unit.Admission.*`, including a
      randomized progress check under all three policies. A pause with
      several paused cohort peers waits for M4's concurrency.
- [x] **Task lanes** (D-048): scheduler, storage, device submission, device
      completion and CPU workers, with bounded queues; measure worker
      counts, queue sizes and wakeup and polling behavior. Real
      multithreaded lost-wakeup and memory-ordering tests, with ARM stress
      on a Spark; the deterministic simulator does not prove them
      ([async model](async-model.md)).
      *Landed:* first the primitives. `base/` gains a bounded queue with a
      cleanup reserve and a coalesced wake flag. `src/scheduler/` gains:
      - the completion board: per-operation mailboxes with
        generation-tagged acceptance and terminal words, closed only once
        reconciled;
      - lanes that drain on close;
      - the task table, whose operations reserve task lifetime before
        provider handoff and whose trees unwind only when drained, with
        its ready queue.

      Threaded tests (`unit.BoundedQueue.*`, `unit.WakeFlag.*`,
      `unit.CompletionBoard.*`, `unit.Lane.*`) pass under ThreadSanitizer
      on `spark` in `check:spark`, alongside `unit.BoardTest.*`,
      `unit.TaskTable.*` and `unit.ReadyQueue.*`. The
      [measurements](experiments/task-lanes/README.md) found two things
      (RE-017):
      - on the Spark, a sleeping owner takes hundreds of microseconds to
        wake, so the scheduler polls while a critical-path completion is
        imminent;
      - a lane queue should have at most four workers.

      Then the lanes and the loop. `services.h` wires the storage lane
      (whole reads over any `Storage`, one per operation), the device
      service's submission and completion lanes (copies on owned streams,
      fences queried independently of submission, released only between
      submission calls) and the CPU worker lane's handler to the
      providers, each reporting through the board. `scheduler.h` is the
      scheduler thread: each turn takes bounded batches of controls, board
      observations and ready tasks, retries what a full lane refused, and
      then polls within a window while a critical operation is in flight,
      or sleeps on the wake flag. Tasks are explicit state machines that
      materialize closures, submit device and CPU work, and spawn
      children:
      - page-ins are the scheduler's, one per extent with bounded waiters;
      - work is prepared under the owner (leases, task hold, record)
        before publication, and its leases retire only on proof of no
        further access;
      - unproven or contradictory results quarantine what they hold and
        stop admission;
      - cancellation rolls back only what no lane took;
      - shutdown drains, or faults on quarantined work.

      Tests:
      - `unit.SchedulerTest.*` on the fakes, in every profile: cancellation
        through submission, completion and memory retirement (extent state,
        backing generation, leases and occupancy afterwards), shared
        page-ins, full queues during cleanup, late, duplicate and
        contradictory observations, unknown fences and unproven reads,
        repeated cancellations sharing one queued intent, a tag started
        again while an earlier start drains, a storage lane that makes
        room without publishing, exhausted operation identities (which
        stop admission; mailboxes and task slots are reused in rotation,
        `unit.CompletionBoard.*`, `unit.TaskTable.*`), fence queries and
        releases the provider keeps refusing, task-tree unwind and
        shutdown; and
        `unit.StorageLaneTest.*` (io_uring, skipped under qemu-user),
        where a cancellation reaches a read that never completes.
      - Its threaded cases, `TheOwnerLosesNoWakeupAmongManyPublishers`,
        `ReusedTagsLoseNoCancellationAmongManyThreads` and
        `ThreadedLanesKeepEveryHandoffOrdered`, run the owner with an
        hour-long tick, so a lost wakeup or cancellation hangs, and send
        data from storage to device to CPU worker with sources evicted (in
        the catalog only: backing release is not wired) and reloaded and
        requests cancelled. `unit.UringTest.NoWakeIsLostAmongManyProducers`
        does the same for the storage lane's eventfd wake.
      - `unit.CudaLanes.*` (`gpu`) do the same over io_uring, CUDA host
        VMM and CUDA streams, and cancel a read and device work in flight.

      All pass on a Spark (`spark-b`) in `check:spark` under the cross, ASan and TSan
      builds, and the fake cases pass on the workstation and under
      qemu-user in `check`. Mapping and backing release on the device
      lane arrived with the D-081 page-in path (backend proof, P2), and
      write-back of live state with P4 (D-086). Not
      yet wired: victim selection on a miss, spill as retention (M4), and
      admission driving task starts.
- [x] **Providers** (D-026): device-memory, device-execution and storage
      interfaces, each with a deterministic poison-filling fake; the CUDA VMM
      provider at D-033's 2 MiB extents; direct-I/O reads into host VMM
      (D-034) that handle short transfers, alignment, retries, cancellation
      and duplicate-load coalescing explicitly, with storage queue depths
      and run sizes measured (M4 tunes them against spill write-back).
      *Landed:* `src/providers/` gains three interfaces.
      - **Device memory** (`device_memory.h`): `VmmProvider` enforces the
        rules every implementation shares. Backing is mapped whole into
        holes, access is explicit and set per run of one backing kind,
        release and free wait for unmap, and an unknown outcome leaves what
        it touched undetermined, never retried.
      - **Device execution** (`device_execution.h`): streams, copies, waits,
        and fences queried without blocking. A fence is released only once
        seen complete, and a stream is destroyed only when all its work is
        fenced. It is safe for a submission lane and a completion lane
        together.
      - **Storage** (`storage.h`): submissions resolve as not started,
        accepted or unknown, and cancellation never reports its own result.
        Implemented over a raw io_uring ring (`platform/io_uring.h`, no
        library).

      `direct_reader.h` gives whole reads over storage: alignment checks,
      short-transfer continuation, bounded retries, coalesced duplicate
      reads with bounded waiters, and a last-waiter cancel that still
      drains.

      Every build has deterministic fakes (`providers/fake/`), poison-filled
      memory where absent backing faults, and scripted completions and
      failures. The CUDA implementations (`providers/cuda/`) use the
      driver API, and only errors the driver reports before acting count as
      known.

      Tests: `unit.DeviceMemoryTest.*`, `unit.DeviceExecutionTest.*`,
      `unit.ReaderTest.*` and `unit.UringTest.*`. On `spark` (`gpu`),
      `unit.CudaDeviceMemory.*` and `unit.CudaDeviceExecution.*` cover io_uring
      reads landing in host VMM backing and host-device-host copies on
      jitLLM's streams.

      The [storage measurement](experiments/storage-queue/README.md) reaches
      about 14.9 GB/s with 8 MiB in flight as 2 MiB requests, without
      registered buffers. The workstation's btrfs quietly buffers direct I/O
      (RE-018).
- [ ] **Backend integration proof** ([scope](backend-proof.md), stages
      P0–P6). The frozen bounds and performance protocol are approved at
      P0 (or, since D-079, pre-registered), before any native output is
      seen. The GGML subset and the
      ExLlamaV3 files enter through D-057 with reviewed patches, and GGML's
      C++ CUDA launchers are checked for throws at the pinned revision
      (D-066). The FP16 control and both EXL3 fixtures run from v0 prepared
      artifacts built by M0's prototype. D-053 dispatch records each
      operation's K-C or K-L choice, selects between at least two
      implementations of one operation by plan, and alternates FP16 and
      EXL3 in one process. The five-rung oracle ladder applies throughout.
      The GEMV provenance gate closed on 2026-09-27 (D-080): GEMV is
      core-eligible, and BP-F2 is gated against upstream with GEMV on
      (EXL3-O), its cases fixed at P3 entry. Weights and state run from
      device VMM, loaded through a host-VMM landing zone (D-081).

      *P0 measured and partly approved:*
      - Both toolchain bridges reproduce their references bit for bit.
      - Every reference arm repeats and restores exactly.
      - ExLlamaV3's tuned launch plans are decoded and frozen.
      - A second pass, after review and challenge, added: every prefill
        row, a cuBLAS 13.8 substitution arm, an FP64 oracle with per-layer
        captures, the FP16 bridge's recorded executed plan and pool peaks,
        and A/A timing controls
        ([report](experiments/backend-proof-p0/README.md)).

      The owner approved on 2026-09-26 the
      [numerical profiles](backend-proof.md#p0-declarations) and the exact
      tier FP16 work needs: the FP16 gate with its recorded-plan match,
      rungs 4 and 5, and the EXL3 packed linears. Native FP16 work (P1, P2)
      can start.

      On 2026-09-26 the owner also approved the EXL3 parts:
      - the Tier C bound, calibrated on 15 legitimate arms and 7 faults;
      - reconstruction-GEMM exactness, through pinned cuBLASLt algorithms
        that a probe shows bit-exact;
      - the kernel-timing rule, validated on holdout sessions;
      - the native EXL3 operation plan and its machine-checkable record
        (GGML's F32 vector attention and F32 RoPE, from a measured
        operation study);
      - the operation-level gate;
      - the persistent-workspace limit.

      Pre-registered at P3 entry (D-079, 2026-09-27), once the ExLlamaV3
      port existed: BP-F2's reference arm (EXL3-O with GEMV on since
      D-080, fused gate/up cases, ExLlamaV3's bias add, one tuning cache,
      and the port's SASS match), and the EXL3 phase memory limits,
      tightened against native's buffer plan (since D-085, the plan's
      declared budget; a native EXL3 run is judged by the coarse memory
      check against EXL3-O, and BP-F2 does not run).

      The rest (the FP16 memory limits, BP-F1's calibration, the
      declared-departure contingency, the P3-entry items, the
      retained-backing criteria and M2's acceptance of the GEMM-only gap)
      the owner delegated on 2026-09-26 with
      their defaults (D-079): each is pre-registered in backend-proof.md
      before the native output it governs. cuBLAS links dynamically from
      the SDK (D-076).

      *P1 started:*
      - **GGML enters the build** as the locked llama.cpp archive, narrowed
        by `archive.keep`, with two patches (D-077). jitLLM compiles only
        the selected files and never `ggml-cuda.cu`, supplying the five
        symbols its launchers need and the context destructor.
      - **`src/kernels/ggml/`** holds:
        - tensor descriptors over jitLLM memory, built with GGML's graph
          functions;
        - the K-C launch context: the provider's stream
          (`DeviceExecution::Submission`), a scratch pool over declared
          workspace, and launch errors returned as faults;
        - the first implementations: RMSNorm, fused RMSNorm-mul, add, mul,
          and matrix multiplication through MMVF and MMF. Each refuses what
          its launcher would assert on; the checks run on the host in every
          profile (`unit.GgmlValidateTest.*`).
      - **On `spark`:** `unit.GgmlMemoryTest.*` shows device and host VMM
        bit-identical to cudaMalloc for all of them, and close to a CPU
        reference. `unit.GgmlKernelsTest.*` shows that launches bind to
        the provider's primary context and its stream order, and that a
        launch error comes back as a fault.
      - **cuBLAS handle and workspace injection.** jitLLM creates the
        cuBLAS handle as upstream sets it up (TF32 math, the provider's
        stream, a declared workspace, 32 MiB as upstream's on GB10) and
        lends it to the launch context (`cublas.h`). The handle refuses a
        cuBLAS other than the pinned 13.8.0 (a system one found first on
        the library path) and any of cuBLAS's numerics switches in the
        environment. GGML's cuBLAS matrix multiplication is a recorded
        jitLLM copy (`mul_mat_cublas.cu`). One host plan
        (`CheckMulMatCublas`) fixes its conversions, cuBLAS entry point,
        operand alignment and scratch bound before launch. It takes only
        operands upstream would route to cuBLAS and that clear both
        workspaces. Tests link cuBLAS dynamically (D-076) and load it from
        the build tree.
        - On every profile, `unit.GgmlValidateTest.*` shows the output
          head's plan drawing exactly GGML's pool peaks recorded in P0
          (5,196,288, 9,781,248 and 156,499,968 bytes at 17, 32 and 512
          rows).
        - On `spark`, `unit.GgmlCublasTest.*` and
          `unit.GgmlCublasMemoryTest.*` show:
          - Sgemm, GemmEx, strided batched and pointer-array batched
            products (F16, BF16 and F32 compute; direct, converted and
            gathered operands; grouping within samples; a 4-byte-aligned
            operand) bit-identical in cudaMalloc memory, device VMM and
            host VMM, and close to a CPU reference;
          - each product's scratch peak equal to its plan's bound, and its
            recorded alignment as predicted;
          - the 17- and 32-row peaks drawn on the device;
          - GGML using the lent handle and creating none.
        - The copied `k_compute_batched_ptrs` has the bridge's recorded
          SASS (`37c97848…`, 200 instructions; cuobjdump 13.0.85 on
          `spark`, 2026-09-26).
      - **BP-F1's rule is calibrated and pre-registered** (D-079;
        [report](experiments/backend-proof-p1/README.md)). A harness
        (`benchmarks/ggml_vmm_bench.cc`) times 53 cases derived from the
        FP16 plan (every GGML kernel jitLLM has, at 1, 16, 17 and 512
        rows), verifying each captured launch against the recorded plan,
        with all their memory in one kind per process. Four A/A
        `cudaMalloc` sessions on `spark` gave a median `σ` of 1.23%; the
        holdout declared in advance passed. The calibration's hash is
        registered in backend-proof.md, and the session driver refuses a
        host-VMM arm under any other.
      - **BP-F1 fails: host VMM is slower** (2026-09-27, on `spark`, under
        the committed pre-registration). 41 of 53 cases failed the primary
        session and its mirrored confirmation, and the aggregate failed in
        both. Every matrix product ran 1.10× to 4.9× slower on host VMM (the
        output head at one row 2.14×), with identical launches and outputs
        ([comparison](experiments/backend-proof-p1/README.md#comparison-host-vmm-against-cudamalloc-bp-f1-gated)).
        This reopened D-034. The owner answered with D-081: weights and
        state move to device VMM, and direct reads land in a bounded
        host-VMM zone that the GPU copies from
        ([diagnosis](experiments/host-vmm-diagnosis/README.md), RE-022;
        direct landing in device memory is impossible here, RE-025).
      - **BP-F1 rule v2 (device VMM) is pre-registered** (2026-09-27;
        [report](experiments/backend-proof-p1/README.md#rule-v2-device-vmm-d-081)).
        The harness gained a device-VMM memory kind. Four new A/A
        `cudaMalloc` sessions on `spark` gave a median `σ` of 1.17%, and
        both holdout sessions passed alone. The new harness, the
        unchanged cases and the calibration are registered as "BP-F1 v2".
      - **BP-F1 passes on device VMM** (2026-09-27, on `spark`, under the
        committed pre-registration `72c7c62`). The primary session had two
        cases over `z` (KQV and the k/v projection at 17 rows); the
        mirrored confirmation had none, and the aggregate passed in both
        (`t` 0.58 and −0.36). Every case's ratio to `cudaMalloc` was
        0.957–1.037, with identical launches and outputs
        ([comparison](experiments/backend-proof-p1/README.md#comparison-device-vmm-against-cudamalloc-bp-f1-rule-v2-gated)).
        D-081 stands.

      - **Plan selection between implementations (D-053).** The
        implementation registry (`src/execution/registry.h`) holds each
        compiled implementation with its identity: source, prepared tree
        digest, a build-time digest of the module's own files, SDK, target,
        device architecture, build type and variant. A plan names
        one implementation per operation and records its identity, and the
        plan's identity is a SHA-256 of them in order. Resolving binds
        every operation or rejects the plan; nothing is substituted. The
        GGML module declares fused RMSNorm-mul and rms_norm then mul
        (`implementations.h`).
        - On every profile, `unit.PlanRegistryTest.*` shows, on fake
          declarations, the plan identity following each implementation's
          (BP-S2), a plan made for an old identity rejected as stale
          (BP-S2), and one naming an implementation the build lacks
          rejected as unsupported, never bound to the other one (BP-S4).
          `unit.GgmlValidateTest.*` covers the unfused implementation's
          operand checks, and `unit.GgmlModuleDigest` (CUDA profiles)
          that the GGML identities cover every file of the module.
        - On `spark`, `unit.GgmlPlanTest.*` runs both selections by plan
          in one process: each is bit-identical across cudaMalloc, device
          VMM and host VMM and close to a CPU reference, at widths 896 and
          4,096. Fused and unfused outputs were also bit-identical to each
          other there (reported, not gated). A stale or foreign declaration
          binds no kernel. BP-S1's exactness against the bridge arms closes
          in P2.
      - **The FP16 plan's remaining operations.** Every GGML kernel that
        `fp16-plan.json` records has an implementation over GGML's launcher
        (`ops.h`): get_rows (both kernels), the KV write (set_rows, F32
        into F16), NEOX RoPE, RoPE fused with the K write, soft_max (both
        column variants), cont of the merged heads (a copy or the scalar
        kernel), SwiGLU, and MMVF fused with a bias or residual add and
        with gate, up and SwiGLU. Each refuses what its launcher asserts
        on; the row indices get_rows, set_rows and the fused K write read
        from device memory are the plan's to bound. The checks run in
        every profile (`unit.GgmlOpsValidateTest.*`). The registry declares the nine
        implementations (`implementations.h`). `fusion.h` reproduces
        upstream's gates for the three fusions over a graph's nodes in
        GGML's order, uses and data ranges included
        (`unit.GgmlFusionTest.*`); the device's MMVF selection completes
        the two MMVF gates (`MulMatVecFusible`). The gates build no GGML
        graph object, which trips UBSan (RE-021). The
        new launchers' sources hold no `throw`, `try` or `catch` at the
        pin (D-066).
        - On `spark-b`, `unit.GgmlOpsMemoryTest.*` shows each bit-identical
          in cudaMalloc memory, device VMM and host VMM, and close to a CPU
          reference. There the fused bias add equalled MMVF then add bit
          for bit, and the fused K write equalled RoPE then set_rows; the
          fused gate and up product, accumulating in F32, differed from the
          unfused one in every element (reported, not gated).
        - `unit.GgmlOpsPlanMatchTest.*` records each launch at the CUDA
          runtime's entry points and compares it with the recorded plan at
          the model's shapes and the recorded row counts (1, 16, 17, 32,
          512): kernel name (NVCC's per-file hash normalized), grid, block
          and dynamic shared memory, or copy size, on the context's stream.
          It matches kernels 3, 6, 7, 8 and 24–28 at every row count the
          plan launches them at, the one-row cont's 3,584-byte copy, the
          fused decode products (17, 19, 20) and the unfused ones (16, 18,
          and the bias add, 4). Not yet matched this way: attention's
          products (9–15, 21), the norms (22, 23), mul (5) and the cuBLAS
          path (0–2, 29–44).
        - The test binary's SASS for kernels 3–28 has the bridge's recorded
          text and encoding hashes (cuobjdump 13.0.85 on `spark-b`,
          2026-09-27).
      - **Per-launch host cost (BP-F4, reported).** On `spark`, one
        decode-row RMSNorm-mul costs about 1.76 µs of host time in GGML's
        fused launcher alone and 1.88 µs through a bound plan; unfused,
        3.6 µs and 3.95 µs (medians;
        [report](experiments/launch-overhead/README.md)). Per-token figures
        against upstream's decode wait for a native decode step (P2).

      - **ExLlamaV3's GEMM kernels enter the build** as the source lock's
        `exllamav3` component: the reference revision's archive
        (`6b84a21b`), narrowed by `archive.keep` to exactly the closure of
        upstream's compilation units for the mcg codebook at K = 4, 5, 6
        and 8 (19 files and `LICENSE`), with two patches, as GGML entered
        (D-077). Patch 0001 removes `util.cuh`'s exiting error checks
        and drops four no-op `register` specifiers that NVCC rejects with
        a Clang host compiler; 0002 adds jitLLM's build, which reproduces
        the P0 reference's device flags. No ATen host wrapper is kept; a
        tooling test fails if `keep` differs from the units' include
        closure (quoted and angle includes) or a kept file can end the
        process. The GEMV and the reconstruction, Hadamard and bias-add
        kernels (whose `.cu` files they share with ATen wrappers) came with
        the launchers in P3 (below). The per-file audit is in
        [licensing.md](licensing.md#exllamav3-gemm-kernels-in-the-core-m2).
        The owner cleared the kernels and `ptx.cuh`'s direct libcu++
        include on 2026-09-27 (D-080). Only tests link them so far (the
        lock's `use: test`).
        - In every profile, `unit.Exl3ContextTest.*` checks the device
          context's sizes from the kept `exl3_devctx.cuh` (4,202,760 B of
          lock slots, a 16 MiB workspace). In every CUDA profile,
          `unit.Exl3KernelTablesTest.*` links the 64 kernels' tables. On
          `spark-b`, `unit.Exl3KernelsGpuTest.*` loads each from its sm_121
          SASS, and each of the 14 GEMM kernels the P0 launch record names
          uses the recorded registers.
        - The 14 kernels' SASS equals the recorded hashes in the `native`
          and `cross` builds' archives and in the linked test binary
          (cuobjdump 13.0.85 on `spark-b`, 2026-09-27). All 64 kernels'
          SASS is identical across both builds and a standalone compile
          with upstream's exact PyTorch flags. P0's `sass_hashes.py` hashes
          the last function of each cubin together with the next member's
          header, so that hash depends on its container; none of the 14 is
          affected, and the other 12 recorded ExLlamaV3 hashes are
          re-derived before use ([P0 report](experiments/backend-proof-p0/README.md)).
        - Each CUDA profile compiles the four units in 75 to 113
          CPU-seconds (19 to 28 s each; the workstation idle, then loaded).

      Remaining in P1: nothing of the allocation census, which D-085
      replaced with a coarse memory check (P2, below). The first native
      EXL3 linear landed with P3 (below). Its
      launchers keep `src/kernels/exl3/launch_contract.h`: every
      cooperative kernel (`cooperative_groups`' grid sync traps when the
      launch was not, and the split-K locks and the multi-GEMM's group
      barrier spin forever if the grid is not all resident) launches
      cooperatively within the co-resident limit, on lock slots no
      concurrently running launch shares. The kept files hold no trap of
      their own; all 160 in the four GEMM units' SASS are that sync's
      (checked 2026-09-27).

      *P2 prerequisites* ([scope](backend-proof.md#p2-prerequisites)):
      - **FP16 memory limits and the census rule,** pre-registered under
        D-079 before any native FP16 run
        ([memory and workspace](backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd)).
        - Each phase kind's `E` is itemized from the recorded plan:
          activations of n × 611,328 bytes (which reproduce both bridge
          compute buffers to the byte), the recorded pool peaks, input
          copies and logits.
        - `E` runs from 1.17 MiB for a single-token step to 748.3 MiB for
          the 512-row prefill. KV, weights and the 32 MiB cuBLAS workspace
          are limited outside it.
        - The census reconciles the catalog with `MemAvailable`,
          `SUnreclaim` and `RssAnon` at quiescent points, with controls.
          Opaque growth goes to `F` only up to the bridge's own; after the
          warm-up, it goes to the phase's `E`.
      - **The plan comparator.** `tests/support/` holds a launch recorder
        for tests and benchmarks; configure fails if a production binary
        links it. `ggml_ops_test` now uses it.
        `experiments/backend-proof-p2/plan_compare.py` compares a
        recording, with cuBLAS's logs, SASS hashes and an nsys trace, with
        `fp16-plan.json`.
        - `tools/tests/test_plan_compare.py` (in `mise run check`) shows:
          - every bridge arm matches itself;
          - 22 mutations are each caught and located, and the allowed
            differences pass;
          - a fragment, or a run without its nsys trace, is never a
            complete match;
          - a full synthetic recording with logs, SASS and trace converts
            and matches.
        - `unit.PlanRecordTest.*` holds the writer to the sample the
          converter reads.
        - On `spark-b`, `unit.GgmlOpsPlanMatchTest.*` recorded the fused
          decode step's first seven launches. With the test's nsys trace,
          SASS from cuobjdump 13.0.85 and the libraries' hashes, they
          matched `control-fused` from token 6, registers and SASS
          included. The result is incomplete by design: it is a fragment,
          with no cuBLAS call and so no cuBLAS logs.

      *P2 started:* **`src/artifact/`** reads v0 prepared artifacts
      natively (no importer; artifacts still come from M0's prototype).
      `Artifact::Open` treats the directory as untrusted input and makes
      the checks the prototype makes when a loader opens an artifact
      (`verify` without payload hashing), in its order and under its rule
      names:
      - the directory, its links and file set, and every size before a read;
      - a strict parser for the canonical JSON subset (integers only, no
        escapes, the prototype's caps), with the artifact ID as the
        manifest's SHA-256;
      - manifest and index schema, profile, groups, chunks, resource and
        expert-slice placement, representation sizes and GGML over-read,
        EXL3 closure and expert structure, all in checked arithmetic;
      - each shard header rebuilt from the index and compared byte for
        byte.

      It yields groups, chunk ranges, placements and closures, row lookups
      and coalesced vectored read plans, and opens a shard for direct reads
      only if it is still the file it validated. It reads no payload and does
      not parse kept `.kv.gguf` metadata. The prototype decides
      `unit.ArtifactCorpusTest.*` in every profile: a corpus it builds and
      judges at build time. Its views of three synthetic artifacts match
      line for line, and its verdicts on 188 mutated ones and 800 seeded
      random mutations (plus probes of the two documented divergences, a
      string escape and the unparsed metadata) match, each divergence
      checked to its exact rule. `unit.ArtifactFilesTest.*`
      adds links, special files, caps on sparse files, the owner policy and
      shards replaced or rewritten after open. On `spark-b`,
      `unit.ArtifactFixtureTest.*` opens the M0 fixtures (Qwen2.5 FP16, both
      EXL3 rates and Gemma 4) and matches the prototype's views of them.
      Next: page-in into device VMM through the landing zone (B2, D-081).

      *Native FP16 at rung 3* ([report](experiments/backend-proof-p2/README.md)):
      - **The model and its graph.** `src/model/` holds the Qwen2 adapter:
        a compiled-in Qwen2.5-0.5B profile bound to the artifact's resources
        at load, and each chunk's host-built inputs (n_kv padded to 256,
        the cell layout, mask, positions, output rows). A reference-side
        script checks the profile against the GGUF. `kernels/ggml/` builds
        each chunk's graph as llama.cpp does (941 GPU nodes; the bridge
        logs 942 with its CPU embedding lookup), plans it with upstream's
        fusion gates in upstream's order and the device's kernel family,
        places activations, and runs the plan bound through the registry,
        which now declares RMSNorm, add, mul and the three product families.
        `unit.Qwen2Test.*` and `unit.Qwen2GraphTest.*` cover the profile,
        the inputs, the node order, both plans and the placement (activations
        exactly A) in every profile.
      - **The FP16 gate passes on all four arms** (2026-09-27, `spark-b`).
        `benchmarks/fp16_exec.cc` loads the FP16 artifact with direct reads,
        every chunk's digest checked, into `cudaMalloc`, and runs both
        trajectories recorded from process start. For FP16-U and FP16-F,
        `control` and `heldout`, `plan_compare.py` reports a complete match
        (exit 0, with cuBLAS's logs, SASS and an nsys trace); only then were
        the logits compared, and they are bit-identical to the bridge's, a
        second evaluation equal to the first. BP-S1 closes.
      - **The census fails on all four arms** under its pre-registered
        rule. The bridge's `F` cap was measured first and written into
        backend-proof.md, with one amendment made before any native census:
        `MemAvailable` misses host memory on the per-CPU page lists
        (RE-024), so readings add them. Native's 1.5 s reading wait was
        set after the first (void) native readings were seen. Every native
        phase places exactly its limit, and native's growth ends each
        evaluation below the cap; the failures are sub-R excesses at
        single warm-up steps and growth that reverses within a few
        intervals. In a batch set aside for the final binary, FP16-F
        `control` passed. For the owner: the rule as written resolves
        neither, nor says how many runs a verdict takes.
      - **D-085 replaced the census with a coarse memory check, and it
        passes** ([check](backend-proof.md#memory-and-workspace-the-m2-gate-in-exl3-bringupmd),
        [report](experiments/backend-proof-p2/README.md#memory-check-d-085)).
        Memory is judged loosely: native's peak, read from `MemAvailable`,
        may be at most about 10% above the bridge's. Native's rung-3 peaks
        on `spark` were 0.81 to 0.96 times the bridge's on all four arms.
        Native EXL3's peaks (EXL3-O arm, rung 3) were 0.82 (4.0 bpw) and
        0.87 (4.5 bpw) times upstream ExLlamaV3's in its reference
        container. That comparison is not like for like: the reference's
        peak includes PyTorch's context and cache.
        - Two later census rules stopped under D-085 (in Git history). v2
          (counters plus an exact nsys tier) failed its bridge holdout on
          opaque counter jumps. v3 (the nsys tier alone) passed its holdout
          and was dropped before registration.
        - Their finding stays: cuBLAS's handle keeps a 64.1 MiB default
          workspace pool that `cublasSetWorkspace` does not free (RE-028).
          The owner kept it outside the 32 MiB workspace figure
          (2026-09-27).

      *The D-081 page-in path, and rungs 4 and 5*
      ([report](experiments/backend-proof-p2/README.md#rungs-4-and-5-paged-into-device-vmm-through-the-landing-zone)):
      - **The page-in path** (`scheduler.h`, `pagein.cc`). A page-in runs
        in stages, each an operation proven complete before the next: the
        VMM lane (or, without one, the device lane) creates, maps and
        opens the extent's backing (D-033);
        the load waits in order for a slot of the landing zone, a
        persistent host-VMM pool; the storage lane reads the chunk into it
        with direct I/O, in the order loads were published; the device
        lane copies it into device VMM on the
        zone's stream; and the extent is published, and the slot freed,
        only once the copy's fence completes. Failed or withdrawn loads
        unmap and release what they mapped; unproven reads, copies or
        unmaps quarantine the extent and its slot. Eviction unmaps and
        releases the backing on the same lane. The device lane also runs
        kernel jobs (`LaunchWork`) whose leases hold until their fence.
        `unit.VmmWork/PageInTest.*` (every profile, with VMM work on the
        device lane and on a VMM lane) covers stage order,
        publication only after the fence, slot reuse only after the copy,
        a full zone's order, failed and short reads, backing failures and
        unknown outcomes (an unmap refused because an earlier unknown
        outcome left its reservation undetermined quarantines the
        extent; a map refused so fails its load cleanly), cancellation in
        every stage (a withdrawn load starts no new stage, even when its
        copy completes in the same turn), a stage that can never get a
        mailbox (quarantined, so the stop reports the fault; one that
        will free is waited for), eviction, reload and relocation, and
        a threaded stress; `unit.VmmWork/CudaPageIn.*` (`gpu`, both
        ways) runs io_uring into host VMM and the copy into device VMM on
        `spark-b`, with eviction, relocation, and cancellation met during
        a read and during a copy.
      - **Rungs 4 and 5 pass on all four FP16 arms** (2026-09-27,
        `spark-b`). `benchmarks/fp16_paged.cc` pages the artifact into
        device VMM through that path (the token table's host copy read in
        place into host VMM for the CPU's embedding lookup) and runs each
        chunk as a device job holding a lease on everything it touches.
        With the plan recorded over four evaluations (a repeat and two
        restores that evict every weight, release its backing and page it
        back in, the second at another reservation), `plan_compare.py`
        reports a complete match, the first evaluation's logits equal the
        bridge's, and the rest equal it bit for bit. BP-A1's in-process
        check found every bound tensor in cataloged device extents of its
        class. Page-in through the zone ran at 11.4 GB/s (median) against
        13.8 in place with backing mapped once, and 6.9 in place with
        backing made per load: a measured shortfall against D-081's
        in-place condition with like backing.
      - **Page-in at disk speed**
        ([report](experiments/pagein-perf/README.md), 2026-09-27,
        `spark` and `spark-b`). The shortfall was the runtime's: the direct reader
        started a load's reads out of file order (RE-026), the storage and
        device submission lanes slept between reads and copies (RE-017),
        and VMM work queued ahead of the copies. The reader now starts
        reads in arrival order, both lanes poll for 200 µs after their last
        progress, and a VMM lane (`BackingService`) makes and releases
        managed backing. Over 8 GiB at four in flight on an idle `spark`
        the zone ran at 14.93 GB/s with backing mapped once and 14.69 made
        per load, against 14.70 in place and 14.90 for the standalone
        probe (before: 12.35, 12.04 and 14.54); rungs 4 and 5 re-ran
        exact. That file was freshly
        written: this SSD reads files at rest, the FP16 artifact's
        among them, at ~13.3 GB/s by any path (RE-027). Coalesced
        chunk-closure reads (BP-P1) remain open: they would cut
        operations, not raise bandwidth.
      - **Rows this closes or advances:** BP-N3 (FP16), BP-A1's in-process
        pointer coverage (FP16), BP-A3 for FP16 (no F32 conversion and no
        CPU extra buffer type; every weight once in device VMM, and the
        one duplicate, the token table's 272 MB host copy for the CPU's
        embedding lookup, declared and cataloged as the bridge's
        `CUDA_Host` copy is, so not hidden; the row's "permanent FP16
        shadow", which its next sentence pairs with transient
        reconstruction, is EXL3's reconstructed weights, for P3), and in part BP-P1
        (every weight evicted and restored bit-identically, but with one
        read per chunk, not the coalesced chunk-closure reads the row
        names), BP-L2 (cancellation in every stage drains before a slot or
        backing is reused, on the fakes; on `spark-b` a request cancelled
        while a slot is busy, and one cancelled while a copy is in flight
        (it completes, publishes whole bytes, then frees its slot), drain,
        but the `io_uring` cancellation race itself is not observed), BP-P5 (relocation, descriptors rebuilt),
        BP-P6 (restore times reported), BP-L1 (a job's lease holds until
        its fence, on the fake) and BP-V2 (VMM create and map failures
        unwind, on the fake).
      Left in P2: the census on the paged harness (BP-A1's
      reconciliation, BP-A2, BP-A5; replaced by D-085's coarse peak check)
      and BP-A4's stale binding and negative controls (landed with P5,
      below). Next for the pager: write-back and state spill through the
      zone (BP-P4, landed with P4 below), coalesced vectored reads, and
      the D-033 handoff of a victim's backing.

      *P3 started: native EXL3 linears exact against upstream*
      ([report](experiments/backend-proof-p3/README.md)):
      - **The lock** adds to the `exllamav3` component the K = 4 GEMV
        kernel (core since D-080) and the reconstruction, Hadamard and
        bias-add kernels, whose `.cu` files patch 0003 reduces to their
        kernels (removing the ATen wrappers). jitLLM's instance unit
        (patch 0002) instantiates only what the linear launches, with the
        reference's device flags; every ExLlamaV3 function in the build has
        the reference's SASS. The per-file records are in
        [licensing.md](licensing.md#exllamav3-gemm-kernels-in-the-core-m2).
      - **`src/kernels/exl3/`** holds the launchers (K-L): a launch context
        on a provider stream with a declared, zeroed lock area no other
        live context shares, cooperative launches bounded by the device's
        co-resident limit, faults returned; host checks of every launch in
        every profile (`unit.Exl3ValidateTest.*`); the reconstruction GEMM
        on cuBLASLt with its pinned algorithms; each path of the linear;
        a recorded copy of upstream's GEMV choice; and five registry
        declarations whose identities cover the module's files.
      - **BP-N5 passes** (2026-09-27, `spark-b`): every real projection of
        both fixtures at rows 1, 8, 9, 16, 32, 33, 144, 145, 1,023 and
        1,024, in EXL3-G and EXL3-O (6,280 cases, 268 through the GEMV),
        is bit-identical to upstream's extension at the same forced plan
        and inputs, every intermediate buffer included, and so are every
        linear's reconstructed weights. It holds with operands in
        `cudaMalloc` memory, at their minimum alignments, and flush against
        unmapped VMM granules: no kernel the sweep launches over-reads at
        these rates, shapes and plans, so
        v0 needs no over-read reservation for these fixtures' EXL3
        resources ([artifact-format.md](artifact-format.md)).
      - On `spark-b`, `unit.Exl3LinearTest.*` runs every path in
        `cudaMalloc` memory and device VMM, the over-read probe (also at
        every fixture shape for the packed variants the sweep never
        launched: the GEMV's wide configuration and its 2–8-row mode,
        every tile shape each shape takes, tile shape 4 on a synthetic
        shape since no fixture shape takes it), two contexts' cooperative
        grids at the co-resident limit on two streams, and the refusals
        (a non-co-resident grid, a shared lock area, a fault, a pin
        recorded for another GEMM, stale multi-GEMM tables).

      *P3 part 2: native EXL3 end to end*
      ([report](experiments/backend-proof-p3/README.md#part-2-the-native-model)):
      - **`src/model/qwen2_exl3.h`** binds an EXL3 artifact and plans each
        phase in the approved record's order (every operation, owner,
        implementation and dtype, each linear's forced launch plan from the
        frozen tuning caches and pins), refusing unrecorded phase kinds
        and missing cases, and places each phase's tensors by lifetime;
        **`src/kernels/exl3/qwen2.h`** binds a phase through the registry
        (plan identity: every implementation's identity and the launch
        plan) and runs it on one stream. New GGML implementations: the
        casts, the BF16 embedding and the forced vector attention; new
        EXL3 declaration: the bias add. The launch recorder wraps the EXL3
        launch entry points and cuBLASLt.
      - **Pre-registered at P3 entry (D-079), before any native EXL3
        timing or memory result:** BP-F2's reference arm and its 184 cases
        (EXL3-O, the fused gate/up cases, ExLlamaV3's bias add, one frozen
        tuning cache per case set, new calibration and holdout), and the
        EXL3 phase memory limits tightened to native's itemized buffer plan;
        a phase kind the trajectories reach (the step with K padded to
        1,024) recorded, with EXL3-O's record, before native ran it.
      - **Results** (2026-09-27, `spark-b`), both fixtures in EXL3-G and
        EXL3-O: the executed plan equals the record (85 phases of 8 kinds);
        Tier E at operation level exact (EXL3-G, every GGML operation
        recomputed by the bridge's library); Tier C passes (all 750
        statistics within bounds); rung 3 repeats, and rungs 4 and 5 (paged
        into device VMM through the zone, evicted, restored, relocated) are
        bit-identical to rung 3.

      BP-F2 and the EXL3 census do not run (D-085): parity is judged end to
      end once serving works, and memory by a coarse peak check, which
      passes on every FP16 arm and both EXL3 fixtures (D-085, D-086).

      *P4, P5 and P6: paging, lifetime and the contract*
      ([report](experiments/backend-proof/README.md), D-086):
      - **Write-back** (`scheduler.h`, `pagein.cc`; D-081's reverse path).
        Evicting live state whose source is its write-back place copies it
        into a landing slot, fenced, writes the slot with direct I/O
        (`ReadSpec::kind`), and then unmaps and releases the backing. The
        catalog (`BeginEvict(…, write_back)`) then keeps the content
        generation and marks the contents preserved, so a later load
        restores them. A failed or short write abandons the eviction with
        the state resident and intact; an unproven copy quarantines it and
        its slot; a load of a place with nothing preserved is refused, and
        so is a new source naming another range for preserved contents.
        Tests: `unit.CatalogTest.AWriteBackEvictionPreservesTheContentGeneration`,
        the write-back cases of `unit.VmmWork/PageInTest.*` and
        `unit.VmmWork/CudaWriteBack.*` (`gpu`: io_uring, device VMM, a
        poisoned premapped pair and a managed pair).
      - **BP-P2, P3, P4 on both representations** (2026-09-27, `spark-b`).
        `jitllm_fp16_paged` and `jitllm_exl3_paged` gained `--partial`,
        `--spill` and, for FP16, `--embeddings shared`. All four FP16 arms
        and all four EXL3 arms were run. Every evaluation equalled the
        first bit for bit, and the first equalled rung 3.
        - Six partial evictions (one layer, side vectors and biases, the
          trellis only in EXL3, a shared small-tensor chunk, padded tails,
          a tensor crossing a chunk boundary) each evicted exactly their
          extents. A launch over the incomplete closure was refused
          before it ran.
        - The cache was written back and restored after a prefill and
          mid-decode, managed or poisoned while premapped, and came back
          byte-identical.
        - FP16 with its token table held once, in device VMM, gave the
          duplicated arm's logits.
        - The EXL3 head is checked to be its own resource.
      - **BP-L and BP-V on the real providers:**
        - a cancelled 1,023-row EXL3 prefill gated behind a stream wait
          keeps its lease on every extent until its fence
          (`--cancel-in-flight`; BP-L1, L3);
        - repeated cancellations at varying points never corrupt a
          reassigned slot (BP-L2);
        - io_uring starts of unknown outcome and duplicated completions
          change nothing, and an unknown fence quarantines and stops
          admission (`unit.VmmWork/CudaPermutations.*`, BP-L4);
        - backing the driver cannot create or map unwinds cleanly
          (BP-V2);
        - a GGML kernel over unmapped backing faults in a child process
          (`unit.GgmlStaleMemoryDeathTest.*`, BP-A4's negative control);
        - a tight budget admits the EXL3 fixture's largest reconstruction
          phase or refuses the plan, explained (`unit.ProgramPlanTest.*`,
          BP-V3).
      - D-050's rows: the pause, lifetime and registration rows gained
        tests (`unit.Admission.*`, `unit.ShapeScenarioTest.*`,
        `unit.VmmWork/PageInTest.*`).
      - **BP-S3** (2026-09-27, `spark-b`): the paged harnesses became
        model runners (`benchmarks/fp16_runner.h`, `exl3_runner.h`) on a
        shared node (`tests/support/paged_node.h`): one catalog domain,
        scheduler, landing zone and workspace (activations and GGML
        pool), each model on its own stream. `jitllm_alternate_paged`
        alternates FP16 and EXL3 for three rounds under a budget that
        holds one model's weights and half the other's; each acquisition
        (`AcquireProgram`, the memory module's planning) evicts only the
        other model's weights. Every evaluation equals its model's rung-3
        logits bit for bit, with no coverage violation, on two pairings.
        Tests: `unit.AcquireTest.*` (fake backend) and
        `unit.CudaPagedNodeTest.*` (`gpu`).
      - Open: BP-P1's coalesced reads, which the owner chose to build in
        M2. D-050 rows whose features arrive later move to those
        features' milestones (owner, 2026-09-27).
- [x] **Retained-backing comparison** ([scope](backend-proof.md#retained-backing-comparison)):
      build the cross-model swap trace, have the retain/amend criteria
      approved, then keep or amend D-033.
      *Landed:* the [swap trace](experiments/retained-backing/README.md)
      and its seeded generator. It has eight synthetic models with D-056's
      measured sizes, in episodes shaped like the frozen A→B→A trace, with
      routes from the paging-feasibility captures. It holds a primary and a
      confirmation seed at three budgets. The
      [criteria](backend-proof.md#retained-backing-comparison) are
      pre-registered under D-079 with the trace's identity.
      *Landed:* part (a)'s [deterministic replay](experiments/retained-backing/replay.md)
      (`benchmarks/retained_backing/`, on an address-only fake provider)
      of D-033 and the 12 slab designs, on the primary seed at 64, 53 and
      40 GiB; every replay agreed with its twin. No slab or hybrid design
      meets every deterministic criterion at every budget, so none can
      replace D-033 on this seed. Nine pass at 64 GiB only (less waste
      and lost content than D-033's padding costs): contiguous-run
      eviction and compaction at each slab size, size classes at 32 MiB
      and the hybrid at 32 and 256 MiB. At 53 and 40 GiB every design's
      peak waste is above D-033's. The confirmation seed stays unread.
      *D-033 is retained* (2026-09-27): no design beats it at every
      budget on part (a). The timed sessions and part (b) do not run
      (D-085).
- [x] **Shape expressibility** (D-068): fake-provider scenarios for draft
      rejection and rollback, a canvas across boundaries, block output and a
      two-artifact context.
      *Landed:* internal contract types, which the P6 decision settles. The
      resource core is unchanged and names no architecture.
      - `src/model/`: state representations declare their blocks and
        capabilities (append; truncate to any position or only to a
        snapshot). A request's live state keeps its committed prefix apart
        from tentative positions. Model contexts compose components in
        roles, each an artifact or a declared part of one, and count
        shared resources once.
      - `src/execution/program.h`: phase kinds with validated widths,
        closures and working sets; decoding modes; transient working
        state; and `PlanProgram`, which turns a finite request into its
        envelope. Everything a program keeps across a completed boundary
        goes in `R_i`: live states at the bound, working states such as a
        canvas, and the output buffer. `E_i` is the largest phase closure,
        with shared extents counted once, plus its working set. A cursor
        refuses phases past the admitted program, and an output buffer
        refuses output past the admitted bound and makes a phase's output
        visible whole or not at all. A contract that could only fail once
        running (no phase emits, colliding item names, a closure in
        another domain) is refused at planning.
      - `unit.ShapeScenarioTest.*` drives these through admission, the
        ledgers, the catalog and materialization, with the fake
        device-memory provider backing every extent. After every step it
        checks D-050's guarantee and the ledgers' invariants. The
        scenarios:
        - a stored draft layer at a budget of exactly `F + R_i + E_i`:
          partly rejected drafts roll back, freeing whole blocks and never
          the committed prefix. A draft paused or cancelled before its
          verify commits nothing.
        - paged KV, snapshot-only recurrent state and a drafter's own KV
          through verify cycles at every acceptance count, a pause after
          one that accepted every draft, and verifies narrowing to the
          bound; a failed verify rolls back through the snapshot at the
          prefix.
        - a companion drafter in a second artifact: one closure over both
          artifacts, with the shared embeddings leased and charged once. A
          stale closure's lease is refused whole.
        - a 256-position canvas paused between denoising steps (D-069):
          the canvas stays in `R_i` and unchanged, and the substitute
          evicts only idle weights. The negative control charges the
          canvas to the phase instead; that request is admitted beside the
          substitute, whose first phase then cannot materialize (a
          circular wait). With the canvas in `R_i`, the same substitute
          queues without a grant. A cancelled paused canvas never commits.
        - block output: a commit waits for output room for the whole
          block, a failed commit rolls back and publishes nothing, and the
          last block is clamped to the output bound. A failed phase ends
          its request, so block output needs no truncation.
      `unit.ProgramPlanTest.*`, `unit.StateCursorTest.*` and
      `unit.ModelContextTest.*` cover the types. No contract gap needed a
      special case in the core.
- [x] **Explainable plans:** plans expose their validated phase widths,
      envelopes and rejection reasons, and the proof records each phase
      kind's guaranteed bound against its observed peak.
      *Landed:* a `ProgramPlan` lists each phase kind's
      width, validated widths, closure, working set, envelope and phase
      count, and itemizes `R_i`. A rejection names the phase kind, the
      width, the required bytes and the shortfall
      (`unit.ProgramPlanTest.*`). The paged harnesses record each phase
      kind's bound (activation extent or region, and pool bound) against
      the highest byte its bound tensors reach and the pool's peak in it
      (`paging.json`). No peak exceeded its bound: FP16 reached every
      bound exactly, and EXL3 reached them or stayed below (the region by
      up to 1,833,216 bytes, the pool by up to 176)
      ([report](experiments/backend-proof/README.md#p6-bounds-against-peaks)).
- [x] Find which OS counters include VMM backing on the Spark driver, so
      the [memory breakdown](architecture.md#memory-breakdown) reconciles.
      *Landed:* on `spark` (driver 580.178.04), device-local and host
      backing both leave `MemAvailable` when created and return when
      released, and the driver's free memory equals `MemAvailable`. Neither
      is charged to the process's cgroup (RE-019). Device backing never
      appears in RSS; host backing appears there only while mapped with
      access. The driver's bookkeeping costs about 34 KiB of unreclaimable
      slab per 2 MiB extent. The breakdown now reconciles against
      `MemAvailable` and shows that bookkeeping as its own line
      ([measurement](experiments/vmm-counters/README.md)).
- [x] **Discrete NVIDIA target** (D-082, owner 2026-09-27): build and
      smoke only; no model runs on it in M2.
      *Landed:* the `native` build compiles every CUDA unit for `sm_121`
      and `sm_86`, GGML's and ExLlamaV3's included (a GB10-only diagnostic
      benchmark excepted; about +46% CPU time for its CUDA objects when
      measured before P2; Spark builds stay GB10-only). `jitllm doctor`
      reports each GPU as unified or discrete and judges a targeted
      discrete GPU as it does the GB10, GPU 0 only (multi-GPU hosts are
      out of scope). The `gpu-discrete` tests (`mise run test -- native
      --gpu`: the providers, the lanes, D-081's page-in path, a GGML
      kernel smoke, EXL3 kernel loading, jitLLM's EXL3 launchers and a
      native EXL3 phase, the CUDA contract and `smoke.doctor.discrete`)
      pass on the workstation's RTX 3080 Ti; tests that compare with GB10
      records stay GB10-only. Fast-swap validation on it is M4's.
- [x] Record the operation contract, registry, patch set, phase envelopes
      and `F` per profile in a decision entry, and the aggregate report in
      `experiments/backend-proof/`.
      *Landed:* D-086, with the per-operation K-C and K-L choices and the
      status of D-052, D-053 and D-081; `F` is judged loosely at the
      process level (owner, 2026-09-27). The
      [aggregate report](experiments/backend-proof/README.md) holds the
      case matrix's status.

**Exit criteria:**

- Every M2 row of D-050's
  [adversarial matrix](reservation-policy.md#worked-cases-and-implementation-gates),
  with D-069's pause cases, passes on the fake backend with no vendor SDK
  present (`check`); rows that need real allocation pass as their BP cases.
- The BP [case matrix](backend-proof.md#case-matrix) passes, its
  fake-backend and CPU-only cases on the workstation and the rest on
  `spark` (`check:spark`), including repeated map/load/evict/restore.
  BP-V1's importer cases run against M0's prototype, and report-only cases
  are reported. The FP16 control and both EXL3 fixtures run native prefill
  and decode that stay within the declared oracle bounds before and after
  restoration.
- The EXL3 kernel-time and workspace gates (BP-F2) pass, or the owner has
  approved an explicit tradeoff. *The owner did, on 2026-09-27 (D-085):*
  BP-F2 is not run; each engine is instead held to at least its
  reference's speed end to end once its code is operational. A loader
  or an FP16 conversion is not EXL3 support (D-052).
- D-048's lanes pass their real concurrency, lost-wakeup and task-tree
  unwind tests, including ARM memory-ordering stress in `check:spark`.
- D-033 is explicitly retained or amended, and the operation-contract
  decision can express D-068's shapes without executing them.

## M3 — One resident model, end to end  `pending`

Goal: import the small fixtures, serve them resident through native GGML and
EXL3 execution, and complete chats from named clients over the three
baseline protocols, with bounded, explainable memory.

**Entry:** M2 exit with its operation-contract decision. The provenance of
llama.cpp's generated Unicode tables is cleared under D-017 before the native
tokenizer is adopted ([first-slice.md](first-slice.md)).

**Scope:**

- [ ] **Importer and verifier** (D-009, D-056): the C++ importer and the
      standalone verifier, with M0's Python prototype as the exact
      accept/reject oracle, running on the workstation and on a Spark.
      Confined import jobs take sources only from the configured stores
      (the `checkpoints` role or the long-term store; D-054, D-063) or the
      Hugging Face Hub (resumable and verified; the token comes from a
      `.env` or config file and is never logged). Hard-linked and symlinked
      sources are rejected, and the user docs say to copy them in or
      download with `--local-dir` (D-063). Removal waits for
      quiescence. Jobs keep their records, locks and grants and report
      progress, status, cancellation and retry (D-041). An install that
      lacks space fails before transferring, reporting the space it needs
      and the removal candidates.
- [ ] **Registration and first use** ([model lifecycle](architecture.md#model-lifecycle)):
      the compact installed-model index cache, shallow checks on every
      startup, and detail and plans built on first use inside an `F`
      reservation. Startup rejects an installed store that fails the
      direct-I/O probe (D-054).
- [ ] **Resident execution:** the FP16 and both EXL3 fixtures under jitLLM
      dispatch, finite context and chunk profiles within the 8K context,
      and state block sizes and KV layouts for each adapter. Graph capture
      enters here, behind a relocation proof, only if BP-F4 shows parity
      needs it.
- [ ] **Tokenization and output:** the native tokenizer, D-067 renderers for
      both template hashes, the tool-call parser, stop rules and seeded
      sampling; host versus device sampling is settled against D-052's
      decode gates.
- [ ] **Front door** ([M3 surface](client-api-baseline.md#m3-surface)):
      Chat Completions, stateless Responses, Messages with token counting,
      `/v1/models` in both shapes with D-046's metadata, and the discovery
      document (D-041); D-045's listener, authentication, CORS, `Host`,
      status and keepalive rules; D-047's non-streaming and storage rules;
      the Claude Code profile; and explicit rejection of `transforms` and
      `plugins`. Numeric intake bounds are fixed before any external input
      is accepted, with the total input limit reconciled with named-client
      tests ([cluster design](cluster-design.md)).
- [ ] **Local management API and CLI** (D-064): versioned routes for import,
      listing, representation inspection, removal, jobs and node health.
- [ ] **TLS** (D-065): per-name certificate files selected by SNI, reload on
      change, key-match and expiry checks, the name-constrained local CA,
      the certbot deploy hook, and the Tailscale certificate timer. The
      root-run hook and timer touch only `/etc/jitllm/tls/`, their units are
      sandboxed to it, and they take the heavy path.
- [ ] **Surface definitions:** front-door, alias and TLS configuration keys;
      the individual `jitllm-` header and body-field names (D-062); the
      HTTP, TLS and JSON libraries, chosen under D-017, D-057 and D-066.
- [ ] Start the model support matrix, recording template hashes.

**Exit criteria:**

- Teacher-forced logits and declared intermediates for the three fixtures
  match their pinned references within bounds declared before evaluation
  ([first-slice.md](first-slice.md), [exl3-bringup.md](exl3-bringup.md)).
  Rendered bytes and token IDs match the golden fixtures.
- EXL3 resident prefill, time to first token and decode inter-token
  p50/p95/p99 meet the predeclared upstream parity bounds against upstream
  serving controls in matched and normal views, with memory inside the
  declared bounds; a regression needs a fix or an explicit owner-approved
  tradeoff (D-052).
- The [M3 acceptance cases](client-api-baseline.md#acceptance-owed-in-m3)
  pass as scoped there. At least one named client completes a chat
  unmodified with each representation, and Cursor stays an explicit gap
  unless resolved. The context-compacted release moves to M4, which
  delivers D-041's close, and the Ollama-native checks move to M8 with the
  Ollama profile.
- Finite default context and output bounds bound every admitted request,
  and the M3 rows of D-050's matrix pass. Memory use is bounded and
  explained by the memory breakdown.
- The importer, verifier and front-door parsers pass their adversarial
  challenge; an interrupted import never appears valid.
- Measured and recorded: startup time and metadata memory as the installed
  library grows, the cold-switch cost of on-demand detail, and both
  contexts' state bytes, from which D-055's
  [capacity values](retention-policy.md#bounds-and-defaults) are pinned.

## M4 — First useful product: A→B→A with partial retention  `pending`

Goal: two small model contexts share one local budget. Switching to B
displaces only what B needs, retained conversation state lets A resume
without a full re-prefill, and the switching policy is chosen from
measurement. Useful without MoE or sharding.

**Entry:** M3 exit with its capacity values pinned, plus everything
[M4 entry pins](retention-policy.md#what-m4-entry-pins): the frozen
transcript, budgets and reference paths among them, and the spill write
budget from the drive's rated endurance. EXL3 switching budgets come from
new matched controls ([exl3-bringup.md](exl3-bringup.md)).

**Scope:**

- [ ] **Retention** ([policy](retention-policy.md); D-024, D-031, D-055):
      prefix and continuation entries with their identity, restore
      boundaries, branches, sharing and refresh; capacity-driven expiry with
      24-hour idle caps; the victim-order baseline; spill with its protected
      directory, preallocation, direct I/O, digest check on restore and
      deletion at startup; miss reasons and fallback reporting.
- [ ] **Partial eviction** of a quiescent model, reloading only missing
      dependencies (D-008).
- [ ] **Concurrency when it fits:** all-resident cohorts under
      full-envelope checks, reporting whether requests ran concurrently or
      time-sliced.
- [ ] **Switching policies** (D-069): the priority-aware default,
      run-to-completion and time-slicing with their guards, their
      configuration keys and per-alias overrides.
- [ ] **Request control** (D-042): interactive and background classes,
      maximum queue waits, cancellation and bounded progress events.
- [ ] **Release** (D-041, D-045): the final-turn flag, idempotent
      continuation close and the context-compacted release, with their wire
      names fixed here.
- [ ] **Warm jobs** (D-041): capacity-constrained, never an implicit
      download.
- [ ] **Diagnostics:** status, the admission what-if query, Perfetto trace
      export and eviction explanations; management controls for priorities,
      residency policies and trace capture; Prometheus `/metrics` with
      compatible health and load queries (D-044).
- [ ] **Storage scheduling:** demand reads mixed with spill write-back,
      inside the pinned write budget.
- [ ] Add the A→B→A workload and its regression thresholds to `check:spark`.
- [ ] **Discrete GPU, secondary** (D-082): the A→B→A fast swap, one model
      active and partial retention within device memory, on the
      workstation's discrete GPU (`mise run test -- native --gpu`), with
      its PCIe restore rate reported, within a configured device budget and
      with the landing zone reported apart from it. Whole-model swaps and
      paging within device memory only; no on-demand expert paging from the
      SSD there. Not an exit criterion.

**Exit criteria:**

- D-055's [timed workload](retention-policy.md#m4-acceptance-workload)
  passes its pass rule: Qwen2.5-0.5B FP16 and EXL3 4.0 bpw in both
  orientations, six jitLLM arms against fresh interleaved references, 72
  accepted repetitions per arm, orientation and cache condition. For each
  floor arm (J-partial, J-spill), orientation, direction and cache
  condition, at the median and at p95, jitLLM's one-sided 97.5% upper bound
  is at most the smallest one-sided 97.5% lower bound among the valid
  reference arms; a comparison with no valid reference arm does not pass
  (D-036). The report covers latency distributions, bytes read and written,
  peak memory and spill, prompt tokens reused versus recomputed, and deltas
  against jitLLM's whole-model control (M7's comparator).
- The [correctness gates](retention-policy.md#correctness-gates) pass: exact
  outputs and bit-identical teacher-forced logits against
  provenance-matched controls, and catalog state and events show that only
  selected extents were displaced.
- The [functional and adversarial cases](retention-policy.md#functional-and-adversarial-cases)
  and the M4 rows of D-050's matrix pass with an EXL3 context in the
  matrix; cache expiry never destroys admitted suspended work.
- Through at least one unmodified named client: a long conversation on A,
  B under pressure, then A resumed, over both resident reuse and forced
  spill/restore. The context-compacted release case deferred from M3
  passes.
- B arriving while A is still generating is measured under each D-069
  policy, with queue delay reported apart from paging and switch time and
  from first-token compute, together with pauses and bytes reloaded. The
  owner keeps or changes the default on these results.
- An all-resident control shows concurrent progress when both complete
  envelopes fit.

## M4a — Configured placement across nodes  `pending`

Goal: one configured conductor places whole models on enrolled nodes and
routes requests with retained-state affinity, so a subagent's model runs on
the other Spark while the main model stays resident. It follows M4,
independently of M5; sharding is M6.

**Entry:** M4 exit.

**Scope:**

- [ ] **Setup and discovery** (D-038, D-039, [cluster design](cluster-design.md)):
      inventory, trusted SSH, the mDNS window, layout classification,
      bounded dedicated-QSFP subnet scans, one-time enrollment and path
      re-detection for enrolled members.
- [ ] **Configuration and trust:** the shared and node-local v2 documents,
      the private CA with TLS 1.3 mutual authentication, epochs, session
      fencing and restart reconciliation, and protocol v1 with its bounded
      state, schema tests and exact per-message field catalog.
- [ ] **Conductor** (D-037): placement, affinity routing, single-attempt
      dispatch, credit-based streaming, health states and authoritative
      per-node admission; stale or aggregate reports never admit.
- [ ] **Availability** (D-041, D-046): the per-model `endpoints` shape.
- [ ] **One import per cluster** (D-054): peer replication of verified
      prepared artifacts over the cluster link, archive to and install from
      the long-term store, and the explicit archive-or-delete choice when a
      node lacks space.
- [ ] The package gains its rdma-core dependencies when M4a links them
      (D-063).
- [ ] Decide whether a worker node serves its own loopback management
      listener.

**Exit criteria:**

- The [required validation](cluster-design.md#required-validation-and-handoff)
  challenge conditions and the conductor's fake-transport and Spark
  scenarios ([architecture](architecture.md#conductor-ownership-and-admission))
  pass.
- An unmodified standard client completes A→B→A through one endpoint, with B
  placed on the other node while A stays resident. Each node enforces its
  full local budget and compatible state reuse, and models run concurrently
  when placement permits.
- Stale capacity reports, node loss and cancellation end in bounded failure
  or unwind, never in unsafe admission or silent replay of a started
  stream.
- Remote access requires authentication and transport protection (D-014,
  D-065). A replicated or archived artifact is published only after
  verification against an identity held outside its source.

## M5 — Demand-paged MoE and the first daily drivers  `pending`

Goal: exact demand-paged routed-expert execution on the named Gemma 4
26B-A4B and Ornith 1.5 35B-A3B pair, and those two models usable day to day
through the named clients, reasoning and constrained output included (owner,
2026-09-23).

**Entry:** M4 exit; M4a is independent. The pair's checkpoints and
representations are pinned, and their GGML source closures, tokenizers and
chat templates are selected and audited (D-013, D-057, D-067). Approved
before measurement: the performance protocols; a resident-performance bound
against the pinned llama.cpp reference (prefill, time to first token and
decode inter-token p50/p95/p99, matched and normal views); the named
budgets, each with its loading policy, including at least one per model at
which its routed experts do not all fit beside its other extents, state and
headroom, so selected experts miss during both prefill and decode; and the
sustained-use schedule and duration.

**Scope:**

- [ ] **Routing boundary** (D-008, [architecture](architecture.md#routing-boundary-for-moe-7)):
      selected-expert leases, asynchronous misses and resumable tasks,
      brought up on a synthetic or tiny MoE before the named pair
      (features.md). Envelopes cover the worst-case union of every allowed
      closure and assume no fixed number of positions per phase (D-068).
- [ ] **Loading policies:** eager active-model loading that keeps inactive
      extents (the feasibility study's recommended first policy) and routed
      demand paging, both selectable, compared at the named budgets.
- [ ] **Expert layout:** the pointer-table or uniform-stride dispatch
      decision and expert compaction from the M5 GGML proof, and the MoE
      mapping in v0 artifacts.
- [ ] **Shapes that come with these models:** hybrid sliding-window and
      global attention (Gemma 4) and recurrent or linear-attention layers
      (Ornith), with their state adapters and explicit restore coverage
      (RE-004, RE-007). M4's retention matrix extends to them.
- [ ] **Traces:** native routing-trace capture and policy replay, checked
      against the M0 reference experiment; captured traces stay outside Git
      and replay by verified hash in the gate.
- [ ] **Reasoning** (D-043, D-046, D-047): protocol-specific reasoning,
      final and tool fields and streaming, model-supported thinking
      controls, signed blocks, `reasoning` and `reasoning_details`, and
      cached-token usage from real prefix reuse.
- [ ] **Constrained output** (D-043): `response_format` JSON object and
      schema, strict tool arguments and vLLM's `structured_outputs.json`,
      over a documented schema subset with explicit rejection of the rest.
- [ ] **Sustained use:** a bounded multi-hour agent/subagent session on the
      pair with repeated switches, branches, cancellations (during I/O and
      while paused included), expiry by capacity and by test-shortened idle
      caps, and spill and restore.
- [ ] EXL3 MoE only if claimed, with its own routed-expert closure, kernel,
      quality and performance baselines (D-052).
- [ ] Assess artifact compatibility guarantees with M4's dense and M5's
      MoE evidence, in a separate decision (D-018).

**Exit criteria:**

- Acceptance exercises real demand misses. At each miss-forcing budget,
  catalog state and events record nonzero selected-expert misses in prefill
  and in decode, with their count, bytes and wait time and the phase and
  layer at which each suspended. There, a measured window without misses is
  inconclusive for the paging gates, and the next two criteria are judged
  on runs with misses.
- No unselected expert loads beyond declared metadata and read-ahead, and
  no expert is substituted or its contribution dropped, verified from
  catalog state and events.
- Numerics stay correct against the pinned references after eviction and
  restoration and across within-step misses, and the M5 rows of D-050's
  matrix pass with real routes.
- D-036's generation limits hold on the pair: at most 10% added generation
  time, continuation time to first token included, and at most 20 ms p95 /
  100 ms p99 added token gaps against a resident control with matched state
  provenance, at every named budget, the miss-forcing ones included. Pause
  gaps are reported separately (D-069). M4's switching floor still holds.
- Resident prefill, time to first token and decode inter-token p50/p95/p99
  on both models meet the approved bound against the pinned llama.cpp
  reference, or the owner approves an explicit tradeoff, as D-052 requires
  for EXL3. Paging limits compare jitLLM with itself, so they cannot stand
  in for this.
- The sustained-use run ends with every completed, cancelled and paused
  request retired and no lease, grant, task or I/O outstanding. Memory, the
  catalog and retained-entry metadata, task and job records and queue
  depths stay within their bounds and level off instead of growing with
  elapsed time; task and job records and queue depths return to their idle
  levels after each switch cycle. Spill stays within `S_spill` and the write
  budget, and the memory breakdown reconciles with OS counters at the end.
- Measured and reported: the routing boundary's resident-hit overhead, each
  routed phase kind's bound against its observed peak, and whether idle
  retained state starves weight residency.
- Named clients complete reasoning and tool round trips on both models,
  including acceptance case 7's reasoning checks, and constrained-output
  requests pass their pinned compatibility fixtures. Any OpenRouter-mode
  claim rests on D-046's pinned-client run. The support matrix records the
  client versions.

## M6 — Sharded model execution (two Sparks)  `pending`

Goal: a flagship model sharded across both Sparks, correct under asymmetric
pressure, cancellation and controlled failure. Placement-only use already
works at M4a.

**Entry:** M4a and M5 exits. By entry: model-parallel artifact partitioning
is decided ([artifact-format.md](artifact-format.md#deliberately-open)); the
flagship checkpoint and its validated parallelism recipe are named (the
MiMo-V2.6-Flash-RL TP=2/EP=2 reference and the MiaAI-Lab recipes are the
candidates); and the two-node admission design chooses between
prepare/commit and the deferred mirrored-ledger shortcut (features.md).

**Scope:**

- [ ] Explicit sharding with conductor-issued distributed phase IDs,
      porting the recipe's parallelism first (TP, PP and EP are different
      plans).
- [ ] Coordinated admission: node-issued reservations and every rank ready
      before commit, with prepare failures unwound; an unknown completion
      never frees another rank's buffers.
- [ ] A separately budgeted, stable communication-buffer pool that honors
      NCCL's registration and threading contracts; ordered collective
      submission and completion fences.
- [ ] Per-step collective latency measured before bandwidth.
- [ ] EXL3 sharded cases only if claimed (D-052).

**Exit criteria:**

- Both ranks stay correct against the pinned reference under asymmetric
  pressure, cancellation and controlled failure, including partial
  preparation, a lost commit, mismatched rank generations and node loss
  during a collective. No timeout is treated as proof of reclaimed memory.
- Sharded performance is reported against the recipe's reference deployment
  in both views.

## M7 — Performance and new decoding modes  `pending`

Goal: meet D-036's benefit target on a library larger than memory, and
execute the speculative and block-diffusion shapes designed since M0
(D-068).

**Entry:** M5 exit. M7 may start before M6 exits, but MiMo's stored MTP
layers run only on M6's sharded execution, so that work waits for M6 and M7
exits after it. Before M7 planning, the bounded DiffusionGemma reference
study measures the per-step expert closures of wide phases, and untriggered
deferrals are reviewed ([features.md](features.md) and the table below).
Before the new modes execute, manifest references to another artifact by
ID, with shared resources counted once, are decided as a format change, and
numerical and statistical bounds and trace protocols for speculative and
diffusion decoding are declared. The named configurations, including the
over-memory library, and the partial-retention benefit workload are pinned
before acceptance runs (D-036).

**Scope:**

- [ ] **Larger-than-memory library:** DeepSeek V4 Flash with Qwen3.8 Flash
      Next on one node is the canonical pair (D-036). It brings compressed
      attention with an indexer, Qwen3.8's sparse n-gram rows and its
      linear-attention layers. If the pair cannot be validated, name another
      whose prepared weights exceed physical memory rather than pass M7 on
      the small pair alone.
- [ ] **Speculative decoding** (D-068): stored MTP layers (Ornith and
      Qwen3.8; MiMo's once it runs sharded under M6) and Gemma 4 companion
      drafters.
- [ ] **Block diffusion** (D-068): DiffusionGemma-26B-A4B.
- [ ] **Execution speed:** selective CUDA graphs behind a relocation proof,
      further kernels and plans (D-053), and target-assisted import tuning
      as an explicit, separately keyed mode.
- [ ] **Victim policy:** compare global LRU, frequency/recency and the
      cost-aware heuristic on identical recorded traces, with the confirmed
      hysteresis, minimum-residency and reload-cost refinements
      ([baseline](architecture.md#victim-selection-initial-baseline)) and
      per-model statistics so no model monopolizes reclaimable bytes. A
      candidate replaces the baseline, and the cost-aware heuristic becomes
      the default, only if replay shows fewer miss bytes and reloads.
- [ ] Deferred optimizations whose triggers have fired (prefetch, residency
      warm-start and the others in features.md), each against its
      demand-only control.

**Exit criteria:**

- D-036's benefit target: at least 25% lower median return-switch latency
  than jitLLM's own whole-model control, with identical state handling at
  the same budget, on the agreed partial-retention workload, with at least
  one named library exceeding physical memory, while the switching floor
  and generation limits still hold.
- Measured and reported on the over-memory library: retention under
  physical pressure and whether idle retained state starves weight
  residency (D-055).
- The speculative verify path's teacher-forced logits match plain decoding
  within declared bounds, with top-1 agreement reported and free-running
  divergence reported at its first position (RE-008).
- Speculative sampling preserves the target distribution. On recorded target
  and draft distributions, acceptance, rejection and residual resampling
  match a reference implementation of the rule exactly under fixed seeds;
  sampled outputs match plain sampling's distribution within declared
  statistical bounds, at the supported sampling settings.
- Rollback leaves exactly the accepted prefix. After rejected drafts,
  teacher-forced continuation matches a control that drafted only the
  accepted tokens, bit-identical where the plan is the same at both verify
  widths and otherwise within the declared bounds. No rejected position's
  KV, recurrent update or drafter state survives, none enters a retained
  entry (D-055, D-068), cancelling between draft and verify retires all
  draft work, and a pause there commits none of it before verify.
- Diffusion stays within declared bounds of its reference, compared step by
  step under a fixed seed. A cancelled canvas never commits, and a paused one
  commits nothing until it resumes and finishes.
- Matched and normal reference comparisons are reported, with no numerical
  or lifetime regression in any supported configuration.

## M8 — Product and first release  `pending`

Goal: the remaining confirmed product scope, and a first tagged 0.x release
that a stranger can install from the project's package repository and run
(owner, 2026-09-23).

**Entry:** M7 exit. The owner may pull an item forward once its dependencies
exist (for example the Ollama profile or tokenization endpoints after M3, or
the dashboard after M4's management controls); the release itself waits for
every earlier exit. Each model added for embeddings, reranking or file
inputs is named at entry with its pinned reference engine and numerical
bounds, declared before native evaluation.

**Scope:**

- [ ] **Dashboard** (D-064): a separate service that calls the management
      API from its own server side, including setting the Hugging Face
      token.
- [ ] **MCP management adapter** (D-042): a separate process exposing
      discovery, status and explicitly authorized actions.
- [ ] **Application permissions** (D-042): per-application credentials and
      scopes for inference, read-only status and model administration.
- [ ] **Session and hint extensions** (D-022, D-046): the optional session
      ID and release, client warm hints, and D-046's `session_id`, `user`
      and `metadata` hints. M3's front door already accepts them as
      advisory preferences, and M4 records `session_id` for release lookup.
- [ ] **Ollama subset** (D-041, D-045): listing, details, chat and
      generation with the `keep_alive` mapping, tested with a named
      Ollama-native client including its load-time bound.
- [ ] **Tokenization and diagnostics** (D-043, D-044): vLLM-compatible
      tokenize, detokenize, tokenizer information and prompt rendering; raw
      Completions with standard log-probabilities and bounded token
      diagnostics.
- [ ] **Pooled outputs** (D-042, D-044): embeddings and reranking on
      validated models named at entry.
- [ ] **File inputs** (D-042): text resources, images (starting with Gemma
      4's vision encoder) and audio files, on models validated for them,
      with bounded preprocessing.
- [ ] **Packaging** (D-027): the signed arm64 apt repository and its signing
      keys, optional copyleft modules in the default install with a
      build-time opt-out (D-080), and drain-before-restart upgrades.
- [ ] **Release readiness** (D-061, D-062): the release checklist, the
      published support matrix, notices and source obligations for every
      shipped profile, and user documentation.

**Exit criteria:**

- Each new route passes its pinned compatibility fixtures, including
  negative and interrupted-stream cases, with the unmodified clients its
  decision names; unsupported features fail explicitly.
- Every new model and output path meets numerical acceptance against its
  pinned reference within the declared bounds before any support claim:
  embedding vectors with the model's pooling and normalization; rerank
  scores and orderings; encoder outputs and the language model's
  teacher-forced logits on image and audio prompts, with preprocessing
  matched to the reference's pinned implementation. Raw Completions
  log-probabilities agree with the validated teacher-forced logits, and
  tokenize and render output matches inference byte for byte and ID for ID.
  Generated text alone is not evidence.
- On a fresh Spark, installing from the apt repository and following only
  the checked-in docs reaches a running supported model (vision.md).
- The default and copyleft-disabled builds meet D-017 with a complete,
  audited closure; builds with optional modules enabled ship matching
  notices and source.
- The release commit passes `check`, `check:full` from a fresh clone and
  `check:spark` (D-061), and the owner tags the first 0.x release.

## Deferred delivery and proposals

Confirmed scope stays confirmed when its implementation is deferred. A
candidate stays unapproved until revisited; reaching a trigger is a reason
to evaluate it. Deferred optimization triggers (predictive prefetch,
dependency-group scoring, optimistic MoE) live in features.md.

| Item | Earliest work / revisit trigger | Scope |
| --- | --- | --- |
| Load/temperature-aware GPU operating policy | Earliest M7 evaluation, or earlier diagnosis if reproducible throttling or unexplained shutdowns occur | [Proposed telemetry and optional adaptive clock ceiling](features.md#load--and-temperature-aware-operating-policy-proposal); measure stock/fixed/adaptive policies first, no assumed fault or automatic host changes |
| Ollama registry and other management compatibility | After the basic subset and relevant native management operation, when a named client needs them | Deferred D-041 candidate; lifecycle mapping needs separate proof |
| Regex/grammar constrained output | After validated JSON/schema support, when a concrete client requires it | D-043 deferral; no automatic milestone delivery |
| LoRA adapters | Earliest M7 planning after validated base-model execution, when a concrete adapter workload needs them | D-044 deferral; no automatic delivery |
| Classification/reward/generic pooling APIs | Earliest M7 planning after validated base-model execution, when a concrete model/task workload needs them | D-044 deferral; not implied by embedding/reranking support |
| Live audio/video input | Earliest M7 planning after initial file-input evidence, when a concrete workload establishes streaming/synchronization requirements | Deferred D-042 candidate; not an automatic M7 deliverable |
| Batch/background inference jobs | Earliest M7 planning after validated request scheduling, when a concrete workload justifies scheduling/storage needs | Deferred D-042 candidate; ordinary background request priority is already confirmed |
| Automatic membership changes | After M4a, when configured enrollment and explicit restart cannot reasonably serve membership churn | Candidate mechanism under D-023/D-038; bootstrap discovery and path refresh for enrolled nodes are already M4a scope |
| Conductor election | After M4a, when conductor failover becomes an explicit requirement; first define fencing and in-flight request handling | Candidate mechanism under D-023; one configured conductor initially |
| Automatic replica placement and balancing | After M4a, when measured overlapping demand on a small model causes waiting while another node has sufficient headroom | Confirmed D-023 scope with deferred delivery; preserve affinity and include duplicated weights/state in budgets |

Review untriggered items during M7 and M8 planning; they do not
automatically enter either milestone's scope or block earlier milestone exits.
