# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note.

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `in progress`

Goal: turn the design brief ([ideation.md](ideation.md)) into a settled
vision, feature matrix, architecture, and milestone ladder — through planning
conversations with the project owner plus targeted spikes where a decision
needs evidence from the real hardware.

- [x] Repo scaffolding for the AI-directed workflow (this scaffold,
      2026-09-20).
- [ ] Feature triage: walk [features.md](features.md) with the owner; promote
      or reject every `proposed` row; answer the open questions; record
      significant calls in [decisions.md](decisions.md).
- [x] Confirm the original-code license (2026-09-20: Apache-2.0 accepted,
      D-003; dependency categories and tiers clarified in D-017, superseding
      D-015; process weight for an externally consumed project recorded,
      D-016).
- [ ] Decide NOTICE and SPDX-header conventions and the contribution policy
      (external PRs accepted? DCO or CLA?). Record in decisions.md.
- [x] Inventory the environments without changing drivers or security
      settings (2026-09-20, read-only, no sudo): workstation baseline and both
      Sparks recorded in
      [architecture.md](architecture.md#target-nodes-dgx-sparks). The
      interconnect half is the separate cabling task below.
- [ ] Spike — **toolchain smoke** (answers open question 6): one end-to-end
      Clang C++23 native build, AArch64 cross build, and NVCC (Clang host
      compiler) CUDA object for GB10, deployed and run on a Spark over SSH.
      Output: exact LLVM / libstdc++ / CUDA / sysroot pins and the GB10
      architecture spelling → decision entry.
- [ ] Spike — **VMM microbench** (open question 1): on a Spark, measure VMM
      granularity, map/unmap latency vs extent size, cost under concurrent
      kernels, and physical-pool retention behaviour. Output: numbers in
      architecture.md plus an extent-size and pool decision entry.
- [ ] Spike — **I/O path comparison** (open question 2): cuFile compatibility
      mode vs native file I/O with pinned staging vs direct I/O, under
      concurrent compute and memory pressure; page-cache duplication and read
      amplification. Output: numbers, a storage-backend decision entry, and
      staging-budget guidance.
- [ ] Spike — **paging feasibility**: use a pinned reference engine and a
      representative quantized MoE to capture prefill and decode expert routes
      across representative requests and batch sizes, including interleaved
      requests to a smaller model in the workload. Replay partial
      retention and whole-model switching against the same request trace and
      total memory budgets, accounting for non-expert weights, live state,
      scratch, staging, and headroom. Combine miss bytes and read sizes with
      the staged-I/O measurements to estimate exposed stalls; keep estimates
      distinct from measurements and state overlap assumptions. Output:
      miss-byte curves versus memory budget, model-switch recovery costs,
      workload/configuration provenance, and owner-agreed generation-stall
      and mixed-workload benefit criteria. Follow the comparison protocol in
      [architecture.md](architecture.md#performance-evidence). Establish which
      workloads justify proceeding, or narrow the scope before M2. This may
      finish in early M1 if reference setup requires it; reference runs do
      not require jitLLM's model implementation or trace recorder.
- [ ] Re-inventory the Spark-to-Spark direct link once the QSFP/NCCL cable
      is installed (expected 2026-09-21): link state, RDMA devices, NCCL
      version, and the bandwidth a plain host-buffer transfer achieves
      between `spark` and `spark-b`. Record in architecture.md. Two-node work
      (M6) waits on this.
- [ ] Decide the async/task and completion model (open question 3), ideally
      prototyped against the fake-backend design.
- [ ] Decide the initial reservation guarantee and progress envelopes (open
      question 9) before M2: guaranteed versus opportunistic grants, retained
      continuation memory, bounded state growth, safe admission/serialization,
      and rejection or a validated alternative when a phase cannot fit.
      Record the policy and the adversarial cases it must pass; see
      [architecture.md](architecture.md#reservation-progress-gate).
- [ ] Decide the first vertical-slice checkpoint, backend, and numerical
      reference (open question 4); record provenance and license status of
      every reused unit.
- [ ] Choose an experimental artifact encoding and layout ABI (open question
      5, D-018), including validation, version rejection, and re-import rules.
      Compatibility guarantees wait for dense and MoE execution and restore
      evidence; they are not an M0 requirement.
- [ ] Decide the C++ source-dependency mechanism (open question 7).
- [ ] Toolchain decisions: build-system conventions (CMake presets / Ninja /
      LLD as proposed), test framework, format and lint pins, CI shape
      including the copyleft-disabled profile, license/provenance tooling
      (REUSE?), and versioning/changelog conventions for an externally
      consumed project (D-016). Record in decisions.md.
- [ ] First full draft of [architecture.md](architecture.md).
- [ ] Rewrite the provisional ladder below into real milestones with exit
      criteria.

**Exit criteria:** the owner has walked features.md and says the plan is good
enough to build from; open questions 1–7 and 9 are answered or explicitly
deferred with a reason and a milestone deadline; toolchain pins exist; M1+
milestones have scopes. The paging-feasibility result and question 9's policy
are required before M2, even if deferred out of M0. Question 8 may remain
deferred to M7. M0 exits on the owner's call, not on a checklist reaching
zero. Both Sparks are reachable now (`spark`, `spark-b`), so the three hardware
spikes can start. Paging feasibility also needs a pinned reference; a
two-node reference run waits on the direct link, just as M6 does.

## Provisional milestone ladder  `pending — to be rewritten in M0`

This is the owner's staged plan from ideation §19 translated into milestones,
ordered by risk: substrate, then the resource core, then one end-to-end model
path, then paging breadth, then two nodes, then performance and product.
Sketch only — do not start work from these entries. They freely reference
`proposed` features.md rows; nothing here pre-empts the M0 triage. No stage
has a promised date; each should leave a usable, testable result.

- **M1 — Bootstrap.** Repository skeleton, declarative SDK setup (mise plus
  provisioning), C++23/Clang native build, Spark cross build, ARM/CUDA smoke
  binary running over SSH, CI with the copyleft-disabled profile, initial
  license and provenance tooling. *Gate:* clean host and container setup;
  native tests pass; smoke binary runs on a Spark; exact pins recorded;
  any deferred paging-feasibility experiment and reservation policy are
  complete before M2.
- **M2 — Resource core.** Catalog, reservation/lease state machine,
  deterministic fake backend, real VMM smoke harness. *Prerequisites:*
  paging-feasibility result supports the agreed scope; reservation policy
  and progress envelopes are recorded. *Gate:* adversarial completion,
  cancellation, competing suspended-phase, state-growth, and impossible-phase
  tests pass; repeated map/load/evict/restore checks succeed on a Spark.
- **M3 — One resident model, end to end.** Import a manageable model (small
  dense first), native backend execution, tokenizer/state/sampling baseline.
  *Gate:* teacher-forced and intermediate comparisons against a pinned
  reference; bounded, explainable memory usage. M0 may interleave M2 and M3
  so numerical plumbing is derisked alongside the pager.
- **M4 — Partial retention.** Two persistent model contexts, shared local
  budget, partial eviction of a quiescent model, basic status API. *Gate:*
  only selected extents displaced; untouched data resident; resumption
  reloads only the missing dependencies; dense-model numerics remain correct
  after eviction and restoration from the experimental artifact.
- **M5 — Demand-paged MoE.** Routing boundary, selected-expert leases,
  asynchronous misses, resumable tasks, native trace capture and policy replay
  checked against the early reference experiment.
  *Gate:* no unselected expert loads beyond declared metadata/read-ahead; no
  substitution; resident-hit and miss overhead measured using the comparison
  protocol; small-MoE numerics remain correct after eviction and restoration.
  With M4's dense evidence, assess artifact compatibility guarantees in a
  separate decision (D-018).
- **M6 — Two Sparks.** Explicit sharding, coordinated admission, stable
  communication buffers, ordered collectives. Needs the direct
  Spark-to-Spark link (cable expected 2026-09-21). *Gate:* both ranks correct
  under asymmetric pressure, cancellation, and controlled failure.
- **M7 — Performance and product.** Alternative compatible kernels and plans,
  prefetch, selective CUDA graphs, dashboard, inference API surface,
  packaging with notices. *Gate:* measured results meet the agreed workload
  benefit and generation-stall criteria; matched-configuration and normal
  reference-configuration comparisons are reported; no numerical or lifetime
  regression; compliant optional-backend builds.
