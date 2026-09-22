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
- [x] Feature triage: walk [features.md](features.md) with the owner; confirm,
      reject, or defer proposals (2026-09-21: 46 proposed or open rows
      triaged; 30 confirmed, 13 deferred with a reason, trigger, and
      earliest milestone, 2 rejected, 1 still open. Load-bearing calls
      recorded as D-028 GGML-first substrate with build-time optional
      backends, D-029 contribution and compliance conventions, D-030
      Claude Code and the Anthropic Messages format). A deferral does not
      become approval when its trigger fires.
- [x] Confirm the original-code license (2026-09-20: Apache-2.0 accepted,
      D-003; dependency categories and tiers clarified in D-017, superseding
      D-015; process weight for an externally consumed project recorded,
      D-016).
- [x] Decide NOTICE and SPDX-header conventions and the contribution policy
      (2026-09-21: REUSE copyright/license metadata plus an embedded-header
      check in CI and a NOTICE file from M1, SBOM with packaging, external
      PRs accepted under DCO; D-029).
- [x] Inventory the environments without changing drivers or security
      settings (2026-09-20, read-only, no sudo): workstation baseline and both
      Sparks recorded in
      [architecture.md](architecture.md#target-nodes-dgx-sparks). The
      interconnect half is the separate cabling task below.
- [x] Spike — **toolchain smoke** (2026-09-21; open question 6,
      D-032): Clang C++23 native and AArch64 cross builds passed;
      NVCC 13.4.92 (Toolkit 13.4.2) with Clang 22.1.8 produced
      C++23 `sm_121` CUDA objects that ran on `spark`, as did the native
      Spark fallback. C++23 host/device
      feature checks and GPU results passed on driver 580.178.04 with
      PTX JIT disabled; the older 13.0 dialect limit is RE-001.
      Exact compiler, library, CUDA component, and target-sysroot pins,
      hashes, commands, and limits are in the
      [smoke report](experiments/toolchain-smoke/README.md).
      M1 still owns declarative provisioning and CMake presets.
- [x] Spike — **VMM microbench** (2026-09-21; open question 1, D-033):
      three runs on `spark` measured 2 MiB minimum/recommended granularity,
      allocation/map/access/unmap/release costs across 2–128 MiB extents,
      and costs with independent background kernels. A retained, unmapped
      1 GiB pool kept its physical footprint and contents; releasing its
      handles returned capacity. Initial policy: 2 MiB independent extents,
      completion-safe backing handoff, no standing unused-handle cache.
      [Aggregate report and harness](experiments/vmm-microbench/README.md);
      numbers in architecture.md. I/O and model-load optimization remain
      separate measurements, not conclusions of this allocation experiment.
- [x] Spike — **I/O path comparison** (2026-09-21; open question 2, D-034):
      compared buffered/direct files, pinned staging, cuFile compatibility,
      native asynchronous I/O, and GPU in-place access on `spark`.
      Direct regular files into host VMM reached about 15 GB/s, sustained
      14.962 GB/s for 180 seconds, and matched device-VMM GPU scan speed.
      Concurrent compute, 100 GiB held-memory pressure, cache reclamation,
      sparse/small reads, verified write bursts, and DMA-bounce tracing are
      in the [aggregate report and harness](experiments/io-path/README.md).
      Initial budget: two to four 2 MiB destination slots, no extra Spark
      staging copy. Raw-device alternatives need a dedicated unmounted SSD;
      no raw performance claim is made. Controller interrupt coalescing was
      tested separately with its original setting restored. Actual GGML
      behavior and full registration/reclaim lifetimes remain M2 proof work;
      mixed read/write and sustained-write/endurance policy remain M4 inputs.
- [x] Pick the reference engine for the feasibility spike: llama.cpp with
      MoE GGUFs and a small router-logging patch (decided 2026-09-20; the
      lightest install and the owner's preference).
- [x] Install llama.cpp on a Spark in a container so the host baseline in
      architecture.md stays clean (2026-09-21): digest-pinned ARM64 CUDA
      image, source `b29c606e2`, and hash-verified Gemma 4 UD-Q4_K_M execute
      on `spark`. GPU inference and cross-process slot restore passed;
      Gemma requires `--swa-full` for the tested reuse path (RE-004), with
      its additional memory cost recorded. Exact identities, expert/state
      accounting, reproducible harness, and limits are in the
      [reference setup report](experiments/reference-setup/README.md).
      No host toolchain/driver/security changes or HF key were needed.
      Ornith (MIT, confirmed by owner) is now validated in the switching
      experiment below; Qwen remains an unexecuted candidate. Installation
      alone did not complete switching or route-trace experiments.
      Candidate trace models, owner-provided, with facts from their model
      cards as read on 2026-09-20 (repository access/revisions/licenses
      rechecked 2026-09-21; Gemma's structure and memory accounting were
      verified in setup, Ornith's in the subsequent A→B→A experiment):
      - [unsloth Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
        125B core parameters, 6B active, 512 experts, top-10 plus 1 shared, expert
        intermediate dim 640, plus a 51B n-gram embedding table and a 4B MTP
        head; Gated DeltaNet plus sparse MQA attention; 262K context. GGUFs
        run from about 72 GB
        (1-bit) through 82 to 90 GB (3-bit) to 111 GB (Q4_K_XL). On a
        121 GiB node that is a single-model budget sweep: 3-bit fits, Q4 is
        borderline, Q5 and up exceed the node, so it covers the
        forces-paging axis by itself. Its n-gram table is the first concrete
        sparse-lookup component: row requests resolve to containing extents
        (D-035), and its MTP head matters
        for matched-configuration comparisons (D-021).
      - [unsloth gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF):
        25.2B total, 3.8B active, 128 experts, top-8 plus 1 shared; hybrid
        1024-token sliding-window and global attention; 256K context.
        Q4_K_M about 17 GB. The owner has experience with Gemma.
        Fits easily: the subagent and switch-latency case, with mixed KV
        lifetimes for the spill study.
      - [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF):
        36B total, about 3B active on the card, `qwen35moe` architecture;
        the selected GGUF verifies 256 experts, top-8 plus a shared FFN,
        40 primary layers and one stored MTP layer; 262K context.
        Q4_K_M about 22 GB. A second small model for replica and
        multi-model-switching traces.
      Gaps these three leave, to fill only if cheap: a few-large-experts,
      low-top-k model (Mixtral-style) for the high per-miss-cost end, and a
      plain full-KV GQA model where KV grows linearly and spill cost is
      highest. Record for each model used: expert count, top-k, shared
      experts, expert bytes at the chosen quant, total bytes, KV or recurrent
      state bytes per token, and headroom on one node. The owner's local
      Ollama blobs are plain GGUFs usable for workstation-side dry runs on
      the RTX 3080 Ti where they fit.
- [x] **Reference A→B→A experiment** (2026-09-21, D-025): 27 verified cycles
      on `spark`, Gemma 4 UD-Q4_K_M → Ornith 1.5 Q4_K_M → Gemma, with an
      18,339-token continuation. Three repeats of nine cases distinguish
      resident, warm/cold file cache, restore/recompute, 80 GiB locked
      physical pressure, and default-SWA/cache optimizations. Under the
      matched forced-displacement budget, median first-token waits are
      21.232 s outward and 18.304 s back with retained state; recomputing
      A's prompt takes 25.236 s on return. Warm restore returns in 4.062 s;
      live residency in 0.089 s. Native LRU, durable save/restore, actual
      block I/O, state bytes, memory and bounded observed swap are verified.
      A reuses 18,297 tokens and processes 42; all outputs match, and a
      separate early/late notebook recall probe passes. B's recurrent-state
      continuation also passes. Default SWA still re-prefills after restore
      (RE-004); decode speeds differ by state lifecycle and are reported.
      Exact pins, external trace identity, measured ranges, tooling and
      limitations are in the [aggregate report](experiments/reference-aba/README.md).
      Normal optimization probes use the same forced one-model policy;
      normal concurrent placement remains unvalidated. The later full feasibility
      study includes a separate large-model observation and finds that Gemma
      sequence snapshots can lack SWA coverage after prompt rollback (RE-007).
      The short matching restored output above is historical evidence, not a
      general continuation-correctness proof; the validated recompute arm is
      the usable correctness floor. M4 can replay this exact input trace.
      Comparator datapoint: Athena's Engine reports a 46 s measured switch
      between DeepSeek V4 Flash and Qwen3.8 Flash Next on one GB10 and a
      2.1 s restore of a 141k-token conversation from disk (creator-reported;
      see [architecture.md](architecture.md#comparator-athenas-engine-closed-source-creator-reported)).
      Optionally install it on a Spark as a second comparator for the
      normal-reference view if its terms still allow personal use; never cite
      its numbers in place of our own measurement.
      The author's X thread adds that the 46 s includes checkpointing the
      session and that the pair does not both fit in 128 GB at those bit
      depths, which makes DeepSeek V4 Flash plus Qwen3.8 Flash Next the
      canonical two-large-model switching workload for the feasibility spike
      if a DeepSeek V4 Flash GGUF is available at pick time.
- [x] Spike — **paging feasibility** (bounded study complete, 2026-09-22):
      the [full study](experiments/paging-feasibility/full-study.md) extends the
      [first cut](experiments/paging-feasibility/README.md) with varied longer
      Gemma/Ornith conversations, the exact reference A→B→A trace, concurrent
      four-sequence decode, and the larger-than-memory DeepSeek/Qwen library.
      Arrival schedules model minutes-to-hours sessions; they are not hardware
      soaks and do not test expiry. Matched budget sweeps compare demand paging,
      eager active-model loading with partial inactive retention, and whole-model
      switching, including non-expert extents, live state, workspace, headroom,
      bounded spill, restore/recompute, and isolated/packed 2 MiB layouts.
      Per-layer reuse, held-out prediction, switch-back dependencies, measured
      reference waits, and explicit storage/overlap sensitivities are recorded.
      Results are offline byte/service estimates, **not measured jitLLM speedups**.
      Qwen comparisons remain conditional on captured trajectories because
      exact prediction equivalence failed; no numerical tolerance is invented.
      Gemma rollback can require state recomputation despite byte-identical
      snapshots (RE-007), and unvalidated large-model spill continuations use
      conservative recompute scenarios. The evidence supports M4 retention
      before M5 expert paging and makes restore coverage an explicit validation
      requirement; it neither proves runtime admission nor enables prefetch.
      Performance acceptance targets are recorded in D-036 below.
- [x] Agree switching-benefit and generation-stall criteria (2026-09-22,
      D-036): M4 onward must meet the fastest correct full-swap reference
      arm at median/p95 in both directions; M5 onward permits at most 10%
      added generation time, continuation time to first token included, and
      20 ms p95 / 100 ms p99 added token gaps against a resident control
      with matched state provenance; M7 requires at least 25% lower median
      return-switch latency than jitLLM's own whole-model control on an
      agreed partial-retention workload, with at least one named library
      that exceeds physical memory. Targets apply to named supported
      configurations; correctness is mandatory and inconclusive comparisons
      do not pass. Pin workloads, trial counts, and measurement methods
      before acceptance runs; these are targets, not measured implementation
      results.
- [x] Complete the Spark-to-Spark direct-link baseline. The owner configured
      the `sparky` DAC cluster on 2026-09-21; the
      [baseline](experiments/interconnect/README.md) passed 78 host-buffer
      test pairs and 27 pinned NCCL runs. The two PCIe interfaces share one
      physical 200 Gb/s port: combined writes measured 184.76 Gb/s in either
      direction, reads 150.10 Gb/s with default queue settings. Large NCCL
      SendRecv/AllReduce reached 22.35/22.20 GB/s; supported result checks
      passed. Channel logs and counters verify actual HCA use; source and
      allocation logs establish GPU access to mapped host communication
      buffers, not GPUDirect RDMA. Message-size sweeps, ranges, pins and
      limitations are recorded with the report and in architecture.md.
      Sharded-model, asymmetric-pressure and failure tests remain M6 work.
- [ ] Inventory which MiaAI-Lab reference files are actually AGPL versus MIT
      ExLlamaV3 upstream before designing the optional-module boundary, and
      note AGPL's network clause for a served process in the licensing docs.
- [ ] Decide where the conductor lives (inside its node's runtime process or
      a sidecar) and how cluster-wide admission and placement are represented
      (D-020). Record in decisions.md.
- [ ] Define the initial configured cluster (D-023): configuration format,
      one designated conductor, configured membership, capability and health
      probes, per-node admission, and affinity to retained compatible state.
      M4a needs no discovery service, election, or automatic replica placement;
      their revisit triggers are below. Nothing about node names or counts
      in code; the owner's `spark`/`spark-b` are one deployment's config.
- [ ] Verify the endpoint set the named clients need (Cursor, OpenCode,
      Codex, Claude Code: chat completions, Responses API, Anthropic Messages
      format, streaming and tool-call details) against their current docs;
      record the baseline surface as a D-022/D-030 follow-up.
- [ ] Decide the async/task and completion model (open question 3), ideally
      prototyped against the fake-backend design.
- [ ] Decide the initial reservation guarantee and progress envelopes (open
      question 9) before M2: guaranteed versus opportunistic grants, retained
      continuation memory, bounded state growth, safe admission/serialization,
      and rejection or a validated alternative when a phase cannot fit.
      Record the policy and the adversarial cases it must pass; see
      [architecture.md](architecture.md#reservation-progress-gate).
      Direction settled 2026-09-21: turn/step-scoped leases with eviction
      only at scheduler-established completion boundaries (features.md);
      the policy entry is still owed.
- [ ] Decide the first vertical-slice checkpoint and numerical reference
      (open question 4); the substrate is GGML-first (D-028, 2026-09-21).
      Record provenance and license status of every reused unit.
- [ ] Scope the **early backend integration proof**, executed alongside M2:
      a small dense model runs from a prepared experimental artifact with
      jitLLM-owned weight/state backing, explicit workspace and completion
      tracking, and all backend allocations accounted for. Include D-034's
      GPU-accessible host VMM, registered-I/O buffer lifetimes, and reclaim
      after all consumers complete. Exercise D-035's imported extent layout,
      packed small tensors, and padded tails with whole-extent reads. Match reference
      logits, then repeat after eviction and restoration of weights and
      state at a completed boundary on a Spark. Exercise cancellation with
      pending work. Use the result to settle internal interfaces before M3;
      the operation contract is settled from this proof, not from the fake
      backend alone (there is no runtime plugin ABI, D-028).
- [ ] Compare retained backing strategies in the M2 backend/paging proof
      (owner follow-up 2026-09-21, D-035): D-033's small independent handles
      versus larger persistently mapped slabs, including 1 GiB, with software
      suballocation and actual executable tensor views. Keep ordinary paging
      free of avoidable create/release cycles. Measure warm reuse, remapping
      and registration costs where required, fragmentation, growing/shrinking
      the shared pool, concurrent compute, and end-to-end restore latency.
      Prove alias/captured-pointer/late-I/O safety for any address changes.
      Both designs retain useful contents within budget; disk transfer size
      remains independent. Use the result to retain or amend D-033 explicitly.
      Include checkpoint batches mixing small and bulk transfers: compare
      serial and bounded asynchronous submission, scheduling order/depth,
      time to the last required completion, and consumer stalls. M4 extends
      this to simultaneous demand reads and state write-back with dependency
      safety and bounded queues. The M0 I/O spike did not measure these mixes.
- [ ] Define the M4 A→B→A acceptance trace and the bounded retention policy
      (D-024, D-031): memory/spill/metadata limits, independent shared-prefix
      and conversation-continuation reuse/expiry policies, cleanup, cache
      identity, restore boundaries, and fallback/error behavior. Choose
      numeric defaults from measured state sizes and available headroom
      before M4. Include resident reuse, forced spill/restore, branch/edit
      cases, expiry and spill exhaustion; protect admitted suspended work.
      Include independent conversations sharing a system prefix: releasing
      or expiring one continuation preserves eligible shared-prefix reuse,
      and hits on that prefix do not refresh unrelated continuations.
- [ ] Choose an experimental artifact encoding and layout ABI (open question
      5, D-018), including validation, version rejection, and re-import rules.
      D-035 settles import-time repacking and the initial Spark profile:
      2 MiB aligned payload extents, whole-extent weight reads, explicit
      expert/tensor indexing, and no CPU payload repacking during page-in.
      The immutable data reuses a known aligned container and jitLLM owns
      the manifest and resource index (settled 2026-09-21); pick the
      container here. Work through dense, expert-axis, tied-weight, small-tensor,
      sparse-row, and final-tail examples: map stored ranges to executable
      GGML views, report padding and read amplification, reject invalid ranges,
      and account for shared extents and backend-readable padding. Include
      extent-boundary reads, a leased tensor sharing an otherwise evictable
      extent, resident holes between misses, and interrupted multi-file import.
      Distinguish logical suballocations/reusable holes from released physical
      backing; small state blocks do not inherit a 2 MiB logical size.
      Reject reuse of an immutable extent's padding/unused slots that conflicts
      with its whole-extent restoration or integrity footprint.
      Mutable spill encoding remains separate.
      Compatibility guarantees wait for dense and MoE
      execution and restore evidence; they are not an M0 requirement.
- [ ] Decide the C++ source-dependency mechanism (open question 7).
- [ ] Toolchain decisions: build-system conventions (CMake presets / Ninja /
      LLD, confirmed 2026-09-21), test framework, format and lint pins, CI
      shape including the copyleft-disabled profile, REUSE lint, a separate
      embedded-header check for commentable source/docs, and an
      installable `.deb` build from M1 (D-029), versioning/changelog
      conventions for an externally consumed project (D-016), and the
      installed layout that packaging will need (FHS paths, service user,
      systemd unit; D-027, confirmed 2026-09-21). Record the pins and the
      layout in decisions.md.
- [ ] First full draft of [architecture.md](architecture.md).
- [ ] Rewrite the provisional ladder below into real milestones with exit
      criteria.

**Exit criteria:** the owner has walked features.md and says the plan is good
enough to build from; open questions 1–7 and 9 are answered or explicitly
deferred with a reason and a milestone deadline; toolchain pins exist; M1+
milestones have scopes. Proposed optimizations may remain deferred with a
reason and trigger; they need not be accepted or rejected to exit M0. The
paging-feasibility result, measured reference switching baseline, agreed
performance criteria, and question 9's policy are required before M2, even if
deferred out of M0; feasibility sizes M4/M5 rather than gating viability
(D-021, D-025). Question 8 is resolved by
D-022; endpoint verification is an M0/M1 task. M0 exits on the owner's call,
not on a checklist reaching zero. Both Sparks are reachable (`spark`, `spark-b`); the toolchain, VMM, I/O,
interconnect, reference switching, and bounded paging-feasibility evidence
is recorded above. Sharded two-node reference execution remains separate
from these single-node model captures and is still future work.

## Provisional milestone ladder  `pending — to be rewritten in M0`

This is the owner's staged plan from ideation §19 translated into milestones,
ordered by risk: substrate, then the resource core, then one end-to-end model
path, then the first useful product at M4 (A→B→A with retention and conversation
state reuse), then configured placement across nodes at M4a, then MoE paging,
sharding, and further performance/product work. M4a keeps the existing M5–M7
identifiers stable and has no dependency on demand-paged MoE.
Sketch only — do not start work from these entries. They freely reference
`proposed` features.md rows; nothing here pre-empts the M0 triage. No stage
has a promised date; each should leave a usable, testable result.

- **M1 — Bootstrap.** Repository skeleton, declarative SDK setup (mise plus
  provisioning), C++23/Clang native build, Spark cross build, ARM/CUDA smoke
  binary running over SSH, CI with the copyleft-disabled profile, initial
  license and provenance tooling (REUSE lint, embedded-header check, NOTICE),
  a first-cut capability probe (the future `doctor` task), and an installable
  `.deb` build. *Gate:* clean host and container setup;
  native tests pass, including a CPU-only configuration with no CUDA toolkit
  present (D-026 guardrail); smoke binary runs on a Spark; exact pins recorded;
  any deferred feasibility experiment, measured reference switching baseline,
  agreed performance criteria, and reservation policy are complete before M2.
- **M2 — Resource core.** Catalog, reservation/lease state machine,
  deterministic fake backend, real VMM smoke harness. *Prerequisites:*
  paging-feasibility results recorded and M4/M5 scope adjusted if warranted
  (D-021, D-025); measured reference cycle and acceptance criteria recorded;
  reservation policy and progress envelopes are recorded. *Gate:* adversarial completion,
  cancellation, competing suspended-phase, state-growth, and impossible-phase
  tests pass with the fake backend and no vendor SDK present; repeated
  map/load/evict/restore checks succeed on a Spark.
  Run the early backend integration proof alongside this work; it must pass
  before settling the internal contract and closing M2. The proof covers one
  small dense model and its state, with correctness checked before and after
  restoration; full serving integration follows in M3.
- **M3 — One resident model, end to end.** Import a manageable model (small
  dense first) with the standalone artifact verifier, GGML-backed native
  execution (D-028), tokenizer/state/sampling baseline, and the
  OpenAI-compatible endpoint plus the Anthropic Messages format (D-022,
  D-030). *Gate:* teacher-forced and
  intermediate comparisons against a pinned reference; bounded, explainable
  memory usage; at least one named client completes a chat through the
  endpoint unmodified. M2's backend proof supplies the integration evidence;
  M2/M3 implementation may overlap while their gates remain explicit.
- **M4 — First useful product: A→B→A with partial retention.** Two small
  supported model contexts, one shared local budget, partial eviction of a
  quiescent model, bounded prefix/continuation retention (D-024, D-031), and
  basic status/diagnostics including the admission what-if query and Perfetto
  trace export. Through at least one unmodified named client, build a
  long conversation on A, request B under pressure, then resume A. *Gate:*
  only selected extents displaced; untouched data remains resident; reload
  only missing dependencies. Exercise both resident state reuse and forced
  spill/restore. A compatible retained prefix resumes without a full
  re-prefill; process only new input and any declared cache-block tail.
  Report switch/switch-back latency distributions, bytes read/written,
  peak memory/spill use, and prompt tokens reused versus recomputed against
  D-025's measured reference cycle, D-036's median/p95 switching floor, and
  jitLLM's own whole-model control at the same budget, which M7's benefit
  target uses as its comparator. Teacher-forced numerics stay correct after
  weight/state restoration. Branching histories,
  edits, incompatible identity, expiry, and spill exhaustion yield correct
  reuse, recomputation from supplied history, or explicit errors as appropriate;
  cache expiry never destroys admitted suspended work. Independent
  conversations reuse a shared system prefix with isolated mutable suffixes;
  continuation release/expiry leaves eligible shared-prefix reuse intact,
  while shared-prefix hits do not refresh unrelated continuations. Exercise
  shorter-prefix fallback after continuation eviction, independent prefix
  expiry, and shared-byte accounting, with numerical reference checks (D-031).
  An all-resident control demonstrates concurrent progress without paging
  when both complete execution envelopes fit. This milestone is useful
  without MoE or sharding.
- **M4a — Configured placement across nodes.** After M4, independently of M5:
  one configured conductor, configured nodes with capability/health probes,
  whole-model placement and request routing with state affinity. Run B on
  another node while A stays resident; use the existing network. *Gate:*
  an unmodified standard client completes the A→B→A flow through one endpoint;
  each node enforces its full local budget and compatible state reuse; models
  run concurrently when placement permits. Stale capacity reports, node loss,
  and cancellation cause bounded failure/unwind without unsafe admission or
  silent replay of a started stream. Honor remote-access protection (D-014).
  Discovery, election, and automatic replicas follow the separate triggers below.
- **M5 — Demand-paged MoE.** Routing boundary, selected-expert leases,
  asynchronous misses, resumable tasks, native trace capture and policy replay
  checked against the early reference experiment.
  *Gate:* no unselected expert loads beyond declared metadata/read-ahead; no
  substitution; resident-hit and miss overhead measured using the comparison
  protocol and D-036's generation limits (at most 10% added generation time,
  continuation time to first token included, and 20 ms p95 / 100 ms p99
  added token gaps) on the named Gemma/Ornith pair; the canonical
  two-large-model pair is an M7 configuration because the study's estimates
  place its demand-paging stalls at the gap limit without overlap. Small-MoE
  numerics remain correct after eviction and restoration. The M4 switching
  floor continues to apply.
  With M4's dense evidence, assess artifact compatibility guarantees in a
  separate decision (D-018).
- **M6 — Sharded model execution (two Sparks here).** Build on M4a's
  configured cluster; add explicit sharding for the flagship, coordinated
  admission, stable communication buffers, and ordered collectives. Needs
  the direct Spark-to-Spark link baseline (DAC configured 2026-09-21) and the relevant
  single-node model/paging evidence. *Gate:* both ranks remain correct under
  asymmetric pressure, cancellation, and controlled failure; no timeout is
  treated as proof of reclaimed memory. Placement-only use already works at M4a.
- **M7 — Performance and product.** Alternative compatible kernels and plans,
  prefetch, selective CUDA graphs, dashboard, optional API extensions
  (sessions, hints; D-022),
  packaging as signed apt packages for Spark with notices (D-027). *Gate:* measured results meet the agreed workload
  targets in D-036, including at least 25% lower median return-switch latency
  than jitLLM's own whole-model control on the agreed partial-retention
  workload, at least one named library exceeding physical memory, the
  switching floor, and generation limits; matched-configuration and normal
  reference-configuration comparisons are reported; no numerical or lifetime
  regression; compliant optional-backend builds.

## Deferred delivery and proposals

Confirmed scope stays confirmed when its implementation is deferred. A
candidate stays unapproved until revisited; reaching a trigger is a reason
to evaluate it. Deferred optimization triggers (predictive prefetch,
dependency-group scoring, optimistic MoE) live in features.md.

| Item | Earliest work / revisit trigger | Scope |
| --- | --- | --- |
| Automatic membership discovery | After M4a, when a deployment needs membership changes that configured nodes and explicit reload cannot reasonably serve | Candidate mechanism under D-023; configured topology is the initial path |
| Conductor election | After M4a, when conductor failover becomes an explicit requirement; first define fencing and in-flight request handling | Candidate mechanism under D-023; one configured conductor initially |
| Automatic replica placement and balancing | After M4a, when measured overlapping demand on a small model causes waiting while another node has sufficient headroom | Confirmed D-023 scope with deferred delivery; preserve affinity and include duplicated weights/state in budgets |

Review untriggered items during M7 planning; they do not automatically enter
M7's implementation scope or block earlier milestone exits.
