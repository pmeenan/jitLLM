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
- [ ] Spike — **VMM microbench** (open question 1): on a Spark, measure VMM
      granularity, map/unmap latency vs extent size, cost under concurrent
      kernels, and physical-pool retention behaviour. Output: numbers in
      architecture.md plus an extent-size and pool decision entry.
- [ ] Spike — **I/O path comparison** (open question 2): cuFile compatibility
      mode vs native file I/O with pinned staging vs direct I/O, under
      concurrent compute and memory pressure; page-cache duplication and read
      amplification. Treat direct I/O as the default hypothesis on unified
      memory, and test GPU in-place access to system-allocated memory read
      straight from NVMe, which would remove the staging copy on Spark.
      Measure sustained read throughput over minutes to catch thermal
      throttling. Output: numbers, a storage-backend decision entry, and
      staging-budget guidance.
- [x] Pick the reference engine for the feasibility spike: llama.cpp with
      MoE GGUFs and a small router-logging patch (decided 2026-09-20; the
      lightest install and the owner's preference).
- [ ] Install llama.cpp on a Spark in a container so the host baseline in
      architecture.md stays clean; record exactly what was installed.
      Candidate trace models, owner-provided, with facts from their model
      cards as read on 2026-09-20 (re-verify at install time):
      - [unsloth Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
        125B core parameters, 6B active, 512 experts, top-10 plus 1 shared, expert
        intermediate dim 640, plus a 51B n-gram embedding table and a 4B MTP
        head; Gated DeltaNet plus sparse MQA attention; 262K context. GGUFs
        run from about 72 GB
        (1-bit) through 82 to 90 GB (3-bit) to 111 GB (Q4_K_XL). On a
        121 GiB node that is a single-model budget sweep: 3-bit fits, Q4 is
        borderline, Q5 and up exceed the node, so it covers the
        forces-paging axis by itself. Its n-gram table is the first concrete
        sparse-lookup component to page by rows, and its MTP head matters
        for matched-configuration comparisons (D-021).
      - [unsloth gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF):
        25.2B total, 3.8B active, 128 experts, top-8 plus 1 shared; hybrid
        1024-token sliding-window and global attention; 256K context.
        Q4_K_M about 17 GB. The owner has experience with Gemma.
        Fits easily: the subagent and switch-latency case, with mixed KV
        lifetimes for the spill study.
      - [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF):
        36B total, about 3B active, `qwen35moe` architecture (expert count
        not on the card; read it from the GGUF metadata); 262K context.
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
- [ ] **Reference A→B→A experiment** (D-025): once the pinned reference setup
      runs, measure a long conversation on A, a request to B under memory
      pressure, and a continuation on A. Use an all-resident control and a
      budget that forces displacement. Include end-to-end first-token waits,
      state save/restore or re-prefill, bytes read/written, and reused versus
      recomputed prompt tokens. Verify applicable reference routing and
      prompt-cache save/restore on the chosen checkpoints; record unavailable
      paths and any harness actions. Pin the trace, settings, engine revision,
      and target environment for M4 to repeat. Separate cold storage, warm OS
      cache, and warm residency; follow the comparison protocol. This is an
      M0/early-M1 deliverable before M2, not a new runtime implementation.
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
- [ ] Spike — **paging feasibility**: use a pinned reference engine and a
      representative quantized MoE to capture prefill and decode expert routes
      across representative requests and batch sizes. Shape the trace as the
      primary workload (D-019): a main model alternating with a subagent on a
      smaller or different model, long contexts, sessions lasting minutes to
      hours. Replay partial retention and whole-model switching against the
      same request trace and total memory budgets, accounting for non-expert
      weights, live state, scratch, staging, and headroom, with and without
      KV spill and restore across switches. Combine miss bytes and read sizes
      with the staged-I/O measurements to estimate exposed stalls and switch
      latencies; keep estimates distinct from measurements and state overlap
      assumptions. From the same traces report per-layer reuse distance and
      next-layer predictability, which decide whether prefetch can hide the
      remaining misses. Output: miss-byte curves versus memory budget, switch
      and switch-back cost estimates against the full-swap floor (D-021,
      D-025: artifact-size/bandwidth estimates for the first cut, then the
      measured reference cycle once setup runs), model-switch
      recovery costs, and workload/configuration provenance. Follow the
      comparison protocol in
      [architecture.md](architecture.md#performance-evidence). The result
      sizes how much partial retention and expert paging gain over a full
      swap and where expert paging earns its complexity; it adjusts M4/M5
      scope rather than gating viability. Time-box a first cut (one MoE, one
      budget sweep) before the full protocol. This may finish in early M1
      if reference setup requires it;
      reference runs do not require jitLLM's model implementation or trace
      recorder.
- [ ] Agree switching-benefit and generation-stall criteria before M2 from
      the measured reference experiment and feasibility evidence. These
      govern M4/M5/M7 acceptance for named workloads; no numeric thresholds
      are assumed in this plan.
- [ ] Re-inventory the Spark-to-Spark direct link once the QSFP/NCCL cable
      is installed (expected 2026-09-21): link state, RDMA devices, NCCL
      version, and the bandwidth a plain host-buffer transfer achieves
      between `spark` and `spark-b`. Record in architecture.md. Sharded
      execution in M6 waits on this; placement and request routing can use
      the existing network.
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
      tracking, and all backend allocations accounted for. Match reference
      logits, then repeat after eviction and restoration of weights and
      state at a completed boundary on a Spark. Exercise cancellation with
      pending work. Use the result to settle internal interfaces before M3;
      the operation contract is settled from this proof, not from the fake
      backend alone (there is no runtime plugin ABI, D-028).
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
      The immutable data reuses a known aligned container and jitLLM owns
      the manifest and resource index (settled 2026-09-21); pick the
      container here. Compatibility guarantees wait for dense and MoE
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
not on a checklist reaching zero. Both Sparks are reachable now
(`spark`, `spark-b`), so the three hardware
spikes can start. Paging feasibility also needs a pinned reference; a
sharded two-node reference run waits on the direct link, as does M6's sharded
execution work.

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
  D-025's measured reference cycle and the agreed criteria. Teacher-forced
  numerics stay correct after weight/state restoration. Branching histories,
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
  protocol; small-MoE numerics remain correct after eviction and restoration.
  With M4's dense evidence, assess artifact compatibility guarantees in a
  separate decision (D-018).
- **M6 — Sharded model execution (two Sparks here).** Build on M4a's
  configured cluster; add explicit sharding for the flagship, coordinated
  admission, stable communication buffers, and ordered collectives. Needs
  the direct Spark-to-Spark link (cable expected 2026-09-21) and the relevant
  single-node model/paging evidence. *Gate:* both ranks remain correct under
  asymmetric pressure, cancellation, and controlled failure; no timeout is
  treated as proof of reclaimed memory. Placement-only use already works at M4a.
- **M7 — Performance and product.** Alternative compatible kernels and plans,
  prefetch, selective CUDA graphs, dashboard, optional API extensions
  (sessions, hints; D-022),
  packaging as signed apt packages for Spark with notices (D-027). *Gate:* measured results meet the agreed workload
  benefit and generation-stall criteria; matched-configuration and normal
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
