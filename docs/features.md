# Feature matrix

The scope ledger for the M0 planning conversations:

- **Confirmed** — stated project scope from the owner's design brief
  ([ideation.md](ideation.md)) and subsequent owner-approved decisions and
  review fixes. Milestone assignment happens in [plan.md](plan.md) as the plan
  firms up.
- **Proposed** — candidate additions awaiting triage by the project owner.
  This covers both approaches the brief itself labelled "proposed design" and
  agent suggestions (marked *agent-suggested*).
- **Deferred** — a candidate parked with a concrete revisit trigger and an
  earliest milestone. It does not block M0 or become accepted automatically
  when the trigger occurs. Confirmed scope can also have a deferred delivery
  date or trigger without being removed from scope.
- **Open questions** — things that shape architecture and need an answer
  during M0 or an explicit deadline before dependent work begins.

Status legend: `confirmed` · `proposed` · `deferred` · `open` · `rejected`

Triage may confirm, reject, or defer a proposal. Rejection and deferral need a
brief reason; only load-bearing decisions need a D-NNN entry. Record a
deferred item's trigger and earliest milestone here or in plan.md.

A confirmed feature is scope, not a support promise: model support is earned
per checkpoint and configuration and tracked in a support matrix (see
"Testing and evidence").

## Model switching and sessions

The primary workload (D-019): one user, or an agent plus subagents, switching
among a library of models larger than memory, with conversations that last
hours. State is retained within D-024's bounds, with safe recomputation from
client-supplied history when an idle cache entry is unavailable.

| Feature | Status | Notes |
| --- | --- | --- |
| Model library larger than memory, loaded on first request, evicted across the whole library | confirmed | D-019 |
| Time-sliced (TDMA-style) execution under memory contention; concurrent execution when a supported placement fits the working sets and complete execution envelopes within each node's budget | confirmed | D-019, D-020; aggregate pool capacity alone is insufficient; switch points are quiescent boundaries; concurrency-when-it-fits is required soon after v0 |
| Single front door: clients talk to the conductor, the cluster's one point of entry, which routes each request to a node running that model | confirmed | D-020, D-022, D-023 |
| Cluster topology discovered or configured at runtime; no node names, counts, or roles baked into the application | confirmed | D-023; M4a starts with configured membership and one configured conductor; discovery and election delivery have separate triggers in plan.md |
| Multiple replicas of the same model across nodes when concurrent demand on a small model justifies it; the conductor balances across replicas with session or prefix affinity | confirmed | D-023; delivery deferred until measured overlapping demand justifies replication after M4a; affinity follows compatible retained state |
| Conversation state preserved across a model switch (KV, compressed, or recurrent state kept resident, spilled to SSD, or reconstructed) | confirmed | D-019, D-024; M4 proves retained-state reuse and correct fallback separately; idle cache retention is bounded |
| Switch-latency metrics: A-to-B first token, B-to-A resume with state, bytes moved per switch, and whether the concurrent or time-sliced path was taken | confirmed | D-019 headline metrics; reported per the comparison protocol |
| Compatible prefix reuse for standard clients; optional explicit session ID and release for cooperating clients | confirmed | D-022 as amended by D-024; prefix identity does not establish session lifetime or a "will resume" guarantee |
| Bounded cache-memory, spill and metadata use; expiry and safe recomputation when reusable state is unavailable | confirmed | D-024; active/suspended admitted requests remain protected; defaults chosen before M4 from measurements |
| Incoming-model prefetch at switch time (load B's non-expert weights and hot experts while A finishes its step) | deferred | *agent-suggested.* Revisit in M7 after M4's measured timeline shows overlap opportunity and spare capacity; request identity is known, timing is speculative |
| Client warm hints as an optional extension (an orchestrator announces a model it is about to use; optional pin or priority) | confirmed | D-022; ignorable by standard clients |
| Conductor places models, or parts of models, per node and routes requests; placement preferred over paging when it suffices | confirmed | D-020; whole-model placement is M4a, independently of M5; sharding is M6. The subagent's model on another node while the main model stays resident needs only the network |

## Runtime core: catalog, reservations, leases

| Feature | Status | Notes |
| --- | --- | --- |
| One native execution process per node managing all local models | confirmed | D-005 |
| Node-wide resource catalog: typed IDs, generations, dependency closure, shared extents charged once | confirmed | D-006; descriptor field groups in ideation §4 |
| Explicit CUDA VMM backing (reserve / create / map / access / unmap via the driver API) | confirmed | D-006 |
| Capacity reservations separate from residency leases; request transaction vs execution lease | confirmed | D-007; initial progress policy (question 9) is required before M2 |
| Lazy commitment: grants never eagerly evict useful cache | confirmed | D-007 |
| Separate commitment and occupancy ledgers | confirmed | D-007 |
| Completion service tracking GPU, I/O, and network consumers before reclaim | confirmed | ideation §3, §14 |
| Resumable continuations: suspend a model phase while I/O is pending and run other ready work | confirmed | ideation §7; number and size of suspended phases is bounded |
| Per-class lifecycle policies (immutable weights, routed experts, dense/attention weights, sparse lookup tables, live KV/state, reusable prefix state, scratch, graph objects, comm buffers, staging) | confirmed | ideation §5 table is the initial policy set. Qwen3.8-Flash-Next's 51B n-gram embedding is the first concrete sparse-lookup component to page by rows |
| Architecture-specific adapters for live and reusable state (KV blocks, compressed attention, sliding window, recurrent) | confirmed | conservative semantics per architecture first |
| Prefix-cache metadata always consistent with physical eviction (no stale hits) | confirmed | pager invariant 4 |
| Deterministic simulated (fake) resource backend for tests | confirmed | Stage 1 deliverable in ideation §19 |
| Conceptual native API: `register_resource` / `reserve_capacity` / `acquire_group` / `submit` / `retire_completed` / `reclaim` / `cancel` | proposed | ideation §20 sketch; types and async primitives to design in M0 |
| Physical-backing pool to amortize allocation overhead | proposed | ideation §8; retained capacity stays in the ledger; sized by the M0 VMM spike |
| Turn/step-scoped leases with eviction only at scheduler-established completion boundaries as the v1 lease model | proposed | *agent-suggested* candidate answer to open question 9. A switch request alone establishes no quiescence; consumers must complete and suspended live state stays protected. Applicability to MoE depends on the progress proof for within-step misses |
| Core free of vendor types; device memory, paging, and transfer operations behind narrow provider interfaces, CUDA VMM the first and only implementation | confirmed | D-026; only where it adds no complexity or penalty on NVIDIA |
| Ledger keyed by memory domain (one domain on unified-memory platforms) so a discrete-GPU platform is a data difference, not a redesign | proposed | *agent-suggested*, only if it costs nothing (D-026) |

## Eviction and retention policy

| Feature | Status | Notes |
| --- | --- | --- |
| Global cross-model victim selection at extent granularity | confirmed | D-008; `shrink(model, N)` may exist as a convenience, not as the boundary |
| Release changes eligibility, not residency; hysteresis, minimum useful residency, reload-cost awareness | confirmed | ideation §9 |
| Real use tracked separately from prefetch and cancelled planned use | confirmed | |
| Per-model statistics combined with global comparison so no model monopolizes reclaimable bytes | confirmed | fairness |
| Compare global LRU vs frequency/recency vs cost-aware heuristic on identical recorded traces | confirmed | report miss bytes, reload repetition, write volume, waiting time — not hit count |
| Routing/access trace capture and offline replay simulator | confirmed | reference-engine feasibility spike in M0/early M1 before M2; native recording and validation in M5; ideation §7, §9 |
| Cost-aware eviction heuristic (`eviction_cost_per_reclaimed_byte`) | proposed | ideation §9 calls it a proposed first heuristic, not a measured predictor |
| Brief wait for an imminent completion instead of evicting expensive contents | proposed | ideation §9 says "consider" |
| Trace-learned expert co-occurrence prefetch | deferred | *agent-suggested.* Revisit in M7 after M5 traces show predictable misses and an overlap window; compare against demand-only execution including unused prefetch bytes |
| Residency warm-start across runtime restarts (persist heat/working-set metadata, re-warm on start) | proposed | *agent-suggested.* Otherwise every restart pays full cold paging; metadata only, never live state |
| Dependency-group value functions for eviction scoring | deferred | *agent-suggested.* Revisit after the M4 baseline when trace replay demonstrates avoidable reload cost caused by ignoring dependency coupling; preserve extent-level reclamation and lifetime rules |

## Storage and I/O

| Feature | Status | Notes |
| --- | --- | --- |
| Common internal read/write completion interface across backends | confirmed | ideation §8 |
| Spark staged path: SSD → pinned host staging → CUDA copy → mapped backing (and the reverse for write-back) | confirmed | D-004 |
| Bounded, explicitly budgeted staging pool reserved before pressure | confirmed | never allocate unbudgeted RAM in order to evict RAM |
| Explicit handling of short transfers, checksum errors, storage exhaustion, alignment, retries, cancellation; bounded queues | confirmed | |
| Coalesce duplicate loads for the same content generation | confirmed | |
| Native file I/O + pinned staging backend (queue depth, priority, cancellation control) | confirmed | one of the two initial Spark candidates |
| cuFile compatibility-mode backend | confirmed | as a benchmark comparison path; whether it ships is the M0 I/O spike's call |
| Buffered vs direct-I/O comparison; page-cache duplication and read amplification measured | confirmed | no system-wide cache flushing as runtime policy |
| Write-back only when preservation requires it; clean weights are never written | confirmed | |
| Native GDS backend on hardware where the direct path is supported | proposed | later; not Spark |
| Remote extent transfer between nodes (logical object/version/extent over a supported transport) | proposed | later; source stays leased until completion; never remote `cuMemMap` |
| Optional crash durability for spill as a separate policy | proposed | ideation §8 distinguishes it from same-process retention |
| Spill encryption at rest | proposed | *agent-suggested.* One concrete way to meet "protect spill files" (D-014) when spill may hold KV/state derived from prompts |
| Direct I/O (`O_DIRECT`) as the default read path on unified memory; buffered reads only where measured to be harmless | proposed | *agent-suggested.* On Spark the page cache and GPU-mapped memory are the same DRAM, so a buffered read double-occupies memory the catalog thinks it owns; this also drives artifact alignment rules |
| Test whether the staging copy is needed at all on Spark: GPU in-place access to system-allocated memory read straight from NVMe | proposed | *agent-suggested* addition to the I/O spike; if in-place access performs acceptably per the Spark porting guide's memory section, the read path loses a copy |
| Sustained-read thermal behaviour and a spill-write budget for the single NVMe | proposed | *agent-suggested.* Measure sustained throughput over minutes, not seconds; reads do not wear the drive but KV spill writes do |

## Model import and prepared artifacts

| Feature | Status | Notes |
| --- | --- | --- |
| Owned import pipeline: validate → select layout → pack/shard → index → hash → atomic publish | confirmed | D-009 |
| Versioned, hashed artifacts supporting bounded range reads without reprocessing | confirmed | D-009, D-018; experimental initially, with explicit rejection of incompatible versions; compatibility guarantees follow dense and MoE restore evidence |
| Checkpoints treated as untrusted input; no code execution; lengths, paths, hashes, metadata validated | confirmed | D-009 |
| Resumable import; free-space and peak-temp checks; interrupted imports never appear valid | confirmed | |
| Workstation-side import; target-assisted tuning as an explicit mode with separately keyed results | confirmed | x86 importer, ARM importer, and architecture-independent artifact format are distinct things |
| Immutable model files separate from mutable spill files | confirmed | |
| Artifact contents: manifest, tokenizer/config, representation catalog, resource index, immutable data, integrity/provenance, optional plan metadata | proposed | ideation §11 "proposed artifact contents"; the encoding is open question 5 |
| Multiple backend layouts per artifact | proposed | only when measured value justifies the disk and import cost |
| Reuse a known container for the immutable blobs (GGUF- or safetensors-style aligned tensor data) and own only the manifest and resource index | proposed | *agent-suggested.* Re-packing experts into contiguous aligned extents is justified; inventing a container is not. Keeps tooling available while D-018 keeps the format experimental |
| Standalone artifact verification tool (checksums, index bounds, manifest consistency) | proposed | *agent-suggested.* Cheap given per-extent checksums; separates "bad artifact" from "pager bug" during bring-up |
| Model support matrix per checkpoint: unsupported → import-only → resident-correct → paged-correct → distributed-correct → performance-validated | confirmed | ideation §19 |
| Direct model download from the Hugging Face Hub in the importer and the management API | confirmed | owner request 2026-09-20; downloads are resumable and verified like any import input (D-009) |
| Hugging Face token from a `.env` or config file tied to the user's HF account, also settable from the web management UI | confirmed | owner request 2026-09-20. Secret handling: never logged, restricted file permissions, `.env` git-ignored, management stays local by default (D-014). Gated repositories download only with a token whose account already has access; the tool cannot grant it |

## Compute backends and execution

| Feature | Status | Notes |
| --- | --- | --- |
| Reuse of kernels, algorithms, and model semantics from vLLM, llama.cpp/GGML, ExLlamaV3/EXL3, FlashInfer, CUTLASS/CuTe under their licenses | confirmed | D-013; which units, per model, is open question 4 |
| Keep a fully resident fused plan and a pageable split plan where both are useful | confirmed | ideation §7, §10 |
| Lossless layout transforms separated from quantization/precision changes (the latter need explicit quality evaluation and metadata) | confirmed | |
| CUDA graphs with dynamic residency decisions outside captured segments; no CUDA API calls from host-function nodes | confirmed | initial approach; ideation §7 |
| One deliberately managed CUDA context per GPU; explicit streams and library handles | confirmed | initial |
| Backend operation contract (declares architectures, layouts, quantization, state, workspace, dependencies, graph restrictions, completion) | proposed | ideation §10 "proposed operation contract" |
| Early backend integration proof using jitLLM-owned memory, explicit workspace and completion, then eviction and restoration | confirmed | Run alongside M2 before settling the internal contract; small dense model, prepared artifact, reference-logit comparisons on a Spark; informs M3 without freezing a plugin ABI |
| Versioned C ABI for optional separately built backends | proposed | ideation §14. Optional implementation modules (D-017) require a fully removable boundary; whether that is a build-time module or a runtime plugin ABI is the open part, and not before the first real backend exposes its requirements |
| Triton AOT as an optional build-time kernel route | proposed | ideation §10; keep provenance of generated code |
| GPU-visible residency table plus compact miss notification | deferred | Earliest M7, after M5 demonstrates material host-boundary overhead; a Boolean check without protection against revocation is unsafe |
| Executing ready experts while other experts load | proposed | later; not an assumed capability |
| Speculative decoding | proposed | not committed; benchmarks require matched decoding features plus the reference's normal configuration, even when its speculative decoding is unavailable in jitLLM. Qwen3.8-Flash-Next ships a 4B MTP head, so its matched comparison must state whether MTP is used |
| Optimistic MoE execution: device-visible residency table, kernels flag a miss, restart from the missed layer | deferred | *agent-suggested.* Earliest M7, after the pessimistic M5 baseline is correct and measured host-boundary cost warrants it; must prove safe revocation, replay of mutable state, and progress when a step's leases fill memory |
| GGML/GGUF as the first compute substrate, with jitLLM supplying the buffers behind tensors; EXL3 kernels ported later for the flagship recipes | proposed | *agent-suggested* framing for open question 4. GGML is MIT, torch-free, has a C API and broad quant and tokenizer coverage; the reference recipes are EXL3 and torch-bound. Pick deliberately GGML also runs on Metal, ROCm/HIP, and Vulkan, so a later port would be mostly a memory-provider job (D-026). |

## Two-node execution

| Feature | Status | Notes |
| --- | --- | --- |
| Conductor placement and request routing across nodes, ahead of sharding | confirmed | D-020, D-023; M4a uses the existing network after M4 and does not depend on M5 |
| Explicit sharding across nodes for the flagship; the conductor issues distributed phase IDs | confirmed | M6, ideation §12; needs the direct link |
| Prepare/commit admission across ranks for sharded models; no rank enters a collective while another can wait indefinitely on an unapproved allocation | confirmed | placement-only execution does not need it |
| Separately budgeted, stable communication-buffer pool honouring NCCL registration and threading contracts | confirmed | |
| Ordered collective submission; completion fences for GPU and network consumers | confirmed | |
| Port the validated target recipe's parallelism first (TP, PP, EP are different plans) | confirmed | |
| Remote paging (see Storage) | proposed | later |
| Master-decides admission with a mirrored replica ledger instead of full prepare/commit for the two-node case | proposed | *agent-suggested* simplification; prepare/commit stays the general design if a third node ever appears |

## Management, inference API, and diagnostics

| Feature | Status | Notes |
| --- | --- | --- |
| CLI / status endpoint and structured events first | confirmed | ideation §13; a dashboard is not a prerequisite for validating the pager |
| Versioned local management API: import/list/remove, representation inspection, priorities, residency policies, cancellation, node health, trace capture | confirmed | budget changes are scheduler requests; removal quiesces users |
| Explainable decisions: memory breakdown, model working sets, request state, eviction decisions (victims, expected cost, bytes recovered, why alternatives were kept), I/O timeline, backend choice | confirmed | ideation §13 views; explainable scheduling is a stated priority |
| Local-only binding by default; auth and transport protection for remote; no prompt/KV logging by default; opaque request IDs | confirmed | D-014 |
| Streaming inference API compatible with standard web-API clients (Cursor, OpenCode, Codex named) | confirmed | D-022; endpoint set verified per client in M0/M1 |
| Web dashboard as a separate process over the management API | confirmed | later (Stage 6); must not own the scheduler or take runtime locks |
| OpenAI-compatible HTTP surface (chat completions at minimum) | confirmed | D-022 baseline; Responses API and streaming/tool-call details verified per named client |
| Anthropic Messages API format alongside the OpenAI-compatible surface | proposed | *agent-suggested.* Several agent clients speak it; include only if a named client needs it Athena's Engine offers both flavours, mild evidence that Spark users expect it. |
| Admission "explain / what-if" query (why can't this request be admitted now; what would need to be evicted) | proposed | *agent-suggested.* Natural extension of explainability and a debugging tool for progress-envelope bugs |
| Trace export in Perfetto / Chrome trace-event format for the I/O timeline and scheduling | proposed | *agent-suggested.* Structured events are confirmed; a standard viewer format avoids building a timeline UI early |
| Target capability probe tool (VMM granularity, GDS mode, RDMA availability, driver/toolkit versions, glibc/ABI) | proposed | *agent-suggested.* The brief says "probe the installed stack"; a first-class tool serves the M0 inventory and the later `doctor` command Platform properties are probed capabilities, not constants (D-026). |
| Per-model memory quota and priority policy (minimum guarantee, maximum share) | proposed | *agent-suggested.* Fairness is confirmed; explicit knobs give the owner control over the flagship-vs-small-model balance |
| Model version hot-swap (publish a new artifact version, drain the old, no runtime restart) | proposed | *agent-suggested.* Removal-with-quiesce is confirmed; this is the add-then-drain composition |

## Toolchain, build, and development environment

| Feature | Status | Notes |
| --- | --- | --- |
| C++23 host runtime, Clang-first; NVCC with Clang host compiler where validated; pinned libstdc++ initially | confirmed | D-010 |
| Cross-compile from x86-64 to Spark; deploy and test over SSH; explicit targets only | confirmed | D-011 |
| `mise.toml` + `mise.lock` for tool setup, environment, and tasks | confirmed | D-012 |
| Project-owned SDK provisioning; native Ubuntu and a reference dev container from the same logic | confirmed | D-012 |
| Copyleft-components-disabled CI profile with audited dependency closure | confirmed | D-002, D-017; excludes optional implementation dependencies and records declared tools/platform runtimes separately |
| Optional implementation modules/plugins selectable at build time; incorporated core implementation uses Apache-2.0 / BSD / MIT / MPL-2.0 | confirmed | D-017; default build may use declared platform dependencies under their actual terms; classification never waives license obligations |
| Reference container pinned by digest; target driver recorded separately from toolkit and library versions | confirmed | ideation §16 |
| Core builds and its tests pass in a CPU-only configuration with no vendor SDK present | confirmed | ideation §16, D-026; the portability guardrail and the fake backend's home |
| CMake presets + Ninja + `compile_commands.json`; LLD where validated; pinned LLVM format/analysis tools | proposed | ideation §14 "proposed engineering conventions"; the obvious default, to confirm in the M0 toolchain decisions |
| Proposed file set: `toolchains/manifest.toml`, `toolchains/artifacts.lock.json`, `tools/setup-toolchain`, `tools/check-toolchain`, `cmake/toolchains/`, `CMakePresets.json`, `.devcontainer/`, `dev` | proposed | ideation §16: "a plan, not files created" |
| `./dev setup / doctor / build / test / deploy` contributor entry point | proposed | ideation §16 "to implement" |
| REUSE-style file-level SPDX identifiers; NOTICE file; SBOM | proposed | ideation §17 "proposed compliance mechanics" |
| Checked-in benchmark trace corpus with regression thresholds in CI | proposed | *agent-suggested.* Benchmark metrics are confirmed; gating on recorded traces catches policy regressions without Spark time for every change |
| Native-on-Spark CMake preset kept as a fallback and diagnostic build alongside the cross build | proposed | *agent-suggested.* The owner reaffirmed cross-compiling as primary (D-011): the C++ side is routine and the toolchain is needed anyway. The one piece with real uncertainty is NVCC with a cross Clang host compiler; a native preset costs little and keeps the first token unblocked if that drags |

## Release and project surface

| Feature | Status | Notes |
| --- | --- | --- |
| Shipped notices and source availability match the actual build configuration (core vs. enabled optional modules) | confirmed | ideation §17, D-017; includes any shipped platform components; a build that self-reports its license profile is the obvious mechanism |
| Versioned releases with a changelog and a compatibility policy for the artifact format and management API | proposed | *agent-suggested.* Release conventions are an M1 decision; artifact compatibility guarantees additionally require D-018's dense and MoE evidence |
| Contribution policy: whether external PRs are accepted; DCO or CLA | open | flagged in ideation §21 alongside the license; single-developer today (D-016) |
| User installation through native package managers: a project-hosted, signed apt repository with arm64 packages for Spark first | confirmed | D-027; the user path, distinct from the developer setup path (D-012) |
| Optional copyleft modules as separate packages in a separate repository component, mirroring D-017's tiers | proposed | *agent-suggested.* apt components make the license profile a visible install choice and keep the default install to the core |
| systemd unit, non-root service user, FHS layout (config under `/etc`, state and artifacts under a configurable data directory), drain-before-restart upgrades | proposed | *agent-suggested* consequences of D-027; settle the layout before M3's endpoint so paths do not move later |
| CI builds installable `.deb` packages from M1, before the repository is published | proposed | *agent-suggested.* Late packaging is where notices, paths, and dependencies go wrong |
| Homebrew and other package managers | deferred | follow their platforms (D-026, D-027) |
| Inventory which MiaAI-Lab files are actually AGPL versus MIT ExLlamaV3 upstream before designing the optional-module boundary; document AGPL's network clause for a served process | proposed | *agent-suggested.* The AGPL exposure may be a small glue and patch set |

## Testing and evidence

| Feature | Status | Notes |
| --- | --- | --- |
| Layered tests: native CPU, deterministic simulated backend, single-Spark CUDA, model semantics, multi-model pressure, two-node, packaging/licensing | confirmed | ideation §18 |
| Mandatory pager invariants (eight, listed in [architecture.md](architecture.md)) enforced by tests | confirmed | |
| Numerical references against pinned known-working engines with documented tolerances; teacher-forced logits and intermediates, not generated text | confirmed | |
| Benchmark set: warm decode rate, inter-token latency distribution, bytes read/written per token, exposed stalls, peak occupancy, mixed-model throughput, per-model waiting/fairness, cold and partial-resume latency, switch and switch-back latency with bytes moved per switch (D-019) | confirmed | full provenance (artifact, backend, toolchain, driver, hardware, policy) with every result |
| Resident-hit path measured independently from the miss path; cold storage vs warm OS cache vs warm residency separated | confirmed | |
| Canonical A→B→A through an unmodified client, with resident reuse, forced spill/restore, and bounded-cache fallback cases | confirmed | M4's first useful product gate; report elapsed time, bytes read/written, reused/recomputed prompt tokens, and numerical checks |
| Paging feasibility assessed before M2 against the full-swap floor; matched-configuration and normal reference-configuration comparisons | confirmed | [performance evidence](architecture.md#performance-evidence); D-021/D-025: first-cut estimates, then a measured reference A→B→A once setup runs; spike sizes M4/M5, M4 validates switching and M5/M7 validate paging/optimizations |
| Athena's Engine as a closed-source comparator for the normal-reference view (46 s full swap and 2.1 s context restore on one GB10, creator-reported) | proposed | *agent-suggested.* Nothing reusable; numbers include speculative decoding; install on a Spark only if its terms allow and only as a comparator The 46 s includes checkpointing the active session; the pair does not both fit in 128 GB. |

## Platforms

| Platform | Status | Notes |
| --- | --- | --- |
| NVIDIA DGX Spark, one or more nodes | confirmed | D-004; the primary target |
| Other NVIDIA CUDA hardware for development and tests | confirmed | the workstation's RTX 3080 Ti for local smoke tests; supported-target status is earned separately and may use more direct transfer paths |
| Apple silicon, single machine | deferred | D-026: not until demand; boundaries kept portable at no cost; memory API verified when a port is considered |
| AMD, single machine | deferred | D-026: same posture |

## Model targets

| Target | Status | Notes |
| --- | --- | --- |
| GLM-5.3-Flash, DeepSeek-v4.1-Flash, Qwen3.8-Flash-Next (via the MiaAI-Lab two-Spark references) | confirmed | as target families and reference recipes; pinned before porting; support earned per checkpoint |
| ~30B-class dense models (suitable Gemma / Llama variants) | confirmed | intended use case, not a promise for every checkpoint |
| Small dense model plus synthetic/tiny MoE as the first bring-up vehicles | confirmed | ideation §19: separate execution, import, and pager bugs before a flagship architecture |
| First vertical-slice checkpoint and backend | open | open question 4 |
| Reference engine for the feasibility spike: llama.cpp with MoE GGUFs | confirmed | decided 2026-09-20. Owner-provided candidates with card-verified configs in plan.md: Qwen3.8-Flash-Next (512 experts, top-10 plus 1 shared, 6B active of 125B, plus a 51B n-gram table; 3-bit fits one node, 4-bit is borderline, 5-bit exceeds it), Gemma 4 26B-A4B (128 experts, top-8 plus 1 shared, hybrid sliding-window attention), Ornith-1.5-35B-A3B (`qwen35moe`). This does not decide the runtime substrate (open question 4) |

## Open questions (answer during M0)

Ordered by how much work a late answer would invalidate. Lesser open items
from ideation §21 (user-space baseline on the Sparks, graph integration proof,
public API scope) ride along as M0 tasks or later-milestone questions.

1. **VMM extent granularity, map/unmap cost, and physical-pool strategy on the
   real Spark driver.** Every pager data structure sizes itself on this.
   → M0 spike "VMM microbench" in [plan.md](plan.md) (needs a Spark). Also
   settle the initial extent policy; dependency-group scoring is deferred
   until the Eviction row's measurement trigger.
2. **Storage I/O path.** cuFile compatibility mode vs native file I/O with
   pinned staging vs direct I/O, measured under concurrent compute and memory
   pressure, including page-cache duplication. Decides the storage service
   design and staging budget. → M0 spike "I/O path comparison" (needs a
   Spark). The spike also tests direct I/O as the default and GPU in-place
   access to system-allocated memory (see the Storage rows).
3. **Async/task and completion model.** Hand-rolled executor with explicit
   continuations, C++20 coroutines, a sender/receiver library, or something
   else. C++23 does not supply the scheduler, and every interface signature
   depends on this. → M0 decision, prototyped against the fake backend design.
4. **First vertical-slice model and backend.** Which checkpoint (revision,
   quantization, tokenizer, kernels, provenance) and which numerical reference
   engine. Decides M3 and gives the license audit its first real inputs. → M0
   decision. Candidate framing: GGML/GGUF-first for a working system versus
   EXL3-first for the flagship recipes (see the Compute rows). Prove the
   selected backend can use jitLLM-owned memory alongside M2 before settling
   the internal contract.
5. **Experimental artifact schema and layout ABI.** Choose metadata encoding,
   alignment, integrity, sharding representation, and version rejection rules
   for bring-up. → M0 decision, likely after question 4. D-018 defers
   compatibility guarantees until a dense model and a small MoE have each
   passed import, execution, eviction, and restoration checks. Experimental
   revisions may require explicit re-import. Also decide whether the blob
   container is reused or bespoke (see the Artifacts rows).
6. **Exact toolchain pins validated as one unit.** LLVM version, libstdc++,
   CUDA toolkit, ARM sysroot, GB10 architecture spelling, Clang as host
   compiler in the cross configuration. Blocks M1. → M0 spike "toolchain
   smoke" (workstation for the build, one Spark for the run).
7. **C++ source-dependency mechanism.** vcpkg, Conan, CPM/FetchContent,
   submodules, or vendoring, independent of toolchain provisioning. Blocks M1
   and interacts with the copyleft-disabled profile. → M0 decision.
8. **Inference API surface.** Resolved 2026-09-20 by D-022: standard web-API
   compatibility (Cursor, OpenCode, Codex) is the baseline, with sessions and
   hints as optional extensions. Remaining M0/M1 task: verify the exact
   endpoint set each named client needs. (The license half closed the same
   day: Apache-2.0 in D-003, dependency policy in D-017.)
9. **Initial reservation guarantee and progress envelopes.** How conservative
   the first scheduler is about serializing phases, and what a "minimum
   feasible phase" envelope includes (activations, state growth, scratch,
   staging, comm, graphs, metadata). Include guaranteed versus opportunistic
   grants, retained continuations, growth limits, and impossible-phase
   handling. → M0 decision, or explicit deferral only until before M2; see
   the [progress gate](architecture.md#reservation-progress-gate). A
   conservative default is acceptable and gets tested in M2 and tuned in M5.
   Candidate: turn/step-scoped leases with eviction only at completed
   boundaries established by the scheduler (see the Runtime core rows).
   Under D-020, v0 has one active phase per node plus suspended state,
   which bounds the envelope problem.
