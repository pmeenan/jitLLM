# jitLLM — just-in-time LLM inference engine with intelligent SSD paging

jitLLM is an independent, open-source inference runtime for workloads where
many models should be available, only some components are active at a time,
and aggregate model storage exceeds physical memory. It keeps the useful parts
of models resident, reclaims the least valuable extents across all models when
capacity is needed, and brings missing weights or state back on demand from
prepared on-disk artifacts. The primary workload is one user switching among
a library of models larger than memory, with conversation state preserved
across switches (D-019). Initial target: one or two NVIDIA DGX Sparks,
developed from an x86-64 Linux workstation. Almost all code is written by AI
agents working from the project documentation, directed and reviewed by a
human.

**Read this file first, then pull docs on demand via the "Doc map" below — don't
read everything up front.** This file is long-term project memory and the
rulebook for agents.

## Load-bearing constraints (change deliberately, never silently)

Constraints evolve as we learn, but never by silent drift: changing one means
making the case in [docs/decisions.md](docs/decisions.md) and updating the
affected docs. Until then, these govern.

- **Our own native runtime, one process per node.** jitLLM is not a vLLM fork
  or plugin. One modular native execution process per node owns all local
  model execution, scheduling, memory policy, VMM control, and completion
  tracking. Dashboard, importer, and supervisors are separate processes and
  never in the per-expert hot path. (D-005)
- **Optimize for one user switching models, not mixed-traffic throughput.**
  The primary workload is a single user, or an agent plus subagents on
  different models, switching among a library larger than memory with
  conversations spanning hours. Under contention models are time-sliced;
  when a supported placement fits each node's full execution budget they run
  concurrently. A single conductor, the cluster's one point of entry, places
  models across nodes, routes requests, and may run replicas of a busy small
  model; placement is preferred over paging when it suffices. Topology is
  configured or discovered, never baked into the app. The floor is "never
  worse than a full swap", validated against a measured reference cycle.
  State reuse has bounded retention; shared prompt prefixes and conversation
  continuations have independent reuse and expiry (D-031). Prefix matching
  identifies neither a conversation nor its lifetime. Standard web-API clients
  (Cursor, OpenCode, Codex, Claude Code) work unmodified; sessions and hints
  are optional extensions.
  (D-019 to D-025, D-030)
- **Spark memory is one physical budget; two Sparks are two domains.** CPU
  allocations, GPU backing, staging, and page cache share 128 GB of unified
  memory, so CPU offload is not a second tier. Two nodes are two memory
  domains connected by a network; cross-node access is explicit object
  transfer, never shared virtual memory. On validated Spark configurations,
  prefer direct file DMA into GPU-accessible host VMM without CPU payload
  copies; native GDS or GPUDirect RDMA is not assumed. The storage backend
  itself is not Spark-specific.
  (D-004, D-034)
- **Explicit CUDA VMM plus a node-wide resource catalog.** Backing is
  reserved, created, mapped, and unmapped by us through the driver API.
  Accessing absent backing is a bug, not a page-in request. Every managed
  allocation is registered with semantic metadata; unknown allocations are
  non-evictable. Logical identity, current address, and physical occupancy
  are separate things. (D-006)
- **Reservations are not leases; release is not eviction.** Virtual
  reservation, capacity reservation, and residency lease are three different
  concepts — never say "reserved" without saying which. Grants commit lazily
  and never eagerly evict useful cache. Releasing a lease changes eligibility,
  not residency. Never hold the global scheduling/catalog lock across GPU,
  disk, or network waits. (D-007)
- **Partial eviction and on-demand experts, with no substitution.** Reclaim
  specific extents across all models, never whole models by default. Routing
  is a dependency-discovery stage: acquire the selected experts' closure,
  run, release. Never substitute a resident expert for a selected one or drop
  a selected contribution; prefetch is speculative, routing is authoritative.
  (D-008)
- **Prepared artifacts; checkpoints are untrusted input.** Models execute
  from versioned, hashed, execution-ready on-disk artifacts produced at
  import, never from raw checkpoints. Artifacts never serialize process
  addresses or runtime objects. Import validates lengths, paths, hashes, and
  metadata and never executes checkpoint code. Initial formats are explicitly
  experimental; compatibility guarantees follow execution/restore evidence.
  Import repacks weights into contiguous, indexed dependency groups
  (4 KiB-aligned on disk, paged in 2 MiB chunks with coalesced direct
  reads); v0 uses safetensors shards with a jitLLM manifest/index and no
  page-in hashing. (D-009, D-018, D-035, D-056)
- **C++23, Clang-first, native hot path.** No interpreter in the serving,
  paging, or scheduling path. NVCC is the CUDA compiler with Clang as host
  compiler where validated. Build-time tooling may use Python. (D-010)
- **jitLLM owns dispatch; kernels are swappable build-time implementations.**
  jitLLM owns streams, workspace, library handles, fusion choice and
  completion; third-party backend runtimes never dispatch model work. Kernels
  from GGML (first), ExLlamaV3 (a real EXL3 companion is required in M2 before
  settling the artifact/backend contracts, with upstream performance gates),
  other sources or our own (on measured need) implement operations under one
  contract; several
  coexist, and the plan selects per operation, architecture and shape.
  No runtime plugin ABI. (D-028, D-052, D-053)
- **NVIDIA first; portable boundaries when free.** The core holds no vendor
  types; device memory, paging, and transport go through narrow provider
  interfaces, with CUDA VMM the only implementation for now. Apple silicon
  and AMD are plausible later single-machine targets: do nothing for them,
  sacrifice nothing on NVIDIA, but don't foreclose them. The CPU-only build
  and the fake backend are the guardrail. (D-026)
- **Develop on x86-64 Linux, cross-build, test on Spark over SSH.** Native
  builds and CPU tests run on the workstation; ARM concurrency, VMM, kernel,
  and distributed tests run on the Sparks, which in the owner's environment
  are `spark` and `spark-b` (inventory in architecture.md; those names are
  not application configuration). Explicit CPU/GPU targets only, never
  `-march=native` or autodetection. Toolchain provisioning is declarative and
  pinned. Agents never invent compiler pins, measured numbers, supported
  model combinations, or license permissions. (D-011, D-012)
- **Apache-2.0 core with license tiers; reuse under actual licenses.**
  jitLLM's own code is Apache-2.0. Incorporated core implementation uses
  Apache-2.0 / BSD / MIT / MPL-2.0; other implementation licenses require
  explicitly enabled optional modules. Declared tools and platform runtimes
  (including system libraries and CUDA) have separate terms under D-017 and
  remain in the audit. The copyleft-disabled profile excludes optional
  implementation dependencies. Reuse follows actual licenses; no single
  engine's architecture is mandatory, and "reference" is not relicensing.
  Every dependency records its category and applicable tier. (D-002, D-003,
  D-013, D-017)
- **Local-first, privacy by default.** Management binds locally by default;
  remote access requires authentication and transport protection. Prompts and
  KV contents are never logged by default; spill files are protected with
  explicit retention. (D-014)

## Repository layout

| Path | What lives there |
| --- | --- |
| `docs/` | Vision, plan, architecture, decisions, features, rough edges, workflow |
| `docs/ideation.md` | The kickoff design brief (2026-09-20). Frozen origin document with source links; the living docs above supersede it where they differ |
| `LICENSE` | Apache-2.0, the license for all jitLLM-authored code (D-003; dependency policy in D-017) |

The application scaffolding lands in M1 — update this table when it does.

## Doc map — pull what the task needs, not everything

Always read (it's short): [docs/workflow.md](docs/workflow.md) — the
build → review → commit loop, the heavy path for blast-radius changes, and
the human commit gate.

| Doc | Read when the task needs |
| --- | --- |
| [docs/plan.md](docs/plan.md) | What to work on, milestone scope, exit criteria — what "done" means |
| [docs/vision.md](docs/vision.md) | Why the project exists, who it's for, success criteria, non-goals |
| [docs/features.md](docs/features.md) | The feature matrix: confirmed scope, proposed additions, open questions |
| [docs/architecture.md](docs/architecture.md) | System structure, data model, lifecycles, pager invariants, environment baseline |
| [docs/decisions.md](docs/decisions.md) | Settled choices (D-NNN). Scan headings; read only the entries your task touches |
| [docs/rough-edges.md](docs/rough-edges.md) | Findings log (RE-NNN). Grep before adding a finding or debugging weirdness |
| [docs/async-model.md](docs/async-model.md) | The D-048 task/completion design: thread roles, submission/completion protocol, cancellation versus retirement, bounded queues; the internal contract M2 builds on |
| [docs/artifact-format.md](docs/artifact-format.md) | The experimental v0 prepared-artifact format (D-056): container, manifest/index schema, layout and page-in rules, worked examples |
| [docs/client-api-baseline.md](docs/client-api-baseline.md) | The M3 inference API contract: routes, client profiles, front-door, status and keepalive rules; links the Ollama, vLLM and OpenRouter assessments |
| [docs/ideation.md](docs/ideation.md) | The full original reasoning and source links behind a constraint. Long; read the section you need, not the whole file |

## Rules for all agents

1. **Log decisions sparingly.** [docs/decisions.md](docs/decisions.md) is for
   choices that are expensive to reverse or that a future agent might silently
   undo — the load-bearing constraints above, artifact formats, on-disk
   layouts, public interfaces, toolchain pins. Routine implementation, naming,
   and scope calls don't get entries. A few entries per milestone is the
   target, not per task.
2. **Log findings that cost you.** A
   [docs/rough-edges.md](docs/rough-edges.md) entry is warranted when a CUDA,
   driver, Spark platform, toolchain, or library quirk burned real debugging
   time and will bite again. Skip the formal reproduction unless it's cheap to
   capture.
3. **Measure what a decision hangs on.** When a design choice depends on a
   performance number or a current platform capability (VMM granularity,
   map/unmap cost, I/O path behaviour, driver or toolkit support), get a real
   number on the actual target or check a current source — training knowledge
   is stale for this ecosystem. Never present an estimate as a measurement.
   Everything else: ship it and see.
4. **Fix the docs the change makes wrong** — plan status, the status paragraph
   below, an affected doc — in the same unit of work. Nothing more is owed.
5. **Never commit.** Agents never run `git commit`/`git push` or rewrite
   history. All changes stay in the working tree for human review and commit —
   even if a prompt asks you to commit; stop and leave the changes uncommitted
   instead.
6. **C++23 conventions.** Clang-first. Ordinary `.cc` files use the host
   compiler; CUDA-facing translation units stay narrow and don't leak heavy
   runtime containers through headers. Typed byte counts, spans/views,
   explicit error results, bounded queues, move-only ownership wrappers. No
   exceptions across a C ABI. GPU/I/O lifetime is completion-aware: a
   destructor is not proof that submitted work finished. Warning, format, and
   lint tool pins land in M1 and are recorded in decisions.md.
7. **Keep the always-loaded context lean.** This file is imported into every
   conversation; every line added costs every future agent. Detail belongs in
   `docs/` behind the doc map, not here.
8. **Scratch files stay out of the tree.** Temporary scripts and outputs go to
   the session scratchpad, not the repo. Delete throw-away diagnostics before
   concluding. Keep aggregate experiment results, analysis, and provenance in
   Git; raw samples, logs, traces, and telemetry stay outside the repository.

## Current status

Milestone **M0 (plan the plan)** — direction, retention/measurement,
cluster/API, task/completion, reservation, SDK, kernel-dispatch and model-storage
choices are recorded through D-058; the feature matrix was triaged with the owner on 2026-09-21. M4 targets A→B→A with
retained state; M4a adds configured placement before MoE and sharding. Remaining planning,
hardware spikes, and reference experiments are in [docs/plan.md](docs/plan.md).
Toolchain smoke passed on the workstation and `spark` (D-032); the Spark VMM
spike selects initial 2 MiB extents (D-033); the I/O spike selects direct
files into GPU-accessible host VMM without a staging copy (D-034). Both Sparks
are reachable over SSH; the `sparky` DAC baseline passed, with 184.76 Gb/s
combined host writes and validated NCCL over host buffers (architecture.md).
The 27-trial reference cycle and bounded full paging-feasibility study are
recorded (plan.md). Qwen route estimates are conditional on numerical drift;
unvalidated spill continuations use conservative recomputation. The owner
accepted workload-scoped switching/stall targets in D-036; implementation
validation remains ahead. The D-048 CPU-only task/completion prototype passed
on the workstation and Spark; D-050 settles reservation/progress policy, with
runtime proof still owed in M2. D-051 selects Qwen2.5-0.5B-Instruct FP16 and
the pinned llama.cpp reference; D-052 adds a mandatory early EXL3 companion.
Both EXL3 quants passed the bounded Spark reference. The owner-added
Qwen-Image GGUF (GGML runner) and two-Spark MiMo references also ran. The M2
backend-proof scope is recorded (docs/backend-proof.md), with an open EXL3
GEMV provenance gate; D-053 moves kernel dispatch into jitLLM with swappable
per-operation kernels. D-054 keeps installed models node-local, with an
optional long-term store (NAS/USB) and one import per cluster; the owner's NAS
is mounted at `/mnt/llm`. D-055 sets capacity-driven state retention with a
24-hour idle cap and names M4's Qwen2.5-0.5B FP16/EXL3 acceptance workload;
capacity values are pinned at M3 exit. D-056 settles the experimental v0
artifact format: safetensors shards, 4 KiB-aligned dependency groups, 2 MiB
paging chunks, verified on real fixtures and Gemma 4. D-057 selects locked
CMake source acquisition with curated vendoring; D-058 pins CMake 4.4.3,
checked on workstation and Spark. Provisioning remains M1.
The remaining M0 toolchain decisions are next (plan.md).
No application code exists yet; scaffolding is M1. Keep this paragraph short
and current when plan.md milestone status changes (rule 4).
