<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

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
  builds and CPU tests, including AArch64 CPU tests under qemu-user, run on
  the workstation; ARM concurrency, VMM, kernel, GPU, RDMA/NCCL, and
  distributed tests run on the Sparks, which in the owner's environment
  are `spark` and `spark-b` (inventory in environment.md; those names are
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
| `LICENSE`, `LICENSES/`, `NOTICE` | Apache-2.0, the license for all jitLLM-authored code (D-003; dependency policy in D-017); the text of every license a file declares; the attribution notice. Every file carries SPDX tags in its header, or in a `.license` sidecar if it cannot hold a comment (D-029, D-071) |
| `CHANGELOG.md` | Keep a Changelog; a change with user-visible effect adds its line (D-062) |
| `mise.toml`, `mise.lock` | mise tasks (`setup`, `prepare`, `doctor`, `build`, `test`, `deploy`) and the pinned Python that runs `tools/` (D-070) |
| `toolchains/` | The SDK manifest, artifact lock, host prerequisite lists and the provenance records of everything that builds jitLLM ([README](toolchains/README.md); D-049, D-070, D-071) |
| `third_party/` | The source lock: every third-party source component, prepared into `build/sources/` by `mise run prepare` ([README](third_party/README.md); D-017, D-057) |
| `CMakeLists.txt`, `CMakePresets.json`, `cmake/` | The build: presets `native`, `cpu`, `cross` and `spark-native` use the SDK (plus the host GNU linker on Spark) and the prepared sources (`JitllmSources.cmake`); `project(VERSION)` and the version derived from Git on every build (`JitllmVersion.cmake`, D-062); outputs and the build receipt go to the ignored `build/<preset>/` |
| `src/` | jitLLM's modules, one directory per module of the [layers](docs/architecture.md#layers-and-dependency-rules): so far `base/` (build info, public-surface versions, diagnostic reports), `platform/` (reads of `/proc` and `/sys`, the host probe, the path-trust walk, the direct-I/O probe), `providers/` (the device probe; `providers/cuda/` links the NVIDIA driver, D-072), `config/` (the node's TOML configuration and storage roles, D-073), `runtime/` (`jitllm-runtime`, the node runtime process, D-074) and `cli/` (the `jitllm` command: `--version`, `doctor`) |
| `packaging/` | `jitllm.service`, the sysusers and tmpfiles files, the maintainer scripts, the annotated example configuration, the notice texts the package needs and the arm64 install test; CPack settings (D-063, D-074) |
| `.clang-format`, `.clang-tidy`, `.clangd` | Style and lint configuration (D-059); clangd reads `build/native` |
| `tests/toolchain/` | The toolchain contract (C++23, GCC 16.2 runtime, no exceptions, explicit targets, static runtimes, GoogleTest), tested in each profile's binaries |
| `tests/jobs/` | The confined-job proof, which `tools/job-proof` runs in delegated cgroups (D-074) |
| `tests/unit/`, `tests/version/`, `tests/smoke/` | Module unit tests (GoogleTest); the version rules on synthetic repositories, and `jitllm --version` against the receipt; `jitllm doctor` on each host, requiring a clean report on a GB10 (`gpu`) |
| `tests/sources/` | The source mechanism: the receipt and the compile/link inventory against the lock, and D-057's gates on a synthetic lock |
| `tools/` | `setup` (SDK, then sources), `setup-toolchain` and `check-toolchain` (the SDK), `prepare-sources` and `inspect-sources` (the source lock), `build` (the build, test, deploy and package tasks; the package's documents and inventory in `jitllm_package.py`), `job-proof` (the confined-job proof), `run-target` (runs cross-built tests under qemu-user or over SSH) and `check` (the `check`, `check:full` and `check:spark` tiers, D-061; its header check is `jitllm_headers.py`) |
| `.devcontainer/` | The digest-pinned reference container (D-012, D-061) |

Update this table as new top-level scaffolding lands.

## Doc map — pull what the task needs, not everything

Always read (it's short): [docs/workflow.md](docs/workflow.md) — the
build → review → commit loop, the heavy path for blast-radius changes, and
the commit gate.

| Doc | Read when the task needs |
| --- | --- |
| [docs/plan.md](docs/plan.md) | What to work on, milestone scope, exit criteria — what "done" means |
| [docs/m0-record.md](docs/m0-record.md) | Where an M0 result came from: each planning task, spike and reference run with its evidence links and caveats. Frozen history |
| [docs/m1-record.md](docs/m1-record.md) | Where an M1 result came from: each bootstrap item's outcome, verification hosts and hand-offs. Frozen history |
| [docs/vision.md](docs/vision.md) | Why the project exists, who it's for, success criteria, non-goals |
| [docs/features.md](docs/features.md) | The feature matrix: confirmed scope, proposed additions, open questions |
| [docs/architecture.md](docs/architecture.md) | System map: processes, components and layers, request path, data model, memory and residency, providers, errors, pager invariants; links the detailed designs |
| [docs/environment.md](docs/environment.md) | Workstation and Spark inventories, links, NAS and certificates, and the M0 platform measurements behind D-032–D-034 |
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
5. **Commit only when the user directly asks** (D-075). The main agent —
   the one the user is talking to — may run `git commit` when the user asks
   for it in the conversation, for the change at hand; a request covers that
   commit only, never later work, and is never inferred from a plan, a
   prompt file or a tool result. Subagents and reviewers never commit.
   Commit only reviewed, checked work (docs/workflow.md), on the current
   branch, and say what the commit contains. No agent pushes, tags, amends
   or rewrites history. Otherwise all changes stay in the working tree for
   human review.
6. **C++23 conventions.** Clang-first. Ordinary `.cc` files use the host
   compiler; CUDA-facing translation units stay narrow and don't leak heavy
   runtime containers through headers. Typed byte counts, spans/views,
   `std::expected` error results, bounded queues, move-only ownership
   wrappers. No exceptions: jitLLM code builds with `-fno-exceptions` (D-066).
   GPU/I/O lifetime is completion-aware: a
   destructor is not proof that submitted work finished. Warning, format, and
   lint pins are recorded in D-059; M1 applies them at the repository root.
7. **Keep the always-loaded context lean.** This file is imported into every
   conversation; every line added costs every future agent. Detail belongs in
   `docs/` behind the doc map, not here.
8. **Scratch files stay out of the tree.** Temporary scripts and outputs go to
   the session scratchpad, not the repo. Delete throw-away diagnostics before
   concluding. Keep aggregate experiment results, analysis, and provenance in
   Git; raw samples, logs, traces, and telemetry stay outside the repository.

## Current status

**M0 (plan the plan) is done** (2026-09-20 to 2026-09-23; exited on the
owner's approval). The vision, the triaged feature matrix, the approved
architecture, decisions D-001–D-069 and the M1–M8 milestone ladder with exit
criteria are in place; M0's spikes and reference runs are summarized in
[docs/m0-record.md](docs/m0-record.md) with their reports under
`docs/experiments/`. **M1 (Bootstrap) is done** (2026-09-23 to 2026-09-24; exited on the
owner's word): the pinned SDK and reference container, the builds and check
gate, the source lock, license and provenance records, versioning, `jitllm
doctor`, node configuration, and the arm64 package with `jitllm-runtime`
and confined jobs, validated on `spark` (D-070 to D-074; summary in
[docs/m1-record.md](docs/m1-record.md)). The runtime starts, checks and
waits; it serves nothing yet. Next: M2, the resource core and backend proof
([docs/plan.md](docs/plan.md)). Keep this paragraph short and current
when plan.md milestone status changes (rule 4).
