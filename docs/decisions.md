# Decision log

Newest first. Every entry: what was decided, why, and what would reopen it.
Entries are for choices that are expensive to reverse or that a future agent
might silently undo — not routine implementation calls; a few per milestone is
the target. Existing entries are never edited into a different decision —
reversing or amending one gets a *new* entry that supersedes it (a status-line
annotation on the old entry is fine). When an entry hangs on a claim about
current technology state, check a current source or run a local experiment —
training knowledge is stale.

**Reading:** scan the D-NNN headings (or grep) and read only the entries your
task touches. Full read is for structural or cross-cutting work.

**Culling:** the log may be periodically pruned — superseded or moot entries
whose context no longer informs anything current are deleted outright; git
history is the archive. D-numbers are never reused.

Format:

```
## D-NNN: Title  (YYYY-MM-DD, status: accepted | proposed | superseded by D-MMM)
Decision / Context / Consequences / Reopen if
```

Seed entries D-001 through D-014 record the directions the owner stated in
[ideation.md](ideation.md) (§ references) at kickoff. D-015 onward record
the owner's M0 triage answers and review fixes from the same day.

---

## D-027: Users install through native package managers; a signed apt repository for Spark first  (2026-09-20, status: accepted)

**Decision.** The user-facing installation path is the platform's package
manager. For DGX Spark that is apt with a project-hosted, signed repository
serving arm64 packages. The developer setup path (mise, project-owned SDK
provisioning, D-012) is separate and is not what users run. Other package
managers follow their platforms (D-026) if and when those are targeted.

**Context.** Owner's direction on 2026-09-20 during M0 triage, noting it
matters little at this stage but should be targeted.

**Consequences.** Packaging is real M7 scope, not an afterthought: a systemd
unit, a non-root service user, an FHS layout (configuration under `/etc`,
state and artifacts under a configurable data directory), an upgrade path
that drains before restart, and package dependencies that express the CUDA
and driver requirements without conflicting with NVIDIA's packages. Optional
copyleft modules can ship as separate packages in a separate repository
component so the default install is the core and enabling a module is an
explicit user action, mirroring D-017; whether to do it that way is a
proposed row. Filesystem and service layout should be settled before the
endpoint lands in M3 so paths do not move later. Repository hosting and
signing keys are an M7 decision.

**Reopen if.** The primary platform stops being Debian-based, or users need
container-only distribution instead.

## D-026: NVIDIA and DGX Spark first; keep the memory, paging, and transport boundaries portable when it costs nothing  (2026-09-20, status: accepted)

**Decision.** The primary target is NVIDIA hardware, DGX Spark first. The
base concepts apply to Apple silicon and AMD equivalents on single machines,
so where abstracting paging, memory, and RDMA or transport operations adds no
complexity and no performance penalty, do it: the core (catalog, ledgers,
reservations and leases, eviction policy, scheduler, conductor, artifact
index, sessions) contains no vendor types; device memory and transfer
operations go through narrow provider interfaces, with CUDA VMM as the first
and only implementation; platform properties (unified memory, VMM
granularity, direct storage path, RDMA) are probed capabilities, not
constants. No work is done for other platforms until there is demand and it
makes sense, and nothing is sacrificed on NVIDIA to enable them.

**Context.** Owner's direction on 2026-09-20 during M0 triage.

**Consequences.** The deterministic fake backend and the CPU-only build are
the portability guardrail: the core builds and its tests pass with no vendor
SDK present. Compute kernels are per-platform by nature and are not
abstracted beyond the backend operation contract; a substrate that already
runs on several platforms would make a port mostly a memory-provider job,
which is a consideration for open question 4. The artifact's canonical form
stays platform-neutral; vendor layouts are optional alternatives (D-009).
Unified memory as one budget (D-004) generalizes to Apple silicon and AMD
APUs; if it costs nothing, the ledger keys by memory domain so a
discrete-GPU platform is a data difference rather than a redesign. Each
platform's memory and transfer APIs are verified when a port is actually
considered, not now.

**Reopen if.** An abstraction is measured to cost performance or clarity on
NVIDIA (NVIDIA wins), or a port is undertaken (which gets its own decisions).

## D-025: Measure the switching baseline once the reference runs  (2026-09-20, status: accepted; amends D-021)

**Decision.** Artifact size divided by measured read bandwidth remains a
first-cut M0 estimate. Once the pinned reference setup runs, measure an
end-to-end A→B→A cycle on the target, including state save/restore or
recomputation, unload/load, and time to the first returned token. M4's
switching gate uses this measured baseline for the same workloads and memory
budgets; estimates alone do not establish the "never worse than a full swap"
claim. Report cold storage, warm OS cache, and warm residency separately.

**Context.** The owner approved the review's planning improvements on
2026-09-20. Transfer bandwidth alone does not describe the user's wait.
[llama.cpp's server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
describes model routing and slot prompt-cache save/restore; verify these on
the pinned build and checkpoint and include applicable reference
optimizations. Do not assume a reference must lose all conversation state.

**Consequences.** D-021's performance floor and scope-sizing purpose remain.
Its exemption from benchmarking the full-swap baseline is superseded. The
reference A→B→A experiment is an M0/early-M1 deliverable, reused by M4;
record unsupported state paths instead of inventing support. Matched and
normal-reference configurations remain separate views under
[the comparison protocol](architecture.md#performance-evidence).

**Reopen if.** The selected reference cannot execute a comparable cycle;
record the limitation and select another validated comparator before making
the corresponding performance claim.

## D-024: Conversation reuse is bounded; prefix identity does not imply session lifetime  (2026-09-20, status: accepted; amends D-019 and D-022)

**Decision.** For clients that resend their history, prefix matching finds
reusable computation. It does not establish a unique conversation, whether
the user has finished, or whether they will return. Retained state between
requests has explicit memory and spill budgets and an expiry policy. A valid
retained prefix resumes from resident or spilled state; when that cache is
absent, expired, incompatible, or invalid, recompute from the supplied history
and report the cache miss. Never claim state reuse when reconstruction ran.

State needed by admitted work, including an internally suspended request,
stays protected by D-007's progress and lifetime rules. Cache expiry cannot
discard that state. If a request lacks the history needed for reconstruction,
unavailable state produces an explicit error; never continue from partial or
unrelated history. Optional session IDs and release hints do not confer
unbounded retention or relax cache-compatibility checks.

**Context.** The owner approved the review's planning improvements on
2026-09-20. A library and its conversations can outgrow both RAM and spill
capacity. Two conversations can share a prefix and then branch. D-019's
preservation goal needs a bounded retention contract, and D-022's prefix
identity needs to be distinguished from explicit session identity.

**Consequences.** Cache identity includes the artifact/model identity,
relevant execution and state-layout configuration, and the exact processed
prefix, including non-text inputs when supported. Reuse shared prefixes
without allowing one branch to mutate another's state. Architecture-specific
adapters declare valid restore boundaries. M4 proves both retained-state
resumption and correct fallback after expiry or cache loss. Set retention
defaults and limits before M4 from measured state sizes and the target
budget; no numeric defaults are chosen here. Details live in
[architecture.md](architecture.md#conversation-state-retention).

**Reopen if.** A client needs a durable server-side conversation without
resending history; define a separate persistence contract and resource
guarantee before promising that behavior.

## D-023: Cluster topology is discovered or configured, never baked in; one conductor; replicas allowed  (2026-09-20, status: accepted)

*Delivery clarification (M0 review): the M0/M1 design obligation below is met
by configured membership, one configured conductor, and capability/health
probes. This is the M4a implementation path. Automatic discovery, election,
and replica delivery follow the explicit revisit triggers in
[plan.md](plan.md#deferred-delivery-and-proposals); those mechanisms do not
block M0/M1. The topology and replica scope remains unchanged.*

**Decision.** The application never hardcodes node names, counts, or roles.
Cluster membership and per-node capabilities come from configuration and
discovery at runtime. There is always exactly one conductor, the single point
of entry, designated by configuration or election, which runs the placement,
routing, and admission role described in D-020. When concurrent demand on a
small model justifies it, the same model may run as multiple replicas on
different nodes; the conductor balances requests across them while keeping a
conversation's state on one replica (session or prefix affinity). The
hostnames `spark` and `spark-b` in these docs describe the owner's
environment only.

**Context.** Owner's clarification on 2026-09-20: the node names are personal
configuration; discovery and configuration should be dynamic and flexible;
there will always be a single conductor and point of entry; multiple copies
of a busy small model should be allowed.

**Consequences.** D-020's "coordinator" is this conductor, and its mention of
the master node's hostname is environment, not design. Cluster configuration
format, discovery mechanism, conductor designation, health and membership,
and per-node capability probing are M0/M1 design items. Replicas add a
load-balancing and affinity dimension to placement; the catalog stays per
node. Code, tests, and docs use role names (conductor, node), never the
owner's hostnames, except in the environment inventory.

**Reopen if.** A deployment needs multiple entry points (federation), which
would revisit the single-conductor rule.

## D-022: Standard web-API compatibility is the baseline; sessions and hints are optional extensions  (2026-09-20, status: accepted; prefix identity and retention amended by D-024)

**Decision.** The inference API works out of the box with existing standard
web APIs and clients; the owner named Cursor, OpenCode, and Codex. The
`model` field of a standard request is the switch signal. For standard
clients, which resend the whole conversation and carry no session ID,
conversation identity comes from prefix matching (prefix-cache identity).
Optional extensions for cooperating clients, all ignorable by standard ones:
an explicit session or conversation ID, an explicit release, and next-model
or warm hints.

**Context.** Owner's answer on 2026-09-20 during M0 triage. Resolves
features.md open question 8.

**Consequences.** OpenAI-compatible chat completions is the minimum surface;
the exact endpoint set the named clients need (Responses API, Anthropic
Messages format, streaming and tool-call details) is verified against their
current documentation in M0/M1 and recorded as a follow-up. Prefix-cache
retention with spill and restore across switches is the baseline mechanism by
which conversation state survives, because standard clients would otherwise
force a full re-prefill. The management API stays separate and local (D-014).
Clients talk to the conductor's endpoint (D-020, D-023).

**Reopen if.** The named clients move to a protocol the baseline does not
cover.

## D-021: The switching bar is "never worse than a full swap"; seamless is the goal  (2026-09-20, status: accepted; baseline measurement amended by D-025)

**Decision.** The minimum acceptable behaviour for a model switch is that it is
no slower than today's practice of suspending one engine instance and
instantiating another, a full unload and load, while adding management
convenience. The goal is as fast and seamless as possible. The full-swap
baseline is estimated from artifact size and measured read bandwidth, not
benchmarked separately, because being slower than it would take effort.

**Context.** Owner's answer on 2026-09-20 during M0 triage.

**Consequences.** The paging-feasibility spike quantifies how much partial
retention and expert paging gain over a full swap and where expert paging
earns its complexity; it adjusts M4/M5 scope rather than gating the
project's viability. M2's prerequisite softens accordingly. Benchmark
reporting still includes switch and switch-back latency and bytes moved.

**Reopen if.** A competing approach raises the practical floor well above a
full swap.

## D-020: The coordinator orchestrates the Spark pool: placement, routing, and time-slicing versus concurrency  (2026-09-20, status: accepted)

*Terminology and scope note, same day (D-023): "coordinator" is the conductor
role; the hostname below is the owner's environment, not application
configuration. Topology is discovered or configured.*

**Decision.** The master node (`spark`) runs the coordinator, which
orchestrates the whole pool: it decides which models, or parts of models, run
on which node; routes each request to the node running that model when the
whole model lives elsewhere; and admits work cluster-wide. When a supported
placement fits the working sets and complete execution envelopes within each
node's budget, models run concurrently with no paging. Aggregate pool capacity
alone is insufficient: unsharded models must each fit on their assigned node.
When no such placement fits, execution is time-sliced; that is the
v0 policy under contention. Concurrency when it fits is required soon after
v0, not deferred to the last milestone.

**Context.** Owner's answers on 2026-09-20 during M0 triage: time-slicing
first but concurrency soon, and the coordinator should make optimal cluster
decisions, placement preferred.

**Consequences.** Node placement is the first two-node capability, ahead of
sharding; the ideation §12 prepare/commit protocol applies to sharded models,
while placement needs only the network. D-005 is unchanged: one execution
process per node; the coordinator is an additional role on the master, and
whether it lives inside that runtime process or a sidecar is an M0 decision.
The inference and management APIs have a single front door on the
coordinator. Capacity accounting is per node with a cluster view. Under time
slicing, question 9 reduces to one active phase per node plus suspended
state. The ladder rewrite places concurrency-when-it-fits around M4/M5.

**Reopen if.** A third node type or a multi-user deployment changes what the
coordinator must balance.

## D-019: Primary workload is one user switching among a library of models, with conversation state preserved  (2026-09-20, status: accepted; retention bounds amended by D-024)

*Switch-boundary clarification (M0 review): the quiescent switch below is a
scheduler-established handoff. A request for another model signals intent;
the scheduler confirms the relevant GPU, I/O, and network completions before
releasing residency leases or reclaiming backing, preserving suspended live
state under D-007. Request arrival alone changes no reclamation eligibility.*

**Decision.** The workload jitLLM is optimized for first is a single user, or
a single user's agent plus subagents, switching automatically among a library
of models that is larger than memory. A conversation spans minutes to hours,
and the main model is expected to resume after a subagent on a different
model finishes. Models are time-sliced rather than run simultaneously
(TDMA-style): two models execute concurrently only when both resident sets
fit in RAM. The headline metrics are therefore switch latency (a request for
model B arrives while A is resident, to B's first token), switch-back latency
(A resumes with its conversation state), steady-state decode parity with an
all-resident run, and the size of the library that stays warm enough.
Multi-tenant fairness and mixed-traffic throughput are secondary.

**Context.** Stated by the owner on 2026-09-20 during M0 triage, sharpening
ideation §1's "one large model plus smaller models handling mixed traffic."

**Consequences.** Model switches are explicit quiescent points, which the
scheduler can use as eviction and lease boundaries. Preserving a suspended
conversation's KV or equivalent state across a switch (residency, spill to
SSD, or reconstruction) matters as much as retaining weights, and a
session or conversation is a first-class scheduling entity with
suspended-versus-finished state. Requests name the next model, so prefetch of
the incoming model's core can overlap the outgoing model's last steps, and
clients may hint upcoming use. With two nodes, placing the subagent's model on
the other node is a legitimate alternative to paging. Steady-state expert
paging during decode must be rare; its value is in resume and warm-up, not
per-token streaming. The feasibility spike's trace is an agent session with
model alternation and long context, not a mixed-traffic benchmark.

**Reopen if.** The project targets multi-user serving, where fairness and
throughput under contention would move up the priority list.

## D-018: Experimental artifacts before compatibility guarantees  (2026-09-20, status: accepted; amends D-009)

**Decision.** M0 chooses an experimental artifact encoding and layout ABI for
bring-up. Every artifact still declares its format and layout versions and
passes the validation, integrity, and atomic-publication requirements of
D-009. Unsupported versions are rejected explicitly; experimental status
never permits guessing a layout or bypassing validation.

Compatibility guarantees wait until a small dense model and a small MoE have
each exercised import, resident execution, eviction, and restoration with
numerical checks. After that evidence, record which versions remain readable,
whether changes require migration or re-import, and how users are notified.
Until then, mark artifacts experimental and document that a format change may
require re-import. Keep the source checkpoint available; never silently
overwrite or reinterpret an incompatible artifact.

**Context.** The owner requested the scaffold review fixes on 2026-09-20.
Choosing a durable compatibility contract before a real backend exercises
the layout risks preserving mistakes. D-009's prepared-artifact and
untrusted-input rules remain in force; only the schema maturity schedule is
amended.

**Consequences.** M3 through M5 validate the experimental format. M1 defines
version identifiers and rejection behavior; format changes remain versioned
decisions under D-016. A stable format is not an M0 exit requirement and is
not implied merely by completing a milestone.

**Reopen if.** External artifact consumers need compatibility guarantees
before both model paths have been validated; that requires an explicit
narrower contract and its own evidence.

## D-017: Dependency policy distinguishes incorporated code, tools, and platform runtimes  (2026-09-20, status: accepted; supersedes D-015)

**Decision.** jitLLM-authored code remains Apache-2.0. Classify dependencies
by how they are used, and audit the full selected build, including its
transitive dependencies:

- **Incorporated implementation.** Imported source, headers, generated code,
  and linked implementation libraries in the core retain the D-015 allowlist:
  Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT, or MPL-2.0. MPL file-level
  obligations still apply. Other licenses default to fully removable optional
  modules, explicitly selected and reviewed for compatibility. Being optional
  does not establish permission to combine or distribute a component.
- **Build tools.** Compilers and general build utilities executed during
  development are recorded under their own licenses. Their executable's
  license does not automatically classify the output. Audit any code,
  templates, headers, or runtime support incorporated into that output under
  the implementation or platform category. Optional backend importers and
  generators remain part of that optional module's dependency closure.
- **Declared platform dependencies.** The default build may use the selected
  platform's C library (initially glibc), libstdc++ and required compiler
  support runtimes, and the NVIDIA driver/CUDA SDK components needed by its
  selected CUDA backend. This exception covers their normal interface headers
  and runtime use, with applicable license exceptions and vendor terms
  recorded. It does not admit arbitrary kernels, copied samples, model
  adapters, or other implementation libraries as platform dependencies.

Each selected component needs a provenance record identifying its role,
exact version, applicable license/exception, linkage or use, and any shipped
files, notices, source, or redistribution obligations. Platform classification
is not a blanket approval of every bundled SDK component or permission to
redistribute it. Unknown or incompatible terms block inclusion. Extending
the platform exception to another dependency family requires a new decision.

**Context.** The owner requested the scaffold review fixes on 2026-09-20.
D-015's unrestricted wording excluded dependencies already required by
D-010 and D-006. [libstdc++ uses GPLv3 with the GCC Runtime Library Exception](https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html);
[CUDA components have NVIDIA and component-specific terms](https://docs.nvidia.com/cuda/eula/index.html).
These sources were checked on 2026-09-20; adoption still checks the exact
pinned components and their applicable terms.

**Consequences.** The copyleft-components-disabled profile excludes optional
implementation modules and their source, headers, generators, generated code,
and binaries from both fetching and building. It may use the declared tools
and platform dependencies above. Its audit reports those dependencies
separately and does not describe the entire deployment as permissive-only.
Optional modules remain removable without losing the scheduler or allocator;
their enabled builds ship matching notices and source obligations. Every new
dependency's handoff records its category and, for implementation code, tier.

**Reopen if.** A selected component's actual terms cannot be met, or the core
allowlist or declared platform dependency families need to change.

## D-016: Externally consumed project — mandatory review pass and evidence-carrying handoffs  (2026-09-20, status: accepted; supersedes D-001)

**Decision.** jitLLM is a single-developer project intended for external
consumption, so the process is heavier than the lean personal-project default.
Unchanged from D-001: AI agents implement from the docs; the human directs,
decides, and is the sole committer; the `docs/` set is long-term memory and
`AGENTS.md` stays lean. Changed:

1. Every change gets a review pass by a separate agent with fresh context
   before handoff, hunting real defects; findings are fixed and re-verified.
2. Changes in blast-radius areas (memory manager and pager invariants, VMM
   mapping and staging, on-disk formats for artifacts and spill, import of
   untrusted checkpoints, management API security defaults, license and
   provenance records, public interfaces) get the heavy path by default: an
   adversarial challenge pass plus fix/verify rounds.
3. Handoff notes carry evidence: which checks ran on which host, test
   results, measurements with provenance. Nothing is handed off with failing
   or silently skipped checks.
4. Behaviour changes come with tests, or the note says why not.
5. Public surfaces (artifact format, management API, CLI, configuration) are
   versioned and their changes get decision entries.
6. Docs that describe capability (README, support matrix) are release
   artifacts; overclaiming is a defect a reviewer flags.

**Context.** Owner's M0 triage answer on 2026-09-20: single-developer, but
meant to be externally consumed, so more rigorous than the lean process used
for personal projects. D-001 had assumed no external users. The loop is
spelled out in [workflow.md](workflow.md).

**Consequences.** Slower per-change turnaround, accepted. The human may
explicitly waive the review pass for a specific trivial change; agents never
waive it themselves and never downgrade a heavy-path change. Versioning,
changelog, and release conventions are decided in M1 and recorded here.

**Reopen if.** The project stops being a public deliverable, or a contributor
base makes a different structure (CI-enforced review, maintainers) more
appropriate.

## D-015: Dependency license tiers — permissive in the core, any license as an optional module  (2026-09-20, status: superseded by D-017)

**Decision.** Any license is allowed somewhere in the tree; the license decides
where. Dependencies and imported code under Apache-2.0, BSD (2- and
3-Clause), MIT, or MPL-2.0 may be used in the **core**, which is what the
default build and the copyleft-components-disabled profile produce. Code under
viral copyleft licenses (AGPL-3.0, GPL, and similar) may appear only in
**optional modules or plugins** that a builder or user explicitly chooses to
include. Parts of the tree are expected to carry optional AGPL code from the
start. Licenses not on the core list (LGPL, EPL, CDDL, source-available
licenses, and anything else) default to the optional tier until the owner
adds them to the core list by a superseding entry.

**Context.** Owner's M0 triage answer on 2026-09-20, refining D-002. The
original-code license is Apache-2.0 (D-003).

**Consequences.** Each imported unit's tier is recorded with its SPDX
identifier and provenance record. MPL-2.0 is file-level copyleft: its files
may live in the core, but modifications to those files stay under MPL-2.0 and
that obligation is tracked, not waived. Enabling an optional copyleft module
changes the license terms and notices of the resulting build, so the build
reports which profile it is and ships matching notices (ideation §17).
Optional modules need a real boundary (build-time module or runtime plugin)
that removes them completely, including headers, generated code, and
transitive dependencies; the mechanism is still a design question in
[features.md](features.md). Every merge of a new dependency states its tier
in the handoff note.

**Reopen if.** A core capability can only be met with a viral-licensed unit
(a product-scope decision to make explicitly), or the owner extends the core
list.

## D-014: Local-first management and privacy defaults  (2026-09-20, status: accepted)

**Decision.** The management API binds to local interfaces by default. Remote
access requires authentication and transport protection. Prompts and KV/state
contents are not logged by default; diagnostic events use opaque request IDs
and aggregate routing information unless detailed capture is deliberately
enabled for a session. Spill files are protected and their retention is
explicit.

**Context.** Stated by the project owner in ideation §13.

**Consequences.** Detailed traces are opt-in per capture. Spill-file location,
permissions, and lifetime are part of the storage design, not an afterthought.
Any remote dashboard needs an authentication story before it ships.

**Reopen if.** A deployment model beyond a single owner's local nodes is
adopted.

## D-013: Reuse existing implementations selectively, under their actual licenses  (2026-09-20, status: accepted)

**Decision.** Existing inference projects (vLLM, llama.cpp/GGML,
ExLlamaV3/EXL3, FlashInfer, CUTLASS/CuTe; Ollama for product-facing lifecycle
behaviour only) are sources of algorithms, kernels, model semantics, and
numerical test references. Compatible implementation units are reused under
their real licenses with provenance recorded. No single project's execution
architecture, allocator, or scheduler is adopted as jitLLM's. Rewriting or
translating copied code does not erase its provenance; "reference" is not a
relicensing mechanism.

**Context.** Ideation §1 decided-direction row "Reuse", §10, §17.

**Consequences.** Every imported unit carries source URL, immutable revision,
license, and a modification record. A reused kernel comes with its
assumptions (layout, quantization, workspace, dispatch); reuse the unit, not
an isolated benchmark winner. No rewrite-to-own without a measured need.

**Reopen if.** A reused unit's license cannot be honoured in a shipped
profile, or a native rewrite is justified by measurement rather than
ownership.

## D-012: Declarative, pinned toolchain provisioning via mise plus project-owned SDK manifests  (2026-09-20, status: accepted)

**Decision.** Tool setup, environment selection, and tasks are declared in a
checked-in `mise.toml` and `mise.lock`. The full LLVM / CUDA / AArch64 SDK is
provisioned by project-owned scripts driven by a toolchain manifest and an
artifact lockfile, with native Ubuntu setup and a reference dev container
sharing the same provisioning logic. Setup is idempotent, previews machine
changes, and never silently changes system default compilers, GPU drivers, or
security controls, accepts model licenses, or downloads optional copyleft
backends without the selected build profile allowing them. Pins are chosen
only after an end-to-end native/cross/CUDA smoke test passes; agents never
invent them.

**Context.** Ideation §1 row "Setup", §16. The C++ source-dependency mechanism
is explicitly a separate, still-open decision; mise is not chosen as a
replacement for every library dependency tool.

**Consequences.** Toolchain identity is reproducible and reportable (a
`doctor` command). Build-only setup requires neither a local GPU nor Spark SSH
access; remote tests require explicit target selection. CI pins the reference
container by digest and records the target driver separately from toolkit
and library versions. Host tools are built separately so an ARM build-time
generator is never run on x86 by accident.

**Reopen if.** mise cannot express a needed pin, or provisioning through it
proves less reliable than a container-only approach.

## D-011: Develop on x86-64 Linux; cross-compile for Spark; deploy and test over SSH  (2026-09-20, status: accepted)

**Decision.** Editing, indexing, native builds, static analysis, and CPU tests
happen on the x86-64 Ubuntu workstation. Spark (AArch64 CPU, GB10 GPU)
binaries are cross-compiled with explicit target triples, sysroot, and CUDA
host compiler; changed executables, libraries, kernel artifacts, and tests are
deployed over SSH; model artifacts stay on the target between runs. ARM
concurrency, VMM, kernel, and distributed tests run on Sparks. Explicit
CPU/GPU targets only, never `-march=native` or workstation autodetection.

**Context.** Ideation §1 row "Development", §15. Workstation baseline captured
2026-09-20 (see architecture.md). Older environments are not a priority;
feature checks are preferred over kernel-version barriers.

**Consequences.** CMake toolchain files for native and Spark builds with
explicit target system, processor, sysroot, find-root policies, and
`CMAKE_CUDA_HOST_COMPILER` set before enabling the CUDA language. ARM runs
early because ARM memory ordering exposes bugs x86 hides; use the C++ memory
model, not hardware assumptions. An optional native ARM build is a diagnostic,
not the primary workflow.

**Reopen if.** Cross CUDA compilation for GB10 cannot be validated, making a
target-side build primary.

## D-010: C++23 host runtime, Clang-first, native hot path  (2026-09-20, status: accepted)

**Decision.** The host runtime is C++23. Clang is the primary compiler for code
we own. NVCC is the default CUDA compiler with Clang as its host compiler
where the validated configuration supports it; localized GCC exceptions for a
backend do not change the primary compiler. Pinned libstdc++ initially
(compiler and standard library are separate choices; never cross incompatible
C++ library/ABI boundaries). No Python or other interpreter in the serving,
paging, or scheduling hot path. Build-time tooling (importers, AOT kernel
compilation such as Triton AOT) may use Python when it produces native
kernels and metadata with recorded provenance.

**Context.** Ideation §1 row "Implementation", §10 "Native runtime does not
prohibit non-native build tools", §14. Current CUDA documentation lists Clang
and C++23 support, but the pinned toolkit/compiler combination must pass our
own integration tests (M0 toolchain smoke spike).

**Consequences.** Ordinary `.cc` files use the host compiler; CUDA-facing
translation units stay narrow and may use a separately validated dialect when
imported code requires it. Exception policy is chosen deliberately in M1; no
exceptions cross a C ABI. Optional separately built backends get a versioned
C ABI with opaque handles and explicit descriptors, no STL objects across it.
GPU/I/O lifetime is completion-aware.

**Reopen if.** The pinned CUDA toolkit does not support Clang as host compiler
with C++23 in the cross configuration.

## D-009: Models run from prepared, versioned artifacts produced by an owned import pipeline  (2026-09-20, status: accepted; schema maturity schedule amended by D-018)

**Decision.** jitLLM owns the conversion from source checkpoint, configuration,
and tokenizer into execution-ready, indexed, hashed, atomically published
on-disk artifacts that allow bounded range reads without reprocessing.
Runtime extents are populated from artifacts, never from raw checkpoints.
Artifacts never serialize process addresses, C++ object layouts, live
mutexes, allocator handles, or executable graph objects; pointer tables and
runtime objects are rebuilt per process. Checkpoints are untrusted input: no
arbitrary code execution; lengths, paths, hashes, and metadata are validated.
Requantization is an explicit transformation, never a side effect of
switching kernels.

**Context.** Ideation §1 row "Model storage", §11.

**Consequences.** The artifact schema (encoding, layout ABI, alignment,
integrity, sharding) is an M0 decision and is expected to version. Import runs
on the workstation where target hardware is not needed; target-assisted
tuning is an explicit mode that writes separately keyed results. Mutable spill
files are separate from immutable model files. Alternative backend layouts
are stored only when measured value justifies them. "Mappable" means indexed
for direct population of runtime extents, not `cuMemMap` of an SSD file.

**Reopen if.** Not as a whole; individual schema versions supersede each
other through their own entries.

## D-008: Partial eviction and on-demand expert acquisition, with no substitution  (2026-09-20, status: accepted)

**Decision.** Under pressure, reclaim the least valuable eligible extents
across all models, not whole models. For routed-expert (MoE) models, routing
is a distinct dependency-discovery stage: execute routing, resolve selected
expert IDs to local shards and storage ranges, acquire residency for their
dependency closure, execute, release leases after all consumers complete.
Never substitute a resident expert for a selected one; never drop a selected
contribution to avoid a miss; prefetch is speculative and actual routing is
authoritative. The initial implementation acquires all selected experts
before launching the numerical implementation and suspends the continuation
while I/O is pending; overlapped expert execution and GPU-visible residency
tables are later fast paths that require a correct baseline first.

**Context.** Ideation §1 row "Paging", §5, §7, §9.

**Consequences.** A batched phase's distinct-expert working set is bounded by
`min(E, k*T)`, so chunk sizes are memory and scheduling decisions. Routing
traces are collected and replayed offline against candidate policies. The
resident-hit path is measured separately from the miss path, including the
cost of the routing synchronization boundary on all-resident paths. A fully
resident fused plan is retained where valuable so not every invocation pays
the maximum flexibility cost.

**Reopen if.** Measurements show the routing boundary's cost is unacceptable
even on all-resident paths and no split plan mitigates it.

## D-007: Capacity reservations are separate from residency leases; commitment is lazy  (2026-09-20, status: accepted)

**Decision.** Three distinct concepts: *virtual reservation* (address-space
operation, no physical capacity), *capacity reservation* (admission commitment
that a bounded operation can obtain memory under a defined progress policy),
and *residency lease* (protection of particular backing while its consumers
execute). A request owns a logical transaction and persistent-state budget;
execution phases acquire concrete leases. Granting capacity never eagerly
evicts useful cached data; unused allowance may remain occupied by revocable
cache until a real dependency miss requires it. Releasing a lease changes
eligibility, not residency. Commitment and occupancy are separate ledgers;
shared extents are charged once; evicting or pending-write-back capacity stays
charged until actually reclaimable. The global scheduling/catalog lock is
never held across GPU, disk, or network waits.

**Context.** Ideation §1 row "Admission", §6, §9.

**Consequences.** Grants are classified as guaranteed under a safe-progress
schedule or opportunistic and deferrable. A request whose minimum feasible
phase can never fit fails or selects another valid plan; it never waits
forever. The first scheduler conservatively serializes phases that cannot
coexist and bounds suspended continuations; overlap is admitted later based
on measured envelopes. Acquisition and eviction have mutually exclusive
transitions with generation checks. Transactions give all-or-nothing
admission at a declared boundary, not database-style rollback of kernel
effects.

**Reopen if.** Not as a whole; the initial guarantee policy (features.md open
question 9) gets its own entry when decided.

## D-006: Explicit CUDA VMM backing management with a node-wide resource catalog  (2026-09-20, status: accepted)

**Decision.** Use the CUDA driver VMM API to reserve addresses, create physical
backing, map it, and set access. jitLLM supplies backing-store policy and
transfer operations; accessing absent backing is a bug, not a page-in request.
Every managed allocation is registered at construction in a node-wide catalog
with semantic metadata (identity, content kind, semantics, layout, recovery,
residency, safety, policy); immutable backing is finalized only after loading,
packing, and postprocessing complete. Unknown or unclassified allocations are
non-evictable. Logical identity (strong typed IDs plus generation checks) is
separate from current address and physical occupancy; raw pointers live only
at the device/backend boundary.

**Context.** Ideation §1 row "Memory", §4, §8. Stable virtual addresses for a
model's loaded lifetime where practical.

**Consequences.** Extent granularity, map/unmap cost, and physical-pool
retention must be measured on the real driver (M0 VMM spike); map/unmap is
not assumed free or asynchronous. Pool-held unmapped extents are reported as
reusable pool capacity, not memory returned to the OS. Stable virtual
addresses do not prove that executable graphs or external registrations
survive backing changes. One deliberately managed CUDA context per GPU
initially, with explicit streams and library handles.

**Reopen if.** Driver behaviour makes fine-grained VMM prohibitively slow and a
hybrid pooled-allocator design is measured to be better.

## D-005: An independent native runtime, one execution process per node  (2026-09-20, status: accepted)

**Decision.** Build jitLLM's own runtime rather than a vLLM fork or plugin. One
modular native execution process per node contains all model execution,
scheduling, memory policy, VMM control, and completion tracking for every
local model. Dashboard, importer, test controller, and supervisor may be
separate processes and are never in the per-expert hot path. The earlier
vLLM-extension, per-model-worker, and external-local-broker proposals are
reference or integration options only; there is no required local IPC
transaction for an expert lease or block eviction.

**Context.** Ideation §1 rows "Runtime ownership" and "Process model", §3.

**Consequences.** "Monolithic" means one authority over local execution state,
not one thread, one undifferentiated codebase, or a global lock held during
I/O. Application-level model contexts are separate from CUDA contexts.
Cluster coordination exchanges logical resource identities, phase IDs,
grants, and transfer metadata, never raw pointers to remote weights. Agents
must not silently revert to a vLLM-controlled process architecture.

**Reopen if.** A hosted engine exposes a memory-management contract that
satisfies D-006, D-007, and D-008 without owning the process.

## D-004: Target platform is NVIDIA DGX Spark, one or two nodes, treated as unified-memory domains over a network  (2026-09-20, status: accepted)

**Decision.** The initial target is local inference on one or two DGX Sparks
(Arm CPU, GB10 GPU at compute capability 12.1, 128 GB unified memory). A
Spark's memory is one physical budget: CPU allocations, GPU backing, staging
buffers, filesystem cache, and OS activity share it, so CPU offload is not an
additional capacity tier. Two Sparks are two memory domains connected by a
network, not a coherent 256 GB space; model parallelism and remote transfers
are explicit. On Spark, GPUDirect Storage runs only in compatibility mode and
GPUDirect RDMA into the relevant device allocations is not supported, so the
storage path is SSD → pinned host staging → CUDA copy → mapped backing (and
the reverse). These are Spark capability facts, not a reason to make the
storage backend Spark-specific.

**Context.** Stated by the owner in ideation §2, with NVIDIA sources checked
on 2026-09-20 (links in ideation §22). CUDA free-memory reporting is not the
complete node budget on Spark; the runtime combines its ledger, system
measurements, pressure signals, and conservative headroom, and never sums
overlapping CPU/GPU measurements as independent pools.

**Consequences.** Probe the installed stack on each node (M0 inventory); never
infer GPUDirect RDMA from a QSFP link. The staging pool is bounded and
reserved before pressure. The MiaAI-Lab two-Spark deployments for
GLM-5.3-Flash, DeepSeek-v4.1-Flash, and Qwen3.8-Flash-Next are starting
points to pin before porting, not validated support claims; feasibility is
measured on the prepared quantized representation, not full-precision sizes.

**Reopen if.** NVIDIA enables native GDS or GPUDirect RDMA on Spark (a direct
path becomes an optional backend; the accounting rules stay), or a second
hardware target is adopted.

## D-003: Apache-2.0 as the original-code license  (2026-09-20, status: accepted)

**Decision.** All jitLLM-authored code is licensed under Apache-2.0, as in the
`LICENSE` file at the repository root.

**Context.** The owner created the repository on GitHub with an Apache-2.0
`LICENSE` (initial commit, 2026-09-20). The design brief written the same day
listed Apache-2.0 as a candidate rather than a final choice (ideation §17), so
this entry was seeded as *proposed*. The owner confirmed Apache-2.0 as final
during M0 triage later on 2026-09-20 and the status changed to accepted. Where
dependencies under other licenses may live is D-015.

**Consequences.** Apache-2.0 notice and modification requirements apply; a
NOTICE file and file-level SPDX headers follow (conventions decided in M0/M1);
each imported component gets a compatibility review against the tiers in
D-015.

**Reopen if.** A contributor-licensing structure (CLA/DCO, foundation) calls
for a different arrangement, or a core dependency turns out to be
incompatible with Apache-2.0 distribution.

## D-002: All original code is open source; optional copyleft must be identifiable and removable  (2026-09-20, status: accepted)

*Scope note (owner, 2026-09-20): model weights are outside the project's
licensing scope. Users download them directly; jitLLM supports loading them
and uses a representative set for testing. The remark below about checkpoint
review is superseded.*

**Decision.** All jitLLM-authored code is open source under a permissive core
license (D-003 for which one). Optional copyleft components, for example
AGPL-derived kernels or importer support, are permitted only if the affected
implementation code, adapters, generated code, importer support, and
transitive dependencies are identifiable and a fork can rebuild a useful core
without them. Losing a model format or an optimization is acceptable; losing
the scheduler or allocator is not. CI maintains a copyleft-components-disabled
profile that neither fetches nor includes such source, headers, code
generation, or binaries, with its dependency closure audited, alongside a
compliant full-feature profile. License compatibility is audited before
merging a new backend, not only before release. Unknown or ambiguous
provenance stays out of distributed builds until resolved.

**Context.** Ideation §1 row "Licensing", §17. Repository-level checks at the
snapshot: vLLM Apache-2.0, llama.cpp MIT, ExLlamaV3 MIT; the MiaAI-Lab
references include AGPL declarations and retained upstream licensing. Those
checks are not substitutes for file-level and transitive review at adoption
time, and their model checkpoints need separate review.

**Consequences.** A provenance record (source URL, immutable revision, SPDX
identifiers, modifications, notices, SBOM) for every imported unit. A
directory, shared library, C ABI, or process boundary does not by itself avoid
copyleft obligations; a build combining covered code complies with them.
Vendor CUDA components have their own terms; "open-source jitLLM" does not
claim every driver, SDK, tool, or model weight in a deployment is open source.
Releases ship notices, build instructions, and source availability for the
actual configuration.

**Reopen if.** A required, non-optional capability turns out to be available
only under copyleft. That is a product-scope decision to make explicitly,
never a silent inclusion.

## D-001: AI-developed, human-directed workflow with a human-only commit gate  (2026-09-20, status: superseded by D-016)

*Superseded the same day: the human commit gate and docs-as-memory carry
forward into D-016; the lean one-pass process does not.*

**Decision.** AI agents implement from the project documentation; the human
directs, decides, reviews, and is the sole committer. The process is lean: one
agent, one pass, human scans the note and diff and commits; reviews happen on
demand; heavyweight multi-agent review only when the human asks for it. The
`docs/` set is the project's long-term memory and `AGENTS.md` is the
always-loaded, deliberately lean rulebook.

**Context.** The owner invoked the scaffold-project workflow on 2026-09-20 for
a single-owner project with no external users, contributors, or
hard-to-reverse deploys at this stage. Details in
[workflow.md](workflow.md).

**Consequences.** Agents never run `git commit` or `git push`. Decisions are
logged here; findings in [rough-edges.md](rough-edges.md); the plan and the
AGENTS.md status paragraph are kept current in the same unit of work. Two-host
development means agents state which checks ran on the workstation and which
needed a Spark.

**Reopen if.** The project gains contributors or users who depend on releases,
which would justify mandatory review passes and a release process, recorded
as a new decision.
