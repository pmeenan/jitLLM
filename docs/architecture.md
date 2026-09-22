# Architecture

> **Status: skeleton.** The first full draft is an M0 exit criterion; what is
> written here now is the load-bearing shape already settled, so drafting can
> build on it rather than re-derive it. The original reasoning and source links
> live in [ideation.md](ideation.md); section numbers (§) refer to it.
> Subsequent policy changes are recorded in [decisions.md](decisions.md).

## Fixed points (from decisions)

- **Ownership split.** Model implementations describe computation and
  dependencies; jitLLM owns storage, residency, scheduling, and execution
  lifetime. One native process per node (D-005).
- **Primary workload.** One user switching among a library of models larger
  than memory; an agent plus subagents on different models; conversations
  spanning minutes to hours; time-sliced under contention, concurrent when
  a supported placement fits each node's complete execution budget. Headline
  metrics are switch latency, switch-back with conversation state preserved,
  and decode parity with an all-resident run. A switch request signals intent;
  the scheduler establishes a completed handoff boundary before releasing
  residency leases or reclaiming backing (D-007, D-019).
- **Cluster coordination.** A single conductor, the cluster's one point of
  entry, places models, or parts of models, per node; routes requests to a
  node running the model; admits work cluster-wide. Placement is the first
  multi-node capability and is preferred over paging when it suffices;
  sharding follows for the flagship. Topology is discovered or configured at
  runtime, never baked into the application, and a busy small model may run
  as replicas across nodes (D-020, D-023).
- **Switching bar and API baseline.** Never worse than a full swap; seamless
  is the goal, validated against a measured reference cycle (D-021, D-025).
  Standard web-API clients work unmodified; the model field drives switching.
  Prefix matching enables bounded state reuse; it identifies neither a
  conversation nor its lifetime. Shared prompt prefixes and conversation
  continuations have independent reuse/expiry policies (D-024, D-031).
  Sessions and hints are optional extensions (D-022).
- **Vocabulary.** *Virtual reservation* = address space. *Capacity
  reservation* = admission commitment under a progress policy. *Residency
  lease* = protection of specific backing while consumers run. Never
  "reserved" unqualified (D-007). Likewise: read-only ≠ always resident;
  unleased ≠ may be lost; reconstructible ≠ already stored in the right
  format; resident ≠ available without admission (§4).
- **Memory model of the target.** One Spark = one 128 GB unified budget. Two
  Sparks = two domains over a network. Validated Spark storage uses direct
  file DMA into GPU-accessible host VMM without a staging copy (D-004, D-034).
- **Explicit VMM plus catalog.** Driver-API VMM; accessing absent backing is
  a bug; every allocation registered; unknown allocations non-evictable;
  typed IDs and generations, raw pointers only at the backend boundary
  (D-006).
- **Paging semantics.** Extent-level cross-model eviction; selected experts
  acquired on demand; no substitution; release ≠ eviction; lazy commitment
  (D-007, D-008).
- **Artifacts.** Prepared, versioned, hashed, atomically published; no
  process state serialized; checkpoints untrusted. Initial encoding and
  layout are experimental; compatibility guarantees require dense and MoE
  import/execution/eviction/restore evidence (D-009, D-018). Import repacks
  weights for whole-extent DMA; the initial Spark profile uses 2 MiB aligned
  payload extents with an expert/tensor index (D-035).
- **Dependency policy.** Own code Apache-2.0; incorporated core implementation
  uses Apache-2.0 / BSD / MIT / MPL-2.0, with other implementation licenses in
  optional, fully removable modules. Declared tools and platform dependencies
  have separate terms and remain in the build audit (D-003, D-017).
- **Language and build.** C++23, Clang-first, NVCC with Clang host compiler
  where validated, pinned libstdc++ initially; cross-built from x86-64 and
  tested on Spark over SSH; declarative pinned toolchain (D-010, D-011,
  D-012).
- **Portability posture.** NVIDIA first. The core holds no vendor types;
  device memory, paging, and transport sit behind narrow provider interfaces
  with CUDA VMM as the only implementation for now; platform properties are
  probed capabilities. Apple silicon and AMD single machines are possible
  later targets; nothing is done or sacrificed for them now (D-026).
- **Distribution.** Users install from a signed apt repository, Spark first;
  the developer toolchain path is separate. Installed layout, service user,
  and unit are settled before the endpoint lands (D-027, D-012).

### Mandatory pager invariants (§18)

Owner-stated; every one gets tests in the simulated backend and on hardware.

1. A consumer never touches absent or incompletely loaded backing.
2. An extent is not remapped, overwritten, or reused until all old consumers
   and relevant registrations are retired.
3. A cancelled request cannot cause late I/O to corrupt a newly assigned
   extent.
4. Mutable content is preserved or deliberately invalidated; cache metadata
   never falsely advertises a valid object.
5. Shared physical backing is counted once, including aliases and pool-held
   allocations.
6. Acquiring leases and revoking eligibility cannot both succeed for
   conflicting generations.
7. The runtime retains enough resources to complete or safely unwind admitted
   work.
8. An unavailable node or unacknowledged completion is not evidence of
   reclaimed memory.

### Reservation progress gate

Before M2, resolve features.md question 9 in a decision that defines the
initial admission policy. A conservative schedule is acceptable. Specify
which grants guarantee progress and which are opportunistic; when a phase
may start or suspend; and how retained activations, live state and its bounded
growth, scratch, staging, communications, graphs, metadata, and OS headroom
fit the budget. Define how an exceeded envelope is handled before unsafe
submission, including safe rejection or a validated alternative plan when
the minimum feasible phase cannot fit.

M2's fake backend must challenge the policy with competing phases that each
want to retain activations while waiting for expert loads, growth of live
state, a permanently impossible phase, and cancellation with late I/O.
Assert bounded occupancy and eventual completion or safe failure/unwind;
deferred work must not strand the resources needed by admitted work. Merely
avoiding an out-of-memory allocation is not proof of progress.

### Performance evidence

The early paging-feasibility spike uses a reference engine before jitLLM's
execution path exists. Record checkpoint revisions, quantization/layout,
expert sizes, request ordering and timing, prefill chunks, decode batches,
context/output lengths, and memory reserved for non-pageable resources.
Capture routes only for deliberately enabled benchmark sessions (D-014).
Compare policies at the same total node budget; preserve the distinction
between cold storage, warm OS cache, and warm runtime residency. Record
assumptions about overlap, mapping overhead, and contention when turning
trace replay and measured I/O into estimates. These estimates guide scope;
actual end-to-end results must later validate them.

Once the pinned reference runs, measure A→B→A on the target (D-025). Start
with one reproducible conversation on A, a request to B, and a continuation
of A's history. Include an all-resident control and a constrained budget
that forces displacement. Measure the full switch and switch-back interval,
including any state writes, unload/load, restore or re-prefill, and first
returned token. Enable applicable reference routing and state-save features,
verify them per checkpoint, and record any harness actions needed to use
them. Pin the trace and settings so M4 can repeat the same experiment.

Every backend/paging performance comparison has two views:

- **Matched configuration:** align checkpoint, numerical policy, request
  workload, context lengths, prefix-cache conditions, and decoding features.
  Disable speculative decoding in both paths if jitLLM lacks it. Compare
  jitLLM's resident and paged paths separately to expose paging overhead.
- **Normal reference configuration:** also run the pinned reference's normal
  documented configuration, including its enabled optimizations. Report its
  actual settings and feature differences. This measures the user-visible
  gap; do not attribute the whole gap to paging.

For speculative runs, record drafter identity, settings, acceptance, and
memory use; report throughput per accepted output token. If a matched run
cannot be made, record why and leave its comparison unvalidated. Before M2,
agree switching-benefit and generation-stall criteria for the selected
workloads using the measured whole-model-switching baseline. Record latency
distributions, bytes read and written, peak memory/spill occupancy, and prompt
tokens reused versus recomputed. Never invent thresholds or measured results.
M4 validates switching, M5 validates MoE paging, and M7 validates subsequent
optimizations against these criteria; scope changes when evidence warrants it.


#### Comparator: Athena's Engine (closed source, creator-reported)

Announced 2026-09-19 on the NVIDIA developer forum by its author; numbers
measured 2026-09-18 on a single GB10. Closed source, so nothing is reusable;
free for personal, research, and small-company use at the time of reading,
so installing it on a Spark as a second comparator is permitted (verify the
current terms first). All figures are the author's, not independently
verified, and include its speculative-decoding sidecar, so they belong in the
normal-reference view, never the matched one.

| Item | Reported |
| --- | --- |
| Models | DeepSeek V4 Flash (IQ2_XXS mix, Q8 projections); Qwen3.8 Flash Next (Unsloth UD-IQ4_XS); GGUF only |
| Prefill | about 1,070 to 1,126 tok/s at 8k context; about 950 to 960 tok/s at 256k |
| Decode, 256 tokens | DeepSeek 21.4 tok/s at 8k, 19.4 at 256k; Qwen3.8 29.9 at 8k, 32.1 at 256k; flat over context |
| Model switch | drain, flush checkpoints, release, verify memory is free, load the other: 46 s measured |
| Context restore | a 141,519-token conversation restored from a 619 MB file in 2.1 s, versus 2 min 20 s to re-prefill |
| API | OpenAI- and Anthropic-compatible, streaming and tool calls |
| Sessions | checkpoints the current agent or session to disk before unloading, so the switch preserves conversation state; one endpoint serves both models |
| Memory | the two models at those bit depths do not both fit in 128 GB, per the author's X thread |

What it tells us. The 46 s switch includes checkpointing the active session,
so it is a real-world floor for A→B→A with state preserved on this exact
model pair on one GB10 (D-021, D-025); M4's target is to beat it clearly at
comparable bit depths, and our own measured baseline still governs. Because
the pair does not both fit in 128 GB, it is the canonical two-large-model
switching workload for the feasibility spike. The restore figure implies roughly 4.4 KB of restorable state per
token for Qwen3.8's hybrid attention, a concrete datapoint for D-024's
retention budgets and the spill/restore gate, and it confirms D-025's
caution that a comparator need not lose conversation state on a swap. Its
"checks the memory is really there" step is the unified-memory accounting
problem of D-004 seen in the wild. Offering both API flavours is mild
evidence for the proposed Anthropic Messages row (D-022).
### Early backend integration proof

Run a small dense model from a prepared experimental artifact alongside M2's
resource-core work, before treating the internal backend contract as settled.
jitLLM supplies the weight and state backing, controls the stream, accounts
for workspace and backend-owned allocations, and tracks completion before
reuse. Unknown allocations remain non-evictable and budgeted. Check
teacher-forced logits against a pinned reference, then evict and restore
weights and retained state at a completed boundary and repeat the comparison
on a Spark. Include cancellation with pending work to exercise lifetime rules.
This proof informs M3 and the interfaces; it does not claim support for
flagship architectures, and there is no runtime plugin ABI to freeze (D-028).

## Expected shape (to be validated in the M0 draft)

The matrix was triaged on 2026-09-21. Where a bullet below leans on a
`deferred` or `open` [features.md](features.md) row, it is a design
assumption, not settled scope.

### Process and components (§3)

```text
standard clients (Cursor, OpenCode, Codex, Claude Code, ...)
        |  OpenAI-compatible and Anthropic Messages endpoint
        v
node A: conductor + jitLLM runtime          (owner's `spark`)
        | routes by placement           | model communication (sharded)
        v                               v
node B..N: jitLLM runtime  <-----------/    (owner's `spark-b`)
        \------------- management API -------------/
                  optional dashboard (separate process)
topology: discovered or configured, never baked in (D-023)

x86-64 workstation: editor / builds / CPU tests / import tools -> SSH deploy
```

| Component | Responsibilities |
| --- | --- |
| Model registry | Identity, tokenizer/config, adapters, immutable manifests, loaded plans, lifecycle |
| Execution planner | Select compatible operations / fused segments; declare dependencies, workspace, yield boundaries |
| Scheduler | Admit requests, advance ready continuations, fairness, distributed phase coordination |
| Resource catalog | Every managed logical resource: storage, backing, lifetime, aliases, statistics |
| Memory manager | Budgets, reservations, leases, cross-model victim selection, mapping policy |
| Storage service | File extents, read/write queues, staging, prefetch, integrity, spill |
| Completion service | Track GPU, I/O, and network consumers before reclamation |
| Compute backends | Execute declared operations with caller-controlled state, streams, workspace |
| Management plane | Config, status, import, metrics, traces, explainable control actions |

"Monolithic" means one authority over local execution state, not one thread
and not a global lock held during I/O. Understandable queues first; lock-free
only where measured. One deliberately managed CUDA context per GPU initially;
application-level model contexts are separate from CUDA contexts.

### Data model (§4)

```text
Logical resource -> tensor/storage byte ranges -> independently reclaimable
backing extents -> zero or more valid stored representations
```

Descriptor field groups: identity, content kind, semantics, layout, recovery,
residency, safety, policy. The categories are orthogonal. Multiple logical
resources may share an extent; the manager knows the full dependency closure
and charges each physical extent once. Shared or tied weights need content
and representation identity, not matching tensor names.

Small tensors and state blocks may be suballocated within backing (D-035).
Their logical sizes, validity, and leases are distinct from the provider's
physical mapping/release granularity and from the storage request size.
Freeing a suballocation can create a reusable hole subject to address,
alignment, and lifetime constraints; the whole backing is still occupied
until every occupant and outstanding registration/consumer permits release.
Immutable weight slots retain their imported extent layout/content identity;
padding or unused slots cannot host unrelated allocations while whole-extent
reloads may overwrite them. General/mutable reuse must also respect restore
footprints, content generations, and representation compatibility.
The ledger distinguishes reusable suballocated bytes from physically released
bytes. Owning all model address spaces in one process does not change the
provider's minimum unmap/release unit. Moving live contents to consolidate
holes would require a separately validated relocation/completion policy;
compaction is not implied by suballocation.

Pool capacity is distinct from allocation and transfer size. Steady-state
paging reuses backing after old consumers complete; release/create is not
required per read. D-033's baseline retains useful contents and directly
hands compatible backing to admitted replacements. A large slab kept mapped
could instead host software-managed slots with tensor views; 1 GiB slabs do
not imply 1 GiB transfers. Compare this owner-proposed alternative against
retained small handles before changing the baseline (D-035). Include address
stability, backend views, registrations, fragmentation, and pressure-driven
shrink; the current CUDA mapping API does not promise arbitrary interior
offset remapping of a large handle. Growth/retention stays within the node
budget and OS headroom; artifact extents do not fix physical handle size.

Conceptual state machine (real transitions also carry content generations,
consumer counts, and cancellation tokens):

```text
NONRESIDENT -> LOADING -> RESIDENT_UNLEASED <-> RESIDENT_LEASED
     ^                        |
     +------ EVICTING <-------+          LOADING  -> FAILED
                                         EVICTING -> RESIDENT_UNLEASED (safe cancel)
```

A dirty extent cannot become nonresident until recovery is secured or an
explicit discard has invalidated its logical contents.

### Memory classes (§5)

Immutable weights (discard clean copies, restore from artifact) · routed
expert weights (acquire the selected closure only) · dense/attention weights
(acquire what the implementation reads; no assumed activation sparsity) ·
sparse lookup and modality components · live KV / compressed attention /
recurrent state (preserve while resumable: residency, valid spill, or
reconstruction) · reusable completed-prefix state (retain by reuse and
recovery value; invalidate correctly) · scratch (recycle after final
consumers; don't spill dead scratch) · graph/runtime objects and kernel code
(coarse cleanup only, initially) · communication buffers (stable backing for
registrations) · transfer staging (bounded, pre-reserved). Live and reusable
state goes through architecture-specific adapters with conservative
semantics.

### Conversation-state retention

D-024 and D-031 distinguish state required by admitted work from reusable
state kept between requests. A suspended continuation retains the resources
needed to complete or safely unwind; expiry of an idle cache entry cannot
invalidate those resources. After response completion, prefix retention is
subject to bounded memory and spill capacity. Optional sessions and hints can guide
policy without making storage unbounded.

Before M4, specify per-node cache-memory and spill-byte limits, metadata/entry
bounds, idle expiry, and spill cleanup. Shared prompt-prefix snapshots and
conversation-continuation snapshots have separate reuse statistics and
retention/expiry decisions within these common bounds. Shared-prefix value
comes from reuse across conversations; continuation value comes from reuse
of that history. A hit on the shared prefix does not refresh unrelated
continuations. Select and record numeric defaults for both policies from
the measured workload and available headroom. Spill-full or expiry
invalidates only eligible reusable entries; active work retains a valid
recovery path or safely fails under the admission policy. No implicit crash
durability or indefinite retention is promised.

Cache identity covers artifact/model version, relevant execution settings
(including position/attention configuration), state representation/layout,
and the exact rendered token prefix from the context origin plus non-text
input identity when supported. The same system-prompt text after different
preceding input is not the same prefix; matching a message label or its text
alone never authorizes a hit. Tokenizer and template changes must not produce
an incompatible hit. Architecture-specific adapters define which boundaries
can be restored; do not assume a recurrent snapshot can be truncated like
full-attention KV.
Coverage includes the attention window required at the first resumed token,
not merely the snapshot's final token or byte identity. Template rewrites can
rewind a common prefix behind the window preserved by a sequence snapshot,
even when the source context used full-SWA allocation (RE-007). In that case
restore an earlier compatible checkpoint or recompute; successful deserialization
and tail removal do not authorize reuse.
Independent branches may share compatible immutable prefixes, with their
mutable continuation state isolated.

Retain the shared system-prompt prefix independently of longer conversation
snapshots at supported restore boundaries. With inputs `S + A` and `S + B`,
where `S` is the same compatible rendered prefix, both requests may reuse
the immutable state for `S`; neither may use the other's divergent suffix.
Expiring or releasing A drops only A's continuation retention, not S's cache
entry or B's state. Reusing S for a new conversation does not keep A alive.
S remains subject to its own bounded retention policy. Expiry of a cache
entry removes its retention claim, not backing still needed by admitted
work or other retained entries; shared extents are counted once. Eviction
updates affected residency and restore metadata; a cache hit requires a
valid resident or stored representation of every dependency needed to restore.

On a compatible hit, restore state and process new input plus any declared
cache-block tail. If a longer continuation is missing or expired, reuse a
compatible shorter prefix at a valid restore boundary and recompute only the
remaining supplied history. Without a valid prefix, recompute from the
request's full history. If required history is unavailable, fail explicitly.
Expose reused/recomputed token counts and miss reasons through diagnostics
without logging prompts or KV. M4 tests branching histories, edits to an
earlier message, incompatible cache identity, expiry, and spill exhaustion,
alongside both resident reuse and forced spill/restore. Include S+A and S+B
with independent release/expiry, continuation eviction while S remains,
shared-prefix expiry while a consumer is suspended, changed rendering or
preceding context that must miss, and shared-byte accounting. Compare each
branch's logits with its uncached reference and report shared-prefix reuse
separately from longer-history reuse.

### Prepared paging layout (§11)

Making a model available includes import into an immutable paging artifact,
with metadata describing architecture, tokenizer, execution representation,
and the index from logical resources to stored extents (D-009, D-035).
Publication follows complete validation; interrupted preparation is not an
available model. The artifact is a logical unit that may have file shards.
Its container and metadata encoding remain open question 5; runtime paging
does not inherit the source checkpoint's tensor ordering.

The initial Spark profile stores payload extents at 2 MiB file boundaries
with initialized tail padding. Each extent populates compatible independent
VMM backing directly; tensor views refer to logical bytes within that backing.
Small tensors with compatible use/lifetimes may share an extent. Expert-local
and layer-local ranges favor bulk reads, while shared weights keep one
representation and shared ownership. Repacking must honor actual backend
strides and quantization blocks without CPU payload transformations at page-in.

Weight misses fetch whole extents; adjacent missing ranges may form larger
requests when both file and destination ranges are contiguous and protected.
Resident holes are not overwritten to manufacture a sequential read. Sparse
row requests also resolve to whole extents initially, with useful-byte/read
amplification measured separately. This layout favors sequential work inside
a resource group; routing can still select distant groups. No physical NAND
placement or all-sequential workload is promised. Mutable state has separate
spill files and generation/retention rules; metadata need not use 2 MiB I/O.

### Lifecycles (§8)

Page-in: commit capacity for the actual missing extents → obtain backing →
map and set access → transfer → verify completion and content identity →
publish resident → grant lease. Duplicate requests for one content generation
are coalesced.

Eviction: select specific eligible extents → atomically exclude new leases →
wait for all consumers and registrations → write back only if preservation
requires it → commit recoverable state / invalidate discarded entries → unmap
and release or recycle → update occupancy and generation.

Storage backends sit behind one read/write completion interface. D-034 selects
native direct-file I/O into GPU-accessible host VMM on validated Spark
configurations, with bounded asynchronous submission and no CPU payload copy.
Device VMM with a validated DMA staging path remains a provider option;
cuFile compatibility mode is a comparison path, not required for the initial
runtime. Native GDS (supported non-Spark targets) and remote extent transfer
remain later backends. Host-VMM GGML execution and full registration/reclaim
lifetimes are part of the M2 integration proof.

### Routing boundary for MoE (§7)

Prepare input and router dependencies → route → selected expert IDs → resolve
local shards and ranges → acquire the dependency closure → expert compute →
combine → release after consumers complete. Initial implementation: compact
GPU-to-host selected-expert report, native residency decision, all selected
experts acquired before launch, continuation suspended while I/O is pending.
CUDA graphs: residency decisions sit outside captured segments; no CUDA API
calls from host-function nodes.

### Two-node flow (§12)

Placement first (D-020, D-023): the conductor decides which node hosts each
model and routes requests there; a subagent's model on another node while the
main model stays resident needs no collective and no direct link. M4a starts
with configured membership and one configured conductor, capability and
health probes, and request routing with affinity to retained compatible
state. The node runtime remains authoritative for local admission; a stale
cluster view cannot authorize unsafe local execution. Unavailable nodes
cause explicit request failure, not assumed reclamation or silent replay of
an already-started stream. Replica placement, automatic discovery, and
conductor election have separate revisit triggers in plan.md. A busy small
model may later run as replicas on several nodes. Concurrent execution
without paging requires a supported placement whose working sets and complete execution envelopes fit
each node's budget; aggregate pool capacity alone is insufficient. M4a depends
on M4, not on demand-paged MoE. Sharding, below, is M6 for the flagship model.

Describe the phase and local requirements per rank → reserve capacity on all
required nodes → establish local residency → commit distributed execution →
preserve collective order → acknowledge completion or cancellation. Local
eviction victims may differ per rank. Communication buffers come from a
separately budgeted pool with stable backing. TP, PP, and EP are different
plans; port the validated recipe's plan first.

### Repository shape (§20; toolchain file set confirmed 2026-09-21, the rest a sketch)

```text
CMakeLists.txt  CMakePresets.json  mise.toml  mise.lock  .devcontainer/
LICENSE  LICENSES/  NOTICE  REUSE.toml
toolchains/{manifest.toml, artifacts.lock.json}   cmake/toolchains/
include/jitllm/{resource,residency,execution,backend,model_artifact}.h
src/{runtime,scheduler,memory,storage,execution,distributed,management}/
backends/{reference,cuda,optional}/   importers/   tools/
tests/{unit,simulation,cuda,model,distributed}/   benchmarks/   dashboard/
third_party/   docs/
```

`docs/` follows this scaffold (single-file decision and findings logs) rather
than the `docs/decisions/` directory in §20. Backend licensing is explicit;
directory names do not establish legal isolation.

### Conceptual native API (§20; confirmed 2026-09-21 as the starting shape; a sketch, not compilable)

`register_resource(descriptor)`, `reserve_capacity(transaction, envelope)`,
`acquire_group(reservation, dependencies)` → ready | deferred | impossible |
cancelled | failed, `submit(plan, lease, context)` → completion token (lease
ownership transfers to completion tracking), `retire_completed(token)`,
`reclaim(extents)` (validates generations, reports actual bytes recovered),
`cancel(transaction)`. Deferred results refer to owned continuations, not a
blocked global scheduler. There is no runtime plugin ABI (D-028); optional
backends are build-time modules behind the operation contract, which is
finalized after the M2 backend proof.

## Development host baseline

Captured 2026-09-20 on the reference workstation, read-only. The brief asked
for this before choosing pins; it is a baseline, not a pin.

| Item | Observed |
| --- | --- |
| OS | Ubuntu 24.04.5 LTS (noble), x86-64 |
| Kernel | 7.0.0-31-generic (Ubuntu-packaged) |
| CPU / RAM | 16 logical CPUs, 62 GiB |
| NVIDIA driver | 595.91.07, open kernel module |
| GPU | NVIDIA GeForce RTX 3080 Ti, 12 GiB, compute capability 8.6 (not a GB10; useful for local CUDA smoke tests, not as a Spark proxy) |
| CUDA toolkit | not installed (no `/usr/local/cuda*`, no `nvcc`) |
| Clang / LLD / clang-tidy | not installed |
| GCC | 13.3.0 (system) |
| CMake | not installed |
| Ninja, clang-format | present only via `~/depot_tools` (Chromium tooling; not a project pin) |
| mise | not installed |
| Docker | 29.8.1 |
| Python | 3.12.3 |

Implication for M1: the workstation needs LLVM, CMake, the CUDA toolkit, and
mise provisioned by the project's own setup path. A local NVIDIA GPU means
some CUDA smoke tests can run on the workstation before a Spark, but Spark
capabilities (VMM behaviour on GB10, GDS mode, unified-memory accounting) are
only measurable on a Spark. The Spark-side inventory follows.

### Toolchain smoke follow-up (2026-09-21)

The [M0 smoke](experiments/toolchain-smoke/README.md) passed native C++23,
AArch64 cross CPU/CUDA execution on `spark`, and a native Spark fallback.
D-032 pins LLVM/LLD 22.1.8, the measured GCC/libstdc++/glibc components,
NVCC/cudart 13.4.92 (Toolkit 13.4.2), and the hashed target snapshot.
Host and CUDA translation units both use C++23, including an `if consteval`
host/device probe; the installed 13.0 comparison has a dialect limit (RE-001).
GB10 was validated with `sm_121` and PTX JIT disabled. The workstation
compiler SDK was extracted to scratch; the baseline above is historical,
and system compiler defaults and drivers were not changed. Full M1
provisioning is still pending.

## Target nodes (DGX Sparks)

Captured 2026-09-20 over SSH, read-only, no sudo. Both nodes are identical in
software. These names are the owner's environment, not application
configuration (D-023). `spark` (master, also `spark-a`) and `spark-b` resolve from the
workstation and from each other, and SSH is configured in both directions
between the nodes (verified 2026-09-20 with a hop from each to the other).
That initial traffic used management Ethernet. The owner configured the
direct DAC cluster on 2026-09-21; its current link inventory follows the
historical platform snapshot below.

| Item | `spark` (hostname `spark-c4e2`) and `spark-b` (hostname `spark-56f5`) |
| --- | --- |
| Platform | NVIDIA DGX Spark, DGX OS 7.5.0 base with OTA 7.6.0 applied 2026-09-20 |
| OS / kernel | Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, aarch64 |
| CPU / RAM | 20 logical CPUs; 121 GiB visible of 128 GB unified memory, about 118 GiB free at idle |
| GPU / driver | NVIDIA GB10, compute capability 12.1; driver 580.178.04, open kernel module. `nvidia-smi` reports no discrete memory total (unified memory, see D-004) |
| CUDA | Toolkit 13.0 (`nvcc` V13.0.88, package cuda-toolkit-13-0 13.0.3-1) at `/usr/local/cuda-13.0` |
| Storage | One Samsung NVMe (MZALC4T0HBL1), 3.7 TB, root filesystem, about 3.5 TB free. No separate data volume |
| GDS / cuFile | GDS 1.15.1.6, libcufile 2.12, gds-tools installed. `use_compat_mode: true`, `allow_compat_mode: true`, `nvidia_fs` not loaded, cuFile RDMA library not loaded. Matches the compatibility-mode-only constraint in D-004 |
| RDMA / interconnect | Initial pre-DAC snapshot: `mlx5_core`, `mlx5_ib`, `ib_core`, `ib_uverbs`, `rdma_cm` loaded; rdma-core 50.0; no devices under `/sys/class/infiniband` or ConnectX netdevs listed. Superseded by the link inventory below |
| NCCL | No `libnccl2` package installed |
| glibc | 2.39 |
| Distro toolchain | clang 18.1.3, gcc 13.3.0, cmake 3.28.3, python 3.12.3, git 2.43, docker 29.6.2, nvidia-container-toolkit 1.20.1. No ninja, no mise. Distro defaults, not project pins |
| Privileges | Passwordless sudo is configured for the SSH user (owner-stated); nothing in this inventory used it |

Implications for the M0 spikes: the target driver is 580.178.04, so the cross
toolchain's CUDA toolkit pin has to stay within that driver's compatibility
range. The installed toolkit remains 13.0; the later D-032 smoke validated
extracted 13.4.2 components using native GB10 code on the existing R580
driver through CUDA minor-version compatibility. New driver-dependent
features and PTX/JIT paths still need separate validation. The workstation
driver (595.91.07) is newer than the targets', so a kernel that runs locally is not proof it runs on Spark. The I/O spike has one
NVMe and one filesystem to work with, shared with the OS. Direct-link
performance was subsequently validated in the M0 baseline below.

### Direct DAC cluster follow-up (2026-09-21)

The owner reports cluster **`sparky`**, two directly connected devices.
Read-only SSH checks on both nodes confirmed the supplied addresses, link
state, local routes and RDMA-device mappings. These are environment inventory,
not hardcoded application topology or a settled jitLLM configuration format.

| SSH alias | Network interface | IPv4 address | RDMA device / port |
| --- | --- | --- | --- |
| `spark-b` | `enp1s0f1np1` | `10.100.208.1/24` | `rocep1s0f1/1` |
| `spark-b` | `enP2p1s0f1np1` | `10.100.209.1/24` | `roceP2p1s0f1/1` |
| `spark` | `enp1s0f1np1` | `10.100.208.2/24` | `rocep1s0f1/1` |
| `spark` | `enP2p1s0f1np1` | `10.100.209.2/24` | `roceP2p1s0f1/1` |

All four interfaces report `UP`, **200,000 Mb/s** link rate and **MTU 1500**;
their RDMA ports report `ACTIVE / LINK_UP`. Each node's route to its peer's
address selects the corresponding interface and local source address.
The other two ConnectX netdevs (`enp1s0f0np0`, `enP2p1s0f0np0`) are down.
Both active interfaces map to the same right-hand physical QSFP port through
separate PCIe Gen5 ×4 paths (measured 32 GT/s ×4 on each node). The
[NVIDIA port map](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html)
explains these two functions; they are not separate 200 Gb/s cables.

Both hosts have `rdma-core 50.0-2ubuntu0.2` and
`perftest 24.01.0+0.38-1build2`; `ib_write_bw --version` reports 6.20.
`ib_write_bw`, `ib_read_bw`, `ib_send_lat` and `ibv_devinfo` are available.
No host `libnccl2` package was reported by `dpkg-query` in the initial link
inventory; the later baseline used a pinned native build in external scratch.
These inventory checks ran without sudo
and changed no network or driver settings. No transfer benchmark, NCCL test,
end-to-end data validation or GPUDirect RDMA validation was performed in
this inventory update.

The subsequent [M0 baseline](experiments/interconnect/README.md) completed
78 host-buffer test pairs and 27 two-GPU NCCL runs on 2026-09-21, without a
reboot or network/driver changes. Three-run medians: each HCA alone reaches
about 109 Gb/s for 8 MiB host writes; together they reach **184.76 Gb/s**
in either direction (consistent with the owner's approximately 185 Gb/s
Sync result). Combined reads reach **150.10 Gb/s** with default queues.
Bidirectional writes total 369.28 Gb/s, about 184.64 Gb/s each way.
Small 8-byte send latency is 1.39–1.40 µs median RTT/2.

Native `sm_121` NCCL 2.30.7 and pinned nccl-tests 2.20.0 reached
**22.35 GB/s SendRecv**, **22.20 GB/s AllReduce** and **20.40 GB/s AllGather
bus bandwidth** at 512 MiB, default HCA selection, out-of-place medians.
All supported result checks passed; SendRecv's in-place check is unsupported
and excluded. A rounded-to-zero AllGather case is excluded from payload
metrics. Small/medium operation latency varied materially across repeats;
the report records size sweeps and ranges rather than extrapolating peak
bandwidth to generation latency.

All 54 rank logs and per-HCA counter snapshots confirm the selected RDMA
paths. CUDA reports GPUDirect RDMA and DMA-BUF support as zero. NCCL channel
and allocation logs, checked against its pinned source, establish **mapped
host communication buffers**: GPU kernels copy/reduce between user buffers
and those buffers, and the NIC performs RDMA on them. This is not direct
registration of user CUDA allocations or zero staging. Exact copy-byte
counts and CUDA timeline tracing were not measured. Sharded execution,
asymmetric memory pressure, cancellation and failure tests remain M6 work.

Independent inventory review (2026-09-21): a separate agent repeated the
read-only address, link, route, RDMA mapping and installed-tool checks on
both nodes and found the inventory consistent. That pre-benchmark review left
measured throughput, aggregate link capacity and GPUDirect support unproven;
M0 baseline testing and M6 execution/failure testing remain distinct.
No settings changed or transfer benchmarks ran during this review.

### VMM microbench follow-up (2026-09-21)

Three runs on `spark` / GB10, driver 580.178.04, using the D-032 cross SDK,
measured device-local pinned VMM allocations with no export handles.
Minimum and recommended granularity were both **2 MiB**. Host-call latency
ranges below are per-run medians, in microseconds; they exclude SSD I/O.

| Extent | Create | Map | Set access | Unmap | Release |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 MiB | 48.72–53.41 | 0.50–0.54 | 35.94–37.70 | 46.66–63.21 | 26.61–27.04 |
| 8 MiB | 178.57–196.75 | 0.72–1.26 | 72.10–81.10 | 116.77–156.06 | 43.62–46.66 |
| 32 MiB | 797.66–925.62 | 4.40–5.06 | 251.44–261.12 | 321.18–327.59 | 134.56–142.52 |
| 128 MiB | 3384.05–3811.23 | 5.34–5.63 | 804.44–835.35 | 985.80–1010.94 | 449.96–472.11 |

These are idle values; the [report](experiments/vmm-microbench/README.md)
summarizes p95/max and concurrency measurements, with source/build provenance,
commands, and limitations; raw output stays outside Git. All 3,600 timed calls
with independent background kernels returned while their completion events
remained pending.
This does not prove no GPU stalls or model-throughput impact. At 128 MiB,
release medians rose to 618–651 µs with background work.

Reserving 1 GiB of virtual addresses did not change observed free memory.
Creating sixteen 64 MiB handles reduced free memory by about 1034 MiB;
unmapping them while retaining the handles left that footprint intact.
Every word survived remapping and verification. Releasing the handles after
unmapping recovered the allocation, with 4–5 MiB baseline drift in the
system-level snapshots. A deliberate corruption verified the check itself.

D-033 starts with 2 MiB independent physical extents, compatible backing
handoff to waiting admitted loads, and no standing unused-handle cache.
Live useful contents remain resident until policy reclaims them. Granularity
is queried, not baked into core identities or on-disk formats; read batches
can span extents. SSD and model measurements may revise this initial policy.

### I/O path follow-up (2026-09-21)

The [M0 comparison](experiments/io-path/README.md) selects **regular files,
direct I/O, and GPU-accessible host-backed VMM** (D-034), amending D-004's
mandatory staging copy. On the Spark's Samsung PCIe 5.0 ×4 SSD, the native
io_uring path measured 14.903–14.968 GB/s with four 2 MiB reads in flight;
the 180-second run sustained 14.962 GB/s without a sustained thermal decline.
GPU scanning of host VMM matched device VMM at about 242 GB/s. CPU submission
and completion work remains; CPU payload copies and a separate staging copy
are absent from the selected path. Capability checks and DMA evidence are
retained in the report, not inferred from unified memory alone.

Start with two 2 MiB requests for latency-sensitive loads and up to four for
bulk reads: 4–8 MiB of catalog-charged destination backing, with no extra
staging allocation on this path. A device-VMM fallback needs its own bounded,
charged DMA staging buffers. Cold/warm OS-cache measurements are separated;
buffered full-file reads under 100 GiB of held memory forced file-cache
reclamation, whereas direct reads kept file cache empty. Concurrent memory
scans lost about 10% throughput with the in-place path; one physical budget
also means shared bandwidth. The scan is not a GGML/model performance proof.

The SSD's observed interrupt-coalescing feature is a separate latency tuning
point, tested with the original value restored afterwards. Raw block,
NVMe passthrough, and SPDK were not timed because the only drive holds mounted
root; no raw performance advantage is claimed. M2 still validates actual
GGML pointers/kernels and cancellation/registration/reclaim lifetimes; M4
settles mixed read/write scheduling and spill retention/write-rate limits.

### Reference-engine follow-up (2026-09-21)

The [pinned llama.cpp container](experiments/reference-setup/README.md) now
runs the Gemma 4 26B A4B UD-Q4_K_M text GGUF on `spark`, with all layers
offloaded and PTX JIT disabled. The 16.95 GB artifact contains 128 experts
per layer, top-8 plus a shared FFN across 30 layers; the report gives exact
expert closures, scales, non-expert bytes, and reference allocations.
Two short synthetic save/restart/restore tests reused all 627 saved tokens
and matched 32 continuation token IDs. This required `--swa-full`: default
windowed retention restored the API counts but re-prefilled the prompt
(RE-004). At context 8192, f16 KV rises from 460 to 1760 MiB with that
workaround. Docker's cgroup statistics/limit do not establish the node's
CUDA occupancy or a validated physical-memory pressure mechanism. No host
baseline settings changed by setup.

The subsequent [A→B→A reference experiment](experiments/reference-aba/README.md)
passed 27 cycles using Gemma as A and MIT Ornith 1.5 Q4_K_M as B. A's
18,339-token continuation reused 18,297 tokens and processed 42 after restore;
all 118 output IDs matched the resident reference. A separate early/late
notebook recall check also matched and returned the correct facts. Ornith's
94-token recurrent/KV state survived unload/reload, reused its prefix and
matched a seven-token continuation. The report includes exact expert closures,
primary/MTP storage accounting, native LRU behavior and durable-save steps.

An 80 GiB verified locked allocation leaves 41.688 GiB of physical capacity;
the matched reference's combined CUDA model/state/compute buffers need
42.603 GiB before host overhead. With cold incoming file caches, median
first-token waits were 21.232 s A→B and 18.304 s B→A with state restored;
full re-prefill returned to A in 25.236 s. Warm-cache restore returned in
4.062 s and live residency in 0.089 s. Each number has three repeats and
an observed range in the report, together with actual block I/O, 622.424 MiB
logical spill, sampled memory and bounded whole-node swap activity.

Default-SWA restore still re-prefilled all 18,339 tokens (RE-004). Those
normal-optimization probes enforce the same one-model policy; simultaneous
normal placement under pressure was not validated. A decode speed also
differs across live full-SWA, restored and default-SWA paths, so the slower
path cannot alone define the generation comparison. The later
[bounded full paging-feasibility study](experiments/paging-feasibility/full-study.md)
is complete: it covers the exact reference trace, longer alternating
Gemma/Ornith requests, four-sequence decode, and a DeepSeek/Qwen library whose
combined storage exceeds one node's memory. Offline replay compares partial
extent retention, eager active-model loading, and whole-model replacement at
matched budgets, with resident state, bounded spill, and recomputation. It
includes actual allocation accounting and explicit storage/overlap scenarios;
these are not measured jitLLM paging or switching speedups.

Qwen's captured-route estimates are conditional: exact prediction equivalence
failed, including between untraced controls. Byte-identical sequence snapshots
also failed to establish continuation correctness. Gemma snapshots omit SWA
history needed after prefix rollback (RE-007); safe coverage checks select
recomputation, and unvalidated large-model spill reuse is modeled conservatively
with recomputation. The earlier matching Gemma continuation above remains a
narrow historical observation; the validated recompute arm supplies the usable
correctness floor. Restore metadata must describe valid context coverage.

The study supports keeping M4 partial retention ahead of M5 expert paging,
without promising one-layer prefetch can hide misses. Actual pager execution,
physical admission safety, and end-to-end latency validation remain runtime
work. Switching-benefit and generation-stall acceptance criteria still require
owner agreement before M2; no numerical targets are inferred from the study.

## Open architecture questions

The architecture-shaping questions are numbered in
[features.md](features.md#open-questions-answer-during-m0): VMM granularity,
I/O path, async model, first vertical slice, artifact schema, toolchain pins,
dependency mechanism, license and API surface, reservation guarantees. Initial
VMM, I/O, and toolchain answers are recorded above (D-033, D-034, and D-032); the matrix
tracks each question's remaining scope. Purely technical additions to resolve
while drafting:

- Exception policy and error-result type for the runtime; what crosses the
  boundary of optional build-time backends (no runtime plugin ABI, D-028).
- Thread topology of the first scheduler: how many service threads for I/O,
  completion polling, and scheduling, and how continuations are handed off.
- Whether the resource catalog is a single-writer structure with sharded read
  paths or partitioned by model from the start.
- How physical-pool capacity, page-cache usage, and OS headroom are reported
  in one honest memory breakdown on unified memory.
- Initial eviction scoring over eligible extents. Dependency-group scoring
  is deferred until trace replay shows a useful improvement over the baseline.
- Lease granularity and progress for the initial scheduler, now directed at
  turn/step-scoped leases (2026-09-21). Optimistic MoE
  execution is deferred until the pessimistic M5 path is correct and measured;
  its future design must still handle a miss when the current step fills RAM.
- The storage queue's final sleep/poll policy and model-driven tuning of
  request sizes/depths; host-VMM GGML and reclaim validation under D-034.
- Where the conductor lives (inside its node's runtime process or a
  sidecar), how it represents cluster-wide capacity,
  placement, and replicas, and how a routed request's streaming response
  flows back through it. M4a uses one configured conductor; election is deferred.
- Cache-memory, spill, metadata, and expiry limits for D-024/D-031's shared
  prompt-prefix and conversation-continuation policies; choose before M4
  from measured state sizes and available headroom.
- The minimal provider interface the pager needs from a device memory and
  transfer backend (reserve, back, map, unmap, copy, fence, event; transport
  send and receive with registration); the ledger keys by memory domain from
  the start (confirmed 2026-09-21, D-026).
