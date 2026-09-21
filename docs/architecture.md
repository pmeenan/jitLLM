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
- **Vocabulary.** *Virtual reservation* = address space. *Capacity
  reservation* = admission commitment under a progress policy. *Residency
  lease* = protection of specific backing while consumers run. Never
  "reserved" unqualified (D-007). Likewise: read-only ≠ always resident;
  unleased ≠ may be lost; reconstructible ≠ already stored in the right
  format; resident ≠ available without admission (§4).
- **Memory model of the target.** One Spark = one 128 GB unified budget. Two
  Sparks = two domains over a network. Storage on Spark is staged through
  pinned host memory (D-004).
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
  import/execution/eviction/restore evidence (D-009, D-018).
- **Dependency policy.** Own code Apache-2.0; incorporated core implementation
  uses Apache-2.0 / BSD / MIT / MPL-2.0, with other implementation licenses in
  optional, fully removable modules. Declared tools and platform dependencies
  have separate terms and remain in the build audit (D-003, D-017).
- **Language and build.** C++23, Clang-first, NVCC with Clang host compiler
  where validated, pinned libstdc++ initially; cross-built from x86-64 and
  tested on Spark over SSH; declarative pinned toolchain (D-010, D-011,
  D-012).

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
agree generation-stall limits and mixed-workload benefit criteria for the
selected workloads, including a whole-model-switching baseline. Never invent
thresholds or measured results. M5/M7 compare actual results to those criteria
and revisit scope when the evidence does not support them.

## Expected shape (to be validated in the M0 draft)

Where a bullet below leans on a `proposed` [features.md](features.md) row, it
is a design assumption to confirm during feature triage, not settled scope.

### Process and components (§3)

```text
x86-64 workstation: editor / builds / CPU tests / import tools
        | SSH deploy
Spark A: jitLLM runtime  <-- model communication -->  Spark B: jitLLM runtime
        \------------- management API -------------/
                  optional dashboard (separate process)
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

### Lifecycles (§8)

Page-in: commit capacity for the actual missing extents → obtain backing →
map and set access → transfer → verify completion and content identity →
publish resident → grant lease. Duplicate requests for one content generation
are coalesced.

Eviction: select specific eligible extents → atomically exclude new leases →
wait for all consumers and registrations → write back only if preservation
requires it → commit recoverable state / invalidate discarded entries → unmap
and release or recycle → update occupancy and generation.

Storage backends sit behind one read/write completion interface: native file
I/O with pinned staging and cuFile compatibility mode (both compared in the M0
I/O spike), native GDS (non-Spark, later), remote extent transfer (later).

### Routing boundary for MoE (§7)

Prepare input and router dependencies → route → selected expert IDs → resolve
local shards and ranges → acquire the dependency closure → expert compute →
combine → release after consumers complete. Initial implementation: compact
GPU-to-host selected-expert report, native residency decision, all selected
experts acquired before launch, continuation suspended while I/O is pending.
CUDA graphs: residency decisions sit outside captured segments; no CUDA API
calls from host-function nodes.

### Two-node flow (§12)

Describe the phase and local requirements per rank → reserve capacity on all
required nodes → establish local residency → commit distributed execution →
preserve collective order → acknowledge completion or cancellation. Local
eviction victims may differ per rank. Communication buffers come from a
separately budgeted pool with stable backing. TP, PP, and EP are different
plans; port the validated recipe's plan first.

### Proposed repository shape (§20; a proposal, not a commitment)

```text
CMakeLists.txt  CMakePresets.json  mise.toml  mise.lock  dev  .devcontainer/
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

### Conceptual native API (§20; a sketch, not compilable)

`register_resource(descriptor)`, `reserve_capacity(transaction, envelope)`,
`acquire_group(reservation, dependencies)` → ready | deferred | impossible |
cancelled | failed, `submit(plan, lease, context)` → completion token (lease
ownership transfers to completion tracking), `retire_completed(token)`,
`reclaim(extents)` (validates generations, reports actual bytes recovered),
`cancel(transaction)`. Deferred results refer to owned continuations, not a
blocked global scheduler. Do not freeze a public plugin ABI before the first
real backend, paging path, and distributed phase expose their requirements.

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

## Target nodes (DGX Sparks)

Captured 2026-09-20 over SSH, read-only, no sudo. Both nodes are identical in
software. `spark` (master, also `spark-a`) and `spark-b` resolve from the
workstation and from each other, and SSH is configured in both directions
between the nodes (verified 2026-09-20 with a hop from each to the other).
That traffic currently rides the management Ethernet; the direct QSFP link is
still uncabled (see the RDMA row).

| Item | `spark` (hostname `spark-c4e2`) and `spark-b` (hostname `spark-56f5`) |
| --- | --- |
| Platform | NVIDIA DGX Spark, DGX OS 7.5.0 base with OTA 7.6.0 applied 2026-09-20 |
| OS / kernel | Ubuntu 24.04.5 LTS, kernel 7.0.0-1019-nvidia, aarch64 |
| CPU / RAM | 20 logical CPUs; 121 GiB visible of 128 GB unified memory, about 118 GiB free at idle |
| GPU / driver | NVIDIA GB10, compute capability 12.1; driver 580.178.04, open kernel module. `nvidia-smi` reports no discrete memory total (unified memory, see D-004) |
| CUDA | Toolkit 13.0 (`nvcc` V13.0.88, package cuda-toolkit-13-0 13.0.3-1) at `/usr/local/cuda-13.0` |
| Storage | One Samsung NVMe (MZALC4T0HBL1), 3.7 TB, root filesystem, about 3.5 TB free. No separate data volume |
| GDS / cuFile | GDS 1.15.1.6, libcufile 2.12, gds-tools installed. `use_compat_mode: true`, `allow_compat_mode: true`, `nvidia_fs` not loaded, cuFile RDMA library not loaded. Matches the compatibility-mode-only constraint in D-004 |
| RDMA / interconnect | `mlx5_core`, `mlx5_ib`, `ib_core`, `ib_uverbs`, `rdma_cm` loaded; rdma-core 50.0. **No devices under `/sys/class/infiniband` and no ConnectX netdevs listed.** The direct QSFP link is not cabled yet (cable expected 2026-09-21); only the management Ethernet port is up |
| NCCL | No `libnccl2` package installed |
| glibc | 2.39 |
| Distro toolchain | clang 18.1.3, gcc 13.3.0, cmake 3.28.3, python 3.12.3, git 2.43, docker 29.6.2, nvidia-container-toolkit 1.20.1. No ninja, no mise. Distro defaults, not project pins |
| Privileges | Passwordless sudo is configured for the SSH user (owner-stated); nothing in this inventory used it |

Implications for the M0 spikes: the target driver is 580.178.04, so the cross
toolchain's CUDA toolkit pin has to stay within that driver's compatibility
range (13.0 is what the nodes have; a newer toolkit means a driver decision
first). The workstation driver (595.91.07) is newer than the targets', so a
kernel that runs locally is not proof it runs on Spark. The I/O spike has one
NVMe and one filesystem to work with, shared with the OS. The interconnect
half of the inventory is a separate plan task after cabling.

## Open architecture questions

The architecture-shaping questions are numbered in
[features.md](features.md#open-questions-answer-during-m0): VMM granularity,
I/O path, async model, first vertical slice, artifact schema, toolchain pins,
dependency mechanism, license and API surface, reservation guarantees. Purely
technical additions to resolve while drafting:

- Exception policy and error-result type for the runtime; what crosses the
  C ABI boundary of optional backends.
- Thread topology of the first scheduler: how many service threads for I/O,
  completion polling, and scheduling, and how continuations are handed off.
- Whether the resource catalog is a single-writer structure with sharded read
  paths or partitioned by model from the start.
- How physical-pool capacity, page-cache usage, and OS headroom are reported
  in one honest memory breakdown on unified memory.
