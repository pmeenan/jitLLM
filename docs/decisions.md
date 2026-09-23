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
the owner's M0 triage answers and review fixes (2026-09-20) and the
feature-matrix triage of 2026-09-21 (D-028 onward).

---

## D-054: Installed artifacts stay node-local; optional long-term store; one import per cluster with peer replication  (2026-09-22, status: accepted; specializes D-009, D-018, D-034 and D-041)

**Decision.** Model storage has three roles, each a configured path; the
paths must not overlap:

- **Installed store (required, per node).** Published prepared artifacts the
  runtime pages from. It must be a local block-device filesystem that passes
  D-034's direct-I/O capability probe; the runtime rejects anything else,
  such as a network, FUSE or memory-backed filesystem, at startup rather
  than paging through a slower, buffered or memory-consuming path. It opens
  files only beneath that path, without following links or crossing mounts.
  Integrity after publication rests on local permissions, so only jitLLM's
  service user writes there. Paging, restore and spill use only this store
  and node-local spill storage (D-014), never the other roles.
- **Checkpoint store.** Downloaded source checkpoints, kept available for
  re-import under D-018. It defaults to node-local storage. A user may
  instead configure an **optional long-term store**: a network mount such as
  a NAS, or a USB-attached drive.
- **Artifact archive (optional, on the long-term store).** Prepared
  artifacts keyed by content identity and the versions and profile they were
  prepared with (D-035), so a reinstall is a copy plus verification instead
  of a re-import. An archived artifact the target no longer accepts is
  ignored, and the model is re-imported from its source (D-018).

jitLLM sees a long-term store only as a filesystem path. Mounting, protocol
and credentials belong to the OS; jitLLM ships no NFS, SMB or rsync client.
Only import/install job processes access it. The runtime process never does,
so a hung or absent mount cannot stall scheduling or paging (D-005, D-048).
An absent store fails the jobs that need it, explicitly and retryably
(D-041); it is never a startup or inference failure, and installed models
keep running. Credentials, spill and conversation state never go there
(D-014).

Entries are named by a fixed-format content hash and published by a
per-writer temporary write, flush and rename. Jobs open only regular files
there and neither create nor follow symlinks, so several nodes can share
one store without locking and a planted link cannot redirect a read or
write. A long-term store is untrusted input like any checkpoint (D-009):
content is verified against identities held outside the store (the origin's
metadata recorded at download, or the identity recorded when jitLLM
published an artifact), never against hash files kept in the same store.
Verification covers the bytes actually used, such as the staged local copy;
a file verified on the store and then read again is unverified. The
importer stages a source onto local storage before repacking by default;
importing directly from the store needs a measured benefit. Jobs keep
their memory, including cached and dirty file data, within the node's
OS/external headroom (D-050), and their I/O yields to serving; the effect
on stalls is measured when built.

**Cluster installation.** One node pulls the source (from the long-term
store or the origin), imports, verifies and publishes it. Each other target
node receives the prepared artifact from that node over the cluster link,
verifies it against the artifact's integrity data and an identity received
over an authenticated session, not only alongside the bulk data, and
publishes it atomically; an interrupted transfer never appears installed.
A target whose provider rejects the artifact's profile (D-035) fails before
any transfer; another profile needs its own import, not a replica.
Nodes do not import the same source independently by default. This is a
management-plane file transfer between enrolled nodes (D-038 identities),
not the deferred remote extent transfer and not shared storage. The install
job names its target nodes; which node imports and the bulk transfer
mechanism are implementation choices, measured when built. An artifact
prepared off the cluster, such as by workstation-side import (D-009),
enters the same way through one node, which first validates it as
untrusted input. On each node, publication and removal are serialized per
model and checked against job and installed generations, so a cancelled,
timed-out, superseded or retried job never publishes late, resurrects a
model deleted after it began, or removes a reinstalled one.

**Local capacity.** Installation never removes installed artifacts on its
own. If a node the install uses lacks space for its artifact and staging,
beyond its spill budget and filesystem headroom, the user chooses
explicitly, as part of the install, which models on that node to archive
or delete. Otherwise the install fails before transferring anything,
reporting the space required and the installed candidates. Jobs allocate
the checked space before writing, so concurrent spill or installs cannot
exhaust it mid-copy. No default or policy selects victims. Archiving
verifies the archive copy, read back from the store, before removing the
local one. Removal quiesces current users and their outstanding I/O
(D-048) before its space counts as free. Deleting an installed artifact
keeps its source checkpoint; removing a local source is a separate explicit
choice, after which re-import needs a new download.

**Context.** Owner direction on 2026-09-22: downloaded models live on the
owner's NAS and only installed models are staged on the Sparks; long-term
stores (NAS, USB drives) are optional for end-user systems; one Spark pulls
from long-term storage and syncs the processed artifact to the other over
the DAC; install-time space is freed by the user's explicit archive/delete
choice. The [measurements](architecture.md#long-term-model-store-2026-09-22),
single `dd`-based samples, found the NAS mount reading 117–118 MB/s per
client, including both Sparks at once, with NAS-side caching uncontrolled.
Against 14.9 GB/s local direct reads (D-034), the NAS is an install-time
source only. A naive single-stream TCP copy between the Sparks' SSDs over
the DAC moved 8 GiB at 1.05 GB/s (0.45 GB/s through ssh with AES-GCM).
Both are far below the SSDs' local rates and the link's 184.76 Gb/s RDMA
baseline, so they are not transport ceilings. Since both Sparks pulled from
the NAS concurrently at line rate in that sample, one import per cluster is
not chosen for time-to-install alone; the peer copy adds about a second per
GB after import at the naive rate. Its value is one import instead of
several, one read of the source (spinning NAS disks or the internet),
identical content on every node, and no import load on a node that may be
serving.

**Consequences.** M1's installed-layout task defines the role paths,
defaults and configuration keys; long-term stores are absent by default,
and the configuration schema is a versioned public interface (D-016). The
storage service's startup probe covers the installed store's filesystem
type and direct-I/O alignment. D-041's install jobs gain staging,
archive/restore-from-archive, per-node space checks, the explicit
archive/delete selection and peer replication; their delivery milestone
remains unassigned, and replication needs M4a enrollment. The owner's
`/mnt/llm` mount is environment, not application configuration (D-023).
No code, configuration format or wire protocol is introduced here.

**Reopen if.** Remote storage passes the direct-I/O probe with acceptable
latency and paging from it is wanted; staging measurably costs more than
direct import from the store; the single importing node becomes a
bottleneck in larger clusters; or users need automatic space management,
which requires its own decision.

## D-053: jitLLM owns kernel dispatch; kernels are swappable build-time implementations selected per operation  (2026-09-22, status: accepted; amends D-028, specializes D-013 and D-052)

**Decision.** jitLLM's runtime owns operation dispatch on every device. That
covers:

- streams and launch order;
- workspace and scratch;
- library handles such as cuBLAS;
- device and context state;
- fusion choices;
- completion and errors.

No third-party backend runtime dispatches model work: not GGML's CUDA
backend, scheduler or graph-compute loop, nor ExLlamaV3's PyTorch
extension. Their launchers and kernels are reused with build-time
adaptation; GGML's context struct survives only as a jitLLM-populated
launcher argument.

**Kernels are implementations of operations** under the operation contract.
They are compiled in as build-time modules from any compatible source:

- GGML/llama.cpp first;
- ExLlamaV3 (D-052);
- later FlashInfer, CUTLASS or other reused units;
- jitLLM-authored kernels where measurement or a missing capability
  justifies them (D-013 still forbids rewrite-to-own).

**Several implementations coexist.** More than one implementation of an
operation can live in one build and one process. Different models, and
different operations within one model, may use different sources at the
same time.

**The planner selects the implementation.** Selection is deterministic,
per operation, architecture, representation/layout, shape range, device
capability and build profile. The choice is bound into the admitted plan
(D-050) and visible in diagnostics. Adding or replacing an implementation
is a build-time change; moving a plan to another compiled implementation
needs a newly validated plan. No kernel choice is permanent.

**What each implementation declares:**

- supported operations, architectures, layouts and quantization;
- shape and alignment constraints;
- numerical behaviour, including accumulation, determinism, fusion and
  tuning data;
- its workspace and peak-memory bound;
- required library handles;
- graph-capture restrictions;
- completion semantics.

**What each implementation must do.** It launches only on the stream and
workspace it is given. It owns no hidden pools, streams or process-wide
flags. It returns errors instead of exiting or aborting on recoverable
failures. Patches, context adapters or lifted kernels that make a source
comply are reviewed build inputs with provenance.

**Implementation identity is part of the numerical plan and of retained
state's cache identity.** That identity is the source, revision, build flags,
variant, launch configuration and tuning data. State produced under one
plan resumes under another only after validated compatibility; otherwise it
is recomputed. Plans change only at request boundaries or by D-050's atomic
envelope replacement at a completed boundary. A new implementation
is not supported until it has:

- reference comparisons;
- an admission envelope;
- performance evidence for its configuration.

**Build profiles.** A model is supported in a build profile only if every
operation in its plan has an eligible implementation in that profile. That
includes the copyleft-disabled profile (D-017). There is never silent
substitution. D-028's rejection of a runtime plugin ABI stands.

**GGML specifics.** For GGML, first try to reuse its CUDA operation
launchers under a jitLLM-supplied context:

- a jitLLM stream;
- a jitLLM cuBLAS handle and workspace;
- a `ggml_cuda_pool` implementation over charged workspace;
- build-time patches for context ownership, the GB10 device-flag side
  effect and access to the `static` matrix-multiply routing;
- preflight scratch sizing and error propagation through the selected
  launchers, replacing upstream's abort paths. Returning null from the
  pool alone is unsafe because upstream consumers do not check it.

Lift a kernel behind an owned launcher when its launcher needs more change.
The M2 proof chooses per operation. Fusion is a jitLLM plan choice:
GGML's fused launchers are separate implementations.

**Context.** Owner direction, 2026-09-22, after the
[backend-proof](backend-proof.md) source reading. Do not run the backend
unmodified; build-time changes are acceptable. Never be stuck with one
kernel. Use several at once, choose the best per model architecture, and
write our own if needed.

At the llama.cpp pin, GGML's CUDA backend owns things jitLLM must control:

- a never-shrinking scratch pool that aborts when it cannot grow;
- cuBLAS workspaces;
- its own streams;
- a process-wide device flag on GB10.

It also has no custom operation, so EXL3 kernels could only run between GGML
graph segments. And GGML's graph loop chooses kernels and fusions
internally, which blocks per-operation choice.

Its operation launchers, by contrast, take a context. That context's
scratch pool is an abstract interface and its stream and handle members can
be pre-set ([common.cuh](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/ggml/src/ggml-cuda/common.cuh#L1207-L1212)).
ExLlamaV3's wrappers are PyTorch-bound, so the EXL3 plan already rewrote them.

**Consequences.**

- **Plan changes.** The backend proof uses owned dispatch, not GGML's
  backend runtime, and runs EXL3 kernels on the same stream instead of
  between GGML graph segments. It must demonstrate coexistence and swapping.
- **Exact matching gets harder.** Bit-exact comparison with the llama.cpp
  toolchain bridge requires reproducing its kernel, fusion and cuBLAS
  choices, or an unfused plan against a fusion-disabled bridge arm.
- **Maintenance moves to jitLLM.** jitLLM maintains selection logic and
  adapted launchers, and tracks upstream by explicit re-pinning. Upstream
  fixes stop being automatic.
- **Portability.** GGML's other device backends become kernel sources for
  later providers, not runtimes to adopt (D-026). A port therefore also
  adapts dispatch; D-028's "mostly a memory-provider job" no longer holds.
- **Still open for M2.** Exact contract types, the implementation registry
  and the patch set are settled by the proof.

**Reopen if.** Reopen this decision if any of these happens:

- Owned dispatch cannot meet D-052/D-036 performance gates that an upstream
  runtime meets, for example launch overhead without graph capture.
- Adapting a kernel source costs more than it returns.
- A requirement emerges to load kernels without rebuilding. That would also
  reopen D-028.

## D-052: Require an EXL3 companion and upstream performance gates in the early backend proof  (2026-09-22, status: accepted; amends D-028 and D-051)

**Decision.** Keep the first GGML/FP16 control, and require native EXL3
execution alongside it in M2, before settling the operation contract and
initial executable artifact layout. Select published Qwen2.5-0.5B-Instruct
4.0 bpw and mixed-rate 4.5 bpw EXL3 fixtures, with immutable identities and
proof obligations in [exl3-bringup.md](exl3-bringup.md). M3 includes resident
EXL3 serving, and M4 includes it in the switching/restore matrix. Initial
EXL3 implementation and performance work is not deferred to flagship models
or M7.

Preserve the packed trellis, per-tensor rates/codebooks and side-tensor
closure; conversion to GGUF, requantization or a permanent FP16 shadow does
not count. Bounded transient reconstruction for a declared large-prefill
plan is allowed and fully charged. Use selected upstream device kernels
behind native ownership/completion boundaries, not Python/PyTorch serving.
The artifact descriptor must represent both GGML and EXL3 requirements;
storing several alternative layouts of one resource remains a separate
deferred feature. Backends remain build-time modules, with no runtime plugin ABI.

Require an identical-artifact ExLlamaV3 numerical and performance reference
on Spark. The target is parity or better within predeclared measured noise:
kernel/workspace gates in M2, full resident prefill/decode gates in M3, and
paging/switching comparisons in M4. Record matched settings and normal
optimized upstream settings; regressions require a fix or an explicit
owner-approved tradeoff, not an automatic pass. Larger representative kernel
shapes and later real MoE/sharded cases are necessary before broader
performance claims. Numerical and performance thresholds are fixed from
reference controls before evaluating native results.

**Context.** The owner requested an EXL3 quant early because its packing
differs substantially and support and performance must be first-class.
Two small quants of the existing architecture exercise mixed per-tensor
rates, a quantized output head, BF16 embedding, FP16 side vectors/biases and
codebook metadata without adding a new model graph. Source/header inspection
establishes that this is not ordinary integer weights plus a scale.

**Consequences.** M2 cannot close with GGML-only evidence. Run and pin the
external EXL3 baseline in M0/early M1, before accepting native results.
At selection time only metadata, full tensor descriptors and upstream source
had been inspected. The subsequent [Spark baseline](experiments/exl3-reference/README.md)
verifies both full payloads and records reference execution; native
correctness/performance remain owed. Selected MIT kernels can be core-eligible
after compiled-closure audit; the EXL3 format does not imply adopting
optional third-party patches. No runtime or license-policy change follows.

**Reopen if.** A fixture's provenance/runtime compatibility blocks its use,
or measured native port costs require a different early EXL3 checkpoint or
execution plan. Replace it with explicit pins and evidence; neither a failed
small fixture nor later flagship work silently removes the early EXL3 gate.

## D-051: Qwen2.5-0.5B-Instruct FP16 with a pinned llama.cpp numerical reference for the first dense slice  (2026-09-22, status: accepted; resolves open question 4, specializes D-028; early EXL3 companion added by D-052)

**Decision.** M2's early backend proof and M3's first model use Qwen's official
`qwen2.5-0.5b-instruct-fp16.gguf` at repository revision
`9217f5db79a29953eb74d5343926648285ec7e67`, SHA-256
`8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc`.
Keep its F16/F32 tensor mix without low-bit quantization. Its embedded Qwen2
BPE tokenizer, chat template and 8,192-token context metadata are authoritative;
do not replace them with the base checkpoint's different template or 32K
context declaration. The [selection contract](first-slice.md) pins the base
cross-check, numerical profile, reuse boundaries and outstanding gates.

The primary numerical reference is the existing digest-pinned llama.cpp
ARM64 image at source `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, executing
CUDA on Spark; its CPU path is diagnostic. Compare full teacher-forced raw
logits at identical token IDs, positions and execution schedules, with F16
KV, Flash Attention and CUDA graphs disabled and CUDA operation fusion
enabled (upstream default) in the initial profile. Fusion changes these
logits, so its setting is part of the numerical plan; the fusion-disabled
arm is recorded for a native plan that does not fuse. M2
uses fixed IDs; M3 separately validates tokenizer/template/sampling semantics.
Cross-implementation tolerances must be measured and declared before native
acceptance; matching sampled text or inventing a tolerance from a discrepancy
does not pass. Storage recovery remains exact.

Use selected GGML operations and Qwen2 graph/tensor semantics behind
jitLLM-owned backing, workspace and completion. The external reference's
libllama scheduler/loader/KV ownership does not become the native runtime.
The [source/provenance inventory](experiments/first-slice/source-audit.json)
distinguishes MIT implementation candidates, model data, tools and platform
terms. No application implementation is incorporated here. Native tokenizer
adoption is blocked until its generated Unicode-data provenance and D-017
eligibility are resolved; fixed-ID M2 work does not need those tables.

**Context.** A small dense full-KV model separates basic import, execution and
paging defects from MoE/recurrent/SWA complexity. The inspected file includes
small F32 tensors and byte-identical stored embedding/output matrices, giving
the artifact and aliasing proof concrete cases. Explicit model license files
and the existing reference environment make the selected input reproducible.
The published GGUF is the canonical input; exact conversion lineage to the
pinned base safetensors has not been established.

**Consequences.** The [bounded reference check](experiments/first-slice/README.md)
passed on Spark: 76 fixed tokens, all 11,547,136 logits repeat exactly within
CPU and CUDA, including context restoration after 32 tokens. CPU/CUDA logits
differ while all 76 top-1 IDs agree. This is reference evidence only, not
native support, 8K-context validation, a paging/restore ABI or a memory
envelope. Low-bit quantization, the prepared-artifact container, compiled
dependency closure and the early backend-proof scope remain separate work.

**Reopen if.** The M2 GGML/owned-backing proof fails, checkpoint-specific
semantics prevent a bounded first slice, or a newly identified provenance
restriction blocks the chosen input. A different artifact/profile needs new
identity and numerical evidence; no silent replacement by a newer model tag.

## D-050: Guarantee bounded requests with retained-state allowance and complete phase envelopes  (2026-09-22, status: accepted; resolves open question 9, specializes D-007 and the D-019/D-020 time-slicing boundary)

**Decision.** Initial inference admission grants guaranteed capacity for a
finite request under a validated plan. Reserve its maximum retained state
and growth through the admitted context/output bounds, plus the largest
additional physical working set required by any phase to complete or safely
unwind. Include backing granularity, copy-on-write/transition peaks, workspace,
I/O, graphs, registrations, metadata, output and independent cleanup capacity.
Unknown allocations remain non-evictable. Numeric envelopes and limits require
implementation evidence; the [policy](reservation-policy.md) supplies the
accounting rule, lifecycle and adversarial acceptance cases.

The scheduling quantum is one client-facing request/response, not a phase.
Start with one active request per node: it holds the execution slot and its
allowance from its first phase, through I/O waits, dependency discovery,
completed phase boundaries and cancellation, until the request retires or
is explicitly terminated and the scheduler proves a completed handoff. The
node never switches to another request or model mid-request. All admitted
requests' retained-state bounds must coexist with the largest phase envelope
and fixed/non-revocable overhead. Other requests run concurrently only when
the sum of the active members' full phase envelopes also fits; M4 proves the
all-resident concurrent case required by D-020. A phase cannot borrow another
admitted request's live-state allowance or depend on its completion to
escape a capacity wait.

Grants remain lazy and separate from physical occupancy and residency leases.
Useful revocable cache may occupy unused allowance. Opportunistic warming or
prefetch never promises inference progress; its accepted operations retain
accounted capacity until retirement and cannot invalidate guaranteed grants.
Queued work owns bounded intake storage, not partially acquired model phases.
Fair selection at request boundaries, bounded queues/output stalls and
independent cleanup are required. Detect an exceeded envelope before unsafe
allocation/submission; reject an impossible minimum phase or choose an already
validated alternative. An envelope upgrade is atomic at a completed boundary
and cannot become an indefinite wait while retaining partial work.

Initially, admitted retained state keeps its full in-memory allowance even
when spilled. Guarantees do not rely on idle cache expiry or future free disk
space. Completed-request state may become bounded reusable cache under D-031;
mandatory preservation remains protected. Unknown completion quarantines
backing and faults affected admission under D-048, never frees capacity.

**Context.** The owner's turn/step-scoped lease direction (2026-09-21), D-007's
lazy commitments, and D-048's completion protocol need a concrete rule against
two suspended phases holding all capacity while each waits for more. Bounding
only current state or predicting expert locality cannot provide that rule.
The conservative serial envelope makes the progress argument explicit without
assuming a working live-state spill/restore scheduler. On 2026-09-22 the
owner set the time-slicing boundary at client-facing request/response
granularity: never switch mid-prompt or mid-response, and run concurrently
only when both fit. Per-step alternation would reload models on every token
in the primary agent/subagent workload.

**Consequences.** Question 9's planning policy is complete; execution evidence
remains M2's fake-backend and GGML/VMM gates, M3's finite request defaults,
M4's retention/concurrency tests and M5's routed-expert bounds. Worst-case
state allowances and conservative phase sums may reject otherwise schedulable
work. No production memory limit, latency guarantee, native model support,
spill encoding or public interface is introduced here.

**Reopen if.** Measured workloads justify credit for spilled admitted state,
a more permissive safe schedule, or shared concurrent-phase allowances.
Require bounded preservation/restore resources, completion-safe lifetime and
adversarial progress evidence before weakening this policy.

## D-049: Provision a complete, persistent project SDK alongside system-managed prerequisites  (2026-09-22, status: accepted; refines D-012)

**Decision.** The owner accepted a project-managed, version-pinned development
SDK as the default for M1. Provision it automatically into a persistent,
versioned location outside the checkout and temporary directories. System
packages manage declared OS prerequisites and drivers; the project manifest
selects the LLVM tools, compiler support runtimes, CUDA components and ARM
sysroot needed by each build profile. The CPU-only profile must work
without CUDA (D-026).

The SDK must include the complete declared development tool set: compiler,
linker, formatter, linter, language server, symbolizer and sanitizer runtimes,
with target runtimes for the enabled native/cross profiles. Preserve D-032's
validated pins; additional tool pins still require validation. mise and CMake
select explicit tools and scope environment changes to project tasks. Setup
does not depend on changing global compiler defaults or shell startup files.
Native workstation setup and the digest-pinned reference container use the
same provisioning logic, as D-012 requires.

**Context.** The extracted M0 SDK passed the recorded build and sanitizer
checks, but the missing compiler-rt package exposed incomplete provisioning.
Version isolation is useful for the native/cross workflow; extraction alone
does not resolve all dependencies or isolate the compiler from host libraries.
System-installed versioned tools may coexist, but their presence does not
select the project's build toolchain.

**Consequences.** M1 must declare and check host dependencies, provision all
selected SDK components, and verify the same setup on a clean host/container
without relying on undeclared workstation libraries or temporary SDK paths.
Its capability probe reports selected tools, runtime availability and host
prerequisites. No fully self-contained or security-isolated environment is
claimed. Existing `/tmp` experiment paths remain historical reproduction
instructions until M1 provides the persistent setup; this decision does not
mark that implementation complete or change installed runtime packaging.

**Reopen if.** Maintaining the dependency set proves less reliable than
version-pinned system packages or a container-only setup. Preserve exact tool
selection, explicit native/cross targets and reproducible provisioning.

## D-048: Explicit native task states, a single catalog writer and completion-owned lifetimes  (2026-09-22, status: accepted; resolves open question 3; reservation policy follows in D-050)

**Decision.** Start with explicit resumable C++23 task state machines and a
single scheduler/catalog writer per node. Bounded storage, device submission,
device completion, network and CPU-worker services exchange owned commands
and generation-tagged observations. Only the scheduler advances continuations
and changes admission/catalog state; no provider resumes a task inline. Keep
completion harvesting independent of potentially blocking provider submission.
The [design](async-model.md) defines thread roles, submission reconciliation,
cancellation, queue saturation and retirement; actual backend types remain
subject to D-028's M2 integration proof.

Reserve bounded task/operation/result/cleanup storage before provider access.
An accepted operation owns its backing leases and registration references
until all relevant accesses stop and required registration retirement is
confirmed. Client termination and task outcome are separate from resource
retirement. Cancellation is intent, not proof of completion; uncertain
submission/completion quarantines charged backing and faults affected
admission. Partial submissions retain every accepted element. Generation
checks reject stale observations but cannot make late DMA into reused memory
safe; lifetime holds prevent that reuse. Lease retirement is not eviction.

**Context.** Question 3 asks for the task/completion foundation before native
interfaces harden. Explicit states expose suspension storage and ownership
without selecting an execution-framework dependency. Coroutines or a
sender/receiver library could express the same policy but would still need
bounded frames, safe destruction, completion joins and provider adapters.
This is a simplicity/inspectability choice, not a speed comparison or a claim
that alternatives lack compiler/library support.

**Consequences.** The [deterministic CPU-only prototype](experiments/async-model/README.md)
passed on the workstation and `spark`: cancellation/late completion, joined
consumers and registrations, saturated records, partial/early submission,
stale generations, invalid reads and unknown completion. It introduces no
runtime code, public API, wire format or source dependency. It does not prove
multithreaded wakeups, provider behavior, task-tree/coalesced-page-in cleanup,
or capacity-reservation progress. Those remain explicit implementation gates;
question 9's policy was subsequently settled in D-050. Runtime queue/worker counts and
polling policy require implementation measurements; fixture sizes are not
defaults. No changes to D-007's lazy commitment or D-019's completed switching
boundaries follow from this decision.

**Reopen if.** Real M2 GGML/VMM integration cannot satisfy this ownership
protocol, measured scheduler contention warrants partitioning, or explicit
state complexity justifies a bounded coroutine/library layer. Preserve
completion ownership, per-node admission authority and deterministic testing
when changing the mechanism.

## D-047: Correct reasoning wire formats, stateless storage validation and non-streaming response handling  (2026-09-22, status: accepted; amends D-045/D-046)

**Decision.** Following the owner's approval to fix the API review findings:

- Current vLLM and OpenRouter Chat Completions both use `reasoning` for
  reasoning text. `reasoning_content` is a legacy spelling, supported only
  when a pinned older client profile requires and tests it. OpenRouter's
  `reasoning_details` remains the structured round-trip extension.
- jitLLM's signed reasoning blocks use the SDK-supported neutral
  `format: "unknown"`; jitLLM identity and version belong inside the opaque
  signature, not a new `format` enum value or another provider's signature.
  The exact pinned client/provider package must preserve text, signatures,
  ordering and indices through streaming and tool-result pass-back before
  support is advertised. M1 versions the opaque signature representation.
- Stateless Responses accepts `store: false`; omission means false for this
  profile. Explicit `store: true`, non-boolean values, non-null
  `previous_response_id` and conversation references receive a protocol-shaped
  400 before admission. No response retrieval is promised.
- D-045's immediate first event and SSE keepalives apply only to requests
  selecting SSE streaming. Non-streaming requests, including `stream: false`,
  receive one JSON result or protocol-shaped JSON error, with headers held
  until that outcome is known. Never insert SSE events/comments into JSON.
  Each tested client profile records its non-streaming timeout bound;
  unsupported long waits are documented, not hidden by changing transport.
  A server request deadline uses a protocol-shaped 504 error before headers
  and initiates completion-safe cancellation; disconnects also initiate
  cancellation. Neither permits backing reclamation before work completes.
  Pre-admission queue expiry remains D-045's 429. No automatic replay or
  exactly-once inference guarantee follows from a timeout.

**Context.** [Current vLLM documentation](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
renames `reasoning_content` to `reasoning`. OpenRouter's official provider
[reasoning schema](https://github.com/OpenRouterTeam/ai-sdk-provider/blob/1b22b05352cb0f9243a6c3fdd326038dd3705544/src/schemas/reasoning-details.ts)
and [format enum](https://github.com/OpenRouterTeam/ai-sdk-provider/blob/1b22b05352cb0f9243a6c3fdd326038dd3705544/src/schemas/format.ts)
discard blocks with unknown enum values; its neutral `unknown` value is
supported. Its [stateless Responses contract](https://openrouter.ai/docs/api_reference/responses/overview)
rejects explicit storage requests. The baseline promises both JSON and SSE.

**Consequences.** These rules supersede D-046's custom-format and vLLM-field
wording, and narrow D-045's keepalive rule. The baseline and assessments carry
these corrections and acceptance cases. No runtime compatibility is claimed;
M1 versioning and execution evidence remain ahead.

**Reopen if.** A pinned client cannot preserve neutral-format signed blocks,
or a required client needs stored Responses or different timeout semantics.

## D-046: Adopt OpenRouter's extension vocabulary on the OpenAI-shaped routes; exclude its hosted-routing features  (2026-09-22, status: accepted; extends D-041, D-043 and D-045; reasoning wire spelling amended by D-047)

**Decision.** The owner triaged the [OpenRouter assessment](openrouter-api-assessment.md):

- **Model metadata (accepted).** `/v1/models` entries carry OpenRouter's
  metadata fields: `context_length`, `architecture` (modalities, tokenizer,
  instruct type), `top_provider.max_completion_tokens`, `supported_parameters`,
  `default_parameters`, `per_request_limits` and `hugging_face_id`, with values
  from the artifact, configured limits and the implemented profile only.
  Pricing and uptime are omitted, never invented. M4a's cluster-availability
  view uses the per-model `endpoints` shape: one entry per node or replica
  holding a prepared artifact, `quantization` from the artifact representation,
  `status` from admission readiness. The native discovery document remains the
  authoritative superset.
- **Reasoning and cache reporting (accepted).** Chat Completions accepts the
  `reasoning` request object (`effort`, `max_tokens`, `exclude`, `enabled`)
  and emits `reasoning` text and `reasoning_details` blocks, signed by jitLLM
  under its own `format` value, alongside or instead of vLLM's
  `reasoning_content` as the profile selects, all under D-043's reasoning
  contract. Usage reports `prompt_tokens_details.cached_tokens` and
  `cache_write_tokens` from real prefix reuse only.
- **Hints (accepted); fallback spelling reserved.** `session_id`, `user` and
  `metadata` are advisory affinity, attribution and retention preferences
  under D-045's signal rules, never conversation identity, retention grants
  or authorization. Alternative-model fallback remains a D-042 design
  suggestion; if it is ever accepted, OpenRouter's `models` array with
  `provider.require_parameters` and `provider.quantizations` is its opt-in
  spelling, with the served model reported. No fallback behavior is
  implemented by this decision.
- **Excluded (rejected).** `plugins`, `transforms`, the auto-router, routing
  suffixes, `provider` preference fields with no local meaning, pricing,
  credits, `service_tier` and generation stats. `transforms` and `plugins`
  are rejected explicitly at the wire, never ignored; cost fields are
  omitted, never reported as zero.

No "OpenRouter profile" is added; these are spellings on the existing
OpenAI-shaped routes.

**Context.** Owner triage on 2026-09-22 of the owner-requested assessment.
OpenRouter's vocabulary is what most agent clients' "OpenRouter" provider
modes already parse, so adopting it where a gap exists lets unmodified
clients use the feature (D-043). The metadata schema gives D-041's M3
discovery requirement a shape clients already read.

**Consequences.** Delivery: metadata fields ride with M3 discovery, the
endpoints shape with M4a cluster availability, reasoning and cache fields
with D-043's delivery milestone when the ladder is rewritten, hints with the
D-022 session extension. Evidence rule unchanged: a named client run in
OpenRouter mode against a custom base URL (OpenCode is the candidate) with
its version, provider package and configuration pinned before any
OpenRouter-mode compatibility is claimed. `supported_parameters` lists only
what the profile implements; `context_length`, modalities and quantization
come from the artifact and validated support. Pass-back of
`reasoning_details` follows D-043's ordering and immutability rules. M1
versioning names any jitLLM `format` value. No implementation is claimed.

**Reopen if.** A named client requires an excluded field, OpenRouter changes
the schema under a pinned client version, or fallback is accepted under
D-042 and needs the reserved spelling made concrete.

## D-045: Front-door listener, auth and CORS defaults; admission status and keepalive contract; standard-client signals and alias echo  (2026-09-22, status: accepted; extends D-014 and D-040–D-044; OpenRouter vocabulary in D-046; streaming scope amended by D-047)

**Decision.** At the owner's direction after review of the D-040–D-044
documents, the inference front door adopts these public-interface rules:

- **Listeners.** One inference front door per conductor serves `/v1/*`, the
  Ollama `/api/*` profile and read-only discovery on one configurable port.
  The management API is a separate listener, local-only by default (D-014).
  jitLLM does not claim port 11434 by default; Ollama-native clients are
  pointed at the front door. `GET /` liveness text and `GET /api/version` are
  served only with the Ollama profile enabled, and `version` reports the
  Ollama release the profile was tested against alongside a field naming the
  real server version.
- **Authentication.** A loopback-bound front door accepts anonymous requests
  until an inference credential is configured, ignoring any placeholder
  credential a client presents; configuring one turns anonymous access off
  unless explicitly re-enabled, after which a request carrying both
  `Authorization` and `x-api-key` must validate on each. Any non-loopback
  binding requires credentials and transport protection. Authorization is
  decided per operation, never by path prefix; an inference credential never
  carries management authority; discovery output is filtered by caller
  authority.
- **CORS and origin checks.** Loopback origins are allowed by default,
  matching Ollama; other origins require a configured list, and a request
  whose `Origin` is outside it is refused before any work. JSON routes
  require `Content-Type: application/json`, so a browser's no-preflight
  request cannot trigger inference or a release. On a loopback binding the
  `Host` header must name a loopback address, the machine's hostname or a
  configured name (the DNS-rebinding guard Ollama applies). A wildcard origin
  is accepted only on a loopback binding with a credential configured, never
  together with anonymous access.
- **Admission outcomes.** Malformed or unsupported requests and context
  exhaustion are 400 in the protocol's error shape, using documented phrases
  where a client acts on them; unknown model IDs are 404; a known model with
  no prepared artifact anywhere is 503 with `x-should-retry: false` and never
  triggers a download; oversized input is 413; exceeded queue waits, full
  queues and budget refusals are 429 with an integer `retry-after` of at
  most 60 s and `x-should-retry: true`; overload or draining is 503 with the
  same `retry-after` bound. A switch, warm or prefill in progress is not an
  error. After headers, failures use the protocol's in-stream error form and
  end without a success marker.
- **Keepalive.** Response headers and the first protocol event are sent as
  soon as a request is validated and admitted, before weights load; then
  keepalives at a pinned interval (`ping` on Messages, SSE comment lines on
  Chat Completions and Responses) through switches and prefill. Long switches
  are never signalled through `retry-after`. Model listing answers from the
  catalog with no I/O and no redirect. Each profile pins its interval and the
  client bounds it stays inside.
- **Standard-client signals.** Advisory only, never conversation identity or
  a retention grant (D-031): Claude Code's `x-claude-code-request-class`
  maps to D-042 priority classes (`auxiliary` is background, the rest
  interactive); `x-claude-code-context-compacted` releases the prior
  continuation with D-041's close semantics, located by affinity through the
  always-sent `x-claude-code-session-id`/`x-claude-code-agent-id` values
  within the caller's scope, a no-op when nothing matches. Both are opt-in
  hint headers on a custom base URL (`CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`,
  v2.1.273+), which the pinned profile sets; absence means default policy.
  `cache_control`, `prompt_cache_key` and comparable fields are retention
  preferences; Ollama `keep_alive` maps to `0` = release the requester's own
  residency lease after its request (eligibility, not eviction, D-007; other
  consumers untouched), positive = advisory retention preference, negative =
  explicit rejection.
- **Model listing and aliases.** `GET /v1/models` serves the Anthropic list
  shape when the request carries `anthropic-version` or `x-api-key`, else the
  OpenAI shape; `GET /v1/models/{id}` returns one entry. Entries name the
  actual model behind an alias. The response `model` field echoes the
  requested alias; the resolved artifact identity travels in a jitLLM
  response header and in discovery and diagnostics. Extensions use namespaced
  headers on every protocol and namespaced body fields only where the
  protocol tolerates unknown keys; exact names follow M1 versioning.
- **Claude Code profile.** The attribution block is stripped when it arrives
  unchanged as the first `system` entry and consists solely of the block, so
  prefix identity excludes its per-conversation fingerprint; it is never
  logged. Own thinking blocks are
  signed and unverifiable ones rejected with the documented wording; adaptive
  thinking on a non-reasoning model is rejected naming the field; auxiliary
  and background requests alias to the main model by default.

**Context.** The review on 2026-09-22 checked the live Claude Code gateway
protocol page, Codex configuration reference, Ollama FAQ/API references and
OpenRouter documentation. Claude Code's opt-in `GET /v1/models` discovery
(3 s timeout, no redirects, `claude`/`anthropic` ID filter), its 300 s
silence watchdog, its `retry-after` and `x-should-retry` handling, its
opt-in request-class and context-compacted hint headers, its default-on alias
fields and its error-wording recovery paths are all documented and were
missing from the baseline. Codex documents a 300 s stream idle default and 4/5 retries. Ollama
clients send no credentials, probe `GET /` and `/api/version`, and send a
positive `keep_alive` by default; Ollama's server guards loopback bindings
with a `Host` check. Switch latency under D-036 makes time to
first byte the binding client constraint.

**Consequences.** The [baseline](client-api-baseline.md) carries the tables
and per-profile handling; the [capability assessment](api-capabilities.md)
carries the Ollama deployment shape. M1 names the extension headers and
version fields. M3 acceptance verifies every status row, the keepalive rule
through an induced switch, discovery timing, the recovery paths, the
context-compacted release and the anonymous-loopback and CORS defaults.
Anonymous loopback access is a single-owner default under D-014 and vision.md's
multi-tenant non-goal, not an isolation claim. The `keep_alive` mapping is a
documented partial Ollama profile, not lifecycle compatibility. No
implementation, runtime dependency or architecture change is implied.

**Reopen if.** A named client's documented bounds or recovery wording change
under a pinned version, a deployment beyond a single owner's local nodes is
adopted (D-014), or a client requires the resolved identity in the standard
`model` field.

## D-044: Confirm compatible reranking, monitoring and completion APIs; bound specialized scope  (2026-09-22, status: accepted; extends D-043; front-door contract in D-045)

**Decision.** The owner approved the remaining vLLM API-triage group:

- Reranking later alongside embeddings, using existing `/rerank`, `/v1/rerank`
  and `/v2/rerank` contracts where supported, tested with unmodified retrieval
  clients and validated ranking models.
- Prometheus `/metrics` and compatible health/load queries. Reuse metric names
  only where their meanings match; expose jitLLM paging measurements separately.
- OpenAI-compatible `/v1/completions`, standard log-probability fields and
  bounded vLLM-compatible token diagnostics for evaluation/completion tools.
- Defer LoRA until a concrete adapter workload needs it. Classification,
  reward and generic pooling remain workload-driven. Generic worker RPC,
  training controls and split-serving deployment APIs are excluded from the
  client baseline; D-043's compatible prompt-rendering endpoints remain in scope.

**Context.** Owner approval completes vLLM API triage. Direct wire compatibility
under D-043 governs; this does not claim every protocol version, model or field
already works. The [assessment](vllm-api-assessment.md) retains failure cases
and per-feature validation requirements.

**Consequences.** Delivery milestones remain to assign when rewriting the
ladder; existing milestone gates are not expanded by implication. Pin each
selected route's request/response/error/stream contract and test actual clients.
Metric compatibility includes units, labels and aggregation semantics, not just
names; unknown measurements are not invented. Limit logprob/token output and
rerank inputs. Raw completions must preserve their templating semantics.
LoRA and classification/reward/pooling are earliest M7 planning after validated
base-model execution and concrete workload demand, not automatic deliverables.
No execution, runtime dependency or architecture change is implied.

**Reopen if.** Named clients require additional contracts or a concrete workload
justifies one of the deferred specialized capabilities.

## D-043: Prefer direct API compatibility; confirm tokenization, constrained output and reasoning contracts  (2026-09-22, status: accepted; extends D-040–D-042; further scope in D-044)

**Decision.** The owner requires direct compatibility wherever practical:
reuse established routes, request fields, response shapes, streaming and error
behavior, verified with unmodified tooling against a pinned version/feature
profile. jitLLM-specific features use separate extensions. Document unsupported
features explicitly; matching an endpoint name alone is not compatibility.

The first vLLM follow-up triage group is approved:

- vLLM-compatible `/tokenize`, `/detokenize`, `/tokenizer_info` and compatible
  prompt-rendering endpoints for supported request formats, using the same
  tokenizer/template as inference.
- Standard `response_format` JSON-object/JSON-schema requests and strict
  function arguments, plus vLLM's `structured_outputs.json` request form.
  Begin with a documented schema subset; reject unsupported constraints.
  Regex/grammar extensions are deferred until a concrete client needs them,
  earliest after the validated JSON/schema implementation.
- Protocol-specific reasoning/final/tool fields and streaming, including
  vLLM-compatible reasoning output. Expose thinking controls only where the
  model implements them, advertise capabilities and reject unsupported settings.

**Context.** Owner approval after requesting seamless use of existing tooling,
following the [vLLM API assessment](vllm-api-assessment.md). This is a wire
compatibility requirement, not numerical equivalence to vLLM or adoption of
its runtime/process architecture.

**Consequences.** Assign delivery milestones when rewriting the ladder; no
additional M3 gate is implied. Pin compatibility fixtures and client versions,
including negative/error and interrupted-stream cases. Bound preprocessing,
schema compilation and generated state. Rendering cannot expose unauthorized
server prompt material; structured-output truncation is not successful schema
completion. Reasoning controls remain model-specific, and provider signatures
or encrypted state are never fabricated. Exact profiles and schemas precede
implementation; M1 establishes versioning. Remaining vLLM proposals still
await triage.

**Reopen if.** A required client's protocol cannot be supported without violating
resource/lifetime or privacy invariants, or a protocol revision changes the
selected compatibility contract.

## D-042: Confirm staged multimodal input, MCP management, sharing controls and embeddings  (2026-09-22, status: accepted; extends D-041)

**Decision.** The owner approved the second API-triage group:

- Text resources, images and audio files are confirmed input scope, delivered
  incrementally with validated models. Live audio/video is deferred until a
  concrete workload establishes streaming and synchronization requirements.
- An optional MCP management adapter follows the native management API. It
  exposes discovery/status and explicitly authorized actions in a separate
  process; ordinary client tool execution stays outside the inference runtime.
- Application permissions, interactive/background priority, maximum queue
  waits, cancellation and bounded progress events are confirmed for cluster
  sharing within the single-owner workload.
- Embeddings are confirmed later scope with a validated embedding model.
  Batch/background inference is deferred until a concrete workload justifies
  its scheduling and storage requirements. Background priority for ordinary
  requests does not imply durable background jobs.

**Context.** Owner approval of the second group in the
[API assessment](api-capabilities.md), completing the requested feature triage.
Input support is earned per artifact/backend; it does not imply media output,
all model families, or production multitenant isolation.

**Consequences.** Delivery milestones for these additions remain to assign
when rewriting the ladder; existing M3/M4/M4a gates are not expanded by
implication. Deferred live audio/video and batch work are earliest M7 planning,
only if their workload triggers fire, not automatic M7 deliverables. Before
implementation, settle versioned schemas, numeric bounds, permissions and
backend/preprocessing evidence. Media/resource processing must stay budgeted;
MCP cannot bypass management authorization; timeouts do not prove consumers
finished. The assessment's optional model fallback, affinity and other details
not in the approved group remain design suggestions, not accepted interfaces.

**Reopen if.** A supported workload requires different modality transports,
MCP roles, durable jobs or isolation beyond the single-owner contract.

## D-041: Add a tested Ollama subset, discovery, continuation close and management jobs  (2026-09-22, status: accepted; extends D-040; further scope in D-042)

**Decision.** The owner approved the first four API-triage recommendations:

- A tested Ollama subset for model listing/details and chat/generation.
  Preload/unload semantics are a separate follow-up, not silently mapped to
  hints. Ollama registry downloads and other model-management compatibility
  are deferred until a named client needs them; earliest work follows the
  basic subset and the relevant native management operation.
- Machine-readable API schemas, supported features and per-model capabilities
  and limits in M3; cluster availability follows in M4a.
- A final-request flag and explicit idempotent continuation release in M4,
  targeting one conversation, preserving independent shared-prefix retention
  and waiting for outstanding consumers before reclaiming backing.
- Download and warm jobs with progress, status, cancellation and retry support.
  Installation succeeds only after verification, preparation and publication.
  Warming remains capacity-constrained; inference never implicitly downloads
  an absent model.

**Context.** Owner approval during API triage, following the documented
[assessment](api-capabilities.md). Listing, model selection, automatic
activation, separate system content, HF imports and optional release already
had confirmed scope. This settles additions to their public API behavior.

**Consequences.** Ollama compatibility is profile-scoped and requires a named
client test; unsupported lifecycle controls fail explicitly. Native memory and
completion invariants remain authoritative. Exact schemas, numeric bounds,
Ollama subset delivery and job delivery milestones remain planning work, not
new M3 gates by implication. Multimodal input, MCP and broader cluster-sharing
extensions remain proposed. M1 versioning precedes implementation; no released
API exists to bump. The assessment records lifecycle races and import/job
failure cases to test before implementation acceptance.

**Reopen if.** A named client requires additional Ollama semantics, or discovery,
close or job behavior cannot preserve bounded admission and completion safety.

## D-040: Serve Chat Completions, Responses and Messages in the M3 baseline  (2026-09-22, status: accepted; follows D-022/D-030; front-door contract in D-045)

**Decision.** M3 serves `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/responses`, `POST /v1/messages` and
`POST /v1/messages/count_tokens` through the inference front door. The
[client API baseline](client-api-baseline.md) defines the text/tool subset,
JSON/SSE behavior, unsupported-feature policy and required client evidence.
Responses initially uses stateless full-history HTTP/SSE; optional sessions
remain independent. Token counting is included by choice, not because Claude
Code requires it. This planning contract has no released API version to bump;
M1 establishes versioning before implementation.

**Context.** The linked official documentation checked on 2026-09-22 shows
that current Codex requires Responses, OpenCode selects its wire format by
provider, and Claude Code uses Messages. Cursor's BYOK docs describe chat and
server-mediated routing but do not establish arbitrary local-model/tool
compatibility. Its exact custom wire behavior is an explicit M3 validation gap.

**Consequences.** Chat Completions alone cannot fulfill the named-client goal.
All three protocols need schema/stream/tool tests; advertise compatibility only
for executed client versions and configurations. M3 retains its at-least-one-
named-client end-to-end gate, not an all-client compatibility claim. Local-only
operation remains the default; Cursor does not authorize remote exposure.
Unsupported semantic features fail explicitly rather than being silently lost.

**Reopen if.** A required client workflow cannot use this bounded surface via
supported configuration, or client verification establishes another required
endpoint, tool type or transport.

## D-039: Detect canonical QSFP layouts and scan dedicated cluster subnets during setup  (2026-09-22, status: accepted; amends D-038)

**Decision.** Setup explicitly proposes single-node, likely direct pair,
likely direct triangle, and switched/shared-fabric N-node layouts. Count
physical QSFP port groups, not the Spark's two host interfaces per port;
merge addresses into nodes only with verified peer identity. Zero connected
ports suggests single-node only with a complete inventory. One active port
and one reciprocal peer suggests a direct pair; three nodes with two ports
each and reciprocal edges form a triangle. Multiple peers reachable through
one port suggest a shared fabric, including 4+ nodes. Incomplete or conflicting
observations retain a partial graph rather than inventing missing membership.

Under the owner's dedicated-QSFP-network assumption, a setup window scans
existing on-link IPv4 subnets on detected QSFP paths by default, with bounded
probe rate, concurrency, memory, range and time. This replaces D-038's no-sweep
rule. Management interfaces/default routes are excluded; IPv6 uses scoped
neighbor/multicast discovery instead of address-space scanning. Responding
IP/MAC pairs are candidates, never automatic enrollment. ARP can reveal the
same peer MAC through a switch, so a direct cable remains a topology inference
unless independently corroborated. Report observed node counts and scan
coverage separately; silent peers do not become proof of absence.

**Context.** Owner's follow-up on 2026-09-22 specifies single, double-direct,
triple-direct and N-switched patterns and requests subnet scanning for the
dedicated network. [The classifier and scan contract](cluster-design.md#layout-classifier)
turn those patterns into explicit requirements while preserving D-038's
identity and per-port accounting. No additional hardware was available to
validate triangle/switched deployment behavior; no active scan was run here.

**Consequences.** Experimental shared/local TOML schema **v2** adds the
`network.subnet_scan` policy and `initial-v2` limit profile; these supersede
the unimplemented v1 drafts.
Internal transport remains protocol v1. M4a validation covers all four layouts,
ambiguous/partial results and scan bounds. The authority, enrollment, local
admission and failure contracts are unchanged; existing membership never
shrinks automatically when a cable or peer disappears.

**Reopen if.** The dedicated-network assumption is unsuitable for a deployment,
or address allocation/forwarding is needed to make the physical topology
usable. Such provisioning remains an explicit operation outside this detector.

## D-038: Autodetect initial network paths; enroll a configured cluster with fenced control sessions  (2026-09-22, status: accepted; refines D-023 and implements D-037's cluster-design gate)

*Same-day follow-up: D-039 adds canonical layout classification and dedicated-
QSFP subnet scanning; its schema v2 supersedes the v1 draft below.*

**Decision.** Initial setup detects interfaces and candidate peer paths,
particularly the known Spark QSFP layout, and proposes the cluster layout.
The initiating node is the proposed conductor. An explicit enrollment writes
one designated conductor and the approved node identities; later startups
automatically detect and validate paths for those members. Bootstrap discovery
is in M4a scope, as the owner requested. Automatic membership changes,
conductor election and automatic replica placement remain deferred.

A versioned hardware profile groups Spark netdevs by adapter and physical
port metadata, corroborated with PCI/devlink/RDMA information, not by names
or IP conventions. The measured two active PCI paths share one physical
200 Gb/s port; speed is not additive. Generic discovery and explicit
selectors remain available for unrecognized hardware. Detected addresses,
carrier and neighbor entries do not establish peer identity, cabling topology
or authority. Discovery changes no OS networking settings. Existing Sync
configuration is optional evidence, not a prerequisite or a jitLLM trust store.

Use experimental **TOML cluster schema v1**: a shared membership document with
cluster UUID, revision, designated conductor, enrolled node IDs/public-key pins
and path policy, plus a local identity/credential/listener/limit document.
Configuration changes are explicit, atomic and fail closed on mismatched
membership digests; no hot membership/trust reload initially. Internal
control/response transport uses mutually authenticated TLS 1.3 over TCP with
bounded length-prefixed JSON, protocol v1. Bootstrap mDNS is setup-only and
untrusted; enrollment uses authenticated administrative access or local import.
Remote client binding remains protected under D-014 and the named-client
endpoint design; inference/management still have one front door under D-037.

Conductor epochs are durably monotonic; workers retain accepted epoch floors.
Fresh worker-issued sessions fence prior connections at local admission,
qualify request IDs, and use sequence high-water marks to reject old requests
after bounded result-cache retirement. Reconnect reconciles or cancels old
attempts; changing a transport path never silently replays them. Moving the
conductor requires stopping/fencing the old authority, not winning a timeout.
Ready nodes still grant capacity locally under D-007/D-037. Shared state
hints retain D-031's independent prefix/continuation lifetimes.

**Context.** Owner's request on 2026-09-22: autodetect the initial layout as
much as possible from interfaces, particularly QSFP, given the known physical
configuration. Read-only checks on both Sparks established usable adapter/
physical-port identifiers independent of interface naming; current NVIDIA
port documentation corroborates the grouping. The
[design and evidence](cluster-design.md) records what was observed, what
cannot be inferred, config/transport semantics, initial numeric coordination
bounds and required validation. Those bounds are policy choices, not hardware
measurements or a substitute for question 9's progress policy.

**Consequences.** D-023's discovery deferral now concerns automatic membership
changes, not setup assistance or address/path refresh for known members.
M4a gains a bounded setup workflow without mandatory subnet editing. No
runtime code, cryptographic library, parser dependency, network configuration
change, distributed model support or benchmark result is introduced here.
M1 selects audited native dependencies and packaging/diagnostic conventions;
M4a must validate the protocol catalog, discovery, trust, crash recovery,
backpressure and adversarial cases in the design before claiming support.
M6 transport/collectives and numeric resource-progress guarantees remain
separate. Evolving the experimental schema/protocol requires explicit
versioning; no compatibility with an existing released format is implied.

**Reopen if.** The measured hardware metadata ceases to distinguish shared
ports, an unsupported network needs forwarding or automated network creation,
or membership churn/independent conductor upgrades justify a stronger cluster
control protocol. Preserve explicit trust and node-local memory authority.

## D-037: The conductor is an in-process role; admission authority stays per node  (2026-09-22, status: accepted; resolves D-020's process-location question)

*Cluster-design follow-up (2026-09-22, D-038): the initial configuration,
network discovery and fenced control-session design are now recorded; their
implementation validation remains ahead.*

**Decision.** The configured conductor runs inside its node's native jitLLM
runtime, including the single-node deployment. It owns the client inference
and management front door, placement policy, bounded request routing, and a
cluster view of node reports. Every node, including the conductor's own node,
retains sole authority over its catalog, capacity commitments, residency
leases, execution and completion. Local dispatch uses the same admission
contract as remote dispatch; it cannot bypass the local budget.

The cluster view represents **separate memory domains**, not a pooled capacity
reservation. It records node/runtime incarnations and report freshness,
capabilities, complete-budget summaries, model-instance placements and
compatible retained-state hints. It also tracks routed attempts and their
node-issued admission outcomes. Reports and placement intent never grant
capacity. A model instance's identity survives partial eviction; a placement
record does not promise residency or state validity.

For M4a, the conductor selects one node for a whole-model attempt, and that
node atomically validates the execution envelope against its live commitments
before admission. It grants, defers within a bounded queue, or rejects; only
its scheduler can acquire residency and start work. Retransmission of an
attempt must not create a second execution. After uncertain dispatch, the
conductor cannot reroute until the original node establishes that execution
never began and can no longer begin. Timeouts, lost acknowledgements and a
lack of streamed tokens do not establish that fact. Started or uncertain
attempts fail explicitly when they cannot be resolved; they are not silently
replayed. Client retries are new requests, not an exactly-once guarantee.

Responses stream back through the conductor with bounded buffering and
backpressure. Cancellation propagates to the owning node; client completion
or disconnect is distinct from resource retirement. A lost node remains an
unknown domain, not reclaimed capacity. Incarnation checks reject stale
commands and reports; reconnect/restart reconciles existing attempts before
new admission. There is no automatic conductor failover, election or stream
resumption in M4a. Replacing the designated conductor requires fencing the
old authority, not just declaring it unhealthy.

**Context.** The next M0 task under D-020 asks where coordination lives and
how admission and placement are represented. An in-process role gives the
single-node runtime the same front door and ownership model as a cluster,
without a second local scheduling authority or a mandatory sidecar lifecycle.
A sidecar would isolate front-door failures and allow separate restarts, but
those benefits do not yet justify the extra process/control boundary for the
primary workload. This is a design choice, not a measured latency claim.

**Consequences.** The conductor is outside per-expert routing and paging;
those remain node-local. Coordination and network buffers count against the
conductor node's budget. A conductor-runtime failure also loses that node's
execution and the cluster front door; other nodes retain responsibility for
safe unwind. [Architecture](architecture.md#conductor-ownership-and-admission)
records the conceptual ledgers, failure rules and required challenge cases.
M6 adds coordinated prepare/commit across per-rank capacity reservations;
this is not a cluster-wide virtual-memory pool or approval of the deferred
mirrored-ledger shortcut. The configured-cluster schema, wire encoding,
restart fencing mechanism, numeric queue/time bounds, async model, and
question 9's capacity-guarantee policy remain their own planning tasks. No
public API, wire format or implementation is introduced by this decision.

**Reopen if.** Front-door isolation, independent upgrades, a conductor-only
host, or measured coordination contention justifies a sidecar. Preserve the
single front door and authoritative local admission if process placement changes.

## D-036: Workload-scoped switching benefit and generation-stall targets  (2026-09-22, status: accepted; specializes D-021 and D-025)

*Refined the same day at the owner's direction after review: the benefit
comparator, the switch/continuation distinction, which reference arm sets
the floor, a required over-memory configuration, and control matching.*

**Decision.** The owner accepted the following performance targets. These are
acceptance requirements for future measured implementations, not results of
the offline feasibility study.

| Area | Acceptance criterion |
| --- | --- |
| Switching floor, M4 onward | Outward and return switches each take no longer than the fastest correct full-swap reference arm at both median and p95, under the same workload and memory pressure. Include state handling and time to the first returned token. |
| Meaningful benefit, M7 | At least 25% lower median return-switch latency than jitLLM's own whole-model control with identical state handling at the same budget, on an agreed workload where partial retention is possible, while preserving the outward-switch floor. |
| Generation, M5 onward | Paging adds at most 10% to total generation time over the pinned workload, counting added time to first token on continuation requests; added inter-token gaps are at most 20 ms at p95 and 100 ms at p99, relative to the same execution configuration fully resident. |
| Correctness, every milestone | No skipped selected experts, invalid state reuse, or relaxed numerical checks to meet performance targets. Recompute when restoration is unsafe and include its cost. |

Definitions. A **switch** is a request for a model other than the one that
last executed on the node (D-019); every other request is a **continuation**,
including a new turn on a model whose extents were partially reclaimed. The
**correct full-swap reference** is the fastest reference arm, across the
matched and normal-configuration views (D-025), whose state path is valid for
the workload; a fast but invalid restore cannot set the floor. Cold-storage
and warm-page-cache arms are reported separately; where the pinned budget
lets the reference keep the incoming model's files warm, the warm arm is also
a required floor condition, because D-034's direct I/O gets no page-cache
benefit and that is the condition jitLLM can lose. The **whole-model control**
is jitLLM itself evicting complete inactive models under the same budget,
state policy, and workload; it isolates the benefit of extent-level retention
(D-008) from the benefit of state retention.

The targets apply to named supported model/workload/budget combinations, not
arbitrary models or all possible budgets. The named set for each milestone
is fixed before acceptance runs; from M7 it includes at least one
configuration whose library of prepared weights exceeds the node's physical
memory. Smaller budgets may be offered as explicitly slower modes without
claiming target compliance. Favor smooth generation over marginal switching
savings. Retained-state reuse and safe recomputation are reported separately;
a fast but invalid restore cannot set the reference floor or satisfy a gate.

**Context.** The owner accepted the proposed targets after the
[full paging-feasibility study](experiments/paging-feasibility/full-study.md).
Its retention savings and demand-paging stall estimates motivate these
priorities, but do not demonstrate achievement. The measured 25.236-second
Gemma recomputation return is baseline evidence, not a universal fixed deadline.
Qwen's conditional route traces cannot establish a passing result.

The refinements follow from the
[reference cycle](experiments/reference-aba/README.md) and the study. The
reference's restore arm, invalid only for SWA coverage (RE-007), returned in
18.304 s, about 27% below the 25.236 s recompute arm, so a benefit measured
against the reference would credit state retention alone. The
normal-configuration recompute arm returned in 23.377 s, faster than the
matched arm, and warm-cache recompute in 11.215 s, so which arm is "the"
reference must be stated. Reference decode ran at about 27–28 tokens/s on
live full-SWA state and about 46–47 on restored state, cause unisolated, a
difference the size of the gap target. The study's batch-512 prefill touched
83.63 of 128 Gemma experts per layer on average, placing the worst paging
exposure at time to first token rather than after it. Its DeepSeek/Qwen
demand-paging estimate at the single-Spark budget is an 18.221 ms/token
largest-request p95 storage service with no overlap and no kernel or driver
cost, at the gap limit, while eager loading has no modeled misses.

**Consequences.** Before acceptance runs, pin the supported models, numerical
configuration, exact request histories, output lengths, budgets, state policy,
cache conditions, and the minimum trial count per arm for any claimed
percentile. Use the existing reference A→B→A and varied-conversation
workloads where supported; M4's first dense-model implementation needs its own
named, pinned workload and fresh reference measurement. Do not require MoE
support merely to evaluate M4. Name the partial-retention benefit workload in
advance rather than selecting the best result afterward. The Gemma/Ornith pair
is M5's named demand-paging configuration; the canonical two-large-model pair
(DeepSeek V4 Flash and Qwen3.8 Flash Next) is an M7 configuration, where the
estimates say demand paging needs prefetch overlap to meet the generation
limits and eager active-model loading is the expected M5 policy. Both large
models currently show unresolved prediction drift; if neither validates, name
another pair whose prepared weights exceed physical memory rather than passing
M7 on the small pair alone.

The runtime must offer the whole-model control as a selectable policy. Run it
from M4 onward alongside the retained policies so the M7 benefit has a
baseline measured the same way.

Repeat the correct full-swap reference and the whole-model control alongside
implementation tests, interleaving arms within each repetition as the
reference cycle did, and report uncertainty with enough repetitions and token
observations for the claimed percentiles. A percentile from fewer trials than
the pinned minimum is reported without a pass/fail. An inconclusive comparison
does not pass. Choose and record sampling and uncertainty methods before
acceptance runs. Preserve both matched-configuration and normal-reference
views (D-025). The fully resident generation control may require a larger
memory budget; identify that control explicitly, hold execution settings,
token workload, context history, and state provenance (live versus restored)
constant, and do not use it as the same-budget switching comparator. Measure
generation after the first token on switches, counting paging waits during
generation; on continuations, also count added time to first token. The 10%
bound aggregates over the pinned workload's generation phases, not per
request. Measure added token gaps against corresponding gaps in the matched
resident control, not by subtracting two unrelated percentile summaries.
Report bytes moved, peak memory/spill, and prompt tokens reused versus
recomputed alongside timings.

M4 validates the switching floor, M5 adds the generation limits, and M7 must
also demonstrate the 25% return-switch benefit. Correctness remains a separate
required gate throughout. No change is made to the partial-extent eviction,
state-lifetime, or authoritative-routing contracts.

**Reopen if.** Measured supported workloads show these targets require a
product tradeoff the owner wants to change, or a new correct reference changes
the practical comparison. Amend explicitly; never silently loosen a target or
exclude a failing named configuration.

## D-035: Import models into paging artifacts aligned with managed backing  (2026-09-21, status: accepted; specializes D-009 and D-034)

**Decision.** Making a model available includes preparing and atomically
publishing its immutable paging artifact and resource index. Source GGUF or
other checkpoints are import inputs; their tensor order and packing do not
dictate runtime reads. Perform lossless layout changes once at import so
stored payloads directly populate the backend's executable VMM layout, with
no CPU payload copying, unpacking, or reshuffling during page-in. Preserve
quantization and numerical meaning unless an explicit transformation is
requested (D-009).

For the initial Spark representation, use **2 MiB aligned payload extents
and whole-extent weight reads**, matching D-033/D-034's independent backing.
Read one or more complete extents per application request; initialize and
store tail padding so even the final extent can be read in full. This is a
chosen artifact profile, not a hardware minimum I/O size. Record the profile,
extent size, and layout version; query provider compatibility before loading
and reject or explicitly re-import incompatible representations. Smaller
kernel/NVMe requests below the application interface do not violate it.

Logical allocation, physical backing/reclamation, and I/O request sizes are
distinct. Suballocate small tensors and state blocks within backing where
compatible; they need not each occupy 2 MiB. A freed suballocation can be
reused within its address/alignment/lifetime constraints, but it does not
return physical capacity while another occupant retains the same VMM handle.
Smaller objects can have independent validity and leases; the containing
backing remains protected by their union. One process owning all models does
not permit a sub-granularity unmap or release. Do not credit reusable holes
as physically released bytes. The whole-extent weight-read policy is chosen
for bulk DMA, not derived as a requirement from VMM granularity.

Immutable weight suballocations follow the imported extent's fixed layout
and content identity. Padding and unused logical slots are not a general
cross-model free pool while that extent remains in use: a whole-extent reload
or integrity check still covers them. General/mutable suballocation reuse
must also respect population/restore footprints and content-generation and
representation compatibility; pointer fit alone is insufficient.

Backing retention is a separate policy from its allocation unit. Normal
paging reuses admitted backing after old consumers complete; it does not
require a release/create cycle per read. Keep useful contents until capacity
or retention policy requires reclamation (D-007/D-033), within the node's
budget and OS headroom. A permanently mapped large slab with software-managed
slots is a valid alternative to independent small handles; a 1 GiB slab can
contain 512 of these 2 MiB stored extents without requiring 1 GiB I/O.
The artifact layout must not freeze that physical-pool implementation.

The owner's large-slab proposal needs an explicit comparison before changing
D-033's baseline: retained small handles versus larger allocations (including
1 GiB) kept mapped and reused through tensor views. Large slabs couple
physical release and constrain relocation; keeping a pool large does not by
itself require large handles. The current
[CUDA VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html)
requires zero `cuMemMap` allocation offset, so arbitrary interior slots of
one large handle must not be assumed independently remappable. Fixed mapped
slots avoid that operation but require the backend, aliases, registrations,
and any captured pointers to tolerate the chosen address/lifetime scheme.
Measure those choices on a real paging cycle, not isolated create/free costs.

**Layout contract.**

- Pack weights used together into contiguous file ranges: layer-local dense
  resources and each routed expert's private weights, subject to the backend's
  executable tensor shapes, strides, and quantization-block rules. Record
  shared/tied resources separately in the dependency closure; do not duplicate
  them into every expert. Padding is not permission to change tensor shapes.
- Pack compatible small tensors into a shared extent where their use and
  lifetimes fit; do not pad every tensor or row to 2 MiB. Shared extents have
  one physical reclamation lifetime and are protected by all consumers. Charge
  all padded backing bytes once, not just useful tensor bytes. Independently
  pageable expert groups must not accidentally share tail backing with
  unrelated experts; immutable and mutable contents never share an extent.
- The index maps model/layer/expert/tensor identity to extent IDs, file/shard
  offsets, logical lengths, stored lengths, relative destination offsets, and
  integrity information covering deterministic padding as well as payload.
  Tensor views and actual addresses are rebuilt at load; serialized pointers
  and CUDA handles remain forbidden. Admission covers the backend's actual
  read ranges, including any required kernel-readable padding. An extent is
  usable only after its full transfer and required validation complete;
  runtime validation must preserve the no-CPU-payload-copy contract.
- Coalesce adjacent missing extents only when file ranges and destination
  ranges permit direct transfer and all destination backing is admitted and
  protected. Disk adjacency alone does not permit one read into scattered
  destinations. Never overwrite resident/live backing to bridge a gap; extra
  prefetch requires its own bounded admission and accounting.
- A known checkpoint working set can be submitted as bounded asynchronous
  batches, with completion tracked per range and consumers released when
  their dependency closure is ready. Independent small operations may overlap
  bulk I/O; interleaving does not guarantee hidden latency. Prioritize required
  reads over speculative reads and background writes, preserve data and
  capacity dependencies, and measure the last required completion/consumer stall as
  well as bandwidth. A barrier waits for required work, not unrelated spill.
- Sparse immutable tables also start with whole-extent reads: row lookups
  resolve to their containing extents, deduplicate misses, and use the loaded
  rows in place. Measure useful bytes versus transferred/resident bytes.
  A smaller-read path requires explicit evidence and a revised validity and
  accounting contract; it is not an automatic exception to this default.
- Mutable prefix/conversation/KV spill remains separate from immutable model
  artifacts. It may use packed extent-sized I/O, but its dirty generations,
  partial tails, shared ownership, expiry, and recoverable-state publication
  require their own M4 policy; optional crash durability remains deferred.
  Never force tiny cache updates into whole-model rewrites.

**Context and limits.** Owner direction following the I/O spike on 2026-09-21.
The [measured file path](experiments/io-path/README.md) supports bulk direct
DMA; it does not measure this model layout. One logical paging artifact may
use indexed file shards. Reuse a known blob container and own the manifest
and resource index; open question 5 still chooses the experimental encoding.
[GGUF's specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
supports tensor offsets and alignment, but changing alignment alone does not
reorganize expert slices inside a tensor. Container reuse does not imply that
an ordinary model loader can execute the repacked representation unchanged.

Layout reduces fragmented application reads within a dependency group; it
cannot make arbitrary routed experts or sparse row requests globally
sequential, nor guarantee physical SSD placement. M0's schema task must show
dense, expert, and sparse-table layouts with padding/read amplification and
the mapping to executable tensor views. M2 proves dense GGML execution and
evict/restore on that layout; M5 proves the MoE mapping. D-018's compatibility
gate remains. No model-level performance or padding budget is claimed yet.

**Reopen if.** Real model layouts incur unacceptable padding/read amplification,
the backend requires a conflicting layout, another provider has incompatible
granularity, or measured workloads justify a different extent/read policy.

## D-034: Direct regular-file I/O into GPU-accessible host VMM on Spark  (2026-09-21, status: accepted; amends D-004's required staging copy)

**Decision.** Keep prepared artifacts and spill state in regular files. The
preferred Spark payload path is native `O_DIRECT` I/O into host-backed CUDA
VMM allocations that the GPU consumes in place. Use a bounded asynchronous
storage queue; the measured `io_uring` path is the initial implementation
direction. CPU control work submits and completes requests, but weight/state
payloads must not pass through CPU copies. Do not silently substitute a
buffered read or an unverified bounce/copy path when this capability is absent.
Retain a device-VMM/staged-transfer provider option where its DMA path is
validated. Small file metadata is not subject to the payload policy.

This amends the **mandatory staging-copy** part of D-004, not its single
physical budget, explicit network transfers, or lack of native GDS/GPUDirect
RDMA on Spark. Host VMM is managed backing in that same budget, not CPU
offload. Keep D-006/D-033's explicit virtual reservation, physical handles,
mapping/access, and independently reclaimable extents. Query host-VMM support
and allocation granularity; device and host allocation properties are not
interchangeable pool entries. Query file direct-I/O alignment (512 B on the
measured ext4 file), and prepare aligned ranges/tails at import; 2 MiB backing
granularity is not a universal artifact-padding requirement.

**Evidence.** The [I/O experiment](experiments/io-path/README.md), on `spark`
/ GB10 / driver 580.178.04 / Samsung PCIe 5.0 ×4 NVMe, measured **14.903–14.968
GB/s** for direct regular files into host VMM at four 2 MiB requests in flight.
GPU scans of that host backing reached **242.016–242.598 GB/s**, matching
device VMM's **241.869–243.331 GB/s** in the same scan. Direct paths retained
no file page cache; the report summarizes GPU content/negative controls, DMA
bounce tracing, pressure/compute tests, and sustained-read results. Native
GDS remains unavailable on Spark; cuFile compatibility adds no required
capability for this path. Raw block/passthrough/SPDK were not benchmarked:
the sole SSD holds mounted root, and there is no dedicated unmounted device.

**Consequences.** Start with two 2 MiB destination slots for latency-sensitive
loads, up to four for bulk reads: **4–8 MiB of already charged destination
backing**, not an extra staging pool. The baseline two-slot measurement
gave 14.625–14.845 GB/s at 279 µs median completion, versus about 558 µs
at four; eight slots added latency without useful bandwidth here. These are
initial tuning points, not hardcoded queue or extent limits. A staged fallback
needs a separately accounted bounded pool (initial experiment: 4–8 MiB),
allocated only when required, and completion-safe reuse. Scheduler/catalog
locks are never held across I/O waits.

M2 must prove GGML tensor/kernel operation on the chosen host VMM addresses,
plus registration, unmap/remap, reclaim, failure, and cancellation lifetimes.
The scan and transfer spike does not replace that proof. Keep the storage
completion boundary narrow; this does not decide open question 3's overall
executor/coroutine model. An NVMe controller tuning choice is deployment
configuration, not a runtime side effect; the separate interrupt-coalescing
comparison in the report restores the observed original setting.

**Reopen if.** Actual GGML/model kernels regress on host VMM; a target lacks
the required mapping or DMA capability; driver/kernel changes introduce
bouncing; model traces need different request sizes/depths; or a dedicated
raw-device comparison demonstrates a material end-to-end gain worth owning
allocation, metadata, recovery, and tooling below the filesystem. No raw
performance advantage or production tail bound is assumed from this spike.

## D-033: Initial 2 MiB independent VMM extents; reuse backing on demand without a standing free pool  (2026-09-21, status: accepted; implements D-006)

**Decision.** Start the CUDA provider with one physical allocation handle per
independently reclaimable extent, using the queried minimum granularity:
2 MiB for the measured Spark device-local allocation properties. Keep that
as a provider capability, not a universal core or artifact-format constant.
Larger I/O and scheduling batches may span adjacent extents without making
their physical lifetimes indivisible. Do not assume a subrange of one large
physical allocation is independently returnable to the OS.

Start without a standing cache of unused physical handles (idle free-pool
target zero). When reclaiming eligible contents for an already admitted
load, allow a compatible handle to pass directly to that load after all old
consumers complete, with explicit unmap/remap/access and content-readiness
tracking as needed. This avoids unnecessary release/create work; it is not a
measured end-to-end speedup. Otherwise release unused unmapped handles.
Useful resident contents remain cached according to policy: releasing a
residency lease does not evict them or turn their handles into a free pool.
Any handle held during handoff still counts against physical occupancy.
Never pre-evict useful contents to stock a free pool.

**Evidence.** The [retained microbench](experiments/vmm-microbench/README.md)
ran three times on `spark` / GB10, driver 580.178.04, with the D-032
cross-built toolchain. Minimum and recommended granularity were both 2 MiB.
At 2 MiB, per-run idle medians were 49–53 µs create, 0.50–0.54 µs map,
36–38 µs access, 47–63 µs unmap, and 27 µs release (rounded). Mapping alone
is not the cost of making bytes usable. Larger extents reduce some costs
per byte but make reclamation coarser. All 3,600 measured calls with an
independent 10 ms background kernel returned before its completion event;
this is evidence for this test, not an asynchronous/nonblocking API guarantee.
An unmapped 1 GiB pool retained its footprint and verified contents until
its physical handles were released. OS/CUDA memory snapshots then recovered
approximately that capacity.

**Consequences.** The 2 MiB choice prioritizes fine reclamation at the hardware
minimum as the first baseline, not a proven optimal transfer size. The pool
policy avoids holding empty memory in a workload whose useful cache already
exceeds capacity, while preserving direct reuse when there is an actual
consumer. Mapping calls run outside global catalog/scheduling locks; mapping
completion and data-transfer completion are separate readiness conditions.
SSD throughput, page-in latency, model throughput, memory-pressure tails,
and graph/registration survival are not established by this experiment.

**Reopen if.** The I/O spike or model traces show that larger physical extents,
batched driver operations, or a bounded unused-handle cache improve measured
end-to-end latency enough to justify their occupancy/reclamation cost. Reprobe
when device, driver, allocation properties, or sharing requirements change.

## D-032: Validated LLVM 22.1.8 / CUDA 13.4.2 toolchain with C++23 throughout  (2026-09-21, status: accepted; implements D-011/D-012 pins)

*2026-09-22 follow-up:* the [smoke manifest](experiments/toolchain-smoke/artifacts.json)
now includes matching `libclang-rt-22-dev` packages for amd64 and arm64.
Clang ASan/UBSan passed the CPU-only async-model suite natively on the
workstation and cross-built on Spark. This completes the extracted sanitizer
dependencies without changing the compiler pin; clean M1 provisioning is
still owed.

**Decision.** Start M1 with Clang/LLD 22.1.8 from apt.llvm.org
Noble packages `1:22.1.8~++20260714014902+ca7933e47d3a-1~exp1~20260714135019.80`
(exact hashes for both architectures in the smoke manifest), GCC 13
libstdc++ development/support files
(`13.3.0-6ubuntu2~24.04.1`), libstdc++6/libgcc-s1 runtime
`14.2.0-4ubuntu2~24.04.1`, and glibc `2.39-0ubuntu8.9`. Select CUDA
Toolkit 13.4.2's NVCC, CRT, libNVVM, libnvptxcompiler, and cudart/runtime
development components `13.4.92-1`, with CCCL package `13.3.4.3.1-1`,
for both x86-64 and ARM. Ordinary `.cc` files use Clang C++23; narrow
`.cu` files use NVCC `--std=c++23` with Clang as host compiler.

Cross-build with `aarch64-linux-gnu -march=armv8-a`, an explicit Clang
host wrapper, LLD, and the hashed 2026-09-21 Spark sysroot snapshot (DGX
OS 7.5.0 base / 7.6.0 OTA). Add the 13.4.2 SBSA target headers/runtime
from the pinned packages. NVCC selects `--target-directory sbsa-linux`
and `-arch=sm_121`. The workstation CPU target is explicitly
`x86_64-linux-gnu -march=x86-64`. Keep the native Spark diagnostic
profile (Clang/NVCC with GNU binutils `2.42-4ubuntu2.10`) alongside the
cross path; translate experiment profiles to CMake presets in M1.

**Evidence.** The [retained smoke experiment](experiments/toolchain-smoke/README.md)
records package/snapshot hashes, commands, and checks. A native C++23 CPU
executable passed on the workstation. AArch64 CPU and NVCC/Clang CUDA
objects were built there, linked, deployed over SSH, and passed on `spark`
(GB10 12.1, driver 580.178.04). The native Spark fallback passed too.
Both GPU paths exercised a C++23 `if consteval` host/device function and
checked all 257 results with PTX JIT disabled. The installed CUDA 13.0
comparison passed only with C++20 CUDA; 13.4.2 removes that limit (RE-001).

**Compiler choice.** The [comparison](experiments/toolchain-smoke/compiler-choice.md)
retains Clang for the verified cross-build workflow and LLVM tooling fit.
GCC is a modern, supported alternative, not ruled out by C++23 or CUDA;
no performance comparison was made. LLVM 23 is newer but outside CUDA
13.4's supported host-compiler range. 22.1.8 is the newest compatible
release checked on 2026-09-21. The earlier LLVM 18 smoke is retained as
comparison evidence. Clang continues to use the separately pinned libstdc++.

**Consequences.** Open question 6 is answered for the smoke scope. CUDA
13.x minor-version compatibility allowed this native GB10 test on R580;
new driver features and PTX/JIT paths need separate validation and may
require a driver upgrade. No driver/default-toolkit change was made in
this experiment. These are initial integration pins, not complete C++23
library or backend support. M1 still owes declarative provisioning,
CMake/mise/CI, clean-host/container verification, and the dependency audit.
The copied sysroot is a local input, not a redistributable SDK.

**Reopen if.** A backend needs a newer compiler/library/driver feature,
the target OS changes, or M1 clean setup cannot reproduce the smoke.
Validate native, cross, and Spark fallback paths before changing pins.

## D-031: Shared prompt prefixes and conversation continuations have independent reuse and retention  (2026-09-21, status: accepted; clarifies D-024 and D-030; supersedes D-022's prefix-as-conversation identity)

**Decision.** Prefix matching identifies reusable computation, never a unique
conversation or its lifetime. A system-prompt prefix can be cached and reused
across independent conversations with compatible execution identities.
Shared prompt-prefix snapshots are immutable; mutable continuation state is
isolated per branch/request. A conversation's longer history may itself have
reusable immutable snapshots, but a hit on the shared system prefix does not
identify or authorize reuse of any conversation's suffix.

Shared prompt prefixes and conversation continuations have separate reuse
statistics and retention/expiry decisions within the same bounded node
memory, spill, and metadata budgets. Shared-prefix value reflects reuse
across conversations; continuation value reflects reuse of that particular
history. A hit on the shared prefix does not refresh unrelated continuation
entries. Expiring or releasing a conversation does not itself invalidate the
shared prefix or another branch. Neither class is pinned indefinitely, and
shared physical extents are charged once and protected until all live
consumers retire (D-006, D-007).

**Context.** Owner's clarification during review on 2026-09-21: system-prompt
prefixes must be cached independently from a given conversation because their
reuse assumptions differ. This makes D-024's distinction explicit and
corrects D-030's restatement of D-022's superseded identity claim.

**Consequences.** Cache keys cover the exact rendered token prefix from the
context origin, model/artifact and execution identity, and non-text inputs
when supported; matching system-prompt text alone is insufficient. Restore
only at architecture-supported boundaries. If a longer continuation is
unavailable, reuse a compatible shorter prefix when present and recompute
the remaining supplied history. Before M4, define the two retention policies
and measured defaults. M4 tests cross-conversation prefix reuse, independent
expiry/release, isolated branches, spill/restore, and shared-byte accounting;
details live in [architecture.md](architecture.md#conversation-state-retention).

**Reopen if.** A supported state representation cannot preserve a reusable
prompt-prefix boundary, or measured workloads require another retention
class; preserve the separation of cache identity and conversation identity.

## D-030: Claude Code is a named client; the Anthropic Messages format is in the baseline surface  (2026-09-21, status: accepted; amends D-022; prefix policy clarified by D-031)

**Decision.** Claude Code joins Cursor, OpenCode, and Codex as a named
standard client. Because it speaks the Anthropic Messages API, that format
ships in the M3 baseline endpoint alongside OpenAI-compatible chat completions
and whatever else the named clients need, rather than as a later optional
extension. The `model` field remains the switch signal, and sessions and
hints remain optional. Prefix matching identifies reusable computation,
not conversation identity or lifetime (D-024, D-031).

**Context.** Owner's answer on 2026-09-21 during the M0 feature triage. The
primary workload (D-019) is an agent plus subagents on different models, a
pattern Claude Code fits.

**Consequences.** The M0/M1 endpoint verification task covers Claude Code's
exact endpoint, streaming, and tool-call needs. M3 carries a second
request/response translation, including tool-call streaming. M4's canonical
A→B→A may run through any named client.

**Reopen if.** Claude Code moves to a protocol the baseline does not cover,
or maintaining two formats measurably delays M3, in which case the Messages
format drops back to an M7 extension.

## D-029: Contributions under DCO; REUSE-style SPDX headers and a NOTICE file from M1; SBOM with packaging  (2026-09-21, status: accepted)

**Decision.** External pull requests are accepted and must carry a Developer
Certificate of Origin sign-off (`Signed-off-by`); there is no CLA. Every
REUSE-covered file has copyright notices and an `SPDX-License-Identifier`
(Apache-2.0 for jitLLM-authored code, the actual license for incorporated
code), with the corresponding license texts under `LICENSES/`. Commentable
source and documentation files embed this metadata in headers, using
`SPDX-FileCopyrightText` for copyright notices. Uncommentable files may use
`.license` sidecars or `REUSE.toml`. From M1, CI runs REUSE lint for metadata
coverage and a separate check for required embedded headers. A root `NOTICE`
file is seeded in M1 and grown by the dependency audit (D-017). A software
bill of materials is generated when `.deb` packaging lands and is tied to
the build's license profile.

**Context.** Owner's answers on 2026-09-21 during the M0 feature triage,
closing the plan.md item on NOTICE, SPDX headers, and contribution policy.
DCO is the lightest credible sign-off for an Apache-2.0 project and keeps
the barrier low for an externally consumed single-developer project (D-016).

**Consequences.** Merging an external PR still goes through the human commit
gate; agents never merge. A commentable source or documentation file without
its required header fails the separate header check even if REUSE lint finds
metadata elsewhere. REUSE lint alone accepts sidecars and `REUSE.toml`; it
does not enforce embedded headers ([REUSE specification](https://reuse.software/spec-3.3/#licensing-information),
checked 2026-09-21). These checks support, but do not replace, D-017's audit
of the selected dependency closure. The SBOM follows `.deb` packaging rather
than standing as its own deliverable; the provisional ladder places the
first CI `.deb` build in M1.

**Reopen if.** Relicensing flexibility becomes necessary (which would mean a
CLA), or a contributor base needs a maintainer structure that D-016 does not
describe.

## D-028: GGML is the first compute substrate; optional backends are build-time modules, not a runtime plugin ABI  (2026-09-21, status: accepted; amends D-010; EXL3 timing and artifact scope amended by D-052; dispatch ownership amended by D-053)

**Decision.** The first vertical slice (M3) executes on GGML/GGUF with jitLLM
supplying the buffers behind tensors, so weights and state live in
jitLLM-owned VMM backing and GGML computes over them. EXL3 kernels are ported
later for the flagship recipes as further build-time backends behind the same
operation contract. Optional implementation modules (D-017) are build-time
modules selected by build profile; there is no versioned runtime C plugin ABI
for separately built backends, and none is planned.

**Context.** Owner's answers on 2026-09-21 during the M0 feature triage;
settles the direction of open question 4 (the checkpoint, quantization, and
numerical reference remain that question's task). GGML is MIT, torch-free,
has a C API and broad quantization and tokenizer coverage, and matches the
llama.cpp reference engine and the owner-provided GGUF candidates for the
reference spike (plan.md); the MiaAI-Lab reference recipes are EXL3 and
torch-bound. The C ABI row (ideation §14) was
rejected outright: a removable boundary is a build-profile property, and
freezing a plugin ABI before a real backend exposes its requirements was the
risk the row itself named.

**Consequences.** The M2 early backend integration proof runs GGML on
jitLLM-owned memory with explicit workspace and completion tracking; if
GGML's allocator or scheduler assumptions cannot be met that way, this entry
is the first thing to reopen. The operation contract is finalized from that
proof. The immutable artifact data follows GGML's tensor formats, re-packed
into aligned extents (open question 5). The copyleft-disabled profile is a
build profile that omits optional modules; license and notice reporting is
per build, not per loaded plugin. A later Metal, ROCm/HIP, or Vulkan port is
mostly a memory-provider job (D-026), since GGML already has those backends.

**Reopen if.** The M2 proof shows GGML cannot run on externally owned backing
without a fork; the flagship recipes need kernels GGML cannot host; or an
out-of-tree, differently licensed backend must load without rebuilding the
core.

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

## D-025: Measure the switching baseline once the reference runs  (2026-09-20, status: accepted; amends D-021; acceptance targets specialized by D-036)

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

## D-024: Conversation reuse is bounded; prefix identity does not imply session lifetime  (2026-09-20, status: accepted; amends D-019 and D-022; independent prefix retention clarified by D-031)

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

*Discovery refinement (2026-09-22, D-038): interface/bootstrap discovery and
path refresh for enrolled nodes are initial scope; automatic membership
changes and election remain deferred.*

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

## D-022: Standard web-API compatibility is the baseline; sessions and hints are optional extensions  (2026-09-20, status: accepted; prefix-as-conversation identity superseded by D-024 and D-031; named clients amended by D-030)

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

## D-021: The switching bar is "never worse than a full swap"; seamless is the goal  (2026-09-20, status: accepted; baseline measurement amended by D-025; acceptance targets specialized by D-036)

*Comparator datapoint (2026-09-21): Athena's Engine, a closed-source engine
for GB10, reports a 46 s measured full swap, including a session checkpoint, between DeepSeek V4 Flash and
Qwen3.8 Flash Next on one node (creator-reported). D-025's own measured
baseline still governs; this number only makes the floor concrete.*

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

*Process-location follow-up (2026-09-22, D-037): the conductor is a role
inside its node's runtime; node-local admission remains authoritative.*

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

**Evidence.** The [MiaAI-Lab/ExLlamaV3 inventory](licensing.md) (2026-09-22)
records pinned file-level declarations, unresolved modification provenance,
verified upstream MIT headers, and AGPL network-source obligations. It does
not approve incorporation or change this policy.

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

## D-012: Declarative, pinned toolchain provisioning via mise plus project-owned SDK manifests  (2026-09-20, status: accepted; provisioning split refined by D-049)

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

## D-010: C++23 host runtime, Clang-first, native hot path  (2026-09-20, status: accepted; optional-backend C ABI clause superseded by D-028)

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

## D-007: Capacity reservations are separate from residency leases; commitment is lazy  (2026-09-20, status: accepted; initial guarantee policy defined in D-050)

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

## D-004: Target platform is NVIDIA DGX Spark, one or two nodes, treated as unified-memory domains over a network  (2026-09-20, status: accepted; staging-copy requirement amended by D-034)

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
