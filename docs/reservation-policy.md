<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Capacity reservations and bounded progress

D-050 answers M0 question 9. This is the initial internal admission policy
for M2, with concurrent execution validated in M4 and routed expert phases
in M5. It defines guarantees and required tests; it does not claim an
implemented scheduler or measured model envelopes. D-048's
[completion protocol](async-model.md) remains the lifetime authority.

## What a grant promises

A **guaranteed capacity reservation** admits one bounded request with a
validated execution plan. Under a healthy provider and fair scheduling, the
node can supply every phase's declared dependencies and bounded state growth
without another admitted request surrendering its live state. Success still
depends on valid input/artifacts and successful operations. Cancellation,
an operation error or an exceeded contract has an accounted safe unwind;
unknown completion instead retains ownership and faults affected admission.
This is a resource-progress guarantee, not a latency or hardware-availability
promise. A deadline cannot prove retirement.

An **opportunistic allowance** is revocable permission for bounded warming,
prefetch or reusable cache, with no promise to execute an inference request.
Queued requests have no execution grant and own only bounded intake/queue
storage, not model leases or partial activations. Inference becomes admitted
only after the node atomically commits its guaranteed envelope. A queued or
opportunistic item cannot be reported as admitted inference under D-045.

Grants commit lazily. They do not map backing, load weights, clear cache or
evict anything. A residency lease protects concrete dependencies, separately
from the capacity reservation; releasing it makes backing eligible for
reclaim without discarding its useful contents (D-007).

## Describe a finite request before admitting it

The plan binds artifact and backend/layout identity, shape/batch limits,
context and output limits, supported restore boundaries, and a finite set
of phase types. Record peak physical requirements for those bounds, not just
current tensor sizes or one observed successful run. A missing client output
limit resolves to a finite supported limit before admission; no unbounded
request is implicit, and no input is silently truncated to make it fit.
Numeric defaults and backend envelopes require M2/M3 evidence.

For the first dense path, a phase is a bounded prefill chunk or decode step
whose entire dependency closure fits until its completed boundary. A
whole-turn phase is legal only with a bounded full-turn envelope. A smaller
boundary must be supported by the actual backend; a layer name or an I/O
wait alone is not a yield boundary. M5 may use validated routing/expert
subphases. Its bound covers every allowed selected closure for the admitted
batch, including shared experts, containing extents and routing activations;
average expert locality and predicted routes cannot establish a guarantee.
No selected expert is substituted or dropped.

Phases are accounting, lease and cancellation boundaries, **not scheduling
quanta**. The scheduling quantum is one client-facing request/response: a
prompt and the complete response it produces (one API call, including a
streamed one; each tool-call round trip is a separate request). Once a
request starts, the node does not switch to another request or model until
it retires or is explicitly terminated. Other requests run beside it only as
a concurrent cohort whose full envelopes fit (below).

The envelope includes:

| Component | Required bound |
| --- | --- |
| Retained request state | Inputs needed for execution/reconstruction, KV/recurrent state, isolated mutable suffixes, copy-on-write growth and all live continuations at a completed boundary, through the request's maximum admitted context/output |
| Phase dependencies | Unique physical weight extents, selected experts or sparse rows and their containing extents, padding/alias footprints, activations and backend workspace through the next completed boundary |
| Transfer and preservation | Read destinations, temporary old/new copies, spill/restore workspace, any staging, registration lifetimes and concurrent DMA; a destination already counted as a weight extent is not a second allocation |
| Runtime and communication | Graph/captured-pointer holds, backend/runtime allocations, code/workspace caches that cannot be reclaimed, stable communication buffers and per-domain provider requirements |
| Control and output | Bounded task trees, waiters, operation/result records, submission/cleanup credits, tokenizer/plan work, queued input, snapshots and output buffers; cleanup and result harvesting cannot require new unbudgeted space |
| Physical overhead | Actual mapping/release granularity, alignment, allocator/slab fragmentation, non-reusable holes and provider overhead; logical tensor bytes alone are insufficient |

OS, other-process and page-cache headroom is excluded from the node's
execution budget. Spark CPU allocations and GPU backing use that same
physical budget; staging is not another tier. On other providers, apply
the checks to every required memory domain and allocation class. Unknown
allocations are non-evictable and charged conservatively; an unbounded or
unaccounted backend cannot receive a guaranteed plan. Do not assume that
the GPU allocator's reported bytes cover all node memory.

## Admission rule and separate ledgers

Use these conceptual bounds for one memory domain. They describe accounting,
not a public interface or final C++ type:

- `B`: the configured execution budget after external/OS headroom.
- `F`: bounded fixed runtime/control/cleanup storage and other non-revocable
  occupancy outside request envelopes. Its cleanup resources are available
  before work starts, including at full occupancy.
- `R(G)`: an upper bound on all state retained across completed boundaries
  by the set `G` of guaranteed requests, including each request's maximum
  future growth. Initially, admitted state retains its full in-memory
  allowance even when a valid spill copy exists.
- `E_i`: request `i`'s largest additional physical working set to get from
  a completed boundary to the next one or unwind safely. This is a peak
  across all phases at admitted shapes, above the retained-state allowance;
  old/new copies that overlap during transitions are included.
- `J`: the maximum non-revocable footprint of outstanding opportunistic or
  maintenance operations, including any remaining allocations needed to
  complete or unwind them. It lasts until actual retirement, not until a
  cancellation acknowledgement.

Initially one request holds the node's execution slot, from its first phase
until the request retires or is explicitly terminated. For a proposed
admitted set `G`, require, with checked arithmetic:

```text
F + R(G) + J + max(E_i for i in G, default 0) <= B
```

The same admission transaction checks task/queue/result/cleanup credits,
allocation-class feasibility, and supported plan bounds. A sum of free
bytes is insufficient if no compatible physical extents can satisfy the
plan. If a provider cannot supply a defensible fragmentation bound, the
plan is unsupported until it has one. Shared backing is counted once in
physical occupancy. `R(G)` may deduplicate retained storage only with a
stable shared immutable identity and a bound for future divergence; otherwise
use conservative separate allowances. Grant sharing credit only when the
remaining members still pass these inequalities after any member's
completion, cancellation or release, in any order. Retirement can never be
refused, so no release may depend on a later check succeeding.
The conservative `max(E_i)` rule does not borrow another request's retained
state or assume it will finish first.

For a supported concurrent cohort `C`, additionally require:

```text
F + R(G) + J + max(max(E_i for i in G, default 0), sum(E_i for i in C)) <= B
```

Each member keeps its complete envelope during waits. Shared phase weights
can make the sum conservative; physical occupancy still counts them once.
Do not discount the sum based on expected non-overlap or cache hits in this
initial policy. After the cohort drains, the serial bound still allows any
other admitted request to run. A request may join a cohort only when this
check passes; it never displaces a running request. M2 starts with one
active request; M4 must validate
concurrent progress for supported all-resident configurations whose complete
envelopes fit, as D-020 requires. Placement on independent nodes applies the
rule independently; aggregate cluster bytes never satisfy a local deficit.
M6 still owes coordinated per-rank admission and collective-order proof.
Recheck the serial and any active-cohort inequalities on every grant,
envelope replacement, cohort change and increase to `F` or `J`. Categories
are disjoint charges for protected requirements; an operation owned by a
phase is within its `E_i`, not also in `J`. Demand-driven reclamation belongs
to that phase's peak, including any simultaneous victim/destination backing.

**Commitment and occupancy are different ledgers.** The inequalities describe
maximum protected requirements; the occupancy ledger records actual backing,
including useful cache, pending write-back, quarantined extents and retained
pool handles. Do not add all cached occupancy to the commitment sum or count
unmaterialized allowances as physical allocations. Revocable cache may occupy
unused allowance. On a real miss, select eligible extents and recover or hand
off compatible backing, then materialize only the missing dependencies.
Every materialization checks actual occupancy against `B`. Capacity credited
by an intended eviction is unavailable until all consumers/registrations
retire and preservation/invalidation permits reuse. A reusable slot is not
necessarily memory that the provider can release to the OS.

## Holding the slot and reaching a handoff

Acquire the execution slot before the request's first phase-specific
allocation, lease, page-in or launch. Keep it and `E_i` through every phase
of that request: dependency discovery, loading, compute, output transfer,
the completed boundaries between phases and completion-aware unwind. The
scheduler may service completions and unrelated host/control work while the
request waits. It cannot start another request's phase unless the
concurrent-cohort check passes; a completed phase boundary is not an
opportunity to run a different request or model. In particular, suspension
with live routing activations does not release the slot. Cancelling A
cannot start B against the same allowance while A still has accepted I/O or
GPU consumers.

At each completed phase boundary, the scheduler proves every consumer and
relevant registration of the finished phase's resources has retired and
classifies surviving mutable contents as retained request state within
`R(G)`. Only then may that phase's leases end; the backing becomes eligible
for this request's later phases, a concurrent cohort member or reclaim,
while the slot stays with the request. The request handoff applies the same
proof to its last phase before releasing the slot. Compatible persistent
registrations may remain only with their backing charged as non-revocable
retained/fixed storage. Anything still held outside those bounds prevents
the handoff. Idle weight cache remains resident and eligible after leases
end. A request for another model alone proves none of these conditions and
does not interrupt the running request.

An admitted request's state cannot expire while it waits for the slot or is
paused at a completed boundary, for example during an output stall. Its
`R(G)` allowance survives until request retirement; a client
disconnect is insufficient. When a completed request hands state to the
idle prefix/continuation cache, atomically transfer accounting and retention
claims before dropping the grant. D-031's independent cache expiry resumes
then. If a new request promotes cached state to live state, acquire its
guarantee and protection atomically with identity/coverage validation, or
leave it queued without pinning that cache entry.

## Growth, reclamation and failure

All later state growth within the admitted bounds is already covered by
`R(G)`; a token cannot wait for an unpromised increase while holding the
phase's resources. Check actual dependencies, growth and allocator rounding
before allocation/submission. A larger request/shape/plan needs an atomic
envelope replacement at a completed boundary. Preserve the old grant until
the replacement passes; if it does not, reject the extension or end the
request explicitly at the boundary. Do not silently exceed the bound or
hold partial work indefinitely waiting for an upgrade. An unexpected
mid-phase excess is a plan/accounting failure: stop new submissions, drain
accepted operations, preserve or explicitly invalidate mutable results,
and report failure. A larger grant is not the recovery assumption.

Initially, spill is an optional way to preserve useful idle cache, not
collateral for an admitted request's capacity guarantee. Suspended admitted
state retains its full `R(G)` allowance; freeing its physical copy does not
admit extra guarantees against that allowance. A future policy that credits
spilled live state must separately prove bounded disk space, restore and
write-back peaks, coverage/correctness and a safe restore schedule. M4's
completed-request spill/restore tests do not establish that stronger policy.

At full occupancy, clean weights or idle state whose retention is optional
can be discarded with correct metadata invalidation. If preserving a victim
requires a write, secure spill capacity plus transfer/cleanup resources
before choosing that path and keep its backing charged through completion.
When optional spill is full or fails, invalidate eligible cache or select
another victim; no admitted state is discarded to make an unproved reclaim
work. Correct fallback still requires available history and a validated
recompute plan. No valid recovery path means explicit failure, not a false
cache hit. Reclaim must not require allocating scratch from the very space
it is trying to free: use the pre-budgeted cleanup resources or a proven
in-place path. Preservation promises that cannot be revoked belong in the
protected bounds, not in revocable cache.

New speculative/maintenance work may start only if its full `J` still leaves
all guaranteed envelopes feasible. Its completed output may remain as
revocable cache. Pre-existing non-revocable work may defer a new grant until
it drains. Demand and retirement take precedence over opportunistic work;
prefetch cannot repeatedly replace a victim needed by an admitted phase.
Pin hints cannot convert unused guaranteed allowance into non-revocable
cache or bypass admission. Incoming requests may remain in a bounded queue
while useful idle cache remains resident.
Required write-back/cleanup inherits the priority of the demand waiting on
it; a read queue cannot starve the write that makes its destination available.
Cache scoring, hysteresis and minimum-residency preferences cannot veto
reclaim needed by an admitted phase indefinitely.

Distinguish a temporary conflict from an impossible plan. If the minimum
validated plan cannot satisfy the rule even alone with required runtime
overhead, reject it without enqueueing it forever. A smaller prefill chunk,
batch or other numerically valid execution plan can be selected only if
already supported and validated; moving work to another node also requires
that node's grant. Do not invent an expert split or use a different model to
make admission succeed. Temporary conflicts get bounded queueing or a
capacity refusal. D-045's existing external status/error mapping remains
unchanged; this decision does not add wire codes or retry promises.

Choose the next request only at request boundaries: when the slot frees,
service eligible admitted requests round-robin by request, with finite
priority preference/aging so background work cannot starve. A running
request is never preempted to serve another; it releases the slot when it
retires or is explicitly terminated by cancellation, contract excess,
failure or a stall/queue limit. Bound bypasses of older feasible queued
requests; when necessary stop admitting newer work until existing requests
drain. Queue waits and client/output stalls have finite configured limits
and terminate with an explicit outcome; a stalled client cannot indefinitely
retain an execution slot. Output stalls stop new production at a completed
boundary and leave completion harvesting independent; a stall that reaches
its limit terminates the request and starts the handoff. Control/cleanup
work has bounded service
independent of inference queues. The progress argument assumes finite phase
work and provider completion; it promises no wall-clock upper bound on a
GPU kernel or I/O operation.

Unexpected external memory pressure, provider allocation failure or lost
completion stops affected admission and triggers explicit failure/drain or
quarantine under D-048. Never shrink the accounting budget below outstanding
claims and declare the difference reclaimed. A requested budget reduction
waits for retirement or is refused; a real loss of usable capacity is a fault,
not evidence that a grant was opportunistic. Numeric headroom, queue limits,
phase sizes and timeouts require implementation measurements before use.

## Worked cases and implementation gates

These are synthetic accounting examples, not Spark measurements or defaults.
With `B=100`, `F=10`, `J=0`, requests A and B each with retained bound `20`
and phase peak `50` fit serially: `10 + 40 + 50 = 100`. If A pauses after
producing activations, B cannot start: both full envelopes would require
`10 + 40 + 100 = 150`. B also does not start at A's later phase boundaries;
it waits until A's request retires or is terminated. Reserving only their
current state of `4` each would miss the later growth to `20`.

A third request with retained bound `1` and peak `1` is feasible alone but
temporarily deferred (`101` with A and B). A request with retained bound
`41` and minimum phase peak `50` is impossible at this budget even alone
(`101`). Useful revocable cache can fill all presently unused physical
capacity in each example without changing the grant calculation; an actual
miss still requires safe, compatible reclaim before materialization.

The resource-progress argument is deliberately conservative: all admitted
boundary state and its growth can coexist, and at least one complete phase
fits above it. No inactive request may occupy another request's phase
allowance. A running phase needs no new capacity commitment to finish or
unwind. Each admitted request has finitely many bounded phases, so it
retires or is explicitly terminated; fair selection at request boundaries
then serves every admitted request. Provider
uncertainty stops this argument and retains resources instead of pretending
to make progress. M2 must test these premises, not merely absence of OOM.

| Adversarial case | Required result / earliest execution gate |
| --- | --- |
| Two phases each retain activations and await missing weights | M2: serialize before the second phase starts, or admit their full concurrent envelopes; no circular capacity wait |
| Agent and subagent requests on two models whose envelopes do not fit together | M2 fake / M4 real: the running request keeps the slot through all its phases; the other starts after its retirement or explicit termination; no per-step alternation or mid-request model switch |
| State grows from a small prefix to the admitted context/output limit | M2: grow without a grant upgrade; include branch/copy-on-write and old/new transition peaks |
| One selected closure or rounded allocation exceeds its bound | M2 synthetic / M5 routes: detect before submission, drain accepted work and fail; no expert substitution or indefinite upgrade wait |
| Feasible queued request versus permanently impossible minimum phase | M2: bounded deferral for the former; immediate impossible result or validated alternative for the latter |
| Grant with a full useful cache; lease release without pressure | M2: no eager eviction; acquire only missing extents on use; lease release preserves resident contents |
| Full memory, spill full/failed, write-back needs scratch | M2 injected / M4 storage: no reclaim cycle or false recoverability; use budgeted cleanup, invalidate eligible cache, or fail explicitly |
| Shared/tied extents, a fork, a leased tensor in an otherwise idle extent | M2: charge unique physical backing, account divergent growth, protect entire conflicting restore footprint and never reclaim leased neighbors |
| Many suballocation holes, pinned slab/registration, padded tails | M2: logical free bytes do not authorize impossible physical allocation; include non-reclaimable backing and restore footprints |
| Cancelled phase, late DMA, registration still live, then replacement phase | M2: retain execution allowance/occupancy until retirement; stale IDs cannot release it; test with D-048 event permutations |
| Repeated speculation, shared page-in waiters, cancellation of one waiter | M2: full speculative peak stays in `J` or the owning phase; demand makes progress; cancelling one waiter does not reclaim another's dependency |
| All task/result/submission/output queues full during cancellation | M2: bounded metadata and independent result/cleanup capacity drain accepted operations without a new allocation |
| Slow/disconnected client or endless high-priority arrivals | M2 fake / M3 front door: output/queue limits and fairness produce progress or explicit termination; intake cannot starve admitted cleanup |
| Continuation/prefix expires while an admitted request is suspended | M2 lifetime / M4 retention: preserve admitted state and independent shared-prefix claims; idle expiry cannot release its allowance |
| Envelope upgrade, grant retirement and cached-state promotion race | M2: single-owner atomic transitions; no gap in protection, double grant or lost backing charge |
| Smaller configured budget, unexpected external pressure, unknown provider completion | M2 injection / provider proof: refuse/defer policy reduction or fault real capacity loss; quarantine remains charged; timeout is never reclaim |
| All-resident concurrent cohort and subsequent serial handoff | M4: full cohort fits and both progress; bounds still admit the next serial phase after retirement |
| Local fit inferred from aggregate cluster capacity or stale report | M4a: each node rechecks its own guarantees; M6 adds per-rank coordinated admission and failure/collective tests |

M2 supplies deterministic fake-backend execution plus the real GGML/VMM
allocation and retirement evidence. M3 binds finite API request defaults;
M4 chooses measured retention limits and proves concurrent execution;
M5 validates pessimistic routing envelopes before overlap optimizations.
No new runtime, model support, spill format, public configuration schema or
numeric production default is introduced by this policy document.
