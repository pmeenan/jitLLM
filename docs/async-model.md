<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Native tasks and completion ownership

D-048 answers M0 question 3: start with explicit, resumable C++23 task state
machines, one node-local scheduler/catalog writer, and bounded provider
services. This is an internal design, not application code or a frozen backend
ABI. M2's real GGML/VMM proof still settles the operation contract. D-050's
[reservation policy](reservation-policy.md) separately defines guaranteed
capacity and retained-state/phase envelopes.

## Execution and thread ownership

The scheduler owns task states, admission and capacity ledgers, catalog
mutations, residency-lease grants/revocation, and retirement decisions. Other
threads send owned commands or observations; they never modify those ledgers
or invoke a task continuation inline. Readers consume bounded snapshots or
ask the owner. Start with one writer for the entire node, not one per model.

Each scheduler turn drains a bounded batch of completion/control observations
and advances a bounded batch of ready tasks. Ready tasks are deduplicated and
served round-robin within their eligible priority class; priority policy must
include starvation prevention. Each step does bounded host work and either
yields, waits on identified dependencies, finishes, or starts safe unwind.
No task blocks this thread waiting for disk, GPU, network, another task, or a
full downstream queue. Bounded CPU jobs, including integrity verification,
run in a worker lane and return an observation. Waiting continuations own
their state and retain its accounted bytes; a suspended stack cannot hide it.

The initial service roles are:

| Role | Work and boundary |
| --- | --- |
| Scheduler/catalog owner | Short serialized policy and state transitions; local execution and conductor requests use the same admission path |
| Storage service | Bounded direct-file submission and completion harvesting; owns its I/O queue and registration bookkeeping |
| Device submission/VMM service | Ordered backend launches, mapping and backing operations; explicit CUDA context/stream ownership stays in the provider |
| Device completion service | Queries recorded completion fences independently of a possibly blocking submission/VMM call; reports observations |
| Network/front-door service | Bounded protocol work, transport completions, cancellation and backpressure; conductor traffic has no per-expert authority |
| Bounded CPU workers | Tokenization, hashing, plan preparation and other potentially long host work; no catalog mutation |

Start with a dedicated scheduler thread and separate storage, device
submission and device-completion lanes. The network role arrives with M3/M4a.
Worker counts and sleep/poll intervals are implementation settings to measure,
not performance claims or new pins. A lane can contain multiple workers only
if its provider's ordering/context contract permits it. Keep completion
harvesting independent of submission waits. In particular, serializing all
GPU activity on one blocking worker cannot be the only way to observe progress.
No CUDA work or scheduler callback runs from a CUDA host-function node.

## Tasks, operations and three different endings

A request owns a bounded task tree. Each task has a generation-tagged ID,
explicit phase/program counter, bounded continuation storage, wait set,
cancellation state, and one ready-queue membership bit. No raw coroutine or
stack address is a completion identity. A task slot cannot be reused while
any operation or child still refers to it. Generations must not wrap into a
live identity; exhaust the ID space by stopping admission, not by reuse.

Keep these independent:

1. **Client outcome:** a response, cancellation, deadline or connection loss.
2. **Task outcome:** success, failure or cancellation of the computation.
3. **Resource retirement:** all accepted consumers have stopped touching their
   backing, necessary registration retirement has completed, and retained
   mutable data is preserved or deliberately invalidated.

A client deadline may settle the first while the other two remain pending.
Cancellation closes admission of new child work and requests best-effort
provider cancellation. Already accepted operations stay owned. It never
destroys an in-flight record or releases its lease. Child errors propagate
to the parent, which drains its accepted children before destruction. Shared
page-in operations belong to the resource service and have bounded waiter
lists: cancelling one waiter removes only its interest. Other waiters and
accepted DMA remain protected. Only the final uninterested waiter can request
provider cancellation, and even then the underlying operation must drain.

An in-flight operation record owns its leases, backing/content generations,
destination ranges, provider identity, completion obligations and any
registration references. Operation outcome and proof of no further access
are separate fields. A provider error that proves a terminal failed operation
can permit retirement; an error that leaves completion uncertain cannot.
An unknown device/transport state quarantines the affected backing, remains
charged to occupancy, stops affected admission and surfaces a node fault.
A timeout never turns it into free capacity. Recovery must establish that
all relevant access domains are quiescent; otherwise keep ownership until
process/device recovery under a separately validated shutdown procedure.

## Submission and completion protocol

The following names describe semantics, not final C++ signatures:

1. **Prepare under the owner.** Validate dependency/content/backing generations,
   acquire the required leases, and allocate a bounded operation slot plus
   submission, terminal-result and cleanup credits before handing work off.
   Store the record before a provider can access memory or report completion.
2. **Publish command.** Transfer provider work through an owned bounded queue.
   The provider resolves every submission attempt as `not_started`,
   `accepted`, or `unknown`; an exception or ambiguous enqueue result is not
   evidence that no work started. Partial batches resolve each element.
3. **Observe.** A provider publishes value observations keyed by operation ID
   and generation. Completion arriving before the acceptance observation is
   retained; the scheduler does not retire before reconciling both. No
   provider calls the continuation while submitting. Duplicate observations
   are idempotent; contradictory results fault the provider. Stale IDs cannot
   update a new operation, task or backing generation.
4. **Validate and advance.** A read's terminal byte count, range and integrity
   must pass before publishing resident content. Mapping/access readiness and
   content readiness are independent. Short/error reads do not publish a
   tensor. GPU work must have all selected dependencies ready before launch.
   Failed or cancelled writes never publish a recoverable snapshot; spill
   publication/durability belongs to the separate spill-format policy.
5. **Join and retire.** Every GPU stream, disk operation, network consumer and
   registration that can still access a backing identity must be accounted
   for. Retire operation leases only after the relevant completion proofs.
   Releasing a lease changes eligibility; eviction remains a separate catalog
   transition. A still-registered extent may remain resident and reusable
   for a compatible operation, but cannot be remapped/reassigned/freed until
   that registration's retirement is confirmed.

Submission and lease transfer are one logical ownership transaction. Before
provider access, failure can roll back. Afterwards only completion-aware
unwind is allowed. A failed second submission does not roll back the first
accepted operation's ownership. Destructors may assert drained state or hand
ownership to a bounded retirement service; they do not assume completion or
silently detach work. No exceptions cross a C ABI.

Generation checks protect metadata; they do **not** prevent a late DMA write
to a reused address. The lifetime hold prevents reassignment in the first
place. Tensor aliases, captured GPU pointers and registered buffers all join
the same backing retirement conditions, even when their virtual address is
unchanged.

## Bounded queues and progress under saturation

Budget task/continuation bytes, waiters, operation records, ready entries,
submission entries, provider completion storage, cancellation state, cleanup
commands, snapshots and output buffers before admission. Limits are explicit
configuration/internal profile values; M1/M2 choose runtime numbers within
D-050's memory envelope. No unbounded `std::function` queue or task per
tensor is implicit in this design.

An admitted operation owns a result slot until consumed. The initial protocol
can use a per-operation mailbox and a coalesced wake flag: a full notification
queue cannot lose terminal results, and the owner rechecks pending mailboxes
before sleeping. Publishing is release/acquire synchronized. Provider rings
must also have sufficient completion capacity and a harvesting path that
does not depend on scheduler admission. Any cancellation operation requiring
its own completion consumes pre-budgeted control credits; when unavailable,
retain the cancellation intent and let the original operation drain.

Admission/output queues may apply backpressure or reject/defer before
submission. Completion and retirement of already admitted work do not depend
on space in those queues. Reserve a cleanup lane or operation-owned cleanup
slot so deregistration/unwind can run even when normal submission is full.
Output backpressure can suspend production at a completed boundary, but never
the collection of device/I/O completions. Cancel/shutdown intent is a bounded
state flag, not one queued allocation per repeated request.

Ready membership is at most one entry per live task; capacity is at least the
admitted task bound. Register wait conditions and check readiness atomically
with owner state changes to prevent a lost wakeup. With several producers,
their mailbox publication and the owner's sleep/recheck protocol require
real concurrency tests in M2; a deterministic simulator does not prove the
C++ memory ordering.

This ensures infrastructure can drain a full queue, conditional on provider
completion or explicit fault handling. It does not prove a suspended model
phase can obtain more memory. D-050 bounds retained state/growth and keeps
the complete phase allowance through waits to prevent circular capacity
waits; M2 must validate that policy. Time-slicing between requests occurs
only at client-facing request boundaries, after a scheduler-established
completed handoff; no task API authorizes mid-operation eviction. Another
request's phases can run during an I/O wait only when their full concurrent
envelopes pass D-050's admission check.

## Why explicit states first

Explicit states make suspension bytes, cancellation points and transitions
visible in deterministic tests and avoid choosing a third-party execution
framework before the backend proof. They cost more state-machine code.
Coroutines can later be a local notation over these same ownership rules if
frame allocation, destruction, continuation scheduling and cancellation are
equally bounded and tested. They do not replace completion ownership.
Sender/receiver libraries remain an alternative if a concrete integration
need justifies their dependency and adapter costs. No claim about relative
speed, compiler support or ecosystem maturity decides this choice.

## Provider checks and validation gates

Primary-source checks on 2026-09-22 support the adapter requirements:

- [CUDA event documentation](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html):
  a newly created event has no captured work; successful query alone is not
  proof that a launch was fenced. Record after the final consumer on each
  relevant stream, or an explicit join. Do not re-record a live fence for
  another operation. Event destruction can precede device completion and
  does not retire the operation's backing. Query errors can originate from
  earlier asynchronous work; classify them conservatively.
- [liburing cancellation documentation](https://man7.org/linux/man-pages/man3/io_uring_prep_cancel.3.html):
  cancellation has its own result and can race with completion or be too late.
  The storage adapter waits for and reconciles the original request's terminal
  completion; cancellation intent or its acknowledgement alone is never the
  pager's retirement signal.

The [CPU-only experiment](experiments/async-model/README.md) challenges the
ownership/state design with controlled event order and bounded records. It
does not select a new source dependency, implement CUDA/io_uring adapters,
prove multithreaded publication, or validate reservation progress. M2 must
add real concurrency/lost-wakeup tests, coalesced page-in waiters, task-tree
unwind, full queues during cleanup, and the GGML host-VMM execution proof.
M4a/M6 add transport registration, lost-node and collective-order validation.
Shutdown must stop admission, cancel queued work, drain accepted operations
and registrations, then release backing; unreconciled work faults shutdown
instead of reporting that its capacity was reclaimed.
