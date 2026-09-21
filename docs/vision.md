# Vision

## What this is

jitLLM is an independent, open-source inference runtime built around one
premise: **model implementations describe computation and dependencies;
jitLLM owns storage, residency, scheduling, and execution lifetime.**

The target workload is one large model, potentially sharded across two NVIDIA
DGX Sparks, plus smaller models handling mixed traffic, where aggregate model
storage exceeds physical memory and not every model or expert is active at
once. Existing engines treat a model as an indivisible allocation: to make
room, you unload a model. jitLLM instead keeps a node-wide catalog of every
managed extent, reclaims the least valuable eligible extents across all models
when capacity is needed, and brings missing weights or state back only when
the model actually needs them, or when justified prefetching can hide a miss.
Routed-expert models get exactly the experts the router selected, loaded on
demand from prepared on-disk artifacts, with the execution suspended and other
work running while the I/O is in flight.

It began as the design brief in [ideation.md](ideation.md) (2026-09-20), which
consolidated the ideation discussion. The decided directions there are
recorded in [decisions.md](decisions.md); its proposed designs are in
[features.md](features.md) awaiting triage.

Correctness is a prerequisite for every supported configuration and every
optimization. Within that constraint, priorities are useful mixed-model
execution, warm generation performance, memory efficiency, and explainable
scheduling. Cold time-to-first-token is desirable but not primary; repeated
paging stalls during generation matter.

## Who it's for

jitLLM is built by one developer but meant to be consumed externally (D-016),
so the audience is anyone with the problem, not just the owner.

1. **People running local inference on one or two DGX Sparks**, the project
   owner first among them, who want more models available than fit in memory
   without losing warm-generation performance or the ability to explain what
   the runtime did.
2. **Owners of other CUDA hardware with a similar memory-versus-storage
   gap**, once a validated direct storage path or a second target makes
   their configuration a supported one.
3. **Inference-systems developers** interested in a memory-first runtime
   whose catalog, reservation/lease, and paging subsystems are usable and
   inspectable independently of any one engine's kernels.

## Success criteria

Each of these is checkable, and each is scoped to a *supported configuration*
in the model support matrix, not to arbitrary checkpoints.

- **Partial retention works.** With two persistent model contexts on one
  Spark, when the second needs capacity, only selected extents of the first
  are displaced; untouched extents remain resident; resuming the first
  reloads only the missing dependencies. Verified from catalog state and
  structured events, not inferred from timing.
- **Demand-paged MoE is exact.** On a supported routed-expert checkpoint, a
  phase loads only the experts the router selected plus declared
  metadata/granularity/read-ahead, never substitutes an expert, never drops a
  selected contribution, and the overhead of the routing/residency boundary
  on an all-resident path is measured and reported.
- **Numerics match a pinned reference.** Teacher-forced logits and declared
  intermediate outputs match a pinned known-working engine within a
  documented tolerance for each supported checkpoint. Generated text alone is
  not evidence.
- **Two Sparks stay correct under stress.** A sharded model on two nodes
  passes asymmetric-memory-pressure, cancellation, and controlled-failure
  tests with no speculative reuse after timeouts.
- **Every decision is explainable.** For any eviction or admission decision
  the runtime can report victims, expected and actual bytes recovered, the
  cost estimate, and why alternatives were retained, from structured events
  keyed by opaque request IDs.
- **Performance claims carry provenance.** Warm decode rate, inter-token
  latency distribution, bytes per token, and exposed stall time are reported
  for each supported configuration with artifact, backend, toolchain, driver,
  hardware, and policy identities, and compared against the pinned reference
  deployment before a backend or policy is promoted. Report both matched
  decoding configurations and the reference's normal optimized configuration,
  following [the comparison protocol](architecture.md#performance-evidence).
  The early paging-feasibility spike establishes workload benefit and
  generation-stall criteria; measured implementation results must meet them.
  Native code is never assumed to outperform an existing engine.
- **Setup is reproducible by a stranger.** A new user reaches a running
  supported model on a fresh Spark following only the checked-in docs and
  setup path, and the setup reports the exact toolchain, driver, and SDK
  identities it installed or found.
- **The default build meets its dependency policy.** Incorporated core
  implementation satisfies the Apache-2.0 / BSD / MIT / MPL-2.0 allowlist;
  declared tools and platform runtimes are recorded under their separate
  terms (D-017). The profile builds, passes the core tests, and its full
  dependency closure is audited. Optional modules and their dependencies
  are absent unless explicitly selected; enabled builds ship matching
  notices and source obligations.

## Non-goals

Stated by the owner for the first implementation, each with its reason.

- **Universal model support.** Support is earned per checkpoint and
  configuration; a family name is not a support claim.
- **A new tensor compiler, or a universal compiler framework in the first
  milestone.** Start with a small set of explicit operations and validated
  backend adapters.
- **A new quantization scheme.** Preserve the reference representation;
  requantization is an explicit, evaluated transformation.
- **Arbitrary GPU page-fault interception.** Explicit VMM: accessing absent
  backing is a bug, not a request.
- **Transparent cross-node shared virtual memory.** Two Sparks are two
  domains; transfers are explicit objects.
- **Production-grade multi-tenant isolation.** Single-owner local nodes.
- **Training.** Inference only.
- **A vLLM fork or plugin as the product architecture.** Reference and
  integration options only (D-005).
- **Rewriting proven kernels to claim native ownership**, or importing an
  engine's allocator/scheduler because we reuse its kernel. Reuse the unit
  with its assumptions; own the memory system.
- **Inventing dense-model sparsity by rearranging weights.** A layout change
  does not let a dense kernel skip bytes.
- **A Python interpreter in the serving, paging, or scheduling hot path.**
  Build-time tooling may use it (D-010).
