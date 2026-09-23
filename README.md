# jitLLM

Just-in-time LLM inference engine with intelligent SSD paging.

jitLLM is an independent, open-source inference runtime for setups where more
models should be available than fit in memory. Instead of loading and
unloading whole models, it keeps a node-wide catalog of every managed memory
extent, evicts the least valuable extents across all models when capacity is
needed, and pages missing weights or state back in on demand from prepared
on-disk artifacts. Routed-expert (MoE) models get exactly the experts the
router selected, loaded just in time.

The workload it is built for first is one person switching among a library of
models, or running an agent whose subagents use different models, with
conversations that last hours. Useful conversation state survives switches
within configured retention limits; expired caches can be rebuilt from the
history clients provide.
Standard web-API clients such as Cursor, OpenCode, and Codex work unmodified,
and with more than one node a single conductor places models across the
cluster and routes requests. NVIDIA and DGX Spark come first; the memory,
paging, and transport boundaries are kept portable so Apple silicon or AMD
single-machine ports stay possible later.

Initial target: one or two NVIDIA DGX Sparks, developed from an x86-64 Linux
workstation. Model support is earned per checkpoint and tracked in a support
matrix; see [docs/features.md](docs/features.md) for what is confirmed scope
versus still being triaged.

Key properties (confirmed scope):

- **Partial, cross-model eviction** at extent granularity, with capacity
  reservations kept separate from residency leases so admission never
  eagerly evicts useful cache.
- **On-demand expert acquisition** at a routing boundary: no expert
  substitution, no dropped contributions, and execution is suspended while
  I/O is in flight so other work can run.
- **Explicit CUDA virtual memory management** with a semantic resource
  catalog, prepared and hashed model artifacts, and explainable
  eviction and admission decisions.
- **Native C++23 runtime**, Clang-first, no interpreter in the serving path;
  cross-built for Spark and tested over SSH.

Almost all code is written by AI agents working from the project
documentation. Every change gets a separate review pass, and a human directs
the work, reviews it, and is the sole committer.

## Status

**Pre-code. M0 (plan the plan) is done; M1 (bootstrap) is next.** The
design brief is in [docs/ideation.md](docs/ideation.md); the living plan,
feature matrix, architecture, and decision log are in `docs/`. No application
code exists yet. Planned distribution is a signed apt repository for DGX
Spark.

The first useful product target is M4: chat with A, switch to B under memory
pressure, then resume A with retained state, through an unmodified client.
M4a adds configured placement across nodes; demand-paged MoE and sharding
have separate later gates. See [the plan](docs/plan.md).

## License

jitLLM's own code is Apache-2.0 (see [LICENSE](LICENSE)). Incorporated core
implementation dependencies use Apache-2.0, BSD, MIT, or MPL-2.0. Build tools
and declared platform dependencies, including system libraries and CUDA,
retain their separate terms and are included in the dependency audit.
AGPL-licensed kernels or importers live only in optional modules you choose
to enable at build time; those builds must report and satisfy the applicable
license, notice, and source obligations. See D-003 and D-017 in
[docs/decisions.md](docs/decisions.md).

## Start here

- [AGENTS.md](AGENTS.md) — constraints, doc map, agent rules
- [docs/vision.md](docs/vision.md) — why, who for, success criteria, non-goals
- [docs/features.md](docs/features.md) — confirmed / proposed / open questions
- [docs/plan.md](docs/plan.md) — the M1–M8 milestone ladder with exit criteria
- [docs/m0-record.md](docs/m0-record.md) — what M0's planning, spikes and reference runs did, with evidence links
- [docs/workflow.md](docs/workflow.md) — how agents and the human collaborate
- [docs/rough-edges.md](docs/rough-edges.md) — findings log
