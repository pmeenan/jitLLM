# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note. When a milestone exits, its section shrinks
to a short summary and its task-by-task history moves to a record beside this
file ([M0](m0-record.md)).

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `done`

Ran 2026-09-20 to 2026-09-23 and exited on the owner's approval of the plan.
M0 turned the [design brief](ideation.md) into the vision, the triaged feature
matrix, the approved architecture and decisions D-001–D-069, and gathered the
evidence they rest on: toolchain, VMM, I/O and interconnect spikes on the
Sparks; the llama.cpp, EXL3, image and MiMo reference runs; the measured A→B→A
reference cycle; the paging-feasibility study; and the v0 artifact layout
study. The [M0 record](m0-record.md) keeps each task's outcome, evidence links
and caveats. Everything M0 left open is owned by the milestone exits below.

## Milestone ladder

Rewritten from the provisional ladder at the end of M0 (2026-09-23; see the
[M0 record](m0-record.md)) and ordered by risk: the substrate; the resource core with the backend proof; one
resident model end to end; the first useful product at M4 (A→B→A with
retention); configured placement at M4a; demand-paged MoE with the first
daily-driver models at M5; sharding at M6; performance and the new decoding
modes at M7; and the remaining product scope with the first tagged release at
M8. Each milestone leaves a usable, testable result. None has a promised date.

| Milestone | Result | Needs |
| --- | --- | --- |
| M1 Bootstrap | Pinned SDK, builds, local check gate, package skeleton, confined-job proof | M0 (done) |
| M2 Resource core | Catalog, admission and leases on a fake backend and a Spark; the backend proof settles the operation contract | M1 |
| M3 One resident model | Import, native GGML/EXL3 serving and the three client protocols on the small fixtures | M2 |
| M4 First useful product | A→B→A with partial retention; switching policy chosen from measurement | M3 |
| M4a Configured placement | Conductor, enrolled nodes, whole-model placement and routing | M4 |
| M5 Demand-paged MoE | Exact expert paging; Gemma 4 and Ornith as daily drivers with reasoning and constrained output | M4 |
| M6 Sharding | A flagship model sharded across both Sparks | M4a, M5 |
| M7 Performance | D-036's benefit target on a library larger than memory; speculative and diffusion decoding | M5; M6 before exit |
| M8 Product and release | Dashboard, remaining API scope, signed apt repository, first tagged 0.x release | M7, for the release |

How to read the milestones below:

- **Exit criteria are the contract.** Scope lists say what a milestone
  builds; its entry work refines them into tasks. An exit criterion changes
  here, visibly and with the owner, never by drift.
- **Detail lives in the linked designs.** Where a gate cites a matrix or an
  acceptance section, every row that section assigns to the milestone is
  part of the gate.
- **Evidence rules apply to every gate.** Thresholds are declared before the
  measurement they judge, and inconclusive comparisons do not pass (D-036).
  Every result names the check tiers and hosts that produced it (D-061,
  [workflow.md](workflow.md)). No performance gate is met by giving up
  correctness.
- **Heavy path.** Work in a [blast-radius area](workflow.md#blast-radius-changes-get-the-heavy-path-by-default)
  passes its adversarial challenge before the milestone exits.
- **Support is earned per checkpoint.** From M3 the support matrix records
  what each exit validated; a milestone exit names its configurations.
- **Release.** The first tagged 0.x release (an owner-signed tag, D-062) is
  cut at M8's exit, after every milestone has exited (owner, 2026-09-23). The
  repository is already public (D-061); until then it carries only `-dev`
  builds.

## M1 — Bootstrap  `in progress`

Goal: a reproducible, pinned developer setup and repository skeleton that
every later milestone builds, tests and packages on, with D-061's local check
gate in force and the first `jitllm` binary running on a Spark.

**Entry:** M0 exited on 2026-09-23.

**Scope:**

- [x] **SDK provisioning** (D-012, D-049): `mise.toml`/`mise.lock`,
      `toolchains/manifest.toml`, `toolchains/artifacts.lock.json`,
      `tools/setup-toolchain`, `tools/check-toolchain`, and the
      digest-pinned reference container (`.devcontainer/`) built from the
      same logic. A persistent, versioned SDK outside the checkout and `/tmp`
      holds every profile's tools at their pins: Clang/LLD and the LLVM
      formatter, linter, language server and symbolizer 22.1.8 with matching
      compiler-rt sanitizer runtimes (D-032, D-059); CUDA Toolkit 13.4.2
      (NVCC 13.4.92); CMake 4.4.3 (D-058); Ninja 1.13.2; and the source-built
      GCC 16.2 runtime for each architecture, the AArch64 copy inside the
      target sysroot (D-060). Host prerequisites stay system-managed and are
      declared and checked, `qemu-user-static` with binfmt among them. The
      mise tasks are `setup`, `doctor`, `build`, `test` and `deploy`.
      *Landed (D-070):* the manifest, lock, prerequisite lists, `setup` and
      `doctor` tasks and reference container, with the sysroot built from
      pinned Ubuntu packages and the AArch64 GCC runtime cross-built. Setup
      passed on the workstation, on `spark` and in a clean reference
      container. `build`, `test` and `deploy` landed with the Build item.
- [x] **Build** (D-010, D-011): `CMakePresets.json` and `cmake/toolchains/`
      for native x86-64, the AArch64 cross build (CPU, and CUDA for
      `sm_121`), the native-Spark fallback, and a CPU-only configuration
      with no CUDA toolkit visible (D-026). C++23 for `.cc` and `.cu`;
      static libstdc++, libgcc and cudart (D-060); `compile_commands.json`.
      At the repository root: `.clang-format`, `.clang-tidy` (findings are
      errors), D-059's warning set with `-Werror`,
      `CMAKE_CXX_SCAN_FOR_MODULES OFF` and `-fno-exceptions`, tests
      included (D-066).
      *Landed:* presets `native`, `cpu`, `cross` and `spark-native`, whose
      toolchain files use the SDK (plus the host GNU linker on Spark) and
      check its receipt against the checkout; `tools/build` behind
      `mise run build`, `test` and `deploy`; and `tools/run-target`, which
      runs cross-built tests under qemu-user or, with `--host`, on a Spark over SSH. The
      `tests/toolchain/` contract tests (C++23 with the GCC 16.2 library, no
      exceptions and an aborting throw path, the explicit CPU baseline,
      glibc-only dynamic dependencies, no symbol version above 2.39 for Spark
      binaries, no RPATH, and a CPU-only build that never sees CUDA) passed
      in every preset, cross under qemu-user and on `spark`; there the sm_121
      CUDA test ran cross-built and natively built. NVCC's host pass uses
      D-059's warnings less `-Wold-style-cast`, which CUDA's own headers trip. GoogleTest under
      `-fno-exceptions` is proven with the Source dependencies item.
- [ ] **Source dependencies** (D-057): the source lock, a preparation step
      separate from SDK setup, lock validation and the build receipt, proven
      on GoogleTest 1.18.0 with gMock through every
      [M1 gate](source-dependencies.md#upgrades-packaging-and-m1-gates).
      Kernel closures (the GGML subset and the ExLlamaV3 files) enter
      through this mechanism at M2's P0 stage, not here.
- [ ] **Local check gate** (D-061): the `check`, `check:full` and
      `check:spark` tasks with D-061's contents. Benchmarks join the gate as
      regressions when their harnesses and baselines exist (the EXL3 kernel
      and parity benchmarks from M2/M3, trace replay from M4/M5); thresholds
      follow measurement.
- [ ] **License and provenance** (D-017, D-029): `LICENSES/`, REUSE
      metadata and lint, the embedded-header check, a root `NOTICE`, and an
      SBOM generated with the package. Audit the notices of what ships and
      what builds it: CMake, Ninja, LLVM, glibc, the CUDA runtime, and the
      static GCC runtime with its embedded components.
- [ ] **Versioning** (D-062): `project(VERSION)`, the dev-version
      derivation and its Debian `~` mapping, `jitllm --version` and the
      receipt (version, commit, license profile, SDK identity), a root
      `CHANGELOG.md`, and the version of D-047's opaque reasoning-signature
      representation.
- [ ] **Smoke binary and capability probe** (D-026): a `jitllm` binary,
      from the cross build and from the native Spark fallback, that runs on
      `spark` over SSH, with a first-cut `doctor` reporting VMM granularity,
      GDS mode, RDMA availability, driver and toolkit versions, glibc and
      ABI, the selected tools and host prerequisites, and any data role
      relocated onto a read-only filesystem (D-063).
- [ ] **Package and installed layout** (D-027, D-063): an arm64 `.deb`
      (M1 chooses CPack or debhelper) in the
      [installed layout](architecture.md#installed-layout): the `jitllm`
      user; `jitllm.service` with sandboxing that keeps GPU and RDMA device
      access, and its restart policy; the per-node process lock; core dumps
      off and a non-dumpable process, checked on each host by an abort that
      leaves no core file or apport report; `/etc/jitllm` and the `/var/lib/jitllm` roles
      with their modes. It depends on `libc6` (`GLIBC_2.38`) and a versioned
      `libcuda.so.1` floored at NVIDIA's minimum driver for the pinned
      toolkit. The install test runs in an arm64 container; M1 decides
      whether it also starts the unit.
- [ ] **Node configuration** (D-063): a strict TOML 1.0 parser (toml++ is
      the candidate under D-017, D-057 and D-066) and D-063's node-local
      keys, `[storage]` included, validated fail-closed with exhaustive
      diagnostics. Front-door and TLS keys are fixed in M3 and
      switching-policy keys in M4 (D-069).
- [ ] **Confined job proof** ([job rules](architecture.md#import-install-and-archive-jobs)):
      choose the mechanism (a delegated cgroup or a subreaper) together with
      the unit's sandboxing; unprivileged `unshare` and `bwrap` are blocked
      on these hosts (RE-013). Prove, under the `jitllm` account, containment
      of a child that outlives its job, the job lock inherited across exec,
      and correct handling across a runtime restart, before M3 builds the
      importer on it. Any system change the proof needs on a host is the
      owner's to approve.
- [ ] Update AGENTS.md's repository-layout table as the scaffolding lands.

**Exit criteria:**

- Setup succeeds on a clean workstation host and in the reference container
  from the declared prerequisites alone, with no global compiler or
  environment changes and no dependency on temporary SDK directories.
  `doctor` reports the exact toolchain, driver and SDK identities.
- `check` passes on the workstation, including the CPU-only configuration
  and the AArch64 CPU tests under qemu-user. `check:full` passes from a fresh
  clone: the sanitizer builds, the copyleft-disabled build from empty caches,
  the network-denied container build, D-057's gates, the `.deb` and its
  install test, and the package inventory against the receipt, NOTICE and
  SBOM. `check:spark` runs the CUDA smoke, `doctor`, LeakSanitizer and
  ThreadSanitizer on `spark`. The confined-job proof passes on its chosen
  host, and an abort leaves no core file or apport report on each host.
- Every new dependency or pin records its D-017 category and tier, with a
  decision entry where AGENTS.md rule 1 calls for one.

## M2 — Resource core and backend proof  `pending`

Goal: the node-wide catalog, ledgers, reservation and lease state machine and
D-048's task lanes, deterministic on a fake backend and real on a Spark.
Alongside them, the early backend integration proof runs real GGML FP16 and
EXL3 kernels on jitLLM-owned memory and settles the internal operation
contract. M2 stays small (owner, 2026-09-23), so it adds no tokenizer, C++
importer, HTTP or new model shapes. Future shapes appear only as fake-provider scenarios,
and the concrete interfaces come from the real GGML/EXL3 proof.

**Entry:** M1 exit. The prerequisites the provisional ladder listed (paging
feasibility, the measured reference cycle, D-036's criteria and D-050's
reservation policy) were recorded in M0.

**Scope:**

- [ ] **Catalog and ledgers** (D-006, D-007): typed IDs and generations,
      dependency closures, shared extents charged once, the memory-domain
      key, separate commitment and occupancy ledgers, unknown allocations
      non-evictable, and the deterministic LRU
      [victim-selection baseline](architecture.md#victim-selection-initial-baseline).
- [ ] **Admission and leases:** D-050's guarantee for one active request at
      a time (concurrency arrives in M4), with D-069's pause rules and
      guards.
- [ ] **Task lanes** (D-048): scheduler, storage, device submission, device
      completion and CPU workers, with bounded queues; measure worker
      counts, queue sizes and wakeup and polling behavior. Real
      multithreaded lost-wakeup and memory-ordering tests, with ARM stress
      on a Spark; the deterministic simulator does not prove them
      ([async model](async-model.md)).
- [ ] **Providers** (D-026): device-memory, device-execution and storage
      interfaces, each with a deterministic poison-filling fake; the CUDA VMM
      provider at D-033's 2 MiB extents; direct-I/O reads into host VMM
      (D-034) that handle short transfers, alignment, retries, cancellation
      and duplicate-load coalescing explicitly, with storage queue depths
      and run sizes measured (M4 tunes them against spill write-back).
- [ ] **Backend integration proof** ([scope](backend-proof.md), stages
      P0–P6). The owner approves the frozen bounds and performance protocol
      at P0, before any native output is seen. The GGML subset and the
      ExLlamaV3 files enter through D-057 with reviewed patches, and GGML's
      C++ CUDA launchers are checked for throws at the pinned revision
      (D-066). The FP16 control and both EXL3 fixtures run from v0 prepared
      artifacts built by M0's prototype. D-053 dispatch records each
      operation's K-C or K-L choice, selects between at least two
      implementations of one operation by plan, and alternates FP16 and
      EXL3 in one process. The five-rung oracle ladder applies throughout. The plan stays GEMM-only until the
      GEMV provenance gate closes, and the gap to upstream is measured.
- [ ] **Retained-backing comparison** ([scope](backend-proof.md#retained-backing-comparison)):
      build the cross-model swap trace, have the retain/amend criteria
      approved, then keep or amend D-033.
- [ ] **Shape expressibility** (D-068): fake-provider scenarios for draft
      rejection and rollback, a canvas across boundaries, block output and a
      two-artifact context.
- [ ] **Explainable plans:** plans expose their validated phase widths,
      envelopes and rejection reasons, and the proof records each phase
      kind's guaranteed bound against its observed peak.
- [ ] Find which OS counters include VMM backing on the Spark driver, so
      the [memory breakdown](architecture.md#memory-breakdown) reconciles.
- [ ] Record the operation contract, registry, patch set, phase envelopes
      and `F` per profile in a decision entry, and the aggregate report in
      `experiments/backend-proof/`.

**Exit criteria:**

- Every M2 row of D-050's
  [adversarial matrix](reservation-policy.md#worked-cases-and-implementation-gates),
  with D-069's pause cases, passes on the fake backend with no vendor SDK
  present (`check`); rows that need real allocation pass as their BP cases.
- The BP [case matrix](backend-proof.md#case-matrix) passes, its
  fake-backend and CPU-only cases on the workstation and the rest on
  `spark` (`check:spark`), including repeated map/load/evict/restore.
  BP-V1's importer cases run against M0's prototype, and report-only cases
  are reported. The FP16 control and both EXL3 fixtures run native prefill
  and decode that stay within the declared oracle bounds before and after
  restoration.
- The EXL3 kernel-time and workspace gates (BP-F2) pass, or the owner has
  approved an explicit tradeoff. A loader or an FP16 conversion is not EXL3
  support (D-052).
- D-048's lanes pass their real concurrency, lost-wakeup and task-tree
  unwind tests, including ARM memory-ordering stress in `check:spark`.
- D-033 is explicitly retained or amended, and the operation-contract
  decision can express D-068's shapes without executing them.

## M3 — One resident model, end to end  `pending`

Goal: import the small fixtures, serve them resident through native GGML and
EXL3 execution, and complete chats from named clients over the three
baseline protocols, with bounded, explainable memory.

**Entry:** M2 exit with its operation-contract decision. The provenance of
llama.cpp's generated Unicode tables is cleared under D-017 before the native
tokenizer is adopted ([first-slice.md](first-slice.md)).

**Scope:**

- [ ] **Importer and verifier** (D-009, D-056): the C++ importer and the
      standalone verifier, with M0's Python prototype as the exact
      accept/reject oracle, running on the workstation and on a Spark.
      Confined import jobs take sources only from the configured stores
      (the `checkpoints` role or the long-term store; D-054, D-063) or the
      Hugging Face Hub (resumable and verified; the token comes from a
      `.env` or config file and is never logged). Hard-linked and symlinked
      sources are rejected, and the user docs say to copy them in or
      download with `--local-dir` (D-063). Removal waits for
      quiescence. Jobs keep their records, locks and grants and report
      progress, status, cancellation and retry (D-041). An install that
      lacks space fails before transferring, reporting the space it needs
      and the removal candidates.
- [ ] **Registration and first use** ([model lifecycle](architecture.md#model-lifecycle)):
      the compact installed-model index cache, shallow checks on every
      startup, and detail and plans built on first use inside an `F`
      reservation. Startup rejects an installed store that fails the
      direct-I/O probe (D-054).
- [ ] **Resident execution:** the FP16 and both EXL3 fixtures under jitLLM
      dispatch, finite context and chunk profiles within the 8K context,
      and state block sizes and KV layouts for each adapter. Graph capture
      enters here, behind a relocation proof, only if BP-F4 shows parity
      needs it.
- [ ] **Tokenization and output:** the native tokenizer, D-067 renderers for
      both template hashes, the tool-call parser, stop rules and seeded
      sampling; host versus device sampling is settled against D-052's
      decode gates.
- [ ] **Front door** ([M3 surface](client-api-baseline.md#m3-surface)):
      Chat Completions, stateless Responses, Messages with token counting,
      `/v1/models` in both shapes with D-046's metadata, and the discovery
      document (D-041); D-045's listener, authentication, CORS, `Host`,
      status and keepalive rules; D-047's non-streaming and storage rules;
      the Claude Code profile; and explicit rejection of `transforms` and
      `plugins`. Numeric intake bounds are fixed before any external input
      is accepted, with the total input limit reconciled with named-client
      tests ([cluster design](cluster-design.md)).
- [ ] **Local management API and CLI** (D-064): versioned routes for import,
      listing, representation inspection, removal, jobs and node health.
- [ ] **TLS** (D-065): per-name certificate files selected by SNI, reload on
      change, key-match and expiry checks, the name-constrained local CA,
      the certbot deploy hook, and the Tailscale certificate timer. The
      root-run hook and timer touch only `/etc/jitllm/tls/`, their units are
      sandboxed to it, and they take the heavy path.
- [ ] **Surface definitions:** front-door, alias and TLS configuration keys;
      the individual `jitllm-` header and body-field names (D-062); the
      HTTP, TLS and JSON libraries, chosen under D-017, D-057 and D-066.
- [ ] Start the model support matrix, recording template hashes.

**Exit criteria:**

- Teacher-forced logits and declared intermediates for the three fixtures
  match their pinned references within bounds declared before evaluation
  ([first-slice.md](first-slice.md), [exl3-bringup.md](exl3-bringup.md)).
  Rendered bytes and token IDs match the golden fixtures.
- EXL3 resident prefill, time to first token and decode inter-token
  p50/p95/p99 meet the predeclared upstream parity bounds against upstream
  serving controls in matched and normal views, with memory inside the
  declared bounds; a regression needs a fix or an explicit owner-approved
  tradeoff (D-052).
- The [M3 acceptance cases](client-api-baseline.md#acceptance-owed-in-m3)
  pass as scoped there. At least one named client completes a chat
  unmodified with each representation, and Cursor stays an explicit gap
  unless resolved. The context-compacted release moves to M4, which
  delivers D-041's close, and the Ollama-native checks move to M8 with the
  Ollama profile.
- Finite default context and output bounds bound every admitted request,
  and the M3 rows of D-050's matrix pass. Memory use is bounded and
  explained by the memory breakdown.
- The importer, verifier and front-door parsers pass their adversarial
  challenge; an interrupted import never appears valid.
- Measured and recorded: startup time and metadata memory as the installed
  library grows, the cold-switch cost of on-demand detail, and both
  contexts' state bytes, from which D-055's
  [capacity values](retention-policy.md#bounds-and-defaults) are pinned.

## M4 — First useful product: A→B→A with partial retention  `pending`

Goal: two small model contexts share one local budget. Switching to B
displaces only what B needs, retained conversation state lets A resume
without a full re-prefill, and the switching policy is chosen from
measurement. Useful without MoE or sharding.

**Entry:** M3 exit with its capacity values pinned, plus everything
[M4 entry pins](retention-policy.md#what-m4-entry-pins): the frozen
transcript, budgets and reference paths among them, and the spill write
budget from the drive's rated endurance. EXL3 switching budgets come from
new matched controls ([exl3-bringup.md](exl3-bringup.md)).

**Scope:**

- [ ] **Retention** ([policy](retention-policy.md); D-024, D-031, D-055):
      prefix and continuation entries with their identity, restore
      boundaries, branches, sharing and refresh; capacity-driven expiry with
      24-hour idle caps; the victim-order baseline; spill with its protected
      directory, preallocation, direct I/O, digest check on restore and
      deletion at startup; miss reasons and fallback reporting.
- [ ] **Partial eviction** of a quiescent model, reloading only missing
      dependencies (D-008).
- [ ] **Concurrency when it fits:** all-resident cohorts under
      full-envelope checks, reporting whether requests ran concurrently or
      time-sliced.
- [ ] **Switching policies** (D-069): the priority-aware default,
      run-to-completion and time-slicing with their guards, their
      configuration keys and per-alias overrides.
- [ ] **Request control** (D-042): interactive and background classes,
      maximum queue waits, cancellation and bounded progress events.
- [ ] **Release** (D-041, D-045): the final-turn flag, idempotent
      continuation close and the context-compacted release, with their wire
      names fixed here.
- [ ] **Warm jobs** (D-041): capacity-constrained, never an implicit
      download.
- [ ] **Diagnostics:** status, the admission what-if query, Perfetto trace
      export and eviction explanations; management controls for priorities,
      residency policies and trace capture; Prometheus `/metrics` with
      compatible health and load queries (D-044).
- [ ] **Storage scheduling:** demand reads mixed with spill write-back,
      inside the pinned write budget.
- [ ] Add the A→B→A workload and its regression thresholds to `check:spark`.

**Exit criteria:**

- D-055's [timed workload](retention-policy.md#m4-acceptance-workload)
  passes its pass rule: Qwen2.5-0.5B FP16 and EXL3 4.0 bpw in both
  orientations, six jitLLM arms against fresh interleaved references, 72
  accepted repetitions per arm, orientation and cache condition. For each
  floor arm (J-partial, J-spill), orientation, direction and cache
  condition, at the median and at p95, jitLLM's one-sided 97.5% upper bound
  is at most the smallest one-sided 97.5% lower bound among the valid
  reference arms; a comparison with no valid reference arm does not pass
  (D-036). The report covers latency distributions, bytes read and written,
  peak memory and spill, prompt tokens reused versus recomputed, and deltas
  against jitLLM's whole-model control (M7's comparator).
- The [correctness gates](retention-policy.md#correctness-gates) pass: exact
  outputs and bit-identical teacher-forced logits against
  provenance-matched controls, and catalog state and events show that only
  selected extents were displaced.
- The [functional and adversarial cases](retention-policy.md#functional-and-adversarial-cases)
  and the M4 rows of D-050's matrix pass with an EXL3 context in the
  matrix; cache expiry never destroys admitted suspended work.
- Through at least one unmodified named client: a long conversation on A,
  B under pressure, then A resumed, over both resident reuse and forced
  spill/restore. The context-compacted release case deferred from M3
  passes.
- B arriving while A is still generating is measured under each D-069
  policy, with queue delay reported apart from paging and switch time and
  from first-token compute, together with pauses and bytes reloaded. The
  owner keeps or changes the default on these results.
- An all-resident control shows concurrent progress when both complete
  envelopes fit.

## M4a — Configured placement across nodes  `pending`

Goal: one configured conductor places whole models on enrolled nodes and
routes requests with retained-state affinity, so a subagent's model runs on
the other Spark while the main model stays resident. It follows M4,
independently of M5; sharding is M6.

**Entry:** M4 exit.

**Scope:**

- [ ] **Setup and discovery** (D-038, D-039, [cluster design](cluster-design.md)):
      inventory, trusted SSH, the mDNS window, layout classification,
      bounded dedicated-QSFP subnet scans, one-time enrollment and path
      re-detection for enrolled members.
- [ ] **Configuration and trust:** the shared and node-local v2 documents,
      the private CA with TLS 1.3 mutual authentication, epochs, session
      fencing and restart reconciliation, and protocol v1 with its bounded
      state, schema tests and exact per-message field catalog.
- [ ] **Conductor** (D-037): placement, affinity routing, single-attempt
      dispatch, credit-based streaming, health states and authoritative
      per-node admission; stale or aggregate reports never admit.
- [ ] **Availability** (D-041, D-046): the per-model `endpoints` shape.
- [ ] **One import per cluster** (D-054): peer replication of verified
      prepared artifacts over the cluster link, archive to and install from
      the long-term store, and the explicit archive-or-delete choice when a
      node lacks space.
- [ ] The package gains its rdma-core dependencies when M4a links them
      (D-063).
- [ ] Decide whether a worker node serves its own loopback management
      listener.

**Exit criteria:**

- The [required validation](cluster-design.md#required-validation-and-handoff)
  challenge conditions and the conductor's fake-transport and Spark
  scenarios ([architecture](architecture.md#conductor-ownership-and-admission))
  pass.
- An unmodified standard client completes A→B→A through one endpoint, with B
  placed on the other node while A stays resident. Each node enforces its
  full local budget and compatible state reuse, and models run concurrently
  when placement permits.
- Stale capacity reports, node loss and cancellation end in bounded failure
  or unwind, never in unsafe admission or silent replay of a started
  stream.
- Remote access requires authentication and transport protection (D-014,
  D-065). A replicated or archived artifact is published only after
  verification against an identity held outside its source.

## M5 — Demand-paged MoE and the first daily drivers  `pending`

Goal: exact demand-paged routed-expert execution on the named Gemma 4
26B-A4B and Ornith 1.5 35B-A3B pair, and those two models usable day to day
through the named clients, reasoning and constrained output included (owner,
2026-09-23).

**Entry:** M4 exit; M4a is independent. The pair's checkpoints and
representations are pinned, and their GGML source closures, tokenizers and
chat templates are selected and audited (D-013, D-057, D-067). Approved
before measurement: the performance protocols; a resident-performance bound
against the pinned llama.cpp reference (prefill, time to first token and
decode inter-token p50/p95/p99, matched and normal views); the named
budgets, each with its loading policy, including at least one per model at
which its routed experts do not all fit beside its other extents, state and
headroom, so selected experts miss during both prefill and decode; and the
sustained-use schedule and duration.

**Scope:**

- [ ] **Routing boundary** (D-008, [architecture](architecture.md#routing-boundary-for-moe-7)):
      selected-expert leases, asynchronous misses and resumable tasks,
      brought up on a synthetic or tiny MoE before the named pair
      (features.md). Envelopes cover the worst-case union of every allowed
      closure and assume no fixed number of positions per phase (D-068).
- [ ] **Loading policies:** eager active-model loading that keeps inactive
      extents (the feasibility study's recommended first policy) and routed
      demand paging, both selectable, compared at the named budgets.
- [ ] **Expert layout:** the pointer-table or uniform-stride dispatch
      decision and expert compaction from the M5 GGML proof, and the MoE
      mapping in v0 artifacts.
- [ ] **Shapes that come with these models:** hybrid sliding-window and
      global attention (Gemma 4) and recurrent or linear-attention layers
      (Ornith), with their state adapters and explicit restore coverage
      (RE-004, RE-007). M4's retention matrix extends to them.
- [ ] **Traces:** native routing-trace capture and policy replay, checked
      against the M0 reference experiment; captured traces stay outside Git
      and replay by verified hash in the gate.
- [ ] **Reasoning** (D-043, D-046, D-047): protocol-specific reasoning,
      final and tool fields and streaming, model-supported thinking
      controls, signed blocks, `reasoning` and `reasoning_details`, and
      cached-token usage from real prefix reuse.
- [ ] **Constrained output** (D-043): `response_format` JSON object and
      schema, strict tool arguments and vLLM's `structured_outputs.json`,
      over a documented schema subset with explicit rejection of the rest.
- [ ] **Sustained use:** a bounded multi-hour agent/subagent session on the
      pair with repeated switches, branches, cancellations (during I/O and
      while paused included), expiry by capacity and by test-shortened idle
      caps, and spill and restore.
- [ ] EXL3 MoE only if claimed, with its own routed-expert closure, kernel,
      quality and performance baselines (D-052).
- [ ] Assess artifact compatibility guarantees with M4's dense and M5's
      MoE evidence, in a separate decision (D-018).

**Exit criteria:**

- Acceptance exercises real demand misses. At each miss-forcing budget,
  catalog state and events record nonzero selected-expert misses in prefill
  and in decode, with their count, bytes and wait time and the phase and
  layer at which each suspended. There, a measured window without misses is
  inconclusive for the paging gates, and the next two criteria are judged
  on runs with misses.
- No unselected expert loads beyond declared metadata and read-ahead, and
  no expert is substituted or its contribution dropped, verified from
  catalog state and events.
- Numerics stay correct against the pinned references after eviction and
  restoration and across within-step misses, and the M5 rows of D-050's
  matrix pass with real routes.
- D-036's generation limits hold on the pair: at most 10% added generation
  time, continuation time to first token included, and at most 20 ms p95 /
  100 ms p99 added token gaps against a resident control with matched state
  provenance, at every named budget, the miss-forcing ones included. Pause
  gaps are reported separately (D-069). M4's switching floor still holds.
- Resident prefill, time to first token and decode inter-token p50/p95/p99
  on both models meet the approved bound against the pinned llama.cpp
  reference, or the owner approves an explicit tradeoff, as D-052 requires
  for EXL3. Paging limits compare jitLLM with itself, so they cannot stand
  in for this.
- The sustained-use run ends with every completed, cancelled and paused
  request retired and no lease, grant, task or I/O outstanding. Memory, the
  catalog and retained-entry metadata, task and job records and queue
  depths stay within their bounds and level off instead of growing with
  elapsed time; task and job records and queue depths return to their idle
  levels after each switch cycle. Spill stays within `S_spill` and the write
  budget, and the memory breakdown reconciles with OS counters at the end.
- Measured and reported: the routing boundary's resident-hit overhead, each
  routed phase kind's bound against its observed peak, and whether idle
  retained state starves weight residency.
- Named clients complete reasoning and tool round trips on both models,
  including acceptance case 7's reasoning checks, and constrained-output
  requests pass their pinned compatibility fixtures. Any OpenRouter-mode
  claim rests on D-046's pinned-client run. The support matrix records the
  client versions.

## M6 — Sharded model execution (two Sparks)  `pending`

Goal: a flagship model sharded across both Sparks, correct under asymmetric
pressure, cancellation and controlled failure. Placement-only use already
works at M4a.

**Entry:** M4a and M5 exits. By entry: model-parallel artifact partitioning
is decided ([artifact-format.md](artifact-format.md#deliberately-open)); the
flagship checkpoint and its validated parallelism recipe are named (the
MiMo-V2.6-Flash-RL TP=2/EP=2 reference and the MiaAI-Lab recipes are the
candidates); and the two-node admission design chooses between
prepare/commit and the deferred mirrored-ledger shortcut (features.md).

**Scope:**

- [ ] Explicit sharding with conductor-issued distributed phase IDs,
      porting the recipe's parallelism first (TP, PP and EP are different
      plans).
- [ ] Coordinated admission: node-issued reservations and every rank ready
      before commit, with prepare failures unwound; an unknown completion
      never frees another rank's buffers.
- [ ] A separately budgeted, stable communication-buffer pool that honors
      NCCL's registration and threading contracts; ordered collective
      submission and completion fences.
- [ ] Per-step collective latency measured before bandwidth.
- [ ] EXL3 sharded cases only if claimed (D-052).

**Exit criteria:**

- Both ranks stay correct against the pinned reference under asymmetric
  pressure, cancellation and controlled failure, including partial
  preparation, a lost commit, mismatched rank generations and node loss
  during a collective. No timeout is treated as proof of reclaimed memory.
- Sharded performance is reported against the recipe's reference deployment
  in both views.

## M7 — Performance and new decoding modes  `pending`

Goal: meet D-036's benefit target on a library larger than memory, and
execute the speculative and block-diffusion shapes designed since M0
(D-068).

**Entry:** M5 exit. M7 may start before M6 exits, but MiMo's stored MTP
layers run only on M6's sharded execution, so that work waits for M6 and M7
exits after it. Before M7 planning, the bounded DiffusionGemma reference
study measures the per-step expert closures of wide phases, and untriggered
deferrals are reviewed ([features.md](features.md) and the table below).
Before the new modes execute, manifest references to another artifact by
ID, with shared resources counted once, are decided as a format change, and
numerical and statistical bounds and trace protocols for speculative and
diffusion decoding are declared. The named configurations, including the
over-memory library, and the partial-retention benefit workload are pinned
before acceptance runs (D-036).

**Scope:**

- [ ] **Larger-than-memory library:** DeepSeek V4 Flash with Qwen3.8 Flash
      Next on one node is the canonical pair (D-036). It brings compressed
      attention with an indexer, Qwen3.8's sparse n-gram rows and its
      linear-attention layers. If the pair cannot be validated, name another
      whose prepared weights exceed physical memory rather than pass M7 on
      the small pair alone.
- [ ] **Speculative decoding** (D-068): stored MTP layers (Ornith and
      Qwen3.8; MiMo's once it runs sharded under M6) and Gemma 4 companion
      drafters.
- [ ] **Block diffusion** (D-068): DiffusionGemma-26B-A4B.
- [ ] **Execution speed:** selective CUDA graphs behind a relocation proof,
      further kernels and plans (D-053), and target-assisted import tuning
      as an explicit, separately keyed mode.
- [ ] **Victim policy:** compare global LRU, frequency/recency and the
      cost-aware heuristic on identical recorded traces, with the confirmed
      hysteresis, minimum-residency and reload-cost refinements
      ([baseline](architecture.md#victim-selection-initial-baseline)) and
      per-model statistics so no model monopolizes reclaimable bytes. A
      candidate replaces the baseline, and the cost-aware heuristic becomes
      the default, only if replay shows fewer miss bytes and reloads.
- [ ] Deferred optimizations whose triggers have fired (prefetch, residency
      warm-start and the others in features.md), each against its
      demand-only control.

**Exit criteria:**

- D-036's benefit target: at least 25% lower median return-switch latency
  than jitLLM's own whole-model control, with identical state handling at
  the same budget, on the agreed partial-retention workload, with at least
  one named library exceeding physical memory, while the switching floor
  and generation limits still hold.
- Measured and reported on the over-memory library: retention under
  physical pressure and whether idle retained state starves weight
  residency (D-055).
- The speculative verify path's teacher-forced logits match plain decoding
  within declared bounds, with top-1 agreement reported and free-running
  divergence reported at its first position (RE-008).
- Speculative sampling preserves the target distribution. On recorded target
  and draft distributions, acceptance, rejection and residual resampling
  match a reference implementation of the rule exactly under fixed seeds;
  sampled outputs match plain sampling's distribution within declared
  statistical bounds, at the supported sampling settings.
- Rollback leaves exactly the accepted prefix. After rejected drafts,
  teacher-forced continuation matches a control that drafted only the
  accepted tokens, bit-identical where the plan is the same at both verify
  widths and otherwise within the declared bounds. No rejected position's
  KV, recurrent update or drafter state survives, none enters a retained
  entry (D-055, D-068), cancelling between draft and verify retires all
  draft work, and a pause there commits none of it before verify.
- Diffusion stays within declared bounds of its reference, compared step by
  step under a fixed seed. A cancelled canvas never commits, and a paused one
  commits nothing until it resumes and finishes.
- Matched and normal reference comparisons are reported, with no numerical
  or lifetime regression in any supported configuration.

## M8 — Product and first release  `pending`

Goal: the remaining confirmed product scope, and a first tagged 0.x release
that a stranger can install from the project's package repository and run
(owner, 2026-09-23).

**Entry:** M7 exit. The owner may pull an item forward once its dependencies
exist (for example the Ollama profile or tokenization endpoints after M3, or
the dashboard after M4's management controls); the release itself waits for
every earlier exit. Each model added for embeddings, reranking or file
inputs is named at entry with its pinned reference engine and numerical
bounds, declared before native evaluation.

**Scope:**

- [ ] **Dashboard** (D-064): a separate service that calls the management
      API from its own server side, including setting the Hugging Face
      token.
- [ ] **MCP management adapter** (D-042): a separate process exposing
      discovery, status and explicitly authorized actions.
- [ ] **Application permissions** (D-042): per-application credentials and
      scopes for inference, read-only status and model administration.
- [ ] **Session and hint extensions** (D-022, D-046): the optional session
      ID and release, client warm hints, and D-046's `session_id`, `user`
      and `metadata` hints. M3's front door already accepts them as
      advisory preferences, and M4 records `session_id` for release lookup.
- [ ] **Ollama subset** (D-041, D-045): listing, details, chat and
      generation with the `keep_alive` mapping, tested with a named
      Ollama-native client including its load-time bound.
- [ ] **Tokenization and diagnostics** (D-043, D-044): vLLM-compatible
      tokenize, detokenize, tokenizer information and prompt rendering; raw
      Completions with standard log-probabilities and bounded token
      diagnostics.
- [ ] **Pooled outputs** (D-042, D-044): embeddings and reranking on
      validated models named at entry.
- [ ] **File inputs** (D-042): text resources, images (starting with Gemma
      4's vision encoder) and audio files, on models validated for them,
      with bounded preprocessing.
- [ ] **Packaging** (D-027): the signed arm64 apt repository and its signing
      keys, a separate component for optional copyleft modules, and
      drain-before-restart upgrades.
- [ ] **Release readiness** (D-061, D-062): the release checklist, the
      published support matrix, notices and source obligations for every
      shipped profile, and user documentation.

**Exit criteria:**

- Each new route passes its pinned compatibility fixtures, including
  negative and interrupted-stream cases, with the unmodified clients its
  decision names; unsupported features fail explicitly.
- Every new model and output path meets numerical acceptance against its
  pinned reference within the declared bounds before any support claim:
  embedding vectors with the model's pooling and normalization; rerank
  scores and orderings; encoder outputs and the language model's
  teacher-forced logits on image and audio prompts, with preprocessing
  matched to the reference's pinned implementation. Raw Completions
  log-probabilities agree with the validated teacher-forced logits, and
  tokenize and render output matches inference byte for byte and ID for ID.
  Generated text alone is not evidence.
- On a fresh Spark, installing from the apt repository and following only
  the checked-in docs reaches a running supported model (vision.md).
- The default and copyleft-disabled builds meet D-017 with a complete,
  audited closure; builds with optional modules enabled ship matching
  notices and source.
- The release commit passes `check`, `check:full` from a fresh clone and
  `check:spark` (D-061), and the owner tags the first 0.x release.

## Deferred delivery and proposals

Confirmed scope stays confirmed when its implementation is deferred. A
candidate stays unapproved until revisited; reaching a trigger is a reason
to evaluate it. Deferred optimization triggers (predictive prefetch,
dependency-group scoring, optimistic MoE) live in features.md.

| Item | Earliest work / revisit trigger | Scope |
| --- | --- | --- |
| Load/temperature-aware GPU operating policy | Earliest M7 evaluation, or earlier diagnosis if reproducible throttling or unexplained shutdowns occur | [Proposed telemetry and optional adaptive clock ceiling](features.md#load--and-temperature-aware-operating-policy-proposal); measure stock/fixed/adaptive policies first, no assumed fault or automatic host changes |
| Ollama registry and other management compatibility | After the basic subset and relevant native management operation, when a named client needs them | Deferred D-041 candidate; lifecycle mapping needs separate proof |
| Regex/grammar constrained output | After validated JSON/schema support, when a concrete client requires it | D-043 deferral; no automatic milestone delivery |
| LoRA adapters | Earliest M7 planning after validated base-model execution, when a concrete adapter workload needs them | D-044 deferral; no automatic delivery |
| Classification/reward/generic pooling APIs | Earliest M7 planning after validated base-model execution, when a concrete model/task workload needs them | D-044 deferral; not implied by embedding/reranking support |
| Live audio/video input | Earliest M7 planning after initial file-input evidence, when a concrete workload establishes streaming/synchronization requirements | Deferred D-042 candidate; not an automatic M7 deliverable |
| Batch/background inference jobs | Earliest M7 planning after validated request scheduling, when a concrete workload justifies scheduling/storage needs | Deferred D-042 candidate; ordinary background request priority is already confirmed |
| Automatic membership changes | After M4a, when configured enrollment and explicit restart cannot reasonably serve membership churn | Candidate mechanism under D-023/D-038; bootstrap discovery and path refresh for enrolled nodes are already M4a scope |
| Conductor election | After M4a, when conductor failover becomes an explicit requirement; first define fencing and in-flight request handling | Candidate mechanism under D-023; one configured conductor initially |
| Automatic replica placement and balancing | After M4a, when measured overlapping demand on a small model causes waiting while another node has sufficient headroom | Confirmed D-023 scope with deferred delivery; preserve affinity and include duplicated weights/state in budgets |

Review untriggered items during M7 and M8 planning; they do not
automatically enter either milestone's scope or block earlier milestone exits.
