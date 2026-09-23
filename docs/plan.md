# Plan

**This is a living document.** Milestones will be re-scoped, re-ordered, split,
or added as planning conversations and findings come in. That churn is
expected; what is *not* allowed is silent change. Update scope and progress
here as work lands. Only consequential choices that change a load-bearing
constraint or are expensive to reverse get a decision-log entry; routine
scope, ordering, and implementation changes do not (AGENTS.md rule 1).

Check a box only when the item is done and verified; partially done items stay
unchecked, optionally with a note.

**Status legend:** `pending` · `in progress` · `done` · `parked`

## M0 — Plan the plan  `in progress`

Goal: turn the design brief ([ideation.md](ideation.md)) into a settled
vision, feature matrix, architecture, and milestone ladder — through planning
conversations with the project owner plus targeted spikes where a decision
needs evidence from the real hardware.

- [x] Repo scaffolding for the AI-directed workflow (this scaffold,
      2026-09-20).
- [x] Feature triage: walk [features.md](features.md) with the owner; confirm,
      reject, or defer proposals (2026-09-21: 46 proposed or open rows
      triaged; 30 confirmed, 13 deferred with a reason, trigger, and
      earliest milestone, 2 rejected, 1 still open. Load-bearing calls
      recorded as D-028 GGML-first substrate with build-time optional
      backends, D-029 contribution and compliance conventions, D-030
      Claude Code and the Anthropic Messages format). A deferral does not
      become approval when its trigger fires.
- [x] Confirm the original-code license (2026-09-20: Apache-2.0 accepted,
      D-003; dependency categories and tiers clarified in D-017, superseding
      D-015; process weight for an externally consumed project recorded,
      D-016).
- [x] Decide NOTICE and SPDX-header conventions and the contribution policy
      (2026-09-21: REUSE copyright/license metadata plus an embedded-header
      check in CI and a NOTICE file from M1, SBOM with packaging, external
      PRs accepted under DCO; D-029).
- [x] Inventory the environments without changing drivers or security
      settings (2026-09-20, read-only, no sudo): workstation baseline and both
      Sparks recorded in
      [architecture.md](architecture.md#target-nodes-dgx-sparks). The
      interconnect half is the separate cabling task below.
- [x] Spike — **toolchain smoke** (2026-09-21; open question 6,
      D-032): Clang C++23 native and AArch64 cross builds passed;
      NVCC 13.4.92 (Toolkit 13.4.2) with Clang 22.1.8 produced
      C++23 `sm_121` CUDA objects that ran on `spark`, as did the native
      Spark fallback. C++23 host/device
      feature checks and GPU results passed on driver 580.178.04 with
      PTX JIT disabled; the older 13.0 dialect limit is RE-001.
      Exact compiler, library, CUDA component, and target-sysroot pins,
      hashes, commands, and limits are in the
      [smoke report](experiments/toolchain-smoke/README.md).
      Matching compiler-rt packages and Clang ASan/UBSan CPU checks on the
      workstation and Spark were added 2026-09-22; M1 provisioning must
      include those runtimes. M1 still owns declarative provisioning and
      CMake presets.
- [x] Spike — **VMM microbench** (2026-09-21; open question 1, D-033):
      three runs on `spark` measured 2 MiB minimum/recommended granularity,
      allocation/map/access/unmap/release costs across 2–128 MiB extents,
      and costs with independent background kernels. A retained, unmapped
      1 GiB pool kept its physical footprint and contents; releasing its
      handles returned capacity. Initial policy: 2 MiB independent extents,
      completion-safe backing handoff, no standing unused-handle cache.
      [Aggregate report and harness](experiments/vmm-microbench/README.md);
      numbers in architecture.md. I/O and model-load optimization remain
      separate measurements, not conclusions of this allocation experiment.
- [x] Spike — **I/O path comparison** (2026-09-21; open question 2, D-034):
      compared buffered/direct files, pinned staging, cuFile compatibility,
      native asynchronous I/O, and GPU in-place access on `spark`.
      Direct regular files into host VMM reached about 15 GB/s, sustained
      14.962 GB/s for 180 seconds, and matched device-VMM GPU scan speed.
      Concurrent compute, 100 GiB held-memory pressure, cache reclamation,
      sparse/small reads, verified write bursts, and DMA-bounce tracing are
      in the [aggregate report and harness](experiments/io-path/README.md).
      Initial budget: two to four 2 MiB destination slots, no extra Spark
      staging copy. Raw-device alternatives need a dedicated unmounted SSD;
      no raw performance claim is made. Controller interrupt coalescing was
      tested separately with its original setting restored. Actual GGML
      behavior and full registration/reclaim lifetimes remain M2 proof work;
      mixed read/write and sustained-write/endurance policy remain M4 inputs.
- [x] Pick the reference engine for the feasibility spike: llama.cpp with
      MoE GGUFs and a small router-logging patch (decided 2026-09-20; the
      lightest install and the owner's preference).
- [x] Install llama.cpp on a Spark in a container so the host baseline in
      architecture.md stays clean (2026-09-21): digest-pinned ARM64 CUDA
      image, source `b29c606e2`, and hash-verified Gemma 4 UD-Q4_K_M execute
      on `spark`. GPU inference and cross-process slot restore passed;
      Gemma requires `--swa-full` for the tested reuse path (RE-004), with
      its additional memory cost recorded. Exact identities, expert/state
      accounting, reproducible harness, and limits are in the
      [reference setup report](experiments/reference-setup/README.md).
      No host toolchain/driver/security changes or HF key were needed.
      Ornith (MIT, confirmed by owner) is now validated in the switching
      experiment below; Qwen remains an unexecuted candidate. Installation
      alone did not complete switching or route-trace experiments.
      Candidate trace models, owner-provided, with facts from their model
      cards as read on 2026-09-20 (repository access/revisions/licenses
      rechecked 2026-09-21; Gemma's structure and memory accounting were
      verified in setup, Ornith's in the subsequent A→B→A experiment):
      - [unsloth Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF):
        125B core parameters, 6B active, 512 experts, top-10 plus 1 shared, expert
        intermediate dim 640, plus a 51B n-gram embedding table and a 4B MTP
        head; Gated DeltaNet plus sparse MQA attention; 262K context. GGUFs
        run from about 72 GB
        (1-bit) through 82 to 90 GB (3-bit) to 111 GB (Q4_K_XL). On a
        121 GiB node that is a single-model budget sweep: 3-bit fits, Q4 is
        borderline, Q5 and up exceed the node, so it covers the
        forces-paging axis by itself. Its n-gram table is the first concrete
        sparse-lookup component: row requests resolve to containing extents
        (D-035), and its MTP head matters
        for matched-configuration comparisons (D-021).
      - [unsloth gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF):
        25.2B total, 3.8B active, 128 experts, top-8 plus 1 shared; hybrid
        1024-token sliding-window and global attention; 256K context.
        Q4_K_M about 17 GB. The owner has experience with Gemma.
        Fits easily: the subagent and switch-latency case, with mixed KV
        lifetimes for the spill study.
      - [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF):
        36B total, about 3B active on the card, `qwen35moe` architecture;
        the selected GGUF verifies 256 experts, top-8 plus a shared FFN,
        40 primary layers and one stored MTP layer; 262K context.
        Q4_K_M about 22 GB. A second small model for replica and
        multi-model-switching traces.
      Gaps these three leave, to fill only if cheap: a few-large-experts,
      low-top-k model (Mixtral-style) for the high per-miss-cost end, and a
      plain full-KV GQA model where KV grows linearly and spill cost is
      highest. Record for each model used: expert count, top-k, shared
      experts, expert bytes at the chosen quant, total bytes, KV or recurrent
      state bytes per token, and headroom on one node. The owner's local
      Ollama blobs are plain GGUFs usable for workstation-side dry runs on
      the RTX 3080 Ti where they fit.
- [x] **Image-generation reference experiment — Qwen-Image-2.1**
      (bounded reference study complete, 2026-09-22): the
      [report, pins and harness](experiments/image-reference/README.md)
      establish a standalone BF16 Diffusers reference on `spark`, with
      checkpoint/dependency provenance and research-license boundaries.
      Twelve image cases cover 512–2048 pixel resolutions, 4/40 steps,
      editing, repeated outputs, prefix reuse, completed-step cancellation
      and phase-boundary backing release. The full pipeline has 32.44 GB of
      parameter storage in execution, not just the advertised 7B generation
      component; the 2048²/40-step request takes 252.796 s and peaks at
      56.521 GiB of CUDA allocations. Exact pixels survive cancellation
      recovery and discarding encoder/denoiser backing after their completed
      phases; all weights were initially loaded, so lower-budget admission
      and subsequent image-state restoration remain unproven.
      Short text → image → text cycles return in 0.044 s with text resident
      and 3.103 s after state restoration, processing only 12 new tokens.
      Full recomputation takes 4.155 s in a separate diagnostic and matches
      its fresh-context control, **but differs from resident text**; the
      original failed comparison is preserved (RE-008). These are single
      observations, not latency distributions or measured jitLLM speedups.
      Image loading is slow and variable; outward harness timings include
      checksum verification. No cold-cache, forced-pressure or
      larger-than-memory image result is claimed. Native GGML image execution,
      pending-work cancellation, output API scope and delivery milestones
      remain separate follow-up work; existing image-input API scope implies
      no native image-generation capability.
- [x] **Additional reference candidates** (owner-added; bounded references
      complete 2026-09-22; [candidate matrix](experiments/model-candidates.md)).
      **Qwen-Image-2.1 GGUF** ([report](experiments/image-gguf/README.md)):
      pinned stable-diffusion.cpp (GGML) runs the full pipeline from Q4_K_M,
      Q8_0 and a BF16 control on `spark`; repackaged components were
      reconciled tensor by tensor with the BF16 baseline. Repeats, cold/warm,
      a second node, phase-released parameters and budgeted disk-backed
      denoising (against a same-VAE-tiling control) give exact pixels;
      quantization shrinks denoiser parameters
      (4.29 vs 13.25 GiB) but not step time, and this runner is 2–3× slower per
      step than diffusers BF16. Its 2048² VAE decode buffer is 38.3 GiB;
      upstream GGML matches the fork through 1024² but aborts there on a
      32-bit stride assert (RE-011). Releasing encoder/VAE parameters outside
      their phases cut sampled host memory 30.4→17.2 GiB at 1024²/40; a 3 GiB
      budget ran the request in about 5 GiB at 2.6× the time. Pixel agreement
      is not a quality score. **MiMo-V2.6-Flash-RL**
      ([report](experiments/mimo-reference/README.md)): TP=2/EP=2 on both
      Sparks with packed MXFP4 experts, 81.9–83.8 GiB weights per rank, both
      RoCE HCAs, correct greedy smoke, about 2,360 prefill tokens/s, prefix-hit
      TTFT 8.68→0.36 s at 20K tokens, and 18.5 decode tokens/s (54 ms median,
      62 ms p99 gaps) without speculation or graphs. Startup needed audited
      config remote code and `torchcodec` (RE-012). One boot and single
      requests; no switching, state restore, pressure, concurrency or
      multimodal cases. Neither establishes native support, replaces the
      canonical DeepSeek/Qwen switching pair, or blocks M1; smaller MiMo
      quants remain a revisit trigger.
- [x] **Reference A→B→A experiment** (2026-09-21, D-025): 27 verified cycles
      on `spark`, Gemma 4 UD-Q4_K_M → Ornith 1.5 Q4_K_M → Gemma, with an
      18,339-token continuation. Three repeats of nine cases distinguish
      resident, warm/cold file cache, restore/recompute, 80 GiB locked
      physical pressure, and default-SWA/cache optimizations. Under the
      matched forced-displacement budget, median first-token waits are
      21.232 s outward and 18.304 s back with retained state; recomputing
      A's prompt takes 25.236 s on return. Warm restore returns in 4.062 s;
      live residency in 0.089 s. Native LRU, durable save/restore, actual
      block I/O, state bytes, memory and bounded observed swap are verified.
      A reuses 18,297 tokens and processes 42; all outputs match, and a
      separate early/late notebook recall probe passes. B's recurrent-state
      continuation also passes. Default SWA still re-prefills after restore
      (RE-004); decode speeds differ by state lifecycle and are reported.
      Exact pins, external trace identity, measured ranges, tooling and
      limitations are in the [aggregate report](experiments/reference-aba/README.md).
      Normal optimization probes use the same forced one-model policy;
      normal concurrent placement remains unvalidated. The later full feasibility
      study includes a separate large-model observation and finds that Gemma
      sequence snapshots can lack SWA coverage after prompt rollback (RE-007).
      The short matching restored output above is historical evidence, not a
      general continuation-correctness proof; the validated recompute arm is
      the usable correctness floor. M4 can replay this exact input trace.
      Comparator datapoint: Athena's Engine reports a 46 s measured switch
      between DeepSeek V4 Flash and Qwen3.8 Flash Next on one GB10 and a
      2.1 s restore of a 141k-token conversation from disk (creator-reported;
      see [architecture.md](architecture.md#comparator-athenas-engine-closed-source-creator-reported)).
      Optionally install it on a Spark as a second comparator for the
      normal-reference view if its terms still allow personal use; never cite
      its numbers in place of our own measurement.
      The author's X thread adds that the 46 s includes checkpointing the
      session and that the pair does not both fit in 128 GB at those bit
      depths, which makes DeepSeek V4 Flash plus Qwen3.8 Flash Next the
      canonical two-large-model switching workload for the feasibility spike
      if a DeepSeek V4 Flash GGUF is available at pick time.
- [x] Spike — **paging feasibility** (bounded study complete, 2026-09-22):
      the [full study](experiments/paging-feasibility/full-study.md) extends the
      [first cut](experiments/paging-feasibility/README.md) with varied longer
      Gemma/Ornith conversations, the exact reference A→B→A trace, concurrent
      four-sequence decode, and the larger-than-memory DeepSeek/Qwen library.
      Arrival schedules model minutes-to-hours sessions; they are not hardware
      soaks and do not test expiry. Matched budget sweeps compare demand paging,
      eager active-model loading with partial inactive retention, and whole-model
      switching, including non-expert extents, live state, workspace, headroom,
      bounded spill, restore/recompute, and isolated/packed 2 MiB layouts.
      Per-layer reuse, held-out prediction, switch-back dependencies, measured
      reference waits, and explicit storage/overlap sensitivities are recorded.
      Results are offline byte/service estimates, **not measured jitLLM speedups**.
      Qwen comparisons remain conditional on captured trajectories because
      exact prediction equivalence failed; no numerical tolerance is invented.
      Gemma rollback can require state recomputation despite byte-identical
      snapshots (RE-007), and unvalidated large-model spill continuations use
      conservative recompute scenarios. The evidence supports M4 retention
      before M5 expert paging and makes restore coverage an explicit validation
      requirement; it neither proves runtime admission nor enables prefetch.
      Performance acceptance targets are recorded in D-036 below.
- [x] Agree switching-benefit and generation-stall criteria (2026-09-22,
      D-036): M4 onward must meet the fastest correct full-swap reference
      arm at median/p95 in both directions; M5 onward permits at most 10%
      added generation time, continuation time to first token included, and
      20 ms p95 / 100 ms p99 added token gaps against a resident control
      with matched state provenance; M7 requires at least 25% lower median
      return-switch latency than jitLLM's own whole-model control on an
      agreed partial-retention workload, with at least one named library
      that exceeds physical memory. Targets apply to named supported
      configurations; correctness is mandatory and inconclusive comparisons
      do not pass. Pin workloads, trial counts, and measurement methods
      before acceptance runs; these are targets, not measured implementation
      results.
- [x] Complete the Spark-to-Spark direct-link baseline. The owner configured
      the `sparky` DAC cluster on 2026-09-21; the
      [baseline](experiments/interconnect/README.md) passed 78 host-buffer
      test pairs and 27 pinned NCCL runs. The two PCIe interfaces share one
      physical 200 Gb/s port: combined writes measured 184.76 Gb/s in either
      direction, reads 150.10 Gb/s with default queue settings. Large NCCL
      SendRecv/AllReduce reached 22.35/22.20 GB/s; supported result checks
      passed. Channel logs and counters verify actual HCA use; source and
      allocation logs establish GPU access to mapped host communication
      buffers, not GPUDirect RDMA. Message-size sweeps, ranges, pins and
      limitations are recorded with the report and in architecture.md.
      Sharded-model, asymmetric-pressure and failure tests remain M6 work.
- [x] Inventory MiaAI-Lab reference licensing versus MIT ExLlamaV3 upstream
      (2026-09-22): [pinned file/group inventory](licensing.md) separates
      AGPL defaults, explicit MIT/Apache notices, historical MIT grants and
      unresolved downstream modifications. Nine vendored ExLlamaV3 headers
      matched pinned upstream bytes and hashes. AGPL network-source duties
      and adoption blockers are recorded; no code or module boundary is
      approved by this inventory.
- [x] Decide conductor location and admission/placement ownership (2026-09-22,
      D-037): conductor inside the designated node runtime, advisory per-node
      cluster view and placements, authoritative local admission, bounded
      streaming through the front door, and no replay of uncertain attempts.
      [Architecture](architecture.md#conductor-ownership-and-admission) records
      lifecycle/failure requirements and future validation cases; D-038 below
      supplies the configuration and wire/fencing design.
- [x] Define the initial cluster (2026-09-22, D-038/D-039):
      [design](cluster-design.md) settles experimental TOML v2, one enrolled
      conductor/membership, mutual authentication, session fencing and restart
      reconciliation, bounded queues/records/timeouts, node health/admission
      and independent retained-state affinity. Owner-requested initial
      interface/QSFP detection proposes the layout; read-only probes on both
      Sparks verified adapter/physical-port grouping for the two host paths
      sharing one cable. D-039 adds single/pair/triangle/switched-N layout
      classification and bounded dedicated-QSFP subnet scans. Setup discovery
      and enrolled-member path refresh
      are M4a scope; automatic membership changes, election and replicas
      remain deferred. Implementation and adversarial execution are still owed.
- [x] Verify the named-client API requirements against current official docs
      (2026-09-22, D-040): the [baseline](client-api-baseline.md) specifies
      Chat Completions, Responses (required by Codex), Messages and token
      counting, model listing, JSON/SSE and tool round trips. Cursor's exact
      custom-endpoint behavior is not fully documented and remains an explicit
      M3 validation gap, as does executed compatibility for every client.
      Local-only defaults remain intact; no serving implementation is claimed.
- [x] Triage the owner-requested [API capability assessment](api-capabilities.md)
      (2026-09-22, D-041/D-042): confirmed Ollama subset, discovery (M3/M4a),
      continuation close (M4), download/warm jobs, staged text-resource/image/
      audio-file inputs, MCP management, single-owner sharing controls and
      later embeddings. Ollama registry/management compatibility, live
      audio/video and batch/background jobs have explicit deferral triggers.
      Exact schemas and unassigned delivery milestones remain planning work;
      no implementation support is claimed.
- [x] Triage the follow-up [vLLM API comparison](vllm-api-assessment.md)
      (2026-09-22, D-043/D-044): direct compatibility tested with unmodified
      clients; tokenization/rendering, JSON/schema/strict tools, reasoning,
      reranking, metrics/health/load and raw Completions/token diagnostics
      confirmed. Regex/grammar, LoRA and classification/reward/pooling remain
      workload-driven deferrals; generic worker RPC, training and split-serving
      deployment controls are excluded from the client baseline. Delivery
      milestones and precise compatibility profiles remain planning work.
- [x] Review the API contract documents against live client docs and fix the
      gaps (2026-09-22, D-045): Claude Code's Anthropic-shape model discovery,
      default-on alias fields, error-wording recovery, request-class and
      context-compacted signals and the attribution block; Codex timeouts,
      retries and returned reasoning items; front-door listener, auth and
      CORS defaults; admission status codes and the keepalive rule;
      `keep_alive`, alias echo and extension carriage. Execution evidence is
      still owed in M3.
- [x] Triage the [OpenRouter assessment](openrouter-api-assessment.md)
      (2026-09-22, D-046): model-metadata fields ride with M3 discovery and
      the per-model endpoints shape with M4a availability; the `reasoning`
      object, `reasoning_details` and cached-token usage join D-043's
      Chat Completions contract; `session_id`/`user`/`metadata` are hints;
      the `models` array is reserved as the spelling should fallback ever be
      accepted, which it is not; plugins, transforms, auto-router, pricing,
      credits and generation stats are excluded. No fourth protocol;
      execution evidence still owed. D-047 corrects reasoning wire fields and
      signed-block format, rejects unsupported Responses storage, and scopes
      SSE keepalives separately from non-streaming JSON/deadline handling;
      the baseline carries their acceptance cases.
- [x] Decide the async/task and completion model (2026-09-22, D-048):
      [explicit native task states](async-model.md), one node-local
      scheduler/catalog writer, bounded provider services and operation-owned
      completion/cleanup storage. Cancellation never substitutes for resource
      retirement. The [CPU-only prototype](experiments/async-model/README.md)
      passed on the workstation and `spark`, including 216 event schedules,
      saturation, partial submission and uncertain completion. Real threading,
      provider and GGML/VMM lifetime validation remain M2; reservation progress
      is the separate question 9 decision below.
- [x] Decide the initial reservation guarantee and progress envelopes
      (2026-09-22, D-050): [policy and adversarial cases](reservation-policy.md)
      settle guaranteed bounded requests, maximum retained-state/growth
      allowances, complete phase envelopes held through waits and unwind,
      request/response-granularity switching (never mid-request; owner
      2026-09-22) and full-envelope checks for supported concurrency. Grants remain lazy;
      opportunistic work cannot invalidate them, and spilled admitted state
      keeps its in-memory allowance. Impossible phases fail or use an already
      validated alternative. Numeric envelopes and executed progress/lifetime
      proof remain M2, with retention/concurrency in M4 and routed phases in M5.
- [x] Decide the first vertical-slice checkpoint and numerical reference
      (2026-09-22, D-051): [Qwen2.5-0.5B-Instruct official FP16 GGUF](first-slice.md),
      exact artifact/tokenizer/template identity and pinned llama.cpp CUDA
      reference, with CPU diagnostics. The [bounded reference check](experiments/first-slice/README.md)
      passes 76-token repeat/context-restore comparisons on Spark; native
      support, numeric acceptance thresholds and context-size validation
      remain M2/M3. Source-unit licenses/provenance are recorded; generated
      Unicode data needs explicit clearance before native tokenizer adoption.
- [x] Require a real EXL3 companion early (2026-09-22, D-052):
      [contract and pinned small fixtures](exl3-bringup.md) add same-model
      4.0 bpw and mixed-rate 4.5 bpw EXL3 to M2 before settling the artifact
      layout/operation contract. Preserve packed execution and side tensors;
      upstream kernel performance is an M2 gate, resident serving performance
      an M3 gate, and EXL3 switching/restore an M4 gate.
- [x] Run the **small EXL3 reference baseline** (2026-09-22): both full-hash
      D-052 quants execute on Spark in the pinned ExLlamaV3 environment;
      tokenizer/template identities, repeated-logit and in-place cache restore
      controls pass. The [report](experiments/exl3-reference/README.md) records
      four direct Model API profiles and 176 real/synthetic kernel cases,
      dispatch boundaries, tracked memory and statistical comparison rules.
      Full Generator serving controls remain M3 work. ARM host
      helpers require a bounded patch; device kernels are unchanged. Native
      numerical tolerances, complete physical-memory envelopes and reference
      cases flagged unstable remain acceptance gates, not inferred passes.
- [x] Scope the remaining **early backend integration proof**, executed alongside M2
      (2026-09-22): the [proof scope](backend-proof.md) fixes entry conditions,
      stages P0–P6, a five-rung numerical oracle ladder (reference, toolchain
      bridge, native dispatch on conventional memory, host VMM, restored) and
      the BP case matrix for backing/accounting, numerics, paging, lifetime,
      failure, performance and kernel coexistence/swapping.
      Source reading at both pins (not measurement): GGML's CUDA backend hides
      a never-shrinking, aborting scratch pool, cuBLAS workspaces, its own
      streams and a GB10 device-flag side effect, chooses kernels/fusions
      internally and has no custom operation. The owner therefore set
      **D-053**: jitLLM owns dispatch, and GGML, ExLlamaV3, later or
      jitLLM-authored kernels are swappable build-time implementations
      selected per operation and plan, several at once. GGML's operation
      launchers take a context whose pool, stream and handle jitLLM can supply,
      with build-time patches for context ownership, device initialization,
      abort paths and the `static` matrix-multiply routing.
      ExLlamaV3 device kernels separate from their ATen wrappers; autotuned
      grids, output dtypes and compile flags are part of the numerical plan.
      Its GEMV kernel cites GPL-3.0 QTIP code as its structural model: an open
      provenance gate ([licensing](licensing.md#early-exl3-companion-d-052));
      until resolved the native plan runs the GEMM kernel where upstream
      selects GEMV. Execution, thresholds and the contract remain M2 work.
- [x] Defer the retained backing comparison to M2 (owner follow-up
      2026-09-21, D-035; deferred 2026-09-23). It compares D-033's small
      independent handles with larger persistently mapped slabs, including
      1 GiB, and serial with bounded asynchronous submission for mixed
      checkpoint batches. It needs the M2 catalog, leases, storage and completion
      services on the backend proof's P4 harness, so it cannot run in M0.
      Deadline: D-033 is explicitly retained or amended before the internal
      contract is settled and M2 closes.
      The [scope](backend-proof.md#retained-backing-comparison) records the
      open prerequisites: a cross-model swap trace (none exists yet),
      synthetic extents because the dense ~0.5B fixtures are too small to
      fragment slabs realistically, and retain/amend criteria set before
      measurement. It also records that expert compaction waits for M5's
      dispatch choice.
- [x] Define the M4 A→B→A acceptance trace and the bounded retention policy
      (2026-09-22, D-055): the [retention policy](retention-policy.md)
      settles entry identity, adapter restore boundaries, immutable shared
      blocks charged once, refresh by branch, D-041 close semantics,
      capacity-driven expiry with 24-hour per-class idle caps (owner
      decision), an initial victim order, lazy digest-verified spill deleted
      at startup, and fallback/reporting. The owner named M4's workload:
      Qwen2.5-0.5B FP16 GGUF and EXL3 4.0 bpw in both orientations on a
      frozen synthetic transcript under policy-forced budgets. It has six
      jitLLM arms, including whole-model controls, fresh interleaved
      llama.cpp/ExLlamaV3 references, at least 72 repetitions per arm and
      distribution-free 97.5% bounds, plus exact outputs and logits against
      provenance-matched controls. The functional matrix covers shared
      prefixes, branches, release races, expiry, spill failures and an
      unmodified client. Capacity values are pinned at M3 exit from measured
      state sizes. The transcript, budgets and reference paths are pinned at
      M4 entry. No retention code or measurement exists yet.
- [x] Choose an experimental artifact encoding and layout ABI (open question
      5, 2026-09-22, D-056): the [v0 format](artifact-format.md) uses
      safetensors shards with explicit zero pads, strict JSON manifest/index,
      a manifest-digest artifact ID with deterministic import and one-rename
      publication, exact version/profile rejection with re-import, and
      verification at install/replication/on demand but never at page-in.
      After owner questions during the task, D-056 amends D-035: dependency
      groups (a dense layer, one expert's closure) are single 4 KiB-aligned
      file ranges paged in 2 MiB group-relative chunks, and adjacent misses
      coalesce into vectored direct reads.
      The [layout study](experiments/artifact-layout/README.md) covers seven
      real models: 4 KiB groups leave ≤0.083% disk padding versus 3.49–10.87%
      at 2 MiB. Worked examples cover dense, expert, tied (Qwen2.5's embedding
      and head stored once), small-tensor, sparse-row, tail, chunk-boundary,
      shared-chunk-lease, resident-hole and interrupted-import cases, plus
      EXL3 descriptors and closure rules, and GGML's row-padding over-read.
      Both D-051/D-052 fixtures (FP16 and EXL3) and Gemma 4 built, verified,
      paged back byte-exact with direct reads and passed the pinned upstream
      safetensors reader. The verifier was hardened until a tenth adversarial
      challenge round came back clean (104 unit tests). On `spark`, 4 KiB offsets
      cost at most 2.6% raw read throughput but win on useful bytes; SHA-256
      runs at 2.49 GB/s per core; one process could reserve 128 TiB of GPU
      VA in one range. The pinned GGML expert stride makes uniform-stride
      expert views cost large VA; a per-expert pointer table is recommended
      for M5. Model-parallel partitioning is deferred to M6 entry.
      The C++ importer/verifier are M3; compatibility guarantees stay behind
      D-018's gate.
- [x] Decide the C++ source-dependency mechanism (2026-09-23, D-057):
      [locked CMake FetchContent acquisition and curated vendoring](source-dependencies.md)
      for adapted source units, separate from D-049's SDK. Select and audit
      each profile's complete closure before fetching; configure/build uses
      verified local inputs, with no implicit downloads or implementation
      library substitution. Official tool documentation checked; M1 still
      owes implementation, concrete dependency pins, native/cross offline
      checks and proof of copyleft-disabled exclusion. Existing source
      provenance gates remain open.
- [ ] Toolchain decisions: build-system conventions (CMake presets / Ninja /
      LLD, confirmed 2026-09-21), test framework, format and lint pins, CI
      shape including the copyleft-disabled profile, REUSE lint, a separate
      embedded-header check for commentable source/docs, and an
      installable `.deb` build from M1 (D-029), versioning/changelog
      conventions for an externally consumed project (D-016), and the
      installed layout that packaging will need (FHS paths, service user,
      systemd unit; D-027, confirmed 2026-09-21), including D-054's
      storage-role paths and defaults. Record the pins and the
      layout in decisions.md. Provisioning split settled 2026-09-22 (D-049):
      a complete, persistent project SDK, declared system prerequisites and
      shared workstation/container setup. CMake 4.4.3 pinned 2026-09-23
      (D-058), with both Linux archive hashes and seven FetchContent semantic
      checks passing on workstation and Spark; CMake-driven native CPU,
      AArch64 cross CPU/CUDA and native Spark CPU/CUDA smoke also passed
      ([report](experiments/cmake-fetchcontent/README.md)). D-059 (2026-09-23)
      pins Ninja 1.13.2, GoogleTest 1.18.0 (owner's choice), the LLVM 22.1.8
      formatter/linter/language server/symbolizer and GCC 14.2 libstdc++
      headers (amending D-032; the headers were replaced by D-060's static GCC
      16.2 runtime), with candidate style, check and warning sets.
      Native, sanitizer, cross-to-Spark CTest and native Spark runs passed
      ([report](experiments/dev-tools/README.md)). The CI shape,
      versioning/changelog conventions, installed layout, implementation and
      application build validation are still owed.
- [x] Evaluate static runtime linking with GCC 16.2 (owner follow-up
      2026-09-23, D-060): GCC 16.2 built from GPG-verified source on both
      hosts. libstdc++, libgcc and cudart are linked statically, and Clang
      stays 22.1.8 (NVCC's host limit; owner declined a compiler split).
      Binaries need only glibc (`GLIBC_2.38` at most) plus the driver's
      `libcuda`, which is loaded at run time. The C++23 probe (`<flat_map>`,
      `<mdspan>`, `<print>`), NVCC, the CUDA smoke on the GB10, GoogleTest
      (native, cross over SSH, Spark) and sanitizers all passed
      ([report](experiments/gcc16-static/README.md)). NCCL must be static or
      built with `-static-libstdc++`; cuBLAS static pending its review. The
      owner confirmed that GCC's runtime exception permits the static
      linking. Release builds do not link `libstdc++exp.a`, so libbacktrace and
      its notice stay out.
- [ ] First full draft of [architecture.md](architecture.md).
- [ ] Rewrite the provisional ladder below into real milestones with exit
      criteria, carrying the deferred retained-backing comparison into M2
      and including unassigned D-041–D-044 API delivery (Ollama subset,
      warm/install jobs with D-054's archive and peer replication, file
      modalities, MCP, sharing controls, embeddings,
      compatible tokenization/rendering, constrained output, reasoning,
      reranking, metrics/health/load, raw Completions/token diagnostics and
      D-046's OpenRouter reasoning/cache spellings and hint fields).

**Exit criteria:** the owner has walked features.md and says the plan is good
enough to build from; open questions 1–7 and 9 are answered or explicitly
deferred with a reason and a milestone deadline; toolchain pins exist; M1+
milestones have scopes. Proposed optimizations may remain deferred with a
reason and trigger; they need not be accepted or rejected to exit M0. The
paging-feasibility result, measured reference switching baseline, agreed
performance criteria, and question 9's policy are required before M2, even if
deferred out of M0; feasibility sizes M4/M5 rather than gating viability
(D-021, D-025). Question 8 is resolved by
D-022; endpoint documentation verification is recorded in D-040 and D-045;
executed client compatibility remains M3 work. M0 exits on the owner's call,
not on a checklist reaching zero. Both Sparks are reachable (`spark`, `spark-b`); the toolchain, VMM, I/O,
interconnect, reference switching, and bounded paging-feasibility evidence
is recorded above. Sharded two-node reference execution remains separate
from these single-node model captures and is still future work.

## Provisional milestone ladder  `pending — to be rewritten in M0`

This is the owner's staged plan from ideation §19 translated into milestones,
ordered by risk: substrate, then the resource core, then one end-to-end model
path, then the first useful product at M4 (A→B→A with retention and conversation
state reuse), then configured placement across nodes at M4a, then MoE paging,
sharding, and further performance/product work. M4a keeps the existing M5–M7
identifiers stable and has no dependency on demand-paged MoE.
Sketch only — do not start work from these entries. They freely reference
`proposed` features.md rows; nothing here pre-empts the M0 triage. No stage
has a promised date; each should leave a usable, testable result.

- **M1 — Bootstrap.** Repository skeleton, declarative SDK setup (mise plus
  shared workstation/container provisioning into a persistent, versioned
  project SDK; D-012/D-049), C++23/Clang native build, Spark cross build, ARM/CUDA
  smoke binary running over SSH, CI with the copyleft-disabled profile, initial
  license and provenance tooling (REUSE lint, embedded-header check, NOTICE),
  a first-cut capability probe (the future `doctor` task), and an installable
  `.deb` build. *Gate:* clean host and container setup with declared OS
  prerequisites and the complete tool/runtime set for each profile, including
  formatter, linter, language server, symbolizer and sanitizer runtimes;
  project tool selection works without global compiler/environment changes
  or dependency on temporary SDK directories;
  native tests pass, including a CPU-only configuration with no CUDA toolkit
  present (D-026 guardrail); D-057's [source-dependency gates](source-dependencies.md#upgrades-packaging-and-m1-gates)
  pass, including fresh offline builds and disabled-module exclusion before
  fetching; smoke binary runs on a Spark; exact pins recorded;
  any deferred feasibility experiment, measured reference switching baseline,
  agreed performance criteria, and reservation policy are complete before M2.
- **M2 — Resource core.** Catalog, reservation/lease state machine,
  deterministic fake backend, real VMM smoke harness. *Prerequisites:*
  paging-feasibility results recorded and M4/M5 scope adjusted if warranted
  (D-021, D-025); measured reference cycle and acceptance criteria recorded;
  reservation policy and progress envelopes are recorded. *Gate:* every
  M2 row of D-050's [adversarial matrix](reservation-policy.md#worked-cases-and-implementation-gates)
  passes with the fake backend and no vendor SDK present — including
  completion, cancellation with late I/O, request-boundary switching,
  competing suspended phases, state growth, impossible phases, spill/reclaim
  saturation, fragmentation, sharing/forks, envelope-upgrade races, full
  queues during cancellation, budget reduction and unknown completion;
  repeated map/load/evict/restore checks succeed on a Spark.
  Run the early backend integration proof ([scope](backend-proof.md))
  alongside this work; it must pass
  before settling the internal contract and closing M2. The deferred
  [retained-backing comparison](backend-proof.md#retained-backing-comparison)
  must explicitly retain or amend D-033 by the same point, under criteria
  approved before measurement. The proof covers the
  small dense FP16 control **and both real EXL3 fixtures** (D-052), their
  native prefill/decode and state, with correctness before and after
  restoration. EXL3 kernel time/workspace gates against the pinned Spark
  reference must pass; a loader or FP16 conversion is insufficient.
  Full serving integration follows in M3.
- **M3 — One resident model, end to end.** Import a manageable model (small
  dense first) with the standalone artifact verifier, GGML-backed native
  execution plus the D-052 native EXL3 companion, tokenizer/state/sampling
  baseline, and the
  Chat Completions, Responses and Anthropic Messages surfaces, model listing
  and Messages token counting, plus machine-readable API/model capability
  discovery, with `/v1/models` entries in D-046's OpenRouter metadata shape,
  under the front-door contract (D-040/D-041/D-045/D-046/D-047;
  [contract and client tests](client-api-baseline.md)). *Gate:* teacher-forced and
  intermediate comparisons against a pinned reference; finite default
  context/output bounds bound every admitted API request (D-050); bounded, explainable
  memory usage; at least one named client completes a chat through the
  endpoint unmodified with each representation. EXL3 resident prefill/decode,
  time-to-first-token measurements meet the predeclared upstream parity gates,
  with memory within the declared bounds in [the contract](exl3-bringup.md);
  performance is not left
  until M7. M2's backend proof supplies the integration evidence;
  M2/M3 implementation may overlap while their gates remain explicit.
  At exit, measure both supported contexts' state bytes and pin D-055's
  retention capacity values in [the policy](retention-policy.md#bounds-and-defaults).
- **M4 — First useful product: A→B→A with partial retention.** Two small
  supported model contexts, one shared local budget, partial eviction of a
  quiescent model, bounded prefix/continuation retention (D-024, D-031), and
  basic status/diagnostics including the admission what-if query and Perfetto
  trace export. Through at least one unmodified named client, build a
  long conversation on A, request B under pressure, then resume A. D-055
  names the timed workload (Qwen2.5-0.5B FP16 and EXL3 4.0 bpw, both
  orientations) and its arms, statistics and functional matrix
  ([acceptance workload](retention-policy.md#m4-acceptance-workload)); M4
  entry pins its transcript, budgets and reference paths. *Gate:*
  only selected extents displaced; untouched data remains resident; reload
  only missing dependencies. Exercise both resident state reuse and forced
  spill/restore, including an EXL3 model context in the matrix (D-052), with
  representation-specific state identity and matched EXL3 reference controls.
  A compatible retained prefix resumes without a full
  re-prefill; process only new input and any declared cache-block tail.
  Report switch/switch-back latency distributions, bytes read/written,
  peak memory/spill use, and prompt tokens reused versus recomputed against
  D-025's measured reference cycle, D-036's median/p95 switching floor, and
  jitLLM's own whole-model control at the same budget, which M7's benefit
  target uses as its comparator. Teacher-forced numerics stay correct after
  weight/state restoration. Branching histories,
  edits, incompatible identity, expiry, and spill exhaustion yield correct
  reuse, recomputation from supplied history, or explicit errors as appropriate;
  cache expiry never destroys admitted suspended work. D-041 adds a final-turn
  flag and explicit idempotent release of one continuation, completion-safe
  and independent of shared-prefix retention. Independent
  conversations reuse a shared system prefix with isolated mutable suffixes;
  continuation release/expiry leaves eligible shared-prefix reuse intact,
  while shared-prefix hits do not refresh unrelated continuations. Exercise
  shorter-prefix fallback after continuation eviction, independent prefix
  expiry, and shared-byte accounting, with numerical reference checks (D-031).
  An all-resident control demonstrates concurrent progress without paging
  when both complete execution envelopes fit. This milestone is useful
  without MoE or sharding.
- **M4a — Configured placement across nodes.** After M4, independently of M5:
  one configured conductor and enrolled nodes, interface/QSFP bootstrap
  discovery and automatic path detection under D-038, capability/health probes,
  whole-model placement and request routing with state affinity, plus
  discoverable cluster model availability (D-041, in D-046's per-model
  endpoints shape). Run B on
  another node while A stays resident; use the existing network. *Gate:*
  the D-038/D-039 topology/discovery/configuration/authentication/fencing and bounded-state
  challenge cases pass; an unmodified standard client completes the A→B→A
  flow through one endpoint; each node enforces its full local budget and compatible state reuse; models
  run concurrently when placement permits. Stale capacity reports, node loss,
  and cancellation cause bounded failure/unwind without unsafe admission or
  silent replay of a started stream. Honor remote-access protection (D-014).
  Discovery, election, and automatic replicas follow the separate triggers below.
- **M5 — Demand-paged MoE.** Routing boundary, selected-expert leases,
  asynchronous misses, resumable tasks, native trace capture and policy replay
  checked against the early reference experiment.
  *Gate:* no unselected expert loads beyond declared metadata/read-ahead; no
  substitution; resident-hit and miss overhead measured using the comparison
  protocol and D-036's generation limits (at most 10% added generation time,
  continuation time to first token included, and 20 ms p95 / 100 ms p99
  added token gaps) on the named Gemma/Ornith pair; the canonical
  two-large-model pair is an M7 configuration because the study's estimates
  place its demand-paging stalls at the gap limit without overlap. Small-MoE
  numerics remain correct after eviction and restoration. The M4 switching
  floor continues to apply. Extend the early EXL3 proof to representative
  routed-expert closures and kernel/performance baselines before claiming
  EXL3 MoE support; dense-only evidence does not cover expert batching (D-052).
  With M4's dense evidence, assess artifact compatibility guarantees in a
  separate decision (D-018).
- **M6 — Sharded model execution (two Sparks here).** Build on M4a's
  configured cluster; add explicit sharding for the flagship, coordinated
  admission, stable communication buffers, and ordered collectives. Needs
  the direct Spark-to-Spark link baseline (DAC configured 2026-09-21) and the relevant
  single-node model/paging evidence. *Gate:* both ranks remain correct under
  asymmetric pressure, cancellation, and controlled failure; no timeout is
  treated as proof of reclaimed memory. Placement-only use already works at M4a.
- **M7 — Performance and product.** Further compatible kernels and plans
  beyond the required M2/M3 EXL3 baseline (D-052),
  prefetch, selective CUDA graphs, dashboard, optional API extensions
  (sessions, hints; D-022),
  packaging as signed apt packages for Spark with notices (D-027). *Gate:* measured results meet the agreed workload
  targets in D-036, including at least 25% lower median return-switch latency
  than jitLLM's own whole-model control on the agreed partial-retention
  workload, at least one named library exceeding physical memory, the
  switching floor, and generation limits; matched-configuration and normal
  reference-configuration comparisons are reported; no numerical or lifetime
  regression; compliant optional-backend builds.

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

Review untriggered items during M7 planning; they do not automatically enter
M7's implementation scope or block earlier milestone exits.
