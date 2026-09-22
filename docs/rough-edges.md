# Rough edges — findings log

CUDA driver, DGX Spark platform, toolchain, and library bugs, quirks,
surprising limits, performance cliffs, and missing capabilities encountered
while building jitLLM. Log the ones that burned real debugging time and will
bite again — this is a save-future-you log, not a compliance artifact.

**Before adding:** grep for the API/library involved to avoid duplicates.
**Before debugging weirdness:** check here first — it may be known.

A good entry says what environment it happened in (workstation or Spark,
driver, toolkit, compiler versions) and what was observed vs. expected;
include a reproduction when it's cheap to capture. Platform constraints that
were known from documentation before any code existed (Spark's
compatibility-mode-only GDS, no GPUDirect RDMA) are recorded as fixed points
in [architecture.md](architecture.md) and D-004, not here; this log is for
what the documentation did not tell us.

Format:

```
## RE-NNN: Title  (YYYY-MM-DD, status: open | fixed-upstream | worked-around | wontfix)
Environment / Repro or measurement / Observed / Expected / Impact / Links
```

Newest first. RE-numbers are never reused.

---

## RE-008: Extended reference runs do not always preserve exact top-1 predictions  (2026-09-21, status: open)

Pinned llama.cpp `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, GB10,
driver 580.178.04, CUDA fusion and graphs disabled. Two distinct limits surfaced
in the [full paging study](experiments/paging-feasibility/full-study.md):

- DeepSeek V4 Flash sequence snapshots reserialize identically after restore,
  but append-only continuation differs from live state on 16/1,536 top-1
  predictions (0/6/10 by turn). A same-host repeat reproduces those differences;
  restored predictions also match across the two Sparks. Raw/compressed KV and
  compressor state are present in the serializer. Cache compaction changes
  attention shapes, but neither a numerical explanation nor missing semantic
  state has been established. This is not the Gemma rollback failure in RE-007.
- Qwen3.8 Flash Next's identical untraced binary repeated on one host differs
  on 6/1,536 predictions (3/2/1). A restore probe differs on 10, including five
  before any restore, so restoration is not an isolated cause. Qwen also prunes
  the last prefill layer to output rows (`models/qwen4exp.cpp:400`); assuming
  every layer routes the whole input batch aborts capture. Record the actual
  final-layer output-only dependency, without reducing consumed input tokens.

No numerical tolerance is inferred from these counts. Capture defaults remain
strict; Qwen's explicit drift-recording mode labels failed equivalence, and
replay/locality analysis require a separate opt-in. Both large-model spill
returns use conservative recomputation in the study. Raw evidence remains in
external `paging/large-capture-2`, `deepseek-restore-probe-2`, `qwen-capture-1`,
`qwen-capture-3`, and `large-restore-probe-1`; their identities and comparisons
are retained in the study's aggregate evidence. Do not promote these reference
observations into a jitLLM numerical or restore-compatibility guarantee.

## RE-007: Gemma sequence snapshots lose SWA history needed after prompt rollback  (2026-09-21, status: worked-around)

On Spark GB10/driver 580.178.04 and pinned llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`, the extended four-turn
Gemma restore probe changed **58/3,072** teacher-forced next-token argmax
predictions versus live full-SWA state (per turn: 0, 12, 19, 27). Ornith
changed **0/3,072**. CUDA fusion and graphs were disabled in both arms.
Each snapshot reserialized byte-for-byte identically after restoration into
a fresh context; that checks the stored bytes, not sufficient context coverage.
Raw negative evidence is `paging/restore-probe-1` on Spark, with the matched
live controls in `paging/small-capture-2`.

Pinned `src/llama-kv-cache.cpp:2080` drops cells outside the final SWA window
when serializing an individual sequence, even with full-SWA allocation.
Gemma uses standard SWA of 1,024 tokens. Its first snapshot ends at position
8,105 and retains SWA positions 7,082–8,105. The canonical next prompt shares
only 7,335 tokens: its first resumed query needs positions 6,312–7,335,
including **770 positions absent from the snapshot**. The next two returns
have the same missing-window count. Successful tail removal does not detect
this gap. Physical cache compaction also changes attention shapes
(`llama-kv-cache.cpp:1250`), but the missing dependencies alone invalidate
assuming equivalent restored execution; the prediction changes are not
classified as harmless numerical noise.

The native session and restore harnesses now check the earliest retained
position against the model's actual SWA window before reuse, including after
full-SWA restoration. They reset and recompute when coverage is insufficient,
following the conservative checkpoint-coverage principle in pinned
`tools/server/server-context.cpp:3297` and `:3349`. They have no older
checkpoint to restore. Replay uses captured recompute routes for Gemma
sequence-spill returns with prefix rollback, and rejects apparent reuse from
legacy normal-SWA captures by selecting the conservative recompute scenario.
Serialized byte identity and a successful short response do not establish
that a checkpoint supports arbitrary template rewrites. The earlier A→B→A
six-token rollback's matching output remains a narrow observation, not a
proof of complete retained-window coverage. Full-history snapshots would be
a different, larger spill contract and are not established by this probe.

## RE-006: Reading MoE routes through the llama.cpp callback changes the CUDA path  (2026-09-21, status: worked-around)

On the pinned reference `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`,
GB10/driver 580.178.04, a callback at `ffn_moe_topk-N` changed 51 of
3,072 teacher-forced next-token argmax predictions over four Gemma turns.
A repeated untraced control matched all 3,072 original control predictions;
the first traced difference occurred before any conversation reuse. The
short first-cut 118-prediction probe had matched, so it did not expose this.

The scheduler splits and synchronizes the graph at requested callback nodes.
The pinned CUDA implementation has a fused routing operation spanning the
selected top-k view, so this observation point changes the execution path.
Disabling both CUDA fusion and CUDA graphs in **both** traced and control
runs (`GGML_CUDA_DISABLE_FUSION=1`, `GGML_CUDA_DISABLE_GRAPHS=1`) restored
exact prediction equality in the extended retained and recomputed Gemma
captures. Those two settings were changed together; this does not isolate
their individual effects or prove bitwise-logit equivalence.

Keep instrumented-route experiments explicitly matched to their control
configuration, and keep their timing separate from the reference's normal
optimized path. A later observation point might preserve normal fusion but
needs its own alias/lifetime and numerical checks. See the
[paging-feasibility experiment](experiments/paging-feasibility/README.md).

## RE-005: Spark perftest warmup option stalled an RDMA-CM sweep  (2026-09-21, status: worked-around)

Environment: both Sparks, kernel `7.0.0-1019-nvidia`, ConnectX firmware
`28.45.4028`, `rdma-core 50.0-2ubuntu0.2`, Ubuntu perftest
`24.01.0+0.38-1build2` (binary reports 6.20), RoCE v2 / MTU 1024.
An `ib_write_bw -d rocep1s0f1 -R -p 18700 -a -n 1000 --report_gbits
--perform_warm_up` server/client pair failed to finish within 90 seconds;
both remote timeouts returned 124. Removing the optional warmup flag
completed the sweep, and the subsequent repeated baseline passed. Duration
tests discard a one-second start/end margin instead.

This is an observed option-combination timeout, not an isolated root cause
or evidence that the DAC requires a reboot. The excluded pilot receipt and
logs remain in workstation `/tmp/jitllm-interconnect/host-sweep/`; the
[baseline report](experiments/interconnect/README.md) records the working
protocol. Also, `ib_write_bw --version` prints `Version: 6.20` but exits 1;
do not treat that informational exit as a failed transfer.

## RE-004: llama.cpp Gemma slot restore reports success but default SWA re-prefills  (2026-09-21, status: worked-around)

Environment: `spark-c4e2`, GB10, driver 580.178.04; pinned llama.cpp
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4` in the CUDA 13.3 ARM64 container;
Gemma 4 26B A4B Unsloth UD-Q4_K_M. Full identities and the retained
[smoke harness](experiments/reference-setup/README.md) accompany the result.

With an 8192-token context, f16 K/V, one slot, batch/microbatch 512, and
default SWA retention, the slot save and fresh-process restore endpoints
both reported **627 tokens / 141,268,896 bytes**. A 639-token continuation
then re-evaluated **all 639 tokens**; the server logged a full prompt
re-processing fallback due to missing cache data. The equivalent resident
continuation evaluated 18 tokens after an internal checkpoint rollback.
The prompt was shorter than the 1024-token sliding window. Successful
serialization counters therefore do not establish useful continuation reuse.

The pinned [server source](https://github.com/ggml-org/llama.cpp/blob/b29c606e28a01b1bc8c1351026a0fa6e616bf6c4/tools/server/server-context.cpp)
uses SWA coverage thresholds and context checkpoints when deciding whether
the common prefix can be reused. `--swa-full` disables that SWA checkpoint
path. With it, two runs restored the 627-token prefix, evaluated only the
12-token extension, and matched all 32 resident-continuation output token
IDs. This is a measured workaround for this pinned configuration, not a
general fix for all hybrid models or proof of long-context coverage.

Cost: logged KV allocation grows from **460 MiB** (160 global + 300 SWA)
to **1760 MiB** (160 global + 1600 full-SWA) at this context. Keep the
normal/windowed and full-SWA reference configurations separate in the
A→B→A experiment and include their actual memory costs. To reproduce the
negative control, remove only `--swa-full` from an external copy of
`smoke.py`; its restore assertions must fail rather than report a pass from
the matching API counters. The two differing continuation token streams in
the negative control alone are not evidence of corrupted KV: they used
different prefill paths. No upstream fix is claimed.

Long-context follow-up: the [A→B→A report](experiments/reference-aba/README.md)
reproduces this behavior at context 32,768. Three default-SWA restore trials
reported 18,303 restored tokens but processed all 18,339 continuation input
tokens. Full-SWA restore reused 18,297 and processed 42, matching every
118-token output. The six-token difference between saved and reusable
counts is legitimate: Gemma's canonical chat template removes the empty
generation-only thinking marker from completed turns. Compare the actual
longest common token prefix, not just the save API's count. A separate
early/late notebook recall check passed. Also retain the measured decode
speed distinction (about 27–28 tokens/s live/recomputed full-SWA versus
46–47 after restore/default SWA); its cause was not isolated here.

## RE-003: nvme-cli 2.8 feature control requires --value, and zero has a different printed form  (2026-09-21, status: worked-around)

Environment: Spark, installed nvme-cli 2.8-1ubuntu0.1. During the bounded
interrupt-coalescing comparison, a command using `set-feature -f 8 -v 263`
set zero: `-v` is **verbosity**, not value, in this version; the value option
is `-V` or `--value`. Also, `get-feature` prints nonzero as
`Current value:0x00000107` but zero as `Current value:00000000`. A parser
requiring `0x` rejected zero, including during the attempted cleanup.

The hardware readback exposed the mismatch before any A/B measurement was
retained. The original value was restored with
`nvme set-feature /dev/nvme0 --feature-id=8 --value=263`, then independently
read back as `0x107`. The comparison was rerun in full with long options,
both output forms accepted, and verified restoration. Do not trust mocks
based on remembered short flags for a device-control command: check the
installed tool's help and read back the actual state. The retained
[coalescing harness](experiments/io-path/coalescing.py) records the original
value before changes and bounds/restores its temporary setting.

## RE-002: cuFile compatibility mode rejects a descriptor opened with O_NOFOLLOW  (2026-09-21, status: worked-around)

Environment: `spark`, GB10, driver 580.178.04, installed libcufile package
1.15.1.6-1 (reported API version 2.12), Linux 7.0.0-1019-nvidia, ext4.
The I/O spike opened its private regular file with
`O_RDONLY | O_DIRECT | O_CLOEXEC | O_NOFOLLOW`; `cuFileHandleRegister`
failed with **5019 / CU_FILE_INVALID_FILE_OPEN_FLAG**. The library log
reported unsupported open flags `229376`. Native direct reads on the same
descriptor worked.

The comparison harness keeps its validated original descriptor open,
reopens `/proc/self/fd/<fd>` with `O_RDONLY | O_DIRECT | O_CLOEXEC`, and
checks device/inode identity before registering that new descriptor with
cuFile. This preserves file identity without following the original user
pathname again. Registration and subsequent GPU-verified reads then passed.
Do not respond by removing path protections from the original file open.
See the [I/O experiment](experiments/io-path/README.md) and its retained
`main.cc`. This workaround is confined to the comparison backend; the
native direct-file candidate does not need it.

## RE-001: CUDA 13.0 NVCC rejects C++23 even with a C++23-capable Clang host  (2026-09-21, status: worked-around)

Environment: x86-64 Ubuntu 24.04, NVCC V13.0.88, Ubuntu Clang 18.1.3;
AArch64 cross target using the Spark DGX OS 7.6.0 snapshot, GB10 `sm_121`.
Passing `-std=c++23` to NVCC fails before compilation with
`Value 'c++23' is not defined for option 'std'`. Clang's support for C++23
does not establish NVCC front-end support. This corrects the expectation
in D-010's original context; its allowance for a separate CUDA dialect
already covers the fix.

The older-toolkit comparison used C++23 `.cc` files and C++20 `.cu`
files and passed cross-built and native Spark runs. The selected D-032
pin instead uses Toolkit 13.4.2 / NVCC 13.4.92, which accepts C++23;
both execution paths passed with an explicit C++23 host/device feature
probe. M1 should use C++23 throughout for that selected SDK. Do not
silently fall back to the installed 13.0 compiler and assume the same
dialect support. See the [reproduction and pins](experiments/toolchain-smoke/README.md)
and [NVCC 13.4 options](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
