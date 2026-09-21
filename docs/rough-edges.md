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
