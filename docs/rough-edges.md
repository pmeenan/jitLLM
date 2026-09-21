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
