<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

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

**Pre-code. M0 (plan the plan) is done; M1 (bootstrap) is in progress.** The
design brief is in [docs/ideation.md](docs/ideation.md); the living plan,
feature matrix, architecture, and decision log are in `docs/`. No application
code exists yet. Planned distribution is a signed apt repository for DGX
Spark.

The first useful product target is M4: chat with A, switch to B under memory
pressure, then resume A with retained state, through an unmodified client.
M4a adds configured placement across nodes; demand-paged MoE and sharding
have separate later gates. See [the plan](docs/plan.md).

## Development setup

Development happens on an x86-64 Ubuntu 24.04 workstation. Spark binaries
are cross-built there and tested over SSH. The compilers, CUDA, CMake and
Ninja come from a pinned SDK that the project provisions under your home
directory. Setup changes no system compilers, packages or shell files.

1. Install [mise](https://mise.jdx.dev) 2026.9.12 or newer and the packages
   in [toolchains/prerequisites/](toolchains/prerequisites/). Setup prints
   the `apt-get` command for any that are missing.
2. In the checkout, run `mise trust` and then `mise run setup`. Setup
   downloads the artifacts pinned in
   [toolchains/artifacts.lock.json](toolchains/artifacts.lock.json),
   checking each SHA-256, builds the GCC 16.2 runtimes and assembles the
   SDK. It then prepares the third-party sources pinned in
   [third_party/sources.lock.json](third_party/sources.lock.json) into
   `build/sources/` (`mise run prepare` does that step alone). Later runs
   return at once.
3. `mise run doctor` checks the SDK and reports the host, driver and
   toolchain identities.

[toolchains/README.md](toolchains/README.md) describes the SDK. The
reference container in [.devcontainer/](.devcontainer/) runs the same setup
on a clean Ubuntu image.

## Building and testing

`mise run build` configures and builds with the SDK's CMake, Clang and
Ninja; `mise run test` builds and runs the tests. Both take CMake preset
names after `--` (default: `native` on x86-64, `spark-native` on a Spark),
and write to `build/<preset>/`:

| Preset | Builds | Tests run |
| --- | --- | --- |
| `native` | x86-64, CUDA for `sm_121` | On the workstation, GPU tests skipped |
| `cpu` | x86-64 with no CUDA toolkit | On the workstation |
| `cross` | AArch64 for DGX Spark, CUDA for `sm_121` | Under qemu-user, GPU tests skipped; or on a Spark with `--host` |
| `spark-native` | AArch64, built on a Spark (a diagnostic fallback) | On that Spark |
| `cpu-asan` | `cpu` with ASan, UBSan and LeakSanitizer | On the workstation |
| `cross-asan` | `cross` with ASan and UBSan | Under qemu-user without leak detection; or on a Spark with it, with `--host` |
| `cross-tsan` | `cross` with ThreadSanitizer | Only on a Spark, with `--host` |

For example, `mise run test -- native cpu cross` runs every workstation
profile, and arguments after a second `--` go to CTest
(`mise run test -- cpu -- -R contract`). `mise run test -- cross --host
<spark>` copies the cross build to the named SSH host, under
`~/.cache/jitllm/deploy/`, and runs its CPU and GPU executables there; build
and binary inspections stay on the workstation. `mise run deploy --
--host <spark>` builds and copies without running tests. Neither ever picks
a host for you. Deploying needs `ssh` and `rsync` on the workstation and
`rsync` on the Spark.

To run CMake by hand, point `JITLLM_SDK` at the SDK and use its `cmake`:

```bash
export JITLLM_SDK=$(tools/setup-toolchain --print-root)
"$JITLLM_SDK/bin/cmake" --preset native && "$JITLLM_SDK/bin/cmake" --build --preset native
```

## Checks

Before handing off a change, run the local check gate (D-061; there is no
hosted CI yet). Each tier ends with a summary of its steps and the host,
commit and SDK they ran with:

| Task | When | Runs |
| --- | --- | --- |
| `mise run check` | Every change | clang-format, REUSE lint, the embedded-header check, the tooling tests, the `native`, `cpu` and `cross` builds and tests (cross under qemu-user), and clang-tidy |
| `mise run check:full` | Toolchain, dependency, packaging and blast-radius changes, and releases | `check`, then `cpu-asan` and `cross-asan`, and the reference build: the checkout copied into the [reference container](.devcontainer/), its sources prepared from an empty cache, then `native`, `cpu` and `cross` built and tested with no network. Needs Docker |
| `mise run check:spark -- --host <spark>` | Anything that needs the hardware | The `cross`, `cross-asan` and `cross-tsan` tests on that Spark, GPU tests and leak detection included |

Check builds configure afresh with the build tool's `--locked`: the core
profile, from locked sources only, with nothing kept from a build
directory's cache. The steps ignore the caller's compiler, CMake, test and
sanitizer environment (`CXXFLAGS`, `GTEST_FILTER`, `ASAN_OPTIONS` and the
like; `tools/check` lists them). The reference build provisions its SDK inside the
container, into the `jitllm-sdk` and `jitllm-cache` volumes the dev
container also uses. The first run takes as long as `mise run setup`.

Every file carries its copyright and license as SPDX tags
([REUSE](https://reuse.software/spec-3.3/)): in a header comment wherever the
format allows one, otherwise in a `.license` sidecar beside it (`data.json`
and `data.json.license`). A new file copies the header of its neighbours:

```text
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
```

with the file's own comment syntax (`//` in C++ and CUDA, `<!-- -->` in
Markdown). The header check fails on a file type it does not know yet; add
the type to [tools/jitllm_headers.py](tools/jitllm_headers.py).

## License

jitLLM's own code is Apache-2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE));
[LICENSES/](LICENSES/) holds the text of every license a file in this
repository declares. Incorporated core
implementation dependencies use Apache-2.0, BSD, MIT, or MPL-2.0. Build tools
and declared platform dependencies, including system libraries and CUDA,
retain their separate terms and are included in the dependency audit.
AGPL-licensed kernels or importers live only in optional modules you choose
to enable at build time; those builds must report and satisfy the applicable
license, notice, and source obligations. See D-003 and D-017 in
[docs/decisions.md](docs/decisions.md), and
[docs/licensing.md](docs/licensing.md) for what each dependency and tool
contributes to a build and what its notices require.

## Start here

- [AGENTS.md](AGENTS.md) — constraints, doc map, agent rules
- [docs/vision.md](docs/vision.md) — why, who for, success criteria, non-goals
- [docs/features.md](docs/features.md) — confirmed / proposed / open questions
- [docs/plan.md](docs/plan.md) — the M1–M8 milestone ladder with exit criteria
- [docs/m0-record.md](docs/m0-record.md) — what M0's planning, spikes and reference runs did, with evidence links
- [docs/workflow.md](docs/workflow.md) — how agents and the human collaborate
- [docs/rough-edges.md](docs/rough-edges.md) — findings log
