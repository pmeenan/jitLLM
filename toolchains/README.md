<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development SDK

jitLLM builds with a pinned, project-provisioned SDK (D-012, D-049, D-070):
Clang/LLD and the LLVM developer tools 22.1.8, CUDA 13.4.92, CMake 4.4.3,
Ninja 1.13.2, the source-built GCC 16.2 C++ runtime that executables link
statically (D-060) and, on x86-64 hosts, the Spark target sysroot. System
packages supply only the declared host prerequisites; setup never installs
packages or changes system defaults.

| File | Holds |
| --- | --- |
| [manifest.toml](manifest.toml) | What each build host's SDK contains: components, the GCC configure flags and used outputs, and the tools linked into `bin/` |
| [artifacts.lock.json](artifacts.lock.json) | Every downloaded byte: URLs, SHA-256 and size, and under `sources` how each was verified when pinned |
| [prerequisites/](prerequisites/) | The Ubuntu 24.04 packages each build host needs, checked by `mise run doctor` |

`mise run setup` ([tools/setup-toolchain](../tools/setup-toolchain)) provisions
the SDK; `mise run doctor` ([tools/check-toolchain](../tools/check-toolchain))
verifies it and reports its identity; `mise run doctor --deep` also checks
every SDK file against its recorded digest. `--dry-run` previews setup (or
`--prune`) without changing files, and setup exits early when the SDK already exists.

## Layout

The SDK lives at `~/.local/share/jitllm/sdk/<identity>` and its download and
build cache at `~/.cache/jitllm`; `JITLLM_SDK_HOME` and `JITLLM_CACHE_HOME`
(or the XDG variables) move them. `tools/setup-toolchain --print-root` prints
the root for the current checkout.

| Path | Contents |
| --- | --- |
| `bin/` | `clang`, `clang++`, `ld.lld`, `clang-format`, `clang-tidy`, `clangd`, `clang-scan-deps`, `llvm-symbolizer` and other LLVM binary tools, `cmake`, `ctest`, `cpack`, `ninja` |
| `llvm/` | LLVM 22.1.8. The resource directory holds the host's compiler-rt and, on x86-64 hosts, the AArch64 archives for cross sanitizer builds |
| `cuda/` | NVCC, CRT, libNVVM, libnvptxcompiler, cudart (including `libcudart_static.a`) and CCCL; x86-64 hosts also get `targets/sbsa-linux` for cross builds. Run NVCC as `cuda/bin/nvcc`: it finds its configuration beside the path it is invoked by, so a symlink elsewhere breaks it |
| `cmake/`, `ninja/` | CMake and Ninja release binaries |
| `gcc/<triple>/` | The host's GCC 16.2 runtime: `include/c++/16`, `lib64/{libstdc++,libsupc++,libatomic}.a` and `lib/gcc/<triple>/16/{crt*.o,libgcc.a,libgcc_eh.a}`. Select it with `--gcc-install-dir=<root>/gcc/<triple>/lib/gcc/<triple>/16` |
| `sysroot/aarch64-linux-gnu/` | x86-64 hosts only: the Spark sysroot (glibc 2.39 and kernel headers from pinned Ubuntu arm64 packages) with the cross-built GCC runtime at `opt/gcc`, in the same layout |
| `pkgs/` | The unpacked package trees behind `llvm/` and `cuda/`, including their copyright files |
| `sdk.json` | The receipt: identity, input digests, versions, each component's artifacts and GCC build inputs, and a digest of the whole tree |

Nothing in the SDK needs `LD_LIBRARY_PATH`. The tools' remaining shared
libraries are the host prerequisites, which `doctor` checks with `ldd`.

## Identity

The identity is the host architecture plus a digest of `manifest.toml`,
`artifacts.lock.json`, `tools/setup-toolchain` and `tools/jitllm_sdk.py`.
Changing any of them gives a new SDK in a new directory; SDKs from different
inputs never mix, and a checkout always finds the SDK its inputs describe.
Old SDKs stay until `tools/setup-toolchain --prune` removes them. Downloads
are cached by SHA-256 and re-verified on every use. GCC runtime builds are
cached by their own inputs (sources, flags, target, sysroot packages and the
build host's compiler and binutils), so an unrelated pin change does not
rebuild GCC.

Each cached GCC build records its inputs and a content digest, checked before
reuse. If a cache is incomplete, modified, or predates this integrity record,
setup stops and identifies the cache directory to remove before retrying.
The SDK itself and other cached builds stay intact.

The tooling regression tests run without downloads or an installed SDK:
`mise exec -- python3 -B -m unittest discover -s tools/tests`.

## Changing a pin

1. Validate the new version end to end first (D-012): native x86-64, the
   cross build run on a Spark, and the native Spark fallback.
2. Update `manifest.toml` and the lock entries, with URLs, SHA-256 and sizes
   taken from a verified source, and record the verification in `sources`.
3. A new or changed pin of a load-bearing component gets a decision entry
   (AGENTS.md rule 1).
4. Run `mise run setup` and `mise run doctor` on each build host.
