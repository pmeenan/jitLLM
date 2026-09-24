<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Third-party sources

Every third-party source component a jitLLM build may use is recorded in
[sources.lock.json](sources.lock.json) (D-017, D-057): its exact bytes, where
they came from, how its license was classified and how it builds. Nothing
else enters the build. [source-dependencies.md](../docs/source-dependencies.md)
explains the mechanism and why it was chosen.

| Step | Where | Does |
| --- | --- | --- |
| Prepare | `mise run prepare` ([tools/prepare-sources](../tools/prepare-sources)); `mise run setup` runs it after the SDK | Validates the lock, selects the profile's closure, then fetches each archive into the persistent cache (`~/.cache/jitllm/downloads/<sha256>/`, shared with the SDK and re-verified on every use). It refuses an archive with anything but plain files and directories, unpacks it with the SDK's CMake through FetchContent's script mode ([populate.cmake](../cmake/sources/populate.cmake)), applies the recorded patches and checks the tree digest before the tree appears as `build/sources/<id>-<tree>` |
| Configure | [cmake/JitllmSources.cmake](../cmake/JitllmSources.cmake) | Validates the whole lock with the same Python code ([tools/inspect-sources](../tools/inspect-sources)), selects the same closure, checks each prepared tree's digest, and only then adds the components as `SYSTEM`, `EXCLUDE_FROM_ALL` subprojects with their locked options. It never downloads. It rejects `FETCHCONTENT_SOURCE_DIR_*`, dependency providers and project-include hooks, makes FetchContent population of a declared dependency fail inside components (whatever their policy level), and fails on any `find_package()` lookup the lock does not declare |
| Build | [cmake/sources/verify.cmake](../cmake/sources/verify.cmake) | Checks each tree on every build before anything that uses it compiles, so an edit after configure (an added file, a mode change, an edit that keeps the timestamp) fails the build |
| Receipt | `build/<preset>/jitllm-receipt.json` | Records what configure used: the lock's digest, the SDK identity, the license profile and modules, and each component's version, license, archive digest, patches, tree, options and source directory. It is official only when no component came from an override |

`--dry-run` shows what `prepare` would do, and `--check` validates the whole
lock, every patch file included; `prepare` and configure read only the patch
files of the selected closure. A missing tree stops configure with the
command that prepares it. A modified tree stops configure and `prepare`
alike: remove the tree and prepare again, or use an override (below). The
`sources.*` tests check the lock, the receipt, and the compile and link
inventory recorded by Ninja against the receipt. The inventory accepts a
build-tree file only if a current build rule or a selected component's
build directory produced it. No compiled or linked object or archive outside
the SDK may contain exception support (D-066). See
[tests/sources/](../tests/sources/).

## Profiles

The **core** profile, the default, is the copyleft-disabled profile of D-002:
the lock's `core` components only. An optional module adds its components
when named at both steps: `mise run prepare -- --modules <m>`, then configure
with `-DJITLLM_MODULES=<m>`. Neither step fetches, reads or builds another
module's sources. A build directory that has built a module never builds a
profile without it: configure refuses, because the module's payloads could
be anywhere in that tree. It keeps that history in
`jitllm-modules-built.txt` and also refuses a directory that has component
outputs but no such record. Build the other profile in a new directory. The
record guards against mistakes, not a deliberate edit; a module's leftover
that a later build uses still fails the inventory check. The lock has no
optional modules yet. `tests/sources/` checks all of this on a
synthetic one.

## What the checks cannot see

Component build scripts run with the user's rights; CMake has no sandbox.
Configure blocks the ways a component could substitute a dependency that
CMake exposes: declared FetchContent population, source overrides,
dependency providers and package lookups. It cannot stop a script that
downloads with `file(DOWNLOAD)` or a tool, or copies host files into its own
build directory, which the inventory check accepts as that component's
output. So:

- auditing a component covers everything its build scripts read, write and
  run, and `license.scope` records that;
- the network-denied build (D-061's `check:full`, in the reference
  container with `--network none`) stops any download a build attempts.

## The lock (schema 1)

`modules` names each optional module with a `description`. Each entry of
`components` is keyed by a lowercase id and has:

| Field | Meaning |
| --- | --- |
| `version`, `upstream` | The release, its repository, tag and full commit |
| `kind` | `archive`, a hash-pinned upstream archive. Vendored units (`third_party/<id>/` in Git) arrive with M2's first adapted kernel |
| `category`, `tier`, `module` | D-017's classification. `implementation` is incorporated code. `core` needs an allowlisted license (Apache-2.0, BSD-2-Clause, BSD-3-Clause, MIT or MPL-2.0); `optional` needs a `module`. A core component never depends on an optional one |
| `use` | `test` if only test executables link it (never shipped), else `product` |
| `machine` | `target`: built with the profile's target toolchain. Build-host tools and generators are not supported until the first one needs a host build |
| `archive` | `file`, `urls` (https), `sha256` and `size`. Bytes that change under the same URL are an error, not a lock update. Members must be plain files and directories |
| `patches` | Ordered `path` (relative to this directory, conventionally `patches/<id>/`) and `sha256`, applied exactly: git-style unified diffs of text files with no fuzz, renames, mode changes or binary hunks. Text before the first file and git's signature are skipped; any other line between files is an error |
| `tree_sha256` | The prepared tree's digest: SHA-256 over one `<sha256> <x or -> <path>` line per file, sorted by path, where `x` marks an owner-executable file. Symbolic links, special files and empty trees are refused |
| `depends` | Other components that must be added first |
| `cmake` | `subdirectory` holding the project, `options` set for it alone, the `platform_packages` it may look up with `find_package()` (D-017's declared platform), and the `targets` jitLLM links. Option values are plain words, never paths. Names may not start with `_`, `CMAKE_`, `JITLLM_` or `FETCHCONTENT_`, or be `BUILD_SHARED_LIBS` |
| `license` | `expression` (SPDX identifiers joined by ` AND `), the license `files` in the tree, shipped `notices`, the audited `scope` (what is compiled and executed), the `evidence`, and the `obligations` |
| `verification` | How the pin was checked |

## Adding or changing a component

1. Choose the release and confirm the archive is immutable. Download it and
   record its SHA-256 and size. Compare it with the tagged commit where you can.
2. Audit what the build compiles, includes, generates and executes, and
   what its scripts read, write and download: license headers, bundled
   code, generators and `find_package()` calls. Record the result under
   `license` and `verification`. A new dependency's handoff note names its
   D-017 category and tier. A decision entry is needed where AGENTS.md
   rule 1 calls for one.
3. Add the entry and lock every option the project declares, turning off
   tests, examples, installers and downloads. `mise run prepare` then
   reports the tree digest to record. Review that tree before recording it.
4. Build and test every workstation preset.

## Local development overrides

To build a component from edited source, point
`JITLLM_SOURCE_OVERRIDE_<ID>` (the id in upper case, `-` as `_`) at a copy
of its tree. Configure warns, and the receipt records the override and
whether it differs from the lock, and is marked unofficial.
`JITLLM_REQUIRE_LOCKED_SOURCES=ON`, for check and release builds, rejects
any override.
