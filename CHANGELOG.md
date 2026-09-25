<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to jitLLM are recorded here, in the format of
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/). jitLLM
follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html) and
stays at 0.x, where a minor release may break compatibility, until 1.0
(D-062). A change to a public surface names its version bump here.

## [Unreleased]

### Added

- The `jitllm` command, with `--version`: the product version (`X.Y.Z` for a
  release, `X.Y.Z-dev.N+g<commit>` otherwise), the commit, the license
  profile, the SDK identity and the target. The build receipt records the
  same version, with its Debian form (`X.Y.Z~dev.N+g<commit>-1`).
- `jitllm doctor`, a capability probe: the build, the host (kernel, glibc,
  memory), RDMA ports, the NVIDIA driver and each GPU's compute capability,
  compute mode, VMM support and backing granularity. It exits 1 when the
  host cannot run the build, which needs a GB10 with VMM and host-backed
  VMM.
- CUDA builds of `jitllm` need the NVIDIA driver (`libcuda.so.1`) to start.
- The node's configuration, `schema_version = 2`: `/etc/jitllm/jitllm.toml`
  and the fragments in `/etc/jitllm/jitllm.d/`, strict TOML 1.0 in which
  every key has one owning file, with the `[storage]` roles, `[limits]` and
  a cluster member's keys. Every problem is reported with its file, line
  and column, and files or directories that other users could change are
  refused.
- An arm64 Debian package, `jitllm`: the `jitllm` command, the node runtime
  `/usr/libexec/jitllm/jitllm-runtime` and `jitllm.service`, which runs it
  as the new `jitllm` system user with `/var/lib/jitllm` as its data
  directory. It depends on the NVIDIA driver 580 or newer. The runtime reads
  its configuration, prepares its storage, checks the host and waits; it
  serves nothing yet. It exits on a fatal signal instead of dumping core.
- `jitllm doctor --config FILE` reads a configuration other than the
  default, and doctor now reports the configuration and the storage roles:
  their owners, modes and filesystems.
