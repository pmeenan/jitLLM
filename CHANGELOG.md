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
