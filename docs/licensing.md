<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Licensing and provenance

jitLLM-authored code is Apache-2.0. [D-002, D-003 and D-017](decisions.md)
govern incorporation, optional modules, tools and platform dependencies.
The first three sections record how the repository declares licenses and
what the pinned toolchain puts into a jitLLM binary (M1, checked
**2026-09-24**; D-071). The rest is the M0 reference inventory, checked on
**2026-09-22**: evidence for the first dense slice, early EXL3 proof and
later optional-backend design, not approval to import these repositories or
a license audit of their built containers. No implementation was
incorporated.

## Repository license metadata

Every file follows the [REUSE specification 3.3](https://reuse.software/spec-3.3/)
(D-029, D-071):

- A file whose format has comments starts with `SPDX-FileCopyrightText` and
  `SPDX-License-Identifier` tags in a header comment. jitLLM's own files say
  `2026 jitLLM contributors` and `Apache-2.0`; third-party material names its
  holders and actual license, as the two experiment patches with MIT
  upstream context do.
- A file that cannot hold a comment (JSON, a patch, plain text, `NOTICE`, or
  `mise.lock`, which mise rewrites) has a `.license` sidecar with the same
  tags. There is no `REUSE.toml`: it could override a file's own header.
- [LICENSES/](../LICENSES/) holds the text of every license a file declares:
  Apache-2.0 (the same bytes as the root `LICENSE`) and MIT (SPDX License
  List 3.29.0).
- [NOTICE](../NOTICE) carries jitLLM's attribution and names the
  third-party material in the repository.

`mise run check` enforces this with two steps. `reuse` runs REUSE lint 6.2.0
from the SDK. `headers` ([tools/jitllm_headers.py](../tools/jitllm_headers.py))
checks what REUSE lint accepts but D-029 does not, against REUSE's own JSON
report. Each commentable file must carry its own header rather than a
sidecar, and REUSE must read exactly that header's license from the file
itself. A license or copyright tag anywhere in a file or its sidecar, read
as a person would see it, must repeat an expression and a holder REUSE
reads for that file, so no tag can show a reader a license REUSE skipped.
Every sidecar must have a file, REUSE must cover every file the check does,
and no file type may go unclassified.

## What builds jitLLM

[toolchains/provenance.toml](../toolchains/provenance.toml) records every
locked SDK artifact, host prerequisite and mise tool with its D-017
category, license, what of it reaches a binary and the notices that follow;
a test fails when any of them lacks a record. Third-party source code is
recorded in the [source lock](../third_party/README.md) instead: GoogleTest
links into tests alone, and toml++ (MIT, D-073) into the shipped binaries,
whose notices then carry its MIT text and Bjoern Hoehrmann's copyright line
from its UTF-8 decoder.

| Unit (version) | Category | License | In a packaged binary |
| --- | --- | --- | --- |
| CMake 4.4.3, Ninja 1.13.2 | tool | BSD-3-Clause; Apache-2.0 | Nothing |
| Clang, LLD and the LLVM tools 22.1.8 | tool | Apache-2.0 WITH LLVM-exception (libz3: MIT) | Generated code |
| Clang's resource headers | platform (compiler support) | Apache-2.0 WITH LLVM-exception; `arm_neon.h` and `arm_fp16.h` carry an MIT text | Inline code and macros |
| compiler-rt | platform | Apache-2.0 WITH LLVM-exception | Nothing: sanitizer builds only; executables link libgcc, not its builtins |
| GCC 16.2.0 runtime (D-060) | platform | GPL-3.0-or-later WITH GCC-exception-3.1, with embedded code below | Every executable: libstdc++, libgcc, libgcc_eh, crtbeginS/crtendS. Not libatomic, which is not on the link line |
| glibc 2.39-0ubuntu8.9 (sysroot) | platform | LGPL-2.1-or-later | Start files and `libc_nonshared.a` members, under a linking exception; libc, libm and the loader stay dynamic |
| Linux UAPI headers 6.8.0-142.142 | platform | GPL-2.0-only WITH Linux-syscall-note | Constants and macros |
| CUDA 13.4.92 runtime and headers | platform | NVIDIA CUDA EULA | CUDA builds: `libcudart_static.a`, NVCC's host stubs and registration code, device code |
| CUDA driver link stub 13.4.92 (`cuda-driver-dev-13-4`) | platform | NVIDIA CUDA EULA | Nothing: CUDA builds need `libcuda.so.1`, which the driver supplies (D-072) |
| cuBLAS and cuBLASLt 13.8.0.4 (D-076) | platform | NVIDIA CUDA EULA | Nothing yet. When a binary uses cuBLAS it links `libcublas.so.13` and `libcublasLt.so.13` dynamically, and the package ships them unmodified (EULA Attachment A) |
| CCCL 13.3.4.3.1 (libcu++, `nv/`) | platform | Apache-2.0 WITH LLVM-exception | Through CUDA headers such as `cuda_fp16.h` |
| NVCC, libNVVM, ptxas and the other CUDA tools | tool | NVIDIA CUDA EULA (internal use) | Generated code |
| REUSE lint 6.2.0 and nine wheels | tool | GPL-3.0-or-later and others (provenance.toml) | Nothing |
| GMP, MPFR, MPC, ISL and gettext (GCC's build inputs) | tool | LGPL-3.0-or-later (GMP also GPL-2.0-or-later), MIT, GPL-3.0-or-later | Nothing: none of their symbols are in the runtime |
| Host prerequisites, mise, Python | tool; host glibc is platform | Ubuntu package terms; MIT; PSF-2.0 | Nothing in a cross-built binary |

Embedded code in the static GCC runtime, as link maps show it (this
corrects and extends the [GCC 16.2 report](experiments/gcc16-static/README.md#provenance-and-use)):

- **Linked routinely.** HP and SGI STL code (headers, and `tree.o`,
  `list.o`); Ryu (Apache-2.0 OR BSL-1.0, taken under BSL-1.0) with *any*
  `<format>` or `<print>` use, not only floating point; libiberty's
  `cp-demangle.o` (GPL-2.0-or-later WITH GCC-exception-2.0) through the
  default terminate handler; glibc soft-fp comparison objects (LGPL-2.1 with
  a linking exception) on AArch64; and `<format>`'s tables generated from
  Unicode data.
- **Linked only when used.** fast_float (MIT) with floating-point
  `std::from_chars`; the tz database (public domain) with chrono time zones;
  IBM-HRL code with `ext/pb_ds`; Jeremy Siek's concept checks; `<barrier>`
  (Apache-2.0 WITH LLVM-exception). PSTL headers arrive through
  `<algorithm>`, `<memory>` and `<numeric>` but contribute code only with
  `<execution>`. libbacktrace never: the SDK has no `libstdc++exp.a`.

## What a packaged binary carries

For a CUDA-enabled arm64 `jitllm`, cross-built by the SDK:

- **Always:** jitLLM's `LICENSE` and `NOTICE`; the HP and SGI permission
  notices, which ask to appear in supporting documentation; a statement that
  the CUDA runtime and NVCC-generated code are under the NVIDIA CUDA EULA,
  not Apache-2.0; and the CUDA headers' Disclaimer and U.S. Government End
  Users Notice, which their text requires in user documentation. The SBOM
  must also identify the GCC 16.2.0 runtime and glibc.
- **When the code is used:** the Unicode notice (any `<format>` or `<print>`;
  shipped by default unless the owner decides otherwise), fast_float's MIT
  notice, Norbert Juffa's and SoftFloat's notices for CUDA device math, the
  MIT text of Clang's NEON headers, the IBM-HRL and Siek notices, and
  glibc's 4.4BSD notice for BSD header macros.
- **Not needed:** Ryu, cp-demangle, soft-fp, the glibc start files and
  `libc_nonshared.a`, the tz database, PSTL, `<barrier>`, libcu++, Clang's
  other headers and the kernel headers. Their exceptions or terms ask
  nothing of object code.

No source offer is owed for such a binary. The EULA lists
`libcudart_static.a` as redistributable (Attachment A) inside an
application with material additional functionality. `cudart_static.o`'s
`.comment` names GCC 8.3.0, and it references no C++ runtime, consistent with
the GCC exception's eligible compilation process. NVIDIA's build process
itself is unverified.

Three constraints follow:

- **Never publish the SDK**, its caches or a reference image populated with
  it. The CUDA tools are for internal use only under the EULA, the runtime
  archives would owe GPL and LGPL source, and `cmake-gui` statically links
  LGPL-3.0 Qt.
- **Release packages come from the `cross` profile.** `spark-native` links
  the Spark's own, unpinned `libc6-dev` and GNU linker.
- **Don't adopt CUB or Thrust directly without a D-017 decision.** Used
  directly they are incorporated implementation, and Thrust includes
  BSL-1.0 files, which the core allowlist does not name.

**How the package carries them** (D-074): `tools/jitllm_package.py` writes
`/usr/share/doc/jitllm/THIRD-PARTY-NOTICES` from the source lock's notices
and the text that each shipped unit's notices locate in
[provenance.toml](../toolchains/provenance.toml) (`extract`), including every
"when the code is used" notice and the whole CUDA EULA, whether or not the
build reaches the code; a Debian `copyright` file; and an SPDX 2.3 SBOM
naming the GCC 16.2.0 runtime, glibc and every other shipped unit. The
statement that NVIDIA code is under the EULA opens the notices.

**Decisions** (the owner's answers on 2026-09-24, for the Package item;
D-074):

1. jitLLM's CUDA sources do not carry the CUDA header notice; it ships in
   the documentation (`THIRD-PARTY-NOTICES`), which the headers' text also
   asks for. The headers ask for it "in the user documentation and internal
   comments to the code", and jitLLM's `.cu` files are its own code.
2. The package ships the CUDA EULA's full text, and its `copyright` file
   names the NVIDIA code under `LicenseRef-NVIDIA-CUDA-EULA`. The reading
   that static linking and stripping do not "modify" the runtime object
   code (EULA §2.3) is an interpretation, not confirmed with NVIDIA.
3. The LGPL-2.1 §5 reading for glibc stands. The start files and
   `libc_nonshared.a` carry the linking exception. `csu/init.c` has none but
   contributes one constant, `_IO_stdin_used`. A §6 reading would clash with
   the EULA's ban on reverse engineering the embedded runtime.
4. The Unicode notice ships. The [first dense slice](#first-dense-slice-d-051)
   still treats Unicode-derived tables it would incorporate as needing their
   own provenance.
5. Clang's resource headers and CCCL reached through CUDA headers belong to
   D-017's platform family, as recorded, although D-017 does not name
   compiler headers or `Apache-2.0 WITH LLVM-exception`.
6. The evidence above that `libcudart_static.a` meets the GCC exception's
   eligibility test is accepted, as D-060 asked.
7. Code compiled from CUDA headers that Attachment A does not list ships as
   object code under EULA §1.1.1's "as incorporated in object code format".
   Their own notice prohibits reproducing or disclosing them to third
   parties "notwithstanding" the EULA, and Attachment A names only some
   headers (the fp16, bf16 and fp8 family, `cuda_occupancy.h` and the
   runtime-compilation set). The inline code of `cuda_runtime.h`, the
   host-stub headers NVCC uses and the device math headers reach the binary
   only as object code. The reading is not confirmed with NVIDIA.

**Method.** Link probes cross-built with the SDK's `cross` flags (`base`
using `<vector>`, `<algorithm>`, `<memory>`, `<string>`, `<format>`,
`<print>`, `<expected>`, `<span>`, `<atomic>`, `<thread>`, `<mutex>` and
`<chrono>`; `double` printing; floating-point `from_chars`; time zones), plus
a relink of the cross-built CUDA contract test with the build's own link
line, all with lld link maps and `--why-extract`. License texts come from
the SDK's packages and files, from glibc's sources at `glibc-2.39` (Ubuntu's
`libc6` copyright file is stale at 2.23 and omits the start files'
exception), from the GCC 16.2.0 tarball, from the CUDA EULA (updated
2026-01-26; the packaged copy matches NVIDIA's page) and from SPDX License
List 3.29.0. The HP and SGI notices match `HPND-sell-variant` by manual
comparison; the IBM-HRL and Siek notices match no SPDX identifier. The
probes and maps stay outside the repository.

## First dense slice (D-051)

The [selection](first-slice.md) pins the official Qwen2.5-0.5B-Instruct FP16
GGUF, its base-metadata cross-check and the existing llama.cpp reference.
[Artifact/tool pins](experiments/first-slice/pins.json) and the
[bounded source-unit inventory](experiments/first-slice/source-audit.json)
record immutable revisions, file hashes, actual notices and adoption status.
Seventeen inspected upstream file hashes matched the immutable GitHub source.
The inventory is not clearance of the future compiled dependency closure.

| Unit and use | Category / observed terms / disposition |
| --- | --- |
| Official Qwen GGUF and base metadata/tokenizer files | External model/test data; both repositories supply the same Apache-2.0 license file. No weights redistributed; retain source terms and required notices with future derived artifacts. Exact source-to-GGUF conversion lineage is unverified |
| Pinned llama.cpp executable/libraries and GGUF Python inspector | External reference/inspection tools, root MIT and gguf-py MIT; actual container/runtime terms remain in the reference setup record. Used only outside jitLLM serving |
| Selected GGML core, CPU/CUDA kernels; Qwen2 graph/tensor semantics; native GGUF reader | Future core implementation candidates, root MIT plus local MIT notices (including Mozilla llamafile SGEMM and YaRN authors). Preserve notices and audit the selected compiled closure before adoption. GGML allocator/workspace behavior still needs the M2 proof |
| Tokenizer implementation and generated Unicode tables | Root MIT implementation is not a blanket grant for derived data. `src/unicode-data.cpp` lacks exact input-data provenance; its generator uses a moving Unicode URL and Python Unicode data. **Blocked from native incorporation** until provenance/terms and D-017 eligibility are resolved |
| Chat-template rendering | The pinned template is model data under its source terms. Future owned native rendering must pass exact byte/token fixtures for enabled branches. Full `common/jinja`, chat/parser and vendor closure is not adopted or cleared |
| HF converter and Python package closure | Inspected only; neither executed nor incorporated. The split converter has remote-code/legacy-checkpoint paths outside the selected model path. Any future use needs independently pinned/audited tools, allowlisted data formats and remote code disabled |
| CUDA, driver and standard runtimes; NumPy/GGUF Python packages | D-017 platform dependencies and external experiment tools, respectively; retain exact image/component identities and their own terms. No new platform exception or source dependency is approved |

The current [Unicode license](https://www.unicode.org/license.txt) is Unicode
License V3 with notice requirements. The exact terms for data underlying the
pinned generated tables must still be traced; do not apply today's text
retroactively as proof. D-017 does not list that license, so regeneration or
copying needs a permitted path or a deliberate policy amendment. Fixed token
IDs keep M2 independent of tokenizer adoption; close this gate before M3.
No whole-vendor-tree, complete-image or redistribution clearance follows from
root MIT.

## Early EXL3 companion (D-052)

The [bring-up contract](exl3-bringup.md) and
[artifact pins](experiments/exl3-reference/pins.json) select two small
third-party Qwen EXL3 conversions for M2. Their downloaded Apache-2.0 license
files match the base Qwen license. The [Spark baseline](experiments/exl3-reference/README.md)
verifies full weight digests, execution and tokenizer/template identity; exact
conversion/calibration lineage remains unknown. These are external model/test
data; no weights are redistributed. Any new quantization job needs cleared
calibration inputs and pinned converter/input provenance.

The external reference uses the ExLlamaV3 revision below, the pinned container
recipe, three additional hash-pinned wheels and recorded compiler/runtime
identities. Its [runtime inventory](experiments/exl3-reference/runtime.json)
records compiled translation units, loaded library hashes and package notice
identities. An ARM host-helper patch includes MIT source context and retains
Turboderp's notice in [UPSTREAM-NOTICE.txt](experiments/exl3-reference/UPSTREAM-NOTICE.txt).
It disables x86 CPU MoE/collective helpers; no device kernels are changed.
No MiaAI patches or calibration corpus are used by this reference.

Selected MIT device kernels remain core-eligible implementation candidates,
pending the native selected-file and compiled-closure audit. The external
PyTorch/driver/toolkit stack is a declared reference tool/platform dependency,
not incorporated native implementation or clearance to redistribute its
whole container. Native wrappers must remove upstream allocator, stream and
scheduling ownership; their actual closure and notices must be audited anew.

**Open provenance question: the GEMV kernel.** At the pinned revision,
[`quant/exl3_gemv_kernel.cuh`](https://github.com/turboderp-org/exllamav3/blob/6b84a21b6f1e5da3f291b9e1019061f0de788279/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh#L3-L4)
describes its small-m path as a "QTIP-style structure". It cites
Cornell-RelaxML/qtip `qtip-kernels/src/inference.cu`.

- The file carries no notice of its own. The same path, with the same QTIP
  reference, is among the nine headers GLM TP3 vendors from upstream
  `02aef45c` (below). That earlier version differs from the pinned file.
- QTIP's repository `LICENSE` at
  [`e90c6688c8dfae326a3a81b5eb032db7c6680ec0`](https://github.com/Cornell-RelaxML/qtip/tree/e90c6688c8dfae326a3a81b5eb032db7c6680ec0)
  (head on 2026-09-22) is the GPL-3.0 text. The cited file exists there.
- ExLlamaV3's README says EXL3 is based on QTIP and calls it a streamlined
  variant of QTIP.

**What the review scan found.** On 2026-09-22 a review compared both GEMV
versions with QTIP's kernel sources at that revision, by token sequence.
Apart from identical PTX `mma` operand strings, it found no shared run of
20 or more tokens. That is an observation, not the structural comparison
required below, and not clearance. This inventory draws no conclusion about
derivation.

**The same gate covers related files.** It applies to any file that
includes this header, directly or through another header, or follows its
body:

- upstream `quant/exl3_gemv.cu`, the host wrapper, which includes the header
  and creates its variants;
- upstream `quant/exl3_moe_coop_kernel.cuh`, and the files that include it
  (`quant/exl3_moe_coop.cu`, `comp_units/exl3_moe_coop_inst_*.cu`);
- upstream `comp_units/exl3_gemv_half_inst.cu`;
- the GLM, GLM TP3 and DeepSeek `cooperative_moe_kernel.cuh` derivatives.

**Before any of these enters a core-eligible native module:**

1. compare the code structurally with the cited file at a pinned revision;
2. record the result;
3. have the owner resolve its D-017 disposition.

**What the proof does meanwhile.** The [backend proof](backend-proof.md)
runs the EXL3 GEMM kernel wherever upstream would select GEMV (m ≤ 8).
Other dispatch paths are unchanged.

**The other selected kernels are not cleared either.** None of them cites
QTIP or another third-party source in its own text. The remaining QTIP
comments sit in host files that dispatch to this path (`exl3_gemm.cu`,
`exl3_gemv.cuh`). The absence of a citation is not clearance: these
kernels still need the audit above.

## Reference instrumentation

The [fused-routes experiment](experiments/fused-routes/README.md) reads MoE
routes from the unmodified pinned llama.cpp image with jitLLM-authored
Apache-2.0 harness code. It retains `route-outputs.patch`, a rejected
libllama change with MIT upstream context and The ggml authors' notice in
`UPSTREAM-NOTICE.txt`, only to reproduce a negative result. The runner does
not build it, and nothing from it enters jitLLM.

## Additional reference candidates

The [Qwen-Image GGUF study](experiments/image-gguf/README.md) runs
stable-diffusion.cpp and its patched GGML fork (both MIT) as external
reference tools built from pinned sources; the GGUFs and Comfy-Org files
remain under the Qwen Research License, like the BF16 study. The
[MiMo reference](experiments/mimo-reference/README.md) uses the MIT-declared
checkpoint, the pinned SGLang (Apache-2.0) image and hash-pinned `torchcodec`
0.16.0 (BSD-3-Clause, Meta), plus one audited checkpoint configuration file
executed for config parsing only. The AGPL-3.0 MiaAI MiMo recipe was read as
documentation; none of its code or patches ran or entered this repository.
No weights are redistributed, and neither study clears a container's full
component closure or approves any of these for jitLLM's core.

## Pinned reference inventory

The links below identify the exact trees inspected; moving branch names are
not audit identities. All tracked paths were enumerated (including hidden
files), license files and declarations were inspected, and source notices
were checked separately from license strings embedded in test/patch payloads.
Counts include docs, data and configuration, not just implementation files.

| Key | Repository and immutable revision | Tracked paths |
| --- | --- | ---: |
| GLM | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks, c1b7d4c9f98c16af65640fef05fa493bbde52ccf][glm] | 252 |
| DeepSeek | [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks, 6f7d1590ad49a2b8995188e45d7b9db31e677452][ds] | 96 |
| Qwen | [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks, d2f54b78c0d2f9d74ac61aa56200e3c40fac3f22][qwen] | 226 |
| ExLlamaV3 | [turboderp-org/exllamav3, 6b84a21b6f1e5da3f291b9e1019061f0de788279][exl] | 743 |

The following is a **declared-license inventory and adoption disposition**.
A retained upstream notice establishes upstream terms; it does not alone
establish the terms of every downstream modification. Unresolved cases are
explicitly blocked from permissive-core incorporation under D-017. The
inventory is complete for this planning task; clearing a selected component's
history and full dependency closure remains an adoption-time requirement.

### Repository defaults and historical notices

| Scope | Observed declaration | Disposition |
| --- | --- | --- |
| GLM scripts, overlays and docs without a more specific declaration | README License section and root `LICENSE`: AGPL version 3 | Optional implementation tier if copied/adapted; no blanket MIT permission |
| DeepSeek launcher/overlay and other files without a more specific declaration | README and root `LICENSE`: AGPL version 3; README also mentions MIT files | Optional implementation tier; resolve file-specific exceptions below |
| Qwen files without a more specific declaration, including `tp1/` | README: `AGPL-3.0-or-later` | Optional implementation tier |
| GLM and DeepSeek root `LICENSE.MIT`; GLM `extensions/cooperative_moe/tp3/LICENSE.MIT` | Preface limits retained MIT notice to contributions before 2026-09-07 | Historical evidence only; not a dual-license offer for the current tree |
| Direct ExLlamaV3 upstream source | Root `LICENSE`: MIT, copyright Turboderp; `exllamav3/vendor/fla/LICENSE` separately retains MIT and its authors | Candidate for core-allowed implementation tier after selected-file/dependency audit; preserve actual notices |

GLM/DeepSeek say AGPL-3.0 without an explicit project-wide “or later” grant.
Do not infer that grant from the sample application notice at the end of the
standard license text. Record the literal declaration here; resolve the exact
SPDX expression for any adopted unit. Qwen explicitly provides “or later”.
For old MIT contributions, obtain a pinned historical version and verify the
actual code and notices before reuse; this audit did not clear old revisions.

### File-specific declarations and mixed provenance

Paths in a row are relative to its pinned repository. Unlisted MiaAI-Lab paths
retain the repository default above **for screening**, not a claim that
third-party material loses its original license. Full builds may combine
these categories.

| Repository and paths | Evidence and classification |
| --- | --- |
| GLM `overlay/ablit_runtime.py` | File starts with an MIT SPDX license identifier. Explicit MIT candidate; this does not cover its checkpoint inputs or the surrounding importer/launcher |
| GLM `overlay/exl3.py`, `overlay/dflash2_speculator.py`, `overlay/qwen3_dflash2.py` | File headers declare Apache-2.0; do not label these MIT merely because they integrate EXL3 |
| GLM `overlay/tp3/vllm/model_executor/parameter.py`, `overlay/tp3/vllm/model_executor/layers/vocab_parallel_embedding.py`, `overlay/tp3/vllm/model_executor/model_loader/weight_utils.py`, `overlay/tp3/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Apache-2.0 headers with vLLM attribution; modified overlay provenance still needs review before adoption |
| GLM `tests/fixtures/flashinfer_mla_sparse_sm120-487ecf187.py.txt`, `tests/fixtures/kda-487ecf187.py.txt` | Apache-2.0 file headers; fixtures are also incorporated source if copied |
| GLM `tests/test_gen_defaults.py` | Describes an embedded vLLM fixture as Apache-2.0; that notice does not independently license the entire test harness |
| DeepSeek `overlay/exl3.py` | Apache-2.0 file header, distinct from the AGPL-default patch scripts and native overlays |
| Qwen `files/ple_layer_patched.py`; `files/ple_offload/{connector,ple_offload_layer,protocol,worker}.py`; `files/ple_offload/orig/{connector,ple_offload_layer,protocol,worker}.py`; `tp1/files/ple_offload/orig/{connector,ple_offload_layer,protocol,worker}.py` | Apache-2.0/vLLM headers; README expressly preserves file-specific SPDX terms under `files/`. Record headers, but audit modifications and the `tp1/` copies before adoption |
| Qwen `bench/sweep.py`, `files/build_draft_vocab.py`, `files/evict_page_cache.py`, `files/patch_qsa_fp8_kv.py` | Explicit AGPL-3.0-or-later headers; `files/patch_mtp_draft_vocab.py` also attributes an AGPL-3.0-or-later origin |
| GLM/DeepSeek `overlay/exl3_fat_gemm.{cu,cuh}`, `overlay/exl3_fat_moe.{cu,cuh}`, `overlay/build_exl3_fat_moe_ext.py`; DeepSeek `overlay/e3v2/exl3_fat_moe.{cu,cuh}`, `overlay/row_store.cpp`, `overlay/engram_{file_backend,layout}.py` | No separate permissive grant found for these current files; use AGPL-default optional classification. Including MIT ExLlamaV3 headers does not make these files MIT |
| GLM/DeepSeek `extensions/cooperative_moe/native/{cooperative_moe.cu,cooperative_moe_kernel.cuh,exl3_moe_coop.cuh}` and GLM TP3 counterparts plus `tp3/native/dispatch.cu` | Modified native implementations with retained `LICENSE.exllamav3` and documented MIT upstream origin. Downstream specialization is not proven MIT by that notice. Treat as mixed provenance, optional/blocked for permissive-core reuse pending clarification; preserve MIT notices in any permitted combined work. `cooperative_moe_kernel.cuh` (native and TP3) includes and follows the GEMV kernel, so its open QTIP question applies |
| GLM `extensions/cooperative_moe/tp3/vendor/exllamav3_ext/` headers listed below | Nine byte-identical upstream MIT headers verified against the pinned upstream tree; separable MIT candidates, not evidence that the surrounding TP3 module is MIT. Exception: `quant/exl3_gemv_kernel.cuh` carries the open QTIP provenance question in [Early EXL3 companion](#early-exl3-companion-d-052) |
| GLM `overlay/patch_sparse_mla_slice.py` | Attributes a patch to punkjazz-labs under MIT, but does not provide a separate whole-file license declaration. Resolve source revision, copied portion and later changes; do not promote the entire script to MIT |
| GLM `overlay/patch_flashkda_tp3.py` and `docs/licenses/Apache-2.0-FlashKDA.txt` | Script names external adaptation and vLLM origins; accompanying Apache license text does not prove every adaptation is Apache. External origin and modifications remain adoption blockers for a permissive classification |
| ExLlamaV3 `exllamav3/conversion/standard_cal_data/*.utf8` | Calibration corpora contain third-party text, not just upstream-authored implementation. In `code.utf8`, embedded source carries GPL-2.0 SPDX notices and a separate All Rights Reserved notice. These do not license the entire corpus under either term, and the root MIT grant does not clear the embedded material. Corpus use or redistribution needs its own provenance review; no corpus is cleared here |

The Apache/MIT headers above are observed declarations, not a finding that all
modifications have independently verified grants. In particular GLM and
DeepSeek combine blanket AGPL descriptions with inherited Apache headers;
selecting a file requires resolving that scope against its history. Prefer
retrieving the needed permissive implementation directly from a verified
upstream revision when downstream changes are not required.

`overlay/patch_dflash2.py`, tests such as `test_suppress_stops.py`, and other
patch/fixture builders contain SPDX strings **inside generated content**.
Those strings do not license the enclosing script. Copied patch payloads,
generated source, headers and native binaries follow their actual provenance;
a generator is not an escape from D-017's implementation audit.

The ExLlamaV3 calibration-data exception also matters when packaging: its
`pyproject.toml` includes `conversion/standard_cal_data/*.utf8` as package
data, and `exllamav3/conversion/calibration_data.py` loads that corpus. An
audit of selected MIT kernels does not clear the whole upstream package or
its calibration inputs.

### Verified ExLlamaV3 boundary

The cooperative-kernel READMEs identify upstream kernel origin
[`58d4d7322a1b3bd70aae8412487b21cc5e205cf4`][kernel-origin]. The GLM build
script and DeepSeek archive helper instead pin support headers to
[`02aef45cd681b960a00afcd0749a4ab99e6c1bfe`][header-origin]. These are different
identities. Both revisions' root licenses were fetched and checked as MIT.
The source kernel family is
`exllamav3/exllamav3_ext/quant/exl3_moe_coop.{cu,cuh}` and
`exl3_moe_coop_kernel.cuh`; downstream fixed-shape specializations must not
be represented as unchanged upstream files.

GLM TP3 `PROVENANCE.json` names the header pin and nine SHA-256 hashes.
All nine vendored files matched both those recorded hashes and the actual
upstream Git blobs under `exllamav3/exllamav3_ext/`:

- `compat.cuh`, `util.cuh`, `ptx.cuh`, `util.h`;
- `quant/codebook.cuh`, `quant/hadamard_inner.cuh`, `quant/exl3_dq.cuh`,
  `quant/exl3_gemv_kernel.cuh`, `quant/exl3_kernel_map.cuh`.

The vendored `LICENSE.exllamav3` retains the MIT notice. This comparison proves
identity of those files, not the validity of every archive hash or the whole
TP3 provenance chain. TP3 also ships `LICENSE.upstream-AGPL-3.0` and a
historical `LICENSE.MIT`; those must not be dropped when considering the
combined module. GLM/DeepSeek's Dockerfiles pin other ExLlamaV3 revisions
(`c5d9c657966ffeeaa9353f0cc899f18629da4a13` and
`e648f1a131365aae15920073e761a3fa5a527654`, respectively); their built images
and transitive dependencies were not audited here.

## AGPL in a served process

[AGPLv3 section 13][agpl13] requires a modified covered program that supports
remote network interaction to prominently offer interacting users its
Corresponding Source, with free access through a customary network copying
mechanism. A served modified runtime must account for that obligation even
when no binary is distributed. [Section 1][agpl1] defines the source scope,
including relevant build/install/run scripts and covered dependencies;
[sections 4–6][agpl4] separately govern conveying source and object code.

For a future enabled AGPL backend, review the **actual combined program** and
provide source for the covered version and build, preserving MIT/Apache and
other required notices too. A C ABI, shared library, optional build flag,
sidecar or reverse proxy does not by itself settle whether works are separate
or limit the required source to one kernel. Do not interpret “network” as
only public Internet access. Choose and verify the user-facing source-offer
mechanism before serving that configuration; this inventory does not add an
API or approve an optional-module boundary.

This does not relicense independently authored jitLLM source files. It means
the combined configuration must meet its applicable terms. The copyleft-disabled
profile must exclude the optional implementation's entire source/header/
generator/generated-code/binary closure from fetching and building, while
retaining an independently useful scheduler and allocator (D-002/D-017).
Tools and platform runtimes are recorded separately under D-017, and model
weights remain user-supplied and outside project licensing scope per D-002.

## Adoption gate and verification handoff

Before selecting any of these units, record its exact upstream path/revision,
license and copyright notices, modifications, implementation/tool/platform
role, dependency closure, and shipped notice/source obligations. Resolve the
mixed-provenance cases above or use a verified alternative; neither the EXL3
format name nor a repository's top-level license clears all implementations.
M1 builds the provenance tooling; D-052 advances selected upstream EXL3
integration to M2. Other optional recipes remain future work under D-028.
The early proof changes scheduling, not D-017's license policy.

Builder verification on the x86-64 workstation: clean initial working tree;
read-only Git snapshots of all four listed revisions; tracked-path and notice
inspection; fetched both cooperative upstream revisions; nine header hash
and byte-identity comparisons passed. Documentation links and whitespace were
checked. No runtime behavior changed, so no application tests, builds, Spark
runs or upstream recipe execution were needed. Upstream raw snapshots remain
outside the repository. All changes remain uncommitted for human review.

Independent review on the x86-64 workstation covered the complete uncommitted
documentation change against D-017 and the M0 inventory task. The reviewer
rechecked all four checkout identities and tracked-path counts, scanned
tracked files for SPDX/copyright/license notices, inspected the repository
defaults and historical MIT prefaces, and compared the network-source wording
with the pinned AGPL section 13. This found and fixed an omitted ExLlamaV3
calibration-corpus exception; package-data inclusion and its loader were
checked too. No remaining findings for this planning scope. The separate
adversarial pass challenged historical MIT, modified native files, embedded
SPDX strings, the AGPL service boundary and nine-header upstream identity.
Neither pass clears complete dependency closures, corpus rights, old MIT
revisions or built containers. No runtime tests or Spark runs were warranted
for these documentation-only changes.

[glm]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/tree/c1b7d4c9f98c16af65640fef05fa493bbde52ccf
[ds]: https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/6f7d1590ad49a2b8995188e45d7b9db31e677452
[qwen]: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks/tree/d2f54b78c0d2f9d74ac61aa56200e3c40fac3f22
[exl]: https://github.com/turboderp-org/exllamav3/tree/6b84a21b6f1e5da3f291b9e1019061f0de788279
[kernel-origin]: https://github.com/turboderp-org/exllamav3/tree/58d4d7322a1b3bd70aae8412487b21cc5e205cf4
[header-origin]: https://github.com/turboderp-org/exllamav3/tree/02aef45cd681b960a00afcd0749a4ab99e6c1bfe
[agpl13]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/LICENSE#L540-L559
[agpl1]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/LICENSE#L101-L140
[agpl4]: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/c1b7d4c9f98c16af65640fef05fa493bbde52ccf/LICENSE#L185-L328
