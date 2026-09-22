<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Licensing and reference provenance

jitLLM-authored code is Apache-2.0. [D-002, D-003 and D-017](decisions.md)
govern incorporation, optional modules, tools and platform dependencies.
This M0 inventory, checked on **2026-09-22**, is evidence for the first dense
slice, early EXL3 proof and later optional-backend design, not approval to import these repositories or a
license audit of their built containers. No implementation was incorporated.

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
| GLM `overlay/ablit_runtime.py` | File starts with `SPDX-License-Identifier: MIT`. Explicit MIT candidate; this does not cover its checkpoint inputs or the surrounding importer/launcher |
| GLM `overlay/exl3.py`, `overlay/dflash2_speculator.py`, `overlay/qwen3_dflash2.py` | File headers declare Apache-2.0; do not label these MIT merely because they integrate EXL3 |
| GLM `overlay/tp3/vllm/model_executor/parameter.py`, `overlay/tp3/vllm/model_executor/layers/vocab_parallel_embedding.py`, `overlay/tp3/vllm/model_executor/model_loader/weight_utils.py`, `overlay/tp3/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py` | Apache-2.0 headers with vLLM attribution; modified overlay provenance still needs review before adoption |
| GLM `tests/fixtures/flashinfer_mla_sparse_sm120-487ecf187.py.txt`, `tests/fixtures/kda-487ecf187.py.txt` | Apache-2.0 file headers; fixtures are also incorporated source if copied |
| GLM `tests/test_gen_defaults.py` | Describes an embedded vLLM fixture as Apache-2.0; that notice does not independently license the entire test harness |
| DeepSeek `overlay/exl3.py` | Apache-2.0 file header, distinct from the AGPL-default patch scripts and native overlays |
| Qwen `files/ple_layer_patched.py`; `files/ple_offload/{connector,ple_offload_layer,protocol,worker}.py`; `files/ple_offload/orig/{connector,ple_offload_layer,protocol,worker}.py`; `tp1/files/ple_offload/orig/{connector,ple_offload_layer,protocol,worker}.py` | Apache-2.0/vLLM headers; README expressly preserves file-specific SPDX terms under `files/`. Record headers, but audit modifications and the `tp1/` copies before adoption |
| Qwen `bench/sweep.py`, `files/build_draft_vocab.py`, `files/evict_page_cache.py`, `files/patch_qsa_fp8_kv.py` | Explicit AGPL-3.0-or-later headers; `files/patch_mtp_draft_vocab.py` also attributes an AGPL-3.0-or-later origin |
| GLM/DeepSeek `overlay/exl3_fat_gemm.{cu,cuh}`, `overlay/exl3_fat_moe.{cu,cuh}`, `overlay/build_exl3_fat_moe_ext.py`; DeepSeek `overlay/e3v2/exl3_fat_moe.{cu,cuh}`, `overlay/row_store.cpp`, `overlay/engram_{file_backend,layout}.py` | No separate permissive grant found for these current files; use AGPL-default optional classification. Including MIT ExLlamaV3 headers does not make these files MIT |
| GLM/DeepSeek `extensions/cooperative_moe/native/{cooperative_moe.cu,cooperative_moe_kernel.cuh,exl3_moe_coop.cuh}` and GLM TP3 counterparts plus `tp3/native/dispatch.cu` | Modified native implementations with retained `LICENSE.exllamav3` and documented MIT upstream origin. Downstream specialization is not proven MIT by that notice. Treat as mixed provenance, optional/blocked for permissive-core reuse pending clarification; preserve MIT notices in any permitted combined work |
| GLM `extensions/cooperative_moe/tp3/vendor/exllamav3_ext/` headers listed below | Nine byte-identical upstream MIT headers verified against the pinned upstream tree; separable MIT candidates, not evidence that the surrounding TP3 module is MIT |
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
