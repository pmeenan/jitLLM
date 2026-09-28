<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen-Image-2.1 BF16, native (M3)

M3's third model slice ([plan](../../plan.md#m3--single-spark-fast-full-swap-in-progress)):
the Qwen-Image-2.1 pipeline (Qwen3-VL-8B text encoder, single-stream DiT,
VAE) run by jitLLM natively on one Spark from its D-056 artifacts, one per
component, joined by a D-089 composition, against diffusers in BF16.

## Pre-registration (fixed before jitLLM's first run on the model)

Inputs, fixed in the repository (D-087's entry rule):

- **Checkpoint:** `Qwen/Qwen-Image-2.1@790c9263`, pinned in
  [fast-swap/pins.json](../fast-swap/pins.json).
- **Oracle:** diffusers `8b3c707e` in the image-reference container
  ([image-reference](../image-reference/README.md), local image
  `sha256:700d6668…`, PyTorch 2.14.0+cu130), BF16 on the GPU, the fast-swap
  prompt set's `image` entry: the teapot prompt, 1024×1024, 40 steps, seed
  42, `true_cfg_scale` 1.0 (no guidance), prefix KV cache on.
- **Reference tensors:** [reference.py](reference.py) re-runs the pipeline's
  denoising loop by hand, in the same calls, order and dtypes as
  `QwenImage21Pipeline.__call__`, and keeps every tensor jitLLM is compared
  with. Its image is pixel for pixel the baseline's (RGB SHA-256
  `7d00b052…`, [baselines.md](../fast-swap/baselines.md#qwen-image-21-diffusers-bf16)),
  so the kept tensors are the baseline's. They stay outside Git on `spark`
  under `~/.local/share/jitllm/references/qwen-image-2.1/native/ref1/`
  (raw little-endian files; `reference.json` records each one's shape and
  SHA-256).
- **Seeded noise:** jitLLM does not reproduce PyTorch's CUDA generator. It
  starts from diffusers' initial latents for seed 42 (`latents_init`, the
  packed `randn_tensor` output), the simpler of the two routes the slice
  allowed.
- **Tokens:** jitLLM's native tokenizer and prompt renderer
  (`src/tokenizer`, `src/chat`) must give the
  reference's 39 token IDs exactly, and drop the 14 system-turn tokens.

**Calibration** (the scale of "small", from the same reference run on
`spark`, 2026-09-28): each component re-run in FP32 on the same inputs,
the pipeline with its DiT in FP32 over all 40 steps, the pipeline's own
cache-off sample and an unrelated sample (seed 43):

| Control | Result |
| --- | --- |
| Text encoder, BF16 vs FP32 (kept hidden states) | rel. RMS 0.101, cosine 0.99495 (a few massive-activation channels dominate) |
| DiT step 0, BF16 vs FP32 (same inputs) | rel. RMS 0.0063, cosine 0.999980 |
| DiT step 1, BF16 vs FP32 (same inputs) | rel. RMS 0.0229, cosine 0.999771 |
| VAE, BF16 vs FP32 (same latents), decoded tensor | rel. RMS 0.0027 |
| VAE, BF16 vs FP32, image | PSNR 53.5 dB, SSIM 0.9985 |
| Whole pipeline, DiT in FP32, image | PSNR 38.8 dB, SSIM 0.9922 |
| Prefix cache off, image | pixel-identical |
| Seed 43 instead of 42, image | PSNR 11.6 dB, SSIM 0.733 |

PSNR is over 8-bit RGB, SSIM the image-gguf study's 11×11 Gaussian
luminance SSIM ([compare.py](../image-gguf/compare.py)); relative RMS is
‖a − b‖ / ‖b‖ with the diffusers BF16 tensor as b.

**Bounds.** A correct BF16 re-implementation rounds differently from
diffusers, roughly as far as diffusers' BF16 is from FP32, so each bound
allows twice the measured BF16-versus-FP32 distance (twice the relative
RMS, twice the SSIM deficit, 6 dB below the PSNR), rounded outward:

1. **Tokens:** the reference's IDs exactly.
2. **Text encoder** (native tokens, the kept 25 rows × 4,096): relative
   RMS ≤ 0.20 and cosine ≥ 0.99 against diffusers' `prompt_embeds`.
3. **DiT, first step** (diffusers' embeddings and initial latents as
   inputs): relative RMS ≤ 0.0125 and cosine ≥ 0.99996 against diffusers'
   step-0 noise prediction.
4. **DiT, every step, teacher-forced** (each step fed diffusers' latents
   for that step): relative RMS ≤ 0.046 per step (twice step 1's
   calibration, the larger).
5. **VAE** (diffusers' final latents decoded): decoded tensor relative RMS
   ≤ 0.0055, image PSNR ≥ 47.5 dB against the reference image.
6. **Image, end to end** (native tokens and text encoder, the DiT's own
   trajectory from the seed's latents, the VAE): **PSNR ≥ 32.0 dB and SSIM
   ≥ 0.98** against the reference image. The unrelated-sample control
   (11.6 dB, 0.733) is far outside; the FP32-DiT control (38.8 dB, 0.9922)
   well inside.

Performance and memory are reported beside diffusers' (plan.md's image
criterion: full generation not more than 10% slower than diffusers'
52.6 s, D-085): per-step time, full generation (text encode, 40 steps, VAE
decode, with the weights resident as the baseline holds them) over plain
runs, and peak memory as the drop in `MemAvailable`.

## What runs

- **Artifacts** (D-089, [artifact-format.md](../../artifact-format.md#compositions)):
  [import_m3.py](../artifact-layout/import_m3.py) `component` imported the
  three components from the pinned checkpoint on `spark` (text encoder
  92 s, denoiser 75 s, VAE 6 s, full verify included) and `compose` wrote
  composition `eca21baad38229e471a44cb2479d392ffcf745fb812e8a41668f336139fa1acd`
  naming text encoder `ed89ed27…`, denoiser `d1184efd…` and VAE
  `44c1a20a…`, in `~/.local/share/jitllm/m3-artifacts/`.
- **Load** (`jitllm_qwen_image_exec`): the composition and each component
  opened as untrusted input (`artifact/composition.h`, `artifact.h`),
  bound to the compiled-in profile (`model/qwen_image.h`), and the groups
  a phase reads loaded with direct reads through pinned staging into
  `cudaMalloc` memory: the text encoder's table and 36 layers (15.14 GB,
  not its vision tower or `lm_head`), the denoiser (14.23 GB), the VAE's
  decoder (1.35 GB of F32, rounded to BF16 on the host as
  `from_pretrained(torch_dtype=bfloat16)` does, 0.51 GB on the device).
- **Phases:** encode (Qwen3-VL's text path, 39 tokens, the 14 of the system
  turn dropped), denoise (40 steps of the block-causal DiT: the first over
  the joint sequence of 25 text and 4,096 image rows, filling the text's
  K/V prefix cache, the rest over the image rows only), decode (the VAE
  decoder, one frame). `--phases released` loads each component at its
  phase's start and frees it at the end; `resident` holds all three.
- **Kernels** ([kernels/image](../../../src/kernels/image/ops.h)): BF16
  products on cuBLAS (`cublasGemmEx`, F32 accumulation, the call PyTorch
  makes for a BF16 `nn.Linear`), jitLLM's FlashAttention-2 forward in BF16
  for the denoiser's attention, and jitLLM's fused BF16 kernels for the
  norms, modulation, rotary embeddings, gated residuals, SwiGLU, the Euler
  step and the VAE's operations (im2col and cuBLAS for its convolutions).
  Each kernel rounds to BF16 where the pinned PyTorch code materializes a
  BF16 tensor, so fusing does not change the values: the scheduler's step,
  for one, rounds `dt` to BF16 before multiplying (0 of 10.2 million values
  differ from diffusers' steps that way, 287,316 with an F32 `dt`).

## Results (`spark`, 2026-09-28)

GB10, driver 580.178.04, the SDK's CUDA and cuBLAS (D-076). Raw outputs in
`~/.local/share/jitllm/m3img-20260928/` on `spark` (the runs `full1`,
`first`, `forced`, `vae` and `released`, and compare.py's verdicts in
`verdicts.json`). A first attempt at the reference run exhausted `spark`'s
memory at 03:39 (its FP32 controls with the BF16 pipeline still resident);
the kernel's OOM killer ended it and several system services (tailscaled,
polkit, fwupd, the DGX telemetry, the user session), which systemd
restarted within two minutes. The reference was re-run with reference.py
as checked in and finished at 03:52; every jitLLM run here started after
04:00 on the recovered host, and diffusers' full-generation and per-step
timings are the earlier baseline's, so no measurement overlaps the
incident.

**Correctness** ([compare.py](compare.py), bounds 1–6): every bound passes.

| Check | Result | Bound |
| --- | --- | --- |
| 1. Tokens | 39 of 39 equal, 14 dropped | exact |
| 2. Text encoder | rel. RMS 0.0437, cosine 0.99905 | ≤ 0.20, ≥ 0.99 |
| 3. DiT, first step | rel. RMS 0.00644, cosine 0.999980 | ≤ 0.0125, ≥ 0.99996 |
| 4. DiT, 40 teacher-forced steps | worst rel. RMS 0.0094 (step 39), median 0.0041 | ≤ 0.046 |
| 5. VAE on diffusers' latents | rel. RMS 0.0035; PSNR 55.9 dB, SSIM 0.9992 | ≤ 0.0055, ≥ 47.5 dB |
| 6. **Image, end to end** | **PSNR 41.8 dB, SSIM 0.99604** | ≥ 32.0 dB, ≥ 0.98 |

End to end, jitLLM's final latents are 0.0174 from diffusers' in relative
RMS, closer than the FP32-DiT control's 0.0254; its image and the
reference, inspected side by side at 512², show the same red teapot, pose
and framing with no visible difference. With `--phases released` the pixels are identical to the
resident run's.

**Performance and memory** (reported; plan.md's image gate is ≤ 10% slower
than diffusers, D-085):

| | jitLLM | diffusers ([baselines](../fast-swap/baselines.md#qwen-image-21-diffusers-bf16)) | Ratio |
| --- | --- | --- | ---: |
| Full generation, weights resident (plain runs) | 36.94 / 36.99 s | 52.55 / 52.71 / 52.73 s | 0.70 |
| Per denoising step, median (synchronized) | 0.893 s [0.885–0.900] | 1.259 s [1.255–1.263] | 0.71 |
| First step (with the text's K/V prefill), from the loop's start | 0.889 s | 1.753 s | 0.51 |
| Text encode / VAE decode | 0.137 s / 1.08 s | 1.95 s / 1.46 s (this study's reference run) | |
| Peak memory, drop in `MemAvailable`, resident | 31.2 GiB | 43.4 GiB | 0.72 |
| Full generation, components released outside their phases | 39.89 / 40.15 s (loads 1.59 + 1.51 + 0.83 s, warm page cache) | | 0.76 |
| Peak memory, released | 15.9 GiB | | 0.37 |

Every run's image is the same pixels (RGBA SHA-256 `95fbcbc5…`), resident
or released, first or plain.

Where a step's time goes (nsys, steps 0–2, per image-row step): the 224
products 0.69 s (cuBLAS picks `nvjet` kernels for the MLP's and a CUTLASS
kernel at 1.76 ms for each 4,096³ one, where PyTorch's cuBLASLt path takes
`nvjet` at about 1.54 ms), attention 0.108 s (3.37 ms per block; PyTorch's
`flash_fwd_kernel` 3.38 ms), everything else about 0.10 s (diffusers'
unfused elementwise kernels: about 0.38 s per step, from its torch.profiler
table of steps 0 and 1).

**Attention, chosen by speed** (D-053): GGML's tensor-core kernel covers
D = 128 with one query head per KV head once its ncols2 = 1 instances are
built (`kernels/ggml/fattn_mma_d128.cu`, `--attention ggml`; RE-030's sinks
issue does not arise without sinks) and matches FP64 within upstream's
bound, but takes 19.9 ms per block call here (0.64 s per step, 64 columns
per tile, stream-k), about 14 TFLOPS. jitLLM's own FlashAttention-2 kernel
(`kernels/image/flash_attention.cu`, BF16 `mma.sync`) takes 3.37 ms, as
PyTorch's does, so neither cuDNN (whose license D-017 would first have to
admit) nor a cuBLAS formulation was needed.

## Judgement calls

- **Components as separate artifacts plus a composition** (D-089)
  rather than one artifact: no change to v0 artifacts or the pinned
  planner, and a shared text encoder is one artifact.
- **Importer policy:** `import_m3.py` runs the pinned `layout.py` with a
  per-component layer pattern (the language layers, the transformer
  blocks, the VAE decoder's up blocks) and the diffusers `_class_name` as
  architecture, both patched into its module instance; GGUF imports write
  exactly what `m3-1` wrote and keep its converter version.
- **Seeded noise from diffusers' saved latents** rather than a
  reimplementation of PyTorch's Philox generator.
- **Own kernels for the elementwise and norm operations** and for
  attention: GGML computes the former in F32 between operations where the
  reference rounds each to BF16, and its attention was 6× slower here.
- **Bounds from calibration:** twice the measured BF16-versus-FP32 distance
  per component, fixed before jitLLM's first run.

## Limits

- One prompt, size (1024²), step count and seed; text-to-image only (no
  condition images, no guidance, the prefix cache on). Other sizes are
  accepted (multiples of 32, 64–2,048) but not compared: one 512², 8-step
  run from seeded noise of jitLLM's own (not diffusers') gave a coherent
  image, and the attention kernel's tests cover that size's sequence
  lengths.
- The phases here run on `cudaMalloc` memory. On the paged node each phase
  is a device job over its own component's closure, a generation is one
  request leasing all three components once (D-093), and the image is
  this harness's pixel for pixel (RGBA SHA-256 `95fbcbc5…`,
  [swap](../fast-swap/swap.md#qwen-image-21-on-the-paged-node)). In both,
  the kernels are called directly: the image path's operations are not yet
  declared in the registry or dispatched through a bound plan (D-053,
  D-086).
- The VAE's convolutions use im2col (0.47 s of the 1.07 s decode).
- Load times are with a warm page cache.

## Reproduce

On `spark`, with the checkpoint in the M3 model store and a build of
`jitllm_qwen_image_exec` (`benchmarks/`):

```sh
M=$HOME/.local/share/jitllm/models/Qwen/Qwen-Image-2.1@790c9263
S=$HOME/.local/share/jitllm/m3-artifacts
cd docs/experiments/artifact-layout
for role in text_encoder transformer vae; do
  python3 import_m3.py component $S ../fast-swap/pins.json qwen-image-2.1 $M $role \
    $([ $role = text_encoder ] && echo --meta text_encoder/generation_config.json)
done
python3 import_m3.py compose $S ../fast-swap/pins.json qwen-image-2.1 $M \
  text_encoder=<id> transformer=<id> vae=<id> --meta scheduler/scheduler_config.json \
  $(cd $M && for f in processor/*; do echo --meta $f; done)
# The reference (about 11 minutes, in the image-reference container, under
# torch.no_grad; 51 GB of CUDA allocations at its peak). Check MemAvailable
# first; --memory caps the container's host allocations:
sudo -n docker run --rm --memory 96g --device nvidia.com/gpu=all --network none -u $(id -u):$(id -g) \
  -e HOME=/tmp -v $M:/model:ro -v $PWD/..:/exp:ro -v $REF_PARENT:/out \
  jitllm-image-reference:20260922 /exp/qwen-image-native/reference.py /model /out/ref1 \
  /exp/fast-swap/prompts.json
# jitLLM: end to end (with two plain timed runs), then each component alone.
B=build/spark-native/benchmarks/jitllm_qwen_image_exec; C=eca21baa...; R=$REF_PARENT/ref1
$B --store $S --composition $C --out full --reference $R --runs 2
$B --store $S --composition $C --out first --reference $R --embeds reference --stop-after 1 --no-vae
$B --store $S --composition $C --out forced --reference $R --embeds reference --force-latents --no-vae
$B --store $S --composition $C --out vae --reference $R --embeds reference --stop-after 1 --decode-reference
$B --store $S --composition $C --out released --reference $R --phases released --runs 1
# The verdicts, in the image-reference container:
python3 compare.py $R --tokens full --text full --image full --dit-first first \
  --dit-forced forced --vae vae
```
