// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/implementations.h"

#include <array>
#include <cstddef>
#include <expected>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/jitllm_ops.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/ggml/validate.h"
#include "kernels/ggml/validate_ext.h"

// The build's part of each identity, from CMakeLists.txt.
#if !defined(JITLLM_GGML_SOURCE_TREE) || !defined(JITLLM_GGML_SDK) ||           \
    !defined(JITLLM_GGML_TARGET) || !defined(JITLLM_GGML_CUDA_ARCHITECTURES) || \
    !defined(JITLLM_GGML_BUILD_TYPE) || !defined(JITLLM_GGML_SANITIZE)
#error "implementations.cc needs the GGML source tree, SDK, target, architectures and build type"
#endif

namespace jitllm::kernels::ggml {

struct RmsNormMulKernel::Entry {
  std::string_view name;
  std::string_view variant;
  std::expected<void, KernelFailure> (*check)(const ggml_tensor* norm, const ggml_tensor* mul);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch, ggml_tensor* norm,
                                            ggml_tensor* mul);
};

struct Kernel::Entry {
  std::string_view name;
  execution::Operation operation;
  std::string_view variant;
  std::size_t arity;
  // Called with exactly `arity` nodes.
  std::expected<void, KernelFailure> (*check)(std::span<const ggml_tensor* const> nodes);
  std::expected<void, KernelFailure> (*run)(LaunchContext& launch,
                                            std::span<ggml_tensor* const> nodes);
};

namespace {

// Whether this build keeps asserts, GGML's device asserts among them.
#ifdef NDEBUG
constexpr std::string_view kAsserts = "NDEBUG";
#else
constexpr std::string_view kAsserts = "asserts";
#endif
// And whether it checks libstdc++'s preconditions (D-083).
#ifdef _GLIBCXX_ASSERTIONS
constexpr std::string_view kLibraryAsserts = "libstdc++ assertions";
#else
constexpr std::string_view kLibraryAsserts = "no libstdc++ assertions";
#endif

constexpr std::array<RmsNormMulKernel::Entry, 2> kRmsNormMul = {{
    {.name = "ggml.rms_norm_mul.fused",
     .variant = "ggml_cuda_op_rms_norm_fused; upstream launch configuration",
     .check = &CheckRmsNormMul,
     .run = &RmsNormMul},
    {.name = "ggml.rms_norm_mul.unfused",
     .variant = "ggml_cuda_op_rms_norm, then ggml_cuda_op_mul; upstream launch configuration",
     .check = &CheckRmsNormThenMul,
     .run = &RmsNormThenMul},
}};

// The other implementations (implementations.h), each checking and running
// its nodes through ops.h.
using Nodes = std::span<ggml_tensor* const>;
using ConstNodes = std::span<const ggml_tensor* const>;

constexpr std::array<Kernel::Entry, 48> kKernels = {{
    {.name = "ggml.rms_norm",
     .operation = execution::Operation::kRmsNorm,
     .variant = "ggml_cuda_op_rms_norm: rms_norm_f32<block, false, false>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRmsNorm(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RmsNorm(launch, n[0]); }},
    {.name = "ggml.add",
     .operation = execution::Operation::kAdd,
     .variant = "ggml_cuda_op_add: k_bin_bcast<op_add, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_ADD); },
     .run = [](LaunchContext& launch, Nodes n) { return Add(launch, n[0]); }},
    {.name = "ggml.mul",
     .operation = execution::Operation::kMul,
     .variant = "ggml_cuda_op_mul: k_bin_bcast<op_mul, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_MUL); },
     .run = [](LaunchContext& launch, Nodes n) { return Mul(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmvf",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_vec_f: mul_mat_vec_f<T, type_acc, ncols, block, false, false> "
                "as upstream selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecF(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmf",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_f: mul_mat_f<T, warp, cols, nwarps, false> as upstream "
                "selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatF(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatF(launch, n[0]); }},
    {.name = "ggml.mul_mat.cublas",
     .operation = execution::Operation::kMatMul,
     .variant = "GGML's cuBLAS path (mul_mat_cublas.cu): conversions, GemmEx, strided or "
                "pointer-array batched GEMM on the lent handle, as upstream plans them",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatCublas(launch, n[0]); }},
    {.name = "ggml.get_rows",
     .operation = execution::Operation::kGetRows,
     .variant = "ggml_cuda_op_get_rows: F32 rows through k_get_rows_float_vec on 16-byte "
                "vectors, aligned rows and at least 128 blocks, else k_get_rows_float; BF16 rows "
                "through k_get_rows_float<nv_bfloat16, float>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGetRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GetRows(launch, n[0]); }},
    {.name = "ggml.set_rows",
     .operation = execution::Operation::kSetRows,
     .variant = "ggml_cuda_op_set_rows: k_set_rows<float, int64_t, half>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSetRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SetRows(launch, n[0]); }},
    {.name = "ggml.rope.neox",
     .operation = execution::Operation::kRope,
     .variant = "ggml_cuda_op_rope: rope_neox<true, false, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRope(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Rope(launch, n[0]); }},
    {.name = "ggml.rope_set_rows.fused",
     .operation = execution::Operation::kRopeSetRows,
     .variant = "ggml_cuda_op_rope_fused: rope_neox<true, false, float, half> writing the KV "
                "destination; upstream launch configuration",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckRopeSetRows(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return RopeSetRows(launch, n[0], n[1]); }},
    {.name = "ggml.soft_max",
     .operation = execution::Operation::kSoftMax,
     .variant = "ggml_cuda_op_soft_max: soft_max_f32<true, ncols, block, float> for 32 to 4,096 "
                "columns in powers of two, else soft_max_f32<true, 0, 0, float>, rows in shared "
                "memory only; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSoftMax(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SoftMax(launch, n[0]); }},
    {.name = "ggml.cont",
     .operation = execution::Operation::kCont,
     .variant = "ggml_cuda_dup: cudaMemcpyAsync if contiguous, cudaMemcpy2DAsync for a pitched "
                "block, else cpy_scalar<cpy_1_scalar<float, float>>; no tiled transpose; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) -> std::expected<void, KernelFailure> {
       if (auto checked = CheckCont(n[0]); !checked) {
         return std::unexpected(checked.error());
       }
       return {};
     },
     .run = [](LaunchContext& launch, Nodes n) { return Cont(launch, n[0]); }},
    {.name = "ggml.swiglu",
     .operation = execution::Operation::kSwiGlu,
     .variant = "ggml_cuda_op_swiglu: unary_gated_op_kernel<op_silu, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSwiGlu(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SwiGlu(launch, n[0]); }},
    {.name = "ggml.convert",
     .operation = execution::Operation::kConvert,
     .variant = "ggml_cuda_cpy between packed tensors: cpy_scalar_contiguous<float, half> or "
                "<half, float>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckConvert(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Convert(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.vec",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ggml_cuda_flash_attn_ext_vec_case<64, F16, F16>, forced: "
                "flash_attn_mask_to_KV_max<ncols> from 1,024 query rows, flash_attn_ext_vec<64, "
                "1 or 2, F16, F16, false>, flash_attn_combine_results<64> over parallel blocks; "
                "launch_fattn's launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnVec(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnVec(launch, n[0]); }},
    {.name = "ggml.mul_mat_add.mmvf_fused",
     .operation = execution::Operation::kMulMatAdd,
     .variant = "ggml_cuda_mul_mat_vec_f with x_bias, writing the add: mul_mat_vec_f<T, "
                "type_acc, 1, block, true, false>, the add's precision; upstream launch "
                "configuration",
     .arity = 2,
     .check = [](ConstNodes n) { return CheckMulMatVecBias(n[0], n[1]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecBias(launch, n[0], n[1]); }},
    {.name = "ggml.mul_mat_glu.mmvf_fused",
     .operation = execution::Operation::kMulMatGlu,
     .variant = "ggml_cuda_mul_mat_vec_f with gate and SwiGLU, writing the GLU: "
                "mul_mat_vec_f<T, type_acc, 1, block, true, false>, the GLU's parameters as "
                "precision; upstream launch configuration",
     .arity = 3,
     .check = [](ConstNodes n) { return CheckMulMatVecGlu(n[0], n[1], n[2]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecGlu(launch, n[0], n[1], n[2]); }},
    // DeepSeek V4 Flash and Qwen3.8 Flash (ops_ext.h).
    {.name = "ggml.mul_mat.mmvq",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_vec_q: quantize_row_q8_1_cuda, then mul_mat_vec_q<type, "
                "ncols_dst> as upstream selects; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecQ(launch, n[0]); }},
    {.name = "ggml.mul_mat.mmq",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_mul_mat_q: quantize_mmq_q8_1_cuda (or the native FP4 quantization for "
                "MXFP4 on Blackwell), then mul_mat_q<type, J, fallback> with J and stream-k as "
                "upstream selects, and its fixup; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatQ(launch, n[0]); }},
    {.name = "ggml.mul_mat.fwht",
     .operation = execution::Operation::kMatMul,
     .variant = "ggml_cuda_op_fwht for GGML_HINT_SRC0_IS_HADAMARD: fwht_cuda<n>, the weights "
                "never read; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatHadamard(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatHadamard(launch, n[0]); }},
    {.name = "ggml.mul_mat_id.mmvq",
     .operation = execution::Operation::kMulMatId,
     .variant =
         "ggml_cuda_mul_mat_vec_q with ids: quantize_row_q8_1_cuda, then mul_mat_vec_q<type, "
         "tokens> over each token's selected experts; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatVecQ(launch, n[0]); }},
    {.name = "ggml.mul_mat_id.mmq",
     .operation = execution::Operation::kMulMatId,
     .variant = "ggml_cuda_mul_mat_q with ids: ggml_cuda_launch_mm_ids_helper, the activations "
                "quantized (or scattered) per selected expert, then mul_mat_q<type, J, fallback> "
                "over each expert's tokens and its fixup; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMulMatIdQ(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return MulMatQ(launch, n[0]); }},
    {.name = "ggml.sub",
     .operation = execution::Operation::kSub,
     .variant = "ggml_cuda_op_sub: k_bin_bcast<op_sub, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_SUB); },
     .run = [](LaunchContext& launch, Nodes n) { return Sub(launch, n[0]); }},
    {.name = "ggml.div",
     .operation = execution::Operation::kDiv,
     .variant = "ggml_cuda_op_div: k_bin_bcast<op_div, float, float, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckBinary(n[0], GGML_OP_DIV); },
     .run = [](LaunchContext& launch, Nodes n) { return Div(launch, n[0]); }},
    {.name = "ggml.scale",
     .operation = execution::Operation::kScale,
     .variant = "ggml_cuda_op_scale: scale_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckScale(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Scale(launch, n[0]); }},
    {.name = "ggml.unary",
     .operation = execution::Operation::kUnary,
     .variant = "ggml_cuda_op_<function> for abs, sgn, neg, silu, tanh, relu, sigmoid, exp, "
                "softplus and sqrt: unary_op_kernel<op_<function>, float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckUnary(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Unary(launch, n[0]); }},
    {.name = "ggml.clamp",
     .operation = execution::Operation::kClamp,
     .variant = "ggml_cuda_op_clamp: op_clamp_kernel<float>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckClamp(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Clamp(launch, n[0]); }},
    {.name = "ggml.fill",
     .operation = execution::Operation::kFill,
     .variant = "ggml_cuda_op_fill: fill_kernel<float or half>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFill(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Fill(launch, n[0]); }},
    {.name = "ggml.repeat",
     .operation = execution::Operation::kRepeat,
     .variant = "ggml_cuda_op_repeat: k_bin_bcast<op_repeat, float, float, float>; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRepeat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Repeat(launch, n[0]); }},
    {.name = "ggml.concat",
     .operation = execution::Operation::kConcat,
     .variant = "ggml_cuda_op_concat: concat_cont<T, dim> per sample for contiguous operands, two "
                "copies along the samples, else concat_non_cont<T, dim>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckConcat(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Concat(launch, n[0]); }},
    {.name = "ggml.sum_rows",
     .operation = execution::Operation::kSumRows,
     .variant = "ggml_cuda_op_sum_rows: reduce_rows_f32<false>, 512 threads per row below two "
                "rows per multiprocessor, else 32 or 128; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSumRows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SumRows(launch, n[0]); }},
    {.name = "ggml.argsort.bitonic",
     .operation = execution::Operation::kArgsort,
     .variant = "ggml_cuda_op_argsort for rows of at most 1,024 that fit shared memory: "
                "k_argsort_f32_i32<order>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckArgsort(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return Argsort(launch, n[0]); }},
    {.name = "ggml.top_k.radix",
     .operation = execution::Operation::kTopK,
     .variant = "ggml_cuda_op_top_k without CUB: top_k_radix_cuda (8-bit radix select, unordered) "
                "for rows over 1,024, else k_argsort_f32_i32<DESC> and a pitched copy of the first "
                "k; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckTopK(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return TopK(launch, n[0]); }},
    {.name = "ggml.swiglu_clamp",
     .operation = execution::Operation::kSwiGluClamp,
     .variant = "ggml_cuda_op_swiglu_clamp: swiglu_clamp_kernel<float>; upstream launch "
                "configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSwiGluClamp(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SwiGluClamp(launch, n[0]); }},
    {.name = "ggml.rope.ext",
     .operation = execution::Operation::kRope,
     .variant = "ggml_cuda_op_rope or ggml_cuda_op_rope_back: rope_norm, rope_neox or rope_multi "
                "<forward, false, float, float> by the node's mode, with its offset and YaRN; "
                "upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckRopeExt(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RopeExt(launch, n[0]); }},
    {.name = "ggml.get_rows.ext",
     .operation = execution::Operation::kGetRows,
     .variant = "ggml_cuda_op_get_rows: k_get_rows<qk, qr, dequantize> for Q8_0, k_get_rows_kq<"
                "dequantize_type> for the k- and i-quants and MXFP4, k_get_rows_float<int32_t, "
                "int32_t> for I32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGetRowsExt(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GetRowsExt(launch, n[0]); }},
    {.name = "ggml.set_rows.ext",
     .operation = execution::Operation::kSetRows,
     .variant = "ggml_cuda_op_set_rows: k_set_rows<src, I32 or I64, dst> for F32 into F32 or F16 "
                "and F16 into F16; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSetRowsExt(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SetRowsExt(launch, n[0]); }},
    {.name = "ggml.ssm_conv",
     .operation = execution::Operation::kSsmConv,
     .variant = "ggml_cuda_op_ssm_conv unfused: ssm_conv_f32<false, 128, conv> up to 32 tokens, "
                "else ssm_conv_long_token_f32<false, 128, conv, 32>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckSsmConv(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return SsmConv(launch, n[0]); }},
    {.name = "ggml.gated_delta_net",
     .operation = execution::Operation::kGatedDeltaNet,
     .variant = "ggml_cuda_op_gated_delta_net: gated_delta_net_cuda<S, KDA, snapshots>; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckGatedDeltaNet(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return GatedDeltaNet(launch, n[0]); }},
    {.name = "ggml.lightning_indexer.wmma",
     .operation = execution::Operation::kLightningIndexer,
     .variant = "ggml_cuda_lightning_indexer on tensor cores: lightning_indexer_kernel_wmma<8, 32, "
                "128, heads, F16>; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckLightningIndexer(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return LightningIndexer(launch, n[0]); }},
    {.name = "ggml.dsv4_hc_comb",
     .operation = execution::Operation::kHcComb,
     .variant = "ggml_cuda_op_dsv4_hc_comb: dsv4_hc_comb_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcComb(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return HcComb(launch, n[0]); }},
    {.name = "ggml.dsv4_hc_pre",
     .operation = execution::Operation::kHcPre,
     .variant = "ggml_cuda_op_dsv4_hc_pre: dsv4_hc_pre_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcPre(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return HcPre(launch, n[0]); }},
    {.name = "ggml.dsv4_hc_post",
     .operation = execution::Operation::kHcPost,
     .variant = "ggml_cuda_op_dsv4_hc_post: dsv4_hc_post_f32; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckHcPost(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return HcPost(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.mma",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ggml_cuda_flash_attn_ext_mma_f16_case<D, D, 1, 2, 4 or 8, 8> for D 256 and 512 as "
                "switch_ncols1 picks, the sparse gather at D 512 as upstream decides, launch_fattn "
                "with stream-k and its fixup; upstream launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnMma(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnMma(launch, n[0]); }},
    {.name = "ggml.flash_attn_ext.mma_d128",
     .operation = execution::Operation::kFlashAttn,
     .variant = "ggml_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 16, 32 or 64, 1> as "
                "switch_ncols1 picks, no mask, launch_fattn with stream-k and its fixup; upstream "
                "launch configuration",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckFlashAttnMma128(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return FlashAttnMma128(launch, n[0]); }},
    // jitLLM's own (jitllm_ops.h), for Qwen3.8's MXFP8 and NVFP4 tensors.
    {.name = "jitllm.mxfp8.mul_mat_vec",
     .operation = execution::Operation::kMatMul,
     .variant = "Mxfp8Gemv<columns 1-8>: one warp a row, 16-code vectors, F32 block sums",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8MulMatVec(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8MulMatVec(launch, n[0]); }},
    {.name = "jitllm.mxfp8.dequant",
     .operation = execution::Operation::kConvert,
     .variant = "Mxfp8ToBf16: sixteen codes a thread",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckMxfp8Dequant(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunMxfp8Dequant(launch, n[0]); }},
    {.name = "jitllm.nvfp4.get_rows",
     .operation = execution::Operation::kGetRows,
     .variant = "Nvfp4RowsKernel: a block an id, a thread a value",
     .arity = 1,
     .check = [](ConstNodes n) { return CheckNvfp4Rows(n[0]); },
     .run = [](LaunchContext& launch, Nodes n) { return RunNvfp4Rows(launch, n[0]); }},
}};

execution::Implementation Declare(std::string_view name, execution::Operation operation,
                                  std::string_view variant) {
  return {
      .name = std::string(name),
      .operation = operation,
      .source = "ggml",
      .revision = std::format("ggml tree {}; jitllm module {}", JITLLM_GGML_SOURCE_TREE,
                              ModuleSourcesDigest()),
      .build = std::format("sdk {}; target {}; cuda {}; build type {}; {}; {}; sanitizers {}",
                           JITLLM_GGML_SDK, JITLLM_GGML_TARGET, JITLLM_GGML_CUDA_ARCHITECTURES,
                           JITLLM_GGML_BUILD_TYPE, kAsserts, kLibraryAsserts, JITLLM_GGML_SANITIZE),
      .variant = std::string(variant)};
}

execution::Implementation Declare(const RmsNormMulKernel::Entry& entry) {
  return Declare(entry.name, execution::Operation::kRmsNormMul, entry.variant);
}

execution::Implementation Declare(const Kernel::Entry& entry) {
  return Declare(entry.name, entry.operation, entry.variant);
}

}  // namespace

std::vector<execution::Implementation> Implementations() {
  std::vector<execution::Implementation> declared;
  declared.reserve(kRmsNormMul.size() + kKernels.size());
  for (const RmsNormMulKernel::Entry& entry : kRmsNormMul) {
    declared.push_back(Declare(entry));
  }
  for (const Kernel::Entry& entry : kKernels) {
    declared.push_back(Declare(entry));
  }
  return declared;
}

std::expected<RmsNormMulKernel, KernelFailure> RmsNormMulKernel::Bind(
    const execution::Implementation& implementation) {
  for (const Entry& entry : kRmsNormMul) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return RmsNormMulKernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML rms_norm_mul implementation of this build",
                            implementation.name)});
}

std::expected<void, KernelFailure> RmsNormMulKernel::Check(const ggml_tensor* norm,
                                                           const ggml_tensor* mul) const {
  return entry_->check(norm, mul);
}

std::expected<void, KernelFailure> RmsNormMulKernel::Run(LaunchContext& launch, ggml_tensor* norm,
                                                         ggml_tensor* mul) const {
  return entry_->run(launch, norm, mul);
}

std::string_view RmsNormMulKernel::name() const { return entry_->name; }

std::expected<Kernel, KernelFailure> Kernel::Bind(const execution::Implementation& implementation) {
  for (const Entry& entry : kKernels) {
    if (entry.name != implementation.name) {
      continue;
    }
    if (execution::IdentityOf(Declare(entry)) != execution::IdentityOf(implementation)) {
      break;
    }
    return Kernel(entry);
  }
  return std::unexpected(KernelFailure{
      .error = KernelError::kRejected,
      .detail = std::format("{} is not a GGML implementation of this build", implementation.name)});
}

std::expected<void, KernelFailure> Kernel::Check(std::span<const ggml_tensor* const> nodes) const {
  if (nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->check(nodes);
}

std::expected<void, KernelFailure> Kernel::Run(LaunchContext& launch,
                                               std::span<ggml_tensor* const> nodes) const {
  if (nodes.size() != entry_->arity) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail =
            std::format("{} takes {} nodes, not {}", entry_->name, entry_->arity, nodes.size())});
  }
  return entry_->run(launch, nodes);
}

std::string_view Kernel::name() const { return entry_->name; }

execution::Operation Kernel::operation() const { return entry_->operation; }

std::size_t Kernel::arity() const { return entry_->arity; }

}  // namespace jitllm::kernels::ggml
