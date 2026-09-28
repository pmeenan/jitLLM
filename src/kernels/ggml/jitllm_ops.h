// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// jitLLM's own operations on GGML tensors, for the formats GGML has no type
// for (M3, Qwen3.8 Flash Next's ModelOpt checkpoint; the A/B that chose them
// is docs/experiments/qwen38-native/README.md):
//
//   jitllm.mxfp8.mul_mat_vec  y[n, t] = W[n, :] · x[:, t] for up to 8
//                             columns t, W in MXFP8 (E4M3 codes, one E8M0
//                             scale per 32 along k), each 32-element block's
//                             dot product in F32, then scaled, the blocks
//                             summed in F32; memory-bound decode products.
//   jitllm.mxfp8.dequant      W as BF16 [k, n], exactly (an E4M3 value times
//                             a power of two) wherever that is a BF16 value,
//                             for GGML's float products (cuBLAS) over wider
//                             batches. E8M0 0xFF and E4M3 0x7F/0xFF are NaN.
//   jitllm.nvfp4.get_rows     rows of a table in ModelOpt NVFP4, each row its
//                             v/2 code bytes (element 2i in the low nibble)
//                             then its v/16 E4M3 scales, times the table's
//                             global F32 scale, as F32 [v, ids]: Qwen3.8's
//                             n-gram embedding lookup.
//
// and, for Qwen3.8's prefill, fusions of the elementwise work the
// hyper-connections and the MoE output do at four streams' width, each the
// same arithmetic as the GGML nodes it replaces (qwen38_graph.cc), in the
// same order and with GGML's device flags (-use_fast_math), so that its
// result equals theirs bit for bit (tests/unit/qwen38_ops_test.cc):
//
//   jitllm.hc.combine   res[c, k, t] + out[c, t] · (2 · sigmoid(inj[k, t] / hc)):
//                       build_hc_combine's sigmoid, scales, repeat, mul
//                       and add;
//   jitllm.hc.norm      rms_norm over each stream's `width` elements, then
//                       times the norm weight, in F32 or rounded to BF16
//                       (the input of the mixer's products);
//   jitllm.hc.mix       build_hc_mix's gate and fold: the norm again (each
//                       stream's scale recomputed, as GGML's rms_norm
//                       reduces), times the weight and sigmoid(g), summed
//                       over the streams in order, times 1 / hc;
//   jitllm.moe.glu      silu(gate · s_gate[e]) · (up · s_up[e]): the
//                       experts' global scales and SwiGLU;
//   jitllm.moe.combine  Σ_i (down_i · s_down[e_i]) · w_i in expert order,
//                       plus shared · sigmoid(shared gate): the routed
//                       experts' weighted sum and the gated shared expert;
//   jitllm.bf16         F32 to BF16, rounded to nearest (as GGML converts
//                       cuBLAS operands);
//   jitllm.gemm.bf16    y[n, t] = W[k, n] · x[k, t] with BF16 W and x, F32
//                       out, through the lent cuBLAS handle: the call GGML's
//                       cuBLAS product makes for BF16 weights after
//                       converting F32 activations, so one conversion
//                       serves every product of an input.
//
// Each is a GGML_OP_CUSTOM node (ggml_custom_4d) whose function pointer
// names the operation; the function itself is never called (GGML's CPU
// backend never runs these graphs). The builders make the nodes; the plan
// (graph_plan.h) names the implementation from the kind; the checks here
// are what each implementation refuses on the host before anything is
// queued (D-086), and the launchers are in jitllm_ops.cu.
//
// Byte tensors are GGML_TYPE_I8: MXFP8 codes [k, n] and scales [k/32, n],
// NVFP4 table rows [v/2 + v/16, rows].

#ifndef JITLLM_KERNELS_GGML_JITLLM_OPS_H_
#define JITLLM_KERNELS_GGML_JITLLM_OPS_H_

#include <cstdint>
#include <expected>

#include "ggml.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

enum class JitllmOp : std::uint8_t {
  kNone,
  kMxfp8MulMatVec,
  kMxfp8Dequant,
  kNvfp4Rows,
  kHcCombine,
  kHcNorm,
  kHcMix,
  kMoeGlu,
  kMoeCombine,
  kBf16,
  kGemmBf16,
  kMoeRoute,
  kMoeQuantize,
  kMoeGemm,
  kMoeGluQuantize,
  kMoeCombineSorted,
  kMoeGemv,
  kGdnConv,
  kGdnNormGate,
};

// The operation a GGML_OP_CUSTOM node names, or kNone.
JitllmOp JitllmOpOf(const ggml_tensor* node);
// The norms' epsilon, which their builders store after GGML's custom
// parameters.
float JitllmOpEps(const ggml_tensor* node);
// The routed-expert operations' integer parameters, stored in the same
// place: index 0 to 7.
std::int32_t JitllmOpInt(const ggml_tensor* node, int index);
float JitllmOpFloat(const ggml_tensor* node, int index);

// The most columns jitllm.mxfp8.mul_mat_vec takes.
inline constexpr std::int64_t kMxfp8VecColumns = 8;

// Builders. `codes` I8 [k, n], `scales` I8 [k / 32, n], `x` F32 [k, t].
ggml_tensor* Mxfp8MulMatVec(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales,
                            ggml_tensor* x);
ggml_tensor* Mxfp8Dequant(ggml_context* context, ggml_tensor* codes, ggml_tensor* scales);
// `table` I8 [values / 2 + values / 16, rows], `ids` I32 [n], `scale` F32
// [1]: F32 [values, n].
ggml_tensor* Nvfp4Rows(ggml_context* context, ggml_tensor* table, ggml_tensor* ids,
                       ggml_tensor* scale, std::int64_t values);

// The fusions. `res` F32 [width, hc, t], `out` F32 [width, t], `inject` F32
// [hc, t]: F32 [width, hc, t].
ggml_tensor* HcCombine(ggml_context* context, ggml_tensor* res, ggml_tensor* out,
                       ggml_tensor* inject);
// `x` F32 [width, hc, t], `weight` F32 [width · hc]: [width · hc, t] of
// `type` (F32 or BF16).
ggml_tensor* HcNorm(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, float eps,
                    ggml_type type);
// `x` and `weight` as HcNorm's, `gate` F32 [width · hc, t] (before its
// sigmoid): F32 [width, t].
ggml_tensor* HcMix(ggml_context* context, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* gate,
                   float eps);
// `gate` and `up` F32 [n, used, t] (the experts' raw products), `ids` I32
// [used, t] (rows may be strided), `gate_scale` and `up_scale` F32
// [experts]: F32 [n, used, t].
ggml_tensor* MoeGlu(ggml_context* context, ggml_tensor* gate, ggml_tensor* up, ggml_tensor* ids,
                    ggml_tensor* gate_scale, ggml_tensor* up_scale);
// `down` F32 [width, used, t], `ids` as MoeGlu's, `down_scale` F32
// [experts], `weights` F32 [1, used, t], `shared` F32 [width, t],
// `shared_gate` F32 [1, t] (before its sigmoid): F32 [width, t].
ggml_tensor* MoeCombine(ggml_context* context, ggml_tensor* down, ggml_tensor* ids,
                        ggml_tensor* down_scale, ggml_tensor* weights, ggml_tensor* shared,
                        ggml_tensor* shared_gate);
// `x` F32: BF16 of its shape.
ggml_tensor* ToBf16(ggml_context* context, ggml_tensor* x);
// `weights` BF16 [k, n], `x` BF16 [k, t]: F32 [n, t].
ggml_tensor* GemmBf16(ggml_context* context, ggml_tensor* weights, ggml_tensor* x);

// The host checks: operands bound, typed and shaped as above, packed where
// the kernels read them with vector loads (codes and x rows 16-byte
// aligned), extents within the kernels' 32-bit indexing, and outputs
// disjoint from their operands. The row lookup's ids are not read here:
// the caller builds them within the table (model/qwen38.h's n-gram hash
// over offsets and sizes the binding checked); the kernel writes NaN for
// an id outside the table rather than read out of bounds.
std::expected<void, KernelFailure> CheckMxfp8MulMatVec(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMxfp8Dequant(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckNvfp4Rows(const ggml_tensor* node);
// The fusions' checks: every operand bound, typed and shaped as its
// builder's, packed (the expert ids may have a longer row stride), 16-byte
// aligned where a kernel loads four floats, within 32-bit grids, and the
// output disjoint from every operand. The expert ids are not read here: the
// kernels write NaN for an id outside the scales rather than read out of
// bounds. The norms take streams of at least 1,024 elements (GGML's
// rms_norm reduces those over 1,024 threads, which the kernels repeat).
std::expected<void, KernelFailure> CheckHcCombine(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcNorm(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckHcMix(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGlu(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeCombine(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckBf16(const ggml_tensor* node);
// And the product's: cuBLAS's int extents, and both operands' rows k
// elements apart (packed).
std::expected<void, KernelFailure> CheckGemmBf16(const ggml_tensor* node);

// jitllm.gated_delta_net.columns: a second implementation of GGML's
// gated_delta_net node (the GGML implementation is ggml.gated_delta_net,
// ops_ext.h), the same arithmetic per value column as upstream's kernel at
// 128-wide heads, with several columns a warp (jitllm_fused.cu), for one
// sequence, a scalar gate and no state snapshots. Its checks are GGML's
// (validate_ext.h CheckGatedDeltaNet) and those restrictions;
// GatedDeltaNetColumnsFits says whether the node has that shape (the plan's
// structural choice).
bool GatedDeltaNetColumnsFits(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGatedDeltaNetColumns(const ggml_tensor* node);
// jitllm.gated_delta_net.lanes: the same recurrence with each value column
// split over 8 lanes of 16 rows, for prefill (more than
// kGatedDeltaNetLanesTokens tokens; the plan's structural choice): fewer
// shuffles, its F32 sums in another order than upstream's, so not bit for
// bit (NMSE against FP64 as upstream's; tests/unit/qwen38_fused_test.cc).
// Its checks are the columns kernel's, and q, k, v and the state 16-byte
// aligned at 16-byte strides for its vector loads.
inline constexpr std::int64_t kGatedDeltaNetLanesTokens = 16;
bool GatedDeltaNetLanesFits(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGatedDeltaNetLanes(const ggml_tensor* node);

// And two more fusions of Qwen3.8's Gated DeltaNet nodes, the same
// arithmetic in the same order:
//   jitllm.gdn.conv       the causal convolution over the conv state's
//                         history then the chunk's rows (ssm_conv), its
//                         silu, and the query and key heads' L2 norm
//                         (rms_norm with eps / d, times 1 / sqrt(d)), read
//                         straight from the rows (no transposed copy or
//                         concatenation);
//   jitllm.gdn.norm_gate  each head's rms_norm times the norm weight times
//                         sigmoid(z), in F32 or rounded to BF16 (the output
//                         projection's input).
// `x` F32 [channels, t] (the QKV rows), `history` F32 [(k - 1) · channels]
// (the conv state: tap j of channel c at c · (k - 1) + j), `weight` F32 [k,
// channels], `qk_channels` the leading channels (query and key heads of
// `head` values) that are normalized: F32 [channels, t].
ggml_tensor* GdnConv(ggml_context* context, ggml_tensor* x, ggml_tensor* history,
                     ggml_tensor* weight, std::int64_t qk_channels, std::int64_t head, float eps,
                     float scale);
// `o` F32 [d, heads, t] (packed), `weight` F32 [d], `z` F32 [d · heads, t]:
// [d · heads, t] of `type` (F32 or BF16).
ggml_tensor* GdnNormGate(ggml_context* context, ggml_tensor* o, ggml_tensor* weight, ggml_tensor* z,
                         float eps, ggml_type type);
std::expected<void, KernelFailure> CheckGdnConv(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckGdnNormGate(const ggml_tensor* node);

// The routed experts over the CUTLASS layout (moe_layout.h), for
// Qwen3.8's prefill and decode (jitllm_moe.cu; the grouped GEMM is
// CUTLASS's, moe_cutlass.h):
//
//   jitllm.moe.route          the routing's ids sorted by expert: offsets,
//                             each slot's row, each row's token and expert
//                             (moe_layout.h RouteLayout), deterministic
//                             (slots in token order within an expert);
//   jitllm.moe.quantize       each token's activations quantized once to
//                             NVFP4 as GGML's MMQ quantizes them
//                             (quantize_mmq_nvfp4: a row scale amax / 2688,
//                             E4M3 block scales searched as upstream does),
//                             written to the token's sorted rows
//                             (moe_layout.h QuantLayout);
//   jitllm.moe.gemm           CUTLASS's block-scaled grouped GEMM over the
//                             sorted rows, BF16 out;
//   jitllm.moe.glu_quantize   each sorted row's gate and up products times
//                             its row scale and the experts' global scales,
//                             SwiGLU (upstream's silu), quantized again for
//                             the down projection;
//   jitllm.moe.combine_sorted each token's experts' down products (times
//                             their row and global scales and routing
//                             weights) summed in expert order, plus the
//                             gated shared expert: jitllm.moe.combine's
//                             arithmetic over the sorted rows;
//   jitllm.moe.gemv           up to 8 tokens' routed products straight from
//                             the layout, the F32 activations quantized to
//                             8 bits as MMVQ quantizes them, F32 out as
//                             mul_mat_id's (decode's); in its SwiGLU form
//                             (MoeGemvSwiglu) the gate and up rows together,
//                             out jitllm.moe.glu's result.
//
// Each writes and reads the layouts moe_layout.h describes; the builders
// store the extents (experts, experts used, tokens, n, k and offsets) in
// the nodes' parameters, and the checks hold every operand to them.
ggml_tensor* MoeRoute(ggml_context* context, ggml_tensor* ids, std::int64_t experts);
ggml_tensor* MoeQuantize(ggml_context* context, ggml_tensor* x, ggml_tensor* route);
ggml_tensor* MoeGemm(ggml_context* context, ggml_tensor* a, ggml_tensor* route,
                     ggml_tensor* weights, std::int64_t n, std::uint64_t codes_offset,
                     std::uint64_t scales_offset);
ggml_tensor* MoeGluQuantize(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                            ggml_tensor* route, ggml_tensor* gate_scale, ggml_tensor* up_scale);
ggml_tensor* MoeCombineSorted(ggml_context* context, ggml_tensor* d, ggml_tensor* a,
                              ggml_tensor* route, ggml_tensor* down_scale, ggml_tensor* weights,
                              ggml_tensor* shared, ggml_tensor* shared_gate);
ggml_tensor* MoeGemv(ggml_context* context, ggml_tensor* weights, ggml_tensor* x, ggml_tensor* ids,
                     std::int64_t n, std::int64_t row0, std::int64_t rows,
                     std::uint64_t codes_offset, std::uint64_t scales_offset);
// jitllm.moe.gemv's SwiGLU form over a block of 2 · f rows (gate rows,
// then up rows): silu(gate · gate_scale[e]) · (up · up_scale[e]), F32
// [f, used, t].
ggml_tensor* MoeGemvSwiglu(ggml_context* context, ggml_tensor* weights, ggml_tensor* x,
                           ggml_tensor* ids, std::int64_t f, ggml_tensor* gate_scale,
                           ggml_tensor* up_scale, std::uint64_t codes_offset,
                           std::uint64_t scales_offset);
// Whether a jitllm.moe.gemv node is the SwiGLU form.
bool IsMoeGemvSwiglu(const ggml_tensor* node);
// The most tokens jitllm.moe.gemv takes.
inline constexpr std::int64_t kMoeGemvTokens = 8;

std::expected<void, KernelFailure> CheckMoeRoute(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeQuantize(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGemm(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGluQuantize(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeCombineSorted(const ggml_tensor* node);
std::expected<void, KernelFailure> CheckMoeGemv(const ggml_tensor* node);

// The launchers (CUDA builds, jitllm_ops.cu and jitllm_fused.cu): the
// check, then one kernel (or one cuBLAS call) on the context's stream. None
// draws scratch; the product needs the context's cuBLAS handle and writes
// its workspace.
class LaunchContext;
std::expected<void, KernelFailure> RunMxfp8MulMatVec(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMxfp8Dequant(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunNvfp4Rows(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcCombine(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcNorm(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunHcMix(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGlu(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeCombine(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunBf16(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGemmBf16(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGatedDeltaNetColumns(LaunchContext& launch,
                                                           ggml_tensor* node);
std::expected<void, KernelFailure> RunGatedDeltaNetLanes(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGdnConv(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunGdnNormGate(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeRoute(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeQuantize(LaunchContext& launch, ggml_tensor* node);
// The grouped GEMM draws its arguments and CUTLASS's workspace from the
// pool: PlanMoeGemm's bytes.
std::expected<std::uint64_t, KernelFailure> PlanMoeGemm(const LaunchContext& launch,
                                                        const ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGemm(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGluQuantize(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeCombineSorted(LaunchContext& launch, ggml_tensor* node);
std::expected<void, KernelFailure> RunMoeGemv(LaunchContext& launch, ggml_tensor* node);

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_JITLLM_OPS_H_
