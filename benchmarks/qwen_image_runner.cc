// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "qwen_image_runner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <fstream>
#include <initializer_list>
#include <map>
#include <span>
#include <string_view>
#include <tuple>
#include <utility>

#include "artifact/composition.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "kernels/image/gemm.h"
#include "kernels/image/ops.h"
#include "scheduler/commands.h"
#include "scheduler/scheduler.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::benchmarks {

namespace {

namespace ja = jitllm::artifact;
namespace kg = jitllm::kernels::ggml;
namespace ki = jitllm::kernels::image;
namespace md = jitllm::model;
namespace sc = jitllm::scheduler;
namespace ts = jitllm::test_support;
using base::Bytes;
using catalog::ExtentId;
using catalog::MemoryClass;
using catalog::Recovery;
using providers::BackingKind;
using Status = test_support::Status;
using Bf16 = std::uint16_t;
using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kExtent = test_support::kPagedExtent;
// The VAE's im2col buffer: the resident harness's size, which sets how its
// 3x3 convolutions are split, and so their products' shapes.
constexpr std::uint64_t kCol = std::uint64_t{256} << 20U;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

// 256-aligned buffers one after another from a base (0 to measure).
class Carve {
 public:
  explicit Carve(std::uint64_t base) : base_(base) {}
  std::uint64_t Take(std::uint64_t bytes) {
    const std::uint64_t at = base_ + used_;
    used_ += Round(std::max<std::uint64_t>(bytes, 256), 256);
    return at;
  }
  std::uint64_t used() const { return used_; }

 private:
  std::uint64_t base_;
  std::uint64_t used_ = 0;
};

std::expected<std::vector<Bf16>, std::string> ReadBf16(const std::filesystem::path& p,
                                                       std::size_t count) {
  std::ifstream in(p, std::ios::binary);
  if (!in) {
    return Error(std::format("{}: cannot open", p.string()));
  }
  std::vector<Bf16> out(count);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * 2));
  if (static_cast<std::size_t>(in.gcount()) != count * 2 || in.peek() != EOF) {
    return Error(std::format("{}: not {} BF16 values", p.string(), count));
  }
  return out;
}

// The buffers each phase's job uses, in the shared workspace (`work`) or
// the image's own memory (`own`); laid out once, at fixed offsets.
struct Own {
  std::uint64_t embeds = 0, txt = 0, pk = 0, pv = 0, freqs = 0, latents = 0, noise = 0;
};
Own LayOutOwn(std::uint64_t base, const md::QwenImageDenoiserProfile& p, std::int64_t text,
              std::int64_t image, std::uint64_t& bytes) {
  Carve c(base);
  const std::int64_t width = p.width;
  Own o;
  o.embeds = c.Take(static_cast<std::uint64_t>(text * p.context * 2));
  o.txt = c.Take(static_cast<std::uint64_t>(text * width * 2));
  o.pk = c.Take(static_cast<std::uint64_t>(std::int64_t{p.blocks} * text * width * 2));
  o.pv = c.Take(static_cast<std::uint64_t>(std::int64_t{p.blocks} * text * width * 2));
  o.freqs = c.Take(static_cast<std::uint64_t>((text + image) * 128 * 4));
  o.latents = c.Take(static_cast<std::uint64_t>(image * p.in_channels * 2));
  o.noise = c.Take(static_cast<std::uint64_t>(image * p.out_channels * 2));
  bytes = c.used();
  return o;
}

struct TextWork {
  std::uint64_t x = 0, n = 0, q = 0, k = 0, v = 0, attn = 0, o = 0, g = 0, u = 0, ids = 0, bad = 0,
                cos = 0, sin = 0;
};
TextWork LayOutText(std::uint64_t base, const md::QwenImageTextProfile& p, std::int64_t rows,
                    std::uint64_t& bytes) {
  Carve c(base);
  const std::int64_t width = p.width;
  const std::int64_t q = std::int64_t{p.heads} * p.head_dim;
  const std::int64_t kv = std::int64_t{p.kv_heads} * p.head_dim;
  TextWork w;
  w.x = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.n = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.q = c.Take(static_cast<std::uint64_t>(rows * q * 2));
  w.k = c.Take(static_cast<std::uint64_t>(rows * kv * 2));
  w.v = c.Take(static_cast<std::uint64_t>(rows * kv * 2));
  w.attn = c.Take(static_cast<std::uint64_t>(rows * q * 2));
  w.o = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.g = c.Take(static_cast<std::uint64_t>(rows * std::int64_t{p.ffn} * 2));
  w.u = c.Take(static_cast<std::uint64_t>(rows * std::int64_t{p.ffn} * 2));
  w.ids = c.Take(static_cast<std::uint64_t>(rows * 4));
  w.bad = c.Take(4);
  w.cos = c.Take(static_cast<std::uint64_t>(rows * 256));
  w.sin = c.Take(static_cast<std::uint64_t>(rows * 256));
  bytes = c.used();
  return w;
}

struct DitWork {
  std::uint64_t x = 0, n = 0, q = 0, k = 0, v = 0, attn = 0, o = 0, g = 0, u = 0, sin = 0, t1 = 0,
                t2 = 0, temb = 0, mod = 0, nscale = 0;
};
DitWork LayOutDit(std::uint64_t base, const md::QwenImageDenoiserProfile& p, std::int64_t rows,
                  std::uint64_t& bytes) {
  Carve c(base);
  const std::int64_t width = p.width;
  DitWork w;
  w.x = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.n = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.q = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.k = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.v = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.attn = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.o = c.Take(static_cast<std::uint64_t>(rows * width * 2));
  w.g = c.Take(static_cast<std::uint64_t>(rows * std::int64_t{p.mlp} * 2));
  w.u = c.Take(static_cast<std::uint64_t>(rows * std::int64_t{p.mlp} * 2));
  w.sin = c.Take(static_cast<std::uint64_t>(2 * std::int64_t{p.timestep_dim} * 2));
  w.t1 = c.Take(static_cast<std::uint64_t>(2 * width * 2));
  w.t2 = c.Take(static_cast<std::uint64_t>(2 * width * 2));
  w.temb = c.Take(static_cast<std::uint64_t>(2 * width * 2));
  w.mod = c.Take(static_cast<std::uint64_t>(width * 2 * 4 * 2));
  w.nscale = c.Take(static_cast<std::uint64_t>(2 * width * 2));
  bytes = c.used();
  return w;
}

struct VaeWork {
  std::vector<std::uint64_t> weights;  // BF16, by the decoder's tensor list
  std::array<std::uint64_t, md::kVaeBuffers> buf{};
  std::uint64_t stats = 0, col = 0, scores = 0, probs = 0;
  std::uint64_t scores_bytes = 0;
};
VaeWork LayOutVae(std::uint64_t base, const md::QwenImageVaeProfile& p, std::uint32_t grid,
                  std::span<const std::uint64_t> f32_bytes, std::uint64_t& bytes) {
  Carve c(base);
  VaeWork w;
  for (const std::uint64_t b : f32_bytes) {
    w.weights.push_back(c.Take(b / 2));
  }
  const auto steps = md::VaeDecoderPlan(p, grid, grid);
  std::array<std::uint64_t, md::kVaeBuffers> need{};
  std::uint64_t attention_pixels = 0;
  for (const md::VaeStep& st : steps) {
    const std::uint64_t in = std::uint64_t{st.in_channels} * st.height * st.width;
    const std::uint64_t scale = st.kind == md::VaeStep::Kind::kUpsample ? 4 : 1;
    const std::uint64_t out = std::uint64_t{st.out_channels} * st.height * st.width * scale;
    need[st.in] = std::max(need[st.in], st.kind == md::VaeStep::Kind::kAttention ? 3 * in : in);
    need[st.out] = std::max(need[st.out], st.kind == md::VaeStep::Kind::kAddDupUp ? 4 * out : out);
    need[st.aux] = std::max(need[st.aux], in);
    if (st.kind == md::VaeStep::Kind::kAttention) {
      attention_pixels = std::max(attention_pixels, std::uint64_t{st.height} * st.width);
    }
  }
  need[md::kVaeX] = std::max<std::uint64_t>(need[md::kVaeX], std::uint64_t{p.z_dim} * grid * grid);
  for (std::size_t i = 0; i < need.size(); ++i) {
    w.buf.at(i) = c.Take(need.at(i) * 2);
  }
  w.stats = c.Take(std::uint64_t{2} * 64 * 2);
  w.col = c.Take(kCol);
  w.scores_bytes = attention_pixels * attention_pixels * 4;
  w.scores = c.Take(w.scores_bytes);
  w.probs = c.Take(attention_pixels * attention_pixels * 2);
  bytes = c.used();
  return w;
}

struct Component {
  std::optional<ja::Artifact> artifact;
  std::vector<std::uint32_t> bound;
  std::vector<ja::FileDescriptor> shards;
  PagedWeights weights;
  std::vector<std::uint64_t> tensor;  // bound tensors' device addresses
  catalog::Closure closure;           // its phase's
};

}  // namespace

struct QwenImageRunner::State {
  const md::QwenImageProfile& profile = md::QwenImage21();
  std::array<Component, 3> c;  // text encoder, denoiser, VAE
  std::vector<std::int32_t> ids;
  std::size_t drop = 0;
  std::int64_t text = 0;
  std::uint32_t grid = 0;
  std::int64_t image = 0;
  md::QwenImageSchedule schedule;
  std::vector<Bf16> noise0;
  md::RotaryTables rot;
  std::vector<float> freqs;
  std::vector<std::uint64_t> vae_f32_bytes;

  test_support::Mapped own_memory;
  test_support::Mapped cublas_workspace;
  std::uint64_t cublas_bytes = 0;
  std::unique_ptr<kg::CublasHandle> cublas;
  Own own;
  std::uint64_t own_bytes = 0;
  std::uint64_t text_bytes = 0, dit_bytes = 0, vae_bytes = 0;
  TextWork tw;
  DitWork dw;
  VaeWork vw;
  std::vector<ExtentId> staging;
  std::byte* in = nullptr;  // pinned: what a job uploads
  std::uint64_t in_bytes = 0;
  std::byte* out = nullptr;  // pinned: what a job downloads
  std::uint64_t out_bytes = 0;

  // The last generation's timings.
  double encode = 0;
  std::vector<double> steps;
  double decode = 0;
  std::uint64_t generations = 0;
};

QwenImageRunner::QwenImageRunner(test_support::PagedNode& node, const QwenImageOptions& options,
                                 int owner, std::uint32_t stream)
    : node_(node), o_(options), owner_(owner), stream_(stream), s_(std::make_unique<State>()) {}

QwenImageRunner::~QwenImageRunner() = default;

std::vector<ExtentId> QwenImageRunner::weights() const {
  std::vector<ExtentId> all;
  for (const Component& c : s_->c) {
    all.insert(all.end(), c.weights.extents().begin(), c.weights.extents().end());
  }
  return all;
}

std::uint64_t QwenImageRunner::weight_read_bytes() const {
  std::uint64_t bytes = 0;
  for (const Component& c : s_->c) {
    bytes += c.weights.read_bytes();
  }
  return bytes;
}

Status QwenImageRunner::Setup() {
  State& s = *s_;
  const md::QwenImageProfile& profile = s.profile;
  auto composition = ja::OpenComposition(o_.store / o_.composition);
  if (!composition) {
    return Error("composition: " + composition.error().ToString());
  }
  if (composition->architecture() != profile.pipeline_architecture) {
    return Error("the composition is not a " + std::string(profile.pipeline_architecture));
  }
  const std::array<std::tuple<std::string_view, std::string_view, std::vector<md::QwenImageTensor>>,
                   3>
      roles = {
          {{"text_encoder", profile.text_architecture, md::TextEncoderTensors(profile.text)},
           {"transformer", profile.denoiser_architecture, md::DenoiserTensors(profile.denoiser)},
           {"vae", profile.vae_architecture, md::VaeDecoderTensors(profile.vae)}}};
  for (std::size_t i = 0; i < roles.size(); ++i) {
    const auto& [role, arch, tensors] = roles.at(i);
    const ja::CompositionComponent* found = composition->Find(role);
    if (found == nullptr) {
      return Error(std::format("the composition has no {}", role));
    }
    ja::OpenOptions options;
    options.expected_id = found->artifact;
    auto a = ja::Artifact::Open(o_.store / found->artifact, options);
    if (!a) {
      return Error(std::format("{}: {}", role, a.error().ToString()));
    }
    if (a->model().architecture != found->architecture) {
      return Error(
          std::format("{}: the artifact's architecture differs from the composition's", role));
    }
    auto bound = md::BindQwenImageComponent(tensors, arch, *a);
    if (!bound) {
      return Error(std::format("{}: {}", role, bound.error()));
    }
    Component& c = s.c.at(i);
    c.artifact.emplace(std::move(*a));
    c.bound = std::move(*bound);
    for (std::uint32_t sh = 0; sh < c.artifact->shards().size(); ++sh) {
      auto fd = c.artifact->OpenShardForDirectRead(sh);
      if (!fd) {
        return Error(std::format("{}: shard {} cannot be opened for direct reads", role, sh));
      }
      c.shards.push_back(std::move(*fd));
    }
  }

  // The prompt's tokens (the native tokenizer and renderer, D-088).
  const auto tokenizer_json = composition->Metadata("tokenizer.json");
  if (!tokenizer_json) {
    return Error("the composition keeps no tokenizer.json");
  }
  auto spec = jitllm::tokenizer::ReadHfTokenizer(*tokenizer_json);
  if (!spec) {
    return Error("tokenizer: " + spec.error().ToString());
  }
  auto tokenizer = jitllm::tokenizer::Tokenizer::Create(std::move(*spec));
  if (!tokenizer) {
    return Error("tokenizer: " + tokenizer.error().ToString());
  }
  auto rendered = jitllm::chat::RenderQwenImagePrompt(o_.prompt, *tokenizer);
  if (!rendered) {
    return Error("prompt: " + rendered.error().ToString());
  }
  std::vector<jitllm::tokenizer::TokenId> ids;
  if (auto e =
          tokenizer->EncodeMarked(rendered->rendered.text, rendered->rendered.specials, {}, ids);
      !e) {
    return Error("prompt: " + e.error().ToString());
  }
  s.drop = rendered->drop_tokens;
  if (ids.size() <= s.drop || ids.size() > 1024) {
    return Error("a prompt of 1 to 1,024 tokens after the system turn");
  }
  s.ids.assign(ids.begin(), ids.end());
  s.text = static_cast<std::int64_t>(ids.size() - s.drop);
  if (o_.size % 32 != 0 || o_.size < 64 || o_.size > 2048 || o_.steps < 2 || o_.steps > 100) {
    return Error("a size a multiple of 32 in 64-2048, and 2-100 steps");
  }
  s.grid = o_.size / profile.vae.spatial_factor;
  s.image = std::int64_t{s.grid} * s.grid;
  auto schedule = md::QwenImageSchedulerSigmas(profile.scheduler, o_.steps,
                                               static_cast<std::uint32_t>(s.image));
  if (!schedule) {
    return Error(schedule.error());
  }
  s.schedule = std::move(*schedule);
  auto noise = ReadBf16(o_.noise, static_cast<std::size_t>(s.image * profile.denoiser.in_channels));
  if (!noise) {
    return std::unexpected(noise.error());
  }
  s.noise0 = std::move(*noise);
  s.rot = md::QwenImageTextRotary(profile.text, static_cast<std::uint32_t>(s.ids.size()));
  s.freqs =
      md::QwenImageDenoiserRotary(profile.denoiser, static_cast<std::uint32_t>(s.text), s.grid);

  // Each component's weights: the groups its bound tensors are in.
  for (Component& c : s.c) {
    if (!c.artifact) {
      return Error("a component without its artifact");
    }
    const ja::Artifact& a = *c.artifact;
    const auto groups = a.groups();
    std::vector<bool> place(groups.size(), false);
    for (const std::uint32_t r : c.bound) {
      place[a.resources()[r].group] = true;
    }
    if (auto r = c.weights.Reserve(node_, a, c.shards, ArtifactKey(a), place, {}); !r) {
      return r;
    }
  }
  const std::optional<ja::Artifact>& vae_artifact = s.c[2].artifact;
  if (!vae_artifact.has_value()) {
    return Error("the VAE without its artifact");
  }
  const ja::Artifact& vae = *vae_artifact;
  for (const std::uint32_t r : s.c[2].bound) {
    const auto& res = vae.resources()[r];
    if (res.bytes.value() % 4 != 0) {
      return Error(std::format("the VAE's {} is not F32", res.name));
    }
    s.vae_f32_bytes.push_back(res.bytes.value());
  }

  // Working memory: the image's own, and each phase's in the workspace.
  (void)LayOutOwn(0, profile.denoiser, s.text, s.image, s.own_bytes);
  (void)LayOutText(0, profile.text, static_cast<std::int64_t>(s.ids.size()), s.text_bytes);
  (void)LayOutDit(0, profile.denoiser, s.text + s.image, s.dit_bytes);
  (void)LayOutVae(0, profile.vae, s.grid, s.vae_f32_bytes, s.vae_bytes);
  work_bytes_ = Round(std::max({s.text_bytes, s.dit_bytes, s.vae_bytes}), kExtent);
  if (auto r =
          node_.MapResident(s.own_memory, "the image's own memory", s.own_bytes,
                            BackingKind::kDevice, MemoryClass::kScratch, Recovery::kPinned, owner_);
      !r) {
    return r;
  }
  s.own = LayOutOwn(s.own_memory.base, profile.denoiser, s.text, s.image, s.own_bytes);

  cudaDeviceProp prop{};
  (void)cudaGetDeviceProperties(&prop, 0);
  s.cublas_bytes =
      kg::CublasHandle::UpstreamWorkspace((prop.major * 100) + (prop.minor * 10)).value();
  if (auto r =
          node_.MapResident(s.cublas_workspace, "the image's cuBLAS workspace", s.cublas_bytes,
                            BackingKind::kDevice, MemoryClass::kRuntime, Recovery::kPinned, owner_);
      !r) {
    return r;
  }
  auto blas =
      kg::CublasHandle::Create(0, node_.execution(), node_.stream(stream_),
                               {.base = s.cublas_workspace.base, .size = Bytes(s.cublas_bytes)});
  if (!blas) {
    return Error("cuBLAS: " + blas.error().detail);
  }
  s.cublas = std::move(*blas);

  // Staging: the largest upload (the first step's rotary tables and latents,
  // or the text's inputs) and download (the decoded image).
  const std::uint64_t text_in = Round(s.ids.size() * 4, 256) + (2 * Round(s.ids.size() * 256, 256));
  const std::uint64_t step_in = Round(s.freqs.size() * 4, 256) + Round(s.noise0.size() * 2, 256) +
                                Round(2 * std::uint64_t{profile.denoiser.timestep_dim} * 2, 256);
  s.in_bytes = std::max<std::uint64_t>({text_in, step_in, 4096});
  s.out_bytes = std::max<std::uint64_t>(
      std::uint64_t{profile.vae.out_channels} * o_.size * o_.size * 2,
      static_cast<std::uint64_t>(s.image * profile.denoiser.out_channels * 2));
  auto in = node_.Pinned(s.in_bytes, owner_, s.staging);
  auto out = node_.Pinned(s.out_bytes, owner_, s.staging);
  if (!in || !out) {
    return Error("pinned staging for the image");
  }
  s.in = static_cast<std::byte*>(*in);
  s.out = static_cast<std::byte*>(*out);
  return {};
}

Status QwenImageRunner::Register() {
  for (Component& c : s_->c) {
    if (auto r = c.weights.Register(node_, owner_); !r) {
      return r;
    }
  }
  // D-090, for every model: the places stay put for the model's life
  // (never unpinned; the pins go with the scheduler).
  if (auto pinned = node_.scheduler().PinPlaces(weights()); !pinned) {
    return Error(std::format("pinning the image's places: {}", sc::ToString(pinned.error())));
  }
  return {};
}

Status QwenImageRunner::Bind() {
  State& s = *s_;
  auto& catalog = node_.catalog();
  std::vector<ExtentId> common;
  for (const ts::Mapped* mapped : std::initializer_list<const ts::Mapped*>{
           &s.own_memory, &node_.activations(), &s.cublas_workspace}) {
    common.insert(common.end(), mapped->extents.begin(), mapped->extents.end());
  }
  common.insert(common.end(), s.staging.begin(), s.staging.end());
  std::vector<ExtentId> all = common;
  for (Component& c : s.c) {
    std::vector<ExtentId> mine = c.weights.extents();
    mine.insert(mine.end(), common.begin(), common.end());
    c.closure = catalog.ClosureOfExtents(mine).value();
    all.insert(all.end(), c.weights.extents().begin(), c.weights.extents().end());
    c.tensor.clear();
    if (!c.artifact.has_value()) {
      return Error("a component without its artifact");
    }
    const ja::Artifact& a = *c.artifact;
    for (const std::uint32_t r : c.bound) {
      const auto& res = a.resources()[r];
      c.tensor.push_back(c.weights.group_address(res.group) + res.offset.value());
    }
  }
  everything_ = catalog.ClosureOfExtents(all).value();
  fence_ = catalog.ClosureOfExtents(s.own_memory.extents).value();
  const std::uint64_t work = node_.activations().base;
  std::uint64_t bytes = 0;
  s.tw = LayOutText(work, s.profile.text, static_cast<std::int64_t>(s.ids.size()), bytes);
  s.dw = LayOutDit(work, s.profile.denoiser, s.text + s.image, bytes);
  s.vw = LayOutVae(work, s.profile.vae, s.grid, s.vae_f32_bytes, bytes);
  if (std::max({s.text_bytes, s.dit_bytes, s.vae_bytes}) > node_.activations().bytes) {
    return Error("the image's working memory exceeds the workspace");
  }
  return {};
}

// ------------------------------------------------------------------ encode

// The text encoder (qwen_image_exec.cc's TextEncoder::Run): Qwen3-VL's text
// path over every rendered token, the rows after the system turn kept.
Status QwenImageRunner::Encode() {
  State& s = *s_;
  const md::QwenImageTextProfile& p = s.profile.text;
  const Component& c = s.c[0];
  const auto rows = static_cast<std::int64_t>(s.ids.size());
  const std::int64_t width = p.width;
  const std::int64_t q = std::int64_t{p.heads} * p.head_dim;
  const std::int64_t kv = std::int64_t{p.kv_heads} * p.head_dim;
  const TextWork& w = s.tw;
  cublasContext* const blas = s.cublas->native();
  const auto t = [&](std::size_t layer, md::TextTensor which) {
    return At<Bf16>(c.tensor[1 + (layer * static_cast<std::size_t>(md::TextTensor::kCount)) +
                             static_cast<std::size_t>(which)]);
  };
  const auto started = Clock::now();
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    void* const st = native.handle;
    auto* const cs = static_cast<cudaStream_t>(st);
    // The inputs: ids, then the rotary tables.
    std::uint64_t at = 0;
    for (auto [dst, src, bytes] :
         {std::tuple{w.ids, static_cast<const void*>(s.ids.data()), rows * 4},
          {w.cos, s.rot.cos.data(), rows * 256},
          {w.sin, s.rot.sin.data(), rows * 256}}) {
      std::memcpy(s.in + at, src, static_cast<std::size_t>(bytes));
      if (auto r = Cuda(cudaMemcpyAsync(At<void>(dst), s.in + at, static_cast<std::size_t>(bytes),
                                        cudaMemcpyHostToDevice, cs),
                        "text inputs");
          !r) {
        ran = r;
        return sc::JobResult::kUnknown;
      }
      at += Round(static_cast<std::uint64_t>(bytes), 256);
    }
    Status r = Cuda(cudaMemsetAsync(At<void>(w.bad), 0, 4, cs), "a flag");
    r = r ? ki::EmbedRows(At<Bf16>(c.tensor[0]), p.vocab, At<std::int32_t>(w.ids), rows, width,
                          At<Bf16>(w.x), At<std::int32_t>(w.bad), st)
          : r;
    const float scale = 1.0f / std::sqrt(static_cast<float>(p.head_dim));
    for (std::size_t l = 0; r && l < p.layers; ++l) {
      using T = md::TextTensor;
      r = ki::RmsNorm(At<Bf16>(w.x), t(l, T::kInputNorm), At<Bf16>(w.n), rows, width, p.rms_eps,
                      st);
      r = r ? ki::Linear(blas, At<Bf16>(w.n), width, t(l, T::kQ), width, At<Bf16>(w.q), q, rows, q,
                         width)
            : r;
      r = r ? ki::Linear(blas, At<Bf16>(w.n), width, t(l, T::kK), width, At<Bf16>(w.k), kv, rows,
                         kv, width)
            : r;
      r = r ? ki::Linear(blas, At<Bf16>(w.n), width, t(l, T::kV), width, At<Bf16>(w.v), kv, rows,
                         kv, width)
            : r;
      r = r ? ki::HeadNormRopeNeox(At<Bf16>(w.q), q, rows, p.heads, t(l, T::kQNorm),
                                   At<Bf16>(w.cos), At<Bf16>(w.sin), p.rms_eps, st)
            : r;
      r = r ? ki::HeadNormRopeNeox(At<Bf16>(w.k), kv, rows, p.kv_heads, t(l, T::kKNorm),
                                   At<Bf16>(w.cos), At<Bf16>(w.sin), p.rms_eps, st)
            : r;
      r = r ? ki::SmallAttention(At<Bf16>(w.q), q, At<Bf16>(w.k), kv, At<Bf16>(w.v), kv,
                                 At<Bf16>(w.attn), q, rows, p.heads, p.kv_heads, true, scale, st)
            : r;
      r = r ? ki::Linear(blas, At<Bf16>(w.attn), q, t(l, T::kO), q, At<Bf16>(w.o), width, rows,
                         width, q)
            : r;
      r = r ? ki::Add(At<Bf16>(w.x), At<Bf16>(w.o), At<Bf16>(w.x), rows * width, st) : r;
      r = r ? ki::RmsNorm(At<Bf16>(w.x), t(l, T::kPostNorm), At<Bf16>(w.n), rows, width, p.rms_eps,
                          st)
            : r;
      r = r ? ki::Linear(blas, At<Bf16>(w.n), width, t(l, T::kGate), width, At<Bf16>(w.g), p.ffn,
                         rows, p.ffn, width)
            : r;
      r = r ? ki::Linear(blas, At<Bf16>(w.n), width, t(l, T::kUp), width, At<Bf16>(w.u), p.ffn,
                         rows, p.ffn, width)
            : r;
      r = r ? ki::SwiGlu(At<Bf16>(w.g), p.ffn, At<Bf16>(w.u), p.ffn, At<Bf16>(w.g), rows, p.ffn, st)
            : r;
      r = r ? ki::Linear(blas, At<Bf16>(w.g), p.ffn, t(l, T::kDown), p.ffn, At<Bf16>(w.o), width,
                         rows, width, p.ffn)
            : r;
      r = r ? ki::Add(At<Bf16>(w.x), At<Bf16>(w.o), At<Bf16>(w.x), rows * width, st) : r;
    }
    const std::int64_t kept = rows - static_cast<std::int64_t>(s.drop);
    r = r ? Cuda(cudaMemcpyAsync(At<void>(s.own.embeds),
                                 At<Bf16>(w.x) + (static_cast<std::int64_t>(s.drop) * width),
                                 static_cast<std::size_t>(kept * width * 2),
                                 cudaMemcpyDeviceToDevice, cs),
                 "the kept rows")
          : r;
    r = r ? Cuda(cudaMemcpyAsync(s.out, At<void>(w.bad), 4, cudaMemcpyDeviceToHost, cs), "the flag")
          : r;
    if (!r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(c.closure, std::move(job), "the image's encode", stream_);
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return posted;
  }
  std::int32_t flag = 0;
  std::memcpy(&flag, s.out, 4);
  if (flag != 0) {
    return Error("a token id outside the embedding table");
  }
  s.encode = Seconds(Clock::now() - started);
  return {};
}

// ------------------------------------------------------------------ denoise

namespace {

// The denoiser's step (qwen_image_exec.cc's Denoiser, native attention),
// its buffers at fixed addresses.
class DitStep {
 public:
  DitStep(const md::QwenImageDenoiserProfile& p, cublasContext* blas, void* stream,
          const std::vector<std::uint64_t>& tensor, const Own& own, const DitWork& w,
          std::int64_t text, std::int64_t image)
      : p_(p),
        blas_(blas),
        s_(stream),
        tensor_(tensor),
        own_(own),
        w_(w),
        text_(text),
        image_(image) {}

  // txt_in: zero-centred RMSNorm, in_layer, GELU (tanh), out_layer: the
  // text rows of the joint sequence (the resident harness's TextRows, its
  // n_ and q_ as scratch).
  Status TextRows() {
    const std::int64_t width = p_.width;
    Status r = ki::ZeroCenterRmsNorm(At<Bf16>(own_.embeds), G(md::DenoiserGlobal::kTxtNorm),
                                     At<Bf16>(w_.n), text_, p_.context, p_.eps, s_);
    r = r ? ki::Linear(blas_, At<Bf16>(w_.n), p_.context, G(md::DenoiserGlobal::kTxtIn), p_.context,
                       At<Bf16>(w_.q), width, text_, width, p_.context)
          : r;
    r = r ? ki::GeluTanh(At<Bf16>(w_.q), At<Bf16>(w_.q), text_ * width, s_) : r;
    r = r ? ki::Linear(blas_, At<Bf16>(w_.q), width, G(md::DenoiserGlobal::kTxtOut), width,
                       At<Bf16>(own_.txt), width, text_, width, width)
          : r;
    return r;
  }

  // One step: the noise from the latents, with the sinusoid (already in
  // w_.sin). Step 0 runs the whole joint sequence and fills the prefix
  // cache; later steps only the image.
  Status Step(bool first) {
    const std::int64_t width = p_.width;
    const std::int64_t rows = first ? text_ + image_ : image_;
    const std::int64_t row0 = first ? 0 : text_;
    const std::int64_t first_target = first ? text_ : 0;
    using G_ = md::DenoiserGlobal;
    Status r = ki::Linear(blas_, At<Bf16>(w_.sin), p_.timestep_dim, G(G_::kTime1), p_.timestep_dim,
                          At<Bf16>(w_.t1), width, 2, width, p_.timestep_dim);
    r = r ? ki::Silu(At<Bf16>(w_.t1), At<Bf16>(w_.t1), 2 * width, s_) : r;
    r = r ? ki::Linear(blas_, At<Bf16>(w_.t1), width, G(G_::kTime2), width, At<Bf16>(w_.temb),
                       width, 2, width, width)
          : r;
    r = r ? ki::Silu(At<Bf16>(w_.temb), At<Bf16>(w_.t2), 2 * width, s_) : r;
    r = r ? ki::Linear(blas_, At<Bf16>(w_.t2), width, G(G_::kModulation), width, At<Bf16>(w_.mod),
                       4 * width, 2, 4 * width, width)
          : r;
    r = r ? ki::Linear(blas_, At<Bf16>(w_.t2), width, G(G_::kNormOut), width, At<Bf16>(w_.nscale),
                       width, 2, width, width)
          : r;
    if (r && first) {
      r = Cuda(cudaMemcpyAsync(At<void>(w_.x), At<void>(own_.txt),
                               static_cast<std::size_t>(text_ * width * 2),
                               cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(s_)),
               "the text rows");
    }
    r = r ? ki::Linear(blas_, At<Bf16>(own_.latents), p_.in_channels, G(G_::kImgIn), p_.in_channels,
                       Row(w_.x, text_), width, image_, width, p_.in_channels)
          : r;
    for (std::uint32_t b = 0; r && b < p_.blocks; ++b) {
      r = Block(b, first, rows, row0, first_target);
    }
    r = r ? ki::LayerNormModulate(Row(w_.x, text_), Row(w_.n, text_), image_, width, p_.eps,
                                  At<Bf16>(w_.nscale), width, 0, s_)
          : r;
    r = r ? ki::Linear(blas_, Row(w_.n, text_), width, G(G_::kProjOut), width, At<Bf16>(own_.noise),
                       p_.out_channels, image_, p_.out_channels, width)
          : r;
    return r;
  }

 private:
  Bf16* G(md::DenoiserGlobal g) const { return At<Bf16>(tensor_[static_cast<std::size_t>(g)]); }
  Bf16* B(std::uint32_t block, md::BlockTensor t) const {
    return At<Bf16>(tensor_[static_cast<std::size_t>(md::DenoiserGlobal::kCount) +
                            (block * static_cast<std::size_t>(md::BlockTensor::kCount)) +
                            static_cast<std::size_t>(t)]);
  }
  Bf16* Row(std::uint64_t buf, std::int64_t row, std::int64_t width = 0) const {
    return At<Bf16>(buf) + (row * (width == 0 ? p_.width : width));
  }

  Status Block(std::uint32_t b, bool first, std::int64_t rows, std::int64_t row0,
               std::int64_t first_target) {
    using T = md::BlockTensor;
    const std::int64_t width = p_.width;
    const std::int64_t mstride = 4 * width;
    const Bf16* mod = At<Bf16>(w_.mod);
    Bf16* x = Row(w_.x, row0);
    Status r = ki::LayerNormModulate(x, Row(w_.n, row0), rows, width, p_.eps, mod, mstride,
                                     first_target, s_);
    r = r ? ki::Linear(blas_, Row(w_.n, row0), width, B(b, T::kQ), width, Row(w_.q, row0), width,
                       rows, width, width)
          : r;
    r = r ? ki::Linear(blas_, Row(w_.n, row0), width, B(b, T::kK), width, Row(w_.k, row0), width,
                       rows, width, width)
          : r;
    r = r ? ki::Linear(blas_, Row(w_.n, row0), width, B(b, T::kV), width, Row(w_.v, row0), width,
                       rows, width, width)
          : r;
    if (!r) {
      return r;
    }
    // Attention with jitLLM's BF16 FlashAttention: q and k rotated in
    // place, the prefix cache in BF16.
    auto* const cs = static_cast<cudaStream_t>(s_);
    r = ki::HeadNormRopeComplex(Row(w_.q, row0), width, rows, p_.heads, B(b, T::kQNorm),
                                At<float>(own_.freqs) + (row0 * 128), p_.eps, s_);
    r = r ? ki::HeadNormRopeComplex(Row(w_.k, row0), width, rows, p_.heads, B(b, T::kKNorm),
                                    At<float>(own_.freqs) + (row0 * 128), p_.eps, s_)
          : r;
    const auto prefix = static_cast<std::size_t>(text_ * width * 2);
    Bf16* pk = At<Bf16>(own_.pk) + (static_cast<std::int64_t>(b) * text_ * width);
    Bf16* pv = At<Bf16>(own_.pv) + (static_cast<std::int64_t>(b) * text_ * width);
    const float scale = 1.0f / std::sqrt(128.0f);
    if (r && first) {
      r = Cuda(cudaMemcpyAsync(pk, At<void>(w_.k), prefix, cudaMemcpyDeviceToDevice, cs), "cache");
      r = r ? Cuda(cudaMemcpyAsync(pv, At<void>(w_.v), prefix, cudaMemcpyDeviceToDevice, cs),
                   "cache")
            : r;
      r = r ? ki::SmallAttention(At<Bf16>(w_.q), width, At<Bf16>(w_.k), width, At<Bf16>(w_.v),
                                 width, At<Bf16>(w_.attn), width, text_, p_.heads, p_.heads, true,
                                 scale, s_)
            : r;
    } else if (r) {
      r = Cuda(cudaMemcpyAsync(At<void>(w_.k), pk, prefix, cudaMemcpyDeviceToDevice, cs), "cache");
      r = r ? Cuda(cudaMemcpyAsync(At<void>(w_.v), pv, prefix, cudaMemcpyDeviceToDevice, cs),
                   "cache")
            : r;
    }
    r = r ? ki::FlashAttention(Row(w_.q, text_), width, At<Bf16>(w_.k), width, At<Bf16>(w_.v),
                               width, Row(w_.attn, text_), width, image_, text_ + image_, p_.heads,
                               p_.heads, scale, s_)
          : r;
    return Rest(b, rows, row0, first_target, r);
  }

  // The block after attention: out projection, gated residual, MLP.
  Status Rest(std::uint32_t b, std::int64_t rows, std::int64_t row0, std::int64_t first_target,
              Status r) {
    using T = md::BlockTensor;
    const std::int64_t width = p_.width;
    const std::int64_t mlp = p_.mlp;
    const std::int64_t mstride = 4 * width;
    const Bf16* mod = At<Bf16>(w_.mod);
    Bf16* x = Row(w_.x, row0);
    r = r ? ki::Linear(blas_, Row(w_.attn, row0), width, B(b, T::kOut), width, Row(w_.o, row0),
                       width, rows, width, width)
          : r;
    r = r ? ki::GatedResidual(x, Row(w_.o, row0), width, rows, width, mod + width, mstride,
                              first_target, s_)
          : r;
    r = r ? ki::LayerNormModulate(x, Row(w_.n, row0), rows, width, p_.eps, mod + (2 * width),
                                  mstride, first_target, s_)
          : r;
    r = r ? ki::Linear(blas_, Row(w_.n, row0), width, B(b, T::kGate), width, Row(w_.g, row0, mlp),
                       mlp, rows, mlp, width)
          : r;
    r = r ? ki::Linear(blas_, Row(w_.n, row0), width, B(b, T::kProj), width, Row(w_.u, row0, mlp),
                       mlp, rows, mlp, width)
          : r;
    r = r ? ki::SwiGlu(Row(w_.g, row0, mlp), mlp, Row(w_.u, row0, mlp), mlp, Row(w_.g, row0, mlp),
                       rows, mlp, s_)
          : r;
    r = r ? ki::Linear(blas_, Row(w_.g, row0, mlp), mlp, B(b, T::kMlpOut), mlp, Row(w_.o, row0),
                       width, rows, width, mlp)
          : r;
    r = r ? ki::GatedResidual(x, Row(w_.o, row0), width, rows, width, mod + (3 * width), mstride,
                              first_target, s_)
          : r;
    return r;
  }

  const md::QwenImageDenoiserProfile& p_;
  cublasContext* blas_;
  void* s_;
  const std::vector<std::uint64_t>& tensor_;
  const Own& own_;
  const DitWork& w_;
  std::int64_t text_;
  std::int64_t image_;
};

}  // namespace

Status QwenImageRunner::Step(std::uint32_t index, bool hash, std::string* sha) {
  State& s = *s_;
  const md::QwenImageDenoiserProfile& p = s.profile.denoiser;
  const Component& c = s.c[1];
  const bool first = index == 0;
  const auto sinus = md::QwenImageTimestepSinusoid(p, s.schedule.timesteps[index]);
  // dt is a 0-dim F32 tensor times the BF16 noise: PyTorch rounds it to
  // BF16 first (the resident harness's EulerStep, rounding on).
  const float dt = s.schedule.sigmas[index + 1] - s.schedule.sigmas[index];
  const std::int64_t latent_elems = s.image * p.in_channels;
  const auto started = Clock::now();
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    void* const st = native.handle;
    auto* const cs = static_cast<cudaStream_t>(st);
    DitStep dit(p, s.cublas->native(), st, c.tensor, s.own, s.dw, s.text, s.image);
    std::uint64_t at = 0;
    const auto upload = [&](std::uint64_t dst, const void* src, std::uint64_t bytes) -> Status {
      std::memcpy(s.in + at, src, bytes);
      auto r = Cuda(cudaMemcpyAsync(At<void>(dst), s.in + at, bytes, cudaMemcpyHostToDevice, cs),
                    "an upload");
      at += Round(bytes, 256);
      return r;
    };
    Status r;
    if (first) {
      // The resident harness's Prepare (the rotary frequencies, the text
      // rows), then the initial latents.
      r = upload(s.own.freqs, s.freqs.data(), s.freqs.size() * 4);
      r = r ? dit.TextRows() : r;
      r = r ? upload(s.own.latents, s.noise0.data(), s.noise0.size() * 2) : r;
    }
    r = r ? upload(s.dw.sin, sinus.data(), sinus.size() * 2) : r;
    r = r ? dit.Step(first) : r;
    r = r ? ki::EulerStep(At<Bf16>(s.own.latents), At<Bf16>(s.own.noise), At<Bf16>(s.own.latents),
                          dt, true, latent_elems, st)
          : r;
    if (r && hash) {
      r = Cuda(cudaMemcpyAsync(s.out, At<void>(s.own.noise),
                               static_cast<std::size_t>(s.image * p.out_channels * 2),
                               cudaMemcpyDeviceToHost, cs),
               "the noise prediction");
    }
    if (!r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(c.closure, std::move(job), "a denoising step", stream_);
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return posted;
  }
  s.steps.push_back(Seconds(Clock::now() - started));
  if (hash && sha != nullptr) {
    jitllm::base::Sha256 h;
    h.Update(std::span(s.out, static_cast<std::size_t>(s.image * p.out_channels * 2)));
    *sha = jitllm::base::ToHex(h.Finish());
  }
  return {};
}

// ------------------------------------------------------------------ decode

// The VAE's weights rounded to BF16 in the workspace, then the decoder
// (qwen_image_exec.cc's VaeDecoder).
Status QwenImageRunner::Decode(std::string& sha) {
  State& s = *s_;
  const md::QwenImageVaeProfile& p = s.profile.vae;
  const Component& c = s.c[2];
  const VaeWork& w = s.vw;
  const std::uint32_t grid = s.grid;
  const auto steps = md::VaeDecoderPlan(p, grid, grid);
  cublasContext* const blas = s.cublas->native();
  const std::size_t count = std::size_t{p.out_channels} * o_.size * o_.size;
  const auto started = Clock::now();
  Status ran;
  auto job = [&](providers::NativeStream native) -> sc::JobResult {
    void* const st = native.handle;
    auto* const cs = static_cast<cudaStream_t>(st);
    Status r;
    for (std::size_t i = 0; r && i < c.tensor.size(); ++i) {
      r = ki::F32ToBf16(At<float>(c.tensor[i]), At<Bf16>(w.weights[i]),
                        static_cast<std::int64_t>(s.vae_f32_bytes[i] / 4), st);
    }
    const std::int64_t pixels0 = std::int64_t{grid} * grid;
    std::vector<Bf16> stats(std::size_t{2} * 64);
    for (std::size_t ch = 0; ch < 64; ++ch) {
      stats[ch] = md::ToBf16(p.latents_std.at(ch));
      stats[64 + ch] = md::ToBf16(p.latents_mean.at(ch));
    }
    std::memcpy(s.in, stats.data(), stats.size() * 2);
    r = r ? Cuda(cudaMemcpyAsync(At<void>(w.stats), s.in, stats.size() * 2, cudaMemcpyHostToDevice,
                                 cs),
                 "latent statistics")
          : r;
    const auto buf = [&](std::uint8_t i) { return At<Bf16>(w.buf.at(i)); };
    // _unpack_latents, then latents * std + mean in BF16.
    r = r ? ki::Transpose(At<Bf16>(s.own.latents), buf(md::kVaeH), pixels0, p.z_dim, st) : r;
    r = r ? ki::ScaleShiftChannels(buf(md::kVaeH), At<Bf16>(w.stats), At<Bf16>(w.stats) + 64,
                                   buf(md::kVaeX), p.z_dim, pixels0, st)
          : r;
    const auto weight = [&](std::int32_t i) {
      return At<Bf16>(w.weights[static_cast<std::size_t>(i)]);
    };
    for (const md::VaeStep& step : steps) {
      if (!r) {
        break;
      }
      using K = md::VaeStep::Kind;
      const std::int64_t pixels = std::int64_t{step.height} * step.width;
      Bf16* in = buf(step.in);
      Bf16* out = buf(step.out);
      Bf16* aux = buf(step.aux);
      switch (step.kind) {
        case K::kConv1x1:
          r = ki::FillBias(weight(step.bias), out, step.out_channels, pixels, st);
          r = r ? ki::ConvProduct(blas, weight(step.weight), step.out_channels, step.in_channels,
                                  in, pixels, out, pixels, pixels, true)
                : r;
          break;
        case K::kConv3x3: {
          r = ki::FillBias(weight(step.bias), out, step.out_channels, pixels, st);
          const std::int64_t inner = std::int64_t{step.in_channels} * 9;
          std::int64_t chunk = static_cast<std::int64_t>(kCol / 2) / inner;
          chunk = std::min<std::int64_t>(pixels, chunk / step.width * step.width);
          if (chunk <= 0) {
            r = Error("im2col buffer smaller than one image row");
          }
          for (std::int64_t p0 = 0; r && p0 < pixels; p0 += chunk) {
            const std::int64_t n = std::min(chunk, pixels - p0);
            r = ki::Im2Col3x3(in, step.in_channels, step.height, step.width, p0, n, At<Bf16>(w.col),
                              st);
            // col's rows are `n` long, out's `pixels`, and `n` pixels.
            // NOLINTNEXTLINE(readability-suspicious-call-argument)
            r = r ? ki::ConvProduct(blas, weight(step.weight), step.out_channels, inner,
                                    At<Bf16>(w.col), n, out + p0, pixels, n, true)
                  : r;
          }
          break;
        }
        case K::kNormSilu:
        case K::kNorm:
          r = ki::ChannelRmsNorm(in, weight(step.weight), out, step.in_channels, pixels,
                                 step.kind == K::kNormSilu, st);
          break;
        case K::kAdd:
          r = ki::Add(in, aux, out, std::int64_t{step.out_channels} * pixels, st);
          break;
        case K::kCopy:
          r = Cuda(cudaMemcpyAsync(out, in, static_cast<std::size_t>(step.in_channels * pixels * 2),
                                   cudaMemcpyDeviceToDevice, cs),
                   "a block input");
          break;
        case K::kUpsample:
          r = ki::Upsample2x(in, out, step.in_channels, step.height, step.width, st);
          break;
        case K::kAttention: {
          const std::int64_t ch = step.in_channels;
          if (std::cmp_greater(pixels * pixels * 4, w.scores_bytes)) {
            r = Error("the VAE attention's scores exceed their buffer");
            break;
          }
          r = ki::ScoresQtK(blas, in, in + (ch * pixels), At<float>(w.scores), ch, pixels, pixels);
          r = r ? ki::SoftmaxRowsToBf16(At<float>(w.scores), At<Bf16>(w.probs), pixels, pixels,
                                        1.0f / std::sqrt(static_cast<float>(ch)), st)
                : r;
          r = r ? ki::ValuesTimesProbs(blas, in + (2 * ch * pixels), At<Bf16>(w.probs), out, ch,
                                       pixels, pixels)
                : r;
          break;
        }
        case K::kAddDupUp:
          r = ki::AddDupUp(out, aux, step.in_channels, step.out_channels, step.factor_t,
                           step.height, step.width, st);
          break;
      }
    }
    r = r ? Cuda(cudaMemcpyAsync(s.out, buf(md::kVaeX), count * 2, cudaMemcpyDeviceToHost, cs),
                 "the decoded image")
          : r;
    if (!r) {
      ran = r;
      return sc::JobResult::kUnknown;
    }
    return sc::JobResult::kQueued;
  };
  const Status posted = node_.Job(c.closure, std::move(job), "the image's decode", stream_);
  if (!ran) {
    return ran;
  }
  if (!posted) {
    return posted;
  }
  const std::span<const Bf16> decoded(reinterpret_cast<const Bf16*>(s.out), count);
  const auto pixels = md::QwenImagePixels(decoded, p.out_channels, o_.size, o_.size);
  sha =
      jitllm::base::ToHex(jitllm::base::Sha256().Update(std::as_bytes(std::span(pixels))).Finish());
  s.decode = Seconds(Clock::now() - started);
  return {};
}

// ------------------------------------------------------------------ the pipeline

Status QwenImageRunner::FirstOutput(std::string& sha) {
  s_->steps.clear();
  if (auto r = Encode(); !r) {
    return r;
  }
  return Step(0, true, &sha);
}

Status QwenImageRunner::Finish(std::string& sha) {
  for (std::uint32_t i = 1; i < o_.steps; ++i) {
    if (auto r = Step(i, false, nullptr); !r) {
      return r;
    }
  }
  if (auto r = Decode(sha); !r) {
    return r;
  }
  ++s_->generations;
  return {};
}

std::vector<std::filesystem::path> QwenImageRunner::data() const {
  std::vector<std::filesystem::path> dirs;
  for (const Component& c : s_->c) {
    if (c.artifact) {
      dirs.push_back(o_.store / c.artifact->id() / "data");
    }
  }
  return dirs;
}

std::string QwenImageRunner::Report() const {
  const State& s = *s_;
  std::vector<double> later(s.steps.begin() + (s.steps.empty() ? 0 : 1), s.steps.end());
  std::ranges::sort(later);
  std::string components;
  const std::array<std::string_view, 3> names = {"text_encoder", "transformer", "vae"};
  for (std::size_t i = 0; i < s.c.size(); ++i) {
    components +=
        std::format(R"({}"{}":{{"extents":{},"read_bytes":{}}})", i == 0 ? "" : ",", names.at(i),
                    s.c.at(i).weights.extents().size(), s.c.at(i).weights.read_bytes());
  }
  return std::format(
      R"({{"components":{{{}}},"own_bytes":{},"work_bytes":{},"text_rows":{},"generations":{},)"
      R"("last":{{"encode":{:.6f},"first_step":{:.6f},"step_median":{:.6f},"decode":{:.6f}}}}})",
      components, s.own_bytes, work_bytes_, s.text, s.generations, s.encode,
      s.steps.empty() ? 0.0 : s.steps.front(), later.empty() ? 0.0 : later[later.size() / 2],
      s.decode);
}

Status QwenImageRunner::Release() {
  State& s = *s_;
  std::vector<std::string> problems;
  s.cublas.reset();
  auto& memory = node_.memory();
  for (ts::Mapped* mapped : {&s.own_memory, &s.cublas_workspace}) {
    if (!mapped->reservation.valid()) {
      continue;
    }
    bool released =
        mapped->backings.empty() ||
        memory.Unmap(mapped->reservation, Bytes(0), Bytes(mapped->backings.size() * kExtent))
            .has_value();
    for (const auto backing : mapped->backings) {
      released = memory.Release(backing).has_value() && released;
    }
    if (!released || !memory.Free(mapped->reservation)) {
      problems.push_back(std::format("{} could not be released", mapped->name));
    }
    mapped->reservation = {};
  }
  for (Component& c : s.c) {
    if (auto r = c.weights.Release(memory); !r) {
      problems.push_back(r.error());
    }
  }
  if (problems.empty()) {
    return {};
  }
  std::string all;
  for (const std::string& problem : problems) {
    all += (all.empty() ? "" : "; ") + problem;
  }
  return Error(all);
}

}  // namespace jitllm::benchmarks
