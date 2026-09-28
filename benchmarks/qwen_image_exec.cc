// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's third model slice (docs/experiments/qwen-image-native/README.md):
// the Qwen-Image-2.1 BF16 pipeline run natively on one Spark from its D-056
// component artifacts and their D-089 composition, for the comparison with
// diffusers and a coarse speed and memory report.
//
//   jitllm_qwen_image_exec --store DIR --composition ID --out DIR
//                          [--prompt TEXT] [--size N] [--steps N] [--stop-after N]
//                          [--reference DIR] [--embeds native|reference]
//                          [--noise FILE] [--force-latents] [--decode-reference]
//                          [--no-vae] [--phases resident|released]
//                          [--attention native|ggml] [--runs N] [--profile]
//
// - The composition is opened as untrusted input (artifact/composition.h)
//   and each component's artifact by the ID it names; each is bound to the
//   compiled-in profile (model/qwen_image.h) and its groups read with
//   direct I/O through pinned staging into device memory (the VAE's F32
//   weights converted to BF16 on the way, as diffusers loads them). Only
//   the groups a phase reads are loaded: the text encoder's table and
//   layers (not its vision tower or head), the denoiser, the VAE's decoder.
// - Phases, each over its component: encode (Qwen3-VL's text path, the
//   system turn's tokens dropped), denoise (the block-causal DiT with the
//   prefix K/V cache and the flow-matching Euler scheduler, no guidance),
//   decode (the VAE decoder). With --phases released, each component is
//   loaded at its phase's start and released at its end; resident loads
//   all three first (as the diffusers baseline holds them).
// - Kernels: jitLLM's BF16 kernels (kernels/image/ops.h), its
//   FlashAttention-2 kernel for the denoiser, cuBLAS BF16 products
//   (kernels/image/gemm.h); --attention ggml takes GGML's tensor-core flash
//   attention at D = 128 through the K-C launch context instead
//   (kernels/ggml/ops_ext.h FlashAttnMma128), the A/B's other arm.
// - --stop-after N runs only the first N steps of the schedule;
//   --decode-reference decodes diffusers' final latents (the VAE alone).
// - --reference: the diffusers tensors reference.py wrote; each phase's
//   output is compared with them (relative RMS, cosine, max |diff|), and
//   --embeds reference / --force-latents feed diffusers' prompt embeddings
//   and each step's latents instead of the native ones (teacher forcing,
//   isolating one component). --noise takes the initial latents (default:
//   the reference's latents_init).
// - Writes the prompt embeddings, every step's noise prediction, the final
//   latents, the decoded tensor and the image's RGBA pixels as raw files,
//   and report.json with the timings (per phase, per step, per full
//   generation over --runs plain runs), peak memory (the drop in
//   MemAvailable) and the comparisons.

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <print>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/bytes.h"
#include "base/sha256.h"
#include "chat/chat.h"
#include "ggml.h"
#include "kernels/ggml/cublas.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/ops_ext.h"
#include "kernels/ggml/tensors.h"
#include "kernels/image/gemm.h"
#include "kernels/image/ops.h"
#include "model/qwen_image.h"
#include "providers/cuda/cuda_device_execution.h"
#include "providers/device_execution.h"
#include "tokenizer/hf.h"
#include "tokenizer/tokenizer.h"

namespace {

namespace ja = jitllm::artifact;
namespace kg = jitllm::kernels::ggml;
namespace ki = jitllm::kernels::image;
namespace md = jitllm::model;
using jitllm::base::Bytes;
using Status = std::expected<void, std::string>;
using Clock = std::chrono::steady_clock;
using Bf16 = std::uint16_t;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

Status Cuda(cudaError_t result, std::string_view what) {
  if (result != cudaSuccess) {
    return Error(std::format("{}: {}", what, cudaGetErrorString(result)));
  }
  return {};
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }
std::uint64_t Round(std::uint64_t bytes, std::uint64_t to) { return (bytes + to - 1) / to * to; }

template <typename T>
T* At(std::uint64_t address) {
  return reinterpret_cast<T*>(address);  // NOLINT(performance-no-int-to-ptr)
}

std::uint64_t MemAvailable() {
  std::ifstream file("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0;
  std::string unit;
  while (file >> key >> value) {
    std::getline(file, unit);
    if (key == "MemAvailable:") {
      return value * 1024;
    }
  }
  return 0;
}

// The lowest MemAvailable seen while it runs, sampled every 50 ms.
class MemorySampler {
 public:
  MemorySampler() : low_(MemAvailable()), thread_([this] { Loop(); }) {}
  MemorySampler(const MemorySampler&) = delete;
  MemorySampler& operator=(const MemorySampler&) = delete;
  MemorySampler(MemorySampler&&) = delete;
  MemorySampler& operator=(MemorySampler&&) = delete;
  ~MemorySampler() {
    stop_ = true;
    thread_.join();
  }
  std::uint64_t low() const { return low_.load(); }
  void Reset() { low_ = MemAvailable(); }

 private:
  void Loop() {
    while (!stop_) {
      const std::uint64_t now = MemAvailable();
      std::uint64_t seen = low_.load();
      while (now < seen && !low_.compare_exchange_weak(seen, now)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  std::atomic<std::uint64_t> low_;
  std::atomic<bool> stop_{false};
  std::thread thread_;
};

// A cudaMalloc'd region, freed when it goes.
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& o) noexcept
      : address_(std::exchange(o.address_, 0)), bytes_(std::exchange(o.bytes_, 0)) {}
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    std::swap(address_, o.address_);
    std::swap(bytes_, o.bytes_);
    return *this;
  }
  ~DeviceBuffer() { Free(); }
  Status Allocate(std::uint64_t bytes, std::string_view what) {
    Free();
    void* p = nullptr;
    if (auto r = Cuda(cudaMalloc(&p, std::max<std::uint64_t>(bytes, 256)), what); !r) {
      return r;
    }
    address_ = reinterpret_cast<std::uintptr_t>(p);
    bytes_ = bytes;
    return {};
  }
  void Free() {
    if (address_ != 0) {
      (void)cudaFree(At<void>(address_));
      address_ = 0;
      bytes_ = 0;
    }
  }
  std::uint64_t address() const { return address_; }
  std::uint64_t bytes() const { return bytes_; }
  template <typename T>
  T* as() const {
    return At<T>(address_);
  }

 private:
  std::uint64_t address_ = 0;
  std::uint64_t bytes_ = 0;
};

class Device {
 public:
  static std::expected<std::unique_ptr<Device>, std::string> Open() {
    if (auto set = Cuda(cudaSetDevice(0), "cudaSetDevice"); !set) {
      return std::unexpected(set.error());
    }
    if (auto context = Cuda(cudaFree(nullptr), "the CUDA context"); !context) {
      return std::unexpected(context.error());
    }
    auto execution = jitllm::providers::cuda::OpenDeviceExecution(0);
    if (!execution) {
      return Error("OpenDeviceExecution failed");
    }
    auto stream = (*execution)->CreateStream();
    if (!stream) {
      return Error("CreateStream failed");
    }
    auto device = std::unique_ptr<Device>(new Device(std::move(*execution), *stream));  // NOLINT
    const auto native = device->execution_->Submission(device->stream_);
    if (!native) {
      return Error("Submission failed");
    }
    device->native_ = static_cast<cudaStream_t>(native->handle);
    return device;
  }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  ~Device() {
    if (!Finish() || !execution_->DestroyStream(stream_)) {
      std::println(stderr, "the stream could not be retired");
    }
  }
  jitllm::providers::DeviceExecution& execution() { return *execution_; }
  jitllm::providers::StreamId stream() const { return stream_; }
  cudaStream_t native() const { return native_; }
  void* s() const { return native_; }
  Status Finish() {
    const auto fence = execution_->Record(stream_);
    if (!fence) {
      return Error("Record failed");
    }
    const auto deadline = Clock::now() + std::chrono::minutes(5);
    for (;;) {
      const auto state = execution_->Query(*fence);
      if (!state) {
        return Error("Query failed");
      }
      if (*state == jitllm::providers::FenceState::kComplete) {
        break;
      }
      if (Clock::now() > deadline) {
        return Error("the stream did not complete");
      }
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if (!execution_->Release(*fence)) {
      return Error("Release failed");
    }
    return Cuda(cudaGetLastError(), "the stream");
  }

 private:
  Device(std::unique_ptr<jitllm::providers::DeviceExecution> execution,
         jitllm::providers::StreamId stream)
      : execution_(std::move(execution)), stream_(stream) {}
  std::unique_ptr<jitllm::providers::DeviceExecution> execution_;
  jitllm::providers::StreamId stream_;
  cudaStream_t native_ = nullptr;
};

// Waits for the stream when it leaves scope. Declared after a phase's
// scratch buffers, it runs before they are freed on every return path, so
// no buffer is freed under queued work (a free is not a completion).
class FinishFirst {
 public:
  explicit FinishFirst(Device& d) : d_(d) {}
  FinishFirst(const FinishFirst&) = delete;
  FinishFirst& operator=(const FinishFirst&) = delete;
  FinishFirst(FinishFirst&&) = delete;
  FinishFirst& operator=(FinishFirst&&) = delete;
  ~FinishFirst() {
    if (auto r = d_.Finish(); !r) {
      std::println(stderr, "a phase's work did not finish: {}", r.error());
    }
  }

 private:
  Device& d_;
};

// ------------------------------------------------------------------ weights

struct Loaded {
  DeviceBuffer region;
  std::vector<std::uint64_t> group_address;         // 0: not loaded
  std::vector<std::vector<std::byte>> host_groups;  // host loads (the VAE)
  std::vector<std::uint64_t> tensor_address;        // by the component's tensor list
  std::uint64_t bytes_read = 0;
  double seconds = 0;
};

// Reads the groups holding `resources`, into one device region (or, with
// `host`, into host memory), with direct reads through pinned staging.
Status LoadGroups(const ja::Artifact& artifact, std::span<const std::uint32_t> resources, bool host,
                  Device& device, Loaded& w) {
  const auto start = Clock::now();
  const auto groups = artifact.groups();
  std::vector<bool> want(groups.size(), false);
  for (const std::uint32_t r : resources) {
    want[artifact.resources()[r].group] = true;
  }
  w.group_address.assign(groups.size(), 0);
  w.host_groups.assign(groups.size(), {});
  std::uint64_t total = 0;
  std::vector<std::uint64_t> offset(groups.size(), 0);
  for (std::size_t g = 0; g < groups.size(); ++g) {
    if (want[g]) {
      offset[g] = total;
      total += Round(groups[g].stored.value(), 256);
      if (host) {
        w.host_groups[g].assign(groups[g].stored.value(), std::byte{0});
      }
    }
  }
  if (!host) {
    if (auto r = w.region.Allocate(total, "a component's weights"); !r) {
      return r;
    }
    for (std::size_t g = 0; g < groups.size(); ++g) {
      if (want[g]) {
        w.group_address[g] = w.region.address() + offset[g];
      }
    }
  }
  std::vector<ja::ChunkKey> chunks;
  for (std::uint32_t g = 0; g < groups.size(); ++g) {
    for (std::uint32_t c = 0; want[g] && c < groups[g].chunks; ++c) {
      chunks.push_back({.group = g, .chunk = c});
    }
  }
  const ja::ReadLimits limits{};
  const auto runs = artifact.PlanReads(chunks, {}, limits);
  if (!runs) {
    return Error("the artifact's read plan was refused");
  }
  const std::uint64_t staging_bytes = limits.max_run.value();
  std::array<void*, 2> staging{};
  for (void*& s : staging) {
    if (auto r = Cuda(cudaMallocHost(&s, staging_bytes), "pinned staging"); !r) {
      return r;
    }
  }
  std::vector<ja::FileDescriptor> shards;
  for (std::uint32_t s = 0; s < artifact.shards().size(); ++s) {
    auto fd = artifact.OpenShardForDirectRead(s);
    if (!fd) {
      return Error(std::format("shard {} cannot be opened for direct reads", s));
    }
    shards.push_back(std::move(*fd));
  }
  std::array<cudaEvent_t, 2> copied{};
  for (cudaEvent_t& e : copied) {
    if (auto r = Cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "an event"); !r) {
      return r;
    }
  }
  std::array<bool, 2> pending = {false, false};
  std::size_t which = 0;
  Status status;
  for (const auto& run : *runs) {
    if (pending[which]) {
      if (auto r = Cuda(cudaEventSynchronize(copied[which]), "a staging copy"); !r) {
        status = r;
        break;
      }
    }
    auto* bytes = static_cast<std::byte*>(staging[which]);
    std::uint64_t done = 0;
    while (done < run.length.value()) {
      const ssize_t got = pread(shards[run.shard].get(), bytes + done, run.length.value() - done,
                                static_cast<off_t>(run.file_offset.value() + done));
      if (got <= 0) {
        return Error(std::format("a direct read of shard {} failed", run.shard));
      }
      done += static_cast<std::uint64_t>(got);
    }
    w.bytes_read += done;
    std::uint64_t at = 0;
    for (const auto& segment : run.segments) {
      const std::uint64_t n = segment.length.value();
      if (host) {
        std::memcpy(w.host_groups[segment.chunk.group].data() + segment.group_offset.value(),
                    bytes + at, n);
      } else {
        const std::uint64_t dst =
            w.group_address[segment.chunk.group] + segment.group_offset.value();
        if (auto r = Cuda(cudaMemcpyAsync(At<void>(dst), bytes + at, n, cudaMemcpyHostToDevice,
                                          device.native()),
                          "a weight upload");
            !r) {
          return r;
        }
      }
      at += n;
    }
    if (!host) {
      if (auto r = Cuda(cudaEventRecord(copied[which], device.native()), "an event record"); !r) {
        return r;
      }
      pending[which] = true;
    }
    which ^= 1U;
  }
  if (auto r = device.Finish(); !r) {
    return r;
  }
  for (cudaEvent_t e : copied) {
    (void)cudaEventDestroy(e);
  }
  for (void* s : staging) {
    (void)cudaFreeHost(s);
  }
  if (!status) {
    return status;
  }
  w.seconds = Seconds(Clock::now() - start);
  return {};
}

// A BF16 component: device addresses of its bound tensors.
Status LoadBf16(const ja::Artifact& a, std::span<const std::uint32_t> bound, Device& d, Loaded& w) {
  if (auto r = LoadGroups(a, bound, false, d, w); !r) {
    return r;
  }
  w.tensor_address.clear();
  for (const std::uint32_t r : bound) {
    const auto& res = a.resources()[r];
    w.tensor_address.push_back(w.group_address[res.group] + res.offset.value());
  }
  return {};
}

// The VAE: its F32 tensors read to the host, rounded to BF16 (as
// from_pretrained(torch_dtype=bfloat16) casts them) into a device region.
Status LoadVae(const ja::Artifact& a, std::span<const std::uint32_t> bound, Device& d, Loaded& w) {
  if (auto r = LoadGroups(a, bound, true, d, w); !r) {
    return r;
  }
  const auto start = Clock::now();
  std::uint64_t total = 0;
  std::vector<std::uint64_t> offsets;
  for (const std::uint32_t r : bound) {
    offsets.push_back(total);
    total += Round(a.resources()[r].bytes.value() / 2, 256);
  }
  if (auto r = w.region.Allocate(total, "the VAE's BF16 weights"); !r) {
    return r;
  }
  std::vector<Bf16> host(total / 2);
  w.tensor_address.clear();
  for (std::size_t i = 0; i < bound.size(); ++i) {
    const auto& res = a.resources()[bound[i]];
    const std::byte* src = w.host_groups[res.group].data() + res.offset.value();
    const std::uint64_t n = res.bytes.value() / 4;
    for (std::uint64_t e = 0; e < n; ++e) {
      float v = 0;
      std::memcpy(&v, src + (e * 4), 4);
      host[(offsets[i] / 2) + e] = md::ToBf16(v);
    }
    w.tensor_address.push_back(w.region.address() + offsets[i]);
  }
  if (auto r = Cuda(cudaMemcpy(w.region.as<void>(), host.data(), total, cudaMemcpyHostToDevice),
                    "the VAE's weights");
      !r) {
    return r;
  }
  w.host_groups.clear();
  w.seconds += Seconds(Clock::now() - start);
  return {};
}

// ------------------------------------------------------------------ comparison

struct Compared {
  double rel_rms = 0;
  double cosine = 0;
  double max_abs = 0;
};

Compared CompareBf16(std::span<const Bf16> a, std::span<const Bf16> b) {
  double dd = 0;
  double bb = 0;
  double aa = 0;
  double ab = 0;
  double mx = 0;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
    const double x = md::FromBf16(a[i]);
    const double y = md::FromBf16(b[i]);
    dd += (x - y) * (x - y);
    bb += y * y;
    aa += x * x;
    ab += x * y;
    mx = std::max(mx, std::abs(x - y));
  }
  return {.rel_rms = std::sqrt(dd / std::max(bb, 1e-300)),
          .cosine = ab / std::sqrt(std::max(aa * bb, 1e-300)),
          .max_abs = mx};
}

std::string Json(const Compared& c) {
  return std::format(R"({{"rel_rms": {:.6g}, "cosine": {:.9f}, "max_abs": {:.6g}}})", c.rel_rms,
                     c.cosine, c.max_abs);
}

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

Status WriteBytes(const std::filesystem::path& p, const void* data, std::size_t bytes) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  return out ? Status{} : Error(std::format("{}: write failed", p.string()));
}

template <typename T>
std::expected<std::vector<T>, std::string> Download(std::uint64_t address, std::size_t count) {
  std::vector<T> out(count);
  if (auto r =
          Cuda(cudaMemcpy(out.data(), At<void>(address), count * sizeof(T), cudaMemcpyDeviceToHost),
               "a download");
      !r) {
    return std::unexpected(r.error());
  }
  return out;
}

// ------------------------------------------------------------------ text encoder

class TextEncoder {
 public:
  TextEncoder(const md::QwenImageTextProfile& p, cublasContext* blas, Device& d)
      : p_(p), blas_(blas), d_(d) {}

  // ids: every token of the rendered prompt; keeps rows [drop, rows).
  Status Run(const Loaded& w, std::span<const std::int32_t> ids, std::size_t drop,
             DeviceBuffer& embeds) {
    const auto rows = static_cast<std::int64_t>(ids.size());
    const std::int64_t width = p_.width;
    const std::int64_t q = std::int64_t{p_.heads} * p_.head_dim;
    const std::int64_t kv = std::int64_t{p_.kv_heads} * p_.head_dim;
    const auto t = [&](std::size_t layer, md::TextTensor which) {
      return At<Bf16>(
          w.tensor_address[1 + (layer * static_cast<std::size_t>(md::TextTensor::kCount)) +
                           static_cast<std::size_t>(which)]);
    };
    DeviceBuffer x;
    DeviceBuffer n;
    DeviceBuffer qb;
    DeviceBuffer kb;
    DeviceBuffer vb;
    DeviceBuffer attn;
    DeviceBuffer o;
    DeviceBuffer g;
    DeviceBuffer u;
    DeviceBuffer ids_d;
    DeviceBuffer bad;
    DeviceBuffer cos;
    DeviceBuffer sin;
    const FinishFirst finish(d_);  // before the scratch above is freed
    for (auto [buf, bytes] : {std::pair{&x, rows * width * 2},
                              {&n, rows * width * 2},
                              {&qb, rows * q * 2},
                              {&kb, rows * kv * 2},
                              {&vb, rows * kv * 2},
                              {&attn, rows * q * 2},
                              {&o, rows * width * 2},
                              {&g, rows * std::int64_t{p_.ffn} * 2},
                              {&u, rows * std::int64_t{p_.ffn} * 2},
                              {&ids_d, rows * 4},
                              {&bad, 4},
                              {&cos, rows * 256},
                              {&sin, rows * 256}}) {
      if (auto r = buf->Allocate(static_cast<std::uint64_t>(bytes), "text encoder buffers"); !r) {
        return r;
      }
    }
    const auto rot = md::QwenImageTextRotary(p_, static_cast<std::uint32_t>(rows));
    auto* const s = d_.native();
    for (auto [dst, src, bytes] :
         {std::tuple{ids_d.as<void>(), static_cast<const void*>(ids.data()), rows * 4},
          {cos.as<void>(), rot.cos.data(), rows * 256},
          {sin.as<void>(), rot.sin.data(), rows * 256}}) {
      if (auto r = Cuda(
              cudaMemcpyAsync(dst, src, static_cast<std::size_t>(bytes), cudaMemcpyHostToDevice, s),
              "text inputs");
          !r) {
        return r;
      }
    }
    if (auto r = Cuda(cudaMemsetAsync(bad.as<void>(), 0, 4, s), "a flag"); !r) {
      return r;
    }
    if (auto r = ki::EmbedRows(At<Bf16>(w.tensor_address[0]), p_.vocab, ids_d.as<std::int32_t>(),
                               rows, width, x.as<Bf16>(), bad.as<std::int32_t>(), s);
        !r) {
      return r;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(p_.head_dim));
    for (std::size_t l = 0; l < p_.layers; ++l) {
      using T = md::TextTensor;
      Status r =
          ki::RmsNorm(x.as<Bf16>(), t(l, T::kInputNorm), n.as<Bf16>(), rows, width, p_.rms_eps, s);
      r = r ? ki::Linear(blas_, n.as<Bf16>(), width, t(l, T::kQ), width, qb.as<Bf16>(), q, rows, q,
                         width)
            : r;
      r = r ? ki::Linear(blas_, n.as<Bf16>(), width, t(l, T::kK), width, kb.as<Bf16>(), kv, rows,
                         kv, width)
            : r;
      r = r ? ki::Linear(blas_, n.as<Bf16>(), width, t(l, T::kV), width, vb.as<Bf16>(), kv, rows,
                         kv, width)
            : r;
      r = r ? ki::HeadNormRopeNeox(qb.as<Bf16>(), q, rows, p_.heads, t(l, T::kQNorm),
                                   cos.as<Bf16>(), sin.as<Bf16>(), p_.rms_eps, s)
            : r;
      r = r ? ki::HeadNormRopeNeox(kb.as<Bf16>(), kv, rows, p_.kv_heads, t(l, T::kKNorm),
                                   cos.as<Bf16>(), sin.as<Bf16>(), p_.rms_eps, s)
            : r;
      r = r ? ki::SmallAttention(qb.as<Bf16>(), q, kb.as<Bf16>(), kv, vb.as<Bf16>(), kv,
                                 attn.as<Bf16>(), q, rows, p_.heads, p_.kv_heads, true, scale, s)
            : r;
      r = r ? ki::Linear(blas_, attn.as<Bf16>(), q, t(l, T::kO), q, o.as<Bf16>(), width, rows,
                         width, q)
            : r;
      r = r ? ki::Add(x.as<Bf16>(), o.as<Bf16>(), x.as<Bf16>(), rows * width, s) : r;
      r = r ? ki::RmsNorm(x.as<Bf16>(), t(l, T::kPostNorm), n.as<Bf16>(), rows, width, p_.rms_eps,
                          s)
            : r;
      r = r ? ki::Linear(blas_, n.as<Bf16>(), width, t(l, T::kGate), width, g.as<Bf16>(), p_.ffn,
                         rows, p_.ffn, width)
            : r;
      r = r ? ki::Linear(blas_, n.as<Bf16>(), width, t(l, T::kUp), width, u.as<Bf16>(), p_.ffn,
                         rows, p_.ffn, width)
            : r;
      r = r ? ki::SwiGlu(g.as<Bf16>(), p_.ffn, u.as<Bf16>(), p_.ffn, g.as<Bf16>(), rows, p_.ffn, s)
            : r;
      r = r ? ki::Linear(blas_, g.as<Bf16>(), p_.ffn, t(l, T::kDown), p_.ffn, o.as<Bf16>(), width,
                         rows, width, p_.ffn)
            : r;
      r = r ? ki::Add(x.as<Bf16>(), o.as<Bf16>(), x.as<Bf16>(), rows * width, s) : r;
      if (!r) {
        return r;
      }
    }
    const std::int64_t kept = rows - static_cast<std::int64_t>(drop);
    if (auto r = embeds.Allocate(static_cast<std::uint64_t>(kept * width * 2), "prompt embeddings");
        !r) {
      return r;
    }
    if (auto r =
            Cuda(cudaMemcpyAsync(
                     embeds.as<void>(), x.as<Bf16>() + (static_cast<std::int64_t>(drop) * width),
                     static_cast<std::size_t>(kept * width * 2), cudaMemcpyDeviceToDevice, s),
                 "the kept rows");
        !r) {
      return r;
    }
    if (auto r = d_.Finish(); !r) {
      return r;
    }
    std::int32_t flag = 0;
    if (auto r = Cuda(cudaMemcpy(&flag, bad.as<void>(), 4, cudaMemcpyDeviceToHost), "the flag");
        !r) {
      return r;
    }
    return flag == 0 ? Status{} : Error("a token id outside the embedding table");
  }

 private:
  const md::QwenImageTextProfile& p_;
  cublasContext* blas_;
  Device& d_;
};

// ------------------------------------------------------------------ denoiser

// Timing of the step's kernels by kind, with --profile.
struct Profile {
  bool on = false;
  std::map<std::string, double> ms;
  std::vector<std::pair<std::string, cudaEvent_t>> marks;
};

class Denoiser {
 public:
  Denoiser(const md::QwenImageDenoiserProfile& p, cublasContext* blas, kg::LaunchContext& launch,
           Device& d)
      : p_(p), blas_(blas), launch_(launch), d_(d) {}

  // Buffers for `text` prompt rows and a grid x grid image, and the text's
  // projection (txt_in, the same at every step), from the embeddings.
  Status Prepare(const Loaded& w, const DeviceBuffer& embeds, std::int64_t text,
                 std::int64_t grid) {
    w_ = &w;
    const std::int64_t width = p_.width;
    auto* const s = d_.native();
    if (x_.address() != 0) {
      if (text != text_ || grid * grid != image_) {
        return Error("a denoiser's buffers are sized for one prompt length and image size");
      }
      return TextRows(embeds, s);
    }
    text_ = text;
    image_ = grid * grid;
    const std::int64_t rows = text_ + image_;
    for (auto [buf, bytes] : std::initializer_list<std::pair<DeviceBuffer*, std::int64_t>>{
             {&x_, rows * width * 2},
             {&n_, rows * width * 2},
             {&q_, rows * width * 2},
             {&k_, rows * width * 2},
             {&v_, rows * width * 2},
             {&qf_, rows * width * 4},
             {&kh_, rows * width * 2},
             {&vh_, rows * width * 2},
             {&attn_, rows * width * 2},
             {&af_, image_ * width * 4},
             {&o_, rows * width * 2},
             {&g_, rows * std::int64_t{p_.mlp} * 2},
             {&u_, rows * std::int64_t{p_.mlp} * 2},
             {&pk_, std::int64_t{p_.blocks} * text_ * width * 2},
             {&pv_, std::int64_t{p_.blocks} * text_ * width * 2},
             {&freqs_, rows * 128 * 4},
             {&sin_, 2 * std::int64_t{p_.timestep_dim} * 2},
             {&t1_, 2 * width * 2},
             {&t2_, 2 * width * 2},
             {&temb_, 2 * width * 2},
             {&mod_, width * 2 * 4 * 2},
             {&nscale_, 2 * width * 2},
             {&txt_, text_ * width * 2}}) {
      if (auto r = buf->Allocate(static_cast<std::uint64_t>(bytes), "denoiser buffers"); !r) {
        return r;
      }
    }
    const auto freqs = md::QwenImageDenoiserRotary(p_, static_cast<std::uint32_t>(text_),
                                                   static_cast<std::uint32_t>(grid));
    if (auto r = Cuda(
            cudaMemcpy(freqs_.as<void>(), freqs.data(), freqs.size() * 4, cudaMemcpyHostToDevice),
            "rotary frequencies");
        !r) {
      return r;
    }
    return TextRows(embeds, s);
  }

  // txt_in: zero-centred RMSNorm, in_layer, GELU (tanh), out_layer: the
  // text rows of the joint sequence.
  Status TextRows(const DeviceBuffer& embeds, cudaStream_t s) {
    const std::int64_t width = p_.width;
    Status r = ki::ZeroCenterRmsNorm(embeds.as<Bf16>(), G(md::DenoiserGlobal::kTxtNorm),
                                     n_.as<Bf16>(), text_, p_.context, p_.eps, s);
    r = r ? ki::Linear(blas_, n_.as<Bf16>(), p_.context, G(md::DenoiserGlobal::kTxtIn), p_.context,
                       q_.as<Bf16>(), width, text_, width, p_.context)
          : r;
    r = r ? ki::GeluTanh(q_.as<Bf16>(), q_.as<Bf16>(), text_ * width, s) : r;
    r = r ? ki::Linear(blas_, q_.as<Bf16>(), width, G(md::DenoiserGlobal::kTxtOut), width,
                       txt_.as<Bf16>(), width, text_, width, width)
          : r;
    return r;
  }

  // One denoising step: noise[image, out_channels] from latents[image,
  // in_channels], with the sinusoid of step `t`. Step 0 runs the whole
  // joint sequence and fills the prefix cache; later steps only the image.
  Status Step(std::uint32_t index, float t, const Bf16* latents, Bf16* noise, Profile& prof) {
    const bool first = index == 0;
    const std::int64_t width = p_.width;
    const std::int64_t rows = first ? text_ + image_ : image_;
    const std::int64_t row0 = first ? 0 : text_;  // first processed row
    const std::int64_t first_target = first ? text_ : 0;
    auto* const s = d_.native();
    const auto sinus = md::QwenImageTimestepSinusoid(p_, t);
    using G_ = md::DenoiserGlobal;
    Mark(prof, "other");
    Status r = Cuda(
        cudaMemcpyAsync(sin_.as<void>(), sinus.data(), sinus.size() * 2, cudaMemcpyHostToDevice, s),
        "the sinusoid");
    r = r ? ki::Linear(blas_, sin_.as<Bf16>(), p_.timestep_dim, G(G_::kTime1), p_.timestep_dim,
                       t1_.as<Bf16>(), width, 2, width, p_.timestep_dim)
          : r;
    r = r ? ki::Silu(t1_.as<Bf16>(), t1_.as<Bf16>(), 2 * width, s) : r;
    r = r ? ki::Linear(blas_, t1_.as<Bf16>(), width, G(G_::kTime2), width, temb_.as<Bf16>(), width,
                       2, width, width)
          : r;
    r = r ? ki::Silu(temb_.as<Bf16>(), t2_.as<Bf16>(), 2 * width, s) : r;
    r = r ? ki::Linear(blas_, t2_.as<Bf16>(), width, G(G_::kModulation), width, mod_.as<Bf16>(),
                       4 * width, 2, 4 * width, width)
          : r;
    r = r ? ki::Linear(blas_, t2_.as<Bf16>(), width, G(G_::kNormOut), width, nscale_.as<Bf16>(),
                       width, 2, width, width)
          : r;
    // x: [text rows (step 0) | image rows]; the image rows from img_in.
    if (r && first) {
      r = Cuda(
          cudaMemcpyAsync(x_.as<void>(), txt_.as<void>(),
                          static_cast<std::size_t>(text_ * width * 2), cudaMemcpyDeviceToDevice, s),
          "the text rows");
    }
    Mark(prof, "gemm");
    r = r ? ki::Linear(blas_, latents, p_.in_channels, G(G_::kImgIn), p_.in_channels,
                       Row(x_, text_), width, image_, width, p_.in_channels)
          : r;
    if (!r) {
      return r;
    }
    for (std::uint32_t b = 0; b < p_.blocks; ++b) {
      if (auto rb = Block(b, first, rows, row0, first_target, prof); !rb) {
        return rb;
      }
    }
    // norm_out on the image rows, then proj_out.
    Mark(prof, "other");
    r = ki::LayerNormModulate(Row(x_, text_), Row(n_, text_), image_, width, p_.eps,
                              nscale_.as<Bf16>(), width, 0, s);
    Mark(prof, "gemm");
    r = r ? ki::Linear(blas_, Row(n_, text_), width, G(G_::kProjOut), width, noise, p_.out_channels,
                       image_, p_.out_channels, width)
          : r;
    Mark(prof, "other");
    return r;
  }

  // Builds the attention node over this run's buffers: the image rows' q
  // (F32) against every row's K and V (F16), into af_.
  Status BuildAttention(kg::TensorArena& arena) {
    ggml_context* c = arena.context();
    const std::int64_t rows = text_ + image_;
    ggml_tensor* qb = ggml_new_tensor_3d(c, GGML_TYPE_F32, 128, p_.heads, image_);
    ggml_tensor* kb = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, p_.heads, rows);
    ggml_tensor* vb = ggml_new_tensor_3d(c, GGML_TYPE_F16, 128, p_.heads, rows);
    kg::TensorArena::Bind(qb, qf_.address() + static_cast<std::uint64_t>(text_ * p_.width * 4));
    kg::TensorArena::Bind(kb, kh_.address());
    kg::TensorArena::Bind(vb, vh_.address());
    ggml_tensor* q = ggml_permute(c, qb, 0, 2, 1, 3);
    ggml_tensor* k = ggml_permute(c, kb, 0, 2, 1, 3);
    ggml_tensor* v = ggml_permute(c, vb, 0, 2, 1, 3);
    fa_ = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / std::sqrt(128.0f), 0.0f, 0.0f);
    kg::TensorArena::Bind(fa_, af_.address());
    const auto plan = kg::PlanFlashAttnMma128(launch_, fa_);
    if (!plan) {
      return Error("flash attention: " + plan.error().detail);
    }
    fa_scratch_ = plan->scratch;
    fa_columns_ = plan->columns;
    return {};
  }
  std::uint64_t fa_scratch() const { return fa_scratch_; }
  int fa_columns() const { return fa_columns_; }

 private:
  Bf16* G(md::DenoiserGlobal g) const {
    return At<Bf16>(w_->tensor_address[static_cast<std::size_t>(g)]);
  }
  Bf16* B(std::uint32_t block, md::BlockTensor t) const {
    return At<Bf16>(w_->tensor_address[static_cast<std::size_t>(md::DenoiserGlobal::kCount) +
                                       (block * static_cast<std::size_t>(md::BlockTensor::kCount)) +
                                       static_cast<std::size_t>(t)]);
  }
  Bf16* Row(const DeviceBuffer& buf, std::int64_t row, std::int64_t width = 0) const {
    return buf.as<Bf16>() + (row * (width == 0 ? p_.width : width));
  }

  void Mark(Profile& prof, std::string what) {
    if (!prof.on) {
      return;
    }
    cudaEvent_t e = nullptr;
    (void)cudaEventCreate(&e);
    (void)cudaEventRecord(e, d_.native());
    prof.marks.emplace_back(std::move(what), e);
  }

  Status Block(std::uint32_t b, bool first, std::int64_t rows, std::int64_t row0,
               std::int64_t first_target, Profile& prof) {
    using T = md::BlockTensor;
    const std::int64_t width = p_.width;
    const std::int64_t mstride = 4 * width;
    auto* const s = d_.native();
    const Bf16* mod = mod_.as<Bf16>();
    Bf16* x = Row(x_, row0);
    Mark(prof, "other");
    Status r =
        ki::LayerNormModulate(x, Row(n_, row0), rows, width, p_.eps, mod, mstride, first_target, s);
    Mark(prof, "gemm");
    r = r ? ki::Linear(blas_, Row(n_, row0), width, B(b, T::kQ), width, Row(q_, row0), width, rows,
                       width, width)
          : r;
    r = r ? ki::Linear(blas_, Row(n_, row0), width, B(b, T::kK), width, Row(k_, row0), width, rows,
                       width, width)
          : r;
    r = r ? ki::Linear(blas_, Row(n_, row0), width, B(b, T::kV), width, Row(v_, row0), width, rows,
                       width, width)
          : r;
    Mark(prof, "other");
    if (r && native_attention_) {
      return NativeAttention(b, first, rows, row0, first_target, prof, r);
    }
    r = r ? ki::HeadNormRopeComplexF32(Row(q_, row0), width, rows, p_.heads, B(b, T::kQNorm),
                                       freqs_.as<float>() + (row0 * 128), p_.eps,
                                       qf_.as<float>() + (row0 * width), s)
          : r;
    r = r ? ki::HeadNormRopeComplexF16(Row(k_, row0), width, rows, p_.heads, B(b, T::kKNorm),
                                       freqs_.as<float>() + (row0 * 128), p_.eps, Row(kh_, row0), s)
          : r;
    r = r ? ki::Bf16RowsToF16(Row(v_, row0), width, Row(vh_, row0), rows, width, s) : r;
    const auto prefix = static_cast<std::size_t>(text_ * width * 2);
    Bf16* pk = pk_.as<Bf16>() + (static_cast<std::int64_t>(b) * text_ * width);
    Bf16* pv = pv_.as<Bf16>() + (static_cast<std::int64_t>(b) * text_ * width);
    if (r && first) {
      // The prefix cache (K after RoPE, V), and the text rows' own
      // attention: causal among the text.
      r = Cuda(cudaMemcpyAsync(pk, kh_.as<void>(), prefix, cudaMemcpyDeviceToDevice, s), "cache");
      r = r ? Cuda(cudaMemcpyAsync(pv, vh_.as<void>(), prefix, cudaMemcpyDeviceToDevice, s),
                   "cache")
            : r;
      r = r ? ki::SmallAttentionF32F16(qf_.as<float>(), width, kh_.as<Bf16>(), width,
                                       vh_.as<Bf16>(), width, attn_.as<Bf16>(), width, text_,
                                       p_.heads, p_.heads, true, 1.0f / std::sqrt(128.0f), s)
            : r;
    } else if (r) {
      r = Cuda(cudaMemcpyAsync(kh_.as<void>(), pk, prefix, cudaMemcpyDeviceToDevice, s), "cache");
      r = r ? Cuda(cudaMemcpyAsync(vh_.as<void>(), pv, prefix, cudaMemcpyDeviceToDevice, s),
                   "cache")
            : r;
    }
    if (!r) {
      return r;
    }
    Mark(prof, "attention");
    if (auto fa = kg::FlashAttnMma128(launch_, fa_); !fa) {
      return Error("flash attention: " + fa.error().detail);
    }
    Mark(prof, "other");
    r = ki::F32ToBf16(af_.as<float>(), Row(attn_, text_), image_ * width, s);
    return Rest(b, rows, row0, first_target, prof, r);
  }

  // Attention with jitLLM's BF16 FlashAttention: q and k rotated in place,
  // the prefix cache in BF16.
  Status NativeAttention(std::uint32_t b, bool first, std::int64_t rows, std::int64_t row0,
                         std::int64_t first_target, Profile& prof, Status r) {
    using T = md::BlockTensor;
    const std::int64_t width = p_.width;
    auto* const s = d_.native();
    r = ki::HeadNormRopeComplex(Row(q_, row0), width, rows, p_.heads, B(b, T::kQNorm),
                                freqs_.as<float>() + (row0 * 128), p_.eps, s);
    r = r ? ki::HeadNormRopeComplex(Row(k_, row0), width, rows, p_.heads, B(b, T::kKNorm),
                                    freqs_.as<float>() + (row0 * 128), p_.eps, s)
          : r;
    const auto prefix = static_cast<std::size_t>(text_ * width * 2);
    Bf16* pk = pk_.as<Bf16>() + (static_cast<std::int64_t>(b) * text_ * width);
    Bf16* pv = pv_.as<Bf16>() + (static_cast<std::int64_t>(b) * text_ * width);
    const float scale = 1.0f / std::sqrt(128.0f);
    if (r && first) {
      r = Cuda(cudaMemcpyAsync(pk, k_.as<void>(), prefix, cudaMemcpyDeviceToDevice, s), "cache");
      r = r ? Cuda(cudaMemcpyAsync(pv, v_.as<void>(), prefix, cudaMemcpyDeviceToDevice, s), "cache")
            : r;
      r = r ? ki::SmallAttention(q_.as<Bf16>(), width, k_.as<Bf16>(), width, v_.as<Bf16>(), width,
                                 attn_.as<Bf16>(), width, text_, p_.heads, p_.heads, true, scale, s)
            : r;
    } else if (r) {
      r = Cuda(cudaMemcpyAsync(k_.as<void>(), pk, prefix, cudaMemcpyDeviceToDevice, s), "cache");
      r = r ? Cuda(cudaMemcpyAsync(v_.as<void>(), pv, prefix, cudaMemcpyDeviceToDevice, s), "cache")
            : r;
    }
    Mark(prof, "attention");
    r = r ? ki::FlashAttention(Row(q_, text_), width, k_.as<Bf16>(), width, v_.as<Bf16>(), width,
                               Row(attn_, text_), width, image_, text_ + image_, p_.heads, p_.heads,
                               scale, s)
          : r;
    Mark(prof, "other");
    return Rest(b, rows, row0, first_target, prof, r);
  }

  // The block after attention: out projection, gated residual, MLP.
  Status Rest(std::uint32_t b, std::int64_t rows, std::int64_t row0, std::int64_t first_target,
              Profile& prof, Status r) {
    using T = md::BlockTensor;
    const std::int64_t width = p_.width;
    const std::int64_t mlp = p_.mlp;
    const std::int64_t mstride = 4 * width;
    auto* const s = d_.native();
    const Bf16* mod = mod_.as<Bf16>();
    Bf16* x = Row(x_, row0);
    Mark(prof, "gemm");
    r = r ? ki::Linear(blas_, Row(attn_, row0), width, B(b, T::kOut), width, Row(o_, row0), width,
                       rows, width, width)
          : r;
    Mark(prof, "other");
    r = r ? ki::GatedResidual(x, Row(o_, row0), width, rows, width, mod + width, mstride,
                              first_target, s)
          : r;
    r = r ? ki::LayerNormModulate(x, Row(n_, row0), rows, width, p_.eps, mod + (2 * width), mstride,
                                  first_target, s)
          : r;
    Mark(prof, "gemm");
    r = r ? ki::Linear(blas_, Row(n_, row0), width, B(b, T::kGate), width, Row(g_, row0, mlp), mlp,
                       rows, mlp, width)
          : r;
    r = r ? ki::Linear(blas_, Row(n_, row0), width, B(b, T::kProj), width, Row(u_, row0, mlp), mlp,
                       rows, mlp, width)
          : r;
    Mark(prof, "other");
    r = r ? ki::SwiGlu(Row(g_, row0, mlp), mlp, Row(u_, row0, mlp), mlp, Row(g_, row0, mlp), rows,
                       mlp, s)
          : r;
    Mark(prof, "gemm");
    r = r ? ki::Linear(blas_, Row(g_, row0, mlp), mlp, B(b, T::kMlpOut), mlp, Row(o_, row0), width,
                       rows, width, mlp)
          : r;
    Mark(prof, "other");
    r = r ? ki::GatedResidual(x, Row(o_, row0), width, rows, width, mod + (3 * width), mstride,
                              first_target, s)
          : r;
    return r;
  }

  const md::QwenImageDenoiserProfile& p_;
  cublasContext* blas_;
  kg::LaunchContext& launch_;
  Device& d_;
  const Loaded* w_ = nullptr;
  std::int64_t text_ = 0;
  std::int64_t image_ = 0;
  DeviceBuffer x_, n_, q_, k_, v_, qf_, kh_, vh_, attn_, af_, o_, g_, u_, pk_, pv_, freqs_, sin_,
      t1_, t2_, temb_, mod_, nscale_, txt_;
  ggml_tensor* fa_ = nullptr;
  std::uint64_t fa_scratch_ = 0;
  int fa_columns_ = 0;

 public:
  bool native_attention_ = true;
};

// ------------------------------------------------------------------ VAE decoder

class VaeDecoder {
 public:
  VaeDecoder(const md::QwenImageVaeProfile& p, cublasContext* blas, Device& d)
      : p_(p), blas_(blas), d_(d) {}

  // latents: [grid*grid, z] packed; writes [out_channels, 16 grid, 16 grid].
  Status Run(const Loaded& w, const Bf16* latents, std::uint32_t grid, DeviceBuffer& decoded) {
    const auto steps = md::VaeDecoderPlan(p_, grid, grid);
    std::array<std::uint64_t, md::kVaeBuffers> need{};
    for (const md::VaeStep& st : steps) {
      const std::uint64_t in = std::uint64_t{st.in_channels} * st.height * st.width;
      const std::uint64_t scale = st.kind == md::VaeStep::Kind::kUpsample ? 4 : 1;
      const std::uint64_t out = std::uint64_t{st.out_channels} * st.height * st.width * scale;
      need[st.in] = std::max(need[st.in], st.kind == md::VaeStep::Kind::kAttention ? 3 * in : in);
      need[st.out] =
          std::max(need[st.out], st.kind == md::VaeStep::Kind::kAddDupUp ? 4 * out : out);
      need[st.aux] = std::max(need[st.aux], in);
    }
    need[md::kVaeX] =
        std::max<std::uint64_t>(need[md::kVaeX], std::uint64_t{p_.z_dim} * grid * grid);
    std::array<DeviceBuffer, md::kVaeBuffers> buf;
    DeviceBuffer stats;
    DeviceBuffer col;
    DeviceBuffer scores;
    DeviceBuffer probs;
    const FinishFirst finish(d_);  // before the scratch above is freed
    for (std::size_t i = 0; i < buf.size(); ++i) {
      if (auto r = buf[i].Allocate(need[i] * 2, "VAE buffers"); !r) {
        return r;
      }
    }
    auto* const s = d_.native();
    const std::int64_t pixels0 = std::int64_t{grid} * grid;
    if (auto r = stats.Allocate(std::uint64_t{2} * 64 * 2, "latent statistics"); !r) {
      return r;
    }
    std::vector<Bf16> st(std::size_t{2} * 64);
    for (std::size_t c = 0; c < 64; ++c) {
      st[c] = md::ToBf16(p_.latents_std[c]);
      st[64 + c] = md::ToBf16(p_.latents_mean[c]);
    }
    Status r =
        Cuda(cudaMemcpyAsync(stats.as<void>(), st.data(), st.size() * 2, cudaMemcpyHostToDevice, s),
             "latent statistics");
    // _unpack_latents, then latents * std + mean in BF16.
    r = r ? ki::Transpose(latents, buf[md::kVaeH].as<Bf16>(), pixels0, p_.z_dim, s) : r;
    r = r ? ki::ScaleShiftChannels(buf[md::kVaeH].as<Bf16>(), stats.as<Bf16>(),
                                   stats.as<Bf16>() + 64, buf[md::kVaeX].as<Bf16>(), p_.z_dim,
                                   pixels0, s)
          : r;
    if (!r) {
      return r;
    }
    const std::uint64_t col_bytes = std::uint64_t{256} << 20U;
    if (auto a = col.Allocate(col_bytes, "im2col"); !a) {
      return a;
    }
    for (const md::VaeStep& step : steps) {
      if (auto e = Execute(step, w, buf, col, scores, probs); !e) {
        return e;
      }
    }
    decoded = std::move(buf[md::kVaeX]);
    return {};
  }

 private:
  Status Execute(const md::VaeStep& st, const Loaded& w,
                 std::array<DeviceBuffer, md::kVaeBuffers>& buf, DeviceBuffer& col,
                 DeviceBuffer& scores, DeviceBuffer& probs) {
    using K = md::VaeStep::Kind;
    auto* const s = d_.native();
    const std::int64_t pixels = std::int64_t{st.height} * st.width;
    Bf16* in = buf[st.in].as<Bf16>();
    Bf16* out = buf[st.out].as<Bf16>();
    Bf16* aux = buf[st.aux].as<Bf16>();
    const auto weight = [&](std::int32_t i) {
      return At<Bf16>(w.tensor_address[static_cast<std::size_t>(i)]);
    };
    switch (st.kind) {
      case K::kConv1x1: {
        Status r = ki::FillBias(weight(st.bias), out, st.out_channels, pixels, s);
        return r ? ki::ConvProduct(blas_, weight(st.weight), st.out_channels, st.in_channels, in,
                                   pixels, out, pixels, pixels, true)
                 : r;
      }
      case K::kConv3x3: {
        Status r = ki::FillBias(weight(st.bias), out, st.out_channels, pixels, s);
        const std::int64_t inner = std::int64_t{st.in_channels} * 9;
        std::int64_t chunk = static_cast<std::int64_t>(col.bytes() / 2) / inner;
        chunk = std::min<std::int64_t>(pixels, chunk / st.width * st.width);
        if (chunk <= 0) {
          return Error("im2col buffer smaller than one image row");
        }
        for (std::int64_t p0 = 0; r && p0 < pixels; p0 += chunk) {
          const std::int64_t count = std::min(chunk, pixels - p0);
          r = ki::Im2Col3x3(in, st.in_channels, st.height, st.width, p0, count, col.as<Bf16>(), s);
          // col's rows are `count` long, out's `pixels`, and `count` pixels.
          // NOLINTNEXTLINE(readability-suspicious-call-argument)
          r = r ? ki::ConvProduct(blas_, weight(st.weight), st.out_channels, inner, col.as<Bf16>(),
                                  count, out + p0, pixels, count, true)
                : r;
        }
        return r;
      }
      case K::kNormSilu:
      case K::kNorm:
        return ki::ChannelRmsNorm(in, weight(st.weight), out, st.in_channels, pixels,
                                  st.kind == K::kNormSilu, s);
      case K::kAdd:
        return ki::Add(in, aux, out, std::int64_t{st.out_channels} * pixels, s);
      case K::kCopy:
        return Cuda(cudaMemcpyAsync(out, in, static_cast<std::size_t>(st.in_channels * pixels * 2),
                                    cudaMemcpyDeviceToDevice, s),
                    "a block input");
      case K::kUpsample:
        return ki::Upsample2x(in, out, st.in_channels, st.height, st.width, s);
      case K::kAttention: {
        const std::int64_t c = st.in_channels;
        if (scores.bytes() < static_cast<std::uint64_t>(pixels * pixels * 4)) {
          if (auto a = scores.Allocate(static_cast<std::uint64_t>(pixels * pixels * 4), "scores");
              !a) {
            return a;
          }
          if (auto a = probs.Allocate(static_cast<std::uint64_t>(pixels * pixels * 2), "probs");
              !a) {
            return a;
          }
        }
        Status r =
            ki::ScoresQtK(blas_, in, in + (c * pixels), scores.as<float>(), c, pixels, pixels);
        r = r ? ki::SoftmaxRowsToBf16(scores.as<float>(), probs.as<Bf16>(), pixels, pixels,
                                      1.0f / std::sqrt(static_cast<float>(c)), s)
              : r;
        return r ? ki::ValuesTimesProbs(blas_, in + (2 * c * pixels), probs.as<Bf16>(), out, c,
                                        pixels, pixels)
                 : r;
      }
      case K::kAddDupUp:
        return ki::AddDupUp(out, aux, st.in_channels, st.out_channels, st.factor_t, st.height,
                            st.width, s);
    }
    return Error("an unknown VAE step");
  }

  const md::QwenImageVaeProfile& p_;
  cublasContext* blas_;
  Device& d_;
};

// ------------------------------------------------------------------ main

struct Options {
  std::filesystem::path store;
  std::string composition;
  std::filesystem::path out;
  std::string prompt = "A red ceramic teapot on a plain wooden table, soft daylight, no text.";
  std::uint32_t size = 1024;
  std::uint32_t steps = 40;
  std::optional<std::filesystem::path> reference;
  bool reference_embeds = false;
  std::optional<std::filesystem::path> noise;
  bool force_latents = false;
  bool vae = true;
  bool released = false;
  std::uint32_t runs = 0;
  std::uint32_t stop_after = 0;  // 0: every step
  bool profile = false;
  bool ggml_attention = false;
  bool decode_reference = false;
};

std::expected<Options, std::string> Parse(std::span<char*> args) {
  Options o;
  for (std::size_t i = 1; i < args.size(); ++i) {
    const std::string_view a = args[i];
    const auto value = [&]() -> std::expected<std::string_view, std::string> {
      if (i + 1 >= args.size()) {
        return Error(std::format("{} needs a value", a));
      }
      return std::string_view(args[++i]);
    };
    const auto number = [&]() -> std::expected<std::uint32_t, std::string> {
      auto v = value();
      if (!v) {
        return std::unexpected(v.error());
      }
      std::uint32_t n = 0;
      for (const char c : *v) {
        if (c < '0' || c > '9' || n > 100000) {
          return Error(std::format("{}: not a number", a));
        }
        n = (n * 10) + static_cast<std::uint32_t>(c - '0');
      }
      return n;
    };
    if (a == "--store" || a == "--composition" || a == "--out" || a == "--prompt" ||
        a == "--reference" || a == "--embeds" || a == "--noise" || a == "--phases" ||
        a == "--attention") {
      auto v = value();
      if (!v) {
        return std::unexpected(v.error());
      }
      if (a == "--store") o.store = *v;
      if (a == "--composition") o.composition = *v;
      if (a == "--out") o.out = *v;
      if (a == "--prompt") o.prompt = *v;
      if (a == "--reference") o.reference = std::filesystem::path(*v);
      if (a == "--noise") o.noise = std::filesystem::path(*v);
      if (a == "--embeds") {
        if (*v != "native" && *v != "reference") return Error("--embeds native|reference");
        o.reference_embeds = *v == "reference";
      }
      if (a == "--phases") {
        if (*v != "resident" && *v != "released") return Error("--phases resident|released");
        o.released = *v == "released";
      }
      if (a == "--attention") {
        if (*v != "native" && *v != "ggml") return Error("--attention native|ggml");
        o.ggml_attention = *v == "ggml";
      }
    } else if (a == "--size" || a == "--steps" || a == "--runs" || a == "--stop-after") {
      auto n = number();
      if (!n) {
        return std::unexpected(n.error());
      }
      if (a == "--size") o.size = *n;
      if (a == "--steps") o.steps = *n;
      if (a == "--runs") o.runs = *n;
      if (a == "--stop-after") o.stop_after = *n;
    } else if (a == "--force-latents") {
      o.force_latents = true;
    } else if (a == "--no-vae") {
      o.vae = false;
    } else if (a == "--profile") {
      o.profile = true;
    } else if (a == "--decode-reference") {
      o.decode_reference = true;
    } else {
      return Error(std::format("unknown argument {}", a));
    }
  }
  if (o.store.empty() || o.composition.empty() || o.out.empty()) {
    return Error("--store, --composition and --out are required");
  }
  if (o.size % 32 != 0 || o.size < 64 || o.size > 2048 || o.steps < 2 || o.steps > 100) {
    return Error("--size a multiple of 32 in 64-2048, --steps 2-100");
  }
  if ((o.reference_embeds || o.force_latents || o.decode_reference) && !o.reference) {
    return Error("--embeds reference, --force-latents and --decode-reference need --reference");
  }
  return o;
}

struct Component {
  std::optional<ja::Artifact> artifact;
  std::vector<std::uint32_t> bound;
  Loaded weights;
};

Status Run(const Options& o) {
  std::filesystem::create_directories(o.out);
  const md::QwenImageProfile& profile = md::QwenImage21();
  std::ostringstream report;
  report << "{\n";
  MemorySampler memory;
  const std::uint64_t baseline = MemAvailable();

  // The composition and its components, as untrusted input.
  auto composition = ja::OpenComposition(o.store / o.composition);
  if (!composition) {
    return Error("composition: " + composition.error().ToString());
  }
  if (composition->architecture() != profile.pipeline_architecture) {
    return Error("the composition is not a " + std::string(profile.pipeline_architecture));
  }
  std::map<std::string, Component> components;
  const std::array<std::tuple<std::string_view, std::string_view, std::vector<md::QwenImageTensor>>,
                   3>
      roles = {
          {{"text_encoder", profile.text_architecture, md::TextEncoderTensors(profile.text)},
           {"transformer", profile.denoiser_architecture, md::DenoiserTensors(profile.denoiser)},
           {"vae", profile.vae_architecture, md::VaeDecoderTensors(profile.vae)}}};
  for (const auto& [role, arch, tensors] : roles) {
    const ja::CompositionComponent* c = composition->Find(role);
    if (c == nullptr) {
      return Error(std::format("the composition has no {}", role));
    }
    ja::OpenOptions options;
    options.expected_id = c->artifact;
    auto a = ja::Artifact::Open(o.store / c->artifact, options);
    if (!a) {
      return Error(std::format("{}: {}", role, a.error().ToString()));
    }
    if (a->model().architecture != c->architecture) {
      return Error(
          std::format("{}: the artifact's architecture differs from the composition's", role));
    }
    auto bound = md::BindQwenImageComponent(tensors, arch, *a);
    if (!bound) {
      return Error(std::format("{}: {}", role, bound.error()));
    }
    Component& comp = components[std::string(role)];
    comp.artifact.emplace(std::move(*a));
    comp.bound = std::move(*bound);
  }

  // The prompt's tokens (the native tokenizer and renderer).
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
  auto rendered = jitllm::chat::RenderQwenImagePrompt(o.prompt, *tokenizer);
  if (!rendered) {
    return Error("prompt: " + rendered.error().ToString());
  }
  std::vector<jitllm::tokenizer::TokenId> ids;
  if (auto e =
          tokenizer->EncodeMarked(rendered->rendered.text, rendered->rendered.specials, {}, ids);
      !e) {
    return Error("prompt: " + e.error().ToString());
  }
  const std::size_t drop = rendered->drop_tokens;
  if (ids.size() <= drop || ids.size() > 1024) {
    return Error("a prompt of 1 to 1,024 tokens after the system turn");
  }
  const auto text_rows = static_cast<std::int64_t>(ids.size() - drop);
  report << std::format("  \"tokens\": {}, \"drop_tokens\": {}, \"text_rows\": {},\n", ids.size(),
                        drop, text_rows);
  if (o.reference) {
    std::ifstream in(*o.reference / "input_ids.i64", std::ios::binary);
    std::vector<std::int64_t> ref(ids.size() + 1);
    in.read(reinterpret_cast<char*>(ref.data()), static_cast<std::streamsize>(ref.size() * 8));
    const bool same = static_cast<std::size_t>(in.gcount()) == ids.size() * 8 &&
                      std::equal(ids.begin(), ids.end(), ref.begin());
    report << std::format("  \"tokens_match_reference\": {},\n", same);
    if (!same) {
      return Error("the native prompt tokens differ from the reference's");
    }
  }

  auto device = Device::Open();
  if (!device) {
    return std::unexpected(device.error());
  }
  Device& d = **device;
  // cuBLAS (its workspace as upstream sizes it) and the K-C launch context
  // for GGML's flash attention (a pool for its stream-k fixup).
  DeviceBuffer blas_ws;
  DeviceBuffer launch_ws;
  cudaDeviceProp prop{};
  (void)cudaGetDeviceProperties(&prop, 0);
  const Bytes blas_bytes =
      kg::CublasHandle::UpstreamWorkspace((prop.major * 100) + (prop.minor * 10));
  if (auto r = blas_ws.Allocate(blas_bytes.value(), "cuBLAS workspace"); !r) {
    return r;
  }
  if (auto r = launch_ws.Allocate(std::uint64_t{64} << 20U, "launch workspace"); !r) {
    return r;
  }
  auto blas = kg::CublasHandle::Create(0, d.execution(), d.stream(),
                                       {.base = blas_ws.address(), .size = blas_bytes});
  if (!blas) {
    return Error("cuBLAS: " + blas.error().detail);
  }
  auto launch = kg::LaunchContext::Create(0, d.execution(), d.stream(),
                                          {.base = launch_ws.address(), .size = Bytes(64 << 20)},
                                          blas->get());
  if (!launch) {
    return Error("launch context: " + launch.error().detail);
  }
  cublasContext* handle = (*blas)->native();

  const std::uint32_t grid = o.size / profile.vae.spatial_factor;
  const std::int64_t image = std::int64_t{grid} * grid;
  const std::int64_t latent_elems = image * profile.denoiser.in_channels;
  auto schedule =
      md::QwenImageSchedulerSigmas(profile.scheduler, o.steps, static_cast<std::uint32_t>(image));
  if (!schedule) {
    return Error(schedule.error());
  }
  report << std::format(R"(  "mu": {:.17g}, "sigmas": [)", schedule->mu);
  for (std::size_t i = 0; i < schedule->sigmas.size(); ++i) {
    report << std::format("{}{:.9g}", i ? ", " : "", schedule->sigmas[i]);
  }
  report << "],\n";

  // Reference tensors.
  std::vector<Bf16> ref_embeds;
  std::vector<Bf16> noise0;
  if (o.reference) {
    auto e = ReadBf16(*o.reference / "prompt_embeds.bf16",
                      static_cast<std::size_t>(text_rows * profile.denoiser.context));
    if (!e) {
      return std::unexpected(e.error());
    }
    ref_embeds = std::move(*e);
  }
  {
    std::filesystem::path path;
    if (o.noise) {
      path = *o.noise;
    } else if (o.reference) {
      path = *o.reference / "latents_init.bf16";
    }
    if (path.empty()) {
      return Error(
          "the initial latents: --noise or --reference (jitLLM does not reproduce "
          "PyTorch's CUDA generator)");
    }
    auto n = ReadBf16(path, static_cast<std::size_t>(latent_elems));
    if (!n) {
      return std::unexpected(n.error());
    }
    noise0 = std::move(*n);
  }

  auto load = [&](std::string_view role) -> Status {
    Component& c = components[std::string(role)];
    if (c.weights.region.address() != 0) {
      return {};
    }
    return role == "vae" ? LoadVae(*c.artifact, c.bound, d, c.weights)
                         : LoadBf16(*c.artifact, c.bound, d, c.weights);
  };
  // Completion-aware: the phase's queued work finishes before its weights
  // are freed (a free is not proof that the kernels reading them are done).
  auto release = [&](std::string_view role) -> Status {
    if (auto r = d.Finish(); !r) {
      return r;
    }
    Component& c = components[std::string(role)];
    c.weights.region.Free();
    c.weights.tensor_address.clear();
    c.weights.group_address.clear();
    return {};
  };
  std::vector<std::string> load_lines;
  if (!o.released) {
    for (const char* role : {"text_encoder", "transformer", "vae"}) {
      if (auto r = load(role); !r) {
        return r;
      }
      const Loaded& w = components[role].weights;
      load_lines.push_back(
          std::format(R"("{}": {{"bytes_read": {}, "seconds": {:.3f}, "device_bytes": {}}})", role,
                      w.bytes_read, w.seconds, w.region.bytes()));
    }
  }

  TextEncoder text(profile.text, handle, d);
  Denoiser dit(profile.denoiser, handle, **launch, d);
  dit.native_attention_ = !o.ggml_attention;
  VaeDecoder vae(profile.vae, handle, d);
  DeviceBuffer latents;
  DeviceBuffer noise_pred;
  DeviceBuffer embeds;
  const FinishFirst finish(d);  // before these and the phases' buffers are freed
  if (auto r = latents.Allocate(static_cast<std::uint64_t>(latent_elems * 2), "latents"); !r) {
    return r;
  }
  if (auto r = noise_pred.Allocate(static_cast<std::uint64_t>(latent_elems * 2), "noise"); !r) {
    return r;
  }
  auto arena = kg::TensorArena::Create(16);
  if (!arena) {
    return Error("arena: " + arena.error().detail);
  }

  // One full generation; `record` keeps every tensor and the per-step
  // times (synchronized each step).
  struct Timing {
    double encode = 0, denoise = 0, decode = 0, total = 0, first_step = 0;
    std::string pixels_sha256;
    std::vector<double> steps;
    std::vector<std::string> loads;
  };
  std::vector<Compared> step_cmp;
  std::optional<Compared> embeds_cmp;
  std::optional<Compared> final_cmp;
  std::optional<Compared> decoded_cmp;
  std::vector<Bf16> final_latents;
  std::vector<Bf16> decoded_host;
  Profile prof;
  prof.on = o.profile;
  bool attention_built = false;

  auto generate = [&](bool record, Timing& t) -> Status {
    const auto start = Clock::now();
    // --- encode
    if (o.released) {
      const auto l0 = Clock::now();
      if (auto r = load("text_encoder"); !r) return r;
      t.loads.push_back(std::format(R"("text_encoder": {:.3f})", Seconds(Clock::now() - l0)));
    }
    if (o.reference_embeds) {
      if (auto r = embeds.Allocate(ref_embeds.size() * 2, "embeddings"); !r) return r;
      if (auto r = Cuda(cudaMemcpy(embeds.as<void>(), ref_embeds.data(), ref_embeds.size() * 2,
                                   cudaMemcpyHostToDevice),
                        "embeddings");
          !r)
        return r;
    } else {
      if (auto r = text.Run(components["text_encoder"].weights, ids, drop, embeds); !r) return r;
    }
    if (auto r = d.Finish(); !r) return r;
    if (o.released) {
      if (auto r = release("text_encoder"); !r) return r;
    }
    const auto encoded = Clock::now();
    t.encode = Seconds(encoded - start);
    if (record) {
      auto got = Download<Bf16>(embeds.address(),
                                static_cast<std::size_t>(text_rows * profile.denoiser.context));
      if (!got) return std::unexpected(got.error());
      if (auto w = WriteBytes(o.out / "prompt_embeds.bf16", got->data(), got->size() * 2); !w)
        return w;
      if (!ref_embeds.empty()) embeds_cmp = CompareBf16(*got, ref_embeds);
    }
    // --- denoise
    if (o.released) {
      const auto l0 = Clock::now();
      if (auto r = load("transformer"); !r) return r;
      t.loads.push_back(std::format(R"("transformer": {:.3f})", Seconds(Clock::now() - l0)));
    }
    const auto denoise_start = Clock::now();
    if (auto r = dit.Prepare(components["transformer"].weights, embeds, text_rows, grid); !r)
      return r;
    if (!attention_built) {
      if (auto r = dit.BuildAttention(*arena); !r) return r;
      attention_built = true;
    }
    if (auto r = Cuda(cudaMemcpyAsync(latents.as<void>(), noise0.data(), noise0.size() * 2,
                                      cudaMemcpyHostToDevice, d.native()),
                      "latents");
        !r)
      return r;
    auto step_start = Clock::now();
    const std::uint32_t last = o.stop_after != 0 ? std::min(o.stop_after, o.steps) : o.steps;
    for (std::uint32_t i = 0; i < last; ++i) {
      if (o.force_latents) {
        auto in = ReadBf16(*o.reference / std::format("step{:02d}_latents_in.bf16", i),
                           static_cast<std::size_t>(latent_elems));
        if (!in) return std::unexpected(in.error());
        if (auto r = Cuda(
                cudaMemcpy(latents.as<void>(), in->data(), in->size() * 2, cudaMemcpyHostToDevice),
                "forced latents");
            !r)
          return r;
      }
      if (auto r =
              dit.Step(i, schedule->timesteps[i], latents.as<Bf16>(), noise_pred.as<Bf16>(), prof);
          !r)
        return r;
      // dt is a 0-dim F32 tensor times the BF16 noise: PyTorch rounds it to
      // BF16 first (checked against the reference's steps: 0 of 10.2M
      // values differ this way, 287,316 with an F32 dt).
      const float dt = schedule->sigmas[i + 1] - schedule->sigmas[i];
      if (auto r = ki::EulerStep(latents.as<Bf16>(), noise_pred.as<Bf16>(), latents.as<Bf16>(), dt,
                                 true, latent_elems, d.native());
          !r)
        return r;
      if (record) {
        if (auto r = d.Finish(); !r) return r;
        const auto now = Clock::now();
        t.steps.push_back(Seconds(now - step_start));
        step_start = now;
        auto got = Download<Bf16>(noise_pred.address(), static_cast<std::size_t>(latent_elems));
        if (!got) return std::unexpected(got.error());
        if (auto w = WriteBytes(o.out / std::format("step{:02d}_noise_pred.bf16", i), got->data(),
                                got->size() * 2);
            !w)
          return w;
        if (o.reference) {
          auto ref = ReadBf16(*o.reference / std::format("step{:02d}_noise_pred.bf16", i),
                              static_cast<std::size_t>(latent_elems));
          if (ref) step_cmp.push_back(CompareBf16(*got, *ref));
        }
      }
    }
    if (auto r = d.Finish(); !r) return r;
    const auto denoised = Clock::now();
    t.denoise = Seconds(denoised - denoise_start);
    t.first_step = t.steps.empty() ? 0 : t.steps[0];
    if (o.released) {
      if (auto r = release("transformer"); !r) return r;
    }
    if (record) {
      auto got = Download<Bf16>(latents.address(), static_cast<std::size_t>(latent_elems));
      if (!got) return std::unexpected(got.error());
      final_latents = *got;
      if (auto w = WriteBytes(o.out / "latents_final.bf16", got->data(), got->size() * 2); !w)
        return w;
      if (o.reference) {
        auto ref =
            ReadBf16(*o.reference / "latents_final.bf16", static_cast<std::size_t>(latent_elems));
        if (ref) final_cmp = CompareBf16(*got, *ref);
      }
    }
    // --- decode
    if (o.vae) {
      if (o.released) {
        const auto l0 = Clock::now();
        if (auto r = load("vae"); !r) return r;
        t.loads.push_back(std::format(R"("vae": {:.3f})", Seconds(Clock::now() - l0)));
      }
      if (o.decode_reference) {
        // The VAE alone: diffusers' final latents.
        auto ref =
            ReadBf16(*o.reference / "latents_final.bf16", static_cast<std::size_t>(latent_elems));
        if (!ref) return std::unexpected(ref.error());
        if (auto r = Cuda(cudaMemcpy(latents.as<void>(), ref->data(), ref->size() * 2,
                                     cudaMemcpyHostToDevice),
                          "reference latents");
            !r)
          return r;
      }
      const auto decode_start = Clock::now();
      DeviceBuffer decoded;
      if (auto r = vae.Run(components["vae"].weights, latents.as<Bf16>(), grid, decoded); !r)
        return r;
      if (auto r = d.Finish(); !r) return r;
      t.decode = Seconds(Clock::now() - decode_start);
      if (o.released) {
        if (auto r = release("vae"); !r) return r;
      }
      const std::size_t count = std::size_t{profile.vae.out_channels} * o.size * o.size;
      auto got = Download<Bf16>(decoded.address(), count);
      if (!got) return std::unexpected(got.error());
      if (record) {
        decoded_host = *got;
        if (auto w = WriteBytes(o.out / "vae_decoded.bf16", got->data(), got->size() * 2); !w)
          return w;
        if (o.reference) {
          auto ref = ReadBf16(*o.reference / "vae_decoded.bf16", count);
          if (ref) decoded_cmp = CompareBf16(*got, *ref);
        }
      }
      const auto pixels = md::QwenImagePixels(*got, profile.vae.out_channels, o.size, o.size);
      t.pixels_sha256 = jitllm::base::ToHex(
          jitllm::base::Sha256().Update(std::as_bytes(std::span(pixels))).Finish());
      if (record) {
        if (auto w = WriteBytes(o.out / "image.rgba8", pixels.data(), pixels.size()); !w) return w;
      }
    }
    t.total = Seconds(Clock::now() - start);
    return {};
  };

  memory.Reset();
  Timing first;
  if (auto r = generate(true, first); !r) {
    return r;
  }
  const std::uint64_t low_first = memory.low();
  std::vector<double> plain;
  bool same_pixels = true;
  for (std::uint32_t i = 0; i < o.runs; ++i) {
    Timing t;
    if (auto r = generate(false, t); !r) {
      return r;
    }
    plain.push_back(t.total);
    same_pixels = same_pixels && t.pixels_sha256 == first.pixels_sha256;
  }
  report << std::format("  \"pixels_sha256\": \"{}\", \"plain_runs_same_pixels\": {},\n",
                        first.pixels_sha256, same_pixels);
  // The report.
  report << std::format("  \"composition\": \"{}\",\n", composition->id());
  report << "  \"loads\": {";
  for (std::size_t i = 0; i < load_lines.size(); ++i) {
    report << (i ? ", " : "") << load_lines[i];
  }
  report << "},\n  \"released_loads_s\": {";
  for (std::size_t i = 0; i < first.loads.size(); ++i) {
    report << (i ? ", " : "") << first.loads[i];
  }
  report << "},\n";
  std::vector<double> gaps(first.steps.begin() + (first.steps.size() > 1 ? 1 : 0),
                           first.steps.end());
  std::ranges::sort(gaps);
  report << std::format(
      "  \"first_run\": {{\"encode_s\": {:.4f}, \"denoise_s\": {:.4f}, \"decode_s\": {:.4f}, "
      "\"total_s\": {:.4f}, \"first_step_s\": {:.4f}, \"step_s_median\": {:.4f}, "
      "\"step_s_min\": {:.4f}, \"step_s_max\": {:.4f}}},\n",
      first.encode, first.denoise, first.decode, first.total, first.first_step,
      gaps.empty() ? 0.0 : gaps[gaps.size() / 2], gaps.empty() ? 0.0 : gaps.front(),
      gaps.empty() ? 0.0 : gaps.back());
  report << "  \"plain_runs_s\": [";
  for (std::size_t i = 0; i < plain.size(); ++i) {
    report << std::format("{}{:.4f}", i ? ", " : "", plain[i]);
  }
  report << "],\n";
  report << std::format(
      "  \"peak_memavailable_drop_bytes\": {},\n  \"first_run_memavailable_drop_bytes\": {},\n",
      baseline - memory.low(), baseline - low_first);
  report << std::format("  \"flash_attention\": {{\"columns\": {}, \"scratch\": {}}},\n",
                        dit.fa_columns(), dit.fa_scratch());
  if (embeds_cmp) {
    report << "  \"prompt_embeds\": " << Json(*embeds_cmp) << ",\n";
  }
  report << "  \"noise_pred\": [";
  for (std::size_t i = 0; i < step_cmp.size(); ++i) {
    report << (i ? ",\n    " : "\n    ") << Json(step_cmp[i]);
  }
  report << "],\n";
  if (final_cmp) {
    report << "  \"latents_final\": " << Json(*final_cmp) << ",\n";
  }
  if (decoded_cmp) {
    report << "  \"vae_decoded\": " << Json(*decoded_cmp) << ",\n";
  }
  if (prof.on) {
    if (auto r = d.Finish(); !r) return r;
    for (std::size_t i = 0; i + 1 < prof.marks.size(); ++i) {
      float ms = 0;
      (void)cudaEventElapsedTime(&ms, prof.marks[i].second, prof.marks[i + 1].second);
      prof.ms[prof.marks[i].first] += ms;
    }
    report << "  \"profile_ms_all_steps\": {";
    bool comma = false;
    for (const auto& [k, v] : prof.ms) {
      report << std::format("{}\"{}\": {:.1f}", comma ? ", " : "", k, v);
      comma = true;
    }
    report << "},\n";
  }
  report << std::format(
      "  \"options\": {{\"size\": {}, \"steps\": {}, \"embeds\": \"{}\", "
      "\"force_latents\": {}, \"phases\": \"{}\", \"decode_reference\": {}, "
      "\"attention\": \"{}\"}}\n}}\n",
      o.size, o.steps, o.reference_embeds ? "reference" : "native", o.force_latents,
      o.released ? "released" : "resident", o.decode_reference,
      o.ggml_attention ? "ggml" : "native");
  const std::string text_report = report.str();
  if (auto r = WriteBytes(o.out / "report.json", text_report.data(), text_report.size()); !r) {
    return r;
  }
  std::print("{}", text_report);
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  auto options = Parse(std::span(argv, static_cast<std::size_t>(argc)));
  if (!options) {
    std::println(stderr, "{}", options.error());
    return 2;
  }
  if (auto r = Run(*options); !r) {
    std::println(stderr, "error: {}", r.error());
    return 1;
  }
  return 0;
}
