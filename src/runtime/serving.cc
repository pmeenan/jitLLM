// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/serving.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <format>
#include <print>
#include <string>
#include <utility>

#include "artifact/artifact.h"
#include "artifact/composition.h"
#include "base/report.h"
#include "base/sha256.h"
#include "engine/dsv4_plan.h"
#include "engine/dsv4_runner.h"
#include "engine/qwen38_runner.h"
#include "engine/qwen_image_runner.h"
#include "model/dsv4.h"
#include "platform/crash_policy.h"
#include "platform/files.h"
#include "platform/path_trust.h"
#include "scheduler/programs.h"
#include "tokenizer/gguf.h"
#include "tokenizer/hf.h"

namespace jitllm::runtime {
namespace {

namespace fs = std::filesystem;
namespace ja = jitllm::artifact;

constexpr std::uint64_t kExtent = engine::kPagedExtent;
// The most a tokenizer or template file may be.
constexpr std::size_t kMaxTokenizerBytes = std::size_t{64} << 20U;
// Room kept beyond the budget for what the catalog does not count: decode
// graphs, the driver's and cuBLAS's own allocations.
constexpr std::uint64_t kUncountedMargin = std::uint64_t{4} << 30U;
// Extents the page-in observer tracks (the M3 models use about 100,000).
constexpr std::size_t kObservedExtents = std::size_t{1} << 18U;

std::unexpected<std::string> Error(std::string what) { return std::unexpected(std::move(what)); }

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

// A file the configuration names, under its trust rules (D-073): every
// directory on the way and the file itself root's or the runtime user's
// and writable by nobody else; opened without following links, and the
// file the walk approved.
std::expected<std::string, std::string> ReadTrustedFile(const fs::path& path, uid_t trusted,
                                                        std::size_t limit) {
  auto walked = platform::WalkTrusted(path, trusted, false);
  if (!walked) {
    return Error(std::format("{}: {}", path.string(), walked.error()));
  }
  if (!walked->exists) {
    return Error(std::format("{} does not exist", path.string()));
  }
  const int fd =
      ::open(walked->resolved.c_str(), O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC | O_NOCTTY);
  if (fd < 0) {
    return Error(std::format("{}: {}", path.string(), Errno(errno)));
  }
  struct stat status{};
  std::string bytes;
  std::string problem;
  if (::fstat(fd, &status) != 0) {
    problem = Errno(errno);
  } else if (!S_ISREG(status.st_mode) || status.st_dev != walked->status.st_dev ||
             status.st_ino != walked->status.st_ino) {
    problem = "not the regular file the trust walk approved";
  } else if (platform::OthersCanWrite(status, trusted, fd)) {
    problem = "other users can write it";
  } else if (std::cmp_greater(status.st_size, limit)) {
    problem = std::format("larger than {} bytes", limit);
  } else {
    bytes.resize(static_cast<std::size_t>(status.st_size));
    std::size_t done = 0;
    while (done < bytes.size()) {
      const ssize_t n =
          ::pread(fd, bytes.data() + done, bytes.size() - done, static_cast<off_t>(done));
      if (n < 0 && errno == EINTR) {
        continue;
      }
      if (n <= 0) {
        problem = n < 0 ? Errno(errno) : "shorter than it was";
        break;
      }
      done += static_cast<std::size_t>(n);
    }
  }
  (void)::close(fd);
  if (!problem.empty()) {
    return Error(std::format("{}: {}", path.string(), problem));
  }
  return bytes;
}

// An installed artifact, opened under the store's trust rules (D-056's
// integrity rests on them) and its ID checked; the runner opens it again.
std::expected<ja::Artifact, std::string> OpenTrusted(const fs::path& store, const std::string& id) {
  ja::OpenOptions options;
  options.expected_id = id;
  options.trusted_owner = ::geteuid();
  auto opened = ja::Artifact::Open(store / id, options);
  if (!opened) {
    return Error(std::format("artifact {}: {}", id, opened.error().ToString()));
  }
  return std::move(*opened);
}

std::string Hex(std::string_view bytes) {
  return base::ToHex(base::Sha256().Update(bytes).Finish());
}

std::string GraphJson(const GraphCounts& g) {
  return std::format(R"({{"eager":{},"captured":{},"replayed":{},"refused":{},"kept":{}}})",
                     g.eager, g.captured, g.replayed, g.refused, g.kept);
}

GraphCounts Sum(const engine::Dsv4GraphStats& a, const engine::Dsv4GraphStats& b,
                std::size_t kept) {
  return {.eager = a.eager + b.eager,
          .captured = a.captured + b.captured,
          .replayed = a.replayed + b.replayed,
          .refused = a.refused + b.refused,
          .kept = kept};
}

// ---------------------------------------------------------------- the models

// DeepSeek V4 Flash (engine/dsv4_runner.h), with DSpark as its drafter.
class Dsv4 final : public Llm {
 public:
  Dsv4(engine::PagedNode& node, const config::ModelEntry& entry, const config::RuntimeRoles& roles,
       bool plain, int index)
      : artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    node_ = &node;
    speculate_ = !drafter_id_.empty() && entry.speculation && !plain;
    context_ = entry.context;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = entry.context;
    options_.max_rows = max_rows_;
    options_.graphs = true;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  Status Setup() override {
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "deepseek4") {
      return Error(std::format("artifact {} is a {}, not DeepSeek V4 (deepseek4)", artifact_id_,
                               artifact->model().architecture));
    }
    if (speculate_) {
      if (auto drafter = OpenTrusted(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template from the artifact's kept GGUF metadata
    // (import rule 7): the first shard's.
    std::string kv;
    for (const ja::ListedFile& f : artifact->files()) {
      if (f.role == ja::FileRole::kSourceMetadata && f.path.ends_with(".kv.gguf") &&
          (kv.empty() || f.path.contains("-00001-of-"))) {
        kv = f.path.substr(5);
      }
    }
    if (kv.empty()) {
      return Error("the artifact keeps no GGUF metadata, so no tokenizer");
    }
    auto bytes = artifact->ReadMetadata(kv);
    if (!bytes) {
      return Error(std::format("{}: {}", kv, bytes.error().ToString()));
    }
    auto read = tokenizer::ReadGgufTokenizer(std::as_bytes(std::span(*bytes)));
    if (!read) {
      return Error(std::format("{}: {}", kv, read.error().ToString()));
    }
    auto created = tokenizer::Tokenizer::Create(std::move(read->spec));
    if (!created) {
      return Error(std::format("{}: {}", kv, created.error().ToString()));
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
    if (read->has_chat_template) {
      template_ = chat::FindTemplate(Hex(read->chat_template));
    }
    if (template_ != nullptr) {
      auto stops = chat::StopTokens(template_->stop, *tokenizer_);
      if (!stops) {
        return Error(stops.error().ToString());
      }
      stops_.assign(stops->begin(), stops->end());
    }
    if (tokenizer_->eos() && std::ranges::find(stops_, *tokenizer_->eos()) == stops_.end()) {
      stops_.push_back(*tokenizer_->eos());
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  Status AfterLoad() override { return runner_.CheckHashRouting(); }
  Status CheckPlaces() override { return runner_.CheckPlaces(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    return std::format(
        R"({{"architecture":"deepseek4","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"graphs":{}}})",
        speculate_ ? "\"dspark\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& c) const override {
    // As llama-server's /apply-template rendered the references' prompts:
    // its default turns the template's thinking on.
    if (!c.enable_thinking) {
      c.enable_thinking = true;
    }
  }

 protected:
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return runner_.Chunk(n_past, all.subspan(n_past), logits, {},
                         inject ? engine::Dsv4ChunkKind::kInject : engine::Dsv4ChunkKind::kPlain);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    // The anchor and its drafts, within the tokens left, the verify's
    // bound, the context and the steps' mask widths (D-092).
    auto rows = std::min<std::uint32_t>(
        {options_.draft_rows + 1, options_.max_verify, left, options_.context - pos});
    while (rows > 1 && !model::Dsv4SameWidths(runner_.state_layout(), pos, rows)) {
      --rows;
    }
    std::vector<std::int32_t> drafts;
    std::vector<float> verified;
    if (auto r = runner_.DraftVerify(pos, all.back(), rows, drafts, verified); !r) {
      return r;
    }
    drafts.resize(rows - 1);
    const std::uint32_t vocab = runner_.vocab();
    const auto row = [&](std::uint32_t i) {
      return std::span<const float>(verified).subspan(std::size_t{i} * vocab, vocab);
    };
    // Accept drafts while the target agrees; the first disagreement, or the
    // row after the last draft, gives the next token.
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (; m < rows - 1; ++m) {
      const std::int32_t want = engine::Argmax(row(m));
      if (want != drafts[m]) {
        next = want;
        break;
      }
    }
    if (next < 0) {
      next = engine::Argmax(row(m));
    }
    if (auto r = runner_.Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      for (std::uint32_t i = 0; i <= m; ++i) {
        logits->emplace_back(row(i).begin(), row(i).end());
      }
    }
    drafted += rows - 1;
    return {};
  }
  Status Settle() override { return runner_.Rollback(); }
  Status ClearState() override { return runner_.Clear(); }
  std::uint64_t target_state_base() const override { return runner_.state_base(); }
  std::uint64_t target_state_bytes() const override { return runner_.state_bytes(); }
  std::uint64_t drafter_state_base() const override { return runner_.drafter_state_base(); }
  std::uint64_t drafter_state_bytes() const override { return runner_.drafter_state_bytes(); }

 private:
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  fs::path store_;
  engine::Dsv4Options options_;  // before the runner, which keeps a reference
  engine::Dsv4Runner runner_;
};

// Qwen3.8 Flash Next (engine/qwen38_runner.h), with its MTP block as its
// drafter.
class Qwen38 final : public Llm {
 public:
  Qwen38(engine::PagedNode& node, const config::ModelEntry& entry,
         const config::RuntimeRoles& roles, bool plain, int index)
      : artifact_id_(entry.artifact.value_or("")),
        drafter_id_(entry.drafter.value_or("")),
        tokenizer_path_(entry.tokenizer),
        template_path_(entry.chat_template),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    node_ = &node;
    speculate_ = !drafter_id_.empty() && entry.speculation && !plain;
    context_ = entry.context;
    options_.artifact = roles.installed / artifact_id_;
    options_.out = roles.spill;
    options_.context = entry.context;
    options_.max_rows = max_rows_;
    options_.graphs = true;
    if (speculate_) {
      options_.drafter = roles.installed / drafter_id_;
    }
  }

  engine::PagedModel& paged() override { return runner_; }

  Status Setup() override {
    auto artifact = OpenTrusted(store_, artifact_id_);
    if (!artifact) {
      return std::unexpected(artifact.error());
    }
    if (artifact->model().architecture != "qwen4exp") {
      return Error(std::format("artifact {} is a {}, not Qwen3.8 Flash Next (qwen4exp)",
                               artifact_id_, artifact->model().architecture));
    }
    if (speculate_) {
      if (auto drafter = OpenTrusted(store_, drafter_id_); !drafter) {
        return std::unexpected(drafter.error());
      }
    }
    // The tokenizer and template: kept in the artifact's metadata, or (M3's
    // import kept config.json only) the checkpoint's, which the
    // configuration names.
    const auto source = [&](std::string_view kept, const std::optional<fs::path>& named,
                            std::string_view key) -> std::expected<std::string, std::string> {
      auto meta = artifact->ReadMetadata(kept);
      if (meta) {
        return std::move(*meta);
      }
      if (meta.error().rule != ja::Rule::kFileSet) {  // kept, but not as listed
        return Error(std::format("{}: {}", kept, meta.error().ToString()));
      }
      if (!named.has_value()) {
        return Error(
            std::format("the artifact keeps no {} and models.{}.{} is not set", kept, name_, key));
      }
      return ReadTrustedFile(named.value_or(fs::path()), ::geteuid(), kMaxTokenizerBytes);
    };
    auto json = source("tokenizer.json", tokenizer_path_, "tokenizer");
    if (!json) {
      return std::unexpected(json.error());
    }
    auto spec = tokenizer::ReadHfTokenizer(*json);
    if (!spec) {
      return Error("tokenizer: " + spec.error().ToString());
    }
    auto created = tokenizer::Tokenizer::Create(std::move(*spec));
    if (!created) {
      return Error("tokenizer: " + created.error().ToString());
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(std::move(*created));
    auto text = source("chat_template.jinja", template_path_, "chat_template");
    if (!text) {
      return std::unexpected(text.error());
    }
    template_ = chat::FindTemplate(Hex(*text));
    if (template_ == nullptr) {
      return Error("no native renderer has this chat template's hash (D-067)");
    }
    auto stops = chat::StopTokens(template_->stop, *tokenizer_);
    if (!stops) {
      return Error(stops.error().ToString());
    }
    stops_.assign(stops->begin(), stops->end());
    if (tokenizer_->eos() && std::ranges::find(stops_, *tokenizer_->eos()) == stops_.end()) {
      stops_.push_back(*tokenizer_->eos());
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return runner_.pool_needed(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  std::vector<catalog::ExtentId> state() const override { return runner_.state(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  Status AfterLoad() override { return runner_.ReadPleHash(); }
  void DropPlans() override { runner_.DropPlans(); }
  double plan_seconds() const override { return runner_.plan_seconds(); }
  double demand_seconds() const override { return runner_.ple().seconds; }
  std::uint64_t demand_bytes() const override { return runner_.ple().read_bytes; }
  GraphCounts graphs() const override {
    return Sum(runner_.graph_stats(), runner_.draft_stats(), runner_.graphs());
  }
  std::string violations() const override {
    return runner_.coverage_violations() == 0
               ? std::string()
               : std::format("{} bound tensors outside cataloged extents of their class; first {}",
                             runner_.coverage_violations(), runner_.first_violation());
  }
  std::string extra() const override {
    const engine::PleStats& p = runner_.ple();
    return std::format(
        R"({{"architecture":"qwen4exp","speculation":{},"coverage_tensors":{},)"
        R"("slab_padding":{},"state_bytes":{},"drafter_state_bytes":{},"ple_table_bytes":{},)"
        R"("ple":{{"chunks":{},"lookups":{},"rows":{},"reads":{},"read_bytes":{},)"
        R"("seconds":{:.6f}}},"graphs":{}}})",
        speculate_ ? "\"mtp\"" : "null", runner_.coverage_tensors(), runner_.slab_padding(),
        runner_.state_bytes(), runner_.drafter_state_bytes(), runner_.table_bytes(), p.chunks,
        p.lookups, p.rows, p.reads, p.read_bytes, p.seconds, GraphJson(graphs()));
  }
  void Defaults(chat::Conversation& /*c*/) const override {
    // The template's own defaults (thinking on), as the oracle's /tokenize
    // rendered the references' prompts.
  }

 protected:
  Status RunChunk(std::span<const std::int32_t> all, std::uint32_t n_past, bool inject,
                  std::vector<float>& logits) override {
    return runner_.Chunk(all, n_past, logits, inject);
  }
  Status SpecStep(std::span<const std::int32_t> all, std::uint32_t pos, std::uint32_t left,
                  std::vector<std::int32_t>& kept, std::vector<std::vector<float>>* logits,
                  std::uint64_t& drafted) override {
    if (pos + runner_.draft_rows() > options_.context) {
      return Error("the drafts would pass the context");
    }
    std::vector<std::int32_t> drafts;
    if (auto r = runner_.Draft(all, drafts); !r) {
      return r;
    }
    // The verify: the anchor and its drafts, within the tokens left.
    const auto rows = std::min<std::uint32_t>(
        {static_cast<std::uint32_t>(drafts.size()) + 1, left, options_.context - pos});
    drafts.resize(rows - 1);
    std::vector<std::int32_t> input(all.begin(), all.end());
    input.insert(input.end(), drafts.begin(), drafts.end());
    std::vector<std::int32_t> argmax;
    std::vector<float> verified;
    if (auto r = runner_.Verify(input, pos, argmax, logits != nullptr ? &verified : nullptr); !r) {
      return r;
    }
    std::uint32_t m = 0;
    std::int32_t next = -1;
    for (; m < rows - 1; ++m) {
      if (argmax[m] != drafts[m]) {
        next = argmax[m];
        break;
      }
    }
    if (next < 0) {
      next = argmax[m];
    }
    if (auto r = runner_.Accept(m + 1); !r) {
      return r;
    }
    kept.assign(drafts.begin(), drafts.begin() + m);
    kept.push_back(next);
    if (logits != nullptr) {
      const std::uint32_t vocab = runner_.vocab();
      for (std::uint32_t i = 0; i <= m; ++i) {
        const auto row = std::span<const float>(verified).subspan(std::size_t{i} * vocab, vocab);
        logits->emplace_back(row.begin(), row.end());
      }
    }
    drafted += rows - 1;
    return {};
  }
  Status Settle() override { return runner_.Rollback(); }
  Status ClearState() override { return runner_.Clear(); }
  std::uint64_t target_state_base() const override { return runner_.state_base(); }
  std::uint64_t target_state_bytes() const override { return runner_.state_bytes(); }
  std::uint64_t drafter_state_base() const override { return runner_.drafter_state_base(); }
  std::uint64_t drafter_state_bytes() const override { return runner_.drafter_state_bytes(); }
  std::uint32_t cursor() const override { return runner_.pending_rows(); }
  void set_cursor(std::uint32_t value) override { runner_.set_pending_rows(value); }

 private:
  std::string artifact_id_;
  std::string drafter_id_;  // empty: none
  std::optional<fs::path> tokenizer_path_;
  std::optional<fs::path> template_path_;
  fs::path store_;
  engine::Qwen38Options options_;  // before the runner, which keeps a reference
  engine::Qwen38Runner runner_;
};

// The Qwen-Image-2.1 pipeline (engine/qwen_image_runner.h).
class QwenImage final : public Image {
 public:
  QwenImage(engine::PagedNode& node, const config::ModelEntry& entry,
            const config::RuntimeRoles& roles, const ServingOptions& serving, int index)
      : composition_id_(entry.composition.value_or("")),
        store_(roles.installed),
        runner_(node, options_, index, static_cast<std::uint32_t>(index)) {
    name_ = entry.name;
    options_.store = roles.installed;
    options_.composition = composition_id_;
    options_.noise = serving.image_noise;
    options_.out = roles.spill;
    options_.prompt = serving.image_prompt;
  }

  engine::PagedModel& paged() override { return runner_; }
  Status Setup() override {
    // Every component under the store's trust rules first.
    const fs::path root = store_ / composition_id_;
    auto walked = platform::WalkTrusted(root, ::geteuid(), false);
    if (!walked || !walked->exists || !S_ISDIR(walked->status.st_mode) ||
        platform::OthersCanWrite(walked->status, ::geteuid(), root)) {
      return Error(
          std::format("composition {}: not a directory only root and this user can "
                      "change{}",
                      composition_id_, walked ? "" : ": " + walked.error()));
    }
    auto composition = ja::OpenComposition(root, composition_id_);
    if (!composition) {
      return Error(
          std::format("composition {}: {}", composition_id_, composition.error().ToString()));
    }
    for (const ja::CompositionComponent& c : composition->components()) {
      if (auto opened = OpenTrusted(store_, c.artifact); !opened) {
        return std::unexpected(opened.error());
      }
    }
    return runner_.Setup();
  }
  std::uint64_t activations_needed() const override { return runner_.activations_needed(); }
  std::uint64_t pool_needed() const override { return engine::QwenImageRunner::pool_needed(); }
  Status Register() override { return runner_.Register(); }
  Status Bind() override { return runner_.Bind(); }
  std::vector<catalog::ExtentId> weights() const override { return runner_.weights(); }
  const catalog::Closure& everything() const override { return runner_.everything(); }
  std::uint64_t weight_read_bytes() const override { return runner_.weight_read_bytes(); }
  std::string extra() const override { return runner_.Report(); }
  Status FirstOutput(std::string& sha) override { return runner_.FirstOutput(sha); }
  Status Finish(std::string& sha) override { return runner_.Finish(sha); }

 private:
  std::string composition_id_;
  fs::path store_;
  engine::QwenImageOptions options_;  // before the runner, which keeps a reference
  engine::QwenImageRunner runner_;
};

}  // namespace

// ---------------------------------------------------------------- memory

MemorySampler::MemorySampler() : start_(Available()) {
  low_ = start_;
  all_ = start_;
  thread_ = std::jthread([this] {
    if (auto stack = platform::InstallThreadSignalStack(); !stack) {
      std::println(stderr, "the memory sampler runs without a signal stack: {}", stack.error());
    }
    Loop();
  });
}

MemorySampler::~MemorySampler() {
  stop_ = true;
  if (thread_.joinable()) {
    thread_.join();
  }
}

std::uint64_t MemorySampler::Available() {
  auto text = platform::ReadSmallFile("/proc/meminfo");
  if (!text) {
    return 0;
  }
  const std::string_view key = "MemAvailable:";
  const std::size_t at = text->find(key);
  if (at == std::string::npos) {
    return 0;
  }
  std::string_view rest = std::string_view(*text).substr(at + key.size());
  while (!rest.empty() && rest.front() == ' ') {
    rest.remove_prefix(1);
  }
  std::uint64_t kib = 0;
  (void)std::from_chars(rest.data(), rest.data() + rest.size(), kib);
  return kib * 1024;
}

void MemorySampler::Reset() { low_ = Available(); }

void MemorySampler::Loop() {
  const auto lower = [](std::atomic<std::uint64_t>& to, std::uint64_t now) {
    std::uint64_t seen = to.load();
    while (now < seen && !to.compare_exchange_weak(seen, now)) {
    }
  };
  while (!stop_) {
    const std::uint64_t now = Available();
    if (now != 0) {
      lower(low_, now);
      lower(all_, now);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void ResidentTimes::Staged(catalog::ExtentId extent, scheduler::PageInEvent event) {
  if (event == scheduler::PageInEvent::kResident && extent.index() < at_.size()) {
    at_[extent.index()] = Clock::now();
  }
}

Clock::time_point ResidentTimes::Latest(std::span<const catalog::ExtentId> extents) const {
  Clock::time_point latest{};
  for (const catalog::ExtentId extent : extents) {
    if (extent.index() < at_.size()) {
      latest = std::max(latest, at_[extent.index()]);
    }
  }
  return latest;
}

// ---------------------------------------------------------------- an LLM

std::expected<std::vector<std::int32_t>, std::string> Llm::EncodeText(std::string_view text) const {
  std::vector<tokenizer::TokenId> ids;
  if (auto r = tokenizer_->Encode(text, {.add_bos_eos = true}, ids); !r) {
    return Error(r.error().ToString());
  }
  if (tokenizer_->bos() && (ids.empty() || ids.front() != *tokenizer_->bos())) {
    ids.insert(ids.begin(), *tokenizer_->bos());  // BOS first, as the references fed it
  }
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::expected<std::vector<std::int32_t>, std::string> Llm::RenderChat(
    const chat::Conversation& conversation) const {
  if (template_ == nullptr) {
    return Error(std::format("{} has no chat template a native renderer supports (D-067)", name_));
  }
  auto rendered = template_->render(conversation);
  if (!rendered) {
    return Error(rendered.error().ToString());
  }
  std::vector<tokenizer::TokenId> ids;
  if (auto r = tokenizer_->EncodeMarked(rendered->text, rendered->specials, {}, ids); !r) {
    return Error(r.error().ToString());
  }
  return std::vector<std::int32_t>(ids.begin(), ids.end());
}

std::string Llm::Detokenize(std::span<const std::int32_t> tokens) const {
  std::string text;
  if (auto r = tokenizer_->Decode(tokens, {}, text); !r) {
    return "(not decodable)";
  }
  return text;
}

Status Llm::Clear() {
  if (auto r = ClearState(); !r) {
    return r;
  }
  history_.clear();
  needs_clear_ = false;
  return {};
}

Status Llm::Prefill(std::span<const std::int32_t> tokens, std::vector<float>& last) {
  if (needs_clear_) {
    if (auto r = Clear(); !r) {
      return r;
    }
  }
  if (tokens.empty()) {
    return Error("nothing to prefill");
  }
  std::vector<std::int32_t> all = history_;
  all.insert(all.end(), tokens.begin(), tokens.end());
  if (all.size() >= context_) {
    return Error(
        std::format("{} tokens do not fit {}'s context of {}", all.size(), name_, context_));
  }
  for (auto at = static_cast<std::uint32_t>(history_.size()); at < all.size(); at += max_rows_) {
    const auto n = static_cast<std::uint32_t>(std::min<std::size_t>(max_rows_, all.size() - at));
    if (auto r = RunChunk(std::span(all).first(at + n), at, speculate_, last); !r) {
      needs_clear_ = true;
      history_.clear();
      return Error(std::format("{}'s prefill at {}: {}", name_, at, r.error()));
    }
  }
  history_ = std::move(all);
  return {};
}

Status Llm::Generate(const std::vector<float>& last, const GenerateOptions& options,
                     Generation& out) {
  const auto is_stop = [&](std::int32_t token) {
    return options.stop && std::ranges::find(stops_, token) != stops_.end();
  };
  out.tokens = {engine::Argmax(last)};
  if (options.keep_logits) {
    out.logits = {last};
  }
  out.stopped = is_stop(out.tokens.back());
  // Every token so far, the anchor (the last generated, not yet in the
  // state) last; pos: how many the state holds.
  std::vector<std::int32_t> all = history_;
  all.push_back(out.tokens.back());
  auto pos = static_cast<std::uint32_t>(history_.size());
  const auto start = Clock::now();
  Status ran;
  while (!out.stopped && out.tokens.size() < options.max_tokens) {
    if (pos + 1 >= context_) {
      ran = Error(std::format("{}'s conversation reached its context of {}", name_, context_));
      break;
    }
    const auto left = static_cast<std::uint32_t>(options.max_tokens - out.tokens.size());
    std::vector<std::int32_t> kept;
    if (speculate_) {
      ran =
          SpecStep(all, pos, left, kept, options.keep_logits ? &out.logits : nullptr, out.drafted);
      if (ran) {
        out.accepted += kept.size() - 1;
      }
    } else {
      std::vector<float> row;
      ran = RunChunk(all, pos, false, row);
      if (ran) {
        kept = {engine::Argmax(row)};
        if (options.keep_logits) {
          out.logits.push_back(std::move(row));
        }
      }
    }
    if (!ran) {
      break;
    }
    if (++out.steps == 1) {
      out.first_step = Clock::now();
    }
    // The anchor and the accepted drafts are in the state now; the last
    // kept token is the next anchor. The generation ends at a stop token,
    // though the state holds what was accepted after it.
    pos += static_cast<std::uint32_t>(kept.size());
    all.insert(all.end(), kept.begin(), kept.end());
    for (const std::int32_t token : kept) {
      out.tokens.push_back(token);
      if (is_stop(token)) {
        out.stopped = true;
        break;
      }
    }
  }
  out.decode_seconds = Seconds(Clock::now() - start);
  if (!ran) {
    needs_clear_ = true;
    history_.clear();
    return ran;
  }
  history_.assign(all.begin(), all.begin() + pos);
  if (out.tokens.size() > options.max_tokens) {
    out.tokens.resize(options.max_tokens);
  }
  if (out.logits.size() > out.tokens.size()) {
    out.logits.resize(out.tokens.size());
  }
  if (auto settled = Settle(); !settled) {
    needs_clear_ = true;
    history_.clear();
    return settled;
  }
  return {};
}

std::uint64_t Llm::state_snapshot_bytes() const {
  return target_state_bytes() + drafter_state_bytes();
}

Status Llm::SaveState(void* host) {
  if (auto r = Settle(); !r) {
    return r;
  }
  engine::PagedModel& p = paged();
  if (auto r = node_->Copy(p.stream(), p.fence_closure(), target_state_base(), host,
                           target_state_bytes(), true, "saving a conversation state");
      !r) {
    return r;
  }
  if (drafter_state_bytes() != 0) {
    if (auto r = node_->Copy(p.stream(), p.fence_closure(), drafter_state_base(),
                             static_cast<std::byte*>(host) + target_state_bytes(),
                             drafter_state_bytes(), true, "saving a drafter's state");
        !r) {
      return r;
    }
  }
  saved_history_ = history_;
  saved_cursor_ = cursor();
  return {};
}

Status Llm::RestoreState(void* host) {
  if (auto r = Settle(); !r) {
    return r;
  }
  engine::PagedModel& p = paged();
  if (auto r = node_->Copy(p.stream(), p.fence_closure(), target_state_base(), host,
                           target_state_bytes(), false, "restoring a conversation state");
      !r) {
    return r;
  }
  if (drafter_state_bytes() != 0) {
    if (auto r = node_->Copy(p.stream(), p.fence_closure(), drafter_state_base(),
                             static_cast<std::byte*>(host) + target_state_bytes(),
                             drafter_state_bytes(), false, "restoring a drafter's state");
        !r) {
      return r;
    }
  }
  history_ = saved_history_;
  set_cursor(saved_cursor_);
  needs_clear_ = false;
  return {};
}

// ---------------------------------------------------------------- the server

Server::Server(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               const ServingOptions& options, std::FILE* log)
    : config_(config),
      roles_(roles),
      options_(options),
      log_(log),
      times_(kObservedExtents),
      node_({.compute_streams = std::max<std::size_t>(config.models.size(), 1),
             .slots = engine::kPagedSlots,
             .inline_lanes = false,
             .coalesce = false,
             .copy_lane = true,
             .slot_bytes = engine::kSlabSlotBytes,
             .observer = &times_}) {}

Server::~Server() {
  if (started_ && !torn_down_) {
    if (auto stopped = TearDown(); !stopped) {
      Log("stopping: " + stopped.error());
    }
  }
}

void Server::Log(std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log_);
  (void)std::fflush(log_);
}

Status Server::Make(const config::ModelEntry& entry, int index) {
  if (entry.composition) {
    if (options_.image_noise.empty()) {
      Log(
          std::format("model {}: not registered: an image pipeline needs --image-noise (its "
                      "initial latents) in this build",
                      entry.name));
      return {};
    }
    models_.push_back(std::make_unique<QwenImage>(node_, entry, roles_, options_, index));
    return {};
  }
  auto artifact = OpenTrusted(roles_.installed, entry.artifact.value_or(""));
  if (!artifact) {
    return Error(std::format("model {}: {}", entry.name, artifact.error()));
  }
  const std::string& architecture = artifact->model().architecture;
  if (architecture == "deepseek4") {
    models_.push_back(std::make_unique<Dsv4>(node_, entry, roles_, options_.plain, index));
  } else if (architecture == "qwen4exp") {
    models_.push_back(std::make_unique<Qwen38>(node_, entry, roles_, options_.plain, index));
  } else {
    return Error(std::format("model {}: no runner for architecture {}", entry.name, architecture));
  }
  return {};
}

Status Server::Start(bool snapshot) {
  if (started_) {
    return Error("the server started already");
  }
  started_ = true;
  if (config_.models.empty()) {
    return Error("the configuration names no models ([models.<name>], D-096)");
  }
  if (auto r = node_.Open(); !r) {
    return r;
  }
  // Each model's stream and owner: its place among the registered.
  for (const config::ModelEntry& entry : config_.models) {
    if (auto r = Make(entry, static_cast<int>(models_.size())); !r) {
      return r;
    }
  }
  if (models_.empty()) {
    return Error("no configured model could be registered");
  }
  std::uint64_t activations = 0;
  std::uint64_t pool = 0;
  std::uint64_t largest = 0;
  std::uint64_t state = 0;
  for (const auto& m : models_) {
    const auto started = Clock::now();
    if (auto r = m->Setup(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
    activations = std::max(activations, m->activations_needed());
    pool = std::max(pool, m->pool_needed());
    largest = std::max<std::uint64_t>(largest, m->weights().size() * kExtent);
    if (m->llm()) {
      state = std::max(state, static_cast<Llm&>(*m).state_snapshot_bytes());
    }
    Log(std::format("model {}: set up in {:.2f} s, {} weight extents ({:.2f} GB read a load)",
                    m->name(), Seconds(Clock::now() - started), m->weights().size(),
                    static_cast<double>(m->weight_read_bytes()) / 1e9));
  }
  if (auto r = node_.MapWorkspace(activations, pool); !r) {
    return r;
  }
  if (snapshot && state != 0) {
    std::vector<catalog::ExtentId> staging;
    auto pinned = node_.Pinned(state, engine::kShared, staging);
    if (!pinned) {
      return std::unexpected(pinned.error());
    }
    snapshot_ = *pinned;
  }
  // The budget (D-050's B): everything mapped for the node's life (the zone,
  // each model's own memory, the workspace, the staging) and the largest
  // model's weights, so a full swap is the only way in.
  fixed_ = node_.catalog().OccupancyOf(node_.domain()).Total().value();
  budget_ = fixed_ + largest;
  const std::uint64_t available = MemorySampler::Available();
  if (available != 0 && largest + kUncountedMargin > available) {
    return Error(std::format(
        "the largest model's weights ({:.1f} GiB) and a {:.0f} GiB margin do not fit the {:.1f} "
        "GiB available beside this node's fixed memory ({:.1f} GiB)",
        static_cast<double>(largest) / (1ULL << 30U),
        static_cast<double>(kUncountedMargin) / (1ULL << 30U),
        static_cast<double>(available) / (1ULL << 30U),
        static_cast<double>(fixed_) / (1ULL << 30U)));
  }
  if (auto r = node_.Start(base::Bytes(budget_)); !r) {
    return r;
  }
  for (const auto& m : models_) {
    if (auto r = m->Register(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  for (const auto& m : models_) {
    if (auto r = m->Bind(); !r) {
      return Error(std::format("model {}: {}", m->name(), r.error()));
    }
  }
  node_.Run();
  Log(std::format("serving {} models; budget {:.2f} GiB ({:.2f} GiB fixed); {:.2f} GiB available",
                  models_.size(), static_cast<double>(budget_) / (1ULL << 30U),
                  static_cast<double>(fixed_) / (1ULL << 30U),
                  static_cast<double>(available) / (1ULL << 30U)));
  return {};
}

Status Server::TearDown() {
  if (torn_down_) {
    return {};
  }
  torn_down_ = true;
  std::vector<engine::PagedModel*> models;
  models.reserve(models_.size());
  for (const auto& m : models_) {
    models.push_back(&m->paged());
  }
  return node_.TearDown(models);
}

Served* Server::Find(std::string_view name) {
  for (const auto& m : models_) {
    if (m->name() == name) {
      return m.get();
    }
  }
  return nullptr;
}

Status Server::InRequest(Served& m, const std::function<Status()>& body) {
  if (auto r = node_.BeginRequest(m.paged().stream(), m.everything(),
                                  std::format("{}'s request", m.name()));
      !r) {
    return r;
  }
  Status ran = body();
  Status ended = node_.EndRequest(m.paged().stream());
  return !ran ? ran : ended;
}

Status Server::WaitReleased() {
  const auto give_up = Clock::now() + std::chrono::minutes(2);
  for (;;) {
    std::size_t left = 0;
    if (auto r = node_.Call(
            [&]() -> Status {
              left = node_.scheduler().evictions();
              return {};
            },
            "counting evictions");
        !r) {
      return r;
    }
    if (left == 0) {
      return {};
    }
    if (Clock::now() > give_up) {
      return Error("backing no load took was not released");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
}

Status Server::FinishSwap(SwapParts& parts) {
  if (auto r = WaitReleased(); !r) {
    return r;
  }
  parts.release = Seconds(Clock::now() - parts.ready);
  auto after = node_.Stats();
  if (!after) {
    return std::unexpected(after.error());
  }
  parts.handed_off = after->handed_off - swap_before_.handed_off;
  parts.released_unused = after->released_unused - swap_before_.released_unused;
  return {};
}

Status Server::Activate(Served& m, SwapParts& parts, std::optional<bool> spill_state) {
  parts = SwapParts{};
  parts.to = m.name();
  if (resident_ == &m) {
    return {};
  }
  auto before = node_.Stats();
  if (!before) {
    return std::unexpected(before.error());
  }
  swap_before_ = *before;
  const bool restoring = std::ranges::find(spilled_, &m) != spilled_.end();
  parts.requested = Clock::now();
  Clock::time_point evicted = parts.requested;
  Clock::time_point loaded;
  if (resident_ == nullptr) {
    // Nothing resident: the first load of its weights (its state is
    // resident since setup).
    std::vector<engine::LoadStats> log;
    if (auto r = node_.Load(m.weights(), std::format("{}'s first load", m.name()), log); !r) {
      return r;
    }
    loaded = Clock::now();
    parts.loaded = log.empty() ? 0 : log.front().extents;
  } else {
    Served& out = *resident_;
    parts.from = out.name();
    const bool conversation = out.llm() && !static_cast<Llm&>(out).history().empty();
    parts.with_state = out.llm() && spill_state.value_or(conversation);
    std::vector<catalog::ExtentId> extents;
    if (parts.with_state) {
      extents = out.state();  // first: its write-backs start first
      parts.spilled_bytes = extents.size() * kExtent;
    }
    const std::vector<catalog::ExtentId> weights = out.weights();
    extents.insert(extents.end(), weights.begin(), weights.end());
    scheduler::SwapReport report;
    if (auto r = node_.Swap(std::move(extents), m.everything(), handoff_, report); !r) {
      return r;
    }
    if (parts.with_state) {
      spilled_.push_back(&out);
    }
    evicted = report.evicted;
    loaded = report.loaded;
    parts.evicted = report.evictions;
    parts.loaded = report.loads;
  }
  parts.evict = Seconds(evicted - parts.requested);
  Clock::time_point restored = evicted;
  if (restoring) {
    restored = std::max(restored, times_.Latest(m.state()));
    parts.restore = Seconds(restored - evicted);
    std::erase(spilled_, &m);
  }
  parts.page_in = Seconds(loaded - restored);
  parts.read_bytes = m.weight_read_bytes() + (restoring ? m.state().size() * kExtent : 0);
  resident_ = &m;
  if (auto r = m.AfterLoad(); !r) {
    return Error(std::format("{} after its load: {}", m.name(), r.error()));
  }
  parts.ready = Clock::now();
  parts.setup = Seconds(parts.ready - loaded);
  parts.total = Seconds(parts.ready - parts.requested);
  // Its places still pinned where it registered them (D-090).
  std::size_t unpinned = 0;
  const std::vector<catalog::ExtentId> managed = m.paged().managed_extents();
  if (auto r = node_.Call(
          [&]() -> Status {
            for (const catalog::ExtentId extent : managed) {
              unpinned += node_.scheduler().PlacePinned(extent) ? 0 : 1;
            }
            return {};
          },
          "checking places");
      !r) {
    return r;
  }
  if (unpinned != 0) {
    return Error(std::format("{} of {}'s {} extents are not pinned at their places", unpinned,
                             m.name(), managed.size()));
  }
  if (auto r = m.CheckPlaces(); !r) {
    return Error(std::format("{}'s places: {}", m.name(), r.error()));
  }
  return {};
}

}  // namespace jitllm::runtime
