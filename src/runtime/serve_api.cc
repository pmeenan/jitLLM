// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The service with models configured (D-097; docs/runtime-serving.md#the-chat-route):
// the configured models registered on the node (serving.h), the loopback
// chat route (api_server.h) over them, and the runtime's signals watched
// through a signalfd on the node's driver thread, which runs every request.

#include <sys/random.h>
#include <sys/signalfd.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <format>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/report.h"
#include "chat/chat.h"
#include "platform/job.h"
#include "platform/sd_notify.h"
#include "runtime/api.h"
#include "runtime/api_server.h"
#include "runtime/commands.h"
#include "runtime/runtime.h"
#include "runtime/serving.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::runtime {
namespace {

void Say(std::FILE* log, std::string_view text) {
  const std::string line = std::format("jitllm-runtime: {}\n", base::Printable(text));
  (void)std::fwrite(line.data(), 1, line.size(), log);
  (void)std::fflush(log);
}

api::Error Failure(int status, std::string message, std::string code = {}, std::string param = {}) {
  return api::Error{.status = status,
                    .type = status >= 500 ? "server_error" : "invalid_request_error",
                    .message = std::move(message),
                    .param = std::move(param),
                    .code = std::move(code)};
}

std::uint64_t RandomSeed() {
  std::uint64_t seed = 0;
  if (::getrandom(&seed, sizeof seed, 0) != static_cast<ssize_t>(sizeof seed)) {
    seed = static_cast<std::uint64_t>(Clock::now().time_since_epoch().count());
  }
  return seed;
}

// The chat route's requests as turns on the node: one model resident, a
// swap when another is asked for, the conversation state reused when the
// rendered request extends it, and each request one lease (D-093).
class NodeBackend final : public api::Backend {
 public:
  NodeBackend(Server& server, const config::NodeConfig& config, std::FILE* log)
      : server_(server), config_(config), log_(log) {}

  std::vector<api::ModelInfo> Models() const override {
    std::vector<api::ModelInfo> models;
    for (const config::ModelEntry& entry : config_.models) {
      if (entry.composition) {
        models.push_back({.name = entry.name, .chat = false, .context = 0});
        continue;
      }
      Served* m = server_.Find(entry.name);
      if (m != nullptr && m->llm()) {
        models.push_back(
            {.name = entry.name, .chat = true, .context = static_cast<Llm&>(*m).usable_context()});
      }
    }
    return models;
  }

  std::expected<api::Completion, api::Error> Complete(const api::ChatRequest& request,
                                                      api::Exchange& exchange) override {
    Served* m = server_.Find(request.model);
    if (m == nullptr || !m->llm()) {
      return std::unexpected(Failure(404, "The model does not exist", "model_not_found", "model"));
    }
    auto& l = static_cast<Llm&>(*m);
    chat::Conversation conversation;
    for (const api::Message& message : request.messages) {
      chat::Role role = chat::Role::kUser;
      if (message.role == api::Role::kSystem) {
        role = chat::Role::kSystem;
      } else if (message.role == api::Role::kAssistant) {
        role = chat::Role::kAssistant;
      }
      conversation.messages.push_back({.role = role,
                                       .content = message.content,
                                       .reasoning_content = message.reasoning,
                                       .tool_calls = {}});
    }
    l.Defaults(conversation);
    auto rendered = l.RenderChat(conversation);
    if (!rendered) {
      return std::unexpected(
          Failure(400, "the conversation cannot be rendered: " + rendered.error(), {}, "messages"));
    }
    const std::vector<std::int32_t>& tokens = *rendered;
    const std::uint32_t usable = l.usable_context();
    const auto prompt = static_cast<std::uint32_t>(tokens.size());
    if (prompt >= usable) {
      return std::unexpected(
          Failure(400,
                  std::format("This model's maximum context length is {} tokens. However, your "
                              "messages resulted in {} tokens. Please reduce the length of the "
                              "messages.",
                              usable, prompt),
                  "context_length_exceeded", "messages"));
    }
    const std::uint32_t max_tokens = request.max_tokens.value_or(usable - prompt);
    if (std::uint64_t{prompt} + max_tokens > usable) {
      return std::unexpected(Failure(
          400,
          std::format("This model's maximum context length is {} tokens. However, you requested "
                      "{} tokens ({} in the messages, {} in the completion). Please reduce the "
                      "length of the messages or completion.",
                      usable, std::uint64_t{prompt} + max_tokens, prompt, max_tokens),
          "context_length_exceeded", "messages"));
    }
    // Whether the rendered prompt leaves the model inside a reasoning
    // block: its last reasoning marker opens one.
    bool reasoning = false;
    if (l.think_start() && l.think_end()) {
      const auto marker = std::ranges::find_if(tokens.rbegin(), tokens.rend(), [&](std::int32_t t) {
        return t == *l.think_start() || t == *l.think_end();
      });
      reasoning = marker != tokens.rend() && *marker == *l.think_start();
    }
    if (!exchange.Admit({.prompt_tokens = prompt})) {
      return api::Completion{};
    }

    swapped_ = server_.resident() != m;
    if (auto r = server_.Activate(*m, parts_); !r) {
      Fail(std::format("making {} resident: {}", m->name(), r.error()));
      swapped_ = false;
      return std::unexpected(Failure(503, "the model could not be made resident"));
    }

    GenerateOptions options{.max_tokens = max_tokens,
                            .stop = true,
                            .keep_logits = false,
                            .sampling = std::nullopt,
                            .seed = 0,
                            .on_tokens = {}};
    if (request.temperature > 0) {
      options.sampling =
          execution::SamplingParams{.temperature = static_cast<float>(request.temperature),
                                    .top_k = 0,
                                    .top_p = static_cast<float>(request.top_p),
                                    .min_p = 0.0F};
      options.seed = request.seed.value_or(RandomSeed());
    }
    tokenizer::StreamDecoder decoder(l.tokenizer(), {});
    bool thinking = reasoning;
    bool any_content = false;
    options.on_tokens = [&](std::span<const std::int32_t> fresh) {
      std::string piece;
      bool go = true;
      bool said = false;
      const auto send = [&]() {
        if (!piece.empty()) {
          go = (thinking ? exchange.Reasoning(piece) : exchange.Content(piece)) && go;
          any_content = any_content || !thinking;
          said = true;
          piece.clear();
        }
      };
      for (const std::int32_t token : fresh) {
        if (thinking && l.think_end() && token == *l.think_end()) {
          decoder.Finish(piece);
          if (piece.empty()) {
            go = exchange.Reasoning({}) && go;  // an empty block: the answer is still trimmed
          }
          send();
          thinking = false;
          continue;
        }
        if (!thinking && !any_content && piece.empty() && l.think_start() &&
            token == *l.think_start()) {
          thinking = true;  // the model opened reasoning itself
          continue;
        }
        (void)decoder.Push(token, piece);  // a token it cannot decode adds nothing
      }
      send();
      return said ? go : exchange.Continue();
    };

    Generation generation;
    std::uint32_t reused = 0;
    auto ran = server_.InRequest(*m, [&]() -> Status {
      const std::vector<std::int32_t>& history = l.history();
      // The state holds a prefix of this request's tokens: only the rest
      // is prefilled; otherwise the conversation starts over (as chat).
      if (history.empty() || history.size() >= tokens.size() ||
          !std::equal(history.begin(), history.end(), tokens.begin())) {
        if (auto r = l.Clear(); !r) {
          return r;
        }
      } else {
        reused = static_cast<std::uint32_t>(history.size());
      }
      std::vector<float> last;
      if (auto r = l.Prefill(std::span(tokens).subspan(reused), last); !r) {
        return r;
      }
      return l.Generate(last, options, generation);
    });
    if (!ran) {
      Fail(std::format("{}'s request: {}", m->name(), ran.error()));
      return std::unexpected(Failure(500, "the generation failed; the runtime is stopping"));
    }
    std::string rest;
    decoder.Finish(rest);
    if (!rest.empty()) {
      (void)(thinking ? exchange.Reasoning(rest) : exchange.Content(rest));
    }
    return api::Completion{
        .completion_tokens = static_cast<std::uint32_t>(generation.tokens.size()),
        .cached_tokens = reused,
        .stopped = generation.stopped};
  }

  void AfterResponse() override {
    if (!swapped_) {
      return;
    }
    swapped_ = false;
    if (auto r = server_.FinishSwap(parts_); !r) {
      Fail("finishing the swap: " + r.error());
      return;
    }
    Say(log_,
        std::format("swap {} -> {}: ready in {:.3f} s (evict {:.3f}, restore {:.3f}, "
                    "page-in {:.3f}, setup {:.3f}); backing released {:.3f} s later",
                    parts_.from.empty() ? "(nothing)" : parts_.from, parts_.to, parts_.total,
                    parts_.evict, parts_.restore, parts_.page_in, parts_.setup, parts_.release));
  }

  bool healthy() const override { return failure_.empty(); }
  std::string failure() const override { return failure_; }

 private:
  void Fail(std::string what) {
    Say(log_, what);
    if (failure_.empty()) {
      failure_ = std::move(what);
    }
  }

  Server& server_;
  const config::NodeConfig& config_;
  std::FILE* log_;
  SwapParts parts_;
  bool swapped_ = false;
  std::string failure_;  // a node failure: the service stops
};

}  // namespace

int RunService(const config::NodeConfig& config, const config::RuntimeRoles& roles,
               std::FILE* log) {
  const ServingOptions serving;  // speculative where there is a drafter; no image prompt
  int status = kExitOk;
  {
    Server server(config, roles, serving, log);
    auto started = server.Start(false);
    std::optional<NodeBackend> backend;
    std::optional<api::Server> http;
    if (started) {
      backend.emplace(server, config, log);
      api::ServerOptions options;
      options.bind = config.client;
      options.log = log;
      http.emplace(*backend, std::move(options));
      if (auto port = http->Listen(); !port) {
        started =
            std::unexpected(std::format("the chat route cannot listen on {}:{}: {}",
                                        config.client.address, config.client.port, port.error()));
      } else {
        Say(log, std::format("the chat route listens on {}{}{}:{} (/v1/chat/completions, "
                             "/v1/models)",
                             config.client.ipv6 ? "[" : "", config.client.address,
                             config.client.ipv6 ? "]" : "", *port));
      }
    }
    if (!started) {
      Say(log, "refusing to serve: " + started.error());
      status = kExitFailure;
    } else {
      sigset_t signals;
      (void)::sigemptyset(&signals);
      for (const int s : {SIGTERM, SIGINT, SIGHUP, SIGCHLD}) {
        (void)::sigaddset(&signals, s);
      }
      const int wake = ::signalfd(-1, &signals, SFD_CLOEXEC | SFD_NONBLOCK);
      if (wake < 0) {
        Say(log, "cannot watch signals (signalfd)");
        status = kExitFailure;
      } else {
        if (auto notified = platform::NotifyServiceManager(
                std::format("READY=1\nSTATUS=serving {} models on the loopback chat route",
                            backend->Models().size()));
            !notified) {
          Say(log, notified.error());
        }
        Say(log, "ready");
        const auto on_wake = [&]() {
          bool stop = false;
          signalfd_siginfo info{};
          while (::read(wake, &info, sizeof info) == static_cast<ssize_t>(sizeof info)) {
            if (info.ssi_signo == SIGCHLD) {
              (void)platform::ReapExited();
            } else if (info.ssi_signo == SIGHUP) {
              Say(log,
                  "the configuration is read only at startup; restart the runtime to apply a "
                  "change");
            } else {
              stop = true;
            }
          }
          return stop;
        };
        auto ran = http->Run(wake, on_wake);
        (void)::close(wake);
        (void)platform::NotifyServiceManager("STOPPING=1");
        if (!ran) {
          Say(log, "stopping after a failure: " + ran.error());
          status = kExitFailure;
        } else {
          Say(log, "stopping");
        }
      }
    }
    http.reset();
    if (auto stopped = server.TearDown(); !stopped) {
      Say(log, "stopping: " + stopped.error());
      status = kExitFailure;
    }
  }
  return status;
}

}  // namespace jitllm::runtime
