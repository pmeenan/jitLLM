// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/api_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/random.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <expected>
#include <format>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jitllm::runtime::api {
namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kJson = "application/json";

Error Refusal(int status, std::string message, std::string code = {}) {
  std::string type = "invalid_request_error";
  if (status == 429) {
    type = "rate_limit_error";
  } else if (status == 403) {
    type = "permission_error";
  } else if (status >= 500) {
    type = "server_error";
  }
  return Error{.status = status,
               .type = std::move(type),
               .message = std::move(message),
               .param = {},
               .code = std::move(code)};
}

bool Readable(int fd) {
  pollfd p{.fd = fd, .events = POLLIN, .revents = 0};
  return ::poll(&p, 1, 0) > 0 && (p.revents & POLLIN) != 0;
}

void Signal(int eventfd) {
  const std::uint64_t one = 1;
  (void)!::write(eventfd, &one, sizeof one);
}

void Drain(int eventfd) {
  std::uint64_t count = 0;
  (void)!::read(eventfd, &count, sizeof count);
}

std::string RequestId() {
  std::array<std::uint8_t, 12> bytes{};
  if (::getrandom(bytes.data(), bytes.size(), 0) != static_cast<ssize_t>(bytes.size())) {
    // Unique within the process is enough for an opaque ID.
    static std::uint64_t counter = 0;
    const std::uint64_t n = ++counter;
    for (std::size_t i = 0; i < 8; ++i) {
      bytes[i] = static_cast<std::uint8_t>(n >> (8 * i));
    }
  }
  std::string id = "chatcmpl-";
  for (const std::uint8_t b : bytes) {
    id += std::format("{:02x}", b);
  }
  return id;
}

double Seconds(Clock::duration d) { return std::chrono::duration<double>(d).count(); }

}  // namespace

bool IsLoopbackHost(std::string_view host) {
  std::string_view name = host;
  if (name.starts_with('[')) {
    const std::size_t close = name.find(']');
    if (close == std::string_view::npos) {
      return false;
    }
    const std::string_view rest = name.substr(close + 1);
    if (!rest.empty() && !rest.starts_with(':')) {
      return false;
    }
    name = name.substr(1, close - 1);
    const std::string address(name);
    in6_addr six{};
    return ::inet_pton(AF_INET6, address.c_str(), &six) == 1 &&
           std::memcmp(&six, &in6addr_loopback, sizeof six) == 0;
  }
  name = name.substr(0, name.find(':'));
  // A host name is case-insensitive (RFC 9110, section 4.2.3).
  if (name.size() == 9 &&
      std::ranges::equal(name, std::string_view("localhost"), [](char a, char b) {
        return (a >= 'A' && a <= 'Z' ? static_cast<char>(a - 'A' + 'a') : a) == b;
      })) {
    return true;
  }
  const std::string address(name);
  in_addr four{};
  return ::inet_pton(AF_INET, address.c_str(), &four) == 1 && (ntohl(four.s_addr) >> 24U) == 127U;
}

// ---------------------------------------------------------------- Stream

// One admitted request's response, as it goes out.
class Server::Stream final : public Exchange {
 public:
  Stream(Server& server, Pending& pending, int wake_fd, const std::function<bool()>& on_wake)
      : server_(server),
        pending_(pending),
        wake_fd_(wake_fd),
        on_wake_(on_wake),
        created_(static_cast<std::int64_t>(std::time(nullptr))),
        deadline_(Clock::now() + server.options_.deadline),
        text_(pending.request.stop) {}

  bool Admit(const Admission& admission) override {
    usage_.prompt_tokens = admission.prompt_tokens;
    if (pending_.request.stream && !StartStream()) {
      return false;
    }
    return Check();
  }
  bool Reasoning(std::string_view text) override {
    Emit(text_.Reasoning(text));
    return Check();
  }
  bool Content(std::string_view text) override {
    Emit(text_.Content(text));
    return !text_.stopped() && Check();
  }
  bool Continue() override { return Check(); }

  // Writes the rest of the response for the backend's result; the status
  // for the log.
  int End(const std::expected<Completion, Error>& result) {
    const int fd = pending_.fd.get();
    if (!result) {
      if (streaming_) {
        (void)Event(ErrorJson(result.error()));
        return result.error().status;
      }
      server_.Respond(std::move(pending_.fd), result.error());
      return result.error().status;
    }
    Emit(text_.Finish());
    usage_.completion_tokens = result->completion_tokens;
    usage_.cached_tokens = result->cached_tokens;
    if (gone_) {
      return 499;  // the client closed the request (nginx's code; logged only)
    }
    if (timed_out_ || stopping_) {
      const Error error = timed_out_
                              ? Refusal(504, std::format("the request passed its {} s deadline",
                                                         server_.options_.deadline.count() / 1000))
                              : Refusal(503, "the runtime is stopping");
      if (streaming_) {
        (void)Event(ErrorJson(error));
      } else {
        server_.Respond(std::move(pending_.fd), error);
      }
      return error.status;
    }
    const Finish finish = result->stopped || text_.stopped() ? Finish::kStop : Finish::kLength;
    const std::string& model = pending_.request.model;
    if (pending_.request.stream) {
      if (!streaming_ && !StartStream()) {
        return 499;
      }
      (void)(Event(ChunkJson(pending_.id, created_, model, Delta::kFinish, {}, finish)) &&
             (!pending_.request.include_usage ||
              Event(UsageChunkJson(pending_.id, created_, model, usage_))) &&
             http::WriteAll(fd, "data: [DONE]\n\n"));
      return 200;
    }
    const std::string body = CompletionJson(
        pending_.id, created_, model, content_,
        reasoning_.empty() ? std::nullopt : std::optional<std::string>(reasoning_), finish, usage_);
    (void)(http::WriteAll(fd, http::Head(200, kJson, body.size())) && http::WriteAll(fd, body));
    return 200;
  }

  const Usage& usage() const { return usage_; }

 private:
  bool StartStream() {
    streaming_ = true;
    const int fd = pending_.fd.get();
    if (!http::WriteAll(fd, http::Head(200, "text/event-stream", std::nullopt)) ||
        !Event(ChunkJson(pending_.id, created_, pending_.request.model, Delta::kRole, {}))) {
      gone_ = true;
      return false;
    }
    return true;
  }

  bool Event(std::string_view json) {
    std::string event = "data: ";
    event += json;
    event += "\n\n";
    if (!http::WriteAll(pending_.fd.get(), event)) {
      gone_ = true;
      return false;
    }
    return true;
  }

  void Emit(const OutputText::Out& out) {
    if (!streaming_) {
      reasoning_ += out.reasoning;
      content_ += out.content;
      return;
    }
    if (!out.reasoning.empty() && !gone_) {
      (void)Event(ChunkJson(pending_.id, created_, pending_.request.model, Delta::kReasoning,
                            out.reasoning));
    }
    if (!out.content.empty() && !gone_) {
      (void)Event(
          ChunkJson(pending_.id, created_, pending_.request.model, Delta::kContent, out.content));
    }
  }

  // Whether the generation goes on.
  bool Check() {
    if (gone_ || timed_out_ || stopping_) {
      return false;
    }
    if (http::PeerGone(pending_.fd.get())) {
      gone_ = true;
      return false;
    }
    if (Clock::now() > deadline_) {
      timed_out_ = true;
      return false;
    }
    if (Readable(wake_fd_) && on_wake_()) {
      stopping_ = true;
      const std::scoped_lock lock(server_.mutex_);
      server_.stopping_ = true;
      return false;
    }
    return true;
  }

  Server& server_;
  Pending& pending_;
  int wake_fd_;
  const std::function<bool()>& on_wake_;
  std::int64_t created_;
  Clock::time_point deadline_;
  OutputText text_;
  Usage usage_;
  std::string reasoning_;
  std::string content_;
  bool streaming_ = false;
  bool gone_ = false;
  bool timed_out_ = false;
  bool stopping_ = false;
};

// ---------------------------------------------------------------- Server

Server::Server(Backend& backend, ServerOptions options)
    : backend_(backend),
      options_(std::move(options)),
      models_(backend.Models()),
      created_(static_cast<std::int64_t>(std::time(nullptr))),
      stop_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)),
      ready_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {}

Server::~Server() {
  if (acceptor_.joinable()) {
    Signal(stop_.get());
    acceptor_.join();
  }
}

std::expected<std::uint16_t, std::string> Server::Listen() {
  if (!stop_.valid() || !ready_.valid()) {
    return std::unexpected("eventfd failed");
  }
  auto listener = http::ListenLoopback(options_.bind);
  if (!listener) {
    return std::unexpected(listener.error());
  }
  listener_ = std::move(listener->fd);
  return listener->port;
}

void Server::Log(std::string_view line) {
  if (options_.log == nullptr) {
    return;
  }
  const std::string text = std::format("jitllm-runtime: {}\n", line);
  (void)std::fwrite(text.data(), 1, text.size(), options_.log);
  (void)std::fflush(options_.log);
}

void Server::Respond(http::Fd fd, const Error& error, std::vector<std::string> extra) {
  const std::string body = ErrorJson(error);
  if (error.status == 429 || error.status == 503) {
    extra.push_back(std::format("Retry-After: {}", kRetryAfterSeconds));
    extra.emplace_back("x-should-retry: true");
  }
  (void)(http::WriteAll(fd.get(), http::Head(error.status, kJson, body.size(), extra)) &&
         http::WriteAll(fd.get(), body));
  http::Finish(std::move(fd));
}

std::vector<Server::Pending> Server::Expired() {
  std::vector<Pending> out;
  const std::scoped_lock lock(mutex_);
  const auto now = Clock::now();
  while (!queue_.empty() && now - queue_.front().queued > options_.queue_wait) {
    out.push_back(std::move(queue_.front()));
    queue_.pop_front();
  }
  return out;
}

void Server::Accept() {
  for (;;) {
    std::array<pollfd, 2> fds{{{.fd = listener_.get(), .events = POLLIN, .revents = 0},
                               {.fd = stop_.get(), .events = POLLIN, .revents = 0}}};
    const int n = ::poll(fds.data(), fds.size(), 1000);
    if (n < 0 && errno != EINTR) {
      Log(std::format("the chat route's listener failed: poll: {}", errno));
      return;
    }
    if ((fds[1].revents & POLLIN) != 0) {
      return;
    }
    for (Pending& p : Expired()) {
      Log(std::format("request {}: 429 after waiting {} s in the queue", p.id,
                      options_.queue_wait.count() / 1000));
      Respond(std::move(p.fd),
              Refusal(429, "the request waited too long behind others; retry later"));
    }
    if ((fds[0].revents & POLLIN) != 0) {
      http::Fd fd = http::Accept(listener_.get(), options_.write_timeout);
      if (fd.valid()) {
        Handle(std::move(fd));
      }
    }
  }
}

void Server::Handle(http::Fd fd) {
  const auto accepted = Clock::now();
  const http::Limits limits{.max_header_bytes = kMaxHeaderBytes,
                            .max_headers = kMaxHeaders,
                            .max_target_bytes = kMaxTargetBytes,
                            .max_body_bytes = kMaxBodyBytes};
  auto read = http::ReadRequest(fd.get(), limits, accepted + options_.head_timeout,
                                accepted + options_.body_timeout);
  if (!read) {
    if (read.error().status == 0) {
      http::Finish(std::move(fd));
    } else {
      Log(std::format("refused a request: {}", read.error().status));
      Respond(std::move(fd), Refusal(read.error().status, read.error().message));
    }
    return;
  }
  const http::Request& request = *read;
  const auto refuse = [&](const Error& error) {
    Log(std::format("refused a request: {}", error.status));
    Respond(std::move(fd), error);
  };
  // Browser guards (D-045, D-064): the Host names this loopback listener,
  // and nothing arrives from a web page.
  const std::string* host = request.Find("host");
  if (host == nullptr || !IsLoopbackHost(*host)) {
    refuse(Refusal(403, "the Host header must name the loopback listener"));
    return;
  }
  const std::string* site = request.Find("sec-fetch-site");
  if (request.Find("origin") != nullptr ||
      (site != nullptr && *site != "none" && *site != "same-origin")) {
    refuse(Refusal(403, "requests from web pages are refused on this route"));
    return;
  }
  const auto method_not_allowed = [&](std::string_view allow) {
    Log("refused a request: 405");
    Respond(std::move(fd), Refusal(405, std::format("use {}", allow)),
            {std::format("Allow: {}", allow)});
  };
  if (request.path == "/v1/models" || request.path.starts_with("/v1/models/")) {
    if (request.method != "GET") {
      method_not_allowed("GET");
      return;
    }
    std::string body;
    if (request.path == "/v1/models") {
      body = ModelsJson(models_, created_);
    } else {
      const std::string_view id = std::string_view(request.path).substr(11);
      const auto it = std::ranges::find(models_, id, &ModelInfo::name);
      if (it == models_.end()) {
        refuse(Refusal(404, "The model does not exist", "model_not_found"));
        return;
      }
      body = ModelJson(*it, created_);
    }
    (void)(http::WriteAll(fd.get(), http::Head(200, kJson, body.size())) &&
           http::WriteAll(fd.get(), body));
    http::Finish(std::move(fd));
    return;
  }
  if (request.path != "/v1/chat/completions") {
    refuse(Refusal(404, "no such route; this runtime serves /v1/chat/completions and /v1/models"));
    return;
  }
  if (request.method != "POST") {
    method_not_allowed("POST");
    return;
  }
  const std::string* type = request.Find("content-type");
  std::string media;
  if (type != nullptr) {
    for (const char c : std::string_view(*type).substr(0, type->find(';'))) {
      if (c != ' ' && c != '\t') {
        media += c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c;
      }
    }
  }
  if (media != kJson) {
    refuse(Refusal(415, "the body must be Content-Type: application/json"));
    return;
  }
  auto parsed = ParseChatRequest(request.body);
  if (!parsed) {
    refuse(parsed.error());
    return;
  }
  const auto model = std::ranges::find(models_, parsed->model, &ModelInfo::name);
  if (model == models_.end()) {
    Error error = Refusal(404, "The model does not exist or is not configured on this node",
                          "model_not_found");
    error.param = "model";
    refuse(error);
    return;
  }
  if (!model->chat) {
    Error error = Refusal(400,
                          "This is not a chat model and thus not supported in the "
                          "v1/chat/completions endpoint",
                          "model_not_supported");
    error.param = "model";
    refuse(error);
    return;
  }
  Pending pending{.fd = std::move(fd),
                  .request = std::move(*parsed),
                  .queued = Clock::now(),
                  .id = RequestId()};
  bool stopping = false;
  {
    const std::scoped_lock lock(mutex_);
    stopping = stopping_;
    if (!stopping && queue_.size() < options_.max_queued) {
      queue_.push_back(std::move(pending));
      Signal(ready_.get());
      return;
    }
  }
  if (stopping) {
    Log(std::format("request {}: 503, the runtime is stopping", pending.id));
    Respond(std::move(pending.fd), Refusal(503, "the runtime is stopping"));
    return;
  }
  Log(std::format("request {}: 429, {} requests queued", pending.id, options_.max_queued));
  Respond(std::move(pending.fd),
          Refusal(429, std::format("{} requests are already waiting; retry later",
                                   options_.max_queued)));
}

void Server::Serve(Pending& pending, int wake_fd, const std::function<bool()>& on_wake) {
  const auto started = Clock::now();
  if (http::PeerGone(pending.fd.get())) {
    Log(std::format("request {}: the client left while it was queued", pending.id));
    http::Finish(std::move(pending.fd));
    return;
  }
  Stream stream(*this, pending, wake_fd, on_wake);
  const std::expected<Completion, Error> result = backend_.Complete(pending.request, stream);
  const int status = stream.End(result);
  if (pending.fd.valid()) {
    http::Finish(std::move(pending.fd));
  }
  const Usage& u = stream.usage();
  Log(std::format(
      "request {}: {} {}, {} prompt tokens ({} cached), {} generated, {:.3f} s "
      "({:.3f} s queued)",
      pending.id, pending.request.model, status, u.prompt_tokens, u.cached_tokens,
      u.completion_tokens, Seconds(Clock::now() - started), Seconds(started - pending.queued)));
  backend_.AfterResponse();
}

std::expected<void, std::string> Server::Run(int wake_fd, const std::function<bool()>& on_wake) {
  if (!listener_.valid()) {
    return std::unexpected("the chat route is not listening");
  }
  acceptor_ = std::jthread([this] { Accept(); });
  std::expected<void, std::string> result;
  for (;;) {
    {
      const std::scoped_lock lock(mutex_);
      if (stopping_) {
        break;
      }
    }
    std::array<pollfd, 2> fds{{{.fd = wake_fd, .events = POLLIN, .revents = 0},
                               {.fd = ready_.get(), .events = POLLIN, .revents = 0}}};
    const int n = ::poll(fds.data(), fds.size(), -1);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      result = std::unexpected(std::format("poll: {}", errno));
      break;
    }
    if ((fds[0].revents & POLLIN) != 0 && on_wake()) {
      break;
    }
    if ((fds[1].revents & POLLIN) == 0) {
      continue;
    }
    Drain(ready_.get());
    for (;;) {
      std::optional<Pending> next;
      {
        const std::scoped_lock lock(mutex_);
        if (stopping_ || queue_.empty()) {
          break;
        }
        next.emplace(std::move(queue_.front()));
        queue_.pop_front();
      }
      Serve(*next, wake_fd, on_wake);
      if (!backend_.healthy()) {
        result = std::unexpected(backend_.failure());
        break;
      }
      if (Readable(wake_fd) && on_wake()) {
        const std::scoped_lock lock(mutex_);
        stopping_ = true;
      }
    }
    if (!result) {
      break;
    }
  }
  std::deque<Pending> left;
  {
    const std::scoped_lock lock(mutex_);
    stopping_ = true;
    left.swap(queue_);
  }
  Signal(stop_.get());
  acceptor_.join();
  // Anything the acceptor queued before it ended.
  {
    const std::scoped_lock lock(mutex_);
    for (Pending& p : queue_) {
      left.push_back(std::move(p));
    }
    queue_.clear();
  }
  for (Pending& p : left) {
    Log(std::format("request {}: 503, the runtime is stopping", p.id));
    Respond(std::move(p.fd), Refusal(503, "the runtime is stopping"));
  }
  return result;
}

}  // namespace jitllm::runtime::api
