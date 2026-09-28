// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's loopback chat route as a server (D-097;
// docs/runtime-serving.md#the-chat-route): `GET /v1/models`,
// `GET /v1/models/{id}` and `POST /v1/chat/completions` on one loopback
// listener, over a Backend that runs the requests. Vendor-free, so the CPU
// tests drive it with a fake backend.
//
// Threads. An acceptor thread reads each connection's request (one at a
// time, under http.h's bounds and the head and body timeouts), refuses what
// is malformed, out of bounds, from a browser or for an unknown model,
// answers the model list itself, and queues a valid chat request: at most
// kMaxQueued wait behind the running one, each at most kQueueWaitMs, and a
// request beyond either gets a 429 with Retry-After. The thread that calls
// Run is the backend's (the node's one driver): it takes queued requests
// in order and runs each to its end, one at a time, while it watches the
// caller's wake descriptor (the runtime's signals) between requests and
// between generation steps.
//
// A request's end. Non-streaming: one JSON body once the outcome is known
// (a 504 past kDeadlineMs, a 503 when the runtime stops). Streaming: the
// headers and the role chunk once the backend admits it (its tokens
// counted against the model's context), then a chunk per step's text, the
// finish chunk, the usage chunk if asked for, and `[DONE]`; a failure after
// the headers is an `error` event and the stream ends without `[DONE]`.
// A peer that goes away, or stops reading for kWriteTimeoutMs, ends the
// generation at the next step. Nothing a request holds is logged: each
// request logs an opaque ID, the model, the status and token counts.

#ifndef JITLLM_RUNTIME_API_SERVER_H_
#define JITLLM_RUNTIME_API_SERVER_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <expected>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "config/node_config.h"
#include "runtime/api.h"
#include "runtime/http.h"

namespace jitllm::runtime::api {

// One admitted request as the backend sees it. Each call returns whether
// to go on: false ends the generation (a stop string, the peer gone, the
// deadline, the runtime stopping).
class Exchange {
 public:
  Exchange() = default;
  Exchange(const Exchange&) = delete;
  Exchange& operator=(const Exchange&) = delete;
  Exchange(Exchange&&) = delete;
  Exchange& operator=(Exchange&&) = delete;
  virtual ~Exchange() = default;

  struct Admission {
    std::uint32_t prompt_tokens = 0;  // the whole conversation, rendered
  };
  // Once, after the model's own checks and before any model work: a
  // streaming response starts here.
  virtual bool Admit(const Admission& admission) = 0;
  // Generated text, in order. Reasoning with empty text: the reasoning
  // block ended having said nothing.
  virtual bool Reasoning(std::string_view text) = 0;
  virtual bool Content(std::string_view text) = 0;
  // Between steps that produced no text.
  virtual bool Continue() = 0;
};

struct Completion {
  std::uint32_t completion_tokens = 0;  // generated, a stop token included
  std::uint32_t cached_tokens = 0;      // the prompt's tokens the state already held
  bool stopped = false;                 // ended at the model's stop token
};

class Backend {
 public:
  Backend() = default;
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  Backend(Backend&&) = delete;
  Backend& operator=(Backend&&) = delete;
  virtual ~Backend() = default;

  // The models, once, when the server is made.
  virtual std::vector<ModelInfo> Models() const = 0;
  // Runs one request on Run's thread: an Error before Admit refuses it (a
  // 4xx, or a 5xx); after Admit, an Error is a failure mid-response.
  virtual std::expected<Completion, Error> Complete(const ChatRequest& request,
                                                    Exchange& exchange) = 0;
  // After the response is out: work off the request's path.
  virtual void AfterResponse() {}
  // False once a failure left the backend unable to serve: Run returns.
  virtual bool healthy() const { return true; }
  virtual std::string failure() const { return {}; }
};

struct ServerOptions {
  config::ClientEndpoint bind;
  std::chrono::milliseconds head_timeout{kHeaderTimeoutMs};
  std::chrono::milliseconds body_timeout{kBodyTimeoutMs};
  std::chrono::milliseconds write_timeout{kWriteTimeoutMs};
  std::chrono::milliseconds queue_wait{kQueueWaitMs};
  std::chrono::milliseconds deadline{kDeadlineMs};
  std::size_t max_queued = kMaxQueued;
  std::FILE* log = nullptr;  // one line a request; nullptr: none
};

class Server {
 public:
  Server(Backend& backend, ServerOptions options);
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;
  Server(Server&&) = delete;
  Server& operator=(Server&&) = delete;
  ~Server();

  // Binds the listener; the port it got.
  std::expected<std::uint16_t, std::string> Listen();
  // Serves until `wake_fd` is readable and `on_wake` (which consumes what
  // made it so) returns true, or the backend fails; then refuses what is
  // still queued (503) and returns. The error: the backend's failure.
  std::expected<void, std::string> Run(int wake_fd, const std::function<bool()>& on_wake);

 private:
  struct Pending {
    http::Fd fd;
    ChatRequest request;
    std::chrono::steady_clock::time_point queued;
    std::string id;
  };
  class Stream;

  void Accept();
  void Handle(http::Fd fd);
  void Serve(Pending& pending, int wake_fd, const std::function<bool()>& on_wake);
  void Respond(http::Fd fd, const Error& error, std::vector<std::string> extra = {});
  void Log(std::string_view line);
  std::vector<Pending> Expired();

  Backend& backend_;
  ServerOptions options_;
  std::vector<ModelInfo> models_;
  std::int64_t created_ = 0;
  http::Fd listener_;
  http::Fd stop_;   // eventfd: the acceptor ends
  http::Fd ready_;  // eventfd: something was queued
  std::mutex mutex_;
  std::deque<Pending> queue_;
  bool stopping_ = false;
  std::jthread acceptor_;
};

// Whether a Host header names this listener's loopback (DNS-rebinding
// guard): localhost, 127.0.0.0/8 or [::1], with any port.
bool IsLoopbackHost(std::string_view host);

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_API_SERVER_H_
