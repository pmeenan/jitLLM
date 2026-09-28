// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// A small, bounded HTTP/1.1 server's pieces for M3's loopback chat route
// (D-097; docs/runtime-serving.md#the-chat-route): a loopback listener,
// one request read from a connection under byte and time bounds, and
// responses written with a per-write timeout. No third-party code.
//
// What a request may be: a request line of a method, an origin-form
// target and HTTP/1.1 (or 1.0); header lines ending in CRLF, without
// folding or control characters; a body only by Content-Length (a
// Transfer-Encoding is refused), which `Expect: 100-continue` is answered
// for. Every response closes the connection, so nothing is pipelined or
// kept alive.

#ifndef JITLLM_RUNTIME_HTTP_H_
#define JITLLM_RUNTIME_HTTP_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "config/node_config.h"

namespace jitllm::runtime::http {

using Clock = std::chrono::steady_clock;

// An owned file descriptor.
class Fd {
 public:
  Fd() = default;
  explicit Fd(int fd) : fd_(fd) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  Fd(Fd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  Fd& operator=(Fd&& other) noexcept;
  ~Fd();
  int get() const { return fd_; }
  bool valid() const { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

struct Limits {
  std::size_t max_header_bytes = 16384;  // the request line and headers
  std::size_t max_headers = 64;
  std::size_t max_target_bytes = 2048;
  std::size_t max_body_bytes = std::size_t{4} << 20U;
};

struct Header {
  std::string name;  // lower case
  std::string value;
};

struct Request {
  std::string method;
  std::string target;  // as sent
  std::string path;    // the target without its query
  std::vector<Header> headers;
  std::string body;

  // The value of the header named `lowercase` (the first), or nullptr.
  const std::string* Find(std::string_view lowercase) const;
};

// Why a request could not be read: the status to answer with, or 0 when
// there is nobody to answer (the peer closed, or never sent a byte).
struct Failure {
  int status = 400;
  std::string message;
};

// Reads one request from a connected socket: its head by `head_by`, its
// body by `body_by`.
std::expected<Request, Failure> ReadRequest(int fd, const Limits& limits, Clock::time_point head_by,
                                            Clock::time_point body_by);

// A status line's reason phrase.
std::string_view Reason(int status);

// A response head: the status line, Content-Type, Content-Length (when
// given, else none), Connection: close, Cache-Control: no-store, and
// `extra` header lines (each "Name: value").
std::string Head(int status, std::string_view content_type, std::optional<std::size_t> length,
                 const std::vector<std::string>& extra = {});

// Writes all of data; false when the peer is gone or a write stalled past
// the socket's send timeout (SO_SNDTIMEO, set at accept).
bool WriteAll(int fd, std::string_view data);

// Whether the peer has closed or reset the connection (without blocking).
bool PeerGone(int fd);

// Ends a connection: half-close, then discard what the peer still sends
// for a moment (so a response is not lost to a reset), and close.
void Finish(Fd fd);

// A listening TCP socket on a loopback endpoint (port 0: the kernel's
// choice) and the port it got.
struct Listener {
  Fd fd;
  std::uint16_t port = 0;
};
std::expected<Listener, std::string> ListenLoopback(const config::ClientEndpoint& endpoint);

// Accepts one connection (non-blocking listener), with its send and
// receive timeouts set; an invalid Fd when none is waiting.
Fd Accept(int listener, std::chrono::milliseconds write_timeout);

}  // namespace jitllm::runtime::http

#endif  // JITLLM_RUNTIME_HTTP_H_
