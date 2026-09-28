// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "runtime/http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jitllm::runtime::http {
namespace {

std::string Errno(int error) { return std::strerror(error); }  // NOLINT(concurrency-mt-unsafe)

std::unexpected<Failure> Fail(int status, std::string message) {
  return std::unexpected(Failure{.status = status, .message = std::move(message)});
}

char Lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

bool EqualsIgnoringCase(std::string_view a, std::string_view b) {
  return a.size() == b.size() &&
         std::ranges::equal(a, b, [](char x, char y) { return Lower(x) == Lower(y); });
}

// RFC 9110's tchar.
bool IsTokenChar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}

// Waits until fd is readable or `by` passes; false on timeout.
bool WaitReadable(int fd, Clock::time_point by) {
  for (;;) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(by - Clock::now());
    if (left.count() <= 0) {
      return false;
    }
    pollfd p{.fd = fd, .events = POLLIN, .revents = 0};
    const int n = ::poll(&p, 1, static_cast<int>(std::min<std::int64_t>(left.count(), 1000)));
    if (n > 0) {
      return true;  // readable, closed or failed: recv tells which
    }
    if (n < 0 && errno != EINTR) {
      return true;
    }
  }
}

// One recv into out (appending up to `most` bytes): the count, 0 at the
// peer's end, -1 on error.
ssize_t ReadSome(int fd, std::string& out, std::size_t most) {
  std::array<char, 16384> chunk{};
  for (;;) {
    const ssize_t n = ::recv(fd, chunk.data(), std::min(most, chunk.size()), 0);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n > 0) {
      out.append(chunk.data(), static_cast<std::size_t>(n));
    }
    return n;
  }
}

// A decimal Content-Length, or -1.
std::int64_t ParseLength(std::string_view text) {
  if (text.empty() || text.size() > 18 ||
      !std::ranges::all_of(text, [](char c) { return c >= '0' && c <= '9'; })) {
    return -1;
  }
  std::int64_t n = 0;
  for (const char c : text) {
    n = (n * 10) + (c - '0');
  }
  return n;
}

}  // namespace

Fd& Fd::operator=(Fd&& other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) {
      (void)::close(fd_);
    }
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

Fd::~Fd() {
  if (fd_ >= 0) {
    (void)::close(fd_);
  }
}

const std::string* Request::Find(std::string_view lowercase) const {
  for (const Header& h : headers) {
    if (h.name == lowercase) {
      return &h.value;
    }
  }
  return nullptr;
}

std::expected<Request, Failure> ReadRequest(int fd, const Limits& limits, Clock::time_point head_by,
                                            Clock::time_point body_by) {
  std::string buf;
  std::size_t head_end = std::string::npos;
  while (head_end == std::string::npos) {
    if (!WaitReadable(fd, head_by)) {
      return buf.empty() ? Fail(0, "no request") : Fail(408, "the request head took too long");
    }
    const ssize_t n = ReadSome(
        fd, buf, limits.max_header_bytes + 4 - std::min(buf.size(), limits.max_header_bytes));
    if (n <= 0) {
      return Fail(0, "the peer closed the connection");
    }
    head_end = buf.find("\r\n\r\n");
    if (head_end == std::string::npos && buf.size() >= limits.max_header_bytes) {
      return Fail(413,
                  std::format("the request head is longer than {} bytes", limits.max_header_bytes));
    }
  }
  if (head_end + 4 > limits.max_header_bytes + 4) {
    return Fail(413,
                std::format("the request head is longer than {} bytes", limits.max_header_bytes));
  }
  const std::string_view head = std::string_view(buf).substr(0, head_end + 2);  // last CRLF kept
  // Every line ends in CRLF: no bare CR or LF.
  for (std::size_t i = 0; i < head.size(); ++i) {
    if ((head[i] == '\r' && (i + 1 >= head.size() || head[i + 1] != '\n')) ||
        (head[i] == '\n' && (i == 0 || head[i - 1] != '\r'))) {
      return Fail(400, "request lines must end in CRLF");
    }
  }
  Request request;
  std::size_t at = head.find("\r\n");
  const std::string_view line = head.substr(0, at);
  {
    const std::size_t a = line.find(' ');
    const std::size_t b = a == std::string_view::npos ? a : line.find(' ', a + 1);
    if (b == std::string_view::npos || line.find(' ', b + 1) != std::string_view::npos) {
      return Fail(400, "the request line must be METHOD TARGET HTTP/1.1");
    }
    const std::string_view method = line.substr(0, a);
    const std::string_view target = line.substr(a + 1, b - a - 1);
    const std::string_view version = line.substr(b + 1);
    if (method.empty() || method.size() > 16 ||
        !std::ranges::all_of(method, [](char c) { return c >= 'A' && c <= 'Z'; })) {
      return Fail(400, "the method is not valid");
    }
    if (target.empty() || target.front() != '/' || target.size() > limits.max_target_bytes ||
        !std::ranges::all_of(target, [](char c) { return c > 0x20 && c < 0x7F; })) {
      return Fail(target.size() > limits.max_target_bytes ? 414 : 400,
                  "the target must be an origin-form path of printable ASCII, at most " +
                      std::to_string(limits.max_target_bytes) + " bytes");
    }
    if (version != "HTTP/1.1" && version != "HTTP/1.0") {
      return Fail(version.starts_with("HTTP/") ? 505 : 400, "only HTTP/1.1 is served");
    }
    request.method = std::string(method);
    request.target = std::string(target);
    request.path = std::string(target.substr(0, target.find('?')));
  }
  std::int64_t length = -1;
  while (at + 2 < head.size()) {
    const std::size_t start = at + 2;
    at = head.find("\r\n", start);
    const std::string_view header = head.substr(start, at - start);
    if (request.headers.size() >= limits.max_headers) {
      return Fail(413, std::format("the request has more than {} headers", limits.max_headers));
    }
    const std::size_t colon = header.find(':');
    if (colon == std::string_view::npos || colon == 0 ||
        !std::ranges::all_of(header.substr(0, colon), IsTokenChar)) {
      return Fail(400, "a header line is not valid (or is folded)");
    }
    std::string_view value = header.substr(colon + 1);
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
      value.remove_prefix(1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
      value.remove_suffix(1);
    }
    if (std::ranges::any_of(value, [](char c) {
          const auto b = static_cast<unsigned char>(c);
          return (b < 0x20 && b != '\t') || b == 0x7F;
        })) {
      return Fail(400, "a header value holds a control character");
    }
    Header h;
    h.name.reserve(colon);
    for (const char c : header.substr(0, colon)) {
      h.name += Lower(c);
    }
    h.value = std::string(value);
    if (h.name == "host" && request.Find("host") != nullptr) {
      return Fail(400, "the request has more than one Host header");  // RFC 9112, section 3.2
    }
    if (h.name == "transfer-encoding") {
      return Fail(501, "request bodies are read only by Content-Length");
    }
    if (h.name == "content-length") {
      const std::int64_t n = ParseLength(h.value);
      if (n < 0 || (length >= 0 && n != length)) {
        return Fail(400, "Content-Length is not valid");
      }
      length = n;
    }
    request.headers.push_back(std::move(h));
  }
  if (length < 0 && request.method == "POST") {
    return Fail(411, "a POST needs Content-Length");
  }
  const auto body_bytes = static_cast<std::size_t>(std::max<std::int64_t>(length, 0));
  if (body_bytes > limits.max_body_bytes) {
    return Fail(413, std::format("the body is longer than {} bytes", limits.max_body_bytes));
  }
  request.body = buf.substr(head_end + 4, body_bytes);
  if (const std::string* expect = request.Find("expect")) {
    if (!EqualsIgnoringCase(*expect, "100-continue")) {
      return Fail(417, "only Expect: 100-continue is understood");
    }
    if (request.body.size() < body_bytes && !WriteAll(fd, "HTTP/1.1 100 Continue\r\n\r\n")) {
      return Fail(0, "the peer is gone");
    }
  }
  while (request.body.size() < body_bytes) {
    if (!WaitReadable(fd, body_by)) {
      return Fail(408, "the request body took too long");
    }
    if (ReadSome(fd, request.body, body_bytes - request.body.size()) <= 0) {
      return Fail(0, "the peer closed the connection");
    }
  }
  return request;
}

std::string_view Reason(int status) {
  switch (status) {
    case 100:
      return "Continue";
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 403:
      return "Forbidden";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 408:
      return "Request Timeout";
    case 411:
      return "Length Required";
    case 413:
      return "Content Too Large";
    case 414:
      return "URI Too Long";
    case 415:
      return "Unsupported Media Type";
    case 417:
      return "Expectation Failed";
    case 429:
      return "Too Many Requests";
    case 500:
      return "Internal Server Error";
    case 501:
      return "Not Implemented";
    case 503:
      return "Service Unavailable";
    case 504:
      return "Gateway Timeout";
    case 505:
      return "HTTP Version Not Supported";
    default:
      return "Error";
  }
}

std::string Head(int status, std::string_view content_type, std::optional<std::size_t> length,
                 const std::vector<std::string>& extra) {
  std::string head =
      std::format("HTTP/1.1 {} {}\r\nContent-Type: {}\r\n", status, Reason(status), content_type);
  if (length) {
    head += std::format("Content-Length: {}\r\n", *length);
  }
  head += "Cache-Control: no-store\r\nConnection: close\r\n";
  for (const std::string& line : extra) {
    head += line + "\r\n";
  }
  head += "\r\n";
  return head;
}

bool WriteAll(int fd, std::string_view data) {
  while (!data.empty()) {
    const ssize_t n = ::send(fd, data.data(), data.size(), MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return false;  // gone, or stalled past SO_SNDTIMEO (EAGAIN)
    }
    data.remove_prefix(static_cast<std::size_t>(n));
  }
  return true;
}

bool PeerGone(int fd) {
  pollfd p{.fd = fd, .events = POLLRDHUP, .revents = 0};
  return ::poll(&p, 1, 0) > 0 && (p.revents & (POLLRDHUP | POLLHUP | POLLERR)) != 0;
}

void Finish(Fd fd) {
  if (!fd.valid()) {
    return;
  }
  (void)::shutdown(fd.get(), SHUT_WR);
  const auto by = Clock::now() + std::chrono::milliseconds(200);
  std::size_t discarded = 0;
  std::string sink;
  while (discarded < (std::size_t{1} << 20U)) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(by - Clock::now());
    if (left.count() <= 0) {
      break;
    }
    pollfd p{.fd = fd.get(), .events = POLLIN, .revents = 0};
    if (::poll(&p, 1, static_cast<int>(left.count())) <= 0) {
      break;
    }
    sink.clear();
    const ssize_t n = ReadSome(fd.get(), sink, 16384);
    if (n <= 0) {
      break;
    }
    discarded += static_cast<std::size_t>(n);
  }
}

std::expected<Listener, std::string> ListenLoopback(const config::ClientEndpoint& endpoint) {
  const int family = endpoint.ipv6 ? AF_INET6 : AF_INET;
  Fd fd(::socket(family, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
  if (!fd.valid()) {
    return std::unexpected("socket: " + Errno(errno));
  }
  const int on = 1;
  (void)::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
  sockaddr_storage storage{};
  socklen_t size = 0;
  if (endpoint.ipv6) {
    (void)::setsockopt(fd.get(), IPPROTO_IPV6, IPV6_V6ONLY, &on, sizeof on);
    sockaddr_in6 six{};
    six.sin6_family = AF_INET6;
    six.sin6_port = htons(endpoint.port);
    if (::inet_pton(AF_INET6, endpoint.address.c_str(), &six.sin6_addr) != 1) {
      return std::unexpected("not an IPv6 address: " + endpoint.address);
    }
    std::memcpy(&storage, &six, sizeof six);
    size = sizeof six;
  } else {
    sockaddr_in four{};
    four.sin_family = AF_INET;
    four.sin_port = htons(endpoint.port);
    if (::inet_pton(AF_INET, endpoint.address.c_str(), &four.sin_addr) != 1) {
      return std::unexpected("not an IPv4 address: " + endpoint.address);
    }
    std::memcpy(&storage, &four, sizeof four);
    size = sizeof four;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
  if (::bind(fd.get(), reinterpret_cast<const sockaddr*>(&storage), size) != 0) {
    return std::unexpected(
        std::format("bind {}:{}: {}", endpoint.address, endpoint.port, Errno(errno)));
  }
  if (::listen(fd.get(), 16) != 0) {
    return std::unexpected("listen: " + Errno(errno));
  }
  sockaddr_storage bound{};
  socklen_t bound_size = sizeof bound;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
  if (::getsockname(fd.get(), reinterpret_cast<sockaddr*>(&bound), &bound_size) != 0) {
    return std::unexpected("getsockname: " + Errno(errno));
  }
  std::uint16_t port = 0;
  if (endpoint.ipv6) {
    sockaddr_in6 six{};
    std::memcpy(&six, &bound, sizeof six);
    port = ntohs(six.sin6_port);
  } else {
    sockaddr_in four{};
    std::memcpy(&four, &bound, sizeof four);
    port = ntohs(four.sin_port);
  }
  return Listener{.fd = std::move(fd), .port = port};
}

Fd Accept(int listener, std::chrono::milliseconds write_timeout) {
  for (;;) {
    const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
    if (fd < 0 && errno == EINTR) {
      continue;
    }
    if (fd < 0) {
      return Fd();
    }
    Fd accepted(fd);
    timeval tv{};
    tv.tv_sec = static_cast<time_t>(write_timeout.count() / 1000);
    tv.tv_usec = static_cast<suseconds_t>((write_timeout.count() % 1000) * 1000);
    (void)::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    const int on = 1;
    (void)::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
    return accepted;
  }
}

}  // namespace jitllm::runtime::http
