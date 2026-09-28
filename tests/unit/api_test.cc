// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The loopback chat route (D-097; runtime/api.h, http.h, api_server.h):
// request parsing and every intake bound, the output's stop strings and
// reasoning split, the JSON shapes, and the server end to end over a real
// loopback socket with a fake backend: routes, browser guards, HTTP
// bounds, timeouts, the queue, streaming and errors after the headers.

#include "runtime/api.h"

#include <arpa/inet.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "base/json.h"
#include "config/node_config.h"
#include "runtime/api_server.h"
#include "runtime/http.h"

namespace {

namespace api = jitllm::runtime::api;
namespace json = jitllm::base::json;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;

// A body's parse result: the request, or the error's status and message.
std::expected<api::ChatRequest, api::Error> Parse(std::string_view body) {
  return api::ParseChatRequest(body);
}

std::string ErrorOf(std::string_view body) {
  auto r = Parse(body);
  if (r.has_value()) {
    return "(accepted)";
  }
  return std::format("{} {} [{}]", r.error().status, r.error().message, r.error().param);
}

constexpr std::string_view kMinimal =
    R"({"model":"m","messages":[{"role":"user","content":"hi"}]})";

std::string WithField(std::string_view field) {
  return std::format(R"({{"model":"m","messages":[{{"role":"user","content":"hi"}}],{}}})", field);
}

TEST(ChatRequest, ReadsTheHonoredFields) {
  auto minimal = Parse(kMinimal);
  ASSERT_TRUE(minimal.has_value());
  EXPECT_EQ(minimal->model, "m");
  ASSERT_EQ(minimal->messages.size(), 1U);
  EXPECT_EQ(minimal->messages[0].content, "hi");
  EXPECT_FALSE(minimal->max_tokens.has_value());
  EXPECT_EQ(minimal->temperature, 1.0);
  EXPECT_EQ(minimal->top_p, 1.0);
  EXPECT_FALSE(minimal->stream);

  auto full = Parse(R"({"model":"deepseek","messages":[
      {"role":"developer","content":"be brief"},
      {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b","cache_control":{}}]},
      {"role":"assistant","content":"c","reasoning_content":"r","refusal":null,"annotations":[]},
      {"role":"user","content":"d","name":"pat"}],
    "max_completion_tokens":32,"temperature":0,"top_p":0.5,"seed":-1,"stop":["x","yz"],
    "stream":true,"stream_options":{"include_usage":true},"n":1,"presence_penalty":0,
    "logprobs":false,"tools":[],"tool_choice":"none","response_format":{"type":"text"},
    "user":"u","metadata":{"k":"v"},"store":false,"parallel_tool_calls":true,
    "prompt_cache_key":"p","frequency_penalty":null})");
  ASSERT_TRUE(full.has_value()) << full.error().message;
  EXPECT_EQ(full->messages[0].role, api::Role::kSystem);
  EXPECT_EQ(full->messages[1].content, "ab");
  EXPECT_EQ(full->messages[2].role, api::Role::kAssistant);
  EXPECT_EQ(full->messages[2].reasoning, std::optional<std::string>("r"));
  EXPECT_EQ(full->max_tokens, std::optional<std::uint32_t>(32));
  EXPECT_EQ(full->temperature, 0.0);
  EXPECT_EQ(full->top_p, 0.5);
  EXPECT_EQ(full->seed, std::optional<std::uint64_t>(~std::uint64_t{0}));
  EXPECT_THAT(full->stop, ::testing::ElementsAre("x", "yz"));
  EXPECT_TRUE(full->stream);
  EXPECT_TRUE(full->include_usage);
  auto one_stop = Parse(WithField(R"("stop":"end","max_tokens":5,"max_completion_tokens":5)"));
  ASSERT_TRUE(one_stop.has_value());
  EXPECT_THAT(one_stop->stop, ::testing::ElementsAre("end"));
  EXPECT_EQ(one_stop->max_tokens, std::optional<std::uint32_t>(5));
}

TEST(ChatRequest, RefusesWhatItDoesNotHonor) {
  EXPECT_THAT(ErrorOf("{"), StartsWith("400 the body is not valid JSON"));
  EXPECT_THAT(ErrorOf("[]"), StartsWith("400 the body must be a JSON object"));
  EXPECT_THAT(ErrorOf(R"({"messages":[{"role":"user","content":"x"}]})"), HasSubstr("[model]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m"})"), HasSubstr("[messages]"));
  EXPECT_THAT(ErrorOf(WithField(R"("frobnicate":1)")),
              HasSubstr("Unrecognized request argument supplied: frobnicate [frobnicate]"));
  EXPECT_THAT(ErrorOf(WithField(R"("frobnicate":null)")), HasSubstr("[frobnicate]"));
  EXPECT_THAT(ErrorOf(WithField(R"("transforms":["middle-out"])")), HasSubstr("[transforms]"));
  EXPECT_THAT(ErrorOf(WithField(R"("n":2)")), HasSubstr("[n]"));
  EXPECT_THAT(ErrorOf(WithField(R"("logprobs":true)")), HasSubstr("[logprobs]"));
  EXPECT_THAT(ErrorOf(WithField(R"("presence_penalty":0.5)")), HasSubstr("[presence_penalty]"));
  EXPECT_THAT(ErrorOf(WithField(R"("tools":[{"type":"function"}])")), HasSubstr("[tools]"));
  EXPECT_THAT(ErrorOf(WithField(R"("response_format":{"type":"json_object"})")),
              HasSubstr("[response_format]"));
  EXPECT_THAT(ErrorOf(WithField(R"("store":true)")), HasSubstr("[store]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stream":"yes")")), HasSubstr("[stream]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stream_options":{"chunk":1})")),
              HasSubstr("[stream_options.chunk]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":4,"max_completion_tokens":5)")),
              HasSubstr("[max_completion_tokens]"));
  // Messages.
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[]})"), HasSubstr("[messages]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"tool","content":"x"}]})"),
              HasSubstr("tools are not supported"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"robot","content":"x"}]})"),
              HasSubstr("[messages[0].role]"));
  EXPECT_THAT(
      ErrorOf(R"({"model":"m","messages":[{"role":"user","content":[{"type":"image_url"}]}]})"),
      HasSubstr("[messages[0].content[0].type]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"user"}]})"),
              HasSubstr("[messages[0].content]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"user","content":"x","x":1}]})"),
              HasSubstr("[messages[0].x]"));
  EXPECT_THAT(
      ErrorOf(R"({"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[{}]}]})"),
      HasSubstr("tool calls are not supported"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"user","content":"a"},
                                                  {"role":"assistant","content":"b"}]})"),
              HasSubstr("the last message must be the user's"));
}

// Every numeric bound, at its edge and one past it.
TEST(ChatRequest, EnforcesItsBounds) {
  // Messages: count and bytes.
  const auto messages = [](std::size_t n, std::size_t bytes) {
    std::string list;
    for (std::size_t i = 0; i < n; ++i) {
      list += std::format(R"({}{{"role":"user","content":"{}"}})", i == 0 ? "" : ",",
                          std::string(bytes, 'a'));
    }
    return std::format(R"({{"model":"m","messages":[{}]}})", list);
  };
  EXPECT_TRUE(Parse(messages(api::kMaxMessages, 1)).has_value());
  EXPECT_THAT(ErrorOf(messages(api::kMaxMessages + 1, 1)), HasSubstr("more than 1024"));
  EXPECT_TRUE(Parse(messages(1, api::kMaxMessageBytes)).has_value());
  EXPECT_THAT(ErrorOf(messages(1, api::kMaxMessageBytes + 1)), HasSubstr("longer than"));
  // Parts: count and joined bytes.
  const auto parts = [](std::size_t n, std::size_t bytes) {
    std::string list;
    for (std::size_t i = 0; i < n; ++i) {
      list += std::format(R"({}{{"type":"text","text":"{}"}})", i == 0 ? "" : ",",
                          std::string(bytes, 'a'));
    }
    return std::format(R"({{"model":"m","messages":[{{"role":"user","content":[{}]}}]}})", list);
  };
  EXPECT_TRUE(Parse(parts(api::kMaxContentParts, 1)).has_value());
  EXPECT_THAT(ErrorOf(parts(api::kMaxContentParts + 1, 1)), HasSubstr("more than 64"));
  EXPECT_THAT(ErrorOf(parts(2, (api::kMaxMessageBytes / 2) + 1)), HasSubstr("longer than"));
  // The model's name.
  EXPECT_TRUE(Parse(std::format(R"({{"model":"{}","messages":[{{"role":"user","content":"x"}}]}})",
                                std::string(api::kMaxModelBytes, 'm')))
                  .has_value());
  EXPECT_THAT(
      ErrorOf(std::format(R"({{"model":"{}","messages":[{{"role":"user","content":"x"}}]}})",
                          std::string(api::kMaxModelBytes + 1, 'm'))),
      HasSubstr("[model]"));
  EXPECT_THAT(ErrorOf(R"({"model":"","messages":[{"role":"user","content":"x"}]})"),
              HasSubstr("[model]"));
  // Tokens, temperature, top_p, seed.
  EXPECT_TRUE(Parse(WithField(R"("max_tokens":262144)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":262145)")), HasSubstr("[max_tokens]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":0)")), HasSubstr("[max_tokens]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":1.5)")), HasSubstr("[max_tokens]"));
  EXPECT_TRUE(Parse(WithField(R"("temperature":2)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("temperature":2.01)")), HasSubstr("[temperature]"));
  EXPECT_THAT(ErrorOf(WithField(R"("temperature":-0.1)")), HasSubstr("[temperature]"));
  EXPECT_THAT(ErrorOf(WithField(R"("temperature":1e999)")), HasSubstr("[temperature]"));
  EXPECT_TRUE(Parse(WithField(R"("top_p":1)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("top_p":0)")), HasSubstr("[top_p]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_p":1.0001)")), HasSubstr("[top_p]"));
  // A top_p the sampler's float rounds to 0 would fail mid-generation.
  EXPECT_THAT(ErrorOf(WithField(R"("top_p":1e-50)")), HasSubstr("[top_p]"));
  EXPECT_TRUE(Parse(WithField(R"("top_p":1e-30)")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("seed":9223372036854775808)")), HasSubstr("[seed]"));
  EXPECT_THAT(ErrorOf(WithField(R"("seed":1.0)")), HasSubstr("[seed]"));
  // Stop strings: count and bytes.
  EXPECT_TRUE(Parse(WithField(R"("stop":["a","b","c","d"])")).has_value());
  EXPECT_THAT(ErrorOf(WithField(R"("stop":["a","b","c","d","e"])")), HasSubstr("[stop]"));
  EXPECT_TRUE(Parse(WithField(std::format(R"("stop":"{}")", std::string(api::kMaxStopBytes, 's'))))
                  .has_value());
  EXPECT_THAT(
      ErrorOf(WithField(std::format(R"("stop":"{}")", std::string(api::kMaxStopBytes + 1, 's')))),
      HasSubstr("[stop]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stop":[""])")), HasSubstr("[stop[0]]"));
  // JSON depth and size.
  std::string deep(api::kMaxJsonDepth + 1, '[');
  deep += std::string(api::kMaxJsonDepth + 1, ']');
  EXPECT_THAT(ErrorOf(WithField(std::format(R"("metadata":{{"a":{}}})", deep))),
              HasSubstr("not valid JSON"));
  EXPECT_THAT(ErrorOf(std::string(api::kMaxBodyBytes + 1, ' ')), HasSubstr("not valid JSON"));
}

TEST(OutputText, EndsAtAStopStringAcrossPieces) {
  api::OutputText text({"STOP", "\n\n"});
  EXPECT_EQ(text.Content("Hello ST").content, "Hello ");  // "ST" might begin STOP
  EXPECT_EQ(text.Content("ORM").content, "STORM");
  EXPECT_EQ(text.Content(" and\n").content, " and");
  EXPECT_EQ(text.Content("\nmore").content, "");
  EXPECT_TRUE(text.stopped());
  EXPECT_EQ(text.Content("ignored").content, "");
  EXPECT_EQ(text.Finish().content, "");
}

TEST(OutputText, FlushesWhatItHeldAtTheEnd) {
  api::OutputText text({"STOP"});
  EXPECT_EQ(text.Content("abcSTO").content, "abc");
  EXPECT_EQ(text.Finish().content, "STO");
  EXPECT_FALSE(text.stopped());
}

TEST(OutputText, SplitsReasoningAndTrimsTheAnswersStart) {
  api::OutputText text({"x"});
  EXPECT_EQ(text.Reasoning("I think x").reasoning, "I think x");  // no stop in reasoning
  EXPECT_EQ(text.Content("\n\n").content, "");
  EXPECT_EQ(text.Content(" Paris").content, "Paris");
  EXPECT_EQ(text.Content("\n ok").content, "\n ok");
  api::OutputText plain({});
  EXPECT_EQ(plain.Content("\nkept").content, "\nkept");
  // A reasoning block that said nothing still trims the answer's start.
  api::OutputText empty({});
  EXPECT_EQ(empty.Reasoning({}).reasoning, "");
  EXPECT_EQ(empty.Content("\n\nParis").content, "Paris");
}

// Untrusted text: NUL and escapes pass through as text; ill-formed UTF-8
// is refused.
TEST(ChatRequest, TakesTextAsText) {
  auto nul = Parse(R"({"model":"m","messages":[{"role":"user","content":"a\u0000b"}]})");
  ASSERT_TRUE(nul.has_value());
  using namespace std::string_view_literals;
  EXPECT_EQ(nul->messages[0].content, "a\0b"sv);
  EXPECT_THAT(ErrorOf("{\"model\":\"m\",\"messages\":[{\"role\":\"user\",\"content\":\"\xC3\"}]}"),
              HasSubstr("not valid JSON"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","model":"n","messages":[{"role":"user","content":"x"}]})"),
              HasSubstr("not valid JSON"));  // a duplicate key
}

TEST(ApiJson, ShapesParseBack) {
  const api::Usage usage{.prompt_tokens = 7, .completion_tokens = 3, .cached_tokens = 2};
  const std::string completion = api::CompletionJson("chatcmpl-1", 5, "m", "a\"b", std::string("r"),
                                                     api::Finish::kLength, usage);
  auto doc = json::Parse(completion);
  ASSERT_TRUE(doc.has_value()) << completion;
  const json::Value choice = doc->root().find("choices")->at(0);
  EXPECT_EQ(choice.find("message")->find("content")->string(), "a\"b");
  EXPECT_EQ(choice.find("message")->find("reasoning")->string(), "r");
  EXPECT_EQ(choice.find("finish_reason")->string(), "length");
  EXPECT_EQ(doc->root().find("usage")->find("total_tokens")->int64(), 10);
  EXPECT_EQ(
      doc->root().find("usage")->find("prompt_tokens_details")->find("cached_tokens")->int64(), 2);
  for (const std::string& chunk :
       {api::ChunkJson("i", 1, "m", api::Delta::kRole, {}),
        api::ChunkJson("i", 1, "m", api::Delta::kContent, "x\n"),
        api::ChunkJson("i", 1, "m", api::Delta::kFinish, {}, api::Finish::kStop),
        api::UsageChunkJson("i", 1, "m", usage), api::ModelsJson({{"a", true, 1}}, 1),
        api::ErrorJson({.status = 400, .type = "t", .message = "m", .param = {}, .code = "c"})}) {
    EXPECT_TRUE(json::Parse(chunk).has_value()) << chunk;
  }
  EXPECT_THAT(api::ChunkJson("i", 1, "m", api::Delta::kFinish, {}, api::Finish::kStop),
              HasSubstr(R"("delta":{},"logprobs":null,"finish_reason":"stop")"));
}

TEST(ApiHost, OnlyLoopbackNames) {
  for (const std::string_view ok : {"localhost", "localhost:8114", "127.0.0.1", "127.0.0.1:8114",
                                    "127.9.9.9:1", "[::1]", "[::1]:8114", "LocalHost:8114"}) {
    EXPECT_TRUE(api::IsLoopbackHost(ok)) << ok;
  }
  for (const std::string_view bad : {"", "example.com", "evil.localhost", "10.0.0.1:8114",
                                     "[::2]:8114", "[::1", "[::1]x", "localhost.:80"}) {
    EXPECT_FALSE(api::IsLoopbackHost(bad)) << bad;
  }
}

// ---------------------------------------------------------------- the server

// Runs requests as the test directs: "block" waits for release; "fail"
// fails before admission, "late" after it; otherwise reasoning, then the
// last message's content echoed in two pieces.
class FakeBackend final : public api::Backend {
 public:
  std::vector<api::ModelInfo> Models() const override {
    return {{.name = "alpha", .chat = true, .context = 100},
            {.name = "image", .chat = false, .context = 0}};
  }
  std::expected<api::Completion, api::Error> Complete(const api::ChatRequest& request,
                                                      api::Exchange& exchange) override {
    const std::string& text = request.messages.back().content;
    if (text == "fail") {
      return std::unexpected(api::Error{.status = 400,
                                        .type = "invalid_request_error",
                                        .message = "too long",
                                        .param = "messages",
                                        .code = "context_length_exceeded"});
    }
    if (!exchange.Admit({.prompt_tokens = 10})) {
      return api::Completion{};
    }
    if (text == "late") {
      return std::unexpected(api::Error{
          .status = 500, .type = "server_error", .message = "failed", .param = {}, .code = {}});
    }
    if (text == "block") {
      started.store(true);
      while (!release.load()) {
        if (!exchange.Continue()) {
          cancelled.store(true);
          return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = false};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
    (void)exchange.Reasoning("hmm");
    const std::size_t half = text.size() / 2;
    const bool go = exchange.Content(text.substr(0, half)) && exchange.Content(text.substr(half));
    return api::Completion{.completion_tokens = go ? 4U : 3U, .cached_tokens = 2, .stopped = go};
  }

  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
  std::atomic<bool> cancelled{false};
};

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override { Start({}); }

  void Start(const api::ServerOptions& overrides) {
    api::ServerOptions options = overrides;
    options.bind = {.address = "127.0.0.1", .ipv6 = false, .port = 0};
    server_.emplace(backend_, options);
    auto port = server_->Listen();
    ASSERT_TRUE(port.has_value()) << port.error();
    port_ = *port;
    wake_ = ::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    thread_ = std::jthread([this] {
      result_ = server_->Run(wake_, [this] {
        std::uint64_t n = 0;
        (void)!::read(wake_, &n, sizeof n);
        return true;
      });
    });
  }

  void TearDown() override { Stop(); }

  void Stop() {
    if (thread_.joinable()) {
      backend_.release.store(true);
      const std::uint64_t one = 1;
      (void)!::write(wake_, &one, sizeof one);
      thread_.join();
      (void)::close(wake_);
    }
  }

  // A connection that has sent `bytes`.
  int Connect(std::string_view bytes) const {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(port_);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
    EXPECT_EQ(::connect(fd, reinterpret_cast<const sockaddr*>(&to), sizeof to), 0);
    timeval tv{.tv_sec = 20, .tv_usec = 0};
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, bytes));
    return fd;
  }

  // Everything the server sends until it closes.
  static std::string ReadAll(int fd) {
    std::string out;
    std::array<char, 4096> buf{};
    for (;;) {
      const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
      if (n <= 0) {
        break;
      }
      out.append(buf.data(), static_cast<std::size_t>(n));
    }
    (void)::close(fd);
    return out;
  }

  std::string Exchange(std::string_view bytes) const { return ReadAll(Connect(bytes)); }

  static std::string Post(std::string_view body, std::string_view extra = "") {
    return std::format(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: "
        "application/json\r\n{}Content-Length: {}\r\n\r\n{}",
        extra, body.size(), body);
  }

  static std::string Chat(std::string_view text, std::string_view fields = "",
                          std::string_view model = "alpha") {
    return std::format(R"({{"model":"{}","messages":[{{"role":"user","content":"{}"}}]{}}})", model,
                       text, fields);
  }

  static std::string BodyOf(const std::string& response) {
    const std::size_t at = response.find("\r\n\r\n");
    return at == std::string::npos ? std::string() : response.substr(at + 4);
  }

  FakeBackend backend_;
  std::optional<api::Server> server_;
  std::uint16_t port_ = 0;
  int wake_ = -1;
  std::expected<void, std::string> result_;
  std::jthread thread_;
};

TEST_F(ServerTest, AnswersAChatCompletion) {
  const std::string response = Exchange(Post(Chat("Hello world")));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(response, HasSubstr("Content-Type: application/json\r\n"));
  EXPECT_THAT(response, HasSubstr("Connection: close\r\n"));
  auto doc = json::Parse(BodyOf(response));
  ASSERT_TRUE(doc.has_value()) << response;
  const json::Value root = doc->root();
  EXPECT_THAT(std::string(root.find("id")->string()), StartsWith("chatcmpl-"));
  EXPECT_EQ(root.find("model")->string(), "alpha");
  const json::Value message = root.find("choices")->at(0).find("message").value();
  EXPECT_EQ(message.find("content")->string(), "Hello world");
  EXPECT_EQ(message.find("reasoning")->string(), "hmm");
  EXPECT_EQ(root.find("choices")->at(0).find("finish_reason")->string(), "stop");
  EXPECT_EQ(root.find("usage")->find("prompt_tokens")->int64(), 10);
  EXPECT_EQ(root.find("usage")->find("completion_tokens")->int64(), 4);
}

TEST_F(ServerTest, StreamsChunksThenDone) {
  const std::string response = Exchange(
      Post(Chat("Hello world", R"(,"stream":true,"stream_options":{"include_usage":true})")));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(response, HasSubstr("Content-Type: text/event-stream\r\n"));
  const std::string body = BodyOf(response);
  EXPECT_THAT(body, StartsWith("data: {"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"role":"assistant","content":""})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"reasoning":"hmm"})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"content":"Hello"})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"content":" world"})"));
  EXPECT_THAT(body, HasSubstr(R"("finish_reason":"stop")"));
  EXPECT_THAT(body, HasSubstr(R"("choices":[],"usage":{"prompt_tokens":10)"));
  EXPECT_TRUE(body.ends_with("data: [DONE]\n\n")) << body;
}

TEST_F(ServerTest, StopStringsEndTheAnswer) {
  const std::string response = Exchange(Post(Chat("Hello world", R"(,"stop":"lo w")")));
  auto doc = json::Parse(BodyOf(response));
  ASSERT_TRUE(doc.has_value()) << response;
  const json::Value choice = doc->root().find("choices")->at(0);
  EXPECT_EQ(choice.find("message")->find("content")->string(), "Hel");
  EXPECT_EQ(choice.find("finish_reason")->string(), "stop");
}

TEST_F(ServerTest, ListsModels) {
  const std::string list = Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost:8114\r\n\r\n");
  EXPECT_THAT(list, StartsWith("HTTP/1.1 200 OK\r\n"));
  EXPECT_THAT(BodyOf(list), HasSubstr(R"({"object":"list","data":[{"id":"alpha")"));
  EXPECT_THAT(Exchange("GET /v1/models/image HTTP/1.1\r\nHost: [::1]\r\n\r\n"),
              HasSubstr(R"({"id":"image","object":"model")"));
  EXPECT_THAT(Exchange("GET /v1/models/beta HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 404 "), HasSubstr("model_not_found")));
}

TEST_F(ServerTest, RefusesBeforeAnyWork) {
  // Model problems: unknown, or not a chat model; and the backend's own
  // refusal before admission.
  EXPECT_THAT(Exchange(Post(Chat("x", "", "beta"))),
              AllOf(StartsWith("HTTP/1.1 404 "), HasSubstr(R"("code":"model_not_found")")));
  EXPECT_THAT(Exchange(Post(Chat("x", "", "image"))),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr(R"("code":"model_not_supported")")));
  EXPECT_THAT(Exchange(Post(Chat("fail"))),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr("context_length_exceeded")));
  EXPECT_THAT(Exchange(Post(Chat("x", R"(,"temperature":3)"))),
              AllOf(StartsWith("HTTP/1.1 400 "), HasSubstr(R"("param":"temperature")")));
  // Routes and methods.
  EXPECT_THAT(Exchange("GET /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 405 "), HasSubstr("Allow: POST\r\n")));
  EXPECT_THAT(Exchange("GET /v1/other HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 404 "));
  // Browser guards.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: evil.example\r\n\r\n"),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\n\r\n"), StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Origin: http://127.0.0.1:3000\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Sec-Fetch-Site: cross-site\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   "Content-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                                   Chat("x").size(), Chat("x"))),
              StartsWith("HTTP/1.1 415 "));
}

TEST_F(ServerTest, BoundsTheHttpRequest) {
  // The body's size, before it is read.
  EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                                   api::kMaxBodyBytes + 1)),
              StartsWith("HTTP/1.1 413 "));
  // The head's size and header count.
  EXPECT_THAT(Exchange(std::format("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX: {}\r\n\r\n",
                                   std::string(api::kMaxHeaderBytes, 'x'))),
              StartsWith("HTTP/1.1 413 "));
  std::string many = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n";
  for (std::size_t i = 0; i < api::kMaxHeaders; ++i) {
    many += std::format("X-{}: y\r\n", i);
  }
  EXPECT_THAT(Exchange(many + "\r\n"), StartsWith("HTTP/1.1 413 "));
  // The target's length.
  EXPECT_THAT(Exchange(std::format("GET /{} HTTP/1.1\r\nHost: localhost\r\n\r\n",
                                   std::string(api::kMaxTargetBytes, 'a'))),
              StartsWith("HTTP/1.1 414 "));
  // Framing: chunked bodies, a missing length, bare LF, folding, versions.
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       "Transfer-Encoding: chunked\r\n\r\n0\r\n\r\n"),
              StartsWith("HTTP/1.1 501 "));
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 411 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\nHost: localhost\n\n\r\n\r\n"),
              StartsWith("HTTP/1.1 400 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n folded\r\n\r\n"),
              StartsWith("HTTP/1.1 400 "));
  EXPECT_THAT(Exchange("GET /v1/models HTTP/2.0\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 505 "));
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Length: 1\r\nContent-Length: 2\r\n\r\nx"),
              StartsWith("HTTP/1.1 400 "));
  // Content-Length's forms: only decimal digits, at most 18 of them.
  for (const std::string_view bad :
       {"+5", "-1", "5, 5", "0x5", "5 5", "", "99999999999999999999"}) {
    EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                     "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                                     bad)),
                StartsWith("HTTP/1.1 400 "))
        << bad;
  }
  // A length and a chunked encoding together (smuggling's shape).
  EXPECT_THAT(Exchange("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"),
              StartsWith("HTTP/1.1 501 "));
  // Two Hosts, and a control byte in a value.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nHost: evil.example\r\n\r\n"),
              StartsWith("HTTP/1.1 400 "));
  using namespace std::string_view_literals;
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nX: a\0b\r\n\r\n"sv),
              StartsWith("HTTP/1.1 400 "));
  // Bytes after the body (a pipelined request) are not served: one
  // response, then the connection closes.
  const std::string first = Chat("One");
  const std::string two = Exchange(Post(first) + Post(Chat("Two")));
  EXPECT_THAT(two, StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(two, HasSubstr("One"));
  EXPECT_THAT(two, Not(HasSubstr("Two")));
  // Expect: 100-continue is answered before the body.
  const std::string body = Chat("Hi there");
  const int fd = Connect(std::format(
      "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
      "Expect: 100-continue\r\nContent-Length: {}\r\n\r\n",
      body.size()));
  std::array<char, 64> interim{};
  const ssize_t got = ::recv(fd, interim.data(), interim.size(), 0);
  ASSERT_GT(got, 0);
  EXPECT_EQ(std::string_view(interim.data(), static_cast<std::size_t>(got)),
            "HTTP/1.1 100 Continue\r\n\r\n");
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, body));
  EXPECT_THAT(ReadAll(fd), StartsWith("HTTP/1.1 200 OK"));
}

TEST_F(ServerTest, TimesOutASlowHead) {
  Stop();
  api::ServerOptions options;
  options.head_timeout = std::chrono::milliseconds(200);
  options.body_timeout = std::chrono::milliseconds(400);
  Start(options);
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n"),
              StartsWith("HTTP/1.1 408 "));
  EXPECT_THAT(
      Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           "Content-Type: application/json\r\nContent-Length: 50\r\n\r\n{{")),
      StartsWith("HTTP/1.1 408 "));
  // A connection that sends nothing is closed without a response.
  EXPECT_EQ(Exchange(""), "");
}

TEST_F(ServerTest, QueuesThenRefusesConcurrentRequests) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.max_queued = 1;
  Start(options);
  const int running = Connect(Post(Chat("block")));
  while (!backend_.started.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const int queued = Connect(Post(Chat("second")));
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // the acceptor queues it
  const std::string refused = Exchange(Post(Chat("third")));
  EXPECT_THAT(refused, StartsWith("HTTP/1.1 429 "));
  EXPECT_THAT(refused, HasSubstr("Retry-After: 10\r\n"));
  EXPECT_THAT(refused, HasSubstr("rate_limit_error"));
  backend_.release.store(true);
  EXPECT_THAT(ReadAll(running), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(ReadAll(queued), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("second")));
}

TEST_F(ServerTest, AFailureAfterTheHeadersEndsTheStreamWithoutDone) {
  const std::string response = Exchange(Post(Chat("late", R"(,"stream":true)")));
  EXPECT_THAT(response, StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(response, HasSubstr(R"(data: {"error":{"message":"failed")"));
  EXPECT_THAT(response, Not(HasSubstr("[DONE]")));
  EXPECT_THAT(Exchange(Post(Chat("late"))), StartsWith("HTTP/1.1 500 "));
}

TEST_F(ServerTest, AClientThatLeavesCancelsItsGeneration) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  while (!backend_.started.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  (void)::close(fd);
  for (int i = 0; i < 400 && !backend_.cancelled.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.cancelled.load());
  // The server goes on serving.
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));
}

TEST_F(ServerTest, StoppingEndsARunningRequest) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  while (!backend_.started.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  const std::uint64_t one = 1;
  (void)!::write(wake_, &one, sizeof one);
  EXPECT_THAT(ReadAll(fd), StartsWith("HTTP/1.1 503 "));
  thread_.join();
  (void)::close(wake_);
  EXPECT_TRUE(result_.has_value());
}

}  // namespace
