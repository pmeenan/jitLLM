// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The chat route (D-097 as amended 2026-09-28; runtime/api.h, http.h,
// binding.h, api_server.h): request parsing and every intake bound, the
// unknown fields' names, the output's stop strings and reasoning split,
// the JSON shapes, the bind resolution over fake interface lists and the
// Host and Origin guard, and the server end to end over real loopback
// sockets with a fake backend: routes, browser guards, HTTP bounds,
// timeouts, keep-alive, pipelining, idle connections, the queue,
// keepalive comments, slow clients, streaming and errors after the
// headers.

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

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "base/json.h"
#include "config/node_config.h"
#include "platform/interfaces.h"
#include "runtime/api_server.h"
#include "runtime/binding.h"
#include "runtime/http.h"

namespace {

namespace api = jitllm::runtime::api;
namespace json = jitllm::base::json;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
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
  // Known fields that ask for what the route does not do are refused, and
  // OpenRouter's transforms and plugins whatever their value (D-046).
  EXPECT_THAT(ErrorOf(WithField(R"("transforms":["middle-out"])")), HasSubstr("[transforms]"));
  EXPECT_THAT(ErrorOf(WithField(R"("transforms":null)")), HasSubstr("[transforms]"));
  EXPECT_THAT(ErrorOf(WithField(R"("plugins":[])")), HasSubstr("[plugins]"));
  EXPECT_THAT(ErrorOf(WithField(R"("n":2)")), HasSubstr("[n]"));
  EXPECT_THAT(ErrorOf(WithField(R"("logprobs":true)")), HasSubstr("[logprobs]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_logprobs":2)")), HasSubstr("[top_logprobs]"));
  EXPECT_THAT(ErrorOf(WithField(R"("presence_penalty":0.5)")), HasSubstr("[presence_penalty]"));
  EXPECT_THAT(ErrorOf(WithField(R"("repetition_penalty":1.1)")), HasSubstr("[repetition_penalty]"));
  EXPECT_THAT(ErrorOf(WithField(R"("tools":[{"type":"function"}])")), HasSubstr("[tools]"));
  EXPECT_THAT(ErrorOf(WithField(R"("tool_choice":"required")")), HasSubstr("[tool_choice]"));
  EXPECT_THAT(ErrorOf(WithField(R"("response_format":{"type":"json_object"})")),
              HasSubstr("[response_format]"));
  EXPECT_THAT(ErrorOf(WithField(R"("logit_bias":{"1":2})")), HasSubstr("[logit_bias]"));
  EXPECT_THAT(ErrorOf(WithField(R"("modalities":["text","audio"])")), HasSubstr("[modalities]"));
  EXPECT_THAT(ErrorOf(WithField(R"("audio":{"voice":"x"})")), HasSubstr("[audio]"));
  EXPECT_THAT(ErrorOf(WithField(R"("store":true)")), HasSubstr("[store]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stream":"yes")")), HasSubstr("[stream]"));
  EXPECT_THAT(ErrorOf(WithField(R"("stream_options":{"include_usage":1})")),
              HasSubstr("[stream_options.include_usage]"));
  EXPECT_THAT(ErrorOf(WithField(R"("max_tokens":4,"max_completion_tokens":5)")),
              HasSubstr("[max_completion_tokens]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_k":-2)")), HasSubstr("[top_k]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_k":2147483648)")), HasSubstr("[top_k]"));
  EXPECT_THAT(ErrorOf(WithField(R"("top_k":1.5)")), HasSubstr("[top_k]"));
  EXPECT_THAT(ErrorOf(WithField(R"("min_p":1.01)")), HasSubstr("[min_p]"));
  EXPECT_THAT(ErrorOf(WithField(R"("min_p":-0.1)")), HasSubstr("[min_p]"));
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
  EXPECT_THAT(
      ErrorOf(R"({"model":"m","messages":[{"role":"assistant","content":"x","tool_calls":[{}]}]})"),
      HasSubstr("tool calls are not supported"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[
                 {"role":"assistant","content":"x","function_call":{"name":"f"}},
                 {"role":"user","content":"y"}]})"),
              HasSubstr("[messages[0].function_call]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[
                 {"role":"assistant","content":"x","audio":{"id":"a"}},
                 {"role":"user","content":"y"}]})"),
              HasSubstr("[messages[0].audio]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[
                 {"role":"assistant","content":null,"refusal":"no"},
                 {"role":"user","content":"y"}]})"),
              HasSubstr("[messages[0].refusal]"));
  EXPECT_THAT(ErrorOf(R"({"model":"m","messages":[{"role":"user","content":"a"},
                                                  {"role":"assistant","content":"b"}]})"),
              HasSubstr("the last message must be the user's"));
}

// Unknown fields are ignored and their names (never their values) kept,
// each once, with their place; known ones are not.
TEST(ChatRequest, IgnoresUnknownFieldsByName) {
  auto r = Parse(R"({"model":"m","frobnicate":{"deep":[1,2]},"zeta":null,"reasoning_effort":"low",
      "messages":[
        {"role":"system","content":"s","x_note":1},
        {"role":"assistant","content":"a","function_call":null,"audio":null,"refusal":null,
         "tool_calls":[],"annotations":[],"x_note":2},
        {"role":"user","content":[{"type":"text","text":"u","x_part":true}],"x_note":3}],
      "stream_options":{"include_usage":true,"x_opt":"v"},
      "top_k":40,"min_p":0.05,"repetition_penalty":1,"user":"u","frobnicate2":1})");
  ASSERT_TRUE(r.has_value()) << r.error().message;
  EXPECT_THAT(r->ignored,
              ::testing::UnorderedElementsAre("frobnicate", "zeta", "reasoning_effort",
                                              "messages[].x_note", "messages[].content[].x_part",
                                              "stream_options.x_opt", "frobnicate2"));
  EXPECT_EQ(r->top_k, 40U);
  EXPECT_FLOAT_EQ(static_cast<float>(r->min_p), 0.05F);
  EXPECT_TRUE(r->include_usage);
  auto off = Parse(WithField(R"("top_k":-1,"min_p":0)"));
  ASSERT_TRUE(off.has_value());
  EXPECT_EQ(off->top_k, 0U);
  EXPECT_TRUE(Parse(kMinimal)->ignored.empty());
  // A long name is cut at a character; a request's names are bounded.
  const std::string long_name = std::string(62, 'a') + "\xC3\xA9\xC3\xA9";
  auto cut = Parse(WithField(std::format(R"("{}":1)", long_name)));
  ASSERT_TRUE(cut.has_value());
  EXPECT_THAT(cut->ignored, ElementsAre(std::string(62, 'a') + "\xC3\xA9"));
  std::string many;
  for (std::size_t i = 0; i < api::kMaxIgnoredPerRequest + 10; ++i) {
    many += std::format(R"({}"u{}":0)", i == 0 ? "" : ",", i);
  }
  auto bounded = Parse(WithField(many));
  ASSERT_TRUE(bounded.has_value());
  EXPECT_EQ(bounded->ignored.size(), api::kMaxIgnoredPerRequest);
}

TEST(IgnoredFields, CountsNamesUpToItsSize) {
  api::IgnoredFields table;
  EXPECT_THAT(table.Record({"a", "b"}, 100), ElementsAre("a", "b"));
  EXPECT_THAT(table.Record({"a", "c"}, 200), ElementsAre("c"));
  auto doc = json::Parse(table.Json());
  ASSERT_TRUE(doc.has_value()) << table.Json();
  const json::Value a = doc->root().find("data")->at(0);
  EXPECT_EQ(a.find("name")->string(), "a");
  EXPECT_EQ(a.find("count")->int64(), 2);
  EXPECT_EQ(a.find("first_seen")->int64(), 100);
  EXPECT_EQ(a.find("last_seen")->int64(), 200);
  for (std::size_t i = 0; i < api::IgnoredFields::kMaxNames + 5; ++i) {
    (void)table.Record({std::format("n{}", i)}, 300);
  }
  auto full = json::Parse(table.Json());
  ASSERT_TRUE(full.has_value());
  EXPECT_EQ(full->root().find("data")->size(), api::IgnoredFields::kMaxNames);
  EXPECT_EQ(full->root().find("unrecorded")->int64(), 8);  // 3 names were in before
  EXPECT_THAT(table.Record({"a"}, 400), IsEmpty());        // a known name is still counted
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
  for (const std::string_view bad :
       {"", "example.com", "evil.localhost", "10.0.0.1:8114", "[::2]:8114", "[::1", "[::1]x",
        "localhost.:80", "localhost:", "localhost:80:80"}) {
    EXPECT_FALSE(api::IsLoopbackHost(bad)) << bad;
  }
}

// ---------------------------------------------------------------- binding

jitllm::platform::InterfaceAddress Address(std::string_view interface, std::string_view text,
                                           bool loopback = false) {
  jitllm::platform::InterfaceAddress a;
  a.interface = std::string(interface);
  a.ipv6 = text.find(':') != std::string_view::npos;
  const std::string owned(text);
  EXPECT_EQ(::inet_pton(a.ipv6 ? AF_INET6 : AF_INET, owned.c_str(), a.bytes.data()), 1) << text;
  a.up = true;
  a.loopback = loopback;
  return a;
}

// spark-b's interfaces as `ip addr` showed them on 2026-09-28, shortened.
std::vector<jitllm::platform::InterfaceAddress> SparkAddresses(bool tailscale = true) {
  std::vector<jitllm::platform::InterfaceAddress> list = {
      Address("lo", "127.0.0.1", true),
      Address("lo", "::1", true),
      Address("enP7s7", "192.168.0.101"),
      Address("enP7s7", "fdd0:5b0c:6852:482c::1"),
      Address("enP7s7", "fe80::1864:c1f4:4501:7925"),
      Address("docker0", "172.17.0.1")};
  if (tailscale) {
    list.push_back(Address("tailscale0", "100.114.118.63"));
    list.push_back(Address("tailscale0", "fd7a:115c:a1e0::2e31:7640"));
    list.push_back(Address("tailscale0", "fe80::a007:23ed:e98f:b08e"));
  }
  return list;
}

// A resolver that knows the tailnet's names, as MagicDNS answers them.
std::optional<std::string> SparkReverse(const jitllm::platform::InterfaceAddress& a) {
  const std::string text = a.Text();
  if (text == "100.114.118.63" || text == "fd7a:115c:a1e0::2e31:7640") {
    return "spark-b.coati-puffin.ts.net";
  }
  if (text == "192.168.0.101") {
    return "spark-56f5.lan";
  }
  return std::nullopt;
}

std::vector<std::string> Endpoints(const api::Listening& l) {
  std::vector<std::string> out;
  for (const auto& e : l.endpoints) {
    out.push_back(e.ipv6 ? std::format("[{}]:{}", e.address, e.port)
                         : std::format("{}:{}", e.address, e.port));
  }
  return out;
}

jitllm::config::ClientConfig Bind(std::vector<std::string_view> entries) {
  jitllm::config::ClientConfig client;
  client.bind.clear();
  for (const std::string_view e : entries) {
    auto entry = jitllm::config::ParseBindEntry(e);
    EXPECT_TRUE(entry.has_value()) << e;
    client.bind.push_back(entry.value_or(jitllm::config::BindEntry{}));
  }
  return client;
}

TEST(Binding, TheDefaultServesLoopbackAndTheTailnet) {
  const api::Listening l = api::ResolveListening(jitllm::config::ClientConfig{}, SparkAddresses(),
                                                 "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l), ElementsAre("127.0.0.1:8114", "[::1]:8114", "100.114.118.63:8114",
                                        "[fd7a:115c:a1e0::2e31:7640]:8114"));
  EXPECT_THAT(l.unauthenticated, IsEmpty());
  EXPECT_THAT(l.notes, ElementsAre(HasSubstr("spark-b.coati-puffin.ts.net")));
  api::HostGuard hosts = l.hosts;
  for (const std::string_view ok :
       {"spark-b.coati-puffin.ts.net", "spark-b.coati-puffin.ts.net:8114", "Spark-B:8114",
        "spark-56f5", "100.114.118.63:8114", "[fd7a:115c:a1e0::2e31:7640]:8114", "localhost"}) {
    EXPECT_TRUE(hosts.AllowsHost(ok)) << ok;
  }
  for (const std::string_view bad :
       {"evil.coati-puffin.ts.net", "spark-b.coati-puffin.ts.net.", "coati-puffin.ts.net",
        "192.168.0.101:8114", "spark-56f5.lan", "100.114.118.64", "evil.example", "[fe80::1]"}) {
    EXPECT_FALSE(hosts.AllowsHost(bad)) << bad;
  }
}

TEST(Binding, WithoutTailscaleItServesLoopbackOnly) {
  auto addresses = SparkAddresses(false);
  // No ::1 either: an IPv6-less host.
  std::erase_if(addresses, [](const auto& a) { return a.ipv6; });
  const api::Listening l =
      api::ResolveListening(jitllm::config::ClientConfig{}, addresses, "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l), ElementsAre("127.0.0.1:8114"));
  EXPECT_THAT(l.notes, ElementsAre(HasSubstr("no tailnet interface found")));
  EXPECT_THAT(l.unauthenticated, IsEmpty());
}

// 100.64.0.0/10 is also carriers' shared address space: only Tailscale's
// interface makes it the tailnet.
TEST(Binding, ACarriersSharedAddressIsNotTheTailnet) {
  std::vector<jitllm::platform::InterfaceAddress> addresses = {Address("lo", "127.0.0.1", true),
                                                               Address("wwan0", "100.72.1.2")};
  api::Listening l = api::ResolveListening(Bind({"tailscale"}), addresses, "h", nullptr);
  EXPECT_THAT(Endpoints(l), IsEmpty());
  // Tailscale's IPv6 range marks a renamed tunnel as the tailnet.
  addresses.push_back(Address("ts9", "100.100.1.2"));
  addresses.push_back(Address("ts9", "fd7a:115c:a1e0::9"));
  l = api::ResolveListening(Bind({"tailscale"}), addresses, "h", nullptr);
  EXPECT_THAT(Endpoints(l), ElementsAre("100.100.1.2:8114", "[fd7a:115c:a1e0::9]:8114"));
  EXPECT_THAT(l.notes, ElementsAre(HasSubstr("MagicDNS name is unknown")));
}

TEST(Binding, ExplicitAddressesAreUnauthenticatedUnlessLoopbackOrTailnet) {
  // A wildcard: every address of its family passes the Host check, it
  // covers loopback's on the same port, and the start log says it is
  // served without authentication.
  api::Listening l = api::ResolveListening(Bind({"loopback", "0.0.0.0", "tailscale"}),
                                           SparkAddresses(), "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l),
              ElementsAre("[::1]:8114", "0.0.0.0:8114", "[fd7a:115c:a1e0::2e31:7640]:8114"));
  ASSERT_THAT(l.unauthenticated, ElementsAre(HasSubstr("serving without authentication")));
  EXPECT_THAT(l.unauthenticated[0], HasSubstr("0.0.0.0:8114"));
  EXPECT_TRUE(l.hosts.AllowsHost("192.168.0.101:8114"));
  EXPECT_TRUE(l.hosts.AllowsHost("172.17.0.1"));
  EXPECT_TRUE(l.hosts.AllowsHost("spark-56f5.lan"));
  EXPECT_TRUE(l.hosts.AllowsHost("spark-b"));
  EXPECT_FALSE(l.hosts.AllowsHost("[fdd0:5b0c:6852:482c::1]"));  // IPv6 is not wildcarded
  // A LAN address is said to be unauthenticated; a tailnet or loopback
  // address is not; a port of its own is kept, and [client] port fills in
  // the rest.
  jitllm::config::ClientConfig client =
      Bind({"192.168.0.101:9000", "100.114.118.63", "127.0.0.2:9001", "[::]:9002"});
  client.port = 9100;
  l = api::ResolveListening(client, SparkAddresses(), "spark-56f5", SparkReverse);
  EXPECT_THAT(Endpoints(l), ElementsAre("192.168.0.101:9000", "100.114.118.63:9100",
                                        "127.0.0.2:9001", "[::]:9002"));
  EXPECT_THAT(
      l.unauthenticated,
      ElementsAre(HasSubstr("on 192.168.0.101:9000, which is neither loopback nor the tailnet"),
                  HasSubstr("on [::]:9002 (every IPv6 interface)")));
  EXPECT_TRUE(l.hosts.AllowsHost("spark-56f5.lan:9000"));
  EXPECT_TRUE(l.hosts.AllowsHost("[fdd0:5b0c:6852:482c::1]:9002"));
  EXPECT_FALSE(l.hosts.AllowsHost("[fe80::1864:c1f4:4501:7925]"));  // link-local
  EXPECT_TRUE(l.hosts.AllowsHost("spark-b.coati-puffin.ts.net"));   // the tailnet address's
  // A tailnet address is named only under ts.net, bound by "tailscale" or
  // by its address: a resolver's other name for it is not trusted.
  const auto spoofed = [](const jitllm::platform::InterfaceAddress& a) {
    return api::InTailnetRange(a) ? std::optional<std::string>("evil.example") : SparkReverse(a);
  };
  for (const std::string_view entry : {"100.114.118.63", "tailscale"}) {
    l = api::ResolveListening(Bind({entry}), SparkAddresses(), "spark-56f5", spoofed);
    EXPECT_FALSE(l.hosts.AllowsHost("evil.example")) << entry;
    EXPECT_FALSE(l.hosts.AllowsHost("evil")) << entry;
    EXPECT_TRUE(l.hosts.AllowsHost("100.114.118.63:8114")) << entry;
  }
  // Reverse lookups are bounded.
  std::size_t lookups = 0;
  std::vector<jitllm::platform::InterfaceAddress> many = {Address("lo", "127.0.0.1", true)};
  for (int i = 1; i <= 40; ++i) {
    many.push_back(Address("eth0", std::format("10.0.0.{}", i)));
  }
  (void)api::ResolveListening(Bind({"0.0.0.0"}), many, "h",
                              [&](const jitllm::platform::InterfaceAddress&) {
                                ++lookups;
                                return std::optional<std::string>();
                              });
  EXPECT_EQ(lookups, api::kMaxReverseLookups);
}

TEST(Binding, OriginsMustBeTheNodeOnItsPort) {
  api::HostGuard hosts;
  hosts.AddName("spark-b.coati-puffin.ts.net");
  hosts.AddPort(8114);
  EXPECT_TRUE(hosts.AllowsOrigin("http://spark-b.coati-puffin.ts.net:8114"));
  EXPECT_TRUE(hosts.AllowsOrigin("HTTP://localhost:8114"));
  EXPECT_TRUE(hosts.AllowsOrigin("http://[::1]:8114"));
  // A scheme alone ("http", "https") once read past its end: the process
  // aborted.
  for (const std::string_view bad :
       {"http://spark-b.coati-puffin.ts.net", "https://spark-b.coati-puffin.ts.net:3000",
        "http://evil.example:8114", "null", "file://", "http://localhost:8114/",
        "ftp://localhost:8114", "http://127.0.0.1:3000", "http", "https", "HTTP", "http:", "http:/",
        "http://", "", "://localhost:8114"}) {
    EXPECT_FALSE(hosts.AllowsOrigin(bad)) << bad;
  }
  hosts.AddPort(443);
  EXPECT_TRUE(hosts.AllowsOrigin("https://spark-b.coati-puffin.ts.net"));
}

// ---------------------------------------------------------------- the server

// Runs requests as the test directs: "block" waits for release; "fail"
// fails before admission, "late" after it; "flood" streams until told to
// stop; "big" answers 16 MiB at once; otherwise reasoning, then the last
// message's content echoed in two pieces.
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
    if (text == "flood") {
      started.store(true);
      const std::string piece(1024, 'x');
      for (int i = 0; i < 200000; ++i) {
        if (!exchange.Content(piece)) {
          flooded.store(true);
          break;
        }
      }
      return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = false};
    }
    if (text == "big") {
      (void)exchange.Content(std::string(std::size_t{16} << 20U, 'y'));
      return api::Completion{.completion_tokens = 1, .cached_tokens = 0, .stopped = true};
    }
    (void)exchange.Reasoning("hmm");
    const std::size_t half = text.size() / 2;
    const bool go = exchange.Content(text.substr(0, half)) && exchange.Content(text.substr(half));
    return api::Completion{.completion_tokens = go ? 4U : 3U, .cached_tokens = 2, .stopped = go};
  }

  std::atomic<bool> started{false};
  std::atomic<bool> release{false};
  std::atomic<bool> cancelled{false};
  std::atomic<bool> flooded{false};
};

// A client socket's reads, with a timeout: false at the end or on none.
bool Recv(int fd, std::string& into) {
  std::array<char, 16384> buf{};
  const ssize_t n = ::recv(fd, buf.data(), buf.size(), 0);
  if (n <= 0) {
    return false;
  }
  into.append(buf.data(), static_cast<std::size_t>(n));
  return true;
}

std::size_t Number(std::string_view text, int base) {
  std::size_t n = 0;
  (void)std::from_chars(text.data(), text.data() + text.size(), n, base);
  return n;
}

// One response from a connection that may persist: its head and its body,
// decoded (by Content-Length, chunks, or to the close); what follows stays
// in `pending`. What a connection sent before closing, if no head.
std::string ReadResponse(int fd, std::string& pending) {
  std::size_t head_end = 0;
  while ((head_end = pending.find("\r\n\r\n")) == std::string::npos) {
    if (!Recv(fd, pending)) {
      return std::exchange(pending, {});
    }
  }
  head_end += 4;
  const std::string head = pending.substr(0, head_end);
  std::string body;
  std::size_t used = 0;
  if (const std::size_t field = head.find("Content-Length: "); field != std::string::npos) {
    const std::size_t length = Number(std::string_view(head).substr(field + 16), 10);
    while (pending.size() < head_end + length && Recv(fd, pending)) {
    }
    body = pending.substr(head_end, length);
    used = std::min(pending.size(), head_end + length);
  } else if (head.find("Transfer-Encoding: chunked\r\n") != std::string::npos) {
    std::size_t at = head_end;
    for (;;) {
      std::size_t eol = 0;
      while ((eol = pending.find("\r\n", at)) == std::string::npos) {
        if (!Recv(fd, pending)) {
          pending.clear();
          return head + body;
        }
      }
      const std::size_t size = Number(std::string_view(pending).substr(at, eol - at), 16);
      if (size == 0) {
        while (pending.size() < eol + 4 && Recv(fd, pending)) {
        }
        used = std::min(pending.size(), eol + 4);
        break;
      }
      while (pending.size() < eol + 2 + size + 2 && Recv(fd, pending)) {
      }
      body += pending.substr(eol + 2, size);
      at = eol + 2 + size + 2;
      if (pending.size() < at) {
        pending.clear();
        return head + body;
      }
    }
  } else {
    while (Recv(fd, pending)) {
    }
    body = pending.substr(head_end);
    used = pending.size();
  }
  pending.erase(0, used);
  return head + body;
}

// Reads until `needle` has arrived (or the connection ends); all of it.
std::string ReadUntil(int fd, std::string& pending, std::string_view needle) {
  while (pending.find(needle) == std::string::npos && Recv(fd, pending)) {
  }
  return pending;
}

// Whether the server has closed the connection (after what it sent).
bool Closed(int fd) {
  std::string ignored;
  while (Recv(fd, ignored)) {
  }
  const ssize_t n = ::recv(fd, ignored.data(), 0, MSG_DONTWAIT);
  return n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK);
}

class ServerTest : public ::testing::Test {
 protected:
  void SetUp() override { Start({}); }

  void Start(const api::ServerOptions& overrides) {
    api::ServerOptions options = overrides;
    options.bind = {{.address = "127.0.0.1", .ipv6 = false, .port = 0}};
    server_.emplace(backend_, options);
    auto ports = server_->Listen();
    ASSERT_TRUE(ports.has_value()) << ports.error();
    port_ = ports->front();
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

  // A connection, with a receive timeout.
  int Open(int receive_buffer = 0) const {
    const int fd = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (receive_buffer > 0) {
      (void)::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &receive_buffer, sizeof receive_buffer);
    }
    sockaddr_in to{};
    to.sin_family = AF_INET;
    to.sin_port = htons(port_);
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast): the sockets API
    EXPECT_EQ(::connect(fd, reinterpret_cast<const sockaddr*>(&to), sizeof to), 0);
    timeval tv{.tv_sec = 20, .tv_usec = 0};
    (void)::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return fd;
  }

  // A connection that has sent `bytes`.
  int Connect(std::string_view bytes) const {
    const int fd = Open();
    EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, bytes));
    return fd;
  }

  // One request's response on a connection of its own.
  std::string Exchange(std::string_view bytes) const {
    const int fd = Connect(bytes);
    std::string pending;
    std::string response = ReadResponse(fd, pending);
    (void)::close(fd);
    return response;
  }

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

  void WaitStarted() {
    for (int i = 0; i < 2000 && !backend_.started.load(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ASSERT_TRUE(backend_.started.load());
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
  EXPECT_THAT(response, HasSubstr("Connection: keep-alive\r\nKeep-Alive: timeout=60\r\n"));
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
  EXPECT_THAT(response, HasSubstr("Transfer-Encoding: chunked\r\n"));
  const std::string body = BodyOf(response);
  EXPECT_THAT(body, StartsWith("data: {"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"role":"assistant","content":""})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"reasoning":"hmm"})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"content":"Hello"})"));
  EXPECT_THAT(body, HasSubstr(R"("delta":{"content":" world"})"));
  EXPECT_THAT(body, HasSubstr(R"("finish_reason":"stop")"));
  EXPECT_THAT(body, HasSubstr(R"("choices":[],"usage":{"prompt_tokens":10)"));
  EXPECT_TRUE(body.ends_with("data: [DONE]\n\n")) << body;
  // HTTP/1.0 has no chunks: the stream ends with the connection.
  const std::string old = Exchange(
      std::format("POST /v1/chat/completions HTTP/1.0\r\nHost: localhost\r\nContent-Type: "
                  "application/json\r\nContent-Length: {}\r\n\r\n{}",
                  Chat("Hi", R"(,"stream":true)").size(), Chat("Hi", R"(,"stream":true)")));
  EXPECT_THAT(old, AllOf(HasSubstr("Connection: close\r\n"), Not(HasSubstr("chunked"))));
  EXPECT_TRUE(BodyOf(old).ends_with("data: [DONE]\n\n")) << old;
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
  EXPECT_THAT(Exchange(Post(Chat("x"), "Origin: null\r\n")), StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Sec-Fetch-Site: cross-site\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(Post(Chat("x"), "Sec-Fetch-Site: same-site\r\n")),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   "Content-Type: text/plain\r\nContent-Length: {}\r\n\r\n{}",
                                   Chat("x").size(), Chat("x"))),
              StartsWith("HTTP/1.1 415 "));
  // The same origin (a page this listener served) passes.
  EXPECT_THAT(Exchange(Post(Chat("x"), std::format("Origin: http://127.0.0.1:{}\r\n", port_))),
              StartsWith("HTTP/1.1 200 "));
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
  // A refusal of the head closes the connection: where the next request
  // would begin is unknown.
  EXPECT_THAT(Exchange("GET /v1/models HTTP/2.0\r\nHost: localhost\r\n\r\n"),
              HasSubstr("Connection: close\r\n"));
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
  std::string pending;
  EXPECT_THAT(ReadResponse(fd, pending), StartsWith("HTTP/1.1 200 OK"));
  (void)::close(fd);
}

// Several requests on one connection, refusals included; Connection:
// close and HTTP/1.0 end it.
TEST_F(ServerTest, KeepsAConnectionAlive) {
  const int fd = Open();
  std::string pending;
  const auto send = [&](const std::string& bytes) {
    EXPECT_TRUE(jitllm::runtime::http::WriteAll(fd, bytes));
    return ReadResponse(fd, pending);
  };
  EXPECT_THAT(send(Post(Chat("One"))), AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("One")));
  const std::string stream = send(Post(Chat("Two", R"(,"stream":true)")));
  EXPECT_THAT(stream, AllOf(HasSubstr(R"("content":"wo")"), HasSubstr("Connection: keep-alive")));
  EXPECT_TRUE(stream.ends_with("data: [DONE]\n\n")) << stream;
  EXPECT_THAT(send("GET /v1/nothing HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 404 "), HasSubstr("Connection: keep-alive")));
  EXPECT_THAT(send(Post(Chat("x", R"(,"n":3)"))), StartsWith("HTTP/1.1 400 "));
  EXPECT_THAT(send("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  EXPECT_THAT(send(Post(Chat("Three"))), HasSubstr("Three"));
  EXPECT_THAT(send("GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("Connection: close\r\n")));
  EXPECT_TRUE(Closed(fd));
  (void)::close(fd);
  const int old = Open();
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(
      old, "GET /v1/models HTTP/1.0\r\nHost: localhost\r\nConnection: keep-alive\r\n\r\n"));
  std::string old_pending;
  EXPECT_THAT(ReadResponse(old, old_pending), HasSubstr("Connection: keep-alive\r\n"));
  EXPECT_TRUE(
      jitllm::runtime::http::WriteAll(old, "GET /v1/models HTTP/1.0\r\nHost: localhost\r\n\r\n"));
  EXPECT_THAT(ReadResponse(old, old_pending), HasSubstr("Connection: close\r\n"));
  EXPECT_TRUE(Closed(old));
  (void)::close(old);
}

// A request sent before the previous response ended is not served: that
// response says Connection: close, and the connection closes after it.
TEST_F(ServerTest, RefusesPipelinedRequests) {
  const int both = Connect(Post(Chat("One")) + Post(Chat("Two")));
  std::string pending;
  EXPECT_THAT(ReadResponse(both, pending), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("One"),
                                                 HasSubstr("Connection: close\r\n")));
  EXPECT_TRUE(Closed(both));
  EXPECT_THAT(pending, Not(HasSubstr("Two")));
  (void)::close(both);
  // The second arrives while the first runs.
  Stop();
  backend_.release.store(false);
  Start({});
  const int later = Connect(Post(Chat("block")));
  WaitStarted();
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(later, Post(Chat("Two"))));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  backend_.release.store(true);
  std::string rest;
  EXPECT_THAT(ReadResponse(later, rest),
              AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("Connection: close\r\n")));
  EXPECT_TRUE(Closed(later));
  EXPECT_THAT(rest, Not(HasSubstr("Two")));
  (void)::close(later);
}

TEST_F(ServerTest, TimesOutASlowHead) {
  Stop();
  api::ServerOptions options;
  options.head_timeout = std::chrono::milliseconds(200);
  options.body_timeout = std::chrono::milliseconds(400);
  options.idle_timeout = std::chrono::milliseconds(300);
  Start(options);
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n"),
              AllOf(StartsWith("HTTP/1.1 408 "), HasSubstr("Connection: close")));
  EXPECT_THAT(
      Exchange(std::format("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                           "Content-Type: application/json\r\nContent-Length: 50\r\n\r\n{{")),
      StartsWith("HTTP/1.1 408 "));
  // A connection that sends nothing is closed, once idle, without a
  // response; so is one kept alive after a response.
  EXPECT_EQ(Exchange(""), "");
  const int kept = Connect("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n");
  std::string pending;
  EXPECT_THAT(ReadResponse(kept, pending), StartsWith("HTTP/1.1 200 "));
  EXPECT_TRUE(Closed(kept));
  (void)::close(kept);
}

// Many idle connections are held; at the limit the oldest idle one makes
// room for a new one.
TEST_F(ServerTest, HoldsManyIdleConnections) {
  std::vector<int> idle;
  for (int i = 0; i < 200; ++i) {
    idle.push_back(Open());
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_THAT(Exchange(Post(Chat("still here"))), HasSubstr("still here"));
  // Each idle one still works.
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(
      idle[150], "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  std::string pending;
  EXPECT_THAT(ReadResponse(idle[150], pending), StartsWith("HTTP/1.1 200 "));
  for (const int fd : idle) {
    (void)::close(fd);
  }
  Stop();
  api::ServerOptions options;
  options.max_connections = 8;
  Start(options);
  idle.clear();
  for (int i = 0; i < 8; ++i) {
    idle.push_back(Open());
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  EXPECT_TRUE(Closed(idle[0]));
  for (const int fd : idle) {
    (void)::close(fd);
  }
}

TEST_F(ServerTest, QueuesThenRefusesConcurrentRequests) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.max_queued = 1;
  Start(options);
  const int running = Connect(Post(Chat("block")));
  WaitStarted();
  const int queued = Connect(Post(Chat("second")));
  std::this_thread::sleep_for(std::chrono::milliseconds(200));  // the I/O thread queues it
  const std::string refused = Exchange(Post(Chat("third")));
  EXPECT_THAT(refused, StartsWith("HTTP/1.1 429 "));
  EXPECT_THAT(refused, HasSubstr("Retry-After: 10\r\n"));
  EXPECT_THAT(refused, HasSubstr("x-should-retry: true\r\n"));
  EXPECT_THAT(refused, HasSubstr("rate_limit_error"));
  backend_.release.store(true);
  std::string a;
  std::string b;
  EXPECT_THAT(ReadResponse(running, a), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(ReadResponse(queued, b), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("second")));
  (void)::close(running);
  (void)::close(queued);
}

// A stream that waits its turn starts once it has waited a keepalive
// interval, and then hears `: keepalive` until its tokens come; so does
// one whose model is busy before its first token. A request whose queue
// wait runs out gets a 429, or, once its stream started, an in-stream
// error without [DONE].
TEST_F(ServerTest, KeepsStreamsAliveWhileTheyWait) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.keepalive = std::chrono::milliseconds(100);
  Start(options);
  const int running = Connect(Post(Chat("block")));
  WaitStarted();
  const int waiting = Connect(Post(Chat("second", R"(,"stream":true)")));
  std::string early;
  const std::string heard = ReadUntil(waiting, early, ": keepalive\n\n");
  EXPECT_THAT(heard, AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("text/event-stream"),
                           HasSubstr(R"("delta":{"role":"assistant","content":""})"),
                           HasSubstr(": keepalive\n\n"), Not(HasSubstr("second"))));
  backend_.release.store(true);
  std::string pending;
  EXPECT_THAT(ReadResponse(running, pending), StartsWith("HTTP/1.1 200 OK"));
  const std::string rest = ReadUntil(waiting, early, "data: [DONE]\n\n");
  EXPECT_THAT(rest, AllOf(HasSubstr(R"("delta":{"content":"sec"})"),
                          HasSubstr(R"("delta":{"content":"ond"})")));
  EXPECT_EQ(rest.find("\"role\":\"assistant\""), rest.rfind("\"role\":\"assistant\""));  // once
  // A running stream hears keepalives before its first token.
  backend_.started.store(false);
  backend_.release.store(false);
  const int busy = Connect(Post(Chat("block", R"(,"stream":true)")));
  std::string busy_pending;
  EXPECT_THAT(ReadUntil(busy, busy_pending, ": keepalive\n\n"), HasSubstr(": keepalive\n\n"));
  backend_.release.store(true);
  EXPECT_THAT(ReadUntil(busy, busy_pending, "data: [DONE]\n\n"), HasSubstr("data: [DONE]"));
  for (const int fd : {running, waiting, busy}) {
    (void)::close(fd);
  }
  // The queue wait runs out.
  Stop();
  backend_.started.store(false);
  backend_.release.store(false);
  options.queue_wait = std::chrono::milliseconds(400);
  Start(options);
  const int blocker = Connect(Post(Chat("block")));
  WaitStarted();
  const int streamed = Connect(Post(Chat("s", R"(,"stream":true)")));
  const int plain = Connect(Post(Chat("p")));
  std::string s;
  std::string p;
  const std::string expired = ReadResponse(streamed, s);
  EXPECT_THAT(expired, AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("rate_limit_error"),
                             Not(HasSubstr("[DONE]"))));
  EXPECT_THAT(ReadResponse(plain, p),
              AllOf(StartsWith("HTTP/1.1 429 "), HasSubstr("Retry-After: 10\r\n")));
  backend_.release.store(true);
  std::string b;
  EXPECT_THAT(ReadResponse(blocker, b), StartsWith("HTTP/1.1 200 OK"));
  for (const int fd : {blocker, streamed, plain}) {
    (void)::close(fd);
  }
}

// A client that stalls mid-request, or stops reading its response, holds
// up nobody else; one that stops reading a stream ends its generation.
TEST_F(ServerTest, SlowClientsDoNotHoldUpOthers) {
  Stop();
  api::ServerOptions options;
  options.max_unsent = std::size_t{64} << 10U;
  options.write_timeout = std::chrono::milliseconds(300);
  Start(options);
  const int stalled = Connect("POST /v1/chat/completions HTTP/1.1\r\nHost: loc");
  const auto t0 = std::chrono::steady_clock::now();
  EXPECT_THAT(Exchange(Post(Chat("quick"))), HasSubstr("quick"));
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(2));
  // A stream nobody reads.
  const int flood = Open(4096);
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(flood, Post(Chat("flood", R"(,"stream":true)"))));
  WaitStarted();
  EXPECT_THAT(Exchange("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 "));
  for (int i = 0; i < 2000 && !backend_.flooded.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.flooded.load());
  EXPECT_THAT(Exchange(Post(Chat("after"))), HasSubstr("after"));
  // A whole response nobody reads: the connection is dropped once its
  // writes stall, and the rest is never buffered further.
  const int big = Open(4096);
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(big, Post(Chat("big"))));
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_THAT(Exchange(Post(Chat("still"))), HasSubstr("still"));
  std::string got;
  while (Recv(big, got)) {
  }
  EXPECT_LT(got.size(), std::size_t{16} << 20U);
  for (const int fd : {stalled, flood, big}) {
    (void)::close(fd);
  }
}

// An unknown field is ignored and counted by name on a loopback-only
// route; its name is logged once, its value never.
TEST_F(ServerTest, IgnoresUnknownFieldsAndCountsThem) {
  Stop();
  std::FILE* log = std::tmpfile();
  ASSERT_NE(log, nullptr);
  api::ServerOptions options;
  options.log = log;
  Start(options);
  EXPECT_THAT(Exchange(Post(Chat("one", R"(,"frobnicate":"secret-value-1")"))),
              AllOf(StartsWith("HTTP/1.1 200 "), HasSubstr("one")));
  EXPECT_THAT(Exchange(Post(Chat("two", R"(,"frobnicate":{"a":"secret-value-2"})"))),
              StartsWith("HTTP/1.1 200 "));
  const std::string table =
      BodyOf(Exchange("GET /jitllm/v1/ignored-fields HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  auto doc = json::Parse(table);
  ASSERT_TRUE(doc.has_value()) << table;
  ASSERT_EQ(doc->root().find("data")->size(), 1U);
  EXPECT_EQ(doc->root().find("data")->at(0).find("name")->string(), "frobnicate");
  EXPECT_EQ(doc->root().find("data")->at(0).find("count")->int64(), 2);
  EXPECT_THAT(Exchange("POST /jitllm/v1/ignored-fields HTTP/1.1\r\nHost: localhost\r\n"
                       "Content-Length: 0\r\n\r\n"),
              StartsWith("HTTP/1.1 405 "));
  Stop();
  (void)std::fflush(log);
  std::rewind(log);
  std::string text;
  std::array<char, 4096> buf{};
  for (std::size_t n = 0; (n = std::fread(buf.data(), 1, buf.size(), log)) > 0;) {
    text.append(buf.data(), n);
  }
  (void)std::fclose(log);
  const std::string line = "an unknown request field is ignored: frobnicate";
  EXPECT_NE(text.find(line), std::string::npos) << text;
  EXPECT_EQ(text.find(line), text.rfind(line));
  EXPECT_THAT(text, Not(HasSubstr("secret-value")));
}

// The node's other names, as the resolution gives them (the tailnet's
// here), pass the Host and Origin guards; others do not.
TEST_F(ServerTest, AcceptsTheNodesNames) {
  Stop();
  api::ServerOptions options;
  options.hosts.AddName("spark-b.coati-puffin.ts.net");
  options.hosts.AddName("spark-b");
  std::array<std::uint8_t, 16> tailnet{100, 114, 118, 63};
  options.hosts.AddAddress(false, tailnet);
  Start(options);
  const auto get = [&](std::string_view host, std::string_view extra = "") {
    return Exchange(std::format("GET /v1/models HTTP/1.1\r\nHost: {}\r\n{}\r\n", host, extra));
  };
  for (const std::string_view ok :
       {"spark-b.coati-puffin.ts.net:8114", "SPARK-B", "100.114.118.63:8114", "localhost"}) {
    EXPECT_THAT(get(ok), StartsWith("HTTP/1.1 200 ")) << ok;
  }
  for (const std::string_view bad : {"evil.coati-puffin.ts.net", "spark-b.coati-puffin.ts.net.",
                                     "100.114.118.64", "[fd7a:115c:a1e0::1]"}) {
    EXPECT_THAT(get(bad), StartsWith("HTTP/1.1 403 ")) << bad;
  }
  EXPECT_THAT(
      get("spark-b", std::format("Origin: http://spark-b.coati-puffin.ts.net:{}\r\n", port_)),
      StartsWith("HTTP/1.1 200 "));
  EXPECT_THAT(get("spark-b", "Origin: http://spark-b.coati-puffin.ts.net:3000\r\n"),
              StartsWith("HTTP/1.1 403 "));
  EXPECT_THAT(get("spark-b", std::format("Origin: http://evil.example:{}\r\n", port_)),
              StartsWith("HTTP/1.1 403 "));
  // A bare scheme once aborted the process; it is a 403, and the route
  // goes on serving.
  for (const std::string_view origin : {"http", "https", "null"}) {
    EXPECT_THAT(get("localhost", std::format("Origin: {}\r\n", origin)),
                StartsWith("HTTP/1.1 403 "))
        << origin;
  }
  EXPECT_THAT(get("localhost"), StartsWith("HTTP/1.1 200 "));
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
  WaitStarted();
  (void)::close(fd);
  for (int i = 0; i < 400 && !backend_.cancelled.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.cancelled.load());
  // The server goes on serving.
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));
}

// A connection kept alive after its request keeps none of the request's
// or the response's bytes allocated: the accounting sees a body arrive,
// and then nothing large on connections that sit idle.
TEST_F(ServerTest, IdleConnectionsKeepNoLargeBuffers) {
  const std::string text(std::size_t{900} << 10U, 'a');  // under a message's 1 MiB
  const std::string whole = Post(Chat(text));
  const auto wait_for = [&](const auto& done) {
    for (int i = 0; i < 400 && !done(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return done();
  };
  // Half a body: the buffer the rest arrives in is held and counted.
  const int first = Connect(std::string_view(whole).substr(0, whole.size() / 2));
  EXPECT_TRUE(wait_for([&] { return server_->held_bytes() >= whole.size(); }))
      << server_->held_bytes();
  EXPECT_TRUE(
      jitllm::runtime::http::WriteAll(first, std::string_view(whole).substr(whole.size() / 2)));
  std::vector<int> kept{first};
  for (int i = 0; i < 8; ++i) {
    if (i > 0) {
      kept.push_back(Connect(whole));
    }
    std::string pending;
    const std::string response = ReadResponse(kept.back(), pending);
    EXPECT_THAT(response,
                AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("Connection: keep-alive")));
    EXPECT_GT(response.size(), text.size());  // the answer echoes it
  }
  // Nine requests of about 1 MiB each, answered at about 1 MiB each: the
  // connections, all still open, hold at most a small buffer each way.
  const std::size_t bound = kept.size() * 2 * api::kKeptBufferBytes;
  EXPECT_TRUE(wait_for([&] { return server_->held_bytes() <= bound; })) << server_->held_bytes();
  // Each still serves.
  EXPECT_TRUE(jitllm::runtime::http::WriteAll(
      kept[3], "GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"));
  std::string pending;
  EXPECT_THAT(ReadResponse(kept[3], pending), StartsWith("HTTP/1.1 200 "));
  for (const int fd : kept) {
    (void)::close(fd);
  }
}

// A client that shuts its sending side after a whole request still gets
// its response (after an interim 102 when it half-closed first, and only
// when not streaming), then the connection closes; the generation is not
// cancelled. A shutdown before the request is whole is a disconnect: no
// response.
TEST_F(ServerTest, AHalfClosedClientGetsItsResponse) {
  constexpr std::string_view kInterim = "HTTP/1.1 102 Processing\r\n\r\n";
  const auto half = [&](const std::string& bytes, bool strip = true) {
    const int fd = Connect(bytes);
    EXPECT_EQ(::shutdown(fd, SHUT_WR), 0);
    std::string all;
    while (Recv(fd, all)) {
    }
    (void)::close(fd);
    if (strip && all.starts_with(kInterim)) {
      all.erase(0, kInterim.size());  // sent when the shutdown was seen first
    }
    return all;
  };
  // The reviewer's case: a whole POST, then SHUT_WR, then read.
  const std::string plain = half(Post(Chat("Hello world")));
  EXPECT_THAT(plain, AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr("Connection: close\r\n"),
                           HasSubstr(R"("content":"Hello world")")));
  // A stream is never sent a 1xx: its probe is its own start.
  const std::string streamed = half(Post(Chat("Hello world", R"(,"stream":true)")), false);
  EXPECT_THAT(streamed, AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr(R"("content":"Hello")"),
                              HasSubstr("data: [DONE]\n\n")));
  EXPECT_TRUE(streamed.ends_with("0\r\n\r\n")) << streamed;
  EXPECT_THAT(half("GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
              StartsWith("HTTP/1.1 200 OK\r\n"));
  // Before the request is whole: nobody to answer.
  EXPECT_EQ(half("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: "
                 "application/json\r\nContent-Length: 50\r\n\r\n{"),
            "");
  EXPECT_EQ(half("GET /v1/models HTTP/1.1\r\nHost: loc"), "");
  // A generation under way when the client half-closes: the interim
  // response comes at once, the generation goes on, the answer follows.
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  WaitStarted();
  ASSERT_EQ(::shutdown(fd, SHUT_WR), 0);
  std::string got;
  EXPECT_THAT(ReadUntil(fd, got, "\r\n\r\n"), StartsWith(kInterim));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_FALSE(backend_.cancelled.load());
  backend_.release.store(true);
  while (Recv(fd, got)) {
  }
  (void)::close(fd);
  EXPECT_FALSE(backend_.cancelled.load());
  EXPECT_THAT(got.substr(std::min(got.size(), kInterim.size())),
              AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr("Connection: close\r\n"),
                    HasSubstr(R"("content":"block")")));
}

// A client that closes entirely while a stream waits for its first token
// is told apart from a half-close by the probe, and its generation ends.
TEST_F(ServerTest, AStreamsClientThatLeavesCancelsItsGeneration) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block", R"(,"stream":true)")));
  WaitStarted();
  std::string head;
  EXPECT_THAT(ReadUntil(fd, head, "\n\n"), StartsWith("HTTP/1.1 200 OK"));  // admitted
  (void)::close(fd);
  for (int i = 0; i < 400 && !backend_.cancelled.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_TRUE(backend_.cancelled.load());
  EXPECT_THAT(Exchange(Post(Chat("after"))), StartsWith("HTTP/1.1 200 OK"));
}

// A queued stream whose client half-closes starts at once (headers and
// role chunk, no 1xx, well before `keepalive`) and is answered after its
// turn. A queued request whose client closes entirely leaves the queue
// (the probe draws a reset), streaming or not: its place is free.
TEST_F(ServerTest, QueuedRequestsTellAHalfCloseFromALeave) {
  Stop();
  backend_.release.store(false);
  api::ServerOptions options;
  options.max_queued = 1;
  options.keepalive = std::chrono::seconds(30);
  Start(options);
  const int running = Connect(Post(Chat("block")));
  WaitStarted();
  const int waiting = Connect(Post(Chat("second", R"(,"stream":true)")));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));  // the I/O thread queues it
  ASSERT_EQ(::shutdown(waiting, SHUT_WR), 0);
  std::string early;
  EXPECT_THAT(ReadUntil(waiting, early, "\n\n"),
              AllOf(StartsWith("HTTP/1.1 200 OK\r\n"), HasSubstr("Connection: close\r\n"),
                    HasSubstr(R"("delta":{"role":"assistant","content":""})")));
  backend_.release.store(true);
  std::string pending;
  EXPECT_THAT(ReadResponse(running, pending), StartsWith("HTTP/1.1 200 OK"));
  EXPECT_THAT(ReadUntil(waiting, early, "data: [DONE]\n\n"), HasSubstr(R"("content":"ond")"));
  (void)::close(running);
  (void)::close(waiting);

  for (const bool stream : {true, false}) {
    backend_.release.store(false);
    backend_.started.store(false);
    const int busy = Connect(Post(Chat("block")));
    WaitStarted();
    const int leaving =
        Connect(Post(Chat("second", stream ? R"(,"stream":true)" : std::string_view{})));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    (void)::close(leaving);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // The queue's one place is free again.
    const int next = Connect(Post(Chat("third")));
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    backend_.release.store(true);
    std::string a;
    std::string b;
    EXPECT_THAT(ReadResponse(busy, a), StartsWith("HTTP/1.1 200 OK")) << stream;
    EXPECT_THAT(ReadResponse(next, b), AllOf(StartsWith("HTTP/1.1 200 OK"), HasSubstr("third")))
        << stream;
    (void)::close(busy);
    (void)::close(next);
  }
}

TEST_F(ServerTest, StoppingEndsARunningRequest) {
  backend_.release.store(false);
  const int fd = Connect(Post(Chat("block")));
  WaitStarted();
  const int queued = Connect(Post(Chat("queued")));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const std::uint64_t one = 1;
  (void)!::write(wake_, &one, sizeof one);
  std::string a;
  std::string b;
  EXPECT_THAT(ReadResponse(fd, a),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("Connection: close\r\n")));
  EXPECT_THAT(ReadResponse(queued, b),
              AllOf(StartsWith("HTTP/1.1 503 "), HasSubstr("Retry-After: 10\r\n")));
  thread_.join();
  (void)::close(wake_);
  (void)::close(fd);
  (void)::close(queued);
  EXPECT_TRUE(result_.has_value());
}

}  // namespace
