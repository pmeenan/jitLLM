// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// M3's minimal chat route (D-097; docs/runtime-serving.md#the-chat-route):
// an OpenAI-shaped POST /v1/chat/completions request read from untrusted
// bytes under fixed numeric bounds, and the JSON the route answers with.
// The full front door (client-api-baseline.md) is M5's; this is a subset of
// its Chat Completions profile, vendor-free and CPU-tested.
//
// What a request may hold. `model`, `messages` (system, developer, user and
// assistant text; a string or an array of text parts), `max_tokens` or
// `max_completion_tokens`, `temperature`, `top_p`, `seed`, `stop`,
// `stream` and `stream_options.include_usage` are honored. A documented
// set is accepted only at the value that means "off" (n = 1, zero
// penalties, no logprobs, no tools, text responses), and a documented set
// of metadata is ignored (user, metadata, prompt_cache_key,
// safety_identifier, service_tier, parallel_tool_calls, store = false):
// client-api-baseline.md's "harmless metadata may be ignored only under a
// documented rule". Anything else is refused with a 400 naming the field,
// never silently dropped.

#ifndef JITLLM_RUNTIME_API_H_
#define JITLLM_RUNTIME_API_H_

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace jitllm::runtime::api {

// The intake bounds, fixed before the route accepts input
// (client-api-baseline.md#shared-correctness-and-limits); the reasons are
// in runtime-serving.md.
inline constexpr std::size_t kMaxHeaderBytes = std::size_t{16} << 10U;  // request line + headers
inline constexpr std::size_t kMaxHeaders = 64;
inline constexpr std::size_t kMaxTargetBytes = 2048;
inline constexpr std::size_t kMaxBodyBytes = std::size_t{4} << 20U;
inline constexpr std::size_t kMaxJsonDepth = 16;
inline constexpr std::size_t kMaxJsonValues = std::size_t{1} << 18U;
inline constexpr std::size_t kMaxMessages = 1024;
inline constexpr std::size_t kMaxMessageBytes = std::size_t{1} << 20U;  // one message's text
inline constexpr std::size_t kMaxContentParts = 64;
inline constexpr std::size_t kMaxModelBytes = 64;
inline constexpr std::size_t kMaxStops = 4;
inline constexpr std::size_t kMaxStopBytes = 128;
inline constexpr std::uint32_t kMaxTokensCeiling = 262144;  // config::kMaxContext
inline constexpr double kMaxTemperature = 2.0;
// Timeouts and the queue, in the server (api_server.h).
inline constexpr std::uint32_t kHeaderTimeoutMs = 10'000;
inline constexpr std::uint32_t kBodyTimeoutMs = 30'000;
inline constexpr std::uint32_t kWriteTimeoutMs = 30'000;
inline constexpr std::uint32_t kQueueWaitMs = 120'000;
inline constexpr std::uint32_t kDeadlineMs = 600'000;
inline constexpr std::size_t kMaxQueued = 4;
inline constexpr std::uint32_t kRetryAfterSeconds = 10;

// An OpenAI-shaped error: the HTTP status and the body's error object.
struct Error {
  int status = 400;
  std::string type = "invalid_request_error";
  std::string message;
  std::string param;  // empty: null
  std::string code;   // empty: null
};

enum class Role : std::uint8_t { kSystem, kUser, kAssistant };

struct Message {
  Role role = Role::kUser;               // "developer" is read as system
  std::string content;                   // text parts joined
  std::optional<std::string> reasoning;  // an assistant's reasoning, sent back
};

struct ChatRequest {
  std::string model;
  std::vector<Message> messages;  // 1 to kMaxMessages, the last the user's
  std::optional<std::uint32_t> max_tokens;
  double temperature = 1.0;  // [0, 2]; 0 is greedy (OpenAI's default is 1)
  double top_p = 1.0;        // (0, 1]
  std::optional<std::uint64_t> seed;
  std::vector<std::string> stop;  // matched against the answer's text
  bool stream = false;
  bool include_usage = false;
};

// Parses and checks a request body (JSON), before any model work.
std::expected<ChatRequest, Error> ParseChatRequest(std::string_view body);

// ---------------------------------------------------------------- output

enum class Finish : std::uint8_t { kStop, kLength };

struct Usage {
  std::uint32_t prompt_tokens = 0;
  std::uint32_t completion_tokens = 0;
  std::uint32_t cached_tokens = 0;  // the prompt's tokens the conversation state already held
};

// {"error":{...}}
std::string ErrorJson(const Error& error);
// A non-streaming response (object "chat.completion").
std::string CompletionJson(std::string_view id, std::int64_t created, std::string_view model,
                           std::string_view content, const std::optional<std::string>& reasoning,
                           Finish finish, const Usage& usage);
// One streamed chunk (object "chat.completion.chunk"): a delta of the role,
// the reasoning or the content, or with `finish` the last one.
enum class Delta : std::uint8_t { kRole, kReasoning, kContent, kFinish };
std::string ChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                      Delta delta, std::string_view text, Finish finish = Finish::kStop);
// The usage chunk (stream_options.include_usage): empty choices.
std::string UsageChunkJson(std::string_view id, std::int64_t created, std::string_view model,
                           const Usage& usage);

// A model on the route's list (GET /v1/models).
struct ModelInfo {
  std::string name;
  bool chat = true;  // false: an image pipeline, which this route does not serve
  std::uint32_t context = 0;
};
std::string ModelJson(const ModelInfo& model, std::int64_t created);
std::string ModelsJson(const std::vector<ModelInfo>& models, std::int64_t created);

// The generated text on its way out: the reasoning (before the template's
// end-of-reasoning marker) and the answer, with the request's stop strings
// matched in the answer. What might still become a stop string is held
// back until it cannot; the answer's leading whitespace after reasoning is
// dropped, as the chat command does.
class OutputText {
 public:
  explicit OutputText(std::vector<std::string> stops) : stops_(std::move(stops)) {}

  struct Out {
    std::string reasoning;
    std::string content;
  };
  // Adds generated text; returns what may be sent now. After a stop
  // string, stopped() is true and nothing more is taken. Reasoning with
  // empty text marks a reasoning block that said nothing, so the answer's
  // leading whitespace is still dropped.
  Out Reasoning(std::string_view text);
  Out Content(std::string_view text);
  // At the end: whatever was held back.
  Out Finish();
  bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  std::string held_;  // answer text that may begin a stop string
  bool had_reasoning_ = false;
  bool content_started_ = false;
  bool stopped_ = false;
};

}  // namespace jitllm::runtime::api

#endif  // JITLLM_RUNTIME_API_H_
