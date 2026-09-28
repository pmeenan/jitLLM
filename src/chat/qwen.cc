// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Qwen3.8 Flash Next's chat format, as the NVFP4 checkpoint's
// chat_template.jinja (SHA-256 c3cf9e34...) renders it through Hugging Face
// transformers, and Qwen-Image 2.1's text-to-image prompt. Written from the
// format, not translated from the template (docs/tokenizer.md).

#include <cstddef>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/json.h"
#include "chat/chat.h"
#include "chat/writer.h"
#include "tokenizer/tokenizer.h"

namespace jitllm::chat {
namespace {

constexpr std::string_view kImStart = "<|im_start|>";
constexpr std::string_view kImEnd = "<|im_end|>";
constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kToolCall = "<tool_call>";
constexpr std::string_view kToolCallEnd = "</tool_call>";
constexpr std::string_view kToolResponse = "<tool_response>";
constexpr std::string_view kToolResponseEnd = "</tool_response>";

constexpr std::string_view kEffortXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, "
    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the "
    "final "
    "answer.";
constexpr std::string_view kEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</"
    "parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can "
    "span\nmultiple "
    "lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be "
    "nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function "
    "call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n</IMPORTANT>";

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::size_t item = kNoItem) {
  return std::unexpected(Error{rule, reason, item});
}

std::string_view Content(const Message& m) {
  return PythonStrip(m.content ? *m.content : std::string_view());
}

}  // namespace

std::expected<Rendered, Error> RenderQwen38(const Conversation& c) {
  if (c.messages.empty()) {
    return Fail(Rule::kInvalid, "no messages");
  }
  std::string_view instructions;
  if (!c.enable_thinking || *c.enable_thinking) {
    const std::string_view effort =
        c.reasoning_effort ? std::string_view(*c.reasoning_effort) : "xhigh";
    if (effort == "xhigh") {
      instructions = kEffortXhigh;
    } else if (effort == "low") {
      instructions = kEffortLow;
    } else if (effort != "medium") {
      return Fail(Rule::kInvalid, "reasoning_effort other than xhigh, medium or low");
    }
  }
  const bool preserve = !c.preserve_thinking || *c.preserve_thinking;
  const Message& first = c.messages[0];

  Writer w;
  if (!c.tools.empty()) {
    w.Special(kImStart);
    w.Text("system\n");
    if (!instructions.empty()) {
      w.Text(instructions);
      w.Text("\n\n");
    }
    w.Text("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for (const base::json::Value tool : c.tools) {
      w.Text("\n");
      w.Json(tool);
    }
    w.Text("\n</tools>");
    w.TextWithTokens(kToolInstructions, {kToolCall, kToolCallEnd});
    if (first.role == Role::kSystem) {
      if (const std::string_view content = Content(first); !content.empty()) {
        w.Text("\n\n");
        w.Text(content);
      }
    }
    w.Special(kImEnd);
    w.Text("\n");
  } else {
    const std::string_view content =
        first.role == Role::kSystem ? Content(first) : std::string_view();
    if (!content.empty() || !instructions.empty()) {
      w.Special(kImStart);
      w.Text("system\n");
      w.Text(instructions);
      if (!content.empty() && !instructions.empty()) {
        w.Text("\n\n");
      }
      w.Text(content);
      w.Special(kImEnd);
      w.Text("\n");
    }
  }
  w.Mark(BoundaryKind::kPrefixEnd);

  // The last user query: the last user message that is not only a tool
  // response.
  std::optional<std::size_t> last_query;
  for (std::size_t i = c.messages.size(); i-- > 0;) {
    if (c.messages[i].role != Role::kUser) {
      continue;
    }
    const std::string_view content = Content(c.messages[i]);
    if (!(content.starts_with(kToolResponse) && content.ends_with(kToolResponseEnd))) {
      last_query = i;
      break;
    }
  }
  if (!last_query) {
    return Fail(Rule::kInvalid, "no user query in the messages");
  }

  for (std::size_t i = 0; i < c.messages.size(); ++i) {
    const Message& m = c.messages[i];
    const std::string_view content = Content(m);
    if (m.role != Role::kAssistant && (!m.tool_calls.empty() || m.reasoning_content)) {
      return Fail(Rule::kUnsupported,
                  "tool calls or reasoning on a message that is not the assistant's", i);
    }
    switch (m.role) {
      case Role::kSystem:
        if (i != 0) {
          return Fail(Rule::kInvalid, "a system message after the first", i);
        }
        break;
      case Role::kUser:
        w.Special(kImStart);
        w.Text("user\n");
        w.Text(content);
        w.Special(kImEnd);
        w.Text("\n");
        break;
      case Role::kAssistant: {
        const std::string_view reasoning = PythonStrip(
            m.reasoning_content ? std::string_view(*m.reasoning_content) : std::string_view());
        w.Special(kImStart);
        w.Text("assistant\n");
        if (preserve || i > *last_query) {
          w.Special(kThinkStart);
          w.Text("\n");
          w.Text(reasoning);
          w.Text("\n");
          w.Special(kThinkEnd);
          w.Text("\n\n");
        }
        w.Text(content);
        for (std::size_t k = 0; k < m.tool_calls.size(); ++k) {
          const ToolCall& call = m.tool_calls[k];
          if (!call.arguments.is_object()) {
            return Fail(Rule::kInvalid, "tool call arguments are not an object", i);
          }
          if (k != 0) {
            w.Text("\n");
          } else if (!content.empty()) {
            w.Text("\n\n");
          }
          w.Special(kToolCall);
          w.Text("\n<function=");
          w.Text(call.name);
          w.Text(">\n");
          for (std::size_t a = 0; a < call.arguments.size(); ++a) {
            const base::json::Value v = call.arguments.member(a);
            w.Text("<parameter=");
            w.Text(call.arguments.key(a));
            w.Text(">\n");
            if (v.is_string()) {
              w.Text(v.string());
            } else {
              w.Json(v);
            }
            w.Text("\n</parameter>\n");
          }
          w.Text("</function>\n");
          w.Special(kToolCallEnd);
        }
        w.Special(kImEnd);
        w.Text("\n");
        break;
      }
      case Role::kTool: {
        if (i > 0 && c.messages[i - 1].role != Role::kTool) {
          w.Special(kImStart);
          w.Text("user");
        }
        w.Text("\n");
        w.Special(kToolResponse);
        w.Text("\n");
        w.Text(content);
        w.Text("\n");
        w.Special(kToolResponseEnd);
        if (i + 1 == c.messages.size() || c.messages[i + 1].role != Role::kTool) {
          w.Special(kImEnd);
          w.Text("\n");
        }
        break;
      }
    }
    w.Mark(BoundaryKind::kMessageEnd);
  }
  if (c.add_generation_prompt) {
    w.Mark(BoundaryKind::kGenerationPrompt);
    w.Special(kImStart);
    w.Text("assistant\n");
    w.Special(kThinkStart);
    if (c.enable_thinking && !*c.enable_thinking) {
      w.Text("\n\n");
      w.Special(kThinkEnd);
      w.Text("\n\n");
    } else {
      w.Text("\n");
    }
  }
  return w.Take();
}

std::expected<ImagePrompt, Error> RenderQwenImagePrompt(std::string_view prompt,
                                                        const tokenizer::Tokenizer& tokenizer) {
  constexpr std::string_view kSystem = "Comprehend and analyze the provided prompt.";
  Writer w;
  w.Special(kImStart);
  w.Text("system\n");
  w.Text(kSystem);
  w.Special(kImEnd);
  w.Text("\n");
  w.Mark(BoundaryKind::kPrefixEnd);
  const Rendered system = [&] {
    Writer s;
    s.Special(kImStart);
    s.Text("system\n");
    s.Text(kSystem);
    s.Special(kImEnd);
    s.Text("\n");
    return s.Take();
  }();
  w.Special(kImStart);
  w.Text("user\n");
  w.Text(prompt.empty() ? std::string_view(" ") : prompt);
  w.Special(kImEnd);
  w.Text("\n");
  w.Mark(BoundaryKind::kGenerationPrompt);
  w.Special(kImStart);
  w.Text("assistant\n");

  std::vector<tokenizer::TokenId> ids;
  if (auto e = tokenizer.EncodeMarked(system.text, system.specials, {}, ids); !e) {
    return Fail(Rule::kInvalid, "the image prompt's system segment does not encode");
  }
  return ImagePrompt{w.Take(), ids.size()};
}

}  // namespace jitllm::chat
