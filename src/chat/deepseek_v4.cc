// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// DeepSeek V4 Flash 0731's chat format, as the 0731 GGUF's embedded template
// (SHA-256 e643c31f..., Unsloth's port of DeepSeek's encoding_dsv4.py at
// deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b) renders it for the roles and
// options jitLLM supports. Written from the format, not translated from the
// template; the fixtures hold both references (docs/tokenizer.md).

#include <cstddef>
#include <expected>
#include <string>
#include <string_view>

#include "base/json.h"
#include "chat/chat.h"
#include "chat/writer.h"

namespace jitllm::chat {
namespace {

constexpr std::string_view kBos = "<｜begin▁of▁sentence｜>";
constexpr std::string_view kEos = "<｜end▁of▁sentence｜>";
constexpr std::string_view kUser = "<｜User｜>";
constexpr std::string_view kAssistant = "<｜Assistant｜>";
constexpr std::string_view kThinkStart = "<think>";
constexpr std::string_view kThinkEnd = "</think>";
constexpr std::string_view kDsml = "｜DSML｜";

constexpr std::string_view kEffortHigh =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to "
    "resolve the "
    "root cause, rigorously stress-testing your logic against all potential paths, edge cases, and "
    "adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, "
    "considered "
    "alternative, and rejected hypothesis to ensure absolutely no assumption is left "
    "unchecked.\n\n";
constexpr std::string_view kEffortMax =
    "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n"
    "You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: "
    "exhaustively "
    "decompose the problem into its most fundamental components, trace every causal chain to its "
    "root, and "
    "resolve the underlying cause rather than any surface symptom.\n"
    "Do not stop reasoning until you have independently verified the solution from multiple angles "
    "and are "
    "certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

constexpr std::string_view kToolsHeader =
    "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can "
    "invoke tools "
    "by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
    "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" "
    "string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "...\n</｜DSML｜invoke>\n<｜DSML｜invoke "
    "name=\"$TOOL_NAME2\">\n...\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>\n\n"
    "String parameters should be specified as is and set `string=\"true\"`. For all other types "
    "(numbers, "
    "booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning "
    "inside "
    "<think>...</think> BEFORE any tool calls or final response.\n\n"
    "Otherwise, output directly after </think> with tool calls or final response.\n\n"
    "### Available Tool Schemas\n\n";
constexpr std::string_view kToolsFooter =
    "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool "
    "calls.\n";

std::unexpected<Error> Fail(Rule rule, std::string_view reason, std::size_t item = kNoItem) {
  return std::unexpected(Error{rule, reason, item});
}

bool NonEmpty(const std::optional<std::string>& s) { return s && !s->empty(); }

}  // namespace

std::expected<Rendered, Error> RenderDeepSeekV4(const Conversation& c) {
  if (c.preserve_thinking) {
    return Fail(Rule::kUnsupported, "preserve_thinking is a Qwen3.8 option");
  }
  const bool thinking = c.enable_thinking.value_or(false);
  std::string_view effort_prefix;
  if (c.reasoning_effort) {
    const std::string_view e = *c.reasoning_effort;
    if (e == "high") {
      effort_prefix = kEffortHigh;
    } else if (e == "max") {
      effort_prefix = kEffortMax;
    } else if (e != "low") {
      return Fail(Rule::kInvalid, "reasoning_effort other than low, high or max");
    }
  }
  const bool has_tools = !c.tools.empty();

  // The system prompt: every system message, then the tool schemas.
  std::string system;
  bool first_system = true;
  for (const Message& m : c.messages) {
    if (m.role != Role::kSystem) {
      continue;
    }
    if (!first_system) {
      system += "\n\n";
    }
    system += m.content.value_or("");
    first_system = false;
  }
  std::string schemas;
  for (std::size_t i = 0; i < c.tools.size(); ++i) {
    const base::json::Value tool = c.tools[i];
    const auto type = tool.find("type");
    if (!type || type->string() != "function") {
      continue;  // the template renders only function tools
    }
    const auto function = tool.find("function");
    if (!function) {
      return Fail(Rule::kInvalid, "a function tool without its function", i);
    }
    AppendPythonJson(*function, schemas);
    schemas += '\n';
  }

  Writer w;
  w.Special(kBos);
  if (thinking) {
    w.Text(effort_prefix);
  }
  w.Text(system);
  if (has_tools) {
    if (!first_system) {
      w.Text("\n\n");
    }
    w.TextWithTokens(kToolsHeader, {kDsml, kThinkStart, kThinkEnd});
    w.Text(schemas);
    w.Text(kToolsFooter);
  }
  w.Mark(BoundaryKind::kPrefixEnd);

  std::size_t last_user = c.messages.size();  // none
  for (std::size_t i = 0; i < c.messages.size(); ++i) {
    if (c.messages[i].role == Role::kUser || c.messages[i].role == Role::kTool) {
      last_user = i;
    }
  }
  bool in_user = false;
  for (std::size_t i = 0; i < c.messages.size(); ++i) {
    const Message& m = c.messages[i];
    switch (m.role) {
      case Role::kSystem:
        break;  // rendered above
      case Role::kUser:
      case Role::kTool:
        if (!m.tool_calls.empty() || m.reasoning_content) {
          return Fail(Rule::kUnsupported, "tool calls or reasoning on a user or tool message", i);
        }
        if (in_user) {
          w.Text("\n\n");
        } else {
          w.Special(kUser);
          in_user = true;
        }
        if (m.role == Role::kUser) {
          w.Text(m.content.value_or(""));
        } else {
          w.Text("<tool_result>");
          w.Text(m.content.value_or(""));
          w.Text("</tool_result>");
        }
        break;
      case Role::kAssistant: {
        in_user = false;
        const bool after_user = i > 0 && (c.messages[i - 1].role == Role::kUser ||
                                          c.messages[i - 1].role == Role::kTool);
        const bool keep_reasoning = has_tools || last_user == c.messages.size() || i > last_user;
        if (after_user) {
          w.Special(kAssistant);
          if (keep_reasoning && thinking) {
            w.Special(kThinkStart);
            w.Text(m.reasoning_content.value_or(""));
            w.Special(kThinkEnd);
          } else {
            w.Special(kThinkEnd);
          }
        } else if (keep_reasoning && thinking) {
          w.Text(m.reasoning_content.value_or(""));
          w.Special(kThinkEnd);
        }
        if (NonEmpty(m.content)) {
          w.Text(*m.content);
        }
        if (!m.tool_calls.empty()) {
          w.Text("\n\n<");
          w.Special(kDsml);
          w.Text("tool_calls>\n");
          for (const ToolCall& call : m.tool_calls) {
            if (!call.arguments.is_object()) {
              return Fail(Rule::kInvalid, "tool call arguments are not an object", i);
            }
            w.Text("<");
            w.Special(kDsml);
            w.Text("invoke name=\"");
            w.Text(call.name);
            w.Text("\">\n");
            for (std::size_t k = 0; k < call.arguments.size(); ++k) {
              const base::json::Value v = call.arguments.member(k);
              w.Text("<");
              w.Special(kDsml);
              w.Text("parameter name=\"");
              w.Text(call.arguments.key(k));
              if (v.is_string()) {
                w.Text(R"(" string="true">)");
                w.Text(v.string());
              } else {
                w.Text(R"(" string="false">)");
                w.Json(v);
              }
              w.Text("</");
              w.Special(kDsml);
              w.Text("parameter>\n");
            }
            w.Text("</");
            w.Special(kDsml);
            w.Text("invoke>\n");
          }
          w.Text("</");
          w.Special(kDsml);
          w.Text("tool_calls>");
        }
        w.Special(kEos);
        break;
      }
    }
    w.Mark(BoundaryKind::kMessageEnd);
  }
  if (c.add_generation_prompt) {
    w.Mark(BoundaryKind::kGenerationPrompt);
    w.Special(kAssistant);
    w.Special(thinking ? kThinkStart : kThinkEnd);
  }
  return w.Take();
}

}  // namespace jitllm::chat
