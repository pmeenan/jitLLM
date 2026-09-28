// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Builds a Rendered: text, the special tokens a template places (marked so
// the tokenizer encodes exactly those as control tokens) and boundaries.
// Internal to the renderers.

#ifndef JITLLM_CHAT_WRITER_H_
#define JITLLM_CHAT_WRITER_H_

#include <initializer_list>
#include <string>
#include <string_view>
#include <utility>

#include "base/json.h"
#include "chat/chat.h"
#include "chat/pyjson.h"

namespace jitllm::chat {

class Writer {
 public:
  void Text(std::string_view text) { r_.text += text; }
  void Special(std::string_view token) {
    r_.specials.push_back({r_.text.size(), token.size()});
    r_.text += token;
  }
  // Text in which each occurrence of the given tokens is marked special.
  void TextWithTokens(std::string_view text, std::initializer_list<std::string_view> tokens) {
    while (!text.empty()) {
      std::size_t best = text.size();
      std::string_view found;
      for (const std::string_view t : tokens) {
        const std::size_t at = text.find(t);
        if (at < best || (at == best && at != text.size() && t.size() > found.size())) {
          best = at;
          found = t;
        }
      }
      Text(text.substr(0, best));
      if (found.empty()) {
        return;
      }
      Special(found);
      text.remove_prefix(best + found.size());
    }
  }
  void Json(base::json::Value value) { AppendPythonJson(value, r_.text); }
  void Mark(BoundaryKind kind) { r_.boundaries.push_back({kind, r_.text.size()}); }
  Rendered Take() { return std::move(r_); }

 private:
  Rendered r_;
};

}  // namespace jitllm::chat

#endif  // JITLLM_CHAT_WRITER_H_
