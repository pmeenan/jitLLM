// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// JSON written as Python's json.dumps(value, ensure_ascii=False) writes a
// value json.loads read: what chat templates' `tojson` filter (Hugging Face
// transformers) and DeepSeek's encoder print into prompts, so a renderer's
// bytes match the reference's.
//
// Separators ", " and ": "; object members in document order; strings with
// `"`, `\`, \n, \r, \t, \b, \f escaped and other control characters as
// \u00XX, everything else as is; integers as Python ints print them (-0 is
// 0); other numbers as Python's float repr (shortest round trip, 1e+16 and
// 1e-05 style exponents, Infinity for overflow).

#ifndef JITLLM_CHAT_PYJSON_H_
#define JITLLM_CHAT_PYJSON_H_

#include <string>

#include "base/json.h"

namespace jitllm::chat {

void AppendPythonJson(base::json::Value value, std::string& out);

// Python's repr of a finite or infinite double, as json.dumps writes it.
void AppendPythonFloat(double value, std::string& out);

}  // namespace jitllm::chat

#endif  // JITLLM_CHAT_PYJSON_H_
