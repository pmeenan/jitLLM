// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "chat/pyjson.h"

#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <string_view>
#include <system_error>

#include "base/json.h"

namespace jitllm::chat {

void AppendPythonFloat(double value, std::string& out) {
  if (std::isinf(value)) {
    out += value < 0 ? "-Infinity" : "Infinity";
    return;
  }
  if (std::isnan(value)) {
    out += "NaN";
    return;
  }
  std::array<char, 64> buffer{};
  const auto [end, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                       std::chars_format::scientific);
  if (ec != std::errc()) {
    out += "NaN";  // unreachable: 64 bytes hold any double
    return;
  }
  std::string_view s(buffer.data(), static_cast<std::size_t>(end - buffer.data()));
  if (s.starts_with('-')) {
    out.push_back('-');
    s.remove_prefix(1);
  }
  const std::size_t e = s.find('e');
  std::string digits;
  for (const char c : s.substr(0, e)) {
    if (c != '.') {
      digits.push_back(c);
    }
  }
  int exponent = 0;
  const std::string_view exp_text = s.substr(e + 1);
  std::from_chars(exp_text.data() + (exp_text.starts_with('+') ? 1 : 0),
                  exp_text.data() + exp_text.size(), exponent);
  const int decpt = exponent + 1;  // digits d1 d2 ... are 0.d1d2... x 10^decpt
  const auto n = static_cast<int>(digits.size());
  if (decpt <= -4 || decpt > 16) {
    out.push_back(digits[0]);
    if (n > 1) {
      out.push_back('.');
      out.append(digits, 1);
    }
    const int x = decpt - 1;
    out.push_back('e');
    out.push_back(x < 0 ? '-' : '+');
    const int magnitude = std::abs(x);
    if (magnitude < 10) {
      out.push_back('0');
    }
    out += std::to_string(magnitude);
    return;
  }
  if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-decpt), '0');
    out += digits;
  } else if (decpt < n) {
    out.append(digits, 0, static_cast<std::size_t>(decpt));
    out.push_back('.');
    out.append(digits, static_cast<std::size_t>(decpt));
  } else {
    out += digits;
    out.append(static_cast<std::size_t>(decpt - n), '0');
    out += ".0";
  }
}

namespace {

void AppendString(std::string_view text, std::string& out) {
  // Python's json escapes exactly what RFC 8259's minimal escaping does,
  // with lower-case hex, and leaves DEL and everything non-ASCII alone.
  base::json::AppendQuoted(text, out);
}

}  // namespace

void AppendPythonJson(base::json::Value value, std::string& out) {
  using base::json::Kind;
  switch (value.kind()) {
    case Kind::kNull:
      out += "null";
      return;
    case Kind::kTrue:
      out += "true";
      return;
    case Kind::kFalse:
      out += "false";
      return;
    case Kind::kString:
      AppendString(value.string(), out);
      return;
    case Kind::kNumber:
      if (value.is_integer()) {
        const std::string_view t = value.number();
        out += t.find_first_not_of("-0") == std::string_view::npos ? "0" : t;
      } else if (const auto x = value.float64()) {
        AppendPythonFloat(*x, out);
      }
      return;
    case Kind::kArray:
      out.push_back('[');
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        AppendPythonJson(value.at(i), out);
      }
      out.push_back(']');
      return;
    case Kind::kObject:
      out.push_back('{');
      for (std::size_t i = 0; i < value.size(); ++i) {
        if (i != 0) {
          out += ", ";
        }
        AppendString(value.key(i), out);
        out += ": ";
        AppendPythonJson(value.member(i), out);
      }
      out.push_back('}');
      return;
  }
}

}  // namespace jitllm::chat
