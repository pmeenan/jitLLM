// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Internal invariants (D-066): an expected failure is an error value; a
// violated invariant, such as an impossible state transition or a ledger
// underflow, is fatal. It writes one bounded line with no request content
// and aborts, which the runtime's crash policy turns into an exit
// (platform/crash_policy.h).

#ifndef JITLLM_BASE_CHECK_H_
#define JITLLM_BASE_CHECK_H_

#include <source_location>
#include <string_view>

namespace jitllm::base {

[[noreturn]] void Fatal(std::string_view what,
                        std::source_location where = std::source_location::current());

// Fatal unless condition holds. `what` names the invariant, not data.
inline void Check(bool condition, std::string_view what,
                  std::source_location where = std::source_location::current()) {
  if (!condition) [[unlikely]] {
    Fatal(what, where);
  }
}

}  // namespace jitllm::base

#endif  // JITLLM_BASE_CHECK_H_
