// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The service manager's readiness protocol (sd_notify(3)), without
// libsystemd: one datagram to the socket that $NOTIFY_SOCKET names.

#ifndef JITLLM_PLATFORM_SD_NOTIFY_H_
#define JITLLM_PLATFORM_SD_NOTIFY_H_

#include <expected>
#include <string>
#include <string_view>

namespace jitllm::platform {

// Sends state (such as "READY=1\nSTATUS=serving") if $NOTIFY_SOCKET is
// set; returns false if it is not, and an error if sending fails.
std::expected<bool, std::string> NotifyServiceManager(std::string_view state);

}  // namespace jitllm::platform

#endif  // JITLLM_PLATFORM_SD_NOTIFY_H_
