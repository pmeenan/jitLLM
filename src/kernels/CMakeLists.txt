# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Build-time implementations of operations, one directory per kernel source
# (D-053; docs/architecture.md#layers-and-dependency-rules).
add_subdirectory(ggml)
add_subdirectory(exl3)
add_subdirectory(image)
