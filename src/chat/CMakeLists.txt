# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# Native chat-template renderers (M3; D-067): the pinned DeepSeek V4 and
# Qwen3.8 templates and Qwen-Image's prompt, selected by template hash,
# with each model's stop tokens; and JSON written as Python's json.dumps
# writes it, as templates print tools and arguments. The model layer
# (docs/architecture.md#layers-and-dependency-rules), after the tokenizer.
add_library(jitllm_chat STATIC chat.cc deepseek_v4.cc pyjson.cc qwen.cc)
target_include_directories(jitllm_chat PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(jitllm_chat PUBLIC jitllm_json jitllm_tokenizer PRIVATE jitllm_warnings)
