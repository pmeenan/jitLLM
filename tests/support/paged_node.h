// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The paged node (engine/paged_node.h) under the names the paged harnesses
// and their tests use.

#ifndef JITLLM_TESTS_SUPPORT_PAGED_NODE_H_
#define JITLLM_TESTS_SUPPORT_PAGED_NODE_H_

#include "engine/paged_node.h"
#include "paged_programs.h"

namespace jitllm::test_support {

using engine::CountingStorage;
using engine::kPagedDepth;
using engine::kPagedExtent;
using engine::kPagedSlots;
using engine::kShared;
using engine::LoadStats;
using engine::Mapped;
using engine::NodeSettings;
using engine::PagedModel;
using engine::PagedNode;
using engine::Span;
using engine::Status;
using engine::StepTimes;

}  // namespace jitllm::test_support

#endif  // JITLLM_TESTS_SUPPORT_PAGED_NODE_H_
