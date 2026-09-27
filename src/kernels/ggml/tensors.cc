// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

#include "kernels/ggml/tensors.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "base/check.h"
#include "ggml.h"

namespace jitllm::kernels::ggml {

void TensorArena::Free::operator()(ggml_context* context) const { ggml_free(context); }

std::expected<TensorArena, KernelFailure> TensorArena::Create(std::size_t tensors) {
  const std::size_t overhead = ggml_tensor_overhead();
  if (tensors == 0 || tensors > std::numeric_limits<std::size_t>::max() / overhead) {
    return std::unexpected(
        KernelFailure{.error = KernelError::kRejected,
                      .detail = std::format("no arena holds {} tensors", tensors)});
  }
  const std::size_t size = tensors * overhead;
  std::vector<std::byte> buffer(size);
  // GGML requires its memory pool aligned to GGML_MEM_ALIGN (16).
  base::Check(reinterpret_cast<std::uintptr_t>(buffer.data()) % 16 == 0,
              "operator new returns 16-byte aligned memory");
  ggml_context* context =
      ggml_init(ggml_init_params{.mem_size = size, .mem_buffer = buffer.data(), .no_alloc = true});
  return TensorArena(std::move(buffer), context, size);
}

TensorArena::TensorArena(std::vector<std::byte> buffer, ggml_context* context, std::size_t capacity)
    : buffer_(std::move(buffer)), context_(context), capacity_(capacity) {}

TensorArena::TensorArena(TensorArena&&) noexcept = default;
TensorArena& TensorArena::operator=(TensorArena&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  // The context goes before the buffer it uses.
  context_ = std::move(other.context_);
  buffer_ = std::move(other.buffer_);
  capacity_ = std::exchange(other.capacity_, 0);
  return *this;
}
TensorArena::~TensorArena() = default;

std::expected<void, KernelFailure> TensorArena::Reserve(std::size_t tensors) const {
  const std::size_t used = context_ == nullptr ? capacity_ : ggml_used_mem(context_.get());
  const std::size_t room = (capacity_ - used) / ggml_tensor_overhead();
  if (tensors > room) {
    return std::unexpected(KernelFailure{
        .error = KernelError::kRejected,
        .detail = std::format("the arena has room for {} more tensors, not {}", room, tensors)});
  }
  return {};
}

void TensorArena::Bind(ggml_tensor* tensor, std::uint64_t address) {
  tensor->data = reinterpret_cast<void*>(address);  // NOLINT(performance-no-int-to-ptr)
}

}  // namespace jitllm::kernels::ggml
