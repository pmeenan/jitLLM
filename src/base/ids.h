// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// Typed, generation-checked identities (D-006, D-048): an index into a
// table and the generation the slot had when the identity was issued. A
// slot's generation advances whenever it is reused, so a stale identity
// never names the slot's new occupant. Generation 0 is never issued: a
// default identity names nothing.

#ifndef JITLLM_BASE_IDS_H_
#define JITLLM_BASE_IDS_H_

#include <compare>
#include <cstdint>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace jitllm::base {

template <typename Tag>
class Id {
 public:
  constexpr Id() = default;
  constexpr Id(std::uint32_t index, std::uint32_t generation)
      : index_(index), generation_(generation) {}

  constexpr std::uint32_t index() const { return index_; }
  constexpr std::uint32_t generation() const { return generation_; }
  constexpr bool valid() const { return generation_ != 0; }

  constexpr auto operator<=>(const Id&) const = default;

  std::string ToString() const { return std::format("{}#{}.{}", Tag::kName, index_, generation_); }

 private:
  std::uint32_t index_ = 0;
  std::uint32_t generation_ = 0;
};

// A table of slots addressed by Id<Tag>. Erasing a slot advances its
// generation; a slot whose generation would wrap is retired, never reused
// (D-048: exhaust the ID space by refusing, not by reuse).
template <typename Tag, typename T>
class SlotTable {
 public:
  // The identity of the new element; invalid if the table is exhausted.
  template <typename... Args>
  Id<Tag> Insert(Args&&... args) {
    std::uint32_t index = 0;
    if (!free_.empty()) {
      index = free_.back();
      free_.pop_back();
    } else {
      if (slots_.size() >= kMaxSlots) {
        return {};
      }
      index = static_cast<std::uint32_t>(slots_.size());
      slots_.push_back(Slot{});
    }
    Slot& slot = slots_[index];
    slot.value.emplace(std::forward<Args>(args)...);
    ++live_;
    return {index, slot.generation};
  }

  // The element an identity names, or nullptr if it is stale or invalid.
  T* Find(Id<Tag> id) {
    if (!id.valid() || id.index() >= slots_.size()) {
      return nullptr;
    }
    Slot& slot = slots_[id.index()];
    return slot.generation == id.generation() && slot.value ? &*slot.value : nullptr;
  }
  const T* Find(Id<Tag> id) const { return const_cast<SlotTable*>(this)->Find(id); }  // NOLINT

  // Removes the element; false if the identity is stale or invalid.
  bool Erase(Id<Tag> id) {
    if (Find(id) == nullptr) {
      return false;
    }
    Slot& slot = slots_[id.index()];
    slot.value.reset();
    --live_;
    if (slot.generation == UINT32_MAX) {
      return true;  // retired: its generation cannot advance
    }
    ++slot.generation;
    free_.push_back(id.index());
    return true;
  }

  std::size_t size() const { return live_; }

  // Calls fn(id, value) for every live element, in index order.
  template <typename Fn>
  void ForEach(Fn&& fn) const {
    for (std::uint32_t i = 0; i < slots_.size(); ++i) {
      const Slot& slot = slots_[i];
      if (slot.value.has_value()) {
        fn(Id<Tag>(i, slot.generation),
           slot.value.value());  // NOLINT(bugprone-unchecked-optional-access)
      }
    }
  }

 private:
  static constexpr std::size_t kMaxSlots = UINT32_MAX;
  struct Slot {
    std::uint32_t generation = 1;
    std::optional<T> value;
  };
  std::vector<Slot> slots_;
  std::vector<std::uint32_t> free_;
  std::size_t live_ = 0;
};

}  // namespace jitllm::base

#endif  // JITLLM_BASE_IDS_H_
