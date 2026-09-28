// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The node's configuration (D-063, D-073): one logical document of strict
// TOML 1.0, a main file plus the fragments in its drop-in directory, in
// cluster-design.md's node-local schema version 2 extended with [storage]
// and the models the node serves ([models.<name>], D-096).
//
// Loading reads the files, refusing any that users other than root and the
// runtime's user could replace (platform/path_trust.h), parses each on its
// own, merges them so that every key has exactly one owning file, and
// validates the result against the schema. Every problem found is
// reported, not just the first: a file that does not parse contributes its
// syntax error, and the rest are still checked.

#ifndef JITLLM_CONFIG_NODE_CONFIG_H_
#define JITLLM_CONFIG_NODE_CONFIG_H_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "base/surface_versions.h"

namespace jitllm::config {

inline constexpr std::int64_t kSchemaVersion = surface::kConfigSchemaVersion;
// The merged document's size limit (cluster-design.md).
inline constexpr std::size_t kDocumentLimit = std::size_t{1} << 20;
inline constexpr std::string_view kDefaultConfigFile = "/etc/jitllm/jitllm.toml";
// The enrollment anchor's fixed path when packaged (D-063).
inline constexpr std::string_view kDefaultAnchor = "/var/lib/jitllm/enrollment";
inline constexpr std::string_view kDefaultDataDir = "/var/lib/jitllm";
inline constexpr std::string_view kLimitsProfile = "initial-v2";
// At most this many fragments form the document.
inline constexpr std::size_t kDropInLimit = 256;
// At most this many problems are reported, then a count of the rest.
inline constexpr std::size_t kDiagnosticLimit = 100;

// The storage roles (D-054, D-055, D-063), as absolute paths in normal
// form: relative values resolved against data_dir, or long_term for
// archive. Nothing here has touched the filesystem.
struct Storage {
  std::filesystem::path data_dir;
  std::filesystem::path installed;
  std::filesystem::path spill;
  std::filesystem::path state;
  std::filesystem::path checkpoints;
  std::optional<std::filesystem::path> long_term;
  std::optional<std::filesystem::path> archive;
};

// A local interface selector: "ifname:<name>" or
// "port:<phys_switch_id>/<phys_port_name>" (cluster-design.md).
struct Control {
  std::uint16_t port = 0;
  // Empty means "auto": the validated interconnects.
  std::vector<std::string> interfaces;
  // Member UUID to a local interface selector.
  std::map<std::string, std::string> peer_scopes;
};

struct Credentials {
  std::filesystem::path ca_file;
  std::filesystem::path certificate_file;
  std::filesystem::path private_key_file;
};

// The keys a cluster member adds; a standalone node has none of them.
struct Membership {
  std::filesystem::path cluster_file;
  std::string node_id;
  Credentials credentials;
  Control control;
};

// The models a node serves (D-096): `[models.<name>]`, one table a model.
inline constexpr std::size_t kMaxModels = 16;
inline constexpr std::size_t kMaxModelName = 64;
// A model's conversation state, in tokens: the default, and its bounds.
inline constexpr std::uint32_t kDefaultContext = 8704;
inline constexpr std::uint32_t kMinContext = 512;
inline constexpr std::uint32_t kMaxContext = 262144;

// One model: the name the CLI (and later the API) asks for, and what the
// installed store holds for it. Nothing here has touched the store.
struct ModelEntry {
  // 1-64 characters of [a-z0-9._-], starting with a letter or digit.
  std::string name;
  // Exactly one of: an installed artifact's ID (a model of one component),
  // or an installed composition's ID (a pipeline, D-089). Each is 64
  // lowercase hex digits, the directory's name under storage.installed.
  std::optional<std::string> artifact;
  std::optional<std::string> composition;
  // With an artifact: its speculative drafter's artifact (DSpark, MTP;
  // D-089's drafter binding). Speculation is then the default decode.
  std::optional<std::string> drafter;
  bool speculation = true;
  // With an artifact: the tokens of conversation state its runner holds.
  std::uint32_t context = kDefaultContext;
  // With an artifact whose kept metadata has no tokenizer or chat template
  // (M3's Qwen3.8 import kept config.json only): the checkpoint's
  // tokenizer.json and chat template, absolute paths the runtime reads
  // under the same trust rules as the configuration.
  std::optional<std::filesystem::path> tokenizer;
  std::optional<std::filesystem::path> chat_template;
};

// The chat route's listener (M3's minimal /v1/chat/completions, D-097):
// `[client] bind`, "<address>:<port>" with a loopback address, IPv4 in
// 127.0.0.0/8 or IPv6 "[::1]". A non-loopback address needs the front
// door's authentication and transport protection (D-014, D-045; M5).
inline constexpr std::string_view kDefaultClientBind = "127.0.0.1:8114";

struct ClientEndpoint {
  std::string address;  // "127.0.0.1" or "::1": what inet_pton reads
  bool ipv6 = false;
  std::uint16_t port = 0;
};

// Parses a loopback bind ("127.0.0.1:8114", "[::1]:8114"); the error says
// what is wrong. The port is 1-65535.
std::expected<ClientEndpoint, std::string> ParseLoopbackBind(std::string_view text);

struct NodeConfig {
  std::optional<Membership> membership;
  // Where the service listens for the chat route (the default unless set).
  ClientEndpoint client{.address = "127.0.0.1", .ipv6 = false, .port = 8114};
  std::string limits_profile{kLimitsProfile};
  Storage storage;
  // Sorted by name.
  std::vector<ModelEntry> models;
  // The files the document was formed from, in the order read. Empty when
  // the built-in defaults apply.
  std::vector<std::filesystem::path> files;
};

// One problem, located where the document allows: a file (empty for the
// document as a whole) and a 1-based line and column (0 when unknown).
struct Diagnostic {
  std::string file;
  std::uint32_t line = 0;
  std::uint32_t column = 0;
  std::string message;
};

// "file:line:column: message", with what is unknown left out.
std::string FormatDiagnostic(const Diagnostic& diagnostic);

// One file's text and the name its diagnostics use.
struct SourceText {
  std::string name;
  std::string text;
};

// Parses, merges and validates files already read, in order; the anchor
// is the enrollment anchor's path, which no storage role may equal,
// contain or lie inside. No files at all yields the built-in defaults.
std::expected<NodeConfig, std::vector<Diagnostic>> ParseNodeConfig(
    std::span<const SourceText> files, const std::filesystem::path& anchor);

struct LoadOptions {
  // The main file; its drop-in directory is the same path with ".toml"
  // replaced by ".d". It must end in ".toml".
  std::filesystem::path main_file{kDefaultConfigFile};
  // Only the default main file may be absent (its fragments alone, or
  // nothing at all, then form the document; D-063): its reader says so.
  bool main_file_optional = false;
  std::filesystem::path anchor{kDefaultAnchor};
  // Besides root, the user whose files are trusted: the runtime's.
  uid_t trusted_uid = 0;
};

// Reads the main file and every regular "*.toml" file in the drop-in
// directory, in bytewise order, skipping other names and not recursing,
// then parses them as ParseNodeConfig does.
std::expected<NodeConfig, std::vector<Diagnostic>> LoadNodeConfig(const LoadOptions& options);

}  // namespace jitllm::config

#endif  // JITLLM_CONFIG_NODE_CONFIG_H_
