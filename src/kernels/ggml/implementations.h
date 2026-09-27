// SPDX-FileCopyrightText: 2026 jitLLM contributors
// SPDX-License-Identifier: Apache-2.0

// The GGML module's entries in the implementation registry (D-053;
// execution/registry.h), CUDA builds only: a build without the device code
// declares none, so a plan naming one is unsupported there (BP-S4).
//
// For backend-proof P1 the module declares the one operation it offers two
// implementations of, RMSNorm scaled by a weight:
//   ggml.rms_norm_mul.fused    GGML's fused launcher (ops.h RmsNormMul), as
//                              the FP16-F profile runs it;
//   ggml.rms_norm_mul.unfused  rms_norm's launcher, then mul's (ops.h
//                              RmsNormThenMul), as FP16-U runs it.
// Each identity covers everything that decides what an implementation
// computes and launches:
//   - the prepared GGML tree's digest, which covers upstream's bytes,
//     jitLLM's patches and the build of GGML's files;
//   - jitLLM's own code in this module: a digest of every file in
//     src/kernels/ggml, written at build time (module_digest.cmake), so any
//     edit here changes every identity the module declares;
//   - the SDK, target, device architecture, build type (NDEBUG, and with it
//     GGML's device asserts) and sanitizers;
//   - the name, and a variant naming the launcher sequence.
// Code outside the module, such as the provider that supplies the stream,
// is not covered: it does not choose what is launched.
// The other implementations in ops.h join when a planner selects them.
//
// A bound plan's implementation becomes a kernel here once, when the plan
// is bound; each launch then runs that kernel's host checks and launchers
// and looks nothing up.

#ifndef JITLLM_KERNELS_GGML_IMPLEMENTATIONS_H_
#define JITLLM_KERNELS_GGML_IMPLEMENTATIONS_H_

#include <expected>
#include <string_view>
#include <vector>

#include "execution/registry.h"
#include "ggml.h"
#include "kernels/ggml/launch.h"
#include "kernels/ggml/tensors.h"

namespace jitllm::kernels::ggml {

// What this module declares to the registry.
std::vector<execution::Implementation> Implementations();

// The digest of the module's own files, generated at build time.
std::string_view ModuleSourcesDigest();

// One GGML implementation of RMSNorm-mul, over a ggml_rms_norm node and
// the ggml_mul that scales it (ops.h).
class RmsNormMulKernel {
 public:
  // Refused unless `implementation` is one this module declares, identity
  // and all: a stale or foreign declaration never selects a kernel.
  static std::expected<RmsNormMulKernel, KernelFailure> Bind(
      const execution::Implementation& implementation);

  // The implementation's operand checks, on the host (validate.h).
  std::expected<void, KernelFailure> Check(const ggml_tensor* norm, const ggml_tensor* mul) const;
  // Checks, then launches on the context's stream.
  std::expected<void, KernelFailure> Run(LaunchContext& launch, ggml_tensor* norm,
                                         ggml_tensor* mul) const;
  std::string_view name() const;

  struct Entry;

 private:
  explicit RmsNormMulKernel(const Entry& entry) : entry_(&entry) {}

  const Entry* entry_;
};

}  // namespace jitllm::kernels::ggml

#endif  // JITLLM_KERNELS_GGML_IMPLEMENTATIONS_H_
