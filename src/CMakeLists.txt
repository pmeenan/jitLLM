# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0

# jitLLM's modules, lower layers first (docs/architecture.md#layers-and-dependency-rules).
# Sources include each other's headers as "<module>/<file>.h" from here.
add_subdirectory(base)
add_subdirectory(platform)
add_subdirectory(providers)
add_subdirectory(catalog)
add_subdirectory(memory)
add_subdirectory(scheduler)
add_subdirectory(config)
add_subdirectory(cli)
add_subdirectory(runtime)
