#!/bin/bash
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -eu
source /home/pmeenan/.local/share/jitllm/interconnect/env.sh
exec "$OPAL_PREFIX/bin/orted" "$@"
