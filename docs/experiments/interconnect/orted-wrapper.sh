#!/bin/bash
set -eu
source /home/pmeenan/.local/share/jitllm/interconnect/env.sh
exec "$OPAL_PREFIX/bin/orted" "$@"
