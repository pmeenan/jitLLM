#!/bin/sh
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Runs one FP16 arm of jitllm_fp16_exec (benchmarks/fp16_exec.cc) on a Spark
# for the FP16 Tier E gate, in the gate's order (backend-proof.md):
#   run_native.sh plan   BUILD ARTIFACT HARNESS WORK ARM
#   run_native.sh logits BUILD ARTIFACT HARNESS WORK ARM
#   run_native.sh census BUILD ARTIFACT HARNESS WORK ARM
# BUILD is a deployed cross build, ARTIFACT the installed FP16 fixture,
# HARNESS a copy of docs/experiments (backend-proof-p0 and -p2), WORK the
# working directory holding control-tokens.txt (the bridge's tokens.txt)
# and heldout-ids.i64le, and ARM one of control-fused, control-unfused,
# heldout-fused, heldout-unfused.
#
# plan:   the recorded run under nsys (--trace=cuda) with cuBLAS's and
#         cuBLASLt's logs, the binary's SASS hashes (cuobjdump), then
#         plan_compare.py convert and compare against fp16-plan.json. Exits
#         with plan_compare's status; only 0 lets the logits be compared.
# logits: the same run's logits SHA-256 against the arm's recorded one.
# census: a separate run with the census readings (no nsys, no logs, no
#         recording), attributed and judged by census.py; void (exit 4) if
#         another process of this user held an NVIDIA device meanwhile.
set -u
STAGE=$1 BUILD=$2 ARTIFACT=$3 HARNESS=$4 WORK=$5 ARM=$6
TRAJECTORY=${ARM%-*}
case ${ARM#*-} in fused) FUSION=on ;; unfused) FUSION=off ;; *) echo "unknown arm $ARM" >&2; exit 2 ;; esac
case $TRAJECTORY in
  control) TOKENS=$WORK/control-tokens.txt ;;
  heldout) TOKENS=$WORK/heldout-ids.i64le ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
BIN=$BUILD/benchmarks/jitllm_fp16_exec
CUOBJDUMP=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
P0=$HARNESS/backend-proof-p0
P2=$HARNESS/backend-proof-p2
OUT=$WORK/$ARM
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo "another process is using the GPU" >&2
  exit 3
fi
export CUDA_DISABLE_PTX_JIT=1
case $STAGE in
  plan)
    rm -rf "$OUT" && mkdir -p "$OUT"
    env CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas.log" \
        CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE="$OUT/cublaslt.log" \
      nsys profile --trace=cuda --sample=none --cpuctxsw=none --export=sqlite -o "$OUT/trace" \
      "$BIN" --artifact "$ARTIFACT" --trajectory "$TRAJECTORY" --tokens "$TOKENS" \
             --fusion $FUSION --out "$OUT/run" --record > "$OUT/run.log" 2>&1 || {
      echo "the run failed:" >&2; tail -5 "$OUT/run.log" >&2; exit 1; }
    "$CUOBJDUMP" -sass "$BIN" | python3 -B "$P0/fp16_plan.py" sass-hash --label executable > "$OUT/sass.jsonl"
    python3 -B "$P2/plan_compare.py" convert "$OUT/run/recording.jsonl" \
      --cublas-log "$OUT/cublas.log" --cublaslt-log "$OUT/cublaslt.log" \
      --sass "$OUT/sass.jsonl" --nsys "$OUT/trace.sqlite" --out "$OUT/native.json" || exit 1
    python3 -B "$P2/plan_compare.py" compare --reference "$P0/fp16-plan.json" --arm "$ARM" \
      --native "$OUT/native.json"
    ;;
  logits)
    python3 -B - "$P0/fp16-plan.json" "$ARM" "$OUT/run/summary.json" "$OUT/run/logits.f32le" <<'EOF'
import hashlib, json, sys
record, arm, summary, logits = sys.argv[1:]
want = next(a for a in json.load(open(record))["arms"] if a["arm"] == arm)["expected_logits_sha256"]
got = hashlib.sha256(open(logits, "rb").read()).hexdigest()
s = json.load(open(summary))
assert s["logits_sha256"] == got, "the summary's hash is not the file's"
print(f"{arm}: native {got}, bridge {want}: {'BIT-IDENTICAL' if got == want else 'DIFFERENT'}; "
      f"repeat differences {s['repeat_bit_differences']}")
sys.exit(0 if got == want and s["repeat_bit_differences"] == 0 else 1)
EOF
    ;;
  census)
    # No other GPU work may run (the census rule). A monitor, started first
    # so its own memory has settled before the harness's first reading,
    # looks every second for any other process of this user holding an
    # NVIDIA device (from /proc, without touching the driver or starting
    # processes); anything found voids the run.
    rm -rf "$OUT-census" && mkdir -p "$OUT-census"
    python3 -B - "$BIN" "$OUT-census/foreign-gpu.txt" "$OUT-census/done" <<'PY' &
import os, sys, time
binary, found, done = os.path.realpath(sys.argv[1]), sys.argv[2], sys.argv[3]
me = os.getpid()
while not os.path.exists(done):
    for pid in [p for p in os.listdir("/proc") if p.isdigit() and int(p) != me]:
        try:
            if os.path.realpath(f"/proc/{pid}/exe") == binary:
                continue
            fds = os.listdir(f"/proc/{pid}/fd")
            if any(os.readlink(f"/proc/{pid}/fd/{fd}").startswith("/dev/nvidia") for fd in fds):
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
                with open(found, "a") as out:
                    out.write(f"{time.strftime('%T')} {pid} {cmd[:160]}\n")
        except OSError:
            continue
    time.sleep(1)
PY
    MONITOR=$!
    sleep 3
    "$BIN" --artifact "$ARTIFACT" --trajectory "$TRAJECTORY" --tokens "$TOKENS" --fusion $FUSION \
           --out "$OUT-census" --census > "$OUT-census.log" 2>&1
    STATUS=$?
    touch "$OUT-census/done"
    wait $MONITOR
    [ $STATUS = 0 ] || { echo "the run failed:" >&2; tail -5 "$OUT-census.log" >&2; exit 1; }
    if [ -s "$OUT-census/foreign-gpu.txt" ]; then
      echo "VOID: other GPU work ran during the census:" >&2
      sort -u -k2,2 "$OUT-census/foreign-gpu.txt" | head -5 >&2
      exit 4
    fi
    python3 -B "$P2/census.py" native "$OUT-census/census.json" --caps "$P2/fp16-f-caps.json" \
      --arm "$ARM" --out "$OUT-census/judged.json"
    ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
