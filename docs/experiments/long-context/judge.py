# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Judges jitLLM at depth against its oracle (README.md, "Correctness").

    judge.py inputs ORACLE.json OUTDIR NAME
        The oracle record's prompt IDs and greedy tokens as harness input
        lines (NAME.prompt.tsv, NAME.force.tsv) for jitllm_dsv4_exec or
        jitllm_qwen38_exec --prompts ... --force ... --generate N.
    judge.py noise A B NAME --vocab V
        The near-tie bound, recorded before the comparison: two of jitLLM's
        own paths forced on the same tokens (its default fast plan and its
        reference form); at each step the change in A's top-two logit margin
        when B scores the same two tokens; the bound is the 99th percentile.
    judge.py greedy ORACLE.json HARNESS NAME --vocab V --bound B
        jitLLM forced on the oracle's greedy tokens: at each step its argmax
        is the oracle's token, or the oracle's log-probability margin between
        its token and jitLLM's argmax is below B (a near-tie; when jitLLM's
        argmax is outside the oracle's top list, the margin is at least the
        oracle token's lead over the list's last). Every exception is listed.
    judge.py repeat A B NAME --vocab V
        Two runs of the same forced prompt: the first step whose logits
        differ, and the largest difference (RE-031).
    judge.py ppl ORACLE HARNESS --ctx L [--within F]
        Perplexity over the window's second half (tokens L/2+1 .. L-1):
        ORACLE is a vLLM ppl-L.nll.json or a number (llama-perplexity's),
        HARNESS the jitLLM harness's ppl.nll.f64. Within F (default 0.03).

Runs on a Spark in a container whose Python has NumPy (the pinned PyTorch
image). Prints one JSON document; exits 1 when a bound fails.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_logits(directory, name, vocab):
    a = np.fromfile(Path(directory) / f"{name}.logits.f32", dtype=np.float32)
    if a.size % vocab:
        raise SystemExit(f"{directory}/{name}: {a.size} logits are not whole rows of {vocab}")
    if not np.all(np.isfinite(a)):
        raise SystemExit(f"{directory}/{name}: non-finite logits")
    return a.reshape(-1, vocab).astype(np.float64)


def arg(flag, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def inputs(oracle_path, outdir, name):
    record = json.loads(Path(oracle_path).read_text())
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.prompt.tsv").write_text(
        name + "\t" + " ".join(map(str, record["prompt_ids"])) + "\n")
    (out / f"{name}.force.tsv").write_text(name + "\t" + " ".join(map(str, record["ids"])) + "\n")
    return {"prompt_tokens": len(record["prompt_ids"]), "generated": len(record["ids"])}


def noise(a_dir, b_dir, name, vocab):
    a, b = load_logits(a_dir, name, vocab), load_logits(b_dir, name, vocab)
    steps = min(len(a), len(b))
    moves = []
    for k in range(steps):
        top = np.argsort(a[k])[-2:][::-1]
        margin_a = a[k][top[0]] - a[k][top[1]]
        margin_b = b[k][top[0]] - b[k][top[1]]
        moves.append(abs(margin_a - margin_b))
    moves = np.array(moves)
    return {"steps": steps, "p50": float(np.percentile(moves, 50)),
            "p99": float(np.percentile(moves, 99)), "max": float(moves.max()),
            "argmax_equal": int(sum(np.argmax(a[k]) == np.argmax(b[k]) for k in range(steps)))}


def greedy(oracle_path, harness, name, vocab, bound):
    record = json.loads(Path(oracle_path).read_text())
    logits = load_logits(harness, name, vocab)
    steps = min(len(logits), len(record["steps"]))
    agree, near, violations = 0, [], []
    for k in range(steps):
        step = record["steps"][k]
        mine = int(np.argmax(logits[k]))
        if mine == step["id"]:
            agree += 1
            continue
        top = {int(t): float(v) for t, v in step["top"]}
        lead = step["logprob"] - top.get(mine, min(top.values()) if top else -math.inf)
        entry = {"step": k, "oracle": step["id"], "jitllm": mine, "oracle_margin": lead,
                 "in_top": mine in top}
        (near if lead < bound and mine in top else violations).append(entry)
    return {"steps": steps, "agree": agree, "near_ties": near, "violations": violations,
            "bound": bound, "pass": not violations}


def repeat(a_dir, b_dir, name, vocab):
    a, b = load_logits(a_dir, name, vocab), load_logits(b_dir, name, vocab)
    steps = min(len(a), len(b))
    differ = [k for k in range(steps) if not np.array_equal(a[k], b[k])]
    return {"steps": steps, "identical": not differ, "first_differing_step":
            differ[0] if differ else None, "steps_differing": len(differ),
            "max_abs_difference": float(np.abs(a[:steps] - b[:steps]).max())}


def ppl(oracle, harness, ctx, within):
    mine = np.fromfile(harness, dtype=np.float64)
    half = mine[ctx // 2:ctx - 1]
    ours = math.exp(float(half.mean()))
    if Path(oracle).exists():
        theirs_nll = np.array(json.loads(Path(oracle).read_text()))[ctx // 2:ctx - 1]
        theirs = math.exp(float(theirs_nll.mean()))
    else:
        theirs = float(oracle)
    ratio = ours / theirs
    return {"window": ctx, "scored": int(half.size), "jitllm_ppl": ours, "oracle_ppl": theirs,
            "ratio": ratio, "within": within, "pass": abs(ratio - 1) <= within}


def main():
    command = sys.argv[1]
    vocab = arg("--vocab", 0, int)
    if command == "inputs":
        result = inputs(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "noise":
        result = noise(sys.argv[2], sys.argv[3], sys.argv[4], vocab)
    elif command == "greedy":
        result = greedy(sys.argv[2], sys.argv[3], sys.argv[4], vocab, arg("--bound", 1.0, float))
    elif command == "repeat":
        result = repeat(sys.argv[2], sys.argv[3], sys.argv[4], vocab)
    elif command == "ppl":
        result = ppl(sys.argv[2], sys.argv[3], arg("--ctx", 0, int), arg("--within", 0.03, float))
    else:
        raise SystemExit(__doc__)
    print(json.dumps(result, indent=1))
    if result.get("pass") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
