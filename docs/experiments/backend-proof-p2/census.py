#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Attributes a census run's memory growth under backend-proof.md's census rule.

A census run (jitllm-census/1 JSON: fp16_census.cc on the bridge, or the
native harness) holds readings at quiescent points and the rule's controls.
For each interval between readings, the unexplained bytes are

    drop in available memory - declared (catalog) growth - SUnreclaim's growth
                         - RssAnon's growth + declared bytes RssAnon also counts

(a byte both the catalog and RssAnon count is subtracted once), where
available memory is MemAvailable plus the pages on the per-CPU page lists
(/proc/zoneinfo), which MemAvailable leaves out (the amendment in
backend-proof.md's census rule; RE-024). Declared
bytes are the reading's own `declared` map. For the bridge, the buffers its
API cannot report are added from fp16-plan.json's record of the same arm:
the GGML pool's committed bytes after each chunk (the pool-peak build) and
the 32 MiB cuBLAS workspace, from the first chunk that calls cuBLAS. A
declared buffer counts in RssAnon if it is ordinary host memory (`host:`
or the bridge's `CPU` buffers), or pinned host memory (`CUDA_Host`, `pinned:`)
when the run's pinned probe moved RssAnon by at least half its size.

  census.py bridge RUN.json... --plan fp16-plan.json --arm ARM [--out SUMMARY.json]
  census.py native RUN.json [--caps fp16-f-caps.json --arm ARM] [--out SUMMARY.json]

`bridge` reports each run's controls, resolution R and cumulative
unexplained growth at every warm-up step, and the F cap: at each step, the
largest cumulative growth of the runs given. `native` does the same for a
native run and, with --caps, judges it (see backend-proof.md).
"""
import argparse
import json
import pathlib
import sys

CONTROL_BYTES = 64 << 20
CUBLAS_WORKSPACE = 32 << 20


class CensusError(ValueError):
    pass


def load(path):
    run = json.loads(pathlib.Path(path).read_text())
    if run.get("format") != "jitllm-census/1":
        raise CensusError(f"{path}: not a jitllm-census/1 run")
    return run


def available(reading):
    """MemAvailable corrected for the per-CPU page lists (backend-proof.md's census amendment)."""
    return reading["mem_available"] + reading["pcp"]


def controls(run):
    """Each control's moves and gaps, the resolution R and whether the run is void."""
    rows, void, pinned_rss = [], [], []
    for c in run["controls"]:
        b, h, a = c["before"], c["held"], c["after"]
        kind = c["kind"]
        moves = {}
        take = available(b) - available(h)
        give = available(a) - available(h)
        if kind == "vmm":  # net of the driver's per-extent bookkeeping
            take -= h["sunreclaim"] - b["sunreclaim"]
            give -= h["sunreclaim"] - a["sunreclaim"]
        moves["MemAvailable"] = (take, give)
        if kind in ("host", "pinned-probe"):
            moves["RssAnon"] = (h["rss_anon"] - b["rss_anon"], h["rss_anon"] - a["rss_anon"])
        gaps = {k: max(abs(CONTROL_BYTES - m) for m in v) for k, v in moves.items()}
        row = {"kind": kind, "repeat": c["repeat"], "where": c["where"], "moves": moves, "gaps": gaps}
        rows.append(row)
        if kind == "pinned-probe":
            pinned_rss.append(moves["RssAnon"][0] >= CONTROL_BYTES // 2)
            continue
        for counter, (t, g) in moves.items():
            if t < CONTROL_BYTES // 2 or g < CONTROL_BYTES // 2:
                void.append(f"{kind} repeat {c['repeat']} ({c['where']}): {counter} moved {t:+} / {g:+}")
    resolution = max((max(r["gaps"].values()) for r in rows if r["kind"] != "pinned-probe"), default=None)
    if pinned_rss and len(set(pinned_rss)) != 1:
        void.append("the pinned probe moved RssAnon in some repeats and not others")
    return rows, resolution, void, bool(pinned_rss and pinned_rss[0])


def in_rss(name, pinned_rss):
    if name.startswith("host:") or name.endswith(":CPU"):
        return True
    return pinned_rss and (name.startswith("pinned:") or name.endswith(":CUDA_Host"))


def bridge_declared(run, record, arm_name):
    """The bridge's declared buffers at each reading, with the record's pool and cuBLAS workspace."""
    arm = next((a for a in record["arms"] if a["arm"] == arm_name), None)
    if arm is None:
        raise CensusError(f"fp16-plan.json has no arm {arm_name}")
    if run["chunks"] != arm["summary"]["chunks"]:
        raise CensusError("the run's chunks are not the arm's")
    if run["logits_sha256"] != arm["expected_logits_sha256"]:
        raise CensusError(f"the run's logits {run['logits_sha256'][:16]} are not the arm's "
                          f"{arm['expected_logits_sha256'][:16]}: its counts are not the bridge's")
    sequences = {s["id"]: s for s in arm["sequences"]}
    chunks = arm["chunks_first_evaluation"]
    first_cublas = next(c["chunk"] for c in chunks if sequences[c["sequence"]]["cublas_calls"])
    # Model buffers are allocated by the model load; the API reports them
    # once a context exists.
    model = next({k: v for k, v in r["declared"].items() if k.startswith("model:")}
                 for r in run["readings"] if any(k.startswith("model:") for k in r["declared"]))
    out, loaded = [], False
    for r in run["readings"]:
        declared = dict(r["declared"])
        loaded = loaded or r["step"] == "model"
        if loaded:
            declared.update(model)
        if r["step"] == "chunk":
            declared["pool:CUDA0"] = chunks[r["chunk"]]["pool_committed_after"]
            if r["chunk"] >= first_cublas:
                declared["cublas-workspace:CUDA0"] = CUBLAS_WORKSPACE
        out.append(declared)
    return out, first_cublas


def attribute(run, declared, pinned_rss):
    """Each interval's terms and the cumulative unexplained growth after it."""
    readings = run["readings"]
    rows, cumulative = [], 0
    for i in range(1, len(readings)):
        p, r = readings[i - 1], readings[i]
        dp, dr = declared[i - 1], declared[i]
        names = set(dp) | set(dr)
        catalog = sum(dr.get(n, 0) - dp.get(n, 0) for n in names)
        overlap = sum(dr.get(n, 0) - dp.get(n, 0) for n in names if in_rss(n, pinned_rss))
        drop = available(p) - available(r)
        slab = r["sunreclaim"] - p["sunreclaim"]
        rss = r["rss_anon"] - p["rss_anon"]
        unexplained = drop - catalog - slab - rss + overlap
        cumulative += unexplained
        rows.append({"step": r["step"], "evaluation": r.get("evaluation", 0), "chunk": r["chunk"], "drop": drop, "catalog": catalog, "sunreclaim": slab,
                     "rss_anon": rss, "counted_twice": overlap, "unexplained": unexplained,
                     "cumulative": cumulative})
    return rows


def mib(n):
    return f"{n / 1048576:+.2f}"


def summarize(path, run, declared):
    control_rows, resolution, void, pinned_rss = controls(run)
    rows = attribute(run, declared, pinned_rss)
    return {"run": str(path), "trajectory": run["trajectory"], "fusion": run["fusion"],
            "logits_sha256": run["logits_sha256"], "resolution": resolution, "void": void,
            "pinned_in_rss_anon": pinned_rss, "controls": control_rows, "intervals": rows}


def warmup_steps(summary):
    """(label, cumulative) at each warm-up step: every interval of the first evaluation."""
    return [(f"{r['step']}{'' if r['chunk'] < 0 else ' ' + str(r['chunk'])}", r["cumulative"])
            for r in summary["intervals"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bridge")
    b.add_argument("runs", nargs="+")
    b.add_argument("--plan", required=True)
    b.add_argument("--arm", required=True)
    b.add_argument("--out")
    b.add_argument("--caps-out", help="the committed caps record: this arm's entry is added or replaced")
    b.add_argument("--note", action="append", default=[], help="provenance for the caps record")
    n = sub.add_parser("native")
    n.add_argument("run")
    n.add_argument("--caps")
    n.add_argument("--arm")
    n.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "bridge":
            record = json.loads(pathlib.Path(a.plan).read_text())
            summaries = []
            for path in a.runs:
                run = load(path)
                declared, first_cublas = bridge_declared(run, record, a.arm)
                summaries.append(summarize(path, run, declared))
            steps = [warmup_steps(s) for s in summaries]
            labels = [label for label, _ in steps[0]]
            if any([label for label, _ in s] != labels for s in steps):
                raise CensusError("the runs' steps differ")
            caps = [{"step": label, "cap": max(s[i][1] for s in steps), "runs": [s[i][1] for s in steps]}
                    for i, label in enumerate(labels)]
            out = {"format": "jitllm-census-caps/1", "arm": a.arm, "first_cublas_chunk": first_cublas,
                   "resolution": max(s["resolution"] for s in summaries),
                   "void": [v for s in summaries for v in s["void"]], "caps": caps, "runs": summaries}
            if a.caps_out:
                write_caps(pathlib.Path(a.caps_out), a.arm, out, a.note)
            for s in summaries:
                print(f"{s['run']}: R {s['resolution']} bytes ({s['resolution'] / 1048576:.2f} MiB), "
                      f"pinned in RssAnon: {s['pinned_in_rss_anon']}, void: {s['void'] or 'no'}")
            for c in caps:
                if not c["step"].startswith("chunk") or c["step"] in ("chunk 0", "chunk 1", "chunk 2") \
                        or c is caps[-1]:
                    print(f"  {c['step']:>16}: cap {mib(c['cap'])} MiB  runs {[mib(x) for x in c['runs']]}")
        else:
            run = load(a.run)
            summary = summarize(a.run, run, [r["declared"] for r in run["readings"]])
            out = summary
            print(f"{a.run}: R {summary['resolution']} bytes, void: {summary['void'] or 'no'}")
            if a.caps:
                if not a.arm:
                    raise CensusError("--caps needs --arm")
                judged = judge(summary, json.loads(pathlib.Path(a.caps).read_text()), a.arm, run)
                out = dict(summary, judged=judged)
                for step in judged["steps"]:
                    if not step["step"].startswith("chunk") or step["excess"] or step is judged["steps"][-1]:
                        print(f"  F at {step['step']:>16}: native {mib(step['native'])} MiB, cap "
                              f"{mib(step['cap'])} MiB, excess {step['excess']}")
                w = judged["persistent_workspace"]
                print(f"  persistent library workspace {w['observed']} of {w['limit']}: "
                      f"{'pass' if w['pass'] else 'FAIL'}")
                print(f"  KV {judged['kv']['declared']} (layout {judged['kv']['layout']}): "
                      f"{'pass' if judged['kv']['pass'] else 'FAIL'}")
                kinds = {}
                for v in judged["phases"]:
                    kinds.setdefault((v["evaluation"], v["rows"], v["n_kv"]), []).append(v)
                for (e, rows, n_kv), vs in sorted(kinds.items()):
                    worst = max(vs, key=lambda v: v["observed"])
                    print(f"  evaluation {e}, {rows} rows at n_kv {n_kv} ({len(vs)} phases): largest "
                          f"{worst['observed']} (A {worst['A']} S {worst['S']} I {worst['I']} L {worst['L']} "
                          f"charged {worst['charged']}) of {worst['limit']}: "
                          f"{'pass' if all(v['pass'] for v in vs) else 'FAIL'}")
                print(f"  census: {'PASS' if judged['pass'] else 'FAIL'}")
                if not judged["pass"]:
                    if a.out:
                        pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
                    return 1
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
        return 1 if out.get("void") else 0
    except (CensusError, KeyError, StopIteration) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def write_caps(path, arm, out, notes):
    """The compact caps record kept in Git: each arm's cumulative cap at every warm-up step."""
    record = json.loads(path.read_text()) if path.exists() else {
        "format": "jitllm-fp16-f-caps/1",
        "description": "The FP16 bridge's cumulative unexplained memory growth at each warm-up step "
                       "(backend-proof.md's census rule): native F's cap. Bytes; each cap is the largest of "
                       "the runs listed.",
        "notes": [], "arms": {}}
    for note in notes:
        if note not in record["notes"]:
            record["notes"].append(note)
    record["arms"][arm] = {
        "first_cublas_chunk": out["first_cublas_chunk"],
        "runs": [{"logits_sha256": s["logits_sha256"], "resolution": s["resolution"]} for s in out["runs"]],
        "steps": [[c["step"], c["cap"], c["runs"]] for c in out["caps"]]}
    record["arms"] = dict(sorted(record["arms"].items()))
    path.write_text(json.dumps(record, indent=1) + "\n")


# backend-proof.md, "Memory and workspace": each FP16 phase kind's limit E,
# by (trajectory, rows, n_kv); KV's declared layout; the persistent library
# workspace's limit.
PHASE_LIMITS = {
    ("control", 32, 256): 48_972_288,
    ("control", 1, 256): 1_224_720,
    ("heldout", 16, 256): 19_595_520,
    ("heldout", 17, 256): 26_016_528,
    ("heldout", 1, 256): 1_224_720,
    ("heldout", 512, 768): 784_605_184,
    ("heldout", 1, 768): 1_226_768,
}
KV_BYTES = {"control": 6_291_456, "heldout": 12_582_912}
PERSISTENT_WORKSPACE = 33_554_432
# Native's readings aligned with the bridge's warm-up steps (the F cap's
# pre-registration): the context, the controls, the end of setup (the
# bridge's context creation) and each chunk of the first evaluation.
ALIGNED = {("context", 0, -1): "context", ("controls", 0, -1): "controls",
           ("setup", 0, -1): "context-created"}


def judge(summary, caps, arm, run):
    """The census rule applied to a native run: F against the bridge's caps, then every phase's E."""
    if arm not in caps["arms"]:
        raise CensusError(f"the caps record has no arm {arm}")
    cap_at = {label: cap for label, cap, _ in caps["arms"][arm]["steps"]}
    resolution = summary["resolution"]
    phases = {(p["evaluation"], p["chunk"]): p for p in run["phases"]}
    charges = {key: 0 for key in phases}
    persistent_excess, charged, steps = 0, 0, []
    for row in summary["intervals"]:
        key = (row["step"], row["evaluation"], row["chunk"])
        label = ALIGNED.get(key) or (f"chunk {row['chunk']}" if row["step"] == "chunk" and row["evaluation"] == 1
                                     else None)
        if label is not None:
            cap = cap_at[label]
            excess = max(0, row["cumulative"] - cap - charged)
            charged += excess
            if excess:
                if row["step"] == "chunk":
                    charges[(1, row["chunk"])] += excess
                else:
                    persistent_excess += excess
            steps.append({"step": label, "native": row["cumulative"], "cap": cap,
                          "F": min(row["cumulative"], cap), "excess": excess})
        elif row["evaluation"] >= 2 and row["unexplained"] > resolution:
            # Lazy growth after the warm-up, beyond R: to the phase it falls
            # in, or the next one when between two ("before" readings).
            charges[(row["evaluation"], row["chunk"])] += row["unexplained"]
    last = run["readings"][-1]["declared"]
    workspace = last.get("device:cublas-workspace", 0) + persistent_excess
    verdicts = []
    for (e, k), p in sorted(phases.items()):
        limit = PHASE_LIMITS.get((run["trajectory"], p["rows"], p["n_kv"]))
        observed = p["A"] + p["S"] + p["I"] + p["L"] + charges[(e, k)]
        verdicts.append({"evaluation": e, "chunk": k, "rows": p["rows"], "n_kv": p["n_kv"], "A": p["A"],
                         "S": p["S"], "I": p["I"], "L": p["L"], "charged": charges[(e, k)],
                         "observed": observed, "limit": limit,
                         "pass": limit is not None and observed <= limit})
    kv = last.get("device:kv")
    return {"resolution": resolution, "void": summary["void"], "steps": steps,
            "persistent_workspace": {"observed": workspace, "limit": PERSISTENT_WORKSPACE,
                                     "excess_before_the_first_phase": persistent_excess,
                                     "pass": workspace <= PERSISTENT_WORKSPACE},
            "kv": {"declared": kv, "layout": KV_BYTES[run["trajectory"]],
                   "pass": kv == KV_BYTES[run["trajectory"]]},
            "phases": verdicts,
            "pass": (not summary["void"] and workspace <= PERSISTENT_WORKSPACE
                     and kv == KV_BYTES[run["trajectory"]] and all(v["pass"] for v in verdicts))}


if __name__ == "__main__":
    sys.exit(main())
