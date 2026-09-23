#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Upstream-reader conformance for built artifacts.

Run inside the pinned jitllm-exl3-reference image (safetensors 0.8.0, torch,
PYTHONPATH=/app/gguf-py) with artifacts and sources mounted read-only:

  conform.py ARTIFACT [SOURCE_GGUF]

Every shard must open with the upstream safetensors reader (which rejects
gaps, overlaps and size mismatches) and every entry, including pads, must
equal the file bytes the jitLLM index names. A source-metadata .kv.gguf must
parse with upstream gguf-py as a zero-tensor GGUF whose fields equal the
source checkpoint's fields.
"""
import json
from pathlib import Path
import sys

import safetensors
from safetensors import safe_open
import torch


def main(root, source=None):
    root = Path(root)
    index = json.loads((root / "index.json").read_bytes())
    out = {"safetensors": safetensors.__version__, "shards": [], "kv_gguf": None}
    for s in index["shards"]:
        path = root / s["path"]
        raw = path.read_bytes()
        n = 0
        with safe_open(str(path), framework="pt") as f:
            header = json.loads(raw[8:8 + int.from_bytes(raw[:8], "little")])
            header.pop("__metadata__")
            for name in f.keys():
                a, b = header[name]["data_offsets"]
                got = f.get_tensor(name).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                if got != raw[s["data_offset"] + a:s["data_offset"] + b]:
                    raise SystemExit(f"byte mismatch {name}")
                n += 1
        out["shards"].append({"path": s["path"], "entries_read": n})
    if source:
        from gguf import GGUFReader
        kv = next(root.glob("meta/*.kv.gguf"))
        mine, theirs = GGUFReader(str(kv)), GGUFReader(source)
        if len(mine.tensors) != 0 or list(mine.fields) != list(theirs.fields) \
                or int(mine.fields["GGUF.tensor_count"].parts[-1][0]) != 0:
            raise SystemExit("kv.gguf fields differ")
        for key in theirs.fields:
            if key == "GGUF.tensor_count":
                continue
            a, b = mine.fields[key], theirs.fields[key]
            if a.types != b.types or [bytes(p) for p in a.parts] != [bytes(p) for p in b.parts]:
                raise SystemExit(f"kv.gguf field {key} differs")
        out["kv_gguf"] = {"fields": len(mine.fields), "tensors": len(mine.tensors)}
    print(json.dumps(out))


if __name__ == "__main__":
    main(*sys.argv[1:])
