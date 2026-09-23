#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Reference model of the experimental v0 prepared-artifact layout.

Plans, writes and verifies jitLLM paging artifacts from GGUF or safetensors
checkpoints under the rules in docs/artifact-format.md. Stdlib only. This is
M0 design evidence and the executable oracle for the M3 C++ importer and
verifier. verify() treats the artifact as untrusted input and fails closed
with a named rule; the M3 verifier must match its accept/reject behaviour.

  plan [--tie-check] SOURCE...        layout statistics as JSON (headers only)
  build OUT SOURCE... [--meta=FILE]  write, verify and atomically publish under OUT
  verify ARTIFACT                    full verification (schema, bounds, hashes)
  loadcheck ARTIFACT SOURCE...       closure reads with O_DIRECT versus source bytes
  coldload ARTIFACT [MAX_RUN_MIB]    timed whole-artifact coalesced vectored load
"""

import collections
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import stat
import struct
import sys

FILE_ALIGN = 4096      # group start/length granularity on disk (O_DIRECT)
CHUNK = 2 << 20        # group-relative paging/backing chunk (D-033 handle size)
MEMBER_ALIGN = 256     # resource alignment inside a group
FORMAT = "jitllm-artifact"
INDEX_FORMAT = "jitllm-index"
FORMAT_VERSION = 0
PROFILE = "spark-v0"
PROFILE_VERSION = 0
SHARD_TARGET = 4 << 30
MAX_ST_HEADER = 100_000_000  # upstream safetensors reader limit
MAX_DOC = 64 << 20           # index.json size limit
MAX_MANIFEST = 1 << 20       # manifest.json size limit (real manifests are a few KB)
MAX_JSON_DEPTH = 8           # deepest real document nests 4 levels

MAX_INT = (1 << 63) - 1
MAX_ENTRIES = 1 << 18        # resources + expert slices per artifact (Qwen3.8 needs ~75k)
MAX_CONTAINERS = 1 << 19     # JSON arrays + objects per document; Qwen3.8's index has ~29k
MAX_VALUES = 1 << 21         # JSON separators (, and :) per document; Qwen3.8's index has ~460k
MAX_ARCH = 200               # general.architecture length (the NAME pattern's limit)
MAX_ALIASES = 8              # alias roles per resource (real ties add one)
MAX_META_TOTAL = 128 << 20   # all kept metadata together (EXL3 fixtures: ~12 MB; Gemma kv: 15.8 MB)
MAX_META = 64 << 20          # kept metadata file limit (largest real vocab section: 15.8 MB)
PAD_PREFIX = "~pad."
PAD_NAME = re.compile(r"~pad\.(0|[1-9][0-9]{0,9})")
NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,199}")   # resource/role names
SLICE_NAME = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_.\-]{0,199})#(0|[1-9][0-9]{0,6})")
HEX = re.compile(r"[0-9a-f]{64}")
SHARD_PATH = re.compile(r"data/[0-9]{5}\.safetensors")
META_PATH = re.compile(r"meta/[A-Za-z0-9_][A-Za-z0-9_.\-]{0,199}")
RESERVED = {"__metadata__"}
KINDS = {"table", "layer", "expert", "global", "head"}
FAMILIES = {"ggml", "exl3", "plain"}
GGML_ROW_PADDING = 512  # ggml-cuda common.cuh MATRIX_ROW_PADDING at the pin
EXL3_SUPPORTED = {"mcg": {4, 5, 6, 8}}  # codebook -> K rates with fixtures (exl3-bringup.md)

# GGML type id -> (name, elements per block, bytes per block). Pinned from
# llama.cpp b29c606e2 gguf-py/gguf/constants.py GGML_QUANT_SIZES.
QK_K = 256
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40),
    10: ("Q2_K", 256, 2 + 2 + QK_K // 16 + QK_K // 4),
    11: ("Q3_K", 256, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: ("Q4_K", 256, 2 + 2 + QK_K // 2 + 12),
    13: ("Q5_K", 256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: ("Q6_K", 256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    15: ("Q8_K", 256, 4 + QK_K + QK_K // 8),
    16: ("IQ2_XXS", 256, 2 + QK_K // 4), 17: ("IQ2_XS", 256, 2 + QK_K // 4 + QK_K // 32),
    18: ("IQ3_XXS", 256, 2 + QK_K // 4 + QK_K // 8),
    19: ("IQ1_S", 256, 2 + QK_K // 8 + QK_K // 16), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: ("IQ2_S", 256, 2 + QK_K // 4 + QK_K // 16),
    23: ("IQ4_XS", 256, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 2 + 4 * 13), 35: ("TQ2_0", 256, 2 + 64),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
GGML_BY_NAME = {v[0]: v for v in GGML_TYPES.values()}
ST_SIZES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2,
            "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
ST_NATIVE = {"F32", "F16", "BF16", "I8", "I16", "I32", "I64", "F64"}  # ggml names == st names

EXPERT_SLICED = re.compile(r"blk\.(\d+)\.ffn_(gate|up|gate_up|down)_exps\.weight")
LAYER = re.compile(r"(?:blk|model\.layers)\.(\d+)\.")
ROW_TABLES = {"token_embd.weight", "per_layer_token_embd.weight", "model.embed_tokens.weight"}
HEADS = re.compile(r"output\.weight|lm_head\..+")
EXL3_PART = re.compile(r"(.+)\.(trellis|suh|svh|su|sv|mcg|mul1|bias)")


class ArtifactError(Exception):
    """Verification failure; `code` names the violated rule."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code


def align(n, a):
    return (n + a - 1) // a * a


# ---------------------------------------------------------------- strict typed checks

def _is_int(v):
    return type(v) is int


def _int(v, where, lo=0, hi=MAX_INT):
    if not _is_int(v) or not lo <= v <= hi:
        raise ArtifactError("schema", f"{where}: expected integer in [{lo}, {hi}]")
    return v


def _str(v, where, pattern=None):
    if type(v) is not str or (pattern is not None and not pattern.fullmatch(v)):
        raise ArtifactError("schema", f"{where}: bad string {v!r}")
    return v


def _list(v, where, lo=0):
    if type(v) is not list or len(v) < lo:
        raise ArtifactError("schema", f"{where}: expected list")
    return v


def _obj(v, keys, where, optional=()):
    if type(v) is not dict or not set(keys) <= set(v) or set(v) - set(keys) - set(optional):
        got = sorted(v) if type(v) is dict else type(v).__name__
        raise ArtifactError("schema", f"{where}: keys {got}")
    return v


def _exact(v, want):
    """Equality that also requires identical JSON types (False != 0, 4096.0 != 4096)."""
    if type(v) is not type(want):
        return False
    if type(want) is dict:
        return set(v) == set(want) and all(_exact(v[k], want[k]) for k in want)
    if type(want) is list:
        return len(v) == len(want) and all(_exact(a, b) for a, b in zip(v, want))
    return v == want


def strict_json(data, where="json"):
    """Parse strict UTF-8 JSON (no BOM, no floats, no non-finite, no duplicate keys)."""
    if type(data) is not bytes:
        raise ArtifactError("json", f"{where}: expected bytes")
    if not data.isascii():  # canonical documents are ASCII (dumps escapes everything else)
        raise ArtifactError("json", f"{where}: non-ASCII bytes")

    def pairs(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)):
            raise ArtifactError("json", f"{where}: duplicate key")
        return dict(items)

    def no_float(s):
        raise ArtifactError("json", f"{where}: non-integer number {s[:20]}")

    def bounded_int(s):
        if len(s) > 20:
            raise ArtifactError("json", f"{where}: integer too long")
        return int(s)

    _prescan(data, where)
    try:
        text = data.decode("utf-8", errors="strict")
        return json.loads(text, object_pairs_hook=pairs, parse_float=no_float,
                          parse_int=bounded_int, parse_constant=no_float)
    except ArtifactError:
        raise
    except (UnicodeError, ValueError, RecursionError) as e:
        raise ArtifactError("json", f"{where}: {type(e).__name__}") from e


# A JSON string, or an unterminated one running to the end: once started the
# match cannot fail, so finditer never retries inside it (linear time).
_JSON_STRING = re.compile(rb'"(?:[^"\\]++|\\.)*+"?', re.DOTALL)
_JSON_BRACKET = re.compile(rb"[\[\]{}]")
MAX_STRINGS = MAX_VALUES + MAX_CONTAINERS + 1  # valid JSON: strings <= separators + containers + 1


def _prescan(data, where):
    """Linear, copy-free scan before parsing: count containers, separators,
    strings and bracket depth outside strings, failing as soon as a cap is
    exceeded, so a small hostile document cannot make the parser (or this
    scan) build millions of objects."""
    containers = values = strings = depth = 0

    def gap(lo, hi):
        nonlocal containers, values, depth
        containers += data.count(b"[", lo, hi) + data.count(b"{", lo, hi)
        values += data.count(b",", lo, hi) + data.count(b":", lo, hi)
        if containers > MAX_CONTAINERS:
            raise ArtifactError("json", f"{where}: more than {MAX_CONTAINERS} arrays/objects")
        if values > MAX_VALUES:
            raise ArtifactError("json", f"{where}: more than {MAX_VALUES} values")
        for m in _JSON_BRACKET.finditer(data, lo, hi):
            depth += 1 if m[0] in b"[{" else -1
            if depth > MAX_JSON_DEPTH or depth < 0:
                raise ArtifactError("json", f"{where}: nesting deeper than {MAX_JSON_DEPTH} or unbalanced")

    pos = 0
    for m in _JSON_STRING.finditer(data):
        gap(pos, m.start())
        strings += 1
        if strings > MAX_STRINGS:
            raise ArtifactError("json", f"{where}: more than {MAX_STRINGS} strings")
        pos = m.end()
    gap(pos, len(data))


def dumps(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()


# ---------------------------------------------------------------- representations

def ggml_row_bytes(type_name, elems):
    _, blck, tsize = GGML_BY_NAME[type_name]
    return elems // blck * tsize


def repr_bytes(rep, where):
    """Validate a representation descriptor and return the byte size it implies."""
    if type(rep) is not dict:
        raise ArtifactError("repr", f"{where}: not an object")
    family = rep.get("family")
    if type(family) is not str:
        raise ArtifactError("repr", f"{where}: family must be a string")
    if family == "ggml":
        _obj(rep, {"family", "type", "ne"}, where)
        if type(rep["type"]) is not str or rep["type"] not in GGML_BY_NAME:
            raise ArtifactError("repr", f"{where}: unknown ggml type {rep['type']!r}")
        ne = _list(rep["ne"], where, 1)
        if len(ne) > 4 or not all(_is_int(n) and 1 <= n <= MAX_INT for n in ne):
            raise ArtifactError("repr", f"{where}: bad ne")
        _, blck, _ = GGML_BY_NAME[rep["type"]]
        if ne[0] % blck:
            raise ArtifactError("repr", f"{where}: row not a multiple of the block")
        return ggml_row_bytes(rep["type"], ne[0]) * math.prod(ne[1:])
    if family in ("exl3", "plain"):
        extra = {"k_bits", "in_features", "out_features", "codebook"} if rep.get("role") == "trellis" else set()
        _obj(rep, {"family", "dtype", "shape"} | ({"role"} | extra if family == "exl3" else set()), where)
        if type(rep["dtype"]) is not str or rep["dtype"] not in ST_SIZES:
            raise ArtifactError("repr", f"{where}: dtype {rep['dtype']!r}")
        if family == "exl3" and (type(rep["role"]) is not str or type(rep.get("codebook", "")) is not str):
            raise ArtifactError("repr", f"{where}: role/codebook must be strings")
        shape = _list(rep["shape"], where)
        if len(shape) > 8 or not all(_is_int(n) and 0 <= n <= MAX_INT for n in shape):
            raise ArtifactError("repr", f"{where}: bad shape")
        n = math.prod(shape) * ST_SIZES[rep["dtype"]]
        if family == "exl3":
            role = rep["role"]
            if role == "trellis":
                k = rep["k_bits"]
                ok = (rep["dtype"] == "I16" and len(shape) == 3 and _is_int(k)
                      and rep["codebook"] in EXL3_SUPPORTED and k in EXL3_SUPPORTED[rep["codebook"]]
                      and shape[2] == 16 * k and _exact(rep["in_features"], 16 * shape[0])
                      and _exact(rep["out_features"], 16 * shape[1]))
            elif role in ("suh", "svh"):
                ok = rep["dtype"] == "F16" and len(shape) == 1
            elif role == "mcg":
                ok = rep["dtype"] == "I32" and shape == []
            else:
                ok = False  # su/sv packed signs, mul1 and 3inst need their own fixtures
            if not ok:
                raise ArtifactError("repr", f"{where}: unsupported or inconsistent EXL3 {role!r}")
        if n == 0:
            raise ArtifactError("repr", f"{where}: empty tensor")
        return n
    raise ArtifactError("repr", f"{where}: family {family!r}")


def exl3_closure_errors(reprs, first_only=False):
    """EXL3 dependency completeness over {name: repr}. Every trellis needs its
    full-length suh (in_features) and svh (out_features) and its codebook
    marker; side vectors and markers never appear without their trellis."""
    errors, prefixes = [], {}
    for name, rep in reprs.items():
        m = EXL3_PART.fullmatch(name)
        if m and m[2] != "bias" and rep.get("family") != "exl3":
            errors.append(f"{name}: EXL3 part name with family {rep.get('family')!r}")
            if first_only:
                return errors
        if rep.get("family") == "exl3":
            m = EXL3_PART.fullmatch(name)
            if not m or m[2] != rep.get("role"):
                errors.append(f"{name}: EXL3 role does not match its name")
                continue
            prefixes.setdefault(m[1], {})[m[2]] = rep
    for prefix, parts in prefixes.items():
        if first_only and errors:
            return errors
        t = parts.get("trellis")
        if t is None:
            errors.append(f"{prefix}: side vectors or marker without a trellis")
            continue
        want = {"trellis", "suh", "svh", t.get("codebook")}
        if set(parts) != want:
            errors.append(f"{prefix}: EXL3 parts {sorted(parts)} but need {sorted(want)}")
        elif parts["suh"]["shape"] != [t["in_features"]] or parts["svh"]["shape"] != [t["out_features"]]:
            errors.append(f"{prefix}: side-vector lengths disagree with the trellis")
    return errors


def readable_for(rep, nbytes):
    """Bytes a backend may read. Pinned GGML CUDA pads quantized rows whose ne0
    is not a multiple of 512 and reads that zeroed padding (ggml-cuda.cu:917,
    763); EXL3 over-read is not yet established (M2 proof), so none is added."""
    if rep["family"] == "ggml":
        _, blck, _ = GGML_BY_NAME[rep["type"]]
        ne0 = rep["ne"][0]
        if blck > 1 and ne0 % GGML_ROW_PADDING:
            return nbytes + ggml_row_bytes(rep["type"], GGML_ROW_PADDING - ne0 % GGML_ROW_PADDING)
    return nbytes


def st_view(rep, nbytes):
    """Container dtype and shape for a resource (informational; the index rules)."""
    if rep["family"] == "ggml":
        if rep["type"] in ST_NATIVE:
            return rep["type"], list(reversed(rep["ne"]))
        return "U8", [nbytes]
    return rep["dtype"], rep["shape"]


# ---------------------------------------------------------------- sources (importer input)

class Reader:
    """Bounded header reader. With a hasher, every consumed byte (skipped ones
    included) is hashed, so the digest covers exactly the bytes interpreted."""

    def __init__(self, f, limit, hasher=None):
        self.f, self.limit, self.hasher = f, limit, hasher

    def take(self, n):
        if n > self.limit - self.f.tell():
            raise ValueError("truncated or oversized header")
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError("truncated header")
        if self.hasher:
            self.hasher.update(data)
        return data

    def skip(self, n):
        if n > self.limit - self.f.tell():
            raise ValueError("truncated or oversized header")
        if not self.hasher:
            self.f.seek(n, os.SEEK_CUR)
            return
        while n:
            self.take(min(n, 1 << 20))
            n -= min(n, 1 << 20)

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))

    def string(self):
        (n,) = self.unpack("Q")
        return self.take(n).decode("utf-8")


_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _gguf_value(r, t, keep, nested=False):
    """One GGUF value. Arrays of arrays are invalid (as in pinned gguf.cpp);
    arrays that are not kept are skipped without materializing them."""
    if t == 8:
        if keep:
            return r.string()
        (n,) = r.unpack("Q")
        r.skip(n)
        return None
    if t == 9:
        if nested:
            raise ValueError("GGUF array of arrays")
        (et,), (n,) = r.unpack("I"), r.unpack("Q")
        if et not in _SCALAR and et != 8:
            raise ValueError(f"bad GGUF array element type {et}")
        if keep and n <= 64:
            return [_gguf_value(r, et, True, True) for _ in range(n)]
        if et in _SCALAR:
            r.skip(n * struct.calcsize("<" + _SCALAR[et]))
        elif et == 8:
            for _ in range(n):
                _gguf_value(r, 8, False, True)
        else:
            raise ValueError(f"bad GGUF array type {et}")
        return {"array_len": n}
    if t not in _SCALAR:
        raise ValueError(f"bad GGUF value type {t}")
    return r.unpack(_SCALAR[t])[0]


def read_gguf(path):
    """Bounded GGUF header read: metadata, tensor table, KV-section end."""
    size = os.path.getsize(path)
    header_hash = hashlib.sha256()
    with open(path, "rb") as f:
        r = Reader(f, size, header_hash)
        if r.take(4) != b"GGUF":
            raise ValueError("not GGUF")
        (version,) = r.unpack("I")
        n_tensors, n_kv = r.unpack("QQ")
        if version != 3 or n_tensors > 1 << 20 or n_kv > 1 << 16:
            raise ValueError("unsupported or implausible GGUF header")
        meta, keys = {}, set()
        for _ in range(n_kv):
            key = r.string()
            (t,) = r.unpack("I")
            meta[key] = _gguf_value(r, t, True)
            _check_gguf_key(key, t, meta[key], keys)
        kv_end = f.tell()
        infos = []
        for _ in range(n_tensors):
            name = r.string()
            (nd,) = r.unpack("I")
            if nd > 4:
                raise ValueError("too many dims")
            ne = list(r.unpack("Q" * nd))
            (ty,), (off,) = r.unpack("I"), r.unpack("Q")
            if ty not in GGML_TYPES:
                raise ValueError(f"unknown ggml type {ty}")
            infos.append((name, ne, ty, off))
        header_len = f.tell()
        alignment = meta.get("general.alignment", 32)
        if not _is_int(alignment) or alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be a positive power of two")
        data = align(f.tell(), alignment)
    tensors = []
    for name, ne, ty, off in infos:
        tname, blck, tsize = GGML_TYPES[ty]
        if not ne or not all(ne) or ne[0] % blck:
            raise ValueError(f"{name}: empty or row not a multiple of block")
        nbytes = ne[0] // blck * tsize * math.prod(ne[1:])
        if data + off + nbytes > size:
            raise ValueError(f"{name}: outside file")
        tensors.append(dict(name=name, family="ggml", dtype=tname, ne=ne, nbytes=nbytes,
                            path=str(path), offset=data + off, block=tsize,
                            row_bytes=ne[0] // blck * tsize))
    ordered = sorted(tensors, key=lambda t: t["offset"])
    for a, b in zip(ordered, ordered[1:]):
        if a["offset"] + a["nbytes"] > b["offset"]:
            raise ValueError(f"{a['name']} overlaps {b['name']}")
    return dict(kind="gguf", meta=meta, tensors=tensors, kv_end=kv_end, n_kv=n_kv, path=str(path),
                header_len=header_len, header_sha256=header_hash.hexdigest())


def read_safetensors(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError("truncated safetensors")
        (n,) = struct.unpack("<Q", head)
        if n > MAX_ST_HEADER or 8 + n > size:
            raise ValueError("safetensors header size")
        raw = f.read(n)
    header_sha256 = hashlib.sha256(head + raw).hexdigest()
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as e:
        raise ValueError(f"safetensors header: {e}") from e
    if type(header) is not dict:
        raise ValueError("safetensors header is not an object")
    header.pop("__metadata__", None)
    tensors = []
    for name, info in header.items():
        ok = (type(info) is dict and info.get("dtype") in ST_SIZES and type(info.get("shape")) is list
              and all(_is_int(x) and x >= 0 for x in info["shape"])
              and type(info.get("data_offsets")) is list and len(info["data_offsets"]) == 2
              and all(_is_int(x) and x >= 0 for x in info["data_offsets"]))
        if not ok:
            raise ValueError(f"{name}: malformed safetensors entry")
        start, end = info["data_offsets"]
        if end < start or end - start != math.prod(info["shape"]) * ST_SIZES[info["dtype"]] or 8 + n + end > size:
            raise ValueError(f"{name}: inconsistent safetensors entry")
        tensors.append(dict(name=name, family="st", dtype=info["dtype"], shape=info["shape"],
                            nbytes=end - start, path=str(path), offset=8 + n + start,
                            block=ST_SIZES[info["dtype"]]))
    tensors.sort(key=lambda t: t["offset"])
    for a, b in zip(tensors, tensors[1:]):
        if a["offset"] + a["nbytes"] > b["offset"]:
            raise ValueError(f"{a['name']} overlaps {b['name']}")
    meta, config = {}, Path(path).parent / "config.json"
    config_path = None
    if config.is_file():
        config_path = str(config)
        config_bytes = config.read_bytes()
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        doc = json.loads(config_bytes)
        model_type = doc.get("model_type") if type(doc) is dict else None
        if isinstance(model_type, str):
            meta["general.architecture"] = model_type
    return dict(kind="safetensors", meta=meta, tensors=tensors, config=config_path,
                config_sha256=config_sha256 if config_path else None, path=str(path),
                header_len=8 + n, header_sha256=header_sha256)


def load_sources(paths):
    parts = [read_gguf(p) if Path(p).suffix == ".gguf" else read_safetensors(p) for p in paths]
    meta, tensors, seen = {}, [], set()
    for part in parts:
        for key, value in part["meta"].items():
            if not key.startswith("split.") and key in meta and meta[key] != value:
                raise ValueError(f"source parts disagree on {key}")
        meta.update(part["meta"])
        for t in part["tensors"]:
            if t["name"] in seen:
                raise ValueError("duplicate tensor across shards: " + t["name"])
            seen.add(t["name"])
            tensors.append(t)
    return dict(parts=parts, meta=meta, tensors=tensors)


# ---------------------------------------------------------------- planning

def representation(t, books):
    if t["family"] == "ggml":
        return {"family": "ggml", "type": t["dtype"], "ne": t["ne"]}
    m = EXL3_PART.fullmatch(t["name"])
    if m and m[2] != "bias":
        rep = {"family": "exl3", "role": m[2], "dtype": t["dtype"], "shape": t["shape"]}
        if m[2] == "trellis":
            if len(t["shape"]) != 3 or t["shape"][2] % 16:
                raise ValueError("unexpected EXL3 trellis " + t["name"])
            rep.update(k_bits=t["shape"][2] // 16, in_features=16 * t["shape"][0],
                       out_features=16 * t["shape"][1], codebook=books[m[1]])
        return rep
    return {"family": "plain", "dtype": t["dtype"], "shape": t["shape"]}


def _exl3_codebooks(tensors):
    names = {t["name"] for t in tensors}
    books = {}
    for t in tensors:
        m = EXL3_PART.fullmatch(t["name"])
        if m and m[2] == "trellis":
            found = [b for b in ("mcg", "mul1") if f"{m[1]}.{b}" in names]
            if len(found) > 1:
                raise ValueError("conflicting EXL3 codebook markers for " + m[1])
            books[m[1]] = found[0] if found else "3inst"
    return books


def content_sha256(t):
    h = hashlib.sha256()
    with open(t["path"], "rb") as f:
        f.seek(t["offset"])
        left = t["nbytes"]
        while left:
            chunk = f.read(min(left, 64 << 20))
            if not chunk:
                raise ValueError("source truncated")
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest()


def plan(src, tie_check=False, shard_target=SHARD_TARGET):
    """Assign resources to groups, pack members, place groups into shards and chunks."""
    tensors, meta = src["tensors"], src["meta"]
    arch = meta.get("general.architecture")
    if type(arch) is not str or not NAME.fullmatch(arch):
        raise ValueError(f"unknown or unsafe architecture {arch!r}")
    n_exp = int(meta.get(f"{arch}.expert_count", 0) or 0)
    books = _exl3_codebooks(tensors)
    for t in tensors:
        if not NAME.fullmatch(t["name"]) or t["name"] in RESERVED:
            raise ValueError(f"tensor name not representable: {t['name']!r}")
    ties = {}
    if tie_check:
        by_shape = collections.defaultdict(list)
        for t in tensors:
            if (t["name"] in ROW_TABLES or HEADS.fullmatch(t["name"])) and not (
                    EXL3_PART.fullmatch(t["name"]) and not t["name"].endswith(".bias")):
                by_shape[(t["dtype"], str(t.get("ne") or t.get("shape")), t["nbytes"])].append(t)
        for group in by_shape.values():
            if len(group) > 1:
                digests = {t["name"]: content_sha256(t) for t in group}
                group.sort(key=lambda t: t["name"] not in ROW_TABLES)  # canonical: the table
                keep = group[0]
                for other in group[1:]:
                    if digests[other["name"]] == digests[keep["name"]]:
                        ties[other["name"]] = (keep["name"], digests[keep["name"]])

    reprs = {t["name"]: representation(t, books) for t in tensors}
    for name, rep in reprs.items():
        repr_bytes(rep, name)  # importer rejects what the verifier would
    problems = exl3_closure_errors(reprs)
    if problems:
        raise ValueError("unsupported EXL3 input: " + "; ".join(problems[:3]))

    groups, by_key = [], {}

    def group(key, kind, layer=None, expert=None):
        if key not in by_key:
            by_key[key] = len(groups)
            groups.append(dict(kind=kind, layer=layer, expert=expert, members=[]))
        return groups[by_key[key]]

    expert_arrays, globals_ = [], []
    for t in tensors:
        if t["name"] in ties:
            continue
        rep = reprs[t["name"]]
        member = dict(name=t["name"], src=t, src_offset=0, nbytes=t["nbytes"], repr=rep,
                      readable=readable_for(rep, t["nbytes"]),
                      roles=[t["name"]] + sorted(k for k, v in ties.items() if v[0] == t["name"]))
        m = EXPERT_SLICED.fullmatch(t["name"]) if t["family"] == "ggml" else None
        if m and n_exp and len(t["ne"]) == 3 and t["ne"][2] == n_exp:
            layer, per = int(m[1]), t["nbytes"] // n_exp
            rep = {"family": "ggml", "type": t["dtype"], "ne": t["ne"][:2]}
            arr = dict(name=t["name"], layer=layer, count=n_exp, slice_bytes=per, repr=rep,
                       readable=readable_for(rep, per), block=t["block"], members=[])
            for e in range(n_exp):
                g = group(("x", layer, e), "expert", layer, e)
                mem = dict(member, name=f"{t['name']}#{e}", src_offset=e * per, nbytes=per,
                           readable=arr["readable"], repr=rep, roles=[], array=len(expert_arrays))
                g["members"].append(mem)
                arr["members"].append(mem)
            expert_arrays.append(arr)
            continue
        layer = LAYER.match(t["name"])
        if any(HEADS.fullmatch(r) for r in member["roles"][1:]):
            member["access"] = "rows"  # tied table: dense head use dominates placement
            group(("h",), "head")["members"].append(member)
        elif t["name"] in ROW_TABLES:
            member["access"] = "rows"
            group(("t", t["name"]), "table")["members"].append(member)
        elif HEADS.fullmatch(t["name"]):
            group(("h",), "head")["members"].append(member)
        elif layer:
            group(("l", int(layer[1])), "layer", int(layer[1]))["members"].append(member)
        else:
            globals_.append(member)

    # Small non-layer tensors (final norm, rope factors) share the head group,
    # or their own group when the head is a tied table outside it.
    host = groups[by_key[("h",)]] if ("h",) in by_key else (group(("g",), "global") if globals_ else None)
    if host is not None:
        host["members"][:0] = globals_
    rank = {"table": 0, "layer": 1, "expert": 1, "global": 2, "head": 3}
    groups.sort(key=lambda g: (rank[g["kind"]], g["layer"] if g["layer"] is not None else -1,
                               -1 if g["expert"] is None else g["expert"]))
    chunk, shards = 0, [dict(stored=0, groups=[])]
    for gid, g in enumerate(groups):
        off = 0
        for mem in g["members"]:
            off = align(off, MEMBER_ALIGN)
            mem["group"], mem["offset"] = gid, off
            off += mem["readable"]  # over-read bytes stay zero and exclusive
        g["used"], g["stored"] = off, max(FILE_ALIGN, align(off, FILE_ALIGN))
        g["chunks"] = -(-g["stored"] // CHUNK)
        shard = shards[-1]
        if shard["stored"] and shard["stored"] + g["stored"] > shard_target:
            shards.append(dict(stored=0, groups=[]))
            shard = shards[-1]
        g["id"], g["shard"], g["file_offset"], g["first_chunk"] = gid, len(shards) - 1, shard["stored"], chunk
        shard["groups"].append(gid)
        shard["stored"] += g["stored"]
        chunk += g["chunks"]
    for arr in expert_arrays:
        offsets = {m["offset"] for m in arr["members"]}
        sizes = {groups[m["group"]]["stored"] for m in arr["members"]}
        if len(offsets) != 1 or len(sizes) != 1:
            raise ValueError("expert array not uniform: " + arr["name"])
        arr["group_offset"], arr["first_group"] = offsets.pop(), arr["members"][0]["group"]
    return dict(arch=arch, experts=n_exp, groups=groups, shards=shards,
                expert_arrays=expert_arrays, chunks=chunk, ties=ties)


# ---------------------------------------------------------------- statistics

def _lcm(*xs):
    out = 1
    for x in xs:
        out = out * x // math.gcd(out, x)
    return out


def _row_geometry(r):
    """(n_rows, row_bytes) of a row-access resource, derived from its representation."""
    rep = r["repr"]
    if rep["family"] == "ggml":
        if len(rep["ne"]) != 2:
            raise ArtifactError("repr", f"{r['name']}: row table must be 2D")
        return rep["ne"][1], ggml_row_bytes(rep["type"], rep["ne"][0])
    shape = rep["shape"]
    if len(shape) != 2 or shape[0] < 1:
        raise ArtifactError("repr", f"{r['name']}: row table must be 2D")
    return shape[0], r["bytes"] // shape[0]


def rows_to_chunks(index, name, rows):
    """(group, chunk) keys a sparse row lookup needs, deduplicated and sorted.
    Geometry comes from the verified index, never from the caller; row IDs
    (token IDs) are untrusted and out-of-range rows are rejected, not clamped."""
    r = next((r for r in index["resources"] if r["name"] == name and r.get("access") == "rows"), None)
    if r is None:
        raise ArtifactError("bounds", f"{name!r} is not a row table")
    n_rows, row_bytes = _row_geometry(r)
    out = set()
    for row in rows:
        if not _is_int(row) or not 0 <= row < n_rows:
            raise ArtifactError("bounds", f"row {row!r} outside table of {n_rows}")
        start = r["offset"] + row * row_bytes
        out.update((r["group"], k) for k in closure(index, r["group"], start, row_bytes))
    return sorted(out)


def _pct(total, used):
    """Padding as a share of the total (stored or backing) bytes."""
    return round(100 * (total - used) / total, 3) if total else None


def stats(p):
    G = p["groups"]
    payload = sum(m["nbytes"] for g in G for m in g["members"])
    used = sum(g["used"] for g in G)
    disk = sum(g["stored"] for g in G)
    handles = sum(g["chunks"] for g in G) * CHUNK
    kinds = collections.defaultdict(lambda: dict(groups=0, payload=0, disk=0, handle_backing=0))
    for g in G:
        k = kinds[g["kind"]]
        k["groups"] += 1
        k["payload"] += g["used"]
        k["disk"] += g["stored"]
        k["handle_backing"] += g["chunks"] * CHUNK
    for k in kinds.values():
        k["disk_padding_pct"] = _pct(k["disk"], k["payload"])
        k["handle_padding_pct"] = _pct(k["handle_backing"], k["payload"])
    out = dict(architecture=p["arch"], experts=p["experts"], groups=len(G), chunks=p["chunks"],
               tensor_payload_bytes=payload, group_used_bytes=used, disk_bytes=disk,
               disk_padding_pct=_pct(disk, used),
               # Memory under D-033 independent 2 MiB handles (chunk-granular),
               # which is also what 2 MiB-aligned groups on disk would store.
               handle_backing_bytes=handles, handle_padding_pct=_pct(handles, used),
               readable_overread_bytes=used - payload - sum(
                   m["offset"] - (prev["offset"] + prev["readable"]) for g in G
                   for prev, m in zip(g["members"], g["members"][1:])),
               shards=len(p["shards"]), by_kind=dict(kinds),
               ties={k: v[0] for k, v in p["ties"].items()})
    per_layer = collections.defaultdict(list)
    for arr in p["expert_arrays"]:
        per_layer[arr["layer"]].append(arr)
    shapes = collections.Counter()
    va = collections.Counter()
    for arrs in per_layer.values():
        g = G[arrs[0]["first_group"]]
        n, span = arrs[0]["count"], g["chunks"] * CHUNK
        blocks = [a["block"] for a in arrs]
        va["uniform_stride_one_array"] += n * align(span, _lcm(CHUNK, *blocks))
        va["uniform_stride_per_projection"] += n * sum(align(span, _lcm(CHUNK, b)) for b in blocks)
        va["pointer_table_handles"] += n * span
        va["pointer_table_slots"] += n * g["stored"]
        shapes[(tuple(f"{a['name'].split('.')[2]}:{a['repr']['type']}" for a in arrs),
                g["used"], g["stored"], g["chunks"])] += 1
    if per_layer:
        out["expert_layers"] = [
            dict(projections=list(k[0]), closure_bytes=k[1], disk_bytes=k[2], chunks=k[3],
                 disk_padding_pct=_pct(k[2], k[1]), handle_padding_pct=_pct(k[3] * CHUNK, k[1]), layers=c)
            for k, c in sorted(shapes.items(), key=lambda kv: -kv[1])]
        out["expert_view_va_bytes"] = dict(va)
    tables = []
    for g in G:
        for m in g["members"]:
            if m.get("access") != "rows":
                continue
            t = m["src"]
            rb = t["row_bytes"] if t["family"] == "ggml" else t["nbytes"] // t["shape"][0]
            straddle = sum(1 for k in range(1, g["chunks"])
                           if 0 < k * CHUNK - m["offset"] < m["nbytes"] and (k * CHUNK - m["offset"]) % rb)
            tables.append(dict(name=m["name"], type=t["dtype"], rows=t["nbytes"] // rb, row_bytes=rb,
                               rows_per_chunk=round(CHUNK / rb, 2), straddling_rows=straddle,
                               single_row_read_amplification=round(CHUNK / rb, 1), chunks=g["chunks"]))
    out["row_tables"] = tables
    fills = [g["stored"] - (g["chunks"] - 1) * CHUNK for g in G]
    out["last_chunk_fill_bytes"] = dict(min=min(fills), max=max(fills), mean=round(sum(fills) / len(fills)),
                                        full=sum(f == CHUNK for f in fills))
    out["container"] = container_sizes(p)
    return out


def shard_entries(p, s):
    """Container entries (resources plus explicit zero pads) in offset order."""
    entries, pads = [], 0
    for gid in p["shards"][s]["groups"]:
        g = p["groups"][gid]
        cursor = start = g["file_offset"]
        for m in g["members"]:
            at = start + m["offset"]
            if at > cursor:
                entries.append((f"{PAD_PREFIX}{pads}", "U8", [at - cursor], cursor, at))
                pads += 1
            dtype, shape = st_view(m["repr"], m["nbytes"])
            entries.append((m["name"], dtype, shape, at, at + m["nbytes"]))
            cursor = at + m["nbytes"]
        if start + g["stored"] > cursor:
            entries.append((f"{PAD_PREFIX}{pads}", "U8", [start + g["stored"] - cursor], cursor, start + g["stored"]))
            pads += 1
    return entries


def st_header_text(entries, limit=None):
    """Canonical compact safetensors header JSON (names are pattern-restricted
    ASCII, so no escaping is needed). Stops once it exceeds `limit` bytes."""
    limit = MAX_ST_HEADER if limit is None else limit
    parts = ['{"__metadata__":{"format":"jitllm-shard","format_version":"%d"}' % FORMAT_VERSION]
    size = len(parts[0]) + 1
    for name, dtype, shape, a, b in entries:
        if not (NAME.fullmatch(name) or SLICE_NAME.fullmatch(name) or PAD_NAME.fullmatch(name)):
            raise ArtifactError("schema", f"unrepresentable entry name {name!r}")
        piece = ',"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]}' % (
            name, dtype, ",".join(str(int(x)) for x in shape), a, b)
        size += len(piece)
        if size > limit:
            raise ArtifactError("container", f"shard header exceeds {limit} bytes")
        parts.append(piece)
    return ("".join(parts) + "}").encode()


def st_header(p, s):
    raw = st_header_text(shard_entries(p, s))
    data_offset = align(8 + len(raw), FILE_ALIGN)
    return raw + b" " * (data_offset - 8 - len(raw)), data_offset


def container_sizes(p):
    out = []
    for s in range(len(p["shards"])):
        entries = shard_entries(p, s)
        raw, data_offset = st_header(p, s)
        # GGUF alternative (general.alignment=256): an entry costs
        # 8+len(name) + 4 + 8*dims + 4 + 8 bytes; group pads still needed.
        gguf = 24 + sum(8 + len(n) + 4 + 8 * max(1, len(sh)) + 12 for n, _, sh, _, _ in entries)
        out.append(dict(entries=len(entries), pads=sum(e[0].startswith(PAD_PREFIX) for e in entries),
                        json_bytes=len(raw.rstrip()), data_offset=data_offset, gguf=gguf))
    return dict(shards=len(out), entries=sum(o["entries"] for o in out), pads=sum(o["pads"] for o in out),
                max_safetensors_json_bytes=max(o["json_bytes"] for o in out),
                header_region_bytes=sum(o["data_offset"] for o in out),
                gguf_header_estimate=sum(o["gguf"] for o in out))


# ---------------------------------------------------------------- writing

def index_doc(p, chunk_hashes, shard_meta):
    groups = [{"kind": g["kind"], "layer": g["layer"], "expert": g["expert"], "shard": g["shard"],
               "offset": g["file_offset"], "stored_bytes": g["stored"], "used_bytes": g["used"],
               "first_chunk": g["first_chunk"]} for g in p["groups"]]
    resources = []
    for g in p["groups"]:
        for m in g["members"]:
            if "array" in m:
                continue
            r = {"name": m["name"], "group": m["group"], "offset": m["offset"], "bytes": m["nbytes"],
                 "readable_bytes": m["readable"], "roles": m["roles"], "repr": m["repr"]}
            if m.get("access"):
                r["access"] = m["access"]
            resources.append(r)
    resources.sort(key=lambda r: (r["group"], r["offset"]))
    arrays = [{"name": a["name"], "layer": a["layer"], "count": a["count"], "first_group": a["first_group"],
               "group_offset": a["group_offset"], "slice_bytes": a["slice_bytes"],
               "readable_bytes": a["readable"], "repr": a["repr"]} for a in p["expert_arrays"]]
    arrays.sort(key=lambda a: (a["first_group"], a["group_offset"]))
    return {"format": INDEX_FORMAT, "format_version": FORMAT_VERSION, "file_alignment": FILE_ALIGN,
            "chunk_bytes": CHUNK, "member_alignment": MEMBER_ALIGN, "shards": shard_meta,
            "groups": groups, "resources": resources, "expert_arrays": arrays, "chunk_sha256": chunk_hashes}


def _fsync_write(path, data):
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def _staging_dir(out_root):
    """The store's staging directory; never a symlink (it is deleted from)."""
    staging = Path(out_root) / ".staging"
    try:
        staging.mkdir(parents=True)
    except FileExistsError:
        pass
    if not stat.S_ISDIR(os.lstat(staging).st_mode):
        raise ArtifactError("file-type", f"{staging} is not a real directory")
    return staging


def _identity_pass(identity_files, p, extra=()):
    """Hash every identity file once, and in the same reads hash each byte range
    the build will copy (and any extra (tensor, key) ranges, such as tie
    candidates), so copied and compared bytes are exactly what the digest covers."""
    wanted = collections.defaultdict(list)
    for g in p["groups"]:
        for m in g["members"]:
            t = m["src"]
            wanted[os.path.realpath(t["path"])].append((t["offset"] + m["src_offset"], m["nbytes"], id(m)))
    for path, offset, n, key in extra:
        wanted[os.path.realpath(path)].append((offset, n, key))
    sources, digests = {}, {}
    for f in identity_files:
        ranges = sorted(wanted.get(os.path.realpath(f), []), key=lambda r: (r[0], r[1]))
        hashers = {key: hashlib.sha256() for _, _, key in ranges}
        whole, pos, r_i = hashlib.sha256(), 0, 0
        with open(f, "rb") as fh:
            while block := fh.read(16 << 20):
                whole.update(block)
                end = pos + len(block)
                while r_i < len(ranges) and ranges[r_i][0] + ranges[r_i][1] <= pos:
                    r_i += 1
                for j in range(r_i, len(ranges)):
                    a, n, key = ranges[j]
                    if a >= end:
                        break
                    if a + n > pos:  # ranges may overlap (tie candidates); skip finished ones
                        hashers[key].update(block[max(a, pos) - pos:min(a + n, end) - pos])
                pos = end
        sources[Path(f).name] = (pos, whole.hexdigest())
        digests.update({key: h.hexdigest() for key, h in hashers.items()})
    return sources, digests


def job_name(source_digests, meta_digests, converter):
    """Content-derived staging name: a restarted import of the same inputs
    reuses its own directory; different inputs never collide."""
    key = json.dumps([source_digests, meta_digests, converter], sort_keys=True)
    return "job-" + hashlib.sha256(key.encode()).hexdigest()[:32]


def _try_lock(path):
    """Exclusive non-blocking flock on path; returns the fd or None if held."""
    import fcntl
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def sweep_staging(out_root):
    """Startup cleanup: remove staging leftovers of jobs not currently running
    (their lock is free). Never follows symlinks; only touches .staging."""
    staging = _staging_dir(out_root)
    removed = []
    for d in sorted(staging.iterdir()):
        if d.name.endswith(".lock") or not d.name.startswith("job-"):
            continue
        fd = _try_lock(staging / f"{d.name}.lock")
        if fd is None:
            continue  # a live job holds it
        try:
            if d.is_symlink() or not d.is_dir():
                d.unlink()
            else:
                shutil.rmtree(d)
            removed.append(d.name)
        except FileNotFoundError:
            pass  # the job finished between listing and locking
        finally:
            os.close(fd)
    return removed


def build(p, src, out_root, source_files, meta_files=(), converter=None, crash_after=None,
          expected_sources=None):
    """Write into locked private staging, verify, then publish by one rename.

    expected_sources maps source name -> SHA-256 recorded at download (D-054);
    the sources are hashed before and after writing, so a source that changes
    during import aborts it."""
    out_root = Path(out_root)
    converter = converter or {"name": "artifact-layout/layout.py", "version": "m0-prototype"}
    metas = []
    for rel, d in meta_files:
        if not isinstance(d, (str, Path)) or str(rel) != f"meta/{Path(d).name}":
            raise ValueError(f"kept metadata must be a source file published under its own name: {rel!r}")
        metas.append((str(rel), d))
    for i, (part_path, part) in enumerate(zip(source_files, src["parts"])):
        if part["kind"] == "gguf" and part.get("n_kv"):
            metas.append((f"meta/{Path(part_path).stem}.kv.gguf", ("kv", part_path, part["kv_end"], i)))
    rels = [r for r, _ in metas]
    if len(set(rels)) != len(rels) or not all(META_PATH.fullmatch(r) for r in rels):
        raise ValueError(f"metadata paths must be unique and safe: {rels}")
    names = [Path(f).name for f in source_files]
    if len(set(names)) != len(names) or not all(NAME.fullmatch(n) for n in names):
        raise ValueError(f"source file names must be unique and safe: {names}")
    # Every input is source identity: checkpoints, the config.json that decides a
    # safetensors architecture, and every metadata file kept in the artifact.
    identity_files, seen_paths = [], set()
    for f in (list(source_files) + sorted({part["config"] for part in src["parts"] if part.get("config")})
              + [str(d) for _, d in metas if isinstance(d, (str, Path))]):
        key = (Path(f).name, os.path.realpath(f))  # distinct names may share one blob (HF cache)
        if key not in seen_paths:
            seen_paths.add(key)
            identity_files.append(f)
    names = [Path(f).name for f in identity_files]
    if len(set(names)) != len(names):
        raise ValueError(f"source identity file names must be unique: {names}")
    by_name = {t["name"]: t for t in src["tensors"]}
    tie_names = sorted({n for role, (keep, _) in p["ties"].items() for n in (role, keep)})
    extra = [(by_name[n]["path"], by_name[n]["offset"], by_name[n]["nbytes"], ("tie", n)) for n in tie_names]
    for i, part in enumerate(src["parts"]):
        extra.append((part["path"], 0, part["header_len"], ("hdr", i)))
        if part["kind"] == "gguf" and part.get("n_kv"):
            extra.append((part["path"], 0, part["kv_end"], ("kv", i)))
    sources, member_digests = _identity_pass(identity_files, p, extra)
    for i, part in enumerate(src["parts"]):  # the plan was made from exactly the hashed headers
        if member_digests[("hdr", i)] != part["header_sha256"]:
            raise ValueError(f"{part['path']}: header changed after planning")
        if part.get("config") and sources[Path(part["config"]).name][1] != part["config_sha256"]:
            raise ValueError(f"{part['config']} changed after planning")
    for role, (keep, digest) in p["ties"].items():  # confirm ties on the hashed bytes
        if member_digests[("tie", role)] != digest or member_digests[("tie", keep)] != digest:
            raise ValueError(f"tie {role} -> {keep} no longer holds")
    if expected_sources is not None and {n: d for n, (_, d) in sources.items()} != expected_sources:
        raise ValueError("sources differ from their recorded identities")
    def parsed(s_):
        return ([(t["name"], t["dtype"], t.get("ne") or t.get("shape"), t["offset"], t["nbytes"],
                  os.path.realpath(t["path"])) for t in s_["tensors"]], s_["meta"],
                [(p_["kind"], p_.get("kv_end"), p_.get("n_kv")) for p_ in s_["parts"]])
    if parsed(load_sources(source_files)) != parsed(src):  # the plan must describe the hashed bytes
        raise ValueError("a source changed after planning")
    planned = {id(m["src"]) for g in p["groups"] for m in g["members"]}
    if not planned <= {id(t) for t in src["tensors"]}:
        raise ValueError("the plan was not made from these sources")
    meta_digests = []
    for rel, d in metas:
        blob = [sources[Path(d[1]).name][1], d[2]] if type(d) is tuple else sources[Path(d).name][1]
        meta_digests.append([rel, blob])
    sizes = [src["parts"][d[3]]["kv_end"] if type(d) is tuple else os.path.getsize(d) for _, d in metas]
    if any(n > MAX_META for n in sizes) or sum(sizes) > MAX_META_TOTAL:
        raise ValueError("kept metadata exceeds the verifier's size limits")
    staging = _staging_dir(out_root)
    job = job_name(sorted([n, dg] for n, (_, dg) in sources.items()), meta_digests, converter)
    lock = _try_lock(staging / f"{job}.lock")
    if lock is None:
        raise ArtifactError("busy", f"import {job} is already running")
    try:
        return _build_locked(p, src, out_root, staging / job, identity_files, metas, converter,
                             sources, crash_after, member_digests)
    finally:
        os.close(lock)


def _build_locked(p, src, out_root, work, identity_files, metas, converter, sources, crash_after,
                  member_digests):
    if work.is_symlink():
        work.unlink()
    elif work.exists():
        shutil.rmtree(work)
    (work / "data").mkdir(parents=True)
    (work / "meta").mkdir()
    chunk_hashes, shard_meta, files, handles = [None] * p["chunks"], [], [], {}
    try:
        for s, shard in enumerate(p["shards"]):
            raw, data_offset = st_header(p, s)
            rel = f"data/{s:05d}.safetensors"
            whole = hashlib.sha256()
            with open(work / rel, "wb") as f:
                head = struct.pack("<Q", len(raw)) + raw
                f.write(head)
                whole.update(head)
                for gid in shard["groups"]:
                    g = p["groups"][gid]
                    buf = bytearray(g["stored"])
                    for m in g["members"]:
                        t = m["src"]
                        if t["path"] not in handles:
                            handles[t["path"]] = open(t["path"], "rb")
                        fh = handles[t["path"]]
                        fh.seek(t["offset"] + m["src_offset"])
                        data = fh.read(m["nbytes"])
                        if len(data) != m["nbytes"] or hashlib.sha256(data).hexdigest() != member_digests[id(m)]:
                            raise ValueError("a source changed during import: " + m["name"])
                        buf[m["offset"]:m["offset"] + m["nbytes"]] = data
                    view = memoryview(buf)
                    for k in range(g["chunks"]):
                        chunk_hashes[g["first_chunk"] + k] = hashlib.sha256(view[k * CHUNK:(k + 1) * CHUNK]).hexdigest()
                    f.write(buf)
                    whole.update(buf)
                f.flush()
                os.fsync(f.fileno())
            shard_meta.append({"path": rel, "data_offset": data_offset, "data_bytes": shard["stored"],
                               "header_sha256": hashlib.sha256(head).hexdigest()})
            files.append({"path": rel, "role": "shard", "bytes": data_offset + shard["stored"],
                          "sha256": whole.hexdigest()})
            if crash_after == s:
                raise KeyboardInterrupt("simulated crash")
    finally:
        for fh in handles.values():
            fh.close()
    for rel, data in metas:
        if type(data) is tuple:  # source KV metadata kept as a zero-tensor GGUF
            with open(data[1], "rb") as f:
                head = f.read(data[2])
            if hashlib.sha256(head).hexdigest() != member_digests[("kv", data[3])]:
                raise ValueError(f"{data[1]}: metadata changed during import")
            data = head[:8] + struct.pack("<Q", 0) + head[16:]
        else:
            name = Path(data).name
            data = Path(data).read_bytes()
            if hashlib.sha256(data).hexdigest() != sources[name][1]:
                raise ValueError(f"{name}: metadata changed during import")
        _fsync_write(work / rel, data)
        files.append({"path": rel, "role": "source-metadata", "bytes": len(data),
                      "sha256": hashlib.sha256(data).hexdigest()})
    index = dumps(index_doc(p, chunk_hashes, shard_meta))
    _fsync_write(work / "index.json", index)
    files.append({"path": "index.json", "role": "index", "bytes": len(index),
                  "sha256": hashlib.sha256(index).hexdigest()})
    if _identity_pass(identity_files, {"groups": []})[0] != sources:
        raise ValueError("a source changed during import")
    families = sorted({m["repr"]["family"] for g in p["groups"] for m in g["members"]})
    transformations = ([{"kind": "dedupe-identical", "resource": keep, "role": role, "sha256": digest}
                        for role, (keep, digest) in p["ties"].items()]
                       + [{"kind": "expert-slice", "tensor": a["name"], "count": a["count"]}
                          for a in p["expert_arrays"]])
    manifest = {
        "format": FORMAT, "format_version": FORMAT_VERSION, "experimental": True,
        "layout": LAYOUT,
        "model": {"architecture": p["arch"], "expert_count": p["experts"], "representation": families},
        "source": sorted(({"name": n, "bytes": b, "sha256": d} for n, (b, d) in sources.items()),
                         key=lambda s: s["name"]),
        "transformations": sorted(transformations, key=_canon_key),
        "converter": converter,
        "files": sorted(files, key=lambda f: f["path"]),
    }
    mbytes = dumps(manifest)
    _fsync_write(work / "manifest.json", mbytes)
    artifact_id = hashlib.sha256(mbytes).hexdigest()
    verify(work, deep=True, expected_id=artifact_id)  # never publish what we would reject
    for d in (work / "data", work / "meta", work):
        _fsync_dir(d)
    final = out_root / artifact_id
    if final.exists() or final.is_symlink():
        verify(final)  # raises if the existing copy is damaged; never silently reused
        shutil.rmtree(work)
    else:
        try:
            os.rename(work, final)
        except OSError as e:
            import errno
            if e.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            verify(final)  # another import published the same content first
            shutil.rmtree(work)
    _fsync_dir(work.parent)
    _fsync_dir(out_root)
    return final


def _canon_key(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def installed(out_root):
    """Published artifacts only: 64-hex directory names; staging is never listed."""
    return sorted(d.name for d in Path(out_root).iterdir()
                  if HEX.fullmatch(d.name) and d.is_dir() and not d.is_symlink())


# ---------------------------------------------------------------- verification

LAYOUT = {"profile": PROFILE, "profile_version": PROFILE_VERSION, "file_alignment": FILE_ALIGN,
          "chunk_bytes": CHUNK, "member_alignment": MEMBER_ALIGN, "container": "safetensors",
          "byte_order": "little"}
MANIFEST_KEYS = {"format", "format_version", "experimental", "layout", "model", "source",
                 "transformations", "converter", "files"}
INDEX_KEYS = {"format", "format_version", "file_alignment", "chunk_bytes", "member_alignment", "shards",
              "groups", "resources", "expert_arrays", "chunk_sha256"}
INDEX_HEADER = {"format": INDEX_FORMAT, "format_version": FORMAT_VERSION, "file_alignment": FILE_ALIGN,
                "chunk_bytes": CHUNK, "member_alignment": MEMBER_ALIGN}
GROUP_KEYS = {"kind", "layer", "expert", "shard", "offset", "stored_bytes", "used_bytes", "first_chunk"}
SHARD_META = {"format": "jitllm-shard", "format_version": str(FORMAT_VERSION)}
TEXT = re.compile(r'[ !#-\[\]-~]{1,200}')  # printable ASCII without " or \ (no escaping in canonical bytes)


def _open_regular(path, where, size=None):
    """Open a regular, singly linked file without following links; check its
    size before anything reads it (never trust a declared length)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)  # a swapped-in FIFO cannot block
    except OSError as e:
        raise ArtifactError("file-type", f"{where}: {e.strerror}") from e
    st = os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        os.close(fd)
        raise ArtifactError("file-type", f"{where}: not a regular, singly linked file")
    if size is not None and st.st_size != size:
        os.close(fd)
        raise ArtifactError("file-size", f"{where}: {st.st_size} bytes, expected {size}")
    return os.fdopen(fd, "rb")


def _read_doc(root, rel, size=None, limit=MAX_DOC):
    if size is not None and size > limit:
        raise ArtifactError("file-size", f"{rel} larger than {limit}")
    with _open_regular(root / rel, rel, size) as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ArtifactError("file-size", f"{rel} larger than {limit}")
    return data


def _walk(root):
    """Every non-directory path under root; only data/ and meta/ subdirectories,
    no symlinks anywhere, unreadable directories fail instead of being skipped."""
    present = set()

    def fail(e):
        raise ArtifactError("file-type", f"unreadable: {e.filename}")

    for dirpath, dirs, filenames in os.walk(root, onerror=fail, followlinks=False):
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir not in (".", "data", "meta"):
            raise ArtifactError("file-set", f"unexpected directory {rel_dir}")
        for d in dirs:
            if os.path.islink(os.path.join(dirpath, d)):
                raise ArtifactError("file-type", f"symlink {d}")
        for name in filenames:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            st = os.lstat(os.path.join(dirpath, name))
            if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
                raise ArtifactError("file-type", f"not a regular, singly linked file: {rel}")
            present.add(rel)
        if rel_dir == "." and set(dirs) - {"data", "meta"}:
            raise ArtifactError("file-set", f"unexpected directories {sorted(set(dirs) - {'data', 'meta'})}")
    return present


def _sorted_unique(items, key, where):
    keys = [key(x) for x in items]
    if any(a >= b for a, b in zip(keys, keys[1:])):
        raise ArtifactError("canonical", f"{where} not in canonical order or duplicated")


def _check_manifest(manifest):
    _obj(manifest, MANIFEST_KEYS, "manifest")
    if manifest["experimental"] is not True:
        raise ArtifactError("schema", "experimental must be true before D-018's gate")
    model = _obj(manifest["model"], {"architecture", "expert_count", "representation"}, "model")
    _str(model["architecture"], "model.architecture", NAME)
    _int(model["expert_count"], "model.expert_count", 0, 1 << 20)
    reps = _list(model["representation"], "model.representation", 1)
    if not all(type(r) is str for r in reps) or reps != sorted(set(reps)) or not set(reps) <= FAMILIES:
        raise ArtifactError("schema", "model.representation")
    sources = _list(manifest["source"], "source", 1)
    for s in sources:
        _obj(s, {"name", "bytes", "sha256"}, "source")
        _str(s["name"], "source.name", NAME)
        _int(s["bytes"], "source.bytes")
        _str(s["sha256"], "source.sha256", HEX)
    _sorted_unique(sources, lambda s: s["name"], "source")
    transformations = _list(manifest["transformations"], "transformations")
    for t in transformations:
        kind = t.get("kind") if type(t) is dict else None
        kind = kind if type(kind) is str else None
        if kind == "dedupe-identical":
            _obj(t, {"kind", "resource", "role", "sha256"}, "transformation")
            _str(t["resource"], "transformation.resource", NAME)
            _str(t["role"], "transformation.role", NAME)
            _str(t["sha256"], "transformation.sha256", HEX)
        elif kind == "expert-slice":
            _obj(t, {"kind", "tensor", "count"}, "transformation")
            _str(t["tensor"], "transformation.tensor", NAME)
            _int(t["count"], "transformation.count", 1, 1 << 20)
        else:
            raise ArtifactError("schema", f"transformation kind {kind!r}")
    _sorted_unique(transformations, _canon_key, "transformations")
    conv = _obj(manifest["converter"], {"name", "version"}, "converter")
    _str(conv["name"], "converter.name", TEXT)
    _str(conv["version"], "converter.version", TEXT)
    listed = _list(manifest["files"], "files", 2)
    files = {}
    for f in listed:
        _obj(f, {"path", "role", "bytes", "sha256"}, "file")
        path = _str(f["path"], "file.path")
        _str(f["role"], "file.role")
        pattern = {"index": re.compile(r"index\.json"), "shard": SHARD_PATH,
                   "source-metadata": META_PATH}.get(f["role"])
        if pattern is None or not pattern.fullmatch(path):
            raise ArtifactError("path", f"{path!r} with role {f['role']!r}")
        _int(f["bytes"], "file.bytes")
        _str(f["sha256"], "file.sha256", HEX)
        files[path] = f
    _sorted_unique(listed, lambda f: f["path"], "files")
    if [p for p, f in files.items() if f["role"] == "index"] != ["index.json"]:
        raise ArtifactError("file-set", "exactly one index.json is required")
    return files


def _check_index(index, manifest, files):
    _obj(index, INDEX_KEYS, "index")
    if not all(_exact(index[k], v) for k, v in INDEX_HEADER.items()):
        raise ArtifactError("unsupported-version", "index header")
    shards = _list(index["shards"], "shards", 1)
    for i, s in enumerate(shards):
        _obj(s, {"path", "data_offset", "data_bytes", "header_sha256"}, "shard")
        if _str(s["path"], "shard.path", SHARD_PATH) != f"data/{i:05d}.safetensors":
            raise ArtifactError("file-set", f"shard {i} path {s['path']}")
        if _int(s["data_offset"], "shard", FILE_ALIGN) % FILE_ALIGN or _int(s["data_bytes"], "shard", FILE_ALIGN) % FILE_ALIGN:
            raise ArtifactError("bounds", f"shard {i} alignment")
        _str(s["header_sha256"], "shard.header_sha256", HEX)
        if files.get(s["path"], {}).get("role") != "shard" or \
                files[s["path"]]["bytes"] != s["data_offset"] + s["data_bytes"]:
            raise ArtifactError("file-set", f"shard {s['path']} not listed or wrong size")
    if {p for p, f in files.items() if f["role"] == "shard"} != {s["path"] for s in shards}:
        raise ArtifactError("file-set", "shard list")
    groups = _list(index["groups"], "groups", 1)
    cursor, next_chunk, last_shard = [0] * len(shards), 0, 0
    seen_expert = set()
    for gid, g in enumerate(groups):
        _obj(g, GROUP_KEYS, "group")
        kind = g["kind"]
        if type(kind) is not str or kind not in KINDS:
            raise ArtifactError("schema", f"group {gid} kind {kind!r}")
        if kind in ("layer", "expert"):
            _int(g["layer"], f"group {gid} layer", 0, 1 << 20)
        elif g["layer"] is not None:
            raise ArtifactError("schema", f"group {gid} layer")
        if kind == "expert":
            _int(g["expert"], f"group {gid} expert", 0, 1 << 20)
            if (g["layer"], g["expert"]) in seen_expert:
                raise ArtifactError("expert-array", f"duplicate expert group {g['layer']}/{g['expert']}")
            seen_expert.add((g["layer"], g["expert"]))
        elif g["expert"] is not None:
            raise ArtifactError("schema", f"group {gid} expert")
        s = _int(g["shard"], "group", 0, len(shards) - 1)
        if s < last_shard:
            raise ArtifactError("bounds", f"group {gid} shard order")
        last_shard = s
        stored, used = _int(g["stored_bytes"], "group", FILE_ALIGN), _int(g["used_bytes"], "group", 1)
        if _int(g["offset"], "group") != cursor[s] or stored != align(used, FILE_ALIGN):
            raise ArtifactError("bounds", f"group {gid} offset/size")
        if _int(g["first_chunk"], "group") != next_chunk:
            raise ArtifactError("bounds", f"group {gid} chunk numbering")
        cursor[s] += stored
        next_chunk += -(-stored // CHUNK)
    if cursor != [s["data_bytes"] for s in shards]:
        raise ArtifactError("bounds", "groups do not tile shards")
    hashes = _list(index["chunk_sha256"], "chunk_sha256")
    if len(hashes) != next_chunk or not all(type(h) is str and HEX.fullmatch(h) for h in hashes):
        raise ArtifactError("schema", "chunk_sha256")
    # Resources and expert slices: typed, sized by their representation, unique.
    spans, names, roles, families = collections.defaultdict(list), set(), set(), set()

    def claim(name):
        if name in names:
            raise ArtifactError("schema", f"duplicate name {name}")
        names.add(name)

    resources = _list(index["resources"], "resources")
    arrays = _list(index["expert_arrays"], "expert_arrays")
    if len(groups) > MAX_ENTRIES or len(resources) + len(arrays) > MAX_ENTRIES:
        raise ArtifactError("bounds", f"more than {MAX_ENTRIES} entries")
    entries = len(resources)
    for r in resources:
        _obj(r, {"name", "group", "offset", "bytes", "readable_bytes", "roles", "repr"}, "resource", ("access",))
        name = _str(r["name"], "resource.name", NAME)
        if name in RESERVED:
            raise ArtifactError("schema", f"reserved name {name}")
        claim(name)
        gid = _int(r["group"], name, 0, len(groups) - 1)
        if groups[gid]["kind"] == "expert":
            raise ArtifactError("expert-array", f"{name} placed in an expert group")
        role_list = _list(r["roles"], f"{name}.roles", 1)
        if role_list[0] != name:
            raise ArtifactError("schema", f"{name}: roles[0] must be the name")
        if len(role_list) > 1 + MAX_ALIASES:
            raise ArtifactError("bounds", f"{name}: more than {MAX_ALIASES} alias roles")
        entries += len(role_list) - 1
        if entries > MAX_ENTRIES:
            raise ArtifactError("bounds", f"more than {MAX_ENTRIES} entries")
        if not all(type(x) is str for x in role_list) or any(a >= b for a, b in zip(role_list[1:], role_list[2:])):
            raise ArtifactError("canonical", f"{name}: alias roles must be sorted and unique")
        for role in role_list:
            _str(role, f"{name}.roles", NAME)
            if role in RESERVED:
                raise ArtifactError("schema", f"reserved role {role}")
            if role in roles:
                raise ArtifactError("schema", f"role {role} bound twice")
            roles.add(role)
        if "access" in r and r["access"] != "rows":
            raise ArtifactError("schema", f"{name}: access {r['access']!r}")
        n = repr_bytes(r["repr"], name)
        families.add(r["repr"]["family"])
        if _int(r["bytes"], name, 1) != n:
            raise ArtifactError("repr", f"{name}: bytes {r['bytes']} but representation implies {n}")
        if _int(r["readable_bytes"], name, 1) != readable_for(r["repr"], n):
            raise ArtifactError("repr", f"{name}: readable_bytes must be {readable_for(r['repr'], n)}")
        if "access" in r:
            _row_geometry(r)
        spans[gid].append((_int(r["offset"], name), n, r["readable_bytes"], name, st_view(r["repr"], n)))
    _sorted_unique(resources, lambda r: (r["group"], r["offset"]), "resources")
    arrays_by_layer = collections.defaultdict(list)
    for a in arrays:
        _obj(a, {"name", "layer", "count", "first_group", "group_offset", "slice_bytes",
                 "readable_bytes", "repr"}, "expert_array")
        name = _str(a["name"], "expert_array.name", NAME)
        if name in RESERVED:
            raise ArtifactError("schema", f"reserved name {name}")
        claim(name)
        layer = _int(a["layer"], name, 0, 1 << 20)
        count = _int(a["count"], name, 1, 1 << 20)
        entries += count
        if entries > MAX_ENTRIES:  # checked before anything is expanded per expert
            raise ArtifactError("bounds", f"more than {MAX_ENTRIES} resources and expert slices")
        if count != manifest["model"]["expert_count"]:
            raise ArtifactError("expert-array", f"{name}: count differs from model.expert_count")
        first = _int(a["first_group"], name, 0, len(groups) - count)
        n = repr_bytes(a["repr"], name)
        if a["repr"]["family"] != "ggml" or len(a["repr"]["ne"]) != 2:
            raise ArtifactError("expert-array", f"{name}: slice must be a 2D ggml matrix")
        families.add("ggml")
        if _int(a["slice_bytes"], name, 1) != n or _int(a["readable_bytes"], name, 1) != readable_for(a["repr"], n):
            raise ArtifactError("repr", f"{name}: slice sizes disagree with representation")
        members = [groups[first + e] for e in range(count)]
        if [(g["kind"], g["layer"], g["expert"]) for g in members] != [("expert", layer, e) for e in range(count)] \
                or len({g["stored_bytes"] for g in members}) != 1:
            raise ArtifactError("expert-array", f"{name}: groups not uniform expert groups")
        arrays_by_layer[layer].append((first, count))
        view = st_view(a["repr"], n)
        for e in range(count):
            spans[first + e].append((_int(a["group_offset"], name), n, a["readable_bytes"], f"{name}#{e}", view))
    _sorted_unique(arrays, lambda a: (a["first_group"], a["group_offset"]), "expert_arrays")
    if roles & {a["name"] for a in arrays}:
        raise ArtifactError("schema", "a role names an expert array")
    for layer, arrs in arrays_by_layer.items():
        if len(set(arrs)) != 1:
            raise ArtifactError("expert-array", f"layer {layer}: arrays disagree on their expert groups")
    covered = {(g["layer"], g["expert"]) for g in groups if g["kind"] == "expert"}
    if covered != {(layer, e) for layer, arrs in arrays_by_layer.items() for e in range(arrs[0][1])}:
        raise ArtifactError("expert-array", "expert groups without a covering array")
    bound = {role: r["repr"] for r in resources for role in r["roles"]}
    bound.update({a["name"]: a["repr"] for a in arrays})
    problems = exl3_closure_errors(bound, first_only=True)
    if problems:
        raise ArtifactError("repr", problems[0])
    if sorted(families) != manifest["model"]["representation"]:
        raise ArtifactError("schema", "model.representation disagrees with the index")
    # Layout: aligned, non-overlapping readable ranges; used_bytes is the last readable end.
    for gid, g in enumerate(groups):
        items = sorted(spans.get(gid, ()))
        if not items:
            raise ArtifactError("bounds", f"empty group {gid}")
        prev_end = 0
        for off, n, readable, name, _ in items:
            if off % MEMBER_ALIGN:
                raise ArtifactError("alignment", name)
            if off < prev_end:
                raise ArtifactError("overlap", name)
            if off + readable > g["stored_bytes"]:
                raise ArtifactError("bounds", name)
            prev_end = off + readable
        if prev_end != g["used_bytes"]:
            raise ArtifactError("bounds", f"group {gid} used_bytes")
    # Transformations must describe the index exactly.
    slices = sorted((t["tensor"], t["count"]) for t in manifest["transformations"] if t["kind"] == "expert-slice")
    if slices != sorted((a["name"], a["count"]) for a in arrays):
        raise ArtifactError("schema", "expert-slice transformations disagree with the index")
    role_of = {r["name"]: set(r["roles"][1:]) for r in resources}
    dedupes = [(t["resource"], t["role"]) for t in manifest["transformations"] if t["kind"] == "dedupe-identical"]
    if sorted(dedupes) != sorted((n, role) for n, rs in role_of.items() for role in rs):
        raise ArtifactError("schema", "dedupe transformations disagree with roles")
    return spans


def expected_shard_header(groups, spans, gids):
    """The one canonical shard header for a shard's groups, derived from the
    index: resources, then a single zero pad per gap, pads numbered in order."""
    entries, pads = [], 0
    for gid in gids:
        g = groups[gid]
        cursor = g["offset"]
        for off, n, _, name, (dtype, shape) in sorted(spans[gid]):
            at = g["offset"] + off
            if at > cursor:
                entries.append((f"{PAD_PREFIX}{pads}", "U8", [at - cursor], cursor, at))
                pads += 1
            entries.append((name, dtype, shape, at, at + n))
            cursor = at + n
        end = g["offset"] + g["stored_bytes"]
        if end > cursor:
            entries.append((f"{PAD_PREFIX}{pads}", "U8", [end - cursor], cursor, end))
            pads += 1
    raw = st_header_text(entries)
    data_offset = align(8 + len(raw), FILE_ALIGN)
    if data_offset - 8 > MAX_ST_HEADER:
        raise ArtifactError("container", f"shard header exceeds {MAX_ST_HEADER} bytes")
    return struct.pack("<Q", data_offset - 8) + raw + b" " * (data_offset - 8 - len(raw)), \
        [(a, b) for name, _, _, a, b in entries if name.startswith(PAD_PREFIX)]


def _check_shard(root, s, gids, spans, groups, index, files, deep, dedupe):
    """Compare the header with the one the index implies; when deep, stream the
    data in chunk-sized pieces: chunk hashes, zero pads, dedupe digests and
    the whole-file hash all come from the same bytes."""
    head, pads = expected_shard_header(groups, spans, gids)
    if len(head) != s["data_offset"]:
        raise ArtifactError("container", f"{s['path']}: data offset is not the canonical one")
    with _open_regular(root / s["path"], s["path"], files[s["path"]]["bytes"]) as f:
        got = f.read(s["data_offset"])
        if got != head:
            raise ArtifactError("container", f"{s['path']}: header disagrees with the index")
        if hashlib.sha256(got).hexdigest() != s["header_sha256"]:
            raise ArtifactError("hash", f"{s['path']} header region")
        if not deep:
            return None
        whole = hashlib.sha256(got)
        pad_i = 0
        watch = sorted((groups[gid]["offset"] + off, groups[gid]["offset"] + off + n, name)
                       for gid in gids for off, n, _, name, _ in spans[gid] if name in dedupe)
        hashers = {name: hashlib.sha256() for _, _, name in watch}
        pos, w_i = 0, 0
        for gid in gids:
            g = groups[gid]
            for k in range(-(-g["stored_bytes"] // CHUNK)):
                piece = f.read(min(CHUNK, g["stored_bytes"] - k * CHUNK))
                if len(piece) != min(CHUNK, g["stored_bytes"] - k * CHUNK):
                    raise ArtifactError("file-size", f"{s['path']} truncated")
                whole.update(piece)
                if hashlib.sha256(piece).hexdigest() != index["chunk_sha256"][g["first_chunk"] + k]:
                    raise ArtifactError("chunk-hash", f"chunk {g['first_chunk'] + k}")
                end = pos + len(piece)
                while pad_i < len(pads) and pads[pad_i][0] < end:
                    a, b = pads[pad_i]
                    if any(piece[max(a, pos) - pos:min(b, end) - pos]):
                        raise ArtifactError("pad", f"{s['path']} pad at {a} not zero")
                    if b > end:
                        break
                    pad_i += 1
                while w_i < len(watch) and watch[w_i][1] <= pos:
                    w_i += 1
                for j in range(w_i, len(watch)):
                    a, b, name = watch[j]
                    if a >= end:
                        break
                    if b > pos:
                        hashers[name].update(piece[max(a, pos) - pos:min(b, end) - pos])
                pos = end
        if f.read(1):
            raise ArtifactError("file-size", f"{s['path']} trailing bytes")
        for _, _, name in watch:
            if dedupe[name] != {hashers[name].hexdigest()}:
                raise ArtifactError("hash", f"{name}: stored bytes do not match the dedupe digest")
        return whole.hexdigest()


def _check_gguf_key(key, t, value, keys):
    """Key rules of pinned gguf.cpp: non-empty, unique, and general.alignment
    a u32 power of two."""
    if not key or key in keys:
        raise ValueError(f"empty or duplicate GGUF key {key!r}")
    keys.add(key)
    if key == "general.alignment" and (t != 4 or value <= 0 or value & (value - 1)):
        raise ValueError("general.alignment must be a u32 power of two")
    if key == "general.architecture" and t != 8:
        raise ValueError("general.architecture must be a string")
    if key.endswith(".expert_count") and t != 4:
        raise ValueError(f"{key} must be a u32")


def _check_kv_gguf(data, where):
    """A kept GGUF metadata file must be a complete zero-tensor GGUF header."""
    import io
    try:
        r = Reader(io.BytesIO(data), len(data))
        if r.take(4) != b"GGUF":
            raise ValueError("magic")
        (version,) = r.unpack("I")
        n_tensors, n_kv = r.unpack("QQ")
        if version != 3 or n_tensors != 0 or n_kv > 1 << 16:
            raise ValueError("version or tensor count")
        keys, kept = set(), {}
        for _ in range(n_kv):
            key = r.string()
            (t,) = r.unpack("I")
            keep = key in ("general.alignment", "general.architecture") or key.endswith(".expert_count")
            if keep and t != (8 if key == "general.architecture" else 4):
                raise ValueError(f"{key} has GGUF type {t}")  # checked before reading any value
            if key == "general.architecture":
                (n,) = r.unpack("Q")
                if n > MAX_ARCH:
                    raise ValueError("general.architecture too long")
                value = r.take(n).decode("utf-8")
            else:
                value = _gguf_value(r, t, keep)
            _check_gguf_key(key, t, value, keys)
            if keep:
                kept[key] = value
        if r.f.tell() != len(data):
            raise ValueError("trailing bytes")
    except (ValueError, UnicodeError, struct.error) as e:
        raise ArtifactError("meta", f"{where}: not a zero-tensor GGUF ({e})") from e
    return kept


def verify(root, deep=True, expected_id=None):
    """Verify an artifact; returns (manifest, index). Fails closed with a named rule.

    deep=True is install/replication/on-demand verification: every byte is
    hashed. deep=False (what a loader does at open) checks structure, sizes
    and headers but not payload hashes. Page-in never re-hashes. Every
    document is parsed from the same bytes that were hashed."""
    try:
        return _verify(Path(root), deep, expected_id)
    except ArtifactError:
        raise
    except (TypeError, KeyError, ValueError, IndexError, AttributeError, OverflowError, MemoryError,
            RecursionError) as e:
        # Safety net for the oracle: anything unanticipated still fails closed.
        raise ArtifactError("malformed", f"{type(e).__name__}: {e}") from e
    except OSError as e:
        raise ArtifactError("io", f"{type(e).__name__}: {e}") from e


def _verify(root, deep, expected_id):
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError("file-type", "artifact root must be a real directory")
    present = _walk(root)
    mbytes = _read_doc(root, "manifest.json", limit=MAX_MANIFEST)
    manifest = strict_json(mbytes, "manifest.json")
    if type(manifest) is not dict or manifest.get("format") != FORMAT:
        raise ArtifactError("format", "not a jitLLM artifact")
    if not _exact(manifest.get("format_version"), FORMAT_VERSION):
        raise ArtifactError("unsupported-version",
                            f"format_version {manifest.get('format_version')!r}; re-import required")
    if not _exact(manifest.get("layout"), LAYOUT):
        raise ArtifactError("unsupported-profile", f"{manifest.get('layout')!r}; re-import required")
    if mbytes != dumps(manifest):
        raise ArtifactError("canonical", "manifest.json is not in canonical form")
    if (expected_id or root.name) != hashlib.sha256(mbytes).hexdigest():
        raise ArtifactError("identity", "artifact name is not the manifest digest")
    files = _check_manifest(manifest)
    if present != set(files) | {"manifest.json"}:
        unlisted, missing = sorted(present - set(files) - {"manifest.json"}), sorted(set(files) - present)
        raise ArtifactError("file-set", f"{len(unlisted)} unlisted {unlisted[:5]}, "
                                        f"{len(missing)} missing {missing[:5]}")
    for path, f in files.items():  # sizes first: nothing below trusts a declared length
        if os.lstat(root / path).st_size != f["bytes"]:
            raise ArtifactError("file-size", f"{path}: size differs from the manifest")
    sources = {s_["name"]: s_ for s_ in manifest["source"]}
    meta_total = 0
    for path, f in files.items():
        if f["role"] != "source-metadata":
            continue
        name = path[len("meta/"):]
        meta_total += f["bytes"]
        if f["bytes"] > MAX_META or meta_total > MAX_META_TOTAL:  # same limits in both modes
            raise ArtifactError("file-size", f"kept metadata exceeds {MAX_META} per file or {MAX_META_TOTAL} total")
        if name.endswith(".kv.gguf"):
            if name[:-len(".kv.gguf")] + ".gguf" not in sources:
                raise ArtifactError("file-set", f"{path}: no matching GGUF source")
        elif (sources.get(name, {}).get("bytes"), sources.get(name, {}).get("sha256")) != (f["bytes"], f["sha256"]):
            raise ArtifactError("file-set", f"{path}: not a verbatim copy of a recorded source")
    for path, f in files.items():
        if f["role"] == "source-metadata" and (deep or path.endswith(".kv.gguf")):
            data = _read_doc(root, path, f["bytes"], limit=MAX_META)
            if deep and hashlib.sha256(data).hexdigest() != f["sha256"]:
                raise ArtifactError("hash", path)
            if path.endswith(".kv.gguf"):
                kept = _check_kv_gguf(data, path)
                model = manifest["model"]
                arch = kept.get("general.architecture")
                if arch is not None and arch != model["architecture"]:
                    raise ArtifactError("meta", f"{path}: architecture disagrees with the manifest")
                for key, value in kept.items():
                    if key.endswith(".expert_count") and (key != f"{model['architecture']}.expert_count"
                                                          or not _exact(value, model["expert_count"])):
                        raise ArtifactError("meta", f"{path}: {key} disagrees with the manifest")
    data = None  # metadata is checked before the (larger) index is parsed
    ibytes = _read_doc(root, "index.json", files["index.json"]["bytes"])
    if hashlib.sha256(ibytes).hexdigest() != files["index.json"]["sha256"]:
        raise ArtifactError("hash", "index.json")
    index = strict_json(ibytes, "index.json")
    if ibytes != dumps(index):
        raise ArtifactError("canonical", "index.json is not in canonical form")
    del ibytes
    spans = _check_index(index, manifest, files)
    dedupe = collections.defaultdict(set)
    for t in manifest["transformations"]:
        if t["kind"] == "dedupe-identical":
            dedupe[t["resource"]].add(t["sha256"])
    by_shard = collections.defaultdict(list)
    for gid, g in enumerate(index["groups"]):
        by_shard[g["shard"]].append(gid)
    for i, s in enumerate(index["shards"]):
        whole = _check_shard(root, s, by_shard[i], spans, index["groups"], index, files, deep, dedupe)
        if deep and whole != files[s["path"]]["sha256"]:
            raise ArtifactError("hash", s["path"])
    return manifest, index


# ---------------------------------------------------------------- page-in model

def _group(index, gid):
    if not _is_int(gid) or not 0 <= gid < len(index["groups"]):
        raise ArtifactError("bounds", f"group {gid!r}")
    return index["groups"][gid]


def chunk_range(index, gid, k):
    """File (path, offset, length) of group chunk k: 4 KiB aligned, <= 2 MiB."""
    g = _group(index, gid)
    if not _is_int(k) or not 0 <= k < -(-g["stored_bytes"] // CHUNK):
        raise ArtifactError("bounds", f"chunk {k!r} of group {gid}")
    s = index["shards"][g["shard"]]
    return s["path"], s["data_offset"] + g["offset"] + k * CHUNK, min(CHUNK, g["stored_bytes"] - k * CHUNK)


def closure(index, gid, offset, readable):
    """Group-relative chunks a byte range needs: every chunk its readable bytes touch."""
    g = _group(index, gid)
    if not (_is_int(offset) and _is_int(readable) and offset >= 0 and readable >= 1
            and offset + readable <= g["stored_bytes"]):
        raise ArtifactError("bounds", f"range {offset!r}+{readable!r} outside group {gid}")
    return list(range(offset // CHUNK, (offset + readable - 1) // CHUNK + 1))


MAX_RUN = 64 << 20  # bounded request size; a tuning value, not an ABI field
try:
    IOV_MAX = os.sysconf("SC_IOV_MAX")
except (ValueError, OSError):
    IOV_MAX = 1024
IOV_MAX = IOV_MAX if IOV_MAX > 0 else 1024


def coalesce(index, missing, resident=frozenset(), max_run=MAX_RUN, max_segments=IOV_MAX):
    """Vectored direct reads for missing (group, chunk) pairs.

    A run is one contiguous file range in one shard, scattered into one iovec
    per chunk (each chunk has its own admitted destination, e.g. its own 2 MiB
    handle or slab slot). Runs break at resident chunks, which are never
    overwritten to bridge a gap, at shard boundaries, at max_run bytes and at
    max_segments iovecs (the kernel's IOV_MAX)."""
    keys = list(missing)
    for key in keys:
        if type(key) is not tuple or len(key) != 2 or not all(_is_int(x) for x in key):
            raise ArtifactError("bounds", f"chunk key {key!r}")
    runs = []
    for key in sorted(set(keys)):
        if key in resident:
            raise ArtifactError("state", f"chunk {key} is both missing and resident")
        path, off, n = chunk_range(index, *key)
        last = runs[-1] if runs else None
        if (last and last[0] == path and last[1] + last[2] == off and last[2] + n <= max_run
                and len(last[3]) < max_segments):
            last[2] += n
            last[3].append((key, n))
        else:
            runs.append([path, off, n, [(key, n)]])
    return [tuple(r) for r in runs]


def _read_runs(root, runs, slots, fds):
    """Issue each run as one preadv scattering chunks into their own slots."""
    for path, off, length, segs in runs:
        if path not in fds:
            fds[path] = os.open(Path(root) / path, os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
        iov = [memoryview(slots[key])[:n] for key, n in segs]
        if os.preadv(fds[path], iov, off) != length:
            raise ArtifactError("io", f"short read at {path}:{off}")


def loadcheck(root, src):
    """Page every resource in (readable-range chunk closure, vectored O_DIRECT
    reads into separate 2 MiB slots) and compare with the source's bytes."""
    _, index = verify(root, deep=False)
    by_name = {t["name"]: t for t in src["tensors"]}
    items = [(r["group"], r["offset"], r["bytes"], r["readable_bytes"], r["name"], by_name[r["roles"][0]], 0)
             for r in index["resources"]]
    for a in index["expert_arrays"]:
        items += [(a["first_group"] + e, a["group_offset"], a["slice_bytes"], a["readable_bytes"],
                   f"{a['name']}#{e}", by_name[a["name"]], e * a["slice_bytes"]) for e in range(a["count"])]
    chunks = requests = 0
    fds = {}
    try:
        for gid, off, n, readable, name, t, src_off in items:
            need = [(gid, k) for k in closure(index, gid, off, readable)]
            slots = {key: mmap.mmap(-1, CHUNK) for key in need}
            runs = coalesce(index, need)
            _read_runs(root, runs, slots, fds)
            got = b"".join(bytes(slots[key]) for key in need)
            start = off - need[0][1] * CHUNK
            with open(t["path"], "rb") as f:
                f.seek(t["offset"] + src_off)
                if got[start:start + n] != f.read(n):
                    raise ArtifactError("content", name)
            if any(got[start + n:start + readable]):
                raise ArtifactError("content", f"{name}: over-read bytes not zero")
            chunks += len(need)
            requests += len(runs)
            for m in slots.values():
                m.close()
    finally:
        for fd in fds.values():
            os.close(fd)
    return dict(resources=len(items), chunks_read=chunks, read_requests=requests)


def coldload(root, max_run=MAX_RUN, threads=8):
    """Timed whole-artifact load: every chunk missing, coalesced vectored reads
    into per-chunk 2 MiB slots, `threads` runs in flight. Returns throughput."""
    import concurrent.futures
    import time
    _, index = verify(root, deep=False)
    need = [(gid, k) for gid, g in enumerate(index["groups"]) for k in range(-(-g["stored_bytes"] // CHUNK))]
    runs = coalesce(index, need, max_run=max_run)
    per = max(len(r[3]) for r in runs)
    pools = [[mmap.mmap(-1, CHUNK) for _ in range(per)] for _ in range(threads)]
    fds = {s["path"]: os.open(Path(root) / s["path"], os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
           for s in index["shards"]}

    def one(i):
        path, off, length, segs = runs[i]
        pool = pools[i % threads]
        iov = [memoryview(pool[j])[:n] for j, (_, n) in enumerate(segs)]
        if os.preadv(fds[path], iov, off) != length:
            raise ArtifactError("io", "short read")
        return length

    total = 0
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(threads) as ex:
        for start in range(0, len(runs), threads):
            total += sum(ex.map(one, range(start, min(len(runs), start + threads))))
    dt = time.perf_counter() - t0
    for fd in fds.values():
        os.close(fd)
    return dict(bytes=total, chunks=len(need), requests=len(runs), max_run=max_run, threads=threads,
                seconds=round(dt, 3), gb_per_s=round(total / dt / 1e9, 3))


def main(argv):
    cmd, args = (argv[1], argv[2:]) if len(argv) > 1 else ("", [])
    if cmd == "plan":
        paths = [a for a in args if not a.startswith("--")]
        print(json.dumps(stats(plan(load_sources(paths), tie_check="--tie-check" in args)), indent=1))
    elif cmd == "build":
        out, paths = args[0], [a for a in args[1:] if not a.startswith("--meta=")]
        metas = [(f"meta/{Path(m[7:]).name}", m[7:]) for m in args[1:] if m.startswith("--meta=")]
        src = load_sources(paths)
        print(build(plan(src, tie_check=True), src, out, paths, metas))
    elif cmd == "verify":
        manifest, index = verify(args[0])
        print(json.dumps({"ok": True, "chunks": len(index["chunk_sha256"]), "files": len(manifest["files"])}))
    elif cmd == "loadcheck":
        print(json.dumps(loadcheck(args[0], load_sources(args[1:]))))
    elif cmd == "coldload":
        print(json.dumps(coldload(args[0], int(args[1]) << 20 if len(args) > 1 else MAX_RUN)))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
