#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""M3's importer: the M0 prototype (layout.py) run against pinned sources.

layout.py stays byte-for-byte the planner the retained-backing trace was
measured with (extract_library.py pins its SHA-256), so this file adds what
M3's imports need around it without copying or changing it:

- the sources must be the pinned download (D-054): each source file's name,
  size and SHA-256 must equal an entry of an M3 pins file
  (docs/experiments/fast-swap/pins.json), checked by layout.build in the same
  pass that hashes the bytes it copies;
- the converter identity names this file, its version and layout.py's
  SHA-256, so an artifact records the code that wrote it;
- layout.py is refused unless it is the pinned planner.

  python3 import_m3.py build OUT PINS MODEL_ID SOURCE...
  python3 import_m3.py verify ARTIFACT

GGUF sources (M3's DeepSeek V4 Flash) take layout.py's plan and writer.
Safetensors sources are Qwen3.8 Flash Next's ModelOpt checkpoint, whose
bytes are repacked on the way (modelopt_qwen38.py): its shards and the
config.json beside them are checked against the pins, and layout.py's
container, index and verifier write and check the artifact.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUT_SHA256 = "a0d1980a9eddd1adf60863cf7b825396700691fd17b3cd62ca063fda2e0b6809"
CONVERTER_NAME = "artifact-layout/import_m3.py"
IMPORTER_VERSION = "m3-1"


def load_layout():
    path = HERE / "layout.py"
    code = path.read_bytes()
    if hashlib.sha256(code).hexdigest() != LAYOUT_SHA256:
        raise SystemExit("layout.py is not the pinned M0 planner")
    spec = importlib.util.spec_from_file_location("layout", path)
    module = importlib.util.module_from_spec(spec)
    # Run the bytes that were hashed, not the file read again.
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102
    return module


def load_modelopt():
    """modelopt_qwen38.py, from the bytes its digest (in the converter
    version) is taken over."""
    path = HERE / "modelopt_qwen38.py"
    code = path.read_bytes()
    digest = hashlib.sha256(code).hexdigest()
    loaded = sys.modules.get("modelopt_qwen38")
    if loaded is not None and getattr(loaded, "_import_m3_digest", None) == digest:
        return loaded, digest  # one module object, so its functions pickle by name
    spec = importlib.util.spec_from_file_location("modelopt_qwen38", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modelopt_qwen38"] = module  # worker processes import it by name
    exec(compile(code, str(path), "exec"), module.__dict__)  # noqa: S102
    module._import_m3_digest = digest
    return module, digest


def converter():
    # IMPORTER_VERSION changes with anything here that changes what is
    # written; layout.py's identity is its pinned digest.
    return {"name": CONVERTER_NAME, "version": f"{IMPORTER_VERSION}+layout-{LAYOUT_SHA256[:16]}"}


def pinned_sources(pins_path, model_id, source_paths):
    """The recorded identity (name -> SHA-256) of each source, from the pins.
    Refused if the model is unknown, a source is not one of its files (by
    base name), two sources share a name, or a size differs from the pin."""
    pins = json.loads(Path(pins_path).read_text())
    models = [m for m in pins.get("models", []) if m.get("id") == model_id]
    if len(models) != 1:
        raise SystemExit(f"{model_id}: not exactly one model of that id in {pins_path}")
    files = {}
    for f in models[0].get("files", []):
        files.setdefault(Path(f["path"]).name, []).append(f)
    expected = {}
    for p in source_paths:
        name = Path(p).name
        if len(files.get(name, [])) != 1:
            raise SystemExit(f"{p}: not exactly one pinned file of {model_id} has this name")
        if name in expected:
            raise SystemExit(f"{name}: given twice")
        pin = files[name][0]
        size = Path(p).stat().st_size
        if size != pin["bytes"]:
            raise SystemExit(f"{p}: {size} bytes, pinned {pin['bytes']}")
        expected[name] = pin["sha256"]
    return expected


def main(argv):
    layout = load_layout()
    cmd, args = (argv[1], argv[2:]) if len(argv) > 1 else ("", [])
    if cmd == "build" and len(args) >= 4:
        out, pins, model_id, paths = args[0], args[1], args[2], args[3:]
        if all(Path(p).suffix == ".safetensors" for p in paths):
            modelopt, digest = load_modelopt()
            expected = pinned_sources(pins, model_id, [*paths, str(Path(paths[0]).parent / "config.json")])
            conv = {"name": CONVERTER_NAME,
                    "version": f"{converter()['version']}+modelopt_qwen38-{digest[:16]}"}
            final, _ = modelopt.build(layout, out, paths, expected=expected, converter=conv)
            print(final)
            return
        if any(Path(p).suffix != ".gguf" for p in paths):
            raise SystemExit("import_m3.py takes GGUF sources, or Qwen3.8's safetensors shards")
        expected = pinned_sources(pins, model_id, paths)
        src = layout.load_sources(paths)
        p = layout.plan(src, tie_check=True)
        print(json.dumps(layout.stats(p)), flush=True)
        print(layout.build(p, src, out, paths, (), converter=converter(), expected_sources=expected))
    elif cmd == "verify" and len(args) == 1:
        manifest, index = layout.verify(args[0])
        print(json.dumps({"ok": True, "chunks": len(index["chunk_sha256"]), "files": len(manifest["files"]),
                          "converter": manifest["converter"]}))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
