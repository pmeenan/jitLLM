# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for import_m3.py. Stdlib only:

  python3 -m unittest test_import_m3
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import import_m3
import test_layout


class ImportM3Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        # A tiny GGUF: one F32 table and one layer tensor.
        data = test_layout.gguf_bytes(
            {"general.architecture": (8, "llama")},
            [("token_embd.weight", [8, 4], 0, test_layout.rand(128, 1)),
             ("blk.0.attn_norm.weight", [8], 0, test_layout.rand(32, 2))])
        self.source = self.dir / "model-00001-of-00001.gguf"
        self.source.write_bytes(data)
        self.digest = hashlib.sha256(data).hexdigest()
        self.pins = self.dir / "pins.json"
        self.write_pins(self.digest, len(data))

    def tearDown(self):
        self.tmp.cleanup()

    def write_pins(self, digest, size, extra=()):
        files = [{"path": "Q/model-00001-of-00001.gguf", "bytes": size, "sha256": digest},
                 {"path": "README.md", "bytes": 1, "sha256": "0" * 64},
                 {"path": "drafter/README.md", "bytes": 1, "sha256": "0" * 64}, *extra]
        self.pins.write_text(json.dumps({"models": [{"id": "m", "files": files}]}))

    def test_the_layout_planner_is_the_pinned_one(self):
        layout = import_m3.load_layout()
        self.assertEqual(hashlib.sha256((import_m3.HERE / "layout.py").read_bytes()).hexdigest(),
                         import_m3.LAYOUT_SHA256)
        self.assertTrue(hasattr(layout, "build"))

    def test_sources_take_their_pinned_identity(self):
        # Two pinned files may share a base name (README.md) if neither is a source.
        expected = import_m3.pinned_sources(self.pins, "m", [str(self.source)])
        self.assertEqual(expected, {self.source.name: self.digest})

    def test_unpinned_resized_or_ambiguous_sources_are_refused(self):
        with self.assertRaises(SystemExit):
            import_m3.pinned_sources(self.pins, "other", [str(self.source)])
        other = self.dir / "other.gguf"
        other.write_bytes(b"GGUF")
        with self.assertRaises(SystemExit):
            import_m3.pinned_sources(self.pins, "m", [str(other)])
        with self.assertRaises(SystemExit):
            import_m3.pinned_sources(self.pins, "m", [str(self.source), str(self.source)])
        self.write_pins(self.digest, self.source.stat().st_size + 1)
        with self.assertRaises(SystemExit):
            import_m3.pinned_sources(self.pins, "m", [str(self.source)])
        self.write_pins(self.digest, self.source.stat().st_size,
                        extra=[{"path": "copy/model-00001-of-00001.gguf", "bytes": 1, "sha256": "1" * 64}])
        with self.assertRaises(SystemExit):
            import_m3.pinned_sources(self.pins, "m", [str(self.source)])

    def test_a_build_verifies_and_records_its_converter(self):
        out = self.dir / "store"
        import_m3.main(["import_m3.py", "build", str(out), str(self.pins), "m", str(self.source)])
        (artifact,) = [p for p in out.iterdir() if p.name != ".staging"]
        manifest, _ = import_m3.load_layout().verify(artifact)
        self.assertEqual(manifest["converter"], import_m3.converter())
        self.assertEqual(manifest["source"], [{"name": self.source.name, "bytes": self.source.stat().st_size,
                                               "sha256": self.digest}])

    def test_a_source_that_differs_from_its_pin_is_not_published(self):
        self.write_pins("f" * 64, self.source.stat().st_size)
        out = self.dir / "store"
        with self.assertRaises(ValueError):
            import_m3.main(["import_m3.py", "build", str(out), str(self.pins), "m", str(self.source)])
        # Refused before anything is staged or published.
        self.assertFalse(out.exists() and any(p.name != ".staging" for p in out.iterdir()))


if __name__ == "__main__":
    unittest.main()
