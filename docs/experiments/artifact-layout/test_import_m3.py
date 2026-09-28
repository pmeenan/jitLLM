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


def bf16(n, seed):
    return test_layout.rand(2 * n, seed)


class ComponentTest(unittest.TestCase):
    """A tiny diffusers-style checkpoint: two components that share the base
    name config.json, a scheduler and model_index.json."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "ckpt"
        files = {
            "model_index.json": json.dumps({"_class_name": "ToyPipeline", "scheduler": ["d", "S"],
                                            "text_encoder": ["t", "T"], "transformer": ["d", "D"]},
                                           indent=2).encode(),
            "scheduler/scheduler_config.json": b'{"_class_name": "S", "shift": 1}',
            "transformer/config.json": b'{"_class_name": "ToyTransformer2DModel", "num_layers": 2}',
            "transformer/diffusion_pytorch_model.safetensors": test_layout.st_bytes([
                ("img_in.weight", "BF16", [8, 4], bf16(32, 1)),
                ("transformer_blocks.0.attn.to_q.weight", "BF16", [8, 8], bf16(64, 2)),
                ("transformer_blocks.1.attn.to_q.weight", "BF16", [8, 8], bf16(64, 3))]),
            "text_encoder/config.json": b'{"model_type": "toy_vl"}',
            "text_encoder/generation_config.json": b'{"x": 1}',
            "text_encoder/model.safetensors": test_layout.st_bytes([
                ("model.language_model.embed_tokens.weight", "BF16", [16, 8], bf16(128, 4)),
                ("model.language_model.layers.0.mlp.up_proj.weight", "BF16", [8, 8], bf16(64, 5)),
                ("lm_head.weight", "BF16", [16, 8], bf16(128, 6))]),
        }
        pins = []
        for rel, data in files.items():
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_bytes(data)
            pins.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        self.pins = Path(self.tmp.name) / "pins.json"
        self.pins.write_text(json.dumps({"models": [{"id": "toy", "files": pins}]}))
        self.store = Path(self.tmp.name) / "store"

    def tearDown(self):
        self.tmp.cleanup()

    def component(self, role, *meta):
        layout = import_m3.load_layout()
        return import_m3.import_component(layout, self.store, self.pins, "toy", self.root, role, meta)

    def compose(self, *bindings, meta=("scheduler/scheduler_config.json",)):
        layout = import_m3.load_layout()
        return import_m3.compose(layout, self.store, self.pins, "toy", self.root, bindings, meta)

    def test_a_diffusers_component_is_named_by_its_class_and_grouped_by_block(self):
        artifact = self.component("transformer")
        manifest, index = import_m3.load_layout().verify(artifact)
        self.assertEqual(manifest["model"]["architecture"], "ToyTransformer2DModel")
        self.assertEqual(manifest["converter"], import_m3.converter("component"))
        self.assertEqual(sorted((g["kind"], g["layer"]) for g in index["groups"]),
                         [("global", None), ("layer", 0), ("layer", 1)])
        self.assertEqual({s["name"] for s in manifest["source"]},
                         {"config.json", "diffusion_pytorch_model.safetensors"})

    def test_the_text_encoder_keeps_its_table_as_rows_and_its_metadata(self):
        artifact = self.component("text_encoder", "text_encoder/generation_config.json")
        manifest, index = import_m3.load_layout().verify(artifact)
        self.assertEqual(manifest["model"]["architecture"], "toy_vl")
        table = [r for r in index["resources"] if r["name"] == "model.language_model.embed_tokens.weight"]
        self.assertEqual(table[0].get("access"), "rows")
        self.assertIn("meta/generation_config.json", {f["path"] for f in manifest["files"]})
        self.assertIn(("layer", 0), {(g["kind"], g["layer"]) for g in index["groups"]})

    def test_the_gguf_path_is_untouched_by_a_component_import(self):
        self.component("transformer")
        fresh = import_m3.load_layout()  # each load is its own module
        self.assertEqual(fresh.LAYER.pattern, r"(?:blk|model\.layers)\.(\d+)\.")
        self.assertEqual(import_m3.converter(), {"name": import_m3.CONVERTER_NAME,
                                                 "version": "m3-1+layout-" + import_m3.LAYOUT_SHA256[:16]})

    def test_sources_are_pinned_by_path_not_base_name(self):
        (self.root / "transformer/config.json").write_bytes(b'{"_class_name": "Other2DModelXX", "n": 1}')
        with self.assertRaises((SystemExit, ValueError)):
            self.component("transformer")
        with self.assertRaises(SystemExit):
            import_m3.pinned_by_path(self.pins, "toy", self.root, ["../ckpt/model_index.json"])
        with self.assertRaises(SystemExit):
            import_m3.pinned_by_path(self.pins, "toy", self.root,
                                     ["text_encoder/config.json", "transformer/config.json"])
        with self.assertRaises(SystemExit):
            self.component("vision")

    def test_a_composition_names_its_components_and_verifies(self):
        te = self.component("text_encoder")
        dit = self.component("transformer")
        comp = self.compose(f"transformer={dit.name}", f"text_encoder={te.name}")
        layout = import_m3.load_layout()
        doc = import_m3.verify_composition(layout, comp, store=self.store)
        self.assertEqual(doc["model"]["architecture"], "ToyPipeline")
        self.assertEqual([(c["role"], c["artifact"], c["architecture"]) for c in doc["components"]],
                         [("text_encoder", te.name, "toy_vl"),
                          ("transformer", dit.name, "ToyTransformer2DModel")])
        self.assertEqual(sorted(f["path"] for f in doc["files"]),
                         ["meta/model_index.json", "meta/scheduler_config.json"])
        # Deterministic: the same inputs give the same ID, verified, not reused blindly.
        self.assertEqual(self.compose(f"text_encoder={te.name}", f"transformer={dit.name}"), comp)
        # The artifacts themselves are unchanged and still verify.
        layout.verify(te)
        layout.verify(dit)

    def test_bindings_are_refused_unless_published_and_listed(self):
        dit = self.component("transformer")
        with self.assertRaises(SystemExit):
            self.compose(f"vae={dit.name}")  # not an entry of model_index.json
        with self.assertRaises(SystemExit):
            self.compose(f"transformer={dit.name}", f"transformer={dit.name}")
        with self.assertRaises(SystemExit):
            self.compose("transformer=xyz")
        with self.assertRaises(Exception):
            self.compose("transformer=" + "a" * 64)  # no such artifact

    def test_a_damaged_composition_is_refused_with_its_rule(self):
        dit = self.component("transformer")
        comp = self.compose(f"transformer={dit.name}")
        layout = import_m3.load_layout()

        def code(path=comp, store=None, expected_id=None):
            with self.assertRaises(layout.ArtifactError) as e:
                import_m3.verify_composition(layout, path, expected_id=expected_id, store=store)
            return e.exception.code

        meta = comp / "meta" / "scheduler_config.json"
        good = meta.read_bytes()
        meta.write_bytes(good.replace(b"1", b"2"))
        self.assertEqual(code(), "hash")
        meta.write_bytes(good)
        (comp / "meta" / "extra.json").write_bytes(b"{}")
        self.assertEqual(code(), "file-set")
        (comp / "meta" / "extra.json").unlink()
        (comp / "data").mkdir()
        self.assertEqual(code(), "file-set")
        (comp / "data").rmdir()
        self.assertEqual(code(expected_id="0" * 64), "identity")
        manifest = (comp / "manifest.json").read_bytes()
        doc = json.loads(manifest)
        (comp / "manifest.json").write_bytes(json.dumps(doc, indent=1).encode())
        self.assertEqual(code(), "canonical")
        doc["components"][0]["architecture"] = "Another"
        mbytes = layout.dumps(doc)
        (comp / "manifest.json").write_bytes(mbytes)
        self.assertEqual(code(store=self.store, expected_id=hashlib.sha256(mbytes).hexdigest()), "component")
        doc["experimental"] = False
        mbytes = layout.dumps(doc)
        (comp / "manifest.json").write_bytes(mbytes)
        self.assertEqual(code(expected_id=hashlib.sha256(mbytes).hexdigest()), "schema")
        doc["format"] = "jitllm-artifact"
        (comp / "manifest.json").write_bytes(layout.dumps(doc))
        self.assertEqual(code(), "format")
        # An artifact is not a composition, and a composition is not an artifact.
        self.assertEqual(code(path=dit), "format")
        (comp / "manifest.json").write_bytes(manifest)
        with self.assertRaises(layout.ArtifactError):
            layout.verify(comp)


if __name__ == "__main__":
    unittest.main()
