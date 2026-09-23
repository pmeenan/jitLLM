# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the v0 layout reference model. Stdlib only; runs anywhere:

  python3 -m unittest test_layout
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import tempfile
import unittest

import layout as L


def gguf_bytes(kv, tensors, alignment=32):
    """Minimal GGUF v3 writer: kv {key: (type, value)}, tensors [(name, ne, type, data)]."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, (t, v) in kv.items():
        out += s(k) + struct.pack("<I", t) + (s(v) if t == 8 else struct.pack("<" + L._SCALAR[t], v))
    off = 0
    for name, ne, ty, data in tensors:
        out += s(name) + struct.pack("<I", len(ne)) + struct.pack("<" + "Q" * len(ne), *ne) + struct.pack("<IQ", ty, off)
        off = L.align(off + len(data), alignment)
    out += b"\0" * (L.align(len(out), alignment) - len(out))
    for _, _, _, data in tensors:
        out += data + b"\0" * (L.align(len(data), alignment) - len(data))
    return out


def rand(n, seed):
    return random.Random(seed).randbytes(n)


def toy_tensors():
    # Tiny MoE: 2 layers, 4 experts. Q4_K gate slices have ne0=256, so pinned
    # GGML over-reads 144 bytes past each (ne0 % 512 != 0); Q6_K down slices
    # have ne0=3072 (no over-read). A tied F16 embedding/output pair and small
    # F32 norms/routers share groups.
    emb = rand(2 * 256 * 64, 1)
    tensors = [("token_embd.weight", [256, 64], 1, emb)]
    for layer in range(2):
        tensors += [
            (f"blk.{layer}.attn_norm.weight", [256], 0, rand(1024, 10 + layer)),
            (f"blk.{layer}.attn_q.weight", [256, 3000], 1, rand(2 * 256 * 3000, 20 + layer)),
            (f"blk.{layer}.ffn_gate_inp.weight", [256, 4], 0, rand(4 * 256 * 4, 30 + layer)),
            (f"blk.{layer}.ffn_gate_exps.weight", [256, 3000, 4], 12, rand(144 * 3000 * 4, 40 + layer)),
            (f"blk.{layer}.ffn_down_exps.weight", [3072, 256, 4], 14, rand(210 * 12 * 256 * 4, 50 + layer)),
        ]
    return tensors + [("output_norm.weight", [256], 0, rand(1024, 60)), ("output.weight", [256, 64], 1, emb)]


KV = {"general.architecture": (8, "toy"), "toy.expert_count": (4, 4), "general.alignment": (4, 32)}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.src_path = self.tmp / "toy.gguf"
        self.src_path.write_bytes(gguf_bytes(KV, toy_tensors()))
        self.src = L.load_sources([str(self.src_path)])
        self.plan = L.plan(self.src, tie_check=True, shard_target=6 << 20)
        self.out = self.tmp / "installed"

    def build(self, **kw):
        return L.build(self.plan, self.src, self.out, [str(self.src_path)], **kw)


class PlanTests(Fixture):
    def test_groups_start_aligned_and_members_packed(self):
        for g in self.plan["groups"]:
            self.assertEqual(g["file_offset"] % L.FILE_ALIGN, 0)
            self.assertEqual(g["stored"], L.align(g["used"], L.FILE_ALIGN))
            for m in g["members"]:
                self.assertEqual(m["offset"] % L.MEMBER_ALIGN, 0)
        layer0 = next(g for g in self.plan["groups"] if g["kind"] == "layer" and g["layer"] == 0)
        self.assertEqual([m["name"] for m in layer0["members"]],
                         ["blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.ffn_gate_inp.weight"])

    def test_expert_slices_are_contiguous_uniform_groups(self):
        arrays = self.plan["expert_arrays"]
        self.assertEqual([a["name"] for a in arrays], ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
                                                       "blk.1.ffn_gate_exps.weight", "blk.1.ffn_down_exps.weight"])
        gate = arrays[0]
        self.assertEqual(gate["slice_bytes"], 144 * 3000)
        groups = [self.plan["groups"][gate["first_group"] + e] for e in range(4)]
        self.assertEqual([(g["kind"], g["layer"], g["expert"]) for g in groups], [("expert", 0, e) for e in range(4)])
        self.assertEqual(len({g["stored"] for g in groups}), 1)
        self.assertEqual([m["name"] for m in groups[2]["members"]],
                         ["blk.0.ffn_gate_exps.weight#2", "blk.0.ffn_down_exps.weight#2"])
        self.assertEqual(groups[2]["members"][0]["src_offset"], 2 * 144 * 3000)

    def test_ggml_row_padding_over_read_is_reserved(self):
        gate, down = self.plan["expert_arrays"][:2]
        self.assertEqual(gate["readable"], gate["slice_bytes"] + 144)  # 256 more Q4_K elements
        self.assertEqual(down["readable"], down["slice_bytes"])       # ne0 = 3072 = 6 * 512
        self.assertGreaterEqual(down["group_offset"], gate["group_offset"] + gate["readable"])

    def test_tied_identical_tensors_stored_once(self):
        self.assertEqual(self.plan["ties"]["output.weight"][0], "token_embd.weight")
        head = next(g for g in self.plan["groups"] if g["kind"] == "head")
        emb = next(m for m in head["members"] if m["name"] == "token_embd.weight")
        self.assertEqual(emb["roles"], ["token_embd.weight", "output.weight"])
        self.assertIn("output_norm.weight", [m["name"] for m in head["members"]])

    def test_different_tensors_are_not_deduplicated(self):
        path = self.tmp / "untied.gguf"
        path.write_bytes(gguf_bytes({"general.architecture": (8, "toy")},
                                    [("token_embd.weight", [256, 64], 1, rand(32768, 1)),
                                     ("output.weight", [256, 64], 1, rand(32768, 2))]))
        self.assertEqual(L.plan(L.load_sources([str(path)]), tie_check=True)["ties"], {})

    def test_groups_never_span_shards(self):
        self.assertGreater(len(self.plan["shards"]), 1)
        for s in self.plan["shards"]:
            self.assertLessEqual(s["stored"], 6 << 20)

    def test_stats_report_padding_and_views(self):
        st = L.stats(self.plan)
        self.assertLess(st["disk_padding_pct"], 2)
        self.assertGreater(st["handle_padding_pct"], st["disk_padding_pct"])
        self.assertEqual(st["readable_overread_bytes"], 2 * 4 * 144)
        va = st["expert_view_va_bytes"]
        # lcm(2 MiB, 144, 210) is 630 MiB: uniform views dwarf pointer tables.
        self.assertEqual(va["uniform_stride_one_array"], 2 * 4 * 630 * (1 << 20))
        self.assertEqual(va["pointer_table_handles"], 2 * 4 * L.CHUNK)

    def test_unrepresentable_inputs_rejected_by_importer(self):
        bad = self.tmp / "bad.gguf"
        bad.write_bytes(gguf_bytes({"general.architecture": (8, "toy")}, [("blk.0.bad name", [256], 0, rand(1024, 1))]))
        with self.assertRaises(ValueError):
            L.plan(L.load_sources([str(bad)]))
        bad.write_bytes(gguf_bytes({"general.architecture": (8, "toy"), "general.alignment": (4, 0)},
                                   [("x", [256], 0, rand(1024, 1))]))
        with self.assertRaises(ValueError):
            L.load_sources([str(bad)])


def st_bytes(entries):
    """Minimal safetensors writer: entries [(name, dtype, shape, data)]."""
    header, off, blobs = {}, 0, b""
    for name, dtype, shape, data in entries:
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(data)]}
        off += len(data)
        blobs += data
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + blobs


class Exl3ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "config.json").write_text('{"model_type": "qwen2"}')

    def entries(self, k=4, drop=(), extra=()):
        p = "model.layers.0.mlp.up_proj"
        out = [(f"{p}.trellis", "I16", [2, 4, 16 * k], rand(2 * 4 * 16 * k * 2, 1)),
               (f"{p}.suh", "F16", [32], rand(64, 2)), (f"{p}.svh", "F16", [64], rand(128, 3)),
               (f"{p}.mcg", "I32", [], rand(4, 4))]
        return [e for e in out if e[0].rsplit(".", 1)[1] not in drop] + list(extra)

    def plan(self, entries):
        path = self.tmp / "model.safetensors"
        path.write_bytes(st_bytes(entries))
        return L.plan(L.load_sources([str(path)]))

    def test_config_is_source_identity(self):
        path = self.tmp / "model.safetensors"
        path.write_bytes(st_bytes(self.entries()))
        src = L.load_sources([str(path)])
        p = L.plan(src)
        a = L.build(p, src, self.tmp / "out", [str(path)])
        L.verify(a)
        names = [x["name"] for x in json.loads((a / "manifest.json").read_bytes())["source"]]
        self.assertEqual(names, ["config.json", "model.safetensors"])
        (self.tmp / "config.json").write_text('{"model_type": "llama"}')
        with self.assertRaises(ValueError):
            L.build(p, src, self.tmp / "out", [str(path)])

    def test_supported_fixture_shape_plans(self):
        p = self.plan(self.entries())
        rep = next(m["repr"] for g in p["groups"] for m in g["members"] if m["name"].endswith("trellis"))
        self.assertEqual((rep["k_bits"], rep["codebook"], rep["in_features"], rep["out_features"]), (4, "mcg", 32, 64))

    def test_unsupported_variants_rejected(self):
        p = "model.layers.0.mlp.up_proj"
        for entries in (self.entries(k=9), self.entries(k=3), self.entries(drop=("mcg",)),
                        self.entries(drop=("suh",)), self.entries(drop=("svh",)),
                        self.entries(drop=("suh",), extra=[(f"{p}.su", "I16", [2], rand(4, 5))]),
                        self.entries(extra=[("model.layers.1.mlp.up_proj.svh", "F16", [64], rand(128, 6))])):
            with self.assertRaises((ValueError, L.ArtifactError)):
                self.plan(entries)


class RowAndReadTests(unittest.TestCase):
    def index(self, sizes):
        groups, off, chunk = [], 0, 0
        for n in sizes:
            groups.append({"shard": 0, "offset": off, "stored_bytes": n, "first_chunk": chunk})
            off += n
            chunk += -(-n // L.CHUNK)
        return {"groups": groups, "shards": [{"path": "data/00000.safetensors", "data_offset": 4096}]}

    def test_rows_straddling_a_chunk_need_both_chunks(self):
        n = 10**6
        idx = self.index([4096, L.align(90 * n, 4096)])
        idx["resources"] = [{"name": "t", "access": "rows", "group": 1, "offset": 0, "bytes": 90 * n,
                             "repr": {"family": "ggml", "type": "IQ4_NL", "ne": [160, n]}}]
        per = L.CHUNK // 90
        self.assertEqual(L.rows_to_chunks(idx, "t", [0, 1]), [(1, 0)])
        self.assertEqual(L.rows_to_chunks(idx, "t", [per]), [(1, 0), (1, 1)])  # straddles
        self.assertEqual(L.rows_to_chunks(idx, "t", [n - 1]), [(1, (90 * n - 1) // L.CHUNK)])
        for bad in (-1, n, True, 1.0):
            with self.assertRaises(L.ArtifactError):
                L.rows_to_chunks(idx, "t", [bad])
        with self.assertRaises(L.ArtifactError):
            L.rows_to_chunks(idx, "not-a-table", [0])

    def test_closure_covers_readable_bytes_within_the_group(self):
        idx = self.index([3 * L.CHUNK])
        self.assertEqual(L.closure(idx, 0, L.CHUNK - 256, 512), [0, 1])
        self.assertEqual(L.closure(idx, 0, 0, L.CHUNK), [0])
        for args in ((0, 0, 4 * L.CHUNK), (1, 0, 1), (-1, 0, 1), (0, -256, 512), (0, 0, 0)):
            with self.assertRaises(L.ArtifactError):
                L.closure(idx, *args)

    def test_coalesce_merges_file_adjacent_chunks_across_groups(self):
        idx = self.index([L.CHUNK + 8192, 4096, 2 * L.CHUNK])
        runs = L.coalesce(idx, [(0, 0), (0, 1), (1, 0), (2, 0)])
        self.assertEqual(len(runs), 1)
        _, off, length, segs = runs[0]
        self.assertEqual((off, length), (4096, L.CHUNK + 8192 + 4096 + L.CHUNK))
        self.assertEqual([n for _, n in segs], [L.CHUNK, 8192, 4096, L.CHUNK])  # one iovec per chunk

    def test_coalesce_never_bridges_resident_chunks(self):
        idx = self.index([3 * L.CHUNK])
        runs = L.coalesce(idx, [(0, 0), (0, 2)], resident={(0, 1)})
        self.assertEqual([len(r[3]) for r in runs], [1, 1])
        with self.assertRaises(L.ArtifactError):
            L.coalesce(idx, [(0, 1)], resident={(0, 1)})

    def test_coalesce_rejects_out_of_range_keys(self):
        idx = self.index([L.CHUNK, L.CHUNK])
        for key in ((-1, 0), (0, 1), (1, -1), (2, 0), (True, 0), ([0], 0), ("x", 0), (0, 0, 0), 5):
            with self.assertRaises(L.ArtifactError):
                L.coalesce(idx, [key])

    def test_coalesce_bounds_run_size(self):
        idx = self.index([8 * L.CHUNK])
        runs = L.coalesce(idx, [(0, k) for k in range(8)], max_run=4 * L.CHUNK)
        self.assertEqual([r[2] for r in runs], [4 * L.CHUNK, 4 * L.CHUNK])

    def test_coalesce_bounds_iovec_count(self):
        idx = self.index([4096] * 1025)  # 1,025 adjacent small groups
        runs = L.coalesce(idx, [(g, 0) for g in range(1025)], max_segments=1024)
        self.assertEqual([len(r[3]) for r in runs], [1024, 1])
        self.assertLessEqual(max(len(r[3]) for r in L.coalesce(idx, [(g, 0) for g in range(1025)])), L.IOV_MAX)


class BuildVerifyTests(Fixture):
    def test_build_verify_and_deterministic_identity(self):
        a = self.build()
        L.verify(a)
        self.assertEqual(a.name, hashlib.sha256((a / "manifest.json").read_bytes()).hexdigest())
        self.assertEqual(self.build(), a)  # same input, same bytes, same id
        self.assertEqual(L.installed(self.out), [a.name])
        self.assertEqual([d for d in (self.out / ".staging").iterdir() if d.suffix != ".lock"], [])
        with open(a / "data/00000.safetensors", "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            self.assertEqual((8 + n) % L.FILE_ALIGN, 0)

    def test_loadcheck_matches_source_and_zero_over_read(self):
        r = L.loadcheck(self.build(), self.src)
        self.assertEqual(r["resources"], 1 + 2 * 3 + 2 * 2 * 4 + 1)

    def test_kv_metadata_is_zero_tensor_gguf(self):
        a = self.build()
        kv = L.read_gguf(next((a / "meta").glob("*.kv.gguf")))
        self.assertEqual((kv["tensors"], kv["meta"]["toy.expert_count"]), ([], 4))

    def test_interrupted_import_is_never_published_and_is_swept(self):
        with self.assertRaises(KeyboardInterrupt):
            self.build(crash_after=0)
        self.assertEqual(L.installed(self.out), [])
        stale = self.out / ".staging" / "job-99999"  # another process's leftover
        stale.mkdir()
        self.build()  # a restarted job replaces its own deterministic staging directory
        self.assertEqual(len(L.installed(self.out)), 1)
        self.assertEqual(L.sweep_staging(self.out), ["job-99999"])

    def test_existing_damaged_artifact_is_not_reused(self):
        a = self.build()
        path = a / "data/00000.safetensors"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaises(L.ArtifactError):
            self.build()  # never reports success over a damaged copy

    def test_symlinked_staging_is_refused(self):
        victim = self.tmp / "victim"
        (victim / "job-1").mkdir(parents=True)
        (victim / "job-1" / "precious.txt").write_text("x")
        self.out.mkdir()
        os.symlink(victim, self.out / ".staging")
        with self.assertRaises(L.ArtifactError):
            L.sweep_staging(self.out)
        with self.assertRaises(L.ArtifactError):
            self.build()
        self.assertTrue((victim / "job-1" / "precious.txt").exists())

    def test_live_job_is_locked_and_not_swept(self):
        self.build()
        staging = self.out / ".staging"
        (staging / "job-live").mkdir()
        fd = L._try_lock(staging / "job-live.lock")
        try:
            self.assertIsNone(L._try_lock(staging / "job-live.lock"))
            self.assertNotIn("job-live", L.sweep_staging(self.out))
            self.assertTrue((staging / "job-live").is_dir())
        finally:
            os.close(fd)
        self.assertEqual(L.sweep_staging(self.out), ["job-live"])

    def test_job_names_depend_on_content(self):
        a = L.job_name([["model.gguf", "0" * 64]], [], {"name": "x", "version": "1"})
        b = L.job_name([["model.gguf", "1" * 64]], [], {"name": "x", "version": "1"})
        self.assertNotEqual(a, b)

    def test_source_identity_and_ties_rechecked(self):
        with self.assertRaises(ValueError):
            self.build(expected_sources={"toy.gguf": "0" * 64})
        # The planned tie no longer holds once the source changes after planning.
        t = next(t for t in self.src["tensors"] if t["name"] == "output.weight")
        data = bytearray(self.src_path.read_bytes())
        data[t["offset"]] ^= 1
        self.src_path.write_bytes(data)
        with self.assertRaises(ValueError):
            self.build()

    def test_source_changed_after_planning_rejected(self):
        data = self.src_path.read_bytes()
        kv = dict(KV, **{"general.name": (8, "shifted")})  # same tensors, shifted offsets
        self.src_path.write_bytes(gguf_bytes(kv, toy_tensors()))
        self.assertNotEqual(data, self.src_path.read_bytes())
        with self.assertRaises(ValueError):
            self.build()

    def test_source_metadata_changed_after_planning_rejected(self):
        kv = dict(KV, **{"general.architecture": (8, "abc"), "abc.expert_count": (4, 4)})
        del kv["toy.expert_count"]
        self.src_path.write_bytes(gguf_bytes(kv, toy_tensors()))
        with self.assertRaises(ValueError):
            self.build()

    def test_importer_rejects_duplicate_gguf_keys(self):
        s = lambda x: struct.pack("<Q", len(x)) + x
        kv = s(b"general.architecture") + struct.pack("<I", 8) + s(b"llama")
        bad = self.tmp / "dup.gguf"
        bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 2) + kv + kv.replace(b"llama", b"toy!!"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            L.load_sources([str(bad)])

    def test_plan_from_other_sources_rejected(self):
        other = self.tmp / "copy"
        other.mkdir()
        shutil.copy(self.src_path, other / "toy.gguf")
        src2 = L.load_sources([str(other / "toy.gguf")])
        with self.assertRaises(ValueError):
            L.build(self.plan, src2, self.out, [str(other / "toy.gguf")])

    def test_shallow_verify_rejects_hard_linked_metadata(self):
        tok = self.tmp / "tokenizer.json"
        tok.write_text("{}")
        a = self.build(meta_files=[("meta/tokenizer.json", str(tok))])
        os.link(a / "meta" / "tokenizer.json", self.tmp / "elsewhere")
        with self.assertRaises(L.ArtifactError) as cm:
            L.verify(a, deep=False)
        self.assertEqual(cm.exception.code, "file-type")

    def test_conflicting_architectures_across_parts_rejected(self):
        other = self.tmp / "other.gguf"
        other.write_bytes(gguf_bytes({"general.architecture": (8, "llama")}, [("x", [256], 0, rand(1024, 1))]))
        with self.assertRaises(ValueError):
            L.load_sources([str(self.src_path), str(other)])

    def test_expert_count_must_be_u32_and_parts_must_agree(self):
        f32 = self.tmp / "f32.gguf"
        f32.write_bytes(gguf_bytes({"general.architecture": (8, "toy"), "toy.expert_count": (6, 4.0)},
                                   [("x", [256], 0, rand(1024, 1))]))
        with self.assertRaises(ValueError):
            L.load_sources([str(f32)])
        a = self.tmp / "a.gguf"
        b = self.tmp / "b.gguf"
        a.write_bytes(gguf_bytes({"general.architecture": (8, "toy"), "toy.expert_count": (4, 8)},
                                 [("x", [256], 0, rand(1024, 1))]))
        b.write_bytes(gguf_bytes({"general.architecture": (8, "toy"), "toy.expert_count": (4, 4)},
                                 [("y", [256], 0, rand(1024, 2))]))
        with self.assertRaises(ValueError):
            L.load_sources([str(a), str(b)])

    def test_source_changed_between_hash_and_copy_rejected(self):
        original = L._identity_pass
        calls = []

        def racing(*args, **kw):
            result = original(*args, **kw)
            if not calls:  # after the identity pass, before the copy
                t = next(t for t in self.src["tensors"] if t["name"] == "blk.0.attn_q.weight")
                data = bytearray(self.src_path.read_bytes())
                data[t["offset"]] ^= 1
                self.src_path.write_bytes(data)
            calls.append(1)
            return result
        self.patch_attr("_identity_pass", racing)
        with self.assertRaisesRegex(ValueError, "changed during import"):
            self.build()

    def after_identity_pass(self, mutate):
        original = L._identity_pass
        calls = []

        def racing(*args, **kw):
            result = original(*args, **kw)
            if not calls:
                mutate()
            calls.append(1)
            return result
        self.patch_attr("_identity_pass", racing)

    def test_kept_metadata_changed_after_hashing_rejected(self):
        # Change the KV section after hashing and re-parsing (at staging time),
        # so only the kept-metadata digest check can catch it.
        original = L._staging_dir

        def racing(out_root):
            data = self.src_path.read_bytes()
            self.src_path.write_bytes(data.replace(b"toy", b"toz", 1))
            return original(out_root)
        self.patch_attr("_staging_dir", racing)
        with self.assertRaisesRegex(ValueError, "metadata changed during import"):
            self.build()

    def test_header_changed_after_planning_rejected(self):
        data = self.src_path.read_bytes()
        self.src_path.write_bytes(data.replace(b"attn_norm", b"attn_nrom", 1))  # same length, same offsets
        with self.assertRaisesRegex(ValueError, "changed after planning"):
            self.build()

    def test_meta_file_is_source_identity_and_checked(self):
        meta = self.tmp / "tokenizer.json"
        meta.write_text('{"v": 1}')
        a = self.build(meta_files=[("meta/tokenizer.json", str(meta))])
        names = [x["name"] for x in json.loads((a / "manifest.json").read_bytes())["source"]]
        self.assertEqual(names, ["tokenizer.json", "toy.gguf"])
        self.after_identity_pass(lambda: meta.write_text('{"v": 2}'))
        with self.assertRaisesRegex(ValueError, "changed during import"):
            self.build(meta_files=[("meta/tokenizer.json", str(meta))])

    def patch_attr(self, name, value):
        old = getattr(L, name)
        setattr(L, name, value)
        self.addCleanup(setattr, L, name, old)

    def test_colliding_metadata_paths_rejected(self):
        kv = self.tmp / "toy.kv.gguf"
        kv.write_bytes(b"x")
        with self.assertRaises(ValueError):  # collides with the generated kept GGUF metadata
            self.build(meta_files=[("meta/toy.kv.gguf", str(kv))])
        with self.assertRaises(ValueError):
            self.build(meta_files=[("meta/../x", str(kv))])

    def test_kept_metadata_must_be_named_source_files(self):
        tok = self.tmp / "tokenizer.json"
        tok.write_text("{}")
        with self.assertRaises(ValueError):
            self.build(meta_files=[("meta/tokenizer.json", b"{}")])       # not a file
        with self.assertRaises(ValueError):
            self.build(meta_files=[("meta/config.json", str(tok))])      # published under another name

    def test_oversized_metadata_rejected_before_writing(self):
        big = self.tmp / "big.json"
        big.write_bytes(b"x" * 1024)
        self.patch_attr("MAX_META", 512)
        with self.assertRaises(ValueError):
            self.build(meta_files=[("meta/big.json", str(big))])
        self.assertFalse((self.out / ".staging").exists() and any(
            d.suffix != ".lock" for d in (self.out / ".staging").iterdir()))

    def test_shared_blob_names_are_both_identity(self):
        blob = self.tmp / "blob"
        blob.write_text("{}")
        (self.tmp / "hf").mkdir()
        for n in ("special_tokens_map.json", "added_tokens.json"):
            os.symlink(blob, self.tmp / "hf" / n)
        a = self.build(meta_files=[(f"meta/{n}", str(self.tmp / "hf" / n))
                                   for n in ("added_tokens.json", "special_tokens_map.json")])
        names = [x["name"] for x in json.loads((a / "manifest.json").read_bytes())["source"]]
        self.assertEqual(names, ["added_tokens.json", "special_tokens_map.json", "toy.gguf"])


class NegativeTests(Fixture):
    def setUp(self):
        super().setUp()
        self.a = self.build()

    # -- helpers: mutate one document, then re-hash everything else so only the
    # rule under test can fire.
    def publish(self, mbytes):
        (self.a / "manifest.json").write_bytes(mbytes)
        new = self.a.with_name(hashlib.sha256(mbytes).hexdigest())
        os.rename(self.a, new)
        self.a = new

    def relist(self, mutate_manifest=None, raw=None):
        m = json.loads((self.a / "manifest.json").read_bytes())
        for f in m["files"]:
            data = (self.a / f["path"]).read_bytes()
            f["bytes"], f["sha256"] = len(data), hashlib.sha256(data).hexdigest()
        if mutate_manifest:
            mutate_manifest(m)
        self.publish(raw(m) if raw else L.dumps(m))

    def index(self, fn, raw=None):
        path = self.a / "index.json"
        doc = json.loads(path.read_bytes())
        fn(doc)
        path.write_bytes(raw(doc) if raw else L.dumps(doc))
        self.relist()

    def header(self, fn):
        """Rewrite shard 0's JSON header in place (same length) and fix its hash."""
        path = self.a / "data/00000.safetensors"
        data = bytearray(path.read_bytes())
        (n,) = struct.unpack("<Q", data[:8])
        text = fn(bytes(data[8:8 + n]).rstrip(b" "))
        self.assertLessEqual(len(text), n)
        data[8:8 + n] = text + b" " * (n - len(text))
        path.write_bytes(data)
        off = json.loads((self.a / "index.json").read_bytes())["shards"][0]["data_offset"]
        self.index(lambda i: i["shards"][0].update(header_sha256=hashlib.sha256(bytes(data[:off])).hexdigest()))

    def expect(self, code):
        with self.assertRaises(L.ArtifactError) as cm:
            L.verify(self.a)
        self.assertEqual(cm.exception.code, code, str(cm.exception))

    def entries(self):
        path = self.a / "data/00000.safetensors"
        raw = path.read_bytes()
        (n,) = struct.unpack("<Q", raw[:8])
        return raw, 8 + n, json.loads(raw[8:8 + n])

    # -- versions, identity, encoding
    def test_unsupported_version_requires_reimport(self):
        self.relist(lambda m: m.update(format_version=1))
        self.expect("unsupported-version")

    def test_type_confused_versions_rejected(self):
        self.relist(lambda m: m.update(format_version=False))
        self.expect("unsupported-version")

    def test_float_profile_rejected(self):
        self.relist(raw=lambda m: L.dumps(m).replace(b'"file_alignment":4096', b'"file_alignment":4096.0'))
        self.expect("json")

    def test_other_profile_rejected(self):
        self.relist(lambda m: m["layout"].update(chunk_bytes=1 << 20))
        self.expect("unsupported-profile")

    def test_non_canonical_manifest_rejected(self):
        self.relist(raw=lambda m: json.dumps(m, sort_keys=True, indent=1).encode())
        self.expect("canonical")

    def test_bom_and_utf16_manifest_rejected(self):
        self.relist(raw=lambda m: b"\xef\xbb\xbf" + L.dumps(m))
        self.expect("json")
        self.relist(raw=lambda m: L.dumps(m).decode().encode("utf-16"))
        self.expect("json")

    def test_duplicate_json_key_rejected(self):
        self.relist(raw=lambda m: L.dumps(m).replace(b'{"converter"', b'{"format":"x","converter"', 1))
        self.expect("json")

    def test_renamed_directory_breaks_identity(self):
        os.rename(self.a, self.a.with_name("0" * 64))
        self.a = self.a.with_name("0" * 64)
        self.expect("identity")

    def test_symlinked_root_rejected(self):
        link = self.tmp / "link"
        os.symlink(self.a, link)
        with self.assertRaises(L.ArtifactError) as cm:
            L.verify(link, expected_id=self.a.name)
        self.assertEqual(cm.exception.code, "file-type")

    # -- manifest schema
    def test_unknown_or_mistyped_manifest_fields_rejected(self):
        for fn in (lambda m: m.update(extra=1), lambda m: m.update(converter=5),
                   lambda m: m["model"].update(extra=1), lambda m: m["source"][0].update(bytes="1"),
                   lambda m: m.update(transformations=42)):
            self.relist(fn)
            self.expect("schema")

    def test_model_must_match_index(self):
        self.relist(lambda m: m["model"].update(representation=["exl3"]))
        self.expect("schema")
        self.relist(lambda m: m["model"].update(representation=["ggml"], expert_count=8))
        self.expect("meta")  # the kept GGUF metadata says 4 experts
        self.relist(lambda m: m["model"].update(expert_count=4, architecture="llama"))
        self.expect("meta")

    def test_second_index_role_rejected(self):
        self.relist(lambda m: m["files"][-1].update(role="index"))
        self.expect("path")

    def test_transformations_must_describe_the_index(self):
        self.relist(lambda m: m.update(transformations=[t for t in m["transformations"] if t["kind"] != "expert-slice"]))
        self.expect("schema")

    # -- file set and hashes
    def test_file_hash_mismatch(self):
        path = next((self.a / "meta").iterdir())
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        self.expect("hash")

    def test_chunk_hash_localizes_corruption(self):
        raw, data_offset, header = self.entries()
        name = next(k for k in header if not k.startswith(("~", "__")))
        a = data_offset + header[name]["data_offsets"][0]
        path = self.a / "data/00000.safetensors"
        data = bytearray(raw)
        data[a] ^= 1
        path.write_bytes(data)
        self.relist()
        self.expect("chunk-hash")

    def test_nonzero_pad_rejected(self):
        raw, data_offset, header = self.entries()
        pad = next(v for k, v in header.items() if k.startswith("~pad."))
        at = data_offset + pad["data_offsets"][0]
        data = bytearray(raw)
        data[at] = 7
        (self.a / "data/00000.safetensors").write_bytes(data)
        idx = json.loads((self.a / "index.json").read_bytes())
        # Recompute chunk hashes so only the pad rule can fire.
        g = next(g for g in idx["groups"] if g["shard"] == 0 and g["offset"] <= at - data_offset < g["offset"] + g["stored_bytes"])
        buf = bytes(data[data_offset + g["offset"]:data_offset + g["offset"] + g["stored_bytes"]])
        for k in range(-(-g["stored_bytes"] // L.CHUNK)):
            idx["chunk_sha256"][g["first_chunk"] + k] = hashlib.sha256(buf[k * L.CHUNK:(k + 1) * L.CHUNK]).hexdigest()
        self.index(lambda i: i.update(chunk_sha256=idx["chunk_sha256"]))
        self.expect("pad")

    def test_unlisted_file_extra_dir_and_symlink_rejected(self):
        (self.a / "meta" / "extra.txt").write_text("x")
        self.expect("file-set")
        os.remove(self.a / "meta" / "extra.txt")
        (self.a / "junk").mkdir()
        self.expect("file-set")
        os.rmdir(self.a / "junk")
        moved = self.tmp / "index.json"
        os.rename(self.a / "index.json", moved)
        os.symlink(moved, self.a / "index.json")
        self.expect("file-type")

    def test_malformed_index_fails_closed(self):
        original = json.loads((self.a / "index.json").read_bytes())
        for raw in (lambda d: b"not json", lambda d: b"[" * 200000, lambda d: L.dumps([d]),
                    lambda d: L.dumps(d).replace(b'"chunk_bytes":2097152', b'"chunk_bytes":1e400')):
            (self.a / "index.json").write_bytes(raw(original))
            self.relist()
            with self.assertRaises(L.ArtifactError):
                L.verify(self.a)
        (self.a / "index.json").write_bytes(L.dumps(dict(original, resources=None)))
        self.relist()
        self.expect("schema")

    # -- index structure
    def test_overlapping_resources_rejected(self):
        def f(i):
            r = next(r for r in i["resources"] if r["name"] == "blk.0.attn_q.weight")
            r["offset"] -= L.MEMBER_ALIGN
        self.index(f)
        self.expect("overlap")

    def test_misaligned_resource_rejected(self):
        self.index(lambda i: i["resources"][1].update(offset=i["resources"][1]["offset"] + 8))
        self.expect("alignment")

    def test_range_outside_group_rejected(self):
        def f(i):
            r = i["resources"][-1]
            r["offset"] = i["groups"][r["group"]]["stored_bytes"] - L.MEMBER_ALIGN
        self.index(f)
        self.expect("bounds")

    def test_representation_must_imply_the_bytes(self):
        for fn in (lambda i: i["resources"][0]["repr"].update(ne=[256, 100000]),
                   lambda i: i["expert_arrays"][0]["repr"].update(type="F32"),
                   lambda i: i["resources"][0]["repr"].update(type="Q9_Z"),
                   lambda i: i["resources"][0].update(readable_bytes=i["resources"][0]["bytes"] + 256)):
            self.index(fn)
            self.expect("repr")

    def test_roles_cannot_substitute_tensors(self):
        self.index(lambda i: i["resources"][0].update(roles=["blk.1.attn_q.weight"]))
        self.expect("schema")
        self.index(lambda i: i["resources"][0].update(roles=[i["resources"][0]["name"], "blk.1.attn_q.weight"]))
        self.expect("schema")
        self.index(lambda i: i["resources"][0].update(access="columns"))
        self.expect("schema")

    def test_group_fields_are_typed(self):
        self.index(lambda i: i["groups"][0].update(kind="banana"))
        self.expect("schema")
        self.index(lambda i: i["groups"][0].update(kind="banana"))

    def test_bool_layer_rejected(self):
        g = next(k for k, g in enumerate(json.loads((self.a / "index.json").read_bytes())["groups"]) if g["kind"] == "layer")
        self.index(lambda i: i["groups"][g].update(layer=False))
        self.expect("schema")

    def test_exl3_closure_enforced_by_verifier(self):
        def f(i):
            i["resources"][0]["repr"] = {"family": "exl3", "role": "suh", "dtype": "F32", "shape": [256]}
        self.index(f)
        self.expect("repr")

    def test_role_cannot_name_an_expert_array(self):
        def f(i):
            next(r for r in i["resources"] if r["name"] == "blk.0.attn_norm.weight")["roles"].append(
                "blk.0.ffn_gate_exps.weight")
        self.index(f)
        self.relist(lambda m: m["transformations"].insert(0, {
            "kind": "dedupe-identical", "resource": "blk.0.attn_norm.weight",
            "role": "blk.0.ffn_gate_exps.weight", "sha256": "0" * 64}))
        with self.assertRaises(L.ArtifactError) as cm:
            L.verify(self.a)
        self.assertIn(cm.exception.code, {"schema", "canonical"})

    def test_exl3_part_names_require_the_exl3_family(self):
        def f(i):
            r = next(r for r in i["resources"] if r["name"] == "blk.0.attn_norm.weight")
            r["name"] = r["roles"][0] = "model.layers.1.mlp.up_proj.trellis"
        self.index(f)
        self.expect("repr")

    def test_dedupe_digest_is_checked(self):
        self.relist(lambda m: next(t for t in m["transformations"] if t["kind"] == "dedupe-identical").update(
            sha256="f" * 64))
        self.expect("hash")

    def test_one_content_has_one_encoding(self):
        self.index(lambda i: i["resources"].reverse())
        self.expect("canonical")
        self.index(lambda i: i["resources"].reverse())
        self.relist(lambda m: m["files"].reverse())
        self.expect("canonical")
        self.relist(lambda m: m["files"].reverse())
        self.relist(lambda m: m["source"].append(dict(m["source"][0])))
        self.expect("canonical")

    def test_header_must_be_the_one_the_index_implies(self):
        self.header(lambda t: t.replace(b'"~pad.0"', b'"~pad.9"', 1))
        self.expect("container")

    def test_free_text_is_printable_ascii(self):
        for bad in ("\ud800\u0000", 'a"b', "a\\b"):
            self.relist(lambda m: m["converter"].update(name=bad))
            self.expect("schema")

    def test_shallow_verify_checks_sizes(self):
        path = self.a / "data/00000.safetensors"
        with open(path, "r+b") as f:
            f.truncate(4096)
        with self.assertRaises(L.ArtifactError) as cm:
            L.verify(self.a, deep=False)
        self.assertEqual(cm.exception.code, "file-size")

    def test_declared_huge_size_does_not_allocate(self):
        self.relist(lambda m: next(f for f in m["files"] if f["role"] == "source-metadata").update(bytes=1 << 50))
        self.expect("file-size")

    def test_kv_gguf_must_parse_and_files_must_be_singly_linked(self):
        kv = lambda: next((self.a / "meta").glob("*.kv.gguf"))
        good = kv().read_bytes()
        kv().write_bytes(b"garbage")
        self.relist()
        self.expect("meta")
        kv().write_bytes(good)
        self.relist()
        os.link(kv(), self.tmp / "other-name")
        self.expect("file-type")

    def patched(self, name, value):
        old = getattr(L, name)
        setattr(L, name, value)
        self.addCleanup(setattr, L, name, old)

    def test_entry_count_is_capped_before_expansion(self):
        self.patched("MAX_ENTRIES", 10)
        self.expect("bounds")

    def test_shard_header_limit_enforced(self):
        self.patched("MAX_ST_HEADER", 1000)
        self.expect("container")

    def test_exl3_rules_cover_roles_and_array_names(self):
        def via_role(i):
            r = next(r for r in i["resources"] if r["name"] == "blk.0.attn_norm.weight")
            r["roles"].append("model.layers.0.mlp.up_proj.trellis")
        self.index(via_role)
        self.relist(lambda m: m["transformations"].insert(0, {
            "kind": "dedupe-identical", "resource": "blk.0.attn_norm.weight",
            "role": "model.layers.0.mlp.up_proj.trellis", "sha256": "0" * 64}))
        with self.assertRaises(L.ArtifactError) as cm:
            L.verify(self.a)
        self.assertIn(cm.exception.code, {"repr", "canonical"})

    def test_exl3_rule_covers_array_names(self):
        self.index(lambda i: i["expert_arrays"][1].update(name="blk.0.ffn_down_exps.svh"))
        self.expect("repr")

    def test_reserved_role_rejected(self):
        self.index(lambda i: i["resources"][0]["roles"].append("__metadata__"))
        self.expect("schema")

    def test_alias_roles_are_canonically_ordered(self):
        def f(i):
            r = next(r for r in i["resources"] if r["name"] == "token_embd.weight")
            r["roles"] = ["token_embd.weight", "output.weight", "lm_head.weight"]
        self.index(f)
        self.expect("canonical")

    def test_every_dedupe_digest_is_checked(self):
        def f(i):
            r = next(r for r in i["resources"] if r["name"] == "token_embd.weight")
            r["roles"] = ["token_embd.weight", "lm_head.weight", "output.weight"]
        self.index(f)

        def m(man):
            good = next(t for t in man["transformations"] if t["kind"] == "dedupe-identical")
            man["transformations"].append(dict(good, role="lm_head.weight", sha256="0" * 64))
            man["transformations"].sort(key=L._canon_key)
        self.relist(m)
        self.expect("hash")

    def test_json_bombs_rejected_before_parsing(self):
        self.relist(raw=lambda m: L.dumps(m)[:-2] + b',"x":' + b"[" * 50 + b"]" * 50 + b"}\n")
        self.expect("json")
        self.patched("MAX_CONTAINERS", 100)
        self.index(lambda i: None)  # the real index has more containers than that
        self.expect("json")

    def test_prescan_bounds_values_and_balance(self):
        self.assertEqual(L.strict_json(b'["' + b"a" * (1 << 20) + b'"]'), ["a" * (1 << 20)])
        for doc in (b"]" * 1000, b"[]]", b"[" + b'"ab",' * (L.MAX_VALUES + 1) + b'"ab"]'):
            with self.assertRaises(L.ArtifactError):
                L.strict_json(doc)
        self.assertEqual(L.strict_json(b'{"a":"]\\\"[{"}'), {"a": ']"[{'})  # brackets inside strings

    def test_prescan_is_linear_on_unterminated_escapes(self):
        import time
        t = time.perf_counter()
        with self.assertRaises(L.ArtifactError):
            L.strict_json(b'"' + b'\\"' * 500_000)
        self.assertLess(time.perf_counter() - t, 2.0)

    def test_kept_metadata_size_limits_in_both_modes(self):
        self.patched("MAX_META", 64)
        for deep in (True, False):
            with self.assertRaises(L.ArtifactError) as cm:
                L.verify(self.a, deep=deep)
            self.assertEqual(cm.exception.code, "file-size")

    def test_manifest_size_limit(self):
        self.patched("MAX_MANIFEST", 64)
        self.expect("file-size")

    def test_gguf_key_rules_in_kept_metadata(self):
        kv = lambda: next((self.a / "meta").glob("*.kv.gguf"))
        s = lambda x: struct.pack("<Q", len(x)) + x
        for body in ([(b"a", 4, struct.pack("<I", 1)), (b"a", 4, struct.pack("<I", 2))],   # duplicate
                     [(b"", 4, struct.pack("<I", 1))],                                     # empty key
                     [(b"general.alignment", 10, struct.pack("<Q", 32))],                 # not u32
                     [(b"general.alignment", 4, struct.pack("<I", 48))]):                 # not 2^k
            data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(body)) + b"".join(
                s(k) + struct.pack("<I", t) + v for k, t, v in body)
            kv().write_bytes(data)
            self.relist()
            self.expect("meta")

    def test_empty_array_of_invalid_type_rejected(self):
        kv = lambda: next((self.a / "meta").glob("*.kv.gguf"))
        data = bytearray(kv().read_bytes())
        (n_kv,) = struct.unpack("<Q", data[16:24])
        data[16:24] = struct.pack("<Q", n_kv + 1)
        key = b"foo.expert_count"  # a kept key: the case that used to slip through
        data += struct.pack("<Q", len(key)) + key + struct.pack("<IIQ", 9, 99, 0)
        kv().write_bytes(bytes(data))
        self.relist()
        self.expect("meta")

    def test_alias_roles_are_capped(self):
        self.patched("MAX_ALIASES", 1)

        def f(i):
            r = next(r for r in i["resources"] if r["name"] == "token_embd.weight")
            r["roles"] = ["token_embd.weight", "lm_head.weight", "output.weight"]
        self.index(f)
        self.expect("bounds")

    def test_every_kept_expert_count_must_match(self):
        s = lambda x: struct.pack("<Q", len(x)) + x
        rogue = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s(b"toy.expert_count") + struct.pack("<II", 4, 99)
        (self.a / "meta" / "zz.kv.gguf").write_bytes(rogue)

        man = json.loads((self.a / "manifest.json").read_bytes())
        man["files"].append({"path": "meta/zz.kv.gguf", "role": "source-metadata", "bytes": 0, "sha256": "0" * 64})
        man["files"].sort(key=lambda f: f["path"])
        man["source"].append({"name": "zz.gguf", "bytes": 1, "sha256": "0" * 64})
        man["source"].sort(key=lambda x: x["name"])
        (self.a / "manifest.json").write_bytes(L.dumps(man))
        self.relist()  # re-hashes every listed file, the new one included
        self.expect("meta")

    def test_kept_metadata_must_be_covered_by_source_identity(self):
        (self.a / "meta" / "tokenizer.json").write_text("{}")
        man = json.loads((self.a / "manifest.json").read_bytes())
        man["files"].append({"path": "meta/tokenizer.json", "role": "source-metadata", "bytes": 0, "sha256": "0" * 64})
        man["files"].sort(key=lambda f: f["path"])
        (self.a / "manifest.json").write_bytes(L.dumps(man))
        self.relist()
        self.expect("file-set")  # no source entry named tokenizer.json with these bytes
        os.remove(self.a / "meta" / "tokenizer.json")
        man = json.loads((self.a / "manifest.json").read_bytes())
        man["files"] = [f for f in man["files"] if f["path"] != "meta/tokenizer.json"]
        man["source"][0]["name"] = "other.gguf"
        (self.a / "manifest.json").write_bytes(L.dumps(man))
        self.relist()
        self.expect("file-set")  # a kept .kv.gguf needs its GGUF source

    def test_non_ascii_documents_rejected(self):
        self.relist(raw=lambda m: L.dumps(m).replace(b'"m0-prototype"', "\"m0-prototyp\u00e9\"".encode()))
        self.expect("json")

    def test_kept_architecture_is_typed_and_bounded(self):
        kv = lambda: next((self.a / "meta").glob("*.kv.gguf"))
        s = lambda x: struct.pack("<Q", len(x)) + x
        for value in (struct.pack("<I", 8) + s(b"t" * (L.MAX_ARCH + 1)),   # too long
                      struct.pack("<II", 4, 7)):                             # not a string
            kv().write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s(b"general.architecture") + value)
            self.relist()
            self.expect("meta")

    def test_nested_or_deep_gguf_arrays_rejected(self):
        kv = lambda: next((self.a / "meta").glob("*.kv.gguf"))
        head = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1) + b"k"
        nested = head + struct.pack("<I", 9) + struct.pack("<IQ", 9, 1) * 5000
        kv().write_bytes(nested)
        self.relist()
        self.expect("meta")

    def test_expert_structure_rules(self):
        self.index(lambda i: i["groups"][0].update(kind="expert", expert=7, layer=0))
        self.expect("expert-array")

    def test_duplicate_array_name_rejected(self):
        self.index(lambda i: i["expert_arrays"][0].update(name=i["expert_arrays"][1]["name"]))
        self.expect("schema")

    def test_nonuniform_expert_array_rejected(self):
        def f(i):
            a = i["expert_arrays"][0]
            i["groups"][a["first_group"] + 1]["expert"] = 3
        self.index(f)
        self.expect("expert-array")

    # -- container header
    def test_container_must_match_index_types(self):
        self.header(lambda t: t.replace(b'"dtype":"F32"', b'"dtype":"I32"', 1))
        self.expect("container")

    def test_bad_pad_entry_rejected(self):
        self.header(lambda t: t.replace(b'"~pad.0":{"dtype":"U8"', b'"~pad.0":{"dtype":"I8"', 1))
        self.expect("container")

    def test_non_canonical_header_rejected(self):
        self.header(lambda t: t.replace(b'"__metadata__":{', b'"__metadata__": {', 1))
        self.expect("container")

    def test_negative_or_float_shapes_rejected(self):
        self.header(lambda t: t.replace(b'"shape":[256]', b'"shape":[-256]', 1))
        self.expect("container")


if __name__ == "__main__":
    unittest.main()
