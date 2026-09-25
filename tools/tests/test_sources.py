# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the source lock and its tools (no downloads, builds or SDK needed).

tests/sources/ checks the same mechanism end to end with the SDK's CMake.
"""

import contextlib
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import unittest
import zipfile
from unittest import mock

sys.dont_write_bytecode = True
TOOLS = pathlib.Path(__file__).resolve().parent.parent
REPO = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import jitllm_sources as srclib  # noqa: E402
from jitllm_sources import SourceError  # noqa: E402


def load_script(name: str):
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), str(TOOLS / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


prepare_sources = load_script("prepare-sources")
setup = load_script("setup")


def component(**changes) -> dict:
    comp = {
        "version": "1.0", "kind": "archive", "category": "implementation", "tier": "core", "use": "product",
        "machine": "target",
        "upstream": {"repository": "https://example.invalid/x", "commit": "a" * 40},
        "archive": {"file": "x-1.0.tar.gz", "urls": ["https://example.invalid/x-1.0.tar.gz"],
                    "sha256": "b" * 64, "size": 10},
        "patches": [], "tree_sha256": "c" * 64, "depends": [],
        "cmake": {"subdirectory": "", "options": {"X_TESTS": "OFF"}, "platform_packages": ["Threads"],
                  "targets": ["x::x"]},
        "license": {"expression": "MIT", "files": ["LICENSE"], "scope": "all", "evidence": "headers"},
        "verification": "checked",
    }
    comp.update(changes)
    return comp


def lock(components: dict, modules: dict | None = None) -> dict:
    return {"schema": 1, "modules": modules or {}, "components": components}


class CheckedInLock(unittest.TestCase):
    def test_validates_and_uses_https_only(self):
        data = srclib.load_lock()  # raises SourceError listing any problem
        for cid, comp in data["components"].items():
            for url in comp["archive"]["urls"]:
                self.assertTrue(url.startswith("https://"), f"{cid}: {url}")

    def test_core_selection(self):
        self.assertEqual(srclib.select(srclib.load_lock(), []), ["googletest", "tomlplusplus"])

    def test_mise_tasks(self):
        tasks = tomllib.loads((REPO / "mise.toml").read_text())["tasks"]
        self.assertEqual(tasks["setup"]["run"], "python3 tools/setup")
        self.assertEqual(tasks["prepare"]["run"], "python3 tools/prepare-sources")


class Validation(unittest.TestCase):
    def problems(self, data: dict, base: pathlib.Path = REPO) -> str:
        return "\n".join(srclib.validate_lock(data, base))

    def test_a_valid_lock_has_no_problems(self):
        self.assertEqual(self.problems(lock({"x": component()})), "")

    def test_unclassified_and_disallowed_licenses(self):
        comp = component()
        del comp["license"]
        self.assertIn("license must be an object", self.problems(lock({"x": comp})))
        comp = component(license={"files": ["LICENSE"], "scope": "s", "evidence": "e"})
        self.assertIn("unclassified components are rejected", self.problems(lock({"x": comp})))
        for expression in ("GPL-3.0-only", "MIT AND GPL-2.0-only", "BSL-1.0"):
            comp = component(license=dict(component()["license"], expression=expression))
            self.assertIn("outside D-017's core allowlist", self.problems(lock({"x": comp})))
        comp = component(license=dict(component()["license"], expression="MIT OR GPL-2.0-only"))
        self.assertIn("joined by ' AND '", self.problems(lock({"x": comp})))

    def test_optional_components_need_a_declared_module(self):
        comp = component(tier="optional", license=dict(component()["license"], expression="AGPL-3.0-only"))
        self.assertIn("names a declared module", self.problems(lock({"x": comp})))
        comp["module"] = "m"
        self.assertEqual(self.problems(lock({"x": comp}, {"m": {"description": "d"}})), "")
        self.assertIn("no component belongs to it",
                      self.problems(lock({"x": component()}, {"m": {"description": "d"}})))
        self.assertIn("only optional components belong to a module",
                      self.problems(lock({"x": component(module="m")}, {"m": {"description": "d"}})))

    def test_dependencies(self):
        modules = {"m": {"description": "d"}, "n": {"description": "d"}}
        self.assertIn("cannot depend on the optional", self.problems(lock(
            {"x": component(depends=["y"]), "y": component(tier="optional", module="m")}, {"m": modules["m"]})))
        self.assertIn("not in the lock", self.problems(lock({"x": component(depends=["nope"])})))
        self.assertIn("from another module", self.problems(lock(
            {"y": component(tier="optional", module="m"),
             "z": component(tier="optional", module="n", depends=["y"])}, modules)))
        self.assertIn("dependency cycle", self.problems(lock(
            {"a": component(depends=["b"]), "b": component(depends=["a"])})))

    def test_unsupported_kinds_and_identities(self):
        for change, text in (
                ({"kind": "vendored"}, "kind must be one of archive"),
                ({"machine": "build"}, "machine must be one of target"),
                ({"category": "build-tool"}, "category must be one of implementation"),
                ({"tree_sha256": "abc"}, "tree_sha256"),
                ({"upstream": {"repository": "r", "commit": "main"}}, "full commit"),
                ({"archive": dict(component()["archive"], urls=["http://example.invalid/x.tgz"])},
                 "is not an https://"),
                ({"archive": dict(component()["archive"], file="../x.tgz")}, "plain file name"),
                ({"cmake": dict(component()["cmake"], options={"X": "a b"})}, "plain string values"),
                ({"cmake": dict(component()["cmake"], subdirectory="../up")}, "relative path"),
                ({"cmake": dict(component()["cmake"], targets=[])}, "must name what jitLLM links"),
                ({"verification": ""}, "how the pin was checked")):
            with self.subTest(change=change):
                self.assertIn(text, self.problems(lock({"x": component(**change)})))
        self.assertIn("ids are lowercase", self.problems(lock({"X": component()})))
        self.assertIn("schema must be 1", self.problems({"schema": 2, "components": {}}))

    def test_malformed_scalar_values_are_diagnostics(self):
        for change, text in (
                ({"tier": "optional", "module": []}, "names a declared module"),
                ({"tier": "optional", "module": {}}, "names a declared module"),
                ({"archive": dict(component()["archive"], size=True)}, "positive byte count"),
                ({"archive": dict(component()["archive"], urls=["https://[broken"])}, "is not an https://"),
                ({"archive": dict(component()["archive"], urls=["https:missing-host"])}, "is not an https://"),
                ({"tree_sha256": "c" * 64 + "\n"}, "tree_sha256"),
                ({"cmake": dict(component()["cmake"], options={"X\n": "OFF"})}, "plain string values"),
                ({"cmake": dict(component()["cmake"], subdirectory="unsafe;path")}, "relative path"),
                ({"license": dict(component()["license"], files=["../LICENSE"])}, "license.files"),
                ({"license": dict(component()["license"], notices=["/NOTICE"])}, "license.notices")):
            with self.subTest(change=change):
                self.assertIn(text, self.problems(lock({"x": component(**change)})))
        self.assertIn("schema must be 1", self.problems(dict(lock({"x": component()}), schema=True)))
        self.assertIn("ids are lowercase", self.problems(lock({"x\n": component()})))

    def test_options_cannot_rebind_cmake_or_jitllm_variables_or_name_paths(self):
        for options, text in (
                ({"CMAKE_PROJECT_INCLUDE": "x"}, "may not set CMAKE_*"),
                ({"cmake_sysroot": "x"}, "may not set CMAKE_*"),
                ({"JITLLM_PYTHON": "x"}, "may not set"),
                ({"FETCHCONTENT_FULLY_DISCONNECTED": "OFF"}, "may not set"),
                ({"BUILD_SHARED_LIBS": "ON"}, "may not set"),
                ({"_JITLLM_SOURCES_SCRIPTS": "x"}, "cmake.options must map"),
                ({"X_TESTS": "/tmp/evil.cmake"}, "plain string values"),
                ({"X_TESTS": "a;b"}, "plain string values")):
            with self.subTest(options=options):
                comp = component(cmake=dict(component()["cmake"], options=options))
                self.assertIn(text, self.problems(lock({"x": comp})))
        comp = component(cmake=dict(component()["cmake"], options={"gtest_build_tests": "OFF", "X_LEVEL": "3.1"}))
        self.assertEqual(self.problems(lock({"x": comp})), "")

    def test_duplicate_json_keys_are_rejected(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = root / "sources.lock.json"
        path.write_text(json.dumps(lock({"x": component()})).replace('"tier": "core"',
                                                                   '"tier": "optional", "tier": "core"'))
        with self.assertRaisesRegex(SourceError, "duplicate JSON key 'tier'"):
            srclib.load_lock(path)

    def test_patches_must_exist_and_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            (base / "p.patch").write_text("patch")
            good = hashlib.sha256(b"patch").hexdigest()
            self.assertEqual(self.problems(lock({"x": component(patches=[{"path": "p.patch", "sha256": good}])}),
                                           base), "")
            self.assertIn("does not match its recorded sha256", self.problems(
                lock({"x": component(patches=[{"path": "p.patch", "sha256": "0" * 64}])}), base))
            self.assertIn("is missing", self.problems(
                lock({"x": component(patches=[{"path": "q.patch", "sha256": good}])}), base))
            self.assertIn("must stay under", self.problems(
                lock({"x": component(patches=[{"path": "../p.patch", "sha256": good}])}), base))

    def test_patch_symlinks_cannot_escape_the_lock_directory(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = root / "lock"
        base.mkdir()
        outside = root / "outside.patch"
        outside.write_bytes(b"patch")
        (base / "escape.patch").symlink_to(outside)
        self.assertIn("must stay under", self.problems(lock({"x": component(patches=[{
            "path": "escape.patch", "sha256": hashlib.sha256(b"patch").hexdigest()}])}), base))

    def test_excluded_module_patches_are_not_read(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = root / "sources.lock.json"
        data = lock({"x": component(), "o": component(tier="optional", module="m", patches=[{
            "path": "missing.patch", "sha256": "0" * 64}])}, {"m": {"description": "optional module"}})
        path.write_text(json.dumps(data))
        with mock.patch.object(srclib, "sha256_file", side_effect=AssertionError("read an excluded patch")):
            self.assertEqual(srclib.load_lock(path, modules=[]), data)
        for modules in (None, ["m"]):
            with self.subTest(modules=modules), self.assertRaisesRegex(SourceError, "patch missing.patch is missing"):
                srclib.load_lock(path, modules=modules)


class Selection(unittest.TestCase):
    MODULES = {"m": {"description": "d"}}

    def test_core_then_modules_in_dependency_order(self):
        data = lock({"b": component(depends=["a"]), "a": component(),
                     "o": component(tier="optional", module="m", depends=["b"])}, self.MODULES)
        self.assertEqual(srclib.select(data, []), ["a", "b"])
        self.assertEqual(srclib.select(data, ["m"]), ["a", "b", "o"])
        self.assertEqual(srclib.license_profile(["m"]), "core+m")
        self.assertEqual(srclib.license_profile([]), "core")

    def test_unknown_modules_and_unselected_dependencies_are_refused(self):
        data = lock({"a": component()}, {})
        with self.assertRaisesRegex(SourceError, "unknown module"):
            srclib.select(data, ["m"])
        data = lock({"a": component(depends=["o"]), "o": component(tier="optional", module="m")}, self.MODULES)
        with self.assertRaisesRegex(SourceError, "does not select"):
            srclib.select(data, [])

    def test_prepared_dir_names_the_tree(self):
        self.assertEqual(srclib.prepared_dir(pathlib.Path("/s"), "x", component()), pathlib.Path("/s/x-" + "c" * 16))


class TreeDigest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "b").mkdir()
        (self.root / "b" / "y.txt").write_text("y\n")
        (self.root / "a.txt").write_text("a\n")
        (self.root / "empty").mkdir()

    def test_matches_the_documented_format(self):
        lines = "".join(f"{hashlib.sha256(data).hexdigest()} - {name}\n"
                        for name, data in (("a.txt", b"a\n"), ("b/y.txt", b"y\n")))
        self.assertEqual(srclib.tree_digest(self.root), hashlib.sha256(lines.encode()).hexdigest())

    def test_content_names_and_executable_bits_count(self):
        base = srclib.tree_digest(self.root)
        (self.root / "a.txt").chmod(0o755)
        executable = srclib.tree_digest(self.root)
        self.assertNotEqual(base, executable)
        (self.root / "a.txt").chmod(0o644)
        self.assertEqual(srclib.tree_digest(self.root), base)
        (self.root / "b" / "y.txt").write_text("z\n")
        self.assertNotEqual(srclib.tree_digest(self.root), base)

    def test_links_and_unsupported_names_are_refused(self):
        (self.root / "link").symlink_to("a.txt")
        with self.assertRaisesRegex(SourceError, "symbolic links"):
            srclib.tree_digest(self.root)
        (self.root / "link").unlink()
        for name in ("semi;colon", "[bracket]", "new\nline", "trailing\n"):
            path = self.root / name
            path.write_text("x")
            with self.subTest(name=name), self.assertRaisesRegex(SourceError, "unsupported file name"):
                srclib.tree_digest(self.root)
            path.unlink()

    def test_an_empty_tree_is_refused(self):
        empty = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (empty / "only-a-directory").mkdir()
        with self.assertRaisesRegex(SourceError, "has no files"):
            srclib.tree_digest(empty)

    def test_missing_and_symlink_roots_are_refused(self):
        for path in (self.root / "absent", self.root / "a.txt"):
            with self.subTest(path=path), self.assertRaisesRegex(SourceError, "not a prepared directory"):
                srclib.tree_digest(path)
        link = self.root / "link"
        link.symlink_to(self.root / "b", target_is_directory=True)
        with self.assertRaisesRegex(SourceError, "not a prepared directory"):
            srclib.tree_digest(link)

    def test_unreadable_subdirectories_do_not_disappear_from_the_digest(self):
        path = self.root / "b"
        path.chmod(0)
        try:
            if os.geteuid() == 0:
                self.skipTest("root bypasses the unreadable-directory fixture")
            with self.assertRaisesRegex(SourceError, "cannot scan prepared tree"):
                srclib.tree_digest(self.root)
        finally:
            path.chmod(0o755)


class Patches(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "src").mkdir()
        (self.root / "src" / "f.h").write_text("one\ntwo\nthree\n")

    def test_modifies_creates_and_deletes(self):
        (self.root / "gone.txt").write_text("bye\n")
        srclib.apply_patch(self.root, (
            "diff --git a/src/f.h b/src/f.h\n--- a/src/f.h\n+++ b/src/f.h\n"
            "@@ -1,3 +1,4 @@\n one\n-two\n+TWO\n+2.5\n three\n"
            "--- /dev/null\n+++ b/new/n.txt\n@@ -0,0 +1,2 @@\n+a\n+b\n\\ No newline at end of file\n"
            "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-bye\n"))
        self.assertEqual((self.root / "src" / "f.h").read_text(), "one\nTWO\n2.5\nthree\n")
        self.assertEqual((self.root / "new" / "n.txt").read_text(), "a\nb")
        self.assertFalse((self.root / "gone.txt").exists())

    def test_keeps_crlf_and_mode(self):
        path = self.root / "src" / "w.txt"
        path.write_bytes(b"a\r\nb\r\n")
        path.chmod(0o755)
        srclib.apply_patch(self.root, "--- a/src/w.txt\n+++ b/src/w.txt\n@@ -1,2 +1,2 @@\n a\r\n-b\r\n+c\r\n")
        self.assertEqual(path.read_bytes(), b"a\r\nc\r\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o755)

    def test_refuses_mismatches_and_unsafe_patches(self):
        for text, message in (
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1,2 +1,2 @@\n one\n-TWO\n+2\n", "context does not match"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -9,1 +9,1 @@\n-x\n+y\n", "past the end"),
                ("--- a/../x\n+++ b/../x\n@@ -1 +1 @@\n-a\n+b\n", "leaves the source tree"),
                ("--- src/f.h\n+++ src/f.h\n@@ -1 +1 @@\n-one\n+1\n", "must start with a/ or b/"),
                ("--- a/src/f.h\n+++ b/src/g.h\n@@ -1 +1 @@\n-one\n+1\n", "renames are not supported"),
                ("diff --git a/src/f.h b/src/f.h\nold mode 100644\nnew mode 100755\n", "not supported"),
                ("--- /dev/null\n+++ b/src/f.h\n@@ -0,0 +1 @@\n+x\n", "which exists"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1,3 +1,3 @@\n one\n-two\n", "ends inside a hunk"),
                ("just text\n", "changes no files")):
            with self.subTest(message=message), self.assertRaisesRegex(SourceError, message):
                srclib.apply_patch(self.root, text)

    def test_refuses_to_patch_through_a_link(self):
        outside = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        (outside / "f.h").write_text("one\n")
        (self.root / "linked").symlink_to(outside)
        with self.assertRaisesRegex(SourceError, "symbolic link"):
            srclib.apply_patch(self.root, "--- a/linked/f.h\n+++ b/linked/f.h\n@@ -1 +1 @@\n-one\n+1\n")
        self.assertEqual((outside / "f.h").read_text(), "one\n")

    def test_hunk_positions_and_line_counts_are_exact(self):
        for text, message in (
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +99 @@\n-one\n+1\n", "new hunk position"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n+unrecorded\n", "unexpected line after"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1", "truncated hunk line"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n\\ anything\n", "unknown newline marker"),
                ("--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n\\ No newline at end of file\n",
                 "before the end"),
                ("diff --git a/new.sh b/new.sh\nnew file mode 100755\n--- /dev/null\n+++ b/new.sh\n"
                 "@@ -0,0 +1 @@\n+x\n", "not supported")):
            with self.subTest(message=message), self.assertRaisesRegex(SourceError, message):
                srclib.apply_patch(self.root, text)
            self.assertEqual((self.root / "src" / "f.h").read_text(), "one\ntwo\nthree\n")

    def test_lines_between_files_must_be_git_headers(self):
        base = "diff --git a/src/f.h b/src/f.h\nindex 1..2 100644\n--- a/src/f.h\n+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n"
        with self.assertRaisesRegex(SourceError, "unexpected 'stray text' between files"):
            srclib.apply_patch(self.root, base + "stray text\n")
        self.assertEqual((self.root / "src" / "f.h").read_text(), "1\ntwo\nthree\n")  # staged callers discard

    def test_preamble_and_signature_are_skipped(self):
        srclib.apply_patch(self.root, "From 0000 Mon Sep 17 00:00:00 2001\nSubject: change\n\nrename nothing\n---\n"
                                      " src/f.h | 2 +-\n\ndiff --git a/src/f.h b/src/f.h\n--- a/src/f.h\n"
                                      "+++ b/src/f.h\n@@ -1 +1 @@\n-one\n+1\n-- \n2.43.0\n")
        self.assertEqual((self.root / "src" / "f.h").read_text(), "1\ntwo\nthree\n")

    def test_old_file_final_newline_is_exact(self):
        path = self.root / "src" / "last.txt"
        header = "--- a/src/last.txt\n+++ b/src/last.txt\n@@ -1 +1 @@\n"
        for original, removed in (("a\n", "-a\n\\ No newline at end of file\n"), ("a", "-a\n")):
            path.write_text(original)
            with self.subTest(original=original), self.assertRaisesRegex(SourceError, "context does not match"):
                srclib.apply_patch(self.root, header + removed + "+b\n")
            self.assertEqual(path.read_text(), original)
        path.write_text("a")
        srclib.apply_patch(self.root, header + "-a\n\\ No newline at end of file\n+b\n")
        self.assertEqual(path.read_bytes(), b"b\n")


class Archives(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))

    def tar(self, *members: tarfile.TarInfo, name="a.tar.gz") -> pathlib.Path:
        path = self.tmp / name
        with tarfile.open(path, "w:gz") as tar:
            data = b"x\n"
            info = tarfile.TarInfo("x-1.0/file.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            for member in members:
                tar.addfile(member)
        return path

    def member(self, name: str, kind: bytes, link: str = "") -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.type, info.linkname = kind, link
        return info

    def test_plain_files_and_directories_pass(self):
        srclib.check_archive(self.tar(self.member("x-1.0/dir", tarfile.DIRTYPE)))

    def test_links_specials_and_unsafe_paths_are_refused(self):
        for member, text in (
                (self.member("x-1.0/escape", tarfile.SYMTYPE, "../../outside"), "not a file or directory"),
                (self.member("x-1.0/hard", tarfile.LNKTYPE, "x-1.0/file.txt"), "not a file or directory"),
                (self.member("x-1.0/pipe", tarfile.FIFOTYPE), "not a file or directory"),
                (self.member("../x-1.0/up.txt", tarfile.REGTYPE), "unsafe or unsupported path"),
                (self.member("/abs.txt", tarfile.REGTYPE), "unsafe or unsupported path"),
                (self.member("x-1.0/semi;colon", tarfile.REGTYPE), "unsafe or unsupported path")):
            with self.subTest(name=member.name), self.assertRaisesRegex(SourceError, text):
                srclib.check_archive(self.tar(member))

    def test_zip_links_are_refused(self):
        path = self.tmp / "a.zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("x-1.0/file.txt", "x\n")
            link = zipfile.ZipInfo("x-1.0/link")
            link.external_attr = (0o120777 << 16)
            z.writestr(link, "../../outside")
        with self.assertRaisesRegex(SourceError, "not a file or directory"):
            srclib.check_archive(path)

    def test_unreadable_archives_are_refused(self):
        path = self.tmp / "junk.tar.gz"
        path.write_bytes(b"not an archive")
        with self.assertRaisesRegex(SourceError, "cannot read the archive"):
            srclib.check_archive(path)


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.origin = self.tmp / "x-1.0.tar.gz"
        self.origin.write_bytes(b"archive bytes")
        self.cache = self.tmp / "cache"
        self.archive = {"file": "x-1.0.tar.gz", "urls": [self.origin.as_uri()],
                        "sha256": hashlib.sha256(b"archive bytes").hexdigest(), "size": len(b"archive bytes")}

    def fetch(self, archive=None):
        return srclib.fetch(self.cache, "x", archive or self.archive, log=lambda _: None)

    def test_fetches_once_into_the_shared_download_cache(self):
        path = self.fetch()
        self.assertEqual(path, self.cache / "downloads" / self.archive["sha256"] / "x-1.0.tar.gz")
        self.assertEqual(path.read_bytes(), b"archive bytes")
        with mock.patch.object(srclib.urllib.request, "urlopen", side_effect=AssertionError("fetched again")):
            self.assertEqual(self.fetch(), path)

    def test_a_damaged_cache_entry_is_replaced(self):
        path = self.fetch()
        path.write_bytes(b"archive bytez")
        self.assertEqual(self.fetch().read_bytes(), b"archive bytes")

    def test_wrong_or_oversized_bytes_are_refused(self):
        self.origin.write_bytes(b"archive bytez")
        with self.assertRaisesRegex(SourceError, "not the locked"):
            self.fetch()
        self.origin.write_bytes(b"archive bytes and more")
        with self.assertRaisesRegex(SourceError, "more than the locked"):
            self.fetch()
        self.assertEqual([p.name for p in (self.cache / "downloads" / self.archive["sha256"]).iterdir()], [])


class PrepareSources(unittest.TestCase):
    def test_a_modified_prepared_tree_is_an_error(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        tree = root / "x"
        tree.mkdir()
        (tree / "f").write_text("f\n")
        comp = component(tree_sha256=srclib.tree_digest(tree))
        self.assertTrue(prepare_sources.check_prepared(tree, "x", comp))
        self.assertFalse(prepare_sources.check_prepared(root / "absent", "x", comp))
        (tree / "f").write_text("g\n")
        with self.assertRaisesRegex(SourceError, "was modified"):
            prepare_sources.check_prepared(tree, "x", comp)

    def test_termination_during_spawn_still_reaps_the_extractor(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        processes = []
        popen = subprocess.Popen

        def interrupted_spawn(*args, **kwargs):
            proc = popen(*args, **kwargs)
            processes.append(proc)
            prepare_sources._stop(signal.SIGTERM, None)
            return proc

        try:
            with mock.patch.object(prepare_sources.subprocess, "Popen", side_effect=interrupted_spawn), \
                    mock.patch.object(prepare_sources.signal, "signal"), self.assertRaises(SystemExit) as stopped:
                prepare_sources.run_extractor([sys.executable, "-c", "import time; time.sleep(60)"],
                                              cwd=root, env={})
            self.assertEqual(stopped.exception.code, 128 + signal.SIGTERM)
            self.assertEqual(len(processes), 1)
            self.assertEqual(processes[0].returncode, -signal.SIGTERM)
            self.assertFalse(prepare_sources._spawning_extractor)
            self.assertIsNone(prepare_sources._pending_stop)
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate(timeout=10)

    def test_termination_retires_extractor_children_before_removing_staging(self):
        root = pathlib.Path(self.enterContext(tempfile.TemporaryDirectory()))
        started, stopped = root / "worker.pid", root / "worker.stopped"
        extractor_pid = root / "extractor.pid"
        worker = root / "worker.py"
        worker.write_text(
            "import os, pathlib, signal, sys, time\n"
            "def stop(signum, frame):\n"
            f"    pathlib.Path({str(stopped)!r}).write_text('stopped')\n"
            "    sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            f"pathlib.Path({str(started)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(60)\n")
        extractor = root / "cmake"
        extractor.write_text(
            f"#!{sys.executable}\nimport os, pathlib, subprocess, sys\n"
            f"pathlib.Path({str(extractor_pid)!r}).write_text(str(os.getpid()))\n"
            f"subprocess.run([sys.executable, {str(worker)!r}], check=True)\n")
        extractor.chmod(0o755)
        archive = root / "archive.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            data = b"x\n"
            info = tarfile.TarInfo("x-1.0/file.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        path = root / "sources.lock.json"
        path.write_text(json.dumps(lock({"x": component(archive={
            "file": archive.name, "urls": [archive.as_uri()], "size": archive.stat().st_size,
            "sha256": srclib.sha256_file(archive)})})))
        dest = root / "sources"
        proc = subprocess.Popen([sys.executable, str(TOOLS / "prepare-sources"), "--lock", str(path),
                                 "--dest", str(dest), "--cache", str(root / "cache"), "--cmake", str(extractor)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        retired = False
        try:
            deadline = time.monotonic() + 10
            while not started.exists() and proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(started.exists(), "extractor worker did not start")
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 128 + signal.SIGTERM, stdout + stderr)
            self.assertTrue(stopped.exists(), "extractor child survived termination")
            self.assertEqual(list(dest.glob(".staging-*")), [])
            retired = True
        finally:
            if proc.poll() is None:
                proc.kill()
            if not retired:
                for pid_file in (started, extractor_pid):
                    if pid_file.exists():
                        with contextlib.suppress(ProcessLookupError):
                            os.kill(int(pid_file.read_text()), signal.SIGKILL)
            proc.communicate(timeout=10)


class Setup(unittest.TestCase):
    def steps(self, *args: str) -> list[list[str]]:
        calls = []
        with mock.patch.object(setup, "run", side_effect=lambda cmd: calls.append(cmd[1:]) or 0), \
                mock.patch.object(sys, "argv", ["setup", *args]):
            self.assertEqual(setup.main(), 0)
        return [[pathlib.Path(c[0]).name, *c[1:]] for c in calls]

    def test_prepares_sources_after_the_sdk(self):
        self.assertEqual(self.steps(), [["setup-toolchain"], ["prepare-sources"]])
        self.assertEqual(self.steps("--jobs", "4"), [["setup-toolchain", "--jobs", "4"], ["prepare-sources"]])
        self.assertEqual(self.steps("--dry-run"), [["setup-toolchain", "--dry-run"], ["prepare-sources", "--dry-run"]])

    def test_sdk_only_actions_and_failures_stop_there(self):
        for flag in ("--print-root", "--list", "--prune"):
            self.assertEqual(self.steps(flag), [["setup-toolchain", flag]])
        with mock.patch.object(setup, "run", return_value=1) as run, mock.patch.object(sys, "argv", ["setup"]):
            self.assertEqual(setup.main(), 1)
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
