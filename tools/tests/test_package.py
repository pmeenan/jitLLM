# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""The package's documents and inventory (tools/jitllm_package.py, D-074)."""

import io
import os
import pathlib
import sys
import tarfile
import tempfile
import tomllib
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TOOLS))
import jitllm_package as package  # noqa: E402
import jitllm_sdk as sdklib  # noqa: E402
import jitllm_sources as srclib  # noqa: E402


class Extract(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.dir.name)
        (self.root / "header.h").write_text(
            "/*\n * Copyright (c) 1994\n * Someone\n *\n * Permission granted.\n * As is.\n */\nint x;\n")

    def tearDown(self):
        self.dir.cleanup()

    def test_from_to_plus_and_comment_prefix(self):
        spec = {"file": "sdk:header.h", "from": "Copyright", "to": "Permission", "plus": 1}
        self.assertEqual(package.extract(spec, self.root), "Copyright (c) 1994\nSomeone\n\nPermission granted.\nAs is.\n")

    def test_whole_file(self):
        self.assertTrue(package.extract({"file": "sdk:header.h"}, self.root).startswith("/*\n"))

    def test_missing_markers_fail(self):
        with self.assertRaises(package.PackageError):
            package.extract({"file": "sdk:header.h", "from": "nowhere"}, self.root)
        with self.assertRaises(package.PackageError):
            package.extract({"file": "sdk:header.h", "from": "Copyright", "to": "nowhere"}, self.root)
        with self.assertRaises(package.PackageError):
            package.extract({"file": "sdk:header.h", "from": "As is", "plus": 5}, self.root)
        with self.assertRaises(package.PackageError):
            package.extract({"file": "elsewhere:header.h"}, self.root)

    def test_component_notice_lines(self):
        (self.root / "u.hpp").write_text("a\n\t\t// Copyright (c) 2008 Someone <x@y>\nb\n")
        self.assertEqual(package.component_notice(self.root, "u.hpp:2-2"), "Copyright (c) 2008 Someone <x@y>\n")
        self.assertEqual(srclib.notice_range("LICENSE"), ("LICENSE", None))
        self.assertEqual(srclib.notice_range("a/b.h:3-9"), ("a/b.h", (3, 9)))
        self.assertEqual(srclib.notice_range("a/b.h:9-3"), ("a/b.h:9-3", None))


class ReadDeb(unittest.TestCase):
    def deb(self, members: dict[str, bytes]) -> pathlib.Path:
        out = io.BytesIO()
        out.write(b"!<arch>\n")
        for name, data in members.items():
            out.write(f"{name:<16}{0:<12}{0:<6}{0:<6}{100644:<8}{len(data):<10}`\n".encode())
            out.write(data + (b"\n" if len(data) % 2 else b""))
        fd, name = tempfile.mkstemp(suffix=".deb")
        os.close(fd)
        path = pathlib.Path(name)
        path.write_bytes(out.getvalue())
        self.addCleanup(path.unlink)
        return path

    @staticmethod
    def tar(files: dict[str, bytes]) -> bytes:
        out = io.BytesIO()
        with tarfile.open(fileobj=out, mode="w:gz") as tar:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return out.getvalue()

    def test_members(self):
        path = self.deb({"debian-binary": b"2.0\n", "control.tar.gz": self.tar({"./control": b"Package: x\n"}),
                         "data.tar.gz": self.tar({"./usr/bin/x": b"binary"})})
        parts = package.read_deb(path)
        self.assertEqual(parts["control"]["control"][1], b"Package: x\n")
        self.assertEqual(parts["data"]["usr/bin/x"][1], b"binary")

    def test_not_a_package(self):
        with self.assertRaises(package.PackageError):
            package.read_deb(self.deb({"debian-binary": b"3.0\n"}))


class InTreeUnits(unittest.TestCase):
    """jitLLM's files with third-party data are listed exactly when an executable is built from them (D-088)."""

    def test_listed_only_when_built_from(self):
        tables = (package.REPO / "src/tokenizer/unicode_data.cc").resolve()
        other = (package.REPO / "src/tokenizer/unicode.cc").resolve()
        self.assertEqual([name for name, _ in package.in_tree_units({tables, other})], ["unicode-data"])
        self.assertEqual(package.in_tree_units({other}), [])
        self.assertEqual(package.in_tree_units(set()), [])

    def test_records_match_the_files_and_notices(self):
        notices = tomllib.loads(package.PROVENANCE.read_text())["notices"]
        license_tag = "SPDX-" + "License-Identifier:"  # split, so REUSE does not read it as this file's tag
        for name, unit in package.IN_TREE_UNITS.items():
            with self.subTest(unit=name):
                self.assertIn(unit["notice"], notices)
                self.assertTrue((package.REPO / "LICENSES" / f"{unit['license']}.txt").is_file())
                for path in unit["files"]:
                    header = (package.REPO / path).read_text(encoding="utf-8").splitlines()[:5]
                    declared = next(line.split(":", 1)[1].strip() for line in header if license_tag in line)
                    self.assertIn(unit["license"], declared.split(" AND "))

    def test_license_pointer_names_unicode(self):
        self.assertIn("THIRD-PARTY-NOTICES", package.license_pointer("Unicode-3.0"))
        self.assertNotIn("Unicode", package.license_pointer("MIT"))

    def test_repository_files_are_all_listed(self):
        package.check_in_tree_units()

    def tree(self, files: dict[str, str]) -> pathlib.Path:
        """A synthetic repository root holding files, each with a header declaring its license."""
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        root = pathlib.Path(scratch.name)
        license_tag = "SPDX-" + "License-Identifier:"  # split, so REUSE does not read it as this file's tag
        for path, expression in files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(f"// {license_tag} {expression}\nint x;\n" if expression else "int x;\n")
        return root

    def test_unlisted_or_missing_data_fails_loudly(self):
        listed = next(iter(package.IN_TREE_UNITS.values()))["files"][0]
        mixed = "Apache-2.0 AND Unicode-3.0"
        package.check_in_tree_units(self.tree({listed: mixed, "src/a/b.cc": "Apache-2.0", "src/a/c.h": "Apache-2.0",
                                               "src/a/adapted.cu": "MIT AND Apache-2.0"}))  # a lock component's
        for files in ({listed: mixed, "src/tokenizer/renamed_data.cc": mixed},  # moved or copied tables
                      {listed: mixed, "src/a/other.cc": "Apache-2.0 AND CC-BY-4.0"},  # any other data license
                      {listed: mixed, "src/tokenizer/unicode_data.inc": mixed},  # tables in an included file
                      {listed: mixed, "src/a/no_header.cc": None},
                      {"src/tokenizer/renamed_data.cc": mixed}):  # the listed file is gone
            with self.subTest(files=sorted(files)), self.assertRaises(package.PackageError):
                package.check_in_tree_units(self.tree(files))

    def test_built_from_reads_ninja_and_fails_loudly(self):
        root = self.tree({})
        build = root / "build"
        build.mkdir()
        ninja = root / "ninja"
        ninja.write_text("#!/bin/sh\n[ \"$3 $4 $5 $6\" = \"-t inputs -0 -E\" ] || exit 2\n"
                         "printf '%s\\0' /abs/src/x.cc src/lib.a\n")
        ninja.chmod(0o755)
        self.assertEqual(package.built_from(build, ninja),
                         {pathlib.Path("/abs/src/x.cc"), (build / "src/lib.a").resolve()})
        ninja.write_text("#!/bin/sh\necho \"ninja: error: loading 'build.ninja'\" >&2\nexit 1\n")
        with self.assertRaisesRegex(package.PackageError, "build.ninja"):
            package.built_from(build, ninja)


class Provenance(unittest.TestCase):
    """Every shipped unit's notices can be extracted from this SDK and repository."""

    def test_every_notice_extracts(self):
        try:
            sdk = sdklib.load()
        except sdklib.SdkError as e:
            self.skipTest(str(e))
        if not (sdk.root / "sysroot").is_dir():
            self.skipTest(f"no SDK at {sdk.root}; run `mise run setup`")
        records = tomllib.loads(package.PROVENANCE.read_text())
        for name, notice in records["notices"].items():
            with self.subTest(notice=name):
                self.assertGreater(len(package.extract(notice["extract"], sdk.root)), 100)


if __name__ == "__main__":
    unittest.main()
