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
