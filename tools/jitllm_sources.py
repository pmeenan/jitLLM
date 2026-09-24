# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""The source lock and prepared sources (D-017, D-057): shared by tools/prepare-sources and its tests.

The lock (third_party/sources.lock.json) names every third-party source
component the build may use, with its exact bytes, license classification
and build options; third_party/README.md describes the schema. This module
validates it, selects a profile's closure, fetches and verifies archives,
applies recorded patches and computes the tree digest used during
preparation, configure and build.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import json
import os
import pathlib
import re
import stat
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

REPO = pathlib.Path(__file__).resolve().parent.parent
LOCK = REPO / "third_party" / "sources.lock.json"
# Prepared trees for every preset of this checkout (the isolated build area).
SOURCES_DIR = REPO / "build" / "sources"
SCHEMA = 1

# D-017's allowlist for incorporated implementation in the core.
CORE_LICENSES = frozenset({"Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "MIT", "MPL-2.0"})
CATEGORIES = ("implementation",)  # build tools and generators land with the first one
TIERS = ("core", "optional")
USES = ("product", "test")
MACHINES = ("target",)  # build-host tools need a host build; not supported yet
KINDS = ("archive",)  # vendored units land with M2's first adapted kernel

_ID = re.compile(r"^[a-z][a-z0-9-]{0,39}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LICENSE_ID = re.compile(r"^[A-Za-z0-9.+-]+$")
# Paths inside a prepared tree also pass through CMake lists, which cannot
# hold `;` or brackets; anything else unusual is refused too.
_TREE_NAME = re.compile(r"^[A-Za-z0-9._+@=,~-][A-Za-z0-9._+@=,~ -]*$")
_CMAKE_VAR = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# Option names a lock may not set: CMake's own variables and hooks, jitLLM's
# and FetchContent's controls, and the private names cmake/JitllmSources.cmake
# uses while adding a component (all start with `_`).
_RESERVED_OPTION = re.compile(r"^(_|cmake_|jitllm_|fetchcontent_)|^build_shared_libs$", re.IGNORECASE)
_CMAKE_PACKAGE = re.compile(r"^[A-Za-z][A-Za-z0-9_.+-]*$")
_TARGET = re.compile(r"^[A-Za-z0-9_.+-]+(::[A-Za-z0-9_.+-]+)?$")
_OPTION_VALUE = re.compile(r"^[A-Za-z0-9_.+-]*$")  # plain words, never paths


class SourceError(Exception):
    """A problem the user must fix; the message says how."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


# The lock ---------------------------------------------------------------------------

def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_lock(path: pathlib.Path = LOCK, *, modules: list[str] | None = None) -> dict:
    """Validates metadata, then patch bytes for the selected closure.

    None checks every component's patches (the explicit --check action).
    An empty list checks only core patches, without reading excluded files.
    """
    try:
        lock = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError, SourceError) as e:
        raise SourceError(f"cannot read the source lock {path}: {e}") from None
    problems = validate_lock(lock, path.parent, check_patch_files=False)
    if not problems:
        selected = lock["components"] if modules is None else select(lock, modules)
        for cid in selected:
            problems += _patch_file_problems(cid, lock["components"][cid]["patches"], path.parent)
    if problems:
        raise SourceError(f"{path} is not a valid source lock:\n  " + "\n  ".join(problems))
    return lock


def _relative_path(value: object, *, root: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if value == "":
        return root
    path = pathlib.PurePosixPath(value)
    return (not path.is_absolute() and ".." not in path.parts
            and (root or bool(path.parts)) and all(_TREE_NAME.fullmatch(p) for p in path.parts))


def _patch_file_problems(cid: str, patches: list[dict], base: pathlib.Path) -> list[str]:
    problems = []
    for patch in patches:
        full = base / patch["path"]
        try:
            if not full.resolve().is_relative_to(base.resolve()):
                problems.append(f"component {cid}: patch path {patch['path']!r} must stay under {base}")
            elif not full.is_file():
                problems.append(f"component {cid}: patch {patch['path']} is missing")
            elif sha256_file(full) != patch["sha256"]:
                problems.append(f"component {cid}: patch {patch['path']} does not match its recorded sha256")
        except OSError as e:
            problems.append(f"component {cid}: cannot read patch {patch['path']}: {e}")
    return problems


def _license_problems(where: str, tier: str, license_: object) -> list[str]:
    if not isinstance(license_, dict):
        return [f"{where}: license must be an object"]
    problems = []
    expression = license_.get("expression")
    if not isinstance(expression, str) or not expression:
        return [f"{where}: license.expression is missing; unclassified components are rejected (D-017)"]
    ids = expression.split(" AND ")
    if not all(_LICENSE_ID.fullmatch(i) for i in ids):
        problems.append(f"{where}: license.expression {expression!r} must be SPDX identifiers joined by ' AND '; "
                        "anything else needs a reviewed extension of this schema")
    elif tier == "core" and not set(ids) <= CORE_LICENSES:
        problems.append(f"{where}: {expression} is outside D-017's core allowlist "
                        f"({', '.join(sorted(CORE_LICENSES))}); it can only be an optional module")
    files = license_.get("files")
    if not isinstance(files, list) or not files or not all(_relative_path(f) for f in files):
        problems.append(f"{where}: license.files must list the license texts in the source")
    notices = license_.get("notices", [])
    if not isinstance(notices, list) or not all(_relative_path(f) for f in notices):
        problems.append(f"{where}: license.notices must list paths in the source")
    for key in ("evidence", "scope"):
        if not isinstance(license_.get(key), str) or not license_[key]:
            problems.append(f"{where}: license.{key} must record the audit")
    return problems


def validate_lock(lock: object, base: pathlib.Path, *, check_patch_files: bool = True) -> list[str]:
    """Every problem with a parsed lock; base is the directory patch paths are relative to."""
    if not isinstance(lock, dict) or type(lock.get("schema")) is not int or lock["schema"] != SCHEMA:
        return [f"schema must be {SCHEMA}"]
    problems = []
    modules = lock.get("modules", {})
    components = lock.get("components")
    if not isinstance(modules, dict):
        return ["modules must be an object"]
    if not isinstance(components, dict) or not components:
        return ["components must be a non-empty object"]
    for name, module in modules.items():
        if not isinstance(name, str) or not _ID.fullmatch(name):
            problems.append(f"module {name!r}: names are lowercase letters, digits and '-'")
        if not isinstance(module, dict) or not isinstance(module.get("description"), str):
            problems.append(f"module {name}: needs a description")
    used_modules = set()
    for cid, comp in components.items():
        where = f"component {cid}"
        if not isinstance(cid, str) or not _ID.fullmatch(cid):
            problems.append(f"{where}: ids are lowercase letters, digits and '-', at most 40")
            continue
        if not isinstance(comp, dict):
            problems.append(f"{where}: must be an object")
            continue
        for key, allowed in (("kind", KINDS), ("category", CATEGORIES), ("tier", TIERS), ("use", USES),
                             ("machine", MACHINES)):
            if comp.get(key) not in allowed:
                problems.append(f"{where}: {key} must be one of {', '.join(allowed)}, not {comp.get(key)!r}")
        if not isinstance(comp.get("version"), str) or not comp["version"]:
            problems.append(f"{where}: version is missing")
        tier = comp.get("tier")
        module = comp.get("module")
        if tier == "optional":
            if not isinstance(module, str) or module not in modules:
                problems.append(f"{where}: an optional component names a declared module, not {module!r}")
            else:
                used_modules.add(module)
        elif module is not None:
            problems.append(f"{where}: only optional components belong to a module")
        upstream = comp.get("upstream")
        if not isinstance(upstream, dict) or not isinstance(upstream.get("repository"), str):
            problems.append(f"{where}: upstream.repository is missing")
        elif not isinstance(upstream.get("commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", upstream["commit"]):
            problems.append(f"{where}: upstream.commit must be the full commit the release was made from")
        archive = comp.get("archive")
        if not isinstance(archive, dict):
            problems.append(f"{where}: archive is missing")
        else:
            if not isinstance(archive.get("sha256"), str) or not _SHA256.fullmatch(archive["sha256"]):
                problems.append(f"{where}: archive.sha256 must be 64 lowercase hex digits")
            if type(archive.get("size")) is not int or archive["size"] <= 0:
                problems.append(f"{where}: archive.size must be a positive byte count")
            name = archive.get("file")
            if not isinstance(name, str) or not _TREE_NAME.fullmatch(name) or name in (".", ".."):
                problems.append(f"{where}: archive.file must be a plain file name")
            urls = archive.get("urls")
            if not isinstance(urls, list) or not urls:
                problems.append(f"{where}: archive.urls must list at least one URL")
            else:
                for url in urls:
                    try:
                        parsed = urllib.parse.urlsplit(url) if isinstance(url, str) else None
                        valid = parsed is not None and ((parsed.scheme == "https" and bool(parsed.hostname))
                                                       or (parsed.scheme == "file" and not parsed.netloc
                                                           and parsed.path.startswith("/")))
                    except ValueError:
                        valid = False
                    if not valid:
                        problems.append(f"{where}: {url!r} is not an https:// (or local file://) URL")
        if not isinstance(comp.get("tree_sha256"), str) or not _SHA256.fullmatch(comp["tree_sha256"]):
            problems.append(f"{where}: tree_sha256 must be the prepared tree's digest")
        patches = comp.get("patches")
        if not isinstance(patches, list):
            problems.append(f"{where}: patches must be a list (empty when there are none)")
        else:
            for patch in patches:
                if not isinstance(patch, dict) or not isinstance(patch.get("path"), str) \
                        or not isinstance(patch.get("sha256"), str) or not _SHA256.fullmatch(patch["sha256"]):
                    problems.append(f"{where}: each patch needs a path and a sha256")
                    continue
                if not _relative_path(patch["path"]):
                    problems.append(f"{where}: patch path {patch['path']!r} must stay under {base}")
                    continue
                if check_patch_files:
                    problems += _patch_file_problems(cid, [patch], base)
        depends = comp.get("depends")
        if not isinstance(depends, list) or not all(isinstance(d, str) for d in depends):
            problems.append(f"{where}: depends must be a list of component ids")
        else:
            for dep in depends:
                other = components.get(dep)
                if not isinstance(other, dict):
                    problems.append(f"{where}: depends on {dep!r}, which is not in the lock")
                elif tier == "core" and other.get("tier") != "core":
                    problems.append(f"{where}: a core component cannot depend on the optional {dep} (D-002)")
                elif tier == "optional" and other.get("tier") == "optional" and other.get("module") != module:
                    problems.append(f"{where}: depends on {dep} from another module")
        cmake = comp.get("cmake")
        if not isinstance(cmake, dict):
            problems.append(f"{where}: cmake is missing")
        else:
            sub = cmake.get("subdirectory")
            if not _relative_path(sub, root=True):
                problems.append(f"{where}: cmake.subdirectory must be a relative path in the tree ('' for its root)")
            options = cmake.get("options")
            if not isinstance(options, dict) or not all(
                    isinstance(k, str) and _CMAKE_VAR.fullmatch(k) and isinstance(v, str)
                    and _OPTION_VALUE.fullmatch(v) for k, v in options.items()):
                problems.append(f"{where}: cmake.options must map CMake variable names to plain string values")
            elif any(_RESERVED_OPTION.search(k) for k in options):
                problems.append(f"{where}: cmake.options may not set CMAKE_*, JITLLM_*, FETCHCONTENT_*, "
                                "BUILD_SHARED_LIBS or names starting with '_'")
            for key, pattern in (("platform_packages", _CMAKE_PACKAGE), ("targets", _TARGET)):
                values = cmake.get(key)
                if not isinstance(values, list) or not all(isinstance(v, str) and pattern.fullmatch(v) for v in values):
                    problems.append(f"{where}: cmake.{key} must be a list of names")
            if isinstance(cmake.get("targets"), list) and not cmake["targets"]:
                problems.append(f"{where}: cmake.targets must name what jitLLM links")
        problems += _license_problems(where, tier, comp.get("license"))
        for key in ("verification",):
            if not isinstance(comp.get(key), str) or not comp[key]:
                problems.append(f"{where}: {key} must say how the pin was checked")
    for name in set(modules) - used_modules:
        problems.append(f"module {name}: no component belongs to it")
    if not problems:
        try:
            dependency_order(components, list(components))
        except SourceError as e:
            problems.append(str(e))
    return problems


def dependency_order(components: dict, ids: list[str]) -> list[str]:
    """ids with each component after its dependencies; rejects cycles."""
    order, state = [], {}

    def visit(cid: str, path: tuple[str, ...]) -> None:
        if state.get(cid) == "done":
            return
        if state.get(cid) == "visiting":
            raise SourceError("dependency cycle: " + " -> ".join(path + (cid,)))
        state[cid] = "visiting"
        for dep in components[cid]["depends"]:
            visit(dep, path + (cid,))
        state[cid] = "done"
        order.append(cid)

    for cid in sorted(ids):
        visit(cid, ())
    return order


def select(lock: dict, modules: list[str]) -> list[str]:
    """The closure for a profile: core components plus the enabled modules' own, in dependency order.

    Raises before anything is fetched if a module is unknown or a selected
    component needs one that the profile leaves out.
    """
    unknown = sorted(set(modules) - set(lock.get("modules", {})))
    if unknown:
        declared = ", ".join(sorted(lock.get("modules", {}))) or "none"
        raise SourceError(f"unknown module(s) {', '.join(unknown)}; the lock declares {declared}")
    components = lock["components"]
    chosen = [cid for cid, comp in components.items()
              if comp["tier"] == "core" or comp.get("module") in modules]
    for cid in chosen:
        for dep in components[cid]["depends"]:
            if dep not in chosen:
                raise SourceError(f"{cid} depends on {dep}, which this profile does not select "
                                  f"(module {components[dep].get('module')})")
    return dependency_order(components, chosen)


def license_profile(modules: list[str]) -> str:
    """The profile name receipts use: `core`, or `core+<module>...` with the enabled modules."""
    return "+".join(["core", *sorted(set(modules))])


def prepared_dir(root: pathlib.Path, cid: str, comp: dict) -> pathlib.Path:
    """Where a component's prepared tree lives; cmake/JitllmSources.cmake uses the same name."""
    return root / f"{cid}-{comp['tree_sha256'][:16]}"


# Tree digest --------------------------------------------------------------------------

def tree_entries(root: pathlib.Path) -> list[tuple[str, bool, str]]:
    """(relative path, owner-executable, sha256) for every regular file, sorted by path bytes.

    Symbolic links, special files and names CMake's lists cannot carry are
    refused. An unreadable directory must not silently vanish from the hash.
    """
    if root.is_symlink() or not root.is_dir():
        raise SourceError(f"{root} is not a prepared directory")

    def failed_scan(error: OSError) -> None:
        raise SourceError(f"cannot scan prepared tree {root}: {error}") from error

    entries = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=failed_scan):
        dirnames.sort()
        for name in dirnames + filenames:
            path = pathlib.Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if not _TREE_NAME.fullmatch(name):
                raise SourceError(f"{rel}: unsupported file name in a prepared tree")
            try:
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    raise SourceError(f"{rel}: symbolic links are not supported in prepared trees")
                if stat.S_ISDIR(info.st_mode):
                    continue
                if not stat.S_ISREG(info.st_mode):
                    raise SourceError(f"{rel}: not a regular file")
                entries.append((rel, bool(info.st_mode & stat.S_IXUSR), sha256_file(path)))
            except OSError as e:
                raise SourceError(f"cannot read prepared file {rel}: {e}") from e
    if not entries:
        raise SourceError(f"{root} has no files; an empty tree is never a prepared source")
    entries.sort(key=lambda e: e[0].encode())
    return entries


def tree_digest(root: pathlib.Path) -> str:
    """SHA-256 of one `<sha256> <x|-> <path>` line per file; see cmake/sources/TreeDigest.cmake."""
    text = "".join(f"{digest} {'x' if executable else '-'} {rel}\n" for rel, executable, digest in tree_entries(root))
    return hashlib.sha256(text.encode()).hexdigest()


# Patches ------------------------------------------------------------------------------

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
# git diff extended headers this applier does not implement.
_UNSUPPORTED = ("old mode", "new mode", "rename ", "copy ", "similarity index", "dissimilarity index",
                "Binary files", "GIT binary patch")


# git diff extended headers this applier accepts (a new file's mode is checked).
_GIT_HEADERS = ("index ", "new file mode ", "deleted file mode ")


def _lines(text: str) -> list[str]:
    """Splits on newlines only, keeping them (str.splitlines also splits on \r, \f and more)."""
    return re.findall(r"[^\n]*\n|[^\n]+$", text)


def _read(path: pathlib.Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: pathlib.Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _patch_path(header: str, root: pathlib.Path) -> pathlib.Path | None:
    name = header.rstrip("\n").split("\t", 1)[0]
    if name == "/dev/null":
        return None
    parts = pathlib.PurePosixPath(name).parts
    if len(parts) < 2 or parts[0] not in ("a", "b"):
        raise SourceError(f"patch path {name!r} must start with a/ or b/ (git diff format)")
    rel = pathlib.PurePosixPath(*parts[1:])
    if ".." in rel.parts or not all(_TREE_NAME.fullmatch(p) for p in rel.parts):
        raise SourceError(f"patch path {name!r} leaves the source tree or has an unsupported name")
    return root / rel


def apply_patch(root: pathlib.Path, text: str) -> None:
    """Applies a git-style unified diff exactly: every hunk at its stated line, with no fuzz.

    Handles modified, new (--- /dev/null) and deleted (+++ /dev/null) UTF-8
    text files. A preamble before the first file and git's signature after
    the last are skipped; any other line outside a file's hunks, renames,
    mode changes, binary patches and any mismatch are errors, and the tree
    may be left partly patched: callers patch a staging copy and discard it
    on failure.
    """
    lines = _lines(text)
    i, touched, started = 0, 0, False
    while i < len(lines):
        line = lines[i]
        if not started and not line.startswith(("diff --git ", "--- ")):
            i += 1  # a preamble, such as the message git format-patch writes
            continue
        started = True
        if line == "-- \n":  # git format-patch's signature ends the patch
            break
        if line.startswith(_UNSUPPORTED) or (
                line.startswith("new file mode ") and line.rstrip("\n") != "new file mode 100644"):
            raise SourceError(f"patch line {i + 1}: {line.strip()!r} is not supported")
        if line.startswith(("diff --git ", *_GIT_HEADERS)):
            i += 1
            continue
        if not line.startswith("--- "):
            raise SourceError(f"patch line {i + 1}: unexpected {line.strip()!r} between files")
        if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
            raise SourceError(f"patch line {i + 1}: '---' without '+++'")
        old_path = _patch_path(lines[i][4:], root)
        new_path = _patch_path(lines[i + 1][4:], root)
        target = new_path or old_path
        if target is None:
            raise SourceError(f"patch line {i + 1}: no file named")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise SourceError(f"patch line {i + 1}: renames are not supported")
        rel = target.relative_to(root)
        if any(p.is_symlink() for p in [target, *target.parents] if p.is_relative_to(root) and p != root):
            raise SourceError(f"patch touches {rel} through a symbolic link")
        if old_path is None:
            if target.exists():
                raise SourceError(f"patch creates {rel}, which exists")
            original = []
        else:
            if not target.is_file():
                raise SourceError(f"patch changes {rel}, which is not a regular file")
            original = _lines(_read(target))
        i += 2
        result, cursor, hunks = [], 0, 0
        while i < len(lines) and lines[i].startswith("@@"):
            match = _HUNK.match(lines[i])
            if not match:
                raise SourceError(f"patch line {i + 1}: malformed hunk header")
            old_start, old_count = int(match[1]), int(match[2] if match[2] is not None else 1)
            new_start, new_count = int(match[3]), int(match[4] if match[4] is not None else 1)
            start = old_start - 1 if old_count else old_start
            if start < cursor or start > len(original):
                raise SourceError(f"patch line {i + 1}: hunk out of order or past the end of {rel}")
            result += original[cursor:start]
            new_index = new_start - 1 if new_count else new_start
            if new_index != len(result):
                raise SourceError(f"patch line {i + 1}: new hunk position does not match {rel}")
            cursor = start
            i += 1
            hunks += 1
            seen_old = seen_new = 0
            while seen_old < old_count or seen_new < new_count:
                if i >= len(lines):
                    raise SourceError("patch ends inside a hunk")
                line = lines[i]
                tag, body = line[:1], line[1:]
                if not line.endswith("\n"):
                    raise SourceError(f"patch line {i + 1}: truncated hunk line")
                no_newline = i + 1 < len(lines) and lines[i + 1].startswith("\\")
                if no_newline:
                    if lines[i + 1].rstrip("\n") != "\\ No newline at end of file":
                        raise SourceError(f"patch line {i + 2}: unknown newline marker")
                    body = body[:-1]
                if tag in (" ", "-"):
                    if cursor >= len(original) or original[cursor] != body:
                        raise SourceError(f"patch line {i + 1}: context does not match {rel}:{cursor + 1}")
                    if tag == " ":
                        result.append(original[cursor])
                        seen_new += 1
                    cursor += 1
                    seen_old += 1
                elif tag == "+":
                    result.append(body)
                    seen_new += 1
                else:
                    raise SourceError(f"patch line {i + 1}: unexpected line in hunk")
                i += 2 if no_newline else 1
            if seen_old != old_count or seen_new != new_count:
                raise SourceError(f"patch hunk before line {i + 1} does not match its header counts")
        if (i < len(lines) and lines[i].startswith(("+", "-", " ", "\\")) and not lines[i].startswith("--- ")
                and lines[i] != "-- \n"):
            raise SourceError(f"patch line {i + 1}: unexpected line after hunk")
        if not hunks and old_path is not None:
            raise SourceError(f"patch names {rel} but changes nothing in it")
        result += original[cursor:]
        if any(not line.endswith("\n") for line in result[:-1]):
            raise SourceError(f"patch inserts a missing-newline marker before the end of {rel}")
        if new_path is None:
            if cursor != len(original) or result:
                raise SourceError(f"patch deletes {rel} but does not remove all of it")
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
            _write(target, "".join(result))
            if mode is not None:
                target.chmod(mode)
        touched += 1
    if not touched:
        raise SourceError("patch changes no files")


# Archives ------------------------------------------------------------------------------

def check_archive(path: pathlib.Path) -> None:
    """Refuses an archive with anything but plain files and directories at safe paths.

    Runs before extraction: a hash-matched archive is reviewed bytes, but a
    link member would let extraction write outside the staging directory
    before the tree digest could reject it.
    """
    def check_name(name: str) -> None:
        parts = pathlib.PurePosixPath(name).parts
        if name.startswith("/") or ".." in parts or not all(_TREE_NAME.fullmatch(p) for p in parts if p != "."):
            raise SourceError(f"{path.name}: member {name!r} has an unsafe or unsupported path")

    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    check_name(info.filename.rstrip("/"))
                    kind = stat.S_IFMT(info.external_attr >> 16)
                    if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                        raise SourceError(f"{path.name}: member {info.filename!r} is not a file or directory")
            return
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                check_name(member.name.rstrip("/"))
                if not (member.isfile() or member.isdir()):
                    raise SourceError(f"{path.name}: member {member.name!r} is not a file or directory")
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as e:
        raise SourceError(f"cannot read the archive {path.name}: {e}") from None


# Fetching -----------------------------------------------------------------------------

def download_path(cache: pathlib.Path, archive: dict) -> pathlib.Path:
    """The persistent cache shares tools/setup-toolchain's content-addressed download layout."""
    return cache / "downloads" / archive["sha256"] / archive["file"]


def verified(path: pathlib.Path, archive: dict) -> bool:
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == archive["size"]
            and sha256_file(path) == archive["sha256"])


def fetch(cache: pathlib.Path, cid: str, archive: dict, log=print) -> pathlib.Path:
    """The cached archive, downloaded first if absent or wrong; always re-verified.

    Bytes that differ from the lock under the same URL are an error, never a
    lock update. TLS certificates are verified (urllib's default context).
    """
    path = download_path(cache, archive)
    if verified(path, archive):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in archive["urls"]:
        for attempt in range(3):
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".part-", delete=False) as tmp:
                try:
                    digest, size = hashlib.sha256(), 0
                    request = urllib.request.Request(url, headers={"User-Agent": "jitllm-prepare-sources"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        while chunk := response.read(1 << 20):
                            size += len(chunk)
                            if size > archive["size"]:
                                raise SourceError(f"more than the locked {archive['size']} bytes")
                            digest.update(chunk)
                            tmp.write(chunk)
                    if size < archive["size"]:
                        raise http.client.IncompleteRead(b"", archive["size"] - size)
                    if digest.hexdigest() != archive["sha256"]:
                        raise SourceError(f"got {size} bytes with SHA-256 {digest.hexdigest()}, "
                                          f"not the locked {archive['sha256']}")
                    tmp.close()
                    os.replace(tmp.name, path)
                    log(f"  fetched {cid} from {url}")
                    return path
                except (OSError, http.client.HTTPException, SourceError) as e:
                    errors.append(f"{url} (attempt {attempt + 1}): {e}")
                    permanent = isinstance(e, SourceError) or url.startswith("file:") or (
                        isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500)
                    if permanent:
                        break
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(tmp.name)
    raise SourceError(f"could not fetch {cid}:\n  " + "\n  ".join(errors))
