# SPDX-FileCopyrightText: 2026 jitLLM contributors
# SPDX-License-Identifier: Apache-2.0
"""What the jitLLM package says about itself (D-017, D-029, D-063, D-071, D-074).

From a build's receipt, the source lock, toolchains/provenance.toml and the
SDK, this writes the package's documentation and control facts:

    control.json          the Debian version, architecture and dependencies
    copyright             /usr/share/doc/jitllm/copyright (Debian's machine-readable format)
    THIRD-PARTY-NOTICES   the notices of everything a packaged binary carries
    jitllm.spdx.json      the SBOM, SPDX 2.3

The dependencies follow the binaries: libc6 at the highest GLIBC_ symbol
version they import, and in CUDA builds the driver's libcuda.so.1 at
NVIDIA's minimum for the toolkit's major version (CUDA_DRIVER_FLOOR).
Notices are included for every platform unit that ships and every product
component, whether or not this build's code reaches the part a notice
covers: an extra notice costs nothing, a missing one is a defect.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import json
import pathlib
import re
import subprocess
import tarfile
import tempfile
import tomllib
import uuid

import jitllm_sdk as sdklib
import jitllm_sources as srclib

REPO = sdklib.REPO
PROVENANCE = REPO / "toolchains" / "provenance.toml"
PACKAGE = "jitllm"
HOMEPAGE = "https://github.com/pmeenan/jitLLM"
# The executables the package installs: (build-tree path, installed path).
EXECUTABLES = (("src/cli/jitllm", "usr/bin/jitllm"),
               ("src/runtime/jitllm-runtime", "usr/libexec/jitllm/jitllm-runtime"))
# NVIDIA's minimum driver for CUDA 13.x minor-version compatibility (CUDA
# Toolkit release notes, table 3, ">= 580", checked 2026-09-24); the build
# carries SASS only, so no PTX JIT needs a newer one.
CUDA_DRIVER_FLOOR = "580"
# The shared libraries a packaged binary may need, and where they come from.
ALLOWED_NEEDED = {"libc.so.6": "libc6", "libm.so.6": "libc6", "ld-linux-aarch64.so.1": "libc6",
                  "ld-linux-x86-64.so.2": "libc6", "libcuda.so.1": "libcuda.so.1"}
DEBIAN_ARCH = {"aarch64-linux-gnu": "arm64", "x86_64-linux-gnu": "amd64"}
# Provenance units whose code reaches only CUDA builds.
CUDA_UNITS = ("cuda-runtime", "cccl")


class PackageError(Exception):
    pass


def extract(spec: dict, sdk_root: pathlib.Path) -> str:
    """A notice's text as provenance.toml's `extract` locates it."""
    kind, _, rel = spec["file"].partition(":")
    base = {"sdk": sdk_root, "repo": REPO}.get(kind)
    if base is None:
        raise PackageError(f"notice file {spec['file']!r} is neither sdk: nor repo:")
    path = base / rel
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise PackageError(f"cannot read the notice source {path}: {e}") from None
    if "from" not in spec:
        return "\n".join(lines).strip("\n") + "\n"
    start = next((i for i, line in enumerate(lines) if spec["from"] in line), None)
    if start is None:
        raise PackageError(f"{path} has no line containing {spec['from']!r}")
    end = next((i for i in range(start, len(lines)) if spec.get("to", spec["from"]) in lines[i]), None)
    if end is None:
        raise PackageError(f"{path} has no line containing {spec['to']!r} after {spec['from']!r}")
    end += spec.get("plus", 0)
    if end >= len(lines):
        raise PackageError(f"{path} ends before the notice does")
    return "\n".join(_uncomment(lines[start:end + 1])) + "\n"


def _uncomment(lines: list[str]) -> list[str]:
    """Lines without a comment prefix common to all of them (" * ", "// ", "#")."""
    for prefix in (" * ", "// ", "# ", " *", "//", "#"):
        if all(line.startswith(prefix) or line.strip() in ("", prefix.strip()) for line in lines):
            return [line[len(prefix):] if line.startswith(prefix) else "" for line in lines]
    return lines


def component_notice(source: pathlib.Path, notice: str) -> str:
    """A lock component's notice: a file in its prepared tree, whole or a line range."""
    rel, lines = srclib.notice_range(notice)
    try:
        text = (source / rel).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise PackageError(f"cannot read {source / rel}: {e}") from None
    if lines:
        if lines[1] > len(text):
            raise PackageError(f"{source / rel} has fewer than {lines[1]} lines")
        text = [line.strip().removeprefix("//").strip() for line in text[lines[0] - 1:lines[1]]]
    return "\n".join(text).strip("\n") + "\n"


def binary_facts(readelf: pathlib.Path, binary: pathlib.Path) -> dict:
    """The shared libraries a binary needs, the highest GLIBC_ version it imports, and whether it has a run path."""
    dynamic = subprocess.run([readelf, "--dynamic", "--wide", binary], capture_output=True, text=True)
    versions = subprocess.run([readelf, "--version-info", "--wide", binary], capture_output=True, text=True)
    if dynamic.returncode or versions.returncode:
        raise PackageError(f"llvm-readelf cannot read {binary}: {dynamic.stderr or versions.stderr}")
    needed = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", dynamic.stdout)
    glibc = [tuple(int(p) for p in v.split(".")) for v in re.findall(r"GLIBC_([0-9]+(?:\.[0-9]+)+)", versions.stdout)]
    return {"needed": needed, "glibc": max(glibc) if glibc else None,
            "runpath": bool(re.search(r"\((RPATH|RUNPATH)\)", dynamic.stdout))}


def _unit_version(sdk: sdklib.Sdk, unit: dict, arch: str) -> str:
    versions = set()
    for key in unit.get("artifacts", []):
        artifact = sdk.lock["artifacts"].get(f"{key}/{arch}") or sdk.lock["artifacts"].get(key)
        if artifact:
            versions.add(artifact["version"])
    return ", ".join(sorted(versions)) or "NOASSERTION"


# Where Debian keeps the full text of a license the notices only name.
COMMON_LICENSES = {"GPL-3.0-or-later": "GPL-3", "GPL-2.0-only": "GPL-2", "LGPL-2.1-or-later": "LGPL-2.1",
                   "Apache-2.0": "Apache-2.0"}


def license_pointer(expression: str) -> str:
    """A Debian copyright License paragraph's body: where the terms of expression are."""
    lines = []
    for base, name in COMMON_LICENSES.items():
        if re.search(rf"(^|[ (]){re.escape(base)}([ )]|$)", expression):
            lines.append(f" On Debian systems, the full text of {base} is in /usr/share/common-licenses/{name}.")
    for exception in re.findall(r"WITH ([A-Za-z0-9.-]+)", expression):
        lines.append(f" {exception}: https://spdx.org/licenses/{exception}.html; it lets object code built with the "
                     "component be distributed on the program's own terms.")
    if "LicenseRef-NVIDIA-CUDA-EULA" in expression:
        lines.append(" The NVIDIA CUDA Toolkit End User License Agreement, reproduced in full in")
        lines.append(" /usr/share/doc/jitllm/THIRD-PARTY-NOTICES.")
    if re.search(r"(^|[ (])MIT([ )]|$)", expression):
        lines.append(" The MIT texts, with their copyright notices, are in /usr/share/doc/jitllm/THIRD-PARTY-NOTICES.")
    return "\n".join(lines)


def shipped_units(provenance: dict, cuda: bool) -> list[tuple[str, dict]]:
    return [(name, unit) for name, unit in provenance["units"].items()
            if unit["ships"] and (cuda or name not in CUDA_UNITS)]


def generate(build: pathlib.Path, sdk: sdklib.Sdk, out: pathlib.Path) -> dict:
    """Writes the package's documents for the build in `build` into `out`; returns control.json's content."""
    try:
        receipt = json.loads((build / "jitllm-receipt.json").read_text())
    except (OSError, ValueError) as e:
        raise PackageError(f"cannot read {build / 'jitllm-receipt.json'}: {e}; build first") from None
    arch = DEBIAN_ARCH.get(receipt["target"])
    if arch is None:
        raise PackageError(f"no Debian architecture for {receipt['target']}")
    provenance = tomllib.loads(PROVENANCE.read_text())
    lock = srclib.load_lock(srclib.LOCK, modules=receipt["modules"])
    readelf = sdk.root / "bin" / "llvm-readelf"
    cuda = bool(receipt["cuda"])

    # Dependencies, from what the binaries import.
    glibc, needs_cuda = (0,), False
    for built, installed in EXECUTABLES:
        facts = binary_facts(readelf, build / built)
        unknown = sorted(set(facts["needed"]) - set(ALLOWED_NEEDED))
        if unknown:
            raise PackageError(f"{installed} needs {', '.join(unknown)}, which no dependency provides")
        if facts["runpath"]:
            raise PackageError(f"{installed} has a run path (D-060)")
        glibc = max(glibc, facts["glibc"] or (0,))
        needs_cuda = needs_cuda or "libcuda.so.1" in facts["needed"]
    depends = [f"libc6 (>= {'.'.join(map(str, glibc))})"]
    if needs_cuda:
        depends.append(f"libcuda.so.1 (>= {CUDA_DRIVER_FLOOR})")
    # systemd-sysusers and systemd-tmpfiles run from the maintainer scripts.
    depends.append("systemd")

    products = [c for c in receipt["components"] if c["use"] == "product"]
    units = shipped_units(provenance, cuda)
    out.mkdir(parents=True, exist_ok=True)

    # THIRD-PARTY-NOTICES.
    parts = [
        "jitLLM third-party notices",
        "==========================",
        "",
        f"For jitllm {receipt['version']['product']}, {receipt['target']}, license profile "
        f"{receipt['license_profile']}. jitLLM's own code is under the Apache License 2.0 (LICENSE, NOTICE).",
        "The binaries also carry the code below, under the terms that follow. A notice is included whenever its",
        "component ships, whether or not this build uses the part it covers.",
    ]
    if cuda:
        parts += ["",
                  "The NVIDIA CUDA runtime object code (libcudart_static) and the code NVCC generates in these",
                  "binaries are under the NVIDIA CUDA Toolkit End User License Agreement, reproduced below, not",
                  "under the Apache License. The NVIDIA driver (libcuda.so.1) is not part of this package."]
    seen: set[str] = set()
    for component in products:
        entry = lock["components"][component["id"]]
        source = pathlib.Path(component["source"])
        parts += ["", "-" * 78, f"{component['id']} {component['version']} ({entry['license']['expression']})",
                  entry["upstream"]["repository"], ""]
        for notice in entry["license"]["notices"]:
            parts.append(component_notice(source, notice))
    for name, unit in units:
        for notice in unit["notices"]:
            if notice in seen:
                continue
            seen.add(notice)
            record = provenance["notices"][notice]
            parts += ["", "-" * 78, f"{record['text']} ({name})", f"Applies: {record['when']}.", "",
                      extract(record["extract"], sdk.root)]
    (out / "THIRD-PARTY-NOTICES").write_text("\n".join(parts).rstrip("\n") + "\n")

    # copyright (https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/): everything is
    # jitLLM's, and the two executables also contain the components the notices cover.
    licenses = ["Apache-2.0"]
    contains = []
    for component in products:
        entry = lock["components"][component["id"]]
        contains.append(f" {component['id']} {component['version']}: {entry['license']['expression']}")
        licenses.append(entry["license"]["expression"])
    for name, unit in units:
        contains.append(f" {name} (build toolchain, {unit['category']}): {unit['license']}")
        licenses.append(unit["license"])
    distinct = list(dict.fromkeys(licenses))
    executables = " ".join(installed for _, installed in EXECUTABLES)
    stanzas = [
        "Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/\n"
        f"Upstream-Name: jitLLM\nSource: {HOMEPAGE}",
        "Files: *\nCopyright: 2026 jitLLM contributors\nLicense: Apache-2.0",
        f"Files: {executables}\nCopyright: 2026 jitLLM contributors, and the holders named in THIRD-PARTY-NOTICES\n"
        f"License: {' AND '.join(f'({x})' if ' ' in x else x for x in distinct)}\n"
        "Comment: These executables also contain code from the following, whose notices are in\n"
        " /usr/share/doc/jitllm/THIRD-PARTY-NOTICES; jitllm.spdx.json lists them with their versions.\n"
        + "\n".join(contains),
        "License: Apache-2.0\n On Debian systems, the full text of the Apache License 2.0 is in\n"
        " /usr/share/common-licenses/Apache-2.0, and in /usr/share/doc/jitllm/LICENSE.",
    ]
    for expression in distinct[1:]:
        stanzas.append(f"License: {expression}\n" + license_pointer(expression))
    (out / "copyright").write_text("\n\n".join(stanzas) + "\n")

    # The SBOM (SPDX 2.3).
    version = receipt["version"]
    created = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    root_id = "SPDXRef-Package-jitllm"
    packages = [{
        "SPDXID": root_id, "name": PACKAGE, "versionInfo": version["product"],
        "downloadLocation": f"git+{HOMEPAGE}.git@{version['commit']}" if version.get("commit") else "NOASSERTION",
        "homepage": HOMEPAGE, "licenseConcluded": "NOASSERTION", "licenseDeclared": "Apache-2.0",
        "copyrightText": "2026 jitLLM contributors", "supplier": "Organization: jitLLM contributors",
        "primaryPackagePurpose": "APPLICATION", "filesAnalyzed": False,
        "comment": f"License profile {receipt['license_profile']}; target {receipt['target']}; SDK {receipt['sdk']}; "
                   f"source lock sha256 {receipt['source_lock']['sha256']}.",
    }]
    relationships = [{"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES",
                      "relatedSpdxElement": root_id}]
    for component in products:
        entry = lock["components"][component["id"]]
        ident = f"SPDXRef-Source-{component['id']}"
        packages.append({
            "SPDXID": ident, "name": component["id"], "versionInfo": component["version"],
            "downloadLocation": entry["archive"]["urls"][0],
            "checksums": [{"algorithm": "SHA256", "checksumValue": entry["archive"]["sha256"]}],
            "licenseConcluded": entry["license"]["expression"], "licenseDeclared": entry["license"]["expression"],
            "copyrightText": "NOASSERTION", "filesAnalyzed": False, "primaryPackagePurpose": "LIBRARY",
            "comment": f"Commit {entry['upstream']['commit']}; incorporated implementation, {entry['tier']} tier (D-017).",
        })
        relationships.append({"spdxElementId": root_id, "relationshipType": "CONTAINS", "relatedSpdxElement": ident})
    for name, unit in units:
        ident = f"SPDXRef-Platform-{name}"
        packages.append({
            "SPDXID": ident, "name": name, "versionInfo": _unit_version(sdk, unit, arch),
            "downloadLocation": "NOASSERTION", "licenseConcluded": "NOASSERTION",
            "licenseDeclared": unit["license"],
            "copyrightText": "NOASSERTION", "filesAnalyzed": False, "primaryPackagePurpose": "LIBRARY",
            "comment": f"D-017 {unit['category']} ({unit['license']}): {unit['enters']}",
        })
        # Everything listed contributes code to the executables (glibc: its
        # start files and header code; the shared C library is a dependency
        # the package declares).
        relationships.append({"spdxElementId": root_id, "relationshipType": "CONTAINS", "relatedSpdxElement": ident})
    namespace = uuid.uuid5(uuid.NAMESPACE_URL, f"{HOMEPAGE}/sbom/{version['product']}/{receipt['target']}")
    sbom = {
        "spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{PACKAGE}-{version['product']}-{arch}",
        "documentNamespace": f"{HOMEPAGE}/spdx/{namespace}",
        "creationInfo": {"created": created, "creators": ["Tool: jitLLM tools/jitllm_package.py"]},
        "packages": packages, "relationships": relationships,
    }
    refs = sorted({ref for unit in (u for _, u in units) for ref in re.findall(r"LicenseRef-[A-Za-z0-9.-]+",
                                                                                 unit["license"])})
    if refs:
        sbom["hasExtractedLicensingInfos"] = [
            {"licenseId": ref, "name": ref.removeprefix("LicenseRef-"),
             "extractedText": "See /usr/share/doc/jitllm/THIRD-PARTY-NOTICES, which reproduces its terms."}
            for ref in refs]
    (out / "jitllm.spdx.json").write_text(json.dumps(sbom, indent=2) + "\n")

    control = {"package": PACKAGE, "version": version["debian"], "architecture": arch,
               "depends": ", ".join(depends), "homepage": HOMEPAGE,
               "license_profile": receipt["license_profile"], "official": receipt["official"]}
    (out / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    return control


# The package's inventory ---------------------------------------------------------

# Every path the package installs (under ./), with its type and mode; the
# owner is always root.
EXPECTED_FILES = {
    "usr/bin/jitllm": 0o755,
    "usr/libexec/jitllm/jitllm-runtime": 0o755,
    "usr/lib/systemd/system/jitllm.service": 0o644,
    "usr/lib/sysusers.d/jitllm.conf": 0o644,
    "usr/lib/tmpfiles.d/jitllm.conf": 0o644,
    "usr/share/doc/jitllm/LICENSE": 0o644,
    "usr/share/doc/jitllm/NOTICE": 0o644,
    "usr/share/doc/jitllm/CHANGELOG.md": 0o644,
    "usr/share/doc/jitllm/copyright": 0o644,
    "usr/share/doc/jitllm/THIRD-PARTY-NOTICES": 0o644,
    "usr/share/doc/jitllm/jitllm.spdx.json": 0o644,
    "usr/share/doc/jitllm/examples/jitllm.toml": 0o644,
}
# Files installed unchanged from the repository.
VERBATIM = {"usr/share/doc/jitllm/LICENSE": "LICENSE", "usr/share/doc/jitllm/NOTICE": "NOTICE",
            "usr/share/doc/jitllm/CHANGELOG.md": "CHANGELOG.md",
            "usr/lib/systemd/system/jitllm.service": "packaging/jitllm.service",
            "usr/lib/sysusers.d/jitllm.conf": "packaging/jitllm.sysusers",
            "usr/lib/tmpfiles.d/jitllm.conf": "packaging/jitllm.tmpfiles",
            "usr/share/doc/jitllm/examples/jitllm.toml": "packaging/jitllm.example.toml"}
MAINTAINER_SCRIPTS = ("postinst", "prerm", "postrm")


def read_deb(path: pathlib.Path) -> dict[str, dict[str, tuple[tarfile.TarInfo, bytes | None]]]:
    """A .deb's members: {"control": {...}, "data": {...}}, each name to (entry, content of a regular file)."""
    data = path.read_bytes()
    if not data.startswith(b"!<arch>\n"):
        raise PackageError(f"{path} is not an ar archive")
    members, at = {}, 8
    while at < len(data):
        header = data[at:at + 60]
        name = header[:16].decode().strip().rstrip("/")
        size = int(header[48:58].decode().strip())
        members[name] = data[at + 60:at + 60 + size]
        at += 60 + size + (size & 1)
    if members.get("debian-binary") != b"2.0\n":
        raise PackageError(f"{path} is not a version 2.0 Debian package")
    out = {}
    for part in ("control", "data"):
        name = next((m for m in members if m.startswith(f"{part}.tar")), None)
        if name is None:
            raise PackageError(f"{path} has no {part} archive")
        entries = {}
        with tarfile.open(fileobj=io.BytesIO(members[name])) as tar:
            for info in tar.getmembers():
                content = tar.extractfile(info).read() if info.isfile() else None
                entries[info.name.removeprefix("./").rstrip("/")] = (info, content)
        out[part] = entries
    return out


def check_package(deb: pathlib.Path, build: pathlib.Path, sdk: sdklib.Sdk) -> list[str]:
    """Every way the package differs from what its build's receipt, the repository and its documents say."""
    problems = []
    receipt = json.loads((build / "jitllm-receipt.json").read_text())
    doc = build / "package" / "doc"
    control_facts = json.loads((doc / "control.json").read_text())
    parts = read_deb(deb)
    data, control = parts["data"], parts["control"]

    # The control file.
    fields = dict(re.findall(r"^([A-Za-z-]+): (.*)$", (control.get("control", (None, b""))[1] or b"").decode(),
                             re.MULTILINE))
    for field, want in (("Package", PACKAGE), ("Version", receipt["version"]["debian"]),
                        ("Architecture", DEBIAN_ARCH.get(receipt["target"])), ("Depends", control_facts["depends"])):
        if fields.get(field) != want:
            problems.append(f"control: {field} is {fields.get(field)!r}, not {want!r}")
    for script in MAINTAINER_SCRIPTS:
        entry = control.get(script)
        want = (REPO / "packaging" / "debian" / script).read_bytes()
        if entry is None or entry[1] != want or entry[0].mode & 0o777 != 0o755:
            problems.append(f"control: {script} is missing, differs from packaging/debian/{script} or is not 0755")
    md5sums = dict(line.split("  ", 1)[::-1] for line in (control.get("md5sums", (None, b""))[1] or b"")
                   .decode().splitlines() if "  " in line)

    # The files.
    files = {name: entry for name, entry in data.items() if name and not entry[0].isdir()}
    for name in sorted(set(files) - set(EXPECTED_FILES)):
        problems.append(f"data: {name} is not part of the installed layout")
    for name, mode in EXPECTED_FILES.items():
        entry = files.get(name)
        if entry is None:
            problems.append(f"data: {name} is missing")
            continue
        info, content = entry
        if not info.isfile() or info.mode & 0o7777 != mode or info.uid != 0 or info.gid != 0:
            problems.append(f"data: {name} is not a regular file owned by root with mode {mode:04o} "
                            f"(mode {info.mode & 0o7777:04o}, uid {info.uid})")
        if md5sums.get(name) != hashlib.md5(content or b"").hexdigest():  # noqa: S324 (dpkg's own format)
            problems.append(f"control: md5sums does not match {name}")
        if name in VERBATIM and content != (REPO / VERBATIM[name]).read_bytes():
            problems.append(f"data: {name} differs from {VERBATIM[name]}")
    for name, (info, _) in data.items():
        if info.isdir() and (info.mode & 0o7777 != 0o755 or info.uid != 0):
            problems.append(f"data: directory {name or '.'} is not root's with mode 0755")
    for name in ("copyright", "THIRD-PARTY-NOTICES", "jitllm.spdx.json"):
        entry = files.get(f"usr/share/doc/jitllm/{name}")
        if entry and entry[1] != (doc / name).read_bytes():
            problems.append(f"data: usr/share/doc/jitllm/{name} is not the one generated for this build")

    # The binaries: what they need, against the dependencies.
    with tempfile.TemporaryDirectory() as scratch:
        for _, installed in EXECUTABLES:
            entry = files.get(installed)
            if not entry or entry[1] is None:
                continue
            copy = pathlib.Path(scratch) / pathlib.PurePosixPath(installed).name
            copy.write_bytes(entry[1])
            facts = binary_facts(sdk.root / "bin" / "llvm-readelf", copy)
            unknown = sorted(set(facts["needed"]) - set(ALLOWED_NEEDED))
            if unknown or facts["runpath"]:
                problems.append(f"{installed}: needs {unknown} or has a run path")
            floor = re.search(r"libc6 \(>= ([0-9.]+)\)", control_facts["depends"])
            if facts["glibc"] and (floor is None or facts["glibc"] > tuple(int(p) for p in floor[1].split("."))):
                problems.append(f"{installed}: imports GLIBC_{'.'.join(map(str, facts['glibc']))}, "
                                "above the libc6 dependency")
            if "libcuda.so.1" in facts["needed"] and "libcuda.so.1" not in control_facts["depends"]:
                problems.append(f"{installed}: needs libcuda.so.1, which the package does not depend on")

    # The SBOM and notices, against the receipt.
    sbom = json.loads((doc / "jitllm.spdx.json").read_text())
    listed = {p["name"]: p for p in sbom["packages"]}
    root = listed.get(PACKAGE, {})
    if root.get("versionInfo") != receipt["version"]["product"]:
        problems.append(f"SBOM: jitllm is {root.get('versionInfo')!r}, not {receipt['version']['product']}")
    notices = (doc / "THIRD-PARTY-NOTICES").read_text()
    lock = srclib.load_lock(srclib.LOCK, modules=receipt["modules"])
    for component in receipt["components"]:
        entry = listed.get(component["id"])
        if component["use"] != "product":
            if entry:
                problems.append(f"SBOM: {component['id']} is test-only and never ships")
            continue
        if not entry or entry["versionInfo"] != component["version"] or \
                entry.get("checksums", [{}])[0].get("checksumValue") != component["archive_sha256"]:
            problems.append(f"SBOM: {component['id']} {component['version']} is missing or differs from the receipt")
        for notice in lock["components"][component["id"]]["license"]["notices"]:
            text = component_notice(pathlib.Path(component["source"]), notice)
            if text.strip() not in notices:
                problems.append(f"notices: {component['id']}'s {notice} is missing")
    provenance = tomllib.loads(PROVENANCE.read_text())
    for name, unit in shipped_units(provenance, bool(receipt["cuda"])):
        if name not in listed:
            problems.append(f"SBOM: the shipped platform unit {name} is missing")
        for notice in unit["notices"]:
            if extract(provenance["notices"][notice]["extract"], sdk.root).strip() not in notices:
                problems.append(f"notices: {notice} ({name}) is missing")
    return problems
