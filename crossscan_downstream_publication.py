"""Package and anonymously verify an immutable Cross-scan downstream tree.

The scientific result may be either PASS or FAIL.  A package PASS means only
that the exact sealed tree, its terminal decision, the preregistered locks,
and the already-public model release were verified and bound together.

This helper never uploads or changes GitHub state.  ``pack`` creates a
byte-stable split PAX tar, ``verify`` revalidates it, and ``verify-public``
accepts a downstream GitHub Release only after logged-out metadata and byte
readback followed by full downstream verification.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

import crossscan_release_publication as M
import crossscan_scrollfiesta_adapter as A
import run_crossscan_scrollfiesta_downstream as D


PACKAGE_SCHEMA = "crossscan-scrollfiesta-downstream-package-v1"
PUBLICATION_SCHEMA = "crossscan-scrollfiesta-downstream-publication-v1"
EXPECTED_REPOSITORY = M.EXPECTED_REPOSITORY
ARCHIVE_ROOT = "crossscan_scrollfiesta_v4_downstream"
PACKAGE_ASSET_NAME = "downstream_package_receipt.json"
PUBLICATION_RECEIPT_NAME = "downstream_publication_receipt.json"
TERMINAL_NAME = "terminal_receipt.json"
MAX_PART_BYTES = M.MAX_PART_BYTES
MAX_ARCHIVE_PARTS = 999  # GitHub permits 1000 assets; reserve one for the receipt.
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
TAG_RE = re.compile(r"^crossscan-v4-downstream-[0-9a-f]{12}$")
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
NETWORK_POLICY = "dedicated-no-proxy-no-auth-no-cookie-strict-https-redirect-opener"
SOURCE_CANONICALIZATION = "crlf-to-lf-only"

# The already-public model receipt was produced from this immutable source blob
# on a CRLF checkout.  Keep both the Git-object identity and the canonical LF
# source digest: the former names the public source exactly, while the latter
# lets an LF or CRLF checkout reproduce the same identity without trusting an
# arbitrary script/runtime tuple.
UPSTREAM_PUBLICATION_SOURCE_IDENTITY = {
    "path": "crossscan_release_publication.py",
    "canonicalization": SOURCE_CANONICALIZATION,
    "bytes_lf": 54661,
    "sha256_lf": "91fc3bd701e87857429949ed5e0723ae9ff868d3128c0d28aa9d81472c8fdf36",
    "git_blob_sha1": "fb63d44c82f3d5f67412768567d582ef9b30afe8",
}
UPSTREAM_ALLOWED_HISTORICAL_VERIFIERS = (
    {
        "script_sha256": "da114048ef8db2c7069498ec5cea0e6cf460fdd7fce71b0c81b2dd6985451fb2",
        "python_version": (
            "3.14.6 | packaged by conda-forge | "
            "(main, Jul 24 2026, 16:16:13) [MSC v.1944 64 bit (AMD64)]"
        ),
        "network_policy": NETWORK_POLICY,
    },
)


canonical_json = M.canonical_json
content_hash = M.content_hash
strict_json_loads = M.strict_json_loads
sha256_file = M.sha256_file
sha256_stream = M.sha256_stream
_resolve_tag = M._resolve_tag
_anonymous_json = M._anonymous_json
_download_anonymous = M._download_anonymous


def _canonical_lf_source(payload: bytes) -> bytes:
    """Return the sole permitted cross-platform representation of source."""
    if not isinstance(payload, bytes) or b"\x00" in payload:
        raise ValueError("verifier source must be non-NUL bytes")
    canonical = payload.replace(b"\r\n", b"\n")
    if b"\r" in canonical:
        raise ValueError("verifier source contains a bare carriage return")
    return canonical


def _source_identity(payload: bytes, path: str) -> dict[str, Any]:
    canonical = _canonical_lf_source(payload)
    git_header = f"blob {len(canonical)}\0".encode("ascii")
    return {
        "path": path,
        "canonicalization": SOURCE_CANONICALIZATION,
        "bytes_lf": len(canonical),
        "sha256_lf": hashlib.sha256(canonical).hexdigest(),
        "git_blob_sha1": hashlib.sha1(git_header + canonical).hexdigest(),
    }


def _local_source_identity(path: Path, logical_name: str) -> dict[str, Any]:
    path = Path(path)
    if _is_linklike(path) or not path.is_file():
        raise ValueError(f"verifier source is not a regular non-link file: {path}")
    return _source_identity(path.read_bytes(), logical_name)


def publication_verifier_identity() -> dict[str, Any]:
    """Portable identity recorded by newly produced downstream receipts."""
    return {
        "source": _local_source_identity(
            Path(__file__).resolve(), "crossscan_downstream_publication.py"
        ),
        "python_version": sys.version,
        "network_policy": NETWORK_POLICY,
    }


def _upstream_validation_surrogate(
    publication: dict[str, Any],
    package: dict[str, Any],
    package_payload: bytes,
) -> dict[str, Any]:
    """Validate the exact historical verifier, then reuse the model validator.

    The model validator originally compared receipt provenance to the *current*
    checkout bytes and runtime.  Its scientific/asset checks remain useful, so
    a copy with current local provenance is passed through only after the real
    receipt has been content-hash checked and matched to the one recorded
    historical verifier/runtime and immutable Git blob.
    """
    if (
        not isinstance(publication, dict)
        or publication.get("content_sha256") != content_hash(publication)
    ):
        raise ValueError("upstream publication receipt content hash mismatch")
    if publication.get("verifier") not in UPSTREAM_ALLOWED_HISTORICAL_VERIFIERS:
        raise ValueError("upstream publication used an unapproved verifier/runtime")
    actual_source = _local_source_identity(
        Path(M.__file__).resolve(), "crossscan_release_publication.py"
    )
    if actual_source != UPSTREAM_PUBLICATION_SOURCE_IDENTITY:
        raise ValueError("local upstream verifier source differs from its pinned Git blob")
    surrogate = dict(publication)
    surrogate["verifier"] = {
        "script_sha256": sha256_file(Path(M.__file__).resolve()),
        "python_version": sys.version,
        "network_policy": NETWORK_POLICY,
    }
    surrogate["content_sha256"] = content_hash(surrogate)
    M.validate_publication_receipt(surrogate, package, package_payload)
    return surrogate


def _is_linklike(path: Path) -> bool:
    try:
        return path.is_symlink() or M._is_reparse(path)
    except OSError:
        return True


def _safe_relative(value: str) -> str:
    if not isinstance(value, str) or "\\" in value or "\x00" in value:
        raise ValueError(f"unsafe downstream path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"unsafe downstream path: {value!r}")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValueError(f"downstream path is not portable printable ASCII: {value!r}")
    if any(character in '<>:"|?*' for character in value):
        raise ValueError(f"downstream path contains a non-portable character: {value!r}")
    for part in path.parts:
        if part.endswith((".", " ")) or part.split(".", 1)[0].upper() in WINDOWS_RESERVED:
            raise ValueError(f"downstream path is not Windows-portable: {value!r}")
    return value


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _load_hashed_json(path: Path, schema: str) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    if _is_linklike(path) or not path.is_file():
        raise ValueError(f"JSON receipt must be a regular non-link file: {path}")
    payload = path.read_bytes()
    value = strict_json_loads(payload)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != schema
        or value.get("content_sha256") != content_hash(value)
    ):
        raise ValueError(f"invalid {schema} receipt: {path}")
    return value, payload


def _inventory_tree(root: Path) -> tuple[Path, dict[str, Path]]:
    raw_root = Path(root)
    if _is_linklike(raw_root) or not raw_root.is_dir():
        raise ValueError(f"downstream root must be a real directory: {raw_root}")
    root = raw_root.resolve()
    files: dict[str, Path] = {}
    folded: set[str] = set()

    def walk_error(error: OSError) -> None:
        raise error

    for current, child_directories, child_files in os.walk(
        root, topdown=True, followlinks=False, onerror=walk_error
    ):
        current_path = Path(current)
        child_directories.sort()
        child_files.sort()
        for name in child_directories:
            path = current_path / name
            relative = _safe_relative(path.relative_to(root).as_posix())
            if _is_linklike(path) or not path.is_dir():
                raise ValueError(f"linked or special downstream directory is forbidden: {relative}")
            if relative.casefold() in folded:
                raise ValueError(f"duplicate or case-alias downstream path: {relative}")
            folded.add(relative.casefold())
        for name in child_files:
            path = current_path / name
            relative = _safe_relative(path.relative_to(root).as_posix())
            try:
                metadata = path.lstat()
            except OSError as error:
                raise ValueError(f"unreadable downstream file: {relative}") from error
            if (
                _is_linklike(path)
                or not stat.S_ISREG(metadata.st_mode)
                or not path.is_file()
            ):
                raise ValueError(f"linked or special downstream file is forbidden: {relative}")
            if metadata.st_nlink != 1:
                raise ValueError(f"hard-linked downstream file is forbidden: {relative}")
            if relative.casefold() in folded:
                raise ValueError(f"duplicate or case-alias downstream path: {relative}")
            folded.add(relative.casefold())
            files[relative] = path
    return root, dict(sorted(files.items()))


def _validate_terminal_file_records(
    terminal: dict[str, Any], actual: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    records = terminal.get("files")
    if not isinstance(records, list):
        raise ValueError("terminal receipt files must be a list")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise ValueError("terminal receipt has an invalid file record schema")
        relative = _safe_relative(record.get("path"))
        if (
            relative == TERMINAL_NAME
            or relative.casefold() in seen
            or type(record.get("bytes")) is not int
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or not HEX64_RE.fullmatch(record["sha256"])
        ):
            raise ValueError(f"terminal receipt has an invalid file record: {relative}")
        seen.add(relative.casefold())
        validated.append(record)
    if [record["path"] for record in validated] != sorted(record["path"] for record in validated):
        raise ValueError("terminal receipt file records are not ordinal-sorted")
    if {record["path"] for record in validated} != set(actual):
        raise ValueError("terminal receipt does not bind the exact regular-file universe")
    for record in validated:
        if record != actual[record["path"]]:
            raise ValueError(f"terminal receipt file record mismatch: {record['path']}")
    return validated


def validate_downstream_tree(root: Path) -> dict[str, Any]:
    resolved, files = _inventory_tree(root)
    if TERMINAL_NAME not in files:
        raise ValueError("downstream tree lacks terminal_receipt.json")
    terminal_payload = files[TERMINAL_NAME].read_bytes()
    terminal = strict_json_loads(terminal_payload)
    if not isinstance(terminal, dict):
        raise ValueError("downstream terminal receipt must be a JSON object")
    if terminal.get("content_sha256") != A.content_hash(terminal):
        raise ValueError("downstream terminal receipt content hash mismatch")
    verified = D.verify_result(
        resolved,
        deep=True,
        expected_content_sha256=terminal["content_sha256"],
    )
    if verified != terminal:
        raise ValueError("strict terminal receipt differs from downstream verification")
    downstream_lock = resolved / "provenance" / "crossscan_scrollfiesta_downstream_lock.json"
    metric_lock = resolved / "provenance" / "crossscan_scrollfiesta_metric_lock.json"
    if (
        "provenance/crossscan_scrollfiesta_downstream_lock.json" not in files
        or "provenance/crossscan_scrollfiesta_metric_lock.json" not in files
    ):
        raise ValueError("downstream tree lacks its preregistered lock provenance")
    A.validate_downstream_lock(downstream_lock)
    D.validate_metric_lock(metric_lock)
    actual = {
        relative: _file_record(path, relative)
        for relative, path in files.items()
        if relative != TERMINAL_NAME
    }
    records = _validate_terminal_file_records(terminal, actual)
    if terminal.get("status") not in ("PASS", "FAIL"):
        raise ValueError("downstream terminal status must be PASS or FAIL")
    return {
        "root": resolved,
        "files": files,
        "terminal": terminal,
        "terminal_payload": terminal_payload,
        "terminal_record": {
            **_file_record(files[TERMINAL_NAME], TERMINAL_NAME),
            "content_sha256": terminal["content_sha256"],
        },
        "file_records": records,
    }


def validate_upstream_model_publication(
    package_receipt_path: Path,
    publication_receipt_path: Path,
    *,
    live: bool = True,
) -> dict[str, Any]:
    package_path = Path(package_receipt_path)
    publication_path = Path(publication_receipt_path)
    if package_path.name != "package_receipt.json":
        raise ValueError("upstream model package receipt must use package_receipt.json")
    if publication_path.name != "publication_receipt.json":
        raise ValueError("upstream model publication receipt must use publication_receipt.json")
    package, package_payload = _load_hashed_json(package_path, M.PACKAGE_SCHEMA)
    publication, publication_payload = _load_hashed_json(publication_path, M.PUBLICATION_SCHEMA)
    M._validate_package_receipt(package)
    publication_surrogate = _upstream_validation_surrogate(
        publication, package, package_payload
    )
    if live:
        M.probe_publication(package, publication_surrogate, package_payload)
    expected_tag = f"crossscan-v4-{package['release']['manifest']['content_sha256'][:12]}"
    if (
        publication.get("status") != "PUBLIC_LOGGED_OUT_VERIFIED"
        or publication.get("repository") != EXPECTED_REPOSITORY
        or publication.get("tag") != expected_tag
        or publication.get("code_head") != package.get("code_head")
        or publication.get("release_manifest_content_sha256")
        != package["release"]["manifest"]["content_sha256"]
        or publication.get("package_receipt", {}).get("content_sha256")
        != package["content_sha256"]
    ):
        raise ValueError("upstream model publication does not bind the expected immutable release")
    return {
        "repository": EXPECTED_REPOSITORY,
        "tag": expected_tag,
        "code_head": package["code_head"],
        "release_manifest_content_sha256": package["release"]["manifest"]["content_sha256"],
        "release_manifest": dict(package["release"]["manifest"]),
        "release_package_content_sha256": package["content_sha256"],
        "release_archive": {
            "bytes": package["archive"]["bytes"],
            "sha256": package["archive"]["sha256"],
        },
        "package_receipt": {
            "path": package_path.name,
            "bytes": len(package_payload),
            "sha256": hashlib.sha256(package_payload).hexdigest(),
            "content_sha256": package["content_sha256"],
            "payload_base64": base64.b64encode(package_payload).decode("ascii"),
        },
        "publication_receipt": {
            "path": publication_path.name,
            "bytes": len(publication_payload),
            "sha256": hashlib.sha256(publication_payload).hexdigest(),
            "content_sha256": publication["content_sha256"],
            "payload_base64": base64.b64encode(publication_payload).decode("ascii"),
        },
    }


def _validate_upstream_binding(value: Any) -> dict[str, Any]:
    required = {
        "repository", "tag", "code_head", "release_manifest_content_sha256",
        "release_manifest", "release_package_content_sha256", "release_archive",
        "package_receipt", "publication_receipt",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("downstream package has an invalid upstream model binding")
    expected_tag = f"crossscan-v4-{str(value.get('release_manifest_content_sha256', ''))[:12]}"
    if (
        value.get("repository") != EXPECTED_REPOSITORY
        or value.get("tag") != expected_tag
        or not HEX40_RE.fullmatch(str(value.get("code_head", "")))
        or not HEX64_RE.fullmatch(str(value.get("release_manifest_content_sha256", "")))
        or not HEX64_RE.fullmatch(str(value.get("release_package_content_sha256", "")))
    ):
        raise ValueError("downstream package upstream model identity is invalid")
    archive = value.get("release_archive")
    if (
        not isinstance(archive, dict)
        or set(archive) != {"bytes", "sha256"}
        or type(archive.get("bytes")) is not int
        or archive["bytes"] <= 0
        or not HEX64_RE.fullmatch(str(archive.get("sha256", "")))
    ):
        raise ValueError("downstream package upstream model archive is invalid")
    manifest = value.get("release_manifest")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"path", "bytes", "sha256", "content_sha256"}
        or manifest.get("path") != "release_manifest.json"
        or type(manifest.get("bytes")) is not int
        or manifest["bytes"] <= 0
        or not HEX64_RE.fullmatch(str(manifest.get("sha256", "")))
        or manifest.get("content_sha256") != value["release_manifest_content_sha256"]
    ):
        raise ValueError("downstream package upstream release manifest is invalid")
    for name, expected_path, expected_content in (
        ("package_receipt", "package_receipt.json", value["release_package_content_sha256"]),
        ("publication_receipt", "publication_receipt.json", None),
    ):
        record = value.get(name)
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256", "content_sha256", "payload_base64"}
            or record.get("path") != expected_path
            or type(record.get("bytes")) is not int
            or record["bytes"] <= 0
            or not HEX64_RE.fullmatch(str(record.get("sha256", "")))
            or not HEX64_RE.fullmatch(str(record.get("content_sha256", "")))
            or (expected_content is not None and record["content_sha256"] != expected_content)
        ):
            raise ValueError(f"downstream package upstream {name} is invalid")
        try:
            payload = base64.b64decode(record["payload_base64"], validate=True)
        except (ValueError, TypeError) as error:
            raise ValueError(f"downstream package upstream {name} payload is invalid") from error
        if (
            len(payload) != record["bytes"]
            or hashlib.sha256(payload).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"downstream package upstream {name} payload hash mismatch")
    package_payload = base64.b64decode(value["package_receipt"]["payload_base64"], validate=True)
    publication_payload = base64.b64decode(
        value["publication_receipt"]["payload_base64"], validate=True
    )
    embedded_package = strict_json_loads(package_payload)
    embedded_publication = strict_json_loads(publication_payload)
    if (
        not isinstance(embedded_package, dict)
        or embedded_package.get("content_sha256") != value["release_package_content_sha256"]
        or not isinstance(embedded_publication, dict)
        or embedded_publication.get("content_sha256")
        != value["publication_receipt"]["content_sha256"]
    ):
        raise ValueError("embedded upstream receipt identities differ from their binding")
    M._validate_package_receipt(embedded_package)
    _upstream_validation_surrogate(
        embedded_publication, embedded_package, package_payload
    )
    embedded_manifest = embedded_package["release"]["manifest"]
    if (
        value["repository"] != embedded_publication.get("repository")
        or value["tag"] != embedded_publication.get("tag")
        or value["code_head"] != embedded_package.get("code_head")
        or value["code_head"] != embedded_publication.get("code_head")
        or value["release_manifest"] != embedded_manifest
        or value["release_manifest_content_sha256"]
        != embedded_manifest.get("content_sha256")
        or value["release_manifest_content_sha256"]
        != embedded_publication.get("release_manifest_content_sha256")
        or value["release_package_content_sha256"]
        != embedded_package.get("content_sha256")
        or value["release_archive"] != {
            "bytes": embedded_package["archive"]["bytes"],
            "sha256": embedded_package["archive"]["sha256"],
        }
        or value["package_receipt"]["content_sha256"]
        != embedded_package.get("content_sha256")
        or value["publication_receipt"]["content_sha256"]
        != embedded_publication.get("content_sha256")
    ):
        raise ValueError("upstream model summary differs from its embedded receipts")
    return value


def _embedded_upstream_receipts(
    binding: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    package_payload = base64.b64decode(
        binding["package_receipt"]["payload_base64"], validate=True
    )
    publication_payload = base64.b64decode(
        binding["publication_receipt"]["payload_base64"], validate=True
    )
    package = strict_json_loads(package_payload)
    publication = strict_json_loads(publication_payload)
    if not isinstance(package, dict) or not isinstance(publication, dict):
        raise ValueError("embedded upstream receipts are not JSON objects")
    return package, publication, package_payload


def _probe_embedded_upstream(binding: dict[str, Any]) -> None:
    package, publication, package_payload = _embedded_upstream_receipts(binding)
    surrogate = _upstream_validation_surrogate(
        publication, package, package_payload
    )
    M.probe_publication(package, surrogate, package_payload)


def _validate_downstream_model_link(tree: dict[str, Any], upstream: dict[str, Any]) -> None:
    expected = upstream["release_manifest"]
    for arm in ("candidate-fixed", "candidate-matched-mass"):
        relative = f"inputs/grids/{arm}/provenance/release_manifest.json"
        path = tree["files"].get(relative)
        if path is None:
            raise ValueError(f"downstream candidate grid lacks published release manifest: {arm}")
        record = _file_record(path, "release_manifest.json")
        if record != {
            key: expected[key] for key in ("path", "bytes", "sha256")
        }:
            raise ValueError(f"downstream candidate grid uses a different model release: {arm}")
        embedded = strict_json_loads(path.read_bytes())
        if (
            not isinstance(embedded, dict)
            or embedded.get("content_sha256") != upstream["release_manifest_content_sha256"]
            or embedded.get("content_sha256") != content_hash(embedded)
        ):
            raise ValueError(f"downstream candidate release manifest content differs: {arm}")


def _file_tar_info(relative: str, path: Path) -> tarfile.TarInfo:
    info = tarfile.TarInfo(f"{ARCHIVE_ROOT}/{relative}")
    info.size = path.stat().st_size
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.type = tarfile.REGTYPE
    return info


def expected_tag(package: dict[str, Any]) -> str:
    terminal = package.get("downstream", {}).get("terminal_receipt", {}).get("content_sha256")
    if not isinstance(terminal, str) or not HEX64_RE.fullmatch(terminal):
        raise ValueError("package lacks a valid terminal content digest")
    return f"crossscan-v4-downstream-{terminal[:12]}"


def expected_confirmation(package: dict[str, Any]) -> str:
    status = package.get("downstream", {}).get("scientific_status")
    digest = package.get("downstream", {}).get("terminal_receipt", {}).get("content_sha256")
    code_head = package.get("code_head")
    if (
        status not in ("PASS", "FAIL")
        or not isinstance(digest, str)
        or not HEX64_RE.fullmatch(digest)
        or not isinstance(code_head, str)
        or not HEX40_RE.fullmatch(code_head)
    ):
        raise ValueError("package lacks a valid status, terminal digest, or code head")
    return (
        f"PUBLISH IMMUTABLE CROSSSCAN V4 DOWNSTREAM {status} {digest} "
        f"AT {code_head}"
    )


def _require_external_pins(
    package: dict[str, Any],
    expected_terminal_content_sha256: str,
    expected_upstream_publication_content_sha256: str,
) -> None:
    if not isinstance(expected_terminal_content_sha256, str) or not HEX64_RE.fullmatch(
        expected_terminal_content_sha256
    ):
        raise ValueError("expected terminal content SHA-256 must be 64 lowercase hex characters")
    if not isinstance(expected_upstream_publication_content_sha256, str) or not HEX64_RE.fullmatch(
        expected_upstream_publication_content_sha256
    ):
        raise ValueError(
            "expected upstream publication content SHA-256 must be 64 lowercase hex characters"
        )
    if (
        package["downstream"]["terminal_receipt"]["content_sha256"]
        != expected_terminal_content_sha256
    ):
        raise ValueError("downstream package differs from independently expected terminal digest")
    if (
        package["upstream_model_publication"]["publication_receipt"]["content_sha256"]
        != expected_upstream_publication_content_sha256
    ):
        raise ValueError("downstream package differs from independently expected upstream publication")


def _validate_package_receipt(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "status", "created_utc", "code_head", "publication_tag",
        "upstream_model_publication", "downstream", "archive", "content_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != PACKAGE_SCHEMA
        or value.get("status") != "PASS"
        or not isinstance(value.get("created_utc"), str)
        or not HEX40_RE.fullmatch(str(value.get("code_head", "")))
        or value.get("content_sha256") != content_hash(value)
    ):
        raise ValueError("invalid Cross-scan downstream package receipt")
    _validate_upstream_binding(value.get("upstream_model_publication"))
    downstream = value.get("downstream")
    downstream_keys = {
        "root_name", "scientific_status", "claim_boundary", "file_count",
        "payload_bytes", "terminal_receipt",
        "downstream_lock_content_sha256", "metric_lock_content_sha256", "verification",
    }
    if not isinstance(downstream, dict) or set(downstream) != downstream_keys:
        raise ValueError("downstream package has an invalid tree record")
    scientific_status = downstream.get("scientific_status")
    expected_claim = (
        "bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement"
        if scientific_status == "PASS"
        else "bounded negative result; no downstream-improvement claim is authorized"
    )
    if (
        downstream.get("root_name") != ARCHIVE_ROOT
        or scientific_status not in ("PASS", "FAIL")
        or downstream.get("claim_boundary") != expected_claim
        or type(downstream.get("file_count")) is not int
        or downstream["file_count"] < 1
        or type(downstream.get("payload_bytes")) is not int
        or downstream["payload_bytes"] <= 0
        or downstream.get("downstream_lock_content_sha256") != A.DOWNSTREAM_LOCK_CONTENT_SHA256
        or downstream.get("metric_lock_content_sha256") != D.METRIC_LOCK_CONTENT_SHA256
    ):
        raise ValueError("downstream package tree identity is invalid")
    verification = downstream.get("verification")
    if verification != {
        "deep_requested": True,
        "pass_recomputed": scientific_status == "PASS",
        "sealed_fail_integrity_verified": scientific_status == "FAIL",
    }:
        raise ValueError("downstream package verification scope is invalid")
    terminal = downstream.get("terminal_receipt")
    if (
        not isinstance(terminal, dict)
        or set(terminal) != {"path", "bytes", "sha256", "content_sha256"}
        or terminal.get("path") != TERMINAL_NAME
        or type(terminal.get("bytes")) is not int
        or terminal["bytes"] <= 0
        or not HEX64_RE.fullmatch(str(terminal.get("sha256", "")))
        or not HEX64_RE.fullmatch(str(terminal.get("content_sha256", "")))
        or value.get("publication_tag") != expected_tag(value)
    ):
        raise ValueError("downstream package terminal receipt identity is invalid")
    archive = value.get("archive")
    metadata = {
        "file_mode": "0644",
        "mtime": 0,
        "uid": 0,
        "gid": 0,
        "uname": "",
        "gname": "",
        "member_order": "ordinal-posix-path",
    }
    if (
        not isinstance(archive, dict)
        or archive.get("format") != "uncompressed-pax-tar"
        or archive.get("root_name") != ARCHIVE_ROOT
        or PurePosixPath(str(archive.get("path", ""))).name != archive.get("path")
        or not str(archive.get("path", "")).endswith(".tar")
        or not HEX64_RE.fullmatch(str(archive.get("sha256", "")))
        or type(archive.get("bytes")) is not int
        or archive["bytes"] <= 0
        or archive.get("member_count")
        != downstream["file_count"]
        or archive.get("metadata") != metadata
    ):
        raise ValueError("downstream package archive identity is invalid")
    parts = archive.get("parts")
    part_limit = archive.get("part_size_limit_bytes")
    if (
        not isinstance(parts, list)
        or not parts
        or len(parts) > MAX_ARCHIVE_PARTS
        or type(part_limit) is not int
        or not 1 <= part_limit <= MAX_PART_BYTES
    ):
        raise ValueError("downstream package archive parts are invalid")
    offset = 0
    for index, part in enumerate(parts):
        expected_name = f"{archive['path']}.part-{index:03d}"
        if (
            not isinstance(part, dict)
            or set(part) != {"path", "offset", "bytes", "sha256"}
            or part.get("path") != expected_name
            or part.get("offset") != offset
            or type(part.get("bytes")) is not int
            or not 1 <= part["bytes"] <= part_limit
            or not HEX64_RE.fullmatch(str(part.get("sha256", "")))
        ):
            raise ValueError(f"downstream package archive part {index} is invalid")
        if index < len(parts) - 1 and part["bytes"] != part_limit:
            raise ValueError("only the final downstream archive part may be short")
        offset += part["bytes"]
    if offset != archive["bytes"]:
        raise ValueError("downstream archive parts do not reconstruct its byte length")
    return value


def _same_upstream_binding(
    package: dict[str, Any],
    model_package_receipt: Path | None,
    model_publication_receipt: Path | None,
    *,
    live: bool,
) -> None:
    if model_package_receipt is None and model_publication_receipt is None:
        return
    if model_package_receipt is None or model_publication_receipt is None:
        raise ValueError("both upstream model receipts are required together")
    actual = validate_upstream_model_publication(
        model_package_receipt, model_publication_receipt, live=live
    )
    if actual != package.get("upstream_model_publication"):
        raise ValueError("upstream model receipts differ from the downstream package binding")


def pack_downstream(
    downstream_dir: Path,
    model_package_receipt: Path,
    model_publication_receipt: Path,
    archive: Path,
    receipt_path: Path,
    *,
    code_head: str,
    expected_terminal_content_sha256: str,
    expected_upstream_publication_content_sha256: str,
    part_size_bytes: int = MAX_PART_BYTES,
    probe_upstream: bool = True,
) -> dict[str, Any]:
    if not isinstance(code_head, str) or not HEX40_RE.fullmatch(code_head):
        raise ValueError("code head must be exactly 40 lowercase hexadecimal characters")
    if not isinstance(expected_terminal_content_sha256, str) or not HEX64_RE.fullmatch(
        expected_terminal_content_sha256
    ):
        raise ValueError("expected terminal content SHA-256 must be 64 lowercase hex characters")
    if not isinstance(expected_upstream_publication_content_sha256, str) or not HEX64_RE.fullmatch(
        expected_upstream_publication_content_sha256
    ):
        raise ValueError(
            "expected upstream publication content SHA-256 must be 64 lowercase hex characters"
        )
    if not 1 <= part_size_bytes <= MAX_PART_BYTES:
        raise ValueError(f"archive part size must be in [1,{MAX_PART_BYTES}]")
    tree = validate_downstream_tree(downstream_dir)
    if tree["terminal"]["content_sha256"] != expected_terminal_content_sha256:
        raise ValueError("downstream tree differs from independently expected terminal digest")
    upstream = validate_upstream_model_publication(
        model_package_receipt, model_publication_receipt, live=False
    )
    if (
        upstream["publication_receipt"]["content_sha256"]
        != expected_upstream_publication_content_sha256
    ):
        raise ValueError("upstream model differs from independently expected publication receipt")
    _validate_downstream_model_link(tree, upstream)
    if probe_upstream:
        _probe_embedded_upstream(upstream)
    archive = Path(archive).resolve()
    receipt_path = Path(receipt_path).resolve()
    staging = archive.with_name(archive.name + ".tmp")
    tree_root = tree["root"]
    if (
        archive == receipt_path
        or staging == receipt_path
        or archive.is_relative_to(tree_root)
        or staging.is_relative_to(tree_root)
        or receipt_path.is_relative_to(tree_root)
    ):
        raise ValueError("archive, staging, and receipt must be distinct and outside downstream tree")
    for path in (archive, staging, receipt_path):
        if path.exists() or _is_linklike(path):
            raise FileExistsError(f"refusing to replace downstream publication output: {path}")
    if list(archive.parent.glob(f"{archive.name}.part-*")):
        raise FileExistsError("downstream archive part or staging output already exists")
    archive.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(tree["files"])
    with tarfile.open(staging, mode="x", format=tarfile.PAX_FORMAT, dereference=False) as bundle:
        for relative in entries:
            path = tree["files"][relative]
            with path.open("rb") as stream:
                bundle.addfile(_file_tar_info(relative, path), stream)
    archive_sha256 = sha256_file(staging)
    archive_bytes = staging.stat().st_size
    expected_parts = (archive_bytes + part_size_bytes - 1) // part_size_bytes
    if expected_parts > MAX_ARCHIVE_PARTS:
        raise ValueError(
            f"downstream archive needs {expected_parts} parts; maximum is {MAX_ARCHIVE_PARTS}"
        )
    parts = M._split_archive(staging, archive, part_size_bytes)
    terminal = tree["terminal"]
    downstream = {
        "root_name": ARCHIVE_ROOT,
        "scientific_status": terminal["status"],
        "claim_boundary": terminal["claim_boundary"],
        "file_count": len(tree["files"]),
        "payload_bytes": sum(path.stat().st_size for path in tree["files"].values()),
        "terminal_receipt": tree["terminal_record"],
        "downstream_lock_content_sha256": terminal["downstream_lock_content_sha256"],
        "metric_lock_content_sha256": terminal["metric_lock_content_sha256"],
        "verification": {
            "deep_requested": True,
            "pass_recomputed": terminal["status"] == "PASS",
            "sealed_fail_integrity_verified": terminal["status"] == "FAIL",
        },
    }
    package = {
        "schema_version": PACKAGE_SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_head": code_head,
        "publication_tag": f"crossscan-v4-downstream-{terminal['content_sha256'][:12]}",
        "upstream_model_publication": upstream,
        "downstream": downstream,
        "archive": {
            "path": archive.name,
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "format": "uncompressed-pax-tar",
            "root_name": ARCHIVE_ROOT,
            "member_count": len(entries),
            "part_size_limit_bytes": part_size_bytes,
            "parts": parts,
            "metadata": {
                "file_mode": "0644",
                "mtime": 0,
                "uid": 0,
                "gid": 0,
                "uname": "",
                "gname": "",
                "member_order": "ordinal-posix-path",
            },
        },
    }
    package["content_sha256"] = content_hash(package)
    _validate_package_receipt(package)
    _require_external_pins(
        package,
        expected_terminal_content_sha256,
        expected_upstream_publication_content_sha256,
    )
    verify_archive(
        archive,
        package,
        downstream_dir=downstream_dir,
        full_archive=staging,
        deep=True,
        probe_upstream=False,
        expected_terminal_content_sha256=expected_terminal_content_sha256,
        expected_upstream_publication_content_sha256=(
            expected_upstream_publication_content_sha256
        ),
    )
    staging.unlink()
    verify_archive(
        archive,
        package,
        downstream_dir=downstream_dir,
        deep=True,
        probe_upstream=False,
        expected_terminal_content_sha256=expected_terminal_content_sha256,
        expected_upstream_publication_content_sha256=(
            expected_upstream_publication_content_sha256
        ),
    )
    _write_json_exclusive(receipt_path, package)
    return package


def _validate_member_metadata(member: tarfile.TarInfo) -> None:
    if (
        member.type != tarfile.REGTYPE
        or member.mode != 0o644
        or member.mtime != 0
        or member.uid != 0
        or member.gid != 0
        or member.uname != ""
        or member.gname != ""
        or member.linkname != ""
        or member.devmajor != 0
        or member.devminor != 0
        or getattr(member, "sparse", None) is not None
        or set(member.pax_headers) - {"path"}
        or ("path" in member.pax_headers and member.pax_headers["path"] != member.name)
    ):
        raise ValueError(f"archive member metadata is not deterministic: {member.name}")


def _read_member(
    bundle: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path | None = None,
) -> tuple[bytes | None, str, int]:
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"archive member cannot be read: {member.name}")
    digest = hashlib.sha256()
    total = 0
    blocks: list[bytes] | None = [] if destination is None and member.name.endswith(TERMINAL_NAME) else None
    target: BinaryIO | None = None
    try:
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            target = destination.open("xb")
        while True:
            block = source.read(8 * 1024 * 1024)
            if not block:
                break
            total += len(block)
            digest.update(block)
            if target is not None:
                target.write(block)
            if blocks is not None:
                blocks.append(block)
    finally:
        source.close()
        if target is not None:
            target.close()
    if total != member.size:
        raise ValueError(f"short read from archive member: {member.name}")
    return (b"".join(blocks) if blocks is not None else None), digest.hexdigest(), total


def _inspect_archive(
    source: BinaryIO,
    package: dict[str, Any],
    extraction_root: Path | None,
) -> dict[str, Any]:
    expected_root = f"{ARCHIVE_ROOT}/"
    with tarfile.open(fileobj=source, mode="r:") as bundle:
        members = bundle.getmembers()
        member_map: dict[str, tarfile.TarInfo] = {}
        order: list[str] = []
        folded: set[str] = set()
        for member in members:
            if not member.name.startswith(expected_root):
                raise ValueError(f"archive member escapes its single root: {member.name}")
            raw_relative = member.name[len(expected_root):]
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"archive contains a non-regular member: {member.name}")
            relative = _safe_relative(raw_relative)
            if relative.casefold() in folded:
                raise ValueError(f"archive contains a duplicate/case-alias member: {relative}")
            _validate_member_metadata(member)
            folded.add(relative.casefold())
            member_map[relative] = member
            order.append(relative)
        if order != sorted(order):
            raise ValueError("archive member order is not deterministic")
        expected = package["downstream"]
        if (
            len(members) != package["archive"]["member_count"]
            or TERMINAL_NAME not in member_map
        ):
            raise ValueError("archive member or directory universe differs from package receipt")
        regular_paths = [relative for relative, member in member_map.items() if member.isfile()]
        if len(regular_paths) != expected["file_count"]:
            raise ValueError("archive regular-file count differs from package receipt")
        terminal_destination = extraction_root / TERMINAL_NAME if extraction_root is not None else None
        terminal_payload, terminal_sha, terminal_bytes = _read_member(
            bundle, member_map[TERMINAL_NAME], terminal_destination
        )
        if terminal_payload is None:
            terminal_payload = terminal_destination.read_bytes()
        terminal_record = expected["terminal_receipt"]
        if (
            terminal_bytes != terminal_record["bytes"]
            or terminal_sha != terminal_record["sha256"]
        ):
            raise ValueError("archived terminal receipt bytes differ from package")
        terminal = strict_json_loads(terminal_payload)
        if (
            not isinstance(terminal, dict)
            or terminal.get("schema_version") != D.SCHEMA
            or terminal.get("status") != expected["scientific_status"]
            or terminal.get("content_sha256") != terminal_record["content_sha256"]
            or terminal.get("content_sha256") != A.content_hash(terminal)
            or terminal.get("downstream_lock_content_sha256") != A.DOWNSTREAM_LOCK_CONTENT_SHA256
            or terminal.get("metric_lock_content_sha256") != D.METRIC_LOCK_CONTENT_SHA256
        ):
            raise ValueError("archived terminal receipt identity is invalid")
        D._verify_terminal_logic(terminal)
        actual: dict[str, dict[str, Any]] = {}
        payload_bytes = terminal_bytes
        for relative in sorted(path for path in regular_paths if path != TERMINAL_NAME):
            destination = extraction_root / relative if extraction_root is not None else None
            _, digest, size = _read_member(bundle, member_map[relative], destination)
            actual[relative] = {"path": relative, "bytes": size, "sha256": digest}
            payload_bytes += size
        _validate_terminal_file_records(terminal, actual)
        upstream_manifest = package["upstream_model_publication"]["release_manifest"]
        for arm in ("candidate-fixed", "candidate-matched-mass"):
            relative = f"inputs/grids/{arm}/provenance/release_manifest.json"
            record = actual.get(relative)
            if record is None or {
                "path": "release_manifest.json",
                "bytes": record["bytes"],
                "sha256": record["sha256"],
            } != {
                key: upstream_manifest[key] for key in ("path", "bytes", "sha256")
            }:
                raise ValueError(f"archived downstream grid uses a different model release: {arm}")
        if payload_bytes != expected["payload_bytes"]:
            raise ValueError("archive payload byte count differs from package receipt")
    return {
        "status": "PASS",
        "scientific_status": terminal["status"],
        "terminal_content_sha256": terminal["content_sha256"],
        "members": len(members),
    }


def verify_archive(
    archive: Path,
    package_receipt: dict[str, Any],
    *,
    downstream_dir: Path | None = None,
    model_package_receipt: Path | None = None,
    model_publication_receipt: Path | None = None,
    full_archive: Path | None = None,
    parts_root: Path | None = None,
    deep: bool = True,
    probe_upstream: bool = True,
    materialize_parent: Path | None = None,
    expected_terminal_content_sha256: str,
    expected_upstream_publication_content_sha256: str,
) -> dict[str, Any]:
    package = _validate_package_receipt(package_receipt)
    _require_external_pins(
        package,
        expected_terminal_content_sha256,
        expected_upstream_publication_content_sha256,
    )
    _same_upstream_binding(
        package,
        model_package_receipt,
        model_publication_receipt,
        live=probe_upstream,
    )
    if probe_upstream:
        _probe_embedded_upstream(package["upstream_model_publication"])
    raw_archive = Path(archive)
    raw_parts_root = raw_archive.parent if parts_root is None else Path(parts_root)
    if _is_linklike(raw_parts_root) or not raw_parts_root.is_dir():
        raise ValueError("archive part root must be a real directory")
    archive = raw_archive.resolve()
    if archive.name != package["archive"]["path"]:
        raise ValueError("logical downstream archive name differs from package receipt")
    parts_root_resolved = raw_parts_root.resolve()
    part_paths = M._verified_part_paths(archive, package["archive"], parts_root_resolved)
    if full_archive is not None:
        full_path = Path(full_archive)
        if _is_linklike(full_path) or not full_path.is_file():
            raise ValueError("full downstream staging archive must be a regular file")
        full_path = full_path.resolve()
        if (
            full_path.stat().st_size != package["archive"]["bytes"]
            or sha256_file(full_path) != package["archive"]["sha256"]
        ):
            raise ValueError("full downstream staging archive differs from ordered parts")

        def open_source() -> BinaryIO:
            return full_path.open("rb")
    else:
        def open_source() -> BinaryIO:
            return M.MultipartReader(part_paths)

    if downstream_dir is not None:
        with open_source() as source:
            result = _inspect_archive(source, package, None)
        tree = validate_downstream_tree(downstream_dir) if deep else None
        if tree is not None:
            _validate_downstream_model_link(tree, package["upstream_model_publication"])
            expected = package["downstream"]
            if (
                tree["terminal_record"] != expected["terminal_receipt"]
                or len(tree["files"]) != expected["file_count"]
                or sum(path.stat().st_size for path in tree["files"].values())
                != expected["payload_bytes"]
            ):
                raise ValueError("local downstream tree differs from package receipt")
    elif deep:
        parent = Path(materialize_parent).resolve() if materialize_parent is not None else parts_root_resolved
        if _is_linklike(parent) or not parent.is_dir():
            raise ValueError("downstream materialization parent must be a real directory")
        with tempfile.TemporaryDirectory(prefix="crossscan-downstream-verify-", dir=parent) as temporary:
            extracted = Path(temporary) / ARCHIVE_ROOT
            extracted.mkdir()
            with open_source() as source:
                result = _inspect_archive(source, package, extracted)
            tree = validate_downstream_tree(extracted)
            _validate_downstream_model_link(tree, package["upstream_model_publication"])
            if tree["terminal_record"] != package["downstream"]["terminal_receipt"]:
                raise ValueError("materialized downstream terminal differs from package")
    else:
        with open_source() as source:
            result = _inspect_archive(source, package, None)
    return {
        **result,
        "archive_sha256": package["archive"]["sha256"],
        "archive_bytes": package["archive"]["bytes"],
    }


def canonical_release_body(package: dict[str, Any]) -> str:
    """Return the sole release body authorized by a sealed package."""
    _validate_package_receipt(package)
    downstream = package["downstream"]
    upstream = package["upstream_model_publication"]
    status = downstream["scientific_status"]
    claim = (
        "Bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement."
        if status == "PASS" else
        "Bounded negative result; no downstream-improvement claim is authorized."
    )
    lines = [
        "# Cross-scan v4 downstream result",
        "",
        claim,
        "",
        f"- Scientific status: {status}",
        f"- Downstream tooling code head: {package['code_head']}",
        (
            "- Terminal receipt content SHA-256: "
            f"{downstream['terminal_receipt']['content_sha256']}"
        ),
        (
            "- Downstream lock content SHA-256: "
            f"{downstream['downstream_lock_content_sha256']}"
        ),
        (
            "- Metric lock content SHA-256: "
            f"{downstream['metric_lock_content_sha256']}"
        ),
        f"- Downstream archive SHA-256: {package['archive']['sha256']}",
        f"- Downstream package content SHA-256: {package['content_sha256']}",
        f"- Upstream model tag: {upstream['tag']}",
        f"- Upstream model code head: {upstream['code_head']}",
        (
            "- Upstream model manifest content SHA-256: "
            f"{upstream['release_manifest_content_sha256']}"
        ),
        (
            "- Upstream model package content SHA-256: "
            f"{upstream['release_package_content_sha256']}"
        ),
        (
            "- Upstream model publication receipt content SHA-256: "
            f"{upstream['publication_receipt']['content_sha256']}"
        ),
        "",
        "The ordered parts reconstruct the exact terminal-bound regular-file universe.",
        "Scientific FAIL is a valid sealed result and must not be selectively rerun.",
    ]
    return "\n".join(lines) + "\n"


def _validate_anonymous_download(value: Any, expected: dict[str, Any]) -> None:
    M._validate_anonymous_download(value, expected)


def validate_publication_receipt(
    publication: dict[str, Any],
    package: dict[str, Any],
    package_payload: bytes,
) -> dict[str, Any]:
    _validate_package_receipt(package)
    try:
        payload_value = strict_json_loads(package_payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("downstream package payload is not strict JSON") from error
    if payload_value != package:
        raise ValueError("downstream package payload differs from supplied package object")
    tag = expected_tag(package)
    required = {
        "schema_version", "status", "verified_utc", "repository", "code_head", "tag",
        "scientific_status", "terminal_content_sha256",
        "downstream_lock_content_sha256", "metric_lock_content_sha256",
        "upstream_model_publication", "verifier", "release", "package_receipt",
        "archive", "content_sha256",
    }
    if (
        not isinstance(publication, dict)
        or set(publication) != required
        or publication.get("schema_version") != PUBLICATION_SCHEMA
        or publication.get("status") != "PUBLIC_LOGGED_OUT_VERIFIED"
        or not isinstance(publication.get("verified_utc"), str)
        or publication.get("content_sha256") != content_hash(publication)
        or publication.get("repository") != EXPECTED_REPOSITORY
        or publication.get("code_head") != package["code_head"]
        or publication.get("tag") != tag
        or publication.get("scientific_status") != package["downstream"]["scientific_status"]
        or publication.get("terminal_content_sha256")
        != package["downstream"]["terminal_receipt"]["content_sha256"]
        or publication.get("downstream_lock_content_sha256") != A.DOWNSTREAM_LOCK_CONTENT_SHA256
        or publication.get("metric_lock_content_sha256") != D.METRIC_LOCK_CONTENT_SHA256
        or publication.get("upstream_model_publication")
        != package["upstream_model_publication"]
    ):
        raise ValueError("downstream publication receipt header/package binding is invalid")
    verifier = publication.get("verifier")
    if verifier != publication_verifier_identity():
        raise ValueError("downstream publication verifier identity is invalid")
    release = publication.get("release")
    expected_url = f"https://github.com/{EXPECTED_REPOSITORY}/releases/tag/{tag}"
    if (
        not isinstance(release, dict)
        or not isinstance(release.get("id"), int)
        or release.get("html_url") != expected_url
        or release.get("immutable") is not True
        or release.get("target_commitish") != package["code_head"]
        or not isinstance(release.get("published_at"), str)
        or not isinstance(release.get("created_at"), str)
    ):
        raise ValueError("downstream publication release identity is invalid")
    chain = release.get("tag_object_chain")
    if (
        not isinstance(chain, list)
        or not chain
        or chain[-1].get("type") != "commit"
        or chain[-1].get("sha") != package["code_head"]
    ):
        raise ValueError("downstream publication tag chain is invalid")
    receipt_expected = {
        "bytes": len(package_payload),
        "sha256": hashlib.sha256(package_payload).hexdigest(),
    }
    receipt_url = (
        f"https://github.com/{EXPECTED_REPOSITORY}/releases/download/{tag}/"
        f"{PACKAGE_ASSET_NAME}"
    )
    receipt_record = publication.get("package_receipt")
    if (
        not isinstance(receipt_record, dict)
        or receipt_record.get("content_sha256") != package["content_sha256"]
        or receipt_record.get("bytes") != receipt_expected["bytes"]
        or receipt_record.get("sha256") != receipt_expected["sha256"]
        or receipt_record.get("github_digest") != f"sha256:{receipt_expected['sha256']}"
        or receipt_record.get("url") != receipt_url
        or not isinstance(receipt_record.get("asset_id"), int)
    ):
        raise ValueError("downstream publication package asset is invalid")
    _validate_anonymous_download(receipt_record.get("anonymous_download"), receipt_expected)
    archive = publication.get("archive")
    if (
        not isinstance(archive, dict)
        or archive.get("path") != package["archive"]["path"]
        or archive.get("bytes") != package["archive"]["bytes"]
        or archive.get("sha256") != package["archive"]["sha256"]
    ):
        raise ValueError("downstream publication logical archive is invalid")
    public_parts = archive.get("parts")
    expected_parts = package["archive"]["parts"]
    if not isinstance(public_parts, list) or len(public_parts) != len(expected_parts):
        raise ValueError("downstream publication archive part count is invalid")
    asset_ids = {receipt_record["asset_id"]}
    for expected_part, actual in zip(expected_parts, public_parts):
        expected_part_url = (
            f"https://github.com/{EXPECTED_REPOSITORY}/releases/download/{tag}/"
            f"{expected_part['path']}"
        )
        if (
            not isinstance(actual, dict)
            or any(actual.get(key) != value for key, value in expected_part.items())
            or actual.get("github_digest") != f"sha256:{expected_part['sha256']}"
            or actual.get("url") != expected_part_url
            or not isinstance(actual.get("asset_id"), int)
            or actual["asset_id"] in asset_ids
        ):
            raise ValueError(f"downstream publication archive part is invalid: {expected_part['path']}")
        asset_ids.add(actual["asset_id"])
        _validate_anonymous_download(actual.get("anonymous_download"), expected_part)
    return publication


def probe_publication(
    package: dict[str, Any],
    publication: dict[str, Any],
    package_payload: bytes,
) -> dict[str, Any]:
    validate_publication_receipt(publication, package, package_payload)
    repository = publication["repository"]
    tag = publication["tag"]
    target, chain = _resolve_tag(repository, tag)
    if target != package["code_head"] or chain != publication["release"]["tag_object_chain"]:
        raise ValueError("live downstream tag chain differs from publication receipt")
    encoded = urllib.parse.quote(tag, safe="")
    release = _anonymous_json(f"{M.API_ROOT}/repos/{repository}/releases/tags/{encoded}")
    expected_release = publication["release"]
    if (
        release.get("id") != expected_release["id"]
        or release.get("html_url") != expected_release["html_url"]
        or release.get("tag_name") != tag
        or release.get("target_commitish") != package["code_head"]
        or release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not True
        or release.get("published_at") != expected_release["published_at"]
        or release.get("created_at") != expected_release["created_at"]
    ):
        raise ValueError("live immutable downstream release metadata differs")
    assets = release.get("assets")
    expected_names = {PACKAGE_ASSET_NAME} | {
        part["path"] for part in package["archive"]["parts"]
    }
    if (
        not isinstance(assets, list)
        or len(assets) != len(expected_names)
        or len({asset.get("name") for asset in assets}) != len(assets)
        or {asset.get("name") for asset in assets} != expected_names
    ):
        raise ValueError("live immutable downstream release asset universe differs")
    records = {PACKAGE_ASSET_NAME: publication["package_receipt"]}
    records.update({part["path"]: part for part in publication["archive"]["parts"]})
    for asset in assets:
        record = records[asset["name"]]
        if (
            asset.get("id") != record.get("asset_id")
            or asset.get("node_id") != record.get("asset_node_id")
            or asset.get("state") != "uploaded"
            or asset.get("size") != record.get("bytes")
            or asset.get("digest") != record.get("github_digest")
            or asset.get("browser_download_url") != record.get("url")
            or asset.get("created_at") != record.get("created_at")
            or asset.get("updated_at") != record.get("updated_at")
        ):
            raise ValueError(f"live immutable downstream asset metadata differs: {asset['name']}")
    body = release.get("body")
    if body != canonical_release_body(package):
        raise ValueError("live downstream release notes differ from the authorized exact body")
    return {"status": "PASS", "release_id": release["id"], "tag": tag, "assets": len(assets)}


def verify_public(
    package_receipt_path: Path,
    model_package_receipt: Path | None,
    model_publication_receipt: Path | None,
    repository: str,
    tag: str,
    output: Path,
    expected_code_head: str,
    expected_terminal_content_sha256: str,
    expected_upstream_publication_content_sha256: str,
) -> dict[str, Any]:
    if repository != EXPECTED_REPOSITORY or not TAG_RE.fullmatch(tag):
        raise ValueError("repository or downstream tag differs from publication contract")
    if not isinstance(expected_code_head, str) or not HEX40_RE.fullmatch(expected_code_head):
        raise ValueError("expected code head must be exactly 40 lowercase hexadecimal characters")
    package_path = Path(package_receipt_path)
    if package_path.name != PACKAGE_ASSET_NAME:
        raise ValueError(f"public downstream package receipt must use {PACKAGE_ASSET_NAME}")
    package, package_payload = _load_hashed_json(package_path, PACKAGE_SCHEMA)
    _validate_package_receipt(package)
    _require_external_pins(
        package,
        expected_terminal_content_sha256,
        expected_upstream_publication_content_sha256,
    )
    _same_upstream_binding(
        package,
        model_package_receipt,
        model_publication_receipt,
        live=False,
    )
    if package["code_head"] != expected_code_head:
        raise ValueError("downstream package does not target the independently expected code head")
    if tag != expected_tag(package):
        raise ValueError("downstream release tag is not derived from terminal content hash")
    _probe_embedded_upstream(package["upstream_model_publication"])
    tag_target, tag_chain = _resolve_tag(repository, tag)
    if tag_target != package["code_head"]:
        raise ValueError("immutable downstream tag does not target its pinned code head")
    encoded = urllib.parse.quote(tag, safe="")
    release = _anonymous_json(f"{M.API_ROOT}/repos/{repository}/releases/tags/{encoded}")
    if (
        release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not True
        or release.get("tag_name") != tag
    ):
        raise ValueError("GitHub downstream release is not public, final, and immutable")
    archive_parts = package["archive"]["parts"]
    assets = release.get("assets")
    expected_names = {part["path"] for part in archive_parts} | {PACKAGE_ASSET_NAME}
    if (
        not isinstance(assets, list)
        or len(assets) != len(expected_names)
        or len({asset.get("name") for asset in assets}) != len(assets)
        or {asset.get("name") for asset in assets} != expected_names
    ):
        raise ValueError("downstream release assets must be exactly parts plus package receipt")
    by_name = {asset["name"]: asset for asset in assets}
    expected_assets = {
        part["path"]: {"bytes": part["bytes"], "sha256": part["sha256"]}
        for part in archive_parts
    }
    expected_assets[PACKAGE_ASSET_NAME] = {
        "bytes": len(package_payload),
        "sha256": hashlib.sha256(package_payload).hexdigest(),
    }
    for name, expected in expected_assets.items():
        asset = by_name[name]
        if (
            asset.get("state") != "uploaded"
            or asset.get("size") != expected["bytes"]
            or asset.get("digest") != f"sha256:{expected['sha256']}"
        ):
            raise ValueError(f"GitHub server-side downstream asset digest differs: {name}")
    body = release.get("body")
    if body != canonical_release_body(package):
        raise ValueError("immutable downstream release notes differ from the authorized exact body")
    output = Path(output).resolve()
    if output.exists() or _is_linklike(output):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crossscan-downstream-public-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        downloaded_receipt = temporary_root / PACKAGE_ASSET_NAME
        receipt_download = _download_anonymous(
            by_name[PACKAGE_ASSET_NAME]["browser_download_url"],
            downloaded_receipt,
            expected_assets[PACKAGE_ASSET_NAME]["bytes"],
        )
        if downloaded_receipt.read_bytes() != package_payload:
            raise ValueError("anonymous downstream package receipt differs byte-for-byte")
        part_downloads = []
        for part in archive_parts:
            name = part["path"]
            downloaded = temporary_root / name
            download = _download_anonymous(
                by_name[name]["browser_download_url"],
                downloaded,
                expected_assets[name]["bytes"],
            )
            if download["sha256"] != part["sha256"]:
                raise ValueError(f"anonymous downstream archive part hash differs: {name}")
            part_downloads.append({
                **part,
                "asset_id": by_name[name].get("id"),
                "asset_node_id": by_name[name].get("node_id"),
                "url": by_name[name].get("browser_download_url"),
                "created_at": by_name[name].get("created_at"),
                "updated_at": by_name[name].get("updated_at"),
                "content_type": by_name[name].get("content_type"),
                "github_digest": by_name[name].get("digest"),
                "anonymous_download": download,
            })
        verify_archive(
            temporary_root / package["archive"]["path"],
            package,
            model_package_receipt=model_package_receipt,
            model_publication_receipt=model_publication_receipt,
            parts_root=temporary_root,
            deep=True,
            probe_upstream=False,
            materialize_parent=temporary_root,
            expected_terminal_content_sha256=expected_terminal_content_sha256,
            expected_upstream_publication_content_sha256=(
                expected_upstream_publication_content_sha256
            ),
        )
    publication = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "PUBLIC_LOGGED_OUT_VERIFIED",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "code_head": package["code_head"],
        "tag": tag,
        "scientific_status": package["downstream"]["scientific_status"],
        "terminal_content_sha256": package["downstream"]["terminal_receipt"]["content_sha256"],
        "downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
        "metric_lock_content_sha256": D.METRIC_LOCK_CONTENT_SHA256,
        "upstream_model_publication": package["upstream_model_publication"],
        "verifier": publication_verifier_identity(),
        "release": {
            "id": release.get("id"),
            "html_url": release.get("html_url"),
            "immutable": True,
            "published_at": release.get("published_at"),
            "created_at": release.get("created_at"),
            "target_commitish": release.get("target_commitish"),
            "tag_object_chain": tag_chain,
        },
        "package_receipt": {
            "content_sha256": package["content_sha256"],
            **expected_assets[PACKAGE_ASSET_NAME],
            "asset_id": by_name[PACKAGE_ASSET_NAME].get("id"),
            "asset_node_id": by_name[PACKAGE_ASSET_NAME].get("node_id"),
            "url": by_name[PACKAGE_ASSET_NAME].get("browser_download_url"),
            "created_at": by_name[PACKAGE_ASSET_NAME].get("created_at"),
            "updated_at": by_name[PACKAGE_ASSET_NAME].get("updated_at"),
            "content_type": by_name[PACKAGE_ASSET_NAME].get("content_type"),
            "github_digest": by_name[PACKAGE_ASSET_NAME].get("digest"),
            "anonymous_download": receipt_download,
        },
        "archive": {
            "path": package["archive"]["path"],
            "bytes": package["archive"]["bytes"],
            "sha256": package["archive"]["sha256"],
            "parts": part_downloads,
        },
    }
    publication["content_sha256"] = content_hash(publication)
    validate_publication_receipt(publication, package, package_payload)
    probe_publication(package, publication, package_payload)
    _write_json_exclusive(output, publication)
    return publication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("pack")
    pack.add_argument("--downstream-dir", type=Path, required=True)
    pack.add_argument("--model-package-receipt", type=Path, required=True)
    pack.add_argument("--model-publication-receipt", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--receipt", type=Path, required=True)
    pack.add_argument("--code-head", required=True)
    pack.add_argument("--expected-terminal-content-sha256", required=True)
    pack.add_argument("--expected-upstream-publication-content-sha256", required=True)
    pack.add_argument(
        "--part-size-bytes",
        type=int,
        default=MAX_PART_BYTES,
        help=f"maximum bytes per GitHub asset (default/max: {MAX_PART_BYTES})",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--downstream-dir", type=Path)
    verify.add_argument("--model-package-receipt", type=Path, required=True)
    verify.add_argument("--model-publication-receipt", type=Path, required=True)
    verify.add_argument("--expected-terminal-content-sha256", required=True)
    verify.add_argument("--expected-upstream-publication-content-sha256", required=True)
    public = subparsers.add_parser("verify-public")
    public.add_argument("--package-receipt", type=Path, required=True)
    public.add_argument("--model-package-receipt", type=Path)
    public.add_argument("--model-publication-receipt", type=Path)
    public.add_argument("--repository", default=EXPECTED_REPOSITORY)
    public.add_argument("--tag", required=True)
    public.add_argument("--output", type=Path, required=True)
    public.add_argument("--expected-code-head", required=True)
    public.add_argument("--expected-terminal-content-sha256", required=True)
    public.add_argument("--expected-upstream-publication-content-sha256", required=True)
    confirmation = subparsers.add_parser("confirmation")
    confirmation.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pack":
        result = pack_downstream(
            args.downstream_dir,
            args.model_package_receipt,
            args.model_publication_receipt,
            args.archive,
            args.receipt,
            code_head=args.code_head,
            expected_terminal_content_sha256=args.expected_terminal_content_sha256,
            expected_upstream_publication_content_sha256=(
                args.expected_upstream_publication_content_sha256
            ),
            part_size_bytes=args.part_size_bytes,
        )
    elif args.command == "verify":
        package, _ = _load_hashed_json(args.receipt, PACKAGE_SCHEMA)
        result = verify_archive(
            args.archive,
            package,
            downstream_dir=args.downstream_dir,
            model_package_receipt=args.model_package_receipt,
            model_publication_receipt=args.model_publication_receipt,
            deep=True,
            expected_terminal_content_sha256=args.expected_terminal_content_sha256,
            expected_upstream_publication_content_sha256=(
                args.expected_upstream_publication_content_sha256
            ),
        )
    elif args.command == "verify-public":
        result = verify_public(
            args.package_receipt,
            args.model_package_receipt,
            args.model_publication_receipt,
            args.repository,
            args.tag,
            args.output,
            args.expected_code_head,
            args.expected_terminal_content_sha256,
            args.expected_upstream_publication_content_sha256,
        )
    else:
        package, _ = _load_hashed_json(args.receipt, PACKAGE_SCHEMA)
        _validate_package_receipt(package)
        print(expected_confirmation(package))
        return 0
    print(json.dumps({
        "status": result["status"],
        "scientific_status": result.get(
            "scientific_status",
            result.get("downstream", {}).get("scientific_status"),
        ),
        "content_sha256": result.get("content_sha256"),
        "archive_sha256": result.get("archive_sha256"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
