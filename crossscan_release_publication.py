"""Build and anonymously verify a deterministic cross-scan release archive.

This helper never uploads or changes GitHub state.  ``pack`` creates a byte-stable
uncompressed PAX tar and package receipt.  ``verify-public`` accepts a GitHub
Release only after an anonymous API/readback check, an immutable-release flag,
an exact tag target, server-side asset digests, and full archive revalidation.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


PACKAGE_SCHEMA = "crossscan-release-package-v1"
PUBLICATION_SCHEMA = "crossscan-release-publication-v1"
EXPECTED_REPOSITORY = "TAUIL-Abd-Elilah/vesuvius-repro"
ARCHIVE_ROOT = "crossscan_release_v4"
CHECKSUM_NAME = "SHA256SUMS"
MANIFEST_NAME = "release_manifest.json"
CHECKSUM_RE = re.compile(r"^([0-9a-f]{64})  ([^\r\n]+)$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
TAG_RE = re.compile(r"^crossscan-v4-[0-9a-f]{12}$")
API_ROOT = "https://api.github.com"
USER_AGENT = "crossscan-release-publication/1"
ALLOWED_DOWNLOAD_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "release-assets.githubusercontent.com",
}
MAX_PART_BYTES = 1_900_000_000
PART_SUFFIX_RE = re.compile(r"\.part-([0-9]{3})$")


class MultipartReader(io.RawIOBase):
    """Seekable read-only concatenation of already verified archive parts."""

    def __init__(self, parts: list[tuple[Path, int]]):
        super().__init__()
        self._parts = parts
        self._offsets: list[int] = []
        total = 0
        for _, size in parts:
            self._offsets.append(total)
            total += size
        self._size = total
        self._position = 0
        self._index: int | None = None
        self._stream: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._size + offset
        else:
            raise ValueError(f"invalid seek mode: {whence}")
        if position < 0:
            raise ValueError("negative multipart seek")
        self._position = min(position, self._size)
        return self._position

    def _part_index(self, position: int) -> int:
        for index in range(len(self._parts) - 1, -1, -1):
            if position >= self._offsets[index]:
                return index
        return 0

    def _open_part(self, index: int, local_offset: int) -> None:
        if self._index != index:
            if self._stream is not None:
                self._stream.close()
            self._stream = self._parts[index][0].open("rb")
            self._index = index
        assert self._stream is not None
        self._stream.seek(local_offset)

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("read from closed multipart archive")
        if self._position >= self._size or size == 0:
            return b""
        remaining = self._size - self._position if size < 0 else min(size, self._size - self._position)
        blocks: list[bytes] = []
        while remaining:
            index = self._part_index(self._position)
            part_path, part_size = self._parts[index]
            local_offset = self._position - self._offsets[index]
            self._open_part(index, local_offset)
            take = min(remaining, part_size - local_offset)
            assert self._stream is not None
            block = self._stream.read(take)
            if len(block) != take:
                raise OSError(f"short read from archive part: {part_path}")
            blocks.append(block)
            self._position += take
            remaining -= take
        return b"".join(blocks)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: dict[str, Any]) -> str:
    copy = dict(value)
    copy.pop("content_sha256", None)
    return hashlib.sha256(canonical_json(copy).encode("utf-8")).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def strict_json_loads(payload: bytes | str) -> Any:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    return json.loads(
        text,
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )


def sha256_stream(stream: BinaryIO, *, limit: int | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        block = stream.read(8 * 1024 * 1024)
        if not block:
            break
        total += len(block)
        if limit is not None and total > limit:
            raise ValueError("stream exceeds the expected byte limit")
        digest.update(block)
    return digest.hexdigest(), total


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)[0]


def file_record(path: Path, relative: str | None = None) -> dict[str, Any]:
    return {
        "path": relative if relative is not None else path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _is_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    value = getattr(metadata, "st_file_attributes", 0)
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(value & flag)


def _safe_relative(value: str) -> str:
    if "\\" in value or "\x00" in value or value == CHECKSUM_NAME:
        raise ValueError(f"unsafe checksum path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe checksum path: {value!r}")
    if path.as_posix() != value:
        raise ValueError(f"non-canonical checksum path: {value!r}")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValueError(f"release path is not portable printable ASCII: {value!r}")
    if any(character in '<>:"|?*' for character in value):
        raise ValueError(f"release path contains a non-portable character: {value!r}")
    return value


def _read_checksums(path: Path) -> tuple[bytes, list[str], dict[str, str]]:
    payload = path.read_bytes()
    if not payload or payload.startswith(b"\xef\xbb\xbf"):
        raise ValueError("SHA256SUMS must be nonempty UTF-8 without BOM")
    if b"\r" in payload or not payload.endswith(b"\n") or payload.endswith(b"\n\n"):
        raise ValueError("SHA256SUMS must use exactly one LF after every entry")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("SHA256SUMS is not UTF-8") from error
    lines = text[:-1].split("\n")
    paths: list[str] = []
    values: dict[str, str] = {}
    folded: set[str] = set()
    for line in lines:
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid SHA256SUMS entry: {line!r}")
        digest, relative = match.groups()
        relative = _safe_relative(relative)
        if relative in values or relative.casefold() in folded:
            raise ValueError(f"duplicate or case-alias checksum path: {relative}")
        values[relative] = digest
        paths.append(relative)
        folded.add(relative.casefold())
    if paths != sorted(paths):
        raise ValueError("SHA256SUMS paths are not ordinal-sorted")
    return payload, paths, values


def _inventory(root: Path) -> dict[str, Path]:
    root = Path(root)
    if root.is_symlink() or _is_reparse(root):
        raise ValueError(f"release root must be a real directory: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"release root must be a real directory: {root}")
    result: dict[str, Path] = {}
    folded: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink() or _is_reparse(path):
            raise ValueError(f"links/reparse points are forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"special release entry is forbidden: {path}")
        relative = path.relative_to(root).as_posix()
        _safe_relative(relative) if relative != CHECKSUM_NAME else None
        if relative in result or relative.casefold() in folded:
            raise ValueError(f"duplicate or case-alias release path: {relative}")
        result[relative] = path
        folded.add(relative.casefold())
    return result


def _load_hashed_json(path: Path, schema: str | None = None) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or _is_reparse(path):
        raise ValueError(f"JSON receipt cannot be a link/reparse point: {path}")
    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("content_sha256") != content_hash(value):
        raise ValueError(f"invalid canonical content hash: {path}")
    if schema is not None and value.get("schema_version") != schema:
        raise ValueError(f"unexpected schema in {path}")
    return value


def _validate_positive_manifest(manifest: Any) -> dict[str, Any]:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "crossscan-model-release-v1"
        or manifest.get("status") != "PASS"
        or manifest.get("outcome") != "POSITIVE_DEPLOYABLE"
        or manifest.get("selected_steps") != 4000
    ):
        raise ValueError("release manifest is not an authorized positive v4 release")
    return manifest


def validate_release(root: Path) -> dict[str, Any]:
    raw_root = Path(root)
    files = _inventory(raw_root)
    root = raw_root.resolve()
    checksum_path = root / CHECKSUM_NAME
    manifest_path = root / MANIFEST_NAME
    if CHECKSUM_NAME not in files or MANIFEST_NAME not in files:
        raise ValueError("release lacks SHA256SUMS or release_manifest.json")
    checksum_payload, paths, expected = _read_checksums(checksum_path)
    if set(paths) != set(files) - {CHECKSUM_NAME}:
        raise ValueError("SHA256SUMS does not bind the exact release file universe")
    for relative in paths:
        if sha256_file(files[relative]) != expected[relative]:
            raise ValueError(f"release checksum mismatch: {relative}")
    manifest = _load_hashed_json(manifest_path)
    _validate_positive_manifest(manifest)
    return {
        "root": root,
        "files": files,
        "paths": paths,
        "checksums": expected,
        "checksum_payload": checksum_payload,
        "manifest": manifest,
    }


def write_release_checksums(root: Path) -> dict[str, Any]:
    """Create the one permitted checksum file for an otherwise complete release."""
    raw_root = Path(root)
    files = _inventory(raw_root)
    if CHECKSUM_NAME in files or (raw_root / CHECKSUM_NAME).is_symlink():
        raise FileExistsError(f"refusing to replace {raw_root / CHECKSUM_NAME}")
    root = raw_root.resolve()
    lines = [
        f"{sha256_file(files[relative])}  {relative}"
        for relative in sorted(files)
    ]
    payload = ("\n".join(lines) + "\n").encode("ascii")
    checksum_path = root / CHECKSUM_NAME
    descriptor = os.open(checksum_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    validate_release(root)
    return file_record(checksum_path, CHECKSUM_NAME)


def _tar_info(relative: str, path: Path) -> tarfile.TarInfo:
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


def _part_path(archive: Path, index: int) -> Path:
    return archive.with_name(f"{archive.name}.part-{index:03d}")


def _split_archive(staging: Path, archive: Path, part_size_bytes: int) -> list[dict[str, Any]]:
    if not 1 <= part_size_bytes <= MAX_PART_BYTES:
        raise ValueError(f"archive part size must be in [1,{MAX_PART_BYTES}]")
    records: list[dict[str, Any]] = []
    offset = 0
    with staging.open("rb") as source:
        index = 0
        while offset < staging.stat().st_size:
            output = _part_path(archive, index)
            temporary = output.with_name(output.name + ".tmp")
            if output.exists() or temporary.exists():
                raise FileExistsError(f"refusing to replace archive part: {output} or {temporary}")
            remaining = min(part_size_bytes, staging.stat().st_size - offset)
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            digest = hashlib.sha256()
            written = 0
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as target:
                    while written < remaining:
                        block = source.read(min(8 * 1024 * 1024, remaining - written))
                        if not block:
                            raise OSError("short read while splitting deterministic archive")
                        target.write(block)
                        digest.update(block)
                        written += len(block)
                    target.flush()
                    os.fsync(target.fileno())
            finally:
                os.close(descriptor)
            temporary.rename(output)
            records.append({
                "path": output.name,
                "offset": offset,
                "bytes": written,
                "sha256": digest.hexdigest(),
            })
            offset += written
            index += 1
    return records


def pack_release(
    release_dir: Path,
    archive: Path,
    receipt_path: Path,
    *,
    code_head: str,
    part_size_bytes: int = MAX_PART_BYTES,
) -> dict[str, Any]:
    if not isinstance(code_head, str) or not HEX40_RE.fullmatch(code_head):
        raise ValueError("code head must be exactly 40 lowercase hexadecimal characters")
    release = validate_release(release_dir)
    archive = archive.resolve()
    receipt_path = receipt_path.resolve()
    staging = archive.with_name(archive.name + ".tmp")
    release_root = release["root"]
    if (
        archive == receipt_path
        or staging == receipt_path
        or archive.is_relative_to(release_root)
        or staging.is_relative_to(release_root)
        or receipt_path.is_relative_to(release_root)
    ):
        raise ValueError("archive, staging, and receipt must be distinct and outside the release")
    for path in (archive, staging, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to replace publication output: {path}")
    if list(archive.parent.glob(f"{archive.name}.part-*")):
        raise FileExistsError("archive part or part staging output already exists")
    archive.parent.mkdir(parents=True, exist_ok=True)
    member_paths = sorted(release["files"])
    try:
        with tarfile.open(staging, mode="x", format=tarfile.PAX_FORMAT, dereference=False) as bundle:
            for relative in member_paths:
                path = release["files"][relative]
                with path.open("rb") as stream:
                    bundle.addfile(_tar_info(relative, path), stream)
    except BaseException:
        # A failed temporary is intentionally preserved for diagnosis.
        raise
    archive_sha256 = sha256_file(staging)
    archive_bytes = staging.stat().st_size
    parts = _split_archive(staging, archive, part_size_bytes)
    manifest_path = release["files"][MANIFEST_NAME]
    checksum_path = release["files"][CHECKSUM_NAME]
    receipt = {
        "schema_version": PACKAGE_SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_head": code_head,
        "release": {
            "root_name": ARCHIVE_ROOT,
            "file_count": len(member_paths),
            "payload_bytes": sum(path.stat().st_size for path in release["files"].values()),
            "manifest": {
                **file_record(manifest_path, MANIFEST_NAME),
                "content_sha256": release["manifest"]["content_sha256"],
            },
            "sha256sums": file_record(checksum_path, CHECKSUM_NAME),
        },
        "archive": {
            "path": archive.name,
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "format": "uncompressed-pax-tar",
            "root_name": ARCHIVE_ROOT,
            "member_count": len(member_paths),
            "part_size_limit_bytes": part_size_bytes,
            "parts": parts,
            "metadata": {
                "mtime": 0,
                "mode": "0644",
                "uid": 0,
                "gid": 0,
                "uname": "",
                "gname": "",
                "member_order": "ordinal-relative-path",
            },
        },
    }
    receipt["content_sha256"] = content_hash(receipt)
    verify_archive(
        archive, receipt, release_dir=release_dir, full_archive=staging
    )
    staging.unlink()
    verify_archive(archive, receipt, release_dir=release_dir)
    _write_json_exclusive(receipt_path, receipt)
    return receipt


def _validate_package_receipt(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema_version") != PACKAGE_SCHEMA
        or value.get("status") != "PASS"
        or not isinstance(value.get("code_head"), str)
        or not HEX40_RE.fullmatch(value["code_head"])
        or value.get("content_sha256") != content_hash(value)
    ):
        raise ValueError("invalid cross-scan package receipt")
    archive = value.get("archive")
    release = value.get("release")
    if not isinstance(archive, dict) or not isinstance(release, dict):
        raise ValueError("package receipt lacks archive/release records")
    if (
        archive.get("format") != "uncompressed-pax-tar"
        or archive.get("root_name") != ARCHIVE_ROOT
        or PurePosixPath(str(archive.get("path", ""))).name != archive.get("path")
        or not str(archive.get("path", "")).endswith(".tar")
        or not HEX64_RE.fullmatch(str(archive.get("sha256", "")))
        or not isinstance(archive.get("bytes"), int)
        or archive.get("bytes", 0) <= 0
        or release.get("root_name") != ARCHIVE_ROOT
    ):
        raise ValueError("package receipt has invalid archive identity")
    parts = archive.get("parts")
    part_limit = archive.get("part_size_limit_bytes")
    if (
        not isinstance(parts, list)
        or not parts
        or not isinstance(part_limit, int)
        or not 1 <= part_limit <= MAX_PART_BYTES
    ):
        raise ValueError("package receipt has invalid archive parts")
    offset = 0
    for index, part in enumerate(parts):
        expected_name = f"{archive['path']}.part-{index:03d}"
        if (
            not isinstance(part, dict)
            or part.get("path") != expected_name
            or part.get("offset") != offset
            or not isinstance(part.get("bytes"), int)
            or not 1 <= part["bytes"] <= part_limit
            or not HEX64_RE.fullmatch(str(part.get("sha256", "")))
        ):
            raise ValueError(f"package receipt has invalid archive part {index}")
        if index < len(parts) - 1 and part["bytes"] != part_limit:
            raise ValueError("only the final archive part may be shorter than the limit")
        offset += part["bytes"]
    if offset != archive["bytes"]:
        raise ValueError("archive part bytes do not reconstruct the logical archive")
    return value


def _verified_part_paths(
    archive: Path,
    expected_archive: dict[str, Any],
    parts_root: Path,
) -> list[tuple[Path, int]]:
    expected_names = {part["path"] for part in expected_archive["parts"]}
    actual = list(parts_root.glob(f"{archive.name}.part-*"))
    if {path.name for path in actual} != expected_names:
        raise ValueError("archive part file universe differs from the package receipt")
    overall = hashlib.sha256()
    result: list[tuple[Path, int]] = []
    for part in expected_archive["parts"]:
        raw_path = parts_root / part["path"]
        if raw_path.is_symlink() or _is_reparse(raw_path):
            raise ValueError(f"invalid archive part file: {part['path']}")
        path = raw_path.resolve()
        if (
            path.parent != parts_root
            or not path.is_file()
            or path.is_symlink()
            or _is_reparse(path)
            or path.stat().st_size != part["bytes"]
        ):
            raise ValueError(f"invalid archive part file: {part['path']}")
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as stream:
            while True:
                block = stream.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                overall.update(block)
                total += len(block)
        if total != part["bytes"] or digest.hexdigest() != part["sha256"]:
            raise ValueError(f"archive part checksum mismatch: {part['path']}")
        result.append((path, total))
    if overall.hexdigest() != expected_archive["sha256"]:
        raise ValueError("ordered archive parts do not reconstruct the full archive hash")
    return result


def verify_archive(
    archive: Path,
    package_receipt: dict[str, Any],
    *,
    release_dir: Path | None = None,
    full_archive: Path | None = None,
    parts_root: Path | None = None,
) -> dict[str, Any]:
    receipt = _validate_package_receipt(package_receipt)
    archive = Path(archive)
    raw_parts_root = archive.parent if parts_root is None else Path(parts_root)
    if raw_parts_root.is_symlink() or _is_reparse(raw_parts_root):
        raise ValueError("archive part root cannot be a link/reparse point")
    archive = archive.resolve()
    expected_archive = receipt["archive"]
    if archive.name != expected_archive["path"]:
        raise ValueError("logical archive name differs from the package receipt")
    root = raw_parts_root.resolve()
    part_paths = _verified_part_paths(archive, expected_archive, root)
    if full_archive is not None:
        raw_full_archive = Path(full_archive)
        if raw_full_archive.is_symlink() or _is_reparse(raw_full_archive):
            raise ValueError("full staging archive cannot be a link/reparse point")
        full_archive = raw_full_archive.resolve()
        if (
            not full_archive.is_file()
            or full_archive.stat().st_size != expected_archive["bytes"]
            or sha256_file(full_archive) != expected_archive["sha256"]
        ):
            raise ValueError("full staging archive differs from its ordered parts")
        source_context: Any = full_archive.open("rb")
    else:
        source_context = MultipartReader(part_paths)
    expected_root = f"{ARCHIVE_ROOT}/"
    with source_context as source:
        with tarfile.open(fileobj=source, mode="r:") as bundle:
            members = bundle.getmembers()
            relative_paths: list[str] = []
            member_map: dict[str, tarfile.TarInfo] = {}
            folded: set[str] = set()
            for member in members:
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"archive contains a non-regular member: {member.name}")
                if not member.name.startswith(expected_root):
                    raise ValueError(f"archive member escapes its single root: {member.name}")
                raw_relative = member.name[len(expected_root):]
                relative = CHECKSUM_NAME if raw_relative == CHECKSUM_NAME else _safe_relative(raw_relative)
                if relative in member_map or relative.casefold() in folded:
                    raise ValueError(f"archive contains duplicate/case-alias member: {relative}")
                if (
                    member.type != tarfile.REGTYPE
                    or getattr(member, "sparse", None) is not None
                    or member.linkname != ""
                    or member.devmajor != 0
                    or member.devminor != 0
                    or member.mtime != 0
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or set(member.pax_headers) - {"path"}
                    or ("path" in member.pax_headers and member.pax_headers["path"] != member.name)
                ):
                    raise ValueError(f"archive member metadata is not deterministic: {relative}")
                member_map[relative] = member
                relative_paths.append(relative)
                folded.add(relative.casefold())
            if relative_paths != sorted(relative_paths):
                raise ValueError("archive member order is not deterministic")
            if len(relative_paths) != expected_archive["member_count"] or CHECKSUM_NAME not in member_map:
                raise ValueError("archive member count/universe differs from receipt")
            checksum_stream = bundle.extractfile(member_map[CHECKSUM_NAME])
            if checksum_stream is None:
                raise ValueError("archive SHA256SUMS cannot be read")
            checksum_payload = checksum_stream.read()
            with tempfile.TemporaryDirectory(prefix="crossscan-checksums-") as temporary:
                checksum_copy = Path(temporary) / CHECKSUM_NAME
                checksum_copy.write_bytes(checksum_payload)
                _, checksum_paths, checksums = _read_checksums(checksum_copy)
            if set(checksum_paths) != set(relative_paths) - {CHECKSUM_NAME}:
                raise ValueError("archived SHA256SUMS does not bind its exact member universe")
            for relative in checksum_paths:
                stream = bundle.extractfile(member_map[relative])
                if stream is None:
                    raise ValueError(f"archive member cannot be read: {relative}")
                digest, size = sha256_stream(stream)
                if digest != checksums[relative] or size != member_map[relative].size:
                    raise ValueError(f"archive member checksum mismatch: {relative}")
            checksum_digest = hashlib.sha256(checksum_payload).hexdigest()
            if checksum_digest != receipt["release"]["sha256sums"]["sha256"]:
                raise ValueError("archived SHA256SUMS file hash differs from receipt")
            manifest_stream = bundle.extractfile(member_map[MANIFEST_NAME])
            if manifest_stream is None:
                raise ValueError("archive manifest cannot be read")
            manifest_payload = manifest_stream.read()
            manifest = strict_json_loads(manifest_payload)
            _validate_positive_manifest(manifest)
            if (
                manifest.get("content_sha256") != content_hash(manifest)
                or manifest.get("content_sha256") != receipt["release"]["manifest"]["content_sha256"]
                or hashlib.sha256(manifest_payload).hexdigest() != receipt["release"]["manifest"]["sha256"]
            ):
                raise ValueError("archived release manifest differs from receipt")
    if release_dir is not None:
        release = validate_release(release_dir)
        if set(release["files"]) != set(relative_paths):
            raise ValueError("local release and archive file universes differ")
        for relative, path in release["files"].items():
            stream_digest = (
                receipt["release"]["sha256sums"]["sha256"]
                if relative == CHECKSUM_NAME
                else release["checksums"][relative]
            )
            if sha256_file(path) != stream_digest:
                raise ValueError(f"local release changed after packaging: {relative}")
    return {
        "status": "PASS",
        "archive_sha256": expected_archive["sha256"],
        "archive_bytes": expected_archive["bytes"],
        "members": len(relative_paths),
        "release_manifest_content_sha256": receipt["release"]["manifest"]["content_sha256"],
    }


class _StrictRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_hosts: set[str]):
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(self, request, response, code, message, headers, new_url):
        parsed = urllib.parse.urlsplit(new_url)
        if (
            code not in (301, 302, 303, 307, 308)
            or parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.hostname not in self.allowed_hosts
        ):
            raise ValueError(f"refusing anonymous redirect to {new_url!r}")
        redirected = super().redirect_request(
            request, response, code, message, headers, new_url
        )
        if redirected is None:
            raise ValueError("anonymous redirect could not be constructed")
        for name in ("Authorization", "Cookie", "Proxy-Authorization"):
            redirected.remove_header(name)
            redirected.unredirected_hdrs.pop(name, None)
            redirected.unredirected_hdrs.pop(name.lower(), None)
        return redirected


def _anonymous_opener(allowed_hosts: set[str]) -> urllib.request.OpenerDirector:
    # Deliberately ignores environment proxies and installs no auth/cookie handlers.
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _StrictRedirect(allowed_hosts),
    )


def _anonymous_json(url: str) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.github.com"
        or parsed.username
        or parsed.password
    ):
        raise ValueError("anonymous GitHub API URL is not exact HTTPS api.github.com")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/vnd.github+json",
            "Accept-Encoding": "identity",
        },
    )
    with _anonymous_opener({"api.github.com"}).open(request, timeout=60) as response:
        if response.getcode() != 200 or response.headers.get("Content-Encoding") not in (None, "identity"):
            raise ValueError("anonymous GitHub API response is not exact unencoded HTTP 200")
        payload = response.read(16 * 1024 * 1024 + 1)
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError("anonymous GitHub API response exceeds 16 MiB")
        value = strict_json_loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"GitHub API returned a non-object: {url}")
    return value


def _resolve_tag(repository: str, tag: str) -> tuple[str, list[dict[str, str]]]:
    encoded = urllib.parse.quote(tag, safe="")
    value = _anonymous_json(f"{API_ROOT}/repos/{repository}/git/ref/tags/{encoded}")
    obj = value.get("object")
    chain: list[dict[str, str]] = []
    for _ in range(4):
        if not isinstance(obj, dict):
            raise ValueError("GitHub tag has no object")
        chain.append({
            "type": str(obj.get("type")),
            "sha": str(obj.get("sha")),
            "url": str(obj.get("url")),
        })
        if obj.get("type") == "commit":
            return str(obj.get("sha")), chain
        if obj.get("type") != "tag" or not str(obj.get("url", "")).startswith(f"{API_ROOT}/repos/{repository}/git/tags/"):
            raise ValueError("GitHub tag does not resolve safely to a commit")
        obj = _anonymous_json(str(obj["url"])).get("object")
    raise ValueError("GitHub tag indirection is too deep")


def _download_anonymous(url: str, output: Path, expected_bytes: int) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.hostname != "github.com":
        raise ValueError("release asset URL must be an unauthenticated github.com HTTPS URL")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/octet-stream",
            "Accept-Encoding": "identity",
        },
    )
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as target:
            with _anonymous_opener(ALLOWED_DOWNLOAD_HOSTS).open(request, timeout=120) as response:
                resolved = urllib.parse.urlsplit(response.geturl())
                if (
                    response.getcode() != 200
                    or response.headers.get("Content-Encoding") not in (None, "identity")
                    or resolved.scheme != "https"
                    or resolved.hostname not in ALLOWED_DOWNLOAD_HOSTS
                ):
                    raise ValueError("release download redirected to an unexpected host")
                digest = hashlib.sha256()
                total = 0
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > expected_bytes:
                        raise ValueError("anonymous release download exceeds expected bytes")
                    digest.update(block)
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
                headers = {
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "content_type": response.headers.get("Content-Type"),
                }
                resolved_host = resolved.hostname
    finally:
        os.close(descriptor)
    if total != expected_bytes:
        raise ValueError("anonymous release download byte count differs from receipt")
    return {
        "authorization_header_sent": False,
        "cookies_enabled": False,
        "environment_proxy_enabled": False,
        "accept_encoding": "identity",
        "resolved_host": resolved_host,
        "bytes": total,
        "sha256": digest.hexdigest(),
        "headers": headers,
    }


def _validate_anonymous_download(value: Any, expected: dict[str, Any]) -> None:
    if (
        not isinstance(value, dict)
        or value.get("authorization_header_sent") is not False
        or value.get("cookies_enabled") is not False
        or value.get("environment_proxy_enabled") is not False
        or value.get("accept_encoding") != "identity"
        or value.get("bytes") != expected["bytes"]
        or value.get("sha256") != expected["sha256"]
        or value.get("resolved_host") not in ALLOWED_DOWNLOAD_HOSTS
    ):
        raise ValueError("publication receipt has an invalid anonymous download record")


def validate_publication_receipt(
    publication: dict[str, Any],
    package: dict[str, Any],
    package_payload: bytes,
) -> dict[str, Any]:
    payload_package = strict_json_loads(package_payload)
    if not isinstance(payload_package, dict) or payload_package != package:
        raise ValueError("package receipt payload differs from the supplied package object")
    _validate_package_receipt(payload_package)
    _validate_package_receipt(package)
    tag = f"crossscan-v4-{package['release']['manifest']['content_sha256'][:12]}"
    if (
        publication.get("schema_version") != PUBLICATION_SCHEMA
        or publication.get("status") != "PUBLIC_LOGGED_OUT_VERIFIED"
        or publication.get("content_sha256") != content_hash(publication)
        or publication.get("repository") != EXPECTED_REPOSITORY
        or publication.get("code_head") != package["code_head"]
        or publication.get("tag") != tag
        or publication.get("release_manifest_content_sha256")
        != package["release"]["manifest"]["content_sha256"]
        or publication.get("sha256sums_file_sha256")
        != package["release"]["sha256sums"]["sha256"]
    ):
        raise ValueError("publication receipt header/package binding is invalid")
    verifier = publication.get("verifier")
    if (
        not isinstance(verifier, dict)
        or verifier.get("script_sha256") != sha256_file(Path(__file__).resolve())
        or verifier.get("python_version") != sys.version
        or verifier.get("network_policy")
        != "dedicated-no-proxy-no-auth-no-cookie-strict-https-redirect-opener"
    ):
        raise ValueError("publication receipt verifier identity is invalid")
    release = publication.get("release")
    expected_release_url = f"https://github.com/{EXPECTED_REPOSITORY}/releases/tag/{tag}"
    if (
        not isinstance(release, dict)
        or not isinstance(release.get("id"), int)
        or release.get("html_url") != expected_release_url
        or release.get("immutable") is not True
        or release.get("target_commitish") != package["code_head"]
        or not isinstance(release.get("published_at"), str)
        or not isinstance(release.get("created_at"), str)
    ):
        raise ValueError("publication receipt release identity is invalid")
    chain = release.get("tag_object_chain")
    if (
        not isinstance(chain, list)
        or not chain
        or chain[-1].get("type") != "commit"
        or chain[-1].get("sha") != package["code_head"]
    ):
        raise ValueError("publication receipt tag chain is invalid")
    receipt_record = publication.get("package_receipt")
    receipt_expected = {
        "bytes": len(package_payload),
        "sha256": hashlib.sha256(package_payload).hexdigest(),
    }
    receipt_url = f"https://github.com/{EXPECTED_REPOSITORY}/releases/download/{tag}/package_receipt.json"
    if (
        not isinstance(receipt_record, dict)
        or receipt_record.get("content_sha256") != package["content_sha256"]
        or receipt_record.get("bytes") != receipt_expected["bytes"]
        or receipt_record.get("sha256") != receipt_expected["sha256"]
        or receipt_record.get("github_digest") != f"sha256:{receipt_expected['sha256']}"
        or receipt_record.get("url") != receipt_url
        or not isinstance(receipt_record.get("asset_id"), int)
    ):
        raise ValueError("publication receipt package asset is invalid")
    _validate_anonymous_download(receipt_record.get("anonymous_download"), receipt_expected)
    archive = publication.get("archive")
    if (
        not isinstance(archive, dict)
        or archive.get("path") != package["archive"]["path"]
        or archive.get("bytes") != package["archive"]["bytes"]
        or archive.get("sha256") != package["archive"]["sha256"]
    ):
        raise ValueError("publication receipt logical archive is invalid")
    public_parts = archive.get("parts")
    expected_parts = package["archive"]["parts"]
    if not isinstance(public_parts, list) or len(public_parts) != len(expected_parts):
        raise ValueError("publication receipt archive part count is invalid")
    asset_ids = {receipt_record["asset_id"]}
    for expected, actual in zip(expected_parts, public_parts):
        expected_url = f"https://github.com/{EXPECTED_REPOSITORY}/releases/download/{tag}/{expected['path']}"
        if (
            not isinstance(actual, dict)
            or any(actual.get(key) != value for key, value in expected.items())
            or actual.get("github_digest") != f"sha256:{expected['sha256']}"
            or actual.get("url") != expected_url
            or not isinstance(actual.get("asset_id"), int)
            or actual["asset_id"] in asset_ids
        ):
            raise ValueError(f"publication receipt archive part is invalid: {expected['path']}")
        asset_ids.add(actual["asset_id"])
        _validate_anonymous_download(actual.get("anonymous_download"), expected)
    return publication


def probe_publication(
    package: dict[str, Any],
    publication: dict[str, Any],
    package_payload: bytes,
) -> dict[str, Any]:
    """Live anonymous metadata probe; immutable asset bytes need not be re-downloaded."""
    validate_publication_receipt(publication, package, package_payload)
    repository = publication["repository"]
    tag = publication["tag"]
    target, chain = _resolve_tag(repository, tag)
    if target != package["code_head"] or chain != publication["release"]["tag_object_chain"]:
        raise ValueError("live public tag chain differs from publication receipt")
    encoded = urllib.parse.quote(tag, safe="")
    release = _anonymous_json(f"{API_ROOT}/repos/{repository}/releases/tags/{encoded}")
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
        raise ValueError("live immutable release metadata differs from publication receipt")
    assets = release.get("assets")
    expected_names = {"package_receipt.json"} | {
        part["path"] for part in package["archive"]["parts"]
    }
    if (
        not isinstance(assets, list)
        or len(assets) != len(expected_names)
        or len({asset.get("name") for asset in assets}) != len(assets)
        or {asset.get("name") for asset in assets} != expected_names
    ):
        raise ValueError("live immutable release asset universe differs")
    published_records = {"package_receipt.json": publication["package_receipt"]}
    published_records.update({part["path"]: part for part in publication["archive"]["parts"]})
    for asset in assets:
        record = published_records[asset["name"]]
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
            raise ValueError(f"live immutable asset metadata differs: {asset['name']}")
    required_body_values = (
        package["code_head"],
        package["archive"]["sha256"],
        package["release"]["manifest"]["content_sha256"],
        package["release"]["sha256sums"]["sha256"],
        package["content_sha256"],
    )
    body = release.get("body")
    if not isinstance(body, str) or any(value not in body for value in required_body_values):
        raise ValueError("live immutable release notes omit a required hash")
    return {
        "status": "PASS",
        "release_id": release["id"],
        "tag": tag,
        "assets": len(assets),
    }


def verify_public(
    package_receipt_path: Path,
    repository: str,
    tag: str,
    output: Path,
    expected_code_head: str,
) -> dict[str, Any]:
    if repository != EXPECTED_REPOSITORY or not TAG_RE.fullmatch(tag):
        raise ValueError("repository or release tag differs from the publication contract")
    if not isinstance(expected_code_head, str) or not HEX40_RE.fullmatch(expected_code_head):
        raise ValueError("expected code head must be exactly 40 lowercase hexadecimal characters")
    if package_receipt_path.name != "package_receipt.json":
        raise ValueError("public package receipt must use the fixed asset name")
    package_payload = package_receipt_path.read_bytes()
    package = _load_hashed_json(package_receipt_path, PACKAGE_SCHEMA)
    _validate_package_receipt(package)
    if package["code_head"] != expected_code_head:
        raise ValueError("package receipt does not target the independently expected code head")
    expected_tag = f"crossscan-v4-{package['release']['manifest']['content_sha256'][:12]}"
    if tag != expected_tag:
        raise ValueError("release tag is not derived from the manifest content hash")
    tag_target, tag_chain = _resolve_tag(repository, tag)
    if tag_target != package["code_head"]:
        raise ValueError("immutable release tag does not target the pinned public code head")
    encoded = urllib.parse.quote(tag, safe="")
    release = _anonymous_json(f"{API_ROOT}/repos/{repository}/releases/tags/{encoded}")
    if (
        release.get("draft") is not False
        or release.get("prerelease") is not False
        or release.get("immutable") is not True
        or release.get("tag_name") != tag
    ):
        raise ValueError("GitHub release is not public, final, and immutable")
    archive_name = package["archive"]["path"]
    archive_parts = package["archive"]["parts"]
    receipt_name = package_receipt_path.name
    assets = release.get("assets")
    expected_names = {part["path"] for part in archive_parts} | {receipt_name}
    if (
        not isinstance(assets, list)
        or len(assets) != len(expected_names)
        or len({asset.get("name") for asset in assets}) != len(assets)
        or {asset.get("name") for asset in assets} != expected_names
    ):
        raise ValueError("GitHub release asset universe must be exactly archive parts plus package receipt")
    by_name = {asset["name"]: asset for asset in assets}
    expected_assets = {
        part["path"]: {"bytes": part["bytes"], "sha256": part["sha256"]}
        for part in archive_parts
    }
    expected_assets[receipt_name] = {
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
            raise ValueError(f"GitHub server-side asset digest differs: {name}")
    required_body_values = (
        package["code_head"],
        package["archive"]["sha256"],
        package["release"]["manifest"]["content_sha256"],
        package["release"]["sha256sums"]["sha256"],
        package["content_sha256"],
    )
    body = release.get("body")
    if not isinstance(body, str) or any(value not in body for value in required_body_values):
        raise ValueError("immutable release notes do not anchor all required hashes")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="crossscan-public-download-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        downloaded_receipt = temporary_root / receipt_name
        receipt_download = _download_anonymous(
            by_name[receipt_name]["browser_download_url"],
            downloaded_receipt,
            expected_assets[receipt_name]["bytes"],
        )
        if downloaded_receipt.read_bytes() != package_payload:
            raise ValueError("anonymous package-receipt download differs byte-for-byte")
        part_downloads = []
        for part in archive_parts:
            name = part["path"]
            downloaded_part = temporary_root / name
            download = _download_anonymous(
                by_name[name]["browser_download_url"],
                downloaded_part,
                expected_assets[name]["bytes"],
            )
            if download["sha256"] != part["sha256"]:
                raise ValueError(f"anonymous archive part hash differs: {name}")
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
        verify_archive(temporary_root / archive_name, package, parts_root=temporary_root)
    publication = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "PUBLIC_LOGGED_OUT_VERIFIED",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "code_head": package["code_head"],
        "tag": tag,
        "verifier": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python_version": sys.version,
            "network_policy": "dedicated-no-proxy-no-auth-no-cookie-strict-https-redirect-opener",
        },
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
            **expected_assets[receipt_name],
            "asset_id": by_name[receipt_name].get("id"),
            "asset_node_id": by_name[receipt_name].get("node_id"),
            "url": by_name[receipt_name].get("browser_download_url"),
            "created_at": by_name[receipt_name].get("created_at"),
            "updated_at": by_name[receipt_name].get("updated_at"),
            "content_type": by_name[receipt_name].get("content_type"),
            "github_digest": by_name[receipt_name].get("digest"),
            "anonymous_download": receipt_download,
        },
        "archive": {
            "path": archive_name,
            "bytes": package["archive"]["bytes"],
            "sha256": package["archive"]["sha256"],
            "parts": part_downloads,
        },
        "release_manifest_content_sha256": package["release"]["manifest"]["content_sha256"],
        "sha256sums_file_sha256": package["release"]["sha256sums"]["sha256"],
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
    pack.add_argument("--release-dir", type=Path, required=True)
    pack.add_argument("--archive", type=Path, required=True)
    pack.add_argument("--receipt", type=Path, required=True)
    pack.add_argument("--code-head", required=True)
    pack.add_argument(
        "--part-size-bytes",
        type=int,
        default=MAX_PART_BYTES,
        help=f"maximum bytes per release asset (default/max: {MAX_PART_BYTES})",
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    verify.add_argument("--release-dir", type=Path)
    public = subparsers.add_parser("verify-public")
    public.add_argument("--package-receipt", type=Path, required=True)
    public.add_argument("--repository", default=EXPECTED_REPOSITORY)
    public.add_argument("--tag", required=True)
    public.add_argument("--output", type=Path, required=True)
    public.add_argument("--expected-code-head", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "pack":
        result = pack_release(
            args.release_dir,
            args.archive,
            args.receipt,
            code_head=args.code_head,
            part_size_bytes=args.part_size_bytes,
        )
    elif args.command == "verify":
        receipt = _load_hashed_json(args.receipt, PACKAGE_SCHEMA)
        result = verify_archive(args.archive, receipt, release_dir=args.release_dir)
    else:
        result = verify_public(
            args.package_receipt,
            args.repository,
            args.tag,
            args.output,
            args.expected_code_head,
        )
    print(json.dumps({
        "status": result["status"],
        "content_sha256": result.get("content_sha256"),
        "archive_sha256": result.get("archive_sha256"),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
