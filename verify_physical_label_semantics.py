#!/usr/bin/env python3
"""Verify decoded physical-label semantics against the upstream full census.

This is intentionally stronger than a tarball or chunk hash check.  It opens
the Blosc-compressed Zarr arrays through their codec, counts every bit plane,
and requires the exact census published by the label author after Villa issue
#191 exposed a raw-compressed-byte reader failure in a downstream tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import zarr


UPSTREAM_CENSUS_COMMIT = "4277710452578e802c23244a6ab385a048aed100"
UPSTREAM_CENSUS_URL = (
    "https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit/"
    f"blob/{UPSTREAM_CENSUS_COMMIT}/results/bit_census.json"
)
CORRECTION_URL = (
    "https://github.com/ScrollPrize/villa/issues/191#issuecomment-5261207396"
)
SCHEMA = "crossscan-physical-label-semantic-audit-v1"
BITS = {
    "valid": 1,
    "material": 2,
    "centerline": 4,
    "recto_band": 8,
    "boundary_poor": 16,
}

EXPECTED = {
    "PHerc0139": {
        "path": "labels0139_L1.zarr",
        "bits": {name: BITS[name] for name in (
            "valid", "material", "centerline", "recto_band"
        )},
        "tar": {
            "path": "labels0139_L1.tar",
            "bytes": 392693760,
            "sha256": "42fe53b760c2c9347d9f215bafa68beec8e96121d03549dab56a52a9a0a9e8dd",
        },
        "shape": [1248, 2304, 2208],
        "chunks": [128, 128, 128],
        "tree": {
            "directory_count": 190,
            "file_count": 3242,
            "total_file_bytes": 390147701,
            "tree_content_sha256": "38779143c99ca166bcf2c24f5bf766451b3c508c01d90cf6cc772d2a8d3aaf66",
        },
        "counts": {
            "window_voxels": 6348865536,
            "valid": 2669177928,
            "material": 1328716163,
            "centerline": 137718669,
            "recto_band": 685890373,
            "boundary_poor": 0,
        },
        "containment": {
            "material_not_valid": 0,
            "centerline_not_material": 0,
            "recto_not_material": 204766850,
            "boundary_poor_not_material": 0,
            "boundary_poor_not_valid": 0,
        },
    },
    "PHerc1203": {
        "path": "labels1203_L1.zarr",
        "bits": dict(BITS),
        "tar": {
            "path": "labels1203_L1.tar",
            "bytes": 515379200,
            "sha256": "32a09f6081342b0f015b258ec577d0296ff23a55892af9785689d8a55bff344c",
        },
        "shape": [2016, 3456, 3456],
        "chunks": [128, 128, 128],
        "tree": {
            "directory_count": 238,
            "file_count": 3305,
            "total_file_bytes": 512731505,
            "tree_content_sha256": "745de94d4abdf2c5a4f58a18efd879995aa8ae0274aefb10c9455476db80fd8a",
        },
        "counts": {
            "window_voxels": 24078974976,
            "valid": 5019919087,
            "material": 4124552223,
            "centerline": 101660665,
            "recto_band": 567917451,
            "boundary_poor": 3077792558,
        },
        "containment": {
            "material_not_valid": 0,
            "centerline_not_material": 0,
            "recto_not_material": 156500866,
            "boundary_poor_not_material": 0,
            "boundary_poor_not_valid": 0,
        },
    },
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("ascii")).hexdigest()


def fractions_from_counts(counts: dict) -> dict[str, float]:
    """Derive the only accepted summary fractions from the exact census."""
    material = counts["material"]
    valid = counts["valid"]
    if counts["window_voxels"] <= 0 or valid <= 0 or material <= 0:
        raise ValueError("semantic census denominators must be positive")
    return {
        "valid_of_window": valid / counts["window_voxels"],
        "material_of_valid": material / valid,
        "centerline_of_material": counts["centerline"] / material,
        "recto_band_of_material": counts["recto_band"] / material,
        "boundary_poor_of_material": counts["boundary_poor"] / material,
    }


def validate_audit_receipt(path: Path) -> tuple[dict, bytes]:
    """Validate a completed audit without reopening the multi-gigabyte labels.

    The receipt is accepted only when it binds the exact pinned archives,
    extracted byte trees, codec-decoded shapes, and full published censuses.
    This is intentionally stricter than checking ``status == PASS``.
    """
    path = Path(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"semantic audit is not valid UTF-8 JSON: {path}") from error
    required_header = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "upstream_census_commit": UPSTREAM_CENSUS_COMMIT,
        "upstream_census_url": UPSTREAM_CENSUS_URL,
        "correction_url": CORRECTION_URL,
    }
    for key, expected in required_header.items():
        if value.get(key) != expected:
            raise ValueError(f"semantic audit has invalid {key}")
    if value.get("content_sha256") != content_hash(value):
        raise ValueError("semantic audit content hash mismatch")
    records = value.get("records")
    if not isinstance(records, dict) or set(records) != set(EXPECTED):
        raise ValueError("semantic audit scroll universe mismatch")
    for scroll, expected in EXPECTED.items():
        record = records[scroll]
        if not isinstance(record, dict):
            raise ValueError(f"semantic audit record is invalid: {scroll}")
        if record.get("tar") != expected["tar"]:
            raise ValueError(f"semantic audit tar identity mismatch: {scroll}")
        expected_tree = {"root": expected["path"], **expected["tree"]}
        if record.get("extracted_tree") != expected_tree:
            raise ValueError(f"semantic audit extracted tree mismatch: {scroll}")
        decoded = record.get("decoded_zarr")
        if not isinstance(decoded, dict):
            raise ValueError(f"semantic audit decoded record is invalid: {scroll}")
        exact_decoded = {
            "path": expected["path"],
            "shape": expected["shape"],
            "chunks": expected["chunks"],
            "dtype": "uint8",
            "counts": expected["counts"],
            "containment": expected["containment"],
        }
        for key, expected_value in exact_decoded.items():
            if decoded.get(key) != expected_value:
                raise ValueError(
                    f"semantic audit decoded {key} mismatch: {scroll}"
                )
        if decoded.get("fractions") != fractions_from_counts(expected["counts"]):
            raise ValueError(f"semantic audit fractions mismatch: {scroll}")
    return value, payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_tar_path(name: str, root_name: str, is_directory: bool) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError(f"unsafe tar member path: {name!r}")
    stripped = name[:-1] if is_directory and name.endswith("/") else name
    if not stripped or stripped.startswith("/") or "//" in stripped:
        raise ValueError(f"unsafe tar member path: {name!r}")
    raw_parts = stripped.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"noncanonical tar member path: {name!r}")
    path = PurePosixPath(stripped)
    parts = path.parts
    if (
        not parts
        or parts[0] != root_name
        or any(part in ("", ".", "..") or ":" in part for part in parts)
    ):
        raise ValueError(f"tar member is outside the expected root: {name!r}")
    relative = "/".join(parts[1:])
    if not relative and not is_directory:
        raise ValueError("the tar root must be a directory")
    return relative


def _stream_equal(left, right) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    while True:
        a = left.read(8 * 1024 * 1024)
        b = right.read(8 * 1024 * 1024)
        if a != b:
            raise ValueError("tar member bytes differ from extracted Zarr bytes")
        if not a:
            break
        digest.update(a)
        total += len(a)
    return total, digest.hexdigest()


def verify_tar_tree(tar_path: Path, store_path: Path, root_name: str) -> dict:
    """Bind every extracted compressed Zarr byte to an already pinned tar."""
    tar_path = Path(tar_path)
    store_path = Path(store_path)
    if store_path.name != root_name or not store_path.is_dir() or store_path.is_symlink():
        raise ValueError("extracted Zarr root is missing, renamed, or a symlink")
    root_resolved = store_path.resolve()
    tar_files: dict[str, tuple[int, str]] = {}
    tar_dirs = set()
    folded = {}
    records = []
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"tar contains a link or special member: {member.name!r}")
            relative = _canonical_tar_path(member.name, root_name, member.isdir())
            if not relative:
                continue
            if relative in tar_files or relative in tar_dirs:
                raise ValueError(f"duplicate tar member: {relative}")
            case_key = relative.casefold()
            if case_key in folded and folded[case_key] != relative:
                raise ValueError(f"case-colliding tar members: {folded[case_key]} / {relative}")
            folded[case_key] = relative
            target = store_path.joinpath(*PurePosixPath(relative).parts)
            current = store_path
            for part in PurePosixPath(relative).parts:
                current = current / part
                if current.is_symlink():
                    raise ValueError(f"extracted tree contains a symlink: {current}")
            try:
                target.resolve().relative_to(root_resolved)
            except ValueError as error:
                raise ValueError(f"extracted path escapes its root: {target}") from error
            if member.isdir():
                if not target.is_dir():
                    raise ValueError(f"missing extracted directory: {relative}")
                tar_dirs.add(relative)
                continue
            if not target.is_file():
                raise ValueError(f"missing extracted file: {relative}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot stream tar member: {relative}")
            with source, target.open("rb") as disk:
                size, digest = _stream_equal(source, disk)
            if size != member.size:
                raise ValueError(f"tar member size mismatch: {relative}")
            tar_files[relative] = (size, digest)
            records.append({"path": relative, "bytes": size, "sha256": digest})

    disk_files = set()
    disk_dirs = set()
    for current, directory_names, file_names in os.walk(store_path):
        current_path = Path(current)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"extracted tree contains a directory symlink: {path}")
            disk_dirs.add(path.relative_to(store_path).as_posix())
        for name in file_names:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"extracted tree contains a special file: {path}")
            disk_files.add(path.relative_to(store_path).as_posix())
    if disk_files != set(tar_files) or disk_dirs != tar_dirs:
        raise ValueError("extracted Zarr file/directory universe differs from pinned tar")
    records.sort(key=lambda item: item["path"])
    tree = {"root": root_name, "files": records}
    return {
        "root": root_name,
        "file_count": len(records),
        "directory_count": len(tar_dirs),
        "total_file_bytes": sum(record["bytes"] for record in records),
        "tree_content_sha256": hashlib.sha256(
            canonical_json(tree).encode("ascii")
        ).hexdigest(),
    }


def census_blocks(blocks) -> dict:
    """Count semantic bit planes over an iterable of decoded uint8 blocks."""
    counts = {name: 0 for name in (
        "window_voxels", "valid", "material", "centerline", "recto_band",
        "boundary_poor",
    )}
    containment = {name: 0 for name in (
        "material_not_valid", "centerline_not_material", "recto_not_material",
        "boundary_poor_not_material", "boundary_poor_not_valid",
    )}
    for raw in blocks:
        value = np.asarray(raw)
        if value.dtype != np.uint8 or value.ndim != 3:
            raise ValueError("decoded label blocks must be 3-D uint8")
        valid = (value & BITS["valid"]) != 0
        material_bit = (value & BITS["material"]) != 0
        centerline_bit = (value & BITS["centerline"]) != 0
        recto_bit = (value & BITS["recto_band"]) != 0
        boundary_bit = (value & BITS["boundary_poor"]) != 0
        counts["window_voxels"] += value.size
        counts["valid"] += int(np.count_nonzero(valid))
        counts["material"] += int(np.count_nonzero(material_bit))
        counts["centerline"] += int(np.count_nonzero(centerline_bit))
        counts["recto_band"] += int(np.count_nonzero(recto_bit))
        counts["boundary_poor"] += int(np.count_nonzero(boundary_bit))
        containment["material_not_valid"] += int(
            np.count_nonzero(material_bit & ~valid)
        )
        containment["centerline_not_material"] += int(
            np.count_nonzero(centerline_bit & ~material_bit)
        )
        containment["recto_not_material"] += int(
            np.count_nonzero(recto_bit & ~material_bit)
        )
        containment["boundary_poor_not_material"] += int(
            np.count_nonzero(boundary_bit & ~material_bit)
        )
        containment["boundary_poor_not_valid"] += int(
            np.count_nonzero(boundary_bit & ~valid)
        )
    return {"counts": counts, "containment": containment}


def zarr_blocks(array) -> object:
    grid = tuple(
        (size + chunk - 1) // chunk
        for size, chunk in zip(array.shape, array.chunks)
    )
    for iz in range(grid[0]):
        for iy in range(grid[1]):
            for ix in range(grid[2]):
                yield array.blocks[iz, iy, ix]


def verify_array(path: Path, expected: dict) -> dict:
    array = zarr.open_array(str(path), mode="r")
    if list(array.shape) != expected["shape"]:
        raise ValueError(f"label shape mismatch: {path}")
    if list(array.chunks) != expected["chunks"] or array.dtype != np.dtype("uint8"):
        raise ValueError(f"label chunk/dtype mismatch: {path}")
    attrs = dict(array.attrs)
    if attrs.get("bits") != expected["bits"]:
        raise ValueError(f"label bit schema mismatch: {path}")
    result = census_blocks(zarr_blocks(array))
    if result["counts"] != expected["counts"]:
        raise ValueError(f"decoded full-volume census mismatch: {path}")
    if result["containment"] != expected["containment"]:
        raise ValueError(f"decoded bit-containment census mismatch: {path}")
    material = result["counts"]["material"]
    valid = result["counts"]["valid"]
    result.update({
        "path": path.name,
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
        "compressor": str(array.compressor),
        "fractions": fractions_from_counts(result["counts"]),
    })
    return result


def verify_tar(root: Path, expected: dict) -> dict:
    path = root / expected["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {
        "path": expected["path"],
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if result["bytes"] != expected["bytes"] or result["sha256"] != expected["sha256"]:
        raise ValueError(f"source label tarball identity mismatch: {path}")
    return result


def audit(labels_root: Path) -> dict:
    labels_root = Path(labels_root)
    records = {}
    for scroll, expected in EXPECTED.items():
        tar_result = verify_tar(labels_root, expected["tar"])
        records[scroll] = {
            "tar": tar_result,
            "extracted_tree": verify_tar_tree(
                labels_root / expected["tar"]["path"],
                labels_root / expected["path"],
                expected["path"],
            ),
            "decoded_zarr": verify_array(labels_root / expected["path"], expected),
        }
    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "prove the extracted compressed Zarr bytes match pinned tarballs, "
            "then prove codec-decoded semantics against the upstream census"
        ),
        "upstream_census_commit": UPSTREAM_CENSUS_COMMIT,
        "upstream_census_url": UPSTREAM_CENSUS_URL,
        "correction_url": CORRECTION_URL,
        "records": records,
    }
    result["content_sha256"] = content_hash(result)
    return result


def write_exclusive(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.labels_root)
    write_exclusive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
