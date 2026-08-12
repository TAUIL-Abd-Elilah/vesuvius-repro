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
from datetime import datetime, timezone
from pathlib import Path

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "path": str(path.resolve()),
        "shape": list(array.shape),
        "chunks": list(array.chunks),
        "dtype": str(array.dtype),
        "compressor": str(array.compressor),
        "fractions": {
            "valid_of_window": valid / result["counts"]["window_voxels"],
            "material_of_valid": material / valid,
            "centerline_of_material": result["counts"]["centerline"] / material,
            "recto_band_of_material": result["counts"]["recto_band"] / material,
            "boundary_poor_of_material": (
                result["counts"]["boundary_poor"] / material
            ),
        },
    })
    return result


def verify_tar(root: Path, expected: dict) -> dict:
    path = root / expected["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {
        "path": str(path.resolve()),
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
        records[scroll] = {
            "tar": verify_tar(labels_root, expected["tar"]),
            "decoded_zarr": verify_array(labels_root / expected["path"], expected),
        }
    result = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "prove semantic decoding of compressed physical labels, not only "
            "byte-level provenance"
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
