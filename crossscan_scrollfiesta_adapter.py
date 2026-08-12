#!/usr/bin/env python3
"""Export a locked cross-scan probability ROI and materialize ScrollFiesta grids.

The adapter deliberately stops at ScrollFiesta's documented ``cubes_PRED`` /
``cubes_RAW`` boundary.  It does not reimplement meshing.  Outputs are
create-no-replace and every payload is hash bound so an interrupted export can
only resume when the existing bytes agree with their receipts and the current
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


SHAPE = (256, 256, 256)
CHUNKS = (128, 128, 128)
DEFAULT_ORIGIN = (3840, 3712, 1344)
AXES = ("z", "y", "x")
SCHEMA = "crossscan-scrollfiesta-adapter-v1"


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


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _create_bytes(path: Path, payload: bytes) -> None:
    """Create *path* exclusively and persist its bytes before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _require_exact_file(path: Path, payload: bytes) -> None:
    if not path.is_file() or path.read_bytes() != payload:
        raise ValueError(f"existing metadata differs from the locked export: {path}")


def _metadata(origin: tuple[int, int, int]) -> dict[Path, bytes]:
    root_attrs = {
        "multiscales": [{
            "version": "0.4",
            "name": "crossscan_surface_probability",
            "axes": [{"name": axis, "type": "space"} for axis in AXES],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                    {"type": "translation", "translation": [float(v) for v in origin]},
                ],
            }],
        }],
        "crossscan": {
            "schema_version": SCHEMA,
            "quantity": "recto_surface_probability",
            "world_origin_l0_zyx": list(origin),
        },
    }
    zarray = {
        "zarr_format": 2,
        "shape": list(SHAPE),
        "chunks": list(CHUNKS),
        "dtype": "<f4",
        "compressor": None,
        "fill_value": 0.0,
        "order": "C",
        "filters": None,
    }
    return {
        Path(".zgroup"): _json_bytes({"zarr_format": 2}),
        Path(".zattrs"): _json_bytes(root_attrs),
        Path("0/.zarray"): _json_bytes(zarray),
        Path("0/.zattrs"): _json_bytes({"_ARRAY_DIMENSIONS": list(AXES)}),
    }


def _normalise_probability(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability)
    if value.shape != SHAPE:
        raise ValueError(f"probability shape must be {SHAPE}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError("probability input must be floating point")
    value = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(value).all():
        raise ValueError("probability input contains non-finite values")
    low = float(value.min())
    high = float(value.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"probabilities must be in [0, 1], got [{low}, {high}]")
    return value


def _chunk_records(probability: np.ndarray) -> Iterable[tuple[tuple[int, int, int], bytes]]:
    for iz, z0 in enumerate(range(0, SHAPE[0], CHUNKS[0])):
        for iy, y0 in enumerate(range(0, SHAPE[1], CHUNKS[1])):
            for ix, x0 in enumerate(range(0, SHAPE[2], CHUNKS[2])):
                chunk = np.ascontiguousarray(
                    probability[
                        z0:z0 + CHUNKS[0],
                        y0:y0 + CHUNKS[1],
                        x0:x0 + CHUNKS[2],
                    ],
                    dtype="<f4",
                )
                yield (iz, iy, ix), memoryview(chunk).cast("B").tobytes()


def _gpu_peak_bytes() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return None


def export_probability_zarr(
    probability: np.ndarray,
    output: Path,
    *,
    origin: tuple[int, int, int] = DEFAULT_ORIGIN,
    resume: bool = False,
    source_receipt: Path | None = None,
) -> dict:
    """Write an uncompressed OME-NGFF Zarr v2 ROI with chunk receipts."""
    started = time.perf_counter()
    tracemalloc.start()
    value = _normalise_probability(probability)
    input_sha = sha256_array(value)
    output = Path(output)
    metadata = _metadata(origin)
    source = None
    if source_receipt is not None:
        source_receipt = Path(source_receipt)
        if not source_receipt.is_file():
            raise FileNotFoundError(source_receipt)
        source = {
            "path": str(source_receipt.resolve()),
            "bytes": source_receipt.stat().st_size,
            "sha256": sha256_file(source_receipt),
        }

    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to replace existing output: {output}")
        if not output.is_dir():
            raise ValueError(f"output is not a directory: {output}")
        if (output / "export_receipt.json").exists():
            raise FileExistsError(f"completed output is immutable: {output}")
        for relative, payload in metadata.items():
            _require_exact_file(output / relative, payload)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for relative, payload in metadata.items():
            _create_bytes(output / relative, payload)

    expected_chunk_names = {
        f"{iz}.{iy}.{ix}"
        for iz in range(2) for iy in range(2) for ix in range(2)
    }
    actual_chunk_names = {
        path.name for path in (output / "0").iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    unexpected = actual_chunk_names - expected_chunk_names
    if unexpected:
        raise ValueError(f"unexpected Zarr chunk payloads: {sorted(unexpected)}")
    receipts_dir = output / "receipts"
    actual_receipt_names = (
        {path.name for path in receipts_dir.iterdir() if path.is_file()}
        if receipts_dir.is_dir() else set()
    )
    expected_receipt_names = {f"{name}.json" for name in expected_chunk_names}
    unexpected_receipts = actual_receipt_names - expected_receipt_names
    if unexpected_receipts:
        raise ValueError(f"unexpected chunk receipts: {sorted(unexpected_receipts)}")

    records = []
    bytes_written = 0
    resumed_chunks = 0
    for index, payload in _chunk_records(value):
        name = ".".join(str(v) for v in index)
        relative = f"0/{name}"
        chunk_path = output / relative
        receipt_path = output / "receipts" / f"{name}.json"
        expected_sha = hashlib.sha256(payload).hexdigest()
        record = {
            "index_zyx": list(index),
            "path": relative,
            "bytes": len(payload),
            "sha256": expected_sha,
            "input_probability_sha256": input_sha,
        }
        record["content_sha256"] = content_hash(record)
        receipt_payload = _json_bytes(record)
        chunk_exists = chunk_path.exists()
        receipt_exists = receipt_path.exists()
        if chunk_exists or receipt_exists:
            if not resume or not (chunk_exists and receipt_exists):
                raise ValueError(f"chunk/receipt pair is incomplete or resume is disabled: {name}")
            if receipt_path.read_bytes() != receipt_payload:
                raise ValueError(f"chunk receipt differs from current input: {receipt_path}")
            if chunk_path.stat().st_size != len(payload) or sha256_file(chunk_path) != expected_sha:
                raise ValueError(f"existing chunk does not match its receipt: {chunk_path}")
            resumed_chunks += 1
        else:
            _create_bytes(chunk_path, payload)
            _create_bytes(receipt_path, receipt_payload)
            bytes_written += len(payload) + len(receipt_payload)
        records.append(record)

    roundtrip, roundtrip_origin = read_probability_zarr(output)
    if roundtrip_origin != origin or not np.array_equal(roundtrip, value):
        raise RuntimeError("OME-Zarr round-trip verification failed")
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    receipt = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "format": "OME-NGFF Zarr v2",
        "axes": list(AXES),
        "shape": list(SHAPE),
        "chunks": list(CHUNKS),
        "dtype": "float32",
        "world_origin_l0_zyx": list(origin),
        "coordinate_scale_l0_zyx": [1.0, 1.0, 1.0],
        "input_probability_sha256": input_sha,
        "source_receipt": source,
        "chunk_records": records,
        "roundtrip_equal": True,
        "resource_measurements": {
            "wall_seconds": time.perf_counter() - started,
            "python_peak_ram_bytes": int(peak_ram),
            "peak_vram_bytes": _gpu_peak_bytes(),
            "bytes_written_this_invocation": bytes_written,
            "resumed_chunks": resumed_chunks,
        },
    }
    receipt["content_sha256"] = content_hash(receipt)
    _create_bytes(output / "export_receipt.json", _json_bytes(receipt))
    return receipt


def verify_probability_export(store: Path) -> dict:
    """Verify the immutable final receipt, metadata, and every chunk byte."""
    store = Path(store)
    receipt_path = store / "export_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("content_sha256") != content_hash(receipt):
        raise ValueError("probability export receipt content hash mismatch")
    expected_header = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "format": "OME-NGFF Zarr v2",
        "axes": list(AXES),
        "shape": list(SHAPE),
        "chunks": list(CHUNKS),
        "dtype": "float32",
        "coordinate_scale_l0_zyx": [1.0, 1.0, 1.0],
    }
    for key, expected in expected_header.items():
        if receipt.get(key) != expected:
            raise ValueError(f"probability export receipt has invalid {key}")
    origin = tuple(receipt.get("world_origin_l0_zyx", ()))
    if len(origin) != 3 or not all(isinstance(v, int) for v in origin):
        raise ValueError("probability export receipt has invalid world origin")
    for relative, payload in _metadata(origin).items():
        _require_exact_file(store / relative, payload)

    records = receipt.get("chunk_records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("probability export receipt must bind exactly eight chunks")
    expected_names = {
        f"{iz}.{iy}.{ix}"
        for iz in range(2) for iy in range(2) for ix in range(2)
    }
    actual_names = {
        path.name for path in (store / "0").iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    if actual_names != expected_names:
        raise ValueError("probability export chunk universe mismatch")
    actual_receipts = {
        path.name for path in (store / "receipts").iterdir() if path.is_file()
    }
    if actual_receipts != {f"{name}.json" for name in expected_names}:
        raise ValueError("probability export chunk-receipt universe mismatch")

    seen = set()
    for record in records:
        if record.get("content_sha256") != content_hash(record):
            raise ValueError("chunk record content hash mismatch")
        index = tuple(record.get("index_zyx", ()))
        if len(index) != 3 or any(v not in (0, 1) for v in index):
            raise ValueError("invalid chunk index in receipt")
        name = ".".join(str(v) for v in index)
        if name in seen or record.get("path") != f"0/{name}":
            raise ValueError("duplicate or invalid chunk path in receipt")
        seen.add(name)
        chunk = store / record["path"]
        sidecar = store / "receipts" / f"{name}.json"
        if json.loads(sidecar.read_text(encoding="utf-8")) != record:
            raise ValueError(f"chunk sidecar differs from final receipt: {name}")
        if chunk.stat().st_size != record.get("bytes"):
            raise ValueError(f"chunk byte count mismatch: {name}")
        if sha256_file(chunk) != record.get("sha256"):
            raise ValueError(f"chunk hash mismatch: {name}")
        if record.get("input_probability_sha256") != receipt.get(
            "input_probability_sha256"
        ):
            raise ValueError(f"chunk input binding mismatch: {name}")
    if seen != expected_names:
        raise ValueError("final receipt does not cover the exact chunk universe")
    probability, read_origin = read_probability_zarr(store)
    if read_origin != origin or sha256_array(probability) != receipt.get(
        "input_probability_sha256"
    ):
        raise ValueError("probability round-trip differs from final receipt")
    return receipt


def read_probability_zarr(store: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Read and minimally validate the adapter's deterministic Zarr layout."""
    store = Path(store)
    zarray = json.loads((store / "0/.zarray").read_text(encoding="utf-8"))
    if zarray != json.loads(_metadata(DEFAULT_ORIGIN)[Path("0/.zarray")]):
        raise ValueError("unsupported or altered probability Zarr array metadata")
    attrs = json.loads((store / ".zattrs").read_text(encoding="utf-8"))
    dataset = attrs["multiscales"][0]["datasets"][0]
    transforms = dataset["coordinateTransformations"]
    if transforms[0] != {"type": "scale", "scale": [1.0, 1.0, 1.0]}:
        raise ValueError("probability Zarr scale differs from level 0")
    translation = tuple(int(v) for v in transforms[1]["translation"])
    if list(translation) != transforms[1]["translation"]:
        raise ValueError("probability Zarr translation must be integral")
    result = np.empty(SHAPE, dtype="<f4")
    expected_bytes = int(np.prod(CHUNKS)) * np.dtype("<f4").itemsize
    for index, _ in _chunk_records(result):
        name = ".".join(str(v) for v in index)
        path = store / "0" / name
        payload = path.read_bytes()
        if len(payload) != expected_bytes:
            raise ValueError(f"invalid chunk byte count: {path}")
        chunk = np.frombuffer(payload, dtype="<f4").reshape(CHUNKS)
        z0, y0, x0 = (index[i] * CHUNKS[i] for i in range(3))
        result[
            z0:z0 + CHUNKS[0],
            y0:y0 + CHUNKS[1],
            x0:x0 + CHUNKS[2],
        ] = chunk
    return result, translation


def fixed_threshold_mask(probability: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    value = _normalise_probability(probability)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return np.ascontiguousarray(value >= threshold, dtype=np.uint8) * np.uint8(255)


def matched_mass_mask(probability: np.ndarray, foreground_count: int) -> np.ndarray:
    """Select exactly N voxels by probability, then C-order index for ties."""
    value = _normalise_probability(probability)
    total = value.size
    if not 0 <= foreground_count <= total:
        raise ValueError(f"foreground_count must be in [0, {total}]")
    flat = value.reshape(-1)
    selected = np.zeros(total, dtype=np.uint8)
    if foreground_count:
        # Stable sort preserves the original C-order index for equal values.
        order = np.argsort(-flat, kind="stable")
        selected[order[:foreground_count]] = np.uint8(255)
    return selected.reshape(SHAPE)


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def materialize_scrollfiesta_grid(
    mask: np.ndarray,
    raw_cube_dir: Path,
    output: Path,
    *,
    origin: tuple[int, int, int] = DEFAULT_ORIGIN,
    arm: str,
    probability_receipt: Path | None = None,
) -> dict:
    """Write the exact eight-cube native ScrollFiesta input grid."""
    value = np.asarray(mask)
    if value.shape != SHAPE or value.dtype != np.uint8:
        raise ValueError(f"mask must be uint8 with shape {SHAPE}")
    if not np.isin(value, np.array([0, 255], dtype=np.uint8)).all():
        raise ValueError("mask must contain only 0 and 255")
    raw_cube_dir = Path(raw_cube_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing grid: {output}")
    output.mkdir(parents=True, exist_ok=False)
    pred_dir = output / "cubes_PRED"
    raw_dir = output / "cubes_RAW"
    pred_dir.mkdir()
    raw_dir.mkdir()

    files = []
    for lz in (0, 128):
        for ly in (0, 128):
            for lx in (0, 128):
                wz, wy, wx = origin[0] + lz, origin[1] + ly, origin[2] + lx
                name = f"z{wz:05d}_y{wy:05d}_x{wx:05d}.tif"
                cube = np.ascontiguousarray(value[lz:lz + 128, ly:ly + 128, lx:lx + 128])
                pred_path = pred_dir / name
                tifffile.imwrite(
                    pred_path,
                    cube,
                    photometric="minisblack",
                    compression=None,
                    rowsperstrip=128,
                )
                raw_source = raw_cube_dir / name
                if not raw_source.is_file():
                    raise FileNotFoundError(f"missing locked RAW cube: {raw_source}")
                raw_check = tifffile.imread(raw_source)
                if raw_check.shape != (128, 128, 128) or raw_check.dtype != np.uint8:
                    raise ValueError(f"RAW cube is not uint8 128^3: {raw_source}")
                raw_path = raw_dir / name
                _copy_exclusive(raw_source, raw_path)
                for role, path in (("PRED", pred_path), ("RAW", raw_path)):
                    files.append({
                        "role": role,
                        "path": path.relative_to(output).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    })

    source = None
    if probability_receipt is not None:
        probability_receipt = Path(probability_receipt)
        source_value = json.loads(probability_receipt.read_text(encoding="utf-8"))
        if source_value.get("content_sha256") != content_hash(source_value):
            raise ValueError("probability receipt content hash mismatch")
        source = {
            "path": str(probability_receipt.resolve()),
            "bytes": probability_receipt.stat().st_size,
            "sha256": sha256_file(probability_receipt),
            "content_sha256": source_value["content_sha256"],
        }
    manifest = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "arm": arm,
        "chunk_size": 128,
        "bbox_l0_zyx": [
            origin[0], origin[0] + 256,
            origin[1], origin[1] + 256,
            origin[2], origin[2] + 256,
        ],
        "n_chunks_zyx": [2, 2, 2],
        "foreground_voxels": int(np.count_nonzero(value)),
        "probability_receipt": source,
        "files": files,
    }
    manifest["content_sha256"] = content_hash(manifest)
    _create_bytes(output / "manifest.json", _json_bytes(manifest))
    return manifest


def load_probability(path: Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            selected = key or (names[0] if len(names) == 1 else None)
            if selected is None or selected not in archive:
                raise ValueError(f"NPZ key is required; available keys: {names}")
            value = archive[selected]
    else:
        raise ValueError("probability input must be .npy or .npz")
    if value.shape == (2, *SHAPE):
        value = value[1]
    return _normalise_probability(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-zarr")
    export.add_argument("--probability", type=Path, required=True)
    export.add_argument("--npz-key")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--origin", type=int, nargs=3, default=DEFAULT_ORIGIN)
    export.add_argument("--source-receipt", type=Path)
    export.add_argument("--resume", action="store_true")

    grid = sub.add_parser("make-grid")
    grid.add_argument("--probability-zarr", type=Path, required=True)
    grid.add_argument("--raw-cubes", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument("--arm", choices=("fixed", "matched-mass"), required=True)
    grid.add_argument("--threshold", type=float, default=0.2)
    grid.add_argument("--foreground-count", type=int)
    args = parser.parse_args(argv)

    if args.command == "export-zarr":
        result = export_probability_zarr(
            load_probability(args.probability, args.npz_key),
            args.output,
            origin=tuple(args.origin),
            resume=args.resume,
            source_receipt=args.source_receipt,
        )
    else:
        verify_probability_export(args.probability_zarr)
        probability, origin = read_probability_zarr(args.probability_zarr)
        receipt = args.probability_zarr / "export_receipt.json"
        if args.arm == "fixed":
            mask = fixed_threshold_mask(probability, args.threshold)
        else:
            if args.foreground_count is None:
                parser.error("--foreground-count is required for matched-mass")
            mask = matched_mass_mask(probability, args.foreground_count)
        result = materialize_scrollfiesta_grid(
            mask,
            args.raw_cubes,
            args.output,
            origin=origin,
            arm=args.arm,
            probability_receipt=receipt,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
