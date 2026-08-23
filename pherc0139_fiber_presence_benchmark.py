"""Deterministic PHerc0139 fiber-presence localization benchmark.

The scientific contract lives in ``pherc0139_fiber_presence_lock.json``.  This
runner deliberately requires either ``--plan``/``--dry-run`` or an explicit
``--run``.  A separately tagged ``--resume-transport`` path exists only for
the publicly recorded first-attempt TLS failure. Planning reads only the lock;
it does not open the prediction Zarr or fetch any prediction chunk.

``--prepare-references`` verifies and freezes the public transformed tifxyz
maps without opening any prediction.  Outcome mode accepts only that pinned
manifest, mirrors and hashes the exact required Zarr objects, and samples the
local partial arrays.  Definitive HTTP 404s use the pinned Zarr zero fill;
every other transport or integrity error aborts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import zlib
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import tifffile


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCK = ROOT / "pherc0139_fiber_presence_lock.json"
DEFAULT_OUTCOME_DIR = ROOT / "results" / "pherc0139_fiber_presence_v1"
PREREG_TAG = "pherc0139-fiber-sanity-prereg-v1"
PREREG_COMMIT = "b309c4c4d23686e14fe8b82faa6413e29789275d"
TRANSPORT_AMENDMENT_TAG = "pherc0139-fiber-sanity-transport-amendment-v1"
TRANSPORT_FAILURE_RECORD = ROOT / "pherc0139_fiber_transport_failure.json"
TRANSPORT_AMENDMENT = ROOT / "PHERC0139_FIBER_TRANSPORT_AMENDMENT.md"
REFERENCE_FILES = ("meta.json", "x.tif", "y.tif", "z.tif")
DOC_SPECS = (
    ("metadata", "metadata", "url", "content_sha256", "transport_sha256",
     "content_encoding", "metadata.json"),
    ("inference", "fiber_artifact", "inference_url", "inference_sha256", None,
     None, "inference.json"),
    ("manifest", "fiber_artifact", "manifest_url", "manifest_sha256", None,
     None, "manifest.json"),
)
CHANNEL_NAMES = ("presence", "nx", "ny")
SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class BenchmarkError(RuntimeError):
    """A contract, input-integrity, or execution error."""


@dataclass(frozen=True)
class SurfaceSample:
    """One pre-outcome sample selected from a fixed UV bin."""

    row: int
    column: int
    bin_y: int
    bin_x: int
    selection_sha256: str
    xyz: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _require_int(container: Mapping[str, Any], key: str, minimum: int = 0) -> int:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkError(f"{key!r} must be an integer >= {minimum}")
    return value


def validate_lock(lock: Mapping[str, Any]) -> None:
    """Reject malformed or internally inconsistent preregistration locks."""

    if lock.get("schema_version") != 2:
        raise BenchmarkError("unsupported lock schema_version")
    if lock.get("sample_id") != "PHerc0139":
        raise BenchmarkError("this runner is locked to PHerc0139")
    if lock.get("preregistration_tag") != PREREG_TAG:
        raise BenchmarkError(f"preregistration_tag must remain {PREREG_TAG}")

    artifact = lock.get("fiber_artifact")
    sampling = lock.get("sampling")
    gate = lock.get("primary_gate")
    if not isinstance(artifact, Mapping) or not isinstance(sampling, Mapping):
        raise BenchmarkError("lock requires fiber_artifact and sampling objects")
    if not isinstance(gate, Mapping):
        raise BenchmarkError("lock requires a primary_gate object")

    segments = lock.get("reference_segments")
    if not isinstance(segments, list) or not segments:
        raise BenchmarkError("reference_segments must be a non-empty list")
    if len(set(segments)) != len(segments):
        raise BenchmarkError("reference_segments contains duplicates")
    for segment in segments:
        if not isinstance(segment, str) or not SAFE_SEGMENT.fullmatch(segment):
            raise BenchmarkError(f"unsafe reference segment identifier: {segment!r}")

    template = lock.get("reference_url_template")
    if (not isinstance(template, str) or "{segment}" not in template
            or "{segment_short}" not in template or "{file}" not in template):
        raise BenchmarkError(
            "reference_url_template requires {segment}, {segment_short}, and {file}"
        )
    if not template.startswith("https://"):
        raise BenchmarkError("reference_url_template must use HTTPS")
    reference_type = lock.get("reference_type")
    if not isinstance(reference_type, str) or not SAFE_SEGMENT.fullmatch(reference_type):
        raise BenchmarkError("reference_type must be a safe non-empty identifier")

    for name, section, url_key, hash_key, transport_hash_key, encoding_key, _ in DOC_SPECS:
        obj = lock.get(section)
        if not isinstance(obj, Mapping):
            raise BenchmarkError(f"missing {section} object for {name}")
        url = obj.get(url_key)
        expected = obj.get(hash_key)
        if not isinstance(url, str) or not url.startswith("https://"):
            raise BenchmarkError(f"{name} URL must use HTTPS")
        if not isinstance(expected, str) or not SHA256_HEX.fullmatch(expected):
            raise BenchmarkError(f"{name} SHA-256 is not a lowercase 64-digit hex digest")
        if transport_hash_key is not None:
            transport_hash = obj.get(transport_hash_key)
            if not isinstance(transport_hash, str) or not SHA256_HEX.fullmatch(transport_hash):
                raise BenchmarkError(f"{name} transport SHA-256 is invalid")
            if obj.get(encoding_key) != "gzip":
                raise BenchmarkError(f"{name} content_encoding must be gzip")

    channels = artifact.get("channels")
    if not isinstance(channels, Mapping) or set(channels) != set(CHANNEL_NAMES):
        raise BenchmarkError("fiber_artifact.channels must contain exactly presence, nx, and ny")
    common_shape = common_chunks = None
    for channel_name in CHANNEL_NAMES:
        channel = channels[channel_name]
        if not isinstance(channel, Mapping):
            raise BenchmarkError(f"channel {channel_name} must be an object")
        shape = channel.get("shape_zyx")
        chunks = channel.get("chunks_zyx")
        if (not isinstance(shape, list) or len(shape) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 1 for v in shape)):
            raise BenchmarkError(f"{channel_name}.shape_zyx must contain three integers > 1")
        if (not isinstance(chunks, list) or len(chunks) != 3
                or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in chunks)):
            raise BenchmarkError(f"{channel_name}.chunks_zyx must contain three positive integers")
        if channel.get("dtype") != "uint8":
            raise BenchmarkError(f"{channel_name}.dtype must be uint8")
        if float(channel.get("decode_divisor", 0.0)) <= 0.0:
            raise BenchmarkError(f"{channel_name}.decode_divisor must be positive")
        if not str(channel.get("zarr_url", "")).startswith("https://"):
            raise BenchmarkError(f"{channel_name}.zarr_url must use HTTPS")
        for hash_key in ("zarray_sha256", "zattrs_sha256"):
            expected = channel.get(hash_key)
            if expected is not None and (
                not isinstance(expected, str) or not SHA256_HEX.fullmatch(expected)
            ):
                raise BenchmarkError(f"{channel_name}.{hash_key} is invalid")
        if not isinstance(channel.get("zarray_sha256"), str):
            raise BenchmarkError(f"{channel_name}.zarray_sha256 is required")
        if common_shape is None:
            common_shape, common_chunks = shape, chunks
        elif shape != common_shape or chunks != common_chunks:
            raise BenchmarkError("presence, nx, and ny must have matching shape and chunks")
    if float(channels["presence"]["decode_divisor"]) != 255.0:
        raise BenchmarkError("presence.decode_divisor must remain 255")
    for channel_name in ("nx", "ny"):
        if float(channels[channel_name].get("decode_offset", math.nan)) != 128.0:
            raise BenchmarkError(f"{channel_name}.decode_offset must remain 128")
        if float(channels[channel_name]["decode_divisor"]) != 127.0:
            raise BenchmarkError(f"{channel_name}.decode_divisor must remain 127")

    bins_y = _require_int(sampling, "uv_bins_y", 1)
    bins_x = _require_int(sampling, "uv_bins_x", 1)
    if (bins_y, bins_x) != (16, 32):
        raise BenchmarkError("this preregistration requires the fixed 16x32 UV grid")
    if _require_int(sampling, "minimum_samples_per_segment", 1) != 128:
        raise BenchmarkError("this preregistration requires 128 complete profiles per segment")
    panel_count = _require_int(sampling, "visual_panel_count", 1)
    if panel_count != 12:
        raise BenchmarkError("this preregistration requires exactly 12 visual panels")
    if sampling.get("normal_neighbor_stride") != 1:
        raise BenchmarkError("only one-pixel central-difference normals are preregistered")
    seed = sampling.get("seed")
    if not isinstance(seed, str) or not seed:
        raise BenchmarkError("sampling seed must be a non-empty string")

    offsets = sampling.get("normal_offsets_prediction_voxels")
    if (not isinstance(offsets, list) or len(offsets) != 7
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in offsets)):
        raise BenchmarkError("normal offsets must be the seven locked numeric offsets")
    offsets_f = [float(v) for v in offsets]
    if offsets_f != [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]:
        raise BenchmarkError("normal offsets do not match the preregistered sequence")

    base_scale = float(sampling.get("reference_coordinate_scale_to_base", 0.0))
    prediction_scale = float(lock.get("prediction_scale_in_base_voxels", 0.0))
    coordinate_scale = float(sampling.get("reference_coordinate_scale_to_prediction", 0.0))
    if prediction_scale <= 0.0 or not math.isclose(
        coordinate_scale, base_scale / prediction_scale, rel_tol=0.0, abs_tol=1e-15
    ):
        raise BenchmarkError("reference-to-prediction scale is inconsistent")
    if _require_int(gate, "minimum_analyzable_segments", 1) != 30:
        raise BenchmarkError("minimum_analyzable_segments must remain frozen at 30")
    if _require_int(gate, "minimum_positive_segments", 1) != 24:
        raise BenchmarkError("minimum_positive_segments must remain frozen at 24")
    min_delta = float(gate.get("minimum_median_segment_delta", -math.inf))
    if not math.isfinite(min_delta):
        raise BenchmarkError("invalid primary median threshold")
    if not math.isclose(min_delta, 0.02, rel_tol=0.0, abs_tol=1e-15):
        raise BenchmarkError("minimum_median_segment_delta must remain frozen at 0.02")

    orientation = lock.get("orientation_analysis")
    if not isinstance(orientation, Mapping):
        raise BenchmarkError("orientation_analysis must be an object")
    expected_orientation = {
        "center_presence_subset_threshold": 0.5,
        "minimum_positive_weight_points_per_segment": 32,
        "minimum_analyzable_segments": 30,
        "maximum_global_median_segment_presence_weighted_tangent_angle_degrees": 20.0,
        "minimum_segments_improving_over_matched_baseline": 24,
        "minimum_segment_improvement_degrees": 5.0,
    }
    for key, expected in expected_orientation.items():
        observed = orientation.get(key)
        if isinstance(expected, int):
            if observed != expected:
                raise BenchmarkError(f"orientation_analysis.{key} must remain {expected}")
        elif not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-15):
            raise BenchmarkError(f"orientation_analysis.{key} must remain {expected}")
    if orientation.get("matched_baseline") != "globally hash-sorted surface normals rotated by floor(N/2)":
        raise BenchmarkError("unexpected matched orientation baseline")

    missing_policy = lock.get("missing_chunk_policy")
    if not isinstance(missing_policy, Mapping):
        raise BenchmarkError("missing_chunk_policy must be an object")
    if missing_policy.get("presence_fill_value") != 0:
        raise BenchmarkError("presence_fill_value must remain zero")
    if missing_policy.get("direction_fill_value") != 0:
        raise BenchmarkError("direction_fill_value must remain zero")
    if missing_policy.get("http_404") != (
        "record as the pinned Zarr fill value rather than a transport failure"
    ):
        raise BenchmarkError("unexpected HTTP 404 policy")

    reference_manifest = lock.get("reference_manifest")
    if not isinstance(reference_manifest, Mapping):
        raise BenchmarkError("reference_manifest must be an object")
    manifest_path = reference_manifest.get("path")
    if not isinstance(manifest_path, str) or not manifest_path or Path(manifest_path).is_absolute():
        raise BenchmarkError("reference_manifest.path must be a non-empty relative path")
    manifest_hash = reference_manifest.get("sha256")
    if manifest_hash is not None and (
        not isinstance(manifest_hash, str) or not SHA256_HEX.fullmatch(manifest_hash)
    ):
        raise BenchmarkError("reference_manifest.sha256 must be null or a SHA-256 digest")

    visuals = lock.get("visual_segments")
    if not isinstance(visuals, list) or len(visuals) != 12:
        raise BenchmarkError("visual_segments must contain exactly 12 frozen entries")
    expected_visuals = visual_segment_order(sampling["seed"], segments, 12)
    if visuals != expected_visuals:
        raise BenchmarkError("visual_segments do not match lowest SHA256(seed|visual|segment)")


def load_lock(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        lock = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid lock JSON: {exc}") from exc
    if not isinstance(lock, dict):
        raise BenchmarkError("lock root must be a JSON object")
    validate_lock(lock)
    return lock, {"path": path.name, "bytes": len(raw), "sha256": sha256_bytes(raw)}


def reference_contract_sha256(lock: Mapping[str, Any]) -> str:
    """Hash reference choices while excluding the post-prepare freeze fields."""

    contract = dict(lock)
    contract.pop("reference_manifest", None)
    contract.pop("frozen_at_utc", None)
    return sha256_bytes(canonical_json_bytes(contract))


def selection_key(seed: str, segment: str, row: int, column: int) -> bytes:
    return hashlib.sha256(f"{seed}|{segment}|{row}|{column}".encode("utf-8")).digest()


def bin_cycle(seed: str, segment: str, bin_y: int, bin_x: int, size: int) -> tuple[int, int]:
    """Return the frozen full-cycle start/stride for one row-major UV bin."""

    if size <= 0:
        raise ValueError("bin size must be positive")
    digest = hashlib.sha256(f"{seed}|{segment}|{bin_y}|{bin_x}".encode("utf-8")).digest()
    start = int.from_bytes(digest[:8], "big") % size
    if size == 1:
        return start, 1
    stride = int.from_bytes(digest[8:16], "big") % size
    if stride == 0:
        stride = 1
    while math.gcd(stride, size) != 1:
        stride += 1
        if stride >= size:
            stride = 1
    return start, stride


def visual_segment_order(seed: str, segments: Sequence[str], count: int) -> list[dict[str, str]]:
    ranked = []
    for segment in segments:
        digest = hashlib.sha256(f"{seed}|visual|{segment}".encode("utf-8")).hexdigest()
        ranked.append((digest, segment))
    ranked.sort()
    return [{"segment": segment, "selection_sha256": digest}
            for digest, segment in ranked[:count]]


def reference_url(lock: Mapping[str, Any], segment: str, filename: str) -> str:
    segment_short = segment.split("-", 1)[0]
    return str(lock["reference_url_template"]).format(
        segment=segment, segment_short=segment_short, file=filename
    )


def reference_segment_cache_dir(
    lock: Mapping[str, Any], cache_root: Path, segment: str
) -> Path:
    del lock  # the manifest binds each cached byte to its transformed public URL
    return cache_root / "segments" / segment


def build_plan(lock: Mapping[str, Any], lock_receipt: Mapping[str, Any]) -> dict[str, Any]:
    sampling = lock["sampling"]
    documents = []
    for name, section, url_key, hash_key, transport_hash_key, encoding_key, cache_name in DOC_SPECS:
        obj = lock[section]
        document = {
            "name": name,
            "url": obj[url_key],
            "expected_content_sha256": obj[hash_key],
            "cache_path": f"documents/{cache_name}",
        }
        if transport_hash_key is not None:
            document["expected_transport_sha256"] = obj[transport_hash_key]
            document["content_encoding"] = obj[encoding_key]
        documents.append(document)
    references = [
        {
            "segment": segment,
            "files": [{"name": filename, "url": reference_url(lock, segment, filename)}
                      for filename in REFERENCE_FILES],
        }
        for segment in lock["reference_segments"]
    ]
    return {
        "schema_version": 2,
        "mode": "PLAN_ONLY",
        "experiment_id": lock["experiment_id"],
        "lock": dict(lock_receipt),
        "prediction_array_opened": False,
        "prediction_chunk_reads": False,
        "documents_to_verify": documents,
        "reference_downloads": references,
        "segment_count": len(references),
        "reference_file_count": len(references) * len(REFERENCE_FILES),
        "reference_contract_sha256": reference_contract_sha256(lock),
        "reference_manifest": lock["reference_manifest"],
        "sampling": sampling,
        "primary_gate": lock["primary_gate"],
        "orientation_analysis": lock["orientation_analysis"],
        "predetermined_visual_segments": lock["visual_segments"],
    }


def _receipt(path: Path, url: str, cache_root: Path, expected_sha256: str | None) -> dict[str, Any]:
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise BenchmarkError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256}, received {actual}"
        )
    return {
        "url": url,
        "cache_path": path.relative_to(cache_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": actual,
        **({"expected_sha256": expected_sha256} if expected_sha256 is not None else {}),
    }


def download_cached(
    url: str,
    path: Path,
    cache_root: Path,
    expected_sha256: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Fetch an immutable public object once and always return a content receipt."""

    if not url.startswith("https://"):
        raise BenchmarkError(f"refusing non-HTTPS input URL: {url}")
    if path.is_file():
        return _receipt(path, url, cache_root, expected_sha256)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "pherc0139-fiber-benchmark/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                digest.update(block)
                size += len(block)
            out.flush()
            os.fsync(out.fileno())
        actual = digest.hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise BenchmarkError(
                f"SHA-256 mismatch for {url}: expected {expected_sha256}, received {actual}"
            )
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    if path.stat().st_size != size:
        raise BenchmarkError(f"cached byte count changed after download: {url}")
    return _receipt(path, url, cache_root, expected_sha256)


def decode_transport_payload(payload: bytes, content_encoding: str | None = None) -> tuple[bytes, str]:
    """Decode a pinned HTTP object while preserving its transport-byte receipt."""

    is_gzip = payload.startswith(b"\x1f\x8b")
    if content_encoding == "gzip" and not is_gzip:
        raise BenchmarkError("expected gzip transport bytes but gzip magic is absent")
    if is_gzip:
        try:
            return gzip.decompress(payload), "gzip"
        except (OSError, EOFError) as exc:
            raise BenchmarkError(f"invalid gzip transport body: {exc}") from exc
    return payload, "identity"


def verify_small_documents(
    lock: Mapping[str, Any], cache_root: Path, timeout_seconds: float
) -> list[dict[str, Any]]:
    receipts = []
    for (name, section, url_key, hash_key, transport_hash_key,
         encoding_key, cache_name) in DOC_SPECS:
        obj = lock[section]
        content_path = cache_root / "documents" / cache_name
        transport_path = cache_root / "documents" / "transport" / cache_name
        expected_transport = obj.get(transport_hash_key) if transport_hash_key else None
        download_cached(
            obj[url_key], transport_path, cache_root, expected_transport, timeout_seconds
        )
        transport = transport_path.read_bytes()
        content, observed_encoding = decode_transport_payload(
            transport, obj.get(encoding_key) if encoding_key else None
        )
        content_hash = sha256_bytes(content)
        if content_hash != obj[hash_key]:
            raise BenchmarkError(
                f"content SHA-256 mismatch for {name}: expected {obj[hash_key]}, got {content_hash}"
            )
        if content_path.exists() and content_path.read_bytes() != content:
            raise BenchmarkError(f"cached decoded content disagrees with transport for {name}")
        if not content_path.exists():
            atomic_write(content_path, content)
        try:
            json.loads(content)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"hash-valid {name} document is not valid JSON: {exc}") from exc
        receipts.append({
            "name": name,
            "url": obj[url_key],
            "content_encoding": observed_encoding,
            "content_cache_path": content_path.relative_to(cache_root).as_posix(),
            "content_bytes": len(content),
            "content_sha256": content_hash,
            "expected_content_sha256": obj[hash_key],
            "transport_cache_path": transport_path.relative_to(cache_root).as_posix(),
            "transport_bytes": len(transport),
            "transport_sha256": sha256_bytes(transport),
            **({"expected_transport_sha256": expected_transport}
               if expected_transport is not None else {}),
        })
    return receipts


def fetch_reference_segment(
    lock: Mapping[str, Any], segment: str, cache_root: Path, timeout_seconds: float
) -> dict[str, Any]:
    segment_dir = reference_segment_cache_dir(lock, cache_root, segment)
    receipts = []
    for filename in REFERENCE_FILES:
        receipts.append({
            "name": filename,
            **download_cached(
                reference_url(lock, segment, filename),
                segment_dir / filename,
                cache_root,
                None,
                timeout_seconds,
            ),
        })
    try:
        metadata = json.loads((segment_dir / "meta.json").read_bytes())
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid meta.json for {segment}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise BenchmarkError(f"meta.json for {segment} is not a JSON object")
    if metadata.get("format") != "tifxyz" or metadata.get("type") != "seg":
        raise BenchmarkError(f"unexpected reference metadata for {segment}")
    bbox = metadata.get("bbox")
    if (not isinstance(bbox, list) or len(bbox) != 2
            or any(not isinstance(corner, list) or len(corner) != 3 for corner in bbox)):
        raise BenchmarkError(f"invalid reference bbox for {segment}")
    try:
        bbox_values = [float(value) for corner in bbox for value in corner]
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"non-numeric reference bbox for {segment}") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in bbox_values):
        raise BenchmarkError(f"invalid reference bbox values for {segment}")
    if any(float(bbox[0][axis]) > float(bbox[1][axis]) for axis in range(3)):
        raise BenchmarkError(f"reversed reference bbox for {segment}")
    return {"segment": segment, "files": receipts}


def fetch_all_references(
    lock: Mapping[str, Any],
    cache_root: Path,
    timeout_seconds: float,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Download references concurrently but return everything in locked order."""

    segments = list(lock["reference_segments"])
    successes: dict[str, dict[str, Any]] = {}
    failures: dict[str, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(fetch_reference_segment, lock, segment, cache_root, timeout_seconds): segment
            for segment in segments
        }
        for future in as_completed(futures):
            segment = futures[future]
            try:
                successes[segment] = future.result()
            except Exception as exc:  # failed segments are frozen failures, never replaced
                failures[segment] = {"type": type(exc).__name__, "message": str(exc)}
    return [successes[s] for s in segments if s in successes], failures


def load_xyz_maps(segment_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    planes = []
    for axis in "xyz":
        raw = tifffile.imread(segment_dir / f"{axis}.tif")
        if raw.dtype != np.dtype("float32"):
            raise BenchmarkError(f"{segment_dir.name}/{axis}.tif is not float32")
        value = np.asarray(raw, dtype=np.float32)
        if value.ndim != 2:
            raise BenchmarkError(f"{segment_dir.name}/{axis}.tif is not a 2-D raster")
        planes.append(value)
    if not (planes[0].shape == planes[1].shape == planes[2].shape):
        raise BenchmarkError(f"coordinate TIFF shapes differ for {segment_dir.name}")
    if min(planes[0].shape) < 3:
        raise BenchmarkError(f"coordinate raster is too small for central differences: {segment_dir.name}")
    return planes[0], planes[1], planes[2]


def _coordinate_planes(
    xyz: np.ndarray | Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(xyz, np.ndarray):
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape (rows, columns, 3)")
        return xyz[..., 0], xyz[..., 1], xyz[..., 2]
    if len(xyz) != 3:
        raise ValueError("xyz plane sequence must contain x, y, and z")
    x, y, z = (np.asarray(v) for v in xyz)
    if x.ndim != 2 or x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z planes must be matching 2-D arrays")
    return x, y, z


def valid_coordinate_mask(xyz: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    x, y, z = _coordinate_planes(xyz)
    return (np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            & (x >= 0.0) & (y >= 0.0) & (z >= 0.0))


def eligible_normal_mask(xyz: np.ndarray | Sequence[np.ndarray]) -> np.ndarray:
    """Vectorized four-neighbour and non-degenerate-normal eligibility mask."""

    x, y, z = _coordinate_planes(xyz)
    valid = valid_coordinate_mask((x, y, z))
    eligible = np.zeros_like(valid)
    neighbor_valid = (
        valid[1:-1, 1:-1]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
    )

    # Float32 is exact enough for differences between base-voxel coordinates
    # and keeps the peak temporary footprint bounded for 1--3M-pixel maps.
    d_row = np.empty((*neighbor_valid.shape, 3), dtype=np.float32)
    d_column = np.empty_like(d_row)
    for axis, plane in enumerate((x, y, z)):
        d_row[..., axis] = plane[2:, 1:-1] - plane[:-2, 1:-1]
        d_column[..., axis] = plane[1:-1, 2:] - plane[1:-1, :-2]
    cross = np.cross(d_row, d_column)
    norm_squared = np.einsum("...i,...i->...", cross, cross, optimize=True)
    eligible[1:-1, 1:-1] = (
        neighbor_valid & np.isfinite(norm_squared) & (norm_squared > 1e-24)
    )
    return eligible


def central_difference_normal(
    xyz: np.ndarray | Sequence[np.ndarray], row: int, column: int
) -> np.ndarray | None:
    """Return the deterministic unit normal in XYZ order, or None if invalid."""

    x, y, z = _coordinate_planes(xyz)
    rows, columns = x.shape
    if row <= 0 or column <= 0 or row >= rows - 1 or column >= columns - 1:
        return None
    valid = valid_coordinate_mask((
        np.asarray([[x[row, column], x[row - 1, column], x[row + 1, column],
                     x[row, column - 1], x[row, column + 1]]]),
        np.asarray([[y[row, column], y[row - 1, column], y[row + 1, column],
                     y[row, column - 1], y[row, column + 1]]]),
        np.asarray([[z[row, column], z[row - 1, column], z[row + 1, column],
                     z[row, column - 1], z[row, column + 1]]]),
    ))
    if not bool(valid.all()):
        return None
    return _central_difference_normal_unchecked(x, y, z, row, column)


def _central_difference_normal_unchecked(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, row: int, column: int
) -> np.ndarray | None:
    """Normal helper for callers that already proved all five coordinates valid."""

    d_row = 0.5 * np.array([
        x[row + 1, column] - x[row - 1, column],
        y[row + 1, column] - y[row - 1, column],
        z[row + 1, column] - z[row - 1, column],
    ], dtype=np.float64)
    d_column = 0.5 * np.array([
        x[row, column + 1] - x[row, column - 1],
        y[row, column + 1] - y[row, column - 1],
        z[row, column + 1] - z[row, column - 1],
    ], dtype=np.float64)
    normal = np.cross(d_row, d_column)
    magnitude = float(np.linalg.norm(normal))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        return None
    return normal / magnitude


def select_surface_samples(
    xyz: np.ndarray | Sequence[np.ndarray],
    segment: str,
    seed: str,
    bins_y: int,
    bins_x: int,
    eligibility: np.ndarray | None = None,
) -> list[SurfaceSample]:
    """Choose the first eligible point on each bin's deterministic full cycle."""

    x, y, z = _coordinate_planes(xyz)
    rows, columns = x.shape
    if eligibility is None:
        eligibility = eligible_normal_mask((x, y, z))
    else:
        eligibility = np.asarray(eligibility, dtype=bool)
        if eligibility.shape != (rows, columns):
            raise ValueError("eligibility mask shape differs from coordinate rasters")

    selected: list[SurfaceSample] = []
    for bin_y in range(bins_y):
        row_start = max(1, (bin_y * rows) // bins_y)
        row_stop = min(rows - 1, ((bin_y + 1) * rows) // bins_y)
        if row_stop <= row_start:
            continue
        for bin_x in range(bins_x):
            column_start = max(1, (bin_x * columns) // bins_x)
            column_stop = min(columns - 1, ((bin_x + 1) * columns) // bins_x)
            if column_stop <= column_start:
                continue
            bin_width = column_stop - column_start
            size = (row_stop - row_start) * bin_width
            start, stride = bin_cycle(seed, segment, bin_y, bin_x, size)
            chosen: tuple[int, int, np.ndarray] | None = None
            flat = start
            for _ in range(size):
                row = row_start + flat // bin_width
                column = column_start + flat % bin_width
                if eligibility[row, column]:
                    normal = _central_difference_normal_unchecked(x, y, z, row, column)
                    if normal is not None:
                        chosen = (row, column, normal)
                        break
                flat = (flat + stride) % size
            if chosen is None:
                continue
            row, column, normal = chosen
            selected.append(SurfaceSample(
                row=row,
                column=column,
                bin_y=bin_y,
                bin_x=bin_x,
                selection_sha256=selection_key(seed, segment, row, column).hex(),
                xyz=(float(x[row, column]), float(y[row, column]), float(z[row, column])),
                normal_xyz=(float(normal[0]), float(normal[1]), float(normal[2])),
            ))
    return selected


def selected_samples_sha256(samples: Sequence[SurfaceSample]) -> str:
    payload = [surface_sample_dict(sample) for sample in samples]
    return sha256_bytes(canonical_json_bytes(payload))


def surface_sample_dict(sample: SurfaceSample) -> dict[str, Any]:
    return {
        "bin_x": sample.bin_x,
        "bin_y": sample.bin_y,
        "column": sample.column,
        "normal_xyz": list(sample.normal_xyz),
        "row": sample.row,
        "selection_sha256": sample.selection_sha256,
        "xyz": list(sample.xyz),
    }


def surface_sample_from_dict(
    value: Mapping[str, Any], seed: str, segment: str, bins_y: int, bins_x: int
) -> SurfaceSample:
    try:
        sample = SurfaceSample(
            row=int(value["row"]),
            column=int(value["column"]),
            bin_y=int(value["bin_y"]),
            bin_x=int(value["bin_x"]),
            selection_sha256=str(value["selection_sha256"]),
            xyz=tuple(float(v) for v in value["xyz"]),  # type: ignore[arg-type]
            normal_xyz=tuple(float(v) for v in value["normal_xyz"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BenchmarkError(f"malformed frozen sample for {segment}: {exc}") from exc
    if len(sample.xyz) != 3 or len(sample.normal_xyz) != 3:
        raise BenchmarkError(f"sample vector length is not three for {segment}")
    if not 0 <= sample.bin_y < bins_y or not 0 <= sample.bin_x < bins_x:
        raise BenchmarkError(f"sample bin is outside the frozen grid for {segment}")
    if sample.row < 1 or sample.column < 1:
        raise BenchmarkError(f"sample is outside the one-pixel interior for {segment}")
    if not all(math.isfinite(v) and v >= 0.0 for v in sample.xyz):
        raise BenchmarkError(f"sample has invalid coordinates for {segment}")
    if not all(math.isfinite(v) for v in sample.normal_xyz):
        raise BenchmarkError(f"sample has invalid normal for {segment}")
    if not math.isclose(
        math.sqrt(sum(v * v for v in sample.normal_xyz)), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise BenchmarkError(f"sample normal is not unit length for {segment}")
    expected = selection_key(seed, segment, sample.row, sample.column).hex()
    if sample.selection_sha256 != expected:
        raise BenchmarkError(f"sample selection hash changed for {segment}")
    return sample


def _coordinate_inventory(
    xyz: Sequence[np.ndarray], valid: np.ndarray
) -> tuple[int, dict[str, list[float]]]:
    count = int(np.count_nonzero(valid))
    if count == 0:
        raise BenchmarkError("coordinate raster has no finite non-negative points")
    minimum = [float(np.min(plane[valid])) for plane in xyz]
    maximum = [float(np.max(plane[valid])) for plane in xyz]
    return count, {"minimum_xyz": minimum, "maximum_xyz": maximum}


def reference_manifest_path(lock: Mapping[str, Any], lock_path: Path) -> Path:
    relative = Path(str(lock["reference_manifest"]["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkError("reference_manifest.path must stay below the lock directory")
    return lock_path.resolve().parent / relative


def prepare_reference_manifest(
    lock: Mapping[str, Any],
    lock_path: Path,
    cache_root: Path,
    timeout_seconds: float,
    download_workers: int,
) -> tuple[dict[str, Any], Path]:
    """Freeze every reference and sample without touching a prediction object."""

    documents = verify_small_documents(lock, cache_root, timeout_seconds)
    reference_receipts, failures = fetch_all_references(
        lock, cache_root, timeout_seconds, download_workers
    )
    if failures:
        raise BenchmarkError(
            "reference preparation aborted; every locked segment is required: "
            + json.dumps(failures, sort_keys=True)
        )
    receipts_by_segment = {entry["segment"]: entry for entry in reference_receipts}
    prepared_segments = []
    sampling = lock["sampling"]
    for segment in lock["reference_segments"]:
        xyz = load_xyz_maps(reference_segment_cache_dir(lock, cache_root, segment))
        valid = valid_coordinate_mask(xyz)
        eligibility = eligible_normal_mask(xyz)
        samples = select_surface_samples(
            xyz,
            segment,
            sampling["seed"],
            int(sampling["uv_bins_y"]),
            int(sampling["uv_bins_x"]),
            eligibility,
        )
        valid_count, bbox = _coordinate_inventory(xyz, valid)
        metadata = json.loads(
            (reference_segment_cache_dir(lock, cache_root, segment) / "meta.json").read_bytes()
        )
        metadata_bbox = np.asarray(metadata["bbox"], dtype=np.float64)
        observed_bbox = np.asarray(
            [bbox["minimum_xyz"], bbox["maximum_xyz"]], dtype=np.float64
        )
        if not np.allclose(metadata_bbox, observed_bbox, rtol=0.0, atol=1e-6):
            raise BenchmarkError(f"meta.json bbox disagrees with TIFF coordinates for {segment}")
        prepared_segments.append({
            "segment": segment,
            "files": receipts_by_segment[segment]["files"],
            "raster_shape_rc": [int(v) for v in xyz[0].shape],
            "valid_coordinate_points": valid_count,
            "eligible_normal_points": int(np.count_nonzero(eligibility)),
            "coordinate_bbox": bbox,
            "selected_samples": len(samples),
            "selection_sha256": selected_samples_sha256(samples),
            "samples": [surface_sample_dict(sample) for sample in samples],
        })

    manifest = {
        "schema_version": 1,
        "mode": "REFERENCE_PREPARATION",
        "experiment_id": lock["experiment_id"],
        "reference_contract_sha256": reference_contract_sha256(lock),
        "verified_documents": documents,
        "reference_url_template": lock["reference_url_template"],
        "sampling": lock["sampling"],
        "selection_algorithm": (
            "per bin SHA256(seed|segment|bin_y|bin_x): first 8 digest bytes select start; "
            "next 8 select stride, advanced with wrap to the first coprime value; choose the "
            "first eligible row-major bin position on the full cycle"
        ),
        "visual_segments": lock["visual_segments"],
        "segments": prepared_segments,
        "prediction_array_opened": False,
        "prediction_chunk_reads": False,
    }
    path = reference_manifest_path(lock, lock_path)
    atomic_write(path, canonical_json_bytes(manifest))
    return manifest, path


def _verified_cached_path(
    cache_root: Path, relative_path: str, expected_bytes: int, expected_sha256: str
) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkError(f"unsafe cache path in reference manifest: {relative_path}")
    path = cache_root.resolve() / relative
    if not path.is_file():
        raise BenchmarkError(f"pinned cached input is missing: {relative_path}")
    if path.stat().st_size != int(expected_bytes):
        raise BenchmarkError(f"pinned cached input byte count changed: {relative_path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise BenchmarkError(f"pinned cached input hash changed: {relative_path}")
    return path


def load_pinned_reference_manifest(
    lock: Mapping[str, Any], lock_path: Path, cache_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[SurfaceSample]]]:
    """Verify the frozen manifest and every cached byte before outcome start."""

    frozen_at = lock.get("frozen_at_utc")
    if not isinstance(frozen_at, str) or not frozen_at:
        raise BenchmarkError("outcome requires a non-null frozen_at_utc")
    expected_hash = lock["reference_manifest"].get("sha256")
    if not isinstance(expected_hash, str) or not SHA256_HEX.fullmatch(expected_hash):
        raise BenchmarkError("outcome requires a pinned reference_manifest.sha256")
    path = reference_manifest_path(lock, lock_path)
    if not path.is_file():
        raise BenchmarkError(f"pinned reference manifest is missing: {path}")
    raw = path.read_bytes()
    actual_hash = sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise BenchmarkError(
            f"reference manifest hash changed: expected {expected_hash}, got {actual_hash}"
        )
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid pinned reference manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise BenchmarkError("unsupported reference manifest schema")
    if manifest.get("experiment_id") != lock["experiment_id"]:
        raise BenchmarkError("reference manifest experiment_id changed")
    if manifest.get("reference_contract_sha256") != reference_contract_sha256(lock):
        raise BenchmarkError("lock reference choices changed after reference preparation")
    if manifest.get("prediction_array_opened") is not False or manifest.get("prediction_chunk_reads") is not False:
        raise BenchmarkError("reference manifest is not prediction-free")
    if manifest.get("visual_segments") != lock["visual_segments"]:
        raise BenchmarkError("reference manifest visual IDs differ from the lock")

    documents = manifest.get("verified_documents")
    if not isinstance(documents, list) or len(documents) != len(DOC_SPECS):
        raise BenchmarkError("reference manifest document receipts are incomplete")
    document_specs_by_name = {spec[0]: spec for spec in DOC_SPECS}
    if {receipt.get("name") for receipt in documents if isinstance(receipt, Mapping)} != set(
        document_specs_by_name
    ):
        raise BenchmarkError("reference manifest document names changed")
    for receipt in documents:
        if not isinstance(receipt, Mapping):
            raise BenchmarkError("malformed document receipt")
        name = str(receipt["name"])
        _, section, url_key, hash_key, transport_hash_key, _, _ = document_specs_by_name[name]
        lock_document = lock[section]
        if receipt.get("url") != lock_document[url_key]:
            raise BenchmarkError(f"pinned document URL changed for {name}")
        if receipt.get("content_sha256") != lock_document[hash_key]:
            raise BenchmarkError(f"pinned document content hash changed for {name}")
        if (transport_hash_key is not None
                and receipt.get("transport_sha256") != lock_document[transport_hash_key]):
            raise BenchmarkError(f"pinned document transport hash changed for {name}")
        content_path = _verified_cached_path(
            cache_root,
            str(receipt["content_cache_path"]),
            int(receipt["content_bytes"]),
            str(receipt["content_sha256"]),
        )
        _verified_cached_path(
            cache_root,
            str(receipt["transport_cache_path"]),
            int(receipt["transport_bytes"]),
            str(receipt["transport_sha256"]),
        )
        try:
            json.loads(content_path.read_bytes())
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"cached pinned document is invalid JSON: {exc}") from exc

    segment_entries = manifest.get("segments")
    if not isinstance(segment_entries, list):
        raise BenchmarkError("reference manifest segments must be a list")
    observed_order = [entry.get("segment") for entry in segment_entries if isinstance(entry, Mapping)]
    if observed_order != lock["reference_segments"]:
        raise BenchmarkError("reference manifest segment universe or order changed")
    sampling = lock["sampling"]
    samples_by_segment: dict[str, list[SurfaceSample]] = {}
    for entry in segment_entries:
        if not isinstance(entry, Mapping):
            raise BenchmarkError("malformed reference segment entry")
        segment = str(entry["segment"])
        files = entry.get("files")
        if not isinstance(files, list) or [item.get("name") for item in files] != list(REFERENCE_FILES):
            raise BenchmarkError(f"reference receipts are incomplete for {segment}")
        for item in files:
            if item.get("url") != reference_url(lock, segment, str(item.get("name"))):
                raise BenchmarkError(f"reference URL changed for {segment}/{item.get('name')}")
            _verified_cached_path(
                cache_root,
                str(item["cache_path"]),
                int(item["bytes"]),
                str(item["sha256"]),
            )
        shape = entry.get("raster_shape_rc")
        if (not isinstance(shape, list) or len(shape) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 3 for v in shape)):
            raise BenchmarkError(f"invalid raster shape for {segment}")
        raw_samples = entry.get("samples")
        if not isinstance(raw_samples, list):
            raise BenchmarkError(f"missing frozen samples for {segment}")
        samples = [surface_sample_from_dict(
            item,
            sampling["seed"],
            segment,
            int(sampling["uv_bins_y"]),
            int(sampling["uv_bins_x"]),
        ) for item in raw_samples]
        if len(samples) != int(entry.get("selected_samples", -1)):
            raise BenchmarkError(f"frozen selected sample count changed for {segment}")
        if len({(sample.bin_y, sample.bin_x) for sample in samples}) != len(samples):
            raise BenchmarkError(f"multiple samples occupy a UV bin for {segment}")
        if any(sample.row >= shape[0] - 1 or sample.column >= shape[1] - 1 for sample in samples):
            raise BenchmarkError(f"frozen sample is outside raster interior for {segment}")
        if selected_samples_sha256(samples) != entry.get("selection_sha256"):
            raise BenchmarkError(f"frozen sample list hash changed for {segment}")
        samples_by_segment[segment] = samples

    receipt = {
        "path": str(lock["reference_manifest"]["path"]),
        "bytes": len(raw),
        "sha256": actual_hash,
    }
    return manifest, receipt, samples_by_segment


def build_normal_queries(
    samples: Sequence[SurfaceSample], coordinate_scale: float, offsets: Sequence[float]
) -> np.ndarray:
    """Return queries in prediction-array ZYX order, shape (sample, offset, 3)."""

    if not samples:
        return np.empty((0, len(offsets), 3), dtype=np.float64)
    centers_xyz = np.asarray([sample.xyz for sample in samples], dtype=np.float64)
    normals_xyz = np.asarray([sample.normal_xyz for sample in samples], dtype=np.float64)
    offsets_array = np.asarray(offsets, dtype=np.float64)
    queries_xyz = (centers_xyz[:, None, :] * float(coordinate_scale)
                   + normals_xyz[:, None, :] * offsets_array[None, :, None])
    return queries_xyz[..., ::-1]


class ChunkedArraySampler:
    """Trilinear sampler with a small decoded-chunk LRU above a local Zarr mirror."""

    def __init__(self, array: Any, decode_divisor: float = 1.0, max_chunks: int = 64):
        self.array = array
        self.shape = tuple(int(v) for v in array.shape)
        if len(self.shape) != 3:
            raise BenchmarkError(f"prediction array must be 3-D, got {self.shape}")
        raw_chunks = getattr(array, "chunks", self.shape)
        if isinstance(raw_chunks, int):
            raw_chunks = (raw_chunks,) * 3
        self.chunks = tuple(int(v) for v in raw_chunks)
        if len(self.chunks) != 3 or any(v <= 0 for v in self.chunks):
            raise BenchmarkError(f"invalid array chunks: {raw_chunks}")
        self.decode_divisor = float(decode_divisor)
        if self.decode_divisor <= 0.0:
            raise BenchmarkError("decode_divisor must be positive")
        self.max_chunks = max(1, int(max_chunks))
        self._cache: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()

    def _chunk(self, key: tuple[int, int, int]) -> np.ndarray:
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        starts = tuple(key[d] * self.chunks[d] for d in range(3))
        stops = tuple(min(starts[d] + self.chunks[d], self.shape[d]) for d in range(3))
        value = np.asarray(self.array[
            starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]
        ])
        self._cache[key] = value
        while len(self._cache) > self.max_chunks:
            self._cache.popitem(last=False)
        return value

    def _value(self, index: tuple[int, int, int]) -> float:
        key = tuple(index[d] // self.chunks[d] for d in range(3))
        chunk = self._chunk(key)
        local = tuple(index[d] - key[d] * self.chunks[d] for d in range(3))
        return float(chunk[local])

    def sample(self, points_zyx: np.ndarray) -> np.ndarray:
        points = np.asarray(points_zyx, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_zyx must have shape (n, 3)")
        result = np.full(points.shape[0], np.nan, dtype=np.float64)
        for point_index, point in enumerate(points):
            if not bool(np.isfinite(point).all()):
                continue
            lower = np.floor(point).astype(np.int64)
            # Even an exact coordinate on the last voxel is invalid: its full
            # interpolation cube would leave the published array.
            if any(lower[d] < 0 or lower[d] + 1 >= self.shape[d] for d in range(3)):
                continue
            fraction = point - lower
            value = 0.0
            for dz in (0, 1):
                wz = fraction[0] if dz else 1.0 - fraction[0]
                for dy in (0, 1):
                    wy = fraction[1] if dy else 1.0 - fraction[1]
                    for dx in (0, 1):
                        wx = fraction[2] if dx else 1.0 - fraction[2]
                        index = (int(lower[0] + dz), int(lower[1] + dy), int(lower[2] + dx))
                        value += wz * wy * wx * self._value(index)
            result[point_index] = value / self.decode_divisor
        return result


def trilinear_sample(
    array: Any,
    points_zyx: np.ndarray,
    decode_divisor: float = 1.0,
    max_chunks: int = 64,
) -> np.ndarray:
    """Convenience entry point used by synthetic tests and small callers."""

    return ChunkedArraySampler(array, decode_divisor, max_chunks).sample(points_zyx)


def interpolation_support_chunks(
    point_zyx: Sequence[float], shape: Sequence[int], chunks: Sequence[int]
) -> tuple[tuple[int, int, int], ...] | None:
    """Chunk coordinates for one complete interpolation cube, or None if OOB."""

    point = np.asarray(point_zyx, dtype=np.float64)
    if point.shape != (3,) or not bool(np.isfinite(point).all()):
        return None
    shape_tuple = tuple(int(v) for v in shape)
    chunks_tuple = tuple(int(v) for v in chunks)
    lower = np.floor(point).astype(np.int64)
    if any(lower[d] < 0 or lower[d] + 1 >= shape_tuple[d] for d in range(3)):
        return None
    per_axis = []
    for axis in range(3):
        first = int(lower[axis] // chunks_tuple[axis])
        second = int((lower[axis] + 1) // chunks_tuple[axis])
        per_axis.append((first,) if first == second else (first, second))
    support = []
    for z_chunk in per_axis[0]:
        for y_chunk in per_axis[1]:
            for x_chunk in per_axis[2]:
                support.append((z_chunk, y_chunk, x_chunk))
    return tuple(support)


def required_chunk_coordinates(
    points_zyx: np.ndarray, shape: Sequence[int], chunks: Sequence[int]
) -> tuple[list[tuple[int, int, int]], int]:
    """Return every chunk touched by valid full trilinear cubes."""

    points = np.asarray(points_zyx, dtype=np.float64).reshape(-1, 3)
    required: set[tuple[int, int, int]] = set()
    invalid = 0
    for point in points:
        support = interpolation_support_chunks(point, shape, chunks)
        if support is None:
            invalid += 1
            continue
        required.update(support)
    return sorted(required), invalid


def prediction_chunk_plan(
    lock: Mapping[str, Any], samples_by_segment: Mapping[str, Sequence[SurfaceSample]]
) -> dict[str, dict[str, Any]]:
    sampling = lock["sampling"]
    offsets = [float(value) for value in sampling["normal_offsets_prediction_voxels"]]
    center_index = offsets.index(0.0)
    all_profiles = []
    all_centers = []
    for segment in lock["reference_segments"]:
        queries = build_normal_queries(
            samples_by_segment[segment],
            float(sampling["reference_coordinate_scale_to_prediction"]),
            offsets,
        )
        if len(queries):
            all_profiles.append(queries.reshape(-1, 3))
            all_centers.append(queries[:, center_index, :])
    profile_points = np.concatenate(all_profiles, axis=0) if all_profiles else np.empty((0, 3))
    center_points = np.concatenate(all_centers, axis=0) if all_centers else np.empty((0, 3))
    plan = {}
    for channel_name in CHANNEL_NAMES:
        channel = lock["fiber_artifact"]["channels"][channel_name]
        points = profile_points if channel_name == "presence" else center_points
        chunk_coordinates, invalid = required_chunk_coordinates(
            points, channel["shape_zyx"], channel["chunks_zyx"]
        )
        plan[channel_name] = {
            "query_count": int(len(points)),
            "out_of_bounds_query_count": invalid,
            "chunk_coordinates": chunk_coordinates,
        }
    return plan


def create_outcome_marker(output_root: Path, payload: Mapping[str, Any]) -> Path:
    """Irreversibly claim the single outcome slot before any prediction read."""

    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / "OUTCOME_STARTED"
    result = output_root / "result.json"
    mirror = output_root / "prediction_mirror"
    if result.exists():
        raise BenchmarkError(f"refusing to overwrite existing outcome: {result}")
    if marker.exists():
        raise BenchmarkError(f"outcome has already started: {marker}")
    if mirror.exists():
        raise BenchmarkError(f"refusing pre-existing prediction mirror: {mirror}")
    marker_payload = canonical_json_bytes(dict(payload))
    try:
        with marker.open("xb") as handle:
            handle.write(marker_payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BenchmarkError(f"outcome has already started: {marker}") from exc
    return marker


def _validate_zarray_descriptor(
    descriptor: Mapping[str, Any], channel_name: str, channel: Mapping[str, Any]
) -> str:
    shape = [int(v) for v in descriptor.get("shape", [])]
    chunks = [int(v) for v in descriptor.get("chunks", [])]
    try:
        dtype = str(np.dtype(descriptor.get("dtype")))
    except TypeError as exc:
        raise BenchmarkError(f"invalid .zarray dtype for {channel_name}") from exc
    if shape != channel["shape_zyx"] or chunks != channel["chunks_zyx"]:
        raise BenchmarkError(f".zarray geometry changed for {channel_name}")
    if dtype != channel["dtype"]:
        raise BenchmarkError(f".zarray dtype changed for {channel_name}: {dtype}")
    fill_value = descriptor.get("fill_value")
    if (isinstance(fill_value, bool) or not isinstance(fill_value, (int, float))
            or float(fill_value) != 0.0):
        raise BenchmarkError(
            f"{channel_name} .zarray fill_value must be the pinned numeric zero"
        )
    separator = descriptor.get("dimension_separator", ".")
    if separator not in (".", "/"):
        raise BenchmarkError(f"unsupported dimension_separator for {channel_name}: {separator}")
    return str(separator)


def _chunk_object_key(coordinate: Sequence[int], separator: str) -> str:
    return separator.join(str(int(value)) for value in coordinate)


def is_definitive_missing_chunk(error: BaseException) -> bool:
    """Only an explicit HTTP 404 is evidence for an absent Zarr fill chunk."""

    return isinstance(error, urllib.error.HTTPError) and error.code == 404


def mirror_prediction_channels(
    lock: Mapping[str, Any],
    chunk_plan: Mapping[str, Mapping[str, Any]],
    output_root: Path,
    timeout_seconds: float,
    workers: int,
    pinned_missing_chunks: Mapping[str, set[tuple[int, int, int]]] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, set[tuple[int, int, int]]],
]:
    """Mirror and hash every exact prediction object needed by frozen queries."""

    try:
        import zarr
    except ImportError as exc:
        raise BenchmarkError(
            "outcome mode requires zarr==2.18.7 and a compatible numcodecs"
        ) from exc

    mirror_root = output_root / "prediction_mirror"
    metadata_receipts: dict[str, Any] = {}
    separators: dict[str, str] = {}
    for channel_name in CHANNEL_NAMES:
        channel = lock["fiber_artifact"]["channels"][channel_name]
        channel_dir = mirror_root / channel_name
        zarray_receipt = download_cached(
            channel["zarr_url"].rstrip("/") + "/.zarray",
            channel_dir / ".zarray",
            mirror_root,
            channel["zarray_sha256"],
            timeout_seconds,
        )
        try:
            descriptor = json.loads((channel_dir / ".zarray").read_bytes())
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"invalid .zarray for {channel_name}: {exc}") from exc
        separators[channel_name] = _validate_zarray_descriptor(descriptor, channel_name, channel)
        channel_metadata = {"zarray": zarray_receipt}
        if channel.get("zattrs_sha256") is not None:
            channel_metadata["zattrs"] = download_cached(
                channel["zarr_url"].rstrip("/") + "/.zattrs",
                channel_dir / ".zattrs",
                mirror_root,
                channel["zattrs_sha256"],
                timeout_seconds,
            )
        metadata_receipts[channel_name] = channel_metadata

    pinned_missing = {
        name: set((pinned_missing_chunks or {}).get(name, set())) for name in CHANNEL_NAMES
    }
    tasks = []
    chunk_receipts: dict[str, list[dict[str, Any]]] = {name: [] for name in CHANNEL_NAMES}
    for channel_name in CHANNEL_NAMES:
        channel = lock["fiber_artifact"]["channels"][channel_name]
        planned_coordinates = {
            tuple(int(value) for value in coordinate)
            for coordinate in chunk_plan[channel_name]["chunk_coordinates"]
        }
        if not pinned_missing[channel_name] <= planned_coordinates:
            raise BenchmarkError(f"pinned fill set is outside the chunk plan for {channel_name}")
        for coordinate in chunk_plan[channel_name]["chunk_coordinates"]:
            coordinate_tuple = tuple(int(value) for value in coordinate)
            key = _chunk_object_key(coordinate, separators[channel_name])
            url = channel["zarr_url"].rstrip("/") + "/" + key
            if coordinate_tuple in pinned_missing[channel_name]:
                chunk_receipts[channel_name].append({
                    "chunk_coordinate_zyx": list(coordinate_tuple),
                    "object_key": key,
                    "url": url,
                    "status": "pinned_first_attempt_http_404_fill",
                    "http_status": 404,
                    "fill_value": 0,
                    "network_request_on_resume": False,
                })
                continue
            tasks.append((channel_name, coordinate_tuple, key, url,
                          mirror_root / channel_name / Path(key)))

    missing_chunks: dict[str, set[tuple[int, int, int]]] = {
        name: set(pinned_missing[name]) for name in CHANNEL_NAMES
    }
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_cached, url, path, mirror_root, None, timeout_seconds
            ): (channel_name, coordinate, key, url)
            for channel_name, coordinate, key, url, path in tasks
        }
        for future in as_completed(futures):
            channel_name, coordinate, key, url = futures[future]
            try:
                receipt = future.result()
                chunk_receipts[channel_name].append({
                    "chunk_coordinate_zyx": list(coordinate),
                    "object_key": key,
                    "status": "stored_object",
                    **receipt,
                })
            except Exception as exc:
                if is_definitive_missing_chunk(exc):
                    missing_chunks[channel_name].add(tuple(coordinate))
                    chunk_receipts[channel_name].append({
                        "chunk_coordinate_zyx": list(coordinate),
                        "object_key": key,
                        "url": url,
                        "status": "missing_uses_pinned_fill",
                        "http_status": 404,
                        "fill_value": 0,
                    })
                else:
                    errors.append({
                        "channel": channel_name,
                        "object_key": key,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    })
    if errors:
        raise BenchmarkError(
            "prediction mirror aborted on non-404 transport/integrity errors: "
            + json.dumps(sorted(errors, key=lambda item: (item["channel"], item["object_key"])),
                         sort_keys=True)
        )
    for channel_name in CHANNEL_NAMES:
        chunk_receipts[channel_name].sort(key=lambda item: item["chunk_coordinate_zyx"])

    arrays = {}
    for channel_name in CHANNEL_NAMES:
        arrays[channel_name] = zarr.open(str(mirror_root / channel_name), mode="r")
    receipts = {
        "store": "local partial Zarr mirror; every required object is SHA-256 receipted",
        "chunk_plan": chunk_plan,
        "metadata": metadata_receipts,
        "chunks": chunk_receipts,
        "missing_fill_chunk_counts": {
            name: len(missing_chunks[name]) for name in CHANNEL_NAMES
        },
    }
    return arrays, receipts, missing_chunks


def validate_orientation_sibling_coherence(
    lock: Mapping[str, Any],
    samples_by_segment: Mapping[str, Sequence[SurfaceSample]],
    presence_sampler: ChunkedArraySampler,
    missing_chunks: Mapping[str, set[tuple[int, int, int]]],
) -> dict[str, Any]:
    """Abort when positive presence lacks complete nx/ny interpolation support."""

    sampling = lock["sampling"]
    offsets = [float(value) for value in sampling["normal_offsets_prediction_voxels"]]
    center_index = offsets.index(0.0)
    direction_channel = lock["fiber_artifact"]["channels"]["nx"]
    positive = zero = out_of_bounds = 0
    incoherent: list[dict[str, Any]] = []
    for segment in lock["reference_segments"]:
        samples = samples_by_segment[segment]
        queries = build_normal_queries(
            samples,
            float(sampling["reference_coordinate_scale_to_prediction"]),
            offsets,
        )
        if not samples:
            continue
        centers = queries[:, center_index, :]
        center_presence = presence_sampler.sample(centers)
        for sample, query, presence in zip(samples, centers, center_presence):
            if not math.isfinite(float(presence)):
                out_of_bounds += 1
                continue
            if float(presence) <= 0.0:
                zero += 1
                continue
            positive += 1
            support = interpolation_support_chunks(
                query, direction_channel["shape_zyx"], direction_channel["chunks_zyx"]
            )
            if support is None:
                raise BenchmarkError(
                    f"positive center presence has out-of-bounds direction query: "
                    f"{segment} row={sample.row} column={sample.column}"
                )
            missing_by_channel = {
                channel_name: sorted(set(support) & missing_chunks[channel_name])
                for channel_name in ("nx", "ny")
            }
            if any(missing_by_channel.values()):
                incoherent.append({
                    "segment": segment,
                    "row": sample.row,
                    "column": sample.column,
                    "center_presence": float(presence),
                    "missing_support": {
                        name: [list(value) for value in values]
                        for name, values in missing_by_channel.items() if values
                    },
                })
    if incoherent:
        raise BenchmarkError(
            "positive presence has missing nx/ny sibling chunk support: "
            + json.dumps({"count": len(incoherent), "first": incoherent[:10]}, sort_keys=True)
        )
    return {
        "positive_center_presence_points": positive,
        "zero_center_presence_points_excluded_from_orientation": zero,
        "out_of_bounds_center_points": out_of_bounds,
        "positive_presence_missing_direction_support": 0,
    }


def evaluate_primary_gate(
    segment_deltas: Sequence[float], gate: Mapping[str, Any]
) -> dict[str, Any]:
    finite = [float(value) for value in segment_deltas if math.isfinite(float(value))]
    positive = sum(value > 0.0 for value in finite)
    negative = sum(value < 0.0 for value in finite)
    zero = len(finite) - positive - negative
    global_median = float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None
    checks = {
        "minimum_analyzable_segments": len(finite) >= int(gate["minimum_analyzable_segments"]),
        "minimum_positive_segments": positive >= int(gate["minimum_positive_segments"]),
        "minimum_median_segment_delta": (
            global_median is not None
            and global_median >= float(gate["minimum_median_segment_delta"])
        ),
    }
    return {
        "decision": (
            "LOCALIZATION_SUPPORTED" if all(checks.values())
            else "LOCALIZATION_NOT_SUPPORTED"
        ),
        "analyzable_segments": len(finite),
        "positive_segments": positive,
        "negative_segments": negative,
        "zero_segments": zero,
        "global_median_segment_delta": global_median,
        "checks": checks,
        "thresholds": dict(gate),
        "inference": "descriptive finite-set gate; no p-value",
    }


def _offset_column(offset: float) -> str:
    integer = int(offset)
    return f"presence_m{abs(integer)}" if integer < 0 else f"presence_p{integer}"


def decode_fiber_directions(
    nx_values: np.ndarray,
    ny_values: np.ndarray,
    decode_offset: float = 128.0,
    decode_divisor: float = 127.0,
) -> np.ndarray:
    """Decode uint8 nx/ny interpolation values into unit XYZ directions."""

    nx_array = np.asarray(nx_values, dtype=np.float64)
    ny_array = np.asarray(ny_values, dtype=np.float64)
    if nx_array.shape != ny_array.shape:
        raise ValueError("nx and ny arrays must have matching shape")
    dx = (nx_array - float(decode_offset)) / float(decode_divisor)
    dy = (ny_array - float(decode_offset)) / float(decode_divisor)
    dz = np.sqrt(np.maximum(0.0, 1.0 - dx * dx - dy * dy))
    vectors = np.stack((dx, dy, dz), axis=-1)
    norms = np.linalg.norm(vectors, axis=-1)
    valid = np.isfinite(vectors).all(axis=-1) & np.isfinite(norms) & (norms > 0.0)
    output = np.full(vectors.shape, np.nan, dtype=np.float64)
    output[valid] = vectors[valid] / norms[valid, None]
    return output


def tangent_plane_angles_degrees(
    fiber_xyz: np.ndarray, surface_normal_xyz: np.ndarray
) -> np.ndarray:
    """Unsigned acute angle between a fiber direction and the tangent plane."""

    fibers = np.asarray(fiber_xyz, dtype=np.float64)
    normals = np.asarray(surface_normal_xyz, dtype=np.float64)
    if fibers.shape != normals.shape or fibers.ndim != 2 or fibers.shape[1] != 3:
        raise ValueError("fiber and normal arrays must both have shape (n, 3)")
    result = np.full(fibers.shape[0], np.nan, dtype=np.float64)
    valid = np.isfinite(fibers).all(axis=1) & np.isfinite(normals).all(axis=1)
    if valid.any():
        dot = np.abs(np.einsum("ij,ij->i", fibers[valid], normals[valid]))
        result[valid] = np.degrees(np.arcsin(np.clip(dot, 0.0, 1.0)))
    return result


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float | None:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    if values_array.shape != weights_array.shape:
        raise ValueError("values and weights must have matching shape")
    valid = (
        np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0.0)
    )
    if not valid.any():
        return None
    values_array = values_array[valid]
    weights_array = weights_array[valid]
    order = np.argsort(values_array, kind="stable")
    values_array = values_array[order]
    weights_array = weights_array[order]
    threshold = 0.5 * float(weights_array.sum())
    index = int(np.searchsorted(np.cumsum(weights_array), threshold, side="left"))
    return float(values_array[min(index, len(values_array) - 1)])


def analyze_segment(
    segment: str,
    samples: Sequence[SurfaceSample],
    raster_shape: Sequence[int],
    lock: Mapping[str, Any],
    samplers: Mapping[str, ChunkedArraySampler],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampling = lock["sampling"]
    offsets = [float(value) for value in sampling["normal_offsets_prediction_voxels"]]
    queries = build_normal_queries(
        samples, float(sampling["reference_coordinate_scale_to_prediction"]), offsets
    )
    if samples:
        profiles = samplers["presence"].sample(
            queries.reshape(-1, 3)
        ).reshape(len(samples), len(offsets))
    else:
        profiles = np.empty((0, len(offsets)), dtype=np.float64)
    center_index = offsets.index(0.0)
    if samples:
        nx_values = samplers["nx"].sample(queries[:, center_index, :])
        ny_values = samplers["ny"].sample(queries[:, center_index, :])
    else:
        nx_values = np.empty(0, dtype=np.float64)
        ny_values = np.empty(0, dtype=np.float64)
    nx_spec = lock["fiber_artifact"]["channels"]["nx"]
    fiber_vectors = decode_fiber_directions(
        nx_values,
        ny_values,
        float(nx_spec["decode_offset"]),
        float(nx_spec["decode_divisor"]),
    )
    surface_normals = np.asarray([sample.normal_xyz for sample in samples], dtype=np.float64)
    if not samples:
        surface_normals = np.empty((0, 3), dtype=np.float64)
    tangent_angles = tangent_plane_angles_degrees(fiber_vectors, surface_normals)
    presence_complete = np.isfinite(profiles).all(axis=1)
    center_presence = profiles[:, center_index] if len(profiles) else np.empty(0)
    orientation_valid = (
        presence_complete & (center_presence > 0.0) & np.isfinite(tangent_angles)
    )
    control_indices = [index for index in range(len(offsets)) if index != center_index]
    deltas = np.full(len(samples), np.nan, dtype=np.float64)
    if presence_complete.any():
        deltas[presence_complete] = (
            profiles[presence_complete, center_index]
            - profiles[presence_complete][:, control_indices].mean(axis=1)
        )

    point_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        row: dict[str, Any] = {
            "segment": segment,
            "row": sample.row,
            "column": sample.column,
            "bin_y": sample.bin_y,
            "bin_x": sample.bin_x,
            "selection_sha256": sample.selection_sha256,
            "map_x": sample.xyz[0],
            "map_y": sample.xyz[1],
            "map_z": sample.xyz[2],
            "normal_x": sample.normal_xyz[0],
            "normal_y": sample.normal_xyz[1],
            "normal_z": sample.normal_xyz[2],
            "fiber_nx_raw": float(nx_values[index]) if math.isfinite(float(nx_values[index])) else None,
            "fiber_ny_raw": float(ny_values[index]) if math.isfinite(float(ny_values[index])) else None,
            "fiber_dx": (float(fiber_vectors[index, 0])
                         if math.isfinite(float(fiber_vectors[index, 0])) else None),
            "fiber_dy": (float(fiber_vectors[index, 1])
                         if math.isfinite(float(fiber_vectors[index, 1])) else None),
            "fiber_dz": (float(fiber_vectors[index, 2])
                         if math.isfinite(float(fiber_vectors[index, 2])) else None),
            "tangent_angle_degrees": (float(tangent_angles[index])
                                      if math.isfinite(float(tangent_angles[index])) else None),
            "matched_baseline_tangent_angle_degrees": None,
            "complete_profile": bool(presence_complete[index]),
            "orientation_valid": bool(orientation_valid[index]),
            "delta": float(deltas[index]) if presence_complete[index] else None,
        }
        for offset_index, offset in enumerate(offsets):
            value = profiles[index, offset_index]
            row[_offset_column(offset)] = float(value) if math.isfinite(float(value)) else None
        point_rows.append(row)

    complete_profiles = profiles[presence_complete]
    minimum = int(sampling["minimum_samples_per_segment"])
    summary: dict[str, Any] = {
        "segment": segment,
        "status": "analyzable" if len(complete_profiles) >= minimum else "insufficient_complete_profiles",
        "raster_shape_rc": [int(v) for v in raster_shape],
        "selected_samples": len(samples),
        "complete_profiles": int(presence_complete.sum()),
        "minimum_complete_profiles": minimum,
        "selection_sha256": selected_samples_sha256(samples),
        "median_delta": None,
        "median_profile": None,
        "median_center_presence": None,
        "fraction_center_above_control_mean": None,
        "symmetric_offset_median_deltas": None,
        "orientation": {},
    }
    if summary["status"] == "analyzable":
        complete_deltas = deltas[presence_complete]
        median_profile = np.median(complete_profiles, axis=0)
        center = complete_profiles[:, center_index]
        controls = complete_profiles[:, control_indices].mean(axis=1)
        symmetric = {}
        for distance in (4, 8, 12):
            negative_index = offsets.index(float(-distance))
            positive_index = offsets.index(float(distance))
            distance_delta = center - 0.5 * (
                complete_profiles[:, negative_index] + complete_profiles[:, positive_index]
            )
            symmetric[str(distance)] = float(np.median(distance_delta))
        summary.update({
            "median_delta": float(np.median(complete_deltas)),
            "median_profile": [float(value) for value in median_profile],
            "median_center_presence": float(np.median(center)),
            "fraction_center_above_control_mean": float(np.mean(center > controls)),
            "symmetric_offset_median_deltas": symmetric,
        })
    orientation_angles = tangent_angles[orientation_valid]
    orientation_weights = center_presence[orientation_valid]
    threshold = float(lock["orientation_analysis"]["center_presence_subset_threshold"])
    subset = orientation_weights >= threshold
    orientation_minimum = int(
        lock["orientation_analysis"]["minimum_positive_weight_points_per_segment"]
    )
    orientation_status = (
        "analyzable" if len(orientation_angles) >= orientation_minimum
        else "insufficient_positive_presence_points"
    )
    summary["orientation"] = {
        "status": orientation_status,
        "positive_presence_points": int(len(orientation_angles)),
        "minimum_positive_presence_points": orientation_minimum,
        "positive_presence_coverage_of_complete_profiles": (
            float(len(orientation_angles) / len(complete_profiles)) if len(complete_profiles) else None
        ),
        "presence_weighted_median_tangent_angle_degrees": (
            weighted_median(orientation_angles, orientation_weights)
            if orientation_status == "analyzable" else None
        ),
        "presence_threshold": threshold,
        "presence_threshold_points": int(np.count_nonzero(subset)),
        "presence_threshold_coverage_of_complete_profiles": (
            float(np.count_nonzero(subset) / len(complete_profiles))
            if len(complete_profiles) else None
        ),
        "presence_threshold_median_tangent_angle_degrees": (
            float(np.median(orientation_angles[subset])) if subset.any() else None
        ),
        "matched_baseline_presence_weighted_median_tangent_angle_degrees": None,
        "improvement_over_matched_baseline_degrees": None,
    }
    return summary, point_rows


def apply_orientation_baseline(
    lock: Mapping[str, Any],
    summaries: list[dict[str, Any]],
    point_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen global half-rotation derangement and summarize orientation."""

    eligible_indices = [
        index for index, row in enumerate(point_rows)
        if row.get("orientation_valid")
        and row.get("fiber_dx") is not None
        and row.get("tangent_angle_degrees") is not None
    ]
    seed = str(lock["sampling"]["seed"])
    eligible_indices.sort(key=lambda index: hashlib.sha256(
        f"{seed}|orientation-baseline|{point_rows[index]['segment']}|"
        f"{point_rows[index]['row']}|{point_rows[index]['column']}".encode("utf-8")
    ).digest())
    count = len(eligible_indices)
    if count >= 2:
        shift = count // 2
        if shift == 0:
            shift = 1
        fibers = np.asarray([
            [point_rows[index]["fiber_dx"], point_rows[index]["fiber_dy"],
             point_rows[index]["fiber_dz"]]
            for index in eligible_indices
        ], dtype=np.float64)
        normals = np.asarray([
            [point_rows[index]["normal_x"], point_rows[index]["normal_y"],
             point_rows[index]["normal_z"]]
            for index in eligible_indices
        ], dtype=np.float64)
        baseline_normals = np.roll(normals, -shift, axis=0)
        baseline_angles = tangent_plane_angles_degrees(fibers, baseline_normals)
        for index, angle in zip(eligible_indices, baseline_angles):
            point_rows[index]["matched_baseline_tangent_angle_degrees"] = float(angle)
    else:
        shift = None

    points_by_segment: dict[str, list[dict[str, Any]]] = {}
    for row in point_rows:
        if row.get("orientation_valid"):
            points_by_segment.setdefault(str(row["segment"]), []).append(row)
    for summary in summaries:
        orientation = summary.get("orientation")
        if (not isinstance(orientation, dict)
                or orientation.get("status") != "analyzable"):
            continue
        rows = points_by_segment.get(str(summary["segment"]), [])
        baseline_values = [row["matched_baseline_tangent_angle_degrees"] for row in rows]
        weights = [row["presence_p0"] for row in rows]
        baseline_median = weighted_median(baseline_values, weights)
        observed = orientation["presence_weighted_median_tangent_angle_degrees"]
        orientation["matched_baseline_presence_weighted_median_tangent_angle_degrees"] = baseline_median
        orientation["improvement_over_matched_baseline_degrees"] = (
            float(baseline_median - observed)
            if baseline_median is not None and observed is not None else None
        )

    analyzable = [
        summary for summary in summaries
        if isinstance(summary.get("orientation"), Mapping)
        and summary["orientation"].get("status") == "analyzable"
        and summary["orientation"].get("presence_weighted_median_tangent_angle_degrees") is not None
    ]
    segment_angles = [
        float(summary["orientation"]["presence_weighted_median_tangent_angle_degrees"])
        for summary in analyzable
    ]
    improvements = [
        float(summary["orientation"]["improvement_over_matched_baseline_degrees"])
        for summary in analyzable
        if summary["orientation"].get("improvement_over_matched_baseline_degrees") is not None
    ]
    orientation_spec = lock["orientation_analysis"]
    global_median = float(np.median(segment_angles)) if segment_angles else None
    improved_count = sum(
        value >= float(orientation_spec["minimum_segment_improvement_degrees"])
        for value in improvements
    )
    all_rows = [point_rows[index] for index in eligible_indices]
    all_angles = [row["tangent_angle_degrees"] for row in all_rows]
    all_weights = [row["presence_p0"] for row in all_rows]
    threshold = float(orientation_spec["center_presence_subset_threshold"])
    subset_angles = [
        float(row["tangent_angle_degrees"]) for row in all_rows
        if float(row["presence_p0"]) >= threshold
    ]
    checks = {
        "minimum_analyzable_segments": (
            len(analyzable) >= int(orientation_spec["minimum_analyzable_segments"])
        ),
        "maximum_global_median_segment_presence_weighted_tangent_angle_degrees": (
            global_median is not None
            and global_median <= float(orientation_spec[
                "maximum_global_median_segment_presence_weighted_tangent_angle_degrees"
            ])
        ),
        "minimum_segments_improving_over_matched_baseline": (
            improved_count >= int(orientation_spec[
                "minimum_segments_improving_over_matched_baseline"
            ])
        ),
    }
    return {
        "decision": "TANGENCY_SUPPORTED" if all(checks.values()) else "TANGENCY_NOT_SUPPORTED",
        "status": "secondary descriptive sanity gate",
        "analyzable_segments": len(analyzable),
        "global_median_of_segment_presence_weighted_tangent_angles_degrees": global_median,
        "global_point_presence_weighted_median_tangent_angle_degrees": weighted_median(
            all_angles, all_weights
        ),
        "global_point_unweighted_median_tangent_angle_degrees": (
            float(np.median(all_angles)) if all_angles else None
        ),
        "presence_threshold": threshold,
        "presence_threshold_points": len(subset_angles),
        "presence_threshold_coverage": (
            float(len(subset_angles) / len(all_rows)) if all_rows else None
        ),
        "presence_threshold_global_median_tangent_angle_degrees": (
            float(np.median(subset_angles)) if subset_angles else None
        ),
        "globally_deranged_baseline": {
            "ordering": "SHA256(seed|orientation-baseline|segment|row|column)",
            "rotation": "floor(N/2)",
            "eligible_points": count,
            "rotation_positions": shift,
            "fixed_points": 0 if count >= 2 else None,
        },
        "segments_improving_by_at_least_threshold": improved_count,
        "segment_improvement_threshold_degrees": float(
            orientation_spec["minimum_segment_improvement_degrees"]
        ),
        "checks": checks,
        "thresholds": dict(orientation_spec),
    }


def _float_csv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".17g") if math.isfinite(float(value)) else ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _float_csv(row.get(column)) for column in columns})
    return stream.getvalue().encode("utf-8")


def _artifact_receipt(path: Path, output_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _source_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError(f"required implementation file is missing: {path.name}")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def runtime_receipt() -> dict[str, Any]:
    expected_python = "3.11.9"
    if platform.python_version() != expected_python:
        raise BenchmarkError(
            f"outcome requires CPython {expected_python}, observed {platform.python_version()}"
        )
    requirements_path = ROOT / "requirements-fiber-lock.txt"
    requirements = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise BenchmarkError(f"unfrozen requirement line: {line}")
        package, expected = line.split("==", 1)
        try:
            observed = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError as exc:
            raise BenchmarkError(f"missing locked runtime package: {package}") from exc
        if observed != expected:
            raise BenchmarkError(
                f"runtime package mismatch for {package}: expected {expected}, observed {observed}"
            )
        requirements[package] = observed
    return {
        "python": expected_python,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": requirements,
        "requirements_lock": _source_receipt(requirements_path),
    }


def _annotated_tag_receipt(
    git: Any, required_tag: str, expected_commit: str
) -> dict[str, str]:
    reference = f"refs/tags/{required_tag}"
    if git("cat-file", "-t", reference) != "tag":
        raise BenchmarkError(f"required tag is not annotated: {required_tag}")
    tag_object = git("rev-parse", reference)
    peeled_commit = git("rev-parse", reference + "^{}")
    if peeled_commit != expected_commit:
        raise BenchmarkError(f"required tag does not resolve to HEAD: {required_tag}")
    return {"name": required_tag, "object": tag_object, "commit": peeled_commit}


def git_receipt(
    required_tag: str,
    required_paths: Sequence[str],
    required_ancestor: str | None = None,
) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise BenchmarkError(f"cannot verify preregistration git state: {exc}") from exc
        return completed.stdout.strip()

    commit = git("rev-parse", "HEAD")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise BenchmarkError("outcome requires a clean tracked worktree")
    tags = sorted(value for value in git("tag", "--points-at", "HEAD").splitlines() if value)
    if required_tag not in tags:
        raise BenchmarkError(f"outcome requires HEAD to carry tag {required_tag}")
    required_tag_receipt = _annotated_tag_receipt(git, required_tag, commit)
    original_preregistration_tag = None
    if required_ancestor is not None:
        if re.fullmatch(r"[0-9a-f]{40}", required_ancestor) is None:
            raise BenchmarkError("required ancestor is not a 40-digit lowercase git object ID")
        try:
            git("merge-base", "--is-ancestor", required_ancestor, commit)
        except BenchmarkError as exc:
            raise BenchmarkError(
                f"outcome commit does not descend from {required_ancestor}"
            ) from exc
        original_preregistration_tag = _annotated_tag_receipt(
            git, PREREG_TAG, required_ancestor
        )
    tracked_blobs = {}
    for relative in required_paths:
        observed = git("ls-files", "--error-unmatch", "--", relative)
        if observed != relative:
            raise BenchmarkError(f"required outcome source is not tracked exactly: {relative}")
        tracked_blobs[relative] = git("rev-parse", f"{commit}:{relative}")
    return {
        "commit": commit,
        "tracked_worktree_clean": True,
        "tags_at_head": tags,
        "required_annotated_tag": required_tag_receipt,
        "required_ancestor": required_ancestor,
        "original_preregistration_tag": original_preregistration_tag,
        "tracked_source_blobs": tracked_blobs,
    }


MIRROR_DIGEST_ALGORITHM = (
    "SHA256(json.dumps(sorted entries {path,bytes,sha256,kind}, sort_keys=True, "
    "separators=(',',':'), ensure_ascii=True)); paths are POSIX-relative to "
    "prediction_mirror; no trailing newline"
)


def _receipt_set_sha256(entries: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted((dict(entry) for entry in entries), key=lambda entry: str(entry["path"]))
    payload = json.dumps(
        ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(payload)


def mirror_inventory_receipt(
    mirror_root: Path,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Hash the pre-resume mirror without decoding or sampling any array value."""

    if not mirror_root.is_dir():
        raise BenchmarkError(f"partial prediction mirror is missing: {mirror_root}")
    entries: list[dict[str, Any]] = []
    stored_keys: dict[str, set[str]] = {name: set() for name in CHANNEL_NAMES}
    for path in sorted((value for value in mirror_root.rglob("*") if value.is_file())):
        relative = path.relative_to(mirror_root).as_posix()
        parts = relative.split("/")
        if path.name.endswith(".partial"):
            raise BenchmarkError(f"partial transport file remains: {relative}")
        if not parts or parts[0] not in CHANNEL_NAMES:
            raise BenchmarkError(f"unexpected mirror object: {relative}")
        channel = parts[0]
        if len(parts) == 2 and parts[1] in (".zarray", ".zattrs"):
            kind = "metadata"
        elif len(parts) == 4 and all(part.isdigit() for part in parts[1:]):
            kind = "chunk"
            stored_keys[channel].add("/".join(parts[1:]))
        elif len(parts) == 2 and re.fullmatch(r"\d+\.\d+\.\d+", parts[1]):
            kind = "chunk"
            stored_keys[channel].add(parts[1])
        else:
            raise BenchmarkError(f"unexpected mirror object: {relative}")
        entries.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "kind": kind,
        })

    channel_receipts = {}
    for channel in CHANNEL_NAMES:
        channel_entries = [entry for entry in entries if entry["path"].startswith(channel + "/")]
        chunk_entries = [entry for entry in channel_entries if entry["kind"] == "chunk"]
        channel_receipts[channel] = {
            "stored_chunk_count": len(chunk_entries),
            "stored_chunk_bytes": sum(int(entry["bytes"]) for entry in chunk_entries),
            "chunk_receipt_set_sha256": _receipt_set_sha256(chunk_entries),
            "all_object_receipt_set_sha256": _receipt_set_sha256(channel_entries),
        }
    metadata_entries = [entry for entry in entries if entry["kind"] == "metadata"]
    receipt = {
        "digest_algorithm": MIRROR_DIGEST_ALGORITHM,
        "sha256": _receipt_set_sha256(entries),
        "object_count": len(entries),
        "bytes": sum(int(entry["bytes"]) for entry in entries),
        "metadata_object_count": len(metadata_entries),
        "metadata_bytes": sum(int(entry["bytes"]) for entry in metadata_entries),
        "partial_file_count": 0,
        "channels": channel_receipts,
    }
    return receipt, stored_keys


def verify_transport_failure_state(
    lock: Mapping[str, Any],
    lock_receipt: Mapping[str, Any],
    chunk_plan: Mapping[str, Mapping[str, Any]],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if not TRANSPORT_FAILURE_RECORD.is_file() or not TRANSPORT_AMENDMENT.is_file():
        raise BenchmarkError("transport failure record or amendment is missing")
    try:
        failure = json.loads(TRANSPORT_FAILURE_RECORD.read_bytes())
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid transport failure record: {exc}") from exc
    if (not isinstance(failure, dict)
            or failure.get("event") != "TRANSPORT_FAILURE_BEFORE_ARRAY_OPEN"
            or failure.get("experiment_id") != lock.get("experiment_id")):
        raise BenchmarkError("unexpected transport failure record")
    scientific = failure.get("scientific_state")
    if (not isinstance(scientific, Mapping)
            or scientific.get("zarr_array_opened") is not False
            or scientific.get("prediction_values_sampled") is not False
            or scientific.get("metrics_computed") is not False
            or scientific.get("panels_rendered") is not False
            or scientific.get("scientific_outcome") is not None):
        raise BenchmarkError("failure record does not describe a pre-analysis transport abort")
    preregistration = failure.get("preregistration")
    if not isinstance(preregistration, Mapping):
        raise BenchmarkError("failure record is missing preregistration provenance")

    marker = output_root / "OUTCOME_STARTED"
    expected_marker = failure.get("outcome_marker")
    if not isinstance(expected_marker, Mapping) or not marker.is_file():
        raise BenchmarkError("original outcome marker is missing")
    if (marker.stat().st_size != int(expected_marker.get("bytes", -1))
            or sha256_file(marker) != expected_marker.get("sha256")):
        raise BenchmarkError("original outcome marker changed after the transport failure")
    marker_payload = json.loads(marker.read_bytes())
    if (marker_payload.get("lock_sha256") != lock_receipt["sha256"]
            or marker_payload.get("prediction_chunk_plan_sha256")
            != sha256_bytes(canonical_json_bytes(chunk_plan))
            or marker_payload.get("preregistration_tag") != PREREG_TAG
            or preregistration.get("commit") != PREREG_COMMIT
            or marker_payload.get("git_commit") != preregistration.get("commit")
            or preregistration.get("tag") != PREREG_TAG):
        raise BenchmarkError("original outcome marker is inconsistent with the frozen plan")
    for forbidden in ("result.json", "segments.csv", "points.csv", "panels",
                      "TRANSPORT_RESUME_STARTED", "TRANSPORT_RESUME_FAILURE.json"):
        if (output_root / forbidden).exists():
            raise BenchmarkError(f"unexpected pre-resume outcome artifact: {forbidden}")

    inventory, stored_keys = mirror_inventory_receipt(output_root / "prediction_mirror")
    if inventory != failure.get("mirror_inventory"):
        raise BenchmarkError("partial mirror receipt-set digest differs from the public failure record")
    failure_detail = failure.get("failure")
    failed = failure_detail.get("failed_requests") if isinstance(failure_detail, Mapping) else None
    fills = failure.get("inferred_http_404_fill_keys", {})
    if not isinstance(failed, Mapping) or not isinstance(fills, Mapping):
        raise BenchmarkError("failure record is missing failed or fill object sets")
    failed_count = 0
    for channel in CHANNEL_NAMES:
        for collection_name, collection in (("failed", failed), ("fill", fills)):
            keys = collection.get(channel)
            if (not isinstance(keys, list) or len(keys) != len(set(keys))
                    or any(not isinstance(key, str)
                           or re.fullmatch(r"\d+/\d+/\d+", key) is None for key in keys)):
                raise BenchmarkError(
                    f"invalid {collection_name} object-key set for {channel}"
                )
        if set(failed[channel]) & set(fills[channel]):
            raise BenchmarkError(f"failed and fill sets overlap for {channel}")
        failed_count += len(failed[channel])
        planned = {
            "/".join(str(int(value)) for value in coordinate)
            for coordinate in chunk_plan[channel]["chunk_coordinates"]
        }
        allowed_missing = set(failed.get(channel, [])) | set(fills.get(channel, []))
        actual_missing = planned - stored_keys[channel]
        if actual_missing != allowed_missing:
            raise BenchmarkError(f"pre-resume missing-object set changed for {channel}")
        if stored_keys[channel] - planned:
            raise BenchmarkError(f"pre-resume mirror has unplanned chunks for {channel}")
    if failed_count != int(failure_detail.get("failed_request_count", -1)):
        raise BenchmarkError("failure request count disagrees with its object-key sets")
    return failure, inventory, marker


def create_transport_resume_marker(output_root: Path, payload: Mapping[str, Any]) -> Path:
    marker = output_root / "TRANSPORT_RESUME_STARTED"
    if marker.exists():
        raise BenchmarkError(f"transport resume has already started: {marker}")
    if (output_root / "result.json").exists():
        raise BenchmarkError("refusing to resume a completed outcome")
    try:
        with marker.open("xb") as handle:
            handle.write(canonical_json_bytes(dict(payload)))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise BenchmarkError(f"transport resume has already started: {marker}") from exc
    return marker


def record_transport_resume_failure(output_root: Path, error: BaseException) -> Path | None:
    """Persist a second technical failure only after the exclusive resume began."""

    resume_marker = output_root / "TRANSPORT_RESUME_STARTED"
    result = output_root / "result.json"
    if not resume_marker.is_file() or result.exists():
        return None
    path = output_root / "TRANSPORT_RESUME_FAILURE.json"
    if path.exists():
        return path
    original_marker = output_root / "OUTCOME_STARTED"
    partial_artifacts = []
    for candidate in (output_root / "segments.csv", output_root / "points.csv"):
        if candidate.is_file():
            partial_artifacts.append(_artifact_receipt(candidate, output_root))
    panels = output_root / "panels"
    if panels.is_dir():
        partial_artifacts.extend(
            _artifact_receipt(candidate, output_root)
            for candidate in sorted(panels.glob("*.png")) if candidate.is_file()
        )
    payload = {
        "schema_version": 1,
        "event": "TRANSPORT_RESUME_TECHNICAL_FAILURE",
        "recorded_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "error": {"type": type(error).__name__, "message": str(error)},
        "original_outcome_marker": _artifact_receipt(original_marker, output_root),
        "transport_resume_marker": _artifact_receipt(resume_marker, output_root),
        "result_written": False,
        "outcome_status": "incomplete technical failure; execution stage unknown",
        "scientific_values_may_have_been_sampled": True,
        "partial_artifacts": partial_artifacts,
    }
    try:
        with path.open("xb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return path
    return path


# A deliberately tiny bitmap alphabet keeps panel bytes independent of fonts,
# operating systems, matplotlib versions, and PNG metadata.
_FONT = {
    " ": ("000", "000", "000", "000", "000"),
    "-": ("000", "000", "111", "000", "000"),
    ".": ("000", "000", "000", "000", "010"),
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "A": ("010", "101", "111", "101", "101"),
    "C": ("111", "100", "100", "100", "111"),
    "D": ("110", "101", "101", "101", "110"),
    "E": ("111", "100", "110", "100", "111"),
    "F": ("111", "100", "110", "100", "100"),
    "G": ("111", "100", "101", "101", "111"),
    "I": ("111", "010", "010", "010", "111"),
    "L": ("100", "100", "100", "100", "111"),
    "M": ("101", "111", "111", "101", "101"),
    "N": ("101", "111", "111", "111", "101"),
    "O": ("111", "101", "101", "101", "111"),
    "P": ("110", "101", "110", "100", "100"),
    "R": ("110", "101", "110", "101", "101"),
    "S": ("111", "100", "111", "001", "111"),
    "T": ("111", "010", "010", "010", "010"),
    "U": ("101", "101", "101", "101", "111"),
    "W": ("101", "101", "111", "111", "101"),
}


def _draw_text(image: np.ndarray, x: int, y: int, text: str, color=(0, 0, 0), scale: int = 2) -> None:
    cursor = x
    fallback = ("111", "101", "010", "000", "010")
    for character in text.upper():
        glyph = _FONT.get(character, fallback)
        for glyph_y, bits in enumerate(glyph):
            for glyph_x, bit in enumerate(bits):
                if bit == "1":
                    y0 = y + glyph_y * scale
                    x0 = cursor + glyph_x * scale
                    image[y0:y0 + scale, x0:x0 + scale] = color
        cursor += 4 * scale


def _line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Sequence[int]) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = color
        if x0 == x1 and y0 == y1:
            break
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def _interpolate_colors(value: float, anchors: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    clipped = min(1.0, max(0.0, float(value)))
    position = clipped * (len(anchors) - 1)
    left = min(int(math.floor(position)), len(anchors) - 2)
    fraction = position - left
    return tuple(int(round((1.0 - fraction) * anchors[left][channel]
                           + fraction * anchors[left + 1][channel])) for channel in range(3))


def _presence_color(value: float) -> tuple[int, int, int]:
    return _interpolate_colors(value, (
        (68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)
    ))


def _delta_color(value: float) -> tuple[int, int, int]:
    return _interpolate_colors((value + 1.0) / 2.0, ((49, 54, 149), (255, 255, 255), (165, 0, 38)))


def _tangent_color(value: float) -> tuple[int, int, int]:
    return _interpolate_colors(value / 90.0, ((0, 104, 55), (255, 255, 191), (165, 0, 38)))


def png_rgb_bytes(image: np.ndarray) -> bytes:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must be uint8 RGB")

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    height, width, _ = image.shape
    scanlines = b"".join(b"\x00" + image[row].tobytes(order="C") for row in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + chunk(b"IEND", b""))


def render_panel(
    segment: str,
    summary: Mapping[str, Any] | None,
    points: Sequence[Mapping[str, Any]],
    offsets: Sequence[float],
) -> bytes:
    image = np.full((350, 1310, 3), 255, dtype=np.uint8)
    _draw_text(image, 18, 8, segment, scale=2)
    frames = (
        (18, 50, 310, 285),
        (345, 50, 637, 285),
        (672, 50, 964, 285),
        (999, 50, 1292, 285),
    )
    labels = ("PRESENCE 0..1", "DELTA -1..1", "TANGENT 0..90", "MEDIAN PROFILE 0..1")
    for frame, label in zip(frames, labels):
        left, top, right, bottom = frame
        _draw_text(image, left, 32, label, scale=2)
        _line(image, left, top, right, top, (0, 0, 0))
        _line(image, right, top, right, bottom, (0, 0, 0))
        _line(image, right, bottom, left, bottom, (0, 0, 0))
        _line(image, left, bottom, left, top, (0, 0, 0))

    usable = [point for point in points if point.get("complete_profile")]
    if summary is not None and summary.get("raster_shape_rc") and usable:
        rows, columns = summary["raster_shape_rc"]
        for panel_index, field in ((0, "presence_p0"), (1, "delta")):
            left, top, right, bottom = frames[panel_index]
            for point in usable:
                x = left + int(round(float(point["column"]) / max(1, columns - 1) * (right - left)))
                y = top + int(round(float(point["row"]) / max(1, rows - 1) * (bottom - top)))
                value = float(point[field])
                color = _presence_color(value) if panel_index == 0 else _delta_color(value)
                image[max(top, y - 2):min(bottom + 1, y + 3),
                      max(left, x - 2):min(right + 1, x + 3)] = color

        tangent_points = [
            point for point in points
            if point.get("orientation_valid") and point.get("tangent_angle_degrees") is not None
        ]
        left, top, right, bottom = frames[2]
        for point in tangent_points:
            x = left + int(round(float(point["column"]) / max(1, columns - 1) * (right - left)))
            y = top + int(round(float(point["row"]) / max(1, rows - 1) * (bottom - top)))
            image[max(top, y - 2):min(bottom + 1, y + 3),
                  max(left, x - 2):min(right + 1, x + 3)] = _tangent_color(
                      float(point["tangent_angle_degrees"])
                  )

        for panel_index, color_function in (
            (0, _presence_color), (1, _delta_color), (2, _tangent_color)
        ):
            left, _, right, bottom = frames[panel_index]
            for x in range(left, right + 1):
                normalized = (x - left) / max(1, right - left)
                if panel_index == 0:
                    value = normalized
                elif panel_index == 1:
                    value = 2.0 * normalized - 1.0
                else:
                    value = 90.0 * normalized
                image[bottom + 8:bottom + 17, x] = color_function(value)

    profile = None if summary is None else summary.get("median_profile")
    left, top, right, bottom = frames[3]
    for fraction in (0.0, 0.5, 1.0):
        y = bottom - int(round(fraction * (bottom - top)))
        _line(image, left, y, right, y, (220, 220, 220))
    zero_x = left + int(round((0.0 - min(offsets)) / (max(offsets) - min(offsets)) * (right - left)))
    _line(image, zero_x, top, zero_x, bottom, (200, 200, 200))
    if profile is not None:
        pixels = []
        for offset, value in zip(offsets, profile):
            x = left + int(round((offset - min(offsets)) / (max(offsets) - min(offsets)) * (right - left)))
            y = bottom - int(round(min(1.0, max(0.0, float(value))) * (bottom - top)))
            pixels.append((x, y))
        for first, second in zip(pixels, pixels[1:]):
            _line(image, first[0], first[1], second[0], second[1], (20, 20, 20))
        for x, y in pixels:
            image[max(top, y - 2):min(bottom + 1, y + 3),
                  max(left, x - 2):min(right + 1, x + 3)] = (0, 0, 0)
    else:
        _line(image, left, top, right, bottom, (180, 0, 0))
        _line(image, left, bottom, right, top, (180, 0, 0))
        _draw_text(image, left + 100, top + 110, "NO DATA", color=(180, 0, 0), scale=3)
    return png_rgb_bytes(image)


def render_visual_panels(
    output_root: Path,
    visual_segments: Sequence[Mapping[str, str]],
    summaries: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    offsets: Sequence[float],
) -> list[dict[str, Any]]:
    summary_by_segment = {str(row["segment"]): row for row in summaries}
    points_by_segment: dict[str, list[Mapping[str, Any]]] = {}
    for row in point_rows:
        points_by_segment.setdefault(str(row["segment"]), []).append(row)
    receipts = []
    for rank, selected in enumerate(visual_segments, start=1):
        segment = selected["segment"]
        filename = f"panel_{rank:02d}_{segment}.png"
        path = output_root / "panels" / filename
        atomic_write(path, render_panel(
            segment,
            summary_by_segment.get(segment),
            points_by_segment.get(segment, []),
            offsets,
        ))
        receipts.append({
            "rank": rank,
            "segment": segment,
            "selection_sha256": selected["selection_sha256"],
            **_artifact_receipt(path, output_root),
        })
    return receipts


def run_outcome(
    lock: Mapping[str, Any],
    lock_receipt: Mapping[str, Any],
    lock_path: Path,
    cache_root: Path,
    output_root: Path,
    timeout_seconds: float,
    download_workers: int,
    decoded_chunk_cache: int,
    resume_transport: bool = False,
) -> dict[str, Any]:
    implementation = {
        "runner": _source_receipt(Path(__file__).resolve()),
        "tests": _source_receipt(ROOT / "test_pherc0139_fiber_presence_benchmark.py"),
        "requirements_lock": _source_receipt(ROOT / "requirements-fiber-lock.txt"),
        "preregistration": _source_receipt(ROOT / "PHERC0139_FIBER_PRESENCE_PREREG.md"),
    }
    if resume_transport:
        implementation.update({
            "transport_failure_record": _source_receipt(TRANSPORT_FAILURE_RECORD),
            "transport_amendment": _source_receipt(TRANSPORT_AMENDMENT),
        })
    runtime = runtime_receipt()
    required_tag = TRANSPORT_AMENDMENT_TAG if resume_transport else PREREG_TAG
    required_git_paths = [
        ".gitattributes",
        "PHERC0139_FIBER_PRESENCE_PREREG.md",
        "pherc0139_fiber_presence_benchmark.py",
        "pherc0139_fiber_presence_lock.json",
        "pherc0139_fiber_reference_manifest.json",
        "requirements-fiber-lock.txt",
        "test_pherc0139_fiber_presence_benchmark.py",
    ]
    if resume_transport:
        required_git_paths.extend([
            "PHERC0139_FIBER_TRANSPORT_AMENDMENT.md",
            "pherc0139_fiber_transport_failure.json",
        ])
    source_control = git_receipt(
        required_tag,
        required_git_paths,
        required_ancestor=PREREG_COMMIT if resume_transport else None,
    )
    reference_manifest, reference_manifest_receipt, samples_by_segment = (
        load_pinned_reference_manifest(lock, lock_path, cache_root)
    )
    chunk_plan = prediction_chunk_plan(lock, samples_by_segment)
    transport_resume = None
    pinned_missing_chunks = None
    if resume_transport:
        failure, pre_resume_inventory, marker = verify_transport_failure_state(
            lock, lock_receipt, chunk_plan, output_root
        )
        resume_marker = create_transport_resume_marker(output_root, {
            "schema_version": 1,
            "experiment_id": lock["experiment_id"],
            "original_outcome_marker_sha256": sha256_file(marker),
            "failure_record_sha256": implementation["transport_failure_record"]["sha256"],
            "pre_resume_mirror_sha256": pre_resume_inventory["sha256"],
            "git_commit": source_control["commit"],
            "transport_amendment_tag": TRANSPORT_AMENDMENT_TAG,
            "runner_sha256": implementation["runner"]["sha256"],
        })
        transport_resume = {
            "status": "transport-only resume after pre-analysis TLS failure",
            "original_preregistration": failure["preregistration"],
            "failure_record": implementation["transport_failure_record"],
            "amendment": implementation["transport_amendment"],
            "pre_resume_mirror_inventory": pre_resume_inventory,
            "resume_marker": _artifact_receipt(resume_marker, output_root),
        }
        pinned_missing_chunks = {
            channel: {
                tuple(int(value) for value in key.split("/"))
                for key in failure["inferred_http_404_fill_keys"][channel]
            }
            for channel in CHANNEL_NAMES
        }
    else:
        marker = create_outcome_marker(output_root, {
            "schema_version": 1,
            "experiment_id": lock["experiment_id"],
            "lock_sha256": lock_receipt["sha256"],
            "reference_manifest_sha256": reference_manifest_receipt["sha256"],
            "prediction_chunk_plan_sha256": sha256_bytes(canonical_json_bytes(chunk_plan)),
            "git_commit": source_control["commit"],
            "preregistration_tag": PREREG_TAG,
            "runner_sha256": implementation["runner"]["sha256"],
        })
    arrays, prediction_receipt, missing_chunks = mirror_prediction_channels(
        lock,
        chunk_plan,
        output_root,
        timeout_seconds,
        download_workers,
        pinned_missing_chunks=pinned_missing_chunks,
    )
    samplers = {
        "presence": ChunkedArraySampler(
            arrays["presence"],
            float(lock["fiber_artifact"]["channels"]["presence"]["decode_divisor"]),
            decoded_chunk_cache,
        ),
        "nx": ChunkedArraySampler(arrays["nx"], 1.0, decoded_chunk_cache),
        "ny": ChunkedArraySampler(arrays["ny"], 1.0, decoded_chunk_cache),
    }
    sibling_coherence = validate_orientation_sibling_coherence(
        lock, samples_by_segment, samplers["presence"], missing_chunks
    )
    prediction_receipt["orientation_sibling_coherence"] = sibling_coherence

    summaries: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    manifest_segments = {
        str(entry["segment"]): entry for entry in reference_manifest["segments"]
    }
    for segment in lock["reference_segments"]:
        entry = manifest_segments[segment]
        summary, points = analyze_segment(
            segment,
            samples_by_segment[segment],
            entry["raster_shape_rc"],
            lock,
            samplers,
        )
        summaries.append(summary)
        all_points.extend(points)

    primary = evaluate_primary_gate(
        [row["median_delta"] for row in summaries if row.get("median_delta") is not None],
        lock["primary_gate"],
    )
    offsets = [float(value) for value in lock["sampling"]["normal_offsets_prediction_voxels"]]
    analyzable_profiles = [row["median_profile"] for row in summaries if row.get("median_profile")]
    global_profile = (
        [float(value) for value in np.median(np.asarray(analyzable_profiles), axis=0)]
        if analyzable_profiles else None
    )
    orientation = apply_orientation_baseline(lock, summaries, all_points)

    summary_columns = (
        "segment", "status", "selected_samples", "complete_profiles", "median_delta",
        "median_center_presence", "fraction_center_above_control_mean", "selection_sha256",
        "orientation_presence_weighted_median_tangent_angle_degrees",
        "orientation_matched_baseline_degrees", "orientation_improvement_degrees",
    )
    summary_csv_rows = []
    for summary in summaries:
        row = dict(summary)
        orientation_row = summary.get("orientation") or {}
        row.update({
            "orientation_presence_weighted_median_tangent_angle_degrees": orientation_row.get(
                "presence_weighted_median_tangent_angle_degrees"
            ),
            "orientation_matched_baseline_degrees": orientation_row.get(
                "matched_baseline_presence_weighted_median_tangent_angle_degrees"
            ),
            "orientation_improvement_degrees": orientation_row.get(
                "improvement_over_matched_baseline_degrees"
            ),
        })
        summary_csv_rows.append(row)
    point_columns = (
        "segment", "row", "column", "bin_y", "bin_x", "selection_sha256",
        "map_x", "map_y", "map_z", "normal_x", "normal_y", "normal_z",
        "presence_m12", "presence_m8", "presence_m4", "presence_p0",
        "presence_p4", "presence_p8", "presence_p12", "complete_profile",
        "orientation_valid", "delta",
        "fiber_nx_raw", "fiber_ny_raw", "fiber_dx", "fiber_dy", "fiber_dz",
        "tangent_angle_degrees", "matched_baseline_tangent_angle_degrees",
    )
    summary_path = output_root / "segments.csv"
    points_path = output_root / "points.csv"
    atomic_write(summary_path, csv_bytes(summary_csv_rows, summary_columns))
    atomic_write(points_path, csv_bytes(all_points, point_columns))

    visual_segments = lock["visual_segments"]
    panel_receipts = render_visual_panels(
        output_root, visual_segments, summaries, all_points, offsets
    )
    artifacts = {
        "segments_csv": _artifact_receipt(summary_path, output_root),
        "points_csv": _artifact_receipt(points_path, output_root),
        "panels": panel_receipts,
        "panel_specification": {
            "count": int(lock["sampling"]["visual_panel_count"]),
            "presence_scale": [0.0, 1.0],
            "delta_scale": [-1.0, 1.0],
            "tangent_angle_degrees_scale": [0.0, 90.0],
            "profile_scale": [0.0, 1.0],
            "selection": "lowest SHA256(seed|visual|segment)",
        },
    }
    result = {
        "schema_version": 2,
        "experiment_id": lock["experiment_id"],
        "mode": "OUTCOME_AFTER_TRANSPORT_RESUME" if resume_transport else "OUTCOME",
        "decision": primary["decision"],
        "lock": dict(lock_receipt),
        "implementation": implementation,
        "runtime": runtime,
        "source_control": source_control,
        "outcome_marker": _artifact_receipt(marker, output_root),
        **({"transport_resume": transport_resume} if transport_resume is not None else {}),
        "reference_manifest": reference_manifest_receipt,
        "prediction": prediction_receipt,
        "sampling": lock["sampling"],
        "primary": primary,
        "secondary": {
            "median_of_segment_median_profiles": global_profile,
            "orientation": orientation,
        },
        "segments": summaries,
        "artifacts": artifacts,
        "interpretation_limit": (
            "Tests localization to pre-existing public surface maps; it does not establish "
            "fiber tracing accuracy, surface-map ground truth, pipeline improvement, or ink recovery."
        ),
    }
    result_path = output_root / "result.json"
    atomic_write(result_path, canonical_json_bytes(result))
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--run", action="store_true", help="execute the single locked outcome run")
    action.add_argument(
        "--resume-transport",
        action="store_true",
        help="execute the one publicly amended resume after the recorded TLS failure",
    )
    action.add_argument(
        "--prepare-references",
        action="store_true",
        help="verify/download references and write the prediction-free manifest",
    )
    action.add_argument("--plan", action="store_true", help="write the plan without network or prediction reads")
    action.add_argument("--dry-run", action="store_true", help="alias for --plan")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results" / "pherc0139_fiber_cache")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTCOME_DIR)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--decoded-chunk-cache", type=int, default=64)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    if args.download_workers <= 0:
        parser.error("--download-workers must be positive")
    if args.decoded_chunk_cache <= 0:
        parser.error("--decoded-chunk-cache must be positive")
    if (args.run or args.resume_transport) and args.lock.resolve() != DEFAULT_LOCK.resolve():
        parser.error("outcome execution requires the canonical preregistered lock path")
    if (args.run or args.resume_transport) and args.out_dir.resolve() != DEFAULT_OUTCOME_DIR.resolve():
        parser.error("outcome execution requires the canonical preregistered outcome directory")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lock, lock_receipt = load_lock(args.lock)
        if args.plan or args.dry_run:
            plan = build_plan(lock, lock_receipt)
            path = args.out_dir / "plan.json"
            atomic_write(path, canonical_json_bytes(plan))
            print(f"PLAN_ONLY: {len(lock['reference_segments'])} locked segments")
            print(f"prediction array opened: {plan['prediction_array_opened']}")
            print(path)
            return 0

        if args.prepare_references:
            if lock["reference_manifest"].get("sha256") is not None:
                raise BenchmarkError("refusing to replace an already pinned reference manifest")
            if lock.get("frozen_at_utc") is not None:
                raise BenchmarkError("reference preparation requires null frozen_at_utc")
            manifest, path = prepare_reference_manifest(
                lock,
                args.lock,
                args.cache_dir,
                args.timeout_seconds,
                args.download_workers,
            )
            print(f"REFERENCE_PREPARATION: {len(manifest['segments'])} locked segments")
            print("prediction array opened: False")
            print("prediction chunk reads: False")
            print(f"manifest sha256: {sha256_file(path)}")
            print(path)
            return 0

        result = run_outcome(
            lock,
            lock_receipt,
            args.lock,
            args.cache_dir,
            args.out_dir,
            args.timeout_seconds,
            args.download_workers,
            args.decoded_chunk_cache,
            resume_transport=args.resume_transport,
        )
        print(result["decision"])
        print(args.out_dir / "result.json")
        return 0
    except (BenchmarkError, OSError, ValueError) as exc:
        if getattr(args, "resume_transport", False):
            try:
                failure_path = record_transport_resume_failure(args.out_dir, exc)
                if failure_path is not None:
                    print(f"transport resume failure record: {failure_path}", file=sys.stderr)
            except (BenchmarkError, OSError, ValueError) as record_error:
                print(f"could not persist transport resume failure: {record_error}", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
