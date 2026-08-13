#!/usr/bin/env python3
"""Render and verify a fixed independent-scan review pack.

This is deliberately outside the frozen efficacy decision.  It consumes the eight visual
cases fixed by the execution lock and renders the separately acquired scan in registration
with the automated reference mask and model probabilities.  It neither computes nor changes
the terminal machine verdict.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import map_coordinates

import crossscan_finetune as C
import run_crossscan_finetune as R
import score_crossscan_finetune as S


PACK_SCHEMA = "crossscan-highres-review-pack-v1"
REVIEW_SCHEMA = "crossscan-highres-human-review-v1"
UPSTREAM_COMMIT = "b24e028178f2c8720ba1d16ac53d5f0b6ac00da7"
UPSTREAM_REPOSITORY = "https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit"
REVIEW_ACKNOWLEDGEMENT = (
    "I reviewed every fixed panel against the registered independent-scan image; "
    "the reference masks are automated proxies, not official or human ground truth."
)
REVIEW_RECOMMENDATIONS = {
    "RELEASE_WITH_AGREEMENT_ONLY",
    "RELEASE_WITH_NAMED_IMAGE_SUPPORTED_CASES",
    "DO_NOT_RELEASE",
}
REVIEW_SLICES_L1 = (16, 32, 48)
PUBLIC_L1_VOXEL_UM = 18.724
ALIGNMENT_VALUES = {"PASS", "FAIL", "UNCERTAIN"}
INITIAL_SUPPORT_VALUES = {
    "SUPPORTED", "NOT_SUPPORTED", "MIXED", "NOT_ASSESSABLE"
}
CHANGE_SUPPORT_VALUES = {
    "CORRECTION", "HARM", "MIXED", "NO_VISIBLE_CHANGE", "NOT_ASSESSABLE"
}


SOURCE_SCANS: dict[str, dict[str, Any]] = {
    "PHerc0139": {
        "moving_volume": "20260413113053-1.129um-0.2m-59keV-masked.zarr",
        "fixed_volume": "20250728140407-9.362um-1.2m-113keV-masked.zarr",
        "source_level": 2,
        "source_level_voxel_um": 4.516,
        "caster_level": 4,
        "material_threshold": 65,
        "registration_heldout_um": 4.09,
        "transform_file": "results/pass3_final.npz",
        "transform_sha256": (
            "609a0cc6593550f3ffed4f579a3a81c23b71d1bccda19f35cc8e991033c46681"
        ),
        "transform_kind": "npz_hi_l4_to_lo_l2",
        "published_transform_file": "20260413113053-to-20250728140407.json",
        "published_transform_sha256": (
            "40de4d0cd161cff5b7eff672e8072d260834efc3b444120c74e253e007acb838"
        ),
    },
    "PHerc1203": {
        "moving_volume": "20260319130212-2.403um-0.2m-77keV-masked.zarr",
        "fixed_volume": "20250820131727-9.362um-1.2m-113keV-masked.zarr",
        "source_level": 1,
        "source_level_voxel_um": 4.806,
        "caster_level": 3,
        "material_threshold": 56,
        "registration_heldout_um": 2.38,
        "transform_file": "results/1203/pass3_final.npz",
        "transform_sha256": (
            "2d92f4558ed90408612db74f7238c09af1d6051069d0dffcc460f4d0e3b1af3c"
        ),
        "transform_kind": "npz_hi_l4_to_lo_l2",
    },
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(_canonical(unsigned).encode("ascii")).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "path": relative.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_hashed(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("content_sha256") != _content_hash(value):
        raise ValueError(f"content hash mismatch: {path}")
    return value


def level_coordinate_to_l0(value: np.ndarray | float, level: int):
    """Map a pyramid-level voxel-centre coordinate to level 0."""
    if type(level) is not int or level < 0:
        raise ValueError(f"invalid pyramid level: {level!r}")
    scale = 1 << level
    return np.asarray(value) * scale + (scale - 1) / 2.0


def l0_coordinate_to_level(value: np.ndarray | float, level: int):
    """Map a level-0 voxel-centre coordinate to a pyramid level."""
    if type(level) is not int or level < 0:
        raise ValueError(f"invalid pyramid level: {level!r}")
    scale = 1 << level
    return (np.asarray(value) - (scale - 1) / 2.0) / scale


def load_registration(scroll: str, transform_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if scroll not in SOURCE_SCANS:
        raise ValueError(f"unsupported scroll: {scroll}")
    cfg = SOURCE_SCANS[scroll]
    path = (transform_root / cfg["transform_file"]).resolve()
    path.relative_to(transform_root.resolve())
    if not path.is_file() or _sha256_file(path) != cfg["transform_sha256"]:
        raise ValueError(f"pinned transform identity mismatch: {path}")
    if cfg["transform_kind"] == "npz_hi_l4_to_lo_l2":
        with np.load(path, allow_pickle=False) as value:
            if "M2" not in value.files or "t2" not in value.files:
                raise ValueError("PHerc1203 transform archive omits M2 or t2")
            matrix = np.asarray(value["M2"], dtype=np.float64)
            offset = np.asarray(value["t2"], dtype=np.float64)
        published_check = None
        if scroll == "PHerc0139":
            published_path = (transform_root / cfg["published_transform_file"]).resolve()
            published_path.relative_to(transform_root.resolve())
            if (
                not published_path.is_file()
                or _sha256_file(published_path) != cfg["published_transform_sha256"]
            ):
                raise ValueError("pinned PHerc0139 published transform identity mismatch")
            published = json.loads(published_path.read_text(encoding="utf-8"))
            affine = np.asarray(published.get("transformation_matrix"), dtype=np.float64)
            moving = np.asarray(published.get("moving_landmarks"), dtype=np.float64)
            fixed = np.asarray(published.get("fixed_landmarks"), dtype=np.float64)
            if (
                published.get("schema_version") != "1.0.0"
                or affine.shape != (3, 4)
                or moving.ndim != 2 or moving.shape[1:] != (3,)
                or moving.shape != fixed.shape or len(moving) < 4
                or not np.isfinite(moving).all() or not np.isfinite(fixed).all()
            ):
                raise ValueError("PHerc0139 published transform provenance mismatch")
            errors = np.linalg.norm(
                (affine[:, :3] @ moving.T).T + affine[:, 3] - fixed, axis=1
            )
            if float(np.max(errors)) > 2.0:
                raise ValueError("PHerc0139 published-transform landmark check failed")
            published_check = {
                "file": {
                    "path": cfg["published_transform_file"],
                    "bytes": published_path.stat().st_size,
                    "sha256": cfg["published_transform_sha256"],
                },
                "paired_landmarks": int(len(moving)),
                "median_fixed_l0_voxels": float(np.median(errors)),
                "maximum_fixed_l0_voxels": float(np.max(errors)),
                "note": (
                    "This public L0 transform is an auxiliary landmark check; rendering "
                    "uses the pinned pass3 matrix consumed by the label caster."
                ),
            }
    else:  # pragma: no cover - constant universe is tested
        raise AssertionError(cfg["transform_kind"])
    if (
        matrix.shape != (3, 3)
        or offset.shape != (3,)
        or not np.isfinite(matrix).all()
        or not np.isfinite(offset).all()
        or abs(float(np.linalg.det(matrix))) < 1e-12
    ):
        raise ValueError(f"invalid or singular transform: {path}")
    return matrix, offset, {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "file": {
            "path": cfg["transform_file"],
            "bytes": path.stat().st_size,
            "sha256": cfg["transform_sha256"],
        },
        "mapping": "independent-scan L4 to public-scan L2; exact label-caster matrix",
        "published_transform_landmark_check": published_check,
    }


def registered_source_coordinates(
    scroll: str,
    registration_matrix: np.ndarray,
    registration_offset: np.ndarray,
    low_global_l1_origin: Iterable[int],
    score_slice_l1: int,
    source_level: int,
    size_l1: int = 64,
    pixels_per_l1: int = 4,
) -> np.ndarray:
    """Return source-level z/y/x coordinates for one registered oblique plane."""
    origin = np.asarray(list(low_global_l1_origin), dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("low L1 origin must contain three finite coordinates")
    if type(score_slice_l1) is not int or not 0 <= score_slice_l1 < size_l1:
        raise ValueError("fixed score slice is outside the score cube")
    if type(pixels_per_l1) is not int or pixels_per_l1 < 1:
        raise ValueError("pixels_per_l1 must be a positive integer")
    matrix = np.asarray(registration_matrix, dtype=np.float64)
    offset = np.asarray(registration_offset, dtype=np.float64)
    if matrix.shape != (3, 3) or offset.shape != (3,):
        raise ValueError("registration affine has invalid shape")
    inverse = np.linalg.inv(matrix)
    n = size_l1 * pixels_per_l1
    subpixel = (np.arange(n, dtype=np.float64) + 0.5) / pixels_per_l1 - 0.5
    yy, xx = np.meshgrid(origin[1] + subpixel, origin[2] + subpixel, indexing="ij")
    zz = np.full_like(yy, origin[0] + score_slice_l1)
    low_l1 = np.stack((zz, yy, xx), axis=0).reshape(3, -1)
    # Reproduce pass10.py/pass10_1203.py exactly. Both pinned NPZ transforms map
    # independent-scan L4 to public-scan L2. The caster doubles the low grid,
    # keeps the linear part, and reads the source at cfg['caster_level'].
    if scroll not in SOURCE_SCANS:
        raise ValueError(f"unsupported scroll: {scroll}")
    caster_level = int(SOURCE_SCANS[scroll]["caster_level"])
    if source_level > caster_level:
        raise ValueError("source pyramid level is coarser than the label-caster level")
    high_caster = inverse @ (low_l1 - (2.0 * offset[:, None] + 0.5))
    high_level = high_caster * float(1 << (caster_level - source_level))
    return high_level.reshape(3, n, n)


@dataclass(frozen=True)
class ZarrMetadata:
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    dtype: str
    fill_value: int
    metadata_sha256: str


class HttpRawZarr:
    """Minimal fail-closed reader for the uncompressed uint8 public scan pyramids."""

    def __init__(self, base_url: str, cache_root: Path, cache_name: str, timeout: int = 120):
        if not base_url.startswith("https://"):
            raise ValueError("scan URL must use HTTPS")
        self.base_url = base_url.rstrip("/")
        self.cache_root = (cache_root / cache_name).resolve()
        self.cache_root.relative_to(cache_root.resolve())
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        metadata_payload = self._get(".zarray", allow_missing=False)[0]
        value = json.loads(metadata_payload.decode("utf-8"))
        allowed = {
            "zarr_format": 2,
            "dtype": "|u1",
            "order": "C",
            "filters": None,
            "compressor": None,
            "dimension_separator": "/",
            "fill_value": 0,
        }
        if any(value.get(key) != expected for key, expected in allowed.items()):
            raise ValueError(f"unsupported or changed scan Zarr metadata: {value}")
        shape = tuple(value.get("shape", ()))
        chunks = tuple(value.get("chunks", ()))
        if (
            len(shape) != 3
            or len(chunks) != 3
            or any(type(v) is not int or v <= 0 for v in (*shape, *chunks))
        ):
            raise ValueError("scan Zarr shape/chunks must be positive 3-vectors")
        self.metadata = ZarrMetadata(
            shape=shape,
            chunks=chunks,
            dtype="|u1",
            fill_value=0,
            metadata_sha256=_sha256_bytes(metadata_payload),
        )
        self.access_records: dict[str, dict[str, Any]] = {}

    def _get(self, relative: str, allow_missing: bool) -> tuple[bytes, bool]:
        destination = self.cache_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            payload = destination.read_bytes()
            return payload, False
        url = f"{self.base_url}/{relative.replace(chr(92), '/')}"
        request = urllib.request.Request(url, headers={"User-Agent": "crossscan-review-v1"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise RuntimeError(f"unexpected HTTP status {response.status}: {url}")
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if allow_missing and exc.code == 404:
                return b"", True
            raise RuntimeError(f"scan fetch failed: {url}: HTTP {exc.code}") from exc
        temporary = destination.with_name(destination.name + ".tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.write_bytes(payload)
        temporary.replace(destination)
        return payload, False

    def read_roi(self, start: Iterable[int], stop: Iterable[int]) -> np.ndarray:
        first = np.asarray(list(start), dtype=np.int64)
        last = np.asarray(list(stop), dtype=np.int64)
        shape = np.asarray(self.metadata.shape, dtype=np.int64)
        chunks = np.asarray(self.metadata.chunks, dtype=np.int64)
        if first.shape != (3,) or last.shape != (3,) or np.any(last <= first):
            raise ValueError("ROI bounds must be increasing z/y/x triplets")
        if np.any(first < 0) or np.any(last > shape):
            raise ValueError(f"ROI outside scan shape: {first.tolist()}..{last.tolist()}")
        extent = last - first
        if np.any(extent > 768) or int(np.prod(extent)) > 80_000_000:
            raise ValueError(f"refusing unexpectedly large scan ROI: {extent.tolist()}")
        output = np.zeros(tuple(map(int, extent)), dtype=np.uint8)
        chunk_first = first // chunks
        chunk_last = (last - 1) // chunks
        for cz in range(int(chunk_first[0]), int(chunk_last[0]) + 1):
            for cy in range(int(chunk_first[1]), int(chunk_last[1]) + 1):
                for cx in range(int(chunk_first[2]), int(chunk_last[2]) + 1):
                    index = np.asarray([cz, cy, cx], dtype=np.int64)
                    relative = f"{cz}/{cy}/{cx}"
                    payload, missing = self._get(relative, allow_missing=True)
                    record = {
                        "path": relative,
                        "url": f"{self.base_url}/{relative}",
                        "missing_fill_chunk": bool(missing),
                    }
                    if missing:
                        block = np.zeros(tuple(self.metadata.chunks), dtype=np.uint8)
                        record.update({"bytes": 0, "sha256": None})
                    else:
                        expected_bytes = int(np.prod(chunks))
                        if len(payload) != expected_bytes:
                            raise ValueError(
                                f"raw scan chunk {relative} has {len(payload)} bytes; "
                                f"expected {expected_bytes}"
                            )
                        block = np.frombuffer(payload, dtype=np.uint8).reshape(
                            self.metadata.chunks
                        )
                        record.update({"bytes": len(payload), "sha256": _sha256_bytes(payload)})
                    self.access_records[relative] = record
                    block_start = index * chunks
                    overlap_start = np.maximum(first, block_start)
                    overlap_stop = np.minimum(last, block_start + chunks)
                    if np.any(overlap_stop <= overlap_start):  # pragma: no cover
                        continue
                    src = tuple(
                        slice(int(overlap_start[a] - block_start[a]),
                              int(overlap_stop[a] - block_start[a]))
                        for a in range(3)
                    )
                    dst = tuple(
                        slice(int(overlap_start[a] - first[a]),
                              int(overlap_stop[a] - first[a]))
                        for a in range(3)
                    )
                    output[dst] = block[src]
        return output


def sample_registered_plane(reader: Any, coordinates: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[0] != 3 or not np.isfinite(coords).all():
        raise ValueError("registered source coordinates must have shape (3,H,W) and be finite")
    start = np.floor(coords.reshape(3, -1).min(axis=1)).astype(int) - 1
    stop = np.ceil(coords.reshape(3, -1).max(axis=1)).astype(int) + 2
    roi = reader.read_roi(start, stop)
    local = coords - start[:, None, None]
    plane = map_coordinates(roi, local, order=1, mode="constant", cval=0, prefilter=False)
    if plane.shape != coords.shape[1:] or not np.isfinite(plane).all():
        raise RuntimeError("registered scan resampling failed")
    return plane.astype(np.float32), {
        "source_roi_start": start.tolist(),
        "source_roi_stop": stop.tolist(),
        "interpolation": "trilinear order=1 in zyx voxel-centre coordinates",
    }


def _nearest_up(value: np.ndarray, factor: int) -> np.ndarray:
    out = np.asarray(value)
    for axis in range(2):
        out = np.repeat(out, factor, axis=axis)
    return out


def _context_ct(ct: np.ndarray, score_slice_l1: int) -> np.ndarray:
    value = np.asarray(ct)
    if value.shape != (256, 256, 256) or value.dtype != np.uint8:
        raise ValueError(f"unexpected public CT context: {value.shape} {value.dtype}")
    if type(score_slice_l1) is not int or not 0 <= score_slice_l1 < 64:
        raise ValueError("score slice is outside the central 64-L1 cube")
    z_l0 = 64 + 2 * score_slice_l1
    plane = value[z_l0:z_l0 + 2].mean(axis=0)
    return plane.reshape(128, 2, 128, 2).mean(axis=(1, 3))


def render_case_panel(
    path: Path,
    case_id: str,
    scroll: str,
    public_ct: np.ndarray,
    source_scan: np.ndarray,
    reference_bits: np.ndarray,
    initial_probability: np.ndarray,
    candidate_probability: np.ndarray,
    source_level_voxel_um: float,
    registration_heldout_um: float,
    score_slice_l1: int = 32,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    source = np.asarray(source_scan, dtype=np.float32)
    bits = np.asarray(reference_bits)
    initial = np.asarray(initial_probability, dtype=np.float32)
    candidate = np.asarray(candidate_probability, dtype=np.float32)
    if source.shape != (256, 256):
        raise ValueError(f"registered source panel must be 256x256, got {source.shape}")
    if bits.shape != (64, 64) or bits.dtype.kind not in "ui":
        raise ValueError("reference slice must be a 64x64 integer bit mask")
    if initial.shape != (64, 64) or candidate.shape != (64, 64):
        raise ValueError("probability slices must be 64x64")
    if not (np.isfinite(initial).all() and np.isfinite(candidate).all()):
        raise ValueError("probability slice contains nonfinite values")
    if not (np.logical_and(initial >= 0, initial <= 1).all()
            and np.logical_and(candidate >= 0, candidate <= 1).all()):
        raise ValueError("probability slice outside [0,1]")
    reference = ((bits & 1) != 0) & ((bits & 8) != 0)
    ref4 = _nearest_up(reference, 4)
    initial4 = _nearest_up(initial, 4)
    candidate4 = _nearest_up(candidate, 4)
    delta4 = candidate4 - initial4
    context = _context_ct(public_ct, score_slice_l1)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), constrained_layout=True)
    flat = list(axes.ravel())
    flat[0].imshow(context, cmap="gray", vmin=0, vmax=255)
    flat[0].add_patch(Rectangle((32, 32), 64, 64, fill=False, ec="#ffd34e", lw=2))
    flat[0].set_title("Public 9.362 µm CT context\n(yellow = fixed score cube)")

    for axis in flat[1:5]:
        axis.imshow(source, cmap="gray", vmin=0, vmax=255)
    flat[1].set_title(
        f"Registered independent scan\nsource pyramid ≈ {source_level_voxel_um:.3f} µm/voxel"
    )
    flat[2].contour(ref4.astype(float), levels=[0.5], colors=["#35e86f"], linewidths=1.1)
    flat[2].set_title("Automated scan-derived proxy contour\n(green; not human/official GT)")
    flat[3].contour(initial4, levels=[0.2], colors=["#ff43cf"], linewidths=1.0)
    flat[3].contour(ref4.astype(float), levels=[0.5], colors=["#35e86f"], linewidths=0.8)
    flat[3].set_title("Initial m7 p=0.2 contour\nmagenta; proxy in green")
    flat[4].contour(candidate4, levels=[0.2], colors=["#22d8ff"], linewidths=1.0)
    flat[4].contour(ref4.astype(float), levels=[0.5], colors=["#35e86f"], linewidths=0.8)
    flat[4].set_title("Six-seed fine-tuned mean p=0.2\ncyan; proxy in green")

    probability_images = []
    for axis, value, title in (
        (flat[5], initial4, "Initial m7 probability"),
        (flat[6], candidate4, "Six-seed fine-tuned mean probability"),
    ):
        axis.imshow(source, cmap="gray", vmin=0, vmax=255)
        probability_images.append(axis.imshow(value, cmap="magma", vmin=0, vmax=1, alpha=0.68))
        axis.contour(ref4.astype(float), levels=[0.5], colors=["#35e86f"], linewidths=0.7)
        axis.set_title(title + "\n(proxy contour in green)")
    flat[7].imshow(source, cmap="gray", vmin=0, vmax=255)
    change = flat[7].imshow(delta4, cmap="coolwarm", vmin=-0.5, vmax=0.5, alpha=0.72)
    flat[7].contour(ref4.astype(float), levels=[0.5], colors=["#35e86f"], linewidths=0.7)
    flat[7].set_title("Fine-tuned − initial probability\n(proxy contour in green)")

    for axis in flat:
        axis.set_xticks([])
        axis.set_yticks([])
    # A 500 µm scale bar in the registered-source panels.
    registered_pixel_um = PUBLIC_L1_VOXEL_UM / 4.0
    bar_pixels = 500.0 / registered_pixel_um
    for axis in flat[1:]:
        axis.plot([12, 12 + bar_pixels], [242, 242], color="white", lw=3, solid_capstyle="butt")
        axis.text(12, 234, "500 µm", color="white", fontsize=8, va="bottom")
    fig.colorbar(probability_images[-1], ax=[flat[5], flat[6]], fraction=0.035, pad=0.01)
    fig.colorbar(change, ax=flat[7], fraction=0.046, pad=0.02)
    fig.suptitle(
        f"{case_id} | fixed pre-outcome case | score slice k={score_slice_l1} | {scroll}\n"
        f"Independent-scan registration held-out median error {registration_heldout_um:.2f} µm. "
        "Visual evidence supports review; it is not ground truth.",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=160, metadata={"Software": "crossscan_highres_review.py"})
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"review panel was not written: {path}")


def _case_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return R._case_map(plan)


def _candidate_mean(
    data_root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    case: dict[str, Any],
    selected_steps: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    scroll = str(case["scroll"])
    scope = "primary" if scroll == C.TRAIN_SCROLL else "safety"
    predictions = []
    identities = []
    for seed in C.INFERENTIAL_SEEDS:
        if scroll == C.TRAIN_SCROLL:
            fold = S.fold_for_stratum(plan, int(case["z_stratum"]))
            prediction = S.load_prediction(
                data_root, plan, lock, case["case_id"], "finetuned", scope,
                seed=seed, steps=selected_steps, fold=fold,
            )
            folds = [fold]
        else:
            fold_predictions = [
                S.load_prediction(
                    data_root, plan, lock, case["case_id"], "finetuned", scope,
                    seed=seed, steps=selected_steps, fold=fold,
                ) for fold in ("even", "odd")
            ]
            prediction = np.mean(np.stack(fold_predictions), axis=0)
            folds = ["even", "odd"]
        predictions.append(prediction)
        identities.append({"seed": seed, "folds": folds, "steps": selected_steps})
    return np.mean(np.stack(predictions), axis=0), identities


def build_review_pack(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    data_root = args.data_root.resolve()
    transform_root = args.transform_root.resolve()
    cache_root = args.cache_root.resolve()
    out = args.out.resolve()
    staging = out.with_name(out.name + ".tmp")
    if out.exists() or staging.exists():
        raise FileExistsError(f"review output already exists: {out} or {staging}")
    plan = _load_hashed(repo / "results/crossscan_finetune/plan.json")
    lock = _load_hashed(repo / "results/crossscan_finetune/execution_lock.json")
    result = _load_hashed(data_root / "final_result.json")
    expected_result = {
        "schema_version": "crossscan-final-result-v1",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
    }
    mismatches = [key for key, value in expected_result.items() if result.get(key) != value]
    if mismatches:
        raise ValueError(f"review inputs disagree with the frozen result: {mismatches}")
    selected_steps = result.get("selected_steps")
    if type(selected_steps) is not int or selected_steps <= 0:
        raise ValueError("final result has invalid selected_steps")
    visuals = lock.get("resolved_protocol", {}).get("visual_cases")
    if (
        not isinstance(visuals, list)
        or len(visuals) != 8
        or {v.get("scroll") for v in visuals} != set(SOURCE_SCANS)
    ):
        raise ValueError("execution lock must contain the exact eight two-scroll visual cases")

    staging.mkdir(parents=True)
    cases = _case_map(plan)
    readers: dict[tuple[str, str], HttpRawZarr] = {}
    transforms: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for scroll, cfg in SOURCE_SCANS.items():
        matrix, offset, provenance = load_registration(scroll, transform_root)
        transforms[scroll] = matrix, offset, provenance
        for role, level in (
            ("display", int(cfg["source_level"])),
            ("label_caster", int(cfg["caster_level"])),
        ):
            url = (
                "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
                f"{scroll}/volumes/{cfg['moving_volume']}/{level}"
            )
            readers[(scroll, role)] = HttpRawZarr(
                url, cache_root,
                f"{scroll}-{Path(cfg['moving_volume']).stem}-L{level}",
            )

    records = []
    for visual in visuals:
        case_id = str(visual["case_id"])
        scroll = str(visual["scroll"])
        locked_score_slice = int(visual["score_slice_l1"])
        if case_id not in cases or cases[case_id]["scroll"] != scroll:
            raise ValueError(f"locked visual case is absent from the plan: {case_id}")
        case = cases[case_id]
        ct, bits, _ = R.verify_evaluation_case(plan, lock, data_root, case_id)
        assert ct is not None and bits is not None
        S.truth_masks(bits)
        scope = "primary" if scroll == C.TRAIN_SCROLL else "safety"
        initial = S.load_prediction(data_root, plan, lock, case_id, "initial", scope)
        candidate, candidate_identities = _candidate_mean(
            data_root, plan, lock, case, selected_steps
        )
        local_origin = np.asarray(case["local_origin_l1"], dtype=int)
        label_origin = np.asarray(C.SCROLLS[scroll]["label_origin_l1"], dtype=int)
        global_origin = local_origin + label_origin
        matrix, offset, transform_provenance = transforms[scroll]
        cfg = SOURCE_SCANS[scroll]
        for score_slice in REVIEW_SLICES_L1:
            display_coordinates = registered_source_coordinates(
                scroll, matrix, offset, global_origin, score_slice,
                int(cfg["source_level"]),
            )
            source_plane, sampling = sample_registered_plane(
                readers[(scroll, "display")], display_coordinates
            )
            caster_coordinates = registered_source_coordinates(
                scroll, matrix, offset, global_origin, score_slice,
                int(cfg["caster_level"]), size_l1=64, pixels_per_l1=1,
            )
            caster_plane, caster_sampling = sample_registered_plane(
                readers[(scroll, "label_caster")], caster_coordinates
            )
            label_slice = bits[score_slice]
            valid = (label_slice & 1) != 0
            material = (label_slice & 2) != 0
            reconstructed_material = caster_plane > int(cfg["material_threshold"])
            mismatch = int(np.count_nonzero(reconstructed_material[valid] != material[valid]))
            if mismatch:
                raise ValueError(
                    f"registered source does not reproduce label-caster material bits: "
                    f"{case_id} k={score_slice}, mismatches={mismatch}/{int(valid.sum())}"
                )
            panel_id = f"{case_id}-k{score_slice:02d}"
            path = staging / "panels" / f"{panel_id}.png"
            render_case_panel(
                path, case_id, scroll, ct, source_plane, label_slice,
                initial[score_slice], candidate[score_slice],
                float(cfg["source_level_voxel_um"]),
                float(cfg["registration_heldout_um"]), score_slice,
            )
            records.append({
                "panel_id": panel_id,
                "case_id": case_id,
                "scroll": scroll,
                "z_stratum": int(visual["z_stratum"]),
                "score_slice_l1": score_slice,
                "is_original_machine_panel_slice": score_slice == locked_score_slice,
                "global_score_origin_l1_zyx": global_origin.tolist(),
                "panel": _file_record(path, f"panels/{path.name}"),
                "source_scan": {
                    "volume": cfg["moving_volume"],
                    "display_level": cfg["source_level"],
                    "display_voxel_um": cfg["source_level_voxel_um"],
                    "display_zarr_metadata_sha256": readers[
                        (scroll, "display")
                    ].metadata.metadata_sha256,
                    "label_caster_level": cfg["caster_level"],
                    "label_caster_zarr_metadata_sha256": readers[
                        (scroll, "label_caster")
                    ].metadata.metadata_sha256,
                },
                "registration": transform_provenance,
                "sampling": sampling,
                "label_caster_reproduction": {
                    **caster_sampling,
                    "material_threshold": cfg["material_threshold"],
                    "valid_voxels": int(valid.sum()),
                    "material_voxels": int(material.sum()),
                    "mismatches_inside_valid": mismatch,
                },
                "candidate_ensemble": candidate_identities,
            })
    for visual in visuals:
        case_records = [
            record for record in records if record["case_id"] == visual["case_id"]
        ]
        verified_slices = sum(
            record["label_caster_reproduction"]["valid_voxels"] > 0
            for record in case_records
        )
        if len(case_records) != len(REVIEW_SLICES_L1) or verified_slices < 2:
            raise ValueError(
                f"fixed review slices do not provide two caster-verifiable panels: "
                f"{visual['case_id']}"
            )
    source_access: dict[str, dict[str, Any]] = {}
    for (scroll, role), reader in readers.items():
        source_access.setdefault(scroll, {})[role] = {
            "base_url": reader.base_url,
            "metadata": {
                "shape": list(reader.metadata.shape),
                "chunks": list(reader.metadata.chunks),
                "dtype": reader.metadata.dtype,
                "fill_value": reader.metadata.fill_value,
                "sha256": reader.metadata.metadata_sha256,
            },
            "chunks": [reader.access_records[key] for key in sorted(reader.access_records)],
        }
    manifest = {
        "schema_version": PACK_SCHEMA,
        "status": "RENDERED_NOT_HUMAN_REVIEWED",
        "created_utc": R.utc_now(),
        "claim_boundary": (
            "Panels compare models with automated registered scan-derived proxy masks and "
            "the independent scan image. They are not official or human ground truth."
        ),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "final_result_content_sha256": result["content_sha256"],
        "selected_steps": selected_steps,
        "review_slices_l1": list(REVIEW_SLICES_L1),
        "label_caster_reproduction": {
            "panels": len(records),
            "panels_with_valid_voxels": sum(
                record["label_caster_reproduction"]["valid_voxels"] > 0
                for record in records
            ),
            "valid_voxels": sum(
                record["label_caster_reproduction"]["valid_voxels"]
                for record in records
            ),
            "mismatches_inside_valid": sum(
                record["label_caster_reproduction"]["mismatches_inside_valid"]
                for record in records
            ),
        },
        "cases": records,
        "source_access": source_access,
        "tool": _file_record(Path(__file__).resolve(), Path(__file__).name),
    }
    manifest["content_sha256"] = _content_hash(manifest)
    R.atomic_write_json(staging / "review_pack.json", manifest)
    template = {
        "schema_version": REVIEW_SCHEMA,
        "review_pack_content_sha256": manifest["content_sha256"],
        "reviewer": "",
        "reviewed_utc": "",
        "acknowledgement": REVIEW_ACKNOWLEDGEMENT,
        "release_recommendation": "DO_NOT_RELEASE",
        "supported_panel_ids": [],
        "cases": [{
            "panel_id": case["panel_id"],
            "case_id": case["case_id"],
            "score_slice_l1": case["score_slice_l1"],
            "panel_sha256": case["panel"]["sha256"],
            "registration_alignment": "UNCERTAIN",
            "initial_disagreement_image_support": "NOT_ASSESSABLE",
            "candidate_change_image_support": "NOT_ASSESSABLE",
            "notes": "",
        } for case in records],
    }
    (staging / "human_review_TEMPLATE.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    staging.replace(out)
    return load_review_pack(out)


def load_review_pack(root: Path) -> dict[str, Any]:
    base = root.resolve()
    manifest = _load_hashed(base / "review_pack.json")
    if manifest.get("schema_version") != PACK_SCHEMA:
        raise ValueError("unexpected high-resolution review-pack schema")
    if (
        manifest.get("status") != "RENDERED_NOT_HUMAN_REVIEWED"
        or manifest.get("review_slices_l1") != list(REVIEW_SLICES_L1)
    ):
        raise ValueError("review pack status or fixed slice set changed")
    cases = manifest.get("cases")
    expected_count = len(SOURCE_SCANS) * C.Z_STRATA * len(REVIEW_SLICES_L1)
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise ValueError(f"review pack must contain exactly {expected_count} fixed panels")
    expected_pairs = {
        (scroll, stratum, score_slice)
        for scroll in SOURCE_SCANS
        for stratum in range(C.Z_STRATA)
        for score_slice in REVIEW_SLICES_L1
    }
    if {
        (case.get("scroll"), case.get("z_stratum"), case.get("score_slice_l1"))
        for case in cases
    } != expected_pairs:
        raise ValueError("review pack does not cover each scroll/z-stratum/slice exactly once")
    panel_ids = [case.get("panel_id") for case in cases]
    if any(not isinstance(value, str) or not value for value in panel_ids) or (
        len(panel_ids) != len(set(panel_ids))
    ):
        raise ValueError("review pack panel IDs must be unique nonempty strings")
    for case in cases:
        reproduction = case.get("label_caster_reproduction")
        if (
            not isinstance(reproduction, dict)
            or type(reproduction.get("valid_voxels")) is not int
            or reproduction["valid_voxels"] < 0
            or reproduction.get("mismatches_inside_valid") != 0
        ):
            raise ValueError("review panel failed the label-caster reproduction gate")
    for case_id in {case["case_id"] for case in cases}:
        if sum(
            case["label_caster_reproduction"]["valid_voxels"] > 0
            for case in cases if case["case_id"] == case_id
        ) < 2:
            raise ValueError("review case lacks two caster-verifiable fixed slices")
    aggregate = manifest.get("label_caster_reproduction")
    expected_aggregate = {
        "panels": len(cases),
        "panels_with_valid_voxels": sum(
            case["label_caster_reproduction"]["valid_voxels"] > 0 for case in cases
        ),
        "valid_voxels": sum(
            case["label_caster_reproduction"]["valid_voxels"] for case in cases
        ),
        "mismatches_inside_valid": 0,
    }
    if aggregate != expected_aggregate:
        raise ValueError("review-pack label-caster aggregate is inconsistent")
    expected_tool = _file_record(Path(__file__).resolve(), Path(__file__).name)
    if manifest.get("tool") != expected_tool:
        raise ValueError("review pack was generated by different renderer bytes")
    paths = set()
    for case in cases:
        record = case.get("panel")
        if not isinstance(record, dict):
            raise ValueError("review pack case omits panel record")
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() in paths:
            raise ValueError("unsafe or duplicate review panel path")
        paths.add(relative.as_posix())
        path = (base / relative).resolve()
        path.relative_to(base)
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256_file(path) != record.get("sha256")
        ):
            raise ValueError(f"review panel identity mismatch: {path}")
    return manifest


def _parse_utc(value: object) -> dt.datetime:
    if not isinstance(value, str) or not value or value.endswith("Z") is False:
        raise ValueError("reviewed_utc must be a UTC ISO-8601 timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("reviewed_utc is not valid ISO-8601") from exc
    now = dt.datetime.now(dt.timezone.utc)
    if parsed > now + dt.timedelta(minutes=5):
        raise ValueError("reviewed_utc is in the future")
    return parsed


def validate_human_review(root: Path, receipt_path: Path) -> tuple[dict[str, Any], bytes]:
    pack = load_review_pack(root)
    payload = receipt_path.read_bytes()
    receipt = json.loads(payload.decode("utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError("unexpected human-review schema")
    if receipt.get("review_pack_content_sha256") != pack["content_sha256"]:
        raise ValueError("human review targets a different review pack")
    reviewer = receipt.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("human review requires a named reviewer")
    _parse_utc(receipt.get("reviewed_utc"))
    if receipt.get("acknowledgement") != REVIEW_ACKNOWLEDGEMENT:
        raise ValueError("human reviewer did not acknowledge the proxy claim boundary")
    recommendation = receipt.get("release_recommendation")
    if recommendation not in REVIEW_RECOMMENDATIONS:
        raise ValueError("invalid release recommendation")
    expected = [
        (case["panel_id"], case["case_id"], case["score_slice_l1"],
         case["panel"]["sha256"])
        for case in pack["cases"]
    ]
    reviews = receipt.get("cases")
    if not isinstance(reviews, list) or [
        (case.get("panel_id"), case.get("case_id"), case.get("score_slice_l1"),
         case.get("panel_sha256"))
        for case in reviews
    ] != expected:
        raise ValueError("human review must cover the exact fixed panel hashes in order")
    for case in reviews:
        if case.get("registration_alignment") not in ALIGNMENT_VALUES:
            raise ValueError("invalid registration-alignment review value")
        if case.get("initial_disagreement_image_support") not in INITIAL_SUPPORT_VALUES:
            raise ValueError("invalid initial-disagreement review value")
        if case.get("candidate_change_image_support") not in CHANGE_SUPPORT_VALUES:
            raise ValueError("invalid candidate-change review value")
        if not isinstance(case.get("notes"), str) or not case["notes"].strip():
            raise ValueError("every fixed panel requires nonempty review notes")
    supported = receipt.get("supported_panel_ids")
    if (
        not isinstance(supported, list)
        or len(supported) != len(set(supported))
        or not set(supported) <= {panel_id for panel_id, _, _, _ in expected}
    ):
        raise ValueError("supported_panel_ids must be a unique subset of fixed panels")
    by_id = {case["panel_id"]: case for case in reviews}
    if any(case["registration_alignment"] == "FAIL" for case in reviews):
        if recommendation != "DO_NOT_RELEASE":
            raise ValueError("a failed registration panel blocks release")
    if recommendation != "DO_NOT_RELEASE":
        for case_id in {case["case_id"] for case in reviews}:
            case_reviews = [case for case in reviews if case["case_id"] == case_id]
            if len(case_reviews) != len(REVIEW_SLICES_L1) or sum(
                case["registration_alignment"] == "PASS" for case in case_reviews
            ) < 2:
                raise ValueError(
                    f"release requires two image-alignment passes per fixed case: {case_id}"
                )
    if recommendation == "RELEASE_WITH_NAMED_IMAGE_SUPPORTED_CASES":
        if not supported:
            raise ValueError("named image-supported release requires at least one case")
        for panel_id in supported:
            case = by_id[panel_id]
            if (
                case["registration_alignment"] != "PASS"
                or case["initial_disagreement_image_support"] != "SUPPORTED"
                or case["candidate_change_image_support"] != "CORRECTION"
            ):
                raise ValueError(
                    f"named image-supported panel is not fully supported: {panel_id}"
                )
    elif supported:
        raise ValueError(
            "supported_panel_ids are allowed only for named image-supported release"
        )
    if recommendation == "RELEASE_WITH_AGREEMENT_ONLY" and any(
        case["registration_alignment"] == "FAIL" for case in reviews
    ):
        raise ValueError("agreement-only release cannot include failed alignment")
    receipt = copy.deepcopy(receipt)
    receipt["content_sha256"] = _content_hash(receipt)
    return receipt, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    render.add_argument("--data-root", type=Path, required=True)
    render.add_argument("--transform-root", type=Path, required=True)
    render.add_argument("--cache-root", type=Path, required=True)
    render.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--review-root", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render":
        result = build_review_pack(args)
    else:
        result, payload = validate_human_review(
            args.review_root.resolve(), args.receipt.resolve()
        )
        result["file_sha256"] = _sha256_bytes(payload)
    print(_canonical(result))


if __name__ == "__main__":
    main()
