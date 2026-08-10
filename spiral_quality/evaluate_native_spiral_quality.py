"""Evaluate frozen native Spiral baseline/final previews without visual claims.

This composes the exact pinned sheetcheck CT primitives and spiralcheck
intrinsic checks.  It uses identical TIFXYZ cells for both arms, cluster-
bootstraps paired changes by sampled neighbourhood, and records every ray
profile.  Winding pitch is deliberately not measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
import weakref

import numpy as np


FORMAT = "villa-native-spiral-quality-v1"
RUN_FORMAT = "villa-native-spiral-slab-run-v1"
MATERIALIZATION_FORMAT = "villa-spiral-normal-slab-materialization-v1"
EXPECTED_HEAD = "07bb743eb0382e4d94217f49128b126c4b0a9682"
EXPECTED_PLAN_SHA = "611b0d2a35fd4699cba09aa179d09525d44e633f8a25f12774b5e178d0371dd5"
EXPECTED_REFERENCE_SHA = "458e6ecfcdefef4a1cbd7baa859fd68907f5ea3b18de6f11589893019b5e663f"
EXPECTED_PREREG_SHA = "a8af1607cdb930d13e2c26b1b1e5aaaca7d46a0abea1394234ab6ada9253b504"
EXPECTED_QUALITY_PREREG_SHA = "137a3d78870f642c9599d29d1dd1421ae99c473fc88e5e43e39873ceff6643a9"
EXPECTED_ALIGNMENT_AMENDMENT_SHA = "6501ebd0c94f0c7ab6558bb19b02b34db8b1eac54a40c73212972a2fc94f540b"
EXPECTED_BLOCK_BATCH_AMENDMENT_SHA = "5bc4a12ec86e5e1c110986f0d96f874249d9635f36927e3f0db20db0ea2218a5"
EXPECTED_INTRINSIC_SCHEMA_AMENDMENT_SHA = "87e6f4a53efc9b69cb8bf302b6a3c9b5db7b504cd29de044e28304372a23ea4d"
EXPECTED_FINALIZATION_AMENDMENT_SHA = "9655e549315e1b5aca5b6a4c02e19d57dd76ffbb82a40719b7766104bc7ddd35"
SHEETCHECK_COMMIT = "7d53893abcc6cc7c0542e483c7266d75ea930885"
SPIRALCHECK_COMMIT = "d1b50e2957409a870225fb9f5dcc5e25f7a0f9da"
VOLUME = (
    "s3://vesuvius-challenge-open-data/PHerc0125/volumes/"
    "20250821151825-9.362um-1.2m-113keV-masked.zarr"
)
VOLUME_LEVEL = 1
VOXEL_SIZE_UM = 18.724
SAMPLE_SEED = 20260809
N_SITES = 20
RAYS_PER_SITE = 20
SITE_RADIUS = 6
REACH_UM = 700.0
STEP_VOX = 0.5
BOOTSTRAPS = 10_000
MIN_PAIRED_SITES = 8
MIN_PAIRED_RAYS = 50
MAX_BLOCK_VOXELS = 32_000_000

# Frozen engineering regression tolerances for deterministic intrinsic bins.
MAX_VIOLATION_FRACTION_INCREASE = 0.001
MAX_SPACING_ALERT_FRACTION_INCREASE = 0.005
MAX_MEAN_VALIDITY_DROP = 0.02
MAX_WINDING_VALIDITY_DROP = 0.05
MIN_PITCH_RATIO = 0.5
MAX_PITCH_RATIO = 2.0
FLOAT_TOLERANCE = 1e-12
ALIGNMENT_FORMAT = "normalized-theta-min-width-v1"
ALIGNMENT_CONTRACT = {
    "axis": "per-winding normalized angular (theta) column",
    "common_columns": "minimum native column count per winding",
    "coordinate": "j/(common_columns-1), inclusive endpoints",
    "interpolation": "linear between adjacent native 3-D vertices",
    "validity": "all bracketing native vertices valid and finite",
    "normals": "recomputed after alignment with pinned sheetcheck.Surface.normals",
    "intrinsic_geometry": "original unresampled artifacts",
}
BLOCK_BATCH_CONTRACT = {
    "order": "stored lexicographic cell order",
    "method": "greedy consecutive maximal batches",
    "arms_per_block": "baseline and final together",
    "margin_voxels": 6,
    "max_block_voxels": MAX_BLOCK_VOXELS,
    "cluster_unit_unchanged": True,
}
INTRINSIC_SCHEMA_CONTRACT = {
    "source": "pinned spiralcheck IntrinsicReport.to_dict()",
    "derived_field": "inflated_bin_fraction",
    "formula": "n_inflated / n_bins_checked",
    "provided_fraction_cross_checks": [
        "n_violations / n_bins_checked",
        "n_collapsed / n_bins_checked",
    ],
}


class GateError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def runner_canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise GateError(f"{label} root must be an object")
    return value


def contained(root: Path, candidate: Path, label: str) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise GateError(f"{label} escapes {root}: {candidate}") from error
    return candidate


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
        encoding="utf-8", check=False,
    )
    if result.returncode:
        raise GateError(f"git {' '.join(args)} failed for {repo}: {result.stderr.strip()}")
    return result.stdout.strip()


def assert_external_repo(repo: Path, expected: str, label: str) -> None:
    if git_output(repo, "rev-parse", "HEAD") != expected:
        raise GateError(f"{label} commit mismatch")
    if git_output(repo, "status", "--porcelain"):
        raise GateError(f"{label} worktree is dirty")
    license_path = repo / "LICENSE"
    if not license_path.is_file() or "MIT" not in license_path.read_text(encoding="utf-8", errors="replace"):
        raise GateError(f"{label} MIT license is missing")


def validate_materialization(dataset_root: Path) -> dict[str, Any]:
    path = dataset_root / "materialization.json"
    value = load_object(path, "materialization evidence")
    if (
        value.get("format") != MATERIALIZATION_FORMAT
        or value.get("scroll") != "PHerc0125"
        or value.get("plan_sha256") != EXPECTED_PLAN_SHA
        or value.get("reference_sha256") != EXPECTED_REFERENCE_SHA
        or bool(value.get("fit_executed"))
        or bool(value.get("physical_winding_sense_measured"))
    ):
        raise GateError("materialization identity or claim boundary mismatch")
    if contained(dataset_root, Path(value["dataset_root"]), "materialization dataset root") != dataset_root.resolve():
        raise GateError("materialization dataset root mismatch")
    return value


def validate_run_evidence(
    path: Path,
    *,
    dataset_root: Path,
    request_sha: str,
    required_step: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = load_object(path, "native run evidence")
    if (
        value.get("format") != RUN_FORMAT
        or value.get("complete") is not True
        or value.get("villa_head") != EXPECTED_HEAD
        or value.get("request_sha256") != request_sha
    ):
        raise GateError(f"native evidence identity mismatch: {path}")
    milestones = value.get("milestones")
    if not isinstance(milestones, list):
        raise GateError("native milestones are missing")
    matches = [item for item in milestones if isinstance(item, dict) and item.get("step") == required_step]
    if len(matches) != 1:
        raise GateError(f"expected exactly one step-{required_step} milestone")
    milestone = matches[0]
    preview = milestone.get("preview")
    if not isinstance(preview, dict):
        raise GateError(f"step-{required_step} preview record is missing")
    request = value.get("effective_request")
    if not isinstance(request, dict) or not isinstance(request.get("paths"), dict):
        raise GateError("effective native request is missing")
    output_root = contained(dataset_root, Path(request["paths"]["output_directory"]), "native output root")
    records = preview.get("files")
    if not isinstance(records, list) or not records:
        raise GateError("preview file manifest is empty")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise GateError("preview file record schema mismatch")
        relative = str(record["path"])
        if relative in seen:
            raise GateError(f"repeated preview file: {relative}")
        seen.add(relative)
        file = contained(output_root, output_root / relative, "preview file")
        if (
            not file.is_file()
            or file.stat().st_size != int(record["bytes"])
            or sha256_file(file) != record["sha256"]
        ):
            raise GateError(f"preview file verification failed: {file}")
        checked.append(record)
    if sha256_bytes(runner_canonical_json(checked)) != preview.get("tree_sha256"):
        raise GateError("preview tree SHA-256 mismatch")
    manifest = contained(output_root, Path(preview["manifest_path"]), "preview manifest")
    surface = contained(output_root, Path(preview["surface_path"]), "preview surface")
    if sha256_file(manifest) != preview.get("manifest_sha256") or not surface.is_dir():
        raise GateError("preview manifest or surface mismatch")
    manifest_value = load_object(manifest, "preview manifest")
    if Path(manifest_value.get("surface_path", "")).resolve() != surface:
        raise GateError("preview manifest surface path mismatch")
    return value, milestone


def split_surface(surface: Any) -> tuple[dict[int, Any], dict[int, np.ndarray]]:
    ranges = surface.meta.get("winding_column_ranges")
    ids = surface.meta.get("component_winding_ids")
    if not isinstance(ranges, list) or not isinstance(ids, list) or len(ranges) != len(ids) or len(ids) < 2:
        raise GateError("combined surface winding metadata is invalid")
    expected_start = 0
    family: dict[int, Any] = {}
    normals: dict[int, np.ndarray] = {}
    cls = type(surface)
    for raw_id, raw_range in zip(ids, ranges):
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise GateError("winding column range is invalid")
        start, end = map(int, raw_range)
        winding = int(raw_id)
        if start != expected_start or end <= start or end > surface.shape[1] or winding in family:
            raise GateError("winding ranges are not unique contiguous partitions")
        expected_start = end
        part = cls(
            points=surface.points[:, start:end],
            valid=surface.valid[:, start:end],
            meta={"scale": surface.meta.get("scale"), "winding_id": winding},
            name=f"w{winding:03d}",
        )
        nrm, ok = part.normals()
        family[winding] = part
        normals[winding] = np.where(ok[..., None], nrm, np.nan)
    if expected_start != surface.shape[1]:
        raise GateError("winding ranges do not cover the combined surface")
    return family, normals


def _declared_scale(surface: Any, label: str) -> tuple[float, float]:
    raw = surface.meta.get("scale")
    if not isinstance(raw, list) or len(raw) != 2:
        raise GateError(f"{label} declared scale is invalid")
    scale = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) and item > 0 for item in scale):
        raise GateError(f"{label} declared scale is invalid")
    return scale


def resample_normalized_columns(surface: Any, target_columns: int, *, label: str) -> Any:
    """Conservatively resample one winding on its normalized theta axis."""
    rows, source_columns = map(int, surface.shape)
    if rows < 2 or source_columns < 2 or target_columns < 2 or target_columns > source_columns:
        raise GateError(f"{label} normalized-theta shape is invalid")
    source_position = np.linspace(0.0, float(source_columns - 1), target_columns)
    left_index = np.floor(source_position).astype(np.int64)
    right_index = np.ceil(source_position).astype(np.int64)
    fraction = (source_position - left_index)[None, :, None]
    left = surface.points[:, left_index, :]
    right = surface.points[:, right_index, :]
    points = left * (1.0 - fraction) + right * fraction
    valid = (
        surface.valid[:, left_index]
        & surface.valid[:, right_index]
        & np.isfinite(left).all(axis=-1)
        & np.isfinite(right).all(axis=-1)
        & np.isfinite(points).all(axis=-1)
    )
    points = np.asarray(points, dtype=np.float32)
    points[~valid] = -1.0
    return type(surface)(
        points=points,
        valid=valid,
        meta={"scale": list(_declared_scale(surface, label)), "normalized_theta_columns": target_columns},
        name=f"{surface.name}-theta-{target_columns}",
    )


def align_paired_families(
    base_family: dict[int, Any], final_family: dict[int, Any],
) -> tuple[
    dict[int, Any], dict[int, np.ndarray], dict[int, Any], dict[int, np.ndarray], dict[str, Any]
]:
    """Align adaptive native winding widths on a common normalized theta grid."""
    if set(base_family) != set(final_family):
        raise GateError("baseline/final winding IDs differ")
    base_aligned: dict[int, Any] = {}
    final_aligned: dict[int, Any] = {}
    base_normals: dict[int, np.ndarray] = {}
    final_normals: dict[int, np.ndarray] = {}
    windings: list[dict[str, Any]] = []
    for winding in sorted(base_family):
        base = base_family[winding]
        final = final_family[winding]
        base_rows, base_columns = map(int, base.shape)
        final_rows, final_columns = map(int, final.shape)
        if base_rows != final_rows:
            raise GateError(f"baseline/final winding row count differs: {winding}")
        if _declared_scale(base, f"baseline winding {winding}") != _declared_scale(
            final, f"final winding {winding}"
        ):
            raise GateError(f"baseline/final winding scale differs: {winding}")
        common_columns = min(base_columns, final_columns)
        base_part = resample_normalized_columns(
            base, common_columns, label=f"baseline winding {winding}"
        )
        final_part = resample_normalized_columns(
            final, common_columns, label=f"final winding {winding}"
        )
        base_nrm, base_ok = base_part.normals()
        final_nrm, final_ok = final_part.normals()
        base_aligned[winding] = base_part
        final_aligned[winding] = final_part
        base_normals[winding] = np.where(base_ok[..., None], base_nrm, np.nan)
        final_normals[winding] = np.where(final_ok[..., None], final_nrm, np.nan)
        windings.append({
            "winding": winding,
            "baseline_native_shape": [base_rows, base_columns],
            "final_native_shape": [final_rows, final_columns],
            "common_shape": [base_rows, common_columns],
        })
    if len(windings) < 2:
        raise GateError("aligned winding family is incomplete")
    return base_aligned, base_normals, final_aligned, final_normals, {
        "format": ALIGNMENT_FORMAT,
        "contract": ALIGNMENT_CONTRACT,
        "windings": windings,
    }


def load_surface(surface_path: Path, surface_cls: Any) -> Any:
    value = surface_cls.load(str(surface_path))
    if value.points.dtype != np.float32 or value.points.ndim != 3 or value.points.shape[-1] != 3:
        raise GateError("TIFXYZ surface shape or dtype mismatch")
    if not value.valid.any():
        raise GateError("TIFXYZ surface has no valid vertex")
    return value


def choose_sites(
    base_family: dict[int, Any],
    base_normals: dict[int, np.ndarray],
    final_family: dict[int, Any],
    final_normals: dict[int, np.ndarray],
    *,
    seed: int = SAMPLE_SEED,
    n_sites: int = N_SITES,
    rays_per_site: int = RAYS_PER_SITE,
    radius: int = SITE_RADIUS,
) -> list[dict[str, Any]]:
    if set(base_family) != set(final_family):
        raise GateError("baseline/final winding IDs differ")
    candidates: list[tuple[int, np.ndarray, tuple[int, int], np.ndarray]] = []
    counts: list[int] = []
    for winding in sorted(base_family):
        if base_family[winding].shape != final_family[winding].shape:
            raise GateError(f"baseline/final winding shape differs: {winding}")
        common = (
            np.isfinite(base_normals[winding]).all(axis=-1)
            & np.isfinite(final_normals[winding]).all(axis=-1)
        )
        interior = common[radius : common.shape[0] - radius, radius : common.shape[1] - radius]
        flat = np.flatnonzero(interior)
        if flat.size:
            candidates.append((winding, flat, interior.shape, common))
            counts.append(int(flat.size))
    if not candidates or sum(counts) < n_sites:
        raise GateError("not enough common normal-valid cells for frozen sampling")
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    rng = np.random.default_rng(seed)
    accepted: list[dict[str, Any]] = []
    centres: set[tuple[int, int, int]] = set()
    max_attempts = max(50_000, n_sites * 10_000)
    for _ in range(max_attempts):
        global_index = int(rng.integers(0, total))
        group = int(np.searchsorted(cumulative, global_index, side="right"))
        previous = int(cumulative[group - 1]) if group else 0
        winding, flat, shape, common = candidates[group]
        local_flat = int(flat[global_index - previous])
        row0, col0 = np.unravel_index(local_flat, shape)
        row, col = int(row0 + radius), int(col0 + radius)
        key = (winding, row, col)
        if key in centres or any(
            old_w == winding and abs(old_r - row) <= 2 * radius and abs(old_c - col) <= 2 * radius
            for old_w, old_r, old_c in centres
        ):
            continue
        local = np.argwhere(common[row - radius : row + radius + 1, col - radius : col + radius + 1])
        if len(local) < 5:
            continue
        cells = local + np.array([row - radius, col - radius])
        site_rng = np.random.default_rng(np.random.SeedSequence([seed, winding, row, col]))
        if len(cells) > rays_per_site:
            cells = cells[site_rng.choice(len(cells), rays_per_site, replace=False)]
        cells = cells[np.lexsort((cells[:, 1], cells[:, 0]))]
        accepted.append({
            "site_id": len(accepted),
            "winding": winding,
            "centre": [row, col],
            "cells": cells.astype(int).tolist(),
        })
        centres.add(key)
        if len(accepted) == n_sites:
            return accepted
    raise GateError(f"could select only {len(accepted)}/{n_sites} separated sites")


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _ct_block_request(rays: np.ndarray, volume_shape: tuple[int, ...]) -> dict[str, Any]:
    flattened = np.asarray(rays, dtype=np.float64).reshape(-1, 3)
    raw_lo = flattened.min(axis=0) - BLOCK_BATCH_CONTRACT["margin_voxels"]
    raw_hi = flattened.max(axis=0) + BLOCK_BATCH_CONTRACT["margin_voxels"]
    shape_limit = np.asarray(volume_shape, dtype=np.int64)
    if shape_limit.shape != (3,) or np.any(shape_limit <= 0):
        raise GateError("CT volume shape is invalid")
    clipped_lo = np.maximum(np.floor(raw_lo).astype(np.int64), 0)
    clipped_hi = np.minimum(np.ceil(raw_hi).astype(np.int64) + 1, shape_limit)
    shape = clipped_hi - clipped_lo
    return {
        "raw_lo": raw_lo,
        "raw_hi": raw_hi,
        "shape": shape.astype(int).tolist(),
        "voxels": int(np.prod(shape, dtype=np.int64)) if np.all(shape > 0) else 0,
    }


def plan_ct_batches(
    arm_rays: dict[str, np.ndarray], volume_shape: tuple[int, ...], *, site_id: int,
    max_voxels: int = MAX_BLOCK_VOXELS,
) -> list[dict[str, Any]]:
    if set(arm_rays) != {"baseline", "final"}:
        raise GateError(f"site {site_id} CT arm inventory mismatch")
    counts = {int(value.shape[0]) for value in arm_rays.values()}
    if len(counts) != 1 or not counts or next(iter(counts)) < 1:
        raise GateError(f"site {site_id} CT ray inventory mismatch")
    count = next(iter(counts))
    plans: list[dict[str, Any]] = []
    start = 0
    while start < count:
        accepted: dict[str, Any] | None = None
        for end in range(start + 1, count + 1):
            joined = np.concatenate(
                [arm_rays[name][start:end] for name in ("baseline", "final")], axis=0
            )
            request = _ct_block_request(joined, volume_shape)
            if request["voxels"] > max_voxels:
                break
            accepted = {**request, "cell_start": start, "cell_end": end}
        if accepted is None:
            raise GateError(f"site {site_id} single paired ray exceeds CT block cap")
        plans.append(accepted)
        start = int(accepted["cell_end"])
    return plans


def evaluate_sites(
    sites: list[dict[str, Any]],
    base_family: dict[int, Any],
    base_normals: dict[int, np.ndarray],
    final_family: dict[int, Any],
    final_normals: dict[int, np.ndarray],
    volume: Any,
    *,
    support_scores: Callable[..., tuple[np.ndarray, np.ndarray]],
    find_sheets: Callable[..., np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    voxel_um = float(volume.voxel_size_um)
    if not math.isclose(voxel_um, VOXEL_SIZE_UM, rel_tol=0, abs_tol=1e-6):
        raise GateError(f"CT voxel size mismatch: {voxel_um}")
    offsets = np.arange(-REACH_UM / voxel_um, REACH_UM / voxel_um + 1e-9, STEP_VOX)
    centre = len(offsets) // 2
    rows: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    for site in sites:
        winding = int(site["winding"])
        cells = np.asarray(site["cells"], dtype=np.int64)
        rr, cc = cells[:, 0], cells[:, 1]
        arm_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, family, normals in (
            ("baseline", base_family, base_normals),
            ("final", final_family, final_normals),
        ):
            points = family[winding].points[rr, cc]
            nrm = normals[winding][rr, cc]
            if not np.isfinite(nrm).all():
                raise GateError("sampled a non-finite surface normal")
            level_points = volume.to_level(points)
            rays = level_points[:, None, :] + offsets[None, :, None] * nrm[:, None, :]
            arm_data[name] = (points, rays)
        arm_rays = {name: rays for name, (_, rays) in arm_data.items()}
        plans = plan_ct_batches(arm_rays, tuple(volume.shape), site_id=int(site["site_id"]))
        measured: dict[str, list[dict[str, Any] | None]] = {
            name: [None] * len(cells) for name in arm_rays
        }
        for batch_index, plan in enumerate(plans):
            block, block_lo = volume.read_box(plan["raw_lo"], plan["raw_hi"])
            if (
                list(block.shape) != plan["shape"]
                or block.size == 0
                or block.size > MAX_BLOCK_VOXELS
                or min(block.shape) < 12
            ):
                raise GateError(
                    f"site {site['site_id']} CT batch {batch_index} is unsafe/unusable: "
                    f"planned={plan['shape']}, actual={list(block.shape)}"
                )
            start, end = int(plan["cell_start"]), int(plan["cell_end"])
            for name, rays in arm_rays.items():
                profiles = volume.sample_box(block, block_lo, rays[start:end]).astype(
                    np.float32, copy=False
                )
                usable = np.count_nonzero(profiles, axis=1) > profiles.shape[1] * 0.6
                support, support_ok = support_scores(profiles, centre)
                for local_index, profile in enumerate(profiles):
                    sheets = (
                        find_sheets(profile, STEP_VOX, voxel_um, min_thickness_um=25.0)
                        if usable[local_index]
                        else np.empty(0, dtype=float)
                    )
                    offset_um = (
                        abs(float(sheets[np.argmin(np.abs(sheets))])) * voxel_um
                        if len(sheets)
                        else float("nan")
                    )
                    measured[name][start + local_index] = {
                        "profile_usable": bool(usable[local_index]),
                        "gap_structure": bool(usable[local_index] and support_ok[local_index]),
                        "support": finite_or_none(support[local_index])
                        if usable[local_index] and support_ok[local_index] else None,
                        "offset_um": finite_or_none(offset_um),
                        "profile_sha256": sha256_bytes(profile.tobytes()),
                        "profile": [float(item) for item in profile],
                    }
            batch_records.append({
                "site_id": int(site["site_id"]),
                "batch_index": batch_index,
                "cell_start": start,
                "cell_end": end,
                "shape": plan["shape"],
                "voxels": int(plan["voxels"]),
            })
        if any(item is None for values in measured.values() for item in values):
            raise GateError(f"site {site['site_id']} CT batch coverage is incomplete")
        for index, (row, col) in enumerate(cells):
            rows.append({
                "site_id": int(site["site_id"]),
                "winding": winding,
                "row": int(row),
                "column": int(col),
                "baseline": measured["baseline"][index],
                "final": measured["final"][index],
            })
    return rows, batch_records


def clustered_interval(
    values: dict[int, list[float]],
    *,
    seed: int,
    statistic: Callable[[np.ndarray], float] = np.median,
) -> dict[str, Any]:
    values = {site: items for site, items in values.items() if items}
    pair_count = sum(len(items) for items in values.values())
    if len(values) < MIN_PAIRED_SITES or pair_count < MIN_PAIRED_RAYS:
        return {
            "sufficient": False, "sites": len(values), "pairs": pair_count,
            "estimate": None, "ci_lo": None, "ci_hi": None,
        }
    site_ids = sorted(values)
    observed = np.concatenate([np.asarray(values[site], dtype=float) for site in site_ids])
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    for index in range(BOOTSTRAPS):
        chosen = rng.integers(0, len(site_ids), size=len(site_ids))
        sample = np.concatenate([
            np.asarray(values[site_ids[int(item)]], dtype=float) for item in chosen
        ])
        draws[index] = statistic(sample)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "sufficient": True,
        "sites": len(site_ids),
        "pairs": pair_count,
        "estimate": float(statistic(observed)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def paired_ct_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support: dict[int, list[float]] = {}
    offset: dict[int, list[float]] = {}
    gap: dict[int, list[float]] = {}
    for row in rows:
        site = int(row["site_id"])
        baseline, final = row["baseline"], row["final"]
        if baseline["support"] is not None and final["support"] is not None:
            support.setdefault(site, []).append(float(final["support"] - baseline["support"]))
        if baseline["offset_um"] is not None and final["offset_um"] is not None:
            offset.setdefault(site, []).append(float(baseline["offset_um"] - final["offset_um"]))
        if baseline["profile_usable"] and final["profile_usable"]:
            gap.setdefault(site, []).append(float(final["gap_structure"] - baseline["gap_structure"]))
    result = {
        "support_final_minus_baseline": clustered_interval(support, seed=SAMPLE_SEED + 1),
        "absolute_offset_reduction_um": clustered_interval(offset, seed=SAMPLE_SEED + 2),
        "gap_structure_final_minus_baseline": clustered_interval(
            gap, seed=SAMPLE_SEED + 3, statistic=np.mean
        ),
    }
    sufficient = all(item["sufficient"] for item in result.values())
    improvements = [
        item["ci_lo"] > 0 for item in result.values() if item["sufficient"]
    ]
    degradations = [
        item["ci_hi"] < 0 for item in result.values() if item["sufficient"]
    ]
    result["all_measurements_sufficient"] = sufficient
    result["any_favorable_interval"] = bool(sufficient and any(improvements))
    result["any_adverse_interval"] = bool(sufficient and any(degradations))
    return result


def normalize_intrinsic_report(value: dict[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateError(f"{label} intrinsic report is not an object")
    normalized = dict(value)
    n_bins = normalized.get("n_bins_checked")
    if type(n_bins) is not int or n_bins <= 0:
        raise GateError(f"{label} intrinsic n_bins_checked is not a positive integer")
    count_fields = ("n_violations", "n_collapsed", "n_inflated")
    for field in count_fields:
        count = normalized.get(field)
        if type(count) is not int or not 0 <= count <= n_bins:
            raise GateError(f"{label} intrinsic {field} is outside [0, n_bins_checked]")
    for count_field, fraction_field in (
        ("n_violations", "violated_bin_fraction"),
        ("n_collapsed", "collapsed_bin_fraction"),
    ):
        try:
            observed = float(normalized[fraction_field])
        except (KeyError, TypeError, ValueError) as exc:
            raise GateError(f"{label} intrinsic {fraction_field} is missing or invalid") from exc
        expected = normalized[count_field] / n_bins
        if not math.isfinite(observed) or not math.isclose(
            observed, expected, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise GateError(f"{label} intrinsic {fraction_field} disagrees with counts")
    derived = normalized["n_inflated"] / n_bins
    existing = normalized.get("inflated_bin_fraction")
    if existing is not None:
        try:
            existing_value = float(existing)
        except (TypeError, ValueError) as exc:
            raise GateError(f"{label} intrinsic inflated_bin_fraction is invalid") from exc
        if not math.isfinite(existing_value) or not math.isclose(
            existing_value, derived, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE
        ):
            raise GateError(f"{label} intrinsic inflated_bin_fraction disagrees with counts")
    if not math.isfinite(derived) or not 0.0 <= derived <= 1.0:
        raise GateError(f"{label} intrinsic derived inflated fraction is invalid")
    normalized["inflated_bin_fraction"] = float(derived)
    return normalized


def close_sheetcheck_resources(volume: Any, sheetcheck_io: Any) -> None:
    filesystems: list[Any] = []
    array = getattr(volume, "_arr", None)
    store = getattr(array, "store", None)
    volume_fs = getattr(store, "fs", None)
    if volume_fs is not None:
        filesystems.append(volume_fs)
    metadata_factory = getattr(sheetcheck_io, "_fs", None)
    if metadata_factory is None or not callable(metadata_factory):
        raise GateError("sheetcheck metadata filesystem factory is unavailable")
    metadata_fs = metadata_factory()
    if metadata_fs is not None:
        filesystems.append(metadata_fs)
    seen: set[int] = set()
    try:
        for filesystem in filesystems:
            if id(filesystem) in seen:
                continue
            seen.add(id(filesystem))
            creator = getattr(filesystem, "_s3creator", None)
            client = getattr(filesystem, "_s3", None)
            if creator is None or client is None:
                continue
            if bool(getattr(filesystem, "asynchronous", False)):
                from zarr.core.sync import sync as zarr_sync

                zarr_sync(creator.__aexit__(None, None, None), timeout=30.0)
            else:
                finalizers = []
                for finalizer in list(weakref.finalize._registry):
                    state = finalizer.peek()
                    if state is not None and state[0] is filesystem:
                        finalizers.append(finalizer)
                if len(finalizers) != 1:
                    raise GateError(
                        "sheetcheck synchronous S3 filesystem finalizer inventory mismatch"
                    )
                finalizers[0]()
    except GateError:
        raise
    except Exception as exc:
        raise GateError(f"failed to close sheetcheck S3 resources: {exc}") from exc
    finally:
        cache_clear = getattr(metadata_factory, "cache_clear", None)
        if callable(cache_clear):
            cache_clear()


def intrinsic_summary(base: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    if base["winding_ids"] != final["winding_ids"]:
        raise GateError("intrinsic winding IDs differ")
    base_valid = {int(k): float(v) for k, v in base["validity_per_winding"].items()}
    final_valid = {int(k): float(v) for k, v in final["validity_per_winding"].items()}
    if not base_valid or set(base_valid) != set(final_valid):
        raise GateError("intrinsic winding validity inventories differ or are empty")
    winding_ids = sorted(base_valid)
    base_pitch = float(base["median_pitch"])
    final_pitch = float(final["median_pitch"])
    if not (math.isfinite(base_pitch) and math.isfinite(final_pitch) and base_pitch > 0 and final_pitch > 0):
        raise GateError("intrinsic median pitch is non-positive or non-finite")
    pitch_ratio = final_pitch / base_pitch
    deltas = {
        "violated_bin_fraction": float(final["violated_bin_fraction"] - base["violated_bin_fraction"]),
        "collapsed_bin_fraction": float(final["collapsed_bin_fraction"] - base["collapsed_bin_fraction"]),
        "inflated_bin_fraction": float(final["inflated_bin_fraction"] - base["inflated_bin_fraction"]),
        "mean_validity": float(
            np.mean([final_valid[winding] for winding in winding_ids])
            - np.mean([base_valid[winding] for winding in winding_ids])
        ),
        "worst_winding_validity": float(min(final_valid[w] - base_valid[w] for w in base_valid)),
        "median_pitch_ratio": pitch_ratio,
    }
    alerts = {
        "radial_violation_regression": (
            deltas["violated_bin_fraction"] > MAX_VIOLATION_FRACTION_INCREASE + FLOAT_TOLERANCE
        ),
        "collapsed_spacing_regression": (
            deltas["collapsed_bin_fraction"] > MAX_SPACING_ALERT_FRACTION_INCREASE + FLOAT_TOLERANCE
        ),
        "inflated_spacing_regression": (
            deltas["inflated_bin_fraction"] > MAX_SPACING_ALERT_FRACTION_INCREASE + FLOAT_TOLERANCE
        ),
        "mean_validity_regression": (
            deltas["mean_validity"] < -MAX_MEAN_VALIDITY_DROP - FLOAT_TOLERANCE
        ),
        "single_winding_validity_regression": (
            deltas["worst_winding_validity"] < -MAX_WINDING_VALIDITY_DROP - FLOAT_TOLERANCE
        ),
        "pitch_scale_instability": not MIN_PITCH_RATIO <= pitch_ratio <= MAX_PITCH_RATIO,
    }
    return {"baseline": base, "final": final, "deltas": deltas, "alerts": alerts,
            "no_material_regression": not any(alerts.values())}


def verify_report(path: Path, dataset_root: Path) -> dict[str, Any]:
    value = load_object(path, "native quality report")
    if value.get("format") != FORMAT or value.get("complete") is not True or value.get("scroll") != "PHerc0125":
        raise GateError("native quality report identity mismatch")
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("preregistration_sha256") != EXPECTED_PREREG_SHA:
        raise GateError("native quality report preregistration mismatch")
    expected_inputs = {
        "quality_preregistration_sha256": EXPECTED_QUALITY_PREREG_SHA,
        "quality_alignment_amendment_sha256": EXPECTED_ALIGNMENT_AMENDMENT_SHA,
        "quality_block_batch_amendment_sha256": EXPECTED_BLOCK_BATCH_AMENDMENT_SHA,
        "quality_intrinsic_schema_amendment_sha256": EXPECTED_INTRINSIC_SCHEMA_AMENDMENT_SHA,
        "quality_finalization_amendment_sha256": EXPECTED_FINALIZATION_AMENDMENT_SHA,
        "materialization_sha256": sha256_file(dataset_root / "materialization.json"),
        "baseline_evidence_sha256": sha256_file(dataset_root / "native_smoke_step1_300.json"),
        "final_evidence_sha256": sha256_file(dataset_root / "native_production_step15000.json"),
        "umbilicus_sha256": EXPECTED_REFERENCE_SHA,
        "sheetcheck_commit": SHEETCHECK_COMMIT,
        "spiralcheck_commit": SPIRALCHECK_COMMIT,
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise GateError(f"native quality report input mismatch: {key}")
    contract = value.get("ct_contract")
    expected_contract = {
        "volume": VOLUME, "level": VOLUME_LEVEL, "voxel_size_um": VOXEL_SIZE_UM,
        "sites": N_SITES, "rays_per_site_max": RAYS_PER_SITE,
        "site_radius_cells": SITE_RADIUS, "reach_um": REACH_UM,
        "step_vox": STEP_VOX, "seed": SAMPLE_SEED, "bootstraps": BOOTSTRAPS,
        "cluster_unit": "sampled neighbourhood",
        "statistic": "paired median for support/offset; paired mean for gap structure",
        "pitch_estimator_used": False,
        "angular_alignment": ALIGNMENT_CONTRACT,
        "ct_block_batching": BLOCK_BATCH_CONTRACT,
    }
    if contract != expected_contract:
        raise GateError("native quality report CT contract mismatch")
    if value.get("intrinsic_schema_normalization") != INTRINSIC_SCHEMA_CONTRACT:
        raise GateError("native quality report intrinsic schema contract mismatch")
    sites = value.get("sites")
    samples = value.get("raw_samples")
    if not isinstance(sites, list) or len(sites) != N_SITES or not isinstance(samples, list):
        raise GateError("native quality report sample inventory mismatch")
    expected_samples = sum(len(site.get("cells", [])) for site in sites if isinstance(site, dict))
    if expected_samples != len(samples) or not MIN_PAIRED_RAYS <= len(samples) <= N_SITES * RAYS_PER_SITE:
        raise GateError("native quality report raw-sample count mismatch")
    alignment = value.get("angular_alignment")
    if (
        not isinstance(alignment, dict)
        or alignment.get("format") != ALIGNMENT_FORMAT
        or alignment.get("contract") != ALIGNMENT_CONTRACT
        or not isinstance(alignment.get("windings"), list)
    ):
        raise GateError("native quality report angular alignment mismatch")
    winding_records = alignment["windings"]
    if [item.get("winding") for item in winding_records if isinstance(item, dict)] != list(range(10, 131)):
        raise GateError("native quality report angular winding inventory mismatch")
    for item in winding_records:
        if set(item) != {"winding", "baseline_native_shape", "final_native_shape", "common_shape"}:
            raise GateError("native quality report angular shape schema mismatch")
        base_shape = item["baseline_native_shape"]
        final_shape = item["final_native_shape"]
        common_shape = item["common_shape"]
        if (
            not all(isinstance(shape, list) and len(shape) == 2 for shape in (base_shape, final_shape, common_shape))
            or base_shape[0] != final_shape[0]
            or common_shape != [base_shape[0], min(base_shape[1], final_shape[1])]
            or min(base_shape[0], base_shape[1], final_shape[1]) < 2
        ):
            raise GateError("native quality report angular shape mismatch")
    batches = value.get("ct_batches")
    if not isinstance(batches, list) or not batches:
        raise GateError("native quality report CT batch inventory mismatch")
    by_site: dict[int, list[dict[str, Any]]] = {}
    for item in batches:
        if not isinstance(item, dict) or set(item) != {
            "site_id", "batch_index", "cell_start", "cell_end", "shape", "voxels"
        }:
            raise GateError("native quality report CT batch schema mismatch")
        by_site.setdefault(int(item["site_id"]), []).append(item)
    if sorted(by_site) != list(range(N_SITES)):
        raise GateError("native quality report CT batch site coverage mismatch")
    for site in sites:
        site_id = int(site["site_id"])
        expected_start = 0
        records = by_site[site_id]
        if [int(item["batch_index"]) for item in records] != list(range(len(records))):
            raise GateError("native quality report CT batch index mismatch")
        for item in records:
            shape = item["shape"]
            if (
                not isinstance(shape, list) or len(shape) != 3
                or any(type(value) is not int or value < 12 for value in shape)
                or int(item["cell_start"]) != expected_start
                or int(item["cell_end"]) <= expected_start
                or int(item["voxels"]) != math.prod(shape)
                or int(item["voxels"]) > MAX_BLOCK_VOXELS
            ):
                raise GateError("native quality report CT batch bound mismatch")
            expected_start = int(item["cell_end"])
        if expected_start != len(site["cells"]):
            raise GateError("native quality report CT batch cell coverage mismatch")
    intrinsic = value.get("intrinsic")
    if not isinstance(intrinsic, dict):
        raise GateError("native quality report intrinsic result is missing")
    try:
        normalized_base = normalize_intrinsic_report(intrinsic["baseline"], label="baseline")
        normalized_final = normalize_intrinsic_report(intrinsic["final"], label="final")
    except KeyError as exc:
        raise GateError("native quality report intrinsic arms are missing") from exc
    if intrinsic != intrinsic_summary(normalized_base, normalized_final):
        raise GateError("native quality report intrinsic result mismatch")
    decisions = value.get("decisions")
    expected_decisions = {
        "quantitative_ct_improvement_authorized",
        "pherc0211_execution_authorized",
        "public_accuracy_wording_authorized",
        "letters_or_reading_claim_authorized",
        "physical_winding_sense_claim_authorized",
        "prize_claim_authorized",
    }
    if not isinstance(decisions, dict) or set(decisions) != expected_decisions or any(
        type(decisions[key]) is not bool for key in expected_decisions
    ):
        raise GateError("native quality report decision schema mismatch")
    if (
        decisions["letters_or_reading_claim_authorized"]
        or decisions["physical_winding_sense_claim_authorized"]
        or decisions["prize_claim_authorized"]
    ):
        raise GateError("native quality report over-authorizes a public claim")
    return value


def execute(args: argparse.Namespace) -> Path:
    root = args.root.resolve(strict=True)
    dataset_root = args.dataset_root.resolve(strict=True)
    output = args.output.resolve(strict=False)
    if output.exists():
        raise GateError(f"refusing to overwrite quality report: {output}")
    prereg = root / "SPIRAL_PRIZE_SCROLL_NATIVE_FIT_PREREG.md"
    if sha256_file(prereg) != EXPECTED_PREREG_SHA:
        raise GateError("native-fit preregistration SHA-256 mismatch")
    quality_prereg = root / "SPIRAL_PRIZE_SCROLL_QUALITY_EVALUATION_PREREG.md"
    if sha256_file(quality_prereg) != EXPECTED_QUALITY_PREREG_SHA:
        raise GateError("quality-evaluation preregistration SHA-256 mismatch")
    alignment_amendment = root / "PHERC0125_QUALITY_ALIGNMENT_AMENDMENT.md"
    if sha256_file(alignment_amendment) != EXPECTED_ALIGNMENT_AMENDMENT_SHA:
        raise GateError("quality-alignment amendment SHA-256 mismatch")
    block_batch_amendment = root / "PHERC0125_QUALITY_BLOCK_BATCH_AMENDMENT.md"
    if sha256_file(block_batch_amendment) != EXPECTED_BLOCK_BATCH_AMENDMENT_SHA:
        raise GateError("quality block-batch amendment SHA-256 mismatch")
    intrinsic_schema_amendment = root / "PHERC0125_QUALITY_INTRINSIC_SCHEMA_AMENDMENT.md"
    if sha256_file(intrinsic_schema_amendment) != EXPECTED_INTRINSIC_SCHEMA_AMENDMENT_SHA:
        raise GateError("quality intrinsic-schema amendment SHA-256 mismatch")
    finalization_amendment = root / "PHERC0125_QUALITY_FINALIZATION_AMENDMENT.md"
    if sha256_file(finalization_amendment) != EXPECTED_FINALIZATION_AMENDMENT_SHA:
        raise GateError("quality finalization amendment SHA-256 mismatch")
    sheet_repo = args.sheetcheck.resolve(strict=True)
    spiral_repo = args.spiralcheck.resolve(strict=True)
    assert_external_repo(sheet_repo, SHEETCHECK_COMMIT, "sheetcheck")
    assert_external_repo(spiral_repo, SPIRALCHECK_COMMIT, "spiralcheck")
    sys.path.insert(0, str(sheet_repo))
    sys.path.insert(0, str(spiral_repo / "src"))
    import sheetcheck.io as sheetcheck_io
    from sheetcheck.io import Surface, Volume
    from sheetcheck.profile import find_sheets
    from sheetcheck.support import support_scores
    from spiralcheck.intrinsic import intrinsic_report
    from spiralcheck.io_tifxyz import load_run_windings

    materialization = validate_materialization(dataset_root)
    _, base_milestone = validate_run_evidence(
        dataset_root / "native_smoke_step1_300.json",
        dataset_root=dataset_root,
        request_sha=str(materialization["smoke_request_sha256"]),
        required_step=1,
    )
    _, final_milestone = validate_run_evidence(
        dataset_root / "native_production_step15000.json",
        dataset_root=dataset_root,
        request_sha=str(materialization["production_request_sha256"]),
        required_step=15_000,
    )
    base_surface_path = Path(base_milestone["preview"]["surface_path"]).resolve()
    final_surface_path = Path(final_milestone["preview"]["surface_path"]).resolve()
    base_surface = load_surface(base_surface_path, Surface)
    final_surface = load_surface(final_surface_path, Surface)
    base_native, _ = split_surface(base_surface)
    final_native, _ = split_surface(final_surface)
    base_family, base_normals, final_family, final_normals, alignment = align_paired_families(
        base_native, final_native
    )
    sites = choose_sites(base_family, base_normals, final_family, final_normals)
    volume = Volume(VOLUME, level=VOLUME_LEVEL)
    try:
        rows, ct_batches = evaluate_sites(
            sites, base_family, base_normals, final_family, final_normals, volume,
            support_scores=support_scores, find_sheets=find_sheets,
        )
    finally:
        close_sheetcheck_resources(volume, sheetcheck_io)
    ct = paired_ct_summary(rows)
    umbilicus = dataset_root / "umbilicus.json"
    if sha256_file(umbilicus) != EXPECTED_REFERENCE_SHA:
        raise GateError("materialized umbilicus SHA-256 mismatch")
    umbilicus_value = json.loads(umbilicus.read_text(encoding="utf-8-sig"))
    base_intrinsic = intrinsic_report(
        load_run_windings(base_surface_path.parent), umbilicus=umbilicus_value
    ).to_dict()
    final_intrinsic = intrinsic_report(
        load_run_windings(final_surface_path.parent), umbilicus=umbilicus_value
    ).to_dict()
    base_intrinsic = normalize_intrinsic_report(base_intrinsic, label="baseline")
    final_intrinsic = normalize_intrinsic_report(final_intrinsic, label="final")
    intrinsic = intrinsic_summary(base_intrinsic, final_intrinsic)
    no_ct_degradation = bool(
        ct["all_measurements_sufficient"] and not ct["any_adverse_interval"]
    )
    quantitative_improvement = bool(
        no_ct_degradation and ct["any_favorable_interval"] and intrinsic["no_material_regression"]
    )
    ph0211_authorized = bool(
        no_ct_degradation and intrinsic["no_material_regression"]
    )
    report = {
        "format": FORMAT,
        "complete": True,
        "scroll": "PHerc0125",
        "claim_scope": (
            "Paired CT and intrinsic quality evidence only; no letters, reading, physical "
            "winding direction, generalization, prize, or maintainer-adoption claim."
        ),
        "inputs": {
            "preregistration_sha256": EXPECTED_PREREG_SHA,
            "quality_preregistration_sha256": EXPECTED_QUALITY_PREREG_SHA,
            "quality_alignment_amendment_sha256": EXPECTED_ALIGNMENT_AMENDMENT_SHA,
            "quality_block_batch_amendment_sha256": EXPECTED_BLOCK_BATCH_AMENDMENT_SHA,
            "quality_intrinsic_schema_amendment_sha256": EXPECTED_INTRINSIC_SCHEMA_AMENDMENT_SHA,
            "quality_finalization_amendment_sha256": EXPECTED_FINALIZATION_AMENDMENT_SHA,
            "materialization_sha256": sha256_file(dataset_root / "materialization.json"),
            "baseline_evidence_sha256": sha256_file(dataset_root / "native_smoke_step1_300.json"),
            "final_evidence_sha256": sha256_file(dataset_root / "native_production_step15000.json"),
            "baseline_preview_manifest_sha256": base_milestone["preview"]["manifest_sha256"],
            "final_preview_manifest_sha256": final_milestone["preview"]["manifest_sha256"],
            "umbilicus_sha256": EXPECTED_REFERENCE_SHA,
            "sheetcheck_commit": SHEETCHECK_COMMIT,
            "spiralcheck_commit": SPIRALCHECK_COMMIT,
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "ct_contract": {
            "volume": VOLUME,
            "level": VOLUME_LEVEL,
            "voxel_size_um": VOXEL_SIZE_UM,
            "sites": N_SITES,
            "rays_per_site_max": RAYS_PER_SITE,
            "site_radius_cells": SITE_RADIUS,
            "reach_um": REACH_UM,
            "step_vox": STEP_VOX,
            "seed": SAMPLE_SEED,
            "bootstraps": BOOTSTRAPS,
            "cluster_unit": "sampled neighbourhood",
            "statistic": "paired median for support/offset; paired mean for gap structure",
            "pitch_estimator_used": False,
            "angular_alignment": ALIGNMENT_CONTRACT,
            "ct_block_batching": BLOCK_BATCH_CONTRACT,
        },
        "intrinsic_schema_normalization": INTRINSIC_SCHEMA_CONTRACT,
        "angular_alignment": alignment,
        "ct_batches": ct_batches,
        "sites": sites,
        "raw_samples": rows,
        "ct": ct,
        "intrinsic_thresholds": {
            "max_violation_fraction_increase": MAX_VIOLATION_FRACTION_INCREASE,
            "max_spacing_alert_fraction_increase": MAX_SPACING_ALERT_FRACTION_INCREASE,
            "max_mean_validity_drop": MAX_MEAN_VALIDITY_DROP,
            "max_single_winding_validity_drop": MAX_WINDING_VALIDITY_DROP,
            "median_pitch_ratio_range": [MIN_PITCH_RATIO, MAX_PITCH_RATIO],
        },
        "intrinsic": intrinsic,
        "decisions": {
            "quantitative_ct_improvement_authorized": quantitative_improvement,
            "pherc0211_execution_authorized": ph0211_authorized,
            "public_accuracy_wording_authorized": quantitative_improvement,
            "letters_or_reading_claim_authorized": False,
            "physical_winding_sense_claim_authorized": False,
            "prize_claim_authorized": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json(report))
    temporary.replace(output)
    verify_report(output, dataset_root)
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    result.add_argument("--root", type=Path, default=root)
    result.add_argument(
        "--dataset-root", type=Path,
        default=root / "automatic_umbilicus/phase_c/PHerc0125_native_fit",
    )
    result.add_argument(
        "--sheetcheck", type=Path,
        default=root / "_external_overlap_review_20260809/sheetcheck",
    )
    result.add_argument(
        "--spiralcheck", type=Path,
        default=root / "_external_overlap_review_20260809/spiralcheck",
    )
    result.add_argument(
        "--output", type=Path,
        default=root / "automatic_umbilicus/phase_c/PHerc0125_native_fit_quality.json",
    )
    result.add_argument("--verify-existing", action="store_true")
    return result


def main() -> int:
    try:
        args = parser().parse_args()
        if args.verify_existing:
            dataset_root = args.dataset_root.resolve(strict=True)
            output = args.output.resolve(strict=True)
            verify_report(output, dataset_root)
        else:
            output = execute(args)
    except (GateError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"evaluate_native_spiral_quality.py: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"complete": True, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
