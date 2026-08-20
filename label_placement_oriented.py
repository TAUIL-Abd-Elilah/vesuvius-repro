"""Correct the label-placement signed statistic with a fixed physical orientation.

This is the outcome-blind implementation of the frozen plan in
LABEL_PLACEMENT_ORIENTATION_CORRECTION_PREREG.md and Amendments 01--03.
It first replays the historical seed-0 sample stream, then evaluates all independently
mapped Scroll1A cubes.  Positive corrected offset means the CT ridge is inward of the
label-run centre, toward the pinned Scroll1A axis.

Example:

  python label_placement_oriented.py --images D:/data/images --labels D:/data/labels
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates

from ridge_residual import SIGMA, across_sheet_normals


ROOT = Path(__file__).resolve().parent
DESIGN_ID = "label-placement-orientation-correction-v1"
SEED = 0
PER_SAMPLE = 600
MIN_VALID = 150
STEP = 0.25
MAX_HALF = 8.0
CORRIDORS = (3.0, 4.0, 6.0, 8.0)
ALIGNMENT_THRESHOLDS = (0.0, 0.25, 0.50)
BOOTSTRAPS = 10_000
BLOCK_SIZE = 1280

AXIS_URL = (
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
    "umbilici/umbilicus-scroll1a_zyx.txt"
)
AXIS_SHA256 = "84785853ad918e98bf241656b2dea80bae6b77303a13d57805f9d7854d391cc9"
META_URL = (
    "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
    "volumes/20230205180739/meta.json"
)
META_SHA256 = "d34e437ca3404aa5f7faaaaa731927ce7adfadf84376dfc3a587c400a40d2520"
VOLUME_UUID = "20230205180739"
VOLUME_SHAPE = (14376, 7888, 8096)  # z, y, x
VOXEL_SIZE_UM = 7.91

OVERLAP_NORMALIZED_SHA256 = "cdcce85096236cad8e3dc87a6b498fa50df01ce4850bf4987d2e2785538d60b6"
OLD_RESULT_NORMALIZED_SHA256 = "d10522c35619c900d740e4f4948dee96ce3c16dc484105aa81f240def29dad73"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized_text_sha256(data: bytes) -> str:
    return sha256(data.replace(b"\r\n", b"\n"))


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def read_pinned(path: Path, expected_normalized_sha: str) -> tuple[bytes, dict[str, Any]]:
    data = path.read_bytes()
    got = normalized_text_sha256(data)
    if got != expected_normalized_sha:
        raise RuntimeError(f"unexpected content hash for {path}: {got}")
    return data, {"path": portable_path(path), "sha256": sha256(data), "normalized_sha256": got}


def fetch_pinned(url: str, expected_sha: str, local: Path | None = None) -> bytes:
    data = local.read_bytes() if local is not None else urllib.request.urlopen(url, timeout=60).read()
    got = sha256(data)
    if got != expected_sha:
        raise RuntimeError(f"unexpected content hash for {url}: {got}")
    return data


def load_axis(axis_data: bytes, meta_data: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    meta = json.loads(meta_data)
    observed_shape = (int(meta["slices"]), int(meta["height"]), int(meta["width"]))
    if (
        str(meta["uuid"]) != VOLUME_UUID
        or observed_shape != VOLUME_SHAPE
        or not np.isclose(float(meta["voxelsize"]), VOXEL_SIZE_UM)
    ):
        raise RuntimeError(f"unexpected Scroll1A volume metadata: {meta}")

    rows = []
    for line in axis_data.decode("utf-8").splitlines():
        if line.strip():
            rows.append([float(v.strip()) for v in line.split(",")])
    axis = np.asarray(rows, dtype=np.float64)
    if axis.shape != (241, 3):
        raise RuntimeError(f"expected 241 z,y,x axis rows, got {axis.shape}")
    if len(np.unique(axis[:, 0])) != len(axis):
        raise RuntimeError("axis z coordinates are not unique")
    raw_negative_z_steps = int(np.sum(np.diff(axis[:, 0]) < 0))
    axis = axis[np.argsort(axis[:, 0])]
    if not np.all(np.diff(axis[:, 0]) > 0):
        raise RuntimeError("axis is not strictly increasing after z sort")
    return axis, {
        "rows": int(len(axis)),
        "raw_negative_z_steps": raw_negative_z_steps,
        "handling": "sort unique controls by z; linearly interpolate y and x at global z",
        "z_support": [float(axis[0, 0]), float(axis[-1, 0])],
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def deterministic_seed(key: str) -> int:
    digest = hashlib.sha256(f"{DESIGN_ID}|seed={SEED}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def points_hash(sample: str, points: np.ndarray) -> str:
    pts = np.asarray(points, dtype="<i4")
    return sha256(sample.encode() + b"\0" + pts.tobytes(order="C"))


def select_mapped_points(sample: str, lab: np.ndarray) -> tuple[np.ndarray, int]:
    candidates = np.argwhere(lab == 1)
    if len(candidates) < max(800, PER_SAMPLE):
        raise RuntimeError(f"{sample}: only {len(candidates)} labelled voxels")
    seed = deterministic_seed(f"mapped-points|{sample}")
    rng = np.random.default_rng(seed)
    selected = candidates[rng.choice(len(candidates), PER_SAMPLE, replace=False)]
    return selected.astype(np.float32), seed


def build_replay_plan(labels: Path, old_result: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Consume the historical RNG stream exactly and recover its 30 selected point sets."""
    expected = [row["sample"] for row in old_result["rows"]]
    if len(expected) != 30 or int(old_result["per_volume_points"]) != PER_SAMPLE:
        raise RuntimeError("historical result does not contain the expected 30x600 design")

    rng = np.random.default_rng(SEED)
    names = sorted(p.stem for p in labels.glob("sample_*.tif"))
    order = [names[i] for i in rng.choice(len(names), min(len(names), 120), replace=False)]
    plan: dict[str, np.ndarray] = {}
    failed_after_selection: list[str] = []
    pos = 0
    for sample in order:
        lab = np.asarray(tifffile.imread(str(labels / f"{sample}.tif")))
        candidates = np.argwhere(lab == 1)
        if len(candidates) < 800:
            continue
        selected = candidates[rng.choice(len(candidates), PER_SAMPLE, replace=False)].astype(np.float32)
        if pos < len(expected) and sample == expected[pos]:
            plan[sample] = selected
            pos += 1
            if pos == len(expected):
                break
        else:
            # The old script consumed this draw but rejected the sample after profile analysis.
            failed_after_selection.append(sample)
    if pos != len(expected):
        raise RuntimeError(f"historical replay diverged at row {pos}: expected {expected[pos]}")
    return plan, failed_after_selection


def sample_profiles(
    ct: np.ndarray, lab: np.ndarray, points: np.ndarray, normals: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = np.arange(-MAX_HALF, MAX_HALF + 1e-9, STEP, dtype=np.float32)
    coords = points[:, :, None].astype(np.float32) + normals[:, :, None] * ts[None, None, :]
    flat = coords.transpose(1, 0, 2).reshape(3, -1)
    smoothed = gaussian_filter(ct.astype(np.float32), SIGMA)
    ct_profile = map_coordinates(smoothed, flat, order=1, mode="nearest").reshape(len(points), -1)
    label_profile = map_coordinates(
        lab.astype(np.float32), flat, order=0, mode="nearest"
    ).reshape(len(points), -1)
    return ts, ct_profile, label_profile


def ridge_from_profile(profile: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.argmax(profile, axis=1)
    offset = ts[k].astype(np.float64)
    edge = (k == 0) | (k == len(ts) - 1)
    ok = ~edge
    if ok.any():
        col = k[ok]
        row = np.arange(len(profile))[ok]
        y0, y1, y2 = profile[row, col - 1], profile[row, col], profile[row, col + 1]
        den = y0 - 2 * y1 + y2
        shift = np.where(
            np.abs(den) > 1e-9,
            0.5 * (y0 - y2) / np.where(den == 0, 1, den),
            0.0,
        )
        offset[ok] = ts[col] + np.clip(shift, -1, 1) * STEP
    offset[edge] = np.nan
    return offset, edge


def label_run_centres(profile: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mid_candidates = np.flatnonzero(np.isclose(ts, 0.0))
    if len(mid_candidates) != 1:
        raise RuntimeError("profile does not have one zero coordinate")
    mid = int(mid_candidates[0])
    centres = np.full(len(profile), np.nan, dtype=np.float64)
    truncated = np.zeros(len(profile), dtype=bool)
    for i, row in enumerate(profile):
        if row[mid] != 1:
            truncated[i] = True
            continue
        lo = mid
        while lo > 0 and row[lo - 1] == 1:
            lo -= 1
        hi = mid
        while hi < len(row) - 1 and row[hi + 1] == 1:
            hi += 1
        if lo == 0 or hi == len(row) - 1:
            truncated[i] = True
            continue
        centres[i] = 0.5 * (float(ts[lo]) + float(ts[hi]))
    return centres, truncated


def radial_reference(
    axis: np.ndarray, points: np.ndarray, cube_lo: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    global_points = points.astype(np.float64) + cube_lo[None, :]
    if np.any(global_points < 0) or np.any(global_points >= np.asarray(VOLUME_SHAPE)[None, :]):
        raise RuntimeError("mapped point lies outside the pinned Scroll1A volume")
    z = global_points[:, 0]
    if z.min() < axis[0, 0] or z.max() > axis[-1, 0]:
        raise RuntimeError("mapped point lies outside the pinned axis z support")
    axis_y = np.interp(z, axis[:, 0], axis[:, 1])
    axis_x = np.interp(z, axis[:, 0], axis[:, 2])
    reference = np.column_stack(
        [np.zeros(len(points)), axis_y - global_points[:, 1], axis_x - global_points[:, 2]]
    )
    length = np.linalg.norm(reference, axis=1)
    if np.any(length <= 1e-9):
        raise RuntimeError("point coincides with the axis; inward direction is undefined")
    return reference, reference / length[:, None], global_points


def orient_offsets(
    normals: np.ndarray, raw_offsets: np.ndarray, unit_reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dot = np.sum(normals * unit_reference, axis=1)
    sign = np.where(dot < 0, -1.0, 1.0)
    oriented_normals = normals * sign[:, None]
    corrected = raw_offsets * sign
    alignment = np.abs(dot)
    good = np.isfinite(raw_offsets)
    if good.any():
        raw_vector = normals[good] * raw_offsets[good, None]
        corrected_vector = oriented_normals[good] * corrected[good, None]
        if not np.allclose(raw_vector, corrected_vector, atol=2e-6, rtol=2e-6):
            raise AssertionError("orientation changed the correction vector or landing point")
        if not np.allclose(np.abs(raw_offsets[good]), np.abs(corrected[good]), atol=1e-12):
            raise AssertionError("orientation changed absolute offsets")
    return corrected, oriented_normals, alignment, sign


def analyse_profiles(
    sample: str,
    points: np.ndarray,
    normals: np.ndarray,
    ts_all: np.ndarray,
    ct_profile_all: np.ndarray,
    label_profile_all: np.ndarray,
    axis: np.ndarray | None,
    cube_lo: np.ndarray | None,
) -> dict[str, Any]:
    if axis is not None and cube_lo is not None:
        reference, unit_reference, global_points = radial_reference(axis, points, cube_lo)
        alignment_all = np.abs(np.sum(normals * unit_reference, axis=1))
        block = np.floor((cube_lo + 160.0) / BLOCK_SIZE).astype(int)
        block_id = ",".join(str(int(v)) for v in block)
        radial_distance = np.linalg.norm(reference, axis=1)
    else:
        unit_reference = None
        global_points = None
        alignment_all = None
        block_id = None
        radial_distance = None

    cells = []
    for half in CORRIDORS:
        take = np.abs(ts_all) <= half + 1e-7
        ts = ts_all[take]
        ct_profile = ct_profile_all[:, take]
        label_profile = label_profile_all[:, take]
        ridge, ridge_edge = ridge_from_profile(ct_profile, ts)
        run_centre, run_truncated = label_run_centres(label_profile, ts)
        raw = ridge - run_centre
        valid = np.isfinite(raw)

        if unit_reference is not None:
            corrected, _, alignment, _ = orient_offsets(normals, raw, unit_reference)
        else:
            corrected = np.full_like(raw, np.nan)
            alignment = np.full(len(raw), np.nan)

        for threshold in ALIGNMENT_THRESHOLDS:
            if unit_reference is None:
                retain = valid if threshold == 0 else np.zeros(len(valid), dtype=bool)
            else:
                retain = valid & (alignment >= threshold)
            values_raw = raw[retain]
            values_corrected = corrected[retain]
            eligible = int(retain.sum()) >= MIN_VALID
            cells.append(
                {
                    "corridor_half_vox": half,
                    "alignment_min_abs_cosine": threshold,
                    "n_selected": int(len(points)),
                    "n_ridge_edge": int(ridge_edge.sum()),
                    "ridge_edge_rate": float(ridge_edge.mean()),
                    "n_label_run_truncated": int(run_truncated.sum()),
                    "label_run_truncation_rate": float(run_truncated.mean()),
                    "n_valid_before_alignment": int(valid.sum()),
                    "n_alignment_excluded": int((valid & ~retain).sum()),
                    "n_retained": int(retain.sum()),
                    "eligible_min_150": eligible,
                    "median_unoriented_offset": (
                        float(np.median(values_raw)) if len(values_raw) else None
                    ),
                    "median_oriented_offset": (
                        float(np.median(values_corrected))
                        if unit_reference is not None and len(values_corrected)
                        else None
                    ),
                    "median_abs_offset": (
                        float(np.median(np.abs(values_raw))) if len(values_raw) else None
                    ),
                }
            )

    result: dict[str, Any] = {
        "sample": sample,
        "point_hash": points_hash(sample, points),
        "n_selected": int(len(points)),
        "mapped": cube_lo is not None,
        "block_1280": block_id,
        "cells": cells,
    }
    if alignment_all is not None and global_points is not None and radial_distance is not None:
        result["alignment_abs_cosine"] = {
            "q10": float(np.quantile(alignment_all, 0.10)),
            "median": float(np.median(alignment_all)),
            "q90": float(np.quantile(alignment_all, 0.90)),
        }
        result["radial_distance_vox"] = {
            "min": float(radial_distance.min()),
            "median": float(np.median(radial_distance)),
            "max": float(radial_distance.max()),
        }
        result["global_point_bounds_zyx"] = {
            "min": global_points.min(axis=0).tolist(),
            "max": global_points.max(axis=0).tolist(),
        }
    return result


def keyed_rng(key: str) -> np.random.Generator:
    return np.random.default_rng(deterministic_seed(f"bootstrap|{key}"))


def sample_bootstrap_ci(values: np.ndarray, key: str) -> list[float]:
    rng = keyed_rng(f"sample|{key}")
    n = len(values)
    draws = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        draws[i] = np.median(values[rng.integers(0, n, size=n)])
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def block_bootstrap_ci(values: np.ndarray, blocks: list[str], key: str) -> list[float]:
    by_block: dict[str, list[float]] = defaultdict(list)
    for value, block in zip(values, blocks, strict=True):
        by_block[block].append(float(value))
    labels = sorted(by_block)
    grouped = [np.asarray(by_block[label]) for label in labels]
    rng = keyed_rng(f"block|{key}")
    draws = np.empty(BOOTSTRAPS, dtype=np.float64)
    for i in range(BOOTSTRAPS):
        picked = rng.integers(0, len(labels), size=len(labels))
        draws[i] = np.median(np.concatenate([grouped[j] for j in picked]))
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def find_cell(record: dict[str, Any], half: float, threshold: float) -> dict[str, Any]:
    for cell in record["cells"]:
        if (
            float(cell["corridor_half_vox"]) == half
            and float(cell["alignment_min_abs_cosine"]) == threshold
        ):
            return cell
    raise KeyError((half, threshold))


def summarize(records: list[dict[str, Any]], cohort: str) -> list[dict[str, Any]]:
    summaries = []
    for half in CORRIDORS:
        for threshold in ALIGNMENT_THRESHOLDS:
            all_cells = [(record, find_cell(record, half, threshold)) for record in records]
            kept = [(record, cell) for record, cell in all_cells if cell["eligible_min_150"]]
            values = np.asarray([cell["median_oriented_offset"] for _, cell in kept], dtype=float)
            abs_values = np.asarray([cell["median_abs_offset"] for _, cell in kept], dtype=float)
            if not len(values):
                summaries.append(
                    {
                        "corridor_half_vox": half,
                        "alignment_min_abs_cosine": threshold,
                        "n_samples_total": len(records),
                        "n_samples_eligible": 0,
                        "status": "no cell has 150 retained runs",
                    }
                )
                continue
            key = f"{cohort}|half={half}|alignment={threshold}"
            blocks = [str(record["block_1280"]) for record, _ in kept]
            summaries.append(
                {
                    "corridor_half_vox": half,
                    "alignment_min_abs_cosine": threshold,
                    "n_samples_total": len(records),
                    "n_samples_eligible": len(kept),
                    "n_samples_excluded_below_150": len(records) - len(kept),
                    "retained_points_total": int(sum(cell["n_retained"] for _, cell in kept)),
                    "retained_points_per_sample_min": int(min(cell["n_retained"] for _, cell in kept)),
                    "retained_points_per_sample_median": float(
                        np.median([cell["n_retained"] for _, cell in kept])
                    ),
                    "median_of_sample_oriented_medians": float(np.median(values)),
                    "q10_sample_oriented_medians": float(np.quantile(values, 0.10)),
                    "q90_sample_oriented_medians": float(np.quantile(values, 0.90)),
                    "median_of_sample_median_abs_offsets": float(np.median(abs_values)),
                    "sample_bootstrap_95pct_ci": sample_bootstrap_ci(values, key),
                    "spatial_block_bootstrap_95pct_ci": block_bootstrap_ci(values, blocks, key),
                    "n_spatial_blocks": len(set(blocks)),
                    "ridge_edge_rate_all_selected": float(
                        np.mean([cell["ridge_edge_rate"] for _, cell in all_cells])
                    ),
                    "label_run_truncation_rate_all_selected": float(
                        np.mean([cell["label_run_truncation_rate"] for _, cell in all_cells])
                    ),
                }
            )
    return summaries


def replay_validation(
    records: list[dict[str, Any]], old_result: dict[str, Any]
) -> dict[str, Any]:
    by_name = {record["sample"]: record for record in records}
    checks = []
    for old in old_result["rows"]:
        record = by_name[old["sample"]]
        cell = find_cell(record, 4.0, 0.0)
        observed_signed = round(float(cell["median_unoriented_offset"]), 4)
        observed_abs = round(float(cell["median_abs_offset"]), 4)
        passed = (
            int(cell["n_valid_before_alignment"]) == int(old["n_runs"])
            and observed_signed == float(old["median_signed"])
            and observed_abs == float(old["median_abs"])
        )
        checks.append(
            {
                "sample": old["sample"],
                "passed": passed,
                "expected": {
                    "n_runs": old["n_runs"],
                    "median_unoriented": old["median_signed"],
                    "median_abs": old["median_abs"],
                },
                "observed": {
                    "n_runs": cell["n_valid_before_alignment"],
                    "median_unoriented": observed_signed,
                    "median_abs": observed_abs,
                },
            }
        )
    if not all(check["passed"] for check in checks):
        bad = [check["sample"] for check in checks if not check["passed"]]
        raise RuntimeError(f"historical replay failed for: {', '.join(bad)}")
    return {"passed": True, "n_rows": len(checks), "checks": checks}


def load_checkpoint(path: Path, identity: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("identity") != identity:
            raise RuntimeError(f"stale checkpoint identity at {path}:{number}")
        key = (row["cohort"], row["result"]["sample"])
        if key in records:
            raise RuntimeError(f"duplicate checkpoint record: {key}")
        records[key] = row
    return records


def append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--overlap", type=Path, default=ROOT / "results" / "overlap" / "overlap_report.json"
    )
    parser.add_argument("--old-result", type=Path, default=ROOT / "results" / "label_placement.json")
    parser.add_argument("--axis-file", type=Path)
    parser.add_argument("--meta-file", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "label_placement_oriented.json")
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs" / "label_placement_oriented.jsonl"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.images.is_dir() or not args.labels.is_dir():
        raise SystemExit("--images and --labels must be existing directories")

    overlap_bytes, overlap_hashes = read_pinned(args.overlap, OVERLAP_NORMALIZED_SHA256)
    old_bytes, old_hashes = read_pinned(args.old_result, OLD_RESULT_NORMALIZED_SHA256)
    overlap = json.loads(overlap_bytes)
    old_result = json.loads(old_bytes)
    located_rows = overlap["located"]
    if len(located_rows) != 189 or any(row["vol_key"] != "s1a" for row in located_rows):
        raise RuntimeError("expected exactly 189 independently mapped Scroll1A cubes")
    located = {row["sample"]: row for row in located_rows}

    axis_data = fetch_pinned(AXIS_URL, AXIS_SHA256, args.axis_file)
    meta_data = fetch_pinned(META_URL, META_SHA256, args.meta_file)
    axis, axis_info = load_axis(axis_data, meta_data)

    source_commit = git_commit()
    identity = {
        "design_id": DESIGN_ID,
        "source_commit": source_commit,
        "overlap_normalized_sha256": overlap_hashes["normalized_sha256"],
        "old_result_normalized_sha256": old_hashes["normalized_sha256"],
        "axis_sha256": sha256(axis_data),
        "meta_sha256": sha256(meta_data),
    }
    checkpoint = load_checkpoint(args.checkpoint, identity)

    print("recovering the exact historical seed-0 sample stream", flush=True)
    replay_plan, replay_profile_failures = build_replay_plan(args.labels, old_result)

    plans: dict[tuple[str, str], dict[str, Any]] = {}
    for sample, points in replay_plan.items():
        plans[("original_replay_all", sample)] = {"points": points, "selection_seed": "legacy-stream"}

    print("selecting 600 deterministic points in each of 189 mapped cubes", flush=True)
    for index, row in enumerate(located_rows, 1):
        sample = row["sample"]
        lab = np.asarray(tifffile.imread(str(args.labels / f"{sample}.tif")))
        points, selection_seed = select_mapped_points(sample, lab)
        plans[("mapped_expansion", sample)] = {
            "points": points,
            "selection_seed": selection_seed,
        }
        if index % 25 == 0:
            print(f"  selected {index}/189", flush=True)

    pending_by_sample: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (cohort, sample), plan in plans.items():
        if (cohort, sample) not in checkpoint:
            pending_by_sample[sample].append((cohort, plan))

    total = len(pending_by_sample)
    for index, sample in enumerate(sorted(pending_by_sample), 1):
        jobs = pending_by_sample[sample]
        lab = np.asarray(tifffile.imread(str(args.labels / f"{sample}.tif")))
        ct = np.asarray(tifffile.imread(str(args.images / f"{sample}.tif")))
        lengths = [len(job[1]["points"]) for job in jobs]
        points = np.concatenate([job[1]["points"] for job in jobs], axis=0)
        normals = across_sheet_normals(ct, points)
        ts_all, ct_profile_all, label_profile_all = sample_profiles(ct, lab, points, normals)
        start = 0
        for (cohort, plan), length in zip(jobs, lengths, strict=True):
            stop = start + length
            mapping = located.get(sample)
            cube_lo = np.asarray(mapping["lo"], dtype=np.float64) if mapping else None
            result = analyse_profiles(
                sample,
                points[start:stop],
                normals[start:stop],
                ts_all,
                ct_profile_all[start:stop],
                label_profile_all[start:stop],
                axis if mapping else None,
                cube_lo,
            )
            expected_hash = points_hash(sample, plan["points"])
            if result["point_hash"] != expected_hash:
                raise AssertionError("point selection changed during analysis")
            row = {
                "identity": identity,
                "cohort": cohort,
                "selection_seed": plan["selection_seed"],
                "result": result,
            }
            append_checkpoint(args.checkpoint, row)
            checkpoint[(cohort, sample)] = row
            start = stop
        print(f"analysed {index}/{total}: {sample} ({', '.join(c for c, _ in jobs)})", flush=True)

    replay_records = [
        checkpoint[("original_replay_all", row["sample"])]["result"] for row in old_result["rows"]
    ]
    expansion_records = [
        checkpoint[("mapped_expansion", row["sample"])]["result"] for row in located_rows
    ]
    validation = replay_validation(replay_records, old_result)
    replay_mapped = [record for record in replay_records if record["mapped"]]
    if len(replay_mapped) != 9:
        raise RuntimeError(f"expected 9 mapped historical samples, got {len(replay_mapped)}")

    result = {
        "status": "corrective audit; the old population signed statistic is withdrawn",
        "generated_utc_date": "2026-08-20",
        "design": {
            "id": DESIGN_ID,
            "source_commit": source_commit,
            "frozen_documents": [
                "LABEL_PLACEMENT_ORIENTATION_CORRECTION_PREREG.md",
                "LABEL_PLACEMENT_ORIENTATION_CORRECTION_AMENDMENT_01.md",
                "LABEL_PLACEMENT_ORIENTATION_CORRECTION_AMENDMENT_02.md",
                "LABEL_PLACEMENT_ORIENTATION_CORRECTION_AMENDMENT_03.md",
            ],
            "seed": SEED,
            "points_per_sample": PER_SAMPLE,
            "min_valid_per_cell": MIN_VALID,
            "corridor_half_widths_vox": list(CORRIDORS),
            "alignment_min_abs_cosines": list(ALIGNMENT_THRESHOLDS),
            "bootstraps": BOOTSTRAPS,
            "spatial_block_size_vox": BLOCK_SIZE,
        },
        "inputs": {
            "overlap_report": overlap_hashes,
            "old_result": old_hashes,
            "axis": {"url": AXIS_URL, "sha256": sha256(axis_data), **axis_info},
            "volume_meta": {
                "url": META_URL,
                "sha256": sha256(meta_data),
                "uuid": VOLUME_UUID,
                "shape_zyx": list(VOLUME_SHAPE),
                "voxel_size_um": VOXEL_SIZE_UM,
            },
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "tifffile": tifffile.__version__,
            "imagecodecs": package_version("imagecodecs"),
        },
        "historical_replay": {
            "validation": validation,
            "selected_then_profile_rejected_before_last_row": replay_profile_failures,
            "n_all": len(replay_records),
            "n_independently_mapped": len(replay_mapped),
            "mapped_sample_results": replay_mapped,
        },
        "cohorts": {
            "original_replay_mapped_9": {
                "note": "mapping-available subset of the historical 30; overlaps expansion",
                "summaries": summarize(replay_mapped, "original_replay_mapped_9"),
            },
            "mapped_expansion_189": {
                "note": "all independently mapped Scroll1A cubes; not the original corpus",
                "summaries": summarize(expansion_records, "mapped_expansion_189"),
                "sample_results": expansion_records,
            },
        },
        "interpretation_limits": [
            "The old +0.0077-voxel population signed statistic and the claim that global label snapping has nothing to correct are withdrawn regardless of this outcome.",
            "The replay-9 subset and expansion-189 overlap and are not independent or pooled.",
            "The 189 mapped cubes cover Scroll1A only and do not represent all 892 public pairs.",
            "Absolute offset still mixes real displacement with estimator error on curved multi-sheet CT.",
            "This corrective audit alone is not a prize submission; generated labels and a positive held-out comparison against existing tooling are required.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    primary = next(
        row
        for row in result["cohorts"]["mapped_expansion_189"]["summaries"]
        if row["corridor_half_vox"] == 4.0 and row["alignment_min_abs_cosine"] == 0.0
    )
    print("historical replay: 30/30 exact", flush=True)
    print(
        "primary mapped expansion: "
        f"{primary['median_of_sample_oriented_medians']:+.4f} vox, "
        f"sample-bootstrap 95% CI {primary['sample_bootstrap_95pct_ci']}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
