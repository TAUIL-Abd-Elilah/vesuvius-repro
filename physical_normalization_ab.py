#!/usr/bin/env python3
"""Plan and score a frozen physical-truth A/B for villa #1364/#1386.

The planner reads only released label bits.  It never opens either prediction.  The scorer
accepts one frozen NPZ per selected block, produced by run_physical_normalization_ab.py:

    baseline_l1         uint8/bool published threshold-0.2 prediction
    corrected_pmax_l1   float32 max probability over each 2x2x2 level-0 cell
    metadata_json       canonical JSON tying the arrays to the manifest block

See PHYSICAL_NORMALIZATION_AB_PREREG.md for the non-tuning rules and decision gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROTOCOL_VERSION = 2
MANIFEST_STATUS = "preregistered_no_real_arm_scores_protocol_v2"
SELECTION_SEED = "vesuvius-physical-normalization-ab-v1-2026-08-11"
BOOTSTRAP_SEED = 20260811
SCORE_SIZE_L1 = 64
CANDIDATE_STRIDE_L1 = 128
BLOCKS_PER_Z_STRATUM = 8
Z_STRATA = 4
BLOCKS_PER_SCROLL = BLOCKS_PER_Z_STRATUM * Z_STRATA
Z_SAMPLE_STEP = 4
MIN_VALID_FRACTION = 0.05
MIN_SAMPLED_CENTERLINE = 256
NULL_SHIFT_L1 = 64
METRIC_HALO_L1 = 8
INFERENCE_TRIM_L0 = 64
FIXED_THRESHOLD = 0.2
BOOTSTRAP_DRAWS = 10_000
FP_NONINFERIORITY_MARGIN = 0.01

PR1386_COMMIT = "f74929a643095ce422ea4d9b70c25ae2b233a000"
PR1382_COMMIT = "5408c48d9db0558a78118d24fe9919ee63b204ee"
BROKEN_REPRO_COMMIT = "94ba215963afb6216e380fe2c86131fa5e724c3b"

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
LABEL_RELEASE = (
    "https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit/releases/tag/v1.0"
)

SCROLLS: dict[str, dict[str, Any]] = {
    "PHerc0139": {
        "label_store": "labels0139_L1.zarr",
        "label_archive": "labels0139_L1.tar",
        "label_archive_sha256": (
            "42fe53b760c2c9347d9f215bafa68beec8e96121d03549dab56a52a9a0a9e8dd"
        ),
        "label_shape_l1": [1248, 2304, 2208],
        "label_origin_l1": [1728, 576, 480],
        "ct_volume": "20250728140407-9.362um-1.2m-113keV-masked.zarr",
        "ct_shape_l0": [20974, 6621, 6621],
        "prediction": (
            "20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr"
        ),
        "registration_heldout_um": 4.09,
    },
    "PHerc1203": {
        "label_store": "labels1203_L1.zarr",
        "label_archive": "labels1203_L1.tar",
        "label_archive_sha256": (
            "32a09f6081342b0f015b258ec577d0296ff23a55892af9785689d8a55bff344c"
        ),
        "label_shape_l1": [2016, 3456, 3456],
        "label_origin_l1": [3936, 0, 0],
        "ct_volume": "20250820131727-9.362um-1.2m-113keV-masked.zarr",
        "ct_shape_l0": [18977, 6844, 6844],
        "prediction": (
            "20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr"
        ),
        "registration_heldout_um": 2.38,
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def require_clean_git(root: Path) -> str:
    status = git_output(root, "status", "--porcelain=v1")
    if status:
        raise SystemExit("refusing to preregister from a dirty worktree:\n" + status)
    return git_output(root, "rev-parse", "HEAD")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def as_int_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("min", "p25", "median", "p75", "max")}
    a = np.asarray(values, dtype=np.float64)
    q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "p25", "median", "p75", "max"), map(float, q)))


def _lattice_offsets(scroll: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(f"{SELECTION_SEED}|{scroll}|lattice".encode()).digest()
    # Multiples of 64 preserve the physical evaluator's arc tiling and z phase.
    return tuple((digest[i] & 1) * SCORE_SIZE_L1 for i in range(3))


def _candidate_hash(scroll: str, z: int, y: int, x: int) -> str:
    payload = f"{SELECTION_SEED}|{scroll}|{z}|{y}|{x}".encode("ascii")
    return sha256_bytes(payload)


def block_geometry(
    local_origin_l1: tuple[int, int, int], label_origin_l1: Iterable[int]
) -> dict[str, list[int]]:
    """Return every coordinate box needed for one score cube.

    Boxes are half-open and axis ordered z,y,x.  `prediction_extent_l1` contains the score
    cube, the upstream null source, and the metric halo.  `inference_bbox_l0` additionally
    places that extent inside the blend interior.
    """

    z, y, x = local_origin_l1
    oz, oy, ox = map(int, label_origin_l1)
    s, n, h = SCORE_SIZE_L1, NULL_SHIFT_L1, METRIC_HALO_L1
    ext_local = [z, z + s, y - n - h, y + s + h, x - h, x + s + h]
    ext_global_l1 = [
        oz + ext_local[0],
        oz + ext_local[1],
        oy + ext_local[2],
        oy + ext_local[3],
        ox + ext_local[4],
        ox + ext_local[5],
    ]
    score_global_l1 = [oz + z, oz + z + s, oy + y, oy + y + s, ox + x, ox + x + s]
    ext_global_l0 = [2 * v for v in ext_global_l1]
    score_global_l0 = [2 * v for v in score_global_l1]
    t = INFERENCE_TRIM_L0
    inference = [
        ext_global_l0[0] - t,
        ext_global_l0[1] + t,
        ext_global_l0[2] - t,
        ext_global_l0[3] + t,
        ext_global_l0[4] - t,
        ext_global_l0[5] + t,
    ]
    return {
        "score_local_l1": [z, z + s, y, y + s, x, x + s],
        "score_global_l1": score_global_l1,
        "score_global_l0": score_global_l0,
        "prediction_extent_local_l1": ext_local,
        "prediction_extent_global_l1": ext_global_l1,
        "prediction_extent_global_l0": ext_global_l0,
        "inference_bbox_l0": inference,
    }


def _box_inside_shape(box: Iterable[int], shape: Iterable[int]) -> bool:
    b = list(map(int, box))
    s = list(map(int, shape))
    return all(
        0 <= b[2 * axis] < b[2 * axis + 1] <= s[axis] for axis in range(3)
    )


def _open_zarr(path: Path):
    import zarr

    return zarr.open(str(path), mode="r")


def scan_candidates(scroll: str, labels_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = SCROLLS[scroll]
    store = labels_root / cfg["label_store"]
    if not store.is_dir():
        raise SystemExit(f"missing label store: {store}")
    zarray = load_json(store / ".zarray")
    attrs = load_json(store / ".zattrs")
    if as_int_list(zarray["shape"]) != cfg["label_shape_l1"]:
        raise SystemExit(f"{scroll}: unexpected label shape {zarray['shape']}")
    if as_int_list(attrs["origin_l1"]) != cfg["label_origin_l1"]:
        raise SystemExit(f"{scroll}: unexpected label origin {attrs['origin_l1']}")
    if zarray.get("dtype") != "|u1" or zarray.get("fill_value") != 0:
        raise SystemExit(f"{scroll}: labels must be uint8 with zero fill")

    labels = _open_zarr(store)
    shape = tuple(map(int, labels.shape))
    offsets = _lattice_offsets(scroll)
    candidates: list[dict[str, Any]] = []
    rejected_geometry = rejected_valid = rejected_centerline = 0

    axes: list[list[int]] = []
    for axis, (dim, offset) in enumerate(zip(shape, offsets)):
        vals = list(range(offset, dim - SCORE_SIZE_L1 + 1, CANDIDATE_STRIDE_L1))
        # If the seed chose the boundary lattice, preserve it; geometry checks below remove
        # candidates that lack null/metric/model context.
        axes.append(vals)

    for z in axes[0]:
        sampled_phase = (z + np.arange(SCORE_SIZE_L1)) % Z_SAMPLE_STEP == 0
        z_stratum = min(Z_STRATA - 1, int((z + SCORE_SIZE_L1 / 2) * Z_STRATA / shape[0]))
        for y in axes[1]:
            for x in axes[2]:
                geom = block_geometry((z, y, x), cfg["label_origin_l1"])
                if not _box_inside_shape(
                    geom["prediction_extent_local_l1"], cfg["label_shape_l1"]
                ) or not _box_inside_shape(geom["inference_bbox_l0"], cfg["ct_shape_l0"]):
                    rejected_geometry += 1
                    continue
                block = np.asarray(
                    labels[z : z + SCORE_SIZE_L1,
                           y : y + SCORE_SIZE_L1,
                           x : x + SCORE_SIZE_L1],
                    dtype=np.uint8,
                )
                valid = (block & 1) != 0
                material = (block & 2) != 0
                centerline = (block & 4) != 0
                n_vox = int(block.size)
                valid_count = int(valid.sum())
                if valid_count < math.ceil(MIN_VALID_FRACTION * n_vox):
                    rejected_valid += 1
                    continue
                sampled_counts = centerline[sampled_phase].sum(axis=(1, 2), dtype=np.int64)
                sampled_centerline = int(sampled_counts.sum())
                if sampled_centerline < MIN_SAMPLED_CENTERLINE:
                    rejected_centerline += 1
                    continue
                phase_indices = np.flatnonzero(sampled_phase)
                visual_k = int(phase_indices[int(np.argmax(sampled_counts))])
                material_count = int(material.sum())
                bp_count = int(((block & 16) != 0).sum()) if scroll == "PHerc1203" else 0
                recto_count = int(((block & 8) != 0).sum())
                rank = _candidate_hash(scroll, z, y, x)
                candidates.append(
                    {
                        "scroll": scroll,
                        "rank_sha256": rank,
                        "z_stratum": z_stratum,
                        "local_origin_l1": [z, y, x],
                        "geometry": geom,
                        "label_stats": {
                            "voxel_count": n_vox,
                            "valid_count": valid_count,
                            "valid_fraction": valid_count / n_vox,
                            "material_count": material_count,
                            "material_fraction_of_valid": material_count / max(valid_count, 1),
                            "sampled_centerline_count": sampled_centerline,
                            "recto_count": recto_count,
                            "boundary_poor_count": bp_count,
                            "boundary_poor_fraction_of_material": bp_count / max(material_count, 1),
                        },
                        "visual_slice_local_l1": z + visual_k,
                        "visual_slice_global_l1": cfg["label_origin_l1"][0] + z + visual_k,
                    }
                )

    summary = {
        "lattice_offsets_l1": list(offsets),
        "raw_grid_count": len(axes[0]) * len(axes[1]) * len(axes[2]),
        "rejected_geometry": rejected_geometry,
        "rejected_valid": rejected_valid,
        "rejected_centerline": rejected_centerline,
        "eligible_count": len(candidates),
        "eligible_by_z_stratum": {
            str(i): sum(c["z_stratum"] == i for c in candidates) for i in range(Z_STRATA)
        },
        "eligible_quantiles": {
            "valid_fraction": quantiles(
                [c["label_stats"]["valid_fraction"] for c in candidates]
            ),
            "material_fraction_of_valid": quantiles(
                [c["label_stats"]["material_fraction_of_valid"] for c in candidates]
            ),
            "sampled_centerline_count": quantiles(
                [float(c["label_stats"]["sampled_centerline_count"]) for c in candidates]
            ),
            "boundary_poor_fraction_of_material": quantiles(
                [c["label_stats"]["boundary_poor_fraction_of_material"] for c in candidates]
            ),
        },
    }
    return candidates, summary


def select_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for stratum in range(Z_STRATA):
        pool = sorted(
            (c for c in candidates if c["z_stratum"] == stratum),
            key=lambda c: c["rank_sha256"],
        )
        if len(pool) < BLOCKS_PER_Z_STRATUM:
            raise SystemExit(
                f"z stratum {stratum} has {len(pool)} eligible blocks; "
                f"need {BLOCKS_PER_Z_STRATUM}"
            )
        selected.extend(pool[:BLOCKS_PER_Z_STRATUM])
    selected.sort(key=lambda c: (c["z_stratum"], c["rank_sha256"]))
    for index, block in enumerate(selected):
        z, y, x = block["local_origin_l1"]
        block["selection_index"] = index
        block["block_id"] = f"{block['scroll']}-z{z:04d}-y{y:04d}-x{x:04d}"
        block["array_file"] = f"arrays/{block['block_id']}.npz"
        block["receipt_file"] = f"receipts/{block['block_id']}.json"
    return selected


def _file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_manifest(repo: Path, labels_root: Path, model_dir: Path) -> dict[str, Any]:
    implementation_commit = require_clean_git(repo)
    blocks: list[dict[str, Any]] = []
    inventories: dict[str, Any] = {}
    label_inputs: dict[str, Any] = {}
    for scroll, cfg in SCROLLS.items():
        archive = labels_root / cfg["label_archive"]
        if not archive.is_file():
            raise SystemExit(f"missing label archive: {archive}")
        archive_hash = sha256_file(archive)
        if archive_hash != cfg["label_archive_sha256"]:
            raise SystemExit(
                f"{archive}: SHA-256 {archive_hash} != {cfg['label_archive_sha256']}"
            )
        store = labels_root / cfg["label_store"]
        label_inputs[scroll] = {
            "release": LABEL_RELEASE,
            "archive": _file_record(archive),
            "store": cfg["label_store"],
            "zarray_sha256": sha256_file(store / ".zarray"),
            "zattrs_sha256": sha256_file(store / ".zattrs"),
            "shape_l1": cfg["label_shape_l1"],
            "origin_l1": cfg["label_origin_l1"],
            "registration_heldout_um": cfg["registration_heldout_um"],
        }
        candidates, inventory = scan_candidates(scroll, labels_root)
        inventories[scroll] = inventory
        blocks.extend(select_candidates(candidates))

    required_model = ["plans.json", "dataset.json", "fold_0/checkpoint_best.pth"]
    model_files = {}
    for rel in required_model:
        p = model_dir / rel
        if not p.is_file():
            raise SystemExit(f"missing model file: {p}")
        model_files[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    plans = load_json(model_dir / "plans.json")
    norm = plans["configurations"]["3d_fullres"]["normalization_schemes"]
    if norm != ["CTNormalization"]:
        raise SystemExit(f"model plans unexpectedly declare {norm}")
    intensity = plans["foreground_intensity_properties_per_channel"]["0"]

    implementation_files = {}
    for name in (
        "physical_normalization_ab.py",
        "run_physical_normalization_ab.py",
        "test_physical_normalization_ab.py",
        "PHYSICAL_NORMALIZATION_AB_PREREG.md",
        "PHYSICAL_NORMALIZATION_AB_AMENDMENT_01.md",
        "results/physical_normalization_ab/truth_power_audit.json",
        "requirements.txt",
    ):
        p = repo / name
        if not p.is_file():
            raise SystemExit(f"missing implementation file: {p}")
        implementation_files[name] = sha256_file(p)

    scroll_inputs = {}
    for scroll, cfg in SCROLLS.items():
        scroll_inputs[scroll] = {
            "ct_url": f"{BUCKET}/{scroll}/volumes/{cfg['ct_volume']}/0",
            "ct_shape_l0": cfg["ct_shape_l0"],
            "published_prediction_url": (
                f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
                f"{cfg['prediction']}/0"
            ),
            "published_prediction_artifact": cfg["prediction"],
            "published_threshold": FIXED_THRESHOLD,
        }

    manifest = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "status": MANIFEST_STATUS,
        "question": (
            "Does villa PR #1386 plans-driven CT normalization improve public m7 against "
            "two-scroll physical cross-scan truth?"
        ),
        "implementation": {
            "repo": "https://github.com/TAUIL-Abd-Elilah/vesuvius-repro",
            "implementation_commit": implementation_commit,
            "files_sha256": implementation_files,
            "corrected_villa_commit": PR1386_COMMIT,
            "broken_reproduction_commit": BROKEN_REPRO_COMMIT,
            "physical_metric_reference_commit": PR1382_COMMIT,
        },
        "inputs": {
            "labels": label_inputs,
            "model": {
                "source": "https://huggingface.co/scrollprize/surface_m7_nnunet",
                "files": model_files,
                "normalization_schemes": norm,
                "foreground_intensity_properties_channel_0": intensity,
            },
            "scrolls": scroll_inputs,
        },
        "sampling": {
            "selection_seed": SELECTION_SEED,
            "score_size_l1": SCORE_SIZE_L1,
            "candidate_stride_l1": CANDIDATE_STRIDE_L1,
            "z_strata": Z_STRATA,
            "blocks_per_z_stratum": BLOCKS_PER_Z_STRATUM,
            "z_sample_step": Z_SAMPLE_STEP,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "min_sampled_centerline": MIN_SAMPLED_CENTERLINE,
            "null_shift_l1": NULL_SHIFT_L1,
            "metric_halo_l1": METRIC_HALO_L1,
            "inference_trim_l0": INFERENCE_TRIM_L0,
            "inventories": inventories,
        },
        "analysis": {
            "fixed_threshold": FIXED_THRESHOLD,
            "matched_mass_scope": "one truth-blind valid-mask threshold per scroll",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "primary_metric": "recall_37um_minus_shifted_null_recall_37um",
            "far37_noninferiority_margin_absolute": FP_NONINFERIORITY_MARGIN,
            "minimum_point_blocks_per_scroll": 32,
        },
        "blocks": blocks,
    }
    manifest["content_sha256"] = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    return manifest


def inventory_command(args: argparse.Namespace) -> None:
    labels_root = Path(args.labels_root).resolve()
    report = {}
    for scroll in SCROLLS:
        candidates, summary = scan_candidates(scroll, labels_root)
        summary["selected_preview"] = [
            {
                "origin": c["local_origin_l1"],
                "z_stratum": c["z_stratum"],
                "rank": c["rank_sha256"],
                "stats": c["label_stats"],
            }
            for c in select_candidates(candidates)
        ]
        report[scroll] = summary
    if args.out:
        write_json(Path(args.out), report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


def plan_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    manifest = build_manifest(repo, Path(args.labels_root).resolve(), Path(args.model_dir).resolve())
    out = Path(args.out)
    write_json(out, manifest)
    print(f"wrote {out} ({len(manifest['blocks'])} frozen blocks)")
    print(f"content_sha256={manifest['content_sha256']}")
    print("Commit and push this manifest before running verify/run/score.")


def normal_field(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D material normal and coherence, matching the #1382 instrument."""
    from scipy import ndimage as ndi

    gy, gx = ndi.sobel(g, 0), ndi.sobel(g, 1)
    jyy = ndi.gaussian_filter(gy * gy, 6)
    jyx = ndi.gaussian_filter(gy * gx, 6)
    jxx = ndi.gaussian_filter(gx * gx, 6)
    phi = 0.5 * np.arctan2(2 * jyx, jyy - jxx)
    tr, det = jyy + jxx, jyy * jxx - jyx * jyx
    disc = np.sqrt(np.maximum(tr * tr / 4 - det, 0))
    lam1, lam2 = tr / 2 + disc, tr / 2 - disc
    coherence = (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-9)
    return np.cos(phi), np.sin(phi), coherence


COUNT_KEYS = (
    "n_centerline",
    "hit1",
    "hit2",
    "hit3",
    "null_hit2",
    "n_arcs",
    "arc_hit",
    "arc_gone",
    "null_arc_hit",
    "null_arc_gone",
    "valid_voxels",
    "pred_valid",
    "pred_invalid",
    "pred_far2",
    "pred_far4",
    "side_real_in",
    "side_real_out",
    "side_null_in",
    "side_null_out",
    "side_ideal_in",
    "side_ideal_out",
)


def blank_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def add_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key in COUNT_KEYS:
        dst[key] += int(src[key])


@dataclasses.dataclass
class TruthPlane:
    raw: np.ndarray
    valid: np.ndarray
    material: np.ndarray
    centerline: np.ndarray
    recto: np.ndarray
    components: np.ndarray
    distance_to_material: np.ndarray
    normal_y: np.ndarray | None = None
    normal_x: np.ndarray | None = None
    coherence: np.ndarray | None = None


def prepare_truth_plane(raw: np.ndarray, with_side: bool = True) -> TruthPlane:
    from scipy import ndimage as ndi

    raw = np.asarray(raw, dtype=np.uint8)
    valid = (raw & 1) != 0
    material = (raw & 2) != 0
    centerline = (raw & 4) != 0
    recto = (raw & 8) != 0
    components, _ = ndi.label(centerline, structure=np.ones((3, 3), dtype=np.uint8))
    distance_to_material = (
        ndi.distance_transform_edt(~material)
        if material.any()
        else np.full(material.shape, np.inf, dtype=np.float32)
    )
    ny = nx = coherence = None
    if with_side and material.any():
        ny, nx, coherence = normal_field(
            ndi.gaussian_filter(material.astype(np.float32), 1.5)
        )
        ys, xs = np.nonzero(material)
        cy, cx = float(ys.mean()), float(xs.mean())
        flip = (ny * (np.arange(raw.shape[0])[:, None] - cy)
                + nx * (np.arange(raw.shape[1])[None, :] - cx)) < 0
        ny = np.where(flip, -ny, ny)
        nx = np.where(flip, -nx, nx)
    return TruthPlane(
        raw, valid, material, centerline, recto, components,
        distance_to_material, ny, nx, coherence,
    )


def _distance_field(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi

    return (
        ndi.distance_transform_edt(~binary)
        if binary.any()
        else np.full(binary.shape, np.inf, dtype=np.float32)
    )


def _inward_counts(
    distance: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    ny: np.ndarray,
    nx: np.ndarray,
) -> tuple[int, int]:
    from scipy.ndimage import map_coordinates

    if len(ys) == 0:
        return 0, 0
    step = 2.0
    inward = map_coordinates(distance, [ys - step * ny, xs - step * nx], order=1)
    outward = map_coordinates(distance, [ys + step * ny, xs + step * nx], order=1)
    return int((inward < outward - 0.25).sum()), int((outward < inward - 0.25).sum())


def score_plane(
    truth: TruthPlane,
    prediction_extent: np.ndarray,
    score_y0: int,
    score_x0: int,
    extent_y0: int,
    extent_x0: int,
    with_side: bool = True,
) -> dict[str, int]:
    """Score one full-truth z plane and one local prediction extent.

    `score_*0` and `extent_*0` use label-store coordinates.  The prediction extent must
    include NULL_SHIFT_L1 upstream y voxels and METRIC_HALO_L1 on every in-plane score edge.
    Empty predictions are deliberately scored, never skipped.
    """

    pred = np.asarray(prediction_extent, dtype=bool)
    out = blank_counts()
    s = SCORE_SIZE_L1
    sy = slice(score_y0, score_y0 + s)
    sx = slice(score_x0, score_x0 + s)
    ey = score_y0 - extent_y0
    ex = score_x0 - extent_x0
    if not (
        ey >= NULL_SHIFT_L1 + METRIC_HALO_L1
        and ex >= METRIC_HALO_L1
        and ey + s + METRIC_HALO_L1 <= pred.shape[0]
        and ex + s + METRIC_HALO_L1 <= pred.shape[1]
    ):
        raise ValueError("prediction extent does not contain the frozen null/metric halo")

    valid_score = truth.valid[sy, sx]
    pred_score = pred[ey : ey + s, ex : ex + s]
    out["valid_voxels"] = int(valid_score.sum())
    out["pred_valid"] = int((pred_score & valid_score).sum())
    out["pred_invalid"] = int((pred_score & ~valid_score).sum())
    dmat = truth.distance_to_material[sy, sx]
    out["pred_far2"] = int((pred_score & valid_score & (dmat > 2)).sum())
    out["pred_far4"] = int((pred_score & valid_score & (dmat > 4)).sum())

    ys0, xs0 = np.nonzero(truth.centerline[sy, sx])
    ys = ys0 + score_y0
    xs = xs0 + score_x0
    out["n_centerline"] = int(len(ys))

    null = np.zeros_like(pred)
    null[NULL_SHIFT_L1:] = pred[:-NULL_SHIFT_L1]
    dreal = _distance_field(pred)
    dnull = _distance_field(null)
    yl, xl = ys - extent_y0, xs - extent_x0
    cd = dreal[yl, xl]
    nd = dnull[yl, xl]
    out["hit1"] = int((cd <= 1).sum())
    out["hit2"] = int((cd <= 2).sum())
    out["hit3"] = int((cd <= 3).sum())
    out["null_hit2"] = int((nd <= 2).sum())

    if len(ys):
        component = truth.components[ys, xs].astype(np.int64)
        aid = component * 10_000_000 + (ys // 64).astype(np.int64) * 2000 + xs // 64
        _, inv = np.unique(aid, return_inverse=True)
        count = np.bincount(inv)
        coverage = np.bincount(inv, weights=(cd <= 2)) / count
        null_coverage = np.bincount(inv, weights=(nd <= 2)) / count
        big = count >= 20
        out["n_arcs"] = int(big.sum())
        out["arc_hit"] = int((coverage[big] >= 0.5).sum())
        out["arc_gone"] = int((coverage[big] < 0.1).sum())
        out["null_arc_hit"] = int((null_coverage[big] >= 0.5).sum())
        out["null_arc_gone"] = int((null_coverage[big] < 0.1).sum())

    if (
        with_side
        and len(ys)
        and truth.normal_y is not None
        and truth.normal_x is not None
        and truth.coherence is not None
    ):
        coherent = truth.coherence[ys, xs] > 0.3
        ysc, xsc = ys[coherent], xs[coherent]
        ylc, xlc = ysc - extent_y0, xsc - extent_x0
        ny = truth.normal_y[ysc, xsc]
        nx = truth.normal_x[ysc, xsc]
        arms: list[tuple[str, np.ndarray]] = [("real", dreal), ("null", dnull)]
        recto_extent = truth.recto[
            extent_y0 : extent_y0 + pred.shape[0],
            extent_x0 : extent_x0 + pred.shape[1],
        ]
        arms.append(("ideal", _distance_field(recto_extent)))
        for name, distance in arms:
            selected = distance[ylc, xlc] <= 3
            i, o = _inward_counts(
                distance,
                ylc[selected].astype(np.float64),
                xlc[selected].astype(np.float64),
                ny[selected],
                nx[selected],
            )
            out[f"side_{name}_in"] = i
            out[f"side_{name}_out"] = o
    return out


def metrics(counts: dict[str, int]) -> dict[str, float | int | None]:
    def ratio(num: str, den: str) -> float | None:
        return counts[num] / counts[den] if counts[den] else None

    recall2 = ratio("hit2", "n_centerline")
    null2 = ratio("null_hit2", "n_centerline")
    arc = ratio("arc_hit", "n_arcs")
    narc = ratio("null_arc_hit", "n_arcs")
    pred = counts["pred_valid"]

    side_values: dict[str, float | None] = {}
    for name in ("real", "null", "ideal"):
        den = counts[f"side_{name}_in"] + counts[f"side_{name}_out"]
        side_values[name] = counts[f"side_{name}_in"] / den if den else None
    side_skill = None
    if all(side_values[name] is not None for name in ("real", "null", "ideal")):
        denom = float(side_values["ideal"]) - float(side_values["null"])
        if denom > 1e-9:
            side_skill = (float(side_values["real"]) - float(side_values["null"])) / denom

    return {
        "n_centerline": counts["n_centerline"],
        "recall_19um": ratio("hit1", "n_centerline"),
        "recall_37um": recall2,
        "recall_56um": ratio("hit3", "n_centerline"),
        "null_recall_37um": null2,
        "point_skill": recall2 - null2 if recall2 is not None and null2 is not None else None,
        "n_arcs": counts["n_arcs"],
        "arc_recall": arc,
        "arc_fully_missed": ratio("arc_gone", "n_arcs"),
        "null_arc_recall": narc,
        "null_arc_fully_missed": ratio("null_arc_gone", "n_arcs"),
        "arc_skill": arc - narc if arc is not None and narc is not None else None,
        "valid_voxels": counts["valid_voxels"],
        "pred_valid": pred,
        "predicted_positive_fraction_valid": ratio("pred_valid", "valid_voxels"),
        "pred_invalid": counts["pred_invalid"],
        "pred_far37_fraction": counts["pred_far2"] / pred if pred else 0.0,
        "pred_far75_fraction": counts["pred_far4"] / pred if pred else 0.0,
        "side_n_decided": counts["side_real_in"] + counts["side_real_out"],
        "side_inward": side_values["real"],
        "side_inward_null_full": side_values["null"],
        "side_inward_ideal_full": side_values["ideal"],
        "side_skill_of_ideal_full": side_skill,
    }


def choose_matched_mass_threshold(values: np.ndarray, target_count: int) -> tuple[float, int]:
    """Choose a strict-`>` threshold closest to target_count, conservatively on ties."""

    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite and in [0,1]")
    n = len(values)
    if not 0 <= target_count <= n:
        raise ValueError("target_count outside probability population")
    if n == 0:
        return 1.0, 0
    unique = np.unique(values)
    # Candidate thresholds produce all realizable strict-positive counts.  Include extrema.
    candidates = [float(np.nextafter(unique[-1], np.inf))]
    candidates.extend(float(v) for v in unique[::-1])
    candidates.append(float(np.nextafter(unique[0], -np.inf)))
    best: tuple[int, float, int] | None = None
    for threshold in candidates:
        count = int((values > threshold).sum())
        # Higher threshold wins equal-distance ties: it cannot buy a result with extra mass.
        key = (abs(count - target_count), -threshold, count)
        if best is None or key < best:
            best = key
    assert best is not None
    return -best[1], best[2]


def _load_block_arrays(path: Path, block: dict[str, Any], manifest_hash: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing frozen block array: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {"baseline_l1", "corrected_pmax_l1", "metadata_json"}
        if set(data.files) != required:
            raise SystemExit(f"{path}: expected keys {sorted(required)}, got {sorted(data.files)}")
        baseline = np.asarray(data["baseline_l1"], dtype=np.uint8)
        corrected = np.asarray(data["corrected_pmax_l1"], dtype=np.float32)
        raw_meta = data["metadata_json"]
        meta_text = str(raw_meta.item())
        meta = json.loads(meta_text)
    expected_shape = (
        SCORE_SIZE_L1,
        SCORE_SIZE_L1 + NULL_SHIFT_L1 + 2 * METRIC_HALO_L1,
        SCORE_SIZE_L1 + 2 * METRIC_HALO_L1,
    )
    if baseline.shape != expected_shape or corrected.shape != expected_shape:
        raise SystemExit(
            f"{path}: shape mismatch baseline={baseline.shape} corrected={corrected.shape} "
            f"expected={expected_shape}"
        )
    if not np.isin(baseline, [0, 1]).all():
        raise SystemExit(f"{path}: baseline is not binary")
    if not np.isfinite(corrected).all() or ((corrected < 0) | (corrected > 1)).any():
        raise SystemExit(f"{path}: corrected probabilities are invalid")
    expected_meta = {
        "schema_version": 1,
        "manifest_content_sha256": manifest_hash,
        "block_id": block["block_id"],
        "prediction_extent_global_l1": block["geometry"]["prediction_extent_global_l1"],
    }
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            raise SystemExit(f"{path}: metadata {key}={meta.get(key)!r}, expected {value!r}")
    return {"baseline": baseline.astype(bool), "corrected": corrected, "metadata": meta}


def _target_view(array: np.ndarray) -> np.ndarray:
    y0, x0 = NULL_SHIFT_L1 + METRIC_HALO_L1, METRIC_HALO_L1
    return array[:, y0 : y0 + SCORE_SIZE_L1, x0 : x0 + SCORE_SIZE_L1]


def _bootstrap(values: list[float], seed: int) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    a = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(a, size=(BOOTSTRAP_DRAWS, len(a)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": len(a), "mean": float(a.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def _stratified_bootstrap(
    values_by_group: dict[str, list[float]], seed: int
) -> dict[str, float | int | None]:
    """Equal-weight group bootstrap; selection freezes equal n within each z stratum."""

    groups = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in sorted(values_by_group.items())
        if values
    }
    if not groups:
        return {
            "n": 0,
            "groups": 0,
            "mean": None,
            "ci95_low": None,
            "ci95_high": None,
        }
    rng = np.random.default_rng(seed)
    group_draws = []
    group_means = []
    for values in groups.values():
        group_draws.append(
            rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
        )
        group_means.append(float(values.mean()))
    draws = np.stack(group_draws, axis=1).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "n": int(sum(len(values) for values in groups.values())),
        "groups": len(groups),
        "mean": float(np.mean(group_means)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def compare_and_gate(
    per_block: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, float | int | None], dict[str, bool]]:
    """Build the frozen v2 comparisons and claim gates from block-level metrics."""

    comparisons: dict[str, Any] = {}
    for scroll in SCROLLS:
        rows = [r for r in per_block if r["scroll"] == scroll]
        comparisons[scroll] = {}
        for corrected in ("corrected_fixed", "corrected_matched"):
            point_diffs_by_stratum: dict[str, list[float]] = defaultdict(list)
            arc_diffs_by_stratum: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                corrected_point = row["arms"][corrected]["point_skill"]
                baseline_point = row["arms"]["published"]["point_skill"]
                if corrected_point is not None and baseline_point is not None:
                    point_diffs_by_stratum[str(row["z_stratum"])].append(
                        corrected_point - baseline_point
                    )
                corrected_arc = row["arms"][corrected]["arc_skill"]
                baseline_arc = row["arms"]["published"]["arc_skill"]
                if corrected_arc is not None and baseline_arc is not None:
                    arc_diffs_by_stratum[str(row["z_stratum"])].append(
                        corrected_arc - baseline_arc
                    )
            fp_diffs = [
                r["arms"][corrected]["pred_far37_fraction"]
                - r["arms"]["published"]["pred_far37_fraction"]
                for r in rows
            ]
            comparisons[scroll][corrected] = {
                "point_skill_delta": _stratified_bootstrap(
                    dict(point_diffs_by_stratum),
                    BOOTSTRAP_SEED + (0 if scroll == "PHerc0139" else 100)
                    + (0 if corrected == "corrected_fixed" else 1),
                ),
                "arc_skill_delta_secondary": _stratified_bootstrap(
                    dict(arc_diffs_by_stratum),
                    BOOTSTRAP_SEED + 10_000 + (0 if scroll == "PHerc0139" else 100)
                    + (0 if corrected == "corrected_fixed" else 1),
                ),
                "far37_fraction_delta_macro_mean": (
                    float(np.mean(fp_diffs)) if fp_diffs else None
                ),
            }

    pooled_diffs_by_scroll_stratum: dict[str, list[float]] = defaultdict(list)
    for scroll in SCROLLS:
        for row in (r for r in per_block if r["scroll"] == scroll):
            corrected_skill = row["arms"]["corrected_fixed"]["point_skill"]
            baseline_skill = row["arms"]["published"]["point_skill"]
            if corrected_skill is not None and baseline_skill is not None:
                pooled_diffs_by_scroll_stratum[
                    f"{scroll}:z{row['z_stratum']}"
                ].append(corrected_skill - baseline_skill)
    pooled = _stratified_bootstrap(
        dict(pooled_diffs_by_scroll_stratum), BOOTSTRAP_SEED + 999
    )

    gates = {
        "fixed_point_skill_positive_each_scroll": all(
            comparisons[s]["corrected_fixed"]["point_skill_delta"]["mean"] is not None
            and comparisons[s]["corrected_fixed"]["point_skill_delta"]["mean"] > 0
            for s in SCROLLS
        ),
        "pooled_fixed_ci_excludes_zero": (
            pooled["ci95_low"] is not None and pooled["ci95_low"] > 0
        ),
        "far37_noninferior_each_scroll": all(
            comparisons[s]["corrected_fixed"]["far37_fraction_delta_macro_mean"]
            is not None
            and comparisons[s]["corrected_fixed"]["far37_fraction_delta_macro_mean"]
            <= FP_NONINFERIORITY_MARGIN
            for s in SCROLLS
        ),
        "matched_point_skill_nonnegative_each_scroll": all(
            comparisons[s]["corrected_matched"]["point_skill_delta"]["mean"] is not None
            and comparisons[s]["corrected_matched"]["point_skill_delta"]["mean"] >= 0
            for s in SCROLLS
        ),
        "all_32_point_blocks_each_scroll": all(
            sum(
                r["scroll"] == s
                and r["arms"]["published"]["n_centerline"]
                == r["label_stats"]["sampled_centerline_count"]
                and r["arms"]["published"]["n_centerline"] >= MIN_SAMPLED_CENTERLINE
                for r in per_block
            )
            == BLOCKS_PER_SCROLL
            for s in SCROLLS
        ),
    }
    gates["primary_claim_passes"] = all(gates.values())
    return comparisons, pooled, gates


def verify_manifest_implementation(manifest: dict[str, Any]) -> None:
    """Refuse scoring if any frozen implementation or protocol file has drifted."""

    repo = Path(__file__).resolve().parent
    expected = manifest.get("implementation", {}).get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise SystemExit("manifest does not freeze implementation file hashes")
    for name, digest in sorted(expected.items()):
        path = repo / name
        if not path.is_file():
            raise SystemExit(f"missing frozen implementation file: {path}")
        actual = sha256_file(path)
        if actual != digest:
            raise SystemExit(f"implementation drift: {name} SHA {actual} != {digest}")


def score_command(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    recorded = manifest.pop("content_sha256", None)
    actual = sha256_bytes(canonical_json(manifest).encode("utf-8"))
    manifest["content_sha256"] = recorded
    if recorded != actual:
        raise SystemExit(f"manifest content hash mismatch: {recorded} != {actual}")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise SystemExit("unsupported protocol version")
    if manifest.get("status") != MANIFEST_STATUS:
        raise SystemExit("manifest is not the frozen v2 preregistration")
    verify_manifest_implementation(manifest)

    labels_root = Path(args.labels_root).resolve()
    arrays_root = Path(args.arrays_root).resolve()
    block_arrays = {
        block["block_id"]: _load_block_arrays(
            arrays_root / block["array_file"], block, recorded
        )
        for block in manifest["blocks"]
    }

    matched: dict[str, dict[str, Any]] = {}
    label_stores = {scroll: _open_zarr(labels_root / cfg["label_store"])
                    for scroll, cfg in SCROLLS.items()}
    for scroll in SCROLLS:
        values = []
        baseline_count = valid_count = 0
        for block in (b for b in manifest["blocks"] if b["scroll"] == scroll):
            z0, z1, y0, y1, x0, x1 = block["geometry"]["score_local_l1"]
            labels = np.asarray(label_stores[scroll][z0:z1, y0:y1, x0:x1], dtype=np.uint8)
            valid = (labels & 1) != 0
            arrays = block_arrays[block["block_id"]]
            base = _target_view(arrays["baseline"])
            corr = _target_view(arrays["corrected"])
            baseline_count += int((base & valid).sum())
            valid_count += int(valid.sum())
            values.append(corr[valid])
        population = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
        threshold, realized = choose_matched_mass_threshold(population, baseline_count)
        matched[scroll] = {
            "threshold": threshold,
            "target_published_positive_count": baseline_count,
            "realized_corrected_positive_count": realized,
            "valid_voxel_count": valid_count,
            "absolute_count_error": abs(realized - baseline_count),
        }

    arms = ("published", "corrected_fixed", "corrected_matched")
    aggregate = {scroll: {arm: blank_counts() for arm in arms} for scroll in SCROLLS}
    per_block_counts = {
        block["block_id"]: {arm: blank_counts() for arm in arms}
        for block in manifest["blocks"]
    }
    jobs_by_plane: dict[tuple[str, int], list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for block in manifest["blocks"]:
        z0 = block["geometry"]["score_local_l1"][0]
        for k in range(SCORE_SIZE_L1):
            if (z0 + k) % Z_SAMPLE_STEP == 0:
                jobs_by_plane[(block["scroll"], z0 + k)].append((block, k))

    for (scroll, z), jobs in sorted(jobs_by_plane.items()):
        raw = np.asarray(label_stores[scroll][z, :, :], dtype=np.uint8)
        truth = prepare_truth_plane(raw, with_side=not args.no_side)
        for block, k in jobs:
            geom = block["geometry"]
            _, _, y0, _, x0, _ = geom["score_local_l1"]
            _, _, ey0, _, ex0, _ = geom["prediction_extent_local_l1"]
            arrays = block_arrays[block["block_id"]]
            plane_arms = {
                "published": arrays["baseline"][k],
                "corrected_fixed": arrays["corrected"][k] > FIXED_THRESHOLD,
                "corrected_matched": arrays["corrected"][k] > matched[scroll]["threshold"],
            }
            for arm, pred in plane_arms.items():
                counts = score_plane(
                    truth, pred, y0, x0, ey0, ex0, with_side=not args.no_side
                )
                add_counts(per_block_counts[block["block_id"]][arm], counts)
                add_counts(aggregate[scroll][arm], counts)

    per_block = []
    blocks_by_id = {b["block_id"]: b for b in manifest["blocks"]}
    for block_id, counts_by_arm in per_block_counts.items():
        block = blocks_by_id[block_id]
        per_block.append(
            {
                "block_id": block_id,
                "scroll": block["scroll"],
                "z_stratum": block["z_stratum"],
                "label_stats": block["label_stats"],
                "arms": {arm: metrics(counts) for arm, counts in counts_by_arm.items()},
            }
        )
    per_block.sort(key=lambda row: row["block_id"])

    comparisons, pooled, gates = compare_and_gate(per_block)

    result = {
        "schema_version": 1,
        "manifest_path": str(manifest_path),
        "manifest_content_sha256": recorded,
        "matched_mass": matched,
        "aggregate": {
            scroll: {arm: metrics(counts) for arm, counts in by_arm.items()}
            for scroll, by_arm in aggregate.items()
        },
        "per_block": per_block,
        "comparisons": comparisons,
        "pooled_fixed_point_skill_delta": pooled,
        "gates": gates,
        "side_metrics_computed": not args.no_side,
    }
    result["content_sha256"] = sha256_bytes(canonical_json(result).encode("utf-8"))
    write_json(Path(args.out), result)
    print(json.dumps({"gates": gates, "matched_mass": matched}, indent=2))
    print(f"wrote {args.out}")


def _verify_hashed_json(path: Path) -> dict[str, Any]:
    value = load_json(path)
    recorded = value.pop("content_sha256", None)
    actual = sha256_bytes(canonical_json(value).encode("utf-8"))
    value["content_sha256"] = recorded
    if recorded != actual:
        raise SystemExit(f"{path}: content hash mismatch")
    return value


def _overlay(
    material: np.ndarray,
    centerline: np.ndarray,
    prediction: np.ndarray | None = None,
) -> np.ndarray:
    image = np.zeros((*material.shape, 3), dtype=np.float32)
    image[material] = (0.30, 0.30, 0.30)
    image[centerline] = (0.95, 0.85, 0.10)
    if prediction is not None:
        image[prediction] = (0.10, 0.70, 1.00)
        image[prediction & material] = (0.15, 0.95, 0.55)
        image[prediction & centerline] = (1.00, 1.00, 1.00)
    return image


def figures_command(args: argparse.Namespace) -> None:
    manifest = _verify_hashed_json(Path(args.manifest).resolve())
    result = _verify_hashed_json(Path(args.result).resolve())
    if result.get("manifest_content_sha256") != manifest.get("content_sha256"):
        raise SystemExit("result was not produced from this manifest")
    labels_root = Path(args.labels_root).resolve()
    arrays_root = Path(args.arrays_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    selected = []
    for scroll in SCROLLS:
        for stratum in range(Z_STRATA):
            pool = sorted(
                (
                    b for b in manifest["blocks"]
                    if b["scroll"] == scroll and b["z_stratum"] == stratum
                ),
                key=lambda b: b["rank_sha256"],
            )
            if not pool:
                raise SystemExit(f"no visual block for {scroll} z stratum {stratum}")
            selected.append(pool[0])

    records = []
    stores = {scroll: _open_zarr(labels_root / cfg["label_store"])
              for scroll, cfg in SCROLLS.items()}
    for block in selected:
        scroll = block["scroll"]
        arrays_path = arrays_root / block["array_file"]
        arrays = _load_block_arrays(
            arrays_path, block, manifest["content_sha256"]
        )
        z0, z1, y0, y1, x0, x1 = block["geometry"]["score_local_l1"]
        visual_z = int(block["visual_slice_local_l1"])
        if not z0 <= visual_z < z1 or visual_z % Z_SAMPLE_STEP:
            raise SystemExit(f"invalid frozen visual slice in {block['block_id']}")
        k = visual_z - z0
        raw = np.asarray(stores[scroll][visual_z, y0:y1, x0:x1], dtype=np.uint8)
        valid = (raw & 1) != 0
        material = (raw & 2) != 0
        centerline = (raw & 4) != 0
        baseline = _target_view(arrays["baseline"])[k] & valid
        probability = _target_view(arrays["corrected"])[k]
        fixed = (probability > FIXED_THRESHOLD) & valid
        matched_threshold = float(result["matched_mass"][scroll]["threshold"])
        matched = (probability > matched_threshold) & valid

        difference = np.zeros((*raw.shape, 3), dtype=np.float32)
        both = baseline & fixed
        difference[both] = (0.65, 0.65, 0.65)
        difference[baseline & ~fixed] = (1.00, 0.20, 0.15)
        difference[fixed & ~baseline] = (0.10, 0.55, 1.00)
        difference[centerline] = (1.00, 1.00, 1.00)

        distance = (
            ndi.distance_transform_edt(~material)
            if material.any()
            else np.full(material.shape, np.inf)
        )
        far = np.zeros((*raw.shape, 3), dtype=np.float32)
        far[material] = (0.25, 0.25, 0.25)
        far[centerline] = (1.00, 1.00, 0.10)
        far[fixed & (distance <= 2)] = (0.10, 0.85, 0.45)
        far[fixed & (distance > 2)] = (1.00, 0.15, 0.15)

        panels = [
            ("physical material / centerline", _overlay(material, centerline)),
            ("published instance-zscore", _overlay(material, centerline, baseline)),
            ("corrected CT, threshold 0.2", _overlay(material, centerline, fixed)),
            ("red: published only; blue: corrected only", difference),
            (f"corrected matched mass, t={matched_threshold:.4f}",
             _overlay(material, centerline, matched)),
            ("corrected: green <=37um; red >37um", far),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for axis, (title, image) in zip(axes.ravel(), panels):
            axis.imshow(image, interpolation="nearest", origin="upper")
            axis.set_title(title, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(
            f"{scroll} | {block['block_id']} | global L1 z={block['visual_slice_global_l1']}\n"
            "yellow=physical centerline; cyan/green=model prediction",
            fontsize=11,
        )
        filename = f"{scroll}-z{block['z_stratum']}-{block['block_id']}.png"
        path = out_dir / filename
        fig.savefig(path, dpi=180, metadata={"Software": "vesuvius-repro physical A/B"})
        plt.close(fig)
        records.append(
            {
                "scroll": scroll,
                "z_stratum": block["z_stratum"],
                "block_id": block["block_id"],
                "visual_slice_local_l1": visual_z,
                "visual_slice_global_l1": block["visual_slice_global_l1"],
                "array_file_sha256": sha256_file(arrays_path),
                "figure": filename,
                "figure_sha256": sha256_file(path),
                "baseline_positive": int(baseline.sum()),
                "corrected_fixed_positive": int(fixed.sum()),
                "corrected_matched_positive": int(matched.sum()),
                "physical_centerline": int(centerline.sum()),
            }
        )
    figure_manifest = {
        "schema_version": 1,
        "selection_rule": "first SHA-ranked frozen block per scroll and z stratum",
        "source_manifest_content_sha256": manifest["content_sha256"],
        "source_result_content_sha256": result["content_sha256"],
        "figures": records,
    }
    figure_manifest["content_sha256"] = sha256_bytes(
        canonical_json(figure_manifest).encode("utf-8")
    )
    write_json(out_dir / "manifest.json", figure_manifest)
    print(f"wrote {len(records)} frozen figures to {out_dir}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory", help="label-only candidate census")
    inventory.add_argument("--labels-root", required=True)
    inventory.add_argument("--out")
    inventory.set_defaults(func=inventory_command)

    plan = sub.add_parser("plan", help="write the frozen sample/provenance manifest")
    plan.add_argument("--labels-root", required=True)
    plan.add_argument("--model-dir", required=True)
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=plan_command)

    score = sub.add_parser("score", help="score all complete frozen block arrays")
    score.add_argument("--manifest", required=True)
    score.add_argument("--labels-root", required=True)
    score.add_argument("--arrays-root", required=True)
    score.add_argument("--out", required=True)
    score.add_argument("--no-side", action="store_true", help="diagnostic only")
    score.set_defaults(func=score_command)

    figures = sub.add_parser("figures", help="render the eight manifest-frozen overlays")
    figures.add_argument("--manifest", required=True)
    figures.add_argument("--result", required=True)
    figures.add_argument("--labels-root", required=True)
    figures.add_argument("--arrays-root", required=True)
    figures.add_argument("--out-dir", required=True)
    figures.set_defaults(func=figures_command)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
