#!/usr/bin/env python3
"""Plan and verify a cross-scan physical-truth fine-tuning experiment.

This module is deliberately outcome-blind. The planner reads the released physical label
bits and the already-preregistered block manifest, but it never opens a prediction or a CT
volume. Training, inference, and scoring are separate stages that consume its frozen plan.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from physical_normalization_ab import SCROLLS, scan_candidates


PROTOCOL_VERSION = 1
PLAN_STATUS = "preoutcome_design_lock_no_pilot_or_primary_predictions"
SELECTION_SEED = "vesuvius-crossscan-physical-finetune-v1-2026-08-11"
SOURCE_MANIFEST_FILE_SHA256 = (
    "d0831d7bb8f5a3aa47eaf4f21d414c336d0a217f6ebe5ad0dc9ecd4dc57423eb"
)
SOURCE_MANIFEST_CONTENT_SHA256 = (
    "567a18faa1c8ca7e743c9240133f4200e67e3085823dd4795c4518e3e0e65ac0"
)

TRAIN_SCROLL = "PHerc1203"
SAFETY_SCROLL = "PHerc0139"
Z_STRATA = 4
SCORE_SIZE_L1 = 64
TRAIN_CROP_SIZE_L1 = 96
EVAL_CONTEXT_SIZE_L1 = 128
MIN_VALID_FRACTION = 0.50
MIN_RECTO_COUNT_SCORE_CUBE = 4096
DIFFICULTY_EDGES = (0.40, 0.80, 0.95)
TRAIN_PER_BIN_PER_STRATUM = 16
VALIDATION_PER_BIN_PER_STRATUM = 2
PILOT_PER_BIN_PER_STRATUM = 2

FOLDS = {
    "even": {"train_z_strata": [0, 2], "held_out_z_strata": [1, 3]},
    "odd": {"train_z_strata": [1, 3], "held_out_z_strata": [0, 2]},
}

PILOT_SEED = 39
INFERENTIAL_SEEDS = [40, 41, 42, 43, 44, 45]
PILOT_STEPS = 2000
PILOT_RETRY_STEPS = 4000
PILOT_AP_GATE = 0.005
PRIMARY_AP_EFFECT_GATE = 0.010
SAFETY_AP_MARGIN = 0.005


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def require_public_clean_head(root: Path) -> dict[str, str]:
    status = git_output(root, "status", "--porcelain=v1")
    if status:
        raise SystemExit("refusing to build a lock from a dirty worktree:\n" + status)
    head = git_output(root, "rev-parse", "HEAD")
    try:
        upstream = git_output(root, "rev-parse", "@{u}")
        upstream_name = git_output(root, "rev-parse", "--abbrev-ref", "@{u}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit("branch must have a public upstream before planning") from exc
    if head != upstream:
        raise SystemExit(f"HEAD {head} is not pushed upstream {upstream}")
    return {
        "commit": head,
        "branch": git_output(root, "branch", "--show-current"),
        "upstream": upstream_name,
    }


def difficulty_bin(boundary_poor_fraction: float) -> int:
    value = float(boundary_poor_fraction)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"boundary-poor fraction outside [0,1]: {value}")
    return int(np.searchsorted(np.asarray(DIFFICULTY_EDGES), value, side="right"))


def physical_target_l1(labels: np.ndarray) -> np.ndarray:
    """Map released bit labels to background=0, recto=1, ignore=2.

    Only valid air is a negative. Material away from the measured recto band remains
    ignored because the label volume does not identify a unique correct surface there.
    Recto is applied last so the intended one-voxel dilation outside material stays positive.
    """

    labels = np.asarray(labels)
    if labels.dtype.kind not in "ui":
        raise TypeError(f"physical labels must be integer, got {labels.dtype}")
    valid = (labels & 1) != 0
    material = (labels & 2) != 0
    recto = (labels & 8) != 0
    target = np.full(labels.shape, 2, dtype=np.uint8)
    target[valid & ~material & ~recto] = 0
    target[valid & recto] = 1
    return target


def upsample_target_l0(target_l1: np.ndarray) -> np.ndarray:
    target = np.asarray(target_l1)
    if target.ndim != 3:
        raise ValueError(f"expected 3D target, got shape {target.shape}")
    values = set(map(int, np.unique(target)))
    if not values <= {0, 1, 2}:
        raise ValueError(f"target has unexpected values {sorted(values)}")
    out = target.astype(np.uint8, copy=False)
    for axis in range(3):
        out = np.repeat(out, 2, axis=axis)
    return out


def _box_from_score_origin(origin: Iterable[int], size: int) -> list[int]:
    z, y, x = map(int, origin)
    before = (size - SCORE_SIZE_L1) // 2
    after = size - SCORE_SIZE_L1 - before
    return [z - before, z + SCORE_SIZE_L1 + after,
            y - before, y + SCORE_SIZE_L1 + after,
            x - before, x + SCORE_SIZE_L1 + after]


def _score_box(origin: Iterable[int]) -> list[int]:
    z, y, x = map(int, origin)
    return [z, z + SCORE_SIZE_L1, y, y + SCORE_SIZE_L1, x, x + SCORE_SIZE_L1]


def _global_l0_box(local_l1_box: Iterable[int], label_origin_l1: Iterable[int]) -> list[int]:
    box = list(map(int, local_l1_box))
    origin = list(map(int, label_origin_l1))
    return [2 * (box[2 * axis + edge] + origin[axis])
            for axis in range(3) for edge in range(2)]


def _box_inside(box: Iterable[int], shape: Iterable[int]) -> bool:
    b = list(map(int, box))
    s = list(map(int, shape))
    return all(0 <= b[2 * axis] < b[2 * axis + 1] <= s[axis]
               for axis in range(3))


def boxes_intersect(a: Iterable[int], b: Iterable[int]) -> bool:
    aa = list(map(int, a))
    bb = list(map(int, b))
    return all(aa[2 * axis] < bb[2 * axis + 1]
               and bb[2 * axis] < aa[2 * axis + 1]
               for axis in range(3))


def _allocation_hash(candidate: dict[str, Any]) -> str:
    z, y, x = candidate["local_origin_l1"]
    payload = f"{SELECTION_SEED}|{TRAIN_SCROLL}|{z}|{y}|{x}".encode("ascii")
    return sha256_bytes(payload)


def _case_from_candidate(candidate: dict[str, Any], role: str) -> dict[str, Any]:
    c = copy.deepcopy(candidate)
    origin = list(map(int, c["local_origin_l1"]))
    z, y, x = origin
    train_box = _box_from_score_origin(origin, TRAIN_CROP_SIZE_L1)
    eval_box = _box_from_score_origin(origin, EVAL_CONTEXT_SIZE_L1)
    cfg = SCROLLS[TRAIN_SCROLL]
    return {
        "case_id": f"{TRAIN_SCROLL}-z{z:04d}-y{y:04d}-x{x:04d}-{role}",
        "scroll": TRAIN_SCROLL,
        "role": role,
        "z_stratum": int(c["z_stratum"]),
        "difficulty_bin": difficulty_bin(
            c["label_stats"]["boundary_poor_fraction_of_material"]
        ),
        "allocation_sha256": _allocation_hash(c),
        "local_origin_l1": origin,
        "score_box_local_l1": _score_box(origin),
        "training_label_box_local_l1": train_box,
        "training_ct_box_global_l0": _global_l0_box(
            train_box, cfg["label_origin_l1"]
        ),
        "evaluation_context_box_local_l1": eval_box,
        "evaluation_ct_box_global_l0": _global_l0_box(
            eval_box, cfg["label_origin_l1"]
        ),
        "label_stats_score_cube": copy.deepcopy(c["label_stats"]),
    }


def allocate_cases(
    candidates: list[dict[str, Any]], primary_origins: set[tuple[int, int, int]]
) -> dict[str, list[dict[str, Any]]]:
    eligible: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for stratum in range(Z_STRATA):
        for bin_id in range(len(DIFFICULTY_EDGES) + 1):
            eligible[(stratum, bin_id)] = []

    for candidate in candidates:
        origin = tuple(map(int, candidate["local_origin_l1"]))
        stats = candidate["label_stats"]
        if origin in primary_origins:
            continue
        if float(stats["valid_fraction"]) < MIN_VALID_FRACTION:
            continue
        if int(stats["recto_count"]) < MIN_RECTO_COUNT_SCORE_CUBE:
            continue
        key = (
            int(candidate["z_stratum"]),
            difficulty_bin(stats["boundary_poor_fraction_of_material"]),
        )
        eligible[key].append(candidate)

    allocations: dict[str, list[dict[str, Any]]] = {
        "train": [], "internal_validation": [], "pilot": []
    }
    needed = (
        TRAIN_PER_BIN_PER_STRATUM
        + VALIDATION_PER_BIN_PER_STRATUM
        + PILOT_PER_BIN_PER_STRATUM
    )
    for key, pool in eligible.items():
        ranked = sorted(pool, key=_allocation_hash)
        if len(ranked) < needed:
            raise ValueError(f"stratum/bin {key} has {len(ranked)} cases; need {needed}")
        i = 0
        for role, count in (
            ("train", TRAIN_PER_BIN_PER_STRATUM),
            ("internal_validation", VALIDATION_PER_BIN_PER_STRATUM),
            ("pilot", PILOT_PER_BIN_PER_STRATUM),
        ):
            allocations[role].extend(
                _case_from_candidate(c, role) for c in ranked[i:i + count]
            )
            i += count

    for cases in allocations.values():
        cases.sort(key=lambda c: (c["z_stratum"], c["difficulty_bin"],
                                  c["allocation_sha256"]))
    return allocations


def _primary_case(block: dict[str, Any]) -> dict[str, Any]:
    scroll = str(block["scroll"])
    origin = list(map(int, block["local_origin_l1"]))
    cfg = SCROLLS[scroll]
    context = _box_from_score_origin(origin, EVAL_CONTEXT_SIZE_L1)
    return {
        "block_id": block["block_id"],
        "scroll": scroll,
        "z_stratum": int(block["z_stratum"]),
        "local_origin_l1": origin,
        "score_box_local_l1": _score_box(origin),
        "evaluation_context_box_local_l1": context,
        "evaluation_ct_box_global_l0": _global_l0_box(
            context, cfg["label_origin_l1"]
        ),
        "label_stats_score_cube": copy.deepcopy(block["label_stats"]),
        "source_array_file": block["array_file"],
    }


def _file_record(path: Path, logical_path: str) -> dict[str, Any]:
    return {
        "path": logical_path.replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _with_content_hash(plan: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(plan)
    out.pop("content_sha256", None)
    out["content_sha256"] = sha256_bytes(canonical_json(out).encode("ascii"))
    return out


def content_hash_without_field(value: dict[str, Any]) -> str:
    bare = copy.deepcopy(value)
    bare.pop("content_sha256", None)
    return sha256_bytes(canonical_json(bare).encode("ascii"))


def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    expected_hash = plan.get("content_sha256")
    actual_hash = content_hash_without_field(plan)
    if expected_hash != actual_hash:
        errors.append(f"content hash {expected_hash} != {actual_hash}")
    if plan.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("unexpected protocol version")
    if plan.get("status") != PLAN_STATUS:
        errors.append("unexpected plan status")

    allocations = plan.get("cases", {})
    expected_per_role = {
        "train": Z_STRATA * (len(DIFFICULTY_EDGES) + 1) * TRAIN_PER_BIN_PER_STRATUM,
        "internal_validation": (
            Z_STRATA * (len(DIFFICULTY_EDGES) + 1) * VALIDATION_PER_BIN_PER_STRATUM
        ),
        "pilot": Z_STRATA * (len(DIFFICULTY_EDGES) + 1) * PILOT_PER_BIN_PER_STRATUM,
    }
    seen_origins: set[tuple[int, int, int]] = set()
    for role, expected in expected_per_role.items():
        cases = allocations.get(role, [])
        if len(cases) != expected:
            errors.append(f"{role}: {len(cases)} cases, expected {expected}")
        for stratum in range(Z_STRATA):
            for bin_id in range(len(DIFFICULTY_EDGES) + 1):
                count = sum(
                    c["z_stratum"] == stratum and c["difficulty_bin"] == bin_id
                    for c in cases
                )
                per_bin = {
                    "train": TRAIN_PER_BIN_PER_STRATUM,
                    "internal_validation": VALIDATION_PER_BIN_PER_STRATUM,
                    "pilot": PILOT_PER_BIN_PER_STRATUM,
                }[role]
                if count != per_bin:
                    errors.append(
                        f"{role}: stratum {stratum} bin {bin_id} has {count}, expected {per_bin}"
                    )
        for case in cases:
            origin = tuple(map(int, case["local_origin_l1"]))
            if origin in seen_origins:
                errors.append(f"duplicate allocated origin {origin}")
            seen_origins.add(origin)
            if case["role"] != role:
                errors.append(f"{case['case_id']}: role mismatch")
            if not _box_inside(
                case["training_label_box_local_l1"], SCROLLS[TRAIN_SCROLL]["label_shape_l1"]
            ):
                errors.append(f"{case['case_id']}: training label box out of bounds")
            if not _box_inside(
                case["evaluation_context_box_local_l1"],
                SCROLLS[TRAIN_SCROLL]["label_shape_l1"],
            ):
                errors.append(f"{case['case_id']}: evaluation label box out of bounds")
            if not _box_inside(
                case["training_ct_box_global_l0"], SCROLLS[TRAIN_SCROLL]["ct_shape_l0"]
            ):
                errors.append(f"{case['case_id']}: training CT box out of bounds")
            if not _box_inside(
                case["evaluation_ct_box_global_l0"], SCROLLS[TRAIN_SCROLL]["ct_shape_l0"]
            ):
                errors.append(f"{case['case_id']}: evaluation CT box out of bounds")

    primary = allocations.get("primary", {})
    for scroll in (TRAIN_SCROLL, SAFETY_SCROLL):
        cases = primary.get(scroll, [])
        if len(cases) != 32:
            errors.append(f"{scroll}: {len(cases)} primary cases, expected 32")
        origins = [tuple(map(int, c["local_origin_l1"])) for c in cases]
        if len(set(origins)) != len(origins):
            errors.append(f"{scroll}: duplicate primary origin")
        for case in cases:
            if not _box_inside(
                case["evaluation_context_box_local_l1"], SCROLLS[scroll]["label_shape_l1"]
            ):
                errors.append(f"{case['block_id']}: evaluation label box out of bounds")
            if not _box_inside(
                case["evaluation_ct_box_global_l0"], SCROLLS[scroll]["ct_shape_l0"]
            ):
                errors.append(f"{case['block_id']}: evaluation CT box out of bounds")

    train_primary_origins = {
        tuple(map(int, c["local_origin_l1"])) for c in primary.get(TRAIN_SCROLL, [])
    }
    overlap = seen_origins & train_primary_origins
    if overlap:
        errors.append(f"allocated origins overlap primary origins: {sorted(overlap)}")

    for fold_name, fold in plan.get("folds", {}).items():
        train_strata = set(map(int, fold["train_z_strata"]))
        held_strata = set(map(int, fold["held_out_z_strata"]))
        if train_strata & held_strata or train_strata | held_strata != set(range(Z_STRATA)):
            errors.append(f"{fold_name}: invalid z-stratum partition")
        training_boxes = [
            c["training_label_box_local_l1"]
            for role in ("train", "internal_validation")
            for c in allocations[role]
            if c["z_stratum"] in train_strata
        ]
        held_score_boxes = [
            c["score_box_local_l1"]
            for c in allocations["pilot"]
            if c["z_stratum"] in held_strata
        ] + [
            c["score_box_local_l1"]
            for c in primary.get(TRAIN_SCROLL, [])
            if c["z_stratum"] in held_strata
        ]
        if any(boxes_intersect(a, b) for a in training_boxes for b in held_score_boxes):
            errors.append(f"{fold_name}: training labels intersect held-out score cubes")

    if errors:
        raise ValueError("invalid cross-scan plan:\n- " + "\n- ".join(errors))
    return {
        "status": "PASS",
        "content_sha256": actual_hash,
        "train_cases": len(allocations["train"]),
        "internal_validation_cases": len(allocations["internal_validation"]),
        "pilot_cases": len(allocations["pilot"]),
        "primary_cases": sum(len(v) for v in primary.values()),
    }


def build_plan(
    repo: Path, labels_root: Path, source_manifest_path: Path, model_dir: Path
) -> dict[str, Any]:
    public_git = require_public_clean_head(repo)
    source_sha = sha256_file(source_manifest_path)
    if source_sha != SOURCE_MANIFEST_FILE_SHA256:
        raise SystemExit(
            "source manifest whole-file SHA-256 "
            f"{source_sha} != frozen {SOURCE_MANIFEST_FILE_SHA256}"
        )
    source = load_json(source_manifest_path)
    source_content = source.get("content_sha256")
    if source_content != SOURCE_MANIFEST_CONTENT_SHA256:
        raise SystemExit(
            "source manifest recorded content SHA-256 "
            f"{source_content} != frozen {SOURCE_MANIFEST_CONTENT_SHA256}"
        )
    recomputed_source_content = content_hash_without_field(source)
    if recomputed_source_content != source_content:
        raise SystemExit(
            "source manifest content does not recompute: "
            f"{recomputed_source_content} != {source_content}"
        )
    primary: dict[str, list[dict[str, Any]]] = {TRAIN_SCROLL: [], SAFETY_SCROLL: []}
    for block in source["blocks"]:
        if block["scroll"] in primary:
            primary[block["scroll"]].append(_primary_case(block))
    for cases in primary.values():
        cases.sort(key=lambda c: (c["z_stratum"], c["block_id"]))

    primary_origins = {
        tuple(map(int, c["local_origin_l1"])) for c in primary[TRAIN_SCROLL]
    }
    candidates, inventory = scan_candidates(TRAIN_SCROLL, labels_root)
    allocations = allocate_cases(candidates, primary_origins)
    allocations["primary"] = primary

    folds: dict[str, Any] = {}
    for fold_name, definition in FOLDS.items():
        train_strata = set(definition["train_z_strata"])
        held_strata = set(definition["held_out_z_strata"])
        folds[fold_name] = {
            **copy.deepcopy(definition),
            "train_case_ids": [
                c["case_id"] for c in allocations["train"]
                if c["z_stratum"] in train_strata
            ],
            "internal_validation_case_ids": [
                c["case_id"] for c in allocations["internal_validation"]
                if c["z_stratum"] in train_strata
            ],
            "pilot_case_ids": [
                c["case_id"] for c in allocations["pilot"]
                if c["z_stratum"] in held_strata
            ],
            "primary_block_ids": [
                c["block_id"] for c in primary[TRAIN_SCROLL]
                if c["z_stratum"] in held_strata
            ],
        }

    label_records = {}
    for scroll in (TRAIN_SCROLL, SAFETY_SCROLL):
        cfg = SCROLLS[scroll]
        archive = labels_root / cfg["label_archive"]
        store = labels_root / cfg["label_store"]
        label_records[scroll] = {
            "archive": _file_record(archive, cfg["label_archive"]),
            "zarray": _file_record(store / ".zarray", f"{cfg['label_store']}/.zarray"),
            "zattrs": _file_record(store / ".zattrs", f"{cfg['label_store']}/.zattrs"),
            "shape_l1": cfg["label_shape_l1"],
            "origin_l1": cfg["label_origin_l1"],
        }

    required_model_files = ["plans.json", "dataset.json", "fold_0/checkpoint_best.pth"]
    model_records = {
        rel: _file_record(model_dir / rel, rel) for rel in required_model_files
    }
    plan = {
        "schema_version": "vesuvius-crossscan-finetune-plan-v1",
        "protocol_version": PROTOCOL_VERSION,
        "status": PLAN_STATUS,
        "question": (
            "Does fine-tuning surface-m7 on model-independent PHerc1203 recto labels "
            "improve held-out physical-truth average precision across z strata without "
            "regressing on untouched PHerc0139?"
        ),
        "implementation": {
            **public_git,
            "source_manifest": _file_record(
                source_manifest_path, "results/physical_normalization_ab/manifest.json"
            ),
            "planner": _file_record(
                repo / "crossscan_finetune.py", "crossscan_finetune.py"
            ),
            "preregistration": _file_record(
                repo / "CROSSSCAN_FINETUNE_PREREG.md", "CROSSSCAN_FINETUNE_PREREG.md"
            ),
            "tests": _file_record(
                repo / "test_crossscan_finetune.py", "test_crossscan_finetune.py"
            ),
        },
        "inputs": {
            "label_release": (
                "https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit/releases/tag/v1.0"
            ),
            "label_license": "CC BY-NC 4.0",
            "labels": label_records,
            "model": model_records,
            "ct_urls": {
                scroll: (
                    "https://vesuvius-challenge-open-data.s3.amazonaws.com/"
                    f"{scroll}/volumes/{SCROLLS[scroll]['ct_volume']}/0"
                ) for scroll in (TRAIN_SCROLL, SAFETY_SCROLL)
            },
        },
        "target": {
            "encoding": {"background": 0, "recto_surface": 1, "ignore": 2},
            "background_rule": "valid AND NOT material AND NOT recto_band",
            "positive_rule": "valid AND recto_band",
            "ignore_rule": "all remaining voxels",
            "upsampling_l1_to_l0": "nearest-neighbor 2x on each axis",
        },
        "sampling": {
            "seed": SELECTION_SEED,
            "source_candidate_inventory": inventory,
            "score_size_l1": SCORE_SIZE_L1,
            "training_crop_size_l1": TRAIN_CROP_SIZE_L1,
            "evaluation_context_size_l1": EVAL_CONTEXT_SIZE_L1,
            "minimum_valid_fraction": MIN_VALID_FRACTION,
            "minimum_recto_count_score_cube": MIN_RECTO_COUNT_SCORE_CUBE,
            "difficulty_variable": "boundary_poor_fraction_of_material",
            "difficulty_edges": list(DIFFICULTY_EDGES),
            "train_per_bin_per_stratum": TRAIN_PER_BIN_PER_STRATUM,
            "internal_validation_per_bin_per_stratum": VALIDATION_PER_BIN_PER_STRATUM,
            "pilot_per_bin_per_stratum": PILOT_PER_BIN_PER_STRATUM,
        },
        "training": {
            "initial_checkpoint": "surface-m7 fold_0 checkpoint_best.pth",
            "architecture": "m7 nnU-Net 3d_fullres ResidualEncoderUNet",
            "patch_size_l0": [192, 192, 192],
            "batch_size": 1,
            "optimizer": "AdamW",
            "initial_lr": 0.0001,
            "minimum_lr": 0.000001,
            "weight_decay": 0.00001,
            "schedule": "cosine",
            "deep_supervision": True,
            "pilot_seed": PILOT_SEED,
            "inferential_seeds": INFERENTIAL_SEEDS,
            "pilot_steps": PILOT_STEPS,
            "single_permitted_retry_steps": PILOT_RETRY_STEPS,
            "inferential_steps": PILOT_RETRY_STEPS,
            "checkpoint_for_scoring": "final epoch only",
        },
        "evaluation": {
            "primary_scroll": TRAIN_SCROLL,
            "safety_scroll": SAFETY_SCROLL,
            "primary_endpoint": "pooled supervised-voxel average precision",
            "pilot_ap_gate": PILOT_AP_GATE,
            "pilot_max_allowed_z_stratum_regression": 0.005,
            "primary_ap_effect_gate": PRIMARY_AP_EFFECT_GATE,
            "safety_ap_noninferiority_margin": SAFETY_AP_MARGIN,
            "seed_is_inferential_unit": True,
            "minimum_same_sign_seeds": 5,
            "paired_test": "two-sided one-sample t test over six seed deltas, alpha 0.05",
            "secondary": [
                "matched-initial-positive-mass recto recall, precision, and Dice",
                "fixed-threshold 0.2 recto recall, precision, and Dice",
                "results by z stratum and boundary-poor difficulty bin",
                "hash-selected baseline/fine-tuned/truth slice panels",
            ],
        },
        "folds": folds,
        "cases": allocations,
    }
    plan = _with_content_hash(plan)
    validate_plan(plan)
    return plan


def verify_external_inputs(
    plan: dict[str, Any], repo: Path, labels_root: Path,
    source_manifest: Path, model_dir: Path
) -> dict[str, Any]:
    result = validate_plan(plan)
    errors: list[str] = []
    records: list[tuple[dict[str, Any], Path]] = [
        (plan["implementation"]["source_manifest"], source_manifest),
        (plan["implementation"]["planner"], repo / "crossscan_finetune.py"),
        (
            plan["implementation"]["preregistration"],
            repo / "CROSSSCAN_FINETUNE_PREREG.md",
        ),
        (plan["implementation"]["tests"], repo / "test_crossscan_finetune.py"),
    ]
    records.extend(
        (record, model_dir / rel)
        for rel, record in plan["inputs"]["model"].items()
    )
    for label in plan["inputs"]["labels"].values():
        records.extend(
            (label[key], labels_root / label[key]["path"])
            for key in ("archive", "zarray", "zattrs")
        )
    for record, path in records:
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        if path.stat().st_size != record["bytes"]:
            errors.append(f"size changed for {path}")
            continue
        digest = sha256_file(path)
        if digest != record["sha256"]:
            errors.append(f"SHA-256 changed for {path}: {digest}")
    if errors:
        raise ValueError("input verification failed:\n- " + "\n- ".join(errors))
    result["external_inputs"] = "PASS"
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="build the outcome-blind machine plan")
    plan.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    plan.add_argument("--labels-root", type=Path, required=True)
    plan.add_argument("--source-manifest", type=Path, required=True)
    plan.add_argument("--model-dir", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify", help="verify a frozen plan and all local inputs")
    verify.add_argument("plan", type=Path)
    verify.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    verify.add_argument("--labels-root", type=Path, required=True)
    verify.add_argument("--source-manifest", type=Path, required=True)
    verify.add_argument("--model-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "plan":
        built = build_plan(
            args.repo.resolve(), args.labels_root.resolve(),
            args.source_manifest.resolve(), args.model_dir.resolve()
        )
        write_json(args.out, built)
        print(canonical_json(validate_plan(built)))
    elif args.command == "verify":
        print(canonical_json(verify_external_inputs(
            load_json(args.plan), args.repo.resolve(), args.labels_root.resolve(),
            args.source_manifest.resolve(), args.model_dir.resolve()
        )))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
