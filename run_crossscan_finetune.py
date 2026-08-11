#!/usr/bin/env python3
"""Fail-closed runtime for the cross-scan physical-truth fine-tuning protocol.

The runtime has five stages: freeze, materialize, preprocess, train, and infer. It never
scores an outcome. The separate scorer writes the pilot decision and final verdict.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

import crossscan_finetune as C


EXECUTION_LOCK_STATUS = "public_preoutcome_execution_lock"
DATASET_ID = 913
DATASET_NAME = "Dataset913_CrossScanPhysical"
PLANS_IDENTIFIER = "nnUNetResEncUNetLPlans"
CONFIGURATION = "3d_fullres"
ITERATIONS_PER_EPOCH = 40
VALIDATION_ITERATIONS_PER_EPOCH = 10
TRAINER_NAME = "CrossScanPhysicalTrainer"
INITIAL_LR = 0.0001
WEIGHT_DECAY = 0.00001
MINIMUM_LR = 0.000001
FINGERPRINT_FOREGROUND_VOXEL_BUDGET = 100_000_000
FINGERPRINT_EXTRACTOR = (
    "nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor."
    "DatasetFingerprintExtractor"
)
LOCKED_IMPLEMENTATION_FILES = (
    "CROSSSCAN_FINETUNE_AMENDMENT_01.md",
    "CROSSSCAN_FINETUNE_AMENDMENT_02.md",
    "CROSSSCAN_FINETUNE_AMENDMENT_03.md",
    "CROSSSCAN_FINETUNE_AMENDMENT_04.md",
    "CROSSSCAN_FINETUNE_AMENDMENT_05.md",
    "CROSSSCAN_FINETUNE_AMENDMENT_06.md",
    "CROSSSCAN_FINETUNE_AMENDMENT_07.md",
    "CROSSSCAN_FINETUNE_PREREG.md",
    "CROSSSCAN_FINETUNE_RUNBOOK.md",
    "crossscan_training_memory_smoke.py",
    "crossscan_preprocess_smoke.py",
    "crossscan_finetune.py",
    "run_crossscan_finetune.py",
    "score_crossscan_finetune.py",
    "test_crossscan_finetune.py",
    "test_run_crossscan_finetune.py",
    "test_score_crossscan_finetune.py",
    "results/crossscan_finetune/execution_lock.superseded-20260811-pretraining.json",
    "results/crossscan_finetune/execution_lock.superseded-20260811-pretraining-v2.json",
    "results/crossscan_finetune/execution_lock.superseded-20260811-preverdict-step-selection-v3.json",
    "results/crossscan_finetune/execution_lock.withdrawn-20260811-release-attributes.json",
    "results/crossscan_finetune/execution_lock.withdrawn-20260811-ci-portability.json",
    "results/crossscan_finetune/preprocess_smoke.json",
    "results/crossscan_finetune/plan.json",
)

TRUSTED_CHECKPOINT_STATIC_UNSAFE_GLOBALS = (
    "numpy._core.multiarray.scalar",
    "numpy.dtype",
)


def coerce_locked_training_hyperparameters(trainer: Any) -> dict[str, float]:
    """Validate and normalize values after nnU-Net's JSON-via-PyYAML boundary.

    PyYAML can resolve JSON scientific notation such as ``1e-05`` as text. The
    values are therefore converted only after checking that they are finite and
    exactly equal to the preregistered numeric hyperparameters.
    """
    expected = {
        "initial_lr": INITIAL_LR,
        "weight_decay": WEIGHT_DECAY,
    }
    normalized: dict[str, float] = {}
    for name, locked in expected.items():
        raw = getattr(trainer, name)
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError(f"{name} must be numeric, not boolean")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} is not numeric: {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {raw!r}")
        if value != locked:
            raise ValueError(f"{name} {value!r} != locked value {locked!r}")
        setattr(trainer, name, value)
        normalized[name] = value
    return normalized


def trusted_checkpoint_safe_types() -> tuple[type, ...]:
    """Return the narrow allowlist required by the frozen released checkpoint."""
    return (
        np.dtype,
        np._core.multiarray.scalar,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    )


def trusted_checkpoint_load_record(expected: dict[str, Any]) -> dict[str, Any]:
    safe_types = trusted_checkpoint_safe_types()
    return {
        "mode": "torch-default-weights-only-with-safe-globals",
        "static_unsafe_globals": list(TRUSTED_CHECKPOINT_STATIC_UNSAFE_GLOBALS),
        "allowlisted_types": [f"{value.__module__}.{value.__name__}" for value in safe_types],
        "checkpoint": {
            "bytes": int(expected["bytes"]),
            "sha256": str(expected["sha256"]),
        },
    }


def load_frozen_pretrained_weights(
    network: Any,
    checkpoint: Path,
    expected: dict[str, Any],
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Hash-bind and narrowly allowlist the trusted released nnU-Net checkpoint.

    Villa's pinned loader calls ``torch.load`` without a ``weights_only`` argument.
    The locked PyTorch runtime therefore uses its restricted default. The checkpoint
    contains two statically discoverable NumPy globals and three constructed dtype
    classes; no other pickle global is accepted.
    """
    expected_file = {
        "bytes": int(expected["bytes"]),
        "sha256": str(expected["sha256"]),
    }
    actual = file_record(checkpoint)
    if actual != expected_file:
        raise ValueError(f"frozen checkpoint identity mismatch: {actual} != {expected_file}")

    import torch
    from nnunetv2.run.load_pretrained_weights import load_pretrained_weights

    unsafe = tuple(sorted(torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)))
    if unsafe != TRUSTED_CHECKPOINT_STATIC_UNSAFE_GLOBALS:
        raise ValueError(
            "frozen checkpoint unsafe globals changed: "
            f"{unsafe} != {TRUSTED_CHECKPOINT_STATIC_UNSAFE_GLOBALS}"
        )
    with torch.serialization.safe_globals(trusted_checkpoint_safe_types()):
        load_pretrained_weights(network, str(checkpoint), verbose=verbose)
    return trusted_checkpoint_load_record(expected)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_output(root: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=not binary,
        encoding=None if binary else "utf-8",
    ).strip()


def normalized_text_bytes(value: bytes) -> bytes:
    return value.replace(b"\r\n", b"\n").rstrip(b"\n")


def require_clean_public_head(repo: Path) -> dict[str, str]:
    status = str(git_output(repo, "status", "--porcelain=v1"))
    if status:
        raise SystemExit("runtime requires a clean worktree:\n" + status)
    head = str(git_output(repo, "rev-parse", "HEAD"))
    branch = str(git_output(repo, "branch", "--show-current"))
    if not branch:
        raise SystemExit("runtime refuses a detached HEAD")
    line = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo, text=True, encoding="utf-8",
    ).strip()
    remote = line.split()[0] if line else ""
    if head != remote:
        raise SystemExit(f"public head missing: origin/{branch}={remote or '<absent>'}, HEAD={head}")
    return {"commit": head, "branch": branch, "remote_commit": remote}


def _with_content_hash(value: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(value)
    out.pop("content_sha256", None)
    out["content_sha256"] = C.sha256_bytes(C.canonical_json(out).encode("ascii"))
    return out


def frozen_visual_cases(plan: dict[str, Any]) -> list[dict[str, Any]]:
    selected = []
    for scroll in (C.TRAIN_SCROLL, C.SAFETY_SCROLL):
        cases = plan["cases"]["primary"][scroll]
        for stratum in range(C.Z_STRATA):
            pool = [c for c in cases if int(c["z_stratum"]) == stratum]
            ranked = sorted(
                pool,
                key=lambda c: C.sha256_bytes(
                    f"crossscan-visual-v1|{c['block_id']}".encode("ascii")
                ),
            )
            if not ranked:
                raise ValueError(f"no visual case for {scroll} stratum {stratum}")
            selected.append({
                "scroll": scroll,
                "z_stratum": stratum,
                "case_id": ranked[0]["block_id"],
                "score_slice_l1": 32,
            })
    return selected


def runtime_environment(villa_root: Path) -> dict[str, str]:
    import blosc2
    import nnunetv2
    import scipy
    import tifffile
    import torch
    import yaml
    import zarr

    module = Path(nnunetv2.__file__).resolve()
    expected_root = (villa_root / "segmentation/models/arch/nnunet").resolve()
    try:
        relative_module = module.relative_to(expected_root).as_posix()
    except ValueError as exc:
        raise SystemExit(
            f"imported nnunetv2 is outside the locked villa tree: {module}"
        ) from exc
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "tifffile": tifffile.__version__,
        "pyyaml": yaml.__version__,
        "zarr": zarr.__version__,
        "blosc2": getattr(blosc2, "__version__", "unknown"),
        "nnunetv2_module": relative_module,
    }


def load_content_hashed(path: Path, expected_status: str | None = None) -> dict[str, Any]:
    value = C.load_json(path)
    recorded = value.get("content_sha256")
    actual = C.content_hash_without_field(value)
    if recorded != actual:
        raise SystemExit(f"{path}: content SHA-256 {recorded} != {actual}")
    if expected_status is not None and value.get("status") != expected_status:
        raise SystemExit(f"{path}: status {value.get('status')} != {expected_status}")
    return value


def require_tracked_bytes(repo: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit(f"locked file must be inside repository: {path}") from exc
    try:
        git_output(repo, "ls-files", "--error-unmatch", rel)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"locked file is not tracked: {rel}") from exc
    committed = bytes(git_output(repo, "show", f"HEAD:{rel}", binary=True))
    if normalized_text_bytes(committed) != normalized_text_bytes(path.read_bytes()):
        raise SystemExit(f"working bytes differ from HEAD: {rel}")
    return rel


def build_execution_lock(repo: Path, plan_path: Path, villa_root: Path) -> dict[str, Any]:
    public = require_clean_public_head(repo)
    plan = load_content_hashed(plan_path, C.PLAN_STATUS)
    C.validate_plan(plan)
    if not villa_root.is_dir():
        raise SystemExit(f"missing villa root: {villa_root}")
    villa_status = str(git_output(villa_root, "status", "--porcelain=v1"))
    if villa_status:
        raise SystemExit("villa worktree must be clean:\n" + villa_status)
    files = {}
    for rel in LOCKED_IMPLEMENTATION_FILES:
        path = repo / rel
        if not path.is_file():
            raise SystemExit(f"missing execution file: {path}")
        files[rel] = {"bytes": path.stat().st_size, "sha256": C.sha256_file(path)}
    lock = {
        "schema_version": "vesuvius-crossscan-execution-lock-v1",
        "status": EXECUTION_LOCK_STATUS,
        "created_utc": utc_now(),
        "implementation": {
            **public,
            "repo": "https://github.com/TAUIL-Abd-Elilah/vesuvius-repro",
            "files": files,
        },
        "plan": {
            "path": require_tracked_bytes(repo, plan_path),
            "bytes": plan_path.stat().st_size,
            "file_sha256": C.sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
        },
        "villa": {
            "commit": str(git_output(villa_root, "rev-parse", "HEAD")),
            "nnunet_tree": str(
                git_output(villa_root, "rev-parse", "HEAD:segmentation/models/arch/nnunet")
            ),
        },
        "environment": runtime_environment(villa_root),
        "resolved_protocol": {
            "dataset_id": DATASET_ID,
            "dataset_name": DATASET_NAME,
            "plans_identifier": PLANS_IDENTIFIER,
            "configuration": CONFIGURATION,
            "iterations_per_epoch": ITERATIONS_PER_EPOCH,
            "validation_iterations_per_epoch": VALIDATION_ITERATIONS_PER_EPOCH,
            "foreground_oversampling": "probabilistic 0.5 per batch item",
            "trainer_name": TRAINER_NAME,
            "inference_tile_step_size": 0.5,
            "inference_gaussian": True,
            "inference_mirroring": False,
            "l0_to_l1_reduction": "2x2x2 maximum",
            "matched_mass_ties": (
                "stable descending probability, then plan case order, then C-order voxel index"
            ),
            "visual_cases": frozen_visual_cases(plan),
        },
    }
    return _with_content_hash(lock)


def verify_execution_lock(repo: Path, lock_path: Path, plan_path: Path,
                          villa_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    public = require_clean_public_head(repo)
    lock = load_content_hashed(lock_path, EXECUTION_LOCK_STATUS)
    plan = load_content_hashed(plan_path, C.PLAN_STATUS)
    C.validate_plan(plan)
    require_tracked_bytes(repo, lock_path)
    require_tracked_bytes(repo, plan_path)
    implementation_commit = lock["implementation"]["commit"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, public["commit"]],
        cwd=repo,
    )
    if ancestor.returncode != 0:
        raise SystemExit(f"locked implementation {implementation_commit} is not an ancestor of HEAD")
    for rel, record in lock["implementation"]["files"].items():
        path = repo / rel
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise SystemExit(f"locked implementation file missing or resized: {rel}")
        if C.sha256_file(path) != record["sha256"]:
            raise SystemExit(f"locked implementation file changed: {rel}")
    if C.sha256_file(plan_path) != lock["plan"]["file_sha256"]:
        raise SystemExit("plan whole-file SHA-256 changed")
    if plan["content_sha256"] != lock["plan"]["content_sha256"]:
        raise SystemExit("plan content SHA-256 changed")
    if str(git_output(villa_root, "status", "--porcelain=v1")):
        raise SystemExit("villa worktree is dirty")
    villa_commit = str(git_output(villa_root, "rev-parse", "HEAD"))
    villa_tree = str(git_output(
        villa_root, "rev-parse", "HEAD:segmentation/models/arch/nnunet"
    ))
    if villa_commit != lock["villa"]["commit"] or villa_tree != lock["villa"]["nnunet_tree"]:
        raise SystemExit("villa/nnU-Net identity changed")
    environment = runtime_environment(villa_root)
    if environment != lock.get("environment"):
        raise SystemExit(
            "runtime environment changed:\n"
            f"locked={C.canonical_json(lock.get('environment'))}\n"
            f"actual={C.canonical_json(environment)}"
        )
    return lock, plan


def verify_runtime(
    repo: Path, lock_path: Path, plan_path: Path, villa_root: Path,
    labels_root: Path, source_manifest: Path, model_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock, plan = verify_execution_lock(repo, lock_path, plan_path, villa_root)
    C.verify_external_inputs(plan, repo, labels_root, source_manifest, model_dir)
    return lock, plan


def retry(fn: Callable[[], Any], attempts: int = 8) -> Any:
    delay = 1.0
    for attempt in range(attempts):
        try:
            return fn()
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise AssertionError("unreachable")


def open_remote_zarr(url: str):
    import zarr
    return retry(lambda: zarr.open(url, mode="r"))


def read_box(array: Any, box: Iterable[int], attempts: int = 8) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = map(int, box)
    expected = (z1 - z0, y1 - y0, x1 - x0)
    value = retry(
        lambda: np.asarray(array[z0:z1, y0:y1, x0:x1]), attempts=attempts
    )
    if value.shape != expected:
        raise RuntimeError(f"read returned {value.shape}, expected {expected}")
    return value


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return C.sha256_bytes(array.view(np.uint8).tobytes())


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    C.write_json(tmp, value)
    tmp.replace(path)


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    tmp.replace(path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def file_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": C.sha256_file(path)}


def relative_data_path(data_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside data root: {path}") from exc


def resolve_data_path(data_root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError(f"expected a data-root-relative path, got {relative}")
    resolved = (data_root / value).resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes data root: {relative}") from exc
    return resolved


def _case_map(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for role in ("train", "internal_validation", "pilot"):
        for case in plan["cases"][role]:
            out[case["case_id"]] = case
    for cases in plan["cases"]["primary"].values():
        for case in cases:
            out[case["block_id"]] = case
    return out


def training_cases(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [*plan["cases"]["train"], *plan["cases"]["internal_validation"]]


def evaluation_cases(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *plan["cases"]["pilot"],
        *plan["cases"]["primary"][C.TRAIN_SCROLL],
        *plan["cases"]["primary"][C.SAFETY_SCROLL],
    ]


def case_identifier(case: dict[str, Any]) -> str:
    return str(case.get("case_id", case.get("block_id")))


def validate_ct(array: np.ndarray, expected_shape: tuple[int, int, int]) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != expected_shape:
        raise ValueError(f"CT shape {value.shape} != {expected_shape}")
    if value.dtype.kind not in "ui" or value.size == 0:
        raise ValueError(f"CT must be nonempty integer data, got {value.dtype}")
    minimum, maximum = int(value.min()), int(value.max())
    if minimum < 0 or maximum > 255:
        raise ValueError(f"CT range [{minimum},{maximum}] is outside uint8")
    return value.astype(np.uint8, copy=False)


def verify_evaluation_case(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path, case_id: str,
    verify_ct: bool = True, verify_truth: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    cases = _case_map(plan)
    if case_id not in cases or case_id not in {
        case_identifier(case) for case in evaluation_cases(plan)
    }:
        raise ValueError(f"{case_id}: not a frozen evaluation case")
    case = cases[case_id]
    ct_path = data_root / "evaluation" / case_id / "ct_l0.npy"
    truth_path = data_root / "evaluation" / case_id / "truth_bits_l1.npy"
    receipt_path = (
        data_root / "materialization" / "evaluation_receipts" / f"{case_id}.json"
    )
    if not all(path.is_file() for path in (ct_path, truth_path, receipt_path)):
        raise FileNotFoundError(f"missing evaluation materialization for {case_id}")
    receipt = load_content_hashed(receipt_path, "PASS")
    expected = {
        "schema_version": "crossscan-evaluation-case-v1",
        "case_id": case_id,
        "scroll": case["scroll"],
        "role": case.get("role", "primary"),
        "z_stratum": case["z_stratum"],
        "ct_box_global_l0": case["evaluation_ct_box_global_l0"],
        "score_box_local_l1": case["score_box_local_l1"],
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
    }
    mismatches = [
        key for key, value in expected.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"{case_id}: evaluation receipt mismatch: {mismatches}")
    ct = None
    if verify_ct:
        if file_record(ct_path) != receipt.get("ct_file"):
            raise ValueError(f"{case_id}: materialized CT file hash mismatch")
        ct = validate_ct(np.load(ct_path, allow_pickle=False), (256, 256, 256))
        if array_sha256(ct) != receipt.get("ct_array_sha256"):
            raise ValueError(f"{case_id}: materialized CT array hash mismatch")
    truth = None
    if verify_truth:
        if file_record(truth_path) != receipt.get("truth_file"):
            raise ValueError(f"{case_id}: materialized truth file hash mismatch")
        truth = np.load(truth_path, allow_pickle=False)
        if truth.shape != (64, 64, 64) or truth.dtype.kind not in "ui":
            raise ValueError(f"{case_id}: invalid truth array {truth.shape} {truth.dtype}")
        if array_sha256(truth) != receipt.get("truth_array_sha256"):
            raise ValueError(f"{case_id}: materialized truth array hash mismatch")
    return ct, truth, receipt


def materialize_training(
    plan: dict[str, Any], lock: dict[str, Any], labels_root: Path,
    data_root: Path, read_attempts: int,
) -> dict[str, Any]:
    import tifffile
    import zarr

    raw_dataset = data_root / "nnUNet_raw" / DATASET_NAME
    images = raw_dataset / "imagesTr"
    labels_out = raw_dataset / "labelsTr"
    receipts = data_root / "materialization" / "training_receipts"
    for path in (images, labels_out, receipts):
        path.mkdir(parents=True, exist_ok=True)
    label_store = zarr.open(
        str(labels_root / C.SCROLLS[C.TRAIN_SCROLL]["label_store"]), mode="r"
    )
    ct = open_remote_zarr(plan["inputs"]["ct_urls"][C.TRAIN_SCROLL])
    completed = 0
    for index, case in enumerate(training_cases(plan), 1):
        case_id = case["case_id"]
        image_path = images / f"{case_id}_0000.tif"
        label_path = labels_out / f"{case_id}.tif"
        receipt_path = receipts / f"{case_id}.json"
        existing = [path.is_file() for path in (receipt_path, image_path, label_path)]
        if any(existing):
            if not all(existing):
                raise RuntimeError(f"partial training materialization: {case_id}")
            receipt = load_content_hashed(receipt_path, "PASS")
            expected = {
                "schema_version": "crossscan-training-case-v1",
                "case_id": case_id,
                "role": case["role"],
                "z_stratum": case["z_stratum"],
                "difficulty_bin": case["difficulty_bin"],
                "ct_box_global_l0": case["training_ct_box_global_l0"],
                "label_box_local_l1": case["training_label_box_local_l1"],
                "plan_content_sha256": plan["content_sha256"],
                "execution_lock_content_sha256": lock["content_sha256"],
            }
            if (all(receipt.get(key) == value for key, value in expected.items())
                    and file_record(image_path) == receipt.get("image_file")
                    and file_record(label_path) == receipt.get("label_file")):
                completed += 1
                continue
            raise RuntimeError(f"stale materialization receipt: {case_id}")
        label_l1 = read_box(label_store, case["training_label_box_local_l1"], read_attempts)
        target_l0 = C.upsample_target_l0(C.physical_target_l1(label_l1))
        counts = {str(v): int((target_l0 == v).sum()) for v in (0, 1, 2)}
        if counts["0"] == 0 or counts["1"] == 0:
            raise RuntimeError(f"{case_id}: target lacks background or recto support: {counts}")
        ct_l0 = validate_ct(
            read_box(ct, case["training_ct_box_global_l0"], read_attempts),
            (192, 192, 192),
        )
        image_tmp = image_path.with_name(image_path.stem + ".tmp.tif")
        label_tmp = label_path.with_name(label_path.stem + ".tmp.tif")
        tifffile.imwrite(image_tmp, ct_l0, compression="zlib", compressionargs={"level": 1})
        tifffile.imwrite(label_tmp, target_l0, compression="zlib", compressionargs={"level": 1})
        image_tmp.replace(image_path)
        label_tmp.replace(label_path)
        receipt = {
            "schema_version": "crossscan-training-case-v1",
            "status": "PASS",
            "created_utc": utc_now(),
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
            "case_id": case_id,
            "role": case["role"],
            "z_stratum": case["z_stratum"],
            "difficulty_bin": case["difficulty_bin"],
            "ct_box_global_l0": case["training_ct_box_global_l0"],
            "label_box_local_l1": case["training_label_box_local_l1"],
            "ct_array_sha256": array_sha256(ct_l0),
            "target_array_sha256": array_sha256(target_l0),
            "target_counts": counts,
            "image_file": file_record(image_path),
            "label_file": file_record(label_path),
        }
        atomic_write_json(receipt_path, _with_content_hash(receipt))
        completed += 1
        print(f"[{index}/{len(training_cases(plan))}] {case_id}", flush=True)
    write_dataset_configuration(plan, data_root)
    return {"training_cases": completed, "status": "PASS"}


def _truth_score_box(case: dict[str, Any], labels_root: Path) -> np.ndarray:
    import zarr
    scroll = case["scroll"]
    store = zarr.open(str(labels_root / C.SCROLLS[scroll]["label_store"]), mode="r")
    return read_box(store, case["score_box_local_l1"])


def materialize_evaluation(
    plan: dict[str, Any], lock: dict[str, Any], labels_root: Path,
    data_root: Path, read_attempts: int,
) -> dict[str, Any]:
    base = data_root / "evaluation"
    receipts = data_root / "materialization" / "evaluation_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    remotes: dict[str, Any] = {}
    completed = 0
    cases = evaluation_cases(plan)
    for index, case in enumerate(cases, 1):
        case_id = case_identifier(case)
        case_dir = base / case_id
        ct_path = case_dir / "ct_l0.npy"
        truth_path = case_dir / "truth_bits_l1.npy"
        receipt_path = receipts / f"{case_id}.json"
        existing = [path.is_file() for path in (receipt_path, ct_path, truth_path)]
        if any(existing):
            if not all(existing):
                raise RuntimeError(f"partial evaluation materialization: {case_id}")
            verify_evaluation_case(plan, lock, data_root, case_id)
            completed += 1
            continue
        scroll = case["scroll"]
        if scroll not in remotes:
            remotes[scroll] = open_remote_zarr(plan["inputs"]["ct_urls"][scroll])
        ct_l0 = validate_ct(
            read_box(remotes[scroll], case["evaluation_ct_box_global_l0"], read_attempts),
            (256, 256, 256),
        )
        truth = _truth_score_box(case, labels_root).astype(np.uint8, copy=False)
        if truth.shape != (64, 64, 64):
            raise RuntimeError(f"{case_id}: truth shape {truth.shape}")
        atomic_save_npy(ct_path, ct_l0)
        atomic_save_npy(truth_path, truth)
        receipt = {
            "schema_version": "crossscan-evaluation-case-v1",
            "status": "PASS",
            "created_utc": utc_now(),
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
            "case_id": case_id,
            "scroll": scroll,
            "role": case.get("role", "primary"),
            "z_stratum": case["z_stratum"],
            "ct_box_global_l0": case["evaluation_ct_box_global_l0"],
            "score_box_local_l1": case["score_box_local_l1"],
            "ct_array_sha256": array_sha256(ct_l0),
            "truth_array_sha256": array_sha256(truth),
            "ct_file": file_record(ct_path),
            "truth_file": file_record(truth_path),
        }
        atomic_write_json(receipt_path, _with_content_hash(receipt))
        completed += 1
        print(f"[{index}/{len(cases)}] {case_id}", flush=True)
    return {"evaluation_cases": completed, "status": "PASS"}


def write_dataset_configuration(plan: dict[str, Any], data_root: Path) -> None:
    raw = data_root / "nnUNet_raw" / DATASET_NAME
    pre = data_root / "nnUNet_preprocessed" / DATASET_NAME
    pre.mkdir(parents=True, exist_ok=True)
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "surface": 1, "ignore": 2},
        "numTraining": len(training_cases(plan)),
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "Tiff3DIO",
    }
    # The logical model path is resolved by the caller's model directory during materialization.
    # Store a marker here; materialize_command replaces it with the verified source plans.
    if not (pre / f"{PLANS_IDENTIFIER}.json").is_file():
        raise RuntimeError(
            f"missing copied model plans in {pre}; materialize_command must seed them first"
        )
    C.write_json(raw / "dataset.json", dataset)
    C.write_json(pre / "dataset.json", dataset)
    splits = []
    for fold_name in ("even", "odd"):
        fold = plan["folds"][fold_name]
        splits.append({
            "train": fold["train_case_ids"],
            "val": fold["internal_validation_case_ids"],
        })
    C.write_json(pre / "splits_final.json", splits)


def seed_dataset_plans(plan: dict[str, Any], model_dir: Path, data_root: Path) -> None:
    pre = data_root / "nnUNet_preprocessed" / DATASET_NAME
    pre.mkdir(parents=True, exist_ok=True)
    plans = C.load_json(model_dir / "plans.json")
    plans["dataset_name"] = DATASET_NAME
    plans["plans_name"] = PLANS_IDENTIFIER
    plans["image_reader_writer"] = "Tiff3DIO"
    plans["configurations"][CONFIGURATION]["batch_size"] = 1
    C.write_json(pre / f"{PLANS_IDENTIFIER}.json", plans)


def verify_materialized_training(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
) -> None:
    raw = data_root / "nnUNet_raw" / DATASET_NAME
    receipts = data_root / "materialization" / "training_receipts"
    errors = []
    for case in training_cases(plan):
        case_id = case["case_id"]
        image = raw / "imagesTr" / f"{case_id}_0000.tif"
        label = raw / "labelsTr" / f"{case_id}.tif"
        receipt_path = receipts / f"{case_id}.json"
        if not all(p.is_file() for p in (image, label, receipt_path)):
            errors.append(f"missing files for {case_id}")
            continue
        receipt = load_content_hashed(receipt_path, "PASS")
        expected = {
            "schema_version": "crossscan-training-case-v1",
            "case_id": case_id,
            "role": case["role"],
            "z_stratum": case["z_stratum"],
            "difficulty_bin": case["difficulty_bin"],
            "ct_box_global_l0": case["training_ct_box_global_l0"],
            "label_box_local_l1": case["training_label_box_local_l1"],
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
        }
        mismatches = [
            key for key, value in expected.items() if receipt.get(key) != value
        ]
        if mismatches:
            errors.append(f"receipt mismatch for {case_id}: {mismatches}")
        if file_record(image) != receipt.get("image_file"):
            errors.append(f"image hash changed for {case_id}")
        if file_record(label) != receipt.get("label_file"):
            errors.append(f"label hash changed for {case_id}")
    if errors:
        raise RuntimeError("training materialization invalid:\n- " + "\n- ".join(errors))


def training_receipts_content_sha256(plan: dict[str, Any], data_root: Path) -> str:
    receipts = data_root / "materialization" / "training_receipts"
    records = []
    for case in training_cases(plan):
        receipt = load_content_hashed(receipts / f"{case['case_id']}.json", "PASS")
        records.append({
            "case_id": case["case_id"],
            "content_sha256": receipt["content_sha256"],
        })
    return C.sha256_bytes(C.canonical_json(records).encode("ascii"))


def validate_dataset_fingerprint(
    fingerprint: dict[str, Any], case_count: int,
) -> dict[str, Any]:
    if case_count < 1:
        raise ValueError("fingerprint requires at least one training case")
    required = {
        "spacings",
        "shapes_after_crop",
        "foreground_intensity_properties_per_channel",
        "median_relative_size_after_cropping",
    }
    missing = sorted(required - set(fingerprint))
    if missing:
        raise ValueError(f"dataset fingerprint missing keys: {missing}")
    spacings = fingerprint["spacings"]
    shapes = fingerprint["shapes_after_crop"]
    if len(spacings) != case_count or len(shapes) != case_count:
        raise ValueError(
            "dataset fingerprint case count mismatch: "
            f"spacings={len(spacings)}, shapes={len(shapes)}, expected={case_count}"
        )
    for name, values, require_integer in (
        ("spacing", spacings, False),
        ("shape", shapes, True),
    ):
        for index, triplet in enumerate(values):
            if not isinstance(triplet, (list, tuple)) or len(triplet) != 3:
                raise ValueError(f"fingerprint {name} {index} is not a triplet")
            for value in triplet:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"fingerprint {name} {index} is nonnumeric")
                number = float(value)
                if not np.isfinite(number) or number <= 0:
                    raise ValueError(f"fingerprint {name} {index} is not positive finite")
                if require_integer and number != int(number):
                    raise ValueError(f"fingerprint shape {index} is not integral")
    relative = fingerprint["median_relative_size_after_cropping"]
    if (isinstance(relative, bool) or not isinstance(relative, (int, float))
            or not np.isfinite(float(relative)) or not 0 < float(relative) <= 1):
        raise ValueError("invalid median_relative_size_after_cropping")
    properties = fingerprint["foreground_intensity_properties_per_channel"]
    if not isinstance(properties, dict):
        raise ValueError("fingerprint intensity properties are not a mapping")
    channel = properties.get("0", properties.get(0))
    if not isinstance(channel, dict):
        raise ValueError("fingerprint is missing CT channel 0 statistics")
    statistic_names = (
        "mean", "median", "std", "min", "max",
        "percentile_99_5", "percentile_00_5",
    )
    for name in statistic_names:
        value = channel.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(float(value))):
            raise ValueError(f"invalid fingerprint CT statistic: {name}")
    if float(channel["std"]) < 0 or float(channel["min"]) > float(channel["max"]):
        raise ValueError("inconsistent fingerprint CT statistics")
    unique_shapes = sorted({tuple(int(v) for v in shape) for shape in shapes})
    unique_spacings = sorted({tuple(float(v) for v in spacing) for spacing in spacings})
    return {
        "case_count": case_count,
        "median_relative_size_after_cropping": float(relative),
        "unique_shapes_after_crop": [list(v) for v in unique_shapes],
        "unique_spacings": [list(v) for v in unique_spacings],
    }


def fingerprint_artifact_paths(data_root: Path) -> tuple[Path, Path, Path]:
    base = data_root / "nnUNet_preprocessed" / DATASET_NAME
    return (
        base / "dataset_fingerprint.json",
        base / "fingerprint_generation_intent.json",
        base / "fingerprint_generation_receipt.json",
    )


def fingerprint_intent_fields(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
    num_processes: int,
) -> dict[str, Any]:
    case_count = len(training_cases(plan))
    if case_count < 1:
        raise ValueError("cannot fingerprint an empty training dataset")
    return {
        "schema_version": "crossscan-fingerprint-intent-v1",
        "status": "LOCKED",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "dataset_name": DATASET_NAME,
        "case_count": case_count,
        "extractor": FINGERPRINT_EXTRACTOR,
        "extractor_seed_per_case": 1234,
        "foreground_voxel_budget": FINGERPRINT_FOREGROUND_VOXEL_BUDGET,
        "foreground_samples_per_case": (
            FINGERPRINT_FOREGROUND_VOXEL_BUDGET // case_count
        ),
        "num_processes": num_processes,
        "training_receipts_content_sha256": training_receipts_content_sha256(
            plan, data_root
        ),
        "plans_policy": (
            "generate the standard fingerprint for dataset provenance; retain the "
            "frozen pretrained m7 plans without replanning"
        ),
    }


def verify_dataset_fingerprint(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
) -> dict[str, Any]:
    fingerprint_path, intent_path, receipt_path = fingerprint_artifact_paths(data_root)
    for path in (fingerprint_path, intent_path, receipt_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing fingerprint artifact: {path}")
    intent = load_content_hashed(intent_path, "LOCKED")
    num_processes = intent.get("num_processes")
    if isinstance(num_processes, bool) or not isinstance(num_processes, int) or num_processes < 1:
        raise ValueError("fingerprint intent has invalid num_processes")
    expected_intent = fingerprint_intent_fields(
        plan, lock, data_root, num_processes
    )
    mismatches = [
        key for key, value in expected_intent.items() if intent.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"fingerprint intent mismatch: {mismatches}")
    fingerprint = C.load_json(fingerprint_path)
    summary = validate_dataset_fingerprint(fingerprint, len(training_cases(plan)))
    receipt = load_content_hashed(receipt_path, "PASS")
    expected_receipt = {
        "schema_version": "crossscan-fingerprint-receipt-v1",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "dataset_name": DATASET_NAME,
        "case_count": len(training_cases(plan)),
        "intent_content_sha256": intent["content_sha256"],
        "training_receipts_content_sha256": intent[
            "training_receipts_content_sha256"
        ],
        "fingerprint_summary": summary,
    }
    mismatches = [
        key for key, value in expected_receipt.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"fingerprint receipt mismatch: {mismatches}")
    if receipt.get("fingerprint_file") != file_record(fingerprint_path):
        raise ValueError("dataset fingerprint file differs from its receipt")
    return receipt


def generate_dataset_fingerprint(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
    num_processes: int,
) -> dict[str, Any]:
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    fingerprint_path, intent_path, receipt_path = fingerprint_artifact_paths(data_root)
    if receipt_path.is_file():
        return verify_dataset_fingerprint(plan, lock, data_root)
    if fingerprint_path.is_file() and not intent_path.is_file():
        raise RuntimeError(
            "unreceipted dataset fingerprint exists without a generation intent"
        )
    expected_intent = fingerprint_intent_fields(
        plan, lock, data_root, num_processes
    )
    if intent_path.is_file():
        intent = load_content_hashed(intent_path, "LOCKED")
        mismatches = [
            key for key, value in expected_intent.items() if intent.get(key) != value
        ]
        if mismatches:
            raise ValueError(f"fingerprint generation intent mismatch: {mismatches}")
    else:
        intent = _with_content_hash({
            **expected_intent,
            "created_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
        })
        atomic_write_json(intent_path, intent)
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import (
        DatasetFingerprintExtractor,
    )
    extractor = DatasetFingerprintExtractor(
        DATASET_ID, num_processes=num_processes, verbose=False
    )
    extractor.num_foreground_voxels_for_intensitystats = (
        FINGERPRINT_FOREGROUND_VOXEL_BUDGET
    )
    extractor.run(overwrite_existing=True)
    summary = validate_dataset_fingerprint(
        C.load_json(fingerprint_path), len(training_cases(plan))
    )
    receipt = _with_content_hash({
        "schema_version": "crossscan-fingerprint-receipt-v1",
        "status": "PASS",
        "created_utc": utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "dataset_name": DATASET_NAME,
        "case_count": len(training_cases(plan)),
        "intent_content_sha256": intent["content_sha256"],
        "training_receipts_content_sha256": intent[
            "training_receipts_content_sha256"
        ],
        "fingerprint_summary": summary,
        "fingerprint_file": file_record(fingerprint_path),
    })
    atomic_write_json(receipt_path, receipt)
    return verify_dataset_fingerprint(plan, lock, data_root)


def preprocessed_dataset_folder(data_root: Path) -> Path:
    base = data_root / "nnUNet_preprocessed" / DATASET_NAME
    plans = C.load_json(base / f"{PLANS_IDENTIFIER}.json")
    identifier = plans["configurations"][CONFIGURATION]["data_identifier"]
    return base / str(identifier)


def preprocessed_file_records(data_root: Path) -> dict[str, dict[str, Any]]:
    base = data_root / "nnUNet_preprocessed" / DATASET_NAME
    return {
        relative_data_path(data_root, path): file_record(path)
        for path in sorted(base.rglob("*")) if path.is_file()
    }


def verify_preprocessed(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
) -> dict[str, Any]:
    receipt_path = data_root / "preprocessing_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing preprocessing receipt: {receipt_path}")
    fingerprint_receipt = verify_dataset_fingerprint(plan, lock, data_root)
    receipt = load_content_hashed(receipt_path, "PASS")
    expected = {
        "schema_version": "crossscan-preprocessing-v2",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "dataset_name": DATASET_NAME,
        "configuration": CONFIGURATION,
        "case_count": len(training_cases(plan)),
        "fingerprint_receipt_content_sha256": fingerprint_receipt[
            "content_sha256"
        ],
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise ValueError(f"preprocessing receipt mismatch: {mismatches}")
    actual = preprocessed_file_records(data_root)
    if actual != receipt.get("files"):
        missing = sorted(set(receipt.get("files", {})) - set(actual))
        extra = sorted(set(actual) - set(receipt.get("files", {})))
        changed = sorted(
            key for key in set(actual) & set(receipt.get("files", {}))
            if actual[key] != receipt["files"][key]
        )
        raise ValueError(
            "preprocessed files differ from receipt: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    return receipt


def preprocess_dataset(plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
                       num_processes: int) -> dict[str, Any]:
    receipt_path = data_root / "preprocessing_receipt.json"
    if receipt_path.is_file():
        return verify_preprocessed(plan, lock, data_root)
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    verify_materialized_training(plan, lock, data_root)
    os.environ["nnUNet_raw"] = str(data_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(data_root / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(data_root / "nnUNet_results_unused")
    fingerprint_receipt = generate_dataset_fingerprint(
        plan, lock, data_root, num_processes
    )
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess_dataset as run
    run(
        DATASET_ID,
        plans_identifier=PLANS_IDENTIFIER,
        configurations=[CONFIGURATION],
        num_processes=[num_processes],
        verbose=False,
    )
    folder = preprocessed_dataset_folder(data_root)
    cases = training_cases(plan)
    missing = [
        c["case_id"] for c in cases
        if not (folder / f"{c['case_id']}.b2nd").is_file()
        or not (folder / f"{c['case_id']}.pkl").is_file()
    ]
    if missing:
        raise RuntimeError(f"preprocessing missing {len(missing)} cases, first={missing[:5]}")
    receipt = {
        "schema_version": "crossscan-preprocessing-v2",
        "status": "PASS",
        "created_utc": utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "dataset_name": DATASET_NAME,
        "configuration": CONFIGURATION,
        "case_count": len(cases),
        "fingerprint_receipt_content_sha256": fingerprint_receipt[
            "content_sha256"
        ],
        "output_folder": relative_data_path(data_root, folder),
        "files": preprocessed_file_records(data_root),
    }
    receipt = _with_content_hash(receipt)
    atomic_write_json(receipt_path, receipt)
    return verify_preprocessed(plan, lock, data_root)


def set_deterministic_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["nnUNet_n_proc_DA"] = "0"
    os.environ["nnUNet_compile"] = "0"
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def fold_index(name: str) -> int:
    if name not in ("even", "odd"):
        raise ValueError(f"unknown fold {name}")
    return 0 if name == "even" else 1


def require_pilot_authorization(
    data_root: Path, steps: int, plan_content_sha256: str | None = None,
    lock_content_sha256: str | None = None,
) -> dict[str, Any]:
    verdict_path = data_root / "pilot_verdict.json"
    if not verdict_path.is_file():
        raise SystemExit("inferential runs require a public-protocol pilot_verdict.json")
    verdict = load_content_hashed(verdict_path, "PASS")
    selected_steps = int(verdict.get("selected_steps", -1))
    if selected_steps != C.PILOT_RETRY_STEPS:
        raise SystemExit(
            "passing pilot verdict must authorize the frozen 4,000-step "
            "inferential recipe"
        )
    if selected_steps != int(steps):
        raise SystemExit(
            f"pilot selected {verdict.get('selected_steps')} steps, requested {steps}"
        )
    if (plan_content_sha256 is not None
            and verdict.get("plan_content_sha256") != plan_content_sha256):
        raise SystemExit("pilot verdict belongs to a different plan")
    if (lock_content_sha256 is not None
            and verdict.get("execution_lock_content_sha256") != lock_content_sha256):
        raise SystemExit("pilot verdict belongs to a different execution lock")
    return verdict


def require_any_pilot_authorization(
    data_root: Path, plan_content_sha256: str | None = None,
    lock_content_sha256: str | None = None,
) -> dict[str, Any]:
    verdict_path = data_root / "pilot_verdict.json"
    if not verdict_path.is_file():
        raise SystemExit("primary and safety inference require a passing pilot verdict")
    verdict = load_content_hashed(verdict_path, "PASS")
    if int(verdict.get("selected_steps", -1)) != C.PILOT_RETRY_STEPS:
        raise SystemExit(
            "passing pilot verdict must authorize the frozen 4,000-step "
            "inferential recipe"
        )
    if (plan_content_sha256 is not None
            and verdict.get("plan_content_sha256") != plan_content_sha256):
        raise SystemExit("pilot verdict belongs to a different plan")
    if (lock_content_sha256 is not None
            and verdict.get("execution_lock_content_sha256") != lock_content_sha256):
        raise SystemExit("pilot verdict belongs to a different execution lock")
    return verdict


def training_output_folder(data_root: Path, seed: int, steps: int, fold: str) -> Path:
    base = data_root / "training" / f"steps-{steps}" / f"seed-{seed}" / fold
    return (
        base / DATASET_NAME
        / f"{TRAINER_NAME}__{PLANS_IDENTIFIER}__{CONFIGURATION}"
        / f"fold_{fold_index(fold)}"
    )


def load_training_receipt(
    plan: dict[str, Any], lock: dict[str, Any], data_root: Path,
    seed: int, steps: int, fold: str,
) -> tuple[dict[str, Any], Path]:
    path = data_root / "training_receipts" / f"seed-{seed}-{fold}-steps-{steps}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing training receipt: {path}")
    receipt = load_content_hashed(path, "PASS")
    preprocessing = load_content_hashed(
        data_root / "preprocessing_receipt.json", "PASS"
    )
    expected_checkpoint = training_output_folder(
        data_root, seed, steps, fold
    ) / "checkpoint_final.pth"
    expected_config = (
        data_root / "training" / f"steps-{steps}" / f"seed-{seed}" / fold
        / "training_config.json"
    )
    expected = {
        "schema_version": "crossscan-training-run-v1",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "seed": seed,
        "fold": fold,
        "steps": steps,
        "epochs": steps // ITERATIONS_PER_EPOCH,
        "trainer_hyperparameters": {
            "initial_lr": INITIAL_LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "preprocessing_receipt_content_sha256": preprocessing["content_sha256"],
        "initial_checkpoint_sha256": plan["inputs"]["model"][
            "fold_0/checkpoint_best.pth"
        ]["sha256"],
        "initial_checkpoint_load": trusted_checkpoint_load_record(
            plan["inputs"]["model"]["fold_0/checkpoint_best.pth"]
        ),
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise ValueError(f"training receipt mismatch: {mismatches}")
    checkpoint_record = receipt.get("checkpoint", {})
    expected_relative = relative_data_path(data_root, expected_checkpoint)
    if checkpoint_record.get("relative_path") != expected_relative:
        raise ValueError(
            "training checkpoint path mismatch: "
            f"{checkpoint_record.get('relative_path')} != {expected_relative}"
        )
    checkpoint = resolve_data_path(data_root, checkpoint_record.get("relative_path", ""))
    if not checkpoint.is_file() or file_record(checkpoint) != {
        "bytes": checkpoint_record.get("bytes"),
        "sha256": checkpoint_record.get("sha256"),
    }:
        raise ValueError(f"checkpoint does not match training receipt: {checkpoint}")
    config_record = receipt.get("training_config", {})
    expected_config_relative = relative_data_path(data_root, expected_config)
    if config_record.get("relative_path") != expected_config_relative:
        raise ValueError("training configuration path mismatch")
    config = resolve_data_path(data_root, config_record.get("relative_path", ""))
    if not config.is_file() or file_record(config) != {
        "bytes": config_record.get("bytes"),
        "sha256": config_record.get("sha256"),
    }:
        raise ValueError(f"training configuration does not match receipt: {config}")
    return receipt, checkpoint


def train_model(
    plan: dict[str, Any], lock: dict[str, Any], model_dir: Path,
    data_root: Path, seed: int, steps: int, fold: str,
) -> dict[str, Any]:
    if steps not in (C.PILOT_STEPS, C.PILOT_RETRY_STEPS):
        raise SystemExit(f"steps must be {C.PILOT_STEPS} or {C.PILOT_RETRY_STEPS}")
    if seed == C.PILOT_SEED:
        if steps == C.PILOT_RETRY_STEPS:
            prior_path = data_root / f"pilot_attempt_steps-{C.PILOT_STEPS}.json"
            if not prior_path.is_file():
                raise SystemExit("4,000-step pilot training requires the 2,000-step attempt")
            prior = load_content_hashed(prior_path, "RETRY_REQUIRED")
            if prior.get("decision") != "RETRY_REQUIRED":
                raise SystemExit("4,000-step pilot training is not authorized")
            if (prior.get("plan_content_sha256") != plan["content_sha256"]
                    or prior.get("execution_lock_content_sha256") != lock["content_sha256"]):
                raise SystemExit("2,000-step pilot attempt belongs to a different lock")
    elif seed in C.INFERENTIAL_SEEDS:
        require_pilot_authorization(
            data_root, steps, plan["content_sha256"], lock["content_sha256"]
        )
    else:
        raise SystemExit(f"seed {seed} is not in the frozen protocol")
    if steps % ITERATIONS_PER_EPOCH:
        raise SystemExit("steps must be divisible by iterations per epoch")
    receipt_path = data_root / "training_receipts" / f"seed-{seed}-{fold}-steps-{steps}.json"
    results_root = data_root / "training" / f"steps-{steps}" / f"seed-{seed}" / fold
    output = training_output_folder(data_root, seed, steps, fold)
    checkpoint = output / "checkpoint_final.pth"
    if receipt_path.exists() or results_root.exists():
        raise SystemExit(
            f"refusing to overwrite existing training run: {receipt_path} or {results_root}"
        )
    preprocessed = data_root / "nnUNet_preprocessed"
    preprocessing = verify_preprocessed(plan, lock, data_root)
    os.environ["nnUNet_raw"] = str(data_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(preprocessed)
    os.environ["nnUNet_results"] = str(results_root)
    set_deterministic_seed(seed)
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    class CrossScanPhysicalTrainer(nnUNetTrainer):
        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.network.parameters(), lr=self.initial_lr,
                weight_decay=self.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(self.num_epochs - 1, 1), eta_min=MINIMUM_LR
            )
            return optimizer, scheduler

        def get_dataloaders(self):
            train_loader, val_loader = super().get_dataloaders()
            # Batch size one makes nnU-Net's positional "last N percent" rule collapse
            # to either 0% or 100%. Single-thread augmentation exposes the underlying
            # loaders, so use their supported probabilistic rule to realize 50% exactly.
            for augmenter in (train_loader, val_loader):
                loader = augmenter.data_loader
                loader.get_do_oversample = loader._probabilistic_oversampling
            return train_loader, val_loader

        def on_epoch_end(self):
            # Keep all metrics and best-EMA tracking, but do not rewrite an 800 MB best
            # checkpoint on every improvement. The protocol scores final epoch only.
            original = self.disable_checkpointing
            self.disable_checkpointing = True
            try:
                super().on_epoch_end()
            finally:
                self.disable_checkpointing = original

    CrossScanPhysicalTrainer.__name__ = TRAINER_NAME
    epochs = steps // ITERATIONS_PER_EPOCH
    config_path = results_root / "training_config.json"
    C.write_json(config_path, {
        "project_name": "crossscan-physical-finetune",
        "dataset": DATASET_NAME,
        "wandb_enabled": 0,
        "num_epochs": epochs,
        "initial_lr": INITIAL_LR,
        "weight_decay": WEIGHT_DECAY,
        "num_iterations_per_epoch": ITERATIONS_PER_EPOCH,
        "num_val_iterations_per_epoch": VALIDATION_ITERATIONS_PER_EPOCH,
        "oversample_foreground_percent": 0.5,
        "enable_deep_supervision": True,
    })
    plans = C.load_json(preprocessed / DATASET_NAME / f"{PLANS_IDENTIFIER}.json")
    dataset = C.load_json(preprocessed / DATASET_NAME / "dataset.json")
    trainer = CrossScanPhysicalTrainer(
        plans=plans,
        configuration=CONFIGURATION,
        fold=fold_index(fold),
        dataset_json=dataset,
        unpack_dataset=False,
        device=torch.device("cuda"),
        yaml_config_path=str(config_path),
    )
    trainer_hyperparameters = coerce_locked_training_hyperparameters(trainer)
    trainer.initialize()
    initial_checkpoint_load = load_frozen_pretrained_weights(
        trainer.network,
        model_dir / "fold_0" / "checkpoint_best.pth",
        plan["inputs"]["model"]["fold_0/checkpoint_best.pth"],
        verbose=True,
    )
    started = utc_now()
    trainer.run_training()
    if not checkpoint.is_file():
        raise RuntimeError(f"training ended without final checkpoint: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if int(saved.get("current_epoch", -1)) != epochs:
        raise RuntimeError(f"checkpoint epoch {saved.get('current_epoch')} != {epochs}")
    if saved.get("trainer_name") != TRAINER_NAME:
        raise RuntimeError(
            f"checkpoint trainer {saved.get('trainer_name')} != {TRAINER_NAME}"
        )
    receipt = {
        "schema_version": "crossscan-training-run-v1",
        "status": "PASS",
        "started_utc": started,
        "completed_utc": utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "seed": seed,
        "fold": fold,
        "steps": steps,
        "epochs": epochs,
        "trainer_hyperparameters": trainer_hyperparameters,
        "command": [sys.executable, *sys.argv],
        "preprocessing_receipt_content_sha256": preprocessing["content_sha256"],
        "training_config": {
            "relative_path": relative_data_path(data_root, config_path),
            **file_record(config_path),
        },
        "checkpoint": {
            "relative_path": relative_data_path(data_root, checkpoint),
            **file_record(checkpoint),
        },
        "initial_checkpoint_sha256": plan["inputs"]["model"]["fold_0/checkpoint_best.pth"]["sha256"],
        "initial_checkpoint_load": initial_checkpoint_load,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    receipt = _with_content_hash(receipt)
    atomic_write_json(receipt_path, receipt)
    return receipt


def max_pool_l0_to_l1(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim != 3 or any(d % 2 for d in value.shape):
        raise ValueError(f"expected even 3D L0 array, got {value.shape}")
    z, y, x = value.shape
    return value.reshape(z // 2, 2, y // 2, 2, x // 2, 2).max(axis=(1, 3, 5))


def normalize_ct(ct: np.ndarray, plans: dict[str, Any]) -> np.ndarray:
    props = plans["foreground_intensity_properties_per_channel"]["0"]
    value = np.asarray(ct, dtype=np.float32).copy()
    np.clip(value, props["percentile_00_5"], props["percentile_99_5"], out=value)
    value -= float(props["mean"])
    value /= max(float(props["std"]), 1e-8)
    return value


def select_inference_cases(plan: dict[str, Any], scope: str, fold: str | None) -> list[str]:
    if scope == "pilot":
        if fold is None:
            return [c["case_id"] for c in plan["cases"]["pilot"]]
        return list(plan["folds"][fold]["pilot_case_ids"])
    if scope == "primary":
        if fold is None:
            return [c["block_id"] for c in plan["cases"]["primary"][C.TRAIN_SCROLL]]
        return list(plan["folds"][fold]["primary_block_ids"])
    if scope == "safety":
        return [c["block_id"] for c in plan["cases"]["primary"][C.SAFETY_SCROLL]]
    raise ValueError(scope)


def prediction_root(data_root: Path, kind: str, scope: str, seed: int | None,
                    steps: int | None, fold: str | None) -> Path:
    if kind == "initial":
        return data_root / "predictions" / "initial" / scope
    assert seed is not None and steps is not None and fold is not None
    return (
        data_root / "predictions" / f"steps-{steps}" / f"seed-{seed}" / fold / scope
    )


def load_network_for_inference(checkpoint: Path, model_dir: Path):
    import gc
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans = C.load_json(model_dir / "plans.json")
    dataset = C.load_json(model_dir / "dataset.json")
    manager = PlansManager(plans)
    config = manager.get_configuration(CONFIGURATION)
    label_manager = manager.get_label_manager(dataset)
    network = nnUNetTrainer.build_network_architecture(
        config.network_arch_class_name,
        config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import,
        determine_num_input_channels(manager, config, dataset),
        label_manager.num_segmentation_heads,
        enable_deep_supervision=False,
    )
    saved = torch.load(
        checkpoint, map_location="cpu", weights_only=False, mmap=True
    )
    state = saved["network_weights"]
    network.load_state_dict(state, strict=True)
    del state, saved
    gc.collect()
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device("cuda"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.manual_initialization(
        network, manager, config, parameters=None, dataset_json=dataset,
        trainer_name="nnUNetTrainer", inference_allowed_mirroring_axes=None,
    )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return predictor, plans


def load_verified_prediction(
    array_path: Path, receipt_path: Path, expected_metadata: dict[str, Any],
) -> np.ndarray:
    if not array_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing prediction output: {array_path}")
    receipt = load_content_hashed(receipt_path, "PASS")
    if receipt.get("schema_version") != "crossscan-inference-case-v1":
        raise ValueError("prediction receipt schema mismatch")
    mismatches = [
        key for key, value in expected_metadata.items() if receipt.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"prediction receipt mismatch: {mismatches}")
    if file_record(array_path) != receipt.get("array_file"):
        raise ValueError(f"prediction file hash mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as payload:
        if set(payload.files) != {"probability_l1", "metadata_json"}:
            raise ValueError(f"unexpected prediction NPZ keys: {payload.files}")
        raw = payload["probability_l1"]
        if raw.dtype != np.dtype(np.float32):
            raise ValueError(f"prediction dtype {raw.dtype} != float32")
        probability = np.asarray(raw)
        metadata_text = str(payload["metadata_json"].item())
    metadata = json.loads(metadata_text)
    if metadata != expected_metadata:
        raise ValueError("embedded prediction metadata mismatch")
    if probability.shape != (64, 64, 64) or not np.isfinite(probability).all():
        raise ValueError(f"invalid probability array: {probability.shape}")
    if float(probability.min()) < 0.0 or float(probability.max()) > 1.0:
        raise ValueError("prediction probability outside [0,1]")
    if (float(probability.min()) != receipt.get("probability_min")
            or float(probability.max()) != receipt.get("probability_max")):
        raise ValueError("prediction probability range mismatch")
    if array_sha256(probability) != receipt.get("probability_array_sha256"):
        raise ValueError("prediction array hash mismatch")
    return probability


def infer_cases(
    plan: dict[str, Any], lock: dict[str, Any], model_dir: Path, data_root: Path,
    kind: str, scope: str, seed: int | None, steps: int | None, fold: str | None,
) -> dict[str, Any]:
    import torch

    if scope in ("primary", "safety"):
        require_any_pilot_authorization(
            data_root, plan["content_sha256"], lock["content_sha256"]
        )
    if kind == "initial":
        if any(v is not None for v in (seed, steps, fold)):
            raise SystemExit("initial inference does not take seed, steps, or fold")
        checkpoint = model_dir / "fold_0" / "checkpoint_best.pth"
        expected_initial = plan["inputs"]["model"]["fold_0/checkpoint_best.pth"]
        if file_record(checkpoint) != {
            "bytes": expected_initial["bytes"],
            "sha256": expected_initial["sha256"],
        }:
            raise SystemExit("initial checkpoint does not match the frozen plan")
    elif kind == "finetuned":
        if seed is None or steps is None or fold is None:
            raise SystemExit("fine-tuned inference requires seed, steps, and fold")
        if scope == "pilot":
            if seed != C.PILOT_SEED:
                raise SystemExit("pilot scope requires pilot seed")
        else:
            if seed not in C.INFERENTIAL_SEEDS:
                raise SystemExit("primary/safety scope requires an inferential seed")
            require_pilot_authorization(
                data_root, steps, plan["content_sha256"], lock["content_sha256"]
            )
        _, checkpoint = load_training_receipt(
            plan, lock, data_root, seed, steps, fold
        )
    else:
        raise SystemExit(f"unknown inference kind {kind}")
    output = prediction_root(data_root, kind, scope, seed, steps, fold)
    output.mkdir(parents=True, exist_ok=True)
    case_ids = select_inference_cases(plan, scope, fold if kind == "finetuned" else None)
    checkpoint_sha256 = C.sha256_file(checkpoint)
    predictor, plans = load_network_for_inference(checkpoint, model_dir)
    completed = 0
    for index, case_id in enumerate(case_ids, 1):
        ct, _, _ = verify_evaluation_case(plan, lock, data_root, case_id)
        assert ct is not None
        array_path = output / f"{case_id}.npz"
        receipt_path = output / f"{case_id}.json"
        metadata = {
            "case_id": case_id,
            "kind": kind,
            "scope": scope,
            "seed": seed,
            "steps": steps,
            "fold": fold,
            "checkpoint_sha256": checkpoint_sha256,
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
        }
        if array_path.is_file() or receipt_path.is_file():
            if not (array_path.is_file() and receipt_path.is_file()):
                raise RuntimeError(f"partial inference output: {case_id}")
            load_verified_prediction(array_path, receipt_path, metadata)
            completed += 1
            continue
        normalized = normalize_ct(ct, plans)
        tensor = torch.from_numpy(normalized[None])
        logits = predictor.predict_sliding_window_return_logits(tensor).float().cpu()
        if tuple(logits.shape) != (2, 256, 256, 256):
            raise RuntimeError(f"{case_id}: logits shape {tuple(logits.shape)}")
        probability = torch.softmax(logits, dim=0)[1].numpy()
        central = probability[64:192, 64:192, 64:192]
        p_l1 = max_pool_l0_to_l1(central).astype(np.float32, copy=False)
        if p_l1.shape != (64, 64, 64) or not np.isfinite(p_l1).all():
            raise RuntimeError(f"{case_id}: invalid pooled probability")
        atomic_save_npz(
            array_path,
            probability_l1=p_l1,
            metadata_json=np.asarray(C.canonical_json(metadata)),
        )
        inference_receipt = {
            "schema_version": "crossscan-inference-case-v1",
            "status": "PASS",
            "created_utc": utc_now(),
            "command": [sys.executable, *sys.argv],
            **metadata,
            "probability_array_sha256": array_sha256(p_l1),
            "probability_min": float(p_l1.min()),
            "probability_max": float(p_l1.max()),
            "array_file": file_record(array_path),
        }
        atomic_write_json(receipt_path, _with_content_hash(inference_receipt))
        completed += 1
        print(f"[{index}/{len(case_ids)}] {case_id}", flush=True)
    return {"status": "PASS", "completed": completed, "expected": len(case_ids)}


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    repo = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--lock", type=Path, default=repo / "results/crossscan_finetune/execution_lock.json")
    parser.add_argument("--plan", type=Path, default=repo / "results/crossscan_finetune/plan.json")
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=repo / "results/physical_normalization_ab/manifest.json")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--villa-root", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    materialize = sub.add_parser("materialize")
    add_runtime_args(materialize)
    materialize.add_argument("--scope", choices=("training", "evaluation", "all"), default="all")
    materialize.add_argument("--read-attempts", type=int, default=8)
    preprocess = sub.add_parser("preprocess")
    add_runtime_args(preprocess)
    preprocess.add_argument("--num-processes", type=int, default=4)
    train = sub.add_parser("train")
    add_runtime_args(train)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--fold", choices=("even", "odd"), required=True)
    infer = sub.add_parser("infer")
    add_runtime_args(infer)
    infer.add_argument("--kind", choices=("initial", "finetuned"), required=True)
    infer.add_argument("--scope", choices=("pilot", "primary", "safety"), required=True)
    infer.add_argument("--seed", type=int)
    infer.add_argument("--steps", type=int)
    infer.add_argument("--fold", choices=("even", "odd"))
    return parser


def _resolved(value: Path) -> Path:
    return value.resolve()


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "freeze":
        if args.out.exists():
            raise SystemExit(f"refusing to overwrite execution lock: {args.out}")
        lock = build_execution_lock(_resolved(args.repo), _resolved(args.plan), _resolved(args.villa_root))
        C.write_json(args.out, lock)
        print(C.canonical_json({"status": "PASS", "content_sha256": lock["content_sha256"]}))
        return
    repo = _resolved(args.repo)
    lock, plan = verify_runtime(
        repo, _resolved(args.lock), _resolved(args.plan), _resolved(args.villa_root),
        _resolved(args.labels_root), _resolved(args.source_manifest), _resolved(args.model_dir),
    )
    data_root = _resolved(args.data_root)
    if args.command == "materialize":
        if args.read_attempts < 1:
            raise SystemExit("read-attempts must be positive")
        seed_dataset_plans(plan, _resolved(args.model_dir), data_root)
        results = {}
        if args.scope in ("training", "all"):
            results["training"] = materialize_training(
                plan, lock, _resolved(args.labels_root), data_root, args.read_attempts
            )
        if args.scope in ("evaluation", "all"):
            results["evaluation"] = materialize_evaluation(
                plan, lock, _resolved(args.labels_root), data_root, args.read_attempts
            )
        print(C.canonical_json(results))
    elif args.command == "preprocess":
        print(C.canonical_json(preprocess_dataset(
            plan, lock, data_root, args.num_processes
        )))
    elif args.command == "train":
        print(C.canonical_json(train_model(
            plan, lock, _resolved(args.model_dir), data_root,
            args.seed, args.steps, args.fold,
        )))
    elif args.command == "infer":
        print(C.canonical_json(infer_cases(
            plan, lock, _resolved(args.model_dir), data_root,
            args.kind, args.scope, args.seed, args.steps, args.fold,
        )))
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
