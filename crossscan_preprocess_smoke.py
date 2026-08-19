#!/usr/bin/env python3
"""Run a real one-case nnU-Net fingerprint and preprocessing integration smoke."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import crossscan_finetune as C
import run_crossscan_finetune as R


SMOKE_FINGERPRINT_VOXEL_BUDGET = 4096


def synthetic_plan() -> dict:
    case = {
        "case_id": "synthetic-preprocess-smoke",
        "scroll": C.TRAIN_SCROLL,
        "role": "train",
        "z_stratum": 0,
        "difficulty_bin": 0,
        "training_label_box_local_l1": [0, 96, 0, 96, 0, 96],
        "training_ct_box_global_l0": [0, 192, 0, 192, 0, 192],
    }
    return {
        "content_sha256": "synthetic-preprocess-smoke-plan-v1",
        "inputs": {"ct_urls": {C.TRAIN_SCROLL: "synthetic://ct"}},
        "cases": {"train": [case], "internal_validation": []},
        "folds": {
            "even": {
                "train_case_ids": [case["case_id"]],
                "internal_validation_case_ids": [],
            },
            "odd": {
                "train_case_ids": [case["case_id"]],
                "internal_validation_case_ids": [],
            },
        },
    }


def git_value(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def run_smoke(repo: Path, villa_root: Path, model_dir: Path) -> dict:
    public = R.require_clean_public_head(repo)
    villa_status = git_value(villa_root, "status", "--porcelain=v1")
    if villa_status:
        raise SystemExit("smoke requires a clean villa worktree:\n" + villa_status)
    model_plans = model_dir / "plans.json"
    if not model_plans.is_file():
        raise FileNotFoundError(f"missing model plans: {model_plans}")
    plan = synthetic_plan()
    lock = {"content_sha256": "synthetic-preprocess-smoke-lock-v1"}
    with tempfile.TemporaryDirectory(prefix="crossscan-preprocess-smoke-") as tmp:
        data_root = Path(tmp).resolve()
        labels_root = data_root / "truth"
        label_store = labels_root / C.SCROLLS[C.TRAIN_SCROLL]["label_store"]
        import zarr
        store = zarr.open(
            str(label_store), mode="w", shape=(96, 96, 96),
            chunks=(32, 32, 32), dtype="u1", zarr_format=2,
        )
        truth_bits = np.ones((96, 96, 96), dtype=np.uint8)
        truth_bits[:, 40:56, :] = 11
        store[:] = truth_bits
        ramp = (np.arange(192, dtype=np.uint16) + 1).astype(np.uint8)
        ct = np.broadcast_to(ramp[:, None, None], (192, 192, 192)).copy()

        R.seed_dataset_plans(plan, model_dir, data_root)
        original_open = R.open_remote_zarr
        original_budget = R.FINGERPRINT_FOREGROUND_VOXEL_BUDGET
        try:
            R.open_remote_zarr = lambda _url: ct
            R.materialize_training(plan, lock, labels_root, data_root, 1)
            R.FINGERPRINT_FOREGROUND_VOXEL_BUDGET = (
                SMOKE_FINGERPRINT_VOXEL_BUDGET
            )
            preprocessing = R.preprocess_dataset(
                plan, lock, data_root, num_processes=1
            )
        finally:
            R.open_remote_zarr = original_open
            R.FINGERPRINT_FOREGROUND_VOXEL_BUDGET = original_budget

        fingerprint_path, _, fingerprint_receipt_path = R.fingerprint_artifact_paths(
            data_root
        )
        fingerprint = C.load_json(fingerprint_path)
        summary = R.validate_dataset_fingerprint(fingerprint, 1)
        fingerprint_receipt = R.load_content_hashed(
            fingerprint_receipt_path, "PASS"
        )
        folder = R.preprocessed_dataset_folder(data_root)
        case_id = plan["cases"]["train"][0]["case_id"]
        data_file = folder / f"{case_id}.b2nd"
        properties_file = folder / f"{case_id}.pkl"
        if not data_file.is_file() or not properties_file.is_file():
            raise RuntimeError("real nnU-Net preprocessor did not emit both case files")

        trainer_copy = data_root / "trainer-start-copy" / "dataset_fingerprint.json"
        trainer_copy.parent.mkdir(parents=True)
        shutil.copy(fingerprint_path, trainer_copy)
        if R.file_record(trainer_copy) != R.file_record(fingerprint_path):
            raise RuntimeError("trainer-start fingerprint copy changed bytes")

        return R._with_content_hash({
            "schema_version": "crossscan-preprocess-smoke-v1",
            "status": "PASS",
            "created_utc": R.utc_now(),
            "implementation": public,
            "villa": {
                "commit": git_value(villa_root, "rev-parse", "HEAD"),
                "nnunet_tree": git_value(
                    villa_root, "rev-parse",
                    "HEAD:segmentation/models/arch/nnunet",
                ),
            },
            "environment": R.runtime_environment(villa_root),
            "model_plans": R.file_record(model_plans),
            "synthetic_inputs": {
                "ct_array_sha256": R.array_sha256(ct),
                "truth_bits_array_sha256": R.array_sha256(truth_bits),
            },
            "fingerprint_foreground_voxel_budget": (
                SMOKE_FINGERPRINT_VOXEL_BUDGET
            ),
            "fingerprint_summary": summary,
            "fingerprint_file": R.file_record(fingerprint_path),
            "fingerprint_receipt_content_sha256": fingerprint_receipt[
                "content_sha256"
            ],
            "preprocessing_receipt_content_sha256": preprocessing[
                "content_sha256"
            ],
            "preprocessed_case": {
                "data": R.file_record(data_file),
                "properties": R.file_record(properties_file),
            },
            "trainer_start_copy": R.file_record(trainer_copy),
            "assertions": {
                "official_fingerprint_extractor_completed": True,
                "official_default_preprocessor_completed": True,
                "fingerprint_is_receipted": True,
                "trainer_start_fingerprint_copy_completed": True,
            },
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = Path(__file__).resolve().parent
    parser.add_argument("--repo", type=Path, default=repo)
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite smoke result: {out}")
    result = run_smoke(
        args.repo.resolve(), args.villa_root.resolve(), args.model_dir.resolve()
    )
    C.write_json(out, result)
    print(C.canonical_json(result))


if __name__ == "__main__":
    main()
