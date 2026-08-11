#!/usr/bin/env python3
"""Build a verified, optimizer-free release of a completed cross-scan run."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import crossscan_finetune as C
import run_crossscan_finetune as R


RELEASE_SCHEMA = "crossscan-model-release-v1"
FINAL_TRAINER = "CrossScanPhysicalTrainer"
INFERENCE_TRAINER = "nnUNetTrainer"


def load_hashed(path: Path) -> dict[str, Any]:
    value = C.load_json(path)
    if value.get("content_sha256") != C.content_hash_without_field(value):
        raise ValueError(f"content hash mismatch: {path}")
    return value


def relative_record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {"path": relative, **R.file_record(path)}


def copy_artifact(source: Path, staging: Path, relative: str) -> dict[str, Any]:
    destination = (staging / relative).resolve()
    destination.relative_to(staging.resolve())
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if R.file_record(destination) != R.file_record(source):
        raise RuntimeError(f"copied artifact changed: {source}")
    return relative_record(staging, destination)


def validate_network_state(state: object) -> dict[str, int]:
    import torch

    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint network_weights is not a nonempty mapping")
    tensors = 0
    elements = 0
    unique_storages: dict[tuple[int, int], int] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("network state must map strings to tensors")
        if not torch.isfinite(value).all():
            raise ValueError(f"network tensor is nonfinite: {key}")
        tensors += 1
        elements += int(value.numel())
        storage = value.untyped_storage()
        unique_storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return {
        "state_tensor_count": tensors,
        "state_logical_elements": elements,
        "state_unique_storage_bytes": int(sum(unique_storages.values())),
    }


def export_checkpoint(
    source: Path, destination: Path, release_metadata: dict[str, Any]
) -> dict[str, Any]:
    import torch

    if destination.exists():
        raise FileExistsError(destination)
    source_record = R.file_record(source)
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("trainer_name") != FINAL_TRAINER:
        raise ValueError(
            f"source trainer {saved.get('trainer_name')} != {FINAL_TRAINER}"
        )
    init_args = saved.get("init_args")
    if not isinstance(init_args, dict) or init_args.get("configuration") != R.CONFIGURATION:
        raise ValueError("source checkpoint configuration mismatch")
    state = saved.get("network_weights")
    state_summary = validate_network_state(state)
    exported = {
        "network_weights": state,
        "current_epoch": saved.get("current_epoch"),
        "init_args": {"configuration": R.CONFIGURATION},
        "trainer_name": INFERENCE_TRAINER,
        "inference_allowed_mirroring_axes": None,
        "release_metadata": copy.deepcopy(release_metadata),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(exported, temporary)
    temporary.replace(destination)

    checked = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    if set(checked) != set(exported):
        raise RuntimeError("exported checkpoint key set changed")
    if checked.get("trainer_name") != INFERENCE_TRAINER:
        raise RuntimeError("exported checkpoint trainer is not standard nnUNetTrainer")
    if checked.get("init_args") != {"configuration": R.CONFIGURATION}:
        raise RuntimeError("exported checkpoint configuration changed")
    if checked.get("release_metadata") != release_metadata:
        raise RuntimeError("exported checkpoint release metadata changed")
    checked_state = checked.get("network_weights")
    if list(checked_state) != list(state):
        raise RuntimeError("exported checkpoint state keys changed")
    for key in state:
        if not torch.equal(state[key], checked_state[key]):
            raise RuntimeError(f"exported network tensor changed: {key}")
    destination_record = R.file_record(destination)
    if destination_record["bytes"] >= source_record["bytes"]:
        raise RuntimeError("optimizer-free checkpoint did not shrink")
    del checked_state, checked, state, saved, exported
    gc.collect()
    return {
        "source_checkpoint": source_record,
        "checkpoint": destination_record,
        "removed_training_state": [
            "optimizer_state", "grad_scaler_state", "logging", "_best_ema"
        ],
        **state_summary,
    }


def validate_standard_nnunet_model(model_root: Path, folds: list[int]) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    os.environ.setdefault("nnUNet_compile", "0")
    for fold in folds:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device("cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_root), use_folds=(fold,), checkpoint_name="checkpoint_final.pth"
        )
        if predictor.trainer_name != INFERENCE_TRAINER:
            raise RuntimeError(f"fold {fold} did not load with standard nnUNetTrainer")
        predictor.network.load_state_dict(predictor.list_of_parameters[0], strict=True)
        del predictor
        gc.collect()


def model_card(result: dict[str, Any], records: list[dict[str, Any]]) -> str:
    primary = result["primary_summary"]
    safety = result["safety_summary"]
    return f"""---
license: apache-2.0
library_name: nnunet
pipeline_tag: image-segmentation
tags:
  - vesuvius-challenge
  - surface-detection
  - nnunet
  - physical-truth
---

# Cross-scan physical-truth surface-m7 fine-tunes

This is an optimizer-free inference release generated from the public, outcome-blind
cross-scan experiment. The terminal frozen outcome is **{result['status']}**.

## Frozen evidence

- Primary PHerc1203 cross-fitted AP delta: **{primary['mean']:+.6f}** across six
  training seeds (95% t interval {primary['ci95'][0]:+.6f} to
  {primary['ci95'][1]:+.6f}; two-sided p={primary['two_sided_p']:.6g};
  {primary['positive_seeds']}/6 positive seeds).
- Untouched PHerc0139 safety AP delta: **{safety['mean']:+.6f}** (95% t interval
  {safety['ci95'][0]:+.6f} to {safety['ci95'][1]:+.6f}).
- Result content SHA-256: `{result['content_sha256']}`.
- Models: {len(records)} checkpoints = seeds 40..45 x complementary even/odd
  z-stratum folds. No best seed was selected.

The primary endpoint predicts every PHerc1203 evaluation block only with the fold that
did not train on its z stratum. The safety endpoint is on a different scroll and averages
the two folds within each seed. These are sampled physical-truth blocks, not a whole-scroll
reading result.

## Inference

The `model/` directory is standard nnU-Net layout with folds 0..11. The exported
checkpoints retain network weights exactly and remove optimizer/gradient-scaler/logging
state. `trainer_name` is mapped from the experiment-local dynamic trainer to the standard
`nnUNetTrainer`; the architecture and weights are unchanged.

nnU-Net's default multi-fold path averages logits. The experiment's visual ensemble uses
an arithmetic mean of probabilities, so use the included runner when that distinction
matters:

```bash
python predict_crossscan_probability_ensemble.py \
  --release-dir . --input-dir INPUT_TIFFS --output-dir OUTPUT --save-probabilities
```

This loads one network on the GPU and applies the 12 parameter sets sequentially. Expect
roughly 5 GB of host RAM for checkpoint parameters plus normal nnU-Net working memory.

## Intended use and limits

Use as a research surface-probability model at the m7 plans' native spacing. Preserve the
model's CT normalization metadata. Do not call the sampled AP result whole-scroll quality,
ink detection, surface tracing, or proof that a scroll can be read. The complete result,
fixed visual panels, plan, lock, and training receipts are in `evidence/`.

## Attribution and license

Fine-tuned from `scrollprize/surface_m7_nnunet`, the nnU-Net component of the first-place
Kaggle Vesuvius surface-detection solution. The base model declares Apache-2.0; this
release uses the same license and retains that attribution. If original-author terms
differ, those govern.
"""


def verify_release_files(root: Path, manifest: dict[str, Any]) -> None:
    records = []
    for model in manifest["models"]:
        records.append(model["checkpoint"])
    records.extend(manifest["artifacts"])
    records.extend(manifest["tooling"])
    records.extend((manifest["model_files"][name] for name in sorted(manifest["model_files"])))
    for record in records:
        path = R.resolve_data_path(root, record["path"])
        if R.file_record(path) != {
            "bytes": record["bytes"], "sha256": record["sha256"]
        }:
            raise RuntimeError(f"release file differs from manifest: {path}")


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    data = args.data_root.resolve()
    model_dir = args.model_dir.resolve()
    out = args.out.resolve()
    staging = out.with_name(out.name + ".tmp")
    if out.exists() or staging.exists():
        raise FileExistsError(f"release output already exists: {out} or {staging}")

    lock, plan = R.verify_runtime(
        repo,
        repo / "results/crossscan_finetune/execution_lock.json",
        repo / "results/crossscan_finetune/plan.json",
        args.villa_root.resolve(),
        args.labels_root.resolve(),
        repo / "results/physical_normalization_ab/manifest.json",
        model_dir,
    )
    preprocessing = R.verify_preprocessed(plan, lock, data)
    verdict = R.require_any_pilot_authorization(
        data, plan["content_sha256"], lock["content_sha256"]
    )
    result = load_hashed(data / "final_result.json")
    base_model_card = model_dir / "README.md"
    if not base_model_card.is_file() or "\nlicense: apache-2.0\n" not in (
        "\n" + base_model_card.read_text(encoding="utf-8")
    ):
        raise ValueError("base-model card does not declare license: apache-2.0")
    expected_result = {
        "schema_version": "crossscan-final-result-v1",
        "status": "POSITIVE_DEPLOYABLE",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "pilot_verdict_content_sha256": verdict["content_sha256"],
        "selected_steps": verdict["selected_steps"],
    }
    mismatches = [
        key for key, value in expected_result.items() if result.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "release is allowed only for the frozen POSITIVE_DEPLOYABLE result; "
            f"mismatches={mismatches}"
        )

    staging.mkdir(parents=True)
    model_root = staging / "model"
    model_root.mkdir()
    preprocessed = data / "nnUNet_preprocessed" / R.DATASET_NAME
    plans_source = preprocessed / f"{R.PLANS_IDENTIFIER}.json"
    dataset_source = preprocessed / "dataset.json"
    shutil.copy2(plans_source, model_root / "plans.json")
    shutil.copy2(dataset_source, model_root / "dataset.json")

    records = []
    release_fold = 0
    for seed in C.INFERENTIAL_SEEDS:
        for training_fold in ("even", "odd"):
            receipt, checkpoint = R.load_training_receipt(
                plan, lock, data, seed, int(result["selected_steps"]), training_fold
            )
            destination = (
                model_root / f"fold_{release_fold}" / "checkpoint_final.pth"
            )
            metadata = {
                "schema_version": "crossscan-inference-checkpoint-v1",
                "seed": seed,
                "training_fold": training_fold,
                "release_fold": release_fold,
                "source_checkpoint_sha256": receipt["checkpoint"]["sha256"],
                "final_result_content_sha256": result["content_sha256"],
                "execution_lock_content_sha256": lock["content_sha256"],
            }
            exported = export_checkpoint(checkpoint, destination, metadata)
            records.append({
                "seed": seed,
                "training_fold": training_fold,
                "release_fold": release_fold,
                "training_receipt_content_sha256": receipt["content_sha256"],
                **exported,
                "checkpoint": relative_record(staging, destination),
            })
            release_fold += 1

    evidence = []
    for source, relative in (
        (data / "final_result.json", "evidence/final_result.json"),
        (data / "pilot_verdict.json", "evidence/pilot_verdict.json"),
        (data / "preprocessing_receipt.json", "evidence/preprocessing_receipt.json"),
        (repo / "results/crossscan_finetune/plan.json", "evidence/plan.json"),
        (repo / "results/crossscan_finetune/execution_lock.json", "evidence/execution_lock.json"),
        (base_model_card, "evidence/BASE_MODEL_README.md"),
    ):
        evidence.append(copy_artifact(source, staging, relative))
    for attempt in sorted(data.glob("pilot_attempt_steps-*.json")):
        load_hashed(attempt)
        evidence.append(copy_artifact(
            attempt, staging, f"evidence/{attempt.name}"
        ))
    for seed in C.INFERENTIAL_SEEDS:
        for fold in ("even", "odd"):
            name = f"seed-{seed}-{fold}-steps-{result['selected_steps']}.json"
            evidence.append(copy_artifact(
                data / "training_receipts" / name,
                staging,
                f"evidence/training_receipts/{name}",
            ))
    for figure in result["figures"]:
        source = R.resolve_data_path(data, figure["file"]["path"])
        if R.file_record(source) != {
            "bytes": figure["file"]["bytes"],
            "sha256": figure["file"]["sha256"],
        }:
            raise ValueError(f"final-result figure hash mismatch: {source}")
        evidence.append(copy_artifact(
            source, staging, f"evidence/figures/{source.name}"
        ))

    tooling = []
    for name in (
        "export_crossscan_release.py",
        "predict_crossscan_probability_ensemble.py",
    ):
        tooling.append(copy_artifact(repo / name, staging, name))

    card = staging / "README.md"
    card.write_text(model_card(result, records), encoding="utf-8")
    tooling.append(relative_record(staging, card))

    model_files = {
        "plans": relative_record(staging, model_root / "plans.json"),
        "dataset": relative_record(staging, model_root / "dataset.json"),
    }
    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "status": "PASS",
        "created_utc": R.utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "preprocessing_receipt_content_sha256": preprocessing["content_sha256"],
        "pilot_verdict_content_sha256": verdict["content_sha256"],
        "final_result_content_sha256": result["content_sha256"],
        "outcome": result["status"],
        "selected_steps": result["selected_steps"],
        "base_model": plan["inputs"]["model"],
        "license": "Apache-2.0 (matching the declared base-model license)",
        "ensemble": {
            "aggregation": "arithmetic mean of class probabilities",
            "fold_count": 12,
            "mirroring": False,
        },
        "models": records,
        "model_files": model_files,
        "artifacts": evidence,
        "tooling": tooling,
    }
    manifest["content_sha256"] = C.content_hash_without_field(manifest)
    R.atomic_write_json(staging / "release_manifest.json", manifest)
    verify_release_files(staging, manifest)
    validate_standard_nnunet_model(model_root, list(range(12)))
    staging.replace(out)
    verified = load_hashed(out / "release_manifest.json")
    verify_release_files(out, verified)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(C.canonical_json(build_release(args)))


if __name__ == "__main__":
    main()
