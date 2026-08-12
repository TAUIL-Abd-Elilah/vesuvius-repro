#!/usr/bin/env python3
"""Run an exported cross-scan ensemble by averaging model probabilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RELEASE_LICENSES = {
    "fine_tuned_checkpoints_and_derived_evidence": "CC BY-NC 4.0",
    "base_model": "Apache-2.0",
    "release_tooling": "MIT",
}
PLAN_CONTENT_SHA256 = "3f001515f55f289199350ce807eb89b3a09510307b9c200780f195a6e8b11698"
EXECUTION_LOCK_CONTENT_SHA256 = "e682279a19f1f5e6d98df6e1978ce3533025b51b9b8a632789f43f22ab09805f"
REQUIRED_RELEASE_TOOL_PATHS = {
    "crossscan_scrollfiesta_adapter.py",
    "run_crossscan_scrollfiesta_inference.py",
    "predict_crossscan_probability_ensemble.py",
    "run_crossscan_finetune.py",
    "crossscan_finetune.py",
    "score_crossscan_finetune.py",
    "verify_physical_label_semantics.py",
    "physical_normalization_ab.py",
}
REQUIRED_RELEASE_ARTIFACT_PATHS = {
    "evidence/final_result.json",
    "evidence/execution_lock.json",
    "evidence/physical_label_semantic_audit.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("ascii")).hexdigest()


def _validate_file_record(record: object, expected_path: str | None = None) -> str:
    if not isinstance(record, dict):
        raise ValueError("release manifest contains a non-object file record")
    path = record.get("path")
    digest = record.get("sha256")
    size = record.get("bytes")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
        or (expected_path is not None and path != expected_path)
        or type(size) is not int
        or size < 0
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"invalid release file record: {record!r}")
    return path


def validate_release_manifest_value(manifest: dict) -> dict:
    """Validate the complete pure-data contract, including executable paths."""
    if not isinstance(manifest, dict):
        raise ValueError("release manifest is not an object")
    if manifest.get("content_sha256") != content_hash(manifest):
        raise ValueError("release manifest content hash mismatch")
    required = {
        "schema_version": "crossscan-model-release-v1",
        "status": "PASS",
        "plan_content_sha256": PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": EXECUTION_LOCK_CONTENT_SHA256,
        "outcome": "POSITIVE_DEPLOYABLE",
        "selected_steps": 4000,
    }
    if any(manifest.get(key) != expected for key, expected in required.items()):
        raise ValueError("release manifest does not describe the promoted frozen result")
    if manifest.get("ensemble") != {
        "aggregation": "arithmetic mean of class probabilities",
        "fold_count": 12,
        "mirroring": False,
    }:
        raise ValueError("release manifest does not describe the locked 12-model ensemble")
    if manifest.get("licenses") != RELEASE_LICENSES:
        raise ValueError("release manifest license contract mismatch")
    for key in (
        "final_result_content_sha256",
        "semantic_audit_content_sha256",
        "semantic_audit_file_sha256",
    ):
        value = manifest.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"release manifest has invalid {key}")
    records = manifest.get("models")
    expected_pairs = [
        (seed, training_fold)
        for seed in range(40, 46)
        for training_fold in ("even", "odd")
    ]
    if not isinstance(records, list) or len(records) != len(expected_pairs):
        raise ValueError("release manifest must contain 12 model records")
    all_paths: list[str] = []
    for release_fold, ((seed, training_fold), record) in enumerate(
        zip(expected_pairs, records)
    ):
        if (
            not isinstance(record, dict)
            or record.get("seed") != seed
            or record.get("training_fold") != training_fold
            or record.get("release_fold") != release_fold
        ):
            raise ValueError("release model ordering differs from the frozen 12-model order")
        all_paths.append(_validate_file_record(
            record.get("checkpoint"),
            f"model/fold_{release_fold}/checkpoint_final.pth",
        ))
    model_files = manifest.get("model_files")
    expected_model_paths = {
        "plans": "model/plans.json",
        "dataset": "model/dataset.json",
    }
    if not isinstance(model_files, dict) or set(model_files) != set(expected_model_paths):
        raise ValueError("release model-file universe must be exactly plans and dataset")
    for name, expected_path in expected_model_paths.items():
        all_paths.append(_validate_file_record(model_files[name], expected_path))
    supporting_paths: dict[str, set[str]] = {}
    for field in ("artifacts", "tooling", "reports"):
        values = manifest.get(field)
        if not isinstance(values, list):
            raise ValueError(f"release manifest {field} must be a list")
        paths = {_validate_file_record(record) for record in values}
        if len(paths) != len(values):
            raise ValueError(f"release manifest {field} contains duplicate paths")
        supporting_paths[field] = paths
        all_paths.extend(paths)
    if not REQUIRED_RELEASE_ARTIFACT_PATHS <= supporting_paths["artifacts"]:
        raise ValueError("release manifest omits required provenance artifacts")
    if not REQUIRED_RELEASE_TOOL_PATHS <= supporting_paths["tooling"]:
        raise ValueError("release package omits an imported local inference module")
    if len(set(all_paths)) != len(all_paths):
        raise ValueError("release manifest aliases one file across multiple roles")
    return manifest


def load_release_manifest(release_dir: Path) -> dict:
    manifest_path = release_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_release_manifest_value(manifest)


def verify_release_files(release_dir: Path, manifest: dict) -> list[int]:
    validate_release_manifest_value(manifest)
    records = manifest.get("models")
    def verify(record: dict) -> None:
        relative = Path(record["checkpoint"]["path"])
        expected = record["checkpoint"]
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checkpoint path: {relative}")
        path = (release_dir / relative).resolve()
        path.relative_to(release_dir.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")

    for record in records:
        verify(record)
    supporting = list(manifest.get("model_files", {}).values())
    supporting += list(manifest.get("artifacts", []))
    supporting += list(manifest.get("tooling", []))
    supporting += list(manifest.get("reports", []))
    for expected in supporting:
        relative = Path(expected["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe release path: {relative}")
        path = (release_dir / relative).resolve()
        path.relative_to(release_dir.resolve())
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"release file hash mismatch: {path}")
    return list(range(12))


def probability_mean_as_logits(logits: Iterable["torch.Tensor"]):
    """Return log-probabilities whose softmax is the arithmetic probability mean."""
    import torch

    total = None
    count = 0
    for value in logits:
        if value.ndim < 2 or value.shape[0] != 2 or not torch.isfinite(value).all():
            raise ValueError(f"invalid two-class logits with shape {tuple(value.shape)}")
        probability = torch.softmax(value.float(), dim=0).to(device="cpu", dtype=torch.float64)
        if total is None:
            total = probability
        else:
            if probability.shape != total.shape:
                raise ValueError("ensemble logit shapes differ")
            total += probability
        count += 1
    if total is None or count == 0:
        raise ValueError("cannot average an empty ensemble")
    mean = (total / count).to(dtype=torch.float32)
    tiny = torch.finfo(mean.dtype).tiny
    return torch.log(mean.clamp_min(tiny))


def make_predictor(device: str, tile_step_size: float):
    import torch
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from torch._dynamo import OptimizedModule

    class ProbabilityMeanPredictor(nnUNetPredictor):
        def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
            original_threads = torch.get_num_threads()
            torch.set_num_threads(min(default_num_processes, original_threads))
            try:
                def predictions():
                    for parameters in self.list_of_parameters:
                        network = (
                            self.network._orig_mod
                            if isinstance(self.network, OptimizedModule)
                            else self.network
                        )
                        network.load_state_dict(parameters, strict=True)
                        yield self.predict_sliding_window_return_logits(data).to("cpu")

                result = probability_mean_as_logits(predictions())
                if self.verbose:
                    print("Probability-space ensemble prediction done")
                return result
            finally:
                torch.set_num_threads(original_threads)

    return ProbabilityMeanPredictor(
        tile_step_size=tile_step_size,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )


def write_run_receipt(output_dir: Path, manifest: dict, args: argparse.Namespace) -> None:
    receipt = {
        "schema_version": "crossscan-probability-ensemble-run-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_manifest_content_sha256": manifest["content_sha256"],
        "aggregation": "arithmetic mean of class probabilities",
        "folds": list(range(12)),
        "mirroring": False,
        "tile_step_size": args.tile_step_size,
        "save_probabilities": args.save_probabilities,
        "command": [sys.executable, *sys.argv],
    }
    receipt["content_sha256"] = content_hash(receipt)
    path = output_dir / "crossscan_ensemble_run.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite ensemble receipt: {path}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-step-size", type=float, default=0.5)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--skip-integrity-check", action="store_true")
    args = parser.parse_args()

    release = args.release_dir.resolve()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise SystemExit(f"input directory does not exist: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    if not 0 < args.tile_step_size <= 1:
        raise SystemExit("tile step size must be in (0, 1]")
    if args.num_processes < 1:
        raise SystemExit("num processes must be positive")

    manifest = load_release_manifest(release)
    folds = (
        list(range(12))
        if args.skip_integrity_check
        else verify_release_files(release, manifest)
    )
    os.environ.setdefault("nnUNet_compile", "0")
    predictor = make_predictor(args.device, args.tile_step_size)
    predictor.initialize_from_trained_model_folder(
        str(release / "model"),
        use_folds=tuple(folds),
        checkpoint_name="checkpoint_final.pth",
    )
    output.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        str(source),
        str(output),
        save_probabilities=args.save_probabilities,
        overwrite=False,
        num_processes_preprocessing=args.num_processes,
        num_processes_segmentation_export=args.num_processes,
    )
    write_run_receipt(output, manifest, args)


if __name__ == "__main__":
    main()
