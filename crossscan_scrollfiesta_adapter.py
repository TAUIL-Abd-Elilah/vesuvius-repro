#!/usr/bin/env python3
"""Export a locked cross-scan probability ROI and materialize ScrollFiesta grids.

The adapter deliberately stops at ScrollFiesta's documented ``cubes_PRED`` /
``cubes_RAW`` boundary.  It does not reimplement meshing.  Outputs are
create-no-replace and every payload is hash bound so an interrupted export can
only resume when the existing bytes agree with their receipts and the current
input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile


SHAPE = (256, 256, 256)
CHUNKS = (128, 128, 128)
DEFAULT_ORIGIN = (3840, 3712, 1344)
CONTEXT_ORIGIN = (3776, 3648, 1280)
CONTEXT_SHAPE = (384, 384, 384)
RAW_SOURCE_URI = (
    "s3://vesuvius-challenge-open-data/PHerc0139/volumes/"
    "20250728140407-9.362um-1.2m-113keV-masked.zarr"
)
RAW_SOURCE_SHAPE = (20974, 6621, 6621)
RAW_SOURCE_CHUNKS = (128, 128, 128)
RAW_CONTEXT_ARRAY_SHA256 = (
    "b654ba8428beb5f24378efb2d9e8f5d516cec60410871682bdbff5535b6b665f"
)
RAW_CUBE_ARRAY_SHA256 = {
    "z03840_y03712_x01344.tif": "039d756bff0e084a90b0e1186abfbdfb0abff2195b4156492340a4b7e9accf47",
    "z03840_y03712_x01472.tif": "a7a241ff27f0ca44f5d29560edfe9cd96c8f9bd9d35418a68cdf03f8ea5c796f",
    "z03840_y03840_x01344.tif": "02774b468ca800b22df5fc25f0abc6ba1c34b81892ab1b816704924e2a36ffcb",
    "z03840_y03840_x01472.tif": "e2dad7c007f8fd3e4e09ff9ae91b76981e8fe09f17da6a6282f1629fa788d777",
    "z03968_y03712_x01344.tif": "413db4f7e8b24b6429bc8257fe2312d299d2af4b24761547c966a670350aa141",
    "z03968_y03712_x01472.tif": "e6efb451360e58d1a9ad9cbab12cbdad66dce661cc3940ff194ce03599ecccfb",
    "z03968_y03840_x01344.tif": "5c838de451510b54c6948bf4ea5f63040bdc55831f2ae2f4c61456b43f5ebddd",
    "z03968_y03840_x01472.tif": "e312de6a5c976efb9061d74d2b92ac21908672419387472c601d03320f9ff5b0",
}
RAW_CARVE_SCHEMA = "crossscan-scrollfiesta-raw-carve-v1"
INFERENCE_RUN_SCHEMA = "crossscan-scrollfiesta-locked-inference-v1"
INFERENCE_TOOL_NAMES = (
    "run_crossscan_scrollfiesta_inference.py",
    "crossscan_scrollfiesta_adapter.py",
    "run_crossscan_finetune.py",
    "predict_crossscan_probability_ensemble.py",
    "crossscan_finetune.py",
    "score_crossscan_finetune.py",
    "verify_physical_label_semantics.py",
    "physical_normalization_ab.py",
)
BASE_MODEL_FILES = {
    "checkpoint": {
        "path": "fold_0/checkpoint_best.pth",
        "bytes": 820473701,
        "sha256": "17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e",
    },
    "plans": {
        "path": "plans.json",
        "bytes": 10751,
        "sha256": "5358b6973982815b727384d8ed3d6ef86db047c07ff00d85f8ae390f07c49fbc",
    },
    "dataset": {
        "path": "dataset.json",
        "bytes": 244,
        "sha256": "19b98ea3ce06e35f45afe0bb256d94f889af6b87a037e7024762c4eb5c3eb15e",
    },
}
NORMALIZATION_CONTRACT = {
    "max": 255.0,
    "mean": 87.54424285888672,
    "median": 81.0,
    "min": 0.0,
    "percentile_00_5": 0.0,
    "percentile_99_5": 212.0,
    "std": 47.74376678466797,
}
AXES = ("z", "y", "x")
SCHEMA = "crossscan-scrollfiesta-adapter-v1"
DOWNSTREAM_LOCK_SCHEMA = "crossscan-scrollfiesta-downstream-lock-v2"
DOWNSTREAM_LOCK_CONTENT_SHA256 = (
    "06142dc819c193a462f37d08a4769024c41ab551411013d40dda72db148457f6"
)
PLAN_CONTENT_SHA256 = (
    "3f001515f55f289199350ce807eb89b3a09510307b9c200780f195a6e8b11698"
)
EXECUTION_LOCK_CONTENT_SHA256 = (
    "e682279a19f1f5e6d98df6e1978ce3533025b51b9b8a632789f43f22ab09805f"
)
PILOT_VERDICT_CONTENT_SHA256 = (
    "e1dff71a20aa9cea87882b677d12a25335256b1144b7812696c3820e5171b53a"
)
INFERENTIAL_SEEDS = list(range(40, 46))
FIXED_THRESHOLD = 0.2
GRID_ARMS = ("baseline-fixed", "candidate-fixed", "candidate-matched-mass")
PINNED_RUNTIME_ENVIRONMENT = {
    "blosc2": "4.9.1",
    "nnunetv2_module": "nnunetv2/__init__.py",
    "numpy": "2.4.6",
    "python": "3.14.6",
    "pyyaml": "6.0.3",
    "scipy": "1.18.0",
    "tifffile": "2026.7.14",
    "torch": "2.13.0+cu126",
    "torch_cuda": "12.6",
    "zarr": "3.2.1",
}
PINNED_VILLA = {
    "commit": "94ba215963afb6216e380fe2c86131fa5e724c3b",
    "nnunet_tree": "24941bfa19e7239db6458287c2a39b9ad4bd7f4a",
}
DETERMINISM_CONTRACT = {
    "nnUNet_compile": "0",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_enabled": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": True,
    "float32_matmul_precision": "highest",
    "deterministic_algorithms": False,
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(canonical_json(unsigned).encode("ascii")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(value).cast("B")).hexdigest()


def file_record(path: Path, relative: str | None = None) -> dict:
    path = Path(path)
    record = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if relative is not None:
        record = {"path": relative, **record}
    return record


def _load_hashed_json(path: Path, description: str) -> tuple[dict, bytes]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict) or value.get("content_sha256") != content_hash(value):
        raise ValueError(f"{description} content hash mismatch: {path}")
    return value, payload


def validate_downstream_lock(path: Path | None = None) -> dict:
    lock_path = Path(path) if path is not None else Path(__file__).with_name(
        "crossscan_scrollfiesta_downstream_lock.json"
    )
    value, _ = _load_hashed_json(lock_path, "downstream lock")
    if value.get("schema_version") != DOWNSTREAM_LOCK_SCHEMA:
        raise ValueError("downstream lock schema mismatch")
    if value.get("status") != "preoutcome_lock_no_terminal_pilot_or_final_result_inspected":
        raise ValueError("downstream lock status mismatch")
    if value.get("content_sha256") != DOWNSTREAM_LOCK_CONTENT_SHA256:
        raise ValueError("downstream lock identity mismatch")
    return value


def validate_execution_lock(path: Path) -> tuple[dict, bytes]:
    value, payload = _load_hashed_json(path, "execution lock")
    required = {
        "schema_version": "vesuvius-crossscan-execution-lock-v1",
        "status": "public_preoutcome_execution_lock",
        "content_sha256": EXECUTION_LOCK_CONTENT_SHA256,
        "environment": PINNED_RUNTIME_ENVIRONMENT,
        "villa": PINNED_VILLA,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"execution lock has invalid {key}")
    if value.get("plan", {}).get("content_sha256") != PLAN_CONTENT_SHA256:
        raise ValueError("execution lock references a different plan")
    resolved = value.get("resolved_protocol", {})
    for key, expected in {
        "inference_tile_step_size": 0.5,
        "inference_gaussian": True,
        "inference_mirroring": False,
    }.items():
        if resolved.get(key) != expected:
            raise ValueError(f"execution lock has invalid {key}")
    return value, payload


def _require_runtime_identity(identity: object) -> dict:
    if not isinstance(identity, dict):
        raise ValueError("inference receipt lacks a runtime identity")
    if identity.get("environment") != PINNED_RUNTIME_ENVIRONMENT:
        raise ValueError("inference runtime differs from the frozen environment")
    if identity.get("villa") != PINNED_VILLA:
        raise ValueError("inference runtime differs from the frozen nnU-Net tree")
    if identity.get("determinism") != DETERMINISM_CONTRACT:
        raise ValueError("inference determinism settings differ from the lock")
    device = identity.get("cuda_device")
    if (
        not isinstance(device, dict)
        or set(device) != {
            "index", "name", "uuid", "driver_version", "capability", "total_memory_bytes"
        }
        or device.get("index") != 0
        or not isinstance(device.get("name"), str)
        or not device["name"]
        or not isinstance(device.get("uuid"), str)
        or not device["uuid"]
        or not isinstance(device.get("driver_version"), str)
        or not device["driver_version"]
        or not isinstance(device.get("capability"), list)
        or len(device["capability"]) != 2
        or any(type(value) is not int or value < 0 for value in device["capability"])
        or type(device.get("total_memory_bytes")) is not int
        or device["total_memory_bytes"] <= 0
    ):
        raise ValueError("inference receipt has an invalid CUDA device identity")
    return identity


def _require_promotion_fields(value: dict) -> None:
    required = {
        "schema_version": "crossscan-final-result-v1",
        "status": "POSITIVE_DEPLOYABLE",
        "plan_content_sha256": PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": EXECUTION_LOCK_CONTENT_SHA256,
        "pilot_verdict_content_sha256": PILOT_VERDICT_CONTENT_SHA256,
        "selected_steps": 4000,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"promotion receipt has invalid {key}")
    if [row.get("seed") for row in value.get("seed_rows", [])] != INFERENTIAL_SEEDS:
        raise ValueError("promotion receipt does not contain the six frozen seed rows")
    if set(value.get("comparisons", {})) != {str(seed) for seed in INFERENTIAL_SEEDS}:
        raise ValueError("promotion receipt does not contain the six frozen comparisons")
    expected_gates = {
        "primary_effect": 0.01,
        "minimum_positive_seeds": 5,
        "alpha_two_sided": 0.05,
        "safety_noninferiority_margin": 0.005,
    }
    if value.get("gates") != expected_gates:
        raise ValueError("promotion receipt gate contract mismatch")
    import score_crossscan_finetune as scorer

    primary_deltas = []
    safety_deltas = []
    for seed, row in zip(INFERENTIAL_SEEDS, value["seed_rows"]):
        if not isinstance(row, dict):
            raise ValueError("promotion receipt has an invalid seed row")
        primary_delta = row.get("primary_delta")
        safety_delta = row.get("safety_delta")
        comparison = value["comparisons"][str(seed)]
        if (
            not isinstance(primary_delta, (int, float))
            or not isinstance(safety_delta, (int, float))
            or not isinstance(comparison, dict)
            or comparison.get("primary", {}).get("overall", {}).get(
                "average_precision_delta"
            ) != primary_delta
            or comparison.get("safety", {}).get("overall", {}).get(
                "average_precision_delta"
            ) != safety_delta
        ):
            raise ValueError("promotion seed rows differ from frozen comparisons")
        primary_deltas.append(float(primary_delta))
        safety_deltas.append(float(safety_delta))
    expected_primary = scorer.t_summary(primary_deltas)
    expected_safety = scorer.t_summary(safety_deltas)
    primary = value.get("primary_summary", {})
    safety = value.get("safety_summary", {})
    if primary != expected_primary or safety != expected_safety:
        raise ValueError("promotion receipt summaries do not recompute from seed rows")
    if scorer.outcome_bucket(primary, safety) != value.get("status"):
        raise ValueError("promotion receipt status does not recompute from summaries")
    if (
        primary.get("n_seeds") != 6
        or not isinstance(primary.get("mean"), (int, float))
        or primary["mean"] < 0.01
        or primary.get("positive_seeds", 0) < 5
        or not isinstance(primary.get("two_sided_p"), (int, float))
        or primary["two_sided_p"] >= 0.05
        or safety.get("n_seeds") != 6
        or not isinstance(safety.get("mean"), (int, float))
        or safety["mean"] < -0.005
    ):
        raise ValueError("promotion receipt summaries do not prove POSITIVE_DEPLOYABLE")
    figures = value.get("figures")
    if not isinstance(figures, list) or len(figures) != 8:
        raise ValueError("promotion receipt does not bind the eight frozen figures")


def validate_promotion_receipt(path: Path) -> tuple[dict, bytes]:
    value, payload = _load_hashed_json(path, "promotion receipt")
    _require_promotion_fields(value)
    return value, payload


def _validate_semantic_audit(path: Path) -> tuple[dict, bytes]:
    import verify_physical_label_semantics as semantics

    return semantics.validate_audit_receipt(path)


def _relative_file_record(root: Path, relative: str) -> dict:
    path = (Path(root) / relative).resolve()
    path.relative_to(Path(root).resolve())
    return file_record(path, Path(relative).as_posix())


def validate_model_source(
    model_role: str,
    model_source: Path,
    *,
    promotion_content_sha256: str,
    semantic_audit_content_sha256: str,
) -> dict:
    """Rehash the exact baseline model or all twelve promoted candidates."""
    model_source = Path(model_source)
    if model_role == "baseline":
        records = {
            name: _relative_file_record(model_source, expected["path"])
            for name, expected in BASE_MODEL_FILES.items()
        }
        if records != BASE_MODEL_FILES:
            raise ValueError("baseline model source differs from released m7 fold 0")
        return {
            "kind": "released-m7-fold-0",
            "files": records,
        }
    if model_role != "candidate":
        raise ValueError("inference model role must be baseline or candidate")
    import predict_crossscan_probability_ensemble as ensemble

    manifest = ensemble.load_release_manifest(model_source)
    ensemble.verify_release_files(model_source, manifest)
    release_promotion, _ = validate_promotion_receipt(
        model_source / "evidence/final_result.json"
    )
    release_lock, _ = validate_execution_lock(
        model_source / "evidence/execution_lock.json"
    )
    release_semantic, release_semantic_payload = _validate_semantic_audit(
        model_source / "evidence/physical_label_semantic_audit.json"
    )
    if (
        manifest.get("plan_content_sha256") != PLAN_CONTENT_SHA256
        or manifest.get("execution_lock_content_sha256")
        != EXECUTION_LOCK_CONTENT_SHA256
        or manifest.get("final_result_content_sha256")
        != promotion_content_sha256
        or manifest.get("semantic_audit_content_sha256")
        != semantic_audit_content_sha256
        or manifest.get("outcome") != "POSITIVE_DEPLOYABLE"
        or manifest.get("selected_steps") != 4000
        or release_promotion.get("content_sha256") != promotion_content_sha256
        or release_lock.get("content_sha256") != EXECUTION_LOCK_CONTENT_SHA256
        or release_semantic.get("content_sha256")
        != semantic_audit_content_sha256
        or hashlib.sha256(release_semantic_payload).hexdigest()
        != manifest.get("semantic_audit_file_sha256")
    ):
        raise ValueError("candidate release manifest provenance mismatch")
    manifest_path = model_source / "release_manifest.json"
    return {
        "kind": "promoted-crossscan-release",
        "release_manifest": {
            **file_record(manifest_path, "release_manifest.json"),
            "content_sha256": manifest["content_sha256"],
        },
    }


def _require_inference_contract(receipt: dict) -> str:
    role = receipt.get("model_role")
    expected_aggregation = (
        "single released m7 fold-0 class probability"
        if role == "baseline"
        else "float64 arithmetic mean of twelve class probabilities"
    )
    expected_folds = [0] if role == "baseline" else list(range(12))
    required = {
        "schema_version": INFERENCE_RUN_SCHEMA,
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "plan_content_sha256": PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": EXECUTION_LOCK_CONTENT_SHA256,
        "context_box_l0_zyx": [3776, 4160, 3648, 4032, 1280, 1664],
        "retained_box_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "context_array_sha256": RAW_CONTEXT_ARRAY_SHA256,
        "probability_shape": list(SHAPE),
        "probability_dtype": "float32",
        "tile_step_size": 0.5,
        "gaussian": True,
        "mirroring": False,
        "normalization": NORMALIZATION_CONTRACT,
        "aggregation": expected_aggregation,
        "folds": expected_folds,
    }
    if role not in ("baseline", "candidate"):
        raise ValueError("inference receipt has invalid model role")
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise ValueError(f"inference receipt has invalid {key}")
    _require_runtime_identity(receipt.get("runtime_identity"))
    return role


def validate_inference_run(
    run_dir: Path, model_source: Path, raw_carve: Path,
) -> tuple[dict, np.ndarray]:
    """Validate a self-contained locked inference and its external model bytes."""
    run_dir = Path(run_dir)
    receipt, _ = _load_hashed_json(
        run_dir / "inference_receipt.json", "locked inference receipt"
    )
    role = _require_inference_contract(receipt)
    expected_artifact_paths = {
        "raw_carve_receipt": "provenance/raw_carve_receipt.json",
        "promotion_receipt": "provenance/final_result.json",
        "execution_lock": "provenance/execution_lock.json",
        "semantic_audit": "provenance/physical_label_semantic_audit.json",
    }
    if role == "candidate":
        expected_artifact_paths["release_manifest"] = "provenance/release_manifest.json"
    for field, expected_path in expected_artifact_paths.items():
        if receipt.get(field, {}).get("path") != expected_path:
            raise ValueError(f"inference {field} path mismatch")
    raw = verify_raw_carve(raw_carve)
    embedded_raw = _validate_embedded_artifact(
        run_dir, receipt.get("raw_carve_receipt", {}),
        "inference RAW carve receipt",
    )
    _require_raw_receipt_header(embedded_raw)
    if embedded_raw != raw:
        raise ValueError("inference RAW receipt differs from supplied carve")
    promotion = _validate_embedded_artifact(
        run_dir, receipt.get("promotion_receipt", {}),
        "inference promotion receipt",
    )
    _require_promotion_fields(promotion)
    execution_lock = _validate_embedded_artifact(
        run_dir, receipt.get("execution_lock", {}), "inference execution lock"
    )
    validated_lock, _ = validate_execution_lock(
        run_dir / receipt["execution_lock"]["path"]
    )
    if execution_lock != validated_lock:
        raise ValueError("inference execution lock differs after full validation")
    semantic_record = receipt.get("semantic_audit", {})
    semantic = _validate_embedded_semantic(
        run_dir, semantic_record, "inference semantic audit"
    )
    expected_model = validate_model_source(
        role,
        model_source,
        promotion_content_sha256=promotion["content_sha256"],
        semantic_audit_content_sha256=semantic["content_sha256"],
    )
    if receipt.get("model_source") != expected_model:
        raise ValueError("inference model-source record mismatch")
    probability_record = receipt.get("probability_file", {})
    if probability_record.get("path") != "probability.npy":
        raise ValueError("inference probability path mismatch")
    probability_path = run_dir / "probability.npy"
    if probability_record != file_record(probability_path, "probability.npy"):
        raise ValueError("inference probability file record mismatch")
    probability = _load_inference_probability(probability_path)
    if sha256_array(probability) != receipt.get("probability_array_sha256"):
        raise ValueError("inference probability array hash mismatch")
    expected_files = {
        "inference_receipt.json",
        "probability.npy",
        receipt["raw_carve_receipt"]["path"],
        receipt["promotion_receipt"]["path"],
        receipt["execution_lock"]["path"],
        receipt["semantic_audit"]["path"],
    }
    release_record = receipt.get("release_manifest")
    if role == "candidate":
        embedded_release = _validate_embedded_artifact(
            run_dir, release_record or {}, "inference release manifest"
        )
        if (
            embedded_release.get("content_sha256")
            != expected_model["release_manifest"]["content_sha256"]
            or embedded_release.get("final_result_content_sha256")
            != promotion["content_sha256"]
            or embedded_release.get("semantic_audit_content_sha256")
            != semantic["content_sha256"]
        ):
            raise ValueError("embedded candidate release manifest mismatch")
        expected_files.add(release_record["path"])
    elif release_record is not None:
        raise ValueError("baseline inference must not embed a candidate release manifest")
    tooling = receipt.get("tooling")
    expected_tool_names = {f"provenance/{name}" for name in INFERENCE_TOOL_NAMES}
    if not isinstance(tooling, list) or len(tooling) != len(expected_tool_names):
        raise ValueError("inference receipt must bind every executed tooling file")
    if {record.get("path") for record in tooling} != expected_tool_names:
        raise ValueError("inference tooling universe mismatch")
    for record in tooling:
        relative = record["path"]
        path = run_dir / relative
        if record != file_record(path, relative):
            raise ValueError(f"inference tooling record mismatch: {relative}")
        current = Path(__file__).with_name(Path(relative).name)
        if sha256_file(path) != sha256_file(current):
            raise ValueError(f"inference tooling differs from current verifier: {relative}")
        expected_files.add(relative)
    if _regular_file_universe(run_dir) != expected_files:
        raise ValueError("inference run exact file universe mismatch")
    return receipt, probability


def _validate_embedded_inference(root: Path, record: dict) -> tuple[dict, set[str]]:
    inference = _validate_embedded_artifact(
        root, record, "embedded locked inference receipt"
    )
    role = _require_inference_contract(inference)
    expected_artifact_paths = {
        "raw_carve_receipt": "provenance/raw_carve_receipt.json",
        "promotion_receipt": "provenance/final_result.json",
        "execution_lock": "provenance/execution_lock.json",
        "semantic_audit": "provenance/physical_label_semantic_audit.json",
    }
    if role == "candidate":
        expected_artifact_paths["release_manifest"] = "provenance/release_manifest.json"
    for field, expected_path in expected_artifact_paths.items():
        if inference.get(field, {}).get("path") != expected_path:
            raise ValueError(f"embedded inference {field} path mismatch")
    paths = {record["path"]}
    promotion = _validate_embedded_artifact(
        root, inference.get("promotion_receipt", {}),
        "embedded inference promotion receipt",
    )
    _require_promotion_fields(promotion)
    paths.add(inference["promotion_receipt"]["path"])
    execution_lock = _validate_embedded_artifact(
        root, inference.get("execution_lock", {}),
        "embedded inference execution lock",
    )
    validated_lock, _ = validate_execution_lock(
        Path(root) / inference["execution_lock"]["path"]
    )
    if execution_lock != validated_lock:
        raise ValueError("embedded execution lock differs after full validation")
    paths.add(inference["execution_lock"]["path"])
    raw = _validate_embedded_artifact(
        root, inference.get("raw_carve_receipt", {}),
        "embedded inference RAW receipt",
    )
    _require_raw_receipt_header(raw)
    paths.add(inference["raw_carve_receipt"]["path"])
    semantic_record = inference.get("semantic_audit", {})
    semantic = _validate_embedded_semantic(
        root, semantic_record, "embedded inference semantic audit"
    )
    paths.add(semantic_record["path"])
    model_source = inference.get("model_source")
    if role == "baseline":
        if model_source != {
            "kind": "released-m7-fold-0", "files": BASE_MODEL_FILES
        } or inference.get("release_manifest") is not None:
            raise ValueError("embedded baseline inference model source mismatch")
    else:
        import predict_crossscan_probability_ensemble as ensemble

        if not isinstance(model_source, dict) or model_source.get("kind") != (
            "promoted-crossscan-release"
        ):
            raise ValueError("embedded candidate inference model source mismatch")
        release_record = inference.get("release_manifest", {})
        release = _validate_embedded_artifact(
            root, release_record, "embedded inference release manifest"
        )
        ensemble.validate_release_manifest_value(release)
        expected_release = model_source.get("release_manifest", {})
        if (
            release.get("schema_version") != "crossscan-model-release-v1"
            or release.get("status") != "PASS"
            or release.get("outcome") != "POSITIVE_DEPLOYABLE"
            or release.get("selected_steps") != 4000
            or release.get("plan_content_sha256") != PLAN_CONTENT_SHA256
            or release.get("execution_lock_content_sha256")
            != EXECUTION_LOCK_CONTENT_SHA256
            or release.get("final_result_content_sha256")
            != promotion["content_sha256"]
            or release.get("semantic_audit_content_sha256")
            != semantic["content_sha256"]
            or release.get("semantic_audit_file_sha256")
            != semantic_record.get("sha256")
            or release.get("content_sha256")
            != expected_release.get("content_sha256")
            or release_record.get("bytes") != expected_release.get("bytes")
            or release_record.get("sha256") != expected_release.get("sha256")
        ):
            raise ValueError("embedded candidate release provenance mismatch")
        paths.add(release_record["path"])
    tooling = inference.get("tooling")
    expected_tool_names = {f"provenance/{name}" for name in INFERENCE_TOOL_NAMES}
    if not isinstance(tooling, list) or len(tooling) != len(expected_tool_names):
        raise ValueError("embedded inference tooling universe mismatch")
    if {record.get("path") for record in tooling} != expected_tool_names:
        raise ValueError("embedded inference tooling universe mismatch")
    for tooling_record in tooling:
        relative = tooling_record.get("path", "")
        path = Path(root) / relative
        if tooling_record != file_record(path, relative):
            raise ValueError(f"embedded inference tooling record mismatch: {relative}")
        current = Path(__file__).with_name(Path(relative).name)
        if sha256_file(path) != sha256_file(current):
            raise ValueError(f"embedded inference tooling differs from verifier: {relative}")
        paths.add(relative)
    return inference, paths


def _artifact_record(relative: str, payload: bytes, value: dict) -> dict:
    return {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": value["content_sha256"],
        "schema_version": value.get("schema_version"),
        "status": value.get("status"),
    }


def _validate_embedded_artifact(root: Path, record: dict, description: str) -> dict:
    raw_relative = record.get("path")
    if not isinstance(raw_relative, str):
        raise ValueError(f"unsafe {description} path")
    relative = Path(raw_relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe {description} path")
    resolved_root = Path(root).resolve()
    path = (resolved_root / relative).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"unsafe {description} path") from error
    value, payload = _load_hashed_json(path, description)
    expected = _artifact_record(relative.as_posix(), payload, value)
    if record != expected:
        raise ValueError(f"{description} record mismatch")
    return value


def _validate_embedded_semantic(root: Path, record: dict, description: str) -> dict:
    """Validate both an embedded record envelope and the full semantic receipt."""
    generic = _validate_embedded_artifact(root, record, description)
    semantic, _ = _validate_semantic_audit(Path(root) / record["path"])
    if semantic != generic:
        raise ValueError(f"{description} differs after semantic validation")
    return semantic


def _require_locked_origin(origin: tuple[int, int, int]) -> tuple[int, int, int]:
    value = tuple(origin)
    if value != DEFAULT_ORIGIN:
        raise ValueError(f"world origin must equal locked origin {DEFAULT_ORIGIN}")
    return value


def _regular_file_universe(root: Path) -> set[str]:
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlinks are forbidden in an immutable artifact: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
    return files


def _require_output_disjoint(output: Path, input_roots: Iterable[Path]) -> Path:
    """Reject writes that would mutate an immutable input artifact tree."""
    resolved_output = Path(output).resolve()
    for root in input_roots:
        resolved_root = Path(root).resolve()
        try:
            resolved_output.relative_to(resolved_root)
        except ValueError:
            continue
        raise ValueError(
            f"output must be outside immutable input artifact {resolved_root}: "
            f"{resolved_output}"
        )
    return resolved_output


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _create_bytes(path: Path, payload: bytes) -> None:
    """Create *path* exclusively and persist its bytes before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _create_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _load_locked_context(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if value.shape != CONTEXT_SHAPE or value.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"locked CT context must be uint8 {CONTEXT_SHAPE}, got "
            f"{value.dtype} {value.shape}"
        )
    value = np.ascontiguousarray(value)
    if sha256_array(value) != RAW_CONTEXT_ARRAY_SHA256:
        raise ValueError("locked CT context array hash mismatch")
    return value


def _read_public_context() -> np.ndarray:
    import zarr

    array = zarr.open(
        RAW_SOURCE_URI, path="0", mode="r", storage_options={"anon": True}
    )
    if (
        tuple(array.shape) != RAW_SOURCE_SHAPE
        or tuple(array.chunks) != RAW_SOURCE_CHUNKS
        or np.dtype(array.dtype) != np.dtype(np.uint8)
    ):
        raise ValueError("public RAW Zarr metadata differs from the locked source")
    z0, y0, x0 = CONTEXT_ORIGIN
    value = np.ascontiguousarray(
        array[
            z0:z0 + CONTEXT_SHAPE[0],
            y0:y0 + CONTEXT_SHAPE[1],
            x0:x0 + CONTEXT_SHAPE[2],
        ]
    )
    if value.shape != CONTEXT_SHAPE or sha256_array(value) != RAW_CONTEXT_ARRAY_SHA256:
        raise ValueError("public RAW context differs from the pinned byte identity")
    return value


def carve_locked_raw(output: Path, *, _source_array: np.ndarray | None = None) -> dict:
    """Carve the one locked CT context and all eight native RAW cubes."""
    validate_downstream_lock()
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing RAW carve: {output}")
    context = (
        _read_public_context()
        if _source_array is None
        else np.ascontiguousarray(_source_array)
    )
    if (
        context.shape != CONTEXT_SHAPE
        or context.dtype != np.dtype(np.uint8)
        or sha256_array(context) != RAW_CONTEXT_ARRAY_SHA256
    ):
        raise ValueError("RAW carve source does not match the locked public context")
    output.mkdir(parents=True, exist_ok=False)
    context_path = output / "context.npy"
    _create_npy(context_path, context)
    raw_dir = output / "cubes_RAW"
    raw_dir.mkdir()
    central = context[64:320, 64:320, 64:320]
    cubes = []
    for (lz, ly, lx), name in _cube_specs():
        cube = np.ascontiguousarray(
            central[lz:lz + 128, ly:ly + 128, lx:lx + 128]
        )
        path = raw_dir / name
        tifffile.imwrite(
            path, cube, photometric="minisblack", compression=None, rowsperstrip=128
        )
        cubes.append(file_record(path, f"cubes_RAW/{name}"))
    receipt = {
        "schema_version": RAW_CARVE_SCHEMA,
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "source": {
            "uri": RAW_SOURCE_URI,
            "level": 0,
            "shape": list(RAW_SOURCE_SHAPE),
            "chunks": list(RAW_SOURCE_CHUNKS),
            "dtype": "uint8",
        },
        "context_box_l0_zyx": [3776, 4160, 3648, 4032, 1280, 1664],
        "grid_box_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "context_array_sha256": RAW_CONTEXT_ARRAY_SHA256,
        "context_file": file_record(context_path, "context.npy"),
        "cubes": cubes,
    }
    receipt["content_sha256"] = content_hash(receipt)
    _create_bytes(output / "raw_carve_receipt.json", _json_bytes(receipt))
    return verify_raw_carve(output)


def _require_raw_receipt_header(receipt: dict) -> None:
    expected_header = {
        "schema_version": RAW_CARVE_SCHEMA,
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "source": {
            "uri": RAW_SOURCE_URI,
            "level": 0,
            "shape": list(RAW_SOURCE_SHAPE),
            "chunks": list(RAW_SOURCE_CHUNKS),
            "dtype": "uint8",
        },
        "context_box_l0_zyx": [3776, 4160, 3648, 4032, 1280, 1664],
        "grid_box_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "context_array_sha256": RAW_CONTEXT_ARRAY_SHA256,
    }
    for key, expected in expected_header.items():
        if receipt.get(key) != expected:
            raise ValueError(f"RAW carve receipt has invalid {key}")


def verify_raw_carve(output: Path) -> dict:
    """Verify the public-source fingerprint and every derived RAW cube byte."""
    validate_downstream_lock()
    output = Path(output)
    receipt, _ = _load_hashed_json(
        output / "raw_carve_receipt.json", "RAW carve receipt"
    )
    _require_raw_receipt_header(receipt)
    context_path = output / "context.npy"
    if receipt.get("context_file") != file_record(context_path, "context.npy"):
        raise ValueError("RAW carve context file record mismatch")
    context = _load_locked_context(context_path)
    central = context[64:320, 64:320, 64:320]
    records = receipt.get("cubes")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("RAW carve must bind exactly eight cubes")
    by_path = {record.get("path"): record for record in records}
    expected_paths = {
        f"cubes_RAW/{name}" for _, name in _cube_specs()
    }
    if set(by_path) != expected_paths or len(by_path) != len(records):
        raise ValueError("RAW carve cube universe mismatch")
    for (lz, ly, lx), name in _cube_specs():
        relative = f"cubes_RAW/{name}"
        path = output / relative
        if by_path[relative] != file_record(path, relative):
            raise ValueError(f"RAW carve cube record mismatch: {name}")
        cube = tifffile.imread(path)
        expected = central[lz:lz + 128, ly:ly + 128, lx:lx + 128]
        if (
            cube.dtype != np.dtype(np.uint8)
            or not np.array_equal(cube, expected)
            or sha256_array(cube) != RAW_CUBE_ARRAY_SHA256[name]
        ):
            raise ValueError(f"RAW cube differs from locked CT context: {name}")
    expected_files = expected_paths | {"context.npy", "raw_carve_receipt.json"}
    if _regular_file_universe(output) != expected_files:
        raise ValueError("RAW carve exact file universe mismatch")
    return receipt


def _require_exact_file(path: Path, payload: bytes) -> None:
    if not path.is_file() or path.read_bytes() != payload:
        raise ValueError(f"existing metadata differs from the locked export: {path}")


def _metadata(origin: tuple[int, int, int]) -> dict[Path, bytes]:
    root_attrs = {
        "multiscales": [{
            "version": "0.4",
            "name": "crossscan_surface_probability",
            "axes": [{"name": axis, "type": "space"} for axis in AXES],
            "datasets": [{
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]},
                    {"type": "translation", "translation": [float(v) for v in origin]},
                ],
            }],
        }],
        "crossscan": {
            "schema_version": SCHEMA,
            "quantity": "recto_surface_probability",
            "world_origin_l0_zyx": list(origin),
        },
    }
    zarray = {
        "zarr_format": 2,
        "shape": list(SHAPE),
        "chunks": list(CHUNKS),
        "dtype": "<f4",
        "compressor": None,
        "fill_value": 0.0,
        "order": "C",
        "filters": None,
    }
    return {
        Path(".zgroup"): _json_bytes({"zarr_format": 2}),
        Path(".zattrs"): _json_bytes(root_attrs),
        Path("0/.zarray"): _json_bytes(zarray),
        Path("0/.zattrs"): _json_bytes({"_ARRAY_DIMENSIONS": list(AXES)}),
    }


def _normalise_probability(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability)
    if value.shape != SHAPE:
        raise ValueError(f"probability shape must be {SHAPE}, got {value.shape}")
    if not np.issubdtype(value.dtype, np.floating):
        raise ValueError("probability input must be floating point")
    value = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(value).all():
        raise ValueError("probability input contains non-finite values")
    low = float(value.min())
    high = float(value.max())
    if low < 0.0 or high > 1.0:
        raise ValueError(f"probabilities must be in [0, 1], got [{low}, {high}]")
    return value


def _load_inference_probability(path: Path) -> np.ndarray:
    """Load the receipt format without the coercions allowed by the generic CLI."""
    value = np.load(Path(path), allow_pickle=False)
    if not isinstance(value, np.ndarray):
        raise ValueError("inference probability.npy must contain one ndarray")
    if value.shape != SHAPE or value.dtype != np.dtype(np.float32):
        raise ValueError(
            f"inference probability.npy must be float32 {SHAPE}, got {value.dtype} {value.shape}"
        )
    if not value.flags.c_contiguous:
        raise ValueError("inference probability.npy must be C-contiguous")
    if not np.isfinite(value).all() or float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError("inference probability.npy contains invalid probabilities")
    return value


def _chunk_records(probability: np.ndarray) -> Iterable[tuple[tuple[int, int, int], bytes]]:
    for iz, z0 in enumerate(range(0, SHAPE[0], CHUNKS[0])):
        for iy, y0 in enumerate(range(0, SHAPE[1], CHUNKS[1])):
            for ix, x0 in enumerate(range(0, SHAPE[2], CHUNKS[2])):
                chunk = np.ascontiguousarray(
                    probability[
                        z0:z0 + CHUNKS[0],
                        y0:y0 + CHUNKS[1],
                        x0:x0 + CHUNKS[2],
                    ],
                    dtype="<f4",
                )
                yield (iz, iy, ix), memoryview(chunk).cast("B").tobytes()


def _reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


def _gpu_peak_bytes() -> int | None:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        pass
    return None


def export_probability_zarr(
    inference_run: Path,
    output: Path,
    *,
    origin: tuple[int, int, int] = DEFAULT_ORIGIN,
    model_source: Path,
    raw_carve: Path,
    resume: bool = False,
) -> dict:
    """Write an uncompressed OME-NGFF Zarr v2 ROI with chunk receipts."""
    validate_downstream_lock()
    origin = _require_locked_origin(origin)
    inference_run = Path(inference_run)
    _require_output_disjoint(output, (inference_run, model_source, raw_carve))
    inference, value = validate_inference_run(
        inference_run, model_source, raw_carve
    )
    model_role = inference["model_role"]
    inference_payload = (inference_run / "inference_receipt.json").read_bytes()
    source = _artifact_record(
        "provenance/inference_receipt.json", inference_payload, inference
    )
    provenance_payloads = {
        "provenance/inference_receipt.json": inference_payload,
    }
    for relative in _regular_file_universe(inference_run):
        if relative in ("inference_receipt.json", "probability.npy"):
            continue
        provenance_payloads[relative] = (inference_run / relative).read_bytes()
    started = time.perf_counter()
    tracemalloc.start()
    _reset_gpu_peak()
    value = _normalise_probability(value)
    input_sha = sha256_array(value)
    output = Path(output)
    metadata = _metadata(origin)
    bytes_written = 0

    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to replace existing output: {output}")
        if not output.is_dir():
            raise ValueError(f"output is not a directory: {output}")
        if (output / "export_receipt.json").exists():
            raise FileExistsError(f"completed output is immutable: {output}")
        for relative, payload in metadata.items():
            _require_exact_file(output / relative, payload)
        for relative, payload in provenance_payloads.items():
            _require_exact_file(output / relative, payload)
    else:
        output.mkdir(parents=True, exist_ok=False)
        for relative, payload in metadata.items():
            _create_bytes(output / relative, payload)
            bytes_written += len(payload)
        for relative, payload in provenance_payloads.items():
            _create_bytes(output / relative, payload)
            bytes_written += len(payload)

    expected_chunk_names = {
        f"{iz}.{iy}.{ix}"
        for iz in range(2) for iy in range(2) for ix in range(2)
    }
    actual_chunk_names = {
        path.name for path in (output / "0").iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    unexpected = actual_chunk_names - expected_chunk_names
    if unexpected:
        raise ValueError(f"unexpected Zarr chunk payloads: {sorted(unexpected)}")
    receipts_dir = output / "receipts"
    actual_receipt_names = (
        {path.name for path in receipts_dir.iterdir() if path.is_file()}
        if receipts_dir.is_dir() else set()
    )
    expected_receipt_names = {f"{name}.json" for name in expected_chunk_names}
    unexpected_receipts = actual_receipt_names - expected_receipt_names
    if unexpected_receipts:
        raise ValueError(f"unexpected chunk receipts: {sorted(unexpected_receipts)}")
    allowed_partial = {
        relative.as_posix() for relative in metadata
    } | set(provenance_payloads) | {
        f"0/{name}" for name in expected_chunk_names
    } | {
        f"receipts/{name}.json" for name in expected_chunk_names
    }
    extras = _regular_file_universe(output) - allowed_partial
    if extras:
        raise ValueError(f"unexpected files in partial probability export: {sorted(extras)}")

    records = []
    resumed_chunks = 0
    for index, payload in _chunk_records(value):
        name = ".".join(str(v) for v in index)
        relative = f"0/{name}"
        chunk_path = output / relative
        receipt_path = output / "receipts" / f"{name}.json"
        expected_sha = hashlib.sha256(payload).hexdigest()
        record = {
            "index_zyx": list(index),
            "path": relative,
            "bytes": len(payload),
            "sha256": expected_sha,
            "input_probability_sha256": input_sha,
        }
        record["content_sha256"] = content_hash(record)
        receipt_payload = _json_bytes(record)
        chunk_exists = chunk_path.exists()
        receipt_exists = receipt_path.exists()
        if chunk_exists or receipt_exists:
            if not resume or not (chunk_exists and receipt_exists):
                raise ValueError(f"chunk/receipt pair is incomplete or resume is disabled: {name}")
            if receipt_path.read_bytes() != receipt_payload:
                raise ValueError(f"chunk receipt differs from current input: {receipt_path}")
            if chunk_path.stat().st_size != len(payload) or sha256_file(chunk_path) != expected_sha:
                raise ValueError(f"existing chunk does not match its receipt: {chunk_path}")
            resumed_chunks += 1
        else:
            _create_bytes(chunk_path, payload)
            _create_bytes(receipt_path, receipt_payload)
            bytes_written += len(payload) + len(receipt_payload)
        records.append(record)

    roundtrip, roundtrip_origin = read_probability_zarr(output)
    if roundtrip_origin != origin or not np.array_equal(roundtrip, value):
        raise RuntimeError("OME-Zarr round-trip verification failed")
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    receipt = {
        "schema_version": SCHEMA,
        "artifact_type": "probability-export",
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "model_role": model_role,
        "format": "OME-NGFF Zarr v2",
        "axes": list(AXES),
        "shape": list(SHAPE),
        "chunks": list(CHUNKS),
        "dtype": "float32",
        "world_origin_l0_zyx": list(origin),
        "coordinate_scale_l0_zyx": [1.0, 1.0, 1.0],
        "input_probability_sha256": input_sha,
        "inference_receipt": source,
        "chunk_records": records,
        "roundtrip_equal": True,
        "resource_measurements": {
            "wall_seconds": time.perf_counter() - started,
            "python_tracemalloc_peak_bytes_since_export_start": int(peak_ram),
            "cuda_peak_allocated_bytes_since_export_start": _gpu_peak_bytes(),
            "bytes_written_this_invocation": 0,
            "resumed_chunks": resumed_chunks,
        },
    }
    receipt_payload = b""
    for _ in range(4):
        receipt["content_sha256"] = content_hash(receipt)
        receipt_payload = _json_bytes(receipt)
        total_written = bytes_written + len(receipt_payload)
        if receipt["resource_measurements"]["bytes_written_this_invocation"] == total_written:
            break
        receipt["resource_measurements"]["bytes_written_this_invocation"] = total_written
    receipt["content_sha256"] = content_hash(receipt)
    receipt_payload = _json_bytes(receipt)
    _create_bytes(output / "export_receipt.json", receipt_payload)
    return receipt


def verify_probability_export(store: Path) -> dict:
    """Verify the immutable final receipt, metadata, and every chunk byte."""
    validate_downstream_lock()
    store = Path(store)
    receipt_path = store / "export_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("content_sha256") != content_hash(receipt):
        raise ValueError("probability export receipt content hash mismatch")
    expected_header = {
        "schema_version": SCHEMA,
        "artifact_type": "probability-export",
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "format": "OME-NGFF Zarr v2",
        "axes": list(AXES),
        "shape": list(SHAPE),
        "chunks": list(CHUNKS),
        "dtype": "float32",
        "coordinate_scale_l0_zyx": [1.0, 1.0, 1.0],
    }
    for key, expected in expected_header.items():
        if receipt.get(key) != expected:
            raise ValueError(f"probability export receipt has invalid {key}")
    if receipt.get("model_role") not in ("baseline", "candidate"):
        raise ValueError("probability export receipt has invalid model role")
    origin = _require_locked_origin(tuple(receipt.get("world_origin_l0_zyx", ())))
    for relative, payload in _metadata(origin).items():
        _require_exact_file(store / relative, payload)
    inference, provenance_files = _validate_embedded_inference(
        store, receipt.get("inference_receipt", {})
    )
    if (
        inference.get("model_role") != receipt.get("model_role")
        or inference.get("probability_array_sha256")
        != receipt.get("input_probability_sha256")
    ):
        raise ValueError("probability export differs from locked inference receipt")

    records = receipt.get("chunk_records")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("probability export receipt must bind exactly eight chunks")
    expected_names = {
        f"{iz}.{iy}.{ix}"
        for iz in range(2) for iy in range(2) for ix in range(2)
    }
    actual_names = {
        path.name for path in (store / "0").iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    if actual_names != expected_names:
        raise ValueError("probability export chunk universe mismatch")
    actual_receipts = {
        path.name for path in (store / "receipts").iterdir() if path.is_file()
    }
    if actual_receipts != {f"{name}.json" for name in expected_names}:
        raise ValueError("probability export chunk-receipt universe mismatch")
    expected_files = {
        relative.as_posix() for relative in _metadata(origin)
    } | {
        "export_receipt.json",
    } | provenance_files | {
        f"0/{name}" for name in expected_names
    } | {
        f"receipts/{name}.json" for name in expected_names
    }
    actual_files = _regular_file_universe(store)
    if actual_files != expected_files:
        raise ValueError("probability export exact file universe mismatch")

    seen = set()
    for record in records:
        if record.get("content_sha256") != content_hash(record):
            raise ValueError("chunk record content hash mismatch")
        index = tuple(record.get("index_zyx", ()))
        if len(index) != 3 or any(v not in (0, 1) for v in index):
            raise ValueError("invalid chunk index in receipt")
        name = ".".join(str(v) for v in index)
        if name in seen or record.get("path") != f"0/{name}":
            raise ValueError("duplicate or invalid chunk path in receipt")
        seen.add(name)
        chunk = store / record["path"]
        sidecar = store / "receipts" / f"{name}.json"
        if json.loads(sidecar.read_text(encoding="utf-8")) != record:
            raise ValueError(f"chunk sidecar differs from final receipt: {name}")
        if chunk.stat().st_size != record.get("bytes"):
            raise ValueError(f"chunk byte count mismatch: {name}")
        if sha256_file(chunk) != record.get("sha256"):
            raise ValueError(f"chunk hash mismatch: {name}")
        if record.get("input_probability_sha256") != receipt.get(
            "input_probability_sha256"
        ):
            raise ValueError(f"chunk input binding mismatch: {name}")
    if seen != expected_names:
        raise ValueError("final receipt does not cover the exact chunk universe")
    probability, read_origin = read_probability_zarr(store)
    if read_origin != origin or sha256_array(probability) != receipt.get(
        "input_probability_sha256"
    ):
        raise ValueError("probability round-trip differs from final receipt")
    return receipt


def read_probability_zarr(store: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Read and minimally validate the adapter's deterministic Zarr layout."""
    store = Path(store)
    zarray = json.loads((store / "0/.zarray").read_text(encoding="utf-8"))
    if zarray != json.loads(_metadata(DEFAULT_ORIGIN)[Path("0/.zarray")]):
        raise ValueError("unsupported or altered probability Zarr array metadata")
    attrs = json.loads((store / ".zattrs").read_text(encoding="utf-8"))
    dataset = attrs["multiscales"][0]["datasets"][0]
    transforms = dataset["coordinateTransformations"]
    if transforms[0] != {"type": "scale", "scale": [1.0, 1.0, 1.0]}:
        raise ValueError("probability Zarr scale differs from level 0")
    translation = tuple(int(v) for v in transforms[1]["translation"])
    if list(translation) != transforms[1]["translation"]:
        raise ValueError("probability Zarr translation must be integral")
    _require_locked_origin(translation)
    result = np.empty(SHAPE, dtype="<f4")
    expected_bytes = int(np.prod(CHUNKS)) * np.dtype("<f4").itemsize
    for index, _ in _chunk_records(result):
        name = ".".join(str(v) for v in index)
        path = store / "0" / name
        payload = path.read_bytes()
        if len(payload) != expected_bytes:
            raise ValueError(f"invalid chunk byte count: {path}")
        chunk = np.frombuffer(payload, dtype="<f4").reshape(CHUNKS)
        z0, y0, x0 = (index[i] * CHUNKS[i] for i in range(3))
        result[
            z0:z0 + CHUNKS[0],
            y0:y0 + CHUNKS[1],
            x0:x0 + CHUNKS[2],
        ] = chunk
    return result, translation


def fixed_threshold_mask(probability: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    value = _normalise_probability(probability)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return np.ascontiguousarray(value >= threshold, dtype=np.uint8) * np.uint8(255)


def matched_mass_mask(probability: np.ndarray, foreground_count: int) -> np.ndarray:
    """Select exactly N voxels by probability, then C-order index for ties."""
    value = _normalise_probability(probability)
    total = value.size
    if not 0 <= foreground_count <= total:
        raise ValueError(f"foreground_count must be in [0, {total}]")
    flat = value.reshape(-1)
    selected = np.zeros(total, dtype=np.uint8)
    if foreground_count:
        # Stable sort preserves the original C-order index for equal values.
        order = np.argsort(-flat, kind="stable")
        selected[order[:foreground_count]] = np.uint8(255)
    return selected.reshape(SHAPE)


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _cube_specs(origin: tuple[int, int, int] = DEFAULT_ORIGIN):
    origin = _require_locked_origin(origin)
    for lz in (0, 128):
        for ly in (0, 128):
            for lx in (0, 128):
                world = (origin[0] + lz, origin[1] + ly, origin[2] + lx)
                name = f"z{world[0]:05d}_y{world[1]:05d}_x{world[2]:05d}.tif"
                yield (lz, ly, lx), name


def _require_grid_header(manifest: dict) -> str:
    required = {
        "schema_version": SCHEMA,
        "artifact_type": "scrollfiesta-grid",
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "chunk_size": 128,
        "bbox_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "n_chunks_zyx": [2, 2, 2],
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"grid manifest has invalid {key}")
    arm = manifest.get("arm")
    if arm not in GRID_ARMS:
        raise ValueError("grid manifest has invalid arm")
    expected_role = "baseline" if arm == "baseline-fixed" else "candidate"
    return expected_role


def materialize_scrollfiesta_grid(
    probability_store: Path,
    raw_carve: Path,
    output: Path,
    *,
    arm: str,
    baseline_fixed_manifest: Path | None = None,
) -> dict:
    """Write one of the three frozen native ScrollFiesta grid arms."""
    validate_downstream_lock()
    if arm not in GRID_ARMS:
        raise ValueError(f"arm must be one of {GRID_ARMS}")
    probability_store = Path(probability_store)
    disjoint_inputs = [probability_store, Path(raw_carve)]
    if baseline_fixed_manifest is not None:
        disjoint_inputs.append(Path(baseline_fixed_manifest).parent)
    _require_output_disjoint(output, disjoint_inputs)
    probability_receipt = verify_probability_export(probability_store)
    expected_role = "baseline" if arm == "baseline-fixed" else "candidate"
    if probability_receipt.get("model_role") != expected_role:
        raise ValueError(f"{arm} requires a {expected_role} probability export")
    probability, origin = read_probability_zarr(probability_store)
    _require_locked_origin(origin)

    baseline_record = None
    baseline_payload = None
    if arm == "candidate-matched-mass":
        if baseline_fixed_manifest is None:
            raise ValueError("candidate-matched-mass requires --baseline-fixed-manifest")
        baseline_fixed_manifest = Path(baseline_fixed_manifest)
        if baseline_fixed_manifest.resolve() != (
            baseline_fixed_manifest.parent / "manifest.json"
        ).resolve():
            raise ValueError("baseline fixed manifest must be named manifest.json")
        baseline_grid = verify_scrollfiesta_grid(baseline_fixed_manifest.parent)
        if baseline_grid.get("arm") != "baseline-fixed":
            raise ValueError("matched-mass provenance is not a baseline-fixed grid")
        baseline_payload = baseline_fixed_manifest.read_bytes()
        baseline_record = _artifact_record(
            "provenance/baseline_fixed_manifest.json", baseline_payload, baseline_grid
        )
        foreground_count = baseline_grid["foreground_voxels"]
        mask = matched_mass_mask(probability, foreground_count)
        mask_rule = {
            "name": "matched-mass",
            "foreground_count": foreground_count,
            "selection": "stable descending probability then C-order index",
            "baseline_fixed_manifest": baseline_record,
        }
    else:
        if baseline_fixed_manifest is not None:
            raise ValueError("a baseline manifest is accepted only for candidate-matched-mass")
        mask = fixed_threshold_mask(probability, FIXED_THRESHOLD)
        mask_rule = {
            "name": "fixed-threshold",
            "comparison": ">=",
            "threshold": FIXED_THRESHOLD,
        }

    raw_carve = Path(raw_carve)
    raw_receipt = verify_raw_carve(raw_carve)
    raw_receipt_payload = (raw_carve / "raw_carve_receipt.json").read_bytes()
    raw_receipt_record = _artifact_record(
        "provenance/raw_carve_receipt.json", raw_receipt_payload, raw_receipt
    )
    raw_cube_dir = raw_carve / "cubes_RAW"
    expected_names = {name for _, name in _cube_specs(origin)}
    if not raw_cube_dir.is_dir():
        raise FileNotFoundError(raw_cube_dir)
    raw_names = set()
    for path in raw_cube_dir.iterdir():
        if path.is_symlink():
            raise ValueError(f"RAW cube symlink is forbidden: {path}")
        if path.is_file():
            raw_names.add(path.name)
    if raw_names != expected_names:
        raise ValueError("RAW input must contain exactly the locked eight-cube universe")
    for name in sorted(expected_names):
        raw_check = tifffile.imread(raw_cube_dir / name)
        if (
            raw_check.shape != (128, 128, 128)
            or raw_check.dtype != np.uint8
            or sha256_array(raw_check) != RAW_CUBE_ARRAY_SHA256[name]
        ):
            raise ValueError(f"RAW cube is not uint8 128^3: {raw_cube_dir / name}")

    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing grid: {output}")
    output.mkdir(parents=True, exist_ok=False)
    pred_dir = output / "cubes_PRED"
    raw_dir = output / "cubes_RAW"
    pred_dir.mkdir()
    raw_dir.mkdir()

    files = []
    for (lz, ly, lx), name in _cube_specs(origin):
        cube = np.ascontiguousarray(mask[lz:lz + 128, ly:ly + 128, lx:lx + 128])
        pred_path = pred_dir / name
        tifffile.imwrite(
            pred_path,
            cube,
            photometric="minisblack",
            compression=None,
            rowsperstrip=128,
        )
        raw_path = raw_dir / name
        _copy_exclusive(raw_cube_dir / name, raw_path)
        for role, path in (("PRED", pred_path), ("RAW", raw_path)):
            files.append({
                "role": role,
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })

    probability_payload = (probability_store / "export_receipt.json").read_bytes()
    probability_record = _artifact_record(
        "provenance/probability_export_receipt.json",
        probability_payload,
        probability_receipt,
    )
    _create_bytes(output / probability_record["path"], probability_payload)
    probability_provenance = {
        f"provenance/{path.relative_to(probability_store / 'provenance').as_posix()}":
        path.read_bytes()
        for path in (probability_store / "provenance").rglob("*")
        if path.is_file()
    }
    if probability_provenance.get(raw_receipt_record["path"]) != raw_receipt_payload:
        raise ValueError("probability inference and grid use different RAW carves")
    for relative, payload in probability_provenance.items():
        _create_bytes(output / relative, payload)
    if baseline_record is not None and baseline_payload is not None:
        _create_bytes(output / baseline_record["path"], baseline_payload)

    manifest = {
        "schema_version": SCHEMA,
        "artifact_type": "scrollfiesta-grid",
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "arm": arm,
        "mask_rule": mask_rule,
        "chunk_size": 128,
        "bbox_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "n_chunks_zyx": [2, 2, 2],
        "foreground_voxels": int(np.count_nonzero(mask)),
        "probability_export_receipt": probability_record,
        "raw_carve_receipt": raw_receipt_record,
        "raw_context_array_sha256": RAW_CONTEXT_ARRAY_SHA256,
        "files": files,
    }
    manifest["content_sha256"] = content_hash(manifest)
    _create_bytes(output / "manifest.json", _json_bytes(manifest))
    return verify_scrollfiesta_grid(output)


def verify_scrollfiesta_grid(output: Path) -> dict:
    """Verify a completed grid, its provenance, exact TIFF universe, and hashes."""
    validate_downstream_lock()
    output = Path(output)
    manifest, _ = _load_hashed_json(output / "manifest.json", "grid manifest")
    expected_role = _require_grid_header(manifest)
    export_receipt = _validate_embedded_artifact(
        output, manifest.get("probability_export_receipt", {}),
        "embedded probability export receipt",
    )
    if (
        export_receipt.get("schema_version") != SCHEMA
        or export_receipt.get("artifact_type") != "probability-export"
        or export_receipt.get("status") != "PASS"
        or export_receipt.get("downstream_lock_content_sha256")
        != DOWNSTREAM_LOCK_CONTENT_SHA256
        or export_receipt.get("model_role") != expected_role
    ):
        raise ValueError("embedded probability export receipt is not the locked source")
    inference, inference_provenance = _validate_embedded_inference(
        output, export_receipt.get("inference_receipt", {})
    )
    if (
        inference.get("model_role") != expected_role
        or inference.get("probability_array_sha256")
        != export_receipt.get("input_probability_sha256")
    ):
        raise ValueError("grid inference provenance differs from probability export")
    raw_receipt = _validate_embedded_artifact(
        output, manifest.get("raw_carve_receipt", {}),
        "embedded RAW carve receipt",
    )
    _require_raw_receipt_header(raw_receipt)
    if manifest.get("raw_context_array_sha256") != RAW_CONTEXT_ARRAY_SHA256:
        raise ValueError("grid RAW context identity mismatch")

    arm = manifest["arm"]
    rule = manifest.get("mask_rule")
    extra_provenance = set()
    if arm in ("baseline-fixed", "candidate-fixed"):
        if rule != {
            "name": "fixed-threshold",
            "comparison": ">=",
            "threshold": FIXED_THRESHOLD,
        }:
            raise ValueError("fixed grid does not use the locked inclusive 0.2 rule")
    else:
        if not isinstance(rule, dict) or rule.get("name") != "matched-mass":
            raise ValueError("matched grid has invalid rule")
        baseline_record = rule.get("baseline_fixed_manifest", {})
        baseline = _validate_embedded_artifact(
            output, baseline_record, "embedded baseline fixed manifest"
        )
        if _require_grid_header(baseline) != "baseline" or baseline.get("arm") != "baseline-fixed":
            raise ValueError("matched grid baseline provenance is not baseline-fixed")
        if baseline.get("mask_rule") != {
            "name": "fixed-threshold",
            "comparison": ">=",
            "threshold": FIXED_THRESHOLD,
        }:
            raise ValueError("matched grid baseline did not use the locked fixed rule")
        if rule.get("foreground_count") != baseline.get("foreground_voxels"):
            raise ValueError("matched grid foreground count differs from baseline manifest")
        if rule.get("selection") != "stable descending probability then C-order index":
            raise ValueError("matched grid selection rule mismatch")
        extra_provenance.add(baseline_record["path"])

    expected_names = {name for _, name in _cube_specs()}
    expected_records = {
        (role, f"cubes_{role}/{name}")
        for role in ("PRED", "RAW") for name in expected_names
    }
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != 16:
        raise ValueError("grid manifest must bind exactly sixteen TIFF files")
    actual_records = {(record.get("role"), record.get("path")) for record in records}
    if actual_records != expected_records or len(actual_records) != len(records):
        raise ValueError("grid manifest TIFF universe mismatch")
    foreground = 0
    raw_record_map = {
        record["path"].removeprefix("cubes_RAW/"): record
        for record in raw_receipt.get("cubes", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    if set(raw_record_map) != expected_names:
        raise ValueError("embedded RAW receipt cube universe mismatch")
    for record in records:
        path = output / record["path"]
        if path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"grid TIFF hash mismatch: {record['path']}")
        value = tifffile.imread(path)
        if value.shape != (128, 128, 128) or value.dtype != np.uint8:
            raise ValueError(f"grid TIFF is not uint8 128^3: {record['path']}")
        if record["role"] == "PRED":
            if not np.isin(value, np.array([0, 255], dtype=np.uint8)).all():
                raise ValueError(f"PRED TIFF is not binary: {record['path']}")
            foreground += int(np.count_nonzero(value))
        else:
            raw_expected = raw_record_map[Path(record["path"]).name]
            if (
                record.get("bytes") != raw_expected.get("bytes")
                or record.get("sha256") != raw_expected.get("sha256")
                or sha256_array(value)
                != RAW_CUBE_ARRAY_SHA256[Path(record["path"]).name]
            ):
                raise ValueError(f"RAW grid file differs from carve: {record['path']}")
    if foreground != manifest.get("foreground_voxels"):
        raise ValueError("grid foreground census differs from manifest")
    if arm == "candidate-matched-mass" and foreground != rule["foreground_count"]:
        raise ValueError("matched grid does not have exact baseline foreground mass")

    expected_files = {path for _, path in expected_records} | {
        "manifest.json",
        manifest["probability_export_receipt"]["path"],
    } | inference_provenance | extra_provenance
    if _regular_file_universe(output) != expected_files:
        raise ValueError("grid exact file universe mismatch")
    return manifest


def _manifest_record(label: str, path: Path, value: dict) -> dict:
    payload = Path(path).read_bytes()
    return _artifact_record(label, payload, value)


def _assert_grid_prediction(grid: Path, expected: np.ndarray) -> None:
    for (lz, ly, lx), name in _cube_specs():
        actual = tifffile.imread(Path(grid) / "cubes_PRED" / name)
        wanted = expected[lz:lz + 128, ly:ly + 128, lx:lx + 128]
        if not np.array_equal(actual, wanted):
            raise ValueError(f"grid PRED TIFF differs from recomputed mask: {name}")


def verify_grid_set(
    baseline_probability: Path,
    candidate_probability: Path,
    baseline_fixed_grid: Path,
    candidate_fixed_grid: Path,
    candidate_matched_grid: Path,
    output: Path,
) -> dict:
    """Verify the complete three-arm experiment as one inseparable artifact."""
    baseline_probability = Path(baseline_probability)
    candidate_probability = Path(candidate_probability)
    baseline_fixed_grid = Path(baseline_fixed_grid)
    candidate_fixed_grid = Path(candidate_fixed_grid)
    candidate_matched_grid = Path(candidate_matched_grid)
    _require_output_disjoint(
        output,
        (
            baseline_probability,
            candidate_probability,
            baseline_fixed_grid,
            candidate_fixed_grid,
            candidate_matched_grid,
        ),
    )
    baseline_export = verify_probability_export(baseline_probability)
    candidate_export = verify_probability_export(candidate_probability)
    baseline_manifest = verify_scrollfiesta_grid(baseline_fixed_grid)
    fixed_manifest = verify_scrollfiesta_grid(candidate_fixed_grid)
    matched_manifest = verify_scrollfiesta_grid(candidate_matched_grid)
    if (
        baseline_export.get("model_role") != "baseline"
        or candidate_export.get("model_role") != "candidate"
        or baseline_manifest.get("arm") != "baseline-fixed"
        or fixed_manifest.get("arm") != "candidate-fixed"
        or matched_manifest.get("arm") != "candidate-matched-mass"
    ):
        raise ValueError("grid set does not contain the exact three frozen arms")
    expected_probability_links = (
        (
            baseline_manifest,
            baseline_export,
            baseline_probability / "export_receipt.json",
        ),
        (
            fixed_manifest,
            candidate_export,
            candidate_probability / "export_receipt.json",
        ),
        (
            matched_manifest,
            candidate_export,
            candidate_probability / "export_receipt.json",
        ),
    )
    for manifest, export, source in expected_probability_links:
        expected = _artifact_record(
            "provenance/probability_export_receipt.json",
            source.read_bytes(),
            export,
        )
        if manifest.get("probability_export_receipt") != expected:
            raise ValueError("grid links to a different probability export")
    baseline_inference, _ = _validate_embedded_inference(
        baseline_probability, baseline_export["inference_receipt"]
    )
    candidate_inference, _ = _validate_embedded_inference(
        candidate_probability, candidate_export["inference_receipt"]
    )
    for field in (
        "raw_carve_receipt", "promotion_receipt", "execution_lock", "semantic_audit"
    ):
        baseline_record = baseline_inference.get(field, {})
        candidate_record = candidate_inference.get(field, {})
        if (
            baseline_record.get("sha256") != candidate_record.get("sha256")
            or baseline_record.get("content_sha256")
            != candidate_record.get("content_sha256")
        ):
            raise ValueError(f"baseline and candidate differ in {field}")
    baseline_tools = {
        record["path"]: (record["bytes"], record["sha256"])
        for record in baseline_inference["tooling"]
    }
    candidate_tools = {
        record["path"]: (record["bytes"], record["sha256"])
        for record in candidate_inference["tooling"]
    }
    if baseline_tools != candidate_tools:
        raise ValueError("baseline and candidate used different inference tooling bytes")
    if baseline_inference.get("runtime_identity") != candidate_inference.get(
        "runtime_identity"
    ):
        raise ValueError("baseline and candidate used different runtime or CUDA identities")
    raw_records = [
        baseline_manifest.get("raw_carve_receipt"),
        fixed_manifest.get("raw_carve_receipt"),
        matched_manifest.get("raw_carve_receipt"),
    ]
    expected_inference_raw = [
        baseline_inference["raw_carve_receipt"],
        candidate_inference["raw_carve_receipt"],
        candidate_inference["raw_carve_receipt"],
    ]
    if any(
        grid_record != inference_record
        for grid_record, inference_record in zip(raw_records, expected_inference_raw)
    ):
        raise ValueError("grid RAW receipt differs from its probability inference input")
    if any(record != raw_records[0] for record in raw_records[1:]):
        raise ValueError("three grids do not share one byte-identical RAW receipt")
    raw_maps = []
    for manifest in (baseline_manifest, fixed_manifest, matched_manifest):
        raw_maps.append({
            record["path"]: (record["bytes"], record["sha256"])
            for record in manifest["files"] if record["role"] == "RAW"
        })
    if any(mapping != raw_maps[0] for mapping in raw_maps[1:]):
        raise ValueError("three grids do not share byte-identical RAW TIFFs")
    embedded_baseline = (
        candidate_matched_grid / "provenance/baseline_fixed_manifest.json"
    )
    if embedded_baseline.read_bytes() != (baseline_fixed_grid / "manifest.json").read_bytes():
        raise ValueError("matched-mass grid embeds a different baseline manifest")

    baseline_value, _ = read_probability_zarr(baseline_probability)
    candidate_value, _ = read_probability_zarr(candidate_probability)
    baseline_mask = fixed_threshold_mask(baseline_value, FIXED_THRESHOLD)
    candidate_fixed_mask = fixed_threshold_mask(candidate_value, FIXED_THRESHOLD)
    candidate_matched_mask = matched_mass_mask(
        candidate_value, int(np.count_nonzero(baseline_mask))
    )
    _assert_grid_prediction(baseline_fixed_grid, baseline_mask)
    _assert_grid_prediction(candidate_fixed_grid, candidate_fixed_mask)
    _assert_grid_prediction(candidate_matched_grid, candidate_matched_mask)
    if matched_manifest.get("foreground_voxels") != int(np.count_nonzero(baseline_mask)):
        raise ValueError("matched-mass N differs from recomputed baseline foreground")

    result = {
        "schema_version": "crossscan-scrollfiesta-grid-set-v1",
        "status": "PASS",
        "downstream_lock_content_sha256": DOWNSTREAM_LOCK_CONTENT_SHA256,
        "raw_context_array_sha256": RAW_CONTEXT_ARRAY_SHA256,
        "promotion_content_sha256": baseline_inference["promotion_receipt"][
            "content_sha256"
        ],
        "semantic_audit_content_sha256": baseline_inference["semantic_audit"][
            "content_sha256"
        ],
        "probability_exports": {
            "baseline": _manifest_record(
                "baseline_probability/export_receipt.json",
                baseline_probability / "export_receipt.json",
                baseline_export,
            ),
            "candidate": _manifest_record(
                "candidate_probability/export_receipt.json",
                candidate_probability / "export_receipt.json",
                candidate_export,
            ),
        },
        "grids": {
            "baseline-fixed": _manifest_record(
                "baseline_fixed_grid/manifest.json",
                baseline_fixed_grid / "manifest.json",
                baseline_manifest,
            ),
            "candidate-fixed": _manifest_record(
                "candidate_fixed_grid/manifest.json",
                candidate_fixed_grid / "manifest.json",
                fixed_manifest,
            ),
            "candidate-matched-mass": _manifest_record(
                "candidate_matched_grid/manifest.json",
                candidate_matched_grid / "manifest.json",
                matched_manifest,
            ),
        },
        "recomputed_masks_equal_all_pred_tiffs": True,
        "shared_raw_tiffs": True,
        "shared_context_promotion_and_semantic_audit": True,
        "shared_execution_runtime_and_tooling": True,
        "raw_cube_arrays_match_pinned_public_context": True,
    }
    result["content_sha256"] = content_hash(result)
    _create_bytes(Path(output), _json_bytes(result))
    return result


def load_probability(path: Path, key: str | None = None) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            names = list(archive.files)
            selected = key or (names[0] if len(names) == 1 else None)
            if selected is None or selected not in archive:
                raise ValueError(f"NPZ key is required; available keys: {names}")
            value = archive[selected]
    else:
        raise ValueError("probability input must be .npy or .npz")
    if value.shape == (2, *SHAPE):
        value = value[1]
    return _normalise_probability(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    carve = sub.add_parser("carve-raw")
    carve.add_argument("--output", type=Path, required=True)

    export = sub.add_parser("export-zarr")
    export.add_argument("--inference-run", type=Path, required=True)
    export.add_argument("--model-source", type=Path, required=True)
    export.add_argument("--raw-carve", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--resume", action="store_true")

    grid = sub.add_parser("make-grid")
    grid.add_argument("--probability-zarr", type=Path, required=True)
    grid.add_argument("--raw-carve", type=Path, required=True)
    grid.add_argument("--output", type=Path, required=True)
    grid.add_argument("--arm", choices=GRID_ARMS, required=True)
    grid.add_argument("--baseline-fixed-manifest", type=Path)

    grid_set = sub.add_parser("verify-grid-set")
    grid_set.add_argument("--baseline-probability-zarr", type=Path, required=True)
    grid_set.add_argument("--candidate-probability-zarr", type=Path, required=True)
    grid_set.add_argument("--baseline-fixed-grid", type=Path, required=True)
    grid_set.add_argument("--candidate-fixed-grid", type=Path, required=True)
    grid_set.add_argument("--candidate-matched-grid", type=Path, required=True)
    grid_set.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "carve-raw":
        result = carve_locked_raw(args.output)
    elif args.command == "export-zarr":
        result = export_probability_zarr(
            args.inference_run,
            args.output,
            model_source=args.model_source,
            raw_carve=args.raw_carve,
            resume=args.resume,
        )
    elif args.command == "make-grid":
        result = materialize_scrollfiesta_grid(
            args.probability_zarr,
            args.raw_carve,
            args.output,
            arm=args.arm,
            baseline_fixed_manifest=args.baseline_fixed_manifest,
        )
    else:
        result = verify_grid_set(
            args.baseline_probability_zarr,
            args.candidate_probability_zarr,
            args.baseline_fixed_grid,
            args.candidate_fixed_grid,
            args.candidate_matched_grid,
            args.output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
