#!/usr/bin/env python3
"""Run the two locked PHerc0139 probability arms with complete provenance."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import crossscan_scrollfiesta_adapter as A
import crossscan_finetune as C
import physical_normalization_ab as N
import predict_crossscan_probability_ensemble as P
import run_crossscan_finetune as R
import score_crossscan_finetune as S
import verify_physical_label_semantics as V


LOCAL_TOOL_MODULES = {
    "run_crossscan_scrollfiesta_inference.py": None,
    "crossscan_scrollfiesta_adapter.py": A,
    "run_crossscan_finetune.py": R,
    "predict_crossscan_probability_ensemble.py": P,
    "crossscan_finetune.py": C,
    "score_crossscan_finetune.py": S,
    "verify_physical_label_semantics.py": V,
    "physical_normalization_ab.py": N,
}


def _configure_runtime() -> dict:
    """Make both arms use the frozen non-compiled deterministic CUDA path."""
    os.environ["nnUNet_compile"] = "0"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("locked ScrollFiesta inference requires CUDA device 0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(False)
    actual = {
        "nnUNet_compile": os.environ.get("nnUNet_compile"),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    if actual != A.DETERMINISM_CONTRACT:
        raise RuntimeError("failed to establish the frozen inference determinism settings")
    return actual


def _runtime_identity(villa_root: Path) -> dict:
    villa_root = Path(villa_root).resolve()
    commit = str(R.git_output(villa_root, "rev-parse", "HEAD"))
    tree = str(R.git_output(
        villa_root, "rev-parse", "HEAD:segmentation/models/arch/nnunet"
    ))
    villa = {"commit": commit, "nnunet_tree": tree}
    if villa != A.PINNED_VILLA:
        raise ValueError("Villa/nnU-Net identity differs from the frozen execution lock")
    status = str(R.git_output(
        villa_root, "status", "--porcelain=v1", "--",
        "segmentation/models/arch/nnunet",
    ))
    if status:
        raise ValueError("the frozen Villa nnU-Net tree has working-copy changes")
    environment = R.runtime_environment(villa_root)
    if environment != A.PINNED_RUNTIME_ENVIRONMENT:
        raise ValueError("runtime packages differ from the frozen execution lock")

    import torch

    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    rows = [line.split(", ", 3) for line in query if line.strip()]
    row = next((fields for fields in rows if len(fields) == 4 and fields[0] == "0"), None)
    if row is None:
        raise RuntimeError("nvidia-smi did not identify CUDA device 0")
    identity = {
        "environment": environment,
        "villa": villa,
        "determinism": _configure_runtime(),
        "cuda_device": {
            "index": 0,
            "name": row[1],
            "uuid": row[2],
            "driver_version": row[3],
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": int(properties.total_memory),
        },
    }
    return A._require_runtime_identity(identity)


def _copy_hashed_json(source: Path, staging: Path, relative: str) -> tuple[dict, bytes]:
    value, payload = A._load_hashed_json(source, relative)
    A._create_bytes(staging / relative, payload)
    return A._artifact_record(relative, payload, value), payload


def _copy_semantic_audit(source: Path, staging: Path, relative: str) -> tuple[dict, bytes]:
    value, payload = A._validate_semantic_audit(source)
    A._create_bytes(staging / relative, payload)
    return A._artifact_record(relative, payload, value), payload


def _normalization(plans_path: Path) -> tuple[dict, dict]:
    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    normalization = plans.get("foreground_intensity_properties_per_channel", {}).get("0")
    if normalization != A.NORMALIZATION_CONTRACT:
        raise ValueError("model plans differ from the locked m7 CT normalization")
    return plans, normalization


def _retained_class_probability(logits) -> "object":
    import torch

    if tuple(logits.shape) != (2, *A.CONTEXT_SHAPE):
        raise RuntimeError(f"inference logits have invalid shape {tuple(logits.shape)}")
    starts = tuple(
        (context - retained) // 2
        for context, retained in zip(A.CONTEXT_SHAPE, A.SHAPE)
    )
    if any(
        context < retained or (context - retained) % 2
        for context, retained in zip(A.CONTEXT_SHAPE, A.SHAPE)
    ):
        raise RuntimeError("retained box is not a centered crop of the inference context")
    slices = tuple(slice(start, start + size) for start, size in zip(starts, A.SHAPE))
    return torch.softmax(logits, dim=0)[(1, *slices)]


def _baseline_probability(model_source: Path, context: np.ndarray) -> np.ndarray:
    import torch

    checkpoint = model_source / A.BASE_MODEL_FILES["checkpoint"]["path"]
    predictor, plans = R.load_network_for_inference(checkpoint, model_source)
    _configure_runtime()
    normalized = R.normalize_ct(context, plans)
    tensor = torch.from_numpy(normalized[None])
    logits = predictor.predict_sliding_window_return_logits(tensor).float().cpu()
    probability = _retained_class_probability(logits)
    result = probability.numpy().astype(np.float32, copy=False)
    if result.shape != A.SHAPE:
        raise RuntimeError("baseline retained probability has invalid shape")
    return np.ascontiguousarray(result)


def _candidate_probability(model_source: Path, context: np.ndarray) -> np.ndarray:
    import torch

    plans, _ = _normalization(model_source / "model" / "plans.json")
    normalized = R.normalize_ct(context, plans)
    tensor = torch.from_numpy(normalized[None])
    predictor = P.make_predictor("cuda", 0.5)
    predictor.initialize_from_trained_model_folder(
        str(model_source / "model"),
        use_folds=tuple(range(12)),
        checkpoint_name="checkpoint_final.pth",
    )
    if len(predictor.list_of_parameters) != 12:
        raise RuntimeError("candidate release did not load exactly twelve parameter sets")
    _configure_runtime()
    accumulator = np.zeros(A.SHAPE, dtype=np.float64)
    for index, parameters in enumerate(predictor.list_of_parameters, 1):
        network = getattr(predictor.network, "_orig_mod", predictor.network)
        network.load_state_dict(parameters, strict=True)
        logits = predictor.predict_sliding_window_return_logits(tensor).float().cpu()
        probability = _retained_class_probability(logits)
        accumulator += probability.numpy().astype(np.float64, copy=False)
        del logits, probability
        print(f"[{index}/12] locked candidate member", flush=True)
    result = np.ascontiguousarray(accumulator / 12.0, dtype=np.float32)
    if result.shape != A.SHAPE:
        raise RuntimeError("candidate retained probability has invalid shape")
    return result


def run(args: argparse.Namespace) -> dict:
    A.validate_downstream_lock()
    _configure_runtime()
    execution_lock, _ = A.validate_execution_lock(args.execution_lock.resolve())
    runtime_identity = _runtime_identity(args.villa_root.resolve())
    output = args.output.resolve()
    staging = output.with_name(output.name + ".tmp")
    A._require_output_disjoint(output, (args.model_source, args.raw_carve))
    if output.exists() or staging.exists():
        raise FileExistsError(f"inference output already exists: {output} or {staging}")
    raw_carve = args.raw_carve.resolve()
    raw_receipt = A.verify_raw_carve(raw_carve)
    context = A._load_locked_context(raw_carve / "context.npy")
    promotion, _ = A.validate_promotion_receipt(args.promotion_receipt.resolve())
    semantic, _ = A._validate_semantic_audit(args.semantic_audit.resolve())
    model_source = args.model_source.resolve()
    model_record = A.validate_model_source(
        args.model_role,
        model_source,
        promotion_content_sha256=promotion["content_sha256"],
        semantic_audit_content_sha256=semantic["content_sha256"],
    )
    plans_path = (
        model_source / "plans.json"
        if args.model_role == "baseline"
        else model_source / "model" / "plans.json"
    )
    _, normalization = _normalization(plans_path)

    staging.mkdir(parents=True)
    raw_record, _ = _copy_hashed_json(
        raw_carve / "raw_carve_receipt.json",
        staging,
        "provenance/raw_carve_receipt.json",
    )
    promotion_record, _ = _copy_hashed_json(
        args.promotion_receipt.resolve(), staging, "provenance/final_result.json"
    )
    execution_record, _ = _copy_hashed_json(
        args.execution_lock.resolve(), staging, "provenance/execution_lock.json"
    )
    semantic_record, _ = _copy_semantic_audit(
        args.semantic_audit.resolve(),
        staging,
        "provenance/physical_label_semantic_audit.json",
    )
    release_record = None
    if args.model_role == "candidate":
        release_record, _ = _copy_hashed_json(
            model_source / "release_manifest.json",
            staging,
            "provenance/release_manifest.json",
        )
    tooling = []
    if set(LOCAL_TOOL_MODULES) != set(A.INFERENCE_TOOL_NAMES):
        raise RuntimeError("local inference tooling closure differs from the verifier")
    for name in A.INFERENCE_TOOL_NAMES:
        module = LOCAL_TOOL_MODULES[name]
        source = Path(__file__).resolve() if module is None else Path(module.__file__).resolve()
        if source.name != name:
            raise RuntimeError(f"imported local module path mismatch: {name} -> {source}")
        relative = f"provenance/{source.name}"
        payload = source.read_bytes()
        A._create_bytes(staging / relative, payload)
        tooling.append(A.file_record(staging / relative, relative))

    started = time.perf_counter()
    tracemalloc.start()
    A._reset_gpu_peak()
    probability = (
        _baseline_probability(model_source, context)
        if args.model_role == "baseline"
        else _candidate_probability(model_source, context)
    )
    probability = A._normalise_probability(probability)
    probability_path = staging / "probability.npy"
    A._create_npy(probability_path, probability)
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    receipt = {
        "schema_version": A.INFERENCE_RUN_SCHEMA,
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
        "plan_content_sha256": A.PLAN_CONTENT_SHA256,
        "execution_lock_content_sha256": A.EXECUTION_LOCK_CONTENT_SHA256,
        "runtime_identity": runtime_identity,
        "model_role": args.model_role,
        "model_source": model_record,
        "context_box_l0_zyx": [3776, 4160, 3648, 4032, 1280, 1664],
        "retained_box_l0_zyx": [3840, 4096, 3712, 3968, 1344, 1600],
        "context_array_sha256": A.RAW_CONTEXT_ARRAY_SHA256,
        "probability_shape": list(A.SHAPE),
        "probability_dtype": "float32",
        "probability_array_sha256": A.sha256_array(probability),
        "probability_file": A.file_record(probability_path, "probability.npy"),
        "tile_step_size": 0.5,
        "gaussian": True,
        "mirroring": False,
        "normalization": normalization,
        "aggregation": (
            "single released m7 fold-0 class probability"
            if args.model_role == "baseline"
            else "float64 arithmetic mean of twelve class probabilities"
        ),
        "folds": [0] if args.model_role == "baseline" else list(range(12)),
        "raw_carve_receipt": raw_record,
        "promotion_receipt": promotion_record,
        "execution_lock": execution_record,
        "semantic_audit": semantic_record,
        "release_manifest": release_record,
        "tooling": tooling,
        "resource_measurements": {
            "wall_seconds": time.perf_counter() - started,
            "python_tracemalloc_peak_bytes_since_inference_start": int(peak_ram),
            "cuda_peak_allocated_bytes_since_inference_start": A._gpu_peak_bytes(),
        },
    }
    receipt["content_sha256"] = A.content_hash(receipt)
    A._create_bytes(staging / "inference_receipt.json", A._json_bytes(receipt))
    verified, recovered = A.validate_inference_run(staging, model_source, raw_carve)
    if not np.array_equal(recovered, probability):
        raise RuntimeError("locked inference round-trip mismatch")
    staging.replace(output)
    return verified


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--raw-carve", type=Path, required=True)
    parser.add_argument("--promotion-receipt", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
