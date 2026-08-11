#!/usr/bin/env python3
"""Run the preregistered released-artifact versus PR #1386 comparison.

This wrapper deliberately does not reinterpret the failed causal sentinel.  It verifies
that public failure, binds a separate operational protocol, and then reuses the frozen
64-block inference and scoring artifacts without requiring old-path reproduction.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import physical_normalization_ab as P
import run_physical_normalization_ab as R


PROTOCOL_ID = "physical_released_baseline_operational_v1"
IMPLEMENTATION_REVISION = 2
LOCK_STATUS = "preregistered_before_corrected_physical_outcomes"
OUTPUT_NAMESPACE = "physical_released_baseline_comparison_r2"
SOURCE_MANIFEST_CONTENT_SHA256 = (
    "567a18faa1c8ca7e743c9240133f4200e67e3085823dd4795c4518e3e0e65ac0"
)
FAILED_CAUSAL_STATUS = "closed_fail_closed"
HEAVY_WORK_NAMES = ("logits", "merged.zarr")


def load_protocol_lock(path: Path) -> dict[str, Any]:
    lock = P.load_json(path)
    recorded = lock.pop("content_sha256", None)
    actual = P.sha256_bytes(P.canonical_json(lock).encode("utf-8"))
    lock["content_sha256"] = recorded
    if recorded != actual:
        raise SystemExit(f"protocol lock content SHA mismatch: {recorded} != {actual}")
    expected = {
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": LOCK_STATUS,
        "output_namespace": OUTPUT_NAMESPACE,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
    }
    for key, value in expected.items():
        if lock.get(key) != value:
            raise SystemExit(f"protocol lock {key} mismatch: {lock.get(key)!r} != {value!r}")
    return lock


class CanonicalReleasedBinaryArray:
    """Read-only view mapping accepted uint8 binary encodings to literal {0,1}."""

    def __init__(self, array: Any):
        self._array = array

    def __getitem__(self, key: Any) -> np.ndarray:
        value = np.asarray(self._array[key])
        if value.dtype != np.uint8:
            raise RuntimeError(f"released baseline dtype changed: {value.dtype}")
        if not np.isin(value, [0, 1, 255]).all():
            observed = np.unique(value)
            raise RuntimeError(
                "released baseline contains non-binary uint8 values: "
                f"{observed[:16].tolist()}"
            )
        return (value != 0).astype(np.uint8)


def run_operational_block(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    block: dict[str, Any],
    env: dict[str, str],
) -> str:
    """Run the inherited block implementation with binary storage canonicalized."""

    original_open = R._open_remote_zarr

    def open_canonical(url: str) -> CanonicalReleasedBinaryArray:
        return CanonicalReleasedBinaryArray(original_open(url))

    R._open_remote_zarr = open_canonical
    try:
        return R.run_block(args, manifest, verification, block, env)
    finally:
        R._open_remote_zarr = original_open


def require_output_namespace(out_root: Path) -> Path:
    resolved = out_root.resolve()
    if resolved.name != OUTPUT_NAMESPACE:
        raise SystemExit(
            f"output root must end in {OUTPUT_NAMESPACE!r}, got {str(resolved)!r}"
        )
    return resolved


def verify_protocol_files(
    repo: Path,
    lock: dict[str, Any],
    source_manifest_path: Path,
) -> dict[str, Any]:
    for relative, expected in lock["protocol_files_sha256"].items():
        actual = P.sha256_file(repo / relative)
        if actual != expected:
            raise SystemExit(f"operational protocol drift: {relative} SHA {actual} != {expected}")

    manifest_file_sha = P.sha256_file(source_manifest_path)
    if manifest_file_sha != lock["source_manifest_file_sha256"]:
        raise SystemExit("source manifest file SHA changed")
    source_manifest = R.load_and_verify_manifest(source_manifest_path)
    if source_manifest["content_sha256"] != SOURCE_MANIFEST_CONTENT_SHA256:
        raise SystemExit("source manifest content SHA changed")

    causal_path = repo / lock["failed_causal_result_path"]
    causal_sha = P.sha256_file(causal_path)
    if causal_sha != lock["failed_causal_result_sha256"]:
        raise SystemExit("failed causal result SHA changed")
    causal = P.load_json(causal_path)
    if causal.get("status") != FAILED_CAUSAL_STATUS:
        raise SystemExit("causal protocol is not recorded as fail-closed")
    if causal.get("corrected_arm_run") is not False:
        raise SystemExit("causal result does not prove corrected_arm_run=false")
    if causal.get("manifest_content_sha256") != SOURCE_MANIFEST_CONTENT_SHA256:
        raise SystemExit("causal result refers to a different source manifest")
    failure = P.load_json(repo / lock["preoutcome_failure_path"])
    if failure.get("status") != "failed_before_model_inference":
        raise SystemExit("revision-1 failure was not pre-inference")
    if failure.get("corrected_probability_array_created") is not False:
        raise SystemExit("revision-1 failure does not prove that no corrected array existed")
    if failure.get("failed_attempt_receipt_sha256") != (
        "54b620f8d19d4292b1ee19aa3596f5480ffa30197805682c2fde309379b2df4a"
    ):
        raise SystemExit("revision-1 failed receipt binding changed")
    return {
        "source_manifest": source_manifest,
        "failed_causal_result": causal,
        "preoutcome_failure": failure,
    }


def _write_or_verify_protocol_receipt(
    out_root: Path,
    lock: dict[str, Any],
    public_freeze: dict[str, str],
) -> Path:
    path = out_root / "protocol_receipt.json"
    fixed = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "estimand": "exact_pr1386_pipeline_vs_released_binary_artifact",
        "causal_claim_allowed": False,
        "protocol_lock_content_sha256": lock["content_sha256"],
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "failed_causal_result_sha256": lock["failed_causal_result_sha256"],
        "public_freeze_commit": public_freeze["repo_head"],
        "public_branch": public_freeze["branch"],
        "corrected_villa_commit": P.PR1386_COMMIT,
        "status": "authorized_unscored",
    }
    if path.is_file():
        existing = P.load_json(path)
        for key, value in fixed.items():
            if existing.get(key) != value:
                raise SystemExit(f"existing protocol receipt {key} mismatch")
    else:
        out_root.mkdir(parents=True, exist_ok=True)
        P.write_json(path, {**fixed, "authorized_utc": R.utc_now()})
    return path


def _tree_size(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def cleanup_completed_heavy_work(out_root: Path, block: dict[str, Any]) -> Path:
    """Delete only heavy temporary stores after final artifacts validate."""

    root = out_root.resolve()
    final_receipt_path = (root / block["receipt_file"]).resolve()
    array_path = (root / block["array_file"]).resolve()
    final_receipt_path.relative_to(root)
    array_path.relative_to(root)
    receipt = P.load_json(final_receipt_path)
    if receipt.get("status") != "complete" or receipt.get("block_id") != block["block_id"]:
        raise RuntimeError("heavy cleanup requires a matching complete final receipt")
    if P.sha256_file(array_path) != receipt.get("array_file_sha256"):
        raise RuntimeError("heavy cleanup refused: final array SHA mismatch")

    attempt = int(receipt["attempt"])
    attempt_root = root / "work" / block["block_id"] / f"attempt-{attempt:03d}"
    attempt_root_resolved = attempt_root.resolve()
    attempt_root_resolved.relative_to(root)
    deleted: list[dict[str, Any]] = []
    for name in HEAVY_WORK_NAMES:
        target = attempt_root / name
        if not target.exists():
            continue
        if target.is_symlink():
            raise RuntimeError(f"refusing to clean symlink: {target}")
        resolved = target.resolve()
        resolved.relative_to(attempt_root_resolved)
        if resolved.parent != attempt_root_resolved or resolved.name != name:
            raise RuntimeError(f"refusing unexpected cleanup target: {resolved}")
        file_count, byte_count = _tree_size(resolved)
        shutil.rmtree(resolved)
        deleted.append({"name": name, "files": file_count, "bytes": byte_count})

    cleanup_path = root / "cleanup" / f"{block['block_id']}.json"
    value = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "block_id": block["block_id"],
        "attempt": attempt,
        "final_receipt_sha256": P.sha256_file(final_receipt_path),
        "array_file_sha256": P.sha256_file(array_path),
        "deleted": deleted,
        "text_logs_retained": True,
        "cleanup_utc": R.utc_now(),
    }
    if cleanup_path.is_file():
        existing = P.load_json(cleanup_path)
        for key in ("protocol_id", "block_id", "attempt", "final_receipt_sha256", "array_file_sha256"):
            if existing.get(key) != value[key]:
                raise RuntimeError(f"cleanup receipt {key} mismatch")
        if deleted:
            raise RuntimeError("heavy work reappeared after a completed cleanup")
        return cleanup_path
    cleanup_path.parent.mkdir(parents=True, exist_ok=True)
    P.write_json(cleanup_path, value)
    return cleanup_path


def write_completion_if_ready(
    out_root: Path,
    manifest: dict[str, Any],
    lock: dict[str, Any],
    public_freeze: dict[str, str],
) -> Path | None:
    records = []
    for block in manifest["blocks"]:
        receipt_path = out_root / block["receipt_file"]
        array_path = out_root / block["array_file"]
        cleanup_path = out_root / "cleanup" / f"{block['block_id']}.json"
        if not (receipt_path.is_file() and array_path.is_file() and cleanup_path.is_file()):
            return None
        P._load_block_arrays(array_path, block, manifest["content_sha256"])
        receipt = P.load_json(receipt_path)
        if receipt.get("status") != "complete":
            return None
        records.append(
            {
                "block_id": block["block_id"],
                "array_sha256": P.sha256_file(array_path),
                "receipt_sha256": P.sha256_file(receipt_path),
                "cleanup_sha256": P.sha256_file(cleanup_path),
            }
        )
    path = out_root / "inference_completion_receipt.json"
    fixed = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "status": "inference_complete_unscored",
        "causal_claim_allowed": False,
        "protocol_lock_content_sha256": lock["content_sha256"],
        "source_manifest_content_sha256": manifest["content_sha256"],
        "public_freeze_commit": public_freeze["repo_head"],
        "completed_blocks": len(records),
        "records": records,
    }
    if path.is_file():
        existing = P.load_json(path)
        for key, value in fixed.items():
            if existing.get(key) != value:
                raise RuntimeError(f"completion receipt {key} mismatch")
    else:
        P.write_json(path, {**fixed, "completed_utc": R.utc_now()})
    return path


def verify_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    lock_path = Path(args.lock).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_root = require_output_namespace(Path(args.out_root))
    lock = load_protocol_lock(lock_path)
    freeze = R.require_public_freeze(repo, lock_path)
    verified = verify_protocol_files(repo, lock, manifest_path)
    args.manifest = str(manifest_path)
    base = R.verify_all(args, verified["source_manifest"])
    if base["public_freeze"]["repo_head"] != freeze["repo_head"]:
        raise SystemExit("lock and source manifest are not frozen at the same public commit")
    print(P.canonical_json({"lock": lock, "public_freeze": freeze, "base": base, "out_root": str(out_root)}))


def run_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    lock_path = Path(args.lock).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_root = require_output_namespace(Path(args.out_root))
    lock = load_protocol_lock(lock_path)
    freeze = R.require_public_freeze(repo, lock_path)
    verified = verify_protocol_files(repo, lock, manifest_path)
    manifest = verified["source_manifest"]
    args.manifest = str(manifest_path)
    args.out_root = str(out_root)
    base = R.verify_all(args, manifest)
    if base["public_freeze"]["repo_head"] != freeze["repo_head"]:
        raise SystemExit("lock and source manifest are not frozen at the same public commit")
    fixed_root = Path(args.villa_fixed_root).resolve()
    env, py_info = R.python_environment(Path(args.python).resolve(), fixed_root)
    base["python"] = py_info
    _write_or_verify_protocol_receipt(out_root, lock, freeze)

    blocks = manifest["blocks"]
    if args.block_id:
        blocks = [block for block in blocks if block["block_id"] == args.block_id]
        if not blocks:
            raise SystemExit(f"unknown block ID: {args.block_id}")
    for index, block in enumerate(blocks, 1):
        print(f"[{index}/{len(blocks)}] {block['block_id']}", flush=True)
        status = run_operational_block(args, manifest, base, block, env)
        if status in {"complete", "already_complete"}:
            cleanup_completed_heavy_work(out_root, block)
        print(f"  {status}", flush=True)
    completion = write_completion_if_ready(out_root, manifest, lock, freeze)
    if completion is not None:
        print(f"inference complete: {completion}", flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lock", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--villa-fixed-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--read-retries", type=int, default=12)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    add_common(verify)
    verify.set_defaults(func=verify_command)
    run = commands.add_parser("run")
    add_common(run)
    run.add_argument("--block-id")
    run.add_argument("--dry-run", action="store_true")
    run.set_defaults(func=run_command)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
