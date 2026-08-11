from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import physical_normalization_ab as P
import run_physical_released_baseline_comparison as O


def test_public_protocol_lock_and_files_verify() -> None:
    repo = Path(__file__).resolve().parent
    lock = O.load_protocol_lock(
        repo / "results" / O.OUTPUT_NAMESPACE / "protocol_lock.json"
    )
    verified = O.verify_protocol_files(
        repo,
        lock,
        repo / "results" / "physical_normalization_ab" / "manifest.json",
    )
    assert verified["source_manifest"]["content_sha256"] == O.SOURCE_MANIFEST_CONTENT_SHA256
    assert verified["failed_causal_result"]["corrected_arm_run"] is False


def test_output_namespace_is_fail_closed(tmp_path: Path) -> None:
    good = tmp_path / O.OUTPUT_NAMESPACE
    assert O.require_output_namespace(good) == good.resolve()
    with pytest.raises(SystemExit, match="output root must end"):
        O.require_output_namespace(tmp_path / "physical_normalization_ab")


def test_protocol_lock_content_hash_is_checked(tmp_path: Path) -> None:
    value = {
        "schema_version": 1,
        "protocol_id": O.PROTOCOL_ID,
        "status": O.LOCK_STATUS,
        "output_namespace": O.OUTPUT_NAMESPACE,
        "source_manifest_content_sha256": O.SOURCE_MANIFEST_CONTENT_SHA256,
        "protocol_files_sha256": {},
    }
    value["content_sha256"] = P.sha256_bytes(P.canonical_json(value).encode("utf-8"))
    value["protocol_id"] = "tampered-after-freeze"
    path = tmp_path / "lock.json"
    P.write_json(path, value)
    with pytest.raises(SystemExit, match="content SHA mismatch"):
        O.load_protocol_lock(path)


def _write_final_artifacts(root: Path, block_id: str = "block-a") -> dict:
    array_path = root / "arrays" / f"{block_id}.npz"
    array_path.parent.mkdir(parents=True)
    with array_path.open("wb") as stream:
        np.savez_compressed(stream, value=np.asarray([1], dtype=np.uint8))
    receipt_path = root / "receipts" / f"{block_id}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "block_id": block_id,
        "attempt": 2,
        "array_file_sha256": P.sha256_file(array_path),
    }
    P.write_json(receipt_path, receipt)
    return {
        "block_id": block_id,
        "array_file": f"arrays/{block_id}.npz",
        "receipt_file": f"receipts/{block_id}.json",
    }


def test_cleanup_deletes_only_heavy_stores_and_retains_logs(tmp_path: Path) -> None:
    root = tmp_path / O.OUTPUT_NAMESPACE
    block = _write_final_artifacts(root)
    attempt = root / "work" / block["block_id"] / "attempt-002"
    for name in O.HEAVY_WORK_NAMES:
        target = attempt / name
        target.mkdir(parents=True)
        (target / "chunk").write_bytes(b"large-temporary-bytes")
    log = attempt / "predict.stdout.log"
    log.write_text("normalization resolved to ct\n", encoding="utf-8")

    cleanup = O.cleanup_completed_heavy_work(root, block)
    assert cleanup.is_file()
    assert log.read_text(encoding="utf-8") == "normalization resolved to ct\n"
    assert not (attempt / "logits").exists()
    assert not (attempt / "merged.zarr").exists()
    value = P.load_json(cleanup)
    assert {item["name"] for item in value["deleted"]} == set(O.HEAVY_WORK_NAMES)


def test_cleanup_is_idempotent_after_success(tmp_path: Path) -> None:
    root = tmp_path / O.OUTPUT_NAMESPACE
    block = _write_final_artifacts(root)
    first = O.cleanup_completed_heavy_work(root, block)
    second = O.cleanup_completed_heavy_work(root, block)
    assert first == second


def test_cleanup_refuses_changed_final_array(tmp_path: Path) -> None:
    root = tmp_path / O.OUTPUT_NAMESPACE
    block = _write_final_artifacts(root)
    (root / block["array_file"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="array SHA mismatch"):
        O.cleanup_completed_heavy_work(root, block)
