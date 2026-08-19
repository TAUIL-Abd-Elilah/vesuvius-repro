from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

import physical_normalization_ab as P
import run_physical_released_baseline_comparison as O


BASELINE_FREEZE_COMMIT = "c62fd475b6f7df716e828b3a774e304e7cf43176"


def materialize_baseline_freeze(
    repo: Path,
    destination: Path,
    paths: set[str],
    crlf_paths: set[str],
) -> None:
    """Reconstruct locked protocol files when testing from a descendant branch."""

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_FREEZE_COMMIT, "HEAD"],
        cwd=repo,
        check=False,
    )
    assert ancestor.returncode == 0, "baseline freeze is not an ancestor of HEAD"
    for relative in sorted(paths):
        data = subprocess.check_output(
            ["git", "show", f"{BASELINE_FREEZE_COMMIT}:{relative}"],
            cwd=repo,
        )
        if relative in crlf_paths:
            data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def test_public_protocol_lock_and_files_verify(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parent
    lock_relative = "results/physical_released_baseline_comparison/protocol_lock.json"
    lock_path = repo / lock_relative
    locked_lock_bytes = subprocess.check_output(
        ["git", "show", f"{BASELINE_FREEZE_COMMIT}:{lock_relative}"],
        cwd=repo,
    )
    assert lock_path.read_bytes() == locked_lock_bytes
    lock = O.load_protocol_lock(lock_path)
    assert lock["checkout_line_endings"]["source_manifest"] == "crlf"
    required_paths = set(lock["protocol_files_sha256"])
    required_paths.update(
        {
            lock["source_manifest_path"],
            lock["failed_causal_result_path"],
            lock["preoutcome_failure_path"],
            lock["pre_score_failure_path"],
        }
    )
    materialize_baseline_freeze(
        repo,
        tmp_path,
        required_paths,
        {lock["source_manifest_path"]},
    )
    verified = O.verify_protocol_files(
        tmp_path,
        lock,
        tmp_path / lock["source_manifest_path"],
    )
    assert verified["source_manifest"]["content_sha256"] == O.SOURCE_MANIFEST_CONTENT_SHA256
    assert verified["failed_causal_result"]["corrected_arm_run"] is False
    assert verified["preoutcome_failure"]["model_inference_started"] is False
    assert verified["pre_score_failure"]["blending_started"] is False
    assert verified["pre_score_failure"]["physical_score_computed"] is False


def test_output_namespace_is_fail_closed(tmp_path: Path) -> None:
    good = tmp_path / O.OUTPUT_NAMESPACE
    assert O.require_output_namespace(good) == good.resolve()
    with pytest.raises(SystemExit, match="output root must end"):
        O.require_output_namespace(tmp_path / "physical_normalization_ab")


@pytest.mark.parametrize("foreground", [1, 255])
def test_released_binary_encodings_canonicalize_identically(foreground: int) -> None:
    source = np.asarray([[0, foreground], [foreground, 0]], dtype=np.uint8)
    view = O.CanonicalReleasedBinaryArray(source)
    np.testing.assert_array_equal(
        view[:, :], np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    )


def test_released_binary_canonicalization_rejects_other_values() -> None:
    view = O.CanonicalReleasedBinaryArray(np.asarray([0, 2, 255], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="non-binary"):
        view[:]


def test_operational_block_restores_remote_opener(monkeypatch: pytest.MonkeyPatch) -> None:
    original = O.R._open_remote_zarr

    def fake_run(*args, **kwargs):
        assert O.R._open_remote_zarr is not original
        assert O.R.NORMALIZATION_LOG_TOKEN == O.MODEL_NORMALIZATION_LOG_TOKEN
        opened = O.R._open_remote_zarr("unused")
        np.testing.assert_array_equal(opened[:], np.asarray([0, 1], dtype=np.uint8))
        return "dry_run"

    monkeypatch.setattr(O.R, "_open_remote_zarr", lambda url: np.asarray([0, 255], dtype=np.uint8))
    patched_original = O.R._open_remote_zarr
    monkeypatch.setattr(O.R, "NORMALIZATION_LOG_TOKEN", "inherited-test-token")
    patched_original_token = O.R.NORMALIZATION_LOG_TOKEN
    monkeypatch.setattr(O.R, "run_block", fake_run)
    assert O.run_operational_block(None, {}, {}, {}, {}) == "dry_run"
    assert O.R._open_remote_zarr is patched_original
    assert O.R.NORMALIZATION_LOG_TOKEN == patched_original_token


def test_operational_block_restores_patches_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(O.R, "_open_remote_zarr", lambda url: np.asarray([0], dtype=np.uint8))
    patched_original = O.R._open_remote_zarr
    monkeypatch.setattr(O.R, "NORMALIZATION_LOG_TOKEN", "inherited-error-token")
    patched_original_token = O.R.NORMALIZATION_LOG_TOKEN

    def fail(*args, **kwargs):
        assert O.R._open_remote_zarr is not patched_original
        assert O.R.NORMALIZATION_LOG_TOKEN == O.MODEL_NORMALIZATION_LOG_TOKEN
        raise RuntimeError("injected failure")

    monkeypatch.setattr(O.R, "run_block", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        O.run_operational_block(None, {}, {}, {}, {})
    assert O.R._open_remote_zarr is patched_original
    assert O.R.NORMALIZATION_LOG_TOKEN == patched_original_token


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
