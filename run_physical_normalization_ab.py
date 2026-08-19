#!/usr/bin/env python3
"""Fail-closed, resumable runner for the frozen physical normalization A/B.

This script refuses to infer until the manifest is committed at the current public branch
head, all implementation/model/label hashes match, and the corrected villa worktree is clean
at the exact PR #1386 commit.  It reads the public binary baseline and runs only the corrected
arm; a separate sentinel command reproduces the baseline with the old path before results are
interpreted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

import physical_normalization_ab as P


NORMALIZATION_LOG_TOKEN = (
    "Using model-declared normalization 'ct' instead of CLI value 'instance_zscore'."
)

# The frozen pre-fix villa commit predates villa's Zarr-3 writer repair. Under the current
# declared dependency range (zarr<4), reproduce Zarr-2 writer semantics without editing either
# villa worktree: restore the former Blosc re-export and explicitly select format 2 whenever a
# v2 numcodecs compressor is supplied. This is a no-op under Zarr 2 and is applied identically
# to inference and blending for both model arms.
ZARR2_COMPAT_SHIM = (
    "import numcodecs,zarr\n"
    "if not hasattr(zarr,'Blosc'): zarr.Blosc=numcodecs.Blosc\n"
    "_physical_ab_zarr_open=zarr.open\n"
    "def _physical_ab_zarr_open_v2(*args,**kwargs):\n"
    "    if kwargs.get('compressor') is not None and 'zarr_format' not in kwargs:\n"
    "        kwargs['zarr_format']=2\n"
    "    return _physical_ab_zarr_open(*args,**kwargs)\n"
    "zarr.open=_physical_ab_zarr_open_v2\n"
)
MODULE_BOOTSTRAP = ZARR2_COMPAT_SHIM + (
    "import runpy,sys\n"
    "_physical_ab_module=sys.argv[1]\n"
    "sys.argv=sys.argv[1:]\n"
    "runpy.run_module(_physical_ab_module,run_name='__main__')\n"
)
BLEND_BOOTSTRAP = ZARR2_COMPAT_SHIM + (
    "import sys\n"
    "from vesuvius.models.run import blending as _physical_ab_blending\n"
    "sys.argv=['blending',*sys.argv[1:]]\n"
    "raise SystemExit(_physical_ab_blending.main())\n"
)
ZARR2_COMPAT_SHA256 = P.sha256_bytes(ZARR2_COMPAT_SHIM.encode("utf-8"))
BLEND_BOOTSTRAP_SHA256 = P.sha256_bytes(BLEND_BOOTSTRAP.encode("utf-8"))


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_text(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


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


def load_and_verify_manifest(path: Path) -> dict[str, Any]:
    manifest = P.load_json(path)
    recorded = manifest.pop("content_sha256", None)
    actual = P.sha256_bytes(P.canonical_json(manifest).encode("utf-8"))
    manifest["content_sha256"] = recorded
    if recorded != actual:
        raise SystemExit(f"manifest content SHA mismatch: {recorded} != {actual}")
    if manifest.get("status") != P.MANIFEST_STATUS:
        raise SystemExit(f"manifest status is not preregistered: {manifest.get('status')}")
    if manifest.get("implementation_revision") != P.IMPLEMENTATION_REVISION:
        raise SystemExit(
            "manifest implementation revision is not current: "
            f"{manifest.get('implementation_revision')}"
        )
    if len(manifest.get("blocks", [])) != 64:
        raise SystemExit(f"manifest must freeze 64 blocks, got {len(manifest.get('blocks', []))}")
    return manifest


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=not binary,
        encoding=None if binary else "utf-8",
    ).strip()


def normalized_text_bytes(value: bytes) -> bytes:
    """Compare Git blobs to Windows checkouts without weakening content checks."""

    return value.replace(b"\r\n", b"\n").rstrip(b"\n")


def require_public_freeze(repo: Path, manifest_path: Path) -> dict[str, str]:
    status = str(_git(repo, "status", "--porcelain=v1"))
    if status:
        raise SystemExit("runner requires a clean preregistration worktree:\n" + status)
    head = str(_git(repo, "rev-parse", "HEAD"))
    branch = str(_git(repo, "branch", "--show-current"))
    if not branch:
        raise SystemExit("runner refuses a detached preregistration HEAD")
    try:
        rel = manifest_path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise SystemExit("manifest must live inside the implementation repository") from exc
    try:
        _git(repo, "ls-files", "--error-unmatch", rel)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"manifest is not tracked at HEAD: {rel}") from exc
    committed = bytes(_git(repo, "show", f"HEAD:{rel}", binary=True))
    local = manifest_path.read_bytes()
    if normalized_text_bytes(committed) != normalized_text_bytes(local):
        raise SystemExit("local manifest bytes do not match the committed HEAD blob")
    remote_line = subprocess.check_output(
        ["git", "ls-remote", "origin", f"refs/heads/{branch}"],
        cwd=repo,
        text=True,
        encoding="utf-8",
    ).strip()
    remote_head = remote_line.split()[0] if remote_line else ""
    if remote_head != head:
        raise SystemExit(
            f"public freeze missing: origin/{branch}={remote_head or '<absent>'}, HEAD={head}"
        )
    return {"repo_head": head, "branch": branch, "remote_head": remote_head, "manifest_rel": rel}


def verify_implementation_files(repo: Path, manifest: dict[str, Any]) -> None:
    expected = manifest["implementation"]["files_sha256"]
    for name, digest in expected.items():
        path = repo / name
        actual = P.sha256_file(path)
        if actual != digest:
            raise SystemExit(f"implementation drift: {name} SHA {actual} != {digest}")


def verify_fixed_villa(root: Path) -> dict[str, str]:
    head = str(_git(root, "rev-parse", "HEAD"))
    status = str(_git(root, "status", "--porcelain=v1"))
    if head != P.PR1386_COMMIT:
        raise SystemExit(f"corrected villa HEAD {head} != frozen {P.PR1386_COMMIT}")
    if status:
        raise SystemExit("corrected villa worktree is dirty:\n" + status)
    return {"head": head, "root": str(root)}


def verify_broken_villa(root: Path) -> dict[str, str]:
    head = str(_git(root, "rev-parse", "HEAD"))
    status = str(_git(root, "status", "--porcelain=v1"))
    if head != P.BROKEN_REPRO_COMMIT:
        raise SystemExit(f"broken-path villa HEAD {head} != frozen {P.BROKEN_REPRO_COMMIT}")
    if status:
        raise SystemExit("broken-path villa worktree is dirty:\n" + status)
    return {"head": head, "root": str(root)}


def verify_inputs(labels_root: Path, model_dir: Path, manifest: dict[str, Any]) -> None:
    for scroll, record in manifest["inputs"]["labels"].items():
        archive = labels_root / record["archive"]["name"]
        if archive.stat().st_size != record["archive"]["bytes"]:
            raise SystemExit(f"{archive}: byte size changed")
        if P.sha256_file(archive) != record["archive"]["sha256"]:
            raise SystemExit(f"{archive}: SHA changed")
        store = labels_root / record["store"]
        if P.sha256_file(store / ".zarray") != record["zarray_sha256"]:
            raise SystemExit(f"{scroll}: label .zarray changed")
        if P.sha256_file(store / ".zattrs") != record["zattrs_sha256"]:
            raise SystemExit(f"{scroll}: label .zattrs changed")
    for rel, record in manifest["inputs"]["model"]["files"].items():
        path = model_dir / rel
        if path.stat().st_size != record["bytes"] or P.sha256_file(path) != record["sha256"]:
            raise SystemExit(f"model input drift: {path}")


def fetch_json(url: str) -> dict[str, Any]:
    def once() -> dict[str, Any]:
        with urllib.request.urlopen(url, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    return retry(once)


def verify_remote_shapes(manifest: dict[str, Any]) -> dict[str, Any]:
    report = {}
    for scroll, record in manifest["inputs"]["scrolls"].items():
        ct_meta = fetch_json(record["ct_url"].rstrip("/") + "/.zarray")
        pred_meta = fetch_json(record["published_prediction_url"].rstrip("/") + "/.zarray")
        expected = list(map(int, record["ct_shape_l0"]))
        if list(map(int, ct_meta["shape"])) != expected:
            raise SystemExit(f"{scroll}: remote CT shape changed")
        if list(map(int, pred_meta["shape"])) != expected:
            raise SystemExit(f"{scroll}: remote prediction shape changed")
        if pred_meta.get("dtype") != "|u1":
            raise SystemExit(f"{scroll}: published prediction is no longer uint8")
        report[scroll] = {
            "ct_zarray_sha256": P.sha256_bytes(P.canonical_json(ct_meta).encode()),
            "prediction_zarray_sha256": P.sha256_bytes(P.canonical_json(pred_meta).encode()),
            "shape_l0": expected,
        }
    return report


def python_environment(python: Path, villa_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    source = villa_root / "vesuvius" / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        {
            "nnUNet_compile": "0",
            "TORCHDYNAMO_DISABLE": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    probe = (
        "import json,sys,numpy,scipy,zarr,numcodecs,torch,vesuvius;"
        "print(json.dumps({'executable':sys.executable,'python':sys.version,"
        "'numpy':numpy.__version__,'scipy':scipy.__version__,'zarr':zarr.__version__,"
        "'numcodecs':numcodecs.__version__,'torch':torch.__version__,"
        "'vesuvius':vesuvius.__file__}))"
    )
    result = run_text([str(python), "-c", probe], villa_root, env)
    if result.returncode:
        raise SystemExit("corrected Python environment probe failed:\n" + result.stderr)
    info = json.loads(result.stdout.strip().splitlines()[-1])
    resolved = Path(info["vesuvius"]).resolve()
    try:
        resolved.relative_to(source.resolve())
    except ValueError as exc:
        raise SystemExit(f"Python imported vesuvius from wrong tree: {resolved}") from exc
    return env, info


def verify_all(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    repo = Path(__file__).resolve().parent
    freeze = require_public_freeze(repo, Path(args.manifest))
    verify_implementation_files(repo, manifest)
    villa = verify_fixed_villa(Path(args.villa_fixed_root).resolve())
    verify_inputs(Path(args.labels_root).resolve(), Path(args.model_dir).resolve(), manifest)
    remote = verify_remote_shapes(manifest)
    _, py = python_environment(Path(args.python).resolve(), Path(args.villa_fixed_root).resolve())
    return {
        "verified_utc": utc_now(),
        "manifest_content_sha256": manifest["content_sha256"],
        "public_freeze": freeze,
        "corrected_villa": villa,
        "remote": remote,
        "python": py,
        "platform": platform.platform(),
    }


def max_pool_l0_to_l1(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3 or any(dim % 2 for dim in array.shape):
        raise ValueError(f"expected even 3-D L0 array, got {array.shape}")
    z, y, x = array.shape
    return array.reshape(z // 2, 2, y // 2, 2, x // 2, 2).max(axis=(1, 3, 5))


def _open_remote_zarr(url: str):
    import zarr

    return zarr.open(url, mode="r")


def read_box(array: Any, box: list[int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = map(int, box)
    expected = (z1 - z0, y1 - y0, x1 - x0)
    value = retry(lambda: np.asarray(array[z0:z1, y0:y1, x0:x1]))
    if value.shape != expected:
        raise RuntimeError(f"read returned {value.shape}, expected {expected}")
    return value


def _array_digest(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def _next_attempt(work_root: Path, block_id: str) -> tuple[int, Path]:
    parent = work_root / block_id
    parent.mkdir(parents=True, exist_ok=True)
    used = []
    for path in parent.glob("attempt-*"):
        try:
            used.append(int(path.name.split("-")[-1]))
        except ValueError:
            continue
    number = max(used, default=0) + 1
    path = parent / f"attempt-{number:03d}"
    path.mkdir()
    return number, path


def _write_attempt_receipt(receipts: Path, block_id: str, attempt: int, value: dict[str, Any]) -> Path:
    receipts.mkdir(parents=True, exist_ok=True)
    path = receipts / f"{block_id}.attempt-{attempt:03d}.json"
    P.write_json(path, value)
    return path


def infer_probability_extent(
    args: argparse.Namespace,
    block: dict[str, Any],
    scroll_input: dict[str, Any],
    villa_root: Path,
    env: dict[str, str],
    work: Path,
    required_log_token: str | None,
    normalization_override: str | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Run one ROI and return max-pooled L1 probabilities plus provenance."""

    logits = work / "logits"
    module_args = [
        "--model_path",
        str(Path(args.model_dir).resolve()),
        "--input_dir",
        scroll_input["ct_url"],
        "--output_dir",
        str(logits),
        "--device",
        "cuda",
        "--disable_tta",
        "--batch_size",
        "1",
        "--num_workers",
        str(args.num_workers),
        "--read-retries",
        str(args.read_retries),
        "--bbox",
        ",".join(
            f"{block['geometry']['inference_bbox_l0'][i]}:"
            f"{block['geometry']['inference_bbox_l0'][i + 1]}"
            for i in (0, 2, 4)
        ),
    ]
    if normalization_override is not None:
        module_args.extend(["--normalization", normalization_override])
    command = [
        str(Path(args.python).resolve()),
        "-c",
        MODULE_BOOTSTRAP,
        "vesuvius.models.run.inference",
        *module_args,
    ]
    evidence: dict[str, Any] = {
        "predict_command": command,
        "predict_module_args": module_args,
        "required_normalization_log_token": required_log_token,
        "normalization_override": normalization_override,
        "zarr2_compatibility": {
            "shim_sha256": ZARR2_COMPAT_SHA256,
            "applied_to": ["inference", "blending"],
        },
        "blend_bootstrap_sha256": BLEND_BOOTSTRAP_SHA256,
    }
    if args.dry_run:
        print(command_text(command))
        evidence["dry_run"] = True
        return None, evidence

    predict = run_text(command, villa_root, env)
    stdout_path = work / "predict.stdout.log"
    stderr_path = work / "predict.stderr.log"
    stdout_path.write_text(predict.stdout, encoding="utf-8")
    stderr_path.write_text(predict.stderr, encoding="utf-8")
    combined = predict.stdout + "\n" + predict.stderr
    evidence.update(
        {
            "predict_returncode": predict.returncode,
            "predict_stdout_sha256": P.sha256_file(stdout_path),
            "predict_stderr_sha256": P.sha256_file(stderr_path),
            "required_normalization_log_token_found": (
                required_log_token in combined if required_log_token else None
            ),
        }
    )
    if predict.returncode != 0:
        raise RuntimeError(f"predict failed with exit {predict.returncode}")
    if required_log_token is not None and required_log_token not in combined:
        raise RuntimeError("predict did not emit the required normalization proof")
    if not logits.is_dir():
        raise RuntimeError("predict wrote no logits directory")

    merged = work / "merged.zarr"
    blend_command = [
        str(Path(args.python).resolve()),
        "-c",
        BLEND_BOOTSTRAP,
        str(logits),
        str(merged),
        "--num_workers",
        str(args.num_workers),
    ]
    blend = run_text(blend_command, villa_root, env)
    blend_stdout = work / "blend.stdout.log"
    blend_stderr = work / "blend.stderr.log"
    blend_stdout.write_text(blend.stdout, encoding="utf-8")
    blend_stderr.write_text(blend.stderr, encoding="utf-8")
    evidence.update(
        {
            "blend_command": blend_command,
            "blend_returncode": blend.returncode,
            "blend_stdout_sha256": P.sha256_file(blend_stdout),
            "blend_stderr_sha256": P.sha256_file(blend_stderr),
        }
    )
    if blend.returncode != 0 or not merged.is_dir():
        raise RuntimeError(f"blend failed with exit {blend.returncode}")

    import zarr

    merged_array = zarr.open(str(merged), mode="r")
    expected_store_shape = (2, *map(int, scroll_input["ct_shape_l0"]))
    if tuple(map(int, merged_array.shape)) != expected_store_shape:
        raise RuntimeError(f"merged shape is {merged_array.shape}, expected {expected_store_shape}")
    z0, z1, y0, y1, x0, x1 = block["geometry"]["prediction_extent_global_l0"]
    expected_extent_shape = (z1 - z0, y1 - y0, x1 - x0)
    logit0 = np.asarray(merged_array[0, z0:z1, y0:y1, x0:x1], dtype=np.float32)
    logit1 = np.asarray(merged_array[1, z0:z1, y0:y1, x0:x1], dtype=np.float32)
    if logit0.shape != expected_extent_shape or logit1.shape != expected_extent_shape:
        raise RuntimeError("merged logits do not cover the frozen prediction extent")
    if not np.isfinite(logit0).all() or not np.isfinite(logit1).all():
        raise RuntimeError("merged logits contain non-finite values")
    delta = np.clip(logit1 - logit0, -80.0, 80.0)
    probability_l0 = 1.0 / (1.0 + np.exp(-delta))
    probability_l1 = max_pool_l0_to_l1(probability_l0).astype(np.float32)
    evidence.update(
        {
            "merged_zarray_sha256": P.sha256_file(merged / ".zarray"),
            "merged_zattrs_sha256": P.sha256_file(merged / ".zattrs"),
            "probability_l1_sha256": _array_digest(probability_l1),
            "probability_l1_shape": list(probability_l1.shape),
        }
    )
    return probability_l1, evidence


def run_block(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    verification: dict[str, Any],
    block: dict[str, Any],
    env: dict[str, str],
) -> str:
    out_root = Path(args.out_root).resolve()
    array_path = out_root / block["array_file"]
    final_receipt = out_root / block["receipt_file"]
    if array_path.is_file() and final_receipt.is_file():
        P._load_block_arrays(array_path, block, manifest["content_sha256"])
        return "already_complete"
    if array_path.exists() or final_receipt.exists():
        raise RuntimeError(f"partial final artifact exists for {block['block_id']}; inspect it")

    attempt, work = _next_attempt(out_root / "work", block["block_id"])
    receipts = out_root / "receipts"
    started = utc_now()
    t0 = time.time()
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "block_id": block["block_id"],
        "attempt": attempt,
        "started_utc": started,
        "manifest_content_sha256": manifest["content_sha256"],
        "public_freeze_commit": verification["public_freeze"]["repo_head"],
        "corrected_villa_commit": P.PR1386_COMMIT,
        "geometry": block["geometry"],
    }
    _write_attempt_receipt(receipts, block["block_id"], attempt, receipt)

    try:
        scroll = block["scroll"]
        scroll_input = manifest["inputs"]["scrolls"][scroll]
        ext_l0 = block["geometry"]["prediction_extent_global_l0"]
        published = _open_remote_zarr(scroll_input["published_prediction_url"])
        baseline_l0 = read_box(published, ext_l0)
        if not np.isin(baseline_l0, [0, 1]).all():
            raise RuntimeError("published baseline contains values outside {0,1}")
        baseline_l1 = max_pool_l0_to_l1(baseline_l0).astype(np.uint8)

        corrected_root = Path(args.villa_fixed_root).resolve()
        corrected_pmax_l1, inference_evidence = infer_probability_extent(
            args,
            block,
            scroll_input,
            corrected_root,
            env,
            work,
            required_log_token=NORMALIZATION_LOG_TOKEN,
        )
        receipt["inference"] = inference_evidence
        if corrected_pmax_l1 is None:
            receipt.update(status="dry_run", finished_utc=utc_now())
            _write_attempt_receipt(receipts, block["block_id"], attempt, receipt)
            return "dry_run"
        if baseline_l1.shape != corrected_pmax_l1.shape:
            raise RuntimeError("baseline/corrected L1 shape mismatch")

        metadata = {
            "schema_version": 1,
            "manifest_content_sha256": manifest["content_sha256"],
            "block_id": block["block_id"],
            "scroll": scroll,
            "prediction_extent_global_l1": block["geometry"]["prediction_extent_global_l1"],
            "public_freeze_commit": verification["public_freeze"]["repo_head"],
            "corrected_villa_commit": P.PR1386_COMMIT,
            "normalization": "ct_from_model_plans",
            "fixed_threshold": P.FIXED_THRESHOLD,
        }
        array_path.parent.mkdir(parents=True, exist_ok=True)
        temp = array_path.with_suffix(".npz.incomplete")
        with temp.open("wb") as f:
            np.savez_compressed(
                f,
                baseline_l1=baseline_l1,
                corrected_pmax_l1=corrected_pmax_l1,
                metadata_json=np.asarray(P.canonical_json(metadata)),
            )
        os.replace(temp, array_path)
        P._load_block_arrays(array_path, block, manifest["content_sha256"])

        receipt.update(
            {
                "status": "complete",
                "finished_utc": utc_now(),
                "elapsed_seconds": time.time() - t0,
                "baseline_l1_sha256": _array_digest(baseline_l1),
                "corrected_pmax_l1_sha256": _array_digest(corrected_pmax_l1),
                "array_file": str(array_path),
                "array_file_sha256": P.sha256_file(array_path),
                "array_shape_l1": list(baseline_l1.shape),
                "baseline_positive_count_extent": int(baseline_l1.sum()),
                "corrected_fixed_positive_count_extent": int(
                    (corrected_pmax_l1 > P.FIXED_THRESHOLD).sum()
                ),
            }
        )
        attempt_receipt = _write_attempt_receipt(
            receipts, block["block_id"], attempt, receipt
        )
        final_receipt.parent.mkdir(parents=True, exist_ok=True)
        P.write_json(final_receipt, {**receipt, "attempt_receipt": str(attempt_receipt)})
        return "complete"
    except Exception as exc:
        receipt.update(
            {
                "status": "failed",
                "finished_utc": utc_now(),
                "elapsed_seconds": time.time() - t0,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        _write_attempt_receipt(receipts, block["block_id"], attempt, receipt)
        raise


def _sentinel_blocks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for scroll in P.SCROLLS:
        blocks = sorted(
            (b for b in manifest["blocks"] if b["scroll"] == scroll),
            key=lambda b: b["selection_index"],
        )
        if not blocks:
            raise SystemExit(f"manifest has no sentinel candidate for {scroll}")
        result.append(blocks[0])
    return result


def sentinel_command(args: argparse.Namespace) -> None:
    manifest = load_and_verify_manifest(Path(args.manifest).resolve())
    verification = verify_all(args, manifest)
    broken_root = Path(args.villa_broken_root).resolve()
    broken_info = verify_broken_villa(broken_root)
    env, py_info = python_environment(Path(args.python).resolve(), broken_root)
    out_root = Path(args.out_root).resolve()
    sentinel_root = out_root / "sentinel"
    sentinel_root.mkdir(parents=True, exist_ok=True)

    for block in _sentinel_blocks(manifest):
        scroll = block["scroll"]
        receipt_path = sentinel_root / f"{scroll}.json"
        if receipt_path.is_file():
            existing = P.load_json(receipt_path)
            if (
                existing.get("manifest_content_sha256") != manifest["content_sha256"]
                or existing.get("block_id") != block["block_id"]
                or existing.get("gate_passed") is not True
            ):
                raise SystemExit(f"invalid existing sentinel receipt: {receipt_path}")
            print(f"{scroll}: sentinel already passed")
            continue

        attempt, work = _next_attempt(sentinel_root / "work", block["block_id"])
        started = utc_now()
        t0 = time.time()
        scroll_input = manifest["inputs"]["scrolls"][scroll]
        ext_l0 = block["geometry"]["prediction_extent_global_l0"]
        published = _open_remote_zarr(scroll_input["published_prediction_url"])
        baseline_l1 = max_pool_l0_to_l1(read_box(published, ext_l0)).astype(bool)
        old_pmax, evidence = infer_probability_extent(
            args,
            block,
            scroll_input,
            broken_root,
            env,
            work,
            required_log_token=None,
            normalization_override="instance_zscore",
        )
        if old_pmax is None:
            print(f"{scroll}: sentinel dry run")
            continue
        old_binary = old_pmax > P.FIXED_THRESHOLD
        baseline_target = P._target_view(baseline_l1)
        old_target = P._target_view(old_binary)
        intersection = int((baseline_target & old_target).sum())
        denominator = int(baseline_target.sum() + old_target.sum())
        dice = 2 * intersection / denominator if denominator else 1.0
        disagree = int((baseline_target != old_target).sum())
        agreement = float((baseline_target == old_target).mean())
        passed = dice >= 0.999
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "manifest_content_sha256": manifest["content_sha256"],
            "public_freeze_commit": verification["public_freeze"]["repo_head"],
            "broken_villa": broken_info,
            "python": py_info,
            "scroll": scroll,
            "block_id": block["block_id"],
            "attempt": attempt,
            "started_utc": started,
            "finished_utc": utc_now(),
            "elapsed_seconds": time.time() - t0,
            "scope": "frozen 64^3 L1 score cube",
            "normalization_override": "instance_zscore",
            "threshold": P.FIXED_THRESHOLD,
            "published_positive_count": int(baseline_target.sum()),
            "reproduced_positive_count": int(old_target.sum()),
            "intersection": intersection,
            "dice": dice,
            "agreement": agreement,
            "disagreeing_voxels": disagree,
            "required_dice": 0.999,
            "gate_passed": passed,
            "inference": evidence,
            "published_target_sha256": _array_digest(baseline_target),
            "reproduced_target_sha256": _array_digest(old_target),
        }
        P.write_json(receipt_path, receipt)
        print(f"{scroll}: sentinel Dice={dice:.6f}, disagree={disagree}")
        if not passed:
            raise SystemExit(f"{scroll}: public artifact is not reproduced; corrected run forbidden")


def require_sentinels(out_root: Path, manifest: dict[str, Any]) -> None:
    for block in _sentinel_blocks(manifest):
        path = out_root / "sentinel" / f"{block['scroll']}.json"
        if not path.is_file():
            raise SystemExit(f"missing baseline sentinel: {path}; run `sentinel` first")
        receipt = P.load_json(path)
        if (
            receipt.get("manifest_content_sha256") != manifest["content_sha256"]
            or receipt.get("block_id") != block["block_id"]
            or receipt.get("gate_passed") is not True
        ):
            raise SystemExit(f"baseline sentinel is stale or failed: {path}")


def verify_command(args: argparse.Namespace) -> None:
    manifest = load_and_verify_manifest(Path(args.manifest).resolve())
    report = verify_all(args, manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


def run_command(args: argparse.Namespace) -> None:
    manifest = load_and_verify_manifest(Path(args.manifest).resolve())
    verification = verify_all(args, manifest)
    fixed_root = Path(args.villa_fixed_root).resolve()
    env, py_info = python_environment(Path(args.python).resolve(), fixed_root)
    verification["python"] = py_info
    if not args.dry_run:
        require_sentinels(Path(args.out_root).resolve(), manifest)

    blocks = manifest["blocks"]
    if args.block_id:
        blocks = [b for b in blocks if b["block_id"] == args.block_id]
        if not blocks:
            raise SystemExit(f"unknown block ID: {args.block_id}")
    for index, block in enumerate(blocks, 1):
        print(f"[{index}/{len(blocks)}] {block['block_id']}", flush=True)
        status = run_block(args, manifest, verification, block, env)
        print(f"  {status}", flush=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, func in (("verify", verify_command), ("run", run_command)):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--labels-root", required=True)
        p.add_argument("--model-dir", required=True)
        p.add_argument("--villa-fixed-root", required=True)
        p.add_argument("--python", required=True)
        p.add_argument("--out-root", required=True)
        p.add_argument("--num-workers", type=int, default=2)
        p.add_argument("--read-retries", type=int, default=12)
        p.add_argument("--block-id")
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(func=func)
    sentinel = sub.add_parser("sentinel")
    sentinel.add_argument("--manifest", required=True)
    sentinel.add_argument("--labels-root", required=True)
    sentinel.add_argument("--model-dir", required=True)
    sentinel.add_argument("--villa-fixed-root", required=True)
    sentinel.add_argument("--villa-broken-root", required=True)
    sentinel.add_argument("--python", required=True)
    sentinel.add_argument("--out-root", required=True)
    sentinel.add_argument("--num-workers", type=int, default=2)
    sentinel.add_argument("--read-retries", type=int, default=12)
    sentinel.add_argument("--dry-run", action="store_true")
    sentinel.set_defaults(func=sentinel_command)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
