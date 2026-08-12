#!/usr/bin/env python3
"""Execute and independently verify the locked three-arm ScrollFiesta gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

import crossscan_scrollfiesta_adapter as A
import crossscan_scrollfiesta_metrics as M
import crossscan_scrollfiesta_obj as O
import verify_physical_label_semantics as V


SCHEMA = "crossscan-scrollfiesta-downstream-result-v1"
METRIC_LOCK_SCHEMA = "crossscan-scrollfiesta-metric-lock-v1"
METRIC_LOCK_CONTENT_SHA256 = (
    "70c29b370b1f6ca2bb7f6d78eb284e456187056d2ed7efb86c7b5950e976f42c"
)
ARM_ORDER = ("baseline-fixed", "candidate-fixed", "candidate-matched-mass")
PINNED_BINARIES = {
    "cube_mesh.exe": {
        "bytes": 1173504,
        "sha256": "0f44830af325f9ee667eb2da26a88b929f48f27503e05fb3b6c0d7b3c8aa376d",
    },
    "grid_pipeline.exe": {
        "bytes": 414720,
        "sha256": "1b99299296f5eb787cf8c75d6d95ac224ac5e68f31bf5216b423ff680f96f062",
    },
    "grid_weld.exe": {
        "bytes": 489472,
        "sha256": "0ac850e70aaaccdfc932a99b9a9a5b620406702a002df02fc6a785774c98a583",
    },
}
PIPELINE_TAIL = (
    "--halo", "13", "--trim-inset", "0", "--simplify", "cvt",
    "--max-concurrent", "1", "--threads-per-cube", "1",
)
TRUTH_BOX = (slice(192, 320), slice(1280, 1408), slice(192, 320))
TRUTH_ARRAY_SHA256 = "c0e7fa6c5581522a044b3d7dbcecb3d54744be654f7960b016becb57b3410a6e"
TRUTH_COUNTS = {"valid": 1674807, "material": 1241241, "recto": 444149,
                "boundary_poor": 0}
RENDERER = {
    "bytes": 12343,
    "sha256": "2697486f4b94f13797e16c74a18f937d1860e3a5d728121bf27307aba269d469",
}
TOOL_NAMES = (
    "run_crossscan_scrollfiesta_downstream.py",
    "crossscan_scrollfiesta_metrics.py",
    "crossscan_scrollfiesta_obj.py",
    "crossscan_scrollfiesta_adapter.py",
    "verify_physical_label_semantics.py",
)
CHILD_ENV_ALLOWLIST = {
    "COMSPEC", "LANG", "LC_ALL", "LC_CTYPE", "PATH", "PATHEXT",
    "SYSTEMROOT", "TEMP", "TMP", "TMPDIR", "WINDIR",
}
PIPELINE_TIMEOUT_SECONDS = 12 * 60 * 60
RENDERER_TIMEOUT_SECONDS = 2 * 60 * 60


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json_exclusive(path: Path, value: dict) -> None:
    A._create_bytes(Path(path), _json_bytes(value))


def validate_metric_lock(path: Path | None = None) -> tuple[dict, bytes]:
    lock_path = Path(path) if path is not None else Path(__file__).with_name(
        "crossscan_scrollfiesta_metric_lock.json"
    )
    value, payload = A._load_hashed_json(lock_path, "downstream metric lock")
    expected = {
        "schema_version": METRIC_LOCK_SCHEMA,
        "status": "locked_before_final_or_downstream_candidate_prediction_was_inspected",
        "parent_downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
        "content_sha256": METRIC_LOCK_CONTENT_SHA256,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ValueError(f"downstream metric lock has invalid {key}")
    if value.get("physical_metric", {}).get("truth_box_array_sha256") != TRUTH_ARRAY_SHA256:
        raise ValueError("downstream metric lock truth identity mismatch")
    return value, payload


def _runtime_identity() -> dict:
    import scipy
    import zarr
    from PIL import __version__ as pillow_version

    executable = Path(sys.executable)
    return {
        "platform_system": platform.system(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "tifffile": tifffile.__version__,
        "Pillow": pillow_version,
        "zarr": zarr.__version__,
        "python_executable": {
            "path": str(executable.resolve()),
            **A.file_record(executable),
        },
    }


def validate_runtime(metric_lock: dict) -> dict:
    expected = metric_lock.get("production_runtime")
    actual = _runtime_identity()
    comparable = {key: actual.get(key) for key in (
        "platform_system", "python_implementation", "python_version",
        "numpy", "scipy", "tifffile", "Pillow", "zarr",
    )}
    if expected != comparable:
        raise ValueError(
            f"downstream runtime differs from the metric lock: "
            f"expected={expected}, actual={comparable}"
        )
    return actual


def _same_runtime(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    left_value = dict(left)
    right_value = dict(right)
    for value in (left_value, right_value):
        executable = value.get("python_executable")
        if isinstance(executable, dict):
            executable = dict(executable)
            executable.pop("path", None)
            value["python_executable"] = executable
    return left_value == right_value


def _validate_file(path: Path, expected: dict, description: str) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} is missing or a symlink: {path}")
    record = A.file_record(path)
    if record != expected:
        raise ValueError(f"{description} identity mismatch: {path}")
    return {"path": str(path.resolve()), **record}


def validate_binaries(directory: Path) -> dict[str, dict]:
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"binary directory is missing or a symlink: {directory}")
    records = {}
    for name, expected in PINNED_BINARIES.items():
        records[name] = _validate_file(directory / name, expected, name)
    return records


def _sanitized_environment() -> tuple[dict[str, str], list[str]]:
    environment = {
        key: value for key, value in os.environ.items()
        if key.upper() in CHILD_ENV_ALLOWLIST
    }
    removed = sorted(key for key in os.environ if key not in environment)
    return environment, removed


def _recompute_grid_set(
    receipt_path: Path,
    baseline_probability: Path,
    candidate_probability: Path,
    grids: dict[str, Path],
) -> tuple[dict, bytes]:
    receipt, payload = A._load_hashed_json(receipt_path, "three-arm grid-set receipt")
    if (
        receipt.get("schema_version") != "crossscan-scrollfiesta-grid-set-v1"
        or receipt.get("status") != "PASS"
        or receipt.get("downstream_lock_content_sha256")
        != A.DOWNSTREAM_LOCK_CONTENT_SHA256
    ):
        raise ValueError("grid-set receipt is not the locked PASS artifact")
    with tempfile.TemporaryDirectory(prefix="crossscan-grid-set-recheck-") as tmp:
        check_path = Path(tmp) / "grid_set.json"
        recomputed = A.verify_grid_set(
            baseline_probability,
            candidate_probability,
            grids["baseline-fixed"],
            grids["candidate-fixed"],
            grids["candidate-matched-mass"],
            check_path,
        )
        if receipt != recomputed or payload != check_path.read_bytes():
            raise ValueError("supplied grid-set receipt differs from full recomputation")
    return receipt, payload


def _load_truth(
    labels_root: Path, semantic_audit: Path, grid_set: dict
) -> tuple[np.ndarray, dict, bytes]:
    semantic, semantic_payload = V.validate_audit_receipt(semantic_audit)
    if semantic.get("content_sha256") != grid_set.get("semantic_audit_content_sha256"):
        raise ValueError("truth semantic audit differs from the three-arm grid set")
    import zarr

    store_path = Path(labels_root) / V.EXPECTED["PHerc0139"]["path"]
    if not store_path.is_dir() or store_path.is_symlink():
        raise ValueError(f"PHerc0139 physical label store is missing: {store_path}")
    store = zarr.open(str(store_path), mode="r")
    truth = np.asarray(store[TRUTH_BOX])
    if truth.dtype != np.uint8:
        raise ValueError(f"PHerc0139 truth box must be uint8, got {truth.dtype}")
    if truth.shape != (128, 128, 128) or not truth.flags.c_contiguous:
        truth = np.ascontiguousarray(truth, dtype=np.uint8)
    digest = A.sha256_array(truth)
    counts = {
        "valid": int(np.count_nonzero(truth & np.uint8(1))),
        "material": int(np.count_nonzero(truth & np.uint8(2))),
        "recto": int(np.count_nonzero(truth & np.uint8(8))),
        "boundary_poor": int(np.count_nonzero(truth & np.uint8(16))),
    }
    if digest != TRUTH_ARRAY_SHA256 or counts != TRUTH_COUNTS:
        raise ValueError("PHerc0139 locked truth box identity mismatch")
    return truth, {
        "store": str(store_path.resolve()),
        "box_local_l1_zyx": [192, 320, 1280, 1408, 192, 320],
        "shape": [128, 128, 128],
        "dtype": "uint8",
        "array_sha256": digest,
        "counts": counts,
        "semantic_audit": {
            **A.file_record(semantic_audit),
            "content_sha256": semantic["content_sha256"],
            "file_sha256": hashlib.sha256(semantic_payload).hexdigest(),
        },
    }, semantic_payload


def _assemble_grid_volume(grid: Path, role: str) -> np.ndarray:
    if role not in ("PRED", "RAW"):
        raise ValueError("grid volume role must be PRED or RAW")
    directory = Path(grid) / f"cubes_{role}"
    volume = np.zeros(A.SHAPE, dtype=np.uint8)
    for (z, y, x), name in A._cube_specs():
        value = tifffile.imread(directory / name)
        if value.shape != (128, 128, 128) or value.dtype != np.uint8:
            raise ValueError(f"invalid {role} cube: {directory / name}")
        volume[z:z + 128, y:y + 128, x:x + 128] = value
    if role == "PRED" and not set(map(int, np.unique(volume))) <= {0, 255}:
        raise ValueError("assembled PRED volume is not binary 0/255")
    return volume


def _copy_file_exclusive(source: Path, destination: Path) -> None:
    source = Path(source)
    destination = Path(destination)
    if not source.is_file() or _is_linklike(source):
        raise ValueError(f"snapshot source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _copy_tree_exclusive(source: Path, destination: Path) -> list[dict]:
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir() or _is_linklike(source):
        raise ValueError(f"snapshot source is not a real directory: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    source_files = A._regular_file_universe(source)
    for relative in sorted(source_files):
        _copy_file_exclusive(source / relative, destination / relative)
    return _hash_tree(destination)


def _snapshot_grids(
    output: Path,
    source_grids: dict[str, Path],
    expected_manifests: dict[str, dict],
) -> tuple[dict[str, Path], dict[str, dict]]:
    snapshots = {}
    records = {}
    for arm in ARM_ORDER:
        destination = Path(output) / "inputs" / "grids" / arm
        files = _copy_tree_exclusive(source_grids[arm], destination)
        manifest = A.verify_scrollfiesta_grid(destination)
        if manifest != expected_manifests[arm]:
            raise ValueError(f"{arm} private snapshot differs from verified source grid")
        snapshots[arm] = destination
        records[arm] = {
            "path": destination.relative_to(output).as_posix(),
            "manifest_content_sha256": manifest["content_sha256"],
            "files": files,
        }
    return snapshots, records


def _snapshot_tools(
    output: Path, binary_dir: Path, renderer_script: Path
) -> tuple[Path, Path, dict, dict]:
    destination = Path(output) / "inputs" / "tools"
    destination.mkdir(parents=True, exist_ok=False)
    for name in PINNED_BINARIES:
        _copy_file_exclusive(Path(binary_dir) / name, destination / name)
    renderer = destination / "render_mesh.py"
    _copy_file_exclusive(renderer_script, renderer)
    binaries = validate_binaries(destination)
    renderer_record = _validate_file(renderer, RENDERER, "snapshotted renderer")
    return destination, renderer, binaries, renderer_record


def _write_truth_snapshot(output: Path, truth: np.ndarray) -> dict:
    path = Path(output) / "inputs" / "truth_l1.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, truth, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    roundtrip = np.load(path, allow_pickle=False)
    if (
        roundtrip.dtype != np.uint8
        or roundtrip.shape != (128, 128, 128)
        or A.sha256_array(roundtrip) != TRUTH_ARRAY_SHA256
    ):
        raise ValueError("truth snapshot round trip differs from locked truth")
    return {"path": path.relative_to(output).as_posix(), **A.file_record(path)}


def _expected_cube_ids() -> set[str]:
    return {Path(name).stem for _, name in A._cube_specs()}


def _parse_summary(path: Path) -> dict:
    expected = _expected_cube_ids()
    if not path.is_file() or path.is_symlink():
        raise ValueError("pipeline summary is missing or a symlink")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["cube_id", "exit_code", "wall_seconds"]:
            raise ValueError("pipeline summary header mismatch")
        rows = []
        seen = set()
        for row in reader:
            if set(row) != set(reader.fieldnames):
                raise ValueError("pipeline summary row schema mismatch")
            cube = row["cube_id"]
            if cube in seen:
                raise ValueError("pipeline summary contains a duplicate cube")
            seen.add(cube)
            try:
                exit_code = int(row["exit_code"])
                wall = float(row["wall_seconds"])
            except ValueError as error:
                raise ValueError("pipeline summary contains an invalid number") from error
            if exit_code != 0 or not np.isfinite(wall) or wall < 0:
                raise ValueError(f"pipeline cube did not complete cleanly: {cube}")
            rows.append({"cube_id": cube, "exit_code": exit_code,
                         "wall_seconds": wall})
    if seen != expected:
        raise ValueError("pipeline summary cube universe mismatch")
    return {"cube_count": len(rows), "rows": sorted(rows, key=lambda item: item["cube_id"])}


def _is_linklike(path: Path) -> bool:
    try:
        return path.is_symlink() or (
            hasattr(path, "is_junction") and path.is_junction()
        )
    except OSError:
        return True


def _inventory_tree(
    root: Path, *, excluded: Iterable[str] = ()
) -> tuple[list[dict], list[dict]]:
    root = Path(root)
    if not root.is_dir() or _is_linklike(root):
        raise ValueError(f"artifact root is not a real directory: {root}")
    excluded_set = set(excluded)
    records = []
    invalid = []

    def walk_error(error: OSError) -> None:
        filename = Path(error.filename) if error.filename else root
        try:
            relative = filename.relative_to(root).as_posix()
        except ValueError:
            relative = "."
        invalid.append({"path": relative, "kind": "unreadable-directory"})

    for current, directories, files in os.walk(
        root, topdown=True, followlinks=False, onerror=walk_error
    ):
        current_path = Path(current)
        for name in list(directories):
            path = current_path / name
            if _is_linklike(path) or not path.is_dir():
                directories.remove(name)
                invalid.append({
                    "path": path.relative_to(root).as_posix(),
                    "kind": "linked-or-special-directory",
                })
        for name in files:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative in excluded_set:
                continue
            if _is_linklike(path) or not path.is_file():
                invalid.append({"path": relative, "kind": "linked-or-special-file"})
                continue
            try:
                records.append(A.file_record(path, relative))
            except OSError:
                invalid.append({"path": relative, "kind": "unreadable-file"})
    return (
        sorted(records, key=lambda item: item["path"]),
        sorted(invalid, key=lambda item: (item["path"], item["kind"])),
    )


def _hash_tree(root: Path, *, excluded: Iterable[str] = ()) -> list[dict]:
    records, invalid = _inventory_tree(root, excluded=excluded)
    if invalid:
        raise ValueError(f"artifact contains forbidden entries: {invalid}")
    return records


def _safe_inventory(root: Path) -> tuple[list[dict], list[dict]]:
    root = Path(root)
    if not root.exists() and not _is_linklike(root):
        return [], []
    try:
        return _inventory_tree(root)
    except Exception as error:
        return [], [{
            "path": ".",
            "kind": "unreadable-artifact-root",
            "error_type": type(error).__name__,
        }]


def _pipeline_command(binary_dir: Path, grid: Path, output: Path) -> list[str]:
    return [
        str((Path(binary_dir) / "grid_pipeline.exe").resolve()),
        str(Path(grid).resolve()),
        str(Path(output).resolve()),
        *PIPELINE_TAIL,
        "--exe", str((Path(binary_dir) / "cube_mesh.exe").resolve()),
        "--weld", str((Path(binary_dir) / "grid_weld.exe").resolve()),
    ]


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        process.kill()
    try:
        process.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        process.kill()
        process.wait()


def _run_command(
    command: list[str],
    log_path: Path,
    environment: dict[str, str],
    *,
    timeout_seconds: float,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("xb") as log:
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(
                    subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                )
            else:
                kwargs["start_new_session"] = True
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                **kwargs,
            )
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_tree(process)
                return {
                    "argv": command,
                    "exit_code": None,
                    "timeout_seconds": timeout_seconds,
                    "error": f"TimeoutExpired: command exceeded {timeout_seconds} seconds",
                }
        return {"argv": command, "exit_code": int(exit_code),
                "timeout_seconds": timeout_seconds, "error": None}
    except Exception as error:  # preserve a sealed bounded failure and continue arms
        return {"argv": command, "exit_code": None,
                "timeout_seconds": timeout_seconds,
                "error": f"{type(error).__name__}: {error}"}


def _renderer_command(renderer: Path, obj: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(renderer).resolve()),
        "--vcolors",
        "--box=3840,3712,1344,256",
        str(Path(obj).resolve()),
        str(Path(output).resolve()),
        "1200", "1000", "0.55,0.5,0.67", "1.0", "2.0",
    ]


def _run_renderer(
    renderer: Path,
    obj: Path,
    output: Path,
    environment: dict[str, str],
) -> dict:
    command = _renderer_command(renderer, obj, output)
    return _run_command(
        command,
        output.with_suffix(".log"),
        environment,
        timeout_seconds=RENDERER_TIMEOUT_SECONDS,
    )


def _validate_png(path: Path, expected_size: tuple[int, int], description: str) -> dict:
    from PIL import Image

    path = Path(path)
    if not path.is_file() or _is_linklike(path):
        raise ValueError(f"{description} is missing or linked")
    try:
        with Image.open(path) as opened:
            if (
                opened.format != "PNG"
                or opened.size != expected_size
                or opened.mode != "RGB"
            ):
                raise ValueError(
                    f"{description} must be RGB PNG {expected_size}, got "
                    f"{opened.format} {opened.size} {opened.mode}"
                )
            pixels = np.asarray(opened.copy(), dtype=np.uint8)
    except (OSError, SyntaxError) as error:
        raise ValueError(f"{description} is not a decodable PNG") from error
    if pixels.shape != (expected_size[1], expected_size[0], 3):
        raise ValueError(f"{description} decoded shape mismatch")
    if int(pixels.min()) == int(pixels.max()):
        raise ValueError(f"{description} is a trivial single-colour image")
    return A.file_record(path)


def _validate_arm_output(
    arm: str,
    grid: Path,
    output: Path,
    pipeline: dict,
    renderer: dict,
) -> dict:
    if pipeline.get("exit_code") != 0 or pipeline.get("error") is not None:
        raise ValueError(f"{arm} direct grid_pipeline command failed")
    if renderer.get("exit_code") != 0 or renderer.get("error") is not None:
        raise ValueError(f"{arm} fixed mesh renderer failed")
    required = (
        "pipeline_summary.csv", "rejected_cubes.txt", "scrollslice.source.json",
        "grid_weld.log", "welded.obj", "welded.obj.weld_report.json",
        "welded.obj.bad_edges.obj", "mesh_fixed_camera.png", "mesh_fixed_camera.log",
        "pipeline_driver.log",
    )
    for relative in required:
        path = output / relative
        if not path.is_file() or _is_linklike(path):
            raise ValueError(f"{arm} is missing required output {relative}")
    _validate_png(
        output / "mesh_fixed_camera.png", (1200, 1000),
        f"{arm} fixed-camera mesh image",
    )
    if (output / "rejected_cubes.txt").read_text(encoding="utf-8").strip():
        raise ValueError(f"{arm} introduced a garbage/empty cube rejection")
    summary = _parse_summary(output / "pipeline_summary.csv")
    source = json.loads((output / "scrollslice.source.json").read_bytes())
    if (
        source.get("format") != "vesuvius-scrollslice-source-v1"
        or source.get("mesh_axes") != "zyx"
        or source.get("mesh") != "welded.obj"
        or Path(source.get("dataset", "")).resolve() != Path(grid).resolve()
    ):
        raise ValueError(f"{arm} scrollslice source points to a different grid")
    cube_ids = _expected_cube_ids()
    logs = {path.name for path in (output / "logs").iterdir() if path.is_file()}
    if logs != {f"{cube}.log" for cube in cube_ids}:
        raise ValueError(f"{arm} per-cube log universe mismatch")
    final_meshes = []
    for cube in sorted(cube_ids):
        path = output / "dump" / cube / f"{cube}_step12_final" / f"{cube}_step12_final_all.obj"
        if not path.is_file() or path.is_symlink() or path.stat().st_size < 128:
            raise ValueError(f"{arm} lacks a complete final per-cube mesh: {cube}")
        final_meshes.append(A.file_record(path, path.relative_to(output).as_posix()))
    obj_audit = O.audit_scrollfiesta_obj(
        output / "welded.obj", output / "welded.obj.weld_report.json"
    )
    return {
        "status": "PASS",
        "arm": arm,
        "grid": str(Path(grid).resolve()),
        "pipeline": pipeline,
        "renderer": renderer,
        "summary": summary,
        "final_per_cube_meshes": final_meshes,
        "mesh_audit": obj_audit,
        "files": _hash_tree(output),
    }


def _scrollfiesta_gate(arms: dict[str, dict]) -> dict:
    details = {}
    all_valid = all(arms.get(arm, {}).get("status") == "PASS" for arm in ARM_ORDER)
    if not all_valid:
        return {"pass": False, "all_three_valid": False, "details": details}
    baseline = arms["baseline-fixed"]["mesh_audit"]
    base_report = baseline["weld_report"]["manifold_audit"]
    base_edge = baseline["obj"]["edge_audit"]
    passed = True
    for arm in ("candidate-fixed", "candidate-matched-mass"):
        audit = arms[arm]["mesh_audit"]
        report = audit["weld_report"]["manifold_audit"]
        edge = audit["obj"]["edge_audit"]
        plane_checks = {
            axis: edge["internal_seam_unpaired_edges_by_plane"][axis]
            <= base_edge["internal_seam_unpaired_edges_by_plane"][axis]
            for axis in ("z", "y", "x")
        }
        checks = {
            "non_manifold_zero": report["non_manifold"] == 0,
            "pinch_verts_zero": report["pinch_verts"] == 0,
            "same_dir_not_above_baseline": (
                report["same_dir_pairs"] <= base_report["same_dir_pairs"]
            ),
            "internal_seam_union_not_above_baseline": (
                edge["internal_seam_unpaired_edges_union"]
                <= base_edge["internal_seam_unpaired_edges_union"]
            ),
            "internal_seam_each_plane_not_above_baseline": all(plane_checks.values()),
            "strict_clean_mesh": report["same_dir_pairs"] == 0,
        }
        details[arm] = {"checks": checks, "plane_checks": plane_checks}
        passed &= all(value for key, value in checks.items() if key != "strict_clean_mesh")
    return {"pass": bool(passed), "all_three_valid": True, "details": details}


def _render_cross_sections(
    raw: np.ndarray,
    truth_l1: np.ndarray,
    masks: dict[str, np.ndarray],
    output: Path,
) -> list[dict]:
    from PIL import Image, __version__ as pillow_version

    output.mkdir(parents=True, exist_ok=False)
    truth_positive = ((truth_l1 & 1) != 0) & ((truth_l1 & 8) != 0)
    truth = np.repeat(np.repeat(np.repeat(truth_positive, 2, 0), 2, 1), 2, 2)
    records = []
    colors = {"truth": np.asarray([30, 230, 90], dtype=np.float32),
              "prediction": np.asarray([245, 60, 210], dtype=np.float32)}
    for axis, name in enumerate(("z", "y", "x")):
        selector = [slice(None)] * 3
        selector[axis] = 128
        selector = tuple(selector)
        ct = raw[selector]
        grey = np.repeat(ct[:, :, None], 3, axis=2).astype(np.float32)
        panels = [grey.astype(np.uint8)]
        for mask, color in [(truth[selector], colors["truth"])] + [
            (masks[arm][selector], colors["prediction"]) for arm in ARM_ORDER
        ]:
            panel = grey.copy()
            selected = mask if mask.dtype == np.bool_ else mask == 255
            panel[selected] = panel[selected] * 0.45 + color * 0.55
            panels.append(np.clip(panel, 0, 255).astype(np.uint8))
        image = Image.fromarray(np.concatenate(panels, axis=1), mode="RGB")
        path = output / f"center_{name}.png"
        image.save(path, format="PNG", optimize=False, compress_level=9)
        _validate_png(path, (1280, 256), f"fixed centre {name} cross-section")
        records.append(A.file_record(path, path.relative_to(output.parent).as_posix()))
    return [{"panel_order": ["CT", "truth", *ARM_ORDER],
             "pillow_version": pillow_version, "files": records}]


def _copy_provenance(output: Path, metric_lock_payload: bytes,
                     grid_set_payload: bytes, semantic_payload: bytes,
                     renderer: Path) -> list[dict]:
    sources = {
        "provenance/crossscan_scrollfiesta_downstream_lock.json":
            Path(A.__file__).with_name("crossscan_scrollfiesta_downstream_lock.json").read_bytes(),
        "provenance/crossscan_scrollfiesta_metric_lock.json": metric_lock_payload,
        "provenance/crossscan_scrollfiesta_grid_set.json": grid_set_payload,
        "provenance/physical_label_semantic_audit.json": semantic_payload,
        "provenance/scrollfiesta_render_mesh.py": Path(renderer).read_bytes(),
    }
    for name in TOOL_NAMES:
        sources[f"provenance/{name}"] = Path(__file__).with_name(name).read_bytes()
    records = []
    for relative, payload in sources.items():
        A._create_bytes(output / relative, payload)
        records.append(A.file_record(output / relative, relative))
    return sorted(records, key=lambda item: item["path"])


def _require_boolean(value: object, description: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{description} must be boolean")
    return value


def _verify_terminal_logic(receipt: dict) -> None:
    components = {
        "physical_pass": receipt.get("physical", {}).get("pass"),
        "scrollfiesta_pass": receipt.get("scrollfiesta_gate", {}).get("pass"),
        "visuals_pass": receipt.get("visual_evidence", {}).get("pass"),
        "input_integrity_pass": receipt.get("input_integrity", {}).get("pass"),
        "artifact_integrity_pass": receipt.get("artifact_integrity", {}).get("pass"),
    }
    for key, value in components.items():
        _require_boolean(value, key)
    terminal = receipt.get("terminal_gate")
    if not isinstance(terminal, dict) or set(terminal) != {*components, "pass"}:
        raise ValueError("downstream terminal gate schema mismatch")
    terminal_pass = _require_boolean(terminal.get("pass"), "terminal pass")
    if any(terminal[key] is not value for key, value in components.items()):
        raise ValueError("downstream terminal gate component mismatch")
    expected_pass = all(components.values())
    if terminal_pass != expected_pass:
        raise ValueError("downstream terminal gate conjunction mismatch")
    if (receipt.get("status") == "PASS") != terminal_pass:
        raise ValueError("downstream status differs from terminal gate")
    expected_claim = (
        "bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement"
        if terminal_pass else
        "bounded negative result; no downstream-improvement claim is authorized"
    )
    if receipt.get("claim_boundary") != expected_claim:
        raise ValueError("downstream claim boundary differs from terminal status")


def _deep_verify_pass(output: Path, receipt: dict) -> None:
    metric_lock, _ = validate_metric_lock(
        output / "provenance" / "crossscan_scrollfiesta_metric_lock.json"
    )
    A.validate_downstream_lock(
        output / "provenance" / "crossscan_scrollfiesta_downstream_lock.json"
    )
    if not _same_runtime(validate_runtime(metric_lock), receipt.get("runtime")):
        raise ValueError("current runtime differs from sealed downstream runtime")

    binary_dir = output / "inputs" / "tools"
    renderer = binary_dir / "render_mesh.py"
    actual_binaries = validate_binaries(binary_dir)
    sealed_binaries = receipt.get("binaries")
    if not isinstance(sealed_binaries, dict) or set(sealed_binaries) != set(PINNED_BINARIES):
        raise ValueError("sealed binary record universe mismatch")
    for name in PINNED_BINARIES:
        sealed = sealed_binaries[name]
        actual = actual_binaries[name]
        if (
            not isinstance(sealed, dict)
            or Path(sealed.get("path", "")).name != name
            or {key: sealed.get(key) for key in ("bytes", "sha256")}
            != {key: actual.get(key) for key in ("bytes", "sha256")}
        ):
            raise ValueError(f"sealed {name} record differs from staged tool")
    actual_renderer = _validate_file(renderer, RENDERER, "sealed renderer")
    sealed_renderer = receipt.get("renderer")
    if (
        not isinstance(sealed_renderer, dict)
        or Path(sealed_renderer.get("path", "")).name != "render_mesh.py"
        or {key: sealed_renderer.get(key) for key in ("bytes", "sha256")}
        != {key: actual_renderer.get(key) for key in ("bytes", "sha256")}
    ):
        raise ValueError("sealed renderer record mismatch")

    snapshots = receipt.get("grid_snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != set(ARM_ORDER):
        raise ValueError("sealed result has the wrong grid snapshot universe")
    grids = {}
    manifests = {}
    masks = {}
    for arm in ARM_ORDER:
        expected_path = f"inputs/grids/{arm}"
        record = snapshots[arm]
        if not isinstance(record, dict) or record.get("path") != expected_path:
            raise ValueError(f"{arm} grid snapshot path mismatch")
        grid = output / expected_path
        manifest = A.verify_scrollfiesta_grid(grid)
        if manifest.get("content_sha256") != record.get("manifest_content_sha256"):
            raise ValueError(f"{arm} grid snapshot manifest mismatch")
        if _hash_tree(grid) != record.get("files"):
            raise ValueError(f"{arm} grid snapshot file records mismatch")
        grids[arm] = grid
        manifests[arm] = manifest
        masks[arm] = _assemble_grid_volume(grid, "PRED")

    truth_record = receipt.get("truth_snapshot")
    if not isinstance(truth_record, dict) or truth_record.get("path") != "inputs/truth_l1.npy":
        raise ValueError("sealed truth snapshot record mismatch")
    truth_path = output / truth_record["path"]
    if {"path": truth_record["path"], **A.file_record(truth_path)} != truth_record:
        raise ValueError("sealed truth snapshot file mismatch")
    truth = np.load(truth_path, allow_pickle=False)
    if (
        truth.shape != (128, 128, 128)
        or truth.dtype != np.uint8
        or A.sha256_array(truth) != TRUTH_ARRAY_SHA256
    ):
        raise ValueError("sealed truth snapshot array mismatch")

    physical_path = output / "physical_metrics.json"
    try:
        physical = json.loads(physical_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("sealed physical metrics are not valid JSON") from error
    scores = {arm: M.score_mask(masks[arm], truth) for arm in ARM_ORDER}
    acceptance = M.evaluate_acceptance(
        scores["baseline-fixed"], scores["candidate-fixed"],
        scores["candidate-matched-mass"],
    )
    if physical != {"scores": scores, "acceptance": acceptance}:
        raise ValueError("sealed physical metrics differ from full recomputation")
    if receipt.get("physical") != {
        "result": "physical_metrics.json", "error": None,
        "pass": bool(acceptance["pass"]),
    }:
        raise ValueError("sealed physical gate envelope mismatch")

    embedded_arms = receipt.get("arms")
    if not isinstance(embedded_arms, dict) or set(embedded_arms) != set(ARM_ORDER):
        raise ValueError("sealed arm universe mismatch")
    recomputed_arms = {}
    for arm in ARM_ORDER:
        embedded = embedded_arms[arm]
        if embedded.get("status") != "PASS":
            raise ValueError(f"terminal PASS contains a failed arm: {arm}")
        arm_output = output / "arms" / arm
        recorded_pipeline = embedded.get("pipeline", {}).get("argv")
        if not isinstance(recorded_pipeline, list) or len(recorded_pipeline) < 7:
            raise ValueError(f"{arm} has no recorded pipeline command")
        recorded_grid = Path(embedded.get("grid", ""))
        recorded_output = Path(recorded_pipeline[2])
        recorded_binary_dir = Path(sealed_binaries["grid_pipeline.exe"]["path"]).parent
        expected_pipeline = _pipeline_command(
            recorded_binary_dir, recorded_grid, recorded_output
        )
        if (
            recorded_pipeline != expected_pipeline
            or embedded.get("pipeline", {}).get("timeout_seconds")
            != PIPELINE_TIMEOUT_SECONDS
            or recorded_output.name != arm
            or recorded_grid.name != arm
        ):
            raise ValueError(f"{arm} pipeline command differs from the lock")
        expected_renderer = _renderer_command(
            Path(sealed_renderer["path"]), recorded_output / "welded.obj",
            recorded_output / "mesh_fixed_camera.png",
        )
        expected_renderer[0] = receipt["runtime"]["python_executable"]["path"]
        if (
            embedded.get("renderer", {}).get("argv") != expected_renderer
            or embedded.get("renderer", {}).get("timeout_seconds")
            != RENDERER_TIMEOUT_SECONDS
        ):
            raise ValueError(f"{arm} renderer command differs from the lock")
        recomputed = _validate_arm_output(
            arm, recorded_grid, arm_output,
            embedded["pipeline"], embedded["renderer"],
        )
        if recomputed != embedded:
            raise ValueError(f"{arm} sealed audit differs from recomputation")
        recomputed_arms[arm] = recomputed
    if _scrollfiesta_gate(recomputed_arms) != receipt.get("scrollfiesta_gate"):
        raise ValueError("sealed ScrollFiesta gate differs from recomputation")

    visual = receipt.get("visual_evidence", {})
    cross_sections = visual.get("cross_sections")
    if not isinstance(cross_sections, list) or len(cross_sections) != 1:
        raise ValueError("sealed cross-section evidence schema mismatch")
    files = cross_sections[0].get("files")
    if not isinstance(files, list) or len(files) != 3:
        raise ValueError("sealed cross-section evidence universe mismatch")
    expected_visual_records = []
    for axis in ("z", "y", "x"):
        path = output / "visuals" / f"center_{axis}.png"
        _validate_png(path, (1280, 256), f"sealed centre {axis} cross-section")
        expected_visual_records.append(
            A.file_record(path, f"visuals/center_{axis}.png")
        )
    if files != expected_visual_records:
        raise ValueError("sealed cross-section records mismatch")


def verify_result(
    output: Path,
    *,
    deep: bool = True,
    expected_content_sha256: str | None = None,
) -> dict:
    output = Path(output)
    if not output.is_dir() or _is_linklike(output):
        raise ValueError(f"downstream result root is not a real directory: {output}")
    receipt, _ = A._load_hashed_json(output / "terminal_receipt.json",
                                     "downstream terminal receipt")
    if receipt.get("schema_version") != SCHEMA or receipt.get("status") not in ("PASS", "FAIL"):
        raise ValueError("invalid downstream terminal receipt header")
    if expected_content_sha256 is not None and (
        receipt.get("content_sha256") != expected_content_sha256
    ):
        raise ValueError("downstream receipt differs from the externally pinned digest")
    if (
        receipt.get("downstream_lock_content_sha256")
        != A.DOWNSTREAM_LOCK_CONTENT_SHA256
        or receipt.get("metric_lock_content_sha256")
        != METRIC_LOCK_CONTENT_SHA256
    ):
        raise ValueError("downstream receipt lock identity mismatch")
    actual, invalid = _inventory_tree(
        output, excluded=("terminal_receipt.json",)
    )
    if receipt.get("files") != actual:
        raise ValueError("downstream result file universe or hash mismatch")
    artifact = receipt.get("artifact_integrity")
    if (
        not isinstance(artifact, dict)
        or artifact.get("invalid_output_entries") != invalid
        or artifact.get("pass") is not (not invalid)
    ):
        raise ValueError("downstream invalid-entry inventory mismatch")
    _verify_terminal_logic(receipt)
    if deep and receipt["status"] == "PASS":
        _deep_verify_pass(output, receipt)
    return receipt


def execute(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    inputs = [
        args.grid_set_receipt, args.baseline_probability, args.candidate_probability,
        args.baseline_grid, args.candidate_fixed_grid, args.candidate_matched_grid,
        args.labels_root, args.semantic_audit, args.binary_dir, args.renderer_script,
    ]
    if args.metric_lock is not None:
        inputs.append(args.metric_lock)
    A._require_output_disjoint(output, [Path(value) for value in inputs])
    if output.exists():
        raise FileExistsError(f"refusing to replace downstream output: {output}")
    metric_lock, metric_lock_payload = validate_metric_lock(args.metric_lock)
    A.validate_downstream_lock()
    runtime = validate_runtime(metric_lock)
    validate_binaries(args.binary_dir)
    _validate_file(args.renderer_script, RENDERER, "fixed renderer")
    source_grids = {
        "baseline-fixed": Path(args.baseline_grid),
        "candidate-fixed": Path(args.candidate_fixed_grid),
        "candidate-matched-mass": Path(args.candidate_matched_grid),
    }
    grid_set, grid_set_payload = _recompute_grid_set(
        args.grid_set_receipt, args.baseline_probability,
        args.candidate_probability, source_grids,
    )
    truth, truth_record, semantic_payload = _load_truth(
        args.labels_root, args.semantic_audit, grid_set
    )
    manifests = {
        arm: A.verify_scrollfiesta_grid(grid)
        for arm, grid in source_grids.items()
    }

    output.mkdir(parents=True, exist_ok=False)
    grids, grid_snapshots = _snapshot_grids(output, source_grids, manifests)
    binary_dir, renderer_script, binaries, renderer_record = _snapshot_tools(
        output, args.binary_dir, args.renderer_script
    )
    truth_snapshot = _write_truth_snapshot(output, truth)
    masks = {arm: _assemble_grid_volume(grid, "PRED") for arm, grid in grids.items()}
    raw = _assemble_grid_volume(grids["baseline-fixed"], "RAW")
    for arm, mask in masks.items():
        if int(np.count_nonzero(mask)) != manifests[arm]["foreground_voxels"]:
            raise ValueError(f"{arm} assembled foreground count differs from manifest")

    provenance = _copy_provenance(
        output, metric_lock_payload, grid_set_payload, semantic_payload,
        renderer_script,
    )
    environment, removed_environment = _sanitized_environment()
    arms = {}
    for arm in ARM_ORDER:
        arm_output = output / "arms" / arm
        command = _pipeline_command(binary_dir, grids[arm], arm_output)
        pipeline = {
            "argv": command, "exit_code": None,
            "timeout_seconds": PIPELINE_TIMEOUT_SECONDS,
            "error": "pipeline was not started",
        }
        renderer = {
            "argv": None, "exit_code": None,
            "timeout_seconds": RENDERER_TIMEOUT_SECONDS,
            "error": "renderer was not started",
        }
        try:
            if A.verify_scrollfiesta_grid(grids[arm]) != manifests[arm]:
                raise ValueError(f"{arm} staged grid changed before execution")
            pipeline = _run_command(
                command,
                arm_output / "pipeline_driver.log",
                environment,
                timeout_seconds=PIPELINE_TIMEOUT_SECONDS,
            )
            mesh = arm_output / "welded.obj"
            if mesh.is_file() and not _is_linklike(mesh):
                renderer = _run_renderer(
                    renderer_script, mesh,
                    arm_output / "mesh_fixed_camera.png", environment,
                )
            else:
                renderer["error"] = "welded.obj was not produced"
            if A.verify_scrollfiesta_grid(grids[arm]) != manifests[arm]:
                raise ValueError(f"{arm} staged grid changed during execution")
            arms[arm] = _validate_arm_output(
                arm, grids[arm], arm_output, pipeline, renderer
            )
        except Exception as error:
            files, invalid_entries = _safe_inventory(arm_output)
            arms[arm] = {
                "status": "FAIL", "arm": arm, "pipeline": pipeline,
                "renderer": renderer, "error": f"{type(error).__name__}: {error}",
                "files": files, "invalid_entries": invalid_entries,
            }

    physical = {}
    physical_error = None
    try:
        scores = {arm: M.score_mask(masks[arm], truth) for arm in ARM_ORDER}
        acceptance = M.evaluate_acceptance(
            scores["baseline-fixed"], scores["candidate-fixed"],
            scores["candidate-matched-mass"],
        )
        physical = {"scores": scores, "acceptance": acceptance}
        _write_json_exclusive(output / "physical_metrics.json", physical)
    except Exception as error:
        physical_error = f"{type(error).__name__}: {error}"

    visual_error = None
    visual_evidence = []
    try:
        visual_evidence = _render_cross_sections(
            raw, truth, masks, output / "visuals"
        )
    except Exception as error:
        visual_error = f"{type(error).__name__}: {error}"

    input_revalidation_error = None
    try:
        for arm in ARM_ORDER:
            if A.verify_scrollfiesta_grid(grids[arm]) != manifests[arm]:
                raise ValueError(f"{arm} staged grid changed after execution")
        if validate_binaries(binary_dir) != binaries:
            raise ValueError("staged binaries changed after execution")
        if _validate_file(
            renderer_script, RENDERER, "staged renderer"
        ) != renderer_record:
            raise ValueError("staged renderer changed after execution")
        if not _same_runtime(validate_runtime(metric_lock), runtime):
            raise ValueError("downstream runtime changed during execution")
        staged_truth = np.load(output / truth_snapshot["path"], allow_pickle=False)
        if A.sha256_array(staged_truth) != TRUTH_ARRAY_SHA256:
            raise ValueError("staged truth changed after execution")
    except Exception as error:
        input_revalidation_error = f"{type(error).__name__}: {error}"

    scrollfiesta = _scrollfiesta_gate(arms)
    physical_pass = bool(
        not physical_error and physical.get("acceptance", {}).get("pass")
    )
    mesh_visuals_pass = all(
        arms.get(arm, {}).get("status") == "PASS"
        for arm in ARM_ORDER
    )
    visuals_pass = visual_error is None and mesh_visuals_pass
    files, invalid_output_entries = _inventory_tree(
        output, excluded=("terminal_receipt.json",)
    )
    artifact_integrity_pass = not invalid_output_entries
    input_integrity_pass = input_revalidation_error is None
    terminal_pass = bool(
        physical_pass
        and scrollfiesta["pass"]
        and visuals_pass
        and artifact_integrity_pass
        and input_integrity_pass
    )
    receipt = {
        "schema_version": SCHEMA,
        "status": "PASS" if terminal_pass else "FAIL",
        "created_utc": _utc_now(),
        "downstream_lock_content_sha256": A.DOWNSTREAM_LOCK_CONTENT_SHA256,
        "metric_lock_content_sha256": metric_lock["content_sha256"],
        "grid_set_content_sha256": grid_set["content_sha256"],
        "promotion_content_sha256": grid_set["promotion_content_sha256"],
        "truth": truth_record,
        "truth_snapshot": truth_snapshot,
        "grid_snapshots": grid_snapshots,
        "binaries": binaries,
        "renderer": renderer_record,
        "runtime": runtime,
        "sanitized_environment_removed_keys": removed_environment,
        "provenance": provenance,
        "arms": arms,
        "physical": {
            "result": "physical_metrics.json" if physical else None,
            "error": physical_error,
            "pass": physical_pass,
        },
        "scrollfiesta_gate": scrollfiesta,
        "visual_evidence": {
            "cross_sections": visual_evidence,
            "mesh_fixed_camera_all_arms": mesh_visuals_pass,
            "error": visual_error,
            "pass": visuals_pass,
        },
        "input_integrity": {
            "private_snapshots_revalidated": input_integrity_pass,
            "error": input_revalidation_error,
            "pass": input_integrity_pass,
        },
        "artifact_integrity": {
            "invalid_output_entries": invalid_output_entries,
            "pass": artifact_integrity_pass,
        },
        "terminal_gate": {
            "physical_pass": physical_pass,
            "scrollfiesta_pass": scrollfiesta["pass"],
            "visuals_pass": visuals_pass,
            "input_integrity_pass": input_integrity_pass,
            "artifact_integrity_pass": artifact_integrity_pass,
            "pass": terminal_pass,
        },
        "claim_boundary": (
            "bounded untouched-PHerc0139 probability-to-ScrollFiesta improvement"
            if terminal_pass else
            "bounded negative result; no downstream-improvement claim is authorized"
        ),
        "files": files,
    }
    receipt["content_sha256"] = A.content_hash(receipt)
    _write_json_exclusive(output / "terminal_receipt.json", receipt)
    return verify_result(output, deep=terminal_pass)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--grid-set-receipt", type=Path, required=True)
    run.add_argument("--baseline-probability", type=Path, required=True)
    run.add_argument("--candidate-probability", type=Path, required=True)
    run.add_argument("--baseline-grid", type=Path, required=True)
    run.add_argument("--candidate-fixed-grid", type=Path, required=True)
    run.add_argument("--candidate-matched-grid", type=Path, required=True)
    run.add_argument("--labels-root", type=Path, required=True)
    run.add_argument("--semantic-audit", type=Path, required=True)
    run.add_argument("--binary-dir", type=Path, required=True)
    run.add_argument("--renderer-script", type=Path, required=True)
    run.add_argument("--metric-lock", type=Path)
    run.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument(
        "--expected-content-sha256",
        help="externally published terminal receipt digest",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "verify":
        result = verify_result(
            args.output,
            expected_content_sha256=args.expected_content_sha256,
        )
    else:
        result = execute(args)
    print(json.dumps({"status": result["status"],
                      "content_sha256": result["content_sha256"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
