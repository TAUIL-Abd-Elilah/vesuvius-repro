#!/usr/bin/env python3
"""Export and score two already-fitted sealed Spiral checkpoints.

This runner intentionally has no fitting command and never calls fit_spiral.
It only reconstructs the plain source TIFXYZ with
``flatten_spiral_checkpoint._export_source_surface``, runs SpiralCheck's
leakage-aware held-out scorer, then applies the frozen comparator from
``SEALED_PATCH_PROTOCOL.md``.

All input paths are explicit.  The output root must not exist: a rerun cannot
overwrite, mix, or silently reuse exports/reports from an earlier attempt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Z_BEGIN = 10500
Z_END = 11500
Z_RANGE = f"{Z_BEGIN},{Z_END}"
EXPECTED_STEPS = 5000
EXPECTED_SEEDS = (17, 23, 101)
EXPECTED_SPLIT_SEED = 20260827
EXPECTED_PATCH_COUNT = 89237
EXPECTED_HELDOUT_FRACTION = 0.20
EXPECTED_VILLA_COMMIT = "17dad916c79266f6a19f76abc507bb8b95c63a9b"
EXPECTED_SPIRALCHECK_COMMIT = "d1b50e2957409a870225fb9f5dcc5e25f7a0f9da"
EXPECTED_SPIRALCHECK_VERSION = "0.4.0"
TAU = "6"
UNSEEN_MIN_DIST = "2"
GEOMETRY_FILES = ("x.tif", "y.tif", "z.tif", "mask.tif", "winding.tif")
PROTOCOL_FILENAME = "SEALED_PATCH_PROTOCOL.md"
COMPARATOR_FILENAME = "compare_sealed_patch_reports.py"


class RunnerError(ValueError):
    """A precondition failed before an evaluation result can be trusted."""


def _file(path_arg: str, label: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if not path.is_file():
        raise RunnerError(f"{label} must be an existing file: {path}")
    return path


def _directory(path_arg: str, label: str) -> Path:
    path = Path(path_arg).expanduser().resolve()
    if not path.is_dir():
        raise RunnerError(f"{label} must be an existing directory: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_geometry_sha256(patch: Path) -> str:
    """Match SpiralCheck's manifest geometry hash, but stream large files."""
    digest = hashlib.sha256()
    for name in GEOMETRY_FILES:
        path = patch / name
        if path.exists():
            digest.update(name.encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str], *, env: dict[str, str], cwd: Path, log: Path) -> float:
    """Run one non-interactive subprocess, preserving exact stdout/stderr."""
    started = time.monotonic()
    with log.open("w", encoding="utf-8", newline="") as stream:
        stream.write("command: " + subprocess.list2cmdline(command) + "\n")
        stream.write("cwd: " + str(cwd) + "\n\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = time.monotonic() - started
    if completed.returncode != 0:
        raise RunnerError(
            f"command failed with exit code {completed.returncode}; inspect {log}"
        )
    return seconds


def _load_export_dependencies(spiral_fitting: Path):
    # The runner is outside the villa worktree by design.  Import the exact
    # source path supplied on the command line so a checkpoint is reconstructed
    # with the requested code, not an accidental installed copy.
    source = str(spiral_fitting)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        import torch
        from checkpoint_io import load_checkpoint_cpu
        from flatten_spiral_checkpoint import _checkpoint_config, _export_source_surface
    except ImportError as exc:
        raise RunnerError(
            "could not import Spiral export dependencies from --spiral-fitting. "
            "Run this script with the Spiral fitting Python environment."
        ) from exc
    return torch, load_checkpoint_cpu, _checkpoint_config, _export_source_surface


def _validate_checkpoint(
    checkpoint_path: Path, load_checkpoint_cpu, arm: str, optimizer_seed: int
) -> tuple[dict, dict]:
    checkpoint = load_checkpoint_cpu(str(checkpoint_path))
    if not isinstance(checkpoint, dict):
        raise RunnerError(f"{checkpoint_path}: checkpoint root is not a dictionary")
    try:
        actual_begin, actual_end = int(checkpoint["z_begin"]), int(checkpoint["z_end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError(f"{checkpoint_path}: missing valid z_begin/z_end") from exc
    if (actual_begin, actual_end) != (Z_BEGIN, Z_END):
        raise RunnerError(
            f"{checkpoint_path}: checkpoint z range is [{actual_begin}, {actual_end}), "
            f"but sealed protocol requires [{Z_BEGIN}, {Z_END})"
        )
    if checkpoint.get("completed_iterations") != EXPECTED_STEPS:
        raise RunnerError(
            f"{checkpoint_path}: completed_iterations={checkpoint.get('completed_iterations')!r}; "
            f"sealed protocol requires {EXPECTED_STEPS}"
        )

    config = checkpoint.get("cfg")
    if not isinstance(config, dict):
        raise RunnerError(f"{checkpoint_path}: checkpoint has no Spiral 'cfg' dictionary")
    shared_expected = {
        "z_begin": Z_BEGIN,
        "z_end": Z_END,
        "optimizer_random_seed": optimizer_seed,
        "optimizer_num_training_steps": EXPECTED_STEPS,
        "input_use_verified_patches": True,
        "input_use_unverified_patches": False,
        "input_use_tracks": False,
        "input_use_fibers": False,
        "input_use_pcl_absolute": False,
        "input_use_pcl_relative": False,
        "input_use_pcl_same_winding": False,
        "input_use_pcl_drawn_control_points": False,
        "input_use_normals": False,
        "input_use_surf_sdt": False,
        "input_use_gradient_magnitude": False,
        "input_use_winding_inference": False,
        "input_use_outer_shell": False,
        "patch_sampling_area_exponent": 0.5,
    }
    mismatches = {
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in shared_expected.items()
        if config.get(name) != expected
    }
    if arm == "baseline":
        arm_expected = {
            "patch_uuid_sampling_cap_regex": None,
            "patch_uuid_sampling_cap_fraction": 1.0,
        }
    elif arm == "treatment":
        arm_expected = {
            "patch_uuid_sampling_cap_regex": "^band-seed",
            "patch_uuid_sampling_cap_fraction": 0.75,
        }
    else:  # pragma: no cover - internal programming error
        raise RunnerError(f"unknown sealed arm: {arm}")
    mismatches.update({
        name: {"expected": expected, "actual": config.get(name)}
        for name, expected in arm_expected.items()
        if config.get(name) != expected
    })
    if mismatches:
        raise RunnerError(
            f"{checkpoint_path}: {arm} checkpoint does not match the sealed protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return checkpoint, config


def _patch_names(directory: Path) -> set[str]:
    return {
        item.name
        for item in directory.iterdir()
        if item.is_dir() and (item / "meta.json").is_file()
    }


def _validate_sealed_split(manifest_path: Path, heldout: Path, fit_inputs: Path) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read sealed split manifest: {manifest_path}") from exc
    expected_meta = {
        "seed": EXPECTED_SPLIT_SEED,
        "heldout_frac": EXPECTED_HELDOUT_FRACTION,
        "n_patches": EXPECTED_PATCH_COUNT,
        "grouping": "family",
    }
    mismatches = {
        name: {"expected": expected, "actual": manifest.get(name)}
        for name, expected in expected_meta.items()
        if manifest.get(name) != expected
    }
    if mismatches:
        raise RunnerError(
            "split manifest does not match the sealed protocol: "
            + json.dumps(mismatches, sort_keys=True)
        )
    assignments = manifest.get("assignments")
    geometry_hashes = manifest.get("geometry_sha256")
    if not isinstance(assignments, dict) or len(assignments) != EXPECTED_PATCH_COUNT:
        raise RunnerError("split manifest assignments are missing or incomplete")
    if not isinstance(geometry_hashes, dict) or set(geometry_hashes) != set(assignments):
        raise RunnerError("split manifest geometry hashes do not exactly cover assignments")
    unexpected_sides = sorted(set(assignments.values()) - {"fit", "heldout"})
    if unexpected_sides:
        raise RunnerError(f"split manifest has invalid assignment sides: {unexpected_sides}")
    expected_fit = {name for name, side in assignments.items() if side == "fit"}
    expected_heldout = {name for name, side in assignments.items() if side == "heldout"}
    if manifest.get("n_heldout") != len(expected_heldout):
        raise RunnerError("split manifest n_heldout disagrees with its assignments")
    actual_fit = _patch_names(fit_inputs)
    actual_heldout = _patch_names(heldout)
    if actual_fit != expected_fit:
        raise RunnerError(
            f"fit directory differs from manifest: "
            f"missing={len(expected_fit - actual_fit)}, extra={len(actual_fit - expected_fit)}"
        )
    if actual_heldout != expected_heldout:
        raise RunnerError(
            f"heldout directory differs from manifest: "
            f"missing={len(expected_heldout - actual_heldout)}, "
            f"extra={len(actual_heldout - expected_heldout)}"
        )
    heldout_hashes = {geometry_hashes[name] for name in expected_heldout}
    fit_hashes = {geometry_hashes[name] for name in expected_fit}
    overlap = heldout_hashes & fit_hashes
    if overlap:
        raise RunnerError(
            f"split manifest leaks {len(overlap)} geometry hash(es) across fit and heldout"
        )
    checked = 0
    for side, directory, names in (
        ("fit", fit_inputs, sorted(expected_fit)),
        ("heldout", heldout, sorted(expected_heldout)),
    ):
        for name in names:
            actual_hash = _patch_geometry_sha256(directory / name)
            if actual_hash != geometry_hashes[name]:
                raise RunnerError(
                    f"{side} patch {name!r} geometry differs from the sealed manifest: "
                    f"actual={actual_hash}, expected={geometry_hashes[name]}"
                )
            checked += 1
            if checked % 5000 == 0:
                print(
                    f"sealed split geometry verification: {checked:,}/{EXPECTED_PATCH_COUNT:,}",
                    flush=True,
                )
    return manifest


def _git_text(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RunnerError(
            f"git {' '.join(args)} failed for {repository}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _validate_spiralcheck_checkout(repository: Path) -> str:
    commit = _git_text(repository, "rev-parse", "HEAD")
    if commit != EXPECTED_SPIRALCHECK_COMMIT:
        raise RunnerError(
            f"SpiralCheck checkout is {commit}; sealed protocol requires "
            f"{EXPECTED_SPIRALCHECK_COMMIT}"
        )
    dirty = _git_text(repository, "status", "--porcelain")
    if dirty:
        raise RunnerError("SpiralCheck checkout must be clean for sealed scoring")
    return commit


def _validate_villa_checkout(spiral_fitting: Path) -> str:
    repository = spiral_fitting.parent
    commit = _git_text(repository, "rev-parse", "HEAD")
    if commit != EXPECTED_VILLA_COMMIT:
        raise RunnerError(
            f"villa checkout is {commit}; sealed protocol requires {EXPECTED_VILLA_COMMIT}"
        )
    tracked_dirty = _git_text(repository, "status", "--porcelain", "--untracked-files=no")
    if tracked_dirty:
        raise RunnerError("villa checkout has modified tracked files; sealed export refused")
    return commit


def _validate_score_report(path: Path, expected_heldout: int) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read SpiralCheck report: {path}") from exc
    meta = report.get("meta")
    if not isinstance(meta, dict):
        raise RunnerError(f"{path}: report has no meta object")
    if meta.get("patch_load_errors"):
        raise RunnerError(
            f"{path}: held-out patch load errors make the sealed subset incomplete: "
            f"{len(meta['patch_load_errors'])}"
        )
    if meta.get("fit_inputs_load_errors"):
        raise RunnerError(f"{path}: fit-input load errors weaken the leakage audit")
    expected_meta = {
        "spiralcheck": EXPECTED_SPIRALCHECK_VERSION,
        "variant": "plain",
        "tau": float(TAU),
        "z_range": [float(Z_BEGIN), float(Z_END)],
        "unseen_min_dist": float(UNSEEN_MIN_DIST),
        "fit_inputs_hash_audit": "clean",
        "patches_dir_listed_in_manifest": expected_heldout,
        "manifest_n_heldout": expected_heldout,
    }
    mismatches = {
        name: {"expected": expected, "actual": meta.get(name)}
        for name, expected in expected_meta.items()
        if meta.get(name) != expected
    }
    if mismatches:
        raise RunnerError(
            f"{path}: report metadata violate sealed requirements: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if meta.get("patches_dir_unlisted"):
        raise RunnerError(f"{path}: report includes patches outside the sealed held-out side")
    if not isinstance(report.get("intrinsic"), dict):
        raise RunnerError(f"{path}: report omitted required intrinsic checks")
    if not isinstance(report.get("heldout_aggregate"), dict):
        raise RunnerError(f"{path}: report omitted held-out aggregate")
    if not isinstance(report.get("heldout_patches"), list):
        raise RunnerError(f"{path}: report omitted per-patch held-out results")
    return report


def _export_plain_surface(
    *,
    arm: str,
    checkpoint_path: Path,
    checkpoint: dict,
    config: dict,
    umbilicus: Path,
    export_root: Path,
    torch,
    export_function,
    device: str,
    chunk_size: int,
) -> tuple[Path, float]:
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RunnerError("--device is CUDA but CUDA is unavailable; pass --device cpu intentionally")
    destination = export_root / arm / "plain_source"
    if destination.exists():
        raise RunnerError(f"refusing to reuse export destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    surface = export_function(
        checkpoint,
        config,
        umbilicus,
        destination,
        device=device_obj,
        voxel_size_um=9.6,
        chunk_size=chunk_size,
    )
    surface = Path(surface).resolve()
    if not surface.is_dir() or not (surface / "meta.json").is_file():
        raise RunnerError(f"{arm}: _export_source_surface did not publish a valid TIFXYZ: {surface}")
    metadata = json.loads((surface / "meta.json").read_text(encoding="utf-8"))
    if metadata.get("winding_column_ranges") is None:
        raise RunnerError(f"{arm}: exported surface is not a combined plain Spiral TIFXYZ")
    elapsed = time.monotonic() - started
    print(f"{arm}: exported plain source surface in {elapsed:.1f}s: {surface}", flush=True)
    return surface, elapsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", required=True, help="existing baseline checkpoint_fitted.ckpt")
    parser.add_argument("--treatment-checkpoint", required=True, help="existing cap075 checkpoint_fitted.ckpt")
    parser.add_argument("--spiral-fitting", required=True, help="directory containing flatten_spiral_checkpoint.py")
    parser.add_argument("--spiralcheck-source", required=True, help="SpiralCheck checkout containing src/spiralcheck")
    parser.add_argument("--spiralcheck-python", required=True, help="Python executable with SpiralCheck dependencies")
    parser.add_argument("--manifest", required=True, help="sealed split_manifest.json")
    parser.add_argument("--heldout-patches", required=True, help="sealed held-out patch directory")
    parser.add_argument("--fit-inputs", required=True, help="actual fit-side verified patch directory")
    parser.add_argument("--umbilicus", required=True, help="PHercParis4 umbilicus.json")
    parser.add_argument("--output-root", required=True, help="new evaluation directory; must not already exist")
    parser.add_argument(
        "--optimizer-seed", type=int, choices=EXPECTED_SEEDS, default=17,
        help="frozen optimizer seed represented by both checkpoints",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for export only (default: cuda)")
    parser.add_argument("--chunk-size", type=int, default=65536, help="points per export transform batch")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.chunk_size <= 0:
        raise RunnerError("--chunk-size must be positive")
    evidence_dir = Path(__file__).resolve().parent
    protocol = _file(str(evidence_dir / PROTOCOL_FILENAME), "sealed protocol")
    comparator = _file(str(evidence_dir / COMPARATOR_FILENAME), "sealed comparator")
    spiral_fitting = _directory(args.spiral_fitting, "--spiral-fitting")
    if not (spiral_fitting / "flatten_spiral_checkpoint.py").is_file():
        raise RunnerError(f"--spiral-fitting has no flatten_spiral_checkpoint.py: {spiral_fitting}")
    villa_commit = _validate_villa_checkout(spiral_fitting)
    spiralcheck_source = _directory(args.spiralcheck_source, "--spiralcheck-source")
    if not (spiralcheck_source / "src" / "spiralcheck" / "cli.py").is_file():
        raise RunnerError(f"--spiralcheck-source has no src/spiralcheck/cli.py: {spiralcheck_source}")
    spiralcheck_commit = _validate_spiralcheck_checkout(spiralcheck_source)
    scoring_python = _file(args.spiralcheck_python, "--spiralcheck-python")
    baseline_checkpoint = _file(args.baseline_checkpoint, "--baseline-checkpoint")
    treatment_checkpoint = _file(args.treatment_checkpoint, "--treatment-checkpoint")
    manifest = _file(args.manifest, "--manifest")
    heldout = _directory(args.heldout_patches, "--heldout-patches")
    fit_inputs = _directory(args.fit_inputs, "--fit-inputs")
    umbilicus = _file(args.umbilicus, "--umbilicus")
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        raise RunnerError(f"refusing to reuse existing --output-root: {output_root}")
    if output_root.parent == output_root:
        raise RunnerError("--output-root cannot be a filesystem root")

    # Re-hash the physical split before loading large checkpoints or creating
    # output. This is an evaluation-only runner; it never retrains.
    split_document = _validate_sealed_split(manifest, heldout, fit_inputs)
    torch, load_checkpoint_cpu, checkpoint_config, export_function = _load_export_dependencies(spiral_fitting)
    # Validate both checkpoints completely before creating outputs or consuming GPU time.
    baseline, baseline_raw_config = _validate_checkpoint(
        baseline_checkpoint, load_checkpoint_cpu, "baseline", args.optimizer_seed
    )
    treatment, treatment_raw_config = _validate_checkpoint(
        treatment_checkpoint, load_checkpoint_cpu, "treatment", args.optimizer_seed
    )
    baseline_config = checkpoint_config(baseline)
    treatment_config = checkpoint_config(treatment)

    output_root.mkdir(parents=True, exist_ok=False)
    failure_path = output_root / "FAILED.json"
    run_manifest: dict[str, Any] = {
        "schema": "sealed-spiralcheck-run-v1",
        "status": "running",
        "started_at_utc": _utc(),
        "protocol": {"path": str(protocol), "sha256": _sha256(protocol)},
        "paths": {
            "baseline_checkpoint": str(baseline_checkpoint),
            "treatment_checkpoint": str(treatment_checkpoint),
            "spiral_fitting": str(spiral_fitting),
            "villa_commit": villa_commit,
            "spiralcheck_source": str(spiralcheck_source),
            "spiralcheck_commit": spiralcheck_commit,
            "manifest": str(manifest),
            "heldout_patches": str(heldout),
            "fit_inputs": str(fit_inputs),
            "umbilicus": str(umbilicus),
        },
        "file_sha256": {
            "baseline_checkpoint": _sha256(baseline_checkpoint),
            "treatment_checkpoint": _sha256(treatment_checkpoint),
            "manifest": _sha256(manifest),
            "umbilicus": _sha256(umbilicus),
            "comparator": _sha256(comparator),
            "flatten_spiral_checkpoint": _sha256(spiral_fitting / "flatten_spiral_checkpoint.py"),
        },
        "frozen_parameters": {
            "z_range": Z_RANGE, "steps": EXPECTED_STEPS,
            "optimizer_seed": args.optimizer_seed,
            "split_seed": EXPECTED_SPLIT_SEED, "tau": float(TAU),
            "unseen_min_dist": float(UNSEEN_MIN_DIST),
            "variant": "plain", "export": "flatten_spiral_checkpoint._export_source_surface",
        },
        "checkpoint_configs": {
            "baseline": baseline_raw_config,
            "treatment": treatment_raw_config,
        },
    }
    _write_json(output_root / "run_manifest.json", run_manifest)
    try:
        exports = output_root / "exports"
        baseline_surface, baseline_export_seconds = _export_plain_surface(
            arm="baseline", checkpoint_path=baseline_checkpoint, checkpoint=baseline,
            config=baseline_config, umbilicus=umbilicus, export_root=exports, torch=torch,
            export_function=export_function, device=args.device, chunk_size=args.chunk_size,
        )
        del baseline, baseline_config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        treatment_surface, treatment_export_seconds = _export_plain_surface(
            arm="treatment", checkpoint_path=treatment_checkpoint, checkpoint=treatment,
            config=treatment_config, umbilicus=umbilicus, export_root=exports, torch=torch,
            export_function=export_function, device=args.device, chunk_size=args.chunk_size,
        )
        del treatment, treatment_config
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # SpiralCheck accepts the parent of a single combined TIFXYZ. `plain`
        # is explicit even though the source export has no spliced patches.
        runner_env = os.environ.copy()
        source_path = str(spiralcheck_source / "src")
        runner_env["PYTHONPATH"] = source_path + os.pathsep + runner_env.get("PYTHONPATH", "")
        reports = output_root / "reports"
        reports.mkdir(exist_ok=False)
        score_seconds: dict[str, float] = {}
        for arm, surface in (("baseline", baseline_surface), ("treatment", treatment_surface)):
            report_dir = reports / arm
            command = [
                str(scoring_python), "-m", "spiralcheck.cli", "score",
                "--meshes", str(surface.parent),
                "--patches", str(heldout),
                "--manifest", str(manifest),
                "--fit-inputs", str(fit_inputs),
                "--umbilicus", str(umbilicus),
                "--z-range", Z_RANGE,
                "--variant", "plain",
                "--tau", TAU,
                "--unseen-min-dist", UNSEEN_MIN_DIST,
                "--out", str(report_dir),
            ]
            score_seconds[arm] = _run(
                command, env=runner_env, cwd=spiralcheck_source,
                log=output_root / f"spiralcheck-{arm}.log",
            )
            _validate_score_report(
                report_dir / "report.json", int(split_document["n_heldout"])
            )
        comparison_path = output_root / "sealed_comparison.json"
        comparison_seconds = _run(
            [str(scoring_python), str(comparator), str(reports / "baseline" / "report.json"),
             str(reports / "treatment" / "report.json"), "--output", str(comparison_path)],
            env=runner_env, cwd=evidence_dir, log=output_root / "sealed-comparator.log",
        )
        run_manifest.update({
            "status": "complete",
            "finished_at_utc": _utc(),
            "exports": {"baseline": str(baseline_surface), "treatment": str(treatment_surface)},
            "reports": {"baseline": str(reports / "baseline" / "report.json"), "treatment": str(reports / "treatment" / "report.json")},
            "comparison": str(comparison_path),
            "runtime_seconds": {
                "baseline_export": baseline_export_seconds,
                "treatment_export": treatment_export_seconds,
                **score_seconds,
                "comparison": comparison_seconds,
            },
            "output_sha256": {
                "baseline_report": _sha256(reports / "baseline" / "report.json"),
                "treatment_report": _sha256(reports / "treatment" / "report.json"),
                "comparison": _sha256(comparison_path),
            },
        })
        _write_json(output_root / "run_manifest.json", run_manifest)
        print(f"sealed comparison complete: {comparison_path}", flush=True)
        return 0
    except BaseException as exc:
        failure = {
            "schema": "sealed-spiralcheck-run-failure-v1",
            "failed_at_utc": _utc(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(failure_path, failure)
        run_manifest.update({"status": "failed", "finished_at_utc": _utc(), "failure": str(failure_path)})
        _write_json(output_root / "run_manifest.json", run_manifest)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as exc:
        raise SystemExit(f"sealed runner refused: {exc}") from exc
