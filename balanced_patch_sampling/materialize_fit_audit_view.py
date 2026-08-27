#!/usr/bin/env python3
"""Replay and materialize the exact fit inputs for sealed leakage auditing.

The view contains hard links, not copied geometry. The complete fit partition
remains separate for split validation; this exact view is supplied only to
SpiralCheck's ``--fit-inputs`` audit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_VILLA_COMMIT = "17dad916c79266f6a19f76abc507bb8b95c63a9b"
EXPECTED_PARTITION_COUNT = 71421
EXPECTED_COUNT = 8542
EXPECTED_DROPS = {
    "z ROI prefilter": 62697,
    "erosion": 0,
    "z ROI after erosion": 182,
}
EXPECTED_ID_SHA256 = "48ab9630e757cbf6483da0fc9fff8eb8b0410099a93a56fb75d7784adacc1a10"
Z_BEGIN = 10500
Z_END = 11500
EROSION_CELLS = 1
CONTENT_FILES = ("meta.json", "x.tif", "y.tif", "z.tif", "mask.tif", "winding.tif")
GEOMETRY_FILES = CONTENT_FILES[1:]


class ViewError(ValueError):
    """The requested audit view would not reproduce the sealed fit inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_hashes(patch: Path) -> tuple[str, str]:
    content_digest = hashlib.sha256()
    geometry_digest = hashlib.sha256()
    for name in CONTENT_FILES:
        path = patch / name
        if not path.exists():
            continue
        encoded_name = name.encode()
        content_digest.update(encoded_name)
        if name in GEOMETRY_FILES:
            geometry_digest.update(encoded_name)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                content_digest.update(block)
                if name in GEOMETRY_FILES:
                    geometry_digest.update(block)
    return content_digest.hexdigest(), geometry_digest.hexdigest()


def _id_sha256(ids: set[str] | list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def _read_final_ids(path: Path) -> set[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        rows = document["patches"]
        ids = [row["id"] for row in rows]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ViewError(f"cannot read patch IDs from {path}: {exc}") from exc
    if not all(isinstance(name, str) and name for name in ids):
        raise ViewError(f"{path}: empty or non-string patch ID")
    if len(set(ids)) != len(ids):
        raise ViewError(f"{path}: duplicate patch IDs")
    for name in ids:
        if Path(name).name != name or name in {".", ".."}:
            raise ViewError(f"{path}: unsafe patch ID {name!r}")
    return set(ids)


def _read_rejected_ids(path: Path) -> set[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ViewError(f"cannot read rejected-patch list {path}: {exc}") from exc
    ids = {
        line.strip().replace("\\", "/").rsplit("/", 1)[-1]
        for line in lines
        if line.strip()
    }
    if any(not name or name in {".", ".."} for name in ids):
        raise ViewError(f"{path}: invalid rejected patch path")
    return ids


def _consumed_ids(final_artifact: Path, rejected_artifact: Path) -> set[str]:
    return _read_final_ids(final_artifact) | _read_rejected_ids(rejected_artifact)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ViewError(f"cannot read split manifest {path}: {exc}") from exc
    assignments = document.get("assignments")
    content = document.get("content_sha256")
    geometry = document.get("geometry_sha256")
    if not all(isinstance(item, dict) for item in (assignments, content, geometry)):
        raise ViewError(
            "split manifest lacks assignments, content_sha256, or geometry_sha256"
        )
    if set(assignments) != set(content) or set(assignments) != set(geometry):
        raise ViewError("split manifest hash maps do not cover the assignments exactly")
    return document


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
        raise ViewError(
            f"git {' '.join(args)} failed for {repository}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def _validate_villa(spiral_fitting: Path) -> str:
    if not (spiral_fitting / "spiral_helpers.py").is_file():
        raise ViewError(f"not a spiral-fitting source directory: {spiral_fitting}")
    repository = spiral_fitting.parent
    commit = _git_text(repository, "rev-parse", "HEAD")
    if commit != EXPECTED_VILLA_COMMIT:
        raise ViewError(
            f"villa checkout is {commit}; expected {EXPECTED_VILLA_COMMIT}"
        )
    if _git_text(repository, "status", "--porcelain", "--untracked-files=no"):
        raise ViewError("villa checkout has modified tracked files")
    return commit


def _replay_chunk(job: tuple[str, str, list[str], int]) -> list[tuple[str, bool, str | None, str | None]]:
    """Run the pinned loader in a spawned worker, returning no array payloads."""
    spiral_fitting, source_fit, entries, io_threads = job
    if spiral_fitting not in sys.path:
        sys.path.insert(0, spiral_fitting)
    from spiral_helpers import load_patch_payload_chunk

    rows = load_patch_payload_chunk(
        source_fit, entries, Z_BEGIN, Z_END, EROSION_CELLS, io_threads
    )
    return [
        (entry, payload is not None, error, reason)
        for entry, payload, error, reason in rows
    ]


def replay_loader(
    source_fit: Path,
    spiral_fitting: Path,
    *,
    workers: int,
    io_threads: int,
) -> dict[str, Any]:
    villa_commit = _validate_villa(spiral_fitting)
    entries = sorted(os.listdir(source_fit))
    if len(entries) != EXPECTED_PARTITION_COUNT:
        raise ViewError(
            f"fit partition has {len(entries)} entries; expected {EXPECTED_PARTITION_COUNT}"
        )
    chunks = [entries[start : start + 256] for start in range(0, len(entries), 256)]
    jobs = [
        (str(spiral_fitting), str(source_fit), chunk, io_threads)
        for chunk in chunks
    ]
    retained: set[str] = set()
    drops = {name: 0 for name in EXPECTED_DROPS}
    errors: dict[str, str] = {}
    completed = 0
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(chunks))), mp_context=context
    ) as executor:
        for rows in executor.map(_replay_chunk, jobs, chunksize=1):
            for name, loaded, error, reason in rows:
                if loaded:
                    retained.add(name)
                elif error is not None:
                    errors[name] = error
                elif reason in drops:
                    drops[reason] += 1
                else:
                    errors[name] = f"unrecognized drop reason: {reason!r}"
                completed += 1
            if completed % 4096 < 256:
                print(
                    f"replayed {completed:,}/{len(entries):,} fit entries; "
                    f"retained {len(retained):,}",
                    flush=True,
                )
    if completed != len(entries):
        raise ViewError(f"loader replay returned {completed} rows; expected {len(entries)}")
    if errors:
        first = next(iter(sorted(errors.items())))
        raise ViewError(
            f"pinned loader replay had {len(errors)} error(s); first={first!r}"
        )
    if drops != EXPECTED_DROPS or len(retained) != EXPECTED_COUNT:
        raise ViewError(
            "pinned loader replay counts differ from the pre-result amendment: "
            f"retained={len(retained)}, drops={drops}"
        )
    return {
        "villa_commit": villa_commit,
        "partition_count": len(entries),
        "retained_count": len(retained),
        "retained_id_sha256": _id_sha256(retained),
        "drops": drops,
        "load_errors": 0,
        "z_range": [Z_BEGIN, Z_END],
        "erosion_cells": EROSION_CELLS,
        "patch_uuid_filter_regex": None,
        "retained_ids": retained,
    }


def materialize(
    *,
    baseline_fit_artifact: Path,
    baseline_rejected_artifact: Path,
    treatment_fit_artifact: Path,
    treatment_rejected_artifact: Path,
    source_fit: Path,
    split_manifest: Path,
    spiral_fitting: Path,
    output: Path,
    workers: int = 12,
    io_threads: int = 4,
    expected_count: int = EXPECTED_COUNT,
    expected_id_sha256: str = EXPECTED_ID_SHA256,
    replay_document: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ViewError(f"refusing to reuse output: {output}")
    if output.parent == output:
        raise ViewError("output cannot be a filesystem root")
    required_files = {
        "baseline fit artifact": baseline_fit_artifact,
        "baseline rejected artifact": baseline_rejected_artifact,
        "treatment fit artifact": treatment_fit_artifact,
        "treatment rejected artifact": treatment_rejected_artifact,
        "split manifest": split_manifest,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise ViewError(f"{label} is not a file: {path}")
    if not source_fit.is_dir():
        raise ViewError(f"source fit directory does not exist: {source_fit}")
    if workers <= 0 or io_threads <= 0:
        raise ViewError("workers and io_threads must be positive")

    baseline_ids = _consumed_ids(baseline_fit_artifact, baseline_rejected_artifact)
    treatment_ids = _consumed_ids(treatment_fit_artifact, treatment_rejected_artifact)
    if baseline_ids != treatment_ids:
        raise ViewError(
            "baseline and treatment consumed-ID unions differ: "
            f"baseline_only={len(baseline_ids - treatment_ids)}, "
            f"treatment_only={len(treatment_ids - baseline_ids)}"
        )
    ids = baseline_ids
    actual_id_sha256 = _id_sha256(ids)
    if len(ids) != expected_count or actual_id_sha256 != expected_id_sha256:
        raise ViewError(
            "consumed input IDs do not match the sealed set: "
            f"count={len(ids)} (expected {expected_count}), "
            f"sha256={actual_id_sha256} (expected {expected_id_sha256})"
        )

    manifest = _read_manifest(split_manifest)
    assignments = manifest["assignments"]
    content_hashes = manifest["content_sha256"]
    geometry_hashes = manifest["geometry_sha256"]
    wrong_side = sorted(name for name in ids if assignments.get(name) != "fit")
    if wrong_side:
        raise ViewError(
            f"{len(wrong_side)} consumed ID(s) are not assigned to fit; "
            f"first={wrong_side[0]!r}"
        )

    replay = replay_document or replay_loader(
        source_fit, spiral_fitting, workers=workers, io_threads=io_threads
    )
    replay_ids = replay.get("retained_ids")
    if not isinstance(replay_ids, set) or replay_ids != ids:
        replay_ids = replay_ids if isinstance(replay_ids, set) else set()
        raise ViewError(
            "pinned loader replay differs from consumed artifacts: "
            f"artifact_only={len(ids - replay_ids)}, replay_only={len(replay_ids - ids)}"
        )
    replay_public = {key: value for key, value in replay.items() if key != "retained_ids"}

    # Validate every source before creating output, so a predictable failure
    # cannot leave a partial view.
    files_by_id: dict[str, list[Path]] = {}
    for index, name in enumerate(sorted(ids), 1):
        patch = source_fit / name
        if not patch.is_dir() or not (patch / "meta.json").is_file():
            raise ViewError(f"missing source patch directory or meta.json: {patch}")
        if any(item.is_dir() for item in patch.iterdir()):
            raise ViewError(f"source patch contains nested directories: {patch}")
        files = sorted(item for item in patch.iterdir() if item.is_file())
        if not files:
            raise ViewError(f"source patch contains no files: {patch}")
        actual_content, actual_geometry = _patch_hashes(patch)
        if actual_content != content_hashes.get(name):
            raise ViewError(
                f"source content differs from sealed manifest for {name!r}: "
                f"actual={actual_content}, expected={content_hashes.get(name)}"
            )
        if actual_geometry != geometry_hashes.get(name):
            raise ViewError(
                f"source geometry differs from sealed manifest for {name!r}: "
                f"actual={actual_geometry}, expected={geometry_hashes.get(name)}"
            )
        files_by_id[name] = files
        if index % 1000 == 0:
            print(f"validated {index:,}/{len(ids):,} source patches", flush=True)

    output.mkdir(parents=True, exist_ok=False)
    linked_files = 0
    for index, name in enumerate(sorted(ids), 1):
        destination = output / name
        destination.mkdir()
        for source in files_by_id[name]:
            target = destination / source.name
            try:
                os.link(source, target)
            except OSError as exc:
                raise ViewError(
                    f"cannot hard-link {source} to {target}; source and output "
                    "must be on the same filesystem"
                ) from exc
            if not os.path.samefile(source, target):
                raise ViewError(f"hard-link identity check failed: {target}")
            linked_files += 1
        if index % 1000 == 0:
            print(f"linked {index:,}/{len(ids):,} patch directories", flush=True)

    artifacts = {
        "baseline_fit": baseline_fit_artifact,
        "baseline_rejected": baseline_rejected_artifact,
        "treatment_fit": treatment_fit_artifact,
        "treatment_rejected": treatment_rejected_artifact,
    }
    view_manifest = {
        "schema": "sealed-fit-audit-view-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patch_count": len(ids),
        "patch_id_sha256": actual_id_sha256,
        "band_seed_count": sum(name.startswith("band-seed") for name in ids),
        "legacy_count": sum(not name.startswith("band-seed") for name in ids),
        "linked_file_count": linked_files,
        "consumed_artifacts": {
            label: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for label, path in artifacts.items()
        },
        "source_fit": str(source_fit.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": _sha256(split_manifest),
        "loader_replay": replay_public,
        "materialization": "hardlink",
    }
    manifest_path = output / ".fit_audit_view.json"
    manifest_path.write_text(
        json.dumps(view_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"complete: {manifest_path}", flush=True)
    return view_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-fit-artifact", required=True)
    parser.add_argument("--baseline-rejected-artifact", required=True)
    parser.add_argument("--treatment-fit-artifact", required=True)
    parser.add_argument("--treatment-rejected-artifact", required=True)
    parser.add_argument("--source-fit", required=True, help="complete sealed fit partition")
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--spiral-fitting", required=True, help="pinned villa spiral-fitting source")
    parser.add_argument("--output", required=True, help="new hard-link view; must not exist")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--expected-id-sha256", default=EXPECTED_ID_SHA256)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    resolve = lambda value: Path(value).expanduser().resolve()
    materialize(
        baseline_fit_artifact=resolve(args.baseline_fit_artifact),
        baseline_rejected_artifact=resolve(args.baseline_rejected_artifact),
        treatment_fit_artifact=resolve(args.treatment_fit_artifact),
        treatment_rejected_artifact=resolve(args.treatment_rejected_artifact),
        source_fit=resolve(args.source_fit),
        split_manifest=resolve(args.split_manifest),
        spiral_fitting=resolve(args.spiral_fitting),
        output=resolve(args.output),
        workers=args.workers,
        io_threads=args.io_threads,
        expected_count=args.expected_count,
        expected_id_sha256=args.expected_id_sha256,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ViewError as exc:
        raise SystemExit(f"fit audit view refused: {exc}") from exc
