#!/usr/bin/env python3
"""Freeze the model-independent point scope of the sealed held-out split.

This program does not load a fitted model or a score report.  It verifies the
exact PHercParis4 split and a clean, pinned SpiralCheck checkout, strict-loads
every held-out tifxyz, and records how many of its quad centers lie in the
frozen fit interval.  The output path must be new so a later run cannot
silently replace the pre-result scope declaration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


EXPECTED_SPIRALCHECK_COMMIT = "d1b50e2957409a870225fb9f5dcc5e25f7a0f9da"
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "9a1b226ebde3854728adfeb6f21513026c0cc49d948cc52684d6ee96e3819f31"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "e54534c45cc29bed43c92fa86fdd2a7a863e60bcc057f0dc4bdb6ea3d8282a64"
)
EXPECTED_SOURCE_INDEX_SHA256 = (
    "29f975b1f1615bf7b11e58ec88c77d17c3d7603c4f70e43175e1eb31d65159c2"
)
EXPECTED_SPLIT_SEED = 20260827
EXPECTED_HELDOUT_FRACTION = 0.20
EXPECTED_PATCH_COUNT = 89237
EXPECTED_FAMILY_COUNT = 85848
EXPECTED_FIT_COUNT = 71421
EXPECTED_HELDOUT_COUNT = 17816
EXPECTED_ALL_ID_SHA256 = (
    "7960d3f3071381f16b272d65ffe99dc9e13b3522e18fad5b4a18f66120f22dd7"
)
EXPECTED_FIT_ID_SHA256 = (
    "a6063f00262ddb50ede9ba90d1c61fe2d39d544d206eaa424cb20f19f00b0411"
)
EXPECTED_HELDOUT_ID_SHA256 = (
    "72ffc09a7413d97affa1145c2e48d8a7aa8e846928efea46a4c3b5e6acc3e3b6"
)
Z_BEGIN = 10500.0
Z_END_HALF_OPEN = 11500.0
# SpiralCheck d1b50e's score filter is inclusive at both ends.  This is the
# exact binary64 inclusive upper bound equivalent to [10500, 11500).
SCORE_Z_END = math.nextafter(Z_END_HALF_OPEN, -math.inf)

CONTENT_FILES = ("meta.json", "x.tif", "y.tif", "z.tif", "mask.tif", "winding.tif")
GEOMETRY_FILES = CONTENT_FILES[1:]


class ScopeError(ValueError):
    """An input cannot reproduce the frozen held-out point scope."""


@dataclass(frozen=True)
class SpiralCheckAPI:
    load_tifxyz: Callable[[Path], Any]
    quad_surface_type: type
    source_file: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_name_sha256(names: Iterable[str]) -> str:
    payload = "".join(f"{name}\n" for name in sorted(names)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_index_sha256(rows: Iterable[tuple[str, str]]) -> str:
    payload = "".join(
        f"{name}\t{value}\n" for name, value in sorted(rows)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_scope_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    """Canonical mapping: UTF-8 ``patch_id<TAB>n_points<LF>`` lines."""
    normalized: list[tuple[str, int]] = []
    seen: set[str] = set()
    for row in rows:
        try:
            patch_id = row["patch_id"]
            n_points = row["n_points"]
        except (KeyError, TypeError) as exc:
            raise ScopeError("scope row lacks patch_id or n_points") from exc
        _validate_patch_id(patch_id)
        if patch_id in seen:
            raise ScopeError(f"duplicate scope patch ID: {patch_id!r}")
        if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 0:
            raise ScopeError(f"invalid n_points for {patch_id!r}: {n_points!r}")
        seen.add(patch_id)
        normalized.append((patch_id, n_points))
    return "".join(
        f"{patch_id}\t{n_points}\n" for patch_id, n_points in sorted(normalized)
    ).encode("utf-8")


def canonical_scope_sha256(rows: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_scope_bytes(rows)).hexdigest()


def _validate_patch_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ScopeError(f"empty or non-string patch ID: {value!r}")
    if value in {".", ".."} or any(char in value for char in "\t\r\n/\\"):
        raise ScopeError(f"unsafe patch ID: {value!r}")
    return value


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
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ScopeError(f"git {' '.join(args)} failed in {repository}: {detail}")
    return completed.stdout.strip()


def validate_spiralcheck_checkout(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    source_file = repository / "src" / "spiralcheck" / "io_tifxyz.py"
    if not source_file.is_file():
        raise ScopeError(
            f"SpiralCheck checkout lacks src/spiralcheck/io_tifxyz.py: {repository}"
        )
    commit = _git_text(repository, "rev-parse", "HEAD")
    if commit != EXPECTED_SPIRALCHECK_COMMIT:
        raise ScopeError(
            f"SpiralCheck checkout is {commit}; expected {EXPECTED_SPIRALCHECK_COMMIT}"
        )
    dirty = _git_text(repository, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        first = dirty.splitlines()[0]
        raise ScopeError(f"SpiralCheck checkout is not clean; first change: {first}")
    return {
        "commit": commit,
        "source": "src/spiralcheck/io_tifxyz.py",
        "source_sha256": _sha256(source_file),
    }


def load_spiralcheck_api(repository: Path) -> SpiralCheckAPI:
    """Load the pinned checkout's file directly, never an installed package."""
    source_file = repository.resolve() / "src" / "spiralcheck" / "io_tifxyz.py"
    module_name = "_sealed_scope_spiralcheck_io_tifxyz"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ScopeError(f"cannot import SpiralCheck source: {source_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - pin/import failure is reported
        sys.modules.pop(module_name, None)
        raise ScopeError(f"cannot import SpiralCheck source {source_file}: {exc}") from exc
    try:
        return SpiralCheckAPI(module.load_tifxyz, module.QuadSurface, source_file)
    except AttributeError as exc:
        raise ScopeError("pinned SpiralCheck source lacks load_tifxyz or QuadSurface") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScopeError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScopeError(f"{label} root is not an object: {path}")
    return value


def validate_source_manifest(path: Path) -> dict[str, Any]:
    if path.name != "public_input_manifest.json" or not path.is_file():
        raise ScopeError(
            "--source-manifest must be the frozen public_input_manifest.json"
        )
    actual_sha = _sha256(path)
    if actual_sha != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ScopeError(
            f"source manifest SHA-256 is {actual_sha}; expected "
            f"{EXPECTED_SOURCE_MANIFEST_SHA256}"
        )
    document = _read_json(path, "source manifest")
    patches = document.get("patches")
    if not isinstance(patches, dict):
        raise ScopeError("source manifest lacks its patches object")
    expected = {
        "patches": EXPECTED_PATCH_COUNT,
        "index_sha256": EXPECTED_SOURCE_INDEX_SHA256,
    }
    mismatches = {
        key: {"expected": value, "actual": patches.get(key)}
        for key, value in expected.items()
        if patches.get(key) != value
    }
    if mismatches:
        raise ScopeError(
            "source manifest does not identify the frozen snapshot: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return {
        "name": path.name,
        "sha256": actual_sha,
        "patch_index_sha256": patches["index_sha256"],
    }


def validate_split_manifest(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    if path.name != "split_manifest.json" or not path.is_file():
        raise ScopeError("--split-manifest must name the frozen split_manifest.json")
    actual_sha = _sha256(path)
    if actual_sha != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ScopeError(
            f"split manifest SHA-256 is {actual_sha}; expected "
            f"{EXPECTED_SPLIT_MANIFEST_SHA256}"
        )
    document = _read_json(path, "split manifest")
    expected_fields = {
        "seed": EXPECTED_SPLIT_SEED,
        "heldout_frac": EXPECTED_HELDOUT_FRACTION,
        "grouping": "family",
        "n_patches": EXPECTED_PATCH_COUNT,
        "n_families": EXPECTED_FAMILY_COUNT,
        "n_heldout": EXPECTED_HELDOUT_COUNT,
    }
    mismatches = {
        key: {"expected": value, "actual": document.get(key)}
        for key, value in expected_fields.items()
        if document.get(key) != value
    }
    if mismatches:
        raise ScopeError(
            "split manifest fields differ from the frozen split: "
            + json.dumps(mismatches, sort_keys=True)
        )
    assignments = document.get("assignments")
    content_hashes = document.get("content_sha256")
    geometry_hashes = document.get("geometry_sha256")
    if not all(isinstance(value, dict) for value in (assignments, content_hashes, geometry_hashes)):
        raise ScopeError("split manifest lacks assignment/content/geometry mappings")
    names = sorted(assignments)
    for name in names:
        _validate_patch_id(name)
        if assignments[name] not in {"fit", "heldout"}:
            raise ScopeError(f"invalid assignment for {name!r}: {assignments[name]!r}")
    if set(content_hashes) != set(names) or set(geometry_hashes) != set(names):
        raise ScopeError("split hash mappings do not exactly match assignment names")
    fit_names = [name for name in names if assignments[name] == "fit"]
    heldout_names = [name for name in names if assignments[name] == "heldout"]
    observed = {
        "all_count": len(names),
        "fit_count": len(fit_names),
        "heldout_count": len(heldout_names),
        "all_id_sha256": _canonical_name_sha256(names),
        "fit_id_sha256": _canonical_name_sha256(fit_names),
        "heldout_id_sha256": _canonical_name_sha256(heldout_names),
    }
    expected_names = {
        "all_count": EXPECTED_PATCH_COUNT,
        "fit_count": EXPECTED_FIT_COUNT,
        "heldout_count": EXPECTED_HELDOUT_COUNT,
        "all_id_sha256": EXPECTED_ALL_ID_SHA256,
        "fit_id_sha256": EXPECTED_FIT_ID_SHA256,
        "heldout_id_sha256": EXPECTED_HELDOUT_ID_SHA256,
    }
    if observed != expected_names:
        raise ScopeError(
            "split assignment names differ from the frozen sets: "
            + json.dumps({"expected": expected_names, "actual": observed}, sort_keys=True)
        )
    return document, fit_names, heldout_names


def validate_heldout_directory(
    heldout: Path, split_manifest: Path, expected_names: Sequence[str]
) -> None:
    if not heldout.is_dir() or heldout.name != "heldout":
        raise ScopeError("--heldout-patches must be the sealed directory named heldout")
    if heldout.resolve().parent != split_manifest.resolve().parent:
        raise ScopeError("heldout and split_manifest.json must be siblings in one split")
    actual_names = sorted(child.name for child in heldout.iterdir() if child.is_dir())
    expected = sorted(expected_names)
    if actual_names != expected:
        actual_set, expected_set = set(actual_names), set(expected)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ScopeError(
            "heldout directory names differ from the frozen assignment set: "
            f"missing={len(missing)} first={missing[:3]}, "
            f"extra={len(extra)} first={extra[:3]}"
        )


def _patch_hashes(patch: Path) -> tuple[str, str]:
    content = hashlib.sha256()
    geometry = hashlib.sha256()
    for filename in CONTENT_FILES:
        source = patch / filename
        if not source.exists():
            continue
        encoded = filename.encode("utf-8")
        content.update(encoded)
        if filename in GEOMETRY_FILES:
            geometry.update(encoded)
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                content.update(block)
                if filename in GEOMETRY_FILES:
                    geometry.update(block)
    return content.hexdigest(), geometry.hexdigest()


def count_in_window_points(surface: Any) -> int:
    """Count QuadSurface centers using SpiralCheck's inclusive range rule."""
    try:
        centers, quad_indices = surface.quad_centers()
    except Exception as exc:  # noqa: BLE001 - converted to a strict load failure
        raise ScopeError(f"QuadSurface.quad_centers failed: {exc}") from exc
    if getattr(centers, "ndim", None) != 2 or centers.shape[1] != 3:
        raise ScopeError(f"quad_centers returned an invalid center shape: {centers.shape!r}")
    if getattr(quad_indices, "ndim", None) != 2 or quad_indices.shape[0] != centers.shape[0]:
        raise ScopeError("quad_centers returned misaligned center/index arrays")
    inside = (centers[:, 0] >= Z_BEGIN) & (centers[:, 0] <= SCORE_Z_END)
    return int(inside.sum())


def load_scope_rows(
    *,
    heldout: Path,
    names: Sequence[str],
    split_document: dict[str, Any],
    api: SpiralCheckAPI,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    content_hashes = split_document["content_sha256"]
    geometry_hashes = split_document["geometry_sha256"]
    for index, name in enumerate(sorted(names), 1):
        patch = heldout / name
        try:
            actual_content, actual_geometry = _patch_hashes(patch)
            if actual_content != content_hashes[name]:
                raise ScopeError(
                    f"content SHA-256 {actual_content} != manifest {content_hashes[name]}"
                )
            if actual_geometry != geometry_hashes[name]:
                raise ScopeError(
                    f"geometry SHA-256 {actual_geometry} != manifest {geometry_hashes[name]}"
                )
            surface = api.load_tifxyz(patch)
            if not isinstance(surface, api.quad_surface_type):
                raise ScopeError(
                    f"load_tifxyz returned {type(surface).__name__}, not QuadSurface"
                )
            rows.append({"patch_id": name, "n_points": count_in_window_points(surface)})
        except Exception as exc:  # noqa: BLE001 - all patches are attempted and reported
            errors[name] = f"{type(exc).__name__}: {exc}"
        if index % 1000 == 0:
            print(f"audited {index:,}/{len(names):,} held-out tifxyz directories", flush=True)
    return rows, errors


def _portable_file_record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ScopeError(f"required evidence source is missing: {path}")
    return {"name": path.name, "sha256": _sha256(path)}


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ScopeError(f"refusing to reuse output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            descriptor = -1
            stream.write(rendered)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def generate(
    *,
    spiralcheck_source: Path,
    split_manifest: Path,
    heldout_patches: Path,
    source_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise ScopeError(f"refusing to reuse output: {output}")
    if output.parent == output:
        raise ScopeError("output cannot be a filesystem root")

    evidence_dir = Path(__file__).resolve().parent
    protocol = evidence_dir / "SEALED_PATCH_PROTOCOL.md"
    amendment = evidence_dir / "SEALED_OPERATIONAL_AMENDMENT.md"
    checkout = validate_spiralcheck_checkout(spiralcheck_source)
    api = load_spiralcheck_api(spiralcheck_source)
    if _sha256(api.source_file) != checkout["source_sha256"]:
        raise ScopeError("SpiralCheck source changed between checkout validation and import")
    source_record = validate_source_manifest(source_manifest.resolve())
    split_document, fit_names, heldout_names = validate_split_manifest(
        split_manifest.resolve()
    )
    validate_heldout_directory(
        heldout_patches.resolve(), split_manifest.resolve(), heldout_names
    )
    rows, errors = load_scope_rows(
        heldout=heldout_patches.resolve(),
        names=heldout_names,
        split_document=split_document,
        api=api,
    )
    if errors:
        preview = dict(list(sorted(errors.items()))[:10])
        raise ScopeError(
            f"strict held-out load failed for {len(errors)} of "
            f"{len(heldout_names)} patches; first errors={json.dumps(preview, sort_keys=True)}"
        )
    if len(rows) != EXPECTED_HELDOUT_COUNT:
        raise ScopeError(
            f"strict held-out load returned {len(rows)} rows; expected "
            f"{EXPECTED_HELDOUT_COUNT}"
        )

    skipped = sum(row["n_points"] == 0 for row in rows)
    split_sha = _sha256(split_manifest.resolve())
    document = {
        "schema": "sealed-heldout-point-scope-v1",
        "model_independent": True,
        "z_window": {
            "fit_semantics": "[z_begin,z_end)",
            "z_begin": Z_BEGIN,
            "z_end_exclusive": Z_END_HALF_OPEN,
            "spiralcheck_inclusive_upper": SCORE_Z_END,
        },
        "provenance": {
            "generator": _portable_file_record(Path(__file__).resolve()),
            "spiralcheck": checkout,
            "source_manifest": source_record,
            "split_manifest": {"name": split_manifest.name, "sha256": split_sha},
            "sealed_protocol": _portable_file_record(protocol),
            "operational_amendment": _portable_file_record(amendment),
        },
        "split": {
            "seed": split_document["seed"],
            "heldout_fraction": split_document["heldout_frac"],
            "patch_count": len(fit_names) + len(heldout_names),
            "fit_patch_count": len(fit_names),
            "heldout_patch_count": len(heldout_names),
            "all_id_sha256": _canonical_name_sha256(fit_names + heldout_names),
            "fit_id_sha256": _canonical_name_sha256(fit_names),
            "heldout_id_sha256": _canonical_name_sha256(heldout_names),
            "heldout_content_index_sha256": _canonical_index_sha256(
                (name, split_document["content_sha256"][name]) for name in heldout_names
            ),
            "heldout_geometry_index_sha256": _canonical_index_sha256(
                (name, split_document["geometry_sha256"][name]) for name in heldout_names
            ),
        },
        "scope": {
            "canonical_encoding": "UTF-8 patch_id<TAB>n_points<LF>, sorted by patch_id",
            "canonical_sha256": canonical_scope_sha256(rows),
            "patch_count": len(rows),
            "patches_with_points": len(rows) - skipped,
            "patches_skipped_zero_points": skipped,
            "total_in_window_points": sum(row["n_points"] for row in rows),
            "strict_load_errors": 0,
            "patches": rows,
        },
    }
    _write_json_exclusive(output, document)
    print(f"sealed held-out scope manifest: {output}", flush=True)
    return document


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spiralcheck-source", required=True,
        help="clean SpiralCheck repository pinned to the frozen commit",
    )
    parser.add_argument("--split-manifest", required=True, help="sealed split_manifest.json")
    parser.add_argument("--heldout-patches", required=True, help="sealed heldout directory")
    parser.add_argument(
        "--source-manifest",
        default=str(Path(__file__).resolve().parent / "public_input_manifest.json"),
        help="frozen public_input_manifest.json (defaults beside this program)",
    )
    parser.add_argument("--output", required=True, help="new output JSON; must not exist")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    generate(
        spiralcheck_source=Path(args.spiralcheck_source).expanduser().resolve(),
        split_manifest=Path(args.split_manifest).expanduser().resolve(),
        heldout_patches=Path(args.heldout_patches).expanduser().resolve(),
        source_manifest=Path(args.source_manifest).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScopeError as exc:
        raise SystemExit(f"held-out scope manifest refused: {exc}") from exc
