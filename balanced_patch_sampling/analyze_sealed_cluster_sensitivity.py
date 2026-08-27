#!/usr/bin/env python3
"""Run the pre-result source-family sensitivity for the sealed comparison.

This is an additional robustness analysis. It does not change the frozen
patch-level comparator or any of its pass/fail gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from compare_sealed_patch_reports import (
    FAMILY_BAND,
    FAMILY_OTHER,
    MIN_UNSEEN_POINTS,
    ComparisonError,
    _family_scores,
    _parse_report,
    _validate_pair,
)


EXPECTED_MANIFEST_SHA256 = (
    "9a1b226ebde3854728adfeb6f21513026c0cc49d948cc52684d6ee96e3819f31"
)
PROTOCOL_NAME = "SEALED_CLUSTER_SENSITIVITY_PROTOCOL.md"
SEED = 20260827
DRAWS = 20_000
CI_PERCENTILES = (2.5, 97.5)


class SensitivityError(ValueError):
    """An input cannot support the frozen sensitivity analysis."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(
    path_arg: str, *, expected_sha256: str | None = EXPECTED_MANIFEST_SHA256
) -> tuple[Path, str, dict[str, Any]]:
    path = Path(path_arg).expanduser().resolve()
    if not path.is_file():
        raise SensitivityError(f"split manifest does not exist: {path}")
    sha256 = _sha256(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise SensitivityError(
            f"split manifest SHA-256 is {sha256}; expected {expected_sha256}"
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SensitivityError(f"could not read split manifest {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise SensitivityError("split manifest root must be an object")
    family_of = document.get("family_of")
    assignments = document.get("assignments")
    if not isinstance(family_of, dict) or not isinstance(assignments, dict):
        raise SensitivityError("split manifest must contain family_of and assignments objects")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in family_of.items()):
        raise SensitivityError("split manifest family_of must map strings to strings")
    return path, sha256, document


def _family(patch_id: str) -> str:
    return FAMILY_BAND if patch_id.startswith("band-seed") else FAMILY_OTHER


def _clusters(report: Any, manifest: dict[str, Any]) -> dict[str, list[list[str]]]:
    family_of: dict[str, str] = manifest["family_of"]
    assignments: dict[str, str] = manifest["assignments"]
    grouped: dict[str, dict[str, list[str]]] = {
        FAMILY_BAND: {},
        FAMILY_OTHER: {},
    }
    for patch_id in sorted(report.patches):
        if assignments.get(patch_id) != "heldout":
            raise SensitivityError(
                f"eligible report patch {patch_id!r} is not assigned to heldout"
            )
        family = _family(patch_id)
        cluster_id = family_of.get(patch_id, patch_id)
        grouped[family].setdefault(cluster_id, []).append(patch_id)

    # A source cluster crossing reporting families would make the stated
    # within-family resampling rule ambiguous, so fail instead of splitting it.
    owners: dict[str, str] = {}
    for family, clusters in grouped.items():
        for cluster_id in clusters:
            previous = owners.setdefault(cluster_id, family)
            if previous != family:
                raise SensitivityError(
                    f"source cluster {cluster_id!r} crosses reporting families"
                )
    for family in (FAMILY_BAND, FAMILY_OTHER):
        if not grouped[family]:
            raise SensitivityError(f"no eligible cluster in reporting family {family!r}")
    return {
        family: [grouped[family][key] for key in sorted(grouped[family])]
        for family in (FAMILY_BAND, FAMILY_OTHER)
    }


def _cluster_bootstrap(
    baseline: Any, treatment: Any, clusters: dict[str, list[list[str]]]
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    deltas = np.empty(DRAWS, dtype=np.float64)
    cursor = 0
    block_size = 64
    while cursor < DRAWS:
        count = min(block_size, DRAWS - cursor)
        family_deltas: list[np.ndarray] = []
        for family in (FAMILY_BAND, FAMILY_OTHER):
            members = clusters[family]
            weights = np.asarray(
                [sum(baseline.patches[pid].n_points for pid in ids) for ids in members],
                dtype=np.float64,
            )
            baseline_numerators = np.asarray(
                [
                    sum(
                        baseline.patches[pid].n_points
                        * baseline.patches[pid].sheet_consistency
                        for pid in ids
                    )
                    for ids in members
                ],
                dtype=np.float64,
            )
            treatment_numerators = np.asarray(
                [
                    sum(
                        baseline.patches[pid].n_points
                        * treatment.patches[pid].sheet_consistency
                        for pid in ids
                    )
                    for ids in members
                ],
                dtype=np.float64,
            )
            indices = rng.integers(
                0, len(members), size=(count, len(members)), endpoint=False
            )
            sampled_weights = weights[indices].sum(axis=1)
            baseline_score = baseline_numerators[indices].sum(axis=1) / sampled_weights
            treatment_score = treatment_numerators[indices].sum(axis=1) / sampled_weights
            family_deltas.append(treatment_score - baseline_score)
        deltas[cursor : cursor + count] = (
            family_deltas[0] + family_deltas[1]
        ) / 2.0
        cursor += count
    low, high = np.percentile(deltas, CI_PERCENTILES)
    return {
        "method": (
            "paired source-cluster bootstrap within band-seed/other; resample "
            "clusters uniformly, retain all member patches and unseen-point weights"
        ),
        "draws": DRAWS,
        "seed": SEED,
        "ci_percentiles": list(CI_PERCENTILES),
        "primary_delta_treatment_minus_baseline_ci": [float(low), float(high)],
    }


def _raw_rows(report: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in report.document["heldout_patches"]:
        unseen = row.get("unseen")
        if isinstance(unseen, dict) and unseen.get("n_points", 0) >= MIN_UNSEEN_POINTS:
            rows[row["patch_id"]] = unseen
    if set(rows) != set(report.patches):
        raise SensitivityError("eligible raw rows disagree with parsed report population")
    return rows


def _family_proximity(report: Any) -> dict[str, dict[str, float | int]]:
    rows = _raw_rows(report)
    output: dict[str, dict[str, float | int]] = {}
    for family in (FAMILY_BAND, FAMILY_OTHER):
        selected = [
            (patch_id, row)
            for patch_id, row in rows.items()
            if _family(patch_id) == family
        ]
        points = sum(int(row["n_points"]) for _, row in selected)
        numerator = 0.0
        for patch_id, row in selected:
            value = row.get("frac_within_tau")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SensitivityError(
                    f"{patch_id!r}: unseen.frac_within_tau must be numeric"
                )
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise SensitivityError(
                    f"{patch_id!r}: unseen.frac_within_tau is invalid: {value!r}"
                )
            numerator += int(row["n_points"]) * value
        output[family] = {
            "n_patches": len(selected),
            "n_points": points,
            "frac_within_tau_point_weighted": numerator / points,
        }
    pooled = sum(
        int(row["n_points"]) * float(row["frac_within_tau"])
        for row in rows.values()
    ) / sum(int(row["n_points"]) for row in rows.values())
    published = float(report.unseen_aggregate["frac_within_tau"])
    if not math.isclose(pooled, published, rel_tol=0.0, abs_tol=1e-12):
        raise SensitivityError(
            f"family proximity recombination {pooled} != published pooled {published}"
        )
    return output


def analyze(
    baseline_path: str,
    treatment_path: str,
    manifest_path: str,
    *,
    expected_manifest_sha256: str | None = EXPECTED_MANIFEST_SHA256,
) -> dict[str, Any]:
    try:
        baseline = _parse_report(baseline_path)
        treatment = _parse_report(treatment_path)
        _validate_pair(baseline, treatment)
    except ComparisonError as exc:
        raise SensitivityError(str(exc)) from exc
    manifest_file, manifest_sha256, manifest = _load_manifest(
        manifest_path, expected_sha256=expected_manifest_sha256
    )
    clusters = _clusters(baseline, manifest)
    bootstrap = _cluster_bootstrap(baseline, treatment, clusters)
    baseline_scores = _family_scores(baseline)
    treatment_scores = _family_scores(treatment)
    primary_delta = float(
        treatment_scores["macro_equal_family"]["sheet_consistency"]
    ) - float(baseline_scores["macro_equal_family"]["sheet_consistency"])
    ci_low = bootstrap["primary_delta_treatment_minus_baseline_ci"][0]

    proximity_baseline = _family_proximity(baseline)
    proximity_treatment = _family_proximity(treatment)
    proximity_delta = {
        family: (
            float(proximity_treatment[family]["frac_within_tau_point_weighted"])
            - float(proximity_baseline[family]["frac_within_tau_point_weighted"])
        )
        for family in (FAMILY_BAND, FAMILY_OTHER)
    }
    here = Path(__file__).resolve()
    protocol = here.with_name(PROTOCOL_NAME)
    return {
        "schema": "sealed-phercparis4-cluster-sensitivity-v1",
        "provenance": {
            "baseline_report": str(baseline.path),
            "baseline_report_sha256": baseline.sha256,
            "treatment_report": str(treatment.path),
            "treatment_report_sha256": treatment.sha256,
            "split_manifest": str(manifest_file),
            "split_manifest_sha256": manifest_sha256,
            "protocol": str(protocol),
            "protocol_sha256": _sha256(protocol) if protocol.is_file() else None,
            "analyzer_sha256": _sha256(here),
        },
        "paired_population": {
            "n_patches": len(baseline.patches),
            "n_points": sum(row.n_points for row in baseline.patches.values()),
            "reporting_families": {
                family: {
                    "n_patches": sum(len(ids) for ids in clusters[family]),
                    "n_source_clusters": len(clusters[family]),
                    "largest_cluster_patches": max(len(ids) for ids in clusters[family]),
                }
                for family in (FAMILY_BAND, FAMILY_OTHER)
            },
        },
        "primary": {
            "frozen_point_estimate_treatment_minus_baseline": primary_delta,
            "cluster_bootstrap": bootstrap,
            "cluster_robust": primary_delta > 0.0 and ci_low > 0.0,
        },
        "family_specific_unseen_within_tau": {
            "baseline": proximity_baseline,
            "treatment": proximity_treatment,
            "treatment_minus_baseline": proximity_delta,
            "no_family_specific_regression_beyond_1pp": all(
                value >= -0.01 for value in proximity_delta.values()
            ),
        },
        "interpretation": (
            "Sensitivity only; the original frozen comparator remains the primary. "
            "The held-out outcome was untouched during selection, but cap selection "
            "used development fits on the full public patch collection."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _synthetic_report(rows: list[tuple[str, int, float, float]]) -> dict[str, Any]:
    points = sum(row[1] for row in rows)
    weighted_sheet = sum(row[1] * row[2] for row in rows) / points
    pooled_within = sum(row[1] * row[3] for row in rows) / points
    return {
        "meta": {
            "tau": 6.0,
            "z_range": "10500,11500",
            "manifest": "sealed-manifest.json",
            "unseen_min_dist": 2.0,
            "fit_inputs_hash_audit": "clean",
        },
        "heldout_patches": [
            {
                "patch_id": patch_id,
                "unseen": {
                    "n_points": n_points,
                    "sheet_consistency": sheet,
                    "frac_within_tau": within,
                },
            }
            for patch_id, n_points, sheet, within in rows
        ],
        "heldout_aggregate": {
            "unseen": {
                "unseen_min_dist": 2.0,
                "n_patches": len(rows),
                "n_patches_excluded": 0,
                "n_points": points,
                "frac_within_tau": pooled_within,
                "mean_sheet_consistency": weighted_sheet,
            }
        },
    }


def self_test() -> None:
    baseline_rows = [
        ("band-seed-a", 20, 0.50, 0.90),
        ("band-seed-b", 30, 0.60, 0.80),
        ("legacy-a-1", 10, 0.40, 0.70),
        ("legacy-a-2", 15, 0.45, 0.75),
        ("legacy-b", 25, 0.55, 0.85),
    ]
    treatment_rows = [
        ("band-seed-a", 20, 0.53, 0.90),
        ("band-seed-b", 30, 0.63, 0.81),
        ("legacy-a-1", 10, 0.48, 0.72),
        ("legacy-a-2", 15, 0.53, 0.77),
        ("legacy-b", 25, 0.63, 0.87),
    ]
    with tempfile.TemporaryDirectory(prefix="sealed-cluster-sensitivity-") as temporary:
        root = Path(temporary)
        baseline = root / "baseline.json"
        treatment = root / "treatment.json"
        manifest = root / "split_manifest.json"
        baseline.write_text(json.dumps(_synthetic_report(baseline_rows)), encoding="utf-8")
        treatment.write_text(json.dumps(_synthetic_report(treatment_rows)), encoding="utf-8")
        manifest.write_text(
            json.dumps(
                {
                    "assignments": {
                        patch_id: "heldout" for patch_id, *_ in baseline_rows
                    },
                    "family_of": {
                        "legacy-a-1": "legacy-a",
                        "legacy-a-2": "legacy-a",
                    },
                }
            ),
            encoding="utf-8",
        )
        one = analyze(
            str(baseline), str(treatment), str(manifest),
            expected_manifest_sha256=None,
        )
        two = analyze(
            str(baseline), str(treatment), str(manifest),
            expected_manifest_sha256=None,
        )
        assert one["primary"]["cluster_bootstrap"] == two["primary"]["cluster_bootstrap"]
        assert one["paired_population"]["reporting_families"][FAMILY_OTHER]["n_source_clusters"] == 2
        assert one["paired_population"]["reporting_families"][FAMILY_OTHER]["n_patches"] == 3
        assert one["primary"]["cluster_robust"] is True
        assert one["family_specific_unseen_within_tau"]["no_family_specific_regression_beyond_1pp"] is True
    print("self-test passed: paired source-family cluster sensitivity")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?")
    parser.add_argument("treatment", nargs="?")
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        if any((args.baseline, args.treatment, args.manifest, args.output)):
            parser.error("--self-test cannot be combined with report arguments")
        self_test()
        return 0
    if not all((args.baseline, args.treatment, args.manifest, args.output)):
        parser.error("baseline, treatment, --manifest, and --output are required")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    try:
        result = analyze(args.baseline, args.treatment, args.manifest)
    except SensitivityError as exc:
        parser.exit(2, f"cluster sensitivity refused: {exc}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "cluster sensitivity: "
        + ("ROBUST" if result["primary"]["cluster_robust"] else "UNCERTAIN")
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
