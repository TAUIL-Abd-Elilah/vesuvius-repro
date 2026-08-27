#!/usr/bin/env python3
"""Compare the two sealed SpiralCheck reports in SEALED_PATCH_PROTOCOL.md.

This is deliberately stricter than SpiralCheck's general ``compare`` command:
it refuses to silently drop a patch, mix leakage thresholds, or use different
unseen-point weights.  It reports treatment minus baseline.  The frozen
primary is the equal-family macro average of point-weighted *per-patch*
unseen sheet consistency, with ``band-seed`` and every other patch as the two
families.

Example
-------
python compare_sealed_patch_reports.py baseline/report.json treatment/report.json \
    --output sealed_comparison.json

Run ``--self-test`` before relying on a new environment.  The test makes tiny
synthetic reports, verifies the frozen gates, exact pairing checks, and a
repeatable 20,000-draw confidence interval without requiring a GPU or data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised at invocation time
    raise SystemExit(
        "This comparator requires numpy (the same numerical dependency used by "
        "SpiralCheck). Install it in the environment that runs SpiralCheck."
    ) from exc


PROTOCOL = "SEALED_PATCH_PROTOCOL.md"
SEED = 20260827
DRAWS = 20_000
MIN_UNSEEN_POINTS = 8  # SpiralCheck's frozen unseen-aggregate inclusion floor.
CI_PERCENTILES = (2.5, 97.5)
FAMILY_BAND = "band-seed"
FAMILY_OTHER = "other"


class ComparisonError(ValueError):
    """An input violates the sealed comparison contract."""


@dataclass(frozen=True)
class Patch:
    patch_id: str
    family: str
    n_points: int
    sheet_consistency: float


@dataclass(frozen=True)
class Report:
    path: Path
    sha256: str
    document: dict[str, Any]
    all_patch_ids: frozenset[str]
    unseen_counts: dict[str, int]
    patches: dict[str, Patch]
    unseen_aggregate: dict[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path_arg: str) -> tuple[Path, bytes, dict[str, Any]]:
    path = Path(path_arg).expanduser().resolve()
    if not path.is_file():
        raise ComparisonError(f"{path}: report.json does not exist or is not a file")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"{path}: could not read report: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ComparisonError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(document, dict):
        raise ComparisonError(f"{path}: report root must be a JSON object, got {type(document).__name__}")
    return path, raw, document


def _number(value: Any, where: str, *, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonError(f"{where}: expected a finite number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise ComparisonError(f"{where}: expected a finite number, got {value!r}")
    if lower is not None and out < lower:
        raise ComparisonError(f"{where}: expected a value >= {lower}, got {out}")
    if upper is not None and out > upper:
        raise ComparisonError(f"{where}: expected a value <= {upper}, got {out}")
    return out


def _integer(value: Any, where: str, *, lower: int = 0) -> int:
    number = _number(value, where, lower=float(lower))
    if not number.is_integer():
        raise ComparisonError(f"{where}: expected an integer, got {value!r}")
    return int(number)


def _family(patch_id: str) -> str:
    return FAMILY_BAND if patch_id.startswith("band-seed") else FAMILY_OTHER


def _parse_report(path_arg: str) -> Report:
    path, raw, doc = _load_json(path_arg)
    rows = doc.get("heldout_patches")
    if not isinstance(rows, list):
        raise ComparisonError(
            f"{path}: no heldout_patches list; score with SpiralCheck plus --fit-inputs "
            "so per-patch unseen results are available"
        )
    aggregate = doc.get("heldout_aggregate")
    if not isinstance(aggregate, dict):
        raise ComparisonError(f"{path}: no heldout_aggregate object")
    unseen_aggregate = aggregate.get("unseen")
    if not isinstance(unseen_aggregate, dict):
        raise ComparisonError(
            f"{path}: no heldout_aggregate.unseen object; score with SpiralCheck --fit-inputs"
        )
    # Gate 3 is intentionally the pooled quantity SpiralCheck publishes, not
    # a point-weighted average of per-patch fractions.
    _number(unseen_aggregate.get("frac_within_tau"), f"{path}: heldout_aggregate.unseen.frac_within_tau", lower=0.0, upper=1.0)
    _integer(unseen_aggregate.get("n_points"), f"{path}: heldout_aggregate.unseen.n_points", lower=0)
    _number(unseen_aggregate.get("unseen_min_dist"), f"{path}: heldout_aggregate.unseen.unseen_min_dist", lower=0.0)

    parsed: dict[str, Patch] = {}
    all_patch_ids: set[str] = set()
    unseen_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        where = f"{path}: heldout_patches[{index}]"
        if not isinstance(row, dict):
            raise ComparisonError(f"{where}: expected an object")
        patch_id = row.get("patch_id")
        if not isinstance(patch_id, str) or not patch_id:
            raise ComparisonError(f"{where}.patch_id: expected a nonempty string")
        if patch_id in all_patch_ids:
            raise ComparisonError(f"{path}: duplicate patch_id {patch_id!r}")
        all_patch_ids.add(patch_id)
        unseen = row.get("unseen")
        # SpiralCheck writes {"n_points": 0} for an all-seen patch. Treat it
        # as excluded rather than pretending it has a score.
        if unseen is None:
            n_points = 0
            sheet = 0.0
        else:
            if not isinstance(unseen, dict):
                raise ComparisonError(f"{where}.unseen: expected an object or null")
            n_points = _integer(unseen.get("n_points"), f"{where}.unseen.n_points", lower=0)
            if n_points:
                sheet = _number(
                    unseen.get("sheet_consistency"),
                    f"{where}.unseen.sheet_consistency",
                    lower=0.0,
                    upper=1.0,
                )
            else:
                sheet = 0.0
        unseen_counts[patch_id] = n_points
        if n_points >= MIN_UNSEEN_POINTS:
            parsed[patch_id] = Patch(patch_id, _family(patch_id), n_points, sheet)

    if not parsed:
        raise ComparisonError(
            f"{path}: no patch has >= {MIN_UNSEEN_POINTS} unseen points, so the "
            "sealed unseen comparison cannot be computed"
        )
    # This catches an accidental mixed version/report where per-patch and
    # published unseen aggregates use different eligibility criteria.
    parsed_total = sum(row.n_points for row in parsed.values())
    aggregate_total = _integer(unseen_aggregate["n_points"], f"{path}: heldout_aggregate.unseen.n_points")
    if parsed_total != aggregate_total:
        raise ComparisonError(
            f"{path}: sum of eligible per-patch unseen n_points is {parsed_total}, "
            f"but heldout_aggregate.unseen.n_points is {aggregate_total}; refuse to "
            "mix inconsistent report sections"
        )
    return Report(
        path, _sha256_bytes(raw), doc, frozenset(all_patch_ids), unseen_counts,
        parsed, unseen_aggregate,
    )


def _same(value_a: Any, value_b: Any) -> bool:
    # json values in the comparison provenance must be identical, but make
    # numeric 6 and 6.0 equivalent because JSON writers differ on that detail.
    if isinstance(value_a, (int, float)) and not isinstance(value_a, bool) and isinstance(value_b, (int, float)) and not isinstance(value_b, bool):
        return float(value_a) == float(value_b)
    return value_a == value_b


def _validate_pair(baseline: Report, treatment: Report) -> None:
    baseline_all_ids, treatment_all_ids = baseline.all_patch_ids, treatment.all_patch_ids
    if baseline_all_ids != treatment_all_ids:
        only_baseline = sorted(baseline_all_ids - treatment_all_ids)
        only_treatment = sorted(treatment_all_ids - baseline_all_ids)
        detail = []
        if only_baseline:
            detail.append(f"only baseline ({len(only_baseline)}): {only_baseline[:5]}")
        if only_treatment:
            detail.append(f"only treatment ({len(only_treatment)}): {only_treatment[:5]}")
        raise ComparisonError(
            "sealed comparison requires exactly identical held-out patch IDs; " + "; ".join(detail)
        )
    all_count_mismatch = [
        patch_id for patch_id in sorted(baseline_all_ids)
        if baseline.unseen_counts[patch_id] != treatment.unseen_counts[patch_id]
    ]
    if all_count_mismatch:
        pid = all_count_mismatch[0]
        raise ComparisonError(
            f"{pid!r}: unseen n_points differ (baseline={baseline.unseen_counts[pid]}, "
            f"treatment={treatment.unseen_counts[pid]}); the arms did not score identical "
            "unseen evidence"
        )
    baseline_ids, treatment_ids = set(baseline.patches), set(treatment.patches)
    if baseline_ids != treatment_ids:
        raise ComparisonError(
            "eligible unseen patch sets differ despite matching report IDs; this violates the "
            f"frozen >= {MIN_UNSEEN_POINTS}-point eligibility rule"
        )
    mismatched_counts = [
        patch_id for patch_id in sorted(baseline_ids)
        if baseline.patches[patch_id].n_points != treatment.patches[patch_id].n_points
    ]
    if mismatched_counts:
        pid = mismatched_counts[0]
        raise ComparisonError(
            f"{pid!r}: unseen n_points differ (baseline={baseline.patches[pid].n_points}, "
            f"treatment={treatment.patches[pid].n_points}); the arms did not score identical "
            "unseen evidence"
        )
    # The primary metric does not make sense if the scorer was given different
    # geometric leakage definitions, tau, manifest, or z window.
    keys = ("tau", "z_range", "manifest", "unseen_min_dist", "fit_inputs_hash_audit")
    a_meta = baseline.document.get("meta")
    b_meta = treatment.document.get("meta")
    if not isinstance(a_meta, dict) or not isinstance(b_meta, dict):
        raise ComparisonError("both reports must contain meta objects")
    for key in keys:
        a_value = a_meta.get(key)
        b_value = b_meta.get(key)
        if a_value is None or b_value is None:
            raise ComparisonError(f"both reports must record meta.{key} for a sealed comparison")
        if not _same(a_value, b_value):
            raise ComparisonError(
                f"report metadata disagree for {key}: baseline={a_value!r}, treatment={b_value!r}"
            )
    if a_meta["fit_inputs_hash_audit"] != "clean":
        raise ComparisonError(
            "meta.fit_inputs_hash_audit must be 'clean' in both reports; actual fit-side inputs "
            "must pass SpiralCheck's leakage audit"
        )
    for key in ("unseen_min_dist",):
        if not _same(baseline.unseen_aggregate.get(key), treatment.unseen_aggregate.get(key)):
            raise ComparisonError(f"published unseen aggregate values disagree for {key}")
    for family in (FAMILY_BAND, FAMILY_OTHER):
        if not any(row.family == family for row in baseline.patches.values()):
            raise ComparisonError(
                f"no eligible unseen patch in family {family!r}; equal-family macro cannot be calculated"
            )


def _family_scores(report: Report) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for family in (FAMILY_BAND, FAMILY_OTHER):
        rows = [row for row in report.patches.values() if row.family == family]
        weights = sum(row.n_points for row in rows)
        out[family] = {
            "n_patches": len(rows),
            "n_points": weights,
            "sheet_consistency_point_weighted": sum(
                row.n_points * row.sheet_consistency for row in rows
            ) / weights,
        }
    out["macro_equal_family"] = {
        "sheet_consistency": (
            float(out[FAMILY_BAND]["sheet_consistency_point_weighted"])
            + float(out[FAMILY_OTHER]["sheet_consistency_point_weighted"])
        ) / 2.0
    }
    return out


def _bootstrap(baseline: Report, treatment: Report) -> dict[str, Any]:
    """Paired, within-family bootstrap with bounded memory.

    Chunking is fixed, so the published result is deterministic for the frozen
    seed and this implementation, even for a large sealed side.
    """
    rng = np.random.default_rng(SEED)
    family_ids = {
        family: sorted(pid for pid, row in baseline.patches.items() if row.family == family)
        for family in (FAMILY_BAND, FAMILY_OTHER)
    }
    deltas = np.empty(DRAWS, dtype=np.float64)
    cursor = 0
    block_size = 64
    while cursor < DRAWS:
        count = min(block_size, DRAWS - cursor)
        family_deltas: list[np.ndarray] = []
        for family in (FAMILY_BAND, FAMILY_OTHER):
            ids = family_ids[family]
            weights = np.asarray([baseline.patches[pid].n_points for pid in ids], dtype=np.float64)
            base = np.asarray([baseline.patches[pid].sheet_consistency for pid in ids], dtype=np.float64)
            treat = np.asarray([treatment.patches[pid].sheet_consistency for pid in ids], dtype=np.float64)
            indices = rng.integers(0, len(ids), size=(count, len(ids)), endpoint=False)
            sampled_weights = weights[indices]
            base_score = (base[indices] * sampled_weights).sum(axis=1) / sampled_weights.sum(axis=1)
            treat_score = (treat[indices] * sampled_weights).sum(axis=1) / sampled_weights.sum(axis=1)
            family_deltas.append(treat_score - base_score)
        deltas[cursor: cursor + count] = (family_deltas[0] + family_deltas[1]) / 2.0
        cursor += count
    ci_low, ci_high = np.percentile(deltas, CI_PERCENTILES)
    return {
        "method": "paired, stratified by patch-id family; resample patches with replacement within each family; point-weight each resample",
        "draws": DRAWS,
        "seed": SEED,
        "ci_percentiles": list(CI_PERCENTILES),
        "primary_delta_treatment_minus_baseline_ci": [float(ci_low), float(ci_high)],
    }


def _aggregate_summary(report: Report) -> dict[str, Any]:
    unseen = report.unseen_aggregate
    keys = (
        "n_patches", "n_patches_excluded", "n_points", "dist_p50", "dist_p90",
        "dist_p99", "dist_max", "frac_within_tau", "mean_sheet_consistency",
        "min_sheet_consistency", "normal_angle_p90_deg", "unseen_min_dist",
    )
    return {key: unseen[key] for key in keys if key in unseen}


def _delta(treatment_value: Any, baseline_value: Any) -> float | None:
    if isinstance(treatment_value, bool) or isinstance(baseline_value, bool):
        return None
    if isinstance(treatment_value, (int, float)) and isinstance(baseline_value, (int, float)):
        return float(treatment_value) - float(baseline_value)
    return None


def compare(baseline_path: str, treatment_path: str) -> dict[str, Any]:
    baseline = _parse_report(baseline_path)
    treatment = _parse_report(treatment_path)
    _validate_pair(baseline, treatment)
    baseline_scores = _family_scores(baseline)
    treatment_scores = _family_scores(treatment)
    primary_delta = (
        float(treatment_scores["macro_equal_family"]["sheet_consistency"])
        - float(baseline_scores["macro_equal_family"]["sheet_consistency"])
    )
    bootstrap = _bootstrap(baseline, treatment)
    ci_low, ci_high = bootstrap["primary_delta_treatment_minus_baseline_ci"]
    band_delta = (
        float(treatment_scores[FAMILY_BAND]["sheet_consistency_point_weighted"])
        - float(baseline_scores[FAMILY_BAND]["sheet_consistency_point_weighted"])
    )
    pooled_delta = _delta(
        treatment.unseen_aggregate["frac_within_tau"], baseline.unseen_aggregate["frac_within_tau"]
    )
    assert pooled_delta is not None
    gates = {
        "primary_positive_and_95ci_above_zero": {
            "rule": "treatment - baseline macro primary > 0 and 95% paired-bootstrap CI lower bound > 0",
            "point_estimate": primary_delta,
            "ci_95": [ci_low, ci_high],
            "pass": primary_delta > 0.0 and ci_low > 0.0,
        },
        "band_seed_noninferior_within_3pp": {
            "rule": "treatment - baseline band-seed unseen sheet consistency >= -0.03",
            "delta": band_delta,
            "threshold": -0.03,
            "pass": band_delta >= -0.03,
        },
        "pooled_unseen_within_tau_noninferior_within_1pp": {
            "rule": "treatment - baseline pooled unseen frac_within_tau >= -0.01",
            "delta": pooled_delta,
            "threshold": -0.01,
            "pass": pooled_delta >= -0.01,
        },
    }
    baseline_aggregate = _aggregate_summary(baseline)
    treatment_aggregate = _aggregate_summary(treatment)
    aggregate_delta = {
        key: _delta(treatment_aggregate.get(key), baseline_aggregate.get(key))
        for key in sorted(set(baseline_aggregate) & set(treatment_aggregate))
    }
    return {
        "schema": "sealed-phercparis4-patch-sampling-comparison-v1",
        "protocol": {
            "path": PROTOCOL,
            "frozen_primary": "equal-family macro of point-weighted per-patch unseen sheet_consistency",
            "families": {FAMILY_BAND: "patch_id startswith 'band-seed'", FAMILY_OTHER: "all other patch IDs"},
            "minimum_unseen_points_per_patch": MIN_UNSEEN_POINTS,
        },
        "provenance": {
            "baseline_report": str(baseline.path),
            "baseline_report_sha256": baseline.sha256,
            "treatment_report": str(treatment.path),
            "treatment_report_sha256": treatment.sha256,
            "baseline_meta": baseline.document.get("meta"),
            "treatment_meta": treatment.document.get("meta"),
        },
        "paired_population": {
            "n_patches": len(baseline.patches),
            "n_points": sum(row.n_points for row in baseline.patches.values()),
            "patch_ids_sha256": _sha256_bytes("\n".join(sorted(baseline.patches)).encode("utf-8")),
        },
        "unseen_sheet_consistency": {
            "baseline": baseline_scores,
            "treatment": treatment_scores,
            "treatment_minus_baseline_macro": primary_delta,
            "treatment_minus_baseline_by_family": {
                family: (
                    float(treatment_scores[family]["sheet_consistency_point_weighted"])
                    - float(baseline_scores[family]["sheet_consistency_point_weighted"])
                )
                for family in (FAMILY_BAND, FAMILY_OTHER)
            },
        },
        "bootstrap": bootstrap,
        "unseen_aggregate": {
            "baseline": baseline_aggregate,
            "treatment": treatment_aggregate,
            "treatment_minus_baseline": aggregate_delta,
        },
        "intrinsic": {
            "baseline": baseline.document.get("intrinsic"),
            "treatment": treatment.document.get("intrinsic"),
        },
        "gates": gates,
        "pass_all_frozen_gates": all(gate["pass"] for gate in gates.values()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _synthetic_report(rows: list[tuple[str, int, float, float]], *, within_tau: float) -> dict[str, Any]:
    """Build a valid report: rows are id, unseen points, sheet, unused marker."""
    selected = [row for row in rows if row[1] >= MIN_UNSEEN_POINTS]
    points = sum(row[1] for row in selected)
    weighted_sheet = sum(row[1] * row[2] for row in selected) / points
    patch_rows = []
    for patch_id, n_points, sheet, _marker in rows:
        patch_rows.append({
            "patch_id": patch_id,
            "unseen": {"n_points": n_points, **({"sheet_consistency": sheet} if n_points else {})},
        })
    return {
        "meta": {
            "spiralcheck": "0.4.0", "tau": 6.0, "z_range": "10500,11500",
            "manifest": "sealed-manifest.json", "unseen_min_dist": 2.0,
            "fit_inputs_hash_audit": "clean",
        },
        "heldout_patches": patch_rows,
        "heldout_aggregate": {"unseen": {
            "unseen_min_dist": 2.0, "n_patches": len(selected), "n_patches_excluded": len(rows) - len(selected),
            "n_points": points, "dist_p50": 1.0, "dist_p90": 3.0, "dist_p99": 5.0,
            "dist_max": 7.0, "frac_within_tau": within_tau,
            "mean_sheet_consistency": weighted_sheet, "min_sheet_consistency": min(row[2] for row in selected),
            "normal_angle_p90_deg": 10.0,
        }},
        "intrinsic": {"n_violations": 0, "n_collapsed": 0},
    }


def self_test() -> None:
    """Exercise success, deterministic bootstrap, and refusal of unmatched IDs."""
    baseline_rows = [
        ("band-seed-a", 50, 0.60, 0.0), ("band-seed-b", 30, 0.55, 0.0),
        ("legacy-a", 40, 0.50, 0.0), ("legacy-b", 20, 0.45, 0.0),
        ("all-seen", 0, 0.0, 0.0),
    ]
    treatment_rows = [
        ("band-seed-a", 50, 0.63, 0.0), ("band-seed-b", 30, 0.58, 0.0),
        ("legacy-a", 40, 0.60, 0.0), ("legacy-b", 20, 0.55, 0.0),
        ("all-seen", 0, 0.0, 0.0),
    ]
    with tempfile.TemporaryDirectory(prefix="sealed-patch-comparator-") as temporary:
        root = Path(temporary)
        baseline_path = root / "baseline.json"
        treatment_path = root / "treatment.json"
        baseline_path.write_text(json.dumps(_synthetic_report(baseline_rows, within_tau=0.80)), encoding="utf-8")
        treatment_path.write_text(json.dumps(_synthetic_report(treatment_rows, within_tau=0.805)), encoding="utf-8")
        one = compare(str(baseline_path), str(treatment_path))
        two = compare(str(baseline_path), str(treatment_path))
        assert one["pass_all_frozen_gates"] is True
        assert one["bootstrap"] == two["bootstrap"]
        assert one["paired_population"]["n_patches"] == 4
        assert one["unseen_sheet_consistency"]["treatment_minus_baseline_macro"] > 0
        bad = json.loads(treatment_path.read_text(encoding="utf-8"))
        bad["heldout_patches"] = bad["heldout_patches"][:-1]
        # Keep aggregate invalid only after ID failure is checked: IDs must not
        # silently intersect regardless of the aggregate's numeric contents.
        treatment_path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            compare(str(baseline_path), str(treatment_path))
        except ComparisonError as exc:
            assert "identical held-out patch IDs" in str(exc)
        else:  # pragma: no cover - hard failure if the test expectation changes
            raise AssertionError("unmatched patch IDs were accepted")
    print("self-test passed: paired, stratified 20,000-draw sealed comparator")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", nargs="?", help="baseline SpiralCheck report.json")
    parser.add_argument("treatment", nargs="?", help="treatment SpiralCheck report.json")
    parser.add_argument("--output", help="write machine-readable comparison JSON here")
    parser.add_argument("--self-test", action="store_true", help="run synthetic sealed-contract test and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        if args.baseline or args.treatment or args.output:
            parser.error("--self-test cannot be combined with reports or --output")
        self_test()
        return 0
    if not args.baseline or not args.treatment or not args.output:
        parser.error("baseline, treatment, and --output are required (or use --self-test)")
    try:
        result = compare(args.baseline, args.treatment)
    except ComparisonError as exc:
        parser.exit(2, f"sealed comparison refused: {exc}\n")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "PASS" if result["pass_all_frozen_gates"] else "FAIL"
    print(f"sealed frozen gates: {status}")
    print(f"primary treatment - baseline: {result['unseen_sheet_consistency']['treatment_minus_baseline_macro']:+.6f}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
