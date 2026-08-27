#!/usr/bin/env python3
"""Summarize the three frozen sealed family-cap replications.

The inputs are the three machine-readable ``sealed_comparison.json`` files
from compare_sealed_patch_reports.py, keyed explicitly by optimizer seed. This
script does not pool bootstrap draws or calculate a cross-seed p-value: seeds
23 and 101 are training replications on the *same* held-out split, not new
independent held-out datasets.

Example
-------
python summarize_sealed_replicates.py \
  --seed17 <seed17-comparison.json> \
  --seed23 <seed23-comparison.json> \
  --seed101 <seed101-comparison.json> \
  --output sealed_replication_summary.json
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


SEEDS = (17, 23, 101)
EXPECTED_SCHEMA = "sealed-phercparis4-patch-sampling-comparison-v1"
EXPECTED_VILLA_COMMIT = "17dad916c79266f6a19f76abc507bb8b95c63a9b"
EXPECTED_SPIRALCHECK_COMMIT = "d1b50e2957409a870225fb9f5dcc5e25f7a0f9da"
EXPECTED_SPLIT_MANIFEST_SHA256 = "9a1b226ebde3854728adfeb6f21513026c0cc49d948cc52684d6ee96e3819f31"
PRIMARY_GATE = "primary_positive_and_95ci_above_zero"
BAND_GATE = "band_seed_noninferior_within_3pp"
POOLED_GATE = "pooled_unseen_within_tau_noninferior_within_1pp"


class SummaryError(ValueError):
    """An input comparison cannot support the frozen replication statement."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{where}: expected a finite numeric value, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise SummaryError(f"{where}: expected a finite numeric value, got {value!r}")
    return out


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SummaryError(f"{where}: expected an object")
    return value


def _get(document: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    value: Any = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _first_present(document: dict[str, Any], paths: tuple[tuple[str, ...], ...]) -> tuple[str | None, Any | None]:
    found: list[tuple[str, Any]] = []
    for path in paths:
        value = _get(document, path)
        if value is not None:
            found.append((".".join(path), value))
    if not found:
        return None, None
    # Multiple canonical copies must agree. This catches a stale run manifest
    # embedded alongside a newer top-level provenance field.
    _, first = found[0]
    if any(value != first for _, value in found[1:]):
        detail = ", ".join(f"{path}={value!r}" for path, value in found)
        raise SummaryError(f"conflicting embedded provenance: {detail}")
    return found[0]


def _optional_provenance(document: dict[str, Any], *, seed: int, protocol_hashes: dict[str, str]) -> dict[str, Any]:
    """Verify provenance only when a comparison embeds it.

    The original comparison format contains report hashes and SpiralCheck meta,
    but not the fit-run manifest. Future runner versions may embed the fields
    below. Absence is recorded as ``not_embedded`` rather than guessed.
    """
    expected: dict[str, tuple[Any, tuple[tuple[str, ...], ...]]] = {
        "optimizer_seed": (seed, (
            ("optimizer_seed",), ("provenance", "optimizer_seed"),
            ("provenance", "frozen_parameters", "optimizer_seed"),
            ("provenance", "run_manifest", "frozen_parameters", "optimizer_seed"),
        )),
        "villa_commit": (EXPECTED_VILLA_COMMIT, (
            ("provenance", "villa_commit"), ("provenance", "code", "villa_commit"),
            ("provenance", "run_manifest", "paths", "villa_commit"),
        )),
        "spiralcheck_commit": (EXPECTED_SPIRALCHECK_COMMIT, (
            ("provenance", "spiralcheck_commit"), ("provenance", "code", "spiralcheck_commit"),
            ("provenance", "run_manifest", "paths", "spiralcheck_commit"),
        )),
        "split_manifest_sha256": (EXPECTED_SPLIT_MANIFEST_SHA256, (
            ("provenance", "split_manifest_sha256"), ("provenance", "manifest_sha256"),
            ("provenance", "run_manifest", "file_sha256", "manifest"),
        )),
        "base_protocol_sha256": (protocol_hashes["base"], (
            ("protocol", "sha256"), ("provenance", "base_protocol_sha256"),
        )),
        "replication_protocol_sha256": (protocol_hashes["replication"], (
            ("provenance", "replication_protocol_sha256"),
            ("provenance", "replication_protocol", "sha256"),
        )),
    }
    verified: dict[str, Any] = {}
    for label, (expected_value, paths) in expected.items():
        path, actual = _first_present(document, paths)
        if path is None:
            verified[label] = {"status": "not_embedded", "expected": expected_value}
        elif actual != expected_value:
            raise SummaryError(
                f"seed {seed}: provenance {path}={actual!r}, expected {expected_value!r}"
            )
        else:
            verified[label] = {"status": "verified", "path": path, "value": actual}
    # Current comparisons always name their base protocol. Check the name if
    # present, independently from any optional hash.
    protocol_path = _get(document, ("protocol", "path"))
    if protocol_path is not None and Path(str(protocol_path)).name != "SEALED_PATCH_PROTOCOL.md":
        raise SummaryError(
            f"seed {seed}: protocol.path={protocol_path!r}; expected SEALED_PATCH_PROTOCOL.md"
        )
    return verified


def _load(path_arg: str, *, seed: int, protocol_hashes: dict[str, str]) -> dict[str, Any]:
    path = Path(path_arg).expanduser().resolve()
    if not path.is_file():
        raise SummaryError(f"seed {seed}: comparison file does not exist: {path}")
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SummaryError(f"seed {seed}: invalid JSON in {path}: {exc}") from exc
    document = _mapping(document, f"seed {seed}: comparison root")
    if document.get("schema") != EXPECTED_SCHEMA:
        raise SummaryError(
            f"seed {seed}: schema={document.get('schema')!r}; expected {EXPECTED_SCHEMA!r}"
        )
    protocol = _mapping(document.get("protocol"), f"seed {seed}: protocol")
    if protocol.get("frozen_primary") != "equal-family macro of point-weighted per-patch unseen sheet_consistency":
        raise SummaryError(f"seed {seed}: comparison does not declare the frozen primary metric")
    primary = _finite(
        _get(document, ("unseen_sheet_consistency", "treatment_minus_baseline_macro")),
        f"seed {seed}: primary delta",
    )
    ci = _get(document, ("bootstrap", "primary_delta_treatment_minus_baseline_ci"))
    if not isinstance(ci, list) or len(ci) != 2:
        raise SummaryError(f"seed {seed}: primary bootstrap CI must be a two-value list")
    ci_low = _finite(ci[0], f"seed {seed}: primary CI lower")
    ci_high = _finite(ci[1], f"seed {seed}: primary CI upper")
    if ci_low > ci_high:
        raise SummaryError(f"seed {seed}: primary bootstrap CI is reversed: {ci!r}")
    bootstrap = _mapping(document.get("bootstrap"), f"seed {seed}: bootstrap")
    if bootstrap.get("draws") != 20_000 or bootstrap.get("seed") != 20260827:
        raise SummaryError(
            f"seed {seed}: bootstrap must be 20,000 draws with seed 20260827; "
            f"got draws={bootstrap.get('draws')!r}, seed={bootstrap.get('seed')!r}"
        )
    band_delta = _finite(
        _get(document, ("unseen_sheet_consistency", "treatment_minus_baseline_by_family", "band-seed")),
        f"seed {seed}: band-seed delta",
    )
    pooled_delta = _finite(
        _get(document, ("unseen_aggregate", "treatment_minus_baseline", "frac_within_tau")),
        f"seed {seed}: pooled unseen within-tau delta",
    )
    gates = _mapping(document.get("gates"), f"seed {seed}: gates")
    gate_rows: dict[str, dict[str, Any]] = {}
    for gate_name in (PRIMARY_GATE, BAND_GATE, POOLED_GATE):
        row = _mapping(gates.get(gate_name), f"seed {seed}: gates.{gate_name}")
        if not isinstance(row.get("pass"), bool):
            raise SummaryError(f"seed {seed}: gates.{gate_name}.pass must be boolean")
        gate_rows[gate_name] = row
    # Recompute the two noninferiority checks from the report's metric values.
    # A gate flag that disagrees is evidence of a mismatched/edited report.
    if gate_rows[BAND_GATE]["pass"] != (band_delta >= -0.03):
        raise SummaryError(f"seed {seed}: reported band gate disagrees with band delta {band_delta}")
    if gate_rows[POOLED_GATE]["pass"] != (pooled_delta >= -0.01):
        raise SummaryError(f"seed {seed}: reported pooled gate disagrees with pooled delta {pooled_delta}")
    expected_primary_gate = primary > 0.0 and ci_low > 0.0
    if gate_rows[PRIMARY_GATE]["pass"] != expected_primary_gate:
        raise SummaryError(f"seed {seed}: reported primary gate disagrees with primary delta/CI")
    return {
        "seed": seed,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "primary_delta": primary,
        "primary_ci_95": [ci_low, ci_high],
        "band_seed_delta": band_delta,
        "pooled_unseen_within_tau_delta": pooled_delta,
        "frozen_gates": {name: gate_rows[name]["pass"] for name in (PRIMARY_GATE, BAND_GATE, POOLED_GATE)},
        "provenance": _optional_provenance(document, seed=seed, protocol_hashes=protocol_hashes),
    }


def _mean_range(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values), "full_range": max(values) - min(values)}


def summarize(paths: dict[int, str], *, evidence_dir: Path) -> dict[str, Any]:
    base_protocol = evidence_dir / "SEALED_PATCH_PROTOCOL.md"
    replication_protocol = evidence_dir / "SEALED_REPLICATION_PROTOCOL.md"
    if not base_protocol.is_file() or not replication_protocol.is_file():
        raise SummaryError("both SEALED_PATCH_PROTOCOL.md and SEALED_REPLICATION_PROTOCOL.md must exist beside this script")
    protocol_hashes = {"base": _sha256(base_protocol), "replication": _sha256(replication_protocol)}
    rows = [_load(paths[seed], seed=seed, protocol_hashes=protocol_hashes) for seed in SEEDS]
    all_primary_positive = all(row["primary_delta"] > 0.0 for row in rows)
    all_band_noninferior = all(row["frozen_gates"][BAND_GATE] for row in rows)
    all_pooled_noninferior = all(row["frozen_gates"][POOLED_GATE] for row in rows)
    return {
        "schema": "sealed-phercparis4-patch-sampling-replication-summary-v1",
        "replication_protocol": {
            "path": str(replication_protocol),
            "sha256": protocol_hashes["replication"],
            "base_protocol_path": str(base_protocol),
            "base_protocol_sha256": protocol_hashes["base"],
            "required_optimizer_seeds": list(SEEDS),
        },
        "per_seed": rows,
        "mean_and_full_range_across_seeds": {
            "primary_delta": _mean_range(rows, "primary_delta"),
            "band_seed_delta": _mean_range(rows, "band_seed_delta"),
            "pooled_unseen_within_tau_delta": _mean_range(rows, "pooled_unseen_within_tau_delta"),
        },
        "replication_rule": {
            "all_three_primary_point_estimates_positive": all_primary_positive,
            "all_three_band_seed_noninferiority_gates_pass": all_band_noninferior,
            "all_three_pooled_unseen_within_tau_noninferiority_gates_pass": all_pooled_noninferior,
            "pass": all_primary_positive and all_band_noninferior and all_pooled_noninferior,
        },
        "interpretation": (
            "No cross-seed significance test or pooled confidence interval is reported. "
            "Seeds 23 and 101 are training replications evaluated on the same sealed held-out split; "
            "seed 17 remains the sole confirmatory test under the original protocol."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _comparison(*, primary: float, ci: tuple[float, float], band: float, pooled: float, optimizer_seed: int | None = None) -> dict[str, Any]:
    provenance: dict[str, Any] = {}
    if optimizer_seed is not None:
        provenance = {
            "optimizer_seed": optimizer_seed,
            "villa_commit": EXPECTED_VILLA_COMMIT,
            "spiralcheck_commit": EXPECTED_SPIRALCHECK_COMMIT,
            "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        }
    return {
        "schema": EXPECTED_SCHEMA,
        "protocol": {
            "path": "SEALED_PATCH_PROTOCOL.md",
            "frozen_primary": "equal-family macro of point-weighted per-patch unseen sheet_consistency",
        },
        "provenance": provenance,
        "bootstrap": {"draws": 20_000, "seed": 20260827, "primary_delta_treatment_minus_baseline_ci": list(ci)},
        "unseen_sheet_consistency": {
            "treatment_minus_baseline_macro": primary,
            "treatment_minus_baseline_by_family": {"band-seed": band, "other": primary},
        },
        "unseen_aggregate": {"treatment_minus_baseline": {"frac_within_tau": pooled}},
        "gates": {
            PRIMARY_GATE: {"pass": primary > 0 and ci[0] > 0},
            BAND_GATE: {"pass": band >= -0.03},
            POOLED_GATE: {"pass": pooled >= -0.01},
        },
    }


def self_test() -> None:
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="sealed-replication-summary-") as temp:
        root = Path(temp)
        files: dict[int, str] = {}
        for seed, primary, ci in ((17, 0.04, (0.01, 0.07)), (23, 0.02, (-0.01, 0.05)), (101, 0.01, (-0.02, 0.04))):
            path = root / f"seed{seed}.json"
            path.write_text(json.dumps(_comparison(primary=primary, ci=ci, band=-0.01, pooled=0.0, optimizer_seed=seed)), encoding="utf-8")
            files[seed] = str(path)
        result = summarize(files, evidence_dir=here)
        assert result["replication_rule"]["pass"] is True
        # The later replication intervals crossing zero do not change the
        # declared replication criterion and are not combined into a new test.
        assert result["per_seed"][1]["frozen_gates"][PRIMARY_GATE] is False
        assert result["mean_and_full_range_across_seeds"]["primary_delta"]["min"] == 0.01
        broken = json.loads(Path(files[23]).read_text(encoding="utf-8"))
        broken["provenance"]["optimizer_seed"] = 101
        Path(files[23]).write_text(json.dumps(broken), encoding="utf-8")
        try:
            summarize(files, evidence_dir=here)
        except SummaryError as exc:
            assert "optimizer_seed" in str(exc)
        else:  # pragma: no cover - regression tripwire
            raise AssertionError("mismatched embedded optimizer seed was accepted")
    print("self-test passed: strict three-seed sealed replication summary")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed17", help="seed-17 sealed_comparison.json")
    parser.add_argument("--seed23", help="seed-23 sealed_comparison.json")
    parser.add_argument("--seed101", help="seed-101 sealed_comparison.json")
    parser.add_argument("--output", help="new output JSON file")
    parser.add_argument("--self-test", action="store_true", help="run synthetic test and exit")
    args = parser.parse_args(argv)
    if args.self_test:
        if any((args.seed17, args.seed23, args.seed101, args.output)):
            parser.error("--self-test cannot be combined with input/output arguments")
        self_test()
        return 0
    if not all((args.seed17, args.seed23, args.seed101, args.output)):
        parser.error("--seed17, --seed23, --seed101, and --output are required")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing output: {output}")
    try:
        result = summarize({17: args.seed17, 23: args.seed23, 101: args.seed101}, evidence_dir=Path(__file__).resolve().parent)
    except SummaryError as exc:
        parser.exit(2, f"sealed replication summary refused: {exc}\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status = "PASS" if result["replication_rule"]["pass"] else "FAIL"
    print(f"sealed replication rule: {status}")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
