#!/usr/bin/env python3
"""Create a hash-bound comparison JSON from two evaluator and patch outputs.

This script intentionally has no embedded result values. It follows the primary
development-screen bootstrap described in PREREGISTERED_SCREEN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_bootstrap(baseline: list[dict], treatment: list[dict]) -> dict:
    base = {row["stable_id"]: row for row in baseline}
    trial = {row["stable_id"]: row for row in treatment}
    if set(base) != set(trial):
        raise ValueError("held-out fiber IDs differ")
    ids = sorted(base)
    b_sat = np.asarray([base[key]["satisfied_points"] for key in ids], float)
    t_sat = np.asarray([trial[key]["satisfied_points"] for key in ids], float)
    total = np.asarray([base[key]["total_points"] for key in ids], float)
    if not np.array_equal(total, np.asarray([trial[key]["total_points"] for key in ids], float)):
        raise ValueError("held-out fiber point counts differ")
    rng = np.random.default_rng(20260827)
    samples = rng.integers(0, len(ids), size=(20_000, len(ids)))
    deltas = (t_sat[samples].sum(1) - b_sat[samples].sum(1)) / total[samples].sum(1)
    return {"paired_fibers": len(ids), "baseline_fraction": float(b_sat.sum() / total.sum()), "treatment_fraction": float(t_sat.sum() / total.sum()), "absolute_delta": float((t_sat.sum() - b_sat.sum()) / total.sum()), "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))], "seed": 20260827, "replicates": 20_000}


def family(rows: list[dict], band: bool) -> float | None:
    selected = [row for row in rows if str(row["id"]).startswith("band-seed") is band]
    total = sum(float(row["total_area"]) for row in selected)
    return sum(float(row["satisfied_area"]) for row in selected) / total if total else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-holdout", type=Path, required=True)
    parser.add_argument("--treatment-holdout", type=Path, required=True)
    parser.add_argument("--baseline-patches", type=Path, required=True)
    parser.add_argument("--treatment-patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    b_hold, t_hold = load(args.baseline_holdout), load(args.treatment_holdout)
    b_rows, t_rows = load(args.baseline_patches)["patches"], load(args.treatment_patches)["patches"]
    paired = paired_bootstrap(b_hold["fibers"], t_hold["fibers"])
    b_band, t_band = family(b_rows, True), family(t_rows, True)
    gate = {"heldout_delta_at_least_0.02": paired["absolute_delta"] >= .02, "heldout_ci_above_zero": paired["ci95"][0] > 0, "band_seed_noninferior_within_0.01": b_band is not None and t_band is not None and t_band >= b_band - .01}
    gate["passed"] = all(gate.values())
    output = {"protocol": "PREREGISTERED_SCREEN.md", "inputs": {str(path): digest(path) for path in (args.baseline_holdout, args.treatment_holdout, args.baseline_patches, args.treatment_patches)}, "heldout": paired, "band_seed_satisfied_area": {"baseline": b_band, "treatment": t_band}, "success_gate": gate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
