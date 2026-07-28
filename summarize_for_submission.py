"""Emit the submission-ready numbers straight from the evidence.

Every count in this project that was typed by hand went stale or wrong at least once:
six -> ten -> eleven -> twelve scrolls, a "nine scrolls have public CT" claim that was
really three, and a README table that twice disagreed with the reports behind it. So
the sentences that go into the form and the Discord post are generated too.

Usage:
    python summarize_for_submission.py [--results vesuvius-repro/results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

VOXELS = 128 ** 3
HERE = Path(__file__).resolve().parent


def _best_variant(variants_dir: Path, scroll: str):
    """Return the highest-Dice tagged run for ``scroll``, if present."""
    best = None
    if not variants_dir.is_dir():
        return None
    for path in sorted(variants_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("scroll") != scroll or "dice" not in row:
            continue
        if best is None or row["dice"] > best["dice"]:
            best = row
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--catalog", default=str(HERE / "catalog.json"))
    args = ap.parse_args()

    rows, new_preds = [], []
    for p in sorted(Path(args.results).glob("*.json")):
        if p.stem == "regime_calibration":
            continue
        r = json.loads(p.read_text())
        if p.stem.endswith("_NEW_surface-m7"):
            new_preds.append(r)
        elif "dice" in r and "anomalous" not in p.stem:
            rows.append(r)

    best: dict[str, dict] = {}
    for r in rows:
        s = r["scroll"]
        if s not in best or r["dice"] > best[s]["dice"]:
            best[s] = r
    ok = sorted((r for r in best.values() if r["dice"] > 0.99),
                key=lambda r: -r["dice"])
    bad = sorted((r for r in best.values() if r["dice"] <= 0.99),
                 key=lambda r: r["dice"])

    with open(args.catalog, encoding="utf-8") as f:
        cat = json.load(f)
    m7_artifacts = [c for c in cat
                    if c.get("status") == "verifiable"
                    and "m7" in c.get("model", "")]
    m7_scrolls = {c["scroll"] for c in m7_artifacts}

    l0 = sum(1 for r in ok if r["ct_level"] == 0)
    l2 = sum(1 for r in ok if r["ct_level"] == 2)
    worst, bestd = min(r["dice"] for r in ok), max(r["dice"] for r in ok)
    exact = [r for r in ok if r["disagreeing_voxels"] == 0]
    covered = len(ok) + len(bad)

    print("=" * 72)
    print("NUMBERS (regenerate this rather than retyping any of them)")
    print("=" * 72)
    print(f"published m7 artifacts          : {len(m7_artifacts)}")
    print(f"scrolls carrying those artifacts: {len(m7_scrolls)}")
    print(f"regional spot-checks            : {covered}  "
          f"({100 * covered / max(len(m7_scrolls), 1):.0f}% of scrolls; "
          "one artifact per scroll)")
    print(f"TTA-off region Dice > 0.99      : {len(ok)}   L0={l0}  L2={l2}")
    print(f"Dice range among those          : {worst:.4f} - {bestd:.4f}")
    print(f"exact, zero differing voxels    : {len(exact)}"
          + (f"  ({exact[0]['scroll']})" if exact else ""))
    print(f"TTA-off region Dice <= 0.99     : {len(bad)}"
          + (f"   {[r['scroll'] for r in bad]}" if bad else ""))
    if new_preds:
        print(f"new predictions produced        : {len(new_preds)}   "
              f"{[r['scroll'] for r in new_preds]}")

    # A run with zero differing voxels has an empty near-threshold dict; it satisfies
    # "all disagreement is near the threshold" vacuously rather than failing it.
    near = [r for r in ok
            if r["disagreeing_voxels"] == 0
            or r.get("disagreement_near_threshold", {}).get("0.01", 0) >= 0.999]
    print(f"matching regions with all disagreement within 0.01 of threshold: "
          f"{len(near)}/{len(ok)}"
          + ("" if len(near) == len(ok) else
             f"   <-- CHECK {[r['scroll'] for r in ok if r not in near]}"))
    if len(near) != len(ok):
        print("  WARNING: the generated sentence below claims all matching regions - do not use "
              "it until this reads N/N.")

    print()
    print("=" * 72)
    print("SENTENCE FOR THE FORM / POST")
    print("=" * 72)
    tail = ""
    if bad:
        b = bad[0]
        tail = (f" With TTA off, {b['scroll']} instead scores Dice {b['dice']:.4f} "
                f"({b['disagreeing_voxels']:,} voxels, "
                f"{100 * b['disagreeing_voxels'] / VOXELS:.2f}%).")
        variant = _best_variant(Path(args.results) / "variants", b["scroll"])
        if variant is not None:
            tail += (f" Its tagged TTA run scores {variant['dice']:.4f}; that audit "
                     "exposed missing configuration provenance, which #1253 now "
                     "records and maintainers backfilled on existing artifacts.")
        else:
            tail += " WARNING: no tagged TTA variant was found; do not ship this sentence."
    exact_txt = (f" {exact[0]['scroll']} came back exact, zero differing voxels."
                 if exact else "")
    print(
        f"A selected 256^3 surface-containing region (central 128^3 scored) from one "
        f"m7 artifact on each of {covered} scrolls was checked against public inputs. "
        f"With TTA off, {len(ok)} regions -- {l0} at CT level 0 and {l2} at level 2 -- "
        f"match at Dice {worst:.4f} to {bestd:.4f}; all their disagreement lies within "
        f"0.01 of the 0.2 threshold, consistent with numerical boundary residue."
        f"{exact_txt}{tail}"
    )
    if covered < len(m7_scrolls):
        print(f"\n  NOTE: {len(m7_scrolls) - covered} scroll(s) not yet spot-checked.")
    else:
        remaining = len(m7_artifacts) - covered
        print(f"\n  Scroll coverage is complete, but this is regional evidence, not a "
              f"full-volume comparison. {remaining} additional duplicate-scroll m7 "
              "artifact(s) are outside the one-artifact-per-scroll baseline.")


if __name__ == "__main__":
    main()
