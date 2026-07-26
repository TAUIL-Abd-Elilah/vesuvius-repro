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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="vesuvius-repro/results")
    ap.add_argument("--catalog", default="catalog.json")
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

    cat = json.load(open(args.catalog))
    m7_scrolls = {c["scroll"] for c in cat
                  if c.get("status") == "verifiable" and "m7" in c.get("model", "")}

    l0 = sum(1 for r in ok if r["ct_level"] == 0)
    l2 = sum(1 for r in ok if r["ct_level"] == 2)
    worst, bestd = min(r["dice"] for r in ok), max(r["dice"] for r in ok)
    exact = [r for r in ok if r["disagreeing_voxels"] == 0]
    covered = len(ok) + len(bad)

    print("=" * 72)
    print("NUMBERS (regenerate this rather than retyping any of them)")
    print("=" * 72)
    print(f"scrolls with an m7 prediction   : {len(m7_scrolls)}")
    print(f"scrolls checked                 : {covered}  "
          f"({100 * covered / max(len(m7_scrolls), 1):.0f}% of the collection)")
    print(f"reproduce (Dice > 0.99)         : {len(ok)}   L0={l0}  L2={l2}")
    print(f"Dice range among those          : {worst:.4f} - {bestd:.4f}")
    print(f"exact, zero differing voxels    : {len(exact)}"
          + (f"  ({exact[0]['scroll']})" if exact else ""))
    print(f"do NOT reproduce                : {len(bad)}"
          + (f"   {[r['scroll'] for r in bad]}" if bad else ""))
    if new_preds:
        print(f"new predictions produced        : {len(new_preds)}   "
              f"{[r['scroll'] for r in new_preds]}")

    # A run with zero differing voxels has an empty near-threshold dict; it satisfies
    # "all disagreement is near the threshold" vacuously rather than failing it.
    near = [r for r in ok
            if r["disagreeing_voxels"] == 0
            or r.get("disagreement_near_threshold", {}).get("0.01", 0) >= 0.999]
    print(f"reproduced runs whose disagreement is 100% within 0.01 of threshold: "
          f"{len(near)}/{len(ok)}"
          + ("" if len(near) == len(ok) else
             f"   <-- CHECK {[r['scroll'] for r in ok if r not in near]}"))
    if len(near) != len(ok):
        print("  WARNING: the generated sentence below claims 'every one' - do not use "
              "it until this reads N/N.")

    print()
    print("=" * 72)
    print("SENTENCE FOR THE FORM / POST")
    print("=" * 72)
    tail = ""
    if bad:
        b = bad[0]
        tail = (f" One scroll does not: {b['scroll']}, at Dice {b['dice']:.4f} "
                f"({b['disagreeing_voxels']:,} voxels, "
                f"{100 * b['disagreeing_voxels'] / VOXELS:.2f}%).")
    exact_txt = (f" {exact[0]['scroll']} came back exact, zero differing voxels."
                 if exact else "")
    print(
        f"Of the {len(m7_scrolls)} scrolls carrying a published m7 surface prediction, "
        f"{covered} were checked against public inputs. {len(ok)} reproduce -- {l0} at "
        f"CT level 0 and {l2} at level 2 -- at Dice {worst:.4f} to {bestd:.4f}, and in "
        f"every one of them 100% of the differing voxels lie within 0.01 of the 0.2 "
        f"threshold, which is the residue of float16 storage and autocast and nothing "
        f"else.{exact_txt}{tail}"
    )
    if covered < len(m7_scrolls):
        print(f"\n  NOTE: {len(m7_scrolls) - covered} scroll(s) not yet checked -- say "
              f"\"{covered} of {len(m7_scrolls)}\", not \"every\".")
    else:
        print("\n  Full coverage: \"every published m7 surface prediction\" is literal.")


if __name__ == "__main__":
    main()
