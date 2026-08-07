"""Score any surface model on the 892-volume public set, split by provenance.

WHY THIS EXISTS. The team has asked for work that quantitatively improves or replaces core
pipeline stages. You cannot claim an improvement to a surface model without a way to measure
one, and the measuring stick has to survive two things that are easy to get wrong here:

  1. **Class 2 is the majority of a volume.** m7's own dataset.json names the label classes
     {0: background, 1: surface, 2: ignore}, and `ignore` runs ~59% of a typical volume. Fold
     it into background and precision collapses for no reason -- predictions landing in
     unscored regions get counted as false positives. It is excluded here, everywhere.
  2. **Recall at a fixed threshold confounds discrimination with calibration.** Two models
     that rank voxels identically but differ in confidence will post different recall at
     threshold 0.2. So every endpoint is also reported at a MATCHED PREDICTED-POSITIVE BUDGET,
     with the threshold picked per volume so both models spend the same amount of "sheet".
     Differences that survive that are differences in ranking, not in calibration.

THE POPULATION SPLIT IS THE POINT. @Jinhojeong's normalized-cross-correlation table locates
189 of the 892 volumes inside Scroll1A. The published m7 model behaves like two different
models across that boundary -- median recall **0.777** on the located set against **0.918**
elsewhere (855 volumes scored, `results/m7_recall/`). That gap has survived elimination of
leakage, label artifact, fused geometry and labelled-sheet density. An aggregate score over
all 892 hides it, so every endpoint here is reported per population as well as pooled.

WHAT IS KNOWN NOT TO WORK, so nobody re-runs it: m7's errors are BOUNDARY errors -- false
positives enriched 3.10x within 1 voxel of labelled sheet, 1.85x at 2, at the null by 3, and
depleted beyond. That enrichment is proximity, not a misplaced label boundary: the shell
profile decays geometrically with no step, and shell 1 sits BELOW the trend from shells 2-5
(p=3.2e-09). Ridge-based non-maximum suppression does not recover precision either -- it beats
the best global threshold at matched recall by +0.0136, and a random-direction control gets
+0.0130 of that. m7's probability field is diffuse, and its precision on the hard population is
not recoverable by any decision rule. See PREREGISTER_margin.md and m7_ridge_nms.py.

  python surface_bench.py --validate                      # null controls, no model needed
  python surface_bench.py --pred results/m7_pred_cache    # score a model's probability maps

`--pred` takes a directory of `sample_XXXXX.npy` probability volumes in [0,1]. They may be
full-size or a centred crop; the label is centre-cropped to match, so a model that only
predicts an interior region is scored fairly on that region.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent


def _find(*rel: str) -> Path:
    """Resolve a path that sits in a different place in the repo than in the working tree.

    This file is developed alongside `vesuvius-repro/` and shipped inside it, so
    `results/overlap/overlap_report.json` is one directory up in one layout and adjacent in the
    other. Hardcoding either breaks the other, and it breaks for a cloner rather than for us,
    which is the worst way round. An env var wins if set, so the data can live anywhere.
    """
    for base in (ROOT, ROOT / "vesuvius-repro", ROOT.parent):
        p = base.joinpath(*rel)
        if p.exists():
            return p
    return ROOT.joinpath(*rel)          # non-existent: callers report a clear error


# Volumes are the public 892-volume Kaggle set and are NOT in this repo. Point these anywhere
# with VESUVIUS_DATA=/path/to/kaggle (expects images/ and labels/ beneath it).
_DATA = Path(os.environ.get("VESUVIUS_DATA", "")) if os.environ.get("VESUVIUS_DATA") \
    else _find("data", "kaggle")
IMAGES = _DATA / "images"
LABELS = _DATA / "labels"
OVERLAP = _find("results", "overlap", "overlap_report.json")

THRESH = 0.2       # the published m7 operating point, from its artifact filenames (th0.2)
BUDGET = 0.12      # matched predicted-positive budget: the sheet base rate in scored regions


def centre_crop(a: np.ndarray, shape) -> np.ndarray:
    """Centre-crop `a` to `shape`. Predictions over an interior region score on that region."""
    sl = []
    for full, want in zip(a.shape, shape):
        off = (full - want) // 2
        sl.append(slice(off, off + want))
    return a[tuple(sl)]


def endpoints(p: np.ndarray, lab: np.ndarray, ct: np.ndarray | None = None) -> dict:
    """Every endpoint for one volume, at the fixed threshold and at the matched budget."""
    sheet = lab == 1
    scored = lab != 2                     # class 2 is ~59% of a volume and is NOT background
    n_scored = float(scored.sum())
    if sheet.sum() < 200 or n_scored < 1000:
        return {"status": "no_sheet"}

    out = {"status": "ok",
           "n_sheet": int(sheet.sum()),
           "base_rate": float(sheet.sum()) / n_scored}

    # Per-volume budget threshold: label-independent given the scored mask, so two models are
    # compared at the same spend rather than at the same number.
    t_budget = float(np.quantile(p[scored], 1.0 - BUDGET))
    out["budget_threshold"] = round(t_budget, 4)

    for tag, t in (("", THRESH), ("budget_", t_budget)):
        pred = p > t
        tp = float((pred & sheet).sum())
        fn = float((~pred & sheet).sum())
        fp = float((pred & scored & ~sheet).sum())
        n_pred = float((pred & scored).sum())
        out[f"{tag}recall"] = tp / max(tp + fn, 1.0)
        out[f"{tag}precision"] = tp / max(tp + fp, 1.0)
        out[f"{tag}pred_positive_fraction"] = n_pred / n_scored
        # precision expressed as a multiple of the base rate: a model predicting everything
        # scores 1.0x, so this says how much the model actually knows.
        out[f"{tag}precision_lift"] = out[f"{tag}precision"] / max(out["base_rate"], 1e-9)
        if ct is not None:
            out[f"{tag}pred_on_empty_ct"] = float((pred & (ct == 0)).sum()) / max(n_pred, 1.0)
    return out


def load_pred(path: Path) -> np.ndarray:
    a = np.load(path).astype(np.float32) if path.suffix == ".npy" else \
        np.asarray(tifffile.imread(str(path))).astype(np.float32)
    if a.max() > 1.5:                    # tolerate 0-255 maps
        a = a / 255.0
    return a


def score(pred_dir: Path, limit: int, with_ct: bool) -> dict:
    located = {r["sample"] for r in json.loads(OVERLAP.read_text())["located"]}
    files = sorted(list(pred_dir.glob("*.npy")) + list(pred_dir.glob("*.tif")))[:limit or None]
    rows = []
    for f in files:
        nm = f.stem
        lp = LABELS / f"{nm}.tif"
        if not lp.exists():
            continue
        p = load_pred(f)
        lab = np.asarray(tifffile.imread(str(lp)))
        if p.shape != lab.shape:
            lab = centre_crop(lab, p.shape)
        ct = None
        if with_ct:
            ct = centre_crop(np.asarray(tifffile.imread(str(IMAGES / f"{nm}.tif"))), p.shape)
        r = endpoints(p, lab, ct)
        r["sample"] = nm
        r["located"] = nm in located
        rows.append(r)
    return summarise(rows)


def summarise(rows: list) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    keys = [k for k in ("recall", "precision", "precision_lift", "pred_positive_fraction",
                        "budget_recall", "budget_precision", "budget_precision_lift",
                        "pred_on_empty_ct", "base_rate") if any(k in r for r in ok)]
    out = {"n_volumes": len(ok), "threshold": THRESH, "budget": BUDGET, "rows": rows}
    for tag, sel in (("all", ok),
                     ("located", [r for r in ok if r.get("located")]),
                     ("other", [r for r in ok if not r.get("located")])):
        if not sel:
            continue
        out[tag] = {"n": len(sel), **{
            f"median_{k}": round(float(np.median([r[k] for r in sel if k in r])), 5)
            for k in keys}}
    return out


def validate() -> None:
    """Null controls. No model, no GPU -- these say the harness measures what it claims.

    A benchmark whose own nulls are unknown cannot tell a real result from a bug. Two
    predictors with analytically known answers are scored on real labels:

      uniform random   recall ~= predicted-positive fraction, and precision_lift ~= 1.0,
                       because a coin flip finds sheet at exactly the base rate
      perfect          recall = 1.0 and precision_lift = 1/base_rate, the ceiling

    If the random arm does not land at a lift of 1.0, the scoring is wrong and no number
    produced by this file means anything.
    """
    names = sorted(p.stem for p in LABELS.glob("sample_*.tif"))[:8]
    rng = np.random.default_rng(0)
    print(f"{'sample':<16}{'rand lift':>11}{'rand recall':>13}{'perfect lift':>14}{'base':>8}")
    lifts, recalls = [], []
    for nm in names:
        lab = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        if (lab == 1).sum() < 200:
            continue
        r = endpoints(rng.random(lab.shape).astype(np.float32), lab)
        perfect = endpoints((lab == 1).astype(np.float32), lab)
        if r["status"] != "ok":
            continue
        lifts.append(r["precision_lift"]); recalls.append(r["recall"])
        print(f"{nm:<16}{r['precision_lift']:>11.3f}{r['recall']:>13.3f}"
              f"{perfect['precision_lift']:>14.2f}{r['base_rate']:>8.4f}")
    print(f"\n  random precision lift: median {np.median(lifts):.4f}  "
          f"(MUST be ~1.000 -- a coin flip finds sheet at the base rate)")
    print(f"  random recall: median {np.median(recalls):.4f}  "
          f"(MUST track the threshold's predicted-positive share, here 1-{THRESH} = "
          f"{1-THRESH:.2f})")
    print(f"  perfect predictor lift = 1/base_rate, the ceiling for that volume.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="directory of sample_XXXXX.npy probability volumes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-ct", action="store_true", help="skip the empty-CT guardrail")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "surface_bench.json"))
    a = ap.parse_args()

    # Preflight. The volumes are the public Kaggle set and are deliberately not vendored here,
    # so a fresh clone will not have them. Say so once, clearly, instead of failing later with
    # a FileNotFoundError on some individual sample.
    missing = [str(p) for p in (LABELS, OVERLAP) if not p.exists()]
    if missing:
        print("cannot run: missing\n  " + "\n  ".join(missing), file=sys.stderr)
        print("\nthe 892-volume set is not in this repo. point at it with:\n"
              "  VESUVIUS_DATA=/path/to/kaggle   (expects images/ and labels/ beneath)",
              file=sys.stderr)
        raise SystemExit(2)

    if a.validate:
        validate()
        return
    if not a.pred:
        ap.error("pass --pred or --validate")

    res = score(Path(a.pred), a.limit, not a.no_ct)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"scored {res['n_volumes']} volumes\n")
    hdr = ("population", "n", "recall", "prec", "lift", "predpos", "budget_recall")
    print(f"{hdr[0]:<11}{hdr[1]:>5}{hdr[2]:>9}{hdr[3]:>8}{hdr[4]:>7}{hdr[5]:>9}{hdr[6]:>15}")
    for tag in ("all", "located", "other"):
        if tag not in res:
            continue
        s = res[tag]
        print(f"{tag:<11}{s['n']:>5}{s['median_recall']:>9.4f}"
              f"{s['median_precision']:>8.4f}{s['median_precision_lift']:>7.2f}"
              f"{s['median_pred_positive_fraction']:>9.4f}{s['median_budget_recall']:>15.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
