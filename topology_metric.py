"""⛔ RESULT, 2026-08-07: there is nothing here to fix, and one metric below is degenerate.

Kept as a record. Two findings, in order of importance.

**1. m7 is already topologically clean on the 892-volume public set.** Predicted connected
components equal labelled components exactly (3/3, 4/4, 6/6, 7/7, 6/6 on the first five), island
fraction 0, fragmentation 0, recall 0.982, missed sheet 0.8-3.0%. Applying the Kaggle winner's
recipe (closing r=3, drop small components) at a matched predicted-positive budget only cost
recall, 0.982 -> 0.834. **The $100k competition's post-processing has nothing to repair here**,
because the defects it targets live in hard regions this dataset does not contain. That is the
fifth independent study to die on that same fact.

**2. ⚠ `hole_frac` below is DEGENERATE and always returns 0.** It calls `binary_fill_holes` and
asks what cavities appear. A papyrus sheet is an OPEN surface: it encloses no 3D cavity, so
nothing is ever filled. Holes in a surface are 2D gaps within the manifold and a 3D cavity
filler cannot see them. Measuring that correctly needs an in-manifold operation, not this. The
metric was wrong by construction and the all-zero column is what exposed it.

Original docstring follows.

---

Topology defects in a predicted surface, and whether post-processing fixes them.

WHY THIS EXISTS. The team ran a **$100,000 Kaggle competition** (Surface Detection, closed
6 Feb 2026) judged on a *topology-aware* blend — voxel accuracy **and** surface connectivity,
explicitly "no gaps, holes, sheet-switches or mergers". The winning solutions were nnU-Net
ensembles plus post-processing; the 2nd-place writeup is titled "a postprocessing win", and the
1st-place recipe was binary closing with a radius-3 ball, dropping components under 20K voxels,
and patching holes via a height-map.

**villa's inference applies none of it.** `models/run/finalize_outputs.py` is softmax, threshold,
uint8 — nothing else. ScrollFiesta's `pred_reject.c` only rejects whole solid-slab cubes and
deliberately keeps "dense-but-thin real tangles". And the published m7 artifacts are already
thresholded (`th0.2` in their filenames), so these are binary masks going downstream with no
topology cleanup at all.

So this measures the defects, applies the public recipe, and asks whether it helps — on
predictions regenerated under the correct CT normalization (villa#1364), because everything
measured before that was wrong.

⚠ THE CONTROL THAT MAKES OR BREAKS THIS. Closing a mask makes it bigger, and a bigger mask has
fewer holes for free. So every arm is also scored at a **matched predicted-positive budget**:
after post-processing, the threshold is re-chosen per volume so the arm spends what the baseline
spent. A topology gain that survives matched spend is real; one that does not is just dilation.

Defects measured, chosen because bet 4's gate showed mergers are rare here (median 1 close
component pair per volume, 43% of volumes with none), so holes and fragmentation are what is
actually available to measure:

  hole_frac      labelled sheet that is NOT predicted but sits enclosed by prediction —
                 a gap in the surface rather than its edge
  island_frac    predicted volume sitting in small components with no labelled sheet in them
  frag           predicted components overlapping one labelled component, minus 1

  python topology_metric.py --n 20
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import (binary_closing, binary_dilation, binary_fill_holes,
                           generate_binary_structure, label)

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data" / "kaggle" / "labels"
CACHE = ROOT / "results" / "m7_pred_cache_ctnorm"
SIZE, TRIM = 256, 64
THRESH = 0.2


def ball(r: int) -> np.ndarray:
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (z * z + y * y + x * x) <= r * r


def topology(pred: np.ndarray, sheet: np.ndarray, scored: np.ndarray) -> dict:
    """Hole, island and fragmentation defects of `pred` against labelled `sheet`."""
    st = generate_binary_structure(3, 1)

    # HOLES: labelled sheet the prediction missed, but which is enclosed by prediction rather
    # than lying at its edge. filling the prediction's cavities and asking what that adds is
    # the cheap way to say "enclosed".
    filled = binary_fill_holes(pred)
    enclosed = filled & ~pred
    miss = sheet & ~pred
    hole = float((miss & enclosed).sum()) / max(float(sheet.sum()), 1.0)

    # ISLANDS: predicted components containing no labelled sheet at all
    comp, n = label(pred & scored, structure=st)
    isl = 0.0
    if n:
        has_sheet = np.zeros(n + 1, dtype=bool)
        has_sheet[np.unique(comp[sheet & (comp > 0)])] = True
        sizes = np.bincount(comp.ravel(), minlength=n + 1)
        bad = [i for i in range(1, n + 1) if not has_sheet[i]]
        isl = float(sizes[bad].sum()) / max(float((pred & scored).sum()), 1.0)

    # FRAGMENTATION: how many predicted pieces cover one labelled component
    lcomp, ln = label(sheet, structure=st)
    frags = []
    for i in range(1, min(ln, 40) + 1):
        m = lcomp == i
        if m.sum() < 200:
            continue
        ids = np.unique(comp[m & (comp > 0)])
        frags.append(max(len(ids) - 1, 0))
    return {"hole_frac": hole, "island_frac": isl,
            "frag": float(np.mean(frags)) if frags else 0.0}


def budget_threshold(p: np.ndarray, scored: np.ndarray, budget: float) -> float:
    return float(np.quantile(p[scored], 1.0 - budget))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--close-r", type=int, default=3)
    ap.add_argument("--min-comp-frac", type=float, default=0.002,
                    help="drop predicted components smaller than this fraction of the crop")
    ap.add_argument("--out", default=str(ROOT / "results" / "topology_metric.json"))
    a = ap.parse_args()

    files = sorted(glob.glob(str(CACHE / "*.npy")))[:a.n]
    st_ball = ball(a.close_r)
    rows = []
    for f in files:
        nm = Path(f).stem
        p = np.load(f).astype(np.float32)
        full = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        off = (full.shape[0] - SIZE) // 2
        lo, hi = off + TRIM, off + SIZE - TRIM
        g = full[lo:hi, lo:hi, lo:hi]
        sheet, scored = g == 1, g != 2
        if sheet.sum() < 400:
            continue

        base = p > THRESH
        budget = float((base & scored).sum()) / max(float(scored.sum()), 1.0)

        # POST-PROCESSED, then re-thresholded to the SAME budget so closing cannot win by
        # simply predicting more.
        closed = binary_closing(base, structure=st_ball)
        comp, n = label(closed, structure=generate_binary_structure(3, 1))
        if n:
            sizes = np.bincount(comp.ravel(), minlength=n + 1)
            small = np.where(sizes < a.min_comp_frac * base.size)[0]
            closed &= ~np.isin(comp, small[small > 0])
        # matched budget: closing only ever adds, so trim back by raising the prob threshold
        extra = float((closed & scored).sum()) / max(float(scored.sum()), 1.0)
        if extra > budget:
            t2 = budget_threshold(np.where(closed, np.maximum(p, THRESH + 1e-6), 0.0),
                                  scored, budget)
            post = closed & (np.where(closed, np.maximum(p, THRESH + 1e-6), 0.0) > t2)
        else:
            post = closed

        rb = topology(base, sheet, scored)
        rp = topology(post, sheet, scored)
        rec = lambda m: float((m & sheet).sum()) / max(float(sheet.sum()), 1.0)
        rows.append({"sample": nm,
                     "base": {**rb, "recall": rec(base),
                              "predpos": float((base & scored).sum()) / float(scored.sum())},
                     "post": {**rp, "recall": rec(post),
                              "predpos": float((post & scored).sum()) / float(scored.sum())}})
        if len(rows) % 5 == 0:
            print(f"  {len(rows)}/{len(files)}", flush=True)

    def med(arm, k):
        return float(np.median([r[arm][k] for r in rows]))

    out = {"n_volumes": len(rows), "close_radius": a.close_r,
           "min_comp_frac": a.min_comp_frac, "threshold": THRESH, "rows": rows}
    print(f"\nvolumes {len(rows)}  (post-processing matched to baseline budget)")
    print(f"{'':<14}{'baseline':>12}{'post':>12}{'delta':>12}")
    for k in ("hole_frac", "island_frac", "frag", "recall", "predpos"):
        b, q = med("base", k), med("post", k)
        out[f"median_base_{k}"] = round(b, 5)
        out[f"median_post_{k}"] = round(q, 5)
        print(f"  {k:<12}{b:>12.5f}{q:>12.5f}{q-b:>+12.5f}")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
