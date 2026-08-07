"""Gate zero for PREREGISTER_diagnostic.md: does local sheet resolvability vary, and is it real?

Resolvability is the CT's own answer to "is there a sheet here that could be found", computed
without any model and without the label's geometry beyond where to sample. At a point:

    across-sheet direction  = most-negative-curvature eigenvector of the smoothed CT Hessian
    profile                 = CT sampled along that direction
    prominence              = profile peak minus its shoulders
    resolvability (CNR)     = prominence / local noise

⚠ PROMINENCE, NEVER A HALF-MAX WIDTH. `m7_thickness.py` was discarded because the predicted/CT
FWHM ratio moved 0.94 -> 0.62 purely as the profile half-width went 3 -> 8: a wider window keeps
lowering the baseline and so the half-max threshold. A width is a free parameter of the window.
Prominence is a contrast, and the shoulder is taken at a fixed fraction of the window so it is
at least explicit about where it is measured.

⚠ NOISE IS ESTIMATED FROM THE PROFILE'S SECOND DIFFERENCE, not from a box around the point. A
box near a sheet is dominated by the sheet, which would make bright regions look noisy and
depress exactly the resolvability we are trying to measure. The second difference of a smooth
profile is ~0, so its MAD reads the high-frequency component, which is what noise is.

The two registered gates (section 5):

  1. per-volume median resolvability must span at least 1.5x across its interquartile range,
     or there is nothing to stratify
  2. |Spearman(resolvability, labelled-sheet fraction)| must be <= 0.7, or the measure is a
     density proxy wearing a new name

  python resolvability.py --n 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.stats import spearmanr

from ridge_residual import across_sheet_normals, SIGMA

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"

HALF, STEP = 5.0, 0.25
SHOULDER = 0.70        # |t| beyond this fraction of the window counts as shoulder
PER_VOL = 500


def resolvability(ct: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Contrast-to-noise of the sheet ridge at each point. Model-free."""
    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    n = across_sheet_normals(ct, pts)
    sm = gaussian_filter(ct.astype(np.float32), SIGMA)
    c = (pts[:, :, None] + n[:, :, None] * ts[None, None, :]).transpose(1, 0, 2).reshape(3, -1)
    prof = map_coordinates(sm, c, order=1, mode="nearest").reshape(len(pts), len(ts))

    outer = np.abs(ts) >= SHOULDER * HALF
    peak = prof.max(axis=1)
    shoulder = np.median(prof[:, outer], axis=1)
    prominence = peak - shoulder

    # noise from the profile's second difference: ~0 for smooth structure, so its MAD reads
    # the high-frequency component. 1.4826 makes MAD a sigma estimate; /sqrt(6) undoes the
    # variance inflation of a second difference of independent samples.
    d2 = np.diff(prof, n=2, axis=1)
    mad = np.median(np.abs(d2 - np.median(d2, axis=1, keepdims=True)), axis=1)
    noise = np.maximum(1.4826 * mad / np.sqrt(6.0), 1e-3)
    return prominence / noise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "resolvability_gate.json"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    names = sorted(p.stem for p in LABELS.glob("sample_*.tif"))
    order = [names[i] for i in rng.permutation(len(names))]

    rows = []
    for nm in order:
        if len(rows) >= a.n:
            break
        lab = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        pts = np.argwhere(lab == 1)
        if len(pts) < 800:
            continue
        ct = np.asarray(tifffile.imread(str(IMAGES / f"{nm}.tif")))
        sel = pts[rng.choice(len(pts), PER_VOL, replace=False)].astype(np.float32)
        r = resolvability(ct, sel)
        r = r[np.isfinite(r)]
        if len(r) < 200:
            continue
        scored = lab != 2
        rows.append({"sample": nm,
                     "median_resolvability": round(float(np.median(r)), 4),
                     "sheet_fraction": round(float((lab == 1).sum() / max(scored.sum(), 1)), 5),
                     "n": int(len(r))})
        if len(rows) % 20 == 0:
            print(f"  {len(rows)}/{a.n}", flush=True)

    v = np.array([r["median_resolvability"] for r in rows])
    d = np.array([r["sheet_fraction"] for r in rows])
    q25, q75 = float(np.quantile(v, .25)), float(np.quantile(v, .75))
    ratio = q75 / max(q25, 1e-9)
    rho = float(spearmanr(v, d).statistic)

    g1 = ratio >= 1.5
    g2 = abs(rho) <= 0.7
    out = {"n_volumes": len(rows), "half": HALF, "shoulder_frac": SHOULDER,
           "median": round(float(np.median(v)), 4),
           "q25": round(q25, 4), "q75": round(q75, 4),
           "iqr_ratio_q75_over_q25": round(ratio, 4),
           "spearman_vs_sheet_fraction": round(rho, 4),
           "gate1_spread_ge_1.5": bool(g1),
           "gate2_not_density_proxy_abs_rho_le_0.7": bool(g2),
           "gate_passed": bool(g1 and g2),
           "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"\nvolumes {len(rows)}")
    print(f"  per-volume median resolvability (CNR): median {np.median(v):.3f}  "
          f"q25 {q25:.3f}  q75 {q75:.3f}  min {v.min():.3f}  max {v.max():.3f}")
    print(f"  GATE 1  q75/q25 = {ratio:.3f}   (need >= 1.5)   {'PASS' if g1 else 'FAIL'}")
    print(f"  GATE 2  Spearman vs sheet fraction = {rho:+.3f}   (need |rho| <= 0.7)   "
          f"{'PASS' if g2 else 'FAIL'}")
    print(f"\n  GATE ZERO {'PASSED' if g1 and g2 else 'FAILED'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
