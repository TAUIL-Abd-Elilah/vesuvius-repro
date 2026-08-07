"""Is m7's predicted sheet simply too FAT, and does thinning it beat raising the threshold?

Motivated by the result that killed bet 2. `m7_margin_fp.py` established that m7's scored
false positives are BOUNDARY errors: enriched within 2 voxels of labelled sheet (3.10, 1.85),
at the null by 3 (1.01), and depleted beyond (0.59, 0.31). m7 does not invent sheet in open
space -- it makes sheet slightly too fat.

If that is right, the probability field should contain a thinner, better sheet than any global
threshold can extract, because thresholding cannot tell a fat sheet's flank from a faint
sheet's core -- both sit at middling p. Measured on the 60 cached volumes, thresholding indeed
trades badly: 0.2 -> 0.5 buys precision 0.446 -> 0.509 and costs recall 0.774 -> 0.643.

So: non-maximum suppression along the across-sheet normal of p. Keep a predicted voxel only
where p is a local maximum across the sheet. Compare against the plain threshold sweep AT
MATCHED RECALL, which is the only fair comparison -- any thinning raises precision if you let
recall fall.

⚠ THE CONTROL, BUILT IN BEFORE THE FIRST NUMBER. Thinning by *any* rule raises precision at
matched recall if the discarded voxels are even slightly worse than average. So NMS along the
true Hessian normal is run beside NMS along a RANDOM direction per voxel. If random thinning
matches the real thing, the geometry adds nothing and the result is an artifact of thinning per
se. This is the control that the margin study lacked until it was too late.

  python m7_ridge_nms.py --n 12        # quick look
  python m7_ridge_nms.py --n 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data" / "kaggle" / "labels"
PRED_CACHE = ROOT / "results" / "m7_pred_cache"
SPLIT = ROOT / "vesuvius-repro" / "results" / "margin_split.json"

SIZE, TRIM = 256, 64          # must match m7_margin_fp.predict_m7
SIGMA = 1.0                   # smoothing before the Hessian, as in thin_labels
LOW = 0.10                    # NMS operates on everything above this
DELTA = 1.0                   # NMS comparison offset along the normal, in voxels
THRESHES = (0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def normals_at(field: np.ndarray, pts: np.ndarray, chunk: int = 300_000) -> np.ndarray:
    """Across-sheet unit vectors from the field's own Hessian (most negative curvature)."""
    sm = gaussian_filter(field.astype(np.float32), SIGMA)
    g = np.gradient(sm)
    gg = [np.gradient(g[i]) for i in range(3)]
    out = np.empty((len(pts), 3), dtype=np.float32)
    for s in range(0, len(pts), chunk):
        q = pts[s:s + chunk]
        H = np.empty((len(q), 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                H[:, i, j] = gg[i][j][q[:, 0], q[:, 1], q[:, 2]]
        H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
        _, v = np.linalg.eigh(H)
        d = v[:, :, 0]
        out[s:s + chunk] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    return out


def nms_mask(p: np.ndarray, dirs_fn, seed: int = 0) -> np.ndarray:
    """Keep voxels above LOW whose p is >= p at +-DELTA along the supplied direction."""
    pts = np.argwhere(p > LOW).astype(np.int32)
    if len(pts) == 0:
        return np.zeros_like(p, dtype=bool)
    d = dirs_fn(p, pts, seed)
    keep = np.ones(len(pts), dtype=bool)
    here = p[pts[:, 0], pts[:, 1], pts[:, 2]]
    for sign in (1.0, -1.0):
        c = (pts + sign * DELTA * d).T.astype(np.float32)
        nb = map_coordinates(p, c, order=1, mode="nearest")
        keep &= here >= nb
    out = np.zeros_like(p, dtype=bool)
    sel = pts[keep]
    out[sel[:, 0], sel[:, 1], sel[:, 2]] = True
    return out


def _hessian_dirs(p, pts, seed):
    return normals_at(p, pts)


def _random_dirs(p, pts, seed):
    """THE CONTROL: thinning along a direction that knows nothing about the sheet."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(len(pts), 3)).astype(np.float32)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)


def pr(pred: np.ndarray, sheet: np.ndarray, scored: np.ndarray) -> tuple[float, float]:
    tp = float((pred & sheet).sum())
    fp = float((pred & scored & ~sheet).sum())
    return tp / max(float(sheet.sum()), 1.0), tp / max(tp + fp, 1.0)


def precision_at_recall(p, sheet, scored, target_recall: float) -> float | None:
    """Best precision a GLOBAL THRESHOLD can reach at >= target recall.

    The honest baseline: not the published 0.2, but the whole sweep. Beating a single
    arbitrary threshold would prove nothing.
    """
    best = None
    for t in np.linspace(0.02, 0.99, 60):
        r, pc = pr(p > t, sheet, scored)
        if r >= target_recall:
            best = pc if best is None else max(best, pc)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "results" / "m7_ridge_nms.json"))
    a = ap.parse_args()

    names = sorted(q.stem for q in PRED_CACHE.glob("*.npy"))[:a.n]
    rows, t0 = [], time.time()
    for k, nm in enumerate(names):
        p = np.load(PRED_CACHE / f"{nm}.npy").astype(np.float32)
        full = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        off = (full.shape[0] - SIZE) // 2
        lo, hi = off + TRIM, off + SIZE - TRIM
        g = full[lo:hi, lo:hi, lo:hi]
        sheet, scored = g == 1, g != 2
        if sheet.sum() < 200:
            continue

        r_base, p_base = pr(p > 0.2, sheet, scored)
        row = {"sample": nm, "recall_at_0.2": r_base, "precision_at_0.2": p_base,
               "base_rate": float(sheet.sum()) / float(scored.sum())}

        for tag, fn in (("hessian", _hessian_dirs), ("random", _random_dirs)):
            m = nms_mask(p, fn)
            # NMS then thresholded: sweep to trace its own precision-recall curve
            best = None
            for t in THRESHES:
                r, pc = pr(m & (p > t), sheet, scored)
                if r < 0.05:
                    continue
                ref = precision_at_recall(p, sheet, scored, r)
                if ref is None:
                    continue
                if best is None or (pc - ref) > best["gain"]:
                    best = {"thresh": float(t), "recall": r, "precision": pc,
                            "threshold_only_precision_at_same_recall": ref,
                            "gain": pc - ref}
            row[tag] = best
        rows.append(row)
        h, rr = row.get("hessian"), row.get("random")
        print(f"  [{k+1}/{len(names)}] {nm}  hessian gain "
              f"{h['gain']:+.4f} @recall {h['recall']:.3f}   random gain {rr['gain']:+.4f}"
              f"   {time.time()-t0:.0f}s", flush=True)

    ok = [r for r in rows if r.get("hessian") and r.get("random")]
    hg = np.array([r["hessian"]["gain"] for r in ok])
    rg = np.array([r["random"]["gain"] for r in ok])
    out = {
        "n_volumes": len(ok), "low": LOW, "delta": DELTA, "sigma": SIGMA,
        "median_recall_at_0.2": round(float(np.median([r["recall_at_0.2"] for r in ok])), 4),
        "median_precision_at_0.2": round(float(np.median([r["precision_at_0.2"] for r in ok])), 4),
        "median_hessian_gain": round(float(np.median(hg)), 4),
        "median_random_gain": round(float(np.median(rg)), 4),
        "median_hessian_minus_random": round(float(np.median(hg - rg)), 4),
        "frac_volumes_hessian_beats_random": round(float(np.mean(hg > rg)), 4),
        "reading": ("gain = precision of NMS-then-threshold minus the best precision any GLOBAL "
                    "threshold reaches at the same recall. The random-direction arm is the "
                    "control: if hessian does not clearly beat random, the geometry adds "
                    "nothing and any gain is an artifact of thinning per se."),
        "rows": rows,
    }
    if len(ok) > 3:
        from scipy.stats import wilcoxon
        out["wilcoxon_hessian_vs_random_p"] = float(wilcoxon(hg, rg)[1])
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n  m7 at 0.2: recall {out['median_recall_at_0.2']} "
          f"precision {out['median_precision_at_0.2']}")
    print(f"  median gain over best-threshold-at-same-recall:")
    print(f"    hessian NMS {out['median_hessian_gain']:+.4f}")
    print(f"    random  NMS {out['median_random_gain']:+.4f}   <- control")
    print(f"    difference  {out['median_hessian_minus_random']:+.4f}  "
          f"({out['frac_volumes_hessian_beats_random']:.0%} of volumes)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
