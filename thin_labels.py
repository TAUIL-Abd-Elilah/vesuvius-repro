"""Thin surface labels toward the sheet centre, preserving position.

Registered in PREREGISTER_labelthin.md. The published labels are 3.335 voxels thick against a
~2.4 voxel sheet, with mean signed centroid drift of 0.0002 voxels -- fat, not displaced. So
this narrows each label run about its OWN centroid and never moves it onto the CT ridge:
our measurement says there is nothing systematic to snap to, and the per-voxel offsets are
sign-random and can lock onto a neighbouring sheet where the winding pitch is small.

Method, per labelled voxel:
  1. across-sheet direction from the smoothed-CT Hessian (smallest-curvature eigenvector),
     the same estimator as measure_label_drift.py
  2. walk the label along that direction to find the run it belongs to
  3. keep the voxel only if it lies within w/2 of the run's centroid

Working on runs rather than on a morphological erosion matters: erosion thins isotropically
and eats a sheet along its surface as well as across it, which would remove true sheet at
patch edges and confound the guardrail. This only ever narrows across the sheet.

  python thin_labels.py --check --n 12          # validate, no writes
  python thin_labels.py --width 3.0 --out data/kaggle/labels_w30
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
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"

SIGMA = 1.0        # CT smoothing before the Hessian, as in measure_label_drift.py
HALF = 4.0         # profile half-width in voxels
STEP = 0.25        # profile sampling step


def across_sheet_dirs(ct: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Unit vector across the sheet at each point: smallest-curvature Hessian eigenvector."""
    sm = gaussian_filter(ct.astype(np.float32), SIGMA)
    g = np.gradient(sm)
    H = np.empty((len(pts), 3, 3), dtype=np.float32)
    for i in range(3):
        gi = np.gradient(g[i])
        for j in range(3):
            H[:, i, j] = gi[j][pts[:, 0], pts[:, 1], pts[:, 2]]
    H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
    _, v = np.linalg.eigh(H)
    d = v[:, :, 0]                                   # most negative curvature = across sheet
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.maximum(n, 1e-6)


def thin_volume(ct: np.ndarray, lab: np.ndarray, width: float,
                chunk: int = 200_000) -> tuple[np.ndarray, dict]:
    """Return a thinned copy of `lab` plus before/after statistics."""
    pts = np.argwhere(lab == 1).astype(np.int32)
    out = np.zeros_like(lab)
    if len(pts) == 0:
        return out, {"n_label": 0}

    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    mid = len(ts) // 2
    lab_f = (lab == 1).astype(np.float32)

    kept = 0
    thick_before, thick_after = [], []
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        dirs = across_sheet_dirs(ct, p)
        coords = p[:, :, None].astype(np.float32) + dirs[:, :, None] * ts[None, None, :]
        flat = coords.transpose(1, 0, 2).reshape(3, -1)
        prof = map_coordinates(lab_f, flat, order=0, mode="nearest").reshape(len(p), len(ts))

        for i in range(len(p)):
            row = prof[i]
            if row[mid] < 0.5:                       # direction estimate left the label
                out[p[i, 0], p[i, 1], p[i, 2]] = 1   # keep rather than guess
                kept += 1
                continue
            lo = mid
            while lo > 0 and row[lo - 1] >= 0.5:
                lo -= 1
            hi = mid
            while hi < len(ts) - 1 and row[hi + 1] >= 0.5:
                hi += 1
            if lo == 0 or hi == len(ts) - 1:         # run exits the window: thickness unknown
                out[p[i, 0], p[i, 1], p[i, 2]] = 1
                kept += 1
                continue
            t_centroid = 0.5 * (ts[lo] + ts[hi])
            run_thick = ts[hi] - ts[lo] + STEP
            thick_before.append(float(run_thick))
            # this voxel sits at t = 0 on its own profile; keep it if the run's centre is
            # within w/2 of it. A run already thinner than w is untouched.
            if run_thick <= width or abs(0.0 - t_centroid) <= width / 2.0:
                out[p[i, 0], p[i, 1], p[i, 2]] = 1
                kept += 1
                thick_after.append(min(float(run_thick), width))

    stats = {
        "n_label": int(len(pts)),
        "n_kept": int(kept),
        "frac_kept": round(kept / max(1, len(pts)), 4),
        "mean_thickness_before": round(float(np.mean(thick_before)), 4) if thick_before else None,
        "n_runs_measured": len(thick_before),
    }
    return out, stats


def measured_thickness(ct: np.ndarray, lab: np.ndarray, n: int = 2000,
                       seed: int = 0) -> float | None:
    """Mean label run thickness across the sheet, on a random sample of labelled voxels."""
    pts = np.argwhere(lab == 1)
    if len(pts) < 50:
        return None
    rng = np.random.default_rng(seed)
    pts = pts[rng.choice(len(pts), size=min(n, len(pts)), replace=False)].astype(np.int32)
    dirs = across_sheet_dirs(ct, pts)
    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    coords = pts[:, :, None].astype(np.float32) + dirs[:, :, None] * ts[None, None, :]
    flat = coords.transpose(1, 0, 2).reshape(3, -1)
    prof = map_coordinates((lab == 1).astype(np.float32), flat, order=0,
                           mode="nearest").reshape(len(pts), len(ts))
    mid = len(ts) // 2
    thick = []
    for i in range(len(pts)):
        row = prof[i]
        if row[mid] < 0.5:
            continue
        lo = mid
        while lo > 0 and row[lo - 1] >= 0.5:
            lo -= 1
        hi = mid
        while hi < len(ts) - 1 and row[hi + 1] >= 0.5:
            hi += 1
        if lo == 0 or hi == len(ts) - 1:
            continue
        thick.append(float(ts[hi] - ts[lo] + STEP))
    return float(np.mean(thick)) if len(thick) >= 50 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=float, default=3.0)
    ap.add_argument("--check", action="store_true",
                    help="validate on a sample and write nothing")
    ap.add_argument("--n", type=int, default=12, help="volumes to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=str(ROOT / "results" / "thin_labels_check.json"))
    a = ap.parse_args()

    samples = sorted(p.name for p in LABELS.glob("sample_*.tif"))
    rng = np.random.default_rng(a.seed)

    if a.check:
        pick = [samples[i] for i in rng.choice(len(samples), size=min(a.n, len(samples)),
                                               replace=False)]
        rows = []
        for name in pick:
            ct = tifffile.imread(IMAGES / name)
            lab = tifffile.imread(LABELS / name)
            if (lab == 1).sum() < 200:
                continue
            t0 = time.time()
            before = measured_thickness(ct, lab, seed=a.seed)
            thinned, st = thin_volume(ct, lab, a.width)
            after = measured_thickness(ct, thinned, seed=a.seed)
            row = {"sample": name, "width": a.width,
                   "thickness_before": None if before is None else round(before, 4),
                   "thickness_after": None if after is None else round(after, 4),
                   "frac_voxels_kept": st["frac_kept"],
                   "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"  {name}  thick {row['thickness_before']} -> {row['thickness_after']}"
                  f"   kept {row['frac_voxels_kept']:.3f}   {row['seconds']}s", flush=True)

        ok = [r for r in rows if r["thickness_before"] and r["thickness_after"]]
        summary = {
            "width": a.width, "n_volumes": len(ok),
            "mean_thickness_before": round(float(np.mean([r["thickness_before"] for r in ok])), 4),
            "mean_thickness_after": round(float(np.mean([r["thickness_after"] for r in ok])), 4),
            "mean_frac_kept": round(float(np.mean([r["frac_voxels_kept"] for r in ok])), 4),
            "rows": rows,
            "gate": ("preregistration condition 1: thinning must measurably reduce thickness "
                     "toward w on held-out volumes, or stop before training"),
        }
        Path(a.report).write_text(json.dumps(summary, indent=1))
        print(f"\n  thickness {summary['mean_thickness_before']} -> "
              f"{summary['mean_thickness_after']} (target {a.width}), "
              f"voxels kept {summary['mean_frac_kept']:.3f}")
        print(f"wrote {a.report}")
        return

    out_dir = Path(a.out or (ROOT / "data" / "kaggle" / f"labels_w{int(a.width*10):02d}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    for k, name in enumerate(samples):
        ct = tifffile.imread(IMAGES / name)
        lab = tifffile.imread(LABELS / name)
        thinned, st = thin_volume(ct, lab, a.width)
        tifffile.imwrite(out_dir / name, thinned.astype(lab.dtype))
        st["sample"] = name
        stats.append(st)
        if k % 25 == 0:
            print(f"  {k}/{len(samples)} {name} kept {st['frac_kept']}", flush=True)
    Path(str(out_dir) + "_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
