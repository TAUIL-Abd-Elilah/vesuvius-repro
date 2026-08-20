"""Are the published surface labels actually misplaced relative to the CT sheet?

The 2026 open problems page names **label snapping** — repositioning annotations using the CT
signal — as a proposed but unimplemented fix for "labels... themselves approximate". Before
building a snapper it is worth asking whether there is anything to snap.

⚠ THE OBVIOUS VERSION OF THIS MEASURES THE WRONG THING, and it took a wrong answer to notice.
Sampling every labelled voxel and taking the median |distance to the CT ridge| gives ~1.01 vox,
which reads as a large misplacement. It is not. A labelled sheet is roughly 3 voxels thick, so
its voxels are spread across that thickness and sit about a voxel from the centre-line **by
construction, for a perfect label**. That number measures label THICKNESS, not label PLACEMENT.

Placement is isolated by comparing each labelled run's CENTRE, along the across-sheet normal,
against the CT ridge. The normal returned by the Hessian has arbitrary sign, so this historical
script's population signed median has no directional meaning. Use `label_placement_oriented.py`,
which fixes the normal toward the pinned Scroll1A axis, for the corrective result.

  python label_placement.py --n 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from ridge_residual import across_sheet_normals, ridge_offset

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"
OVERLAP = ROOT / "vesuvius-repro" / "results" / "overlap" / "overlap_report.json"

HALF, STEP = 4.0, 0.25
PER_VOL = 600


def volume_offsets(ct: np.ndarray, lab: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance from each labelled run's centre to the CT ridge, in voxels."""
    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    mid = len(ts) // 2
    n = across_sheet_normals(ct, pts)
    c = (pts[:, :, None] + n[:, :, None] * ts[None, None, :]).transpose(1, 0, 2).reshape(3, -1)
    lp = map_coordinates(lab.astype(np.float32), c, order=0,
                         mode="nearest").reshape(len(pts), len(ts))
    ridge, _ = ridge_offset(ct, pts, n)

    out = []
    for i in range(len(pts)):
        r = lp[i]
        if r[mid] != 1 or np.isnan(ridge[i]):
            continue
        lo = mid
        while lo > 0 and r[lo - 1] == 1:
            lo -= 1
        hi = mid
        while hi < len(r) - 1 and r[hi + 1] == 1:
            hi += 1
        if lo == 0 or hi == len(r) - 1:      # run leaves the window; centre is unknown
            continue
        out.append(ridge[i] - 0.5 * (ts[lo] + ts[hi]))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "label_placement.json"))
    a = ap.parse_args()

    located = {r["sample"] for r in json.loads(OVERLAP.read_text())["located"]}
    rng = np.random.default_rng(a.seed)
    names = sorted(p.stem for p in LABELS.glob("sample_*.tif"))
    order = [names[i] for i in rng.choice(len(names), min(len(names), a.n * 4), replace=False)]

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
        off = volume_offsets(ct, lab, sel)
        if len(off) < 150:
            continue
        rows.append({"sample": nm, "located": nm in located, "n_runs": int(len(off)),
                     "median_signed": round(float(np.median(off)), 4),
                     "median_abs": round(float(np.median(np.abs(off))), 4)})

    s = np.array([r["median_signed"] for r in rows])
    q = np.array([r["median_abs"] for r in rows])
    out = {
        "n_volumes": len(rows), "per_volume_points": PER_VOL,
        "metric_noise_floor_synthetic": 0.164,
        "median_signed_offset": round(float(np.median(s)), 4),
        "q10_signed": round(float(np.quantile(s, 0.10)), 4),
        "q90_signed": round(float(np.quantile(s, 0.90)), 4),
        "median_abs_offset": round(float(np.median(q)), 4),
        "reading": ("WITHDRAWN: Hessian eigenvector signs were not tied to one physical "
                    "direction, so the population signed statistic cannot support a placement "
                    "claim. The correction vector and |offset| are sign-invariant, but |offset| "
                    "still mixes displacement with real-CT estimator error."),
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"volumes {len(rows)}   (synthetic noise floor 0.164 vox)")
    print(f"  UNORIENTED (withdrawn)      : median {np.median(s):+.3f}  "
          f"q10 {np.quantile(s, .1):+.3f}  q90 {np.quantile(s, .9):+.3f}")
    print(f"  |offset|                   : median {np.median(q):.3f}  (NOT a placement number)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
