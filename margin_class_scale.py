"""Gate zero for PREREGISTER_margin.md: is the asserted margin a property of the label SET?

The motivating number - 97.1% of the voxel just outside a labelled sheet run is class 0,
positively asserted background - came from 8 volumes whose ignore-fraction ranges from 0.13
to 0.78. If margin class tracks how heavily a volume was annotated, that pooled percentage
could be carried by a handful of densely labelled volumes and would not describe the set.

So this reports the distribution ACROSS volumes, not a pooled percentage, over >= 200 volumes.

Registered gate: proceed only if the MEDIAN per-volume class-0 margin share is >= 0.80, and
the distribution is not strongly bimodal. Below that, bet 2 is closed unamended.

  python margin_class_scale.py --n 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from thin_labels import across_sheet_dirs, HALF, STEP

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"


def margin_classes(ct: np.ndarray, lab: np.ndarray, n_sample: int, seed: int) -> dict | None:
    """Class of the voxel immediately outside each labelled sheet run, along the normal."""
    pts = np.argwhere(lab == 1)
    if len(pts) < 400:
        return None
    rng = np.random.default_rng(seed)
    pts = pts[rng.choice(len(pts), size=min(n_sample, len(pts)), replace=False)].astype(np.int32)

    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    mid = len(ts) // 2
    dirs = across_sheet_dirs(ct, pts)
    coords = pts[:, :, None].astype(np.float32) + dirs[:, :, None] * ts[None, None, :]
    prof = map_coordinates(lab.astype(np.float32),
                           coords.transpose(1, 0, 2).reshape(3, -1),
                           order=0, mode="nearest").reshape(len(pts), len(ts))

    counts = {0: 0, 1: 0, 2: 0}
    for r in prof:
        if r[mid] != 1:
            continue
        lo = mid
        while lo > 0 and r[lo - 1] == 1:
            lo -= 1
        hi = mid
        while hi < len(r) - 1 and r[hi + 1] == 1:
            hi += 1
        if lo == 0 or hi == len(r) - 1:          # run leaves the window; margin unknown
            continue
        for j in (lo - 1, hi + 1):
            c = int(r[j])
            if c in counts:
                counts[c] += 1
    tot = sum(counts.values())
    if tot < 100:
        return None
    return {"n_margins": tot,
            "frac_background": counts[0] / tot,
            "frac_ignore": counts[2] / tot,
            "frac_surface": counts[1] / tot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="volumes to sample")
    ap.add_argument("--per-volume", type=int, default=900, help="labelled voxels per volume")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "margin_class_scale.json"))
    a = ap.parse_args()

    names = sorted(p.name for p in LABELS.glob("sample_*.tif"))
    rng = np.random.default_rng(a.seed)
    pick = [names[i] for i in rng.choice(len(names), size=min(a.n, len(names)), replace=False)]

    rows, t0 = [], time.time()
    for k, nm in enumerate(pick):
        try:
            ct = tifffile.imread(IMAGES / nm)
            lab = tifffile.imread(LABELS / nm)
        except Exception:                         # noqa: BLE001
            continue
        r = margin_classes(ct, lab, a.per_volume, a.seed)
        if r is None:
            continue
        r["sample"] = nm
        r["vol_frac_ignore"] = float((lab == 2).mean())
        r["vol_frac_surface"] = float((lab == 1).mean())
        rows.append(r)
        if k % 20 == 0:
            print(f"  [{k}/{len(pick)}] {nm}  bg {r['frac_background']:.3f}  "
                  f"ign {r['frac_ignore']:.3f}  (vol ignore {r['vol_frac_ignore']:.2f})  "
                  f"{time.time()-t0:.0f}s", flush=True)

    bg = np.array([r["frac_background"] for r in rows])
    med = float(np.median(bg))
    # bimodality check: share of volumes in the middle of the range. A genuinely bimodal
    # distribution empties the middle.
    mid_share = float(((bg > 0.2) & (bg < 0.8)).mean())

    out = {
        "n_volumes": len(rows),
        "median_frac_background": round(med, 4),
        "mean_frac_background": round(float(bg.mean()), 4),
        "q10": round(float(np.quantile(bg, 0.10)), 4),
        "q90": round(float(np.quantile(bg, 0.90)), 4),
        "frac_volumes_ge_0.80": round(float((bg >= 0.80).mean()), 4),
        "share_in_middle_0.2_0.8": round(mid_share, 4),
        "corr_bg_vs_vol_ignore": round(float(np.corrcoef(
            bg, [r["vol_frac_ignore"] for r in rows])[0, 1]), 4),
        "gate": "proceed only if median_frac_background >= 0.80 and not strongly bimodal",
        "gate_passed": bool(med >= 0.80),
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"\n  volumes measured: {len(rows)}")
    print(f"  per-volume class-0 margin share: median {med:.4f}  "
          f"mean {bg.mean():.4f}  q10 {out['q10']}  q90 {out['q90']}")
    print(f"  volumes with share >= 0.80: {out['frac_volumes_ge_0.80']:.1%}")
    print(f"  share sitting in the middle (0.2-0.8): {mid_share:.1%}")
    print(f"  corr(margin background share, volume ignore fraction): "
          f"{out['corr_bg_vs_vol_ignore']}")
    print(f"\n  GATE {'PASSED' if out['gate_passed'] else 'FAILED'} "
          f"(registered threshold: median >= 0.80)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
