"""⛔ DISCARDED, 2026-08-07. The estimator does not measure what it claims. Kept as a record.

The ratio below is a free parameter, not a measurement. Sweeping the profile half-width over
10 volumes:

    half-width   +-3.0   +-4.0   +-6.0   +-8.0
    predicted     4.00    4.75    5.56    5.62
    CT            4.25    5.25    7.06    8.75
    ratio         0.94    0.88    0.77    0.62

The predicted FWHM saturates near 5.6 while the CT FWHM keeps growing with the window. That is
the tell: `p` is a saturating probability field with a sharp edge, CT intensity decays
gradually, and a wider window keeps lowering the CT profile's baseline and so its half-max
threshold. Their half-max widths are not the same kind of quantity and their ratio means
nothing. It also explains why this reported a CT FWHM of 7.25 against the 3.5 in
`results/ct_sheet_thickness.json`, which used a +-4.0 window centred on LABELLED voxels.

So neither "m7 paints sheets too thick" nor "too thin" is supported. Any future attempt at
this needs a thickness definition that is invariant to the window -- fitting a parametric
profile, or a fixed absolute threshold rather than a half-max relative to a drifting baseline.

Original docstring follows.

---

How thick are m7's predicted sheets, measured against the CT rather than the labels?

Every angle tonight has died on a label confound: the margin enrichment was proximity, and
ridge NMS was matched by random-direction thinning. This asks the thickness question WITHOUT
labels, so annotation density and boundary convention cannot enter.

The motivating number is from the 855-volume benchmark. On the Scroll1A-located population
m7 predicts 3.48x the labelled sheet volume (0.2660 / 0.0764); elsewhere it predicts 2.30x
(0.1359 / 0.0590). Recall is meanwhile WORSE on the located set, 0.777 against 0.918, and
mean confidence collapses, 0.849 against 0.951. That is the signature of confusion, not of
conservatism -- but "3.48x the labelled volume" is only meaningful if the labels are a fair
reference, and they may simply annotate less where sheets are hard.

So compare the predicted sheet to the CT sheet directly, profile by profile, along the
across-sheet normal:

    predicted FWHM / CT FWHM

A ratio near 1 means m7's surfaces are as thick as the papyrus and the over-prediction is an
annotation effect. A ratio well above 1 means m7 genuinely paints surfaces thicker than the
CT sheet -- which matters downstream, because tracing consumes these predictions and a sheet
painted too thick is a sheet that can merge with its neighbour.

⚠ The estimator inflates. `results/thickness_control.json` measured +0.286 vox of pure
voxelisation inflation on synthetic sheets of known thickness. That bias applies to BOTH
profiles here and largely cancels in a ratio, but it is why the absolute FWHMs are reported
alongside the ratio rather than the ratio alone.

  python m7_thickness.py --n 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from thin_labels import across_sheet_dirs

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
PRED_CACHE = ROOT / "results" / "m7_pred_cache"
OVERLAP = ROOT / "vesuvius-repro" / "results" / "overlap" / "overlap_report.json"

SIZE, TRIM = 256, 64
HALF, STEP = 6.0, 0.25
PRED_T = 0.5          # sample the confident core of the prediction, not its flank
N_PROFILES = 700


def fwhm(prof: np.ndarray) -> np.ndarray:
    """Width of the contiguous above-half-max run THROUGH THE PROFILE CENTRE, in voxels.

    ⚠ Not `(prof >= half).sum()`. A profile 12 voxels wide routinely contains the neighbouring
    sheet as well, and counting every sample above half-max silently adds its width: that
    version reported a CT FWHM of 7.00 vox against the 3.5 measured in
    `results/ct_sheet_thickness.json`, which is what exposed it. Only the run containing the
    centre belongs to the sheet the profile is centred on.
    """
    base = prof.min(axis=1, keepdims=True)
    peak = prof.max(axis=1, keepdims=True)
    above = prof >= (base + 0.5 * (peak - base))
    mid = prof.shape[1] // 2
    n = np.zeros(len(prof), dtype=np.float32)
    for i in range(len(prof)):
        if not above[i, mid]:
            n[i] = np.nan
            continue
        lo = mid
        while lo > 0 and above[i, lo - 1]:
            lo -= 1
        hi = mid
        while hi < prof.shape[1] - 1 and above[i, hi + 1]:
            hi += 1
        n[i] = (hi - lo + 1) * STEP
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "results" / "m7_thickness.json"))
    a = ap.parse_args()

    located = {r["sample"] for r in json.loads(OVERLAP.read_text())["located"]}
    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    names = sorted(q.stem for q in PRED_CACHE.glob("*.npy"))[:a.n]

    rows = []
    for k, nm in enumerate(names):
        p = np.load(PRED_CACHE / f"{nm}.npy").astype(np.float32)
        ct = np.asarray(tifffile.imread(str(IMAGES / f"{nm}.tif")))
        off = (ct.shape[0] - SIZE) // 2
        lo, hi = off + TRIM, off + SIZE - TRIM
        ctc = ct[lo:hi, lo:hi, lo:hi].astype(np.float32)

        pts = np.argwhere(p > PRED_T).astype(np.int32)
        if len(pts) < 500:
            continue
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), min(N_PROFILES, len(pts)), replace=False)]

        d = across_sheet_dirs(ctc, pts)
        c = pts[:, :, None].astype(np.float32) + d[:, :, None] * ts[None, None, :]
        cc = c.transpose(1, 0, 2).reshape(3, -1)
        pp = map_coordinates(p, cc, order=1, mode="nearest").reshape(len(pts), len(ts))
        cp = map_coordinates(ctc, cc, order=1, mode="nearest").reshape(len(pts), len(ts))

        rows.append({"sample": nm, "located": nm in located,
                     "pred_fwhm": round(float(np.nanmedian(fwhm(pp))), 3),
                     "ct_fwhm": round(float(np.nanmedian(fwhm(cp))), 3),
                     "n_profiles": int(len(pts))})
        if k % 10 == 0:
            print(f"  [{k+1}/{len(names)}] {nm}  pred {rows[-1]['pred_fwhm']:.2f}  "
                  f"ct {rows[-1]['ct_fwhm']:.2f}", flush=True)

    out = {"n_volumes": len(rows), "pred_threshold": PRED_T,
           "voxelisation_inflation_note": (
               "+0.286 vox on both profiles, from results/thickness_control.json; "
               "largely cancels in the ratio"),
           "rows": rows}
    for tag, sel in (("all", rows),
                     ("located", [r for r in rows if r["located"]]),
                     ("other", [r for r in rows if not r["located"]])):
        if not sel:
            continue
        pf = np.array([r["pred_fwhm"] for r in sel])
        cf = np.array([r["ct_fwhm"] for r in sel])
        out[tag] = {"n": len(sel),
                    "median_pred_fwhm": round(float(np.median(pf)), 3),
                    "median_ct_fwhm": round(float(np.median(cf)), 3),
                    "median_ratio": round(float(np.median(pf / cf)), 3)}
        print(f"{tag:<9} n={len(sel):<3} predicted FWHM {np.median(pf):5.2f} vox   "
              f"CT FWHM {np.median(cf):5.2f} vox   ratio {np.median(pf/cf):4.2f}x")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
