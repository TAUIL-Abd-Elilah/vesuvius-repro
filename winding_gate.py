"""Gate for global winding assignment: is the adjacent-pair separation BIMODAL?  ✅ PASSED

The target is the thing `scrollprize.org/2026_open_problems` calls out most emphatically —
*"Most importantly, relative winding number annotations seem to have a great impact on the
spiral fit. Automating these procedures will boost scalability by a great extent!"* — and the
thing @aistae's scroll-truth explicitly does **not** do (*"not a winding number"*; it answers
pairwise same-wrap only).

**The formulation.** Give every patch an integer winding number w. For an adjacent pair,
  delta = w(j) - w(i)  should be  0  (same wrap)  or  +-1  (the next wrap in or out).
Then a global assignment exists iff every cycle's deltas sum to zero, and ⭐ **a cycle that does
not sum to zero IS a fork** — the same object Will Stevens describes when patch visit ordering
"forks at F, giving a region of both windings". One algorithm yields both the winding numbers
and the fork detector.

**What has to be true first, and is not guaranteed.** Deltas have to be recoverable. Two patches
on the same wrap sit side by side and touch, so their minimum separation is ~0. Two on adjacent
wraps are stacked a wrap pitch apart. So the distribution of minimum separations over adjacent
pairs must be **bimodal**. If it is a smear, no threshold recovers delta and this closes here.

⭐ **RESULT: PASSED on slab 12.** 1,931 patches, 14,513 adjacent pairs at r=25. Peaks at **0.2**
and **9.2** voxels with a dip of 69 against a smaller peak of 434 — a ratio of 0.16 against the
registered 0.60. **The Scroll 4 wrap pitch is ~9-10 voxels** and the same-wrap / next-wrap split
is clean at a threshold near 4.5. Independent sanity check: ScrollFiesta uses `--wrap-pitch 9.5`
on PHerc0139, a different scroll but the same order.

Registered gate, before looking:
  1. the offset distribution must be bimodal, i.e. a dip between two peaks with the dip below
     **60%** of the smaller peak
  2. the upper peak must sit at >= 3 voxels, or "adjacent wrap" is not separable from "same wrap
     measured noisily"

⚠ THE FIRST VERSION OF THIS MEASURED THE WRONG QUANTITY AND FAILED ITS OWN GATE. It projected
the centroid-to-centroid separation onto the mean sheet normal, giving a smear (median 21.3,
q95 166.6) with a dip ratio of 0.88. That was a measurement error, not physics: these patches
span ~200 voxels and curve, so two can touch at their edges while their centroids sit 160 voxels
apart, and centroid offset says nothing about local sheet separation. The gate thresholds were
NOT relaxed — the quantity was corrected to minimum separation, which needs no normal at all.

  python winding_gate.py --slab 12
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
WILL = ROOT / "_will"
SLAB = 250.0
ADJ_R = 25.0        # must be wide enough to SEE the next wrap, not clip it away
GRID_STRIDE = 6     # subsample the 127x127 grid; keeps normals but bounds the point count


def patch_records(slab: int):
    """Every patch in the slab with a subsampled grid, its centroid and a local normal."""
    recs = []
    for tag, zp, pre in (("bad", WILL / "s4_bad_patches.zip", "s4_bad_patches"),
                         ("good", WILL / "s4_good_patches.zip", "s4_good_patches")):
        z = zipfile.ZipFile(zp)
        names = sorted({n.split("/")[1] for n in z.namelist()
                        if n.startswith(pre + "/") and len(n.split("/")) > 2})
        for nm in names:
            try:
                meta = json.loads(z.read(f"{pre}/{nm}/meta.json"))
                b = meta["bbox"]
            except KeyError:
                continue
            cz = 0.5 * (b[0][2] + b[1][2])
            if int(cz // SLAB) != slab:
                continue
            try:
                xs = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/x.tif")))
                ys = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/y.tif")))
                zs = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/z.tif")))
            except KeyError:
                continue
            P = np.stack([zs, ys, xs], axis=-1).astype(np.float64)
            V = (xs > 0) & (ys > 0) & (zs > 0)
            if V.sum() < 600:
                continue

            # local normal from the patch's own tangents, averaged over valid interior nodes
            ok = (V[2:, 1:-1] & V[:-2, 1:-1] & V[1:-1, 2:] & V[1:-1, :-2])
            if ok.sum() < 100:
                continue
            Pu = (0.5 * (P[2:, 1:-1] - P[:-2, 1:-1]))[ok]
            Pv = (0.5 * (P[1:-1, 2:] - P[1:-1, :-2]))[ok]
            nrm = np.cross(Pu, Pv)
            nn = np.linalg.norm(nrm, axis=1)
            nrm = nrm[nn > 1e-9] / nn[nn > 1e-9, None]
            if len(nrm) < 50:
                continue
            # average with sign alignment, since cross products can flip across the grid
            ref = nrm[0]
            nrm = nrm * np.sign(nrm @ ref)[:, None]
            n = nrm.mean(0)
            n /= max(np.linalg.norm(n), 1e-9)

            pts = P[V][::GRID_STRIDE]
            recs.append({"patch": nm, "label": tag, "pts": pts,
                         "centroid": pts.mean(0), "normal": n})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slab", type=int, default=12)
    ap.add_argument("--radius", type=float, default=ADJ_R)
    ap.add_argument("--out", default=str(ROOT / "results" / "winding_gate.json"))
    a = ap.parse_args()

    print(f"loading slab {a.slab}", flush=True)
    recs = patch_records(a.slab)
    print(f"  {len(recs)} patches with usable normals", flush=True)
    if len(recs) < 50:
        raise SystemExit("too few patches")

    allp = np.concatenate([r["pts"] for r in recs])
    owner = np.concatenate([np.full(len(r["pts"]), i) for i, r in enumerate(recs)])
    tree = cKDTree(allp)
    mind = {}
    for i, j in tree.query_pairs(r=a.radius, output_type="ndarray"):
        x, y = owner[i], owner[j]
        if x == y:
            continue
        k = (min(x, y), max(x, y))
        d = float(np.linalg.norm(allp[i] - allp[j]))
        if k not in mind or d < mind[k]:
            mind[k] = d
    print(f"  {len(mind)} adjacent pairs at r={a.radius}", flush=True)

    # ⚠ MINIMUM DISTANCE BETWEEN THE PATCHES, not a centroid offset. The first version of this
    # projected the centroid separation onto the mean normal and produced a smear (median 21.3,
    # q95 166.6) that failed the gate. That was a measurement error, not physics: these patches
    # span ~200 voxels and curve, so two of them touch at their edges while their centroids sit
    # 160 voxels apart, and the centroid offset says nothing about local sheet separation.
    # Same-wrap neighbours touch (min distance ~0); adjacent wraps are stacked a pitch apart.
    offs = np.array([d for d in mind.values()])

    hist, edges = np.histogram(offs, bins=50, range=(0, ADJ_R))
    ctr = 0.5 * (edges[1:] + edges[:-1])
    # find the two largest well-separated peaks
    order = np.argsort(-hist)
    p1 = order[0]
    p2 = next((k for k in order if abs(ctr[k] - ctr[p1]) >= 4.0), None)
    ok_bimodal = False
    dip = peak2 = None
    if p2 is not None:
        lo, hi = sorted([p1, p2])
        dip = float(hist[lo:hi + 1].min())
        smaller = float(min(hist[p1], hist[p2]))
        ok_bimodal = dip < 0.60 * smaller
        peak2 = float(ctr[max(p1, p2)])

    print(f"\n  offset distribution over {len(offs)} pairs:")
    print(f"    median {np.median(offs):.2f}  q75 {np.quantile(offs,.75):.2f}  "
          f"q95 {np.quantile(offs,.95):.2f}")
    if p2 is None:
        print("    only ONE peak found -> not bimodal")
    else:
        print(f"    peaks at {ctr[p1]:.1f} and {ctr[p2]:.1f} vox; dip {dip:.0f} vs "
              f"smaller peak {min(hist[p1],hist[p2]):.0f}  "
              f"(need dip < 60%)")
    g1 = bool(ok_bimodal)
    g2 = bool(peak2 is not None and peak2 >= 3.0)
    print(f"\n  GATE 1 bimodal              {'PASS' if g1 else 'FAIL'}")
    print(f"  GATE 2 upper peak >= 3 vox  {'PASS' if g2 else 'FAIL'}")
    print(f"\n  GATE {'PASSED' if g1 and g2 else 'FAILED'}")

    Path(a.out).write_text(json.dumps(
        {"slab": a.slab, "radius": a.radius, "n_patches": len(recs), "n_pairs": len(offs),
         "hist": hist.tolist(), "centres": ctr.tolist(),
         "median_offset": float(np.median(offs)),
         "gate_bimodal": g1, "gate_peak": g2, "gate_passed": bool(g1 and g2)}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
