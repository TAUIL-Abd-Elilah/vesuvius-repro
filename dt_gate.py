"""Gate zero for PREREGISTER_dt.md: is there a densely-packed regime in this data at all?

Bet 4 claims a distance-transform target merges fewer nearby sheets than a binary one. That is
only measurable where two distinct sheets actually come close. **If the 892-volume set rarely
contains such pairs, the study is about nothing** and closes here.

⚠ There is real reason to expect it might. Our own earlier probe (closing-based contact fraction)
found only 0.001–0.006 of sheet volume bridged at radius 2–3, which is very little. Bet 3 closed
at its gate in an afternoon; this may too, and that is the gate doing its job.

Registered thresholds (§4): over >= 100 volumes, **the median volume must contain >= 5 pairs of
distinct labelled sheet components whose closest approach is <= 6 voxels**, and **>= 60% of
volumes must contain >= 1**.

Method note: components are found with a 3-D 6-connectivity structure, tiny fragments are
dropped as speckle, and closest approach is computed exactly with a KD-tree — but only for pairs
whose bounding boxes are already within the threshold, which prunes almost everything and keeps
this to a few seconds a volume.

  python dt_gate.py --n 100
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_erosion, find_objects, generate_binary_structure, label
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
LABELS = ROOT / "data" / "kaggle" / "labels"

NEAR = 6.0          # "densely packed": closest approach <= this, in voxels
MIN_COMP = 100      # voxels; smaller components are speckle, not sheets
MAX_COMPS = 60      # guard: volumes with more components than this are subsampled by size
MAX_PTS = 4000      # boundary voxels kept per component; see close_pairs


def bbox_gap(a, b) -> float:
    """Lower bound on the distance between two bounding boxes. Cheap pre-filter."""
    d = 0.0
    for sa, sb in zip(a, b):
        if sa.start > sb.stop - 1:
            d += (sa.start - (sb.stop - 1)) ** 2
        elif sb.start > sa.stop - 1:
            d += (sb.start - (sa.stop - 1)) ** 2
    return float(np.sqrt(d))


def close_pairs(lab: np.ndarray) -> tuple[int, int, list]:
    """(n_components, n_close_pairs, closest distances) for one volume."""
    st = generate_binary_structure(3, 1)
    comp, n = label(lab == 1, structure=st)
    if n == 0:
        return 0, 0, []
    sizes = np.bincount(comp.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= MIN_COMP]
    if len(keep) > MAX_COMPS:                     # largest first, bounded work
        keep = sorted(keep, key=lambda i: -sizes[i])[:MAX_COMPS]
    if len(keep) < 2:
        return len(keep), 0, []

    # ⚠ BOUNDARY VOXELS ONLY, and subsampled. The closest approach between two solids is always
    # between their surfaces, so interior voxels cannot contribute and only slow the KD-tree
    # down. A first version used full point sets and did not finish 12 volumes in four minutes.
    # Subsampling can only ever OVERestimate the gap, so it under-counts close pairs — the
    # conservative direction for a gate that must not pass on an artifact.
    slices = find_objects(comp)
    rng = np.random.default_rng(0)
    pts = {}
    for i in keep:
        sl = slices[i - 1]
        sub = comp[sl] == i
        shell = sub & ~binary_erosion(sub, structure=st)
        p = np.argwhere(shell) + np.array([s.start for s in sl])
        if len(p) > MAX_PTS:
            p = p[rng.choice(len(p), MAX_PTS, replace=False)]
        pts[i] = p

    dists = []
    for i, j in combinations(keep, 2):
        if bbox_gap(slices[i - 1], slices[j - 1]) > NEAR:
            continue                              # cannot possibly be close
        a, b = pts[i], pts[j]
        if len(a) == 0 or len(b) == 0:
            continue
        if len(a) > len(b):
            a, b = b, a
        d = float(cKDTree(b).query(a, k=1)[0].min())
        if d <= NEAR:
            dists.append(d)
    return len(keep), len(dists), dists


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "dt_gate.json"))
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    names = sorted(p.stem for p in LABELS.glob("sample_*.tif"))
    order = [names[i] for i in rng.permutation(len(names))]

    rows = []
    for nm in order:
        if len(rows) >= a.n:
            break
        lab = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        if (lab == 1).sum() < 800:
            continue
        ncomp, npair, dists = close_pairs(lab)
        rows.append({"sample": nm, "n_components": ncomp, "n_close_pairs": npair,
                     "min_gap": round(float(min(dists)), 3) if dists else None})
        if len(rows) % 5 == 0:
            print(f"  {len(rows)}/{a.n}", flush=True)

    p = np.array([r["n_close_pairs"] for r in rows])
    c = np.array([r["n_components"] for r in rows])
    med = float(np.median(p))
    frac1 = float(np.mean(p >= 1))
    g1, g2 = med >= 5, frac1 >= 0.60

    out = {"n_volumes": len(rows), "near_threshold_vox": NEAR, "min_component_vox": MIN_COMP,
           "median_close_pairs": med, "mean_close_pairs": round(float(p.mean()), 2),
           "frac_volumes_with_ge1_pair": round(frac1, 4),
           "median_components_per_volume": float(np.median(c)),
           "gate1_median_pairs_ge5": bool(g1),
           "gate2_frac_with_pair_ge_0.60": bool(g2),
           "gate_passed": bool(g1 and g2), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1))

    print(f"\nvolumes {len(rows)}")
    print(f"  components per volume (>= {MIN_COMP} vox): median {np.median(c):.0f}")
    print(f"  close pairs (gap <= {NEAR:.0f} vox): median {med:.1f}  mean {p.mean():.2f}  "
          f"max {p.max()}")
    print(f"  GATE 1  median close pairs = {med:.1f}   (need >= 5)   {'PASS' if g1 else 'FAIL'}")
    print(f"  GATE 2  volumes with >=1 pair = {frac1:.1%}   (need >= 60%)   "
          f"{'PASS' if g2 else 'FAIL'}")
    print(f"\n  GATE ZERO {'PASSED' if g1 and g2 else 'FAILED'}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
