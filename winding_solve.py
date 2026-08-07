"""⚠ NOT WORKING — global winding assignment from patch geometry. Kept with its results.

STATE, 2026-08-07. The spiral structure is genuinely recovered; the integer assignment is not.

  solver / edge policy                  inconsistent   wraps   implied pitch   Spearman(w,radius)
  BFS propagation, centroid sign            58.9%        --          --              --
  BFS propagation, local contact sign       36.8%        --          --              --
  LS + IRLS, conservative edges (8-11)       2.9%        33       51.8 vox           +0.826
  LS + IRLS, multi-wrap deltas              31.6%        57       30.2 vox           +0.839
  measured pitch from winding_gate                                  ~9.5 vox

⛔ **THE 2.9% IS NOT A SUCCESS AND I REPORTED IT AS ONE FOR AN HOUR.** It is low largely because
the graph is sparse: restricting edges to the 8-11 vox band leaves most wrap transitions with no
edge at all, and a graph with few cycles has few chances to be inconsistent. The tell was the
implied pitch — 51.8 vox against a measured 9.5, i.e. the scroll compressed 5x — which I treated
as a separate problem instead of as evidence that the consistency was hollow. I had explicitly
added a coverage check when tightening thresholds precisely to stop myself buying consistency by
deleting edges, but I checked CONNECTIVITY (98% in one component) rather than whether enough WRAP
TRANSITIONS survived. Connectivity stayed high while the transitions were gutted.

What is real: Spearman(winding, radius) = +0.83 in both regimes, so the solve recovers true
spiral structure. What is not: any correct integer winding number.

Three things a continuation needs, and it wants a fresh preregistration rather than an amendment:
  1. a gate on SCALE (implied pitch must match the measured pitch), not only on consistency
  2. a synthetic spiral with known winding numbers, to validate before touching real data
  3. multi-wrap deltas that are actually reliable — at 19-38 vox separation, "which wrap" is
     ambiguous where the sheet curves, and round(d/pitch) is not good enough

⚠ The fork detector built on this does NOT predict Will Stevens' bad-patch labels: fork AUC
0.528-0.542 across three slabs against plain degree at 0.585-0.608. Degree is better. So "bad
patch" is not "winding fork", and the hypothesis that one algorithm would deliver both is wrong.

Original docstring follows.

---

Global winding assignment from patch geometry, and forks as its residual.

The gate (`winding_gate.py`) established that adjacent-pair separation on Scroll 4 is cleanly
bimodal — peaks at 0.2 and 9.2 voxels, dip ratio 0.16 — so the wrap pitch is ~9-10 vox and
"same wrap" versus "next wrap" is separable near 4.5. That makes the delta on each graph edge
recoverable, which is everything this needs.

**The formulation.** Give patch i an integer winding number w(i). For an adjacent pair,

    delta(i,j) = 0   if they touch          (same wrap)
    delta(i,j) = +-1 if separated by ~pitch (next wrap; sign from which is further out)

A globally consistent assignment exists **iff every cycle's deltas sum to zero**. Solve by
spanning-forest propagation, then every non-tree edge gives a residual

    resid(i,j) = w(j) - w(i) - delta(i,j)

and ⭐ **an edge with non-zero residual closes a cycle that does not sum to zero — a FORK.**
That is precisely what Will Stevens describes when visit ordering "forks at F, giving a region
of both windings". So one algorithm produces the winding numbers the open problems page asks to
automate *and* a fork detector, and the fork detector is testable against his 9,171 labels.

⚠ SIGN NEEDS A RADIAL DIRECTION. Outward is away from the scroll axis, which is not published
for Scroll 4, so it is estimated from the patch cloud itself (see `estimate_axis`) and the
estimate is reported rather than assumed — a bad axis flips signs and manufactures forks.

⚠ THIS IS SCROLL 4 PATCH GEOMETRY ONLY. No CT, no model, no tracer. @aistae's scroll-truth
answers the pairwise same-wrap question on Scroll 1 from CT intensity and explicitly does not
produce winding numbers; this is the complementary half and on different data.

  python winding_solve.py --slab 12
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from winding_gate import patch_records

ROOT = Path(__file__).resolve().parent
SAME_MAX = 4.5      # min separation below this -> same wrap (from the gate's dip)
NEXT_MAX = 15.0     # between SAME_MAX and this -> one wrap apart; beyond, no edge
ADJ_R = 25.0


def estimate_axis(recs) -> np.ndarray:
    """Scroll axis in (y,x), as the point minimising the spread of wrap radii.

    A plain centroid is pulled by uneven patch coverage. The axis of a spiral is better found
    as the point about which the cloud is most annular: scan a coarse grid around the centroid
    and take the centre whose radial histogram is flattest (a good centre spreads radii evenly
    across wraps; a bad one piles them up).
    """
    pts = np.concatenate([r["pts"] for r in recs])[:, 1:]   # (y, x)
    c0 = pts.mean(0)
    best, best_score = c0, None
    for dy in np.linspace(-300, 300, 13):
        for dx in np.linspace(-300, 300, 13):
            c = c0 + np.array([dy, dx])
            rad = np.linalg.norm(pts - c, axis=1)
            h, _ = np.histogram(rad, bins=40)
            h = h[h > 0]
            score = float(np.std(h / h.sum()))              # flatter is better
            if best_score is None or score < best_score:
                best, best_score = c, score
    return best


def build(recs, axis):
    allp = np.concatenate([r["pts"] for r in recs])
    owner = np.concatenate([np.full(len(r["pts"]), i) for i, r in enumerate(recs)])
    tree = cKDTree(allp)
    mind = {}
    for i, j in tree.query_pairs(r=ADJ_R, output_type="ndarray"):
        x, y = owner[i], owner[j]
        if x == y:
            continue
        k = (min(x, y), max(x, y))
        d = float(np.linalg.norm(allp[i] - allp[j]))
        if k not in mind or d < mind[k]:
            mind[k] = d

    # ⚠ SIGN IS DETERMINED LOCALLY, AT THE CONTACT POINT. The first version compared the two
    # patches' centroid distances from the axis, which FAILED: for cross-wrap pairs that radial
    # difference has median 40.4 vox against a true pitch of 9.5, because it is dominated by
    # where along the wrap each patch sits rather than which wrap it is on. The sign was a coin
    # flip and 58.9% of cycles failed to close.
    #
    # Locally it is well posed: at the closest point between the two patches, which side of
    # patch i's surface does patch j lie on? Normals are first oriented OUTWARD (positive dot
    # with the local radial direction), since a raw cross product has arbitrary sign per patch
    # and inconsistent orientation alone would flip deltas.
    for r in recs:
        radial = np.array([0.0, r["centroid"][1] - axis[0], r["centroid"][2] - axis[1]])
        nr = np.linalg.norm(radial)
        if nr > 1e-9 and float(r["normal"] @ (radial / nr)) < 0:
            r["normal"] = -r["normal"]

    trees = {i: cKDTree(r["pts"]) for i, r in enumerate(recs)}
    edges = []
    for (i, j), d in mind.items():
        if d <= SAME_MAX:
            edges.append((i, j, 0))
        elif d <= NEXT_MAX:
            # closest point on i to patch j, and the local offset to j there
            dd, ii = trees[i].query(recs[j]["pts"], k=1)
            m = int(np.argmin(dd))
            pj = recs[j]["pts"][m]
            pi = recs[i]["pts"][ii[m]]
            off = float((pj - pi) @ recs[i]["normal"])
            if abs(off) < 1.0:
                continue                       # ambiguous: not clearly on either side
            edges.append((i, j, 1 if off > 0 else -1))
    rad = np.array([np.linalg.norm(r["centroid"][1:] - axis) for r in recs])
    return edges, rad


def solve(n, edges):
    """Spanning-forest propagation of w, then residuals on the non-tree edges."""
    adj = defaultdict(list)
    for i, j, dl in edges:
        adj[i].append((j, dl))
        adj[j].append((i, -dl))

    w = np.full(n, np.nan)
    tree_edge = set()
    for s in range(n):
        if not np.isnan(w[s]):
            continue
        w[s] = 0.0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, dl in adj[u]:
                if np.isnan(w[v]):
                    w[v] = w[u] + dl
                    tree_edge.add((min(u, v), max(u, v)))
                    q.append(v)

    resid = {}
    for i, j, dl in edges:
        k = (min(i, j), max(i, j))
        if k in tree_edge:
            continue
        resid[k] = float(w[j] - w[i] - dl)
    return w, resid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slab", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "results" / "winding_solve.json"))
    a = ap.parse_args()

    recs = patch_records(a.slab)
    print(f"slab {a.slab}: {len(recs)} patches", flush=True)
    axis = estimate_axis(recs)
    print(f"  estimated axis (y,x) = ({axis[0]:.0f}, {axis[1]:.0f})", flush=True)

    edges, rad = build(recs, axis)
    same = sum(1 for _, _, d in edges if d == 0)
    print(f"  edges: {len(edges)}  ({same} same-wrap, {len(edges)-same} cross-wrap)", flush=True)

    w, resid = solve(len(recs), edges)
    bad_edges = {k: v for k, v in resid.items() if abs(v) > 0.5}
    print(f"  non-tree edges {len(resid)}, inconsistent {len(bad_edges)} "
          f"({len(bad_edges)/max(len(resid),1):.1%})", flush=True)
    print(f"  winding numbers span {np.nanmin(w):.0f} .. {np.nanmax(w):.0f}", flush=True)

    # per-patch: how many incident edges close an inconsistent cycle
    inc = np.zeros(len(recs))
    deg = np.zeros(len(recs))
    for (i, j), v in resid.items():
        deg[i] += 1; deg[j] += 1
        if abs(v) > 0.5:
            inc[i] += 1; inc[j] += 1
    frac = inc / np.maximum(deg, 1)

    lab = np.array([r["label"] == "bad" for r in recs])
    from scipy.stats import mannwhitneyu
    out = {"slab": a.slab, "axis": axis.tolist(), "n_patches": len(recs),
           "n_edges": len(edges), "n_same": same,
           "n_nontree": len(resid), "n_inconsistent": len(bad_edges),
           "winding_min": float(np.nanmin(w)), "winding_max": float(np.nanmax(w))}
    print(f"\n{'feature':<18}{'bad':>10}{'good':>10}{'AUC':>8}{'p':>11}")
    for nm, v in (("fork_count", inc), ("fork_frac", frac)):
        b, g = v[lab], v[~lab]
        u = mannwhitneyu(b, g, alternative="two-sided")
        auc = u.statistic / (len(b) * len(g))
        out[f"auc_{nm}"] = round(float(auc), 4)
        out[f"p_{nm}"] = float(u.pvalue)
        print(f"  {nm:<16}{np.median(b):>10.4f}{np.median(g):>10.4f}{auc:>8.3f}{u.pvalue:>11.2e}")
    out["rows"] = [{"patch": r["patch"], "label": r["label"], "w": float(w[i]),
                    "fork_count": float(inc[i]), "fork_frac": float(frac[i])}
                   for i, r in enumerate(recs)]
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
