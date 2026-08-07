"""Does a patch's position in the ADJACENCY GRAPH predict Will Stevens' bad-patch labels?

Two intrinsic properties are already ruled out on his Scroll 4 patches:

  where the patch sits   CT-ridge residual        AUC 0.512  (p 0.38)
  how it is parameterised  metric distortion      AUC ~0.48  (p > 0.28)

Both classes are geometrically pristine — stretch 1.01, shear 0.005 — and sit on the papyrus
equally well. So badness is **relational**, which is also how he describes it: *"if patch visit
ordering starts at S and proceeds along patches in the red winding, it can fork at F, giving a
region of both windings"*. A fork is a property of the graph, not of a patch.

So: build the adjacency graph over every patch in one z-slab, and ask whether graph position
separates his two classes.

Features, chosen because each is a different guess at what "forks the winding" means:

  degree            how many patches this one touches
  clustering        fraction of its neighbour pairs that are themselves adjacent. A patch
                    bridging two groups that do not touch each other has LOW clustering — that
                    is the graph signature of a fork.
  neigh_spread      RMS distance from the patch centroid to its neighbours' centroids,
                    normalised by its own size. A patch whose neighbours sit far apart in
                    different directions is spanning something it maybe should not.
  betweenness_local how many of its neighbour pairs have no other common neighbour — a cheap
                    local cut measure that does not need a global shortest-path computation.

⚠ ADJACENCY IS BUILT WITH ONE KD-TREE OVER ALL PATCH POINTS AT ONCE, not pairwise. A slab holds
~1,900 patches, so pairwise is 1.8M comparisons; a single `query_pairs` is O(N log N) and takes
seconds. The earlier CT check died from exactly this kind of quadratic mistake.

⚠ Good and bad patches are pooled and processed identically before any label is looked at.

  python patch_graph.py --slab 12
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
WILL = ROOT / "_will"
SLAB = 250.0
PTS_PER_PATCH = 160
ADJ_R = 6.0          # two patches are adjacent if any sampled points come within this


def load_slab(slab: int, seed: int):
    """All patches whose bbox centre-z falls in the slab, from both archives."""
    rng = np.random.default_rng(seed)
    recs = []
    for tag, zp, pre in (("bad", WILL / "s4_bad_patches.zip", "s4_bad_patches"),
                         ("good", WILL / "s4_good_patches.zip", "s4_good_patches")):
        z = zipfile.ZipFile(zp)
        names = sorted({n.split("/")[1] for n in z.namelist()
                        if n.startswith(pre + "/") and len(n.split("/")) > 2})
        for nm in names:
            try:
                meta = json.loads(z.read(f"{pre}/{nm}/meta.json"))
            except KeyError:
                continue
            b = meta["bbox"]
            cz = 0.5 * (b[0][2] + b[1][2])
            if int(cz // SLAB) != slab:
                continue
            try:
                xs = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/x.tif")))
                ys = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/y.tif")))
                zs = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/z.tif")))
            except KeyError:
                continue
            v = (xs > 0) & (ys > 0) & (zs > 0)
            if v.sum() < 400:
                continue
            P = np.stack([zs[v], ys[v], xs[v]], axis=1).astype(np.float32)
            if len(P) > PTS_PER_PATCH:
                P = P[rng.choice(len(P), PTS_PER_PATCH, replace=False)]
            recs.append({"patch": nm, "label": tag, "pts": P,
                         "centroid": P.mean(0), "size": float(np.linalg.norm(P.std(0)))})
    return recs


def build_graph(recs, r: float):
    """Adjacency by one KD-tree over every sampled point, mapped back to patch ids."""
    allp = np.concatenate([r["pts"] for r in recs])
    owner = np.concatenate([np.full(len(r["pts"]), i) for i, r in enumerate(recs)])
    tree = cKDTree(allp)
    adj = defaultdict(set)
    for i, j in tree.query_pairs(r=r, output_type="ndarray"):
        a, b = owner[i], owner[j]
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def features(recs, adj):
    out = []
    for i, r in enumerate(recs):
        nb = sorted(adj.get(i, ()))
        deg = len(nb)
        clust = 0.0
        cut = 0.0
        if deg >= 2:
            pairs = 0
            linked = 0
            nocommon = 0
            for a_i in range(deg):
                for b_i in range(a_i + 1, deg):
                    a, b = nb[a_i], nb[b_i]
                    pairs += 1
                    if b in adj.get(a, ()):
                        linked += 1
                    elif not (adj.get(a, set()) & adj.get(b, set()) - {i}):
                        nocommon += 1
            clust = linked / max(pairs, 1)
            cut = nocommon / max(pairs, 1)
        spread = 0.0
        if deg:
            d = np.linalg.norm(np.array([recs[j]["centroid"] for j in nb]) - r["centroid"],
                               axis=1)
            spread = float(np.sqrt((d ** 2).mean()) / max(r["size"], 1e-6))
        out.append({"patch": r["patch"], "label": r["label"], "degree": float(deg),
                    "clustering": clust, "neigh_spread": spread, "local_cut": cut})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slab", type=int, default=12)
    ap.add_argument("--radius", type=float, default=ADJ_R)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "patch_graph.json"))
    a = ap.parse_args()

    print(f"loading slab {a.slab} (z {a.slab*SLAB:.0f}-{(a.slab+1)*SLAB:.0f})", flush=True)
    recs = load_slab(a.slab, a.seed)
    nb_ = sum(1 for r in recs if r["label"] == "bad")
    print(f"  {len(recs)} patches: {nb_} bad, {len(recs)-nb_} good", flush=True)
    if len(recs) < 50:
        raise SystemExit("too few patches in slab")

    adj = build_graph(recs, a.radius)
    feats = features(recs, adj)
    deg = np.array([f["degree"] for f in feats])
    print(f"  adjacency r={a.radius}: median degree {np.median(deg):.1f}, "
          f"isolated {np.mean(deg==0):.1%}", flush=True)

    from scipy.stats import mannwhitneyu
    bad = [f for f in feats if f["label"] == "bad"]
    good = [f for f in feats if f["label"] == "good"]
    out = {"slab": a.slab, "radius": a.radius, "n_bad": len(bad), "n_good": len(good),
           "median_degree": float(np.median(deg)), "rows": feats}
    print(f"\n{'feature':<16}{'bad':>10}{'good':>10}{'AUC':>8}{'p':>11}")
    for k in ("degree", "clustering", "neigh_spread", "local_cut"):
        b = np.array([f[k] for f in bad])
        g = np.array([f[k] for f in good])
        u = mannwhitneyu(b, g, alternative="two-sided")
        auc = u.statistic / (len(b) * len(g))
        out[f"auc_{k}"] = round(float(auc), 4)
        out[f"p_{k}"] = float(u.pvalue)
        print(f"  {k:<14}{np.median(b):>10.4f}{np.median(g):>10.4f}{auc:>8.3f}"
              f"{u.pvalue:>11.2e}")
    print("\n  AUC 0.5 = graph position says nothing about which patches he flagged.")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
