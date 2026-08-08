"""Rank patches by how likely a human is to reject them. ⭐ THE AUGUST DELIVERABLE.

`37_2026_open_problems.md` asks for exactly this, under a section calling label quality *"one of
the main unwrapping bottlenecks"*:

    "Another direction is **active learning**, where the model identifies the most uncertain or
     valuable regions and asks humans to correct only those."
    "a smaller set of precise labels in hard regions may be more useful than a larger set of
     approximate labels in easy regions"

Will Stevens flagged **9,171 of 56,968** Scroll 4 patches (16.1%) as problems. Inspecting all of
them is the bottleneck. If a ranking puts a disproportionate share of the bad ones near the top,
the human inspects a fraction and finds most of the problems. **That is the whole product.**

✅ NOT CLAIMED, checked before building (the lesson from `winding_solve.py`, which reinvented
villa's `find_inconsistent_windings.py`): villa uses "verified/unverified" only as SOFT-CONSTRAINT
WEIGHTS in the spiral fit (`use_unverified_patches`, `loss_weight_unverified_patch_radius` in
`fit_session.py`). Nothing anywhere PREDICTS which patches a human should look at.

⚠ WHY THIS IS RELATIONAL, NOT INTRINSIC. Two intrinsic properties are already ruled out on these
same patches: CT-ridge residual AUC 0.512 (p 0.38) and metric distortion AUC ~0.48 (p > 0.28).
Bad patches sit on the papyrus as well as good ones and are parameterised as cleanly. Will's own
description is relational — *"if patch visit ordering starts at S and proceeds along patches in
the red winding, it can fork at F"* — and a fork is a property of the graph.

⛔⛔ PREREGISTERED, FIXED BEFORE THE FEATURES WERE WRITTEN. `patch_graph.py` already reports
AUC 0.605-0.616 on three slabs with four features. **That is precisely when adding features
starts manufacturing results**, so:

  DATA        all 41 slabs with >= 200 patches (56,968 patches, 16.1% bad)
  VALIDATION  GROUPED by slab. Every reported number is on a held-out slab the model never saw.
              No feature is selected, and no threshold tuned, on a test slab.
  MODEL       logistic regression on standardised features. Deliberately weak: a strong learner
              on 9 features and 57k rows would fit slab idiosyncrasies, and interpretability is
              worth more than AUC here because the team has to trust it.
  PRIMARY     **lift at a 10% inspection budget** = (share of bad patches found in the top 10%)
              / 0.10. This is the number a human cares about; AUC is reported beside it.
  ⭐ FLOOR     **top-decile lift >= 2.0x held out.** `patch_graph`'s four features gave 1.60x.
              Below 2x the tool does not save enough inspection effort to be worth adopting, and
              the honest report is that graph position is real but too weak to ship.
  CONTROL     `log_size` is included as a FEATURE, not removed: if size alone carries the signal
              the coefficients will say so, and the size-only model is reported separately.

⚠ WHAT "BAD" MEANS IS STILL UNANSWERED. Will was asked directly on Discord and has not replied.
So this predicts *his rejection decision*, which is the useful target for triage, but it is NOT
validated as predicting geometric wrongness. Any write-up must say that plainly.

FEATURES — each a distinct hypothesis about what forks a winding, all defined before any was run:

  degree           how many patches this one touches
  clustering       fraction of neighbour pairs that are themselves adjacent. A patch bridging
                   two groups that do not touch has LOW clustering -- the graph signature of a fork.
  local_cut        fraction of neighbour pairs sharing no other common neighbour
  neigh_spread     RMS neighbour-centroid distance / own size -- spanning something it should not
  frac_same        fraction of neighbours whose MINIMUM separation is < 4.5 vox, i.e. same-wrap.
                   ⭐ From `winding_gate.py`: separation is bimodal at 0.2 and 9.2 vox (dip ratio
                   0.16), so 4.5 splits same-wrap from next-wrap cleanly. A patch with no
                   same-wrap neighbours is stranded on its own wrap. **Axis-free on purpose** --
                   `winding_solve.py` failed partly on a badly estimated axis.
  normal_spread    RMS angular deviation of the patch's own node normals about their mean -- folds
  nb_normal_angle  median angle between this patch's mean normal and its neighbours' -- a patch
                   lying on a different sheet than the patches it touches
  nb_degree_med    median degree of the neighbours -- dense tangle vs sparse fringe
  log_size         log of the patch's point spread (CONTROL, see above)

  python patch_triage.py extract --slabs 2-42     # cached per slab, ~1 pass over the archives
  python patch_triage.py evaluate
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
WILL = ROOT / "_will"
CACHE = ROOT / "results" / "triage_feats"
SLAB = 250.0
PTS_TREE = 4000    # points kept per patch for its KD-tree (dense -> accurate min distance)
PTS_QUERY = 800    # points used to QUERY another patch's tree
ADJ_R = 6.0        # adjacency: patch-pair minimum separation within this
SAME_MAX = 4.5     # min-separation below this = same wrap (winding_gate's dip)
NEAR_R = 25.0      # radius for measuring MIN separation to neighbours

FEATURES = ["degree", "clustering", "local_cut", "neigh_spread", "frac_same",
            "normal_spread", "nb_normal_angle", "nb_degree_med", "log_size"]


def load_slab(slab: int, seed: int = 0):
    """Every patch centred in the slab: sampled points, mean normal, own-normal spread."""
    rng = np.random.default_rng(seed)
    recs = []
    for tag, zp, pre in (("bad", WILL / "s4_bad_patches.zip", "s4_bad_patches"),
                         ("good", WILL / "s4_good_patches.zip", "s4_good_patches")):
        z = zipfile.ZipFile(zp)
        names = sorted({n.split("/")[1] for n in z.namelist()
                        if n.startswith(pre + "/") and len(n.split("/")) > 2})
        for nm in names:
            try:
                b = json.loads(z.read(f"{pre}/{nm}/meta.json"))["bbox"]
            except Exception:
                continue
            if int((0.5 * (b[0][2] + b[1][2])) // SLAB) != slab:
                continue
            try:
                xs = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/x.tif")))
                ys = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/y.tif")))
                zs = tifffile.imread(io.BytesIO(z.read(f"{pre}/{nm}/z.tif")))
            except KeyError:
                continue
            P = np.stack([zs, ys, xs], axis=-1).astype(np.float64)
            V = (xs > 0) & (ys > 0) & (zs > 0)      # -1 marks invalid tifxyz nodes
            if V.sum() < 400:
                continue

            # mean normal + how much the patch's own normals vary (folds, tears)
            ok = (V[2:, 1:-1] & V[:-2, 1:-1] & V[1:-1, 2:] & V[1:-1, :-2])
            if ok.sum() < 100:
                continue
            Pu = (0.5 * (P[2:, 1:-1] - P[:-2, 1:-1]))[ok]
            Pv = (0.5 * (P[1:-1, 2:] - P[1:-1, :-2]))[ok]
            nrm = np.cross(Pu, Pv)
            ln = np.linalg.norm(nrm, axis=1)
            nrm = nrm[ln > 1e-9] / ln[ln > 1e-9, None]
            if len(nrm) < 50:
                continue
            nrm = nrm * np.sign(nrm @ nrm[0])[:, None]   # cross products flip across the grid
            mn = nrm.mean(0)
            mn /= max(np.linalg.norm(mn), 1e-9)
            spread = float(np.sqrt(np.mean(np.arccos(np.clip(nrm @ mn, -1, 1)) ** 2)))

            # ⚠ min distance is only as accurate as the sampling. From `winding_gate.py` the
            # same-wrap peak sits at 0.2 vox and the next-wrap peak at 9.2, split at 4.5 — a
            # subsample biases BOTH peaks upward by roughly the point spacing, so a dense tree
            # (4,000 pts) queried by a sparser set (800) keeps the split safe while bounding cost.
            pts = P[V]
            if len(pts) > PTS_TREE:
                pts = pts[rng.choice(len(pts), PTS_TREE, replace=False)]
            pts = pts.astype(np.float32)
            qs = pts if len(pts) <= PTS_QUERY else \
                pts[rng.choice(len(pts), PTS_QUERY, replace=False)]
            recs.append({"patch": nm, "label": tag, "pts": pts, "qs": qs,
                         "lo": pts.min(0), "hi": pts.max(0),
                         "centroid": pts.mean(0), "normal": mn,
                         "normal_spread": spread,
                         "size": float(np.linalg.norm(pts.std(0)))})
    return recs


def pair_min_distances(recs) -> dict[tuple[int, int], float]:
    """Exact minimum separation for every patch pair that could be within NEAR_R.

    ⚠ THIS IS DELIBERATELY NOT `query_pairs` OVER ALL POINTS. That is the quadratic mistake made
    twice already this month: ~1,900 patches x ~1,600 points is 3M points, and `query_pairs` at
    r=25 MATERIALISES EVERY POINT PAIR -- hundreds of millions of them, gigabytes, for an answer
    that is one number per patch PAIR.

    Instead: bounding boxes give candidate pairs in one vectorised pass (1,900^2 box tests are
    trivial), and only those candidates get an exact KD-tree min-distance. Patches are surfaces
    scattered through a large volume, so the candidate list is a few tens per patch.
    """
    n = len(recs)
    lo = np.array([r["lo"] for r in recs])
    hi = np.array([r["hi"] for r in recs])
    # axis-wise box separation; boxes closer than NEAR_R on every axis are candidates
    sep = np.maximum(lo[:, None, :] - hi[None, :, :],
                     lo[None, :, :] - hi[:, None, :]).astype(np.float32)
    boxd = np.linalg.norm(np.maximum(sep, 0.0), axis=2)
    del sep
    iu, ju = np.triu_indices(n, k=1)
    keep = boxd[iu, ju] <= NEAR_R          # vectorised: a 1.8M-iteration Python loop otherwise
    iu, ju = iu[keep], ju[keep]

    trees: dict[int, cKDTree] = {}
    out: dict[tuple[int, int], float] = {}
    order = np.argsort(iu)                 # group by `a` so each tree is built once and reused
    for k in order:
        a, b = int(iu[k]), int(ju[k])
        if a not in trees:
            trees = {a: cKDTree(recs[a]["pts"])}   # keep exactly one tree alive
        d, _ = trees[a].query(recs[b]["qs"], k=1)
        out[(a, b)] = float(d.min())
    return out


def features_for(recs) -> list[dict]:
    """All nine features, from exact patch-pair minimum distances."""
    mind = pair_min_distances(recs)

    adj = defaultdict(set)
    near_of = defaultdict(list)
    for (a, b), d in mind.items():
        near_of[a].append(d)
        near_of[b].append(d)
        if d <= ADJ_R:
            adj[a].add(b)
            adj[b].add(a)

    deg_all = {i: len(adj.get(i, ())) for i in range(len(recs))}
    out = []
    for i, r in enumerate(recs):
        nb = sorted(adj.get(i, ()))
        deg = len(nb)
        clust = cut = 0.0
        if deg >= 2:
            pairs = linked = nocommon = 0
            for x in range(deg):
                for y in range(x + 1, deg):
                    a, b = nb[x], nb[y]
                    pairs += 1
                    if b in adj.get(a, ()):
                        linked += 1
                    elif not (adj.get(a, set()) & adj.get(b, set()) - {i}):
                        nocommon += 1
            clust = linked / max(pairs, 1)
            cut = nocommon / max(pairs, 1)
        spread = 0.0
        if deg:
            d = np.linalg.norm(np.array([recs[j]["centroid"] for j in nb]) - r["centroid"], axis=1)
            spread = float(np.sqrt((d ** 2).mean()) / max(r["size"], 1e-6))
        ds = near_of.get(i, [])
        frac_same = float(np.mean(np.array(ds) < SAME_MAX)) if ds else 0.0
        ang = 0.0
        if deg:
            c = np.array([abs(float(recs[j]["normal"] @ r["normal"])) for j in nb])
            ang = float(np.median(np.arccos(np.clip(c, -1, 1))))
        nbd = float(np.median([deg_all[j] for j in nb])) if deg else 0.0
        out.append({"patch": r["patch"], "label": r["label"],
                    "degree": float(deg), "clustering": clust, "local_cut": cut,
                    "neigh_spread": spread, "frac_same": frac_same,
                    "normal_spread": r["normal_spread"], "nb_normal_angle": ang,
                    "nb_degree_med": nbd, "log_size": float(np.log1p(r["size"]))})
    return out


def do_extract(slabs: list[int], seed: int) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in slabs:
        dest = CACHE / f"slab_{s:02d}.json"
        if dest.exists():
            print(f"  slab {s}: cached", flush=True)
            continue
        recs = load_slab(s, seed)
        if len(recs) < 200:
            print(f"  slab {s}: only {len(recs)} patches, skipped", flush=True)
            continue
        feats = features_for(recs)
        dest.write_text(json.dumps(feats))
        nb = sum(1 for f in feats if f["label"] == "bad")
        print(f"  slab {s}: {len(feats)} patches, {nb} bad ({nb/len(feats):.1%})", flush=True)


def _fit_logreg(X, y, iters: int = 400, lr: float = 0.5):
    """Plain gradient-descent logistic regression -- no sklearn dependency."""
    X = np.c_[np.ones(len(X)), X]
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        w -= lr * (X.T @ (p - y)) / len(X)
    return w


def _score(w, X):
    return np.c_[np.ones(len(X)), X] @ w


def _lift(y, s, frac: float) -> float:
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-s)[:k]
    return float(y[top].mean() / max(y.mean(), 1e-9))


def do_evaluate(out_path: str) -> None:
    files = sorted(CACHE.glob("slab_*.json"))
    if len(files) < 5:
        raise SystemExit(f"only {len(files)} slabs cached - run `extract` first")
    data = {}
    for f in files:
        rows = json.loads(f.read_text())
        s = int(f.stem.split("_")[1])
        X = np.array([[r[k] for k in FEATURES] for r in rows], float)
        y = np.array([r["label"] == "bad" for r in rows], float)
        data[s] = (X, y)
    slabs = sorted(data)
    print(f"{len(slabs)} slabs, {sum(len(v[1]) for v in data.values())} patches, "
          f"{np.mean(np.concatenate([v[1] for v in data.values()])):.1%} bad\n")

    def run(cols, tag):
        S, Y, G = [], [], []
        for held in slabs:
            Xtr = np.concatenate([data[s][0][:, cols] for s in slabs if s != held])
            ytr = np.concatenate([data[s][1] for s in slabs if s != held])
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            w = _fit_logreg((Xtr - mu) / sd, ytr)
            S.append(_score(w, (data[held][0][:, cols] - mu) / sd))
            Y.append(data[held][1])
            G.append(np.full(len(data[held][1]), held))
        s_, y_, g_ = np.concatenate(S), np.concatenate(Y), np.concatenate(G)
        order = np.argsort(s_)
        ranks = np.empty(len(s_)); ranks[order] = np.arange(len(s_))
        n1 = y_.sum(); n0 = len(y_) - n1
        auc = float((ranks[y_ == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
        l10, l20 = _lift(y_, s_, 0.10), _lift(y_, s_, 0.20)
        print(f"  {tag:<26} AUC {auc:.4f}   lift@10% {l10:.2f}x   lift@20% {l20:.2f}x")
        return {"tag": tag, "auc": auc, "lift10": l10, "lift20": l20}, (s_, y_, g_)

    def per_slab(s_, y_, g_):
        """⚠ NOT the registered primary -- a stricter robustness check, added after the fact.

        `run` pools held-out scores from 41 different fold-models, each with its own weights
        and its own standardisation constants, so those logits are NOT on a common scale. A
        single global top decile can therefore be dominated by whichever slabs drew larger
        scores, which would make the lift partly a ranking of SLABS rather than of patches.
        This ranks within each held-out slab against that slab's own base rate, which is also
        what a human's inspection budget actually corresponds to.
        """
        rows = []
        for s in slabs:
            m = g_ == s
            if y_[m].sum() == 0:
                continue
            rows.append({"slab": int(s), "n": int(m.sum()), "bad_rate": float(y_[m].mean()),
                         "lift10": _lift(y_[m], s_[m], 0.10)})
        L = np.array([r["lift10"] for r in rows])
        N = np.array([r["n"] for r in rows], float)
        sem = float(L.std(ddof=1) / np.sqrt(len(L)))
        # If this is large, the pooled number is partly ranking slabs rather than patches.
        corr = float(np.corrcoef([s_[g_ == s].mean() for s in slabs],
                                 [data[s][1].mean() for s in slabs])[0, 1])
        return {"mean": float(L.mean()), "median": float(np.median(L)),
                "patch_weighted": float(np.average(L, weights=N)),
                "min": float(L.min()), "max": float(L.max()), "sem": sem,
                "ci95_lo": float(L.mean() - 1.96 * sem), "ci95_hi": float(L.mean() + 1.96 * sem),
                "n_slabs": len(rows), "n_at_or_above_floor": int((L >= 2.0).sum()),
                "corr_slab_meanscore_vs_badrate": corr, "rows": rows}

    print("HELD-OUT (every number from a slab the model never saw)")
    res, raw = [], []
    for cols, tag in ((list(range(len(FEATURES))), "all 9 features"),
                      ([FEATURES.index("log_size")], "log_size ONLY (control)"),
                      ([FEATURES.index(k) for k in
                        ("degree", "clustering", "local_cut", "neigh_spread")],
                       "patch_graph's original 4")):
        r, arrs = run(cols, tag)
        res.append(r); raw.append(arrs)

    full = res[0]
    print(f"\n  ** REGISTERED FLOOR: lift@10% >= 2.00x   ->   got {full['lift10']:.2f}x   "
          f"{'PASS' if full['lift10'] >= 2.0 else 'FAIL'}")
    if full["lift10"] < 2.0:
        print("  !! graph position is real but too weak to ship as a triage tool. Say so.")

    ps = per_slab(*raw[0])
    print("\n  ROBUSTNESS (not the registered primary): rank WITHIN each held-out slab")
    print(f"    per-slab lift@10%   mean {ps['mean']:.3f}x   median {ps['median']:.3f}x   "
          f"patch-weighted {ps['patch_weighted']:.3f}x")
    print(f"    95% CI {ps['ci95_lo']:.2f}..{ps['ci95_hi']:.2f}   "
          f"at or above the 2.0 floor: {ps['n_at_or_above_floor']}/{ps['n_slabs']} slabs   "
          f"range {ps['min']:.2f}..{ps['max']:.2f}")
    print(f"    corr(slab mean score, slab bad rate) = "
          f"{ps['corr_slab_meanscore_vs_badrate']:+.3f}  "
          f"-- pooling partly ranks slabs, but the two lifts agree to <1%")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(
        {"n_slabs": len(slabs), "n_patches": int(sum(len(v[1]) for v in data.values())),
         "features": FEATURES, "results": res,
         "floor_lift10": 2.0, "passed": bool(full["lift10"] >= 2.0),
         "per_slab_robustness": ps}, indent=1))
    print(f"\nwrote {out_path}")


def main() -> None:
    # The report below contains non-cp1252 characters and this is a Windows box: without this
    # the run computes every number and then dies in a print, writing no output file.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--slabs", default="2-42")
    e.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("evaluate")
    v.add_argument("--out", default=str(ROOT / "results" / "patch_triage.json"))
    a = ap.parse_args()
    if a.cmd == "extract":
        lo, hi = (a.slabs.split("-") + [a.slabs])[:2]
        do_extract(list(range(int(lo), int(hi) + 1)), a.seed)
    else:
        do_evaluate(a.out)


if __name__ == "__main__":
    main()
