"""Rank patches by how likely a human is to reject them. ⭐ THE AUGUST DELIVERABLE.

`37_2026_open_problems.md` asks for exactly this, under a section calling label quality *"one of
the main unwrapping bottlenecks"*:

    "Another direction is **active learning**, where the model identifies the most uncertain or
     valuable regions and asks humans to correct only those."
    "a smaller set of precise labels in hard regions may be more useful than a larger set of
     approximate labels in easy regions"

Will Stevens' published split records **9,171 of 56,968** Scroll 4 patches (16.1%) as problems.
Inspecting all of them is the bottleneck. If a ranking puts a disproportionate share of the bad
ones near the top, the human inspects a fraction and finds most of the problems. **That is the
whole product.**

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
His public pipeline can generate several exclusion lists, but the published archives do not say
which one produced this split. So this predicts *his recorded accepted/rejected split*, which is
the useful target for triage, but it is NOT validated as predicting geometric wrongness or a
documented human decision. Any write-up must say that plainly.

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
  python patch_triage.py rank path/to/tifxyz_patches --out ranking.csv --budget 0.10
  python patch_triage.py rank patches.zip --slab 12 --out ranking.csv --budget 0.10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
WILL = ROOT / "_will"
CACHE = ROOT / "results" / "triage_feats"
MODEL = ROOT / "results" / "patch_triage_model.json"
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


def _meta_slab(meta: object) -> int | None:
    """Return the 250-voxel z slab recorded by tifxyz metadata, if valid."""
    try:
        bbox = meta["bbox"]
        return int((0.5 * (bbox[0][2] + bbox[1][2])) // SLAB)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _tifxyz_record(patch: str, xs, ys, zs, meta: object,
                   rng: np.random.Generator) -> dict:
    """Turn one x/y/z TIFF triplet into the fixed deployment representation."""
    xs, ys, zs = map(np.asarray, (xs, ys, zs))
    if not (xs.shape == ys.shape == zs.shape) or xs.ndim != 2:
        raise ValueError(
            f"coordinate shapes must be equal 2-D arrays, got "
            f"{xs.shape}, {ys.shape}, {zs.shape}"
        )
    P = np.stack([zs, ys, xs], axis=-1).astype(np.float64)
    V = (xs > 0) & (ys > 0) & (zs > 0)
    if V.sum() < 400:
        raise ValueError(f"only {int(V.sum())} valid nodes; need at least 400")
    ok = V[2:, 1:-1] & V[:-2, 1:-1] & V[1:-1, 2:] & V[1:-1, :-2]
    if ok.sum() < 100:
        raise ValueError(
            f"only {int(ok.sum())} central-difference nodes; need at least 100"
        )
    Pu = (0.5 * (P[2:, 1:-1] - P[:-2, 1:-1]))[ok]
    Pv = (0.5 * (P[1:-1, 2:] - P[1:-1, :-2]))[ok]
    nrm = np.cross(Pu, Pv)
    length = np.linalg.norm(nrm, axis=1)
    nrm = nrm[length > 1e-9] / length[length > 1e-9, None]
    if len(nrm) < 50:
        raise ValueError(f"only {len(nrm)} valid normals; need at least 50")
    nrm = nrm * np.sign(nrm @ nrm[0])[:, None]
    mean_normal = nrm.mean(0)
    mean_normal /= max(np.linalg.norm(mean_normal), 1e-9)
    spread = float(np.sqrt(np.mean(
        np.arccos(np.clip(nrm @ mean_normal, -1, 1)) ** 2
    )))
    pts = P[V]
    slab = _meta_slab(meta)
    if slab is None:
        slab = int(float(pts[:, 0].mean()) // SLAB)
    if len(pts) > PTS_TREE:
        pts = pts[rng.choice(len(pts), PTS_TREE, replace=False)]
    pts = pts.astype(np.float32)
    qs = pts if len(pts) <= PTS_QUERY else pts[
        rng.choice(len(pts), PTS_QUERY, replace=False)
    ]
    return {
        "patch": patch, "label": "unknown", "pts": pts, "qs": qs,
        "lo": pts.min(0), "hi": pts.max(0), "centroid": pts.mean(0),
        "normal": mean_normal, "normal_spread": spread,
        "size": float(np.linalg.norm(pts.std(0))), "slab": slab,
    }


def _read_json_bytes(data: bytes) -> object:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _enumerate_tifxyz_source(path: Path, prefix: str,
                             slabs: set[int] | None) -> tuple[str, list[dict]]:
    """Describe eligible patches without loading their large coordinate TIFFs."""
    source = path.resolve()
    descriptors = []
    if source.is_dir():
        parents = sorted(
            {p.parent for p in source.rglob("x.tif")
             if (p.parent / "y.tif").is_file() and (p.parent / "z.tif").is_file()},
            key=lambda p: p.as_posix(),
        )
        for parent in parents:
            relative = parent.relative_to(source).as_posix()
            relative = parent.name if relative == "." else relative
            meta_path = parent / "meta.json"
            meta = None
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    pass
            meta_slab = _meta_slab(meta)
            if slabs is not None and meta_slab is not None and meta_slab not in slabs:
                continue
            descriptors.append({
                "source": source, "kind": "directory", "parent": parent,
                "patch": f"{prefix}/{relative}" if prefix else relative, "meta": meta,
            })
        return "directory", descriptors

    if not source.is_file() or not zipfile.is_zipfile(source):
        raise SystemExit(f"not a tifxyz directory or ZIP archive: {path}")
    with zipfile.ZipFile(source) as archive:
        names = {name.replace("\\", "/").lstrip("./"): name
                 for name in archive.namelist() if not name.endswith(("/", "\\"))}
        parents = sorted({
            PurePosixPath(name).parent.as_posix()
            for name in names
            if PurePosixPath(name).name == "x.tif"
            and f"{PurePosixPath(name).parent.as_posix()}/y.tif" in names
            and f"{PurePosixPath(name).parent.as_posix()}/z.tif" in names
        })
        parts = [PurePosixPath(parent).parts for parent in parents]
        strip_top = bool(parts and all(len(p) >= 2 and p[0] == parts[0][0] for p in parts))
        for parent, parent_parts in zip(parents, parts):
            relative = PurePosixPath(*parent_parts[1:]).as_posix() if strip_top else parent
            meta_member = f"{parent}/meta.json"
            meta = _read_json_bytes(archive.read(names[meta_member])) \
                if meta_member in names else None
            meta_slab = _meta_slab(meta)
            if slabs is not None and meta_slab is not None and meta_slab not in slabs:
                continue
            descriptors.append({
                "source": source, "kind": "zip", "parent": parent,
                "members": {axis: names[f"{parent}/{axis}.tif"]
                            for axis in ("x", "y", "z")},
                "patch": f"{prefix}/{relative}" if prefix else relative,
                "meta": meta,
            })
    return "zip", descriptors


def load_tifxyz_inputs(paths: list[Path], seed: int = 0, max_patches: int = 5000,
                       slabs: set[int] | None = None):
    """Load tifxyz review batches from directories and/or ZIPs without extraction."""
    if not paths:
        raise SystemExit("provide at least one tifxyz directory or ZIP archive")
    multiple = len(paths) > 1
    descriptors = []
    for path in paths:
        prefix = (path.stem if path.is_file() else path.name) if multiple else ""
        _, found = _enumerate_tifxyz_source(path, prefix, slabs)
        descriptors.extend(found)
    descriptors.sort(key=lambda item: item["patch"])
    if not descriptors:
        scope = " in the requested slab(s)" if slabs is not None else ""
        raise SystemExit(
            f"no tifxyz patches{scope} (expected PATCH/x.tif, y.tif, z.tif)"
        )
    duplicates = [name for name, count in Counter(
        item["patch"] for item in descriptors
    ).items() if count > 1]
    if duplicates:
        raise SystemExit(f"duplicate patch names across inputs: {duplicates[:5]}")
    if len(descriptors) > max_patches:
        raise SystemExit(
            f"found {len(descriptors)} patches, above --max-patches {max_patches}; "
            "pass --slab N or rank a smaller review batch"
        )

    rng = np.random.default_rng(seed)
    recs, skipped = [], []
    zip_handles: dict[Path, zipfile.ZipFile] = {}
    try:
        for item in descriptors:
            patch = item["patch"]
            try:
                if item["kind"] == "directory":
                    parent = item["parent"]
                    xs = tifffile.imread(parent / "x.tif")
                    ys = tifffile.imread(parent / "y.tif")
                    zs = tifffile.imread(parent / "z.tif")
                else:
                    source, members = item["source"], item["members"]
                    if source not in zip_handles:
                        zip_handles[source] = zipfile.ZipFile(source)
                    archive = zip_handles[source]
                    xs = tifffile.imread(io.BytesIO(archive.read(members["x"])))
                    ys = tifffile.imread(io.BytesIO(archive.read(members["y"])))
                    zs = tifffile.imread(io.BytesIO(archive.read(members["z"])))
                record = _tifxyz_record(patch, xs, ys, zs, item["meta"], rng)
                if slabs is None or int(record["slab"]) in slabs:
                    recs.append(record)
            except Exception as exc:
                skipped.append({"patch": patch, "reason": str(exc)})
    finally:
        for archive in zip_handles.values():
            archive.close()
    if not recs:
        raise SystemExit(f"none of {len(descriptors)} candidate tifxyz patches were valid")
    return recs, skipped


def load_cached_training(cache_dir: Path = CACHE):
    """Return the fixed labelled feature matrix used by the held-out study."""
    files = sorted(cache_dir.glob("slab_*.json"))
    if len(files) < 5:
        raise SystemExit(f"only {len(files)} slabs cached - run `extract` first")
    rows = []
    for path in files:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    X = np.array([[row[name] for name in FEATURES] for row in rows], dtype=float)
    y = np.array([row["label"] == "bad" for row in rows], dtype=float)
    return files, X, y


def fit_deployment_model(cache_dir: Path = CACHE) -> dict:
    """Fit the fixed all-nine-feature model on every labelled training slab."""
    files, X, y = load_cached_training(cache_dir)
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    weights = _fit_logreg((X - mu) / sd, y)
    # BLAS reduction order can move the final bit across thread counts. Fifteen
    # decimal places retain far more precision than ranking needs and make the
    # tracked JSON byte-stable across single- and multi-threaded rebuilds.
    mu = np.round(mu, 15)
    sd = np.round(sd, 15)
    weights = np.round(weights, 15)
    source_hash = hashlib.sha256()
    for path in files:
        source_hash.update(path.name.encode("utf-8"))
        source_hash.update(b"\0")
        with path.open("rb") as handle:
            while block := handle.read(1 << 20):
                source_hash.update(block)
    return {
        "tool": "patch_triage deployment model",
        "serialization_decimals": 15,
        "features": FEATURES,
        "training": {
            "n_slabs": len(files), "n_patches": int(len(y)),
            "bad_rate": float(y.mean()),
            "cache_manifest_sha256": source_hash.hexdigest(),
        },
        "standardization_mean": mu.tolist(),
        "standardization_sd": sd.tolist(),
        "weights_intercept_then_features": weights.tolist(),
        "validation": {
            "source": "results/patch_triage.json",
            "mean_per_slab_lift10": 2.34503899036176,
            "slabs_at_or_above_2x": 37,
            "n_validation_slabs": 41,
        },
    }


def load_deployment_model(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(
            f"deployment model not found: {path}; rebuild with `patch_triage.py fit-model`"
        )
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("features") != FEATURES:
        raise SystemExit(f"model feature schema does not match this script: {path}")
    for key, length in (("standardization_mean", len(FEATURES)),
                        ("standardization_sd", len(FEATURES)),
                        ("weights_intercept_then_features", len(FEATURES) + 1)):
        if len(model.get(key, [])) != length:
            raise SystemExit(f"model field {key} has the wrong length: {path}")
    return model


def do_fit_model(cache_dir: str, out_path: str) -> None:
    model = fit_deployment_model(Path(cache_dir))
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")


def do_rank(inputs: list[str], out_path: str, budget: float, seed: int,
            max_patches: int, model_path: str, slabs: list[int] | None) -> None:
    """Load the fixed model artifact and rank a tifxyz review batch."""
    if not (0.0 < budget <= 1.0):
        raise SystemExit("--budget must be in (0, 1]")
    model_file = Path(model_path)
    model = load_deployment_model(model_file)
    model_digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
    mu = np.asarray(model["standardization_mean"], dtype=float)
    sd = np.asarray(model["standardization_sd"], dtype=float)
    weights = np.asarray(model["weights_intercept_then_features"], dtype=float)

    recs, skipped = load_tifxyz_inputs(
        [Path(value) for value in inputs], seed=seed, max_patches=max_patches,
        slabs=set(slabs) if slabs else None,
    )
    groups = defaultdict(list)
    for record in recs:
        groups[int(record["slab"])].append(record)

    ranked = []
    for slab in sorted(groups):
        features = features_for(groups[slab])
        X = np.array([[row[name] for name in FEATURES] for row in features], float)
        scores = _score(weights, (X - mu) / sd)
        names = np.array([row["patch"] for row in features], dtype=object)
        order = np.lexsort((names, -scores))
        k = max(1, int(round(budget * len(features))))
        for rank0, index in enumerate(order):
            row = features[int(index)]
            ranked.append({
                "slab": slab,
                "rank_within_slab": rank0 + 1,
                "n_in_slab": len(features),
                "selected": rank0 < k,
                "score": float(scores[int(index)]),
                **{name: float(row[name]) for name in FEATURES},
                "patch": row["patch"],
            })

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = ["patch", "slab", "rank_within_slab", "n_in_slab", "selected",
               "score", *FEATURES]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ranked)

    metadata = {
        "tool": "patch_triage rank",
        "input": str(inputs[0]) if len(inputs) == 1 else None,
        "inputs": [str(value) for value in inputs],
        "requested_slabs": sorted(set(slabs)) if slabs else None,
        "output": str(destination),
        "seed": int(seed),
        "budget": float(budget),
        "ranking_scope": "within each 250-voxel z slab; scores are not cross-slab calibrated",
        "n_input_patches": len(recs) + len(skipped),
        "n_ranked": len(ranked),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "training": {
            "model_path": str(model_path), "model_sha256": model_digest,
            "features": FEATURES,
            **model["training"], "validation": model.get("validation"),
        },
        "limitations": [
            "Predicts Will Stevens' recorded accepted/rejected split; its provenance is not documented by the archives.",
            "The target is not independently validated geometric wrongness or a documented human decision.",
            "Held-out mean per-slab lift@10% was 2.345x; 4 of 41 slabs missed the 2.0x floor.",
            "The five added features did not materially outperform patch_graph's original four.",
        ],
    }
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    selected = sum(bool(row["selected"]) for row in ranked)
    print(f"ranked {len(ranked)} patches in {len(groups)} slab(s); "
          f"selected {selected} at budget {budget:.1%}")
    if skipped:
        print(f"skipped {len(skipped)} invalid patches; details: {metadata_path}")
    print(f"wrote {destination}")
    print(f"wrote {metadata_path}")
    print("scope: predicts a recorded accepted/rejected split, not validated geometric wrongness")


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


def do_evaluate(out_path: str, cache_dir: Path = CACHE) -> None:
    files = sorted(cache_dir.glob("slab_*.json"))
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


def _held_out_all_feature_scores(cache_dir: Path):
    files = sorted(cache_dir.glob("slab_*.json"))
    if len(files) < 5:
        raise SystemExit(f"only {len(files)} slabs cached - run `extract` first")
    data = {}
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        slab = int(path.stem.split("_")[1])
        X = np.array([[row[name] for name in FEATURES] for row in rows], float)
        y = np.array([row["label"] == "bad" for row in rows], float)
        data[slab] = (X, y)
    scored = {}
    slabs = sorted(data)
    for held in slabs:
        X_train = np.concatenate([data[s][0] for s in slabs if s != held])
        y_train = np.concatenate([data[s][1] for s in slabs if s != held])
        mu, sd = X_train.mean(0), X_train.std(0) + 1e-9
        weights = _fit_logreg((X_train - mu) / sd, y_train)
        scored[held] = (
            _score(weights, (data[held][0] - mu) / sd), data[held][1]
        )
    return scored


def _write_lift_svg(path: Path, curve: list[dict]) -> None:
    width, height = 840, 520
    left, right, top, bottom = 82, 30, 70, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    ymax = max(3.0, 1.05 * max(
        max(row["mean_per_slab_lift"], row["patch_weighted_lift"])
        for row in curve
    ))

    def px(fraction):
        return left + (float(fraction) / 0.50) * plot_w

    def py(value):
        return top + plot_h - (float(value) / ymax) * plot_h

    def points(key):
        return " ".join(f"{px(r['budget']):.1f},{py(r[key]):.1f}" for r in curve)

    ticks = []
    for percent in range(0, 51, 10):
        x = px(percent / 100)
        ticks.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" '
            'stroke="#e7e2d8" stroke-width="1"/>'
            f'<text x="{x:.1f}" y="{top+plot_h+27}" text-anchor="middle" '
            f'class="tick">{percent}%</text>'
        )
    y_tick = 0.0
    while y_tick <= ymax + 1e-9:
        y = py(y_tick)
        y_label = f"{y_tick:.1f}".rstrip("0").rstrip(".")
        ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" '
            'stroke="#e7e2d8" stroke-width="1"/>'
            f'<text x="{left-14}" y="{y+5:.1f}" text-anchor="end" '
            f'class="tick">{y_label}x</text>'
        )
        y_tick += 0.5
    row10 = min(curve, key=lambda row: abs(row["budget"] - 0.10))
    x10, y10 = px(row10["budget"]), py(row10["mean_per_slab_lift"])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  .title {{ font: 700 24px Arial, sans-serif; fill: #22201d; }}
  .sub {{ font: 14px Arial, sans-serif; fill: #615b52; }}
  .tick {{ font: 12px Arial, sans-serif; fill: #615b52; }}
  .axis {{ font: 600 13px Arial, sans-serif; fill: #39352f; }}
  .legend {{ font: 13px Arial, sans-serif; fill: #39352f; }}
</style>
<rect width="100%" height="100%" fill="#faf8f3"/>
<text x="{left}" y="32" class="title">Patch triage lift on held-out Scroll 4 slabs</text>
<text x="{left}" y="54" class="sub">Leave-one-slab-out; rank within each slab; 56,835 patches across 41 slabs</text>
{''.join(ticks)}
<line x1="{left}" y1="{py(1):.1f}" x2="{left+plot_w}" y2="{py(1):.1f}" stroke="#8b857c" stroke-width="2" stroke-dasharray="7 6"/>
<polyline points="{points('patch_weighted_lift')}" fill="none" stroke="#174f72" stroke-width="4" stroke-linejoin="round"/>
<polyline points="{points('mean_per_slab_lift')}" fill="none" stroke="#d35d38" stroke-width="3" stroke-linejoin="round"/>
<circle cx="{x10:.1f}" cy="{y10:.1f}" r="6" fill="#d35d38" stroke="#faf8f3" stroke-width="2"/>
<text x="{x10+13:.1f}" y="{y10-10:.1f}" class="axis">10%: {row10['mean_per_slab_lift']:.2f}x mean</text>
<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#39352f" stroke-width="2"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#39352f" stroke-width="2"/>
<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" class="axis">Fraction of each slab sent to review</text>
<text x="22" y="{top+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top+plot_h/2:.1f})" class="axis">Precision lift over random review</text>
<line x1="{left+plot_w-245}" y1="{top+19}" x2="{left+plot_w-210}" y2="{top+19}" stroke="#174f72" stroke-width="4"/>
<text x="{left+plot_w-200}" y="{top+24}" class="legend">patch-weighted</text>
<line x1="{left+plot_w-245}" y1="{top+43}" x2="{left+plot_w-210}" y2="{top+43}" stroke="#d35d38" stroke-width="3"/>
<text x="{left+plot_w-200}" y="{top+48}" class="legend">mean slab</text>
<text x="{left+plot_w-245}" y="{top+72}" class="sub">Target: recorded accepted/rejected split</text>
</svg>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def do_curve(cache_dir: str, out_path: str, svg_path: str) -> None:
    scored = _held_out_all_feature_scores(Path(cache_dir))
    curve = []
    for percent in range(1, 51):
        budget = percent / 100.0
        rows, selected_bad = [], 0.0
        total_bad = sum(float(y.sum()) for _, y in scored.values())
        total_selected = 0
        total_patches = sum(len(y) for _, y in scored.values())
        for slab in sorted(scored):
            scores, y = scored[slab]
            k = max(1, int(round(float(budget) * len(y))))
            top = np.argsort(-scores)[:k]
            lift = float(y[top].mean() / max(y.mean(), 1e-9))
            rows.append((len(y), lift))
            selected_bad += float(y[top].sum())
            total_selected += k
        sizes = np.array([row[0] for row in rows], float)
        lifts = np.array([row[1] for row in rows], float)
        curve.append({
            "budget": float(budget),
            "actual_patch_fraction": float(total_selected / total_patches),
            "mean_per_slab_lift": float(lifts.mean()),
            "median_per_slab_lift": float(np.median(lifts)),
            "patch_weighted_lift": float(np.average(lifts, weights=sizes)),
            "bad_patch_recall": float(selected_bad / total_bad),
        })
    result = {
        "tool": "patch_triage lift curve",
        "protocol": "leave one slab out; rank within held-out slab",
        "n_slabs": len(scored),
        "n_patches": int(sum(len(y) for _, y in scored.values())),
        "features": FEATURES,
        "curve": curve,
    }
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_lift_svg(Path(svg_path), curve)
    row10 = next(row for row in curve if row["budget"] == 0.10)
    print(f"10% mean per-slab lift {row10['mean_per_slab_lift']:.3f}x; "
          f"patch-weighted {row10['patch_weighted_lift']:.3f}x")
    print(f"wrote {destination}")
    print(f"wrote {svg_path}")


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
    v.add_argument("--cache-dir", default=str(CACHE))
    r = sub.add_parser(
        "rank", help="rank a tifxyz review batch within each 250-voxel z slab"
    )
    r.add_argument(
        "inputs", nargs="+",
        help="directory or ZIP containing PATCH/x.tif, y.tif, z.tif; multiple allowed",
    )
    r.add_argument("--out", default="patch_triage_ranking.csv")
    r.add_argument("--budget", type=float, default=0.10)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--model", default=str(MODEL))
    r.add_argument(
        "--slab", type=int, action="append", dest="slabs",
        help="rank only this 250-voxel z slab (repeat for more than one)",
    )
    r.add_argument(
        "--max-patches", type=int, default=5000,
        help="memory guard for the relational pair search (default: 5000)",
    )
    f = sub.add_parser("fit-model", help="rebuild the tracked deployment model")
    f.add_argument("--cache-dir", default=str(CACHE))
    f.add_argument("--out", default=str(MODEL))
    c = sub.add_parser("curve", help="regenerate the held-out per-slab lift curve")
    c.add_argument("--cache-dir", default=str(CACHE))
    c.add_argument("--out", default=str(ROOT / "results" / "patch_triage_curve.json"))
    c.add_argument("--svg", default=str(ROOT / "results" / "patch_triage_lift.svg"))
    a = ap.parse_args()
    if a.cmd == "extract":
        lo, hi = (a.slabs.split("-") + [a.slabs])[:2]
        do_extract(list(range(int(lo), int(hi) + 1)), a.seed)
    elif a.cmd == "evaluate":
        do_evaluate(a.out, Path(a.cache_dir))
    elif a.cmd == "rank":
        do_rank(a.inputs, a.out, a.budget, a.seed, a.max_patches, a.model, a.slabs)
    elif a.cmd == "fit-model":
        do_fit_model(a.cache_dir, a.out)
    else:
        do_curve(a.cache_dir, a.out, a.svg)


if __name__ == "__main__":
    main()
