"""Arms A and B of PREREGISTER_margin.md, run under the split committed before training.

    arm A  baseline        data/kaggle/labels          (published, unmodified)
    arm B  margin ignored  data/kaggle/labels_margin   (margin 0 -> 2, margin_relabel.py)

⚠ THE TRAP THIS FILE IS BUILT AROUND. The intervention edits the labels, so scoring against
the edited labels proves nothing. **Training reads the arm's labels; evaluation always reads
`data/kaggle/labels`**, whichever arm produced the weights. `LABEL_DIRS` is consulted in
exactly one place (`Volumes` construction for train/val) and `EVAL_LABELS` is a separate
constant that no arm can reach. Endpoint code never sees the arm name.

Endpoints are the four registered in section 5, per volume, on the 174 held-out volumes that
locate on Scroll1A:

  primary     recall over class-1 voxels, class 2 excluded, sigmoid(l1-l0) > 0.2
              -- the rule bench_m7_recall.py applies to the published model today
  co-primary  surface localisation error: |offset| from a predicted-sheet voxel to the CT
              ridge along the across-sheet normal. Label-free and thickness-free.
  secondary   predicted positive fraction over the scored region
  guardrail   share of predicted sheet sitting on CT that is identically zero

  python margin_arms.py --arm A --seed 0
  python margin_arms.py --compare
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import tifffile

from ablate_faint_sheet import build_model, set_seed, soft_dice
from thin_labels import across_sheet_dirs

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
SPLIT = ROOT / "vesuvius-repro" / "results" / "margin_split.json"
RESULTS = ROOT / "results" / "margin_arms"

# Training labels, per arm. The ONLY place the arm changes anything.
LABEL_DIRS = {"A": ROOT / "data" / "kaggle" / "labels",
              "B": ROOT / "data" / "kaggle" / "labels_margin"}
# Evaluation labels. Deliberately a separate constant with no arm dependence.
EVAL_LABELS = ROOT / "data" / "kaggle" / "labels"

CROP = 128
THRESH = 0.2          # the published m7 threshold, as bench_m7_recall.py uses it
TRIM = 32             # drop the volume boundary before scoring, as the benchmark does
STRIDE = 96           # 320 = 128 + 3 windows at stride 96, so every voxel is covered
POOL_SIZE = 64        # volume pairs held decompressed; ~4 GB, see Volumes
POOL_REFRESH = 8      # of those, swapped for fresh draws at each epoch boundary
LOC_SAMPLE = 1500     # predicted-sheet voxels sampled per volume for localisation error
HALF, STEP = 4.0, 0.25   # profile half-width / step, as in margin_class_scale.py


class Volumes:
    """A bounded, rotating RAM pool over the training list.

    ⚠ WHY NOT MEMORY-MAPPED, despite what the July ablation's harness claims. These TIFFs
    are LZW-compressed (compression tag 5, 320 pages each), and `tifffile.memmap` raises
    `image data are not memory-mappable` on every one of them. The July `Volumes._open`
    catches that and falls back to `tifffile.imread`, so a class documented as "memory-
    mapped, not read into RAM" in fact reads every volume into RAM. At 96 volumes that was
    3 GB and invisible; at 581 it took this box to 1.2 GB free before it was killed, on a
    machine that shares its RAM with another training job.

    So: hold POOL_SIZE volume pairs decompressed, and swap POOL_REFRESH of them for fresh
    draws at each epoch boundary. RAM is bounded at ~4 GB regardless of the set size, and
    over 80 epochs the run still draws on far more of the 581 than a fixed subset would.
    Both arms use the same pool size, refresh rate and seed, so the sampling is identical
    between them and cannot favour either.
    """

    def __init__(self, names: list[str], label_dir: Path, seed: int = 0,
                 size: int = POOL_SIZE):
        self.names, self.label_dir = list(names), label_dir
        self.rng = random.Random(seed)
        self.size = min(size, len(self.names))
        self.slots: list[int] = []
        self.img: list[np.ndarray] = []
        self.lab: list[np.ndarray] = []
        for _ in range(self.size):
            self._load_into(len(self.img))
        self.n_loads = self.size

    def _pick(self) -> int:
        """Draw a volume not currently resident, so the pool never holds duplicates."""
        for _ in range(200):
            k = self.rng.randrange(len(self.names))
            if k not in self.slots:
                return k
        return self.rng.randrange(len(self.names))

    def _load_into(self, slot: int) -> None:
        k = self._pick()
        nm = self.names[k]
        im = np.asarray(tifffile.imread(str(IMAGES / f"{nm}.tif")))
        lb = np.asarray(tifffile.imread(str(self.label_dir / f"{nm}.tif")))
        if slot < len(self.img):
            self.slots[slot], self.img[slot], self.lab[slot] = k, im, lb
        else:
            self.slots.append(k); self.img.append(im); self.lab.append(lb)

    def refresh(self, n: int = POOL_REFRESH) -> None:
        for _ in range(min(n, self.size)):
            self._load_into(self.rng.randrange(self.size))
            self.n_loads += 1

    def __len__(self) -> int:
        return len(self.img)

    def crop(self, rng: random.Random, size: int = CROP):
        k = rng.randrange(len(self.img))
        v = self.img[k]
        s = tuple(slice(o, o + size) for o in
                  (rng.randrange(0, v.shape[a] - size + 1) for a in range(3)))
        im = self.img[k][s].astype(np.float32) / 255.0
        lb = self.lab[k][s].astype(np.int64)
        for ax in (0, 1, 2):
            if rng.random() < 0.5:
                im, lb = np.flip(im, ax), np.flip(lb, ax)
        return np.ascontiguousarray(im), np.ascontiguousarray(lb)


def class_weights(vols: Volumes, max_volumes: int = 24) -> np.ndarray:
    """Inverse-frequency weights, from the arm's OWN labels.

    Arm B moves ~44% of a sheet volume out of background and into ignore, so its class
    frequencies genuinely differ. Freezing arm A's weights onto arm B would be a second,
    unregistered intervention; letting each arm weight its own label distribution is what
    'train on these labels' means. Both weight vectors are recorded in the result file.
    """
    counts = np.zeros(3, dtype=np.float64)
    for lb in vols.lab[:max_volumes]:
        counts += np.bincount(np.asarray(lb).ravel(), minlength=3)[:3]
    freq = counts / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-6)
    w = w / w.mean()
    print(f"  class frequencies {np.round(freq, 4).tolist()} -> "
          f"weights {np.round(w, 3).tolist()}", flush=True)
    return w.astype(np.float32), freq.tolist()


def predict_volume(model, ct: np.ndarray, device, size: int = CROP, stride: int = STRIDE):
    """Sliding-window logits over the whole volume, overlap-averaged.

    Averaging matters: a single centred window would leave most of a 320^3 volume unscored,
    and corner-only windows (what the July ablation used on its val set) miss the middle
    entirely. Every voxel here is covered by at least one window.
    """
    import torch

    acc = np.zeros((3,) + ct.shape, dtype=np.float32)
    cnt = np.zeros(ct.shape, dtype=np.float32)
    starts = [sorted({*range(0, ct.shape[a] - size + 1, stride), ct.shape[a] - size})
              for a in range(3)]
    with torch.no_grad():
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    s = (slice(z, z + size), slice(y, y + size), slice(x, x + size))
                    patch = ct[s].astype(np.float32) / 255.0
                    t = torch.from_numpy(patch)[None, None].to(device)
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=device == "cuda"):
                        out = model(t)
                    acc[(slice(None),) + s] += out[0].float().cpu().numpy()
                    cnt[s] += 1.0
    return acc / np.maximum(cnt, 1.0)[None]


def localisation_error(ct: np.ndarray, pred: np.ndarray, sl: tuple, rng) -> float | None:
    """Median |offset| from predicted sheet to the CT ridge, along the across-sheet normal.

    Thickness-free and label-free, so neither arm's relabelling can touch it: it asks only
    whether the prediction sits where the CT actually has a sheet. Sampling the profile at
    STEP = 0.25 vox means the answer is quantised to a quarter voxel, which is well inside
    the effect this study is looking for.
    """
    from scipy.ndimage import map_coordinates

    pts = np.argwhere(pred)
    if len(pts) < 200:
        return None
    pts = pts[rng.choice(len(pts), size=min(LOC_SAMPLE, len(pts)), replace=False)]
    pts = (pts + np.array([sl[0].start, sl[1].start, sl[2].start])).astype(np.int32)

    ts = np.arange(-HALF, HALF + 1e-9, STEP, dtype=np.float32)
    dirs = across_sheet_dirs(ct, pts)
    coords = pts[:, :, None].astype(np.float32) + dirs[:, :, None] * ts[None, None, :]
    prof = map_coordinates(ct.astype(np.float32),
                           coords.transpose(1, 0, 2).reshape(3, -1),
                           order=1, mode="nearest").reshape(len(pts), len(ts))
    return float(np.median(np.abs(ts[np.argmax(prof, axis=1)])))


def score_volume(model, name: str, device, rng) -> dict:
    """Every endpoint for one test volume. Reads EVAL_LABELS, never the arm's labels."""
    ct = np.asarray(tifffile.imread(str(IMAGES / f"{name}.tif")))
    gt = np.asarray(tifffile.imread(str(EVAL_LABELS / f"{name}.tif")))
    if ct.shape != gt.shape:
        return {"sample": name, "status": "shape_mismatch"}

    logits = predict_volume(model, ct, device)
    sl = tuple(slice(TRIM, s - TRIM) for s in ct.shape)
    l0, l1 = logits[0][sl], logits[1][sl]
    p = 1.0 / (1.0 + np.exp(-(l1 - l0)))
    g = gt[sl]

    sheet, ignore = g == 1, g == 2
    scored = ~ignore                       # class 2 is ~59% of a volume; it is not background
    if sheet.sum() == 0:
        return {"sample": name, "status": "no_sheet"}
    pred = p > THRESH

    tp = float((pred & sheet).sum())
    fn = float((~pred & sheet).sum())
    fp = float((pred & scored & ~sheet).sum())
    empty = ct[sl] == 0
    n_pred = float((pred & scored).sum())

    return {
        "sample": name, "status": "ok",
        "recall": tp / max(tp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "pred_positive_fraction": n_pred / max(float(scored.sum()), 1.0),
        "loc_error": localisation_error(ct, pred & scored, sl, rng),
        "pred_on_empty_ct": float((pred & empty).sum()) / max(n_pred, 1.0),
        "n_sheet": int(sheet.sum()),
    }


def train(arm: str, seed: int, epochs: int, iters: int, batch: int,
          n_train: int, n_test: int) -> dict:
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    sp = json.loads(SPLIT.read_text())
    tr_names = sp["train"][:n_train] if n_train else sp["train"]
    te_names = sp["test"][:n_test] if n_test else sp["test"]

    print(f"arm={arm} seed={seed} device={device} labels={LABEL_DIRS[arm].name} "
          f"train={len(tr_names)} test={len(te_names)}", flush=True)
    t0 = time.time()
    tr = Volumes(tr_names, LABEL_DIRS[arm], seed=seed)
    print(f"  pool of {len(tr)}/{len(tr_names)} volumes loaded in {time.time()-t0:.0f}s",
          flush=True)

    w_np, freq = class_weights(tr)
    w = torch.from_numpy(w_np).to(device)
    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ce = nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    rng = random.Random(seed)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for _ in range(iters):
            ims, lbs = [], []
            for _ in range(batch):
                im, lb = tr.crop(rng)
                ims.append(im); lbs.append(lb)
            x = torch.from_numpy(np.stack(ims))[:, None].to(device)
            y = torch.from_numpy(np.stack(lbs)).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
                loss = ce(logits, y) + soft_dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss.detach())
        if ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}  loss {tot/iters:.4f}  {time.time()-t0:.0f}s  "
                  f"(pool draws {tr.n_loads})", flush=True)
        if ep < epochs:
            tr.refresh()
    train_min = round((time.time() - t0) / 60, 1)

    model.eval()
    erng = np.random.default_rng(0)          # evaluation sampling is seed-independent
    rows, t0 = [], time.time()
    for k, nm in enumerate(te_names):
        rows.append(score_volume(model, nm, device, erng))
        if k % 20 == 0:
            print(f"  eval [{k}/{len(te_names)}] {time.time()-t0:.0f}s", flush=True)
    ok = [r for r in rows if r.get("status") == "ok"]

    med = lambda k: (round(float(np.median([r[k] for r in ok if r[k] is not None])), 5)
                     if any(r[k] is not None for r in ok) else None)
    rep = {
        "arm": arm, "seed": seed, "labels_used_for_training": LABEL_DIRS[arm].name,
        "labels_used_for_scoring": EVAL_LABELS.name,
        "epochs": epochs, "iters": iters, "batch": batch,
        "n_train": len(tr_names), "n_test_scored": len(ok),
        "pool_size": POOL_SIZE, "pool_refresh_per_epoch": POOL_REFRESH,
        "pool_total_volume_loads": tr.n_loads,
        "class_frequencies": [round(f, 5) for f in freq],
        "class_weights": [round(float(v), 4) for v in w_np],
        "train_minutes": train_min, "eval_minutes": round((time.time() - t0) / 60, 1),
        "median_recall": med("recall"), "median_precision": med("precision"),
        "median_pred_positive_fraction": med("pred_positive_fraction"),
        "median_loc_error": med("loc_error"),
        "median_pred_on_empty_ct": med("pred_on_empty_ct"),
        "rows": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{arm}_seed{seed}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"  -> {out}")
    print(f"  median recall {rep['median_recall']}  prec {rep['median_precision']}  "
          f"loc {rep['median_loc_error']}  empty-CT {rep['median_pred_on_empty_ct']}",
          flush=True)
    return rep


def compare() -> None:
    """Arm B against arm A, paired per volume, under the registered decision rule."""
    from scipy.stats import wilcoxon

    runs = {"A": [], "B": []}
    for p in sorted(RESULTS.glob("*_seed*.json")):
        r = json.loads(p.read_text())
        runs.setdefault(r["arm"], []).append(r)
    if not runs["A"] or not runs["B"]:
        raise SystemExit(f"need both arms in {RESULTS}; have "
                         f"{ {k: len(v) for k, v in runs.items()} }")
    print("seeds per arm: " + ", ".join(f"{k}={len(v)}" for k, v in runs.items()))

    # Per volume, average over seeds first: the unit of analysis is the volume, and the
    # seeds are repeats of the same arm, not independent observations of a volume.
    def by_volume(arm: str, key: str) -> dict:
        acc = {}
        for r in runs[arm]:
            for row in r["rows"]:
                if row.get("status") == "ok" and row.get(key) is not None:
                    acc.setdefault(row["sample"], []).append(row[key])
        return {k: float(np.mean(v)) for k, v in acc.items()}

    print(f"\n{'endpoint':<28}{'A':>10}{'B':>10}{'delta':>10}{'p':>10}  verdict")
    for key, name, floor in (("recall", "PRIMARY median recall", 0.01),
                             ("loc_error", "CO-PRIM loc error (vox)", None),
                             ("pred_positive_fraction", "secondary pred-pos frac", None),
                             ("pred_on_empty_ct", "GUARDRAIL empty-CT FP", None),
                             ("precision", "precision", None)):
        a, b = by_volume("A", key), by_volume("B", key)
        common = sorted(set(a) & set(b))
        if len(common) < 10:
            print(f"{name:<28}{'--':>10}{'--':>10}  too few paired volumes")
            continue
        va = np.array([a[k] for k in common])
        vb = np.array([b[k] for k in common])
        d = float(np.median(vb) - np.median(va))
        try:
            p = float(wilcoxon(vb, va)[1])
        except ValueError:
            p = float("nan")
        if floor is not None and abs(d) < floor:
            verdict = f"NULL (below {floor} floor)"
        elif p < 0.05:
            verdict = "B better" if d > 0 else "B worse"
        else:
            verdict = "not significant"
        print(f"{name:<28}{np.median(va):>10.4f}{np.median(vb):>10.4f}"
              f"{d:>+10.4f}{p:>10.2e}  {verdict}")

    ga, gb = by_volume("A", "pred_on_empty_ct"), by_volume("B", "pred_on_empty_ct")
    common = sorted(set(ga) & set(gb))
    if common:
        rise = float(np.median([gb[k] for k in common]) - np.median([ga[k] for k in common]))
        print(f"\nGUARDRAIL (registered): B fails outright if empty-CT false positives rise "
              f"by\nmore than 2 points. Rise = {rise:+.4f} -> "
              f"{'FIRED, arm B fails' if rise > 0.02 else 'not fired'}")
    print("\nRegistered magnitude floor: a median recall gain below 0.01 is reported as null\n"
          "regardless of p. Abandon condition 4 -- no extra seeds, no second margin width.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["A", "B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=0, help="0 = all 581")
    ap.add_argument("--n-test", type=int, default=0, help="0 = all 174")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.arm:
        train(a.arm, a.seed, a.epochs, a.iters, a.batch, a.n_train, a.n_test)
    else:
        ap.error("pass --arm or --compare")


if __name__ == "__main__":
    main()
