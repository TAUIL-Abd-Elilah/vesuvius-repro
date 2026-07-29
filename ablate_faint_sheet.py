"""Does simulating faint sheets during training recover the sheets m7 misses?

WHY THIS EXPERIMENT EXISTS. Measuring where the published m7 model loses recall across
all 892 public volumes gave a specific answer: missed sheet voxels are about 10.3% darker
than found sheet voxels *inside the same volume* (median CT 96 vs 107), in 161 of 201
volumes, and the effect is graded by severity - volumes under 70% recall show -0.708 sigma
against -0.163 sigma for volumes over 90%. Local sheet thickness and component size show
no difference at all. Posted as ScrollPrize/villa#191.

That is a measurement, not a fix, and it implies one: if the model misses sheets because
they are faint, then training on artificially faintened sheets should recover them. This
is the controlled two-arm test of that implication.

  baseline : random flips only
  faint    : random flips, plus a local intensity attenuation applied to the sheet and
             its immediate surround, which manufactures the failure case on demand

Both arms share architecture, seed, split, schedule and segmentation head. The only
difference is the augmentation.

WHY THE ATTENUATION IS BLURRED, WHICH IS THE ONLY SUBTLE PART. Attenuating exactly the
class-1 mask would stamp an intensity edge along the label boundary, and the model could
then find sheets by looking for that edge - the augmentation would leak the answer and
the arm would "win" for a reason that does not transfer. The mask is therefore Gaussian
blurred before it modulates intensity, so the change is a smooth local density dip with
no label-aligned step. See faint_sheet().

WHAT IS ACTUALLY MEASURED. Overall class-1 Dice is reported, but the hypothesis is about
a subpopulation, so the headline is recall stratified by how bright the sheet is:
GT sheet voxels are split into brightness terciles per volume and recall is reported for
each. The prediction the experiment makes is a gain concentrated in the DARK tercile. A
uniform gain would mean the augmentation just acted as generic regularisation, and a gain
only in the bright tercile would contradict the hypothesis outright.

Usage:
    python ablate_faint_sheet.py --arm baseline --seed 0
    python ablate_faint_sheet.py --arm faint    --seed 0
    python ablate_faint_sheet.py --compare
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np

SEED = 0
CROP = 128
VAL_FRACTION = 0.22
N_VOLUMES = 96           # capped: the whole set is 892 and Volumes holds them in RAM
FAINT_PROB = 0.5         # fraction of crops that get the attenuation
FAINT_MAX = 0.45         # strongest attenuation, as a fraction of local intensity
FAINT_SIGMA = 2.5        # blur applied to the mask before it modulates intensity
CONTACT_R = 3            # gap thickness, in voxels, that counts as "sheets in contact"
RESULTS = Path("results/ablation_faint")


def set_seed(s: int) -> None:
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _shape(path: str):
    import tifffile
    try:
        with tifffile.TiffFile(path) as tf:
            return tuple(tf.series[0].shape)
    except Exception:  # noqa: BLE001
        return None


def _intact(path: str) -> bool:
    """A truncated TIFF still opens and reshapes, so validate the shape, not existence."""
    s = _shape(path)
    return s is not None and len(s) == 3 and len(set(s)) == 1


def load_split(n_volumes: int) -> tuple[list, list]:
    imgs = sorted(glob.glob("data/kaggle/images/*.tif"))
    labs = sorted(glob.glob("data/kaggle/labels/*.tif"))
    pairs = [(i, l) for i, l in zip(imgs, labs)
             if os.path.basename(i) == os.path.basename(l)]
    if not pairs:
        raise SystemExit("no image/label pairs under data/kaggle")

    # Subsample BEFORE the shape check so the check costs O(n_volumes), not O(892).
    rng = random.Random(SEED)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    picked, bad = [], 0
    for i in idx:
        if len(picked) >= n_volumes:
            break
        ip, lp = pairs[i]
        if _intact(ip) and _intact(lp):
            picked.append(pairs[i])
        else:
            bad += 1
    if bad:
        print(f"  skipped {bad} truncated pair(s)")
    n_val = max(1, int(round(len(picked) * VAL_FRACTION)))
    return picked[n_val:], picked[:n_val]


def contact_mask(m: np.ndarray) -> np.ndarray:
    """Sheet voxels sitting across a thin gap from another sheet.

    Why this exists: the ungated `faint` arm was a clean negative -- dark-tercile recall did
    not move while everything else degraded. @Jinhojeong's reading on villa#191 is that the
    attenuation is geometry-blind, so it also darkens contacts where intensity has already
    stopped separating adjacent sheets. Those cases are unresolvable by intensity *by
    construction*, so the arm spends capacity on them while wrecking the bright mass that
    was working. This mask is what the `faint_gated` arm refuses to touch.

    A morphological closing fills any gap thinner than CONTACT_R; whatever the closing adds
    was a thin gap, and the sheet voxels bordering it are the contacts.
    """
    from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure

    st = generate_binary_structure(3, 1)
    closed = binary_closing(m, structure=st, iterations=CONTACT_R)
    thin_gap = closed & ~m
    if not thin_gap.any():
        return np.zeros_like(m)
    return binary_dilation(thin_gap, structure=st, iterations=CONTACT_R) & m


def faint_sheet(im: np.ndarray, lb: np.ndarray, rng: random.Random,
                gated: bool = False) -> np.ndarray:
    """Attenuate the sheet and its surround, simulating a faint sheet.

    The measured failure population is sheet voxels roughly 10% darker than the ones the
    model finds, so the attenuation is drawn up to FAINT_MAX and applied multiplicatively.
    The mask is blurred first: a hard label-shaped edge would let the model locate sheets
    by the artefact rather than by the papyrus.

    `gated=True` removes contact regions from the mask before blurring -- attenuate only
    where intensity still has a chance of separating the sheets. Note the blur is applied
    *after* gating, so the gated arm still has no hard label-shaped edge; the leak
    mitigation is unchanged.
    """
    from scipy.ndimage import gaussian_filter

    m = (lb == 1)
    if not m.any():
        return im
    if gated:
        m = m & ~contact_mask(m)
        if not m.any():
            return im
    soft = gaussian_filter(m.astype(np.float32), FAINT_SIGMA)
    peak = float(soft.max())
    if peak <= 0:
        return im
    soft /= peak                                  # 1 at the sheet core, decaying outward
    strength = rng.uniform(0.15, FAINT_MAX)
    return np.clip(im * (1.0 - strength * soft), 0.0, 1.0)


class Volumes:
    """Volumes are memory-mapped, not read into RAM.

    Holding 96 volumes resident costs ~6 GB and this box shares its RAM with other
    training jobs; an earlier version died on `Unable to allocate 32 MiB` mid-epoch, with
    the GPU barely touched. Crops are small and random, so the OS page cache serves the
    hot parts and dataset size stops being a memory decision. Falls back to a real read
    for anything tifffile cannot map.
    """

    def __init__(self, pairs):
        import tifffile
        self.img, self.lab = [], []
        for ip, lp in pairs:
            self.img.append(self._open(tifffile, ip))
            self.lab.append(self._open(tifffile, lp))

    @staticmethod
    def _open(tifffile, path):
        try:
            return tifffile.memmap(path, mode="r")
        except Exception:  # noqa: BLE001 - compressed or non-contiguous TIFF
            return tifffile.imread(path)

    def __len__(self) -> int:
        return len(self.img)

    def crop(self, rng: random.Random, arm: str, size: int = CROP):
        k = rng.randrange(len(self.img))
        v = self.img[k]
        z = rng.randrange(0, v.shape[0] - size + 1)
        y = rng.randrange(0, v.shape[1] - size + 1)
        x = rng.randrange(0, v.shape[2] - size + 1)
        s = (slice(z, z + size), slice(y, y + size), slice(x, x + size))
        im = self.img[k][s].astype(np.float32) / 255.0
        lb = self.lab[k][s].astype(np.int64)
        for ax in (0, 1, 2):                      # shared by both arms
            if rng.random() < 0.5:
                im, lb = np.flip(im, ax), np.flip(lb, ax)
        im, lb = np.ascontiguousarray(im), np.ascontiguousarray(lb)
        if arm in ("faint", "faint_gated") and rng.random() < FAINT_PROB:
            im = np.ascontiguousarray(
                faint_sheet(im, lb, rng, gated=(arm == "faint_gated")))
        return im, lb


def build_model():
    import torch
    import torch.nn as nn

    def block(i, o):
        return nn.Sequential(
            nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.01, True),
            nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o, affine=True), nn.LeakyReLU(0.01, True))

    class UNet(nn.Module):
        def __init__(self, ch=(24, 48, 96, 192)):
            super().__init__()
            self.e = nn.ModuleList()
            prev = 1
            for c in ch:
                self.e.append(block(prev, c))
                prev = c
            self.pool = nn.MaxPool3d(2)
            self.up = nn.ModuleList(
                [nn.ConvTranspose3d(ch[i + 1], ch[i], 2, 2) for i in range(len(ch) - 1)])
            self.d = nn.ModuleList(
                [block(ch[i] * 2, ch[i]) for i in range(len(ch) - 1)])
            self.head = nn.Conv3d(ch[0], 3, 1)

        def forward(self, x):
            feats = []
            for i, e in enumerate(self.e):
                x = e(x)
                if i < len(self.e) - 1:
                    feats.append(x)
                    x = self.pool(x)
            for i in range(len(self.d) - 1, -1, -1):
                x = self.up[i](x)
                x = self.d[i](torch.cat([x, feats[i]], 1))
            return self.head(x)

    return UNet()


def class_weights(vols: Volumes, max_volumes: int = 24) -> np.ndarray:
    """Inverse-frequency weights. Without these the model collapses to the majority class.

    Estimated from a subset: counting every voxel of every volume would pull the whole
    memory-mapped set through RAM, which is exactly what the mapping avoids. Frequencies
    this coarse are stable well before 24 volumes.
    """
    counts = np.zeros(3, dtype=np.float64)
    for lb in vols.lab[:max_volumes]:
        counts += np.bincount(np.asarray(lb).ravel(), minlength=3)[:3]
    freq = counts / counts.sum()
    w = 1.0 / np.maximum(freq, 1e-6)
    w = w / w.mean()
    print(f"  class frequencies {np.round(freq, 3).tolist()} -> weights {np.round(w, 3).tolist()}")
    return w.astype(np.float32)


def soft_dice(logits, target, eps: float = 1e-5):
    import torch
    import torch.nn.functional as F
    p = F.softmax(logits, 1)
    t = F.one_hot(target, 3).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    num = 2 * (p * t).sum(dims) + eps
    den = p.sum(dims) + t.sum(dims) + eps
    return 1 - (num / den).mean()


def eval_windows(shape, size: int):
    out = []
    for z in (0, shape[0] - size):
        for y in (0, shape[1] - size):
            for x in (0, shape[2] - size):
                out.append((slice(z, z + size), slice(y, y + size), slice(x, x + size)))
    return out


def evaluate(model, val: Volumes, device, size: int = CROP) -> dict:
    """Class-1 Dice, plus recall split by sheet brightness - the hypothesis lives there."""
    import torch
    import torch.nn.functional as F

    model.eval()
    inter = np.zeros(3); psum = np.zeros(3); tsum = np.zeros(3)
    hit = np.zeros(3); tot = np.zeros(3)          # dark / mid / bright GT sheet voxels
    n_windows = 0
    with torch.no_grad():
        for k in range(len(val)):
            v, lb = val.img[k], val.lab[k]
            for s in eval_windows(v.shape, size):
                raw = v[s]
                im = torch.from_numpy(raw.astype(np.float32) / 255.0)[None, None].to(device)
                gt = lb[s].astype(np.int64)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    logits = model(im)
                pred = F.softmax(logits.float(), 1).argmax(1)[0].cpu().numpy()
                for c_ in range(3):
                    p_, t_ = pred == c_, gt == c_
                    inter[c_] += (p_ & t_).sum(); psum[c_] += p_.sum(); tsum[c_] += t_.sum()

                sheet = gt == 1
                if sheet.sum() >= 30:
                    vals = raw[sheet]
                    lo, hi = np.percentile(vals, [33.3, 66.7])
                    found = pred[sheet] == 1
                    for b, sel in enumerate((vals <= lo,
                                             (vals > lo) & (vals <= hi),
                                             vals > hi)):
                        hit[b] += found[sel].sum(); tot[b] += sel.sum()
                n_windows += 1
    dice = (2 * inter / np.maximum(psum + tsum, 1)).tolist()
    rec = (hit / np.maximum(tot, 1)).tolist()
    model.train()
    return {
        "eval_windows": n_windows,
        "dice_per_class": dice,
        "dice_class1_sheet": dice[1],
        "dice_mean": float(np.mean(dice)),
        "recall_sheet_overall": float(hit.sum() / max(tot.sum(), 1)),
        "recall_dark": rec[0], "recall_mid": rec[1], "recall_bright": rec[2],
        "sheet_voxels_scored": int(tot.sum()),
    }


def train(arm: str, epochs: int, iters: int, batch: int, seed: int, n_volumes: int) -> dict:
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)
    tr_pairs, va_pairs = load_split(n_volumes)
    print(f"arm={arm} seed={seed} device={device} "
          f"train={len(tr_pairs)} val={len(va_pairs)} volumes")
    t_load = time.time()
    tr, va = Volumes(tr_pairs), Volumes(va_pairs)
    print(f"  loaded in {time.time()-t_load:.0f}s")

    w = torch.from_numpy(class_weights(tr)).to(device)
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
                im, lb = tr.crop(rng, arm)
                ims.append(im); lbs.append(lb)
            x = torch.from_numpy(np.stack(ims))[:, None].to(device)
            y = torch.from_numpy(np.stack(lbs)).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(x)
                loss = ce(logits, y) + soft_dice(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += float(loss)
        if ep % 10 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}  loss {tot/iters:.4f}  {time.time()-t0:.0f}s", flush=True)

    rep = {"arm": arm, "seed": seed, "epochs": epochs, "iters": iters, "batch": batch,
           "train_volumes": len(tr_pairs), "val_volumes": len(va_pairs),
           "faint_prob": FAINT_PROB if arm in ("faint", "faint_gated") else 0.0,
           "faint_max": FAINT_MAX if arm in ("faint", "faint_gated") else 0.0,
           "contact_r": CONTACT_R if arm == "faint_gated" else None,
           "minutes": round((time.time() - t0) / 60, 1)}
    rep.update(evaluate(model, va, device))
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{arm}_seed{seed}.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"  -> {out}")
    print(f"  sheet Dice {rep['dice_class1_sheet']:.4f}   recall dark/mid/bright "
          f"{rep['recall_dark']:.3f}/{rep['recall_mid']:.3f}/{rep['recall_bright']:.3f}")
    return rep


def compare() -> None:
    """Compare arms across seeds, and refuse to call a difference that noise explains."""
    runs = {}
    for p in sorted(RESULTS.glob("*_seed*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        runs.setdefault(r["arm"], []).append(r)
    if len(runs) < 2:
        raise SystemExit(f"need both arms in {RESULTS}; have {list(runs)}")

    keys = [("dice_class1_sheet", "sheet Dice"), ("recall_sheet_overall", "recall overall"),
            ("recall_dark", "recall DARK tercile"), ("recall_mid", "recall mid"),
            ("recall_bright", "recall bright")]
    print(f"seeds per arm: " + ", ".join(f"{a}={len(v)}" for a, v in runs.items()))
    print(f"\n{'metric':<22}{'baseline':>20}{'faint':>20}{'delta':>12}  verdict")
    for k, name in keys:
        b = np.array([r[k] for r in runs["baseline"]])
        f = np.array([r[k] for r in runs["faint"]])
        d = f.mean() - b.mean()
        spread = max(b.std(ddof=1) if len(b) > 1 else 0.0,
                     f.std(ddof=1) if len(f) > 1 else 0.0)
        verdict = ("not demonstrated" if abs(d) <= spread
                   else ("faint BETTER" if d > 0 else "faint WORSE"))
        print(f"{name:<22}{b.mean():>13.4f} ±{b.std(ddof=1) if len(b)>1 else 0:.4f}"
              f"{f.mean():>13.4f} ±{f.std(ddof=1) if len(f)>1 else 0:.4f}"
              f"{d:>+12.4f}  {verdict}")

    print("\nA difference within the across-seed spread is NOT a result. The hypothesis "
          "predicts\na gain concentrated in the DARK tercile; a uniform gain would mean "
          "generic regularisation,\nand a bright-only gain would contradict it.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["baseline", "faint", "faint_gated"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--volumes", type=int, default=N_VOLUMES)
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare:
        compare()
    elif a.arm:
        train(a.arm, a.epochs, a.iters, a.batch, a.seed, a.volumes)
    else:
        ap.error("pass --arm or --compare")


if __name__ == "__main__":
    main()
