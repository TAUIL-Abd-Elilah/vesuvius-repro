"""Where does a failed reproduction disagree? Structure discriminates the cause.

When verify_region.py rules out calibration, several structural explanations remain
and they leave different fingerprints:

  * patch grid / blending  -> disagreement peaks at particular offsets within the
    patch stride, because that is where the Gaussian weights differ most.
  * region-edge effects    -> disagreement rises towards the faces of the scored box.
  * different input data   -> disagreement is spread roughly uniformly, and tracks
    the CT rather than the patch layout.

This prints all three profiles for a region already scored, so the next experiment
is chosen by evidence instead of by guessing.

Usage:
    python diagnose_structure.py --scroll PHercParis4 --ours outputs/paris4/merged.zarr \
        --bbox 6600:6856,3200:3456,4032:4288 --trim 64 --patch 192 --step 0.5
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import zarr

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def retry(fn, attempts: int = 8):
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--trim", type=int, default=64)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--catalog", default="catalog.json")
    args = ap.parse_args()

    rows = json.load(open(args.catalog))
    row = next(r for r in rows if r.get("scroll") == args.scroll and r.get("prediction"))
    pub_url = (f"{BUCKET}/{args.scroll}/representations/predictions/surfaces/"
               f"{row['prediction']}/0")
    ct_url = f"{BUCKET}/{args.scroll}/volumes/{row['ct_volume']}/{row['declared_level']}"
    thr = row["threshold"]

    b = [int(v) for p in args.bbox.split(",") for v in p.split(":")]
    z0, z1, y0, y1, x0, x1 = b
    m = args.trim
    iz, iy, ix = slice(z0 + m, z1 - m), slice(y0 + m, y1 - m), slice(x0 + m, x1 - m)

    pub = retry(lambda: np.asarray(zarr.open(pub_url, mode="r")[iz, iy, ix])) > 0
    arr = zarr.open(args.ours, mode="r")
    l0 = np.asarray(arr[0, iz, iy, ix]).astype(np.float32)
    l1 = np.asarray(arr[1, iz, iy, ix]).astype(np.float32)
    ours = (1.0 / (1.0 + np.exp(-(l1 - l0)))) > thr
    dis = ours != pub
    print(f"{args.scroll}: {100*dis.mean():.3f}% disagreement over {dis.shape}")

    # --- 1. per-axis profile: is it localised to a slab? ---
    print("\n--- disagreement per axis (deciles of the scored box) ---")
    for ax, name in enumerate("zyx"):
        prof = dis.mean(axis=tuple(a for a in range(3) if a != ax))
        n = len(prof)
        dec = [100 * prof[i * n // 10:(i + 1) * n // 10].mean() for i in range(10)]
        print(f"  {name}: " + " ".join(f"{d:5.2f}" for d in dec))

    # --- 2. distance from the scored-box face: residual edge effect? ---
    print("\n--- disagreement vs distance from the nearest face ---")
    zz, yy, xx = np.meshgrid(*[np.arange(s) for s in dis.shape], indexing="ij")
    d_face = np.minimum.reduce([
        zz, dis.shape[0] - 1 - zz, yy, dis.shape[1] - 1 - yy, xx, dis.shape[2] - 1 - xx])
    for lo, hi in ((0, 8), (8, 16), (16, 32), (32, 48), (48, 999)):
        sel = (d_face >= lo) & (d_face < hi)
        if sel.any():
            print(f"  {lo:3}-{hi if hi<999 else '+':>3} voxels in: "
                  f"{100*dis[sel].mean():5.2f}%  ({sel.sum():,} voxels)")

    # --- 3. phase within the patch stride: grid/blending fingerprint? ---
    stride = int(round(args.patch * args.step))
    print(f"\n--- disagreement vs position modulo the patch stride ({stride}) ---")
    for ax, name in enumerate("zyx"):
        origin = (z0, y0, x0)[ax] + m
        idx = (np.arange(dis.shape[ax]) + origin) % stride
        prof = dis.mean(axis=tuple(a for a in range(3) if a != ax))
        bins = [prof[idx // (stride // 8) == k].mean() for k in range(8)]
        print(f"  {name}: " + " ".join(f"{100*v:5.2f}" for v in bins))

    # --- 4. does the disagreement track the CT, i.e. is it about the input? ---
    print("\n--- disagreement vs CT intensity ---")
    ct = retry(lambda: np.asarray(zarr.open(ct_url, mode="r")[iz, iy, ix])).astype(np.float32)
    qs = np.percentile(ct, [0, 20, 40, 60, 80, 100])
    for i in range(5):
        sel = (ct >= qs[i]) & (ct <= qs[i + 1])
        if sel.any():
            print(f"  CT [{qs[i]:6.1f},{qs[i+1]:6.1f}]: {100*dis[sel].mean():5.2f}%  "
                  f"({sel.sum():,} voxels)")


if __name__ == "__main__":
    main()
