"""Is prediction-over-empty-CT a boundary fringe, or sheet painted deep into nothing?

pred_over_empty_ct.py established that 30 of 36 published m7 surface predictions mark sheet
where the masked CT is identically zero, 39.7% of predicted sheet pooled. That number alone
does not say whether it matters, and there are two very different situations behind it:

  fringe   the mask boundary and the prediction boundary disagree by a voxel or two, which
           is an ordinary dilation/threshold artifact and nearly harmless - a consumer
           intersecting with CT > 0 loses almost nothing.
  deep     the model asserts sheet far out into unscanned space, in which case whole
           predicted structures exist with no evidence under them, and anything built on
           the predictions - instance counts, connectivity, spacing - inherits them.

The discriminator is distance. For every phantom voxel (prediction > 0 AND CT == 0) we
measure the euclidean distance to the nearest voxel that HAS scan data, and report the
distribution. A fringe artifact sits at 1-2 voxels. Anything with mass at 8+ voxels is the
model predicting into nothing.

Also reported, because it is what downstream tools actually consume: connected components
of the prediction, and what fraction of them are wholly phantom. A component that is 100%
over empty CT is a sheet instance that does not exist.

Usage:
    python phantom_sheet_depth.py --scrolls PHerc0343P PHerc1218 --n-blocks 12
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt, label

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
DIST_BINS = [(0, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6)]


def retry(fn, attempts: int = 6):
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)


def analyse(scroll: str, catalog: list[dict], n_blocks: int, block: int,
            seed: int) -> dict | None:
    rows = [r for r in catalog if r.get("scroll") == scroll
            and r.get("status") == "verifiable" and "m7" in r.get("model", "")]
    if not rows:
        return None
    row = rows[0]
    level = row.get("declared_level", 2)
    pred = zarr.open(f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
                     f"{row['prediction']}/0", mode="r")
    ct = zarr.open(f"{BUCKET}/{scroll}/volumes/{row['ct_volume']}/{level}", mode="r")
    if pred.shape != ct.shape:
        return None

    rng = np.random.default_rng(seed)
    depth_hist = np.zeros(len(DIST_BINS), dtype=np.int64)
    n_phantom = n_sheet = 0
    comp_total = comp_phantom = 0
    comp_sheet_vox = comp_phantom_vox = 0
    blocks_used = 0
    all_empty_blocks = all_empty_sheet_vox = 0

    tries = 0
    while blocks_used < n_blocks and tries < n_blocks * 6:
        tries += 1
        z, y, x = (int(rng.integers(0, max(1, s - block))) for s in pred.shape)
        try:
            p = retry(lambda: np.asarray(pred[z:z + block, y:y + block, x:x + block])) > 0
            c = retry(lambda: np.asarray(ct[z:z + block, y:y + block, x:x + block])) > 0
        except Exception:  # noqa: BLE001
            continue
        if p.shape != c.shape or not p.any():
            continue
        phantom = p & ~c
        if not phantom.any():
            continue
        if not c.any():
            # A block holding predicted sheet and NO scan data anywhere has no defined
            # "distance to real data" - the nearest scanned voxel is outside the block.
            # It must NOT go into the depth histogram: dumping it in the last bin
            # manufactures "100% predicted into nothing" out of an unmeasured quantity.
            # It is its own, and stronger, statistic, so it is counted on its own.
            all_empty_blocks += 1
            all_empty_sheet_vox += int(p.sum())
            continue
        blocks_used += 1

        # distance from every voxel to the nearest voxel that has scan data
        dist = distance_transform_edt(~c)
        d = dist[phantom]
        for i, (lo, hi) in enumerate(DIST_BINS):
            depth_hist[i] += int(((d >= lo) & (d <= hi)).sum())
        n_phantom += int(phantom.sum())
        n_sheet += int(p.sum())

        # connected components of the prediction: how many predicted structures are
        # wholly unsupported by scan data?
        lab, n = label(p)
        if n:
            comp_total += n
            for cid in range(1, n + 1):
                m = lab == cid
                sz = int(m.sum())
                comp_sheet_vox += sz
                frac = float(phantom[m].mean())
                if frac >= 0.9:
                    comp_phantom += 1
                    comp_phantom_vox += sz

    if n_phantom == 0 and all_empty_blocks == 0:
        print(f"{scroll:<12} no phantom sheet in {tries} sampled blocks")
        return None
    if n_phantom == 0:
        print(f"{scroll:<12} L{level}  {all_empty_blocks} block(s) of sheet with NO scan "
              f"data at all; no mixed block to measure depth in", flush=True)
        return {"scroll": scroll, "level": level, "blocks_used": 0,
                "all_empty_blocks": all_empty_blocks,
                "all_empty_sheet_voxels": all_empty_sheet_vox,
                "phantom_voxels": 0, "sheet_voxels": 0, "depth_bins": [],
                "frac_within_2vox": None, "frac_beyond_8vox": None,
                "components_total": 0, "components_wholly_phantom": 0,
                "frac_components_phantom": None,
                "frac_sheet_volume_in_phantom_components": None}

    hist = depth_hist / depth_hist.sum()
    out = {
        "scroll": scroll, "level": level, "blocks_used": blocks_used,
        "all_empty_blocks": all_empty_blocks,
        "all_empty_sheet_voxels": all_empty_sheet_vox,
        "phantom_voxels": int(n_phantom), "sheet_voxels": int(n_sheet),
        "depth_bins": [{"lo": lo, "hi": None if hi > 10 ** 5 else hi,
                        "frac": float(f), "n": int(k)}
                       for (lo, hi), f, k in zip(DIST_BINS, hist, depth_hist)],
        "frac_within_2vox": float(hist[0] + hist[1]),
        "frac_beyond_8vox": float(hist[4] + hist[5]),
        "components_total": comp_total,
        "components_wholly_phantom": comp_phantom,
        "frac_components_phantom": (comp_phantom / comp_total) if comp_total else None,
        "frac_sheet_volume_in_phantom_components":
            (comp_phantom_vox / comp_sheet_vox) if comp_sheet_vox else None,
    }
    print(f"{scroll:<12} L{level}  [{all_empty_blocks} all-empty blk]  "
          f"phantom {100*n_phantom/max(n_sheet,1):5.1f}% of sheet | "
          f"within 2vox {100*out['frac_within_2vox']:5.1f}%  "
          f"beyond 8vox {100*out['frac_beyond_8vox']:5.1f}% | "
          f"components wholly phantom {comp_phantom}/{comp_total}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="+", required=True)
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--n-blocks", type=int, default=12)
    ap.add_argument("--block", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ink_reference/phantom_sheet_depth.json")
    args = ap.parse_args()

    catalog = json.load(open(args.catalog))
    print("distance from each phantom voxel to the nearest voxel WITH scan data\n"
          "fringe artifact => mass at 1-2 voxels; predicting into nothing => mass at 8+\n",
          flush=True)
    rows = [r for r in (analyse(s, catalog, args.n_blocks, args.block, args.seed)
                        for s in args.scrolls) if r]
    if not rows:
        return
    meas = [r for r in rows if r["frac_within_2vox"] is not None]
    tot_p = sum(r["phantom_voxels"] for r in meas)
    if not tot_p:
        print("\nno mixed blocks anywhere - depth is undefined for this sample")
        return
    w2 = sum(r["frac_within_2vox"] * r["phantom_voxels"] for r in meas) / tot_p
    b8 = sum(r["frac_beyond_8vox"] * r["phantom_voxels"] for r in meas) / tot_p
    ct_ = sum(r["components_total"] for r in rows)
    cp_ = sum(r["components_wholly_phantom"] for r in rows)
    ae = sum(r.get("all_empty_blocks", 0) for r in rows)
    print(f"\nPOOLED over {len(meas)} scrolls with mixed blocks, {tot_p:,} phantom voxels")
    print(f"  blocks of predicted sheet with NO scan data at all: {ae}")
    print(f"  within 2 voxels of real data : {100*w2:5.1f}%   <- fringe")
    print(f"  beyond 8 voxels              : {100*b8:5.1f}%   <- predicted into nothing")
    print(f"  components wholly phantom    : {cp_}/{ct_} "
          f"({100*cp_/max(ct_,1):.1f}%)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"note": ("Distance from phantom sheet voxels to the nearest voxel with scan data. "
                  "Mass at 1-2 voxels means a boundary artifact; mass at 8+ means the "
                  "model asserts sheet in unscanned space."),
         "pooled": {"frac_within_2vox": w2, "frac_beyond_8vox": b8,
                    "components_total": ct_, "components_wholly_phantom": cp_},
         "scrolls": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
