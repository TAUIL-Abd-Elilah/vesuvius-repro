"""Does --mask_empty_input clip real sheet at the scan boundary?

The three-way on #1173 showed arm C removing 100% of phantom with real sheet "identical to
the voxel" across arms. @Schurkai's objection is that the region was 87.9% empty, so there
was almost no real sheet next to a boundary for the mask to get wrong -- and the overall
real-sheet count would stay identical even if a boundary layer had been shaved, because that
layer is a rounding error against 433,707 voxels.

So this reports the endpoint that can actually see it: real sheet (CT > 0 and predicted
sheet) as a function of distance from the scanned/unscanned interface, per arm. If the mask
over-clips, the loss is concentrated in the first few voxels and shows as a deficit in the
near bins while the far bins stay equal.

    python measure_boundary_clip.py --work outputs/border_three_way \
        --scroll PHerc0139 --bbox 13105:13489,657:1041,3181:3565
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5)
BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 10 ** 6)]


def parse_bbox(s: str):
    z, y, x = s.split(",")
    out = []
    for piece in (z, y, x):
        a, b = piece.split(":")
        out += [int(a), int(b)]
    return out


def prob_from_logits(arr, sl) -> np.ndarray:
    """final.zarr holds 2-channel logits; the published threshold is on the sigmoid.

    Index the channel and the region in ONE operation. `arr[0][sl]` reads the entire
    channel first -- 1.67 TiB for this store -- and only then slices it.
    """
    z, y, x = sl
    l0 = np.asarray(arr[0, z, y, x]).astype(np.float32)
    l1 = np.asarray(arr[1, z, y, x]).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-(l1 - l0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="outputs/border_three_way")
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--arms", default="A_main,C_blendfix_maskinput")
    ap.add_argument("--out", default=str(ROOT / "results" / "boundary_clip.json"))
    a = ap.parse_args()

    z0, z1, y0, y1, x0, x1 = parse_bbox(a.bbox)
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

    catalog = json.load(open(a.catalog))
    row = [r for r in catalog if r.get("scroll") == a.scroll
           and r.get("status") == "verifiable" and "m7" in r.get("model", "")
           and r.get("declared_level", 2) == 0][0]

    work = ROOT / a.work
    ct_cache = work / "ct_block.npy"
    if ct_cache.exists():
        ct = np.load(ct_cache)
    else:
        ct_arr = zarr.open(f"{BUCKET}/{a.scroll}/volumes/{row['ct_volume']}/0", mode="r")
        ct = np.asarray(ct_arr[sl])
        np.save(ct_cache, ct)

    empty = ct == 0
    scanned = ~empty
    # Interface = the scanned skin of the boundary, same construction as find_border_region.
    interface = ndimage.binary_dilation(empty) & scanned
    # Distance from the interface, measured on the scanned side only; unscanned voxels get
    # no distance because real sheet cannot live there by definition.
    dist = ndimage.distance_transform_edt(~interface)

    print(f"{a.scroll} {a.bbox}")
    print(f"  CT empty {empty.mean():.1%}, interface {int(interface.sum()):,} voxels\n")

    results = {}
    for arm in a.arms.split(","):
        f = work / arm / "final.zarr"
        if not f.exists():
            print(f"  {arm}: missing {f}")
            continue
        arr = zarr.open(str(f), mode="r")
        prob = prob_from_logits(arr, sl)
        per_thr = {}
        for t in THRESHOLDS:
            sheet = prob > t
            real = sheet & scanned
            phantom = sheet & empty
            bins = {}
            for lo, hi in BINS:
                m = (dist >= lo) & (dist < hi) & scanned
                bins[f"{lo}-{hi if hi < 10**6 else 'inf'}"] = int((real & m).sum())
            per_thr[str(t)] = {
                "real_sheet": int(real.sum()),
                "phantom": int(phantom.sum()),
                "real_sheet_by_distance": bins,
                "real_sheet_within_4vox": int((real & (dist < 4)).sum()),
            }
        results[arm] = per_thr
        r2 = per_thr["0.2"]
        print(f"  {arm:<26} @0.2  real {r2['real_sheet']:>9,}  "
              f"near(<4) {r2['real_sheet_within_4vox']:>8,}  phantom {r2['phantom']:>9,}")

    arms = list(results)
    payload = {"scroll": a.scroll, "bbox": a.bbox,
               "frac_ct_empty": float(empty.mean()),
               "interface_voxels": int(interface.sum()),
               "bins": [f"{lo}-{hi if hi < 10**6 else 'inf'}" for lo, hi in BINS],
               "endpoint": ("real sheet by distance from the scanned/unscanned interface; "
                            "over-clipping shows as a deficit in the near bins while far "
                            "bins stay equal"),
               "arms": results}
    Path(a.out).write_text(json.dumps(payload, indent=1))

    if len(arms) == 2:
        base, test = arms
        print(f"\n  real sheet by distance from the boundary, threshold 0.2")
        print(f"  {'bin (vox)':<12} {base:>14} {test:>26} {'delta':>10}")
        b = results[base]["0.2"]["real_sheet_by_distance"]
        c = results[test]["0.2"]["real_sheet_by_distance"]
        for k in b:
            d = c[k] - b[k]
            print(f"  {k:<12} {b[k]:>14,} {c[k]:>26,} {d:>10,}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
