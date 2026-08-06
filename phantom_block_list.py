"""Emit the per-block phantom list, so someone else can sample the same blocks.

`pred_over_empty_ct.py` answers "how much phantom sheet is there" and keeps only per-scroll
totals. @anshu231 (inkdx) asked for the block list itself, as a sampling cross-check against
the 55 mixed blocks they extracted independently across PHercMANB / 0841 / 1451 / 1545. This
emits coordinates.

Definitions are theirs, so the two lists are comparable rather than merely similar:

    phantom voxel  =  prediction > 0  AND  CT == 0
    depth          =  Euclidean distance to the nearest scanned (CT > 0) voxel
    mixed block    =  contains both scanned CT and predicted sheet

The distinction that matters is between a block with *no* scan under it at all — where a
phantom prediction is unsurprising and a ct>0 gate removes it outright — and a mixed block,
where scan and prediction coexist and the prediction still sits deep inside material. The
second is the residue a support mask alone does not explain, and it is the interesting half.

CPU and network only; no GPU, no model. Blocks are sampled with the same seed and geometry
as the original run, so block k here is block k there.

    python phantom_block_list.py --scrolls PHercMANB PHerc0841 PHerc1451 PHerc1545
    python phantom_block_list.py --all
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parent

# The four where the deep-in-material residue concentrates, per the all-36 run.
RESIDUE_SCROLLS = ["PHercMANB", "PHerc0841", "PHerc1451", "PHerc1545"]


def retry(fn, attempts: int = 6):
    """This link drops mid-transfer; a whole-collection pass is thousands of reads."""
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)


def edt_depth(empty: np.ndarray, phantom: np.ndarray) -> dict:
    """Depth of each phantom voxel below the nearest scanned voxel.

    scipy's EDT on the empty mask gives, for every empty voxel, the distance to the nearest
    non-empty one -- which is exactly 'how far inside the unscanned region does this
    prediction sit'. Falls back to reporting nothing rather than guessing if scipy is absent,
    because a wrong depth is worse than no depth.
    """
    if not phantom.any():
        return {}
    try:
        from scipy import ndimage
    except ImportError:
        return {"note": "scipy unavailable; depths omitted"}
    if not empty.any() or empty.all():
        # all-or-nothing scan coverage: depth is undefined rather than zero
        return {"note": "block is wholly scanned or wholly unscanned; depth undefined"}
    d = ndimage.distance_transform_edt(empty)[phantom]
    return {
        "depth_mean": round(float(d.mean()), 3),
        "depth_p50": round(float(np.percentile(d, 50)), 3),
        "depth_p90": round(float(np.percentile(d, 90)), 3),
        "depth_max": round(float(d.max()), 3),
        "frac_within_2vox": round(float((d <= 2).mean()), 4),
        "frac_beyond_8vox": round(float((d > 8).mean()), 4),
    }


def blocks_for(scroll: str, catalog: list[dict], n_blocks: int, block: int,
               seed: int) -> list[dict]:
    rows = [r for r in catalog if r.get("scroll") == scroll
            and r.get("status") == "verifiable" and "m7" in r.get("model", "")]
    if not rows:
        print(f"{scroll}: no m7 prediction", flush=True)
        return []
    row = rows[0]
    level = row.get("declared_level", 2)
    pred = zarr.open(f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
                     f"{row['prediction']}/0", mode="r")
    ct = zarr.open(f"{BUCKET}/{scroll}/volumes/{row['ct_volume']}/{level}", mode="r")
    if pred.shape != ct.shape:
        print(f"{scroll}: prediction {pred.shape} vs CT L{level} {ct.shape} differ - skipped",
              flush=True)
        return []

    # Same generator, same order, same block size as pred_over_empty_ct.py, so block index
    # k identifies the same cube in both outputs.
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_blocks):
        z, y, x = (int(rng.integers(0, max(1, s - block))) for s in pred.shape)
        try:
            p = retry(lambda: np.asarray(pred[z:z + block, y:y + block, x:x + block]))
            c = retry(lambda: np.asarray(ct[z:z + block, y:y + block, x:x + block]))
        except Exception as e:  # noqa: BLE001
            print(f"  {scroll} block {k}: read failed ({type(e).__name__})", flush=True)
            continue
        if p.shape != c.shape or p.size == 0:
            continue

        sheet, empty = p > 0, c == 0
        phantom = sheet & empty
        n_phantom = int(phantom.sum())
        if n_phantom == 0:
            continue

        scanned_frac = float((~empty).mean())
        rec = {
            "scroll": scroll, "prediction": row["prediction"], "level": level,
            "block_index": k, "origin_zyx": [z, y, x], "block": block,
            "ct_coverage": round(scanned_frac, 4),
            "sheet_voxels": int(sheet.sum()),
            "phantom_voxels": n_phantom,
            "frac_of_sheet_phantom": round(n_phantom / max(1, int(sheet.sum())), 4),
            # a mixed block has scan AND prediction; the all-empty ones are the easy class
            "mixed": bool(scanned_frac > 0.0 and scanned_frac < 1.0),
            **edt_depth(empty, phantom),
        }
        out.append(rec)
        print(f"  {scroll:<12} blk {k:>3} @ {z:>6},{y:>6},{x:>6}  "
              f"cov {scanned_frac:5.2f}  phantom {n_phantom:>8,}  "
              f"{'mixed' if rec['mixed'] else 'all-empty'}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="+", default=None)
    ap.add_argument("--all", action="store_true", help="every scroll in the catalog")
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--n-blocks", type=int, default=40)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "phantom_block_list.json"))
    a = ap.parse_args()

    catalog = json.load(open(a.catalog))
    if a.all:
        scrolls = sorted({r["scroll"] for r in catalog
                          if r.get("status") == "verifiable" and "m7" in r.get("model", "")})
    else:
        scrolls = a.scrolls or RESIDUE_SCROLLS

    print(f"{len(scrolls)} scrolls, {a.n_blocks} blocks of {a.block}^3, seed {a.seed}\n",
          flush=True)
    records = []
    for s in scrolls:
        records.extend(blocks_for(s, catalog, a.n_blocks, a.block, a.seed))

    mixed = [r for r in records if r["mixed"]]
    payload = {
        "definitions": {
            "phantom": "prediction > 0 and CT == 0",
            "depth": "Euclidean distance to the nearest CT > 0 voxel",
            "mixed": "block contains both scanned CT and predicted sheet",
        },
        "sampling": {"n_blocks": a.n_blocks, "block": a.block, "seed": a.seed,
                     "note": "same rng, order and geometry as pred_over_empty_ct.py, so "
                             "block_index k is the same cube in both outputs"},
        "scrolls": scrolls,
        "n_blocks_with_phantom": len(records),
        "n_mixed_blocks": len(mixed),
        "total_phantom_voxels": sum(r["phantom_voxels"] for r in records),
        "mixed_phantom_voxels": sum(r["phantom_voxels"] for r in mixed),
        "blocks": records,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"\n{len(records)} blocks carry phantom, {len(mixed)} of them mixed "
          f"({payload['mixed_phantom_voxels']:,} of {payload['total_phantom_voxels']:,} "
          f"phantom voxels)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
