"""Find a region with a LONG scanned/unscanned interface, for the #1173 over-clipping control.

find_phantom_region.py deliberately picks the most-empty regions it can, because that makes
phantom unambiguous: over identically-zero CT every predicted sheet voxel is phantom by
definition rather than by threshold. @Schurkai's objection to the resulting three-way is
exactly right — that same property makes the region useless for the opposite question. At
87.9% empty there is almost no real sheet sitting next to a boundary, so a mask that shaved
a layer off real sheet at the scan edge would not show up.

This selects on the opposite criterion:

  * CT coverage near half, not near zero
  * a long interface between scanned and unscanned voxels
  * real sheet (CT > 0 and prediction > 0) actually sitting within a few voxels of that
    interface, since that is the population --mask_empty_input could wrongly clip
  * some phantom present, so arm C still has something legitimate to remove

The endpoint the later run reports is real sheet NEAR THE BOUNDARY, not real sheet overall:
the existing table's "identical to the voxel" would stay true even if a boundary layer were
shaved, because that layer is a rounding error against 433,707 voxels.

    python find_border_region.py --scrolls PHerc0175B PHerc0139 --side 384
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


def describe(p: np.ndarray, c: np.ndarray, near: int) -> dict | None:
    """Interface size and how much real sheet sits within `near` voxels of it."""
    from scipy import ndimage

    sheet, empty = p > 0, c == 0
    if not sheet.any() or not empty.any() or empty.all():
        return None

    # Interface = scanned voxels touching an unscanned one. Dilating the empty mask by one
    # and intersecting with the scanned side gives the scanned skin of the boundary.
    dil = ndimage.binary_dilation(empty)
    interface = dil & (~empty)
    if not interface.any():
        return None

    # Distance from every voxel to the nearest interface voxel.
    dist = ndimage.distance_transform_edt(~interface)
    near_mask = dist <= near

    real_sheet = sheet & (~empty)
    phantom = sheet & empty
    return {
        "frac_ct_empty": float(empty.mean()),
        "interface_voxels": int(interface.sum()),
        "real_sheet_voxels": int(real_sheet.sum()),
        "real_sheet_near_boundary": int((real_sheet & near_mask).sum()),
        "phantom_voxels": int(phantom.sum()),
        "frac_real_sheet_near_boundary": round(
            float((real_sheet & near_mask).sum() / max(1, int(real_sheet.sum()))), 4),
    }


def scan(scroll: str, row: dict, side: int, n_blocks: int, seed: int, probe: int,
         near: int) -> list[dict]:
    pred = zarr.open(f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
                     f"{row['prediction']}/0", mode="r")
    ct = zarr.open(f"{BUCKET}/{scroll}/volumes/{row['ct_volume']}/0", mode="r")
    if pred.shape != ct.shape:
        print(f"{scroll}: pred {pred.shape} vs CT {ct.shape} differ - skipped", flush=True)
        return []

    rng = np.random.default_rng(seed)
    found = []
    for i in range(n_blocks):
        z, y, x = (int(rng.integers(0, max(1, s - side))) for s in pred.shape)
        cz, cy, cx = z + side // 2, y + side // 2, x + side // 2
        h = probe // 2
        try:
            p = retry(lambda: np.asarray(pred[cz - h:cz + h, cy - h:cy + h, cx - h:cx + h]))
            c = retry(lambda: np.asarray(ct[cz - h:cz + h, cy - h:cy + h, cx - h:cx + h]))
        except Exception:  # noqa: BLE001
            continue
        if p.shape != c.shape or p.size == 0:
            continue
        d = describe(p, c, near)
        if d is None or d["phantom_voxels"] == 0 or d["real_sheet_near_boundary"] == 0:
            continue
        # Coverage anywhere near the extremes defeats the purpose of the control.
        if not (0.15 <= d["frac_ct_empty"] <= 0.85):
            continue
        rec = {"scroll": scroll, "bbox": [z, z + side, y, y + side, x, x + side], **d}
        found.append(rec)
        print(f"  {scroll:<12} empty {d['frac_ct_empty']:.2f}  interface {d['interface_voxels']:>7,}"
              f"  real sheet near border {d['real_sheet_near_boundary']:>7,}"
              f"  phantom {d['phantom_voxels']:>7,}", flush=True)

    # Rank by the population the control needs: real sheet next to the boundary.
    found.sort(key=lambda r: (-r["real_sheet_near_boundary"], -r["interface_voxels"]))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="+", required=True)
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--side", type=int, default=384)
    ap.add_argument("--probe", type=int, default=128)
    ap.add_argument("--n-blocks", type=int, default=60)
    ap.add_argument("--near", type=int, default=4, help="voxels from the interface")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "border_region.json"))
    a = ap.parse_args()

    catalog = json.load(open(a.catalog))
    all_found = []
    for scroll in a.scrolls:
        rows = [r for r in catalog if r.get("scroll") == scroll
                and r.get("status") == "verifiable" and "m7" in r.get("model", "")
                and r.get("declared_level", 2) == 0]
        if not rows:
            print(f"{scroll}: no level-0 verifiable m7 prediction", flush=True)
            continue
        print(f"{scroll}:", flush=True)
        all_found.extend(scan(scroll, rows[0], a.side, a.n_blocks, a.seed, a.probe, a.near))

    all_found.sort(key=lambda r: (-r["real_sheet_near_boundary"], -r["interface_voxels"]))
    payload = {"near_voxels": a.near, "side": a.side, "probe": a.probe,
               "criterion": "most real sheet within `near` voxels of the scan interface, "
                            "with CT coverage away from both extremes and phantom present",
               "candidates": all_found[:20]}
    Path(a.out).write_text(json.dumps(payload, indent=1))
    if all_found:
        b = all_found[0]
        z0, z1, y0, y1, x0, x1 = b["bbox"]
        print(f"\nbest: {b['scroll']}  --bbox {z0}:{z1},{y0}:{y1},{x0}:{x1}")
        print(f"  CT empty {b['frac_ct_empty']:.1%}, interface {b['interface_voxels']:,}, "
              f"real sheet near border {b['real_sheet_near_boundary']:,}, "
              f"phantom {b['phantom_voxels']:,}")
    else:
        print("\nno candidate met the criteria")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
