"""Locate a concrete region where the published m7 prediction asserts sheet over zero CT.

pred_over_empty_ct.py measures how OFTEN this happens across the collection. To test
Schurkai's hypothesis in #1173 -- that most of the 99.69% comes from the blend rather
than the input path -- inference has to be re-run three ways, which needs an actual bbox
rather than a prevalence figure.

A region is only useful here if all three of these hold:
  - the CT is (near-)identically zero, so anything predicted there is phantom
  - the PUBLISHED prediction is confidently non-zero there, so the model demonstrably
    does produce phantom in this spot rather than us hoping it will
  - it is large enough to hold the sliding window, so the blend actually runs

Level-0 predictions only: inference runs at level 0, and comparing a level-2 prediction
against a level-0 run would confound the thing being measured.

    python find_phantom_region.py --scrolls PHerc0139 PHerc0175A --side 384
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


def scan(scroll: str, row: dict, side: int, n_blocks: int, seed: int, probe: int):
    """Sample blocks; return the ones with the most phantom, described in global coords."""
    pred = zarr.open(f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
                     f"{row['prediction']}/0", mode="r")
    ct = zarr.open(f"{BUCKET}/{scroll}/volumes/{row['ct_volume']}/0", mode="r")
    if pred.shape != ct.shape:
        print(f"{scroll}: pred {pred.shape} vs CT {ct.shape} differ - skipped")
        return []

    rng = np.random.default_rng(seed)
    found = []
    for _ in range(n_blocks):
        # Probe a small cube first: reading `side`^3 of both arrays for every candidate
        # is far more traffic than this needs, and most candidates are rejected.
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
        sheet, empty = p > 0, c == 0
        n_sheet = int(sheet.sum())
        if not n_sheet:
            continue
        phantom = int((sheet & empty).sum())
        found.append({
            "scroll": scroll,
            "bbox": [z, z + side, y, y + side, x, x + side],
            "probe_frac_ct_empty": float(empty.mean()),
            "probe_frac_sheet": float(sheet.mean()),
            "probe_frac_phantom_of_sheet": phantom / n_sheet,
            "probe_phantom_voxels": phantom,
        })
    found.sort(key=lambda r: (-r["probe_frac_phantom_of_sheet"], -r["probe_phantom_voxels"]))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="+", required=True)
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--side", type=int, default=384, help="bbox edge for the later runs")
    ap.add_argument("--probe", type=int, default=128, help="cube actually read per candidate")
    ap.add_argument("--n-blocks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/phantom_region.json")
    args = ap.parse_args()

    catalog = json.load(open(args.catalog))
    best = []
    for s in args.scrolls:
        rows = [r for r in catalog
                if r.get("scroll") == s and r.get("status") == "verifiable"
                and "m7" in r.get("model", "") and r.get("declared_level", 2) == 0]
        if not rows:
            print(f"{s}: no level-0 m7 prediction")
            continue
        hits = scan(s, rows[0], args.side, args.n_blocks, args.seed, args.probe)
        if hits:
            top = hits[0]
            print(f"{s:<12} best block: phantom {100*top['probe_frac_phantom_of_sheet']:6.2f}% "
                  f"of sheet, sheet {100*top['probe_frac_sheet']:5.2f}%, "
                  f"CT empty {100*top['probe_frac_ct_empty']:5.1f}%  bbox {top['bbox']}",
                  flush=True)
            best.extend(hits[:3])
        else:
            print(f"{s:<12} no sampled block had any predicted sheet", flush=True)

    best.sort(key=lambda r: (-r["probe_frac_phantom_of_sheet"], -r["probe_phantom_voxels"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "note": ("Candidate regions where the published m7 prediction asserts sheet over "
                 "identically-zero CT, for the three-way rerun in villa#1173. Fractions are "
                 "over the probe cube at the block centre, not the whole bbox."),
        "side": args.side, "probe": args.probe, "seed": args.seed,
        "candidates": best[:12],
    }, indent=2))
    print(f"\nwrote {args.out} ({len(best)} candidates)")


if __name__ == "__main__":
    main()
