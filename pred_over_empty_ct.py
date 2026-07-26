"""How much published surface prediction sits over empty CT?

Found while chasing a degenerate ink run. predict_ink.py picked a region for PHerc0500P2
where the m7 surface prediction reports 46.9% sheet, sent it to the ink model, and got a
constant output. The cause was not the ink model: the CT there is identically zero over
the whole block, at both level 2 and level 0, while the surface prediction is confidently
non-zero on exactly the same grid. The prediction asserts papyrus where the scan has no
data.

The masked CT volumes are zero outside the scanned material, so "prediction > 0 where
CT == 0" is prediction with nothing underneath it. This measures how common that is, by
sampling blocks at random across each volume and reporting P(CT == 0 | prediction > 0).

Comparison scrolls are included on purpose: one scroll's number means nothing without
knowing what the others do. Sampling is uniform over the volume, not steered toward the
region that started this, so the figure is not selected to be alarming.

Usage:
    python pred_over_empty_ct.py --scrolls PHerc0500P2 PHerc0332 PHerc0343P
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


def check(scroll: str, catalog: list[dict], n_blocks: int, block: int,
          seed: int) -> dict | None:
    rows = [r for r in catalog if r.get("scroll") == scroll
            and r.get("status") == "verifiable" and "m7" in r.get("model", "")]
    if not rows:
        print(f"{scroll}: no m7 prediction")
        return None
    row = rows[0]
    # Only 14 of the 41 published m7 predictions sit at level 2; the other 27 are at
    # level 0. Reading the CT at whatever level the prediction declares is what makes
    # this a collection-wide audit rather than a third of one.
    level = row.get("declared_level", 2)
    pred = zarr.open(f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
                     f"{row['prediction']}/0", mode="r")
    ct = zarr.open(f"{BUCKET}/{scroll}/volumes/{row['ct_volume']}/{level}", mode="r")
    if pred.shape != ct.shape:
        print(f"{scroll}: prediction {pred.shape} and CT L{level} {ct.shape} differ - "
              "skipped, this test assumes a shared grid")
        return None

    rng = np.random.default_rng(seed)
    tot_pred = tot_pred_on_empty = tot_vox = tot_empty = 0
    hits = 0
    for i in range(n_blocks):
        z, y, x = (int(rng.integers(0, max(1, s - block))) for s in pred.shape)
        try:
            p = retry(lambda: np.asarray(pred[z:z + block, y:y + block, x:x + block]))
            c = retry(lambda: np.asarray(ct[z:z + block, y:y + block, x:x + block]))
        except Exception:  # noqa: BLE001
            continue
        if p.shape != c.shape or p.size == 0:
            continue
        pm, empty = p > 0, c == 0
        tot_vox += p.size
        tot_empty += int(empty.sum())
        tot_pred += int(pm.sum())
        n_bad = int((pm & empty).sum())
        tot_pred_on_empty += n_bad
        if n_bad:
            hits += 1
    if tot_pred == 0:
        print(f"{scroll}: no predicted sheet in {n_blocks} sampled blocks")
        return None
    frac = tot_pred_on_empty / tot_pred
    out = {"scroll": scroll, "prediction": row["prediction"], "level": level,
           "blocks_sampled": n_blocks, "block": block,
           "voxels": tot_vox,
           "frac_ct_empty": tot_empty / tot_vox,
           "frac_predicted_sheet": tot_pred / tot_vox,
           "p_ct_empty_given_pred": frac,
           "blocks_with_any": hits}
    # flush: a 36-scroll run is thousands of network reads and takes a long time. Without
    # this, stdout redirected to a file stays block-buffered and the log looks dead.
    print(f"{scroll:<12} L{level}  CT empty {100*out['frac_ct_empty']:5.1f}%   "
          f"sheet {100*out['frac_predicted_sheet']:5.2f}%   "
          f"P(CT empty | sheet) {100*frac:6.3f}%   "
          f"blocks affected {hits}/{n_blocks}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scrolls", nargs="+", required=True)
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--n-blocks", type=int, default=40)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    # Kept out of results/ink_structure/, which is one file per scroll (see paris4_ink_range.py).
    ap.add_argument("--out", default="results/ink_reference/pred_over_empty_ct.json")
    args = ap.parse_args()

    catalog = json.load(open(args.catalog))
    print(f"sampling {args.n_blocks} blocks of {args.block}^3 per scroll, "
          "each against the CT at its own prediction's declared level\n")
    rows = [r for r in (check(s, catalog, args.n_blocks, args.block, args.seed)
                        for s in args.scrolls) if r]
    if rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"note": ("P(CT empty | predicted sheet) on masked CT volumes. Non-zero means "
                      "the published prediction asserts papyrus where the scan has no "
                      "data."), "scrolls": rows}, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
