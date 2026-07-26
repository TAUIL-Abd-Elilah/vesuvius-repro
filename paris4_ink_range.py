"""What range does REAL ink actually span, across many regions of PHerc. Paris 4?

ink_structure.py measures P(ink | dense CT) - the rate at which a model calls papyrus
"ink". One reference point cannot say whether another scroll's value is out of family, so
this builds the reference distribution from the published Paris 4 artifact, the only ink
in the collection known to be real.

SAMPLING HAS TO BE BLIND TO INK, or the reference is rigged. Regions are chosen by CT
content - blocks holding a decent amount of papyrus - which is the same criterion
predict_ink.py applies on the other scrolls. Choosing them by ink content instead would
select high-ink regions, inflate the reference, and make an over-firing scroll look
normal. The one ink-dependent filter is that the block must have a computed prediction at
all: the artifact is zero outside the region the team ran, and a constant-zero block means
"not computed here", not "no ink here". Those are dropped rather than scored as zero.

Level 4 (~38 um) is scanned to find candidate blocks; every number is then measured at
level 0, the resolution the model ran at, so the values are comparable to the other
scrolls' figures.
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
INK = ("PHercParis4/representations/predictions/ink-3d/"
       "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr")
CT = "PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr"
KNOWN = (34102 + 64, 16346 + 64, 16346 + 64)  # the calibrated region, trimmed
N = 128
SCALE4 = 16


def retry(fn, attempts: int = 8):
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"    transient read, retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-regions", type=int, default=12)
    ap.add_argument("--span-z", type=int, default=48, help="L4 voxels around KNOWN")
    ap.add_argument("--span-yx", type=int, default=160)
    ap.add_argument("--pick", choices=("random", "dense"), default="random")
    ap.add_argument("--seed", type=int, default=0)
    # Aggregates live apart from results/ink_structure/, which holds exactly one file per
    # scroll. Mixing them there makes a plain glob over that directory report "paris4_range"
    # as if it were a scroll - which has already happened once.
    ap.add_argument("--out", default="results/ink_reference/paris4_range.json")
    args = ap.parse_args()

    ink4 = zarr.open(f"{BUCKET}/{INK}/4", mode="r")
    ct4 = zarr.open(f"{BUCKET}/{CT}/4", mode="r")
    print(f"ink L4 {ink4.shape}   ct L4 {ct4.shape}")

    cz, cy, cx = (c // SCALE4 for c in KNOWN)
    z0, z1 = max(0, cz - args.span_z), min(ink4.shape[0], cz + args.span_z)
    y0, y1 = max(0, cy - args.span_yx), min(ink4.shape[1], cy + args.span_yx)
    x0, x1 = max(0, cx - args.span_yx), min(ink4.shape[2], cx + args.span_yx)
    print(f"scanning L4 z {z0}:{z1}  y {y0}:{y1}  x {x0}:{x1}", flush=True)

    iblk = retry(lambda: np.asarray(ink4[z0:z1, y0:y1, x0:x1])).astype(np.float32)
    cblk = retry(lambda: np.asarray(ct4[z0:z1, y0:y1, x0:x1])).astype(np.float32)
    print(f"  ink L4 nonzero {100*(iblk > 0).mean():.2f}%   ct L4 mean {cblk.mean():.1f}")

    # a 128^3 L0 block is 8^3 at L4; score candidates on CT content only
    step = 8
    dense_thr = float(np.quantile(cblk[cblk > 0], 0.6)) if (cblk > 0).any() else 0.0
    cands = []
    for zz in range(0, iblk.shape[0] - step + 1, step):
        for yy in range(0, iblk.shape[1] - step + 1, step):
            for xx in range(0, iblk.shape[2] - step + 1, step):
                cs = cblk[zz:zz + step, yy:yy + step, xx:xx + step]
                isb = iblk[zz:zz + step, yy:yy + step, xx:xx + step]
                if isb.max() == 0:          # prediction not computed here
                    continue
                frac_dense = float((cs >= dense_thr).mean())
                if frac_dense < 0.25:       # not enough papyrus to be a fair test
                    continue
                cands.append((frac_dense, (z0 + zz) * SCALE4,
                              (y0 + yy) * SCALE4, (x0 + xx) * SCALE4))
    print(f"  {len(cands)} candidate blocks with papyrus and a computed prediction")
    if not cands:
        raise SystemExit("no candidates - widen --span-z/--span-yx")

    # Spread the picks out instead of taking one cluster. Ordering by density would bias
    # the sample toward solid-papyrus blocks, which both shifts the ink rate and makes the
    # dense/sparse contrast meaningless (the "sparse" half of an all-papyrus block is
    # still papyrus, not air). Random order keeps the reference honest.
    if args.pick == "dense":
        cands.sort(key=lambda t: -t[0])
    else:
        np.random.default_rng(args.seed).shuffle(cands)
    picks: list[tuple] = []
    for c in cands:
        if all(abs(c[1] - p[1]) >= N or abs(c[2] - p[2]) >= N or abs(c[3] - p[3]) >= N
               for p in picks):
            picks.append(c)
        if len(picks) >= args.n_regions:
            break
    print(f"  measuring {len(picks)} regions at L0\n", flush=True)

    ink0 = zarr.open(f"{BUCKET}/{INK}/0", mode="r")
    ct0 = zarr.open(f"{BUCKET}/{CT}/0", mode="r")
    rows = []
    for i, (fd, z, y, x) in enumerate(picks, 1):
        try:
            ink = retry(lambda: np.asarray(
                ink0[z:z + N, y:y + N, x:x + N])).astype(np.float32) / 255.0
            ct = retry(lambda: np.asarray(
                ct0[z:z + N, y:y + N, x:x + N])).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] {z},{y},{x} read failed: {type(exc).__name__}")
            continue
        if ink.shape != (N, N, N) or ink.std() < 1e-9:
            print(f"  [{i}] {z},{y},{x} skipped (no computed prediction)")
            continue
        hot = ink > 0.5
        hi = ct >= np.quantile(ct, 0.9)
        lo = ct <= np.quantile(ct, 0.5)
        p_hi = float(hot[hi].mean())
        p_lo = float(hot[lo].mean())
        rows.append({"origin_l0": [int(z), int(y), int(x)],
                     "l4_dense_frac": fd,
                     "ink_frac_hot": float(hot.mean()),
                     "p_ink_dense": p_hi, "p_ink_sparse": p_lo,
                     "dense_lift": float(p_hi / p_lo) if p_lo > 0 else float("inf"),
                     "ct_mean": float(ct.mean())})
        print(f"  [{i}] {z},{y},{x}  hot {100*hot.mean():6.3f}%  "
              f"P(ink|dense) {p_hi:.4f}  lift "
              f"{rows[-1]['dense_lift'] if np.isfinite(rows[-1]['dense_lift']) else float('inf'):.1f}x",
              flush=True)

    if not rows:
        raise SystemExit("every candidate turned out to have no computed prediction")

    pd = np.array([r["p_ink_dense"] for r in rows])
    out = {
        "scroll": "PHercParis4",
        "source": "published ink-3d artifact (real ink)",
        "n_regions": len(rows),
        "p_ink_dense": {"min": float(pd.min()), "max": float(pd.max()),
                        "mean": float(pd.mean()), "median": float(np.median(pd)),
                        "p90": float(np.quantile(pd, 0.9))},
        "regions": rows,
        "note": ("Reference range for P(ink | top-decile CT) on ink known to be real. "
                 "Regions were chosen by CT content, blind to ink, so the range is not "
                 "selected for high signal. Compare other scrolls against this."),
    }
    print(f"\nP(ink|dense) over {len(rows)} real-ink regions: "
          f"min {pd.min():.4f}  median {np.median(pd):.4f}  max {pd.max():.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
