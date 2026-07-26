"""Is the L2 surface prediction actually aligned with the L0 CT grid under a x4 scale?

Both predict_ink.py (region picking) and ink_on_surface.py (the ink/sheet discriminator)
assume L0 index = 4 * L2 index. If that assumption is off - by a pyramid offset, a
half-voxel convention, a crop - then every geometric statement built on it is measuring
noise, and it would do so silently.

This checks it without involving the ink model at all. Papyrus is denser than the air
between sheets, so a correctly aligned sheet mask must select higher CT intensity than
its complement. We compute that contrast at the assumed alignment and at a range of
shifts along each axis. If x4 is right, shift 0 wins. If something else wins, the mapping
is wrong and the discriminator has to be corrected before it can be believed.
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
L2_TO_L0 = 4


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


def surface_at_l0(surface_path: str, z0: int, y0: int, x0: int, n: int,
                  pad: int = 32) -> np.ndarray:
    """Upsampled sheet mask covering [z0-pad, z0+n+pad) so shifts can be evaluated."""
    a = zarr.open(f"{BUCKET}/{surface_path}/0", mode="r")
    start = [c - pad for c in (z0, y0, x0)]
    lo = [s // L2_TO_L0 for s in start]
    hi = [-(-(s + n + 2 * pad) // L2_TO_L0) for s in start]
    for i in range(3):
        lo[i] = max(0, lo[i])
        hi[i] = min(hi[i], a.shape[i])
    blk = retry(lambda: np.asarray(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]))
    up = np.repeat(np.repeat(np.repeat(blk, L2_TO_L0, 0), L2_TO_L0, 1), L2_TO_L0, 2)
    off = [s - l * L2_TO_L0 for s, l in zip(start, lo)]
    up = up[off[0]:off[0] + n + 2 * pad,
            off[1]:off[1] + n + 2 * pad,
            off[2]:off[2] + n + 2 * pad]
    return up > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--region", required=True, help="z0,y0,x0 at L0")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--catalog", default="catalog.json")
    args = ap.parse_args()

    z0, y0, x0 = (int(v) for v in args.region.split(","))
    n, pad = args.n, 32
    catalog = json.load(open(args.catalog))
    rows = [r for r in catalog if r.get("scroll") == args.scroll
            and r.get("status") == "verifiable" and "m7" in r.get("model", "")
            and r.get("declared_level") == 2]
    if not rows:
        raise SystemExit(f"no L2 m7 prediction for {args.scroll}")
    row = rows[0]
    sp = f"{args.scroll}/representations/predictions/surfaces/{row['prediction']}"

    ct_url = f"{BUCKET}/{args.scroll}/volumes/{row['ct_volume']}/0"
    ct_arr = zarr.open(ct_url, mode="r")
    ct = retry(lambda: np.asarray(
        ct_arr[z0:z0 + n, y0:y0 + n, x0:x0 + n])).astype(np.float32)
    surf_big = surface_at_l0(sp, z0, y0, x0, n, pad)

    print(f"{args.scroll}  region {z0},{y0},{x0}  n={n}")
    print(f"  CT  mean {ct.mean():.1f}  min {ct.min():.1f}  max {ct.max():.1f}")
    print(f"  L2 prediction shape {zarr.open(f'{BUCKET}/{sp}/0', mode='r').shape}")
    print(f"  CT  L0 shape        {ct_arr.shape}")
    print(f"  ratio {[round(c/p, 3) for c, p in zip(ct_arr.shape, zarr.open(f'{BUCKET}/{sp}/0', mode='r').shape)]}")
    def contrast_at(dz: int, dy: int, dx: int):
        s = surf_big[pad + dz:pad + dz + n,
                     pad + dy:pad + dy + n,
                     pad + dx:pad + dx + n]
        if s.shape != ct.shape or s.all() or not s.any():
            return None
        on, off = float(ct[s].mean()), float(ct[~s].mean())
        return {"frac_sheet": float(s.mean()), "ct_on": on, "ct_off": off,
                "contrast": on - off}

    # Scan each axis on its own - a diagonal scan cannot tell which axis is offset,
    # and the axes have no reason to share an offset.
    grid = list(range(-pad, pad + 1, 2))
    cur = [0, 0, 0]
    for it in range(3):  # coordinate descent; converges immediately if already aligned
        moved = False
        for ax, name in enumerate("zyx"):
            best_s, best_c = cur[ax], None
            row = []
            for sh in grid:
                trial = list(cur)
                trial[ax] = sh
                r = contrast_at(*trial)
                if r is None:
                    continue
                row.append((sh, r["contrast"]))
                if best_c is None or r["contrast"] > best_c:
                    best_s, best_c = sh, r["contrast"]
            if it == 0:
                top = ", ".join(f"{s}:{c:+.1f}" for s, c in row if abs(s) % 8 == 0)
                print(f"  {name}: {top}")
            if best_s != cur[ax]:
                cur[ax], moved = best_s, True
        if not moved:
            break

    base = contrast_at(0, 0, 0)
    best = contrast_at(*cur)
    print()
    print(f"  contrast at assumed x4 alignment (0,0,0): {base['contrast']:+.1f}  "
          f"(sheet {100*base['frac_sheet']:.1f}%, CT on {base['ct_on']:.1f} "
          f"vs off {base['ct_off']:.1f})")
    print(f"  best offset found (dz,dy,dx) = {tuple(cur)}: {best['contrast']:+.1f}  "
          f"(CT on {best['ct_on']:.1f} vs off {best['ct_off']:.1f})")

    # Sheets are locally parallel, so shifting a few voxels usually lands on the same
    # sheet and the contrast curve has a broad, shallow maximum. An argmax that is not
    # exactly 0 is therefore the normal case and is NOT evidence of a mapping bug. Only
    # call it misaligned when moving away from 0 buys a large improvement.
    if base["contrast"] <= 0:
        print("  => the mask does not select denser material at the assumed alignment. "
              "Either the mapping is wrong here or the mask marks something other than "
              "papyrus - check a second scroll before concluding either.")
    elif best["contrast"] > 2.0 * base["contrast"]:
        print(f"  => shifting by {tuple(cur)} more than doubles the contrast; "
              "the x4 mapping looks off for this scroll.")
    else:
        print("  => consistent with the x4 mapping: the mask selects denser material at "
              "offset 0, and the maximum nearby is broad and shallow, as parallel sheets "
              "produce.")


if __name__ == "__main__":
    main()
