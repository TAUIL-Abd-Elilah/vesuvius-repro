"""Does the CT-ridge residual separate Will Stevens' good patches from his bad ones?

He marks 9,171 of 56,968 Scroll 4 patches as problems, and says patches stop growing where the
surface predictions are messy. If the residual we built for snap quality is measuring anything
real, it should be larger on the patches he already knows are bad — with no access to his
labels, his tracer, or any model.

⭐ WHY THIS TEST IS WORTH MORE THAN ITS SIZE. The residual reads the CT only: it finds the sheet
ridge along the across-sheet normal and reports how far the surface point sits from it. It never
sees m7. So unlike everything else we measured this week, **it is immune to the
`instance_zscore` normalization defect** (villa#1364) that invalidated our m7 numbers — a ridge
is a location, and locating a maximum is invariant to any monotone rescaling of intensity.
Scroll 4 is uint16 where the labelled set was uint8, and that does not matter either.

Design notes that keep it cheap and honest:

- **Per patch, one contiguous sub-window of the 127x127 grid**, not scattered points. Its world
  bbox is then small enough to stream as one block, so the whole check costs megabytes rather
  than the ~7 GB that per-point fetching would.
- **Good and bad are drawn from the same z range** and processed by identical code, because the
  scroll's character varies enormously with depth and an unmatched draw would measure that.
- `-1` marks invalid grid points in tifxyz and must be dropped before anything else.

  python will_patch_check.py --n 120
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import tifffile

from ridge_residual import ridge_offset

ROOT = Path(__file__).resolve().parent
WILL = ROOT / "_will"
WIN = 20            # sub-window of the 127x127 grid, per patch
MARGIN = 8          # voxels of CT padding around the sub-window's bbox
MAX_SPAN = 96       # ⚠ COST gate, not a quality one. The volume has 128^3 chunks, so a
                    # 220-span block pulls up to 27 of them (~100 MB) per patch and the run
                    # needs ~12 GB of streaming — the first attempt produced no output in six
                    # minutes for exactly this reason. At 96 it is at most 2x2x2 chunks.


def open_volume():
    import zarr, numcodecs, torch
    if not hasattr(zarr, "Blosc"):
        zarr.Blosc = numcodecs.Blosc
    _o = torch.compiler.disable
    def _d(fn=None, *, recursive=True, reason=None):
        return _o(fn, recursive=recursive)
    torch.compiler.disable = _d
    from vesuvius import list_files
    url = list_files()["4"]["54"]["7.91"]["volume"].rstrip("/") + "/0"
    store = zarr.storage.FsspecStore.from_url(url, read_only=True)
    return zarr.open(store, mode="r")


def patch_points(z: zipfile.ZipFile, base: str, rng) -> np.ndarray | None:
    """A contiguous WINxWIN block of valid (z,y,x) surface points from one tifxyz patch."""
    try:
        xs = tifffile.imread(io.BytesIO(z.read(f"{base}/x.tif")))
        ys = tifffile.imread(io.BytesIO(z.read(f"{base}/y.tif")))
        zs = tifffile.imread(io.BytesIO(z.read(f"{base}/z.tif")))
    except KeyError:
        return None
    h, w = xs.shape
    if h < WIN or w < WIN:
        return None
    valid = (xs > 0) & (ys > 0) & (zs > 0)          # -1 marks missing in tifxyz
    best, bestn = None, 0
    for _ in range(12):                              # find a densely-valid sub-window
        r = rng.integers(0, h - WIN + 1)
        c = rng.integers(0, w - WIN + 1)
        v = valid[r:r + WIN, c:c + WIN]
        if v.sum() > bestn:
            best, bestn = (r, c), int(v.sum())
    if best is None or bestn < WIN * WIN * 0.5:
        return None
    r, c = best
    sl = (slice(r, r + WIN), slice(c, c + WIN))
    m = valid[sl]
    # volume axes are (z, y, x)
    return np.stack([zs[sl][m], ys[sl][m], xs[sl][m]], axis=1).astype(np.float32)


def _fetch(vol, lo, hi, attempts: int = 5):
    """Read a CT block, retrying transient stream failures.

    ⚠ Streaming a 257 GB remote zarr WILL drop chunks. A first run died partway through on
    `ClientPayloadError: received 2096629 of 4194304 bytes` with an SSL record-layer failure —
    a truncated chunk, not a bug in the query. Without a retry the whole run is lost to one
    bad read, which is exactly what happened.
    """
    import time as _t
    for i in range(attempts):
        try:
            return np.asarray(vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
        except Exception:                            # noqa: BLE001 - network, any flavour
            if i == attempts - 1:
                return None
            _t.sleep(1.5 * (i + 1))
    return None


def residual_for(vol, pts: np.ndarray) -> float | None:
    lo = np.floor(pts.min(0)).astype(int) - MARGIN
    hi = np.ceil(pts.max(0)).astype(int) + MARGIN + 1
    if np.any(hi - lo > MAX_SPAN) or np.any(lo < 0):
        return None
    if np.any(hi >= np.array(vol.shape)):
        return None
    raw = _fetch(vol, lo, hi)
    if raw is None:
        return None
    block = raw.astype(np.float32)
    if block.size == 0 or float(block.max()) <= 0:
        return None
    off, _ = ridge_offset(block, (pts - lo).astype(np.float32))
    good = ~np.isnan(off)
    if good.sum() < 200:
        return None
    return float(np.median(np.abs(off[good])))


def collect(zip_path: Path, prefix: str, names: list[str], vol, rng, want: int, zlo, zhi):
    out = []
    scanned = 0
    z = zipfile.ZipFile(zip_path)
    for nm in names:
        if len(out) >= want:
            break
        scanned += 1
        if scanned % 300 == 0:
            print(f"    scanned {scanned}, kept {len(out)}", flush=True)
        meta = json.loads(z.read(f"{prefix}/{nm}/meta.json"))
        b = np.array(meta["bbox"])                   # [[x,y,z],[x,y,z]]
        cz = 0.5 * (b[0][2] + b[1][2])
        if not (zlo <= cz <= zhi):
            continue
        pts = patch_points(z, f"{prefix}/{nm}", rng)
        if pts is None:
            continue
        r = residual_for(vol, pts)
        if r is None:
            continue
        out.append({"patch": nm, "z": float(cz), "residual": r, "n_pts": int(len(pts))})
        if len(out) % 10 == 0:
            print(f"    {len(out)}/{want}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=120, help="patches per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zlo", type=float, default=3000.0)
    ap.add_argument("--zhi", type=float, default=8000.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "will_patch_check.json"))
    a = ap.parse_args()

    vol = open_volume()
    print(f"Scroll4 {vol.shape} {vol.dtype}", flush=True)
    rng = np.random.default_rng(a.seed)

    res = {}
    for tag, zp, prefix in (("bad", WILL / "s4_bad_patches.zip", "s4_bad_patches"),
                            ("good", WILL / "s4_good_patches.zip", "s4_good_patches")):
        # ⚠ Existence is not enough: a download still in flight is a real file that
        # zipfile rejects with BadZipFile. Check it actually opens.
        if not zp.exists():
            print(f"  {tag}: {zp.name} missing, skipped")
            continue
        try:
            z = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            print(f"  {tag}: {zp.name} is not a complete zip (download in flight?), skipped")
            continue
        names = sorted({n.split("/")[1] for n in z.namelist()
                        if n.startswith(prefix + "/") and len(n.split("/")) > 2})
        rng2 = np.random.default_rng(a.seed)
        order = [names[i] for i in rng2.permutation(len(names))]
        print(f"  {tag}: {len(names)} patches available", flush=True)
        res[tag] = collect(zp, prefix, order, vol, rng, a.n, a.zlo, a.zhi)

    out = {"window": WIN, "z_range": [a.zlo, a.zhi], "rows": res}
    for tag in res:
        v = np.array([r["residual"] for r in res[tag]])
        if len(v):
            out[tag] = {"n": len(v), "median": round(float(np.median(v)), 4),
                        "q25": round(float(np.quantile(v, .25)), 4),
                        "q75": round(float(np.quantile(v, .75)), 4)}
            print(f"  {tag}: n={len(v)} median residual {np.median(v):.4f} "
                  f"IQR {np.quantile(v,.25):.3f}-{np.quantile(v,.75):.3f}")
    if "bad" in out and "good" in out:
        from scipy.stats import mannwhitneyu
        b = [r["residual"] for r in res["bad"]]
        g = [r["residual"] for r in res["good"]]
        u = mannwhitneyu(b, g, alternative="greater")
        auc = u.statistic / (len(b) * len(g))
        out["mannwhitney_p_bad_greater"] = float(u.pvalue)
        out["auc_bad_over_good"] = round(float(auc), 4)
        print(f"\n  bad > good?  AUC {auc:.3f}   p {u.pvalue:.3g}")
        print("  AUC 0.5 = the residual says nothing about which patches he flagged.")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
