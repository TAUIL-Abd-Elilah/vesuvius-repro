"""Can whole-scroll inference be STREAMED, or does the patch grid make it read the volume
many times over?

The GPU half of the week-1 gate needs a free GPU. This is the other half, and it is the half
that can kill the plan on its own: villa's sliding window is 192^3 with half-overlap, while
the store is chunked at 128^3. The two grids do not align, so a patch touches up to 3x3x3
chunks and neighbouring patches re-touch the same ones. If the resulting READ AMPLIFICATION
is large, streaming a 1.2 TB scroll means fetching many TB, and "no download" is a fiction.

Everything here is CPU + network only.

Exact byte accounting is possible because the public volumes are `"compressor": null` uint8
at 128^3 -- every chunk is exactly 2 MiB on the wire.

  python bench_stream_inference.py --metadata       # zarr geometry, all 13 eligible scrolls
  python bench_stream_inference.py --amplification  # the decisive simulation
  python bench_stream_inference.py --throughput     # real sustained MB/s from the bucket
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"

# scroll -> masked CT volume, from catalog.json; all 13 grand-prize-eligible volumes
ELIGIBLE = {
    "PHerc0125": "20250821151825-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0191": "20250821151635-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0211": "20250821151803-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0257": "20250821151750-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0268": "20251110183117-8.640um-1.2m-116keV-masked.zarr",
    "PHerc0358": "20250821151737-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0800": "20250521135224-8.640um-1.2m-116keV-masked.zarr",
    "PHerc0813": "20250821151723-9.362um-1.2m-113keV-masked.zarr",
    "PHerc0826": "20250821151701-9.362um-1.2m-113keV-masked.zarr",
    "PHerc1203": "20250820131727-9.362um-1.2m-113keV-masked.zarr",
    "PHerc1218": "20250521120456-8.640um-1.2m-116keV-masked.zarr",
    "PHerc1447": "20250521151220-8.640um-1.2m-116keV-masked.zarr",
    "PHerc1545": "20250821151648-9.362um-1.2m-113keV-masked.zarr",
}

PATCH = 192
STEP = 0.5
CHUNK = 128


def sliding_starts(size: int, patch: int, step: float) -> list[int]:
    """villa/nnU-Net convention: starts spread UNIFORMLY across the axis, not at fixed stride.

    Getting this wrong is not academic -- #1247 was exactly this bug, and an ROI-origin grid
    gave 27 chunks and a flattering 384x where the true full-volume grid gives 125.
    """
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * step))) + 1
    return [int(round(x)) for x in np.linspace(0, size - patch, n)]


def chunks_for_patch(start: tuple[int, int, int], patch: int, chunk: int):
    """Chunk indices a patch overlaps, per axis."""
    return [range(s // chunk, (s + patch - 1) // chunk + 1) for s in start]


def morton(cz: int, cy: int, cx: int, bits: int = 21) -> int:
    """Z-order key over chunk indices.

    chunk_blocked resets locality at every block boundary; a Morton curve keeps it in all
    three axes at once, which is aistae's point in villa#1325 and the reason it wins.
    """
    key = 0
    for b in range(bits):
        key |= ((cz >> b) & 1) << (3 * b + 2)
        key |= ((cy >> b) & 1) << (3 * b + 1)
        key |= ((cx >> b) & 1) << (3 * b)
    return key


def simulate(region, cache_chunks, order="zyx", patch=PATCH, step=STEP, chunk=CHUNK):
    """Count chunk fetches under an LRU cache for one traversal order.

    Returns (fetches, distinct_chunks, n_patches).
    """
    starts = [sliding_starts(region[i], patch, step) for i in range(3)]
    grid = [(z, y, x) for z in starts[0] for y in starts[1] for x in starts[2]]
    if order == "xyz":                      # innermost axis varies slowest
        grid = [(z, y, x) for x in starts[2] for y in starts[1] for z in starts[0]]
    elif order == "chunk_blocked":
        # walk patches grouped by the chunk-slab they start in, so a slab's chunks are
        # touched consecutively and can leave the cache for good afterwards
        grid.sort(key=lambda p: (p[0] // chunk, p[1] // chunk, p[2] // chunk))
    elif order == "morton":
        grid.sort(key=lambda p: morton(p[0] // chunk, p[1] // chunk, p[2] // chunk))

    lru: OrderedDict = OrderedDict()
    fetches = 0
    distinct = set()
    for p in grid:
        rz, ry, rx = chunks_for_patch(p, patch, chunk)
        for cz in rz:
            for cy in ry:
                for cx in rx:
                    key = (cz, cy, cx)
                    distinct.add(key)
                    if cache_chunks <= 0:        # no cache at all: every touch is a fetch
                        fetches += 1
                        continue
                    if key in lru:
                        lru.move_to_end(key)
                        continue
                    fetches += 1
                    lru[key] = True
                    if len(lru) > cache_chunks:
                        lru.popitem(last=False)
    return fetches, len(distinct), len(grid)


def cmd_metadata():
    print(f"{'scroll':<11} {'shape':<24} {'chunks':<14} {'dtype':<6} {'compressor':<11} {'TB':>7}")
    print("-" * 82)
    out = {}
    for s, v in ELIGIBLE.items():
        try:
            r = requests.get(f"{BASE}/{s}/volumes/{v}/0/.zarray", timeout=30)
            r.raise_for_status()
            z = r.json()
        except Exception as e:
            print(f"{s:<11} ERROR {e}")
            continue
        shape, ch = z["shape"], z["chunks"]
        tb = np.prod(shape, dtype=np.float64) / 1024 ** 4
        out[s] = dict(shape=shape, chunks=ch, dtype=z["dtype"], compressor=z["compressor"], tb=tb)
        print(f"{s:<11} {str(shape):<24} {str(ch):<14} {z['dtype']:<6} "
              f"{str(z['compressor']):<11} {tb:7.2f}")
    tot = sum(v["tb"] for v in out.values())
    print(f"\n  13 eligible scrolls total: {tot:.1f} TB  (mean {tot/max(len(out),1):.2f} TB)")
    json.dump(out, open("results/eligible_zarr_geometry.json", "w"), indent=1)
    print("  wrote results/eligible_zarr_geometry.json")


def cmd_amplification(region_side=1536):
    """Simulate a slab of a REAL scroll cross-section, not a toy cube.

    A cube small enough to simulate quickly also fits entirely in the larger caches, which
    makes every policy look perfect (the first version of this reported 1.00x across the
    board). What decides the answer is cache size versus the traversal's WORKING SET, and on
    a real scroll one z-slab of chunks is (7593/128)^2 = 3,519 chunks = 6.9 GiB -- far larger
    than any cache we would use. So simulate the true cross-section and a few patch-planes.
    """
    cross = 7593                                   # PHerc1218 x,y extent
    region = (region_side, cross, cross)
    nch = [int(np.ceil(region[i] / CHUNK)) for i in range(3)]
    vol_chunks = float(np.prod(nch))
    print(f"Slab {region[0]} x {cross} x {cross} of PHerc1218 (real cross-section)")
    print(f"patch {PATCH}^3 step {STEP} | chunk {CHUNK}^3 = 2 MiB")
    print(f"Chunks spanning the slab: {vol_chunks:,.0f}  ({vol_chunks*2/1024:.1f} GiB)")
    print(f"One z-slab of chunks (the reuse working set): {nch[1]*nch[2]:,} chunks "
          f"= {nch[1]*nch[2]*2/1024:.1f} GiB\n")
    print(f"{'traversal':<15} {'cache':>10} {'fetches':>12} {'amplification':>14}")
    print("-" * 56)
    rows = []
    for order in ("zyx", "chunk_blocked", "morton"):
        for cache_mib in (0, 1024, 4096, 8192, 16384):
            cache_chunks = cache_mib // 2          # 2 MiB per chunk
            f, d, n = simulate(region, cache_chunks, order)
            amp = f / d                            # vs reading each needed chunk exactly once
            label = "none" if cache_mib == 0 else f"{cache_mib} MiB"
            print(f"{order:<15} {label:>10} {f:>12,} {amp:>13.2f}x")
            rows.append(dict(order=order, cache_mib=cache_mib, fetches=f,
                             distinct=d, patches=n, amplification=amp))
    best = min(rows, key=lambda r: r["amplification"])
    print(f"\nBest: {best['order']} @ {best['cache_mib']} MiB -> {best['amplification']:.2f}x")
    print(f"Patches simulated: {rows[0]['patches']:,} | distinct chunks: {rows[0]['distinct']:,}")
    # PHerc1218 level 0 is 23247 x 7593 x 7593 uint8, verified from the .zarray. An earlier
    # version of this print hardcoded 1.25 TB, which understates it by 7%.
    vol_tb = 23247 * 7593 * 7593 / 1e12
    bw = 11.0                                  # MB/s measured on this machine, cmd_throughput
    print(f"\nWhat this means for PHerc1218 ({vol_tb:.2f} TB) at {bw:.0f} MB/s:")
    for r in rows:
        if r["cache_mib"] in (0, 4096, 16384):
            fetched = vol_tb * r["amplification"]
            print(f"  {r['order']:<15} {r['cache_mib']:>5} MiB cache -> "
                  f"{fetched:6.2f} TB {fetched * 1e12 / (bw * 1e6) / 3600:8.1f} h")
    json.dump(rows, open("results/stream_read_amplification.json", "w"), indent=1)
    print("\nwrote results/stream_read_amplification.json")


def cmd_throughput(scroll="PHerc1218", n=24, workers=8):
    v = ELIGIBLE[scroll]
    z = requests.get(f"{BASE}/{scroll}/volumes/{v}/0/.zarray", timeout=30).json()
    shape, ch = z["shape"], z["chunks"]
    nz, ny, nx = [shape[i] // ch[i] for i in range(3)]
    rng = np.random.default_rng(0)
    # sample from the middle of the volume; the masked edges are fill_value and may 404
    keys = [(int(rng.integers(nz // 3, 2 * nz // 3)),
             int(rng.integers(ny // 3, 2 * ny // 3)),
             int(rng.integers(nx // 3, 2 * nx // 3))) for _ in range(n)]

    def fetch(k):
        url = f"{BASE}/{scroll}/volumes/{v}/0/{k[0]}/{k[1]}/{k[2]}"
        t = time.time()
        try:
            r = requests.get(url, timeout=60)
            return (len(r.content) if r.status_code == 200 else 0,
                    time.time() - t, r.status_code)
        except Exception:
            return (0, time.time() - t, -1)

    for label, w in (("serial", 1), (f"concurrent x{workers}", workers)):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=w) as ex:
            res = list(ex.map(fetch, keys))
        el = time.time() - t0
        got = sum(r[0] for r in res)
        ok = sum(1 for r in res if r[2] == 200)
        miss = sum(1 for r in res if r[2] == 404)
        lat = np.median([r[1] for r in res])
        print(f"{label:<16} {ok:>2}/{n} ok ({miss} 404=empty)  {got/1e6:7.1f} MB in {el:6.2f}s "
              f"-> {got/1e6/el:7.1f} MB/s  median latency {lat*1000:6.0f} ms")
    print("\n(404 = chunk never written: masked volumes are sparse, which is itself a saving.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--amplification", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    ap.add_argument("--region", type=int, default=1536)
    a = ap.parse_args()
    import os
    os.makedirs("results", exist_ok=True)
    if a.metadata:
        cmd_metadata()
    if a.amplification:
        cmd_amplification(a.region)
    if a.throughput:
        cmd_throughput()
