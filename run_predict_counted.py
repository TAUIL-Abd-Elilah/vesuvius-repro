"""Run vesuvius.predict with every HTTP fetch counted.

Task #6 exists because the first cache A/B was a null: three runs of one identical
configuration came back 41.5 s, 239 s and 254 s, so link variance swamped the effect and
the reading had to be retracted (villa#1325). Adding replicates would fight that noise.
Removing the clock avoids it.

Read amplification is a count, not a duration. zarr 3 opens these volumes as an FsspecStore
over HTTPFileSystem, and every chunk fetch goes through `HTTPFileSystem._cat_file`, so
wrapping that one method yields the exact number of requests and bytes a run issued --
deterministic, and identical whether the link is fast or falling over.

    python run_predict_counted.py --io-counts-out counts.json <normal predict args>

Set VESUVIUS_CHUNK_CACHE_DIR to route reads through an fsspec simplecache chain, which is
the arm under test. Counting sits *outside* the cache, so a cache hit costs no fetch and
therefore shows up as a genuinely absent request rather than a cheaper one.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import zarr  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("VESUVIUS_GPU_MEM_FRACTION", "0.88"))
    )

from fsspec.implementations.http import HTTPFileSystem  # noqa: E402

STATS = {"fetches": 0, "bytes": 0, "ranged": 0}
PER_KEY: Counter = Counter()
_orig_cat_file = HTTPFileSystem._cat_file


async def _counted_cat_file(self, url, start=None, end=None, **kwargs):
    data = await _orig_cat_file(self, url, start=start, end=end, **kwargs)
    STATS["fetches"] += 1
    STATS["bytes"] += len(data) if data is not None else 0
    if start is not None or end is not None:
        STATS["ranged"] += 1
    PER_KEY[url] += 1
    return data


HTTPFileSystem._cat_file = _counted_cat_file

# --- optional chunk cache, the arm under test -------------------------------------------
import vesuvius.data.utils as _vutils  # noqa: E402
import vesuvius.data.vc_dataset as _vds  # noqa: E402
import vesuvius.data.volume as _vvol  # noqa: E402

CACHE_DIR = os.environ.get("VESUVIUS_CHUNK_CACHE_DIR")
_orig_open_zarr = _vutils.open_zarr


def _cached_open_zarr(path, mode="r", storage_options=None, **kwargs):
    if CACHE_DIR and mode == "r" and isinstance(path, str) and path.startswith("http"):
        so = dict(storage_options or {})
        so["simplecache"] = {"cache_storage": CACHE_DIR}
        for k in ("shape", "chunks", "dtype", "compressor", "fill_value", "order", "verbose"):
            kwargs.pop(k, None)
        return zarr.open("simplecache::" + path, mode="r", storage_options=so, **kwargs)
    return _orig_open_zarr(path, mode=mode, storage_options=storage_options, **kwargs)


if CACHE_DIR:
    os.makedirs(CACHE_DIR, exist_ok=True)
    _vutils.open_zarr = _cached_open_zarr
    _vvol.open_zarr = _cached_open_zarr
    _vds.open_zarr = _cached_open_zarr

OUT = None


def _dump():
    if OUT is None:
        return
    unique = len(PER_KEY)
    # Metadata keys (.zarray/.zattrs/.zgroup) are fetched too and are not chunks; counting
    # them as chunk traffic would inflate amplification on small regions.
    chunk_keys = {k: v for k, v in PER_KEY.items()
                  if not k.rsplit("/", 1)[-1].startswith(".")}
    chunk_fetches = sum(chunk_keys.values())
    payload = {
        "cache_dir": CACHE_DIR,
        "total_fetches": STATS["fetches"],
        "total_bytes": STATS["bytes"],
        "ranged_fetches": STATS["ranged"],
        "unique_urls": unique,
        "chunk_fetches": chunk_fetches,
        "unique_chunks": len(chunk_keys),
        "amplification": round(chunk_fetches / len(chunk_keys), 4) if chunk_keys else None,
        "max_refetches_of_one_chunk": max(chunk_keys.values()) if chunk_keys else 0,
        "note": ("amplification = chunk fetches / distinct chunks. 1.0 means every chunk was "
                 "fetched exactly once. Counting wraps HTTPFileSystem._cat_file, so it is "
                 "outside any cache: a cache hit is an absent request, not a cheap one."),
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[io] {chunk_fetches:,} chunk fetches over {len(chunk_keys):,} distinct chunks "
          f"= {payload['amplification']}x, {STATS['bytes'] / 1e6:.0f} MB -> {OUT}",
          file=sys.stderr, flush=True)


atexit.register(_dump)

from vesuvius.models.run import inference  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--io-counts-out" in argv:
        i = argv.index("--io-counts-out")
        OUT = argv[i + 1]
        del argv[i:i + 2]
    sys.argv = ["vesuvius.predict"] + argv
    inference.main()
