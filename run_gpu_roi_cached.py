"""run_gpu_roi.py plus an on-disk chunk cache, for measuring what the cache is worth.

villa#1325 (aistae) reports that the inference read path has no chunk cache, so overlapping
patches re-fetch the same chunks; #1327 is their draft fix. This file exists only to MEASURE
that on a real GPU run with real weights — the half aistae cannot run, having no GPU — and it
deliberately does not change villa. Nothing here belongs upstream; #1327 is the right place
for the actual fix, and this credits it rather than competing with it.

Mechanism: `vesuvius.data.utils.open_zarr` is the single point through which the read path
opens data, but `volume.py` and `vc_dataset.py` bind it at import time, so all three names are
patched. Remote reads are re-opened through an fsspec `simplecache::` chain. The open-data
volumes are immutable, so `simplecache` (no revalidation) is right and `filecache` would only
buy a round trip per chunk.

    VESUVIUS_CHUNK_CACHE_DIR=D:/tmp/chunkcache python run_gpu_roi_cached.py <predict args>

With the variable unset it is byte-for-byte the behaviour of run_gpu_roi.py.
"""

from __future__ import annotations

import os
import sys

# Cap the CUDA allocator BEFORE torch initialises the context (see run_gpu_roi.py, §00.1 r2).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import zarr  # noqa: E402

if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(
        float(os.environ.get("VESUVIUS_GPU_MEM_FRACTION", "0.88"))
    )

# --- shim 1: torch.compiler.disable(reason=...) gained `reason` after 2.4 ---
_orig_disable = torch.compiler.disable


def _disable(fn=None, *, recursive=True, reason=None):  # noqa: ARG001
    return _orig_disable(fn, recursive=recursive)


torch.compiler.disable = _disable

import vesuvius.data.utils as _vutils  # noqa: E402
import vesuvius.data.vc_dataset as _vds  # noqa: E402
import vesuvius.data.volume as _vvol  # noqa: E402

_orig_open_zarr = _vutils.open_zarr
CACHE_DIR = os.environ.get("VESUVIUS_CHUNK_CACHE_DIR")
_stats = {"cached_opens": 0, "passthrough_opens": 0}


def _cached_open_zarr(path, mode="r", storage_options=None, **kwargs):
    remote = isinstance(path, str) and path.startswith(("http://", "https://"))
    if CACHE_DIR and mode == "r" and remote:
        so = dict(storage_options or {})
        so["simplecache"] = {"cache_storage": CACHE_DIR}
        _stats["cached_opens"] += 1
        # strip creation-only kwargs; this branch is read-mode by construction
        for k in ("shape", "chunks", "dtype", "compressor", "fill_value", "order", "verbose"):
            kwargs.pop(k, None)
        return zarr.open("simplecache::" + path, mode="r", storage_options=so, **kwargs)
    _stats["passthrough_opens"] += 1
    return _orig_open_zarr(path, mode=mode, storage_options=storage_options, **kwargs)


if CACHE_DIR:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # all three bindings: utils defines it, volume and vc_dataset imported it by value
    _vutils.open_zarr = _cached_open_zarr
    _vvol.open_zarr = _cached_open_zarr
    _vds.open_zarr = _cached_open_zarr
    print(f"[chunk cache] on, at {CACHE_DIR}", file=sys.stderr)

from vesuvius.models.run import inference  # noqa: E402

if __name__ == "__main__":
    sys.argv = ["vesuvius.predict"] + sys.argv[1:]
    try:
        inference.main()
    finally:
        print(f"[chunk cache] opens: {_stats}", file=sys.stderr)
