"""Week-1 GPU gate, throughput half: can a 3090 keep up with streamed patches?

The I/O half (STRATEGY.md §00.0s B) says a naive tiled traversal reads PHerc1218 30.5x over
and that this machine's link caps at 11 MB/s. That bounds the job from below at ~56 h even
with the best cache policy. This measures the other side: whether inference is anywhere near
the bottleneck, or whether the whole thing is I/O-bound and the GPU idles.

It runs villa's OWN predict path (through run_gpu_roi.py, which only shims two torch-2.4
signature changes) against the remote store over a --bbox region, with no local copy.

  python bench_gpu_stream_gate.py                       # default 768^3 region of PHerc1218
  python bench_gpu_stream_gate.py --side 384 --dry-run  # geometry only, no GPU

Reported: wall-clock, patches actually run, patches/s, chunks touched, implied sustained
MB/s, and peak GPU memory sampled from nvidia-smi (not torch, since predict is a subprocess
and Windows spills silently -- §00.1 rule 2).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PY = "C:/Users/PC/miniconda3/envs/blogging/python.exe"
MODEL = str(ROOT / "model_m7")
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCROLL = "PHerc1218"
VOLUME = "20250521120456-8.640um-1.2m-116keV-masked.zarr"
SHAPE = (23247, 7593, 7593)          # verified from the level-0 .zarray
PATCH, STEP, CHUNK = 192, 0.5, 128


def sliding_starts(size: int, patch: int, step: float) -> list[int]:
    """villa/nnU-Net convention: starts spread uniformly across the axis (see #1247)."""
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * step))) + 1
    return [int(round(x)) for x in np.linspace(0, size - patch, n)]


def geometry(lo: tuple[int, int, int], hi: tuple[int, int, int]) -> dict:
    """Patches the ROI selects from the FULL-volume grid, and the chunks they touch."""
    per_axis, chunk_ranges = [], []
    for i in range(3):
        starts = [s for s in sliding_starts(SHAPE[i], PATCH, STEP)
                  if s < hi[i] and s + PATCH > lo[i]]
        per_axis.append(starts)
        lo_c = min(s // CHUNK for s in starts)
        hi_c = max((s + PATCH - 1) // CHUNK for s in starts)
        chunk_ranges.append(hi_c - lo_c + 1)
    n_patches = int(np.prod([len(s) for s in per_axis]))
    n_chunks = int(np.prod(chunk_ranges))
    return {
        "starts_per_axis": [len(s) for s in per_axis],
        "n_patches": n_patches,
        "n_chunks_spanned": n_chunks,
        "bytes_if_each_chunk_once": n_chunks * CHUNK ** 3,
    }


class GpuSampler(threading.Thread):
    """Poll nvidia-smi rather than torch: predict is a subprocess, and on Windows the
    driver spills into shared system memory without raising, so torch's own counter in
    the parent process would report nothing at all."""

    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period, self.stop_flag, self.peak_mib, self.samples = period, False, 0, 0

    def run(self) -> None:
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10)
                v = int(out.stdout.strip().splitlines()[0])
                self.peak_mib = max(self.peak_mib, v)
                self.samples += 1
            except Exception:
                pass
            time.sleep(self.period)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=768)
    ap.add_argument("--z", type=int, default=11000, help="region origin, mid-volume by default")
    ap.add_argument("--y", type=int, default=3400)
    ap.add_argument("--x", type=int, default=3400)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "stream_gate_gpu.json"))
    a = ap.parse_args()

    lo = (a.z, a.y, a.x)
    hi = tuple(lo[i] + a.side for i in range(3))
    geom = geometry(lo, hi)
    bbox = f"{lo[0]}:{hi[0]},{lo[1]}:{hi[1]},{lo[2]}:{hi[2]}"

    print(f"{SCROLL} level 0 {SHAPE}, region {a.side}^3 at {lo}")
    print(f"  bbox {bbox}")
    print(f"  patches selected from the full-volume grid: {geom['n_patches']} "
          f"({geom['starts_per_axis']} per axis)")
    print(f"  chunks spanned: {geom['n_chunks_spanned']} "
          f"= {geom['bytes_if_each_chunk_once'] / 1e6:.0f} MB if each is read once")
    if a.dry_run:
        return

    work = ROOT / "outputs" / "stream_gate"
    if work.exists():
        import shutil
        shutil.rmtree(work)
    work.mkdir(parents=True)

    url = f"{BASE}/{SCROLL}/volumes/{VOLUME}/0"
    cmd = [PY, "run_gpu_roi.py", "--model_path", MODEL, "--input_dir", url,
           "--output_dir", str(work / "logits"), "--device", "cuda",
           "--disable_tta", "--batch_size", "1", "--num_workers", "2", "--bbox", bbox]
    env = {"nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1", "PYTHONIOENCODING": "utf-8"}

    import os
    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    r = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    elapsed = time.time() - t0
    sampler.stop_flag = True
    sampler.join(timeout=5)

    ok = r.returncode == 0
    res = {
        "scroll": SCROLL, "level": 0, "shape": list(SHAPE), "region_side": a.side,
        "region_lo": list(lo), "bbox": bbox, **geom,
        "ok": ok, "returncode": r.returncode, "elapsed_s": round(elapsed, 1),
        "patches_per_s": round(geom["n_patches"] / elapsed, 3) if ok and elapsed else None,
        "implied_mb_s_if_each_chunk_once":
            round(geom["bytes_if_each_chunk_once"] / 1e6 / elapsed, 2) if ok and elapsed else None,
        "peak_gpu_mib": sampler.peak_mib, "gpu_samples": sampler.samples,
        "note": ("Wall-clock covers streaming AND inference; the two overlap, so "
                 "implied MB/s is a lower bound on link use, not a pure I/O figure."),
    }
    if not ok:
        res["stderr_tail"] = (r.stderr or "")[-1500:]
        res["stdout_tail"] = (r.stdout or "")[-800:]

    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
