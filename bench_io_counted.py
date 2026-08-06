"""Validate the published read-amplification simulation against measured HTTP fetches.

Task #6. The first cache A/B was a null because one configuration varied 41.5 s to 253.7 s
(villa#1325, retracted). The fix is not more replicates -- it is dropping the clock. Read
amplification is a count, so `run_predict_counted.py` wraps `HTTPFileSystem._cat_file` and
counts every chunk request the real inference path issues. Deterministic, and unaffected by
whether the link is fast or falling over.

That makes the interesting comparison possible: I published a *simulated* 30.5x to #1325.
This runs the identical bbox for real and asks whether the simulator predicts the measured
fetch count. If it does, the extrapolation stands. If it does not, I owe that thread a
correction.

Arms, run interleaved rather than blocked so any drift hits both equally:

    sim        bench_stream_inference.simulate() over the same patch grid, no cache
    nocache    measured, no chunk cache
    cache      measured, fsspec simplecache

    python bench_io_counted.py --side 256 --reps 2

Failed fetches raise inside `_cat_file` and are therefore never counted, so a flaky link
costs wall-clock and retries but does not inflate the amplification figure.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PY = "C:/Users/PC/miniconda3/envs/vesuvius/python.exe"
WORKTREE_SRC = ROOT / "_worktrees" / "predict-resume" / "vesuvius" / "src"
MODEL = str(ROOT / "model_m7")
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCROLL, VOLUME = "PHerc1218", "20250521120456-8.640um-1.2m-116keV-masked.zarr"
SHAPE = (23247, 7593, 7593)
PATCH, STEP, CHUNK = 192, 0.5, 128

sys.path.insert(0, str(ROOT / "vesuvius-repro"))
from bench_stream_inference import simulate, sliding_starts  # noqa: E402


def simulate_bbox(lo, hi) -> dict:
    """Predicted fetches for exactly the patches this bbox selects from the full grid.

    Not simulate()'s own region grid: an ROI takes its patches from the FULL-volume grid
    (#1247), so the two must be built the same way or the comparison is meaningless.
    """
    per_axis = [[s for s in sliding_starts(SHAPE[i], PATCH, STEP)
                 if s < hi[i] and s + PATCH > lo[i]] for i in range(3)]
    grid = [(z, y, x) for z in per_axis[0] for y in per_axis[1] for x in per_axis[2]]
    touched, fetches = set(), 0
    for (z, y, x) in grid:
        rz = range(z // CHUNK, (z + PATCH - 1) // CHUNK + 1)
        ry = range(y // CHUNK, (y + PATCH - 1) // CHUNK + 1)
        rx = range(x // CHUNK, (x + PATCH - 1) // CHUNK + 1)
        for cz in rz:
            for cy in ry:
                for cx in rx:
                    touched.add((cz, cy, cx))
                    fetches += 1
    return {"arm": "sim_nocache", "patches": len(grid), "chunk_fetches": fetches,
            "unique_chunks": len(touched),
            "amplification": round(fetches / len(touched), 4) if touched else None}


def run_measured(label, bbox, work: Path, cache_dir: str | None) -> dict:
    arm_dir = work / label
    if arm_dir.exists():
        shutil.rmtree(arm_dir)
    arm_dir.mkdir(parents=True)
    counts = arm_dir / "io.json"
    env = {**os.environ, "PYTHONPATH": str(WORKTREE_SRC),
           "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1", "PYTHONIOENCODING": "utf-8"}
    if cache_dir:
        if Path(cache_dir).exists():
            shutil.rmtree(cache_dir)          # cold cache: measures within-run reuse only
        os.makedirs(cache_dir, exist_ok=True)
        env["VESUVIUS_CHUNK_CACHE_DIR"] = cache_dir
    else:
        env.pop("VESUVIUS_CHUNK_CACHE_DIR", None)

    cmd = [PY, "run_predict_counted.py", "--io-counts-out", str(counts),
           "--model_path", MODEL, "--input_dir", f"{BASE}/{SCROLL}/volumes/{VOLUME}/0",
           "--output_dir", str(arm_dir / "logits"), "--device", "cuda",
           "--disable_tta", "--batch_size", "1",
           # num_workers 0 is required, not a preference: DataLoader workers are separate
           # processes, the counter lives in the parent, and with workers the run reports
           # zero fetches. Found the hard way.
           "--num_workers", "0", "--bbox", bbox]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    (arm_dir / "run.log").write_text((r.stdout or "") + (r.stderr or ""), encoding="utf-8")

    out = {"arm": label, "rc": r.returncode, "elapsed_s": round(elapsed, 1),
           "completed": r.returncode == 0}
    if counts.exists():
        out.update(json.loads(counts.read_text()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--z", type=int, default=11000)
    ap.add_argument("--y", type=int, default=3400)
    ap.add_argument("--x", type=int, default=3400)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--work", default="outputs/io_counted")
    ap.add_argument("--cache-dir", default="C:/Users/PC/AppData/Local/Temp/claude/vesuv_iocache")
    ap.add_argument("--out", default=str(ROOT / "results" / "io_counted.json"))
    a = ap.parse_args()

    lo = (a.z, a.y, a.x)
    hi = tuple(lo[i] + a.side for i in range(3))
    bbox = f"{lo[0]}:{hi[0]},{lo[1]}:{hi[1]},{lo[2]}:{hi[2]}"
    work = ROOT / a.work
    work.mkdir(parents=True, exist_ok=True)

    sim = simulate_bbox(lo, hi)
    print(f"{SCROLL} {a.side}^3 at {lo}\n  bbox {bbox}")
    print(f"  simulated: {sim['patches']} patches, {sim['chunk_fetches']:,} fetches over "
          f"{sim['unique_chunks']:,} chunks = {sim['amplification']}x\n", flush=True)

    runs = [sim]
    for rep in range(a.reps):
        for label, cache in (("nocache", None), ("cache", a.cache_dir)):
            tag = f"{label}_r{rep}"
            print(f"  {tag} ...", flush=True)
            res = run_measured(tag, bbox, work, cache)
            res["rep"] = rep
            res["base_arm"] = label
            runs.append(res)
            amp = res.get("amplification")
            print(f"    rc={res['rc']} {res['elapsed_s']}s  "
                  f"{res.get('chunk_fetches', 0):,} fetches / "
                  f"{res.get('unique_chunks', 0):,} chunks = {amp}x  "
                  f"{res.get('total_bytes', 0) / 1e6:.0f} MB", flush=True)

    ok = [r for r in runs if r.get("completed")]
    payload = {
        "scroll": SCROLL, "bbox": bbox, "side": a.side,
        "method": ("fetch counts wrap HTTPFileSystem._cat_file, so they are exact and "
                   "independent of link speed; failed fetches raise and are not counted"),
        "simulated": sim,
        "runs": runs,
        "completed_runs": len(ok),
    }
    Path(a.out).write_text(json.dumps(payload, indent=1))

    print("\n  arm            fetches   unique      amp   completed")
    for r in runs:
        print(f"  {r['arm']:<14} {r.get('chunk_fetches', 0):>8,} {r.get('unique_chunks', 0):>8,}"
              f" {str(r.get('amplification')):>8}   {r.get('completed', '-')}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
