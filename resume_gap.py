"""Show what an interrupted `vesuvius.predict` costs you, and whether restarting recovers it.

Why this matters at all: on this machine's measured 11 MB/s link, a whole-scroll pass over
PHerc1218 is ~36 h even under the best traversal policy (see bench_stream_inference.py). A
36-hour job on a home connection WILL be interrupted. So the question is not how fast the run
is, it is whether an interrupted run is worth anything afterwards.

`inference.py` opens `logits_part_{id}.zarr` with `mode='w'` and chunks it one-chunk-per-patch
with `write_empty_chunks=False`, so:

  * completed patches are individually visible on disk as chunk files, and
  * restarting the same command re-creates the store from scratch.

This measures both halves rather than asserting them:

  1. run, kill at roughly half, count the patch chunks that survived on disk;
  2. restart the identical command, and count again a few seconds in.

  python resume_gap.py --side 384 --kill-at 0.4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PY = "C:/Users/PC/miniconda3/envs/blogging/python.exe"
MODEL = str(ROOT / "model_m7")
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCROLL, VOLUME = "PHerc1218", "20250521120456-8.640um-1.2m-116keV-masked.zarr"
SHAPE = (23247, 7593, 7593)
PATCH, STEP = 192, 0.5


def sliding_starts(size: int, patch: int, step: float) -> list[int]:
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * step))) + 1
    return [int(round(x)) for x in np.linspace(0, size - patch, n)]


def n_patches(lo, hi) -> int:
    return int(np.prod([
        len([s for s in sliding_starts(SHAPE[i], PATCH, STEP) if s < hi[i] and s + PATCH > lo[i]])
        for i in range(3)
    ]))


def count_patch_chunks(store: Path) -> int:
    """Chunk files present, i.e. patches whose logits actually reached disk.

    zarr 2 writes one file per chunk and names metadata with a leading dot, so anything not
    starting with '.' is a chunk. Chunks are (1, C, pZ, pY, pX) => exactly one per patch.
    """
    if not store.exists():
        return 0
    return sum(1 for p in store.rglob("*") if p.is_file() and not p.name.startswith("."))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=384)
    ap.add_argument("--z", type=int, default=11000)
    ap.add_argument("--y", type=int, default=3400)
    ap.add_argument("--x", type=int, default=3400)
    ap.add_argument("--kill-at", type=float, default=0.4, help="fraction of patches before kill")
    ap.add_argument("--restart-seconds", type=float, default=200.0,
                    help="how long to watch run 2 before giving up")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "resume_gap.json"))
    a = ap.parse_args()

    lo = (a.z, a.y, a.x)
    hi = tuple(lo[i] + a.side for i in range(3))
    bbox = f"{lo[0]}:{hi[0]},{lo[1]}:{hi[1]},{lo[2]}:{hi[2]}"
    total = n_patches(lo, hi)

    work = ROOT / "outputs" / "resume_gap"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    logits = work / "logits" / "logits_part_0.zarr"

    cmd = [PY, "run_gpu_roi.py", "--model_path", MODEL,
           "--input_dir", f"{BASE}/{SCROLL}/volumes/{VOLUME}/0",
           "--output_dir", str(work / "logits"), "--device", "cuda",
           "--disable_tta", "--batch_size", "1", "--num_workers", "2", "--bbox", bbox]
    env = {**os.environ, "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
           "PYTHONIOENCODING": "utf-8"}

    print(f"{total} patches over {bbox}")
    print(f"run 1: killing once {a.kill_at:.0%} of patches have reached disk\n")

    target = max(1, int(total * a.kill_at))
    t0 = time.time()
    p = subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    killed_at = None
    while p.poll() is None:
        done = count_patch_chunks(logits)
        if done >= target:
            killed_at = done
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
            break
        time.sleep(1.0)
    elapsed_1 = time.time() - t0
    time.sleep(2.0)
    survived = count_patch_chunks(logits)

    print(f"  killed after {elapsed_1:.0f} s with {killed_at} patches written")
    print(f"  chunks surviving on disk: {survived} of {total}\n")

    # Sampling once is not enough, and getting this wrong the first time is instructive:
    # a single sample 25 s in still showed all 86 chunks, because the restart had not yet
    # finished loading the model and had not reached _create_output_stores(). The signal is
    # the TRAJECTORY -- if mode='w' recreates the array, the count collapses and then climbs
    # again from zero, so what matters is the minimum reached, not the value at any instant.
    print(f"run 2: identical command, sampling every {a.poll_seconds:.0f} s "
          f"for up to {a.restart_seconds:.0f} s")
    p2 = subprocess.Popen(cmd, env=env, cwd=str(ROOT),
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    trajectory = []
    t1 = time.time()
    while time.time() - t1 < a.restart_seconds and p2.poll() is None:
        trajectory.append((round(time.time() - t1, 1), count_patch_chunks(logits)))
        time.sleep(a.poll_seconds)
    p2.send_signal(signal.SIGTERM)
    try:
        p2.wait(timeout=20)
    except subprocess.TimeoutExpired:
        p2.kill()
    after_restart = min(c for _, c in trajectory) if trajectory else -1
    print("  trajectory (s, chunks): " + ", ".join(f"{t}:{c}" for t, c in trajectory))

    res = {
        "bbox": bbox, "total_patches": total,
        "killed_after_s": round(elapsed_1, 1),
        "patches_written_before_kill": killed_at,
        "chunks_surviving_on_disk": survived,
        "chunks_after_restart_min": after_restart,
        "restart_trajectory": trajectory,
        "restart_watch_seconds": a.restart_seconds,
        "work_preserved": after_restart >= survived,
        "reading": (
            "If the trajectory minimum is far below chunks_surviving_on_disk, the restart "
            "discarded completed work: inference.py opens the logits store with mode='w'."
        ),
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"  minimum chunk count seen during run 2: {after_restart}")
    print(f"\n  work preserved across restart: {res['work_preserved']}")
    print(json.dumps(res, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
