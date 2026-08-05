"""End-to-end proof that --resume recovers an interrupted run.

resume_gap.py measures the problem: kill a run, restart the identical command, and the
completed patches are destroyed because the logits store is opened with mode='w'. This
runs the same sequence against the patched code and checks the opposite.

Runs the villa worktree, not the installed clone, via PYTHONPATH.

  python resume_demo.py --side 384 --kill-at 0.3
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
WORKTREE_SRC = ROOT / "_worktrees" / "predict-resume" / "vesuvius" / "src"
MODEL = str(ROOT / "model_m7")
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
SCROLL, VOLUME = "PHerc1218", "20250521120456-8.640um-1.2m-116keV-masked.zarr"
SHAPE = (23247, 7593, 7593)
PATCH, STEP = 192, 0.5


def sliding_starts(size, patch, step):
    if size <= patch:
        return [0]
    n = int(np.ceil((size - patch) / (patch * step))) + 1
    return [int(round(x)) for x in np.linspace(0, size - patch, n)]


def n_patches(lo, hi):
    return int(np.prod([
        len([s for s in sliding_starts(SHAPE[i], PATCH, STEP) if s < hi[i] and s + PATCH > lo[i]])
        for i in range(3)
    ]))


def count_chunks(store: Path) -> int:
    if not store.exists():
        return 0
    return sum(1 for p in store.rglob("*") if p.is_file() and not p.name.startswith("."))


def read_manifest(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", type=int, default=384)
    ap.add_argument("--z", type=int, default=11000)
    ap.add_argument("--y", type=int, default=3400)
    ap.add_argument("--x", type=int, default=3400)
    ap.add_argument("--kill-at", type=float, default=0.3)
    ap.add_argument("--watch-seconds", type=float, default=240.0)
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "resume_demo.json"))
    a = ap.parse_args()

    lo = (a.z, a.y, a.x)
    hi = tuple(lo[i] + a.side for i in range(3))
    bbox = f"{lo[0]}:{hi[0]},{lo[1]}:{hi[1]},{lo[2]}:{hi[2]}"
    total = n_patches(lo, hi)

    work = ROOT / "outputs" / "resume_demo"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    out_dir = work / "logits"
    logits = out_dir / "logits_part_0.zarr"
    manifest = out_dir / "completed_part_0.json"

    cmd = [PY, "run_gpu_roi.py", "--model_path", MODEL,
           "--input_dir", f"{BASE}/{SCROLL}/volumes/{VOLUME}/0",
           "--output_dir", str(out_dir), "--device", "cuda",
           "--disable_tta", "--batch_size", "1", "--num_workers", "2",
           "--bbox", bbox, "--resume"]
    env = {**os.environ,
           "PYTHONPATH": str(WORKTREE_SRC),
           "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1", "PYTHONIOENCODING": "utf-8"}

    print(f"{total} patches over {bbox}, running the predict-resume worktree\n")

    # --- run 1: interrupt it -------------------------------------------------------
    target = max(1, int(total * a.kill_at))
    t0 = time.time()
    log1 = open(work / "run1.log", "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=log1, stderr=subprocess.STDOUT)
    while p.poll() is None:
        if count_chunks(logits) >= target:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=25)
            except subprocess.TimeoutExpired:
                p.kill()
            break
        time.sleep(1.0)
    time.sleep(3.0)
    survived = count_chunks(logits)
    man1 = read_manifest(manifest)
    done1 = len(man1["completed"]) if man1 else 0
    print(f"run 1 interrupted after {time.time() - t0:.0f} s")
    print(f"  chunks on disk: {survived}")
    print(f"  manifest records complete: {done1}\n")

    # --- run 2: resume -------------------------------------------------------------
    # The manifest is flushed every 64 writes, so it lags the chunks on disk; that is the
    # intended trade (a few recomputed patches instead of an fsync per patch), and the
    # check below is that the count never COLLAPSES, not that it never repeats work.
    print(f"run 2: same command with --resume, sampling every {a.poll_seconds:.0f} s")
    traj = []
    log2 = open(work / "run2.log", "w", encoding="utf-8", errors="replace")
    p2 = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=log2, stderr=subprocess.STDOUT)
    t1 = time.time()
    while time.time() - t1 < a.watch_seconds and p2.poll() is None:
        traj.append((round(time.time() - t1, 1), count_chunks(logits)))
        time.sleep(a.poll_seconds)
    finished = p2.poll() is not None
    if not finished:
        p2.send_signal(signal.SIGTERM)
        try:
            p2.wait(timeout=25)
        except subprocess.TimeoutExpired:
            p2.kill()
    time.sleep(2.0)
    lowest = min(c for _, c in traj) if traj else -1
    man2 = read_manifest(manifest)
    done2 = len(man2["completed"]) if man2 else 0

    print("  trajectory (s:chunks): " + ", ".join(f"{t:.0f}:{c}" for t, c in traj[:20]))
    print(f"  lowest chunk count seen: {lowest} (was {survived} before restart)")
    print(f"  manifest now records: {done2} of {total}")

    res = {
        "bbox": bbox, "total_patches": total,
        "chunks_after_interrupt": survived,
        "manifest_after_interrupt": done1,
        "lowest_chunks_during_resume": lowest,
        "manifest_after_resume": done2,
        "run2_finished": finished,
        "trajectory": traj,
        "work_preserved": lowest >= survived,
        "made_progress": done2 > done1,
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"\n  work preserved across restart: {res['work_preserved']}")
    print(f"  resumed run made progress:     {res['made_progress']}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
