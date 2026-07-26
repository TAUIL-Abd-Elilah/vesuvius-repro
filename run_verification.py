"""End-to-end: pick a region of a scroll, reproduce it, and score it.

Chains the whole check for one scroll - find a region with actual surface in it,
run the official predictor over just that region, blend, and compare against the
published artifact - so a scroll can be verified unattended.

Everything it needs comes from catalog.json, so it works for any scroll with a
published surface prediction rather than a hand-wired one.

Usage:
    python run_verification.py --scroll PHerc0125 --model m7 [--size 256] [--out results/]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
PY_GPU = r"C:/Users/PC/miniconda3/envs/blogging/python.exe"
PY_CPU = r"C:/Users/PC/miniconda3/envs/vesuvius/python.exe"
MODEL = r"D:/Competition/Vesuvius progress prizes/model_m7"


def retry(fn, attempts: int = 8):
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def pick_region(pred_url: str, size: int) -> tuple[int, int, int, float]:
    """Find a region with enough surface in it that Dice is meaningful."""
    a = zarr.open(pred_url, mode="r")
    Z, Y, X = a.shape
    best = None
    for fz in (0.35, 0.5, 0.65):
        z, y, x = int(Z * fz), int(Y * 0.5), int(X * 0.5)
        if z + size > Z or y + size > Y or x + size > X:
            continue
        blk = retry(lambda: np.asarray(a[z:z + 128, y:y + 128, x:x + 128]))
        pos = float((blk > 0).mean())
        print(f"  probe z={z} y={y} x={x}: {100 * pos:.2f}% positive")
        if best is None or pos > best[3]:
            best = (z, y, x, pos)
    if best is None or best[3] < 0.02:
        raise SystemExit("no region with enough surface content found")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--model", default="m7")
    ap.add_argument("--level", type=int, default=None,
                    help="CT pyramid level; needed for scrolls predicted twice")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--trim", type=int, default=64)
    ap.add_argument("--out", default="results")
    ap.add_argument("--catalog", default="catalog.json")
    args = ap.parse_args()

    rows = json.load(open(args.catalog))
    hits = [r for r in rows if r.get("scroll") == args.scroll
            and r.get("prediction") and args.model in r.get("model", "")]
    if args.level is not None:
        hits = [r for r in hits if r["declared_level"] == args.level]
    if len(hits) != 1:
        avail = ", ".join(f"L{r['declared_level']}" for r in hits)
        raise SystemExit(
            f"expected exactly one match, got {len(hits)} ({avail}); pass --level")
    row = hits[0]
    level = row["declared_level"]
    pred_url = (f"{BUCKET}/{args.scroll}/representations/predictions/surfaces/"
                f"{row['prediction']}/0")
    ct_url = f"{BUCKET}/{args.scroll}/volumes/{row['ct_volume']}/{level}"

    print(f"=== {args.scroll}  model {row['model']}  CT level L{level} ===")
    z, y, x, pos = pick_region(pred_url, args.size)
    bbox = f"{z}:{z+args.size},{y}:{y+args.size},{x}:{x+args.size}"
    print(f"  region {bbox}  ({100*pos:.2f}% positive)")

    work = Path("outputs") / f"verify_{args.scroll}"
    if work.exists():
        import shutil
        shutil.rmtree(work)

    env_extra = {"nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
                 "PYTHONIOENCODING": "utf-8"}
    import os
    env = {**os.environ, **env_extra}

    print("  predicting ...")
    r = subprocess.run([PY_GPU, "run_gpu_roi.py", "--model_path", MODEL,
                        "--input_dir", ct_url, "--output_dir", str(work / "logits"),
                        "--device", "cuda", "--disable_tta", "--batch_size", "1",
                        "--num_workers", "2", "--read-retries", "12",
                        "--bbox", bbox], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("predict failed")

    # A run against a tree WITHOUT the global-grid fix (villa#1247) still succeeds -
    # it just anchors the sliding window to the ROI, giving 8 patches at stride 64
    # instead of ~100-125 at the volume's stride. The output looks plausible and
    # scores ~0.6-0.9, which reads as "this scroll does not reproduce" rather than
    # "you are on the wrong branch". That mistake nearly went out as a finding, so
    # it is now a hard failure rather than a silent one.
    n_patches = int(zarr.open(str(work / "logits" / "logits_part_0.zarr"),
                              mode="r").shape[0])
    expected = max(27, (args.size // 64) ** 3)
    if n_patches < expected:
        raise SystemExit(
            f"only {n_patches} patches for a {args.size}^3 region (expected >= "
            f"{expected}). The patch grid was anchored to the ROI, so this tree is "
            f"missing villa#1247. Check out the branch that carries it "
            f"(run-gpu-grid) and rerun; do NOT record this result.")
    print(f"  {n_patches} patches (global grid ok)")

    print("  blending ...")
    blend = (
        "import sys, torch\n"
        "_o=torch.compiler.disable\n"
        "def d(fn=None,*,recursive=True,reason=None): return _o(fn,recursive=recursive)\n"
        "torch.compiler.disable=d\n"
        "from vesuvius.models.run import blending\n"
        f"sys.argv=['b',r'{work / 'logits'}',r'{work / 'merged.zarr'}']\n"
        "blending.main()\n"
    )
    r = subprocess.run([PY_GPU, "-c", blend], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("blend failed")

    Path(args.out).mkdir(exist_ok=True)
    out_json = Path(args.out) / f"{args.scroll}_{row['model']}.json"
    print("  scoring ...")
    cmd = [PY_CPU, "verify_region.py", "--scroll", args.scroll,
           "--model", args.model, "--ours", str(work / "merged.zarr"),
           "--bbox", bbox, "--trim", str(args.trim), "--json", str(out_json)]
    if args.level is not None:
        cmd += ["--level", str(args.level)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
        raise SystemExit("verify failed")


if __name__ == "__main__":
    main()
