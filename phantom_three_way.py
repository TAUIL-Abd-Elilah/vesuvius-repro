"""Three-way rerun for villa#1173: does the blend or the input path produce phantom sheet?

Schurkai's hypothesis, in their words: most of our 99.69% is the blend rather than the
input path, because an all-zero patch is already dropped by `is_empty` (min == max), so
what is left in unscanned blocks is logits leaking outward from the patch edge. They asked
for three arms, which is what this runs:

  A  main                          e224dfc
  B  main + blend fix              Schurkai:fix-1114-mask-empty-input (normalize_blended_logits)
  C  B + --mask_empty_input        the opt-in input-path gate

The region is chosen by find_phantom_region.py so that the CT is IDENTICALLY ZERO across
it while the published m7 prediction asserts sheet. That makes the metric unambiguous:
over a region with no scan data, every predicted sheet voxel is phantom, so no threshold
argument about what counts as "sheet over empty CT" is needed. Reported across a sweep of
thresholds anyway, because a single one invites the reply that it was chosen.

Both worktrees are checked out from the same clone and run via PYTHONPATH, so the only
difference between arms is the villa source and the flag.

    python phantom_three_way.py --scroll PHerc0175A --bbox 6569:6953,3212:3596,6040:6424
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PY = "C:/Users/PC/miniconda3/envs/vesuvius/python.exe"
MODEL = str(ROOT / "model_m7")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"

ARMS = {
    "A_main": {"worktree": "phantom-main", "extra": []},
    "B_blendfix": {"worktree": "phantom-schurkai", "extra": []},
    "C_blendfix_maskinput": {"worktree": "phantom-schurkai", "extra": ["--mask_empty_input"]},
}
# Probability cut-points. 0.2 is what the published m7 predictions use (catalog.json
# `threshold`), the rest bracket it so the answer cannot be read as a chosen cut.
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def retry(fn, attempts: int = 6, timeout_note: str = ""):
    """Retry a flaky remote read. This link drops mid-transfer (STRATEGY 00.1 #8).

    measure() originally read the CT block without this and hung for 20 minutes on a
    stalled HTTP read, with the GPU idle and nothing in any log to say why.
    """
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"      read failed ({type(e).__name__}), retry {i+1}/{attempts-1} {timeout_note}",
                  flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 20.0)


def ct_block_cached(ct, bbox, cache: Path):
    """Fetch the CT block once and keep it. Every arm scores against the same voxels,
    so re-downloading it per arm is three times the traffic and three times the chance
    of a stall."""
    if cache.exists():
        return np.load(cache)
    z0, z1, y0, y1, x0, x1 = bbox
    print("    fetching CT block (cached after this)...", flush=True)
    block = retry(lambda: np.asarray(ct[z0:z1, y0:y1, x0:x1]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, block)
    return block


def parse_bbox(text: str):
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError("--bbox wants z0:z1,y0:y1,x0:x1")
    out = []
    for p in parts:
        lo, hi = p.split(":")
        out += [int(lo), int(hi)]
    return out


def run(cmd, env, log_path, label):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.call(cmd, env=env, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"    {label}: rc={rc} in {dt:.0f}s -> {log_path.name}", flush=True)
    if rc != 0:
        # A failed arm must not be silently scored as "no phantom" -- that would read as
        # the fix working. Same failure mode as the resume demo's swallowed exception.
        raise RuntimeError(f"{label} failed (rc={rc}); see {log_path}")
    return dt


def measure(final_path: Path, ct_block, bbox, arm: str):
    z0, z1, y0, y1, x0, x1 = bbox
    a = zarr.open(str(final_path), mode="r")
    # blend_logits writes the GLOBAL frame, channel-first: (C, Z, Y, X). Slicing the first
    # three axes silently slices the channel axis instead of z and returns an empty block.
    block = np.asarray(a[:, z0:z1, y0:y1, x0:x1]).astype(np.float32)
    if block.shape[0] != 2:
        raise RuntimeError(f"expected 2-channel binary logits, got {block.shape}")
    # Two-channel softmax reduces to a sigmoid of the difference, which avoids overflow.
    prob = 1.0 / (1.0 + np.exp(-(block[1] - block[0])))

    empty = ct_block == 0
    n_empty = int(empty.sum())
    row = {
        "arm": arm,
        "voxels": int(prob.size),
        "frac_ct_empty": float(empty.mean()),
        "prob_max": float(prob.max()),
        "prob_max_on_empty": float(prob[empty].max()) if n_empty else 0.0,
        "prob_mean_on_empty": float(prob[empty].mean()) if n_empty else 0.0,
        "logit_absmax_on_empty": float(np.abs(block[1] - block[0])[empty].max()) if n_empty else 0.0,
        "phantom_by_threshold": {},
    }
    for t in THRESHOLDS:
        sheet = prob > t
        n_sheet = int(sheet.sum())
        n_phantom = int((sheet & empty).sum())
        # Sheet sitting on ACTUAL scan data. Reporting only the phantom count would score
        # "predicts nothing anywhere" as a perfect fix; this is the half that must survive.
        n_real = n_sheet - n_phantom
        row["phantom_by_threshold"][f"{t:g}"] = {
            "sheet_voxels": n_sheet,
            "phantom_voxels": n_phantom,
            "real_sheet_voxels": n_real,
            "frac_of_region": n_phantom / prob.size,
            "frac_of_sheet": (n_phantom / n_sheet) if n_sheet else 0.0,
            "frac_of_scanned": (n_real / (prob.size - n_empty)) if n_empty < prob.size else 0.0,
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--bbox", required=True, help="z0:z1,y0:y1,x0:x1 global voxel coords")
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--work", default="outputs/phantom_three_way")
    ap.add_argument("--out", default="results/phantom_three_way.json")
    ap.add_argument("--keep", action="store_true", help="keep existing arm outputs")
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset, e.g. B_blendfix,C_blendfix_maskinput")
    args = ap.parse_args()

    bbox = parse_bbox(args.bbox)
    catalog = json.load(open(args.catalog))
    rows = [r for r in catalog if r.get("scroll") == args.scroll
            and r.get("status") == "verifiable" and "m7" in r.get("model", "")
            and r.get("declared_level", 2) == 0]
    if not rows:
        raise SystemExit(f"{args.scroll}: no level-0 verifiable m7 prediction in catalog")
    ct_volume = rows[0]["ct_volume"]
    input_url = f"{BUCKET}/{args.scroll}/volumes/{ct_volume}/0"
    ct = zarr.open(input_url, mode="r")

    work = ROOT / args.work
    if work.exists() and not args.keep:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    z0, z1, y0, y1, x0, x1 = bbox
    print(f"{args.scroll}  bbox {bbox}  ({z1-z0}x{y1-y0}x{x1-x0})", flush=True)
    print(f"input {input_url}\n", flush=True)

    ct_block = ct_block_cached(ct, bbox, work / "ct_block.npy")

    results = []
    wanted = set(args.arms.split(",")) if args.arms else set(ARMS)
    for arm, spec in ARMS.items():
        if arm not in wanted:
            print(f"  {arm}: skipped (--arms)", flush=True)
            continue
        src = ROOT / "_worktrees" / spec["worktree"] / "vesuvius" / "src"
        arm_dir = work / arm
        logits_dir = arm_dir / "logits"
        final_path = arm_dir / "final.zarr"
        if final_path.exists() and args.keep:
            print(f"  {arm}: reusing existing output")
        else:
            arm_dir.mkdir(parents=True, exist_ok=True)
            env = {**os.environ,
                   "PYTHONPATH": str(src),
                   "nnUNet_compile": "0",
                   "TORCHDYNAMO_DISABLE": "1",
                   "PYTHONIOENCODING": "utf-8",
                   # Windows spills VRAM into shared RAM instead of OOMing (STRATEGY 00.1 #2)
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            print(f"  {arm}  ({spec['worktree']}{' ' + ' '.join(spec['extra']) if spec['extra'] else ''})",
                  flush=True)
            run([PY, "-m", "vesuvius.models.run.inference",
                 "--model_path", MODEL, "--input_dir", input_url,
                 "--output_dir", str(logits_dir), "--device", "cuda",
                 "--disable_tta", "--batch_size", "1", "--num_workers", "2",
                 "--bbox", args.bbox] + spec["extra"],
                env, arm_dir / "predict.log", "predict")
            run([PY, "-m", "vesuvius.models.run.blending",
                 str(logits_dir), str(final_path)],
                env, arm_dir / "blend.log", "blend+finalize")

        row = measure(final_path, ct_block, bbox, arm)
        top = row["phantom_by_threshold"]["0.2"]
        print(f"    phantom@0.2: {top['phantom_voxels']:,} vox "
              f"({100*top['frac_of_region']:.2f}% of region, "
              f"{100*top['frac_of_sheet']:.1f}% of sheet), "
              f"max prob on empty CT {row['prob_max_on_empty']:.4f}\n", flush=True)
        results.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "note": ("Three-way rerun for villa#1173 over a region whose CT is identically zero. "
                 "Every predicted sheet voxel there is phantom by construction. Arm A is "
                 "villa main, B adds Schurkai's blend normalization fix, C adds "
                 "--mask_empty_input on top."),
        "scroll": args.scroll, "bbox": bbox, "input": input_url,
        "arms": {k: {"worktree": v["worktree"], "extra": v["extra"]} for k, v in ARMS.items()},
        "results": results,
    }, indent=2))
    print(f"wrote {args.out}")

    base = results[0]["phantom_by_threshold"]["0.2"]
    print("\n  at the published threshold 0.2:")
    print(f"  {'arm':<24} {'phantom vox':>12} {'vs main':>9} {'real sheet vox':>15} {'vs main':>9}")
    for r in results:
        cell = r["phantom_by_threshold"]["0.2"]
        n, k = cell["phantom_voxels"], cell["real_sheet_voxels"]
        rp = "-" if r is results[0] else (f"{100*n/base['phantom_voxels']:.1f}%"
                                          if base["phantom_voxels"] else "n/a")
        rr = "-" if r is results[0] else (f"{100*k/base['real_sheet_voxels']:.1f}%"
                                          if base["real_sheet_voxels"] else "n/a")
        print(f"  {r['arm']:<24} {n:>12,} {rp:>9} {k:>15,} {rr:>9}")
    print("\n  A fix that removes phantom by removing everything would show both columns fall.")


if __name__ == "__main__":
    main()
