"""Regenerate the m7 prediction cache with the normalization the model was actually trained on.

Every m7 number we published came from `vesuvius.predict`'s default `--normalization
instance_zscore`, while m7's `plans.json` declares `CTNormalization` with mean 87.544,
std 47.744. The nnU-Net loading path never reads the plans, so the default silently won.
Found by @Jinhojeong (villa#1364, raised on villa#193); confirmed here in the source and
reproduced on 4 volumes, where recall went 0.803 -> 0.940 and precision 0.467 -> 0.742.

There is no flag for this. The CLI offers instance_zscore / global_zscore / instance_minmax /
none, and the `ct` scheme raises because `inference.py` never passes it the intensity
properties. So the workaround, which is @Jinhojeong's third arm, is to apply CT normalization
to the input ourselves and hand the wrapper `--normalization none`.

CT normalization, per nnU-Net: clip to [percentile_00_5, percentile_99_5], then subtract the
foreground mean and divide by the foreground std -- all four constants read from this model's
own plans.json rather than hardcoded, so a different checkpoint cannot silently inherit m7's.

  python m7_renorm.py --n 60
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile

import m7_margin_fp as M

ROOT = Path(__file__).resolve().parent
OUT_CACHE = ROOT / "results" / "m7_pred_cache_ctnorm"
# The split lives beside this file in the published repo and one level down in the working
# tree. Try both, so the published copy is runnable rather than only ours.
SPLIT = next((p for p in (ROOT / "vesuvius-repro" / "results" / "margin_split.json",
                          ROOT / "results" / "margin_split.json") if p.exists()),
             ROOT / "results" / "margin_split.json")
SIZE, TRIM = M.SIZE if hasattr(M, "SIZE") else 256, M.TRIM


def plans_constants(model_dir: Path) -> tuple[float, float, float, float]:
    p = json.loads((model_dir / "plans.json").read_text())
    cfg = p.get("configurations", {}).get("3d_fullres", {})
    schemes = cfg.get("normalization_schemes")
    if schemes != ["CTNormalization"]:
        raise SystemExit(f"expected CTNormalization, plans says {schemes}")
    fp = p["foreground_intensity_properties_per_channel"]["0"]
    return (float(fp["mean"]), float(fp["std"]),
            float(fp["percentile_00_5"]), float(fp["percentile_99_5"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--all", action="store_true", help="every labelled volume, not just the test split")
    a = ap.parse_args()

    model = Path(M.MODEL)
    mean, std, lo, hi = plans_constants(model)
    print(f"plans CTNormalization: mean {mean:.4f} std {std:.4f} clip [{lo}, {hi}]", flush=True)

    import zarr, numcodecs
    if not hasattr(zarr, "Blosc"):
        zarr.Blosc = numcodecs.Blosc

    if a.all:
        # every volume that has both an image and a label, so the corrected cache can replace
        # results/m7_recall/ wholesale rather than only its test slice
        names = sorted(p.stem for p in (M.LABELS).glob("sample_*.tif")
                       if (M.IMAGES / f"{p.stem}.tif").exists())
    else:
        names = json.loads(SPLIT.read_text())["test"][:a.n]
    print(f"volumes to do: {len(names)}", flush=True)
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    work = ROOT / "_renorm_work"
    env = {**os.environ, "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
           "PYTHONIOENCODING": "utf-8"}
    t0 = time.time()
    done = 0
    for k, nm in enumerate(names):
        dest = OUT_CACHE / f"{nm}.npy"
        if dest.exists():
            done += 1
            continue
        ct = np.asarray(tifffile.imread(str(M.IMAGES / f"{nm}.tif")))
        if work.exists():
            shutil.rmtree(work)
        work.mkdir()
        ctn = (np.clip(ct.astype(np.float32), lo, hi) - mean) / std
        z = zarr.open(str(work / "ct.zarr"), mode="w", shape=ct.shape, chunks=(128,) * 3,
                      dtype="float32", zarr_format=2)
        z[:] = ctn

        off = (ct.shape[0] - a.size) // 2
        bbox = f"{off}:{off+a.size},{off}:{off+a.size},{off}:{off+a.size}"
        r = subprocess.run(
            [M.PY, "-c", M.SHIM + "runpy.run_path(r'run_gpu_roi.py', run_name='__main__')",
             "--model_path", str(model), "--input_dir", str(work / "ct.zarr"),
             "--output_dir", str(work / "logits"), "--device", "cuda", "--disable_tta",
             "--batch_size", "1", "--num_workers", "2", "--bbox", bbox,
             "--normalization", "none"],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        lg = work / "logits"
        if not lg.exists() or not any(lg.iterdir()):
            print(f"  {nm}: predict wrote nothing (rc={r.returncode})", flush=True)
            continue
        blend = (M.SHIM + "from vesuvius.models.run import blending\n"
                 f"sys.argv=['b',r'{lg}',r'{work / 'merged.zarr'}']\nblending.main()\n")
        subprocess.run([M.PY, "-c", blend], env=env, capture_output=True, text=True)
        if not (work / "merged.zarr").exists():
            print(f"  {nm}: blend wrote nothing", flush=True)
            continue
        arr = zarr.open(str(work / "merged.zarr"), mode="r")
        sl = slice(off + TRIM, off + a.size - TRIM)
        l0 = np.asarray(arr[0, sl, sl, sl]).astype(np.float32)
        l1 = np.asarray(arr[1, sl, sl, sl]).astype(np.float32)
        np.save(dest, (1.0 / (1.0 + np.exp(-(l1 - l0)))).astype(np.float16))
        done += 1
        if done % 5 == 0:
            print(f"  [{done}/{len(names)}] {time.time()-t0:.0f}s", flush=True)
    if work.exists():
        shutil.rmtree(work)
    print(f"cached {done} volumes -> {OUT_CACHE}")
    print(f"now: python surface_bench.py --pred {OUT_CACHE} --out results/surface_bench_m7_ctnorm.json")


if __name__ == "__main__":
    main()
