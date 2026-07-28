"""What fraction of labelled sheet does the published m7 model MISS?

TWO THINGS TO KNOW BEFORE READING ANY NUMBER OUT OF THIS.

  * The labels have three classes, and m7's own dataset.json names them
    {0: background, 1: surface, 2: ignore}. Class 2 is the MAJORITY of a typical volume
    (~59%) and must be excluded from scoring. An earlier version of this script folded it
    into background, which left recall correct but precision badly understated.
  * That dataset.json also says numTraining: 786, with shapes [320, 314, 314] - the same
    scale and label scheme as this 892-volume set. The close shape counts are strong
    evidence of substantial overlap, but the fingerprint does not expose public sample
    identifiers. Exact membership is therefore unknown, and this is NOT a held-out
    benchmark. The 106-count difference is not a proven held-out subset either - though
    hash_public_volumes.py names two volumes (sample_00853, sample_00854) whose crop
    shape appears nowhere in the fingerprint, so those two are provably outside it.

Everything measured in July asked whether the published predictions are right WHERE THEY
EXIST. This asks the inverse, which is the one that matters for reading: a sheet the model
misses is text nobody will ever recover.

bench_m7_derisk.py answered this on ONE region of ONE sample - 95.3% recall, best Dice
0.4391 over any threshold. (Its 16.7% precision figure is invalid for the same
ignore-class reason described above, and is not repeated.) That single number is currently quoted in
our own record as though it were settled. It is n=1, and this session produced three
separate cases where a small sample gave the wrong answer with an unpredictable sign. So
it gets run at scale before anything is built on it.

The comparison is NOT a Dice benchmark. The Kaggle label has an `ignore` region that must
be excluded, and m7 and the remaining scored labels need not encode identical objects.
Dice is reported only for continuity with the earlier probe. The primary question is
recall: OF THE VOXELS THE LABEL CALLS SHEET, WHAT FRACTION DOES M7 FIND?

Resumable: one JSON per sample in results/m7_recall/, existing ones are skipped.

Usage:
    python bench_m7_recall.py --n 40
    python bench_m7_recall.py --n 200 --size 256
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PY = os.environ.get("VESUVIUS_PYTHON", sys.executable)
MODEL = os.environ.get(
    "VESUVIUS_MODEL_PATH", "hf://scrollprize/surface_m7_nnunet")
PREDICT_SCRIPT = os.environ.get("VESUVIUS_PREDICT_SCRIPT")
THRESH = 0.2  # the published m7 threshold, from the artifact filenames (th0.2)


def run_one(ip: str, lp: str, work: Path, size: int, trim: int) -> dict | None:
    import tifffile
    import zarr

    img = tifffile.imread(ip)
    lab = tifffile.imread(lp)
    if img.shape != lab.shape or len(img.shape) != 3:
        return {"sample": os.path.basename(ip), "status": "shape_mismatch",
                "img": list(img.shape), "lab": list(lab.shape)}
    if min(img.shape) < size:
        size = (min(img.shape) // 64) * 64          # 256^3 samples exist too
        if size <= 2 * trim:
            return {"sample": os.path.basename(ip), "status": "too_small"}

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    zpath = work / "ct.zarr"
    z = zarr.open(str(zpath), mode="w", shape=img.shape, chunks=(128, 128, 128),
                  dtype="uint8", zarr_format=2)
    z[:] = img

    off = (img.shape[0] - size) // 2
    bbox = f"{off}:{off+size},{off}:{off+size},{off}:{off+size}"
    env = {**os.environ, "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
           "PYTHONIOENCODING": "utf-8"}

    predict_entry = ([PY, PREDICT_SCRIPT] if PREDICT_SCRIPT else
                     [PY, "-m", "vesuvius.models.run.inference"])
    r = subprocess.run(predict_entry + ["--model_path", MODEL,
                        "--input_dir", str(zpath), "--output_dir", str(work / "logits"),
                        "--device", "cuda", "--disable_tta", "--batch_size", "1",
                        "--num_workers", "2", "--bbox", bbox],
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return {"sample": os.path.basename(ip), "status": "predict_failed",
                "tail": (r.stderr or "")[-400:]}

    blend = (
        "import sys, torch\n"
        "_o=torch.compiler.disable\n"
        "def d(fn=None,*,recursive=True,reason=None): return _o(fn,recursive=recursive)\n"
        "torch.compiler.disable=d\n"
        "from vesuvius.models.run import blending\n"
        f"sys.argv=['b',r'{work / 'logits'}',r'{work / 'merged.zarr'}']\n"
        "blending.main()\n"
    )
    r = subprocess.run([PY, "-c", blend], env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return {"sample": os.path.basename(ip), "status": "blend_failed",
                "tail": (r.stderr or "")[-400:]}

    a = zarr.open(str(work / "merged.zarr"), mode="r")
    sl = slice(off + trim, off + size - trim)
    l0 = np.asarray(a[0, sl, sl, sl]).astype(np.float32)
    l1 = np.asarray(a[1, sl, sl, sl]).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(l1 - l0)))
    gt = lab[sl, sl, sl]

    # m7's own dataset.json declares labels {0: background, 1: surface, 2: ignore}, and
    # class 2 is the MAJORITY of a typical volume (~59%). It must be excluded from
    # scoring, not folded into background: counting predictions there as false positives
    # understates precision badly. Recall is unaffected either way, since it only ever
    # looks at class-1 voxels - which is why the recall figures predate this fix and
    # remain valid.
    sheet = gt == 1                       # the labelled sheet class
    ignore = gt == 2
    scored = ~ignore                      # background-or-sheet; everything else is unscored
    if sheet.sum() == 0:
        return {"sample": os.path.basename(ip), "status": "no_sheet_in_region"}
    pred = p > THRESH

    tp = float((pred & sheet).sum())
    fn = float((~pred & sheet).sum())
    fp = float((pred & scored & ~sheet).sum())   # false positives in SCORED background only
    recall = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    fp_ignore = float((pred & ignore).sum())     # predictions landing in unscored regions

    # Recall over thresholds: how much does the miss rate depend on where the cut is?
    rec_by_thr = {f"{t:.2f}": float((p > t)[sheet].mean()) for t in
                  (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)}

    # WHERE are the misses? "Depth inside the sheet" does not work here: the labelled
    # sheets are only ~2-3 voxels thick (mean EDT ~1.2), so a ">2 voxels deep" test can
    # barely fire regardless of how bad the model is. It reads as reassuring while
    # measuring nothing. Two metrics that do work on a thin structure:
    #
    #   orphan distance - for each missed sheet voxel, how far to the NEAREST predicted
    #                     voxel. 1 means the model drew the sheet a voxel off; 10 means it
    #                     did not find that sheet at all.
    #   lost components - labelled sheet components with almost none of their voxels found.
    #                     One of those is a sheet nobody will ever recover.
    from scipy.ndimage import distance_transform_edt, label as cc_label
    depth = distance_transform_edt(sheet)
    missed = ~pred & sheet

    dist_to_pred = distance_transform_edt(~pred) if pred.any() else np.full(pred.shape, 999.0)
    orphan = dist_to_pred[missed] if missed.any() else np.array([0.0])

    # LOCAL comparison: missed sheet voxels against FOUND sheet voxels, inside the same
    # volume. Volume-level averages explained almost nothing (all 9 properties jointly
    # gave R^2 = 0.099 over 826 volumes), which is consistent with the failures being
    # local rather than a property of whole volumes. Pairing within a volume makes each
    # volume its own control, so anything that shows up here is not a volume-level
    # confound.
    from scipy.ndimage import uniform_filter
    im = img[sl, sl, sl].astype(np.float32)
    loc_mean = uniform_filter(im, 5)
    loc_std = np.sqrt(np.clip(uniform_filter(im * im, 5) - loc_mean ** 2, 0, None))

    lab_cc, n_cc = cc_label(sheet)
    comp_size = np.zeros_like(lab_cc, dtype=np.float32)
    if n_cc:
        sizes = np.bincount(lab_cc.ravel())
        comp_size = sizes[lab_cc].astype(np.float32)

    local = {}
    if missed.any() and (sheet & pred).any():
        fnd = sheet & pred
        for name, field in (("depth", depth), ("ct", im), ("ct_local_std", loc_std),
                            ("component_size", comp_size)):
            mv, fv = field[missed], field[fnd]
            local[name] = {
                "missed_median": float(np.median(mv)),
                "found_median": float(np.median(fv)),
                # signed difference in units of the found-group spread, so it is
                # comparable across volumes with different scales
                "delta_z": float((np.median(mv) - np.median(fv)) /
                                 (fv.std() + 1e-6)),
            }
    lost = small = 0
    if n_cc:
        for cid in range(1, n_cc + 1):
            m = lab_cc == cid
            if m.sum() < 27:          # a handful of voxels is label noise, not a sheet
                small += 1
                continue
            if float(pred[m].mean()) < 0.10:
                lost += 1
    scored_cc = max(n_cc - small, 0)

    out = {
        "sample": os.path.basename(ip), "status": "ok",
        "region": bbox, "trim": trim,
        "label_sheet_fraction": float(sheet.mean()),
        "label_ignore_fraction": float(ignore.mean()),
        "scored_fraction": float(scored.mean()),
        "pred_in_ignore_frac": float(fp_ignore / max(pred.sum(), 1)),
        "pred_positive_fraction": float(pred.mean()),
        "recall": recall, "precision": precision,
        "miss_fraction": 1.0 - recall,
        "recall_by_threshold": rec_by_thr,
        "sheet_mean_depth": float(depth[sheet].mean()),
        # how isolated are the missed voxels from anything the model did predict
        "orphan_dist_median": float(np.median(orphan)),
        "orphan_dist_p90": float(np.quantile(orphan, 0.9)),
        "missed_far_frac": float((orphan > 3).mean()),
        # whole labelled sheets the model essentially did not find
        "sheet_components": int(scored_cc),
        "sheet_components_lost": int(lost),
        "frac_components_lost": (lost / scored_cc) if scored_cc else 0.0,
        "mean_confidence": float(np.abs(p - 0.5).mean() * 2),
        "local_missed_vs_found": local,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--trim", type=int, default=64)
    ap.add_argument("--out", default="results/m7_recall")
    args = ap.parse_args()

    imgs = sorted(glob.glob("data/kaggle/images/*.tif"))
    labs = sorted(glob.glob("data/kaggle/labels/*.tif"))
    pairs = [(i, l) for i, l in zip(imgs, labs)
             if os.path.basename(i) == os.path.basename(l)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path("outputs") / "m7_recall_work"

    todo = pairs[args.start:args.start + args.n]
    print(f"{len(todo)} sample(s); threshold {THRESH}; results -> {out}\n", flush=True)
    t0 = time.time()
    done = 0
    for ip, lp in todo:
        dest = out / (os.path.basename(ip).replace(".tif", ".json"))
        if dest.exists():
            continue
        try:
            rep = run_one(ip, lp, work, args.size, args.trim)
        except Exception as exc:  # noqa: BLE001
            rep = {"sample": os.path.basename(ip), "status": "error",
                   "error": f"{type(exc).__name__}: {exc}"}
        if rep is None:
            continue
        dest.write_text(json.dumps(rep, indent=2))
        done += 1
        if rep.get("status") == "ok":
            print(f"  {rep['sample']}  recall {100*rep['recall']:5.1f}%  "
                  f"MISSES {100*rep['miss_fraction']:5.1f}%  "
                  f"precision {100*rep['precision']:4.1f}%  "
                  f"lost {rep['sheet_components_lost']}/{rep['sheet_components']}  "
                  f"far-miss {100*rep['missed_far_frac']:4.1f}%", flush=True)
        else:
            print(f"  {rep['sample']}  {rep['status']}", flush=True)
    el = time.time() - t0
    print(f"\n{done} run in {el/60:.1f} min "
          f"({el/max(done,1):.0f}s each)")


if __name__ == "__main__":
    main()
