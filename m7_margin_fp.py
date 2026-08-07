"""Do m7's scored false positives concentrate in the one-voxel asserted margin?

This asks bet 2's question of the PUBLISHED model directly, with no proxy network anywhere
in the path. `margin_arms.py` trains a small UNet from scratch, so a positive result there
says "this label change helps *a* surface model"; it does not say it helps m7. This does,
and it needs no training at all.

The margin is free: `margin_relabel.py` already wrote all 892 volumes, and

    margin = (labels_margin == 2) & (labels == 0)

recovers exactly the voxels it moved -- class 0, within one voxel of a labelled sheet run
along the across-sheet normal.

⚠ THE RAW SHARE IS NOT THE STATISTIC. The margin is a thin shell, a per-cent-scale slice of
the scored non-sheet region, so "3% of false positives are in the margin" could be either
damning or nothing at all depending on how big the shell is. What matters is ENRICHMENT:

    enrichment = (FP in margin / FP total) / (margin voxels / scored non-sheet voxels)

Enrichment of 1 means m7's errors ignore the margin and bet 2's premise does not describe
this model. Enrichment well above 1 means a disproportionate share of what the benchmark
counts against m7 is landing on the half voxel where the label asserts background and the CT
profile says sheet -- i.e. the benchmark is penalising the model for being right.

Scoring matches bench_m7_recall.py exactly: sigmoid(l1 - l0) > 0.2, class 2 excluded.

  python m7_margin_fp.py --check          # analysis math only, no GPU, no m7
  python m7_margin_fp.py --n 60           # the real thing
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

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"
MARGIN = ROOT / "data" / "kaggle" / "labels_margin"
SPLIT = ROOT / "vesuvius-repro" / "results" / "margin_split.json"
PRED_CACHE = ROOT / "results" / "m7_pred_cache"

# ⚠ NOT the blogging env. That env is torch 2.4.1 and can no longer import villa at all --
# `torch.compiler.disable(reason=...)` raises there. bench_m7_recall.py still points at it
# and would fail today; this points at the only env that can run m7.
PY = r"C:/Users/PC/miniconda3/envs/vesuvius/python.exe"
MODEL = str(ROOT / "model_m7")
THRESH = 0.2
TRIM = 64

# Two upstream incompatibilities stand between this env and a working m7 run. Shimmed here
# rather than by editing the villa tree, so that tree stays clean for a PR.
#
#   1. `zarr.Blosc` is a zarr-2 API, REMOVED IN ZARR 3, and villa's inference.py calls it in
#      four places (_get_zarr_compressor). villa's pyproject declares `zarr>=2.18.7,<4`, so
#      zarr 3 is supported on paper and broken in fact -- vesuvius.predict cannot write its
#      output store at all. On zarr 2.18.3, `zarr.Blosc is numcodecs.Blosc` is True, so
#      pointing at numcodecs is a no-op there and a fix here.
#   2. `torch.compiler.disable(reason=...)` -- the blogging env's torch 2.4.1 rejects the
#      keyword, which is why that env can no longer import villa. Kept for symmetry.
#   3. Having got past (1), zarr 3 then rejects the store outright: villa's open_zarr passes
#      a v2-style `compressor=` to zarr.open without ever naming a format, so zarr 3 defaults
#      to v3 and raises `compressor cannot be used for arrays with zarr_format 3`. Pinning
#      zarr_format=2 whenever a compressor is supplied reproduces exactly what zarr 2.18.3
#      did, which is the combination the 855-volume benchmark was produced under.
SHIM = (
    "import sys, runpy, zarr, numcodecs, torch\n"
    "if not hasattr(zarr, 'Blosc'): zarr.Blosc = numcodecs.Blosc\n"
    "_zopen = zarr.open\n"
    "def _zopen2(*a, **k):\n"
    "    if k.get('compressor') is not None and 'zarr_format' not in k:\n"
    "        k['zarr_format'] = 2\n"
    "    return _zopen(*a, **k)\n"
    "zarr.open = _zopen2\n"
    "_o = torch.compiler.disable\n"
    "def _d(fn=None, *, recursive=True, reason=None): return _o(fn, recursive=recursive)\n"
    "torch.compiler.disable = _d\n"
)


def margin_mask(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(labels, margin) for one volume. Margin is recovered, never recomputed."""
    lab = np.asarray(tifffile.imread(str(LABELS / f"{name}.tif")))
    lm = np.asarray(tifffile.imread(str(MARGIN / f"{name}.tif")))
    return lab, (lm == 2) & (lab == 0)


def distance_profile(lab: np.ndarray, pred: np.ndarray, max_d: int = 5) -> dict:
    """⚠ THE CONTROL THAT DECIDES WHETHER THE MARGIN RESULT MEANS ANYTHING.

    Enrichment above 1 in the margin is not by itself evidence for the label-boundary story.
    m7 over-predicts around sheets, so ANY near-sheet region will be enriched in false
    positives whether or not the label boundary is misplaced. Proximity alone would produce
    the headline number.

    So report enrichment as a function of distance from the labelled sheet, in one-voxel
    shells. The two stories make different predictions and the shapes are not subtle:

      label boundary misplaced by ~half a voxel  ->  a STEP: shell 1 far above shell 2,
                                                     then flat, because the true sheet ends
      generic near-sheet over-prediction         ->  SMOOTH DECAY across shells 1,2,3,4

    Euclidean, not restricted to the across-sheet normal: this is deliberately the weaker,
    more conservative geometry. If shell 1 does not stand out even here, the normal-restricted
    version standing out would be an artifact of the normal estimate rather than of the CT.
    """
    from scipy.ndimage import distance_transform_edt

    sheet, ignore = lab == 1, lab == 2
    scored = ~ignore
    d = distance_transform_edt(~sheet)
    fp = pred & scored & ~sheet
    n_fp, n_ns = float(fp.sum()), float((scored & ~sheet).sum())
    if n_fp == 0 or n_ns == 0:
        return {}
    out = {}
    for k in range(1, max_d + 1):
        shell = (d > k - 1) & (d <= k) & scored & ~sheet
        n_shell = float(shell.sum())
        if n_shell < 100:
            out[f"shell_{k}"] = None
            continue
        out[f"shell_{k}"] = round((float((fp & shell).sum()) / n_fp) / (n_shell / n_ns), 3)
    return out


def analyse(lab: np.ndarray, margin: np.ndarray, pred: np.ndarray) -> dict:
    """Enrichment of m7's scored false positives inside the asserted margin."""
    sheet, ignore = lab == 1, lab == 2
    scored = ~ignore
    fp = pred & scored & ~sheet
    non_sheet = scored & ~sheet

    n_fp = float(fp.sum())
    n_ns = float(non_sheet.sum())
    n_margin = float((margin & scored).sum())
    if n_fp == 0 or n_ns == 0 or n_margin == 0:
        return {"status": "degenerate", "n_fp": n_fp, "n_margin": n_margin}

    share_fp = float((fp & margin).sum()) / n_fp
    share_vol = n_margin / n_ns
    return {
        "status": "ok",
        "fp_share_in_margin": round(share_fp, 6),
        "margin_share_of_nonsheet": round(share_vol, 6),
        "enrichment": round(share_fp / share_vol, 3),
        "margin_hit_rate": round(float((pred & margin & scored).sum()) / n_margin, 5),
        "nonmargin_fp_rate": round((n_fp - float((fp & margin).sum()))
                                   / max(n_ns - n_margin, 1.0), 5),
        "n_fp": int(n_fp), "n_margin_scored": int(n_margin),
    }


def predict_m7(name: str, work: Path, size: int = 256) -> np.ndarray | None:
    """m7 logits over a centred cube, via the same two-step run+blend the benchmark uses."""
    import zarr

    img = np.asarray(tifffile.imread(str(IMAGES / f"{name}.tif")))
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

    args = ["--model_path", MODEL, "--input_dir", str(zpath),
            "--output_dir", str(work / "logits"), "--device", "cuda", "--disable_tta",
            "--batch_size", "1", "--num_workers", "2", "--bbox", bbox]
    r = subprocess.run([PY, "-c", SHIM + "runpy.run_path(r'run_gpu_roi.py', "
                        "run_name='__main__')", *args],
                       cwd=ROOT, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # ⚠ DO NOT TRUST THE RETURN CODE. villa's inference and blending both catch their own
    # exceptions, print "--- ... Failed ---" and exit 0. The first version of this script
    # checked returncode, sailed past a dead inference step, and died later on a missing
    # merged.zarr with a traceback that pointed at the wrong stage entirely. Check that the
    # artefact exists instead.
    logits = work / "logits"
    if not logits.exists() or not any(logits.iterdir()):
        print(f"    predict wrote nothing (rc={r.returncode}): "
              f"{(r.stdout or '')[-300:]}", flush=True)
        return None

    blend = (SHIM + "from vesuvius.models.run import blending\n"
             f"sys.argv=['b',r'{work / 'logits'}',r'{work / 'merged.zarr'}']\n"
             "blending.main()\n")
    r = subprocess.run([PY, "-c", blend], cwd=ROOT, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if not (work / "merged.zarr").exists():
        print(f"    blend wrote nothing (rc={r.returncode}): "
              f"{(r.stdout or '')[-300:]}", flush=True)
        return None

    a = zarr.open(str(work / "merged.zarr"), mode="r")
    sl = slice(off + TRIM, off + size - TRIM)
    l0 = np.asarray(a[0, sl, sl, sl]).astype(np.float32)
    l1 = np.asarray(a[1, sl, sl, sl]).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-(l1 - l0))), (off + TRIM, off + size - TRIM)


def check() -> None:
    """Validate the analysis on real labels with STAND-IN predictions. No GPU, no m7.

    Two synthetic predictors bracket the answer, so a broken enrichment cannot look sane:
    a uniform random predictor must land at enrichment ~1 by construction, and one that
    deliberately fires on the margin must land far above it. If the first is not ~1 the
    statistic is wrong, whatever the real run later reports.
    """
    names = json.loads(SPLIT.read_text())["test"][:6]
    rng = np.random.default_rng(0)
    print(f"{'sample':<16}{'random':>10}{'margin-seeking':>16}   margin share of non-sheet")
    for nm in names:
        lab, margin = margin_mask(nm)
        sl = (slice(TRIM, lab.shape[0] - TRIM),) * 3
        lab_c, margin_c = lab[sl], margin[sl]

        rate = 0.20                                   # m7's measured predicted-positive rate
        null = analyse(lab_c, margin_c, rng.random(lab_c.shape) < rate)
        seek = analyse(lab_c, margin_c,
                       margin_c | (rng.random(lab_c.shape) < rate * 0.5))
        print(f"{nm:<16}{null['enrichment']:>10.3f}{seek['enrichment']:>16.3f}"
              f"   {null['margin_share_of_nonsheet']:.5f}")
    print("\n  A uniform random predictor MUST sit at enrichment ~1.000 -- that is the null\n"
          "  the real number is read against. The margin-seeking column shows the statistic\n"
          "  has range. Neither says anything about m7; they say the maths is not broken.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="analysis math only, no GPU")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=str(ROOT / "results" / "m7_margin_fp.json"))
    a = ap.parse_args()

    if a.check:
        check()
        return

    names = json.loads(SPLIT.read_text())["test"][:a.n]
    work = ROOT / "_m7_margin_work"
    rows, t0 = [], time.time()
    for k, nm in enumerate(names):
        got = predict_m7(nm, work)
        if got is None:
            rows.append({"sample": nm, "status": "predict_failed"})
            continue
        p, (lo, hi) = got
        lab, margin = margin_mask(nm)
        sl = (slice(lo, hi),) * 3
        pred = p > THRESH
        r = analyse(lab[sl], margin[sl], pred)
        r["shells"] = distance_profile(lab[sl], pred)
        r["sample"] = nm
        rows.append(r)
        # Cache the probabilities: ~14 MB per volume as float16, and it means any later
        # question about thresholds or geometry costs no GPU at all. Not saving these is
        # what made the arms' first configuration cost a full retrain.
        PRED_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(PRED_CACHE / f"{nm}.npy", p.astype(np.float16))
        if r["status"] == "ok":
            print(f"  [{k+1}/{len(names)}] {nm}  enrichment {r['enrichment']:>7.2f}  "
                  f"fp in margin {r['fp_share_in_margin']:.4f}  {time.time()-t0:.0f}s",
                  flush=True)
    ok = [r for r in rows if r.get("status") == "ok"]
    ens = [r["enrichment"] for r in ok]
    out = {
        "n_volumes": len(ok), "threshold": THRESH, "trim": TRIM,
        "median_enrichment": round(float(np.median(ens)), 3) if ens else None,
        "q10_enrichment": round(float(np.quantile(ens, 0.10)), 3) if ens else None,
        "q90_enrichment": round(float(np.quantile(ens, 0.90)), 3) if ens else None,
        "frac_volumes_enriched": round(float(np.mean([e > 1.0 for e in ens])), 4) if ens else None,
        "median_fp_share_in_margin": round(float(np.median(
            [r["fp_share_in_margin"] for r in ok])), 5) if ok else None,
        "reading": ("enrichment 1.0 = m7's false positives ignore the margin and bet 2's "
                    "premise does not describe this model; >> 1 = the benchmark is counting "
                    "the asserted margin against m7"),
        "shell_enrichment_median": {
            f"shell_{k}": (round(float(np.median(s)), 3) if (s := [
                r["shells"][f"shell_{k}"] for r in ok
                if r.get("shells", {}).get(f"shell_{k}") is not None]) else None)
            for k in range(1, 6)},
        "shell_reading": ("THE CONTROL. A step at shell 1 with shells 2+ far lower supports a "
                          "misplaced label boundary. Smooth decay across shells 1-4 means m7 "
                          "simply over-predicts near sheet and the margin is not special, "
                          "which would sink the headline number."),
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n  median enrichment {out['median_enrichment']}  "
          f"({out['frac_volumes_enriched']:.0%} of volumes > 1)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
