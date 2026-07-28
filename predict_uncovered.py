"""Produce a surface prediction for a scroll the project has not published one for.

Three scrolls in the open-data bucket have public CT at the m7 model's working
resolution but no published surface prediction: PHerc0172, PHerc1667 and PHercParis3.
The model is public, so those predictions are producible.

This differs from run_verification.py in what it can and cannot claim. There is no
published artifact to score against, so there is no Dice here and no "reproduced"
verdict. What it does instead is report the model's *confidence fingerprint*, which
is the diagnostic that caught the PHerc0846A anomaly:

  * the logit range. A healthy region spans roughly [-4, 18]; the degenerate region
    on PHerc0846A spanned [-1.9, 6.5] and pushed 97% of voxels over the threshold.
  * the positive fraction. Published predictions run ~13-77%, typically ~20-25%.
    A number far outside that says the model is not seeing papyrus.

Together those say whether the model is operating in its normal regime on this scroll
or producing mush. That is an honest claim; "this prediction is correct" is not one we
can make without ground truth.

Region selection also cannot lean on a prediction, so it probes the CT directly and
takes the densest candidate - air scores near zero and tells us nothing.

Usage:
    python predict_uncovered.py --scroll PHerc1667 [--size 256]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
HERE = Path(__file__).resolve().parent
PY_GPU = os.environ.get("VESUVIUS_PYTHON", sys.executable)
MODEL = os.environ.get(
    "VESUVIUS_MODEL_PATH", "hf://scrollprize/surface_m7_nnunet")
PREDICT_SCRIPT = os.environ.get("VESUVIUS_PREDICT_SCRIPT")

# Calibrated on regions whose status is already known, not picked by eye - see
# calibrate_regime.py and results/regime_calibration.json.
#
# Logit span turned out NOT to discriminate: PHerc0500P2 reproduced *exactly*
# (Dice 1.0000) with a span of 6.2, below an earlier hand-picked cutoff of 12.0
# that would have branded it degenerate. Mean confidence does discriminate, with a
# 7x gap and no overlap:
#     reproduced regions   0.231 .. 0.944   (n=9)
#     degenerate region    0.032            (PHerc0846A r1, 96.9% positive)
MIN_MEAN_CONFIDENCE = 0.12      # between the two populations, nearer the bad one
PLAUSIBLE_POSITIVE = (0.02, 0.90)   # published regions observed at 19% .. 77%
OBSERVED_POSITIVE = (0.190, 0.768)  # for reporting when outside, not for failing


def retry(fn, attempts: int = 8):
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"    transient read, retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def pick_region(ct_url: str, size: int) -> tuple[int, int, int, float]:
    """Densest of a few candidates. Without a prediction to probe, CT density is
    the only signal for 'there is scroll here rather than air'."""
    a = zarr.open(ct_url, mode="r")
    Z, Y, X = a.shape
    best = None
    for fz in (0.35, 0.5, 0.65):
        z, y, x = int(Z * fz), int(Y * 0.5) - size // 2, int(X * 0.5) - size // 2
        z, y, x = max(0, z), max(0, y), max(0, x)
        if z + size > Z or y + size > Y or x + size > X:
            continue
        blk = retry(lambda: np.asarray(a[z:z + 128, y:y + 128, x:x + 128]))
        score = float((blk > 40).mean())
        print(f"  probe z={z} y={y} x={x}: {100 * score:.1f}% above air, "
              f"mean {blk.mean():.1f}")
        if best is None or score > best[3]:
            best = (z, y, x, score)
    if best is None:
        raise SystemExit("no in-bounds candidate region")
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--survey", default="uncovered.json")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--trim", type=int, default=64)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    survey = {r["scroll"]: r for r in json.load(open(args.survey))}
    if args.scroll not in survey or not survey[args.scroll].get("best"):
        raise SystemExit(f"{args.scroll} has no volume at the model's working resolution")
    best = survey[args.scroll]["best"]
    volume, level = best["volume"], best["level"]
    ct_url = f"{BUCKET}/{args.scroll}/volumes/{volume}/{level}"

    print(f"=== {args.scroll}  {volume}  L{level}  {best['um']}um  {best['shape']} ===")
    z, y, x, dens = pick_region(ct_url, args.size)
    bbox = f"{z}:{z+args.size},{y}:{y+args.size},{x}:{x+args.size}"
    print(f"  region {bbox}  ({100*dens:.1f}% above air)")

    work = Path("outputs") / f"new_{args.scroll}"
    if work.exists():
        shutil.rmtree(work)
    env = {**os.environ, "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
           "PYTHONIOENCODING": "utf-8"}

    print("  predicting ...")
    predict_entry = ([PY_GPU, PREDICT_SCRIPT] if PREDICT_SCRIPT else
                     [PY_GPU, "-m", "vesuvius.models.run.inference"])
    r = subprocess.run(predict_entry + ["--model_path", MODEL,
                        "--input_dir", ct_url, "--output_dir", str(work / "logits"),
                        "--device", "cuda", "--disable_tta", "--batch_size", "1",
                        "--num_workers", "2", "--read-retries", "12",
                        "--bbox", bbox], env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("predict failed")

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
    r = subprocess.run([PY_GPU, "-c", blend], env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("blend failed")

    m = args.trim
    arr = zarr.open(str(work / "merged.zarr"), mode="r")
    iz = slice(z + m, z + args.size - m)
    iy = slice(y + m, y + args.size - m)
    ix = slice(x + m, x + args.size - m)
    l0 = np.asarray(arr[0, iz, iy, ix]).astype(np.float32)
    l1 = np.asarray(arr[1, iz, iy, ix]).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-(l1 - l0)))
    pos = float((p > 0.2).mean())
    span = float(l1.max() - l1.min())
    conf = float(np.abs(p - 0.5).mean() * 2)

    healthy = conf >= MIN_MEAN_CONFIDENCE and PLAUSIBLE_POSITIVE[0] <= pos <= PLAUSIBLE_POSITIVE[1]
    outside = not (OBSERVED_POSITIVE[0] <= pos <= OBSERVED_POSITIVE[1])
    report = {
        "scroll": args.scroll, "volume": volume, "ct_level": level,
        "resolution_um": best["um"], "volume_shape": best["shape"],
        "bbox": [z, z + args.size, y, y + args.size, x, x + args.size],
        "trim": m, "threshold": 0.2,
        "positive_fraction": pos,
        "logit_fg_range": [float(l1.min()), float(l1.max())],
        "logit_bg_range": [float(l0.min()), float(l0.max())],
        "logit_span": span,
        "mean_confidence": conf,
        "confidence_reference": {"reproduced_range": [0.231, 0.944],
                                 "known_degenerate": 0.032},
        "regime": "normal" if healthy else "degenerate",
        "positive_fraction_outside_observed_range": outside,
        "published_prediction": None,
        "note": ("No published prediction exists for this scroll, so this is a new "
                 "artifact and not a reproduction. No Dice is claimable. The regime "
                 "call rests on mean confidence, calibrated against regions of known "
                 "status; logit span was tested and does not discriminate."),
    }
    Path(args.out).mkdir(exist_ok=True)
    out_json = Path(args.out) / f"{args.scroll}_NEW_surface-m7.json"
    Path(out_json).write_text(json.dumps(report, indent=2))

    print(f"\n  positive fraction {100*pos:.2f}%   foreground logits "
          f"[{l1.min():.1f}, {l1.max():.1f}]  span {span:.1f}")
    print(f"  mean confidence {conf:.3f}  "
          f"(reproduced regions 0.231-0.944, known degenerate 0.032)")
    if outside:
        print(f"  NOTE positive fraction is outside the {100*OBSERVED_POSITIVE[0]:.0f}-"
              f"{100*OBSERVED_POSITIVE[1]:.0f}% seen in published regions")
    print(f"  regime: {report['regime']}")
    print(f"  wrote {out_json}")


if __name__ == "__main__":
    main()
