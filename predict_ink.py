"""Run the public 3D ink model on scrolls that have no published ink prediction.

Exactly one scroll in the open-data bucket carries an ink prediction: PHerc. Paris 4.
Thirty-five others have verified surface predictions and no ink at all. The model that
produced the Paris 4 artifact is public (scrollprize/ink_3d_dino_guided,
ckpt_78k_fullsup.pth - the checkpoint name matches the artifact's "v3-78k-fullsup"),
and it reproduces that artifact to the limit of its 8-bit storage (Pearson 0.9806,
mean |diff| 0.0034 against a quantisation step of 0.0039). So the pipeline is verified
before being pointed anywhere new.

TWO THINGS THIS IS NOT.

  * It is not a claim that these predictions are correct. There is no ground truth and
    no published artifact to score against, so no Dice is quoted for any scroll but
    Paris 4.
  * The released checkpoint's final full-supervision configuration points at PHerc.
    Paris 4, while its model card describes a broader teacher/self-distillation lineage.
    Running it elsewhere is still outside that final supervised domain. The output may
    be worthless on some scrolls, and saying which is the point of the exercise rather
    than a caveat on it.

What IS reported is the ink-signal distribution against the one scroll where the model
is known to work, so a reader can see where signal is plausibly recoverable and where
the model simply produces nothing.

The ink model runs at native ~2.4um (level 0), while the surface predictions sit at
level 2 (~9.6um). Regions are chosen by probing the surface prediction - ink lives on
sheets, so a region with no surface has nothing to find - and coordinates are scaled by
4 to reach level 0.

Usage:
    python predict_ink.py --scroll PHerc0332
    python predict_ink.py --all
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
PY = os.environ.get("VESUVIUS_PYTHON", sys.executable)
INK_CKPT = os.environ.get(
    "VESUVIUS_INK_MODEL_PATH", str(HERE / "model_ink3d" / "ckpt_78k_fullsup.pth"))
PREDICT_SCRIPT = os.environ.get("VESUVIUS_PREDICT_SCRIPT")
SIZE = 256

# Measured on PHerc. Paris 4, where this model is known to work (published artifact,
# interior of 34102:34358,16346:16602,16346:16602).
PARIS4_REF = {"mean": 0.020, "frac_gt_0.5": 0.0127, "max": 0.988}

# ...but ONE region of Paris 4 is not a baseline. paris4_ink_range.py measured the
# published artifact over 16 regions picked by CT content, blind to ink: the hot fraction
# runs 0.000%-14.7% and P(ink|dense) runs 0.000-0.374, median 0.059. The single region
# above sits near the bottom of its own scroll's spread, so dividing by it manufactures
# large multiples out of ordinary variation - it is what produced the "PHerc0332 is 4x
# Paris 4" and "PHerc0343P is 13x" readings, both of which fall inside the range above.
# A ratio against one region is therefore not reported. The range is.
PARIS4_RANGE = {
    "n_regions": 16,
    "frac_gt_0.5": {"min": 0.0, "median": 0.03753, "max": 0.14718},
    "p_ink_dense": {"min": 0.0, "median": 0.05875, "max": 0.37388},
    "source": "paris4_ink_range.py, published ink-3d artifact",
}


def classify(rep: dict, ink_std: float | None = None) -> dict:
    """Attach status, verdict and the comparison note. Shared by fresh runs and --refresh.

    A constant output means the run produced nothing. That is a failure of the run and
    must never be reported as "this scroll has no ink" - the two look identical in a
    summary table and mean opposite things.
    """
    lo, hi = PARIS4_RANGE["frac_gt_0.5"]["min"], PARIS4_RANGE["frac_gt_0.5"]["max"]
    rep["paris4_range"] = PARIS4_RANGE
    rep.pop("signal_vs_paris4", None)  # ratio against a single region - see PARIS4_RANGE

    flat = (ink_std is not None and ink_std < 1e-6) or \
           (rep["ink_max"] - rep["ink_mean"] < 1e-6)
    if flat:
        rep["status"] = "degenerate_constant_output"
        rep["verdict"] = (f"FAILED: output is constant at {rep['ink_mean']:.4f} over the "
                          "whole region, so the model produced nothing here. This is not "
                          "a finding about the scroll.")
    elif rep["ink_max"] < 0.5:
        rep["verdict"] = (f"no voxel reaches 0.5 (max {rep['ink_max']:.3f}) - no ink "
                          "signal in this region")
    elif rep["frac_gt_0.5"] > hi:
        rep["verdict"] = ("above every region measured on Paris 4 - check for saturation "
                          "before reading this as more ink")
    else:
        rep["verdict"] = "within the range Paris 4 itself spans across regions"
    rep["note_on_comparison"] = (
        f"Paris 4 spans {100*lo:.3f}%-{100*hi:.3f}% hot across "
        f"{PARIS4_RANGE['n_regions']} regions, so a ratio against any single Paris 4 "
        "region is not meaningful and none is quoted.")
    return rep


def retry(fn, attempts: int = 8):
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"    transient read, retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def pick_region_from_surface(scroll: str, row: dict) -> tuple[int, int, int, float]:
    """Locate a region with real surface at L2, then scale x4 into L0 for the ink model.

    The surface prediction alone is NOT enough to choose a region. On several scrolls the
    published prediction is not masked to the CT support and reports confident sheet where
    the scan has no data at all - 92% of predicted sheet voxels on PHerc0343P, 77% on
    PHerc0500P2, verified at level 0 (pred_over_empty_ct.py). Picking on surface fraction
    alone sent a PHerc0500P2 region with "46.9% surface" and identically zero CT to the
    GPU, which burned an inference and returned a constant 0.5 that then had to be told
    apart from a real "no ink here". So the CT is checked on the same grid and any
    candidate without material underneath it is rejected.
    """
    url = (f"{BUCKET}/{scroll}/representations/predictions/surfaces/"
           f"{row['prediction']}/0")
    a = zarr.open(url, mode="r")
    ct2 = zarr.open(f"{BUCKET}/{scroll}/volumes/{row['ct_volume']}/2", mode="r")
    same_grid = ct2.shape == a.shape
    if not same_grid:
        print(f"  note: CT L2 {ct2.shape} and prediction {a.shape} differ; "
              "cannot pre-check CT on a shared grid")
    Z, Y, X = a.shape
    probe = SIZE // 4  # 64 voxels at L2 == the 256 the ink model will see at L0
    best = None
    for fz in (0.35, 0.5, 0.65):
        z, y, x = int(Z * fz), Y // 2 - probe // 2, X // 2 - probe // 2
        z, y, x = max(0, z), max(0, y), max(0, x)
        if z + probe > Z or y + probe > Y or x + probe > X:
            continue
        blk = retry(lambda: np.asarray(a[z:z + probe, y:y + probe, x:x + probe]))
        frac = float((blk > 0).mean())
        filled = 1.0
        if same_grid:
            cblk = retry(lambda: np.asarray(
                ct2[z:z + probe, y:y + probe, x:x + probe]))
            filled = float((cblk > 0).mean())
        print(f"  surface probe L2 z={z} y={y} x={x}: {100*frac:.1f}% surface, "
              f"{100*filled:.1f}% CT present")
        if filled < 0.5:
            print("    rejected: prediction sits over empty CT here")
            continue
        if best is None or frac > best[3]:
            best = (z, y, x, frac)
    if best is None or best[3] < 0.02:
        raise SystemExit(f"{scroll}: no region with both surface content and CT data")
    z, y, x, frac = best
    return z * 4, y * 4, x * 4, frac


def run_one(scroll: str, catalog: list[dict], out_dir: Path, keep: bool) -> dict | None:
    rows = [r for r in catalog if r.get("scroll") == scroll
            and r.get("status") == "verifiable" and "m7" in r.get("model", "")
            and r.get("declared_level") == 2]
    if not rows:
        print(f"  {scroll}: no L2 surface prediction, so no 2.4um scan to run ink on")
        return None
    row = rows[0]
    ct = row["ct_volume"]
    ct_url = f"{BUCKET}/{scroll}/volumes/{ct}/0"

    print(f"=== {scroll}   {ct}   ink at L0 (~2.4um) ===", flush=True)
    z, y, x, surf = pick_region_from_surface(scroll, row)
    bbox = f"{z}:{z+SIZE},{y}:{y+SIZE},{x}:{x+SIZE}"
    print(f"  L0 region {bbox}  (surface fraction {100*surf:.1f}%)", flush=True)

    work = out_dir / f"ink_{scroll}"
    if work.exists():
        shutil.rmtree(work)
    env = {**os.environ, "nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
           "PYTHONIOENCODING": "utf-8"}

    print("  predicting ...", flush=True)
    predict_entry = ([PY, PREDICT_SCRIPT] if PREDICT_SCRIPT else
                     [PY, "-m", "vesuvius.models.run.inference"])
    r = subprocess.run(predict_entry + ["--model_path", INK_CKPT,
                        "--model-type", "train_py", "--input_dir", ct_url,
                        "--output_dir", str(work / "logits"), "--device", "cuda",
                        "--disable_tta", "--batch_size", "1", "--num_workers", "2",
                        "--read-retries", "12", "--bbox", bbox],
                       env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print((r.stdout or "")[-1500:], (r.stderr or "")[-1500:])
        return {"scroll": scroll, "status": "predict_failed"}

    print("  blending ...", flush=True)
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
        print((r.stdout or "")[-1500:], (r.stderr or "")[-1500:])
        return {"scroll": scroll, "status": "blend_failed"}

    m = 64
    a = zarr.open(str(work / "merged.zarr"), mode="r")
    iz, iy, ix = (slice(z + m, z + SIZE - m), slice(y + m, y + SIZE - m),
                  slice(x + m, x + SIZE - m))
    logit = np.asarray(a[0, iz, iy, ix]).astype(np.float32)
    p = 1.0 / (1.0 + np.exp(-logit))

    rep = {
        "scroll": scroll, "ct_volume": ct, "ct_level": 0,
        "bbox": [z, z + SIZE, y, y + SIZE, x, x + SIZE], "trim": m,
        "surface_fraction_at_region": surf,
        "ink_mean": float(p.mean()), "ink_max": float(p.max()),
        "ink_p99": float(np.quantile(p, 0.99)),
        "frac_gt_0.3": float((p > 0.3).mean()),
        "frac_gt_0.5": float((p > 0.5).mean()),
        "frac_gt_0.7": float((p > 0.7).mean()),
        "paris4_reference": PARIS4_REF,
        "model": "scrollprize/ink_3d_dino_guided ckpt_78k_fullsup.pth",
        "trained_on": "PHercParis4 only - this is cross-scroll generalisation",
        "published_ink_prediction": None,
        "note": ("No published ink prediction exists for this scroll, so this is a new "
                 "artifact and not a reproduction. No Dice is claimable."),
        "status": "ok",
    }
    classify(rep, float(p.std()))
    lo, hi = PARIS4_RANGE["frac_gt_0.5"]["min"], PARIS4_RANGE["frac_gt_0.5"]["max"]
    print(f"  ink mean {rep['ink_mean']:.4f}  max {rep['ink_max']:.3f}  "
          f">0.5 {100*rep['frac_gt_0.5']:.3f}%  "
          f"(Paris 4 spans {100*lo:.2f}-{100*hi:.2f}% across regions)")
    print(f"  verdict: {rep['verdict']}", flush=True)

    if not keep:
        shutil.rmtree(work / "logits", ignore_errors=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--out", default="results/ink")
    ap.add_argument("--work", default="outputs")
    ap.add_argument("--keep-logits", action="store_true")
    ap.add_argument("--refresh-verdicts", action="store_true",
                    help="re-derive verdicts in existing result JSONs, no inference")
    args = ap.parse_args()

    catalog = json.load(open(args.catalog))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.refresh_verdicts:
        for jf in sorted(out.glob("*_ink3d.json")):
            rep = json.loads(jf.read_text())
            if "ink_max" not in rep:
                continue
            before = rep.get("verdict", "")
            classify(rep)
            jf.write_text(json.dumps(rep, indent=2))
            flag = "  <-- CHANGED" if rep["verdict"] != before else ""
            print(f"{rep['scroll']:<12} {rep['verdict']}{flag}")
        return

    if args.all:
        scrolls = sorted({r["scroll"] for r in catalog
                          if r.get("status") == "verifiable"
                          and "m7" in r.get("model", "")
                          and r.get("declared_level") == 2
                          and r["scroll"] != "PHercParis4"})
    elif args.scroll:
        scrolls = [args.scroll]
    else:
        raise SystemExit("pass --scroll NAME or --all")

    print(f"{len(scrolls)} scroll(s) to run\n")
    for i, s in enumerate(scrolls, 1):
        dest = out / f"{s}_ink3d.json"
        if dest.exists():
            print(f"[{i}/{len(scrolls)}] {s}: already done, skipping")
            continue
        print(f"[{i}/{len(scrolls)}]", flush=True)
        try:
            rep = run_one(s, catalog, Path(args.work), args.keep_logits)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if rep:
            dest.write_text(json.dumps(rep, indent=2))
            print(f"  wrote {dest}\n", flush=True)


if __name__ == "__main__":
    main()
