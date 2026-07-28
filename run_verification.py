"""End-to-end: pick a region of a scroll, regenerate it, and score it.

Chains the whole check for one scroll - find a region with actual surface in it,
run the official predictor over just that region, blend, and compare against the
published artifact. This is a regional spot-check, not a full-volume comparison.

Everything it needs comes from catalog.json, so it works for any scroll with a
published surface prediction rather than a hand-wired one.

Usage:
    python run_verification.py --scroll PHerc0125 --model m7
    python run_verification.py --scroll PHercParis4 --model m7 --level 2 --tta
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = os.environ.get(
    "VESUVIUS_MODEL_PATH", "hf://scrollprize/surface_m7_nnunet")


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
    import numpy as np
    import zarr

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
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--catalog", default=str(HERE / "catalog.json"))
    ap.add_argument("--work-dir", default=str(HERE / "outputs"),
                    help="temporary logits and blended output directory")
    ap.add_argument("--python", default=os.environ.get("VESUVIUS_PYTHON", sys.executable),
                    help="Python interpreter containing vesuvius and its model dependencies")
    ap.add_argument("--model-path", default=DEFAULT_MODEL,
                    help="local nnU-Net model directory or hf:// model reference; "
                         "defaults to VESUVIUS_MODEL_PATH or the public m7 model")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--predict-script", default=os.environ.get("VESUVIUS_PREDICT_SCRIPT"),
                    help="optional compatibility wrapper instead of "
                         "`python -m vesuvius.models.run.inference`")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--read-retries", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the artifact and region, then print commands without running")
    ap.add_argument("--bbox", default=None,
                    help="z0:z1,y0:y1,x0:x1 instead of selecting a region")
    ap.add_argument("--tta", action="store_true",
                    help="enable mirroring test-time augmentation (default off)")
    ap.add_argument("--tag", default=None,
                    help="suffix for a variant result, preventing baseline overwrite")
    args = ap.parse_args()

    with open(args.catalog, encoding="utf-8") as f:
        rows = json.load(f)
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
    if args.bbox:
        bbox = args.bbox
        print(f"  region {bbox}  (given)")
    else:
        z, y, x, pos = pick_region(pred_url, args.size)
        bbox = f"{z}:{z+args.size},{y}:{y+args.size},{x}:{x+args.size}"
        print(f"  region {bbox}  ({100*pos:.2f}% positive)")
    if args.tta:
        print("  mirroring TTA ENABLED - expect roughly 8x the inference time")

    suffix = f"_{args.tag}" if args.tag else ""
    work = Path(args.work_dir) / f"verify_{args.scroll}{suffix}"

    env_extra = {"nnUNet_compile": "0", "TORCHDYNAMO_DISABLE": "1",
                 "PYTHONIOENCODING": "utf-8"}
    env = {**os.environ, **env_extra}

    print("  predicting ...")
    predict_entry = ([args.python, str(Path(args.predict_script).resolve())]
                     if args.predict_script else
                     [args.python, "-m", "vesuvius.models.run.inference"])
    cmd_predict = predict_entry + [
        "--model_path", args.model_path,
        "--input_dir", ct_url,
        "--output_dir", str(work / "logits"),
        "--device", args.device,
        "--batch_size", "1",
        "--num_workers", str(args.num_workers),
        "--read-retries", str(args.read_retries),
        "--bbox", bbox,
    ]
    if not args.tta:
        cmd_predict.insert(cmd_predict.index("--device") + 2, "--disable_tta")
    if args.dry_run:
        print("  " + subprocess.list2cmdline(cmd_predict))
        return
    if work.exists():
        import shutil
        shutil.rmtree(work)
    r = subprocess.run(cmd_predict, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("predict failed")

    # A run against a tree WITHOUT the global-grid fix (villa#1247) still succeeds -
    # it just anchors the sliding window to the ROI, giving 8 patches at stride 64
    # instead of the 64+ patches selected from the volume's grid. The output looks plausible and
    # scores ~0.6-0.9, which reads as "this scroll does not reproduce" rather than
    # "you are on the wrong branch". That mistake nearly went out as a finding, so
    # it is now a hard failure rather than a silent one.
    import zarr

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
    r = subprocess.run([args.python, "-c", blend], env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit("blend failed")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out) / "variants" if args.tag else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.scroll}{suffix}_{row['model']}.json"
    print("  scoring ...")
    cmd = [args.python, str(HERE / "verify_region.py"), "--scroll", args.scroll,
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
    report = json.loads(out_json.read_text(encoding="utf-8"))
    scored = args.size - 2 * args.trim
    report["verification_scope"] = (
        f"one selected {args.size}^3 region; central {scored}^3 scored")
    report["runner"] = {
        "tta": args.tta,
        "device": args.device,
        "model_path": args.model_path,
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
