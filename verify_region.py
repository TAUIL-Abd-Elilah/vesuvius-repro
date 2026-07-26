"""Score a locally reproduced surface prediction against the published one.

Given a blended logits store produced by `vesuvius.predict` + `vesuvius.blend_logits`
over some region, this says how closely it matches the published artifact for the
same voxels, and - when it does not match - what class of explanation is still open.

The scroll's published prediction, its CT and the pyramid level they live on are
resolved from catalog.json (see catalog_predictions.py), so this works for any of
the scrolls that have a published surface prediction, not just a hand-wired one.

Two measurements do most of the work:

  * a sweep over every threshold. This is strictly more generous than any global
    precision or calibration change, so its maximum bounds what fp16 autocast,
    float16 storage or a threshold ambiguity could ever buy. If the sweep does not
    reach agreement, the difference is structural and no amount of numerics fixes it.

  * where the disagreement sits relative to the threshold. Voxels far from the
    decision boundary cannot be flipped by a small perturbation, so the fraction of
    disagreement lying far away is a floor on what must be explained structurally.

Usage:
    python verify_region.py --scroll PHerc0139 --ours outputs/repro/merged.zarr \
        --bbox 8000:8256,3200:3456,3200:3456 [--trim 64] [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import zarr

warnings.filterwarnings("ignore")

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"


def retry(fn, attempts: int = 8):
    """This bucket drops connections routinely; see ScrollPrize/villa#1244."""
    delay = 1.0
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"  transient read ({type(exc).__name__}), retry in {delay:.0f}s")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def parse_bbox(text: str) -> tuple[int, ...]:
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError("bbox must be z0:z1,y0:y1,x0:x1")
    out: list[int] = []
    for p in parts:
        lo, hi = p.split(":")
        out += [int(lo), int(hi)]
    return tuple(out)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / s


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return 1.0 if u == 0 else float((a & b).sum()) / u


def load_catalog(path: str) -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


def resolve(catalog: list[dict], scroll: str,
            model: str | None = None, level: int | None = None) -> dict:
    """Pick one published prediction, refusing to guess when it is ambiguous.

    A scroll can carry predictions from different models at different CT levels
    (PHercParis4 has both surface-m7 at L2 and surface-recto-2um-ps256 at L0).
    Silently taking the first scores against the wrong artifact, which looks like
    a catastrophic reproduction failure rather than the mix-up it is.
    """
    hits = [r for r in catalog if r.get("scroll") == scroll and r.get("prediction")]
    if not hits:
        raise SystemExit(f"{scroll} has no published surface prediction in the catalog")
    if model:
        hits = [r for r in hits if model in r["model"]]
    if level is not None:
        hits = [r for r in hits if r["declared_level"] == level]
    if not hits:
        raise SystemExit(f"no prediction for {scroll} matching model={model} level={level}")
    if len(hits) > 1:
        lines = "\n".join(f"    --model {r['model']} --level {r['declared_level']}"
                          for r in hits)
        raise SystemExit(
            f"{scroll} has {len(hits)} published predictions; disambiguate with:\n{lines}")
    return hits[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--ours", required=True, help="blended logits zarr")
    ap.add_argument("--bbox", required=True, help="z0:z1,y0:y1,x0:x1 in prediction voxels")
    ap.add_argument("--trim", type=int, default=64,
                    help="voxels trimmed per face before scoring")
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--model", default=None,
                    help="substring of the model family, e.g. m7 (needed when a "
                         "scroll has more than one published prediction)")
    ap.add_argument("--level", type=int, default=None, help="CT pyramid level")
    ap.add_argument("--json", default=None, help="write a machine-readable report here")
    args = ap.parse_args()

    row = resolve(load_catalog(args.catalog), args.scroll, args.model, args.level)
    level = row["declared_level"]
    published = (f"{BUCKET}/{args.scroll}/representations/predictions/surfaces/"
                 f"{row['prediction']}/0")
    threshold = row["threshold"]

    z0, z1, y0, y1, x0, x1 = parse_bbox(args.bbox)
    m = args.trim
    iz, iy, ix = slice(z0 + m, z1 - m), slice(y0 + m, y1 - m), slice(x0 + m, x1 - m)

    print(f"scroll {args.scroll}  CT level L{level}  published threshold {threshold}")
    print(f"region {(z0, z1, y0, y1, x0, x1)}  interior trim {m}")

    print("reading published prediction ...")
    pub = retry(lambda: np.asarray(zarr.open(published, mode="r")[iz, iy, ix])) > 0

    print("reading our blended logits ...")
    arr = zarr.open(args.ours, mode="r")
    # blend_logits writes raw logits, channel 0 background / 1 foreground. The
    # published artifact is thresholded on the softmax, so recover that rather
    # than thresholding a logit directly - they are not the same number.
    l0 = np.asarray(arr[0, iz, iy, ix]).astype(np.float32)
    l1 = np.asarray(arr[1, iz, iy, ix]).astype(np.float32)
    ours_p = 1.0 / (1.0 + np.exp(-(l1 - l0)))

    ours_b = ours_p > threshold
    base = dice(ours_b, pub)
    disagree = ours_b != pub
    n_dis = int(disagree.sum())

    print(f"\npublished positives {100 * pub.mean():.2f}%   "
          f"ours {100 * ours_b.mean():.2f}%")
    print(f"Dice {base:.4f}   IoU {iou(ours_b, pub):.4f}   "
          f"agreement {100 * (ours_b == pub).mean():.3f}%")
    print(f"disagreeing voxels {n_dis:,} ({100 * disagree.mean():.3f}% of scored region)")

    # bound on any global precision / calibration change
    taus = np.concatenate([np.arange(0.01, 0.10, 0.01), np.arange(0.10, 0.51, 0.02),
                           np.arange(0.55, 0.96, 0.05)])
    best_tau, best_dice = float(threshold), base
    for tau in taus:
        d = dice(ours_p > tau, pub)
        if d > best_dice:
            best_tau, best_dice = float(tau), d
    print(f"best Dice over all thresholds {best_dice:.4f} at tau={best_tau:.2f} "
          f"({best_dice - base:+.4f} headroom)")

    near = {}
    if n_dis:
        d_p = ours_p[disagree]
        for eps in (0.01, 0.05, 0.15):
            near[eps] = float(np.mean(np.abs(d_p - threshold) < eps))
        print(f"disagreement within 0.01 of threshold: {100 * near[0.01]:.1f}%   "
              f"beyond 0.15: {100 * (1 - near[0.15]):.1f}%")

    report = {
        "scroll": args.scroll, "ct_level": level, "prediction": row["prediction"],
        "run_id": row["run_id"], "threshold": threshold,
        "bbox": [z0, z1, y0, y1, x0, x1], "trim": m,
        "scored_shape": list(pub.shape),
        "published_positive_fraction": float(pub.mean()),
        "our_positive_fraction": float(ours_b.mean()),
        "dice": base, "iou": iou(ours_b, pub),
        "agreement": float((ours_b == pub).mean()),
        "disagreeing_voxels": n_dis,
        "best_dice_any_threshold": best_dice, "best_threshold": best_tau,
        "disagreement_near_threshold": {str(k): v for k, v in near.items()},
    }

    print("\n--- verdict ---")
    if base > 0.99:
        print(f"  Reproduced. {n_dis:,} voxels differ; "
              f"{100 * near.get(0.01, 1.0):.0f}% of them sit within 0.01 of the")
        print("  threshold, which is what float16 storage and autocast leave behind.")
        report["verdict"] = "reproduced"
    elif best_dice - base < 0.02:
        print("  Not reproduced, and no threshold does better - so precision and")
        print("  calibration are ruled out. The difference is structural: patch grid,")
        print("  step size, patch size, normalization or preprocessing.")
        report["verdict"] = "structural_difference"
    else:
        print(f"  Not reproduced, but tau={best_tau:.2f} reaches {best_dice:.4f}; "
              "calibration is in play.")
        report["verdict"] = "calibration_difference"

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
