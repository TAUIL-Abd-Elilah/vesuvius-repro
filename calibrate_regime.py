"""What does a healthy m7 output actually look like?

predict_uncovered.py has to say whether the model is operating normally on a scroll
with no published prediction to compare against. That judgement was initially made
against a constant I picked by eye, which is not evidence. This measures the same
statistics on the regions whose status is already known - the eleven that reproduce,
plus the two known-bad cases - so the threshold is calibrated rather than asserted.

Local data only; no network.

Usage:
    python calibrate_regime.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import zarr

warnings.filterwarnings("ignore")

# (store, results json holding the bbox, label, known verdict)
CASES = [
    ("outputs/verify_PHerc0125/merged.zarr",   "results/PHerc0125_surface-m7.json",   "PHerc0125 L0",   "reproduced"),
    ("outputs/verify_PHerc0175A/merged.zarr",  "results/PHerc0175A_surface-m7.json",  "PHerc0175A L0",  "reproduced"),
    ("outputs/verify_PHerc0191/merged.zarr",   "results/PHerc0191_surface-m7.json",   "PHerc0191 L0",   "reproduced"),
    ("outputs/verify_PHerc0211/merged.zarr",   "results/PHerc0211_surface-m7.json",   "PHerc0211 L0",   "reproduced"),
    ("outputs/verify_PHerc0500P2/merged.zarr", "results/PHerc0500P2_surface-m7.json", "PHerc0500P2 L2", "reproduced"),
    ("outputs/verify_PHerc0814/merged.zarr",   "results/PHerc0814_surface-m7.json",   "PHerc0814 L2",   "reproduced"),
    ("outputs/verify_PHerc0841/merged.zarr",   "results/PHerc0841_surface-m7.json",   "PHerc0841 L2",   "reproduced"),
    ("outputs/verify_PHerc1203/merged.zarr",   "results/PHerc1203_surface-m7.json",   "PHerc1203 L2",   "reproduced"),
    ("outputs/p0846_r2/merged.zarr",           "results/PHerc0846A_r2_surface-m7.json", "PHerc0846A r2", "reproduced"),
    ("outputs/repro_gpu_grid/merged.zarr",     "results/results_phec0139_r1.json",    "PHerc0139 L0",   "reproduced"),
    ("outputs/verify_PHerc0846A/merged.zarr",  "results/PHerc0846A_surface-m7.json",  "PHerc0846A r1",  "KNOWN DEGENERATE"),
    ("outputs/paris4/merged.zarr",             "results/results_paris4_r1.json",      "PHercParis4 r1", "not reproduced"),
    ("outputs/new_PHerc1667/merged.zarr",      "results/PHerc1667_NEW_surface-m7.json", "PHerc1667 L2*", "NEW - unknown"),
]


def stats(path: str, meta_path: str) -> dict | None:
    p = Path(path)
    if not p.exists() or not Path(meta_path).exists():
        return None
    meta = json.load(open(meta_path))
    z0, z1, y0, y1, x0, x1 = meta["bbox"]
    m = meta.get("trim", 64)
    a = zarr.open(str(p), mode="r")
    # slice the scored interior directly - the store is full-volume-shaped and
    # sparse, so materializing a whole channel tries to allocate terabytes
    iz, iy, ix = slice(z0 + m, z1 - m), slice(y0 + m, y1 - m), slice(x0 + m, x1 - m)
    f = np.asarray(a[1, iz, iy, ix]).astype(np.float32).ravel()
    b = np.asarray(a[0, iz, iy, ix]).astype(np.float32).ravel()
    if f.size == 0:
        return None
    p_fg = 1.0 / (1.0 + np.exp(-(f - b)))
    return {
        "fg_min": float(f.min()), "fg_max": float(f.max()),
        "span": float(f.max() - f.min()),
        "positive_fraction": float((p_fg > 0.2).mean()),
        "mean_confidence": float(np.abs(p_fg - 0.5).mean() * 2),
        "frac_p_above_0.9": float((p_fg > 0.9).mean()),
    }


def main() -> None:
    print(f"{'case':18} {'verdict':17} {'fg logits':>18} {'span':>6} "
          f"{'pos%':>6} {'conf':>6} {'p>.9':>6}")
    rows = []
    for path, meta_path, label, verdict in CASES:
        s = stats(path, meta_path)
        if s is None:
            print(f"{label:18} {verdict:17} (store missing)")
            continue
        rows.append((label, verdict, s))
        print(f"{label:18} {verdict:17} "
              f"[{s['fg_min']:7.1f},{s['fg_max']:6.1f}] {s['span']:6.1f} "
              f"{100*s['positive_fraction']:6.1f} {s['mean_confidence']:6.3f} "
              f"{100*s['frac_p_above_0.9']:6.1f}")

    good = [s for _, v, s in rows if v == "reproduced"]
    bad = [s for _, v, s in rows if v == "KNOWN DEGENERATE"]
    if good:
        spans = [s["span"] for s in good]
        confs = [s["mean_confidence"] for s in good]
        print(f"\nhealthy span   : min {min(spans):.1f}  max {max(spans):.1f}")
        print(f"healthy conf   : min {min(confs):.3f}  max {max(confs):.3f}")
    if bad:
        print(f"degenerate span: {bad[0]['span']:.1f}   conf {bad[0]['mean_confidence']:.3f}")
    json.dump(rows, open("results/regime_calibration.json", "w"), indent=2)
    print("\nwrote results/regime_calibration.json")


if __name__ == "__main__":
    main()
