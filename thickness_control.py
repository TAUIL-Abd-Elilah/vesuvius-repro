"""Control: what thickness does this estimator report for a PERFECT label?

The label-thinning hypothesis rests on "published labels are 3.335 voxels thick against a
~2.4 voxel sheet, so they are fat". That comparison is only valid if the estimator returns
~2.4 for a correct label. It might not: a binary mask that covers a 2.4-voxel sheet has to
round outward at both faces, so voxelisation alone inflates the measured run.

So synthesise sheets of KNOWN thickness, voxelise them the way a label is, and measure with
the same machinery used on the real labels. If a correct 2.4-voxel sheet already reads ~3.3,
the labels are not fat, the premise is an artifact, and the study stops here.

  python thickness_control.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from thin_labels import across_sheet_dirs, measured_thickness

ROOT = Path(__file__).resolve().parent
SHAPE = (64, 64, 64)


def synth(true_thickness: float, angle_deg: float, seed: int = 0):
    """A planar sheet of known thickness at a given tilt, plus the CT it would produce.

    Tilted deliberately: an axis-aligned sheet is the one case where voxelisation is exact,
    so testing only that would flatter the estimator.
    """
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:SHAPE[0], 0:SHAPE[1], 0:SHAPE[2]].astype(np.float32)
    a = np.deg2rad(angle_deg)
    n = np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float32)   # unit normal
    d = (z - SHAPE[0] / 2) * n[0] + (y - SHAPE[1] / 2) * n[1] + (x - SHAPE[2] / 2) * n[2]

    # CT: bright where |d| is small, i.e. a slab with soft edges, plus noise
    ct = 40.0 + 120.0 * np.exp(-(d ** 2) / (2 * (true_thickness / 2.355) ** 2))
    ct = np.clip(ct + rng.normal(0, 4.0, ct.shape), 0, 255).astype(np.uint8)

    # label: the correct binary mask of that slab -- a voxel is labelled if its CENTRE lies
    # inside the true slab. This is the best a correct annotation can do on this grid.
    lab = (np.abs(d) <= true_thickness / 2).astype(np.uint8)
    return ct, lab


def main() -> None:
    rows = []
    for true_t in (2.0, 2.4, 3.0, 3.5):
        for angle in (0, 15, 30, 45):
            ct, lab = synth(true_t, angle)
            n_lab = int(lab.sum())
            if n_lab < 200:
                continue
            meas = measured_thickness(ct, lab, n=1500, seed=0)
            rows.append({"true_thickness": true_t, "angle_deg": angle,
                         "n_label_voxels": n_lab,
                         "measured_thickness": None if meas is None else round(meas, 4),
                         "inflation": None if meas is None else round(meas - true_t, 4)})
            print(f"  true {true_t:>4}  tilt {angle:>2}deg  ->  measured "
                  f"{rows[-1]['measured_thickness']}   (+{rows[-1]['inflation']})", flush=True)

    ok = [r for r in rows if r["measured_thickness"] is not None]
    by_true = {}
    for r in ok:
        by_true.setdefault(r["true_thickness"], []).append(r["measured_thickness"])
    summary = {
        "note": ("estimator response to a CORRECT label of known thickness; the real labels "
                 "measure 3.335 on the same estimator"),
        "mean_measured_by_true": {str(k): round(float(np.mean(v)), 4) for k, v in by_true.items()},
        "mean_inflation": round(float(np.mean([r["inflation"] for r in ok])), 4),
        "real_label_thickness": 3.3349,
        "rows": rows,
    }
    Path(ROOT / "results" / "thickness_control.json").write_text(json.dumps(summary, indent=1))

    print("\n  mean measured, by true thickness:")
    for k, v in summary["mean_measured_by_true"].items():
        print(f"    true {k}  ->  {v}")
    print(f"\n  mean inflation from voxelisation alone: +{summary['mean_inflation']} vox")
    print(f"  real published labels measure 3.3349 on this estimator")
    m24 = summary["mean_measured_by_true"].get("2.4")
    if m24 is not None:
        print(f"\n  READING: a correct 2.4-voxel sheet reads {m24}. "
              f"{'The premise survives.' if m24 < 3.0 else 'The premise is an ARTIFACT - stop.'}")


if __name__ == "__main__":
    main()
