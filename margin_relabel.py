"""Arm B's label transform: the asserted sheet margin becomes `ignore`, never `surface`.

Registered in PREREGISTER_margin.md section 3. Gate zero (margin_class_scale.py, 200 volumes)
established that 97.4% of the voxel immediately outside a labelled sheet run is class 0 --
positively asserted background -- on roughly half a voxel of what the CT profile calls sheet.

This withdraws that assertion and nothing more:

    class 0, within 1 voxel of a labelled sheet voxel ALONG THE ACROSS-SHEET NORMAL  ->  2

`ignore`, not `surface`. Dilating to surface would assert sheet on the strength of a smoothed
profile's FWHM, which is a generous definition of where papyrus ends. Withdrawing an
unjustified not-sheet claim can cost a little supervision; it cannot teach the model something
false.

**Why stepping one voxel along the normal is the same thing as finding each run's boundary.**
A labelled voxel in the middle of a run steps onto another labelled voxel, which is class 1
and therefore untouched. Only a voxel at the run's edge steps onto class 0. So stepping +-1
from every labelled voxel selects exactly the class-0 voxels within one voxel of the sheet
across the sheet, without a per-run Python loop -- and it is restricted to the normal, so the
sheet's lateral edges (where an annotation simply stops) are left alone.

  python margin_relabel.py --check --n 12
  python margin_relabel.py --out data/kaggle/labels_margin
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

from thin_labels import across_sheet_dirs

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"


def relabel_margin(ct: np.ndarray, lab: np.ndarray,
                   chunk: int = 400_000) -> tuple[np.ndarray, dict]:
    """Return a copy of `lab` with the across-sheet margin moved from 0 to 2."""
    out = lab.copy()
    pts = np.argwhere(lab == 1)
    if len(pts) == 0:
        return out, {"n_sheet": 0, "n_relabelled": 0, "frac_of_volume": 0.0}

    shape = np.array(lab.shape)
    touched = np.zeros(lab.shape, dtype=bool)

    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk].astype(np.int32)
        d = across_sheet_dirs(ct, p)
        for sign in (1.0, -1.0):
            q = np.rint(p + sign * d).astype(np.int64)
            np.clip(q, 0, shape - 1, out=q)
            touched[q[:, 0], q[:, 1], q[:, 2]] = True

    sel = touched & (lab == 0)
    out[sel] = 2
    return out, {
        "n_sheet": int((lab == 1).sum()),
        "n_relabelled": int(sel.sum()),
        "frac_of_volume": round(float(sel.mean()), 6),
        "frac_of_sheet": round(float(sel.sum() / max(1, (lab == 1).sum())), 4),
    }


def class_fracs(lab: np.ndarray) -> dict:
    return {str(int(c)): round(float((lab == c).mean()), 5) for c in (0, 1, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate on a sample, write nothing")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=str(ROOT / "results" / "margin_relabel_check.json"))
    a = ap.parse_args()

    names = sorted(p.name for p in LABELS.glob("sample_*.tif"))

    if a.check:
        rng = np.random.default_rng(a.seed)
        pick = [names[i] for i in rng.choice(len(names), size=min(a.n, len(names)),
                                             replace=False)]
        rows = []
        for nm in pick:
            ct = tifffile.imread(IMAGES / nm)
            lab = tifffile.imread(LABELS / nm)
            if (lab == 1).sum() < 400:
                continue
            t0 = time.time()
            new, st = relabel_margin(ct, lab)
            row = {"sample": nm, "before": class_fracs(lab), "after": class_fracs(new),
                   **st, "seconds": round(time.time() - t0, 1)}
            # the transform must only ever move 0 -> 2, never touch class 1
            assert int((new == 1).sum()) == int((lab == 1).sum()), "surface voxels changed"
            assert int(((lab == 2) & (new != 2)).sum()) == 0, "ignore voxels changed"
            rows.append(row)
            print(f"  {nm}  bg {row['before']['0']:.4f}->{row['after']['0']:.4f}  "
                  f"ign {row['before']['2']:.4f}->{row['after']['2']:.4f}  "
                  f"relabelled {st['n_relabelled']:,} ({st['frac_of_sheet']:.2f}x sheet)  "
                  f"{row['seconds']}s", flush=True)

        d_bg = float(np.mean([r["before"]["0"] - r["after"]["0"] for r in rows]))
        summary = {
            "n_volumes": len(rows),
            "mean_background_fraction_removed": round(d_bg, 5),
            "mean_relabelled_per_volume": int(np.mean([r["n_relabelled"] for r in rows])),
            "mean_relabelled_over_sheet": round(
                float(np.mean([r["frac_of_sheet"] for r in rows])), 4),
            "invariants": "class 1 unchanged; class 2 never reduced; only 0 -> 2",
            "gate": ("preregistration abandon condition 2: the transform must measurably move "
                     "the margin class distribution on held-out volumes"),
            "rows": rows,
        }
        Path(a.report).write_text(json.dumps(summary, indent=1))
        print(f"\n  background fraction removed per volume: {d_bg:.5f}")
        print(f"  relabelled per volume: {summary['mean_relabelled_per_volume']:,} voxels "
              f"({summary['mean_relabelled_over_sheet']:.2f}x the sheet volume)")
        print(f"wrote {a.report}")
        return

    out_dir = Path(a.out or (ROOT / "data" / "kaggle" / "labels_margin"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = []
    t0 = time.time()
    for k, nm in enumerate(names):
        ct = tifffile.imread(IMAGES / nm)
        lab = tifffile.imread(LABELS / nm)
        new, st = relabel_margin(ct, lab)
        tifffile.imwrite(out_dir / nm, new.astype(lab.dtype))
        st["sample"] = nm
        stats.append(st)
        if k % 25 == 0:
            print(f"  [{k}/{len(names)}] {nm}  relabelled {st['n_relabelled']:,}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    Path(str(out_dir) + "_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote {out_dir}  ({len(stats)} volumes)")


if __name__ == "__main__":
    main()
