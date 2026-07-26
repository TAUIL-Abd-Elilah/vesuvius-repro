"""Does predicted ink land on material, or is the model firing on nothing?

The 3D ink model was trained on PHerc. Paris 4 alone. Pointed at another scroll it still
emits a probability everywhere, and it fires at 4x the Paris 4 rate on PHerc0332 and 13x
on PHerc0343P. A rate on its own cannot separate two very different situations:

  (a) there is genuinely more recoverable ink there, or
  (b) the model is out of its training domain and firing on texture.

The discriminator is physical rather than statistical. Ink is carbon laid ON papyrus, so
a true detection has to sit on the sheet - dense material in the CT - and not in the air
between sheets. A model responding to out-of-domain texture has no reason to respect that
geometry. So we ask how the ink probability varies with CT density.

WHY DENSITY AND NOT THE PUBLISHED SURFACE PREDICTION. The first version of this test used
the surface-m7 masks, since those exist for every scroll here. On PHerc. Paris 4 - the one
scroll that has to work, because it is the calibration case - the m7 mask does not track
papyrus: it covers 18.4% of the block with mean CT 64.1 inside versus 68.8 outside, and
P(dense|m7) = 0.187 against P(dense|not m7) = 0.199. Paris 4's other prediction,
surface-recto-2um at L0, behaves the way a sheet mask should on the same block - CT 85.1
inside versus 63.3 outside, P(dense) 0.345 versus 0.157 - but it exists for exactly one
scroll.

That failure is specific to Paris 4 and is NOT a general statement about m7: checked on a
second and third scroll (check_surface_alignment.py), the m7 mask does select denser
material, by +4.6 CT on PHerc0332 and +28.5 on PHerc0343P at the same x4 mapping. Paris 4
being the odd one out is consistent with it already being the exception elsewhere in this
project - it is the only scroll of 36 that needs TTA to reproduce.

Either way m7 cannot be the reference here, because the reference has to hold on the
calibration scroll. CT density is used instead: it is what the ink model was actually
shown, it needs no second model, and it exists for every scroll. Where the recto mask
exists it is reported too, as a check against a real sheet mask.

Nothing here needs a hand-picked threshold. The headline is a profile: ink rate binned by
CT decile. Real ink rises with density. A texture artifact does not have to.

  ct_profile      mean ink probability and hot fraction per CT decile
  dense_lift      P(ink | top CT decile) / P(ink | bottom half of CT)
  shifted null    the same statistic after rolling the ink volume 64 voxels against the
                  CT. Marginals are preserved exactly and only the alignment is destroyed,
                  so it is what the lift reads if ink and material were unrelated. The
                  claim is the gap between the two, never the raw number.

TWO TESTS WERE TRIED AND NEITHER SEPARATES REAL INK FROM OVER-FIRING. Both are kept,
because the numbers are still worth reporting and because the reason each fails is the
useful part.
  * on-material (CT decile): every case passes it, including ones that look like
    saturation, because the model keys on density. Necessary, not sufficient.
  * depth into material (shell vs core): reads 0.64x on the published Paris 4 artifact,
    where the ink is real, and 2.21x on PHerc0343P, the most likely over-firing case.
    Reading it the obvious way inverts the calibration case, so it is reported as a
    descriptive number with no verdict attached. See depth_profile().
Anything built on top of this file needs a third idea, not a re-reading of these two.

CALIBRATION, NOT EYEBALLING. The published Paris 4 ink artifact goes through the identical
measurement first. It is the one place in the collection where the ink is known to be
real, so it defines what a true positive scores. Our own reproduction of it is measured
too, as a check that the measurement does not depend on the pipeline. A scroll is only
interesting if it reaches the published artifact's regime.

Usage:
    python ink_structure.py --case paris4_published
    python ink_structure.py --all
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt

warnings.filterwarnings("ignore")
BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
TRIM = 64
SIZE = 256
L2_TO_L0 = 4

PARIS4_INK_PUB = ("PHercParis4/representations/predictions/ink-3d/"
                  "20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr")
PARIS4_REGION = (34102, 16346, 16346)
PARIS4_RECTO = ("PHercParis4/representations/predictions/surfaces/"
                "20260411134726-surface-20260413141734-surface-recto-2um-ps256-L0-th0.45.zarr")


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fetch(url: str, z: int, y: int, x: int, n: int) -> np.ndarray:
    a = zarr.open(url, mode="r")
    return retry(lambda: np.asarray(a[z:z + n, y:y + n, x:x + n]))


def lift(hot: np.ndarray, ct: np.ndarray, hi_thr: float, lo_thr: float) -> dict:
    hi, lo = ct >= hi_thr, ct <= lo_thr
    if hi.sum() == 0 or lo.sum() == 0:
        return {"undefined": "CT has no spread in this block"}
    p_hi, p_lo = float(hot[hi].mean()), float(hot[lo].mean())
    return {"p_ink_dense": p_hi, "p_ink_sparse": p_lo,
            "dense_lift": float(p_hi / p_lo) if p_lo > 0 else float("inf"),
            "frac_of_ink_in_dense": (float(hot[hi].sum() / hot.sum())
                                     if hot.sum() > 0 else 0.0)}


def measure(ink: np.ndarray, ct: np.ndarray, thr: float = 0.5) -> dict:
    out: dict = {"ink_thresh": thr, "ink_mean": float(ink.mean()),
                 "ink_std": float(ink.std()), "ink_max": float(ink.max()),
                 "ink_frac_hot": float((ink > thr).mean()),
                 "ct_mean": float(ct.mean()), "ct_std": float(ct.std())}

    # A constant output is a failed run, not a finding of "no ink". Catch it here so it
    # can never be reported as a measurement.
    if out["ink_std"] < 1e-6:
        out["status"] = "degenerate_constant_output"
        out["note"] = (f"ink probability is constant at {out['ink_mean']:.4f} across the "
                       "whole block, so the model produced no output here. This is a "
                       "failed run, NOT evidence that the scroll has no ink.")
        return out
    out["status"] = "ok"

    hot = ink > thr
    hi_thr = float(np.quantile(ct, 0.9))
    lo_thr = float(np.quantile(ct, 0.5))
    out["ct_p90"], out["ct_p50"] = hi_thr, lo_thr
    out["aligned"] = lift(hot, ct, hi_thr, lo_thr)

    rolled = np.roll(hot, shift=(TRIM, TRIM, TRIM), axis=(0, 1, 2))
    out["shifted_null"] = lift(rolled, ct, hi_thr, lo_thr)

    a = out["aligned"].get("dense_lift")
    b = out["shifted_null"].get("dense_lift")
    if isinstance(a, float) and isinstance(b, float) and b > 0:
        out["lift_over_null"] = float(a / b)

    ctf, inkf = ct.ravel().astype(np.float64), ink.ravel().astype(np.float64)
    if ctf.std() > 0 and inkf.std() > 0:
        out["pearson_ink_vs_ct"] = float(np.corrcoef(ctf, inkf)[0, 1])

    edges = np.quantile(ct, np.linspace(0, 1, 11))
    prof = []
    for i in range(10):
        lo_e, hi_e = edges[i], edges[i + 1]
        m = (ct >= lo_e) & (ct <= hi_e) if i == 9 else (ct >= lo_e) & (ct < hi_e)
        if m.sum() == 0:
            continue
        prof.append({"decile": i + 1, "ct_lo": float(lo_e), "ct_hi": float(hi_e),
                     "n_voxels": int(m.sum()), "ink_mean": float(ink[m].mean()),
                     "frac_hot": float(hot[m].mean())})
    out["ct_profile"] = prof
    out.update(depth_profile(ink, ct, hot))
    return out


def depth_profile(ink: np.ndarray, ct: np.ndarray, hot: np.ndarray) -> dict:
    """Ink rate against depth INTO the papyrus - the test density cannot do.

    Every case passes the on-material test, because the ink model keys on density: real
    ink and an over-firing model both put their output on the sheet and neither puts it in
    the air. So that test separates nothing.

    Depth does separate them. Ink is carbon laid ON the face of a sheet, so real ink has
    to sit in the first few voxels of material and fall away into the body. A model
    responding to the bulk texture of papyrus has no reason to prefer the face, and will
    fire through the full thickness. At ~2.4 um a sheet is roughly 40-80 voxels thick, so
    there is plenty of interior for the two behaviours to disagree in.

    Material is split from air by Otsu on the block's own CT histogram rather than a fixed
    cut, so the same code works on scrolls with different exposure.
    """
    out: dict = {}
    try:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(ct))
    except Exception:  # noqa: BLE001
        thr = float(np.quantile(ct, 0.5))
        out["depth_threshold_note"] = "skimage unavailable; used the CT median"
    material = ct >= thr
    out["material_threshold"] = thr
    out["material_frac"] = float(material.mean())
    if not material.any() or material.all():
        out["depth_profile"] = []
        return out

    depth = distance_transform_edt(material)  # voxels to the nearest air voxel
    bins = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10 ** 6)]
    prof = []
    for lo, hi in bins:
        m = (depth >= lo) & (depth <= hi)
        if m.sum() == 0:
            continue
        prof.append({"depth_lo": lo, "depth_hi": None if hi > 10 ** 5 else hi,
                     "n_voxels": int(m.sum()), "ink_mean": float(ink[m].mean()),
                     "frac_hot": float(hot[m].mean())})
    out["depth_profile"] = prof

    # One number: surface shell (<=4 voxels in) against the interior (>8).
    #
    # THIS METRIC DOES NOT DISCRIMINATE - measured, not assumed. On the published Paris 4
    # artifact, where the ink is real, it reads 0.64x: real ink is NOT concentrated at the
    # sheet face by this measure, and its rate actually rises again at depth 9-16.
    # PHerc0343P, the case most likely to be saturating, reads 2.21x. So a high value is
    # not evidence of real ink and a low value is not evidence against it - reading it the
    # obvious way inverts the calibration case. It is kept as a descriptive number only.
    #
    # The likely reason is that "depth into material" is not "depth into a sheet": in a
    # rolled scroll adjacent sheets touch, so Otsu's material blobs merge across them and
    # the deep population is pressed-together sheets rather than sheet interior.
    shell = (depth >= 1) & (depth <= 4)
    core = depth > 8
    if shell.sum() and core.sum():
        s, c = float(hot[shell].mean()), float(hot[core].mean())
        out["surface_vs_core"] = {
            "p_ink_shell": s, "p_ink_core": c,
            "shell_over_core": (float(s / c) if c > 0 else
                                (float("inf") if s > 0 else None)),
            "calibration": {"paris4_published_real_ink": 0.64, "paris4_repro": 0.72},
            "does_it_discriminate": "NO - see the comment in depth_profile()",
        }
        if hot.sum() == 0:
            out["surface_vs_core"]["note"] = "no voxel exceeds the threshold; nothing to locate"
    return out


def add_recto(rep: dict, ink: np.ndarray, origin: tuple[int, int, int]) -> None:
    """Paris 4 only: cross-check the density proxy against a real sheet mask."""
    try:
        m = fetch(f"{BUCKET}/{PARIS4_RECTO}/0", *origin, ink.shape[0]) > 0
    except Exception as exc:  # noqa: BLE001
        rep["recto_check"] = {"failed": f"{type(exc).__name__}: {exc}"}
        return
    if not m.any() or m.all():
        rep["recto_check"] = {"failed": "recto mask degenerate over this block"}
        return
    hot = ink > 0.5
    p_on, p_off = float(hot[m].mean()), float(hot[~m].mean())
    rolled = np.roll(hot, shift=(TRIM, TRIM, TRIM), axis=(0, 1, 2))
    rep["recto_check"] = {
        "recto_coverage": float(m.mean()),
        "p_ink_on_sheet": p_on, "p_ink_off_sheet": p_off,
        "sheet_lift": float(p_on / p_off) if p_off > 0 else float("inf"),
        "frac_of_ink_on_sheet": float(hot[m].sum() / hot.sum()) if hot.sum() else 0.0,
        "shifted_null_sheet_lift": (float(rolled[m].mean() / rolled[~m].mean())
                                    if rolled[~m].mean() > 0 else float("inf")),
    }


def build_cases(catalog: list[dict], results_dir: Path, work: Path) -> dict:
    def ct_url_for(scroll: str) -> str:
        r = [x for x in catalog if x.get("scroll") == scroll
             and x.get("status") == "verifiable" and "m7" in x.get("model", "")
             and x.get("declared_level") == 2][0]
        return f"{BUCKET}/{scroll}/volumes/{r['ct_volume']}/0"

    n = SIZE - 2 * TRIM

    def published():
        z, y, x = (c + TRIM for c in PARIS4_REGION)
        ink = fetch(f"{BUCKET}/{PARIS4_INK_PUB}/0", z, y, x, n).astype(np.float32) / 255.0
        ct = fetch(ct_url_for("PHercParis4"), z, y, x, n).astype(np.float32)
        return ink, ct, (z, y, x), True

    def from_merged(merged: Path, region, scroll: str):
        z, y, x = (c + TRIM for c in region)
        a = zarr.open(str(merged), mode="r")
        ink = sigmoid(np.asarray(a[0, z:z + n, y:y + n, x:x + n]).astype(np.float32))
        ct = fetch(ct_url_for(scroll), z, y, x, n).astype(np.float32)
        return ink, ct, (z, y, x), scroll == "PHercParis4"

    cases = {
        "paris4_published": published,
        "paris4_repro": lambda: from_merged(
            work / "ink_repro" / "merged.zarr", PARIS4_REGION, "PHercParis4"),
    }
    for jf in sorted(results_dir.glob("*_ink3d.json")):
        rep = json.loads(jf.read_text())
        if rep.get("status") != "ok":
            continue
        s = rep["scroll"]
        merged = work / f"ink_{s}" / "merged.zarr"
        if not merged.exists():
            continue
        b = rep["bbox"]
        cases[s] = (lambda m=merged, rg=(b[0], b[2], b[4]), sc=s:
                    from_merged(m, rg, sc))
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--results", default="results/ink")
    ap.add_argument("--work", default="outputs")
    ap.add_argument("--out", default="results/ink_structure")
    args = ap.parse_args()

    catalog = json.load(open(args.catalog))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = build_cases(catalog, Path(args.results), Path(args.work))

    if args.all:
        names = list(cases)
    elif args.case:
        if args.case not in cases:
            raise SystemExit(f"unknown case {args.case}; have: {', '.join(cases)}")
        names = [args.case]
    else:
        raise SystemExit(f"pass --case NAME or --all; have: {', '.join(cases)}")

    print(f"{len(names)} case(s): {', '.join(names)}\n", flush=True)
    for name in names:
        print(f"=== {name} ===", flush=True)
        try:
            ink, ct, origin, is_paris4 = cases[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {type(exc).__name__}: {exc}\n", flush=True)
            continue

        rep = measure(ink, ct)
        rep["case"] = name
        rep["origin_l0"] = list(origin)
        rep["n"] = int(ink.shape[0])
        rep["reference"] = "CT density at L0 (what the ink model was shown)"

        if rep["status"] != "ok":
            print(f"  {rep['note']}\n", flush=True)
        else:
            if is_paris4:
                add_recto(rep, ink, origin)
            a, nl = rep["aligned"], rep["shifted_null"]
            print(f"  ink>0.5 {100*rep['ink_frac_hot']:.3f}% of voxels   "
                  f"CT mean {rep['ct_mean']:.1f}")
            print(f"  P(ink|dense) {a.get('p_ink_dense', 0):.4f}   "
                  f"P(ink|sparse) {a.get('p_ink_sparse', 0):.4f}")
            print(f"  DENSE LIFT {a.get('dense_lift', float('nan')):.2f}x   "
                  f"(shifted null {nl.get('dense_lift', float('nan')):.2f}x)")
            print(f"  pearson(ink, CT) {rep.get('pearson_ink_vs_ct', float('nan')):+.4f}")
            for r in rep["ct_profile"]:
                print(f"    decile {r['decile']:>2}  CT {r['ct_lo']:>5.0f}-{r['ct_hi']:>5.0f}  "
                      f"ink_mean {r['ink_mean']:.4f}  hot {100*r['frac_hot']:7.3f}%")
            if rep.get("depth_profile"):
                print(f"  depth into material (Otsu {rep['material_threshold']:.0f}, "
                      f"material {100*rep['material_frac']:.1f}%):")
                for r in rep["depth_profile"]:
                    hi = "+" if r["depth_hi"] is None else f"-{r['depth_hi']}"
                    print(f"    depth {r['depth_lo']}{hi:<4} n={r['n_voxels']:>8}  "
                          f"hot {100*r['frac_hot']:7.3f}%")
                sc = rep.get("surface_vs_core")
                if sc and sc["shell_over_core"] is not None:
                    print(f"  shell/core {sc['shell_over_core']:.2f}x  "
                          f"(real ink reads 0.64x - this does NOT discriminate)")
            rc = rep.get("recto_check")
            if rc and "failed" not in rc:
                print(f"  recto sheet mask: lift {rc['sheet_lift']:.2f}x "
                      f"(null {rc['shifted_null_sheet_lift']:.2f}x), "
                      f"{100*rc['frac_of_ink_on_sheet']:.1f}% of ink on sheet")

        dest = out / f"{name}.json"
        dest.write_text(json.dumps(rep, indent=2))
        print(f"  wrote {dest}\n", flush=True)


if __name__ == "__main__":
    main()
