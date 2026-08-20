"""Is a papyrus sheet REALLY here? A model-free CT contrast test. ⭐ Leg 3 of the 3-way accept.

⛔⛔ FINAL, 2026-08-07: **THE LEG IS DEAD ON REAL DATA.** Registered repair floor was "per-volume
thresholds must keep >= 50% of labelled sheet". Achieved **11.2%** (one global threshold: 7.8%).
The rule was fixed before the run; it failed; it is not being repaired a third time.

  labelled sheet  median contrast 2.4  (q05 0.3, q95 14.5)
  background      median 1.0, **q99 9.6**
  per-volume recall at each volume's own background q99: 68.0 / 17.0 / 12.0 / 8.0 / 6.0 / 5.8 /
  5.0 / 4.2 / 2.8 / 2.5 / 2.0 / 1.5 %   -> mean 11.2%

⚠ TWO LIMITATIONS, STATED BUT **NOT** USED TO RESCUE THE RESULT. Both were visible only after the
failure, which is exactly when a limitation is least trustworthy, so neither is being acted on:

  CONTAMINATED NULL   "background" is `label == 0` at >= 6 vox from sheet-or-ignore, but in a
                      scroll almost everything IS papyrus, and 25-67% of these volumes is class 2
                      (`ignore`). So the negative class very likely contains unlabelled sheet,
                      which is why the background q99 (9.6) towers over its median (1.0). The
                      test may therefore be unfair rather than the leg bad -- **which means the
                      leg's true performance is UNKNOWN, not that it is fine.**
  UNIT OF ANALYSIS    this scores single voxels. A deployed accept would score a connected
                      component of predicted sheet, and averaging would sharpen it.

⭐ AND THE REASON BOTH ARE UNFIXABLE HERE IS THE MONTH'S RECURRING FINDING: **there is no
trustworthy negative class in the public label set**, just as there was no ground truth for the
five studies before this. The 892-volume set cannot tell us where sheet ISN'T any more than it
could tell us where the hard sheet IS.

⛔ CONSEQUENCE FOR THE AUGUST PLAN: the 3-way accept loses the only leg that consults the scan
independently. What remains is m7 AND `surface_recto_3dunet` -- two nnU-Nets on overlapping
labels, i.e. **"two correlated models agreed"**, which this file's own header calls weak evidence.
Do not ship the augmented label set as if it were 3-way verified.

---

RESULT (synthetic gate) 2026-08-07: **G2/G3/G4 PASSED, G1 FAILED, and the G1 failure is a finding.**

  G4 sigma      noise estimator within +3.0% .. +10.3% of truth              PASS
  G1 recovery   measured CNR ran -27.1% / -30.5% / -32.6% / -34.3% low       **FAIL**
  G2 null       no-sheet median 0.25, q99 0.59 vs 6.95 on sheet -> 11.8x     PASS
  G3 separation threshold at the no-sheet q99 recalls 100% of sheet points   PASS

**The statistic discriminates sheet from air superbly and is NOT a calibrated CNR.** Two
deterministic mechanisms, both measured rather than assumed:

  WINDOW TRUNCATION  at half-width +-4 a thick sheet (FWHM 6.0, sigma_s 2.55) never reaches true
                     background inside the window, so the baseline sits ~29% up the peak. At
                     sigma_g 0.25, where smoothing attenuation is 0.995, measured/true was still
                     **0.629** -- so this, not smoothing, is the dominant term.
  PACKING            a neighbouring wrap at pitch 9.5 drops FWHM-6.0 from 0.786 to **0.445**.
                     ⚠ This one is PHYSICS, not error: when wraps nearly merge the gap fills in
                     and the true local contrast really is lower.

⛔ **NO WINDOW CHOICE RESCUES G1** -- widening swaps the bias rather than removing it
(half=8: thin 0.674, thick 0.859, i.e. the trend inverts). `half=6, min` gives 0.667/0.757/0.786
and a single ~1.35x calibration constant WOULD make G1 pass. **That constant is not applied.**
Choosing it from the numbers that failed the gate is precisely what Amendment 3 did, and the
deviation is structured (thickness, packing) rather than a constant offset, so a constant would
be a fiction fitted to four points.

⭐ **THE DESIGN CONSEQUENCE, which is why this was worth finding.** The statistic is depressed in
thick and densely-packed regions -- **exactly the hard regions the augmented label set exists to
add.** A single global accept threshold would therefore admit easy sheet and reject hard sheet,
**reproducing the very bias that makes the current 892-volume set useless for this work**
(the earlier +0.008-vox population signed offset is withdrawn because its normal signs were not
physically oriented; median CNR 272, only 2.9% below 50). So the threshold **must be local or
stratified by packing**, never one number over the whole scroll. Named `sheet_contrast`, not
`cnr`, because it is a relative contrast against the local inter-sheet gap and nothing more.


THE AUGUST DELIVERABLE is a label set built by accepting high-confidence predictions that lie
outside the existing ground truth, on a **3-way** test:

    m7 says sheet   AND   surface_recto_3dunet independently says sheet   AND   >>> THIS <<<

The first two legs are neural and correlated — both are nnU-Nets trained on overlapping labels,
so agreeing with each other is weak evidence. **This leg is the only one that asks the scan
directly**, and it is the reason the dataset is defensible without @Jinhojeong's collaboration
(villa#193). If it does not hold up, the whole plan reduces to "two models agreed".

`37_2026_open_problems.md` calls label quality *"one of the main unwrapping bottlenecks"* and asks
for **label snapping**: *"using the raw CT signal and local geometry to move approximate labels
back onto the most plausible papyrus surface"*. `ridge_residual.py` measures WHERE the sheet is
(validated: recovers 0.502/1.001/2.181, lands 0.164 vox from truth, 0.000 window drift). This
measures WHETHER one is there at all. Same normals, same profile machinery, different question.

WHAT IS MEASURED, at a point p with across-sheet normal n:

    sheet_contrast(p) = ( peak - baseline ) / sigma_noise

  peak      max of the CT profile along n, within +-1.5 vox of p (a sheet must be AT p, not
            merely nearby -- otherwise every point next to a sheet passes)
  baseline  min of the profile over the full +-4 vox window. ⚠ NOT the profile flanks: at a
            wrap pitch of ~9.5 vox and sheet FWHM ~3.5, the flanks land on the NEIGHBOURING
            sheet in densely packed regions, which is exactly where the augmented labels are
            supposed to come from. The window minimum is the inter-sheet gap, which is right.
  sigma     per-volume noise from the MAD of first differences / sqrt(2) -- model-free, exact
            for white noise, and biased HIGH by real structure, which makes CNR conservative.

⛔ THE FAILURE MODE THIS MUST NOT HAVE. `max - min` over ~33 profile samples is **positive on
pure noise**. An estimator that reports CNR 4 on empty air will accept air. So the null is not a
footnote here, it is the whole gate, and it is measured rather than assumed -- the ridge metric's
first gate was circular and returned a suspiciously perfect 0.000, and that is the mistake this
file is written to not repeat.

REGISTERED GATES, fixed before running:
  G1 RECOVERY    on synthetic sheets of known CNR (amp/noise) in {5, 10, 20, 50}, the median
                 estimate is within 30% of truth
  G2 NULL        on synthetic volumes with NO sheet, q99 of the estimate is at least **3x below**
                 the median estimate on true-CNR-10 sheets. Scale-free on purpose: it does not
                 require guessing the null's absolute value in advance.
  G3 SEPARATION  thresholding at the no-sheet q99 recalls >= 90% of true-CNR-10 sheet points
  G4 SIGMA       the noise estimator recovers the true sigma within 25% (else G1 fails for the
                 wrong reason)

ABANDON: if G2 fails, this leg cannot distinguish sheet from air and the 3-way test collapses to
the two correlated neural legs. Say that plainly and stop; do not retune to rescue it.

⭐ REGISTERED FOLLOW-UP — THE LABEL-FREE THRESHOLD (written before the repaired calibration was
read, because it is the only honest moment to fix it).

⚠ THE DEPLOYMENT PROBLEM. A per-volume threshold taken from that volume's *background q99* needs
labels to compute. In use we accept candidates **outside** the ground truth, exactly where clean
background does not exist. So the threshold must come from the volume itself, unlabelled.

  CANDIDATE   tau(V) = percentile_p of `sheet_contrast` at N random points in V. Random points
              are overwhelmingly non-sheet, so a high percentile of that distribution should
              approximate the background q99 without ever consulting a label.
  ⚠ p IS A FREE PARAMETER and picking it after seeing the answer is a forking path. So: **p is
    chosen on a random HALF of the volumes and evaluated on the other half**, split by seed,
    reported both ways.

  REGISTERED PASS, on the held-out half:
    L1  Spearman(tau_labelfree, tau_labelderived) across volumes >= 0.70
    L2  substituting tau_labelfree costs <= 10 percentage points of pooled sheet recall vs the
        label-derived threshold
  FAIL either -> the leg cannot be deployed outside the GT, whatever its calibration looks like.
  Say so; a leg that only works where labels already exist is worth nothing to this dataset.

  python sheet_presence.py --validate
  python sheet_presence.py --calibrate      # reference + null on the real labelled volumes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates

from ridge_residual import IMAGES, LABELS, across_sheet_normals

ROOT = Path(__file__).resolve().parent

SIGMA = 1.0        # CT smoothing before the Hessian / profile, matching ridge_residual
HALF = 4.0         # profile half-width, voxels
STEP = 0.25
CENTRE_WIN = 1.5   # the peak must lie this close to the point to count as "sheet at p"


def noise_sigma_mad(ct: np.ndarray) -> float:
    """⛔ SUPERSEDED — kept because the calibration numbers in this file's header used it.

    MAD of first differences / sqrt(2). Exact for white noise, and it failed on real data for
    two independent mechanism-level reasons, both verified rather than suspected:

      QUANTISED  the CT is uint8, so first differences are integers and their MAD is an INTEGER.
                 Measured: exactly 2.0000, 12.0000, 5.0000 on samples 00047 / 00716 / 00831,
                 i.e. sigma quantised to steps of 1.0486. At sigma ~2 that is a ~25% error, and
                 it is what made sample_00047 report a sheet contrast of 28.4 against ~4 for
                 every comparable volume.
      STRUCTURE  with wraps every ~9.5 vox a large fraction of voxels sit on a sheet edge, so
                 the spread of first differences is dominated by sheet gradient, not noise.
    """
    d = np.diff(ct.astype(np.float32), axis=0).ravel()
    if d.size > 4_000_000:
        d = d[:: d.size // 4_000_000 + 1]
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def noise_sigma(ct: np.ndarray, block: int = 16, pct: float = 10.0) -> float:
    """Noise sigma from the QUIETEST blocks: low percentile of per-block standard deviations.

    Fixes both defects of the MAD estimator above. Structure raises a block's variance, so the
    low percentile of block sigmas is dominated by genuinely quiet regions -- inter-sheet gaps
    and uniform interior -- rather than by sheet edges. And a standard deviation over 4,096
    voxels is continuous, so the uint8 quantisation that pinned the MAD to integers is gone.

    ⚠ This rescales every point in a volume by the same factor, so it CANNOT change the
    sheet-vs-background separation WITHIN a volume. What it fixes is POOLING across volumes,
    which is where the real calibration actually broke.
    """
    a = ct.astype(np.float32)
    s = [n // block * block for n in a.shape]
    a = a[:s[0], :s[1], :s[2]]
    b = a.reshape(s[0] // block, block, s[1] // block, block, s[2] // block, block)
    b = b.transpose(0, 2, 4, 1, 3, 5).reshape(-1, block ** 3)
    return float(max(np.percentile(b.std(axis=1), pct), 1e-3))


def sheet_contrast(ct: np.ndarray, pts: np.ndarray, normals: np.ndarray | None = None,
                   half: float = HALF, step: float = STEP, sigma: float = SIGMA,
                   centre_win: float = CENTRE_WIN, sig_noise: float | None = None):
    """Local sheet contrast at each point, plus the offset of the peak from the point.

    Returns (contrast, peak_offset, normals). `peak_offset` is where along n the in-window maximum
    sits, so it can be cross-checked against `ridge_residual.ridge_offset` -- they should agree
    where both are defined.
    """
    n = across_sheet_normals(ct, pts, sigma) if normals is None else normals
    ts = np.arange(-half, half + 1e-9, step, dtype=np.float32)
    sm = gaussian_filter(ct.astype(np.float32), sigma)

    coords = pts[:, :, None].astype(np.float32) + n[:, :, None] * ts[None, None, :]
    prof = map_coordinates(sm, coords.transpose(1, 0, 2).reshape(3, -1),
                           order=1, mode="nearest").reshape(len(pts), len(ts))

    near = np.abs(ts) <= centre_win
    k = np.argmax(prof[:, near], axis=1)
    peak = prof[:, near][np.arange(len(pts)), k]
    peak_off = ts[near][k].astype(np.float64)
    baseline = prof.min(axis=1)

    s = noise_sigma(ct) if sig_noise is None else sig_noise
    cnr = (peak - baseline) / max(s, 1e-6)
    return cnr, peak_off, n


def _synth(shape=(96, 96, 96), tilt=(0.0, 0.0), thickness=3.5, amp=180.0, noise=8.0,
           bg=40.0, seed=0, sheet=True):
    """One sheet of known contrast at a known plane -- or no sheet at all (`sheet=False`)."""
    rng = np.random.default_rng(seed)
    n = np.array([1.0, np.tan(tilt[0]), np.tan(tilt[1])])
    n /= np.linalg.norm(n)
    c = (np.array(shape) - 1) / 2.0
    vol = np.full(shape, bg, dtype=np.float32)
    if sheet:
        zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
        sd = (zz - c[0]) * n[0] + (yy - c[1]) * n[1] + (xx - c[2]) * n[2]
        vol = vol + amp * np.exp(-0.5 * (sd / (thickness / 2.3548)) ** 2)
    vol = vol + rng.normal(0, noise, size=shape)
    # ⚠ NOT clipped to uint8: at CNR 50 (amp 180, noise 3.6) clipping at 255 would flatten the
    # peak and cap the measurable contrast, so the recovery arm would fail on a rendering
    # artefact rather than on the estimator. Real CT is uint8, but the gate tests the estimator.
    return vol.astype(np.float32), c, n


def validate(per_vol: int = 600, seed: int = 0) -> dict:
    """⛔ THE GATE. Synthetic sheets of KNOWN CNR, plus volumes with no sheet at all."""
    tilts = [(0.0, 0.0), (0.25, 0.0), (0.0, 0.35), (0.3, 0.25)]
    true_cnrs = [5.0, 10.0, 20.0, 50.0]
    amp = 180.0
    rng = np.random.default_rng(seed)

    # G4 first: does the noise estimator work at all?
    sig_rows = []
    for i, tc in enumerate(true_cnrs):
        v, _, _ = _synth(amp=amp, noise=amp / tc, seed=seed + i)
        sig_rows.append((amp / tc, noise_sigma(v)))
    sig_err = max(abs(e - t) / t for t, e in sig_rows)
    print("G4 NOISE SIGMA")
    for t, e in sig_rows:
        print(f"  true {t:6.2f}   estimated {e:6.2f}   ({abs(e-t)/t:+.1%})")
    g4 = bool(sig_err < 0.25)
    print(f"  worst error {sig_err:.1%}  (need <25%)   {'PASS' if g4 else 'FAIL'}\n")

    def points_on(c, n, k):
        b = c + rng.uniform(-20, 20, size=(k, 3))
        return (b - np.outer((b - c) @ n, n)).astype(np.float32)

    print("G1 RECOVERY — estimated CNR on sheets of known contrast")
    rec, ten = [], []
    for tc in true_cnrs:
        got = []
        for ti, tilt in enumerate(tilts):
            v, c, n = _synth(tilt=tilt, amp=amp, noise=amp / tc, seed=seed + ti)
            cnr, _, _ = sheet_contrast(v, points_on(c, n, per_vol))
            got.append(cnr)
            if tc == 10.0:
                ten.append(cnr)
        med = float(np.median(np.concatenate(got)))
        rec.append({"true": tc, "measured": med, "rel": abs(med - tc) / tc})
        print(f"  true CNR {tc:5.1f}   measured {med:7.2f}   ({(med-tc)/tc:+.1%})")
    g1 = bool(all(r["rel"] < 0.30 for r in rec))
    print(f"  {'PASS' if g1 else 'FAIL'}  (need every row within 30%)\n")

    print("G2 NULL — the SAME estimator on volumes containing NO sheet")
    null = []
    for ti, tilt in enumerate(tilts):
        v, c, n = _synth(tilt=tilt, amp=amp, noise=amp / 10.0, seed=seed + 100 + ti, sheet=False)
        # points and normals are whatever the Hessian says in pure noise -- exactly the
        # situation a false candidate presents
        b = (c + rng.uniform(-20, 20, size=(per_vol, 3))).astype(np.float32)
        cnr, _, _ = sheet_contrast(v, b)
        null.append(cnr)
    null = np.concatenate(null)
    ten = np.concatenate(ten)
    q99 = float(np.percentile(null, 99))
    med10 = float(np.median(ten))
    ratio = med10 / max(q99, 1e-9)
    print(f"  no-sheet CNR:  median {np.median(null):.2f}   q99 {q99:.2f}   max {null.max():.2f}")
    print(f"  true-CNR-10 sheet median {med10:.2f}")
    g2 = bool(ratio >= 3.0)
    print(f"  separation {ratio:.1f}x  (need >=3x)   {'PASS' if g2 else 'FAIL'}\n")

    print("G3 SEPARATION — threshold at the no-sheet q99")
    recall = float(np.mean(ten >= q99))
    g3 = bool(recall >= 0.90)
    print(f"  threshold {q99:.2f}   recalls {recall:.1%} of true-CNR-10 sheet points"
          f"  (need >=90%)   {'PASS' if g3 else 'FAIL'}")

    ok = g1 and g2 and g3 and g4
    print(f"\n  GATE {'PASSED' if ok else 'FAILED'}")
    if not ok:
        print("  ⛔ if G2 failed, the 3-way test collapses to two correlated neural legs.")
    return {"sigma_rows": [{"true": t, "est": e} for t, e in sig_rows], "recovery": rec,
            "null_q99": q99, "null_median": float(np.median(null)),
            "cnr10_median": med10, "separation": ratio, "recall_at_q99": recall,
            "g1_recovery": g1, "g2_null": g2, "g3_separation": g3, "g4_sigma": g4,
            "gate_passed": bool(ok)}


def calibrate(n_vol: int = 12, per_vol: int = 400, seed: int = 0) -> dict:
    """The REAL reference: CNR on labelled sheet, and on background far from any label.

    The acceptance threshold for the augmented label set comes from here -- a candidate must
    clear the background null AND sit inside the labelled distribution. ⚠ Background excludes
    anything near class 2 (`ignore`, ~59% of volume) as well as near class 1: `ignore` may well
    contain unlabelled sheet, so using it as a negative would poison the null.
    """
    rng = np.random.default_rng(seed)
    names = sorted(p.stem for p in LABELS.glob("sample_*.tif"))
    if not names:
        raise SystemExit(f"no labels under {LABELS}")
    pos, neg, rows = [], [], []
    for nm in [names[i] for i in rng.permutation(len(names))]:
        if len(rows) >= n_vol:
            break
        lab = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        sheet = lab == 1
        if sheet.sum() < per_vol:
            continue
        ct = np.asarray(tifffile.imread(str(IMAGES / f"{nm}.tif")))
        s = noise_sigma(ct)

        p = np.argwhere(sheet)
        p = p[rng.choice(len(p), per_vol, replace=False)].astype(np.float32)
        cp, _, _ = sheet_contrast(ct, p, sig_noise=s)

        far = distance_transform_edt(~(sheet | (lab == 2))) >= 6.0
        q = np.argwhere(far)
        cq = np.array([])
        if len(q) >= per_vol:
            q = q[rng.choice(len(q), per_vol, replace=False)].astype(np.float32)
            cq, _, _ = sheet_contrast(ct, q, sig_noise=s)
        pos.append(cp)
        if len(cq):
            neg.append(cq)
        # ⭐ PER-VOLUME threshold at this volume's OWN background q99. The pooled threshold kept
        # only 8.2% of labelled sheet, because each volume sits on a different scale; the
        # separation was there all along INSIDE every volume (sheet/bg ratio 1.3-16.8, median ~3).
        thr = float(np.percentile(cq, 99)) if len(cq) else float("nan")
        rec = float(np.mean(cp >= thr)) if len(cq) else float("nan")
        rows.append({"volume": nm, "sigma": s, "contrast_sheet_median": float(np.median(cp)),
                     "contrast_bg_median": float(np.median(cq)) if len(cq) else None,
                     "threshold": thr, "recall": rec, "n_bg": int(len(cq))})
        if len(cq):
            print(f"  {nm}: sigma {s:6.3f}  sheet {np.median(cp):7.2f}  bg {np.median(cq):6.2f}"
                  f"  ratio {np.median(cp)/max(np.median(cq),1e-9):5.1f}"
                  f"  thr {thr:6.2f}  recall {rec:6.1%}", flush=True)
        else:
            print(f"  {nm}: sigma {s:6.3f}  sheet {np.median(cp):7.2f}  bg n/a", flush=True)

    P = np.concatenate(pos)
    out = {"n_volumes": len(rows), "rows": rows,
           "sheet_contrast_median": float(np.median(P)),
           "sheet_contrast_q05": float(np.percentile(P, 5)),
           "sheet_contrast_q95": float(np.percentile(P, 95))}
    print(f"\nLABELLED SHEET   median CNR {out['sheet_contrast_median']:.1f}   "
          f"q05 {out['sheet_contrast_q05']:.1f}   q95 {out['sheet_contrast_q95']:.1f}")
    if neg:
        N = np.concatenate(neg)
        glob_thr = float(np.percentile(N, 99))
        glob_rec = float(np.mean(P >= glob_thr))
        # pooled recall under PER-VOLUME thresholds, weighting each volume by its sample count
        per = [r for r in rows if np.isfinite(r["recall"])]
        pool = float(np.mean([r["recall"] for r in per])) if per else float("nan")
        out.update({"bg_contrast_median": float(np.median(N)),
                    "bg_contrast_q99": glob_thr,
                    "global_threshold": glob_thr, "global_recall": glob_rec,
                    "per_volume_recall": pool,
                    "decision_floor": 0.50,
                    "leg_usable": bool(pool >= 0.50)})
        print(f"BACKGROUND       median {out['bg_contrast_median']:.1f}   q99 {glob_thr:.1f}")
        print(f"\n  ONE GLOBAL threshold {glob_thr:.1f}  ->  keeps {glob_rec:.1%} of labelled sheet")
        print(f"  PER-VOLUME thresholds     ->  keeps {pool:.1%}")
        verdict = ("PASS — the leg is usable with local thresholds" if pool >= 0.50
                   else "FAIL — the leg is dead")
        print(f"\n  REGISTERED FLOOR 50%:  {verdict}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--n-vol", type=int, default=12)
    ap.add_argument("--per-vol", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not (a.validate or a.calibrate):
        ap.error("pass --validate and/or --calibrate")
    res = {}
    if a.validate:
        res["validate"] = validate(per_vol=max(a.per_vol, 600))
    if a.calibrate:
        if a.validate:
            print("\n" + "=" * 70 + "\nCALIBRATION ON REAL LABELLED VOLUMES\n")
        res["calibrate"] = calibrate(a.n_vol, a.per_vol)
    dest = Path(a.out or (ROOT / "results" / "sheet_presence.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
