"""Signed offset from a point to the CT sheet ridge, along the across-sheet normal.

Phase 1 of the §V roadmap, and the shared instrument for both tracks: it measures a mesh
vertex's snap residual (scrollfiesta_public#10) and a label voxel's misplacement (villa#193)
with the same code. Label-free and mesh-free — it only ever asks the CT where the sheet is,
so it cannot reward a geometry for agreeing with whatever produced it.

⚠ WHY RIDGE POSITION AND NOT A WIDTH. The obvious residual compares sheet widths, and that is
not a measurement. Measured on 10 volumes, the predicted/CT FWHM ratio moved

    half-width   +-3.0   +-4.0   +-6.0   +-8.0
    ratio         0.94    0.88    0.77    0.62

purely as a function of the profile window, because a wider window keeps lowering the CT
profile's baseline and therefore the half-max threshold. A width is a free parameter of the
window. A ridge is a location, and `validate()` below checks that claim rather than asserting
it — it is stated publicly in both issues, so it had better hold.

SIGN CONVENTION. Positive means the ridge sits on the +normal side of the point, i.e. the ridge
is at `p + offset * n` for the returned offset. So a snap should move a vertex by
`+offset` along `n`, and a perfect snap returns 0. Hessian eigenvector sign is arbitrary:
`offset*n` and the landing point are valid per point, but population signed offsets may only be
aggregated after every normal is tied to the same independent physical direction.

  python ridge_residual.py --validate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, map_coordinates

ROOT = Path(__file__).resolve().parent
IMAGES = ROOT / "data" / "kaggle" / "images"
LABELS = ROOT / "data" / "kaggle" / "labels"

SIGMA = 1.0        # CT smoothing before the Hessian
HALF = 4.0         # profile half-width, voxels
STEP = 0.25        # profile sampling step, voxels


def across_sheet_normals(ct: np.ndarray, pts: np.ndarray,
                         sigma: float = SIGMA, chunk: int = 200_000) -> np.ndarray:
    """Unit across-sheet vectors: most-negative-curvature Hessian eigenvector of the CT."""
    sm = gaussian_filter(ct.astype(np.float32), sigma)
    g = np.gradient(sm)
    gg = [np.gradient(g[i]) for i in range(3)]
    out = np.empty((len(pts), 3), dtype=np.float32)
    idx = np.rint(pts).astype(np.int64)
    for a in range(3):
        np.clip(idx[:, a], 0, ct.shape[a] - 1, out=idx[:, a])
    for s in range(0, len(pts), chunk):
        q = idx[s:s + chunk]
        H = np.empty((len(q), 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                H[:, i, j] = gg[i][j][q[:, 0], q[:, 1], q[:, 2]]
        H = 0.5 * (H + np.transpose(H, (0, 2, 1)))
        _, v = np.linalg.eigh(H)
        d = v[:, :, 0]
        out[s:s + chunk] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
    return out


def ridge_offset(ct: np.ndarray, pts: np.ndarray, normals: np.ndarray | None = None,
                 half: float = HALF, step: float = STEP,
                 sigma: float = SIGMA) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance from each point to the CT sheet ridge along its normal.

    Sub-voxel by parabolic interpolation about the sampled maximum: a residual quantised to
    `step` would be useless for tuning a snap whose whole job is sub-voxel placement.
    Returns (offset, normals). Offset is NaN where the maximum sits on the window edge, i.e.
    where the ridge is not bracketed and the answer would be a guess.
    """
    n = across_sheet_normals(ct, pts, sigma) if normals is None else normals
    ts = np.arange(-half, half + 1e-9, step, dtype=np.float32)
    sm = gaussian_filter(ct.astype(np.float32), sigma)

    coords = pts[:, :, None].astype(np.float32) + n[:, :, None] * ts[None, None, :]
    prof = map_coordinates(sm, coords.transpose(1, 0, 2).reshape(3, -1),
                           order=1, mode="nearest").reshape(len(pts), len(ts))

    k = np.argmax(prof, axis=1)
    off = ts[k].astype(np.float64)
    edge = (k == 0) | (k == len(ts) - 1)

    # parabolic refinement through the three samples about the peak
    ok = ~edge
    if ok.any():
        i = k[ok]
        r = np.arange(len(pts))[ok]
        y0, y1, y2 = prof[r, i - 1], prof[r, i], prof[r, i + 1]
        den = (y0 - 2 * y1 + y2)
        shift = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
        off[ok] = ts[i] + np.clip(shift, -1, 1) * step
    off[edge] = np.nan
    return off, n


def _sample_sheet_points(n_vol: int, per_vol: int, seed: int):
    """Labelled sheet voxels, which should already sit near the CT ridge."""
    rng = np.random.default_rng(seed)
    names = sorted(p.stem for p in LABELS.glob("sample_*.tif"))
    picked = [names[i] for i in rng.choice(len(names), size=n_vol * 3, replace=False)]
    out = []
    for nm in picked:
        if len(out) >= n_vol:
            break
        lab = np.asarray(tifffile.imread(str(LABELS / f"{nm}.tif")))
        pts = np.argwhere(lab == 1)
        if len(pts) < per_vol:
            continue
        ct = np.asarray(tifffile.imread(str(IMAGES / f"{nm}.tif")))
        sel = pts[rng.choice(len(pts), size=per_vol, replace=False)].astype(np.float32)
        out.append((nm, ct, sel))
    return out


def synthetic_sheet(shape=(96, 96, 96), tilt=(0.0, 0.0), thickness=3.5,
                    amp=180.0, bg=40.0, noise=8.0, seed=0):
    """A volume containing ONE sheet whose true plane is known exactly.

    Gaussian cross-section of the given FWHM, plane through the volume centre with the given
    tilt. Returns (volume, point_on_plane, unit_normal).
    """
    rng = np.random.default_rng(seed)
    n = np.array([1.0, np.tan(tilt[0]), np.tan(tilt[1])])
    n /= np.linalg.norm(n)
    c = (np.array(shape) - 1) / 2.0
    zz, yy, xx = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in shape], indexing="ij")
    sd = (zz - c[0]) * n[0] + (yy - c[1]) * n[1] + (xx - c[2]) * n[2]
    sig = thickness / 2.3548                      # FWHM -> sigma
    vol = bg + amp * np.exp(-0.5 * (sd / sig) ** 2)
    vol = vol + rng.normal(0, noise, size=shape)
    return np.clip(vol, 0, 255).astype(np.uint8), c, n


def validate(n_vol: int = 6, per_vol: int = 600, seed: int = 0) -> dict:
    """⛔ THE PHASE-1 GATE — on SYNTHETIC sheets whose true position is known.

    ⚠ An earlier version of this gate was CIRCULAR and is kept in the git history as a warning.
    It took real CT, moved known-good points by d along their normal, and checked that the
    measured offset changed by -d. That is true BY ALGEBRA for any ridge-finder whatsoever:
    both profiles sample the same line, so the difference is exactly -d regardless of whether
    the located "ridge" has anything to do with a sheet. It returned +0.000 error at every
    displacement and every window, which is what gave it away -- real measurements are not that
    clean.

    The honest test needs ground truth the estimator cannot see. So: build a sheet at a known
    plane, sample points at known signed distances from it, and require the metric to return
    those distances. Two arms:

      RECOVERY   over +-2 vox and four tilts. Recovered offset must match the true one.
      WINDOW     the same at half-widths 3/4/6/8. Recovery must NOT drift with the window --
                 the claim made publicly in scrollfiesta_public#10 and villa#193, against an
                 FWHM ratio that drifted 0.94 -> 0.62 over the same sweep.
    """
    tilts = [(0.0, 0.0), (0.25, 0.0), (0.0, 0.35), (0.3, 0.25)]
    offsets = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    rng = np.random.default_rng(seed)
    rows, win_rows = [], []

    for ti, tilt in enumerate(tilts):
        vol, c, n = synthetic_sheet(tilt=tilt, seed=seed + ti)
        # points on the plane, away from the volume edges, then pushed off by a known amount
        base = c + rng.uniform(-20, 20, size=(per_vol, 3))
        base -= np.outer((base - c) @ n, n)        # project exactly onto the true plane
        for d in offsets:
            pts = (base + n * d).astype(np.float32)
            off, nn = ridge_offset(vol, pts)
            good = ~np.isnan(off)
            # ⚠ CONVENTION-FREE CHECK. eigh returns eigenvectors with ARBITRARY SIGN, so the
            # sign of `off` is only meaningful against the normal returned beside it. Testing
            # `off == -d` therefore tests the eigensolver's sign convention, not the metric.
            # What matters for a snap is where the correction LANDS: p + off*n must sit on the
            # true plane. That is what is measured here, as signed distance to that plane.
            landed = pts + nn * off[:, None]
            resid = (landed - c) @ n
            rows.append({"tilt": ti, "applied": d,
                         "measured_magnitude": float(np.nanmedian(np.abs(off[good]))),
                         "error": float(np.nanmedian(np.abs(resid[good])))})
        for h in (3.0, 4.0, 6.0, 8.0):
            pts = (base + n * 1.0).astype(np.float32)
            off, nn = ridge_offset(vol, pts, half=h)
            g = ~np.isnan(off)
            landed = pts + nn * off[:, None]
            win_rows.append({"tilt": ti, "half": h,
                             "measured": float(np.nanmedian(np.abs(off[g]))),
                             "residual": float(np.nanmedian(np.abs(((landed - c) @ n)[g])))})

    print(f"synthetic sheets: {len(tilts)} tilts x {per_vol} points, "
          f"true FWHM 3.5 vox, noise sigma 8\n")
    print("RECOVERY — |offset| recovered, and where the correction LANDS vs the true plane")
    ok = True
    for d in offsets:
        m = float(np.median([r["measured_magnitude"] for r in rows if r["applied"] == d]))
        e = float(np.median([r["error"] for r in rows if r["applied"] == d]))
        bad = e > 0.25
        ok &= not bad
        print(f"  true |offset| {abs(d):.1f}   recovered {m:.3f}   "
              f"lands {e:.3f} vox off the true plane{'   <-- FAILS' if bad else ''}")

    print("\nWINDOW STABILITY — a point 1.0 vox off the true plane, per half-width:")
    for h in (3.0, 4.0, 6.0, 8.0):
        mm = float(np.median([r["measured"] for r in win_rows if r["half"] == h]))
        rr = float(np.median([r["residual"] for r in win_rows if r["half"] == h]))
        print(f"  half {h:.0f}   recovered {mm:.3f}   lands {rr:.3f} off plane")
    ref = np.median([r["measured"] for r in win_rows if r["half"] == 4.0])
    spread = max(abs(np.median([r["measured"] for r in win_rows if r["half"] == h]) - ref)
                 for h in (3.0, 6.0, 8.0))
    print(f"  max drift from the +-4 reference: {spread:.3f} vox   "
          f"(the FWHM ratio drifted 0.94 -> 0.62 over this same sweep)")

    print(f"\n  GATE {'PASSED' if ok and spread < 0.25 else 'FAILED'}")
    return {"recovery": rows, "window": win_rows,
            "gate_passed": bool(ok and spread < 0.25),
            "note": "synthetic ground truth; the earlier real-CT self-displacement gate was "
                    "circular and returned 0.000 by algebra"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n-vol", type=int, default=6)
    ap.add_argument("--per-vol", type=int, default=600)
    ap.add_argument("--out", default=str(ROOT / "results" / "ridge_residual_validation.json"))
    a = ap.parse_args()
    if not a.validate:
        ap.error("pass --validate")
    res = validate(a.n_vol, a.per_vol)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
