"""Does metric distortion of a tifxyz patch separate Will Stevens' good patches from his bad?

First piece of the meshing work. Will marks 9,171 of 56,968 Scroll 4 patches as problems and
says *"patches stop growing when they encounter areas of high stress — often caused by messy
surface predictions"*. "Stress" in a parameterisation is **metric distortion**, and a tifxyz
patch is a regular (u,v) grid of 3-D points, so the distortion is computable from the patch
geometry alone — no CT, no model, no tracer.

⚠ WE ALREADY KNOW PLACEMENT DOES NOT SEPARATE THEM. The CT-ridge residual gave AUC 0.512
(p=0.38) on 120 of each: his bad patches sit on the papyrus exactly as well as his good ones. So
whatever makes a patch bad is not that it drifted off the sheet, and geometry of the patch
itself is the next place to look.

WHAT IS MEASURED. For each interior grid node, the discrete first fundamental form of the
embedding, from the two grid tangents Pu = dP/du and Pv = dP/dv:

    E = <Pu,Pu>    F = <Pu,Pv>    G = <Pv,Pv>

from which three classical distortion measures, each scale-aware in a different way:

  stretch      sqrt(lambda_max / lambda_min) of [[E,F],[F,G]] — anisotropy. 1 = isometric.
  shear        |F| / sqrt(E*G) — departure from orthogonality. 0 = conformal.
  area_cv      coefficient of variation of sqrt(EG - F^2) across the patch — how unevenly the
               parameterisation stretches area. A clean patch samples the surface uniformly.

⚠ `-1` marks invalid nodes in tifxyz and must be excluded before any difference is taken, or a
missing neighbour manufactures an enormous tangent and the statistic becomes noise.

  python patch_distortion.py --n 150
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parent
WILL = ROOT / "_will"


def patch_grid(z: zipfile.ZipFile, base: str):
    """(H,W,3) grid of world points and a validity mask, or None."""
    try:
        xs = tifffile.imread(io.BytesIO(z.read(f"{base}/x.tif")))
        ys = tifffile.imread(io.BytesIO(z.read(f"{base}/y.tif")))
        zs = tifffile.imread(io.BytesIO(z.read(f"{base}/z.tif")))
    except KeyError:
        return None
    P = np.stack([zs, ys, xs], axis=-1).astype(np.float64)
    valid = (xs > 0) & (ys > 0) & (zs > 0)
    return P, valid


def distortion(P: np.ndarray, valid: np.ndarray) -> dict | None:
    """Stretch, shear and area-CV over the interior nodes with all four neighbours valid."""
    # central differences; a node is usable only if both neighbours on both axes are valid
    ok = (valid[2:, 1:-1] & valid[:-2, 1:-1] & valid[1:-1, 2:] & valid[1:-1, :-2]
          & valid[1:-1, 1:-1])
    if ok.sum() < 200:
        return None
    Pu = 0.5 * (P[2:, 1:-1] - P[:-2, 1:-1])
    Pv = 0.5 * (P[1:-1, 2:] - P[1:-1, :-2])
    Pu, Pv = Pu[ok], Pv[ok]

    E = np.einsum("ij,ij->i", Pu, Pu)
    F = np.einsum("ij,ij->i", Pu, Pv)
    G = np.einsum("ij,ij->i", Pv, Pv)
    good = (E > 1e-9) & (G > 1e-9)
    E, F, G = E[good], F[good], G[good]
    if len(E) < 200:
        return None

    tr, det = E + G, E * G - F * F
    disc = np.maximum(tr * tr / 4.0 - det, 0.0)
    lmax = tr / 2.0 + np.sqrt(disc)
    lmin = np.maximum(tr / 2.0 - np.sqrt(disc), 1e-12)
    stretch = np.sqrt(lmax / lmin)
    shear = np.abs(F) / np.sqrt(E * G)
    area = np.sqrt(np.maximum(det, 0.0))
    area_cv = float(area.std() / max(area.mean(), 1e-9))

    return {"stretch_med": float(np.median(stretch)),
            "stretch_q90": float(np.quantile(stretch, 0.90)),
            "shear_med": float(np.median(shear)),
            "shear_q90": float(np.quantile(shear, 0.90)),
            "area_cv": area_cv,
            "n_nodes": int(len(E))}


def collect(zip_path: Path, prefix: str, want: int, seed: int, zlo: float, zhi: float):
    z = zipfile.ZipFile(zip_path)
    names = sorted({n.split("/")[1] for n in z.namelist()
                    if n.startswith(prefix + "/") and len(n.split("/")) > 2})
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(names)):
        if len(out) >= want:
            break
        nm = names[i]
        try:
            meta = json.loads(z.read(f"{prefix}/{nm}/meta.json"))
        except KeyError:
            continue
        b = meta["bbox"]
        cz = 0.5 * (b[0][2] + b[1][2])
        if not (zlo <= cz <= zhi):
            continue
        g = patch_grid(z, f"{prefix}/{nm}")
        if g is None:
            continue
        d = distortion(*g)
        if d is None:
            continue
        d["patch"] = nm
        d["z"] = float(cz)
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zlo", type=float, default=3000.0)
    ap.add_argument("--zhi", type=float, default=8000.0)
    ap.add_argument("--out", default=str(ROOT / "results" / "patch_distortion.json"))
    a = ap.parse_args()

    res = {}
    for tag, zp, prefix in (("bad", WILL / "s4_bad_patches.zip", "s4_bad_patches"),
                            ("good", WILL / "s4_good_patches.zip", "s4_good_patches")):
        if not zp.exists():
            print(f"  {tag}: missing {zp.name}")
            continue
        res[tag] = collect(zp, prefix, a.n, a.seed, a.zlo, a.zhi)
        print(f"  {tag}: {len(res[tag])} patches", flush=True)

    out = {"n": a.n, "z_range": [a.zlo, a.zhi], "rows": res}
    if "bad" in res and "good" in res and res["bad"] and res["good"]:
        from scipy.stats import mannwhitneyu
        print(f"\n{'metric':<14}{'bad':>10}{'good':>10}{'AUC':>8}{'p':>11}")
        for k in ("stretch_med", "stretch_q90", "shear_med", "shear_q90", "area_cv"):
            b = np.array([r[k] for r in res["bad"]])
            g = np.array([r[k] for r in res["good"]])
            u = mannwhitneyu(b, g, alternative="two-sided")
            auc = u.statistic / (len(b) * len(g))
            out[f"auc_{k}"] = round(float(auc), 4)
            out[f"p_{k}"] = float(u.pvalue)
            print(f"  {k:<12}{np.median(b):>10.4f}{np.median(g):>10.4f}"
                  f"{auc:>8.3f}{u.pvalue:>11.2e}")
        print("\n  AUC 0.5 = the geometry says nothing about which patches he flagged.")
        print("  (the CT-ridge residual gave 0.512 on the same question — placement does not"
              " separate them)")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
