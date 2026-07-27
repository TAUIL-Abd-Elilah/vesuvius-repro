"""Under what DATA conditions does the published surface model miss sheet?

bench_m7_recall.py established that the published m7 path finds a median 90.8% of labelled
sheet across the 868 public Kaggle volumes, with a long left tail: a quarter of volumes
below 80%, and 41 labelled sheets recovered barely at all.

That says how much is missed, not what makes a volume hard. The correlations available
from the model side are circular - "recall is low where confidence is low" is the same
fact stated twice, and tells nobody how to improve anything. This measures the volumes
themselves, using only the CT and the label, and asks which physical conditions the misses
line up with.

Five properties per volume, all computed without the model:

  thickness     mean sheet thickness in voxels (2x the mean EDT inside the sheet)
  spacing       median gap between neighbouring sheets, measured as the distance from
                background voxels to the nearest sheet, doubled
  contrast      separation of sheet from background in CT, in units of background sigma -
                the signal the model actually has to work with
  fragmentation labelled sheet components per volume, and the size of the largest
  flatness      angular dispersion of the local sheet normal. High dispersion = curved or
                crushed, which is the regime villa#191 says the models do worst in.

Written to results/miss_map/, one JSON per volume, joined to recall by sample name.
CPU only, no GPU, no network.

Usage:
    python miss_map.py --n 900
    python miss_map.py --report          # aggregate what is already computed
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def describe(img: np.ndarray, lab: np.ndarray) -> dict:
    from scipy.ndimage import distance_transform_edt, label as cc_label, gaussian_filter

    sheet = lab == 1
    out: dict = {"sheet_fraction": float(sheet.mean())}
    if sheet.sum() < 100:
        out["status"] = "no_sheet"
        return out

    # --- thickness: EDT inside the sheet, doubled (radius -> full width)
    din = distance_transform_edt(sheet)
    out["thickness_mean"] = float(2 * din[sheet].mean())
    out["thickness_p90"] = float(2 * np.quantile(din[sheet], 0.9))

    # --- spacing: how far background sits from the nearest sheet. Doubled, this is the
    # gap between neighbouring sheets. The median is over background voxels, so it
    # reflects typical separation rather than the few big empty regions.
    dout = distance_transform_edt(~sheet)
    bg = ~sheet
    out["spacing_median"] = float(2 * np.median(dout[bg]))
    out["spacing_p10"] = float(2 * np.quantile(dout[bg], 0.10))

    # --- contrast: how separable is sheet from background in the CT itself
    f = img.astype(np.float32)
    mu_s, mu_b = float(f[sheet].mean()), float(f[bg].mean())
    sd_b = float(f[bg].std())
    out["ct_sheet_mean"], out["ct_bg_mean"], out["ct_bg_std"] = mu_s, mu_b, sd_b
    out["contrast_sigma"] = float((mu_s - mu_b) / sd_b) if sd_b > 0 else 0.0

    # --- fragmentation
    cc, n = cc_label(sheet)
    out["components"] = int(n)
    if n:
        sizes = np.bincount(cc.ravel())[1:]
        out["largest_component_frac"] = float(sizes.max() / sizes.sum())
        out["components_over_1k"] = int((sizes > 1000).sum())

    # --- flatness: dispersion of the local sheet normal.
    # The gradient of a smoothed mask points across the sheet, so its direction is the
    # surface normal. Sign is arbitrary per voxel, so orientation is compared through the
    # structure tensor of the normals rather than by averaging vectors, which would cancel.
    sm = gaussian_filter(sheet.astype(np.float32), 1.5)
    gz, gy, gx = np.gradient(sm)
    mag = np.sqrt(gz**2 + gy**2 + gx**2)
    edge = sheet & (mag > 1e-3)
    if edge.sum() > 50:
        v = np.stack([gz[edge], gy[edge], gx[edge]], axis=1)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        T = (v[:, :, None] * v[:, None, :]).mean(axis=0)   # 3x3, sign-invariant
        w = np.linalg.eigvalsh(T)                          # ascending
        # one dominant direction => flat; spread across eigenvalues => curved/crushed
        out["normal_anisotropy"] = float(w[2])             # 1/3 = isotropic, 1 = perfectly flat
        out["flatness"] = float((w[2] - w[1]) / w.sum())
    out["status"] = "ok"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--out", default="results/miss_map")
    ap.add_argument("--recall", default="results/m7_recall")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.report:
        import tifffile
        imgs = sorted(glob.glob("data/kaggle/images/*.tif"))[:args.n]
        for i, ip in enumerate(imgs, 1):
            name = os.path.basename(ip)
            dest = out / name.replace(".tif", ".json")
            if dest.exists():
                continue
            lp = ip.replace("images", "labels")
            if not os.path.exists(lp):
                continue
            try:
                rep = describe(tifffile.imread(ip), tifffile.imread(lp))
            except Exception as exc:  # noqa: BLE001
                rep = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            rep["sample"] = name
            dest.write_text(json.dumps(rep, indent=2))
            if i % 25 == 0:
                print(f"  {i}/{len(imgs)}", flush=True)

    # --- join to recall and report
    rec = {}
    for f in glob.glob(str(Path(args.recall) / "*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("status") == "ok":
            rec[r["sample"]] = r["recall"]
    rows = []
    for f in glob.glob(str(out / "*.json")):
        r = json.loads(Path(f).read_text())
        if r.get("status") == "ok" and r["sample"] in rec:
            r["recall"] = rec[r["sample"]]
            rows.append(r)
    if not rows:
        print("nothing joined yet")
        return

    print(f"\n{len(rows)} volumes with both a recall score and a description\n")
    y = np.array([r["recall"] for r in rows])
    keys = ["thickness_mean", "spacing_median", "spacing_p10", "contrast_sigma",
            "components", "largest_component_frac", "flatness", "normal_anisotropy",
            "sheet_fraction"]
    print("correlation with recall (what the DATA looks like where the model fails):")
    scored = []
    for k in keys:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        m = np.isfinite(v)
        if m.sum() < 20 or v[m].std() == 0:
            continue
        c = float(np.corrcoef(y[m], v[m])[0, 1])
        scored.append((abs(c), c, k))
    for _, c, k in sorted(scored, reverse=True):
        print(f"  {k:24} r = {c:+.3f}")

    print("\nrecall by quintile of the strongest few:")
    for _, _, k in sorted(scored, reverse=True)[:3]:
        v = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        m = np.isfinite(v)
        qs = np.quantile(v[m], [0, .2, .4, .6, .8, 1.0])
        print(f"  {k}:")
        for i in range(5):
            sel = m & (v >= qs[i]) & (v <= qs[i + 1] if i == 4 else v < qs[i + 1])
            if sel.sum():
                print(f"    {qs[i]:8.2f}-{qs[i+1]:<8.2f} n={sel.sum():>4}  "
                      f"median recall {100*np.median(y[sel]):5.1f}%")


if __name__ == "__main__":
    main()
