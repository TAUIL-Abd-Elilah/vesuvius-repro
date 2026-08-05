"""Split the 892-volume m7 recall benchmark by where each volume actually comes from.

Background. `bench_m7_recall.py` reports one median recall over the public labelled set.
In villa#191, @Jinhojeong located those volumes inside the canonical scroll volumes by
normalized cross-correlation, and published the result: 189 of the 892 locate on Scroll1A,
122 of those intersect 202 regions of the published
`Dataset059_s1_s4_s5_patches_frangiedt` patch set, 352 intersecting pairs, 12 of them at
IoU 1.0 on the same extraction grid. None locate on Scroll1B, 4 or 5; Scrolls 2 and 3 were
not searched.

This joins that table to our per-volume recall and asks two questions:

  1. Do the volumes that overlap a published *training* patch set score better, as leakage
     would predict?
  2. Is the single headline median hiding more than one population?

⚠ SCOPE, and it must travel with any number from here. Intersecting a frangiedt patch is
NOT the same as being one of m7's training cases: m7's training set carries no public
identifiers, and this cannot and does not test membership of it. "Locates nowhere" is not
"is not Scroll 1" either — only four canonical volumes were searched.

  python overlap_recall_split.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RECALL_DIR = ROOT / "results" / "m7_recall"
CACHE = ROOT / "results" / "overlap" / "overlap_report.json"

# pinned so the join is reproducible even if the source repo moves on
OVERLAP_URL = (
    "https://raw.githubusercontent.com/Jinhojeong/vesuvius-surface-geometry-diagnostic/"
    "9003cefbda06ae27fdc73550aed0438c69ebd6d7/results/overlap/overlap_report.json"
)
OVERLAP_SHA256 = "cdcce85096236cad8e3dc87a6b498fa50df01ce4850bf4987d2e2785538d60b6"


def fetch_overlap() -> dict:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(OVERLAP_URL) as r:  # noqa: S310 - pinned https URL
            CACHE.write_bytes(r.read())
    blob = CACHE.read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    if got != OVERLAP_SHA256:
        raise SystemExit(f"overlap_report.json sha256 {got} != pinned {OVERLAP_SHA256}")
    return json.loads(blob)


def describe(name: str, vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    if a.size == 0:
        return {"group": name, "n": 0}
    return {
        "group": name,
        "n": int(a.size),
        "median_recall": round(float(np.median(a)), 4),
        "mean_recall": round(float(a.mean()), 4),
        "q1": round(float(np.percentile(a, 25)), 4),
        "q3": round(float(np.percentile(a, 75)), 4),
        "frac_below_0.8": round(float((a < 0.8).mean()), 4),
    }


def mannwhitney_u(x: list[float], y: list[float]) -> tuple[float, float]:
    """Two-sided Mann-Whitney U, normal approximation with tie correction.

    Written out rather than imported so the number does not depend on a scipy version.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = x.size, y.size
    both = np.concatenate([x, y])
    order = both.argsort()
    ranks = np.empty(both.size, float)
    ranks[order] = np.arange(1, both.size + 1)
    srt = both[order]
    i = 0
    while i < srt.size:
        j = i
        while j + 1 < srt.size and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(both, return_counts=True)
    tie = (counts**3 - counts).sum()
    n = n1 + n2
    sd = sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    z = (u1 - n1 * n2 / 2) / sd
    return float(z), float(1 - erf(abs(z) / sqrt(2)))


def main() -> None:
    rep = fetch_overlap()
    recall = {p.stem: json.loads(p.read_text()) for p in sorted(RECALL_DIR.glob("sample_*.json"))}
    scored = {k: v for k, v in recall.items() if v.get("status") == "ok" and "recall" in v}

    located = {r["sample"] for r in rep["located"]}
    overlapping = {p["sample"] for p in rep["pairs"]}
    iou1 = {p["sample"] for p in rep["pairs"] if p["iou"] >= 1.0}
    finetune = {p["sample"] for p in rep["pairs"] if "rand" not in p.get("pool", "")}

    groups = {
        "all_scored": list(scored),
        "locates_on_s1a": [k for k in scored if k in located],
        "s1a_and_intersects_frangiedt": [k for k in scored if k in overlapping],
        "s1a_no_intersection": [k for k in scored if k in located and k not in overlapping],
        "iou_1.0_identical_cube": [k for k in scored if k in iou1],
        "finetune_pool_intersection": [k for k in scored if k in finetune],
        "locates_nowhere_searched": [k for k in scored if k not in located],
    }
    rows = [describe(g, [scored[k]["recall"] for k in ks]) for g, ks in groups.items()]

    # is the split explained by the labels rather than by the volumes?
    def med(keys: list[str], field: str) -> float | None:
        vals = [scored[k][field] for k in keys if field in scored[k]]
        return round(float(np.median(vals)), 4) if vals else None

    confounds = {
        field: {
            "locates_on_s1a": med(groups["locates_on_s1a"], field),
            "locates_nowhere_searched": med(groups["locates_nowhere_searched"], field),
        }
        for field in (
            "label_sheet_fraction",
            "sheet_mean_depth",
            "sheet_components",
            "mean_confidence",
            "pred_positive_fraction",
        )
    }

    z_ov, p_ov = mannwhitney_u(
        [scored[k]["recall"] for k in groups["s1a_and_intersects_frangiedt"]],
        [scored[k]["recall"] for k in groups["locates_nowhere_searched"]],
    )
    z_lo, p_lo = mannwhitney_u(
        [scored[k]["recall"] for k in groups["locates_on_s1a"]],
        [scored[k]["recall"] for k in groups["locates_nowhere_searched"]],
    )
    z_in, p_in = mannwhitney_u(
        [scored[k]["recall"] for k in groups["s1a_and_intersects_frangiedt"]],
        [scored[k]["recall"] for k in groups["s1a_no_intersection"]],
    )

    out = {
        "source_overlap_report": {"url": OVERLAP_URL, "sha256": OVERLAP_SHA256},
        "n_recall_files": len(recall),
        "n_scored": len(scored),
        "groups": rows,
        "median_by_group_confounds": confounds,
        "tests": {
            "intersecting_vs_not_located": {"z": round(z_ov, 3), "p": float(f"{p_ov:.3g}")},
            "located_vs_not_located": {"z": round(z_lo, 3), "p": float(f"{p_lo:.3g}")},
            "intersecting_vs_located_not_intersecting": {
                "z": round(z_in, 3),
                "p": float(f"{p_in:.3g}"),
                "note": "the leakage test: within Scroll1A, does patch-set membership help?",
            },
        },
        "scope": (
            "Overlap is with the published Dataset059_s1_s4_s5_patches_frangiedt patch set, "
            "not with m7's training cases, which carry no public identifiers. "
            "'Locates nowhere' means not found in the four canonical volumes searched "
            "(Scroll1A, 1B, 4, 5); Scrolls 2 and 3 were not searched."
        ),
    }
    dest = ROOT / "results" / "overlap_recall_split.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
