"""Does the located/not-located recall gap survive matching on labelled sheet fraction?

Three explanations for the split in overlap_recall_split.py are now dead. Leakage died on
the join (patch-set membership buys no recall, p = 0.80). A label artifact died because the
located volumes carry MORE labelled sheet, not less. Fused geometry died on @Jinhojeong's
892-volume census run (flagged-site rate 1.10% in both groups, p = 0.33, with the power to
see a 0.1-point shift).

What is left is what was measured directly: the located volumes carry more labelled sheet
(median 0.076 against 0.059), and the model is less confident on them while predicting more
positive. Those are two facts, not one explanation, and the obvious question is whether the
first causes the second - i.e. whether "denser labelled sheet is harder" accounts for the
whole 0.777 against 0.918, or whether something about the located population survives it.

So: stratify by labelled sheet fraction and compare within strata. If the gap collapses when
sheet fraction is held roughly fixed, the signature is density. If it survives, it is not.

  python overlap_recall_matched.py
"""

from __future__ import annotations

import json
from math import erf, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RECALL_DIR = ROOT / "results" / "m7_recall"
OVERLAP = ROOT / "results" / "overlap" / "overlap_report.json"


def mannwhitney(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = x.size, y.size
    if n1 < 3 or n2 < 3:
        return None, None
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
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    u1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(both, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    n = n1 + n2
    sd = sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    z = (u1 - n1 * n2 / 2) / sd
    return float(z), float(1 - erf(abs(z) / sqrt(2)))


def main() -> None:
    rep = json.loads(OVERLAP.read_text())
    located = {r["sample"] for r in rep["located"]}

    rows = []
    for p in sorted(RECALL_DIR.glob("sample_*.json")):
        d = json.loads(p.read_text())
        if d.get("status") != "ok" or "recall" not in d or "label_sheet_fraction" not in d:
            continue
        rows.append({"sample": p.stem, "recall": float(d["recall"]),
                     "sheet": float(d["label_sheet_fraction"]),
                     "located": p.stem in located})

    loc = [r for r in rows if r["located"]]
    non = [r for r in rows if not r["located"]]
    z_all, p_all = mannwhitney([r["recall"] for r in loc], [r["recall"] for r in non])

    # Quintiles of labelled sheet fraction over the whole scored set, so each stratum holds
    # volumes of comparable density and the comparison inside it is like-for-like.
    edges = np.percentile([r["sheet"] for r in rows], [0, 20, 40, 60, 80, 100])
    strata = []
    for k in range(5):
        lo, hi = edges[k], edges[k + 1]
        sel = [r for r in rows if (r["sheet"] >= lo and (r["sheet"] < hi or k == 4))]
        a = [r["recall"] for r in sel if r["located"]]
        b = [r["recall"] for r in sel if not r["located"]]
        z, pv = mannwhitney(a, b)
        strata.append({
            "stratum": k + 1,
            "sheet_range": [round(float(lo), 4), round(float(hi), 4)],
            "n_located": len(a), "n_not_located": len(b),
            "median_located": round(float(np.median(a)), 4) if a else None,
            "median_not_located": round(float(np.median(b)), 4) if b else None,
            "gap": round(float(np.median(b) - np.median(a)), 4) if a and b else None,
            "z": None if z is None else round(z, 3),
            "p": None if pv is None else float(f"{pv:.3g}"),
        })

    out = {
        "n_scored": len(rows),
        "unmatched": {
            "median_located": round(float(np.median([r["recall"] for r in loc])), 4),
            "median_not_located": round(float(np.median([r["recall"] for r in non])), 4),
            "gap": round(float(np.median([r["recall"] for r in non])
                               - np.median([r["recall"] for r in loc])), 4),
            "z": round(z_all, 3), "p": float(f"{p_all:.3g}"),
        },
        "sheet_fraction_medians": {
            "located": round(float(np.median([r["sheet"] for r in loc])), 4),
            "not_located": round(float(np.median([r["sheet"] for r in non])), 4),
        },
        "strata": strata,
        "reading": ("if the gap collapses inside strata, the located signature is labelled "
                    "sheet density; if it survives at similar size, density does not explain it"),
    }
    (ROOT / "results" / "overlap_recall_matched.json").write_text(json.dumps(out, indent=1))

    print(f"scored volumes: {len(rows)}  located {len(loc)}  not located {len(non)}")
    u = out["unmatched"]
    print(f"\nunmatched: located {u['median_located']} vs not-located {u['median_not_located']}"
          f"  gap {u['gap']}  p={u['p']}")
    print(f"\n{'stratum':<8}{'sheet range':<20}{'n loc':>6}{'n non':>7}"
          f"{'med loc':>9}{'med non':>9}{'gap':>8}{'p':>10}")
    for s in strata:
        print(f"{s['stratum']:<8}{str(s['sheet_range']):<20}{s['n_located']:>6}"
              f"{s['n_not_located']:>7}{str(s['median_located']):>9}"
              f"{str(s['median_not_located']):>9}{str(s['gap']):>8}{str(s['p']):>10}")
    gaps = [s["gap"] for s in strata if s["gap"] is not None]
    print(f"\nmean within-stratum gap {np.mean(gaps):.4f} vs unmatched {u['gap']}")


if __name__ == "__main__":
    main()
