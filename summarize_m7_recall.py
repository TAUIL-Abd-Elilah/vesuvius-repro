"""Regenerate every recall figure quoted in the README from results/m7_recall/.

These numbers were maintained by hand until now, which is exactly how the artifact-audit
table drifted six times before it was made generated. Run this instead of retyping:

    python summarize_m7_recall.py                    # the figures
    python summarize_m7_recall.py --check            # non-zero exit if the README disagrees

Every percentage is over volumes with status "ok" -- those with labelled sheet in the
scored interior. Volumes with no labelled sheet there cannot have a recall and are counted
separately rather than folded in as zeros, which would drag every quantile down.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(results_dir / "*.json"))):
        if Path(path).name == "README.md":
            continue
        rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
    return rows


def summarize(rows: list[dict]) -> dict:
    status = Counter(r.get("status", "?") for r in rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    recall = np.array([r["recall"] for r in ok], dtype=float)

    components = sum(r.get("sheet_components", 0) for r in ok)
    lost = sum(r.get("sheet_components_lost", 0) for r in ok)
    volumes_losing = sum(1 for r in ok if r.get("sheet_components_lost", 0) > 0)

    def median_at(threshold: str) -> float | None:
        vals = [r["recall_by_threshold"][threshold] for r in ok
                if threshold in r.get("recall_by_threshold", {})]
        return float(np.median(vals)) if vals else None

    return {
        "attempted": len(rows),
        "status": dict(status),
        "scored": len(ok),
        "median_recall": float(np.median(recall)),
        "p5_recall": float(np.percentile(recall, 5)),
        "below_80": float(100 * np.mean(recall < 0.80)),
        "below_70": float(100 * np.mean(recall < 0.70)),
        "below_50": float(100 * np.mean(recall < 0.50)),
        "components": components,
        "components_lost": lost,
        "volumes_losing_a_component": volumes_losing,
        "median_recall_at_0.05": median_at("0.05"),
        "median_recall_at_0.70": median_at("0.70"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results" / "m7_recall"))
    ap.add_argument("--expect-total", type=int, default=892,
                    help="size of the public set; warns if fewer volumes were attempted")
    args = ap.parse_args()

    rows = load(Path(args.results))
    if not rows:
        raise SystemExit(f"no result JSONs in {args.results}")
    s = summarize(rows)

    print("=" * 68)
    print("m7 RECALL AGAINST THE PUBLIC LABELS -- regenerate, do not retype")
    print("=" * 68)
    print(f"volumes attempted        : {s['attempted']} of {args.expect_total}")
    for name, count in sorted(s["status"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:24}: {count}")
    if s["attempted"] < args.expect_total:
        print(f"  !! {args.expect_total - s['attempted']} public volumes unscored - "
              f"say '{s['attempted']} of {args.expect_total}', never 'all'")
    if s["status"].get("predict_failed"):
        print(f"  !! {s['status']['predict_failed']} predict failures - these were transient "
              f"before; delete those JSONs and rerun to retry")

    print(f"\nscored (labelled sheet in interior): {s['scored']}")
    print(f"  median recall          : {100*s['median_recall']:.1f}%")
    print(f"  5th percentile         : {100*s['p5_recall']:.1f}%")
    print(f"  below 80% recall       : {s['below_80']:.1f}% of volumes")
    print(f"  below 70%              : {s['below_70']:.1f}%")
    print(f"  below 50%              : {s['below_50']:.1f}%")
    print(f"  sheet components       : {s['components_lost']} of {s['components']} "
          f"barely recovered, across {s['volumes_losing_a_component']} volumes")
    if s["median_recall_at_0.05"] is not None:
        print(f"\nnot a thresholding artifact: median recall {100*s['median_recall']:.1f}% "
              f"at the published 0.2")
        print(f"  -> {100*s['median_recall_at_0.05']:.1f}% at 0.05, "
              f"{100*s['median_recall_at_0.70']:.1f}% at 0.70")

    print("\n" + json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
