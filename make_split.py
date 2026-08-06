"""Write the train/val/test split for PREREGISTER_margin.md, before any arm is trained.

Split by PROVENANCE, not at random. @Jinhojeong located 189 of the 892 public volumes inside
Scroll1A by normalized cross-correlation (villa#191); those are the population the current
model does worst on - median recall 0.777 against 0.918 for the rest, a gap that has now
survived elimination of leakage, label artifact, fused geometry and labelled-sheet density.

  test        every located volume that bench_m7_recall.py can score
  val / train  sampled from the volumes that locate nowhere searched, disjoint from test

A random split would put near-duplicate crops of the same region on both sides, and the
located set is exactly where an improvement has to show, so it is held out whole.

  python make_split.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RECALL_DIR = ROOT / "vesuvius-repro" / "results" / "m7_recall"
OVERLAP = ROOT / "vesuvius-repro" / "results" / "overlap" / "overlap_report.json"
LABELS = ROOT / "data" / "kaggle" / "labels"

N_VAL = 100
SEED = 0


def main() -> None:
    blob = OVERLAP.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    rep = json.loads(blob)
    located = {r["sample"] for r in rep["located"]}

    on_disk = {p.stem for p in LABELS.glob("sample_*.tif")}

    scored = set()
    for p in sorted(RECALL_DIR.glob("sample_*.json")):
        d = json.loads(p.read_text())
        if d.get("status") == "ok" and "recall" in d:
            scored.add(p.stem)

    test = sorted((located & scored) & on_disk)
    pool = sorted((on_disk - located) & scored)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pool))
    val = sorted(pool[i] for i in idx[:N_VAL])
    train = sorted(pool[i] for i in idx[N_VAL:])

    assert not (set(test) & set(val)), "test/val overlap"
    assert not (set(test) & set(train)), "test/train overlap"
    assert not (set(val) & set(train)), "val/train overlap"

    out = {
        "seed": SEED,
        "overlap_report_sha256": sha,
        "criterion": ("test = volumes that locate on Scroll1A and are scored by "
                      "bench_m7_recall.py; val/train sampled from volumes that locate "
                      "nowhere among the four canonical volumes searched"),
        "n_test": len(test), "n_val": len(val), "n_train": len(train),
        "test": test, "val": val, "train": train,
    }
    dest = ROOT / "vesuvius-repro" / "results" / "margin_split.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"  test  {len(test):>4}  (located on Scroll1A, scored)")
    print(f"  val   {len(val):>4}")
    print(f"  train {len(train):>4}")
    print(f"  overlap_report sha256 {sha[:16]}...")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
