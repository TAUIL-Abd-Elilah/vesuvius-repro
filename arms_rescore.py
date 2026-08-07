"""Re-score the saved arm checkpoints through `surface_bench.py`, per amendment 3.

Amendment 1 picked the matched-budget threshold on the VAL set. Amendment 3 records why that
is wrong: val is drawn from the 681 volumes that locate nowhere and test is the 174 that locate
on Scroll1A, and those are exactly the two populations whose median recall differs 0.777 /
0.918. A threshold fitted on the easy population does not transfer to the hard one. Arm A seed
0 asked for a 0.12 spend and got 0.0217.

This recomputes every endpoint with the threshold taken **per volume on the volume being
scored**, which spends the budget exactly by construction. No retraining: amendment 1 saved the
checkpoints precisely so an endpoint question would never cost a training run again.

⭐ It deliberately calls `surface_bench.endpoints()` rather than reimplementing the metrics, so
the arms and the published m7 reference are scored by **byte-identical code**. Any difference
between them is then a difference between the models, which is the only thing that should
differ. Reimplementing would reintroduce exactly the class-2 and calibration mistakes that
`surface_bench` exists to prevent.

  python arms_rescore.py --ckpt results/margin_arms/A_seed0.pt --n 4 --device cpu   # smoke
  python arms_rescore.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile

import surface_bench as SB
from ablate_faint_sheet import build_model
from margin_arms import CROP, STRIDE, TRIM, predict_volume

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "margin_arms"
SPLIT = ROOT / "vesuvius-repro" / "results" / "margin_split.json"


def score_checkpoint(ckpt: Path, names: list[str], device: str) -> dict:
    import torch

    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    located = {r["sample"] for r in json.loads(SB.OVERLAP.read_text())["located"]}
    rows, t0 = [], time.time()
    for k, nm in enumerate(names):
        ct = np.asarray(tifffile.imread(str(SB.IMAGES / f"{nm}.tif")))
        lab = np.asarray(tifffile.imread(str(SB.LABELS / f"{nm}.tif")))
        logits = predict_volume(model, ct, device, size=CROP, stride=STRIDE)
        sl = tuple(slice(TRIM, s - TRIM) for s in ct.shape)
        l0, l1 = logits[0][sl], logits[1][sl]
        p = 1.0 / (1.0 + np.exp(-(l1 - l0)))
        r = SB.endpoints(p, lab[sl], ct[sl])
        r["sample"] = nm
        r["located"] = nm in located
        rows.append(r)
        if k % 20 == 0:
            print(f"    [{k}/{len(names)}] {time.time()-t0:.0f}s", flush=True)
    return SB.summarise(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    names = json.loads(SPLIT.read_text())["test"]
    if a.n:
        names = names[:a.n]

    ckpts = sorted(RESULTS.glob("*.pt")) if a.all else [Path(a.ckpt)]
    if not ckpts or not all(c.exists() for c in ckpts):
        raise SystemExit(f"no checkpoints found (looked in {RESULTS})")

    out = {}
    for c in ckpts:
        print(f"  {c.name}", flush=True)
        res = score_checkpoint(c, names, a.device)
        out[c.stem] = res
        s = res.get("all", {})
        print(f"    recall {s.get('median_recall')}  lift {s.get('median_precision_lift')}  "
              f"budget_recall {s.get('median_budget_recall')}  "
              f"predpos {s.get('median_pred_positive_fraction')}", flush=True)

    dest = RESULTS / "rescored.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
