"""What could we predict that the project has not?

Nine scrolls in the open-data bucket have public CT but no published surface
prediction. The m7 model is public, so those predictions are producible - but only
where the CT actually admits the model's working resolution.

The m7 runs all sit near ~9.6 um: the ~9 um scans are predicted at pyramid level 0
and the ~2.4 um scans at level 2. So for each uncovered scroll this asks whether any
pyramid level of any of its volumes lands near that resolution, and how large the
resulting grid would be.

Metadata only, anonymous - a few KB per scroll.

Usage:
    python survey_uncovered.py [--out uncovered.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from catalog_verifiable import BUCKET, fetch, list_prefixes

# ...-<res>um-...  e.g. 20260319104112-2.401um-0.3m-77keV-masked.zarr
RES_RE = re.compile(r"-(?P<res>[0-9.]+)um-")

# what the published m7 runs actually operated on, in microns
TARGET_UM = 9.6
TOLERANCE = 0.25  # fraction; 9.6 +/- 25% covers the 8.6-9.4 and 4x2.4 groups


def shape_of(scroll: str, volume: str, level: int) -> list[int] | None:
    url = f"{BUCKET}/{scroll}/volumes/{volume}/{level}/.zarray"
    try:
        return json.loads(fetch(url))["shape"]
    except Exception:
        return None


def survey(scroll: str) -> dict:
    # list_prefixes yields full keys ("<scroll>/volumes/<name>.zarr/"), so take the
    # basename - using the key as the volume name doubles the prefix and 404s.
    vols = [v.rstrip("/").split("/")[-1] for v in list_prefixes(f"{scroll}/volumes/")
            if v.rstrip("/").endswith(".zarr")]
    out = {"scroll": scroll, "volumes": [], "best": None}
    for v in vols:
        m = RES_RE.search(v)
        if not m:
            continue
        res0 = float(m.group("res"))
        entry = {"volume": v, "level0_um": res0, "levels": []}
        for lvl in range(5):
            um = res0 * (2 ** lvl)
            if abs(um - TARGET_UM) / TARGET_UM > TOLERANCE:
                continue
            shp = shape_of(scroll, v, lvl)
            if shp is None:
                continue
            entry["levels"].append({"level": lvl, "um": round(um, 3), "shape": shp,
                                    "voxels": int(shp[0]) * int(shp[1]) * int(shp[2])})
        if entry["levels"]:
            out["volumes"].append(entry)
    # prefer the level closest to the target, tie-broken by the biggest grid
    cands = [(abs(l["um"] - TARGET_UM), -l["voxels"], e["volume"], l)
             for e in out["volumes"] for l in e["levels"]]
    if cands:
        cands.sort(key=lambda c: (c[0], c[1]))
        _, _, vol, lvl = cands[0]
        out["best"] = {"volume": vol, **lvl}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalog.json")
    ap.add_argument("--out", default="uncovered.json")
    args = ap.parse_args()

    cat = json.load(open(args.catalog))
    scrolls = sorted({r["scroll"] for r in cat if r.get("status") == "no_prediction"})
    print(f"{len(scrolls)} scrolls with public CT and no published surface prediction\n")

    results = []
    for s in scrolls:
        print(f"--- {s}", flush=True)
        try:
            r = survey(s)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        results.append(r)
        if not r["volumes"]:
            print("    no volume has a level near the model's working resolution")
        for e in r["volumes"]:
            for l in e["levels"]:
                mark = " <-- best" if r["best"] and r["best"]["volume"] == e["volume"] \
                    and r["best"]["level"] == l["level"] else ""
                print(f"    {e['volume']}  L{l['level']}  {l['um']}um  "
                      f"{l['shape']}  {l['voxels'] / 1e9:.1f} Gvox{mark}")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    ok = [r for r in results if r["best"]]
    print(f"{len(ok)}/{len(results)} scrolls admit a prediction at ~{TARGET_UM}um")


if __name__ == "__main__":
    main()
