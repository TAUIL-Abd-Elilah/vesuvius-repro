"""Which published surface predictions can be verified against public CT?

The published surface predictions are the input to most downstream segmentation
work, and they are only *checkable* where the prediction and the CT it came from
share a voxel grid - otherwise any comparison needs resampling and stops being
exact. This walks the public Open Data bucket and reports, per scroll:

  * the published surface prediction(s) and the model run behind them
  * the CT volume named in the prediction's own filename
  * whether their level-0 grids are voxel-identical

No credentials, no terms acceptance: everything here is anonymous HTTP against
the public bucket. Metadata only - a few KB per scroll, not voxels.

Usage:
    python catalog_verifiable.py [--out catalog.json] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request

BUCKET = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
# <ct-stamp>-surface-<run-id>-<model...>-L<level>-th<threshold>.zarr
# The model segment is deliberately loose: alongside `surface-m7` the bucket also
# carries e.g. `surface-recto-2um-ps256`, and a stricter pattern silently drops
# whole model families from the catalogue.
PRED_RE = re.compile(
    r"^(?P<ct>\d+)-surface-(?P<run>\d+)-(?P<model>.+?)-L(?P<level>\d+)"
    r"-th(?P<thresh>[0-9.]+)\.zarr$"
)


def fetch(url: str, attempts: int = 5) -> bytes:
    """GET with backoff; this bucket drops connections routinely."""
    delay = 1.0
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            if i == attempts - 1:
                raise
        except Exception:
            if i == attempts - 1:
                raise
        time.sleep(delay)
        delay = min(delay * 2, 16.0)
    raise RuntimeError("unreachable")


def list_prefixes(prefix: str) -> list[str]:
    url = f"{BUCKET}?list-type=2&delimiter=/&max-keys=1000"
    if prefix:
        url += f"&prefix={prefix}"
    body = fetch(url).decode("utf-8", "replace")
    return [p for p in re.findall(r"<Prefix>([^<]+)</Prefix>", body)]


def zarray(path: str, level: int = 0) -> dict | None:
    """Read a pyramid level's .zarray, tolerating stores that do not have one."""
    try:
        return json.loads(fetch(f"{BUCKET}/{path}/{level}/.zarray").decode("utf-8"))
    except Exception:
        return None


def matching_ct_level(ct_path: str, pred_shape: list[int], max_level: int = 6):
    """Find which CT pyramid level the prediction actually sits on.

    A prediction whose filename says L0 is not necessarily on the CT's level 0:
    where it is not, an overlay at level 0 is silently wrong by the level ratio,
    so it is worth naming the level rather than just flagging a mismatch.
    """
    for level in range(max_level):
        meta = zarray(ct_path, level)
        if meta is None:
            break
        if list(meta["shape"]) == list(pred_shape):
            return level, meta["shape"]
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="catalog.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    scrolls = [p.rstrip("/") for p in list_prefixes("")]
    scrolls = [s for s in scrolls if s.startswith("PHerc")]
    if args.limit:
        scrolls = scrolls[: args.limit]
    print(f"{len(scrolls)} scroll prefixes in the bucket\n")

    rows = []
    for i, scroll in enumerate(scrolls, 1):
        base = f"{scroll}/representations/predictions/surfaces/"
        try:
            preds = [p.rstrip("/").split("/")[-1] for p in list_prefixes(base)]
        except Exception as exc:
            print(f"[{i:2}/{len(scrolls)}] {scroll:18} listing failed: {type(exc).__name__}")
            continue
        preds = [p for p in preds if PRED_RE.match(p)]
        if not preds:
            # "no prediction" and "nothing to predict from" are different facts, and
            # conflating them overstates what is missing: six of these prefixes carry
            # only photographs, with no CT volume at all.
            try:
                ct_vols = [v.rstrip("/").split("/")[-1]
                           for v in list_prefixes(f"{scroll}/volumes/")
                           if v.rstrip("/").endswith(".zarr")]
            except Exception:
                ct_vols = []
            status = "no_prediction" if ct_vols else "no_ct"
            note = (f"no published surface prediction ({len(ct_vols)} CT volumes)"
                    if ct_vols else "no CT volumes at all (photos only)")
            print(f"[{i:2}/{len(scrolls)}] {scroll:18} {note}")
            rows.append({"scroll": scroll, "status": status, "ct_volumes": ct_vols})
            continue

        for pred in preds:
            m = PRED_RE.match(pred)
            assert m
            ct_stamp = m.group("ct")
            pmeta = zarray(base + pred)
            # the CT volume whose name starts with the stamp the prediction records
            vols = [v.rstrip("/").split("/")[-1]
                    for v in list_prefixes(f"{scroll}/volumes/")]
            ct_name = next((v for v in vols if v.startswith(ct_stamp)), None)
            cmeta = zarray(f"{scroll}/volumes/{ct_name}") if ct_name else None

            row = {
                "scroll": scroll,
                "prediction": pred,
                "run_id": m.group("run"),
                "model": m.group("model"),
                "threshold": float(m.group("thresh")),
                "ct_volume": ct_name,
                "pred_shape": pmeta.get("shape") if pmeta else None,
                "ct_shape": cmeta.get("shape") if cmeta else None,
            }
            # The filename carries the CT pyramid level the prediction was computed
            # on (the L<k> token). Trust nothing: resolve the level the prediction
            # actually sits on by shape, and check the two agree - that is what
            # makes a comparison against the CT exact rather than approximately right.
            declared = int(m.group("level"))
            row["declared_level"] = declared
            if not ct_name:
                row["actual_level"] = None
                row["status"] = "ct_missing"
            elif not row["pred_shape"]:
                row["actual_level"] = None
                row["status"] = "pred_unreadable"
            else:
                lvl, _ = matching_ct_level(f"{scroll}/volumes/{ct_name}", row["pred_shape"])
                row["actual_level"] = lvl
                if lvl is None:
                    row["status"] = "no_matching_level"
                elif lvl == declared:
                    row["status"] = "verifiable"
                else:
                    row["status"] = "level_mismatch"
            rows.append(row)

            if row["status"] == "verifiable":
                flag = f"OK   L{declared}"
            elif row["status"] == "level_mismatch":
                flag = f"MISM says L{declared} is L{row['actual_level']}"
            else:
                flag = f"?? {row['status']}"
            print(f"[{i:2}/{len(scrolls)}] {scroll:18} {flag:28} "
                  f"pred {row['pred_shape']}")

    preds = [r for r in rows if r.get("prediction")]
    verifiable = [r for r in rows if r.get("status") == "verifiable"]
    mismatch = [r for r in rows if r.get("status") == "level_mismatch"]
    other = [r for r in preds if r.get("status") not in ("verifiable", "level_mismatch")]
    runs = sorted({r["run_id"] for r in preds})
    bylevel: dict = {}
    for r in verifiable:
        bylevel[r["declared_level"]] = bylevel.get(r["declared_level"], 0) + 1
    print(f"\n{'='*70}")
    print(f"scrolls scanned                    : {len(scrolls)}")
    print(f"published surface predictions      : {len(preds)}")
    print(f"declared CT level matches the grid : {len(verifiable)}")
    print(f"  by declared level                : "
          + ", ".join(f"L{k}={v}" for k, v in sorted(bylevel.items())))
    print(f"declared level WRONG               : {len(mismatch)}")
    for r in mismatch:
        f = 2 ** abs(r["actual_level"] - r["declared_level"])
        print(f"    {r['scroll']:18} filename says L{r['declared_level']}, "
              f"grid is CT L{r['actual_level']} -> off by {f}x")
    print(f"unresolved                         : {len(other)}")
    for r in other:
        print(f"    {r['scroll']:18} {r['status']}")
    print(f"distinct model runs                : {len(runs)} {runs}")
    families: dict = {}
    for r in preds:
        key = (r["model"], r["run_id"])
        families[key] = families.get(key, 0) + 1
    print("model families:")
    for (model, run), n in sorted(families.items(), key=lambda kv: -kv[1]):
        print(f"    {model:34} run {run}  {n} prediction(s)")
    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
