"""Run one regional m7 spot-check for every remaining scroll, unattended.

Eleven scrolls were checked by hand; this walks the rest so each scroll carrying an m7
artifact has one checked region. It does not compare complete volumes or every artifact
on scrolls with multiple published predictions.

Each scroll is independent, so a failure is logged and the sweep continues - on this
connection a run dies every so often and aborting the batch would waste the ones
already done. Scrolls with a result JSON already present are skipped, so the sweep is
resumable: rerun it after a crash and it picks up where it stopped.

Usage:
    python sweep_all.py [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=str(HERE / "catalog.json"))
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = json.load(open(args.catalog))
    m7 = [r for r in rows if r.get("status") == "verifiable" and "m7" in r.get("model", "")]

    levels: dict[str, list[int]] = {}
    for r in m7:
        levels.setdefault(r["scroll"], []).append(r["declared_level"])

    done = {p.name.split("_")[0] for p in Path(args.results).glob("*_surface-m7.json")}
    done |= {"PHerc0139", "PHerc0332", "PHercParis4"}  # verified before the naming settled

    todo = sorted(s for s in levels if s not in done)
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(levels)} scrolls with an m7 prediction; {len(levels) - len(todo)} done; "
          f"{len(todo)} to run\n")
    if args.dry_run:
        for s in todo:
            print(f"  {s}  levels {levels[s]}")
        return

    ok, failed = [], []
    for i, scroll in enumerate(todo, 1):
        lv = levels[scroll]
        cmd = [args.python, str(HERE / "run_verification.py"),
               "--scroll", scroll, "--model", "m7",
               "--catalog", args.catalog, "--out", args.results]
        if len(lv) > 1:
            cmd += ["--level", str(max(lv))]
        elif lv:
            cmd += ["--level", str(lv[0])]
        print(f"[{i}/{len(todo)}] {scroll} (L{lv}) ...", flush=True)
        t0 = time.time()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        dt = time.time() - t0
        if r.returncode == 0:
            tail = [l for l in r.stdout.splitlines() if "Dice" in l]
            print(f"    OK  {dt/60:.1f} min  {tail[-1].strip() if tail else ''}",
                  flush=True)
            ok.append(scroll)
        else:
            err = (r.stderr or r.stdout).strip().splitlines()
            print(f"    FAILED {dt/60:.1f} min: {err[-1] if err else '?'}",
                  file=sys.stderr, flush=True)
            failed.append(scroll)

    print(f"\n=== {len(ok)} ok, {len(failed)} failed ===")
    if failed:
        print("failed:", ", ".join(failed))
        print("rerun sweep_all.py to retry only those (results are skipped when present)")


if __name__ == "__main__":
    main()
