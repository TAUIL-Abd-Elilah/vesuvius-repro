"""Run both arms of PREREGISTER_margin.md, three seeds each, one GPU job at a time.

Order is INTERLEAVED -- A0, B0, A1, B1, A2, B2 -- not arm-major. If the box is interrupted
(six of the last seven long GPU runs here died on their own), what survives is a set of
complete A/B pairs at matched seeds, which is analysable. Arm-major order would leave three
baselines and nothing to compare them to.

Already-finished runs are skipped, so this is safe to restart.

  python run_margin_arms.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "margin_arms"
PY = sys.executable
EPOCHS, ITERS, BATCH = 80, 60, 2


def main() -> None:
    jobs = [(arm, seed) for seed in (0, 1, 2) for arm in ("A", "B")]
    log = RESULTS / "run_log.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    done = json.loads(log.read_text()) if log.exists() else []

    for arm, seed in jobs:
        out = RESULTS / f"{arm}_seed{seed}.json"
        if out.exists():
            print(f"[skip] {arm} seed {seed} already done", flush=True)
            continue
        print(f"\n{'='*64}\n[run] arm {arm} seed {seed}  "
              f"{time.strftime('%H:%M:%S')}\n{'='*64}", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [PY, "-u", str(ROOT / "margin_arms.py"), "--arm", arm, "--seed", str(seed),
             "--epochs", str(EPOCHS), "--iters", str(ITERS), "--batch", str(BATCH)],
            cwd=ROOT)
        done.append({"arm": arm, "seed": seed, "rc": r.returncode,
                     "minutes": round((time.time() - t0) / 60, 1),
                     "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        log.write_text(json.dumps(done, indent=1))
        print(f"[done] arm {arm} seed {seed} rc={r.returncode} "
              f"{done[-1]['minutes']} min", flush=True)

    print("\nall jobs attempted; now: python margin_arms.py --compare", flush=True)


if __name__ == "__main__":
    main()
