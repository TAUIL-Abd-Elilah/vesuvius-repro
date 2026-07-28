"""Offline integrity check for a fresh clone of this repository."""

from __future__ import annotations

import compileall
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    if not compileall.compile_dir(HERE, quiet=1):
        raise SystemExit("Python compilation failed")

    catalog = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
    artifacts = [row for row in catalog
                 if row.get("status") == "verifiable"
                 and "m7" in row.get("model", "")]
    scrolls = {row["scroll"] for row in artifacts}
    assert len(artifacts) == 41, len(artifacts)
    assert len(scrolls) == 36, len(scrolls)

    reports = []
    for path in (HERE / "results").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(row, dict) and "scroll" in row and "dice" in row:
            reports.append(row)
    checked = {row["scroll"] for row in reports}
    assert checked == scrolls, (len(checked), sorted(scrolls - checked))

    variants = []
    for path in (HERE / "results" / "variants").glob("*.json"):
        variants.append(json.loads(path.read_text(encoding="utf-8")))
    assert any(row.get("scroll") == "PHercParis4" and row.get("dice", 0) > 0.99
               for row in variants)

    proc = subprocess.run(
        [sys.executable, str(HERE / "summarize_for_submission.py")],
        check=True, capture_output=True, text=True, encoding="utf-8")
    assert "published m7 artifacts          : 41" in proc.stdout
    assert "regional spot-checks            : 36" in proc.stdout

    print("smoke test passed: 41 m7 artifacts, 36 scrolls, 36 regional checks")


if __name__ == "__main__":
    main()
