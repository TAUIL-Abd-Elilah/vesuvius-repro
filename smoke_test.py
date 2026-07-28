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
    report_paths = list((HERE / "results").glob("*.json"))
    report_paths += list((HERE / "results" / "variants").glob("*.json"))
    for path in report_paths:
        row = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(row, dict) and "scroll" in row and "dice" in row:
            reports.append(row)
    checked = {row["scroll"] for row in reports}
    assert checked == scrolls, (len(checked), sorted(scrolls - checked))

    catalog_pairs = {(row["scroll"], row["prediction"]) for row in artifacts}
    matched_pairs = {(row["scroll"], row.get("prediction")) for row in reports
                     if row.get("dice", 0) > 0.99}
    assert catalog_pairs <= matched_pairs, sorted(catalog_pairs - matched_pairs)

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
    assert "artifacts with regional match   : 41/41" in proc.stdout

    print("smoke test passed: 41/41 m7 artifacts have a regional match across 36 scrolls")


if __name__ == "__main__":
    main()
