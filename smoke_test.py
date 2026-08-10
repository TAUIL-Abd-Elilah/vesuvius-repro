"""Offline integrity check for a fresh clone of this repository."""

from __future__ import annotations

import compileall
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
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

    triage = json.loads(
        (HERE / "results" / "patch_triage.json").read_text(encoding="utf-8")
    )
    model = json.loads(
        (HERE / "results" / "patch_triage_model.json").read_text(encoding="utf-8")
    )
    curve = json.loads(
        (HERE / "results" / "patch_triage_curve.json").read_text(encoding="utf-8")
    )
    assert model["features"] == triage["features"]
    assert model["training"]["n_slabs"] == triage["n_slabs"] == 41
    assert model["training"]["n_patches"] == triage["n_patches"] == 56835
    row10 = next(row for row in curve["curve"] if row["budget"] == 0.10)
    robust = triage["per_slab_robustness"]
    assert abs(row10["mean_per_slab_lift"] - robust["mean"]) < 1e-12
    assert abs(row10["median_per_slab_lift"] - robust["median"]) < 1e-12
    assert abs(row10["patch_weighted_lift"] - robust["patch_weighted"]) < 1e-12
    ET.parse(HERE / "results" / "patch_triage_lift.svg")
    rank_help = subprocess.run(
        [sys.executable, str(HERE / "patch_triage.py"), "rank", "--help"],
        check=True, capture_output=True, text=True, encoding="utf-8")
    assert "PATCH/x.tif, y.tif, z.tif" in rank_help.stdout
    assert "directory or ZIP" in rank_help.stdout
    assert "--slab" in rank_help.stdout

    spiral_quality = subprocess.run(
        [sys.executable, str(HERE / "spiral_quality" / "verify_spiral_quality_release.py")],
        check=True, capture_output=True, text=True, encoding="utf-8")
    spiral_result = json.loads(spiral_quality.stdout)
    assert spiral_result["complete"] is True
    assert spiral_result["paired_profiles_rehashed"] == 800
    assert spiral_result["ct_recomputed"] is True
    assert spiral_result["intrinsic_recomputed"] is True
    assert spiral_result["all_authorizations_false"] is True

    print(
        "smoke test passed: 41/41 m7 artifacts match; patch-triage artifacts agree; "
        "Spiral quality release recomputes"
    )


if __name__ == "__main__":
    main()
