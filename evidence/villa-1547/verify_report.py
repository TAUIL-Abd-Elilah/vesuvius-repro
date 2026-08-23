#!/usr/bin/env python3
"""Verify the frozen Villa #1547 report and preregistered gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT_SHA256 = "6a5b6ef6217629452b2717682fa475184e2c9641c2b3011e4813f3d67bc5c65d"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report_one = ROOT / "report-workers-1.json"
    report_four = ROOT / "report-workers-4.json"
    assert report_one.read_bytes() == report_four.read_bytes()
    assert sha256(report_one) == REPORT_SHA256

    report = json.loads(report_one.read_text(encoding="utf-8"))
    assert report["parameters"] == {
        "point_stride": 1,
        "target_index_sampling_stride": 1,
        "tolerance": 2.0,
    }
    assert report["self_pairs_excluded"] is True

    coverage = {
        (source["source_path"], hit["target_path"]):
            hit["source_coverage_fraction"]
        for source in report["sources"]
        for hit in source["hits"]
    }
    positives = [coverage[("w045", "w046")], coverage[("w046", "w045")]]
    controls = [
        coverage[("w044", "w045")],
        coverage[("w045", "w044")],
        coverage[("w046", "w047")],
        coverage[("w047", "w046")],
    ]
    separation = min(positives) / max(controls)
    assert min(positives) > 0.75
    assert max(controls) < 0.05
    assert separation > 15.0

    for window in ("w044", "w045", "w046", "w047"):
        artifacts = [
            ROOT / "legacy" / run / window / "overlapping.json"
            for run in ("run-baseline", "run-report-w1", "run-report-w4")
        ]
        assert len({artifact.read_bytes() for artifact in artifacts}) == 1

    print(
        "PASS: positive min={:.6%}, control max={:.6%}, separation={:.2f}x".format(
            min(positives), max(controls), separation
        )
    )
    print(f"PASS: worker reports identical ({REPORT_SHA256})")
    print("PASS: legacy overlapping.json outputs identical")


if __name__ == "__main__":
    main()
