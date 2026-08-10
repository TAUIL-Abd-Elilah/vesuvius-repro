"""Recompute and verify the compact PHerc0125 Spiral-quality release."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_native_spiral_quality as evaluator


HERE = Path(__file__).resolve().parent
REPORT = HERE / "results" / "PHerc0125_native_fit_quality.json"
LOSS = HERE / "results" / "PHerc0125_loss_summary.json"
MANIFEST = HERE / "manifest.json"

PINNED = {
    "evaluate_native_spiral_quality.py": "4b9fde37c131d99aefab04b4927b4e35f78cec8c8fb9372c568ad5b98e74f8a2",
    "test_evaluate_native_spiral_quality.py": "0f5e014a83707314672348b6455828badf9ce61ccec5a29266d6a296fb3f0b6f",
    "SPIRAL_PRIZE_SCROLL_NATIVE_FIT_PREREG.md": "a8af1607cdb930d13e2c26b1b1e5aaaca7d46a0abea1394234ab6ada9253b504",
    "SPIRAL_PRIZE_SCROLL_QUALITY_EVALUATION_PREREG.md": "137a3d78870f642c9599d29d1dd1421ae99c473fc88e5e43e39873ceff6643a9",
    "PHERC0125_QUALITY_ALIGNMENT_AMENDMENT.md": "6501ebd0c94f0c7ab6558bb19b02b34db8b1eac54a40c73212972a2fc94f540b",
    "PHERC0125_QUALITY_BLOCK_BATCH_AMENDMENT.md": "5bc4a12ec86e5e1c110986f0d96f874249d9635f36927e3f0db20db0ea2218a5",
    "PHERC0125_QUALITY_INTRINSIC_SCHEMA_AMENDMENT.md": "87e6f4a53efc9b69cb8bf302b6a3c9b5db7b504cd29de044e28304372a23ea4d",
    "PHERC0125_QUALITY_FINALIZATION_AMENDMENT.md": "9655e549315e1b5aca5b6a4c02e19d57dd76ffbb82a40719b7766104bc7ddd35",
    "results/PHerc0125_native_fit_quality.json": "a80995fce75d694e13a82e0afc2d36e29b70d8eee718f1c76a79b0d3087d7bec",
}

EXPECTED_DECISIONS = {
    "quantitative_ct_improvement_authorized": False,
    "pherc0211_execution_authorized": False,
    "public_accuracy_wording_authorized": False,
    "letters_or_reading_claim_authorized": False,
    "physical_winding_sense_claim_authorized": False,
    "prize_claim_authorized": False,
}


class ReleaseError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReleaseError(f"not a JSON object: {path}")
    return value


def equivalent(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, bool) or isinstance(right, bool):
        if type(left) is not type(right) or left != right:
            raise ReleaseError(f"value mismatch at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12):
            raise ReleaseError(f"numeric mismatch at {path}: {left!r} != {right!r}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ReleaseError(f"key mismatch at {path}")
        for key in sorted(left):
            equivalent(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ReleaseError(f"length mismatch at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            equivalent(left_item, right_item, path=f"{path}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise ReleaseError(f"value mismatch at {path}: {left!r} != {right!r}")


def verify_profiles(report: dict[str, Any]) -> int:
    rows = report.get("raw_samples")
    if not isinstance(rows, list) or len(rows) != 400:
        raise ReleaseError("expected exactly 400 registered paired rays")
    for index, row in enumerate(rows):
        for arm in ("baseline", "final"):
            value = row.get(arm)
            if not isinstance(value, dict) or not isinstance(value.get("profile"), list):
                raise ReleaseError(f"missing profile at row {index}/{arm}")
            profile = np.asarray(value["profile"], dtype=np.float32)
            actual = hashlib.sha256(profile.tobytes()).hexdigest()
            if actual != value.get("profile_sha256"):
                raise ReleaseError(f"profile hash mismatch at row {index}/{arm}")
            expected_usable = bool(np.count_nonzero(profile) > profile.size * 0.6)
            if value.get("profile_usable") is not expected_usable:
                raise ReleaseError(f"profile usability mismatch at row {index}/{arm}")
    return len(rows) * 2


def verify_loss_summary(report: dict[str, Any]) -> dict[str, Any]:
    loss = load_object(LOSS)
    if loss.get("format") != "pherc0125-native-fit-loss-summary-v1":
        raise ReleaseError("loss summary format mismatch")
    if loss.get("quality_report_sha256") != sha256_file(REPORT):
        raise ReleaseError("loss summary report hash mismatch")
    milestones = loss.get("milestones")
    if not isinstance(milestones, list) or [row.get("step") for row in milestones] != [1, 15000]:
        raise ReleaseError("loss milestone inventory mismatch")
    if any(row.get("warnings") != [] for row in milestones):
        raise ReleaseError("loss summary contains a runner warning")
    initial = float(milestones[0]["total_loss"])
    final = float(milestones[1]["total_loss"])
    expected_reduction = (1.0 - final / initial) * 100.0
    if not math.isclose(expected_reduction, float(loss["loss_reduction_percent"]), abs_tol=1e-12):
        raise ReleaseError("loss reduction arithmetic mismatch")
    if report.get("inputs", {}).get("baseline_evidence_sha256") != loss.get("source_evidence_sha256", {}).get("baseline"):
        raise ReleaseError("baseline evidence provenance mismatch")
    if report.get("inputs", {}).get("final_evidence_sha256") != loss.get("source_evidence_sha256", {}).get("final"):
        raise ReleaseError("final evidence provenance mismatch")
    return loss


def verify_no_machine_paths() -> None:
    # Split the sentinels so this verifier does not flag its own source text.
    forbidden = (
        "d:" + "\\competition",
        "c:" + "\\users\\",
        "d:" + "/competition",
        "c:" + "/users/",
    )
    for path in HERE.rglob("*"):
        if not path.is_file() or path.suffix.lower() in {".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except UnicodeDecodeError:
            continue
        if any(token in text for token in forbidden):
            raise ReleaseError(f"machine-local path leaked into {path.relative_to(HERE)}")


def verify_manifest() -> int:
    manifest = load_object(MANIFEST)
    if manifest.get("format") != "spiral-quality-release-manifest-v1":
        raise ReleaseError("release manifest format mismatch")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ReleaseError("release manifest has no file records")
    expected_paths = {
        path.relative_to(HERE).as_posix()
        for path in HERE.rglob("*")
        if path.is_file()
        and path != MANIFEST
        and "__pycache__" not in path.parts
        and path.suffix.lower() != ".pyc"
    }
    recorded_paths = {record.get("path") for record in records if isinstance(record, dict)}
    if recorded_paths != expected_paths or len(recorded_paths) != len(records):
        raise ReleaseError("release manifest file inventory mismatch")
    for record in records:
        if set(record) != {"path", "bytes", "sha256"}:
            raise ReleaseError("release manifest record schema mismatch")
        path = HERE / record["path"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ReleaseError(f"release manifest hash mismatch: {record['path']}")
    return len(records)


def verify() -> dict[str, Any]:
    manifest_files = verify_manifest()
    for relative, expected in PINNED.items():
        path = HERE / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ReleaseError(f"missing or changed pinned release file: {relative}")
    verify_no_machine_paths()
    report = load_object(REPORT)
    if not (
        report.get("format") == evaluator.FORMAT
        and report.get("complete") is True
        and report.get("scroll") == "PHerc0125"
    ):
        raise ReleaseError("report identity mismatch")
    if report.get("inputs", {}).get("evaluator_sha256") != PINNED["evaluate_native_spiral_quality.py"]:
        raise ReleaseError("report does not bind the released evaluator")
    profile_count = verify_profiles(report)
    recomputed_ct = evaluator.paired_ct_summary(report["raw_samples"])
    equivalent(recomputed_ct, report.get("ct"), path="ct")
    intrinsic = report.get("intrinsic", {})
    base = evaluator.normalize_intrinsic_report(intrinsic.get("baseline"), label="baseline")
    final = evaluator.normalize_intrinsic_report(intrinsic.get("final"), label="final")
    recomputed_intrinsic = evaluator.intrinsic_summary(base, final)
    equivalent(recomputed_intrinsic, intrinsic, path="intrinsic")
    no_ct_degradation = bool(
        recomputed_ct["all_measurements_sufficient"]
        and not recomputed_ct["any_adverse_interval"]
    )
    quantitative_improvement = bool(
        no_ct_degradation
        and recomputed_ct["any_favorable_interval"]
        and recomputed_intrinsic["no_material_regression"]
    )
    expected_decisions = dict(EXPECTED_DECISIONS)
    expected_decisions["quantitative_ct_improvement_authorized"] = quantitative_improvement
    expected_decisions["public_accuracy_wording_authorized"] = quantitative_improvement
    expected_decisions["pherc0211_execution_authorized"] = bool(
        no_ct_degradation and recomputed_intrinsic["no_material_regression"]
    )
    if report.get("decisions") != expected_decisions or expected_decisions != EXPECTED_DECISIONS:
        raise ReleaseError("decision gate does not reproduce the frozen all-false outcome")
    loss = verify_loss_summary(report)
    return {
        "complete": True,
        "manifest_files": manifest_files,
        "report_sha256": sha256_file(REPORT),
        "paired_profiles_rehashed": profile_count,
        "ct_recomputed": True,
        "intrinsic_recomputed": True,
        "all_authorizations_false": True,
        "loss_reduction_percent": loss["loss_reduction_percent"],
        "median_pitch_ratio": recomputed_intrinsic["deltas"]["median_pitch_ratio"],
    }


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
