from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import physical_bridge_split as B
import physical_normalization_ab as P


def source_manifest() -> dict:
    repo = Path(__file__).resolve().parent
    return B.load_hashed_json(repo / "results" / "physical_normalization_ab" / "manifest.json")


def fake_completed_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    blocks = []
    records = []
    for index in range(64):
        block_id = f"block-{index:02d}"
        block = {
            "block_id": block_id,
            "array_file": f"arrays/{block_id}.npz",
            "receipt_file": f"receipts/{block_id}.json",
        }
        blocks.append(block)
        array_path = tmp_path / block["array_file"]
        array_path.parent.mkdir(parents=True, exist_ok=True)
        array_path.write_bytes(f"array-{index}".encode("ascii"))
        array_sha = B.P.sha256_file(array_path)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "block_id": block_id,
            "attempt": 1,
            "public_freeze_commit": B.SOURCE_PUBLIC_FREEZE_COMMIT,
            "manifest_content_sha256": B.SOURCE_MANIFEST_CONTENT_SHA256,
            "corrected_villa_commit": B.P.PR1386_COMMIT,
            "array_file_sha256": array_sha,
            "inference": {
                "predict_returncode": 0,
                "blend_returncode": 0,
                "required_normalization_log_token": B.R.MODEL_NORMALIZATION_LOG_TOKEN,
                "required_normalization_log_token_found": True,
            },
        }
        receipt_path = tmp_path / block["receipt_file"]
        B.P.write_json(receipt_path, receipt)
        receipt_sha = B.P.sha256_file(receipt_path)
        cleanup = {
            "schema_version": 1,
            "protocol_id": B.R.PROTOCOL_ID,
            "implementation_revision": B.R.IMPLEMENTATION_REVISION,
            "block_id": block_id,
            "attempt": 1,
            "final_receipt_sha256": receipt_sha,
            "array_file_sha256": array_sha,
            "text_logs_retained": True,
        }
        cleanup_path = tmp_path / "cleanup" / f"{block_id}.json"
        B.P.write_json(cleanup_path, cleanup)
        records.append(
            {
                "block_id": block_id,
                "array_sha256": array_sha,
                "receipt_sha256": receipt_sha,
                "cleanup_sha256": B.P.sha256_file(cleanup_path),
            }
        )
    completion = {
        "schema_version": 1,
        "protocol_id": B.R.PROTOCOL_ID,
        "implementation_revision": B.R.IMPLEMENTATION_REVISION,
        "status": "inference_complete_unscored",
        "causal_claim_allowed": False,
        "protocol_lock_content_sha256": B.SOURCE_PROTOCOL_LOCK_CONTENT_SHA256,
        "source_manifest_content_sha256": B.SOURCE_MANIFEST_CONTENT_SHA256,
        "public_freeze_commit": B.SOURCE_PUBLIC_FREEZE_COMMIT,
        "completed_blocks": 64,
        "records": records,
    }
    B.P.write_json(tmp_path / "inference_completion_receipt.json", completion)
    monkeypatch.setattr(
        B.P,
        "_load_block_arrays",
        lambda path, block, manifest_hash: {"verified": block["block_id"]},
    )
    return {"blocks": blocks}


def test_split_is_deterministic_balanced_and_disjoint() -> None:
    manifest = source_manifest()
    first = B.split_assignments(manifest)
    second = B.split_assignments(copy.deepcopy(manifest))
    assert first == second
    assert len(first["dev"]) == 32
    assert len(first["holdout"]) == 32
    dev_ids = {item["block_id"] for item in first["dev"]}
    holdout_ids = {item["block_id"] for item in first["holdout"]}
    assert dev_ids.isdisjoint(holdout_ids)
    assert dev_ids | holdout_ids == {block["block_id"] for block in manifest["blocks"]}
    for split in ("dev", "holdout"):
        counts = Counter((item["scroll"], item["z_stratum"]) for item in first[split])
        assert set(counts.values()) == {4}
        assert Counter(item["scroll"] for item in first[split]) == {
            "PHerc0139": 16,
            "PHerc1203": 16,
        }


def test_frozen_candidate_grid_is_valid_and_has_one_low_threshold() -> None:
    assert len(B.CANDIDATE_CONFIGS) == 3
    assert {config.low_threshold for config in B.CANDIDATE_CONFIGS.values()} == {0.2}
    for config in B.CANDIDATE_CONFIGS.values():
        config.validate(3)


def comparison(
    point: float,
    *,
    far: float = 0.0,
    recall: float = 0.0,
    edited_total: int = 4,
    edited_each: int = 2,
    removed: int = 20,
    ci_low: float | None = None,
) -> dict:
    return {
        "pooled_point_skill_delta": {
            "mean": point,
            "ci95_low": point / 2 if ci_low is None else ci_low,
            "ci95_high": point * 1.5,
        },
        "by_scroll": {
            scroll: {
                "point_skill_delta": {"mean": point},
                "far37_fraction_delta_macro_mean": far,
                "recall37_delta_macro_mean": recall,
                "edited_blocks": edited_each,
                "point_blocks": 16,
            }
            for scroll in P.SCROLLS
        },
        "edited_blocks_total": edited_total,
        "removed_voxels_total": removed,
    }


def comparison_bundle(fixed: dict, matched: dict | None = None) -> dict:
    return {
        "vs_fixed": fixed,
        "vs_matched_budget": copy.deepcopy(fixed) if matched is None else matched,
    }


def test_development_selects_largest_eligible_point_gain() -> None:
    names = list(B.CANDIDATE_CONFIGS)
    values = {
        names[0]: comparison_bundle(comparison(0.01)),
        names[1]: comparison_bundle(comparison(0.02)),
        names[2]: comparison_bundle(comparison(-0.01)),
    }
    selected, gates = B.select_development_candidate(values)
    assert selected == names[1]
    assert gates[names[0]]["eligible"] is True
    assert gates[names[2]]["eligible"] is False


def test_development_ties_prefer_far_then_edits_then_name() -> None:
    names = list(B.CANDIDATE_CONFIGS)
    values = {
        names[0]: comparison_bundle(comparison(0.01, far=0.001, removed=10)),
        names[1]: comparison_bundle(comparison(0.01, far=0.0, removed=30)),
        names[2]: comparison_bundle(comparison(0.01, far=0.0, removed=20)),
    }
    selected, _ = B.select_development_candidate(values)
    assert selected == names[2]


@pytest.mark.parametrize(
    "value",
    [
        comparison(0.0),
        comparison(0.01, far=B.FAR37_MARGIN + 1e-6),
        comparison(0.01, recall=B.RECALL37_MARGIN - 1e-6),
        comparison(0.01, edited_total=3),
        comparison(0.01, edited_each=0),
    ],
)
def test_development_gate_failures_do_not_unlock_holdout(value: dict) -> None:
    values = {
        name: comparison_bundle(copy.deepcopy(value)) for name in B.CANDIDATE_CONFIGS
    }
    selected, gates = B.select_development_candidate(values)
    assert selected is None
    assert not any(item["eligible"] for item in gates.values())


def test_holdout_gate_requires_positive_ci_and_every_conjunct() -> None:
    passed = B.holdout_gates(
        comparison_bundle(comparison(0.02, ci_low=0.001))
    )
    assert passed["primary_claim_passes"] is True
    failed = B.holdout_gates(comparison_bundle(comparison(0.02, ci_low=0.0)))
    assert failed["pooled_point_ci_low_positive"] is False
    assert failed["primary_claim_passes"] is False


def test_compare_candidate_uses_all_scroll_strata_deterministically() -> None:
    rows = []
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    for scroll in P.SCROLLS:
        for z_stratum in range(P.Z_STRATA):
            for index in range(4):
                rows.append(
                    {
                        "scroll": scroll,
                        "z_stratum": z_stratum,
                        "arms": {
                            "corrected_fixed": {
                                "point_skill": 0.10,
                                "pred_far37_fraction": 0.05,
                                "recall_37um": 0.80,
                                "arc_fully_missed": 0.10,
                            },
                            candidate: {
                                "point_skill": 0.11,
                                "pred_far37_fraction": 0.049,
                                "recall_37um": 0.80,
                                "arc_fully_missed": 0.09,
                            },
                        },
                        "audits": {candidate: {"removed_voxels": 1 if index == 0 else 0}},
                    }
                )
    first = B.compare_candidate(rows, candidate, "corrected_fixed", 123)
    second = B.compare_candidate(rows, candidate, "corrected_fixed", 123)
    assert first == second
    assert first["pooled_point_skill_delta"]["n"] == 32
    assert first["pooled_point_skill_delta"]["groups"] == 8
    assert first["pooled_point_skill_delta"]["mean"] == pytest.approx(0.01)
    assert first["edited_blocks_total"] == 8


def test_matched_budget_control_is_exact_and_breaks_ties_by_flat_index() -> None:
    probability = np.array(
        [[[0.9, 0.8, 0.8], [0.7, 0.2, 0.1]]], dtype=np.float32
    )
    control, audit = B.probability_rank_mass_control(probability, 2, 0.2)
    expected = np.array(
        [[[True, True, False], [False, False, False]]], dtype=bool
    )
    assert np.array_equal(control, expected)
    assert audit["target_positive_voxels"] == 2
    assert audit["realized_positive_voxels"] == 2
    assert audit["uses_labels"] is False


def test_matched_budget_failure_blocks_development_and_holdout() -> None:
    fixed = comparison(0.02, ci_low=0.001)
    matched = comparison(-0.001, ci_low=-0.002)
    values = {
        name: comparison_bundle(copy.deepcopy(fixed), copy.deepcopy(matched))
        for name in B.CANDIDATE_CONFIGS
    }
    selected, gates = B.select_development_candidate(values)
    assert selected is None
    assert not any(item["eligible"] for item in gates.values())
    holdout = B.holdout_gates(comparison_bundle(fixed, matched))
    assert holdout["matched_pooled_point_ci_low_positive"] is False
    assert holdout["primary_claim_passes"] is False


def test_completed_inference_verifies_all_64_bound_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fake_completed_inference(tmp_path, monkeypatch)
    arrays = B.verify_completed_inference(tmp_path, manifest, {})
    assert len(arrays) == 64
    assert arrays["block-00"] == {"verified": "block-00"}


def test_completed_inference_rejects_rehashed_missing_normalization_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = fake_completed_inference(tmp_path, monkeypatch)
    block = manifest["blocks"][0]
    receipt_path = tmp_path / block["receipt_file"]
    cleanup_path = tmp_path / "cleanup" / f"{block['block_id']}.json"
    completion_path = tmp_path / "inference_completion_receipt.json"

    receipt = B.P.load_json(receipt_path)
    receipt["inference"]["required_normalization_log_token_found"] = False
    B.P.write_json(receipt_path, receipt)
    cleanup = B.P.load_json(cleanup_path)
    cleanup["final_receipt_sha256"] = B.P.sha256_file(receipt_path)
    B.P.write_json(cleanup_path, cleanup)
    completion = B.P.load_json(completion_path)
    completion["records"][0]["receipt_sha256"] = B.P.sha256_file(receipt_path)
    completion["records"][0]["cleanup_sha256"] = B.P.sha256_file(cleanup_path)
    B.P.write_json(completion_path, completion)

    with pytest.raises(SystemExit, match="final receipt binding mismatch"):
        B.verify_completed_inference(tmp_path, manifest, {})


def test_score_and_plan_refuse_existing_outputs(tmp_path: Path) -> None:
    output = tmp_path / "already-exists.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to overwrite existing bridge result"):
        B.score_command(SimpleNamespace(result=str(output)))
    with pytest.raises(SystemExit, match="refusing to overwrite existing protocol lock"):
        B.plan_command(SimpleNamespace(out=str(output)))


def test_hashed_json_rejects_tampering(tmp_path: Path) -> None:
    value = {"schema_version": 1, "status": "clean"}
    value["content_sha256"] = B._content_sha(value)
    path = tmp_path / "value.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert B.load_hashed_json(path)["status"] == "clean"
    value["status"] = "tampered"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SystemExit, match="content SHA mismatch"):
        B.load_hashed_json(path)
