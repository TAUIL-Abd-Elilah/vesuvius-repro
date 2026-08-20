from __future__ import annotations

import copy
import gc
import json
import weakref
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
    loaded = []

    def load(path: Path, block: dict, manifest_hash: str) -> dict:
        loaded.append(block["block_id"])
        return {"verified": block["block_id"]}

    monkeypatch.setattr(B.P, "_load_block_arrays", load)
    array_refs = B.verify_completed_inference(tmp_path, manifest, {})
    assert len(array_refs) == 64
    array_path, array_sha256 = array_refs["block-00"]
    assert array_path == tmp_path / "arrays" / "block-00.npz"
    assert array_sha256 == B.P.sha256_file(array_path)
    assert loaded == [f"block-{index:02d}" for index in range(64)]


def test_streaming_matches_legacy_masks_audits_and_arm_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = B.BridgeSplitConfig(
        low_threshold=0.2,
        persistence_threshold=0.4,
        seed_threshold=0.6,
        cut_ceiling=0.35,
        min_seed_voxels=8,
        min_output_component_voxels=16,
        max_removed_fraction=0.10,
    )
    monkeypatch.setattr(B, "CANDIDATE_CONFIGS", {"test_candidate": config})

    def bridged(bridge_probability: float) -> np.ndarray:
        probability = np.zeros((15, 15, 25), dtype=np.float32)
        probability[4:11, 4:11, 2:8] = 0.9
        probability[4:11, 4:11, 17:23] = 0.9
        probability[7, 7, 8:17] = bridge_probability
        return probability

    blocks = [
        {"block_id": "persistent", "scroll": "PHerc1203", "z_stratum": 1},
        {"block_id": "empty", "scroll": "PHerc0139", "z_stratum": 0},
        {"block_id": "weak", "scroll": "PHerc0139", "z_stratum": 2},
    ]
    probabilities = {
        "weak": bridged(0.25),
        "persistent": bridged(0.5),
        "empty": np.zeros((15, 15, 25), dtype=np.float32),
    }
    arrays = {
        block_id: {"corrected": probability.copy()}
        for block_id, probability in probabilities.items()
    }
    paths = {block_id: tmp_path / f"{block_id}.npz" for block_id in probabilities}
    for block_id, path in paths.items():
        path.write_bytes(block_id.encode("ascii"))
    array_refs = {
        block_id: (path, B.P.sha256_file(path)) for block_id, path in paths.items()
    }

    def summarize(
        selected_blocks: list[dict],
        masks: dict,
        audits: dict,
        labels_root: Path,
    ) -> list[dict]:
        del labels_root
        return sorted(
            [
                {
                    "block_id": block["block_id"],
                    "scroll": block["scroll"],
                    "z_stratum": block["z_stratum"],
                    "arms": {
                        name: {"positive_voxels": int(np.count_nonzero(mask))}
                        for name, mask in masks[block["block_id"]].items()
                    },
                    "audits": audits[block["block_id"]],
                }
                for block in selected_blocks
            ],
            key=lambda row: row["block_id"],
        )

    legacy_masks, legacy_audits = B.build_masks(
        blocks, arrays, B.CANDIDATE_CONFIGS
    )
    legacy_rows = summarize(blocks, legacy_masks, legacy_audits, tmp_path)

    path_to_block = {path: block_id for block_id, path in paths.items()}

    def load(path: Path, block: dict, manifest_hash: str) -> dict:
        assert manifest_hash == B.SOURCE_MANIFEST_CONTENT_SHA256
        assert path_to_block[path] == block["block_id"]
        return {"corrected": probabilities[block["block_id"]].copy()}

    monkeypatch.setattr(B.P, "_load_block_arrays", load)
    monkeypatch.setattr(B, "score_blocks", summarize)
    streaming_rows = B.score_blocks_streaming(
        blocks, array_refs, B.CANDIDATE_CONFIGS, tmp_path
    )

    assert B.P.canonical_json(streaming_rows) == B.P.canonical_json(legacy_rows)
    expected_arms = ["corrected_fixed"]
    for candidate in B.CANDIDATE_CONFIGS:
        expected_arms.extend([candidate, B.matched_budget_arm(candidate)])
    assert list(streaming_rows[0]["arms"]) == expected_arms
    audits_by_id = {row["block_id"]: row["audits"]["test_candidate"] for row in streaming_rows}
    assert audits_by_id["weak"]["accepted_components"] == 1
    assert audits_by_id["weak"]["removed_voxels"] > 0
    assert audits_by_id["persistent"]["accepted_components"] == 0
    assert audits_by_id["persistent"]["removed_voxels"] == 0
    assert audits_by_id["empty"]["accepted_components"] == 0
    assert audits_by_id["empty"]["removed_voxels"] == 0


def test_physical_counts_are_identical_when_blocks_are_scored_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocks = [
        {
            "block_id": "block-b",
            "scroll": "PHerc1203",
            "z_stratum": 1,
            "geometry": {
                "score_local_l1": [1, 3, 0, 1, 0, 1],
                "prediction_extent_local_l1": [1, 3, 0, 1, 0, 1],
            },
        },
        {
            "block_id": "block-a",
            "scroll": "PHerc0139",
            "z_stratum": 0,
            "geometry": {
                "score_local_l1": [0, 2, 0, 1, 0, 1],
                "prediction_extent_local_l1": [0, 2, 0, 1, 0, 1],
            },
        },
    ]
    masks = {
        block["block_id"]: {
            "corrected_fixed": np.array([[[True]], [[False]]]),
            "candidate": np.array([[[True]], [[True]]]),
        }
        for block in blocks
    }
    audits = {block["block_id"]: {"candidate": {"removed_voxels": 1}} for block in blocks}

    class Store:
        def __getitem__(self, key: tuple) -> np.ndarray:
            return np.full((1, 1), int(key[0]), dtype=np.uint8)

    monkeypatch.setattr(B.P, "SCORE_SIZE_L1", 2)
    monkeypatch.setattr(B.P, "Z_SAMPLE_STEP", 1)
    monkeypatch.setattr(B.P, "_open_zarr", lambda path: Store())
    monkeypatch.setattr(B.P, "prepare_truth_plane", lambda raw: int(raw[0, 0]))
    monkeypatch.setattr(B.P, "blank_counts", lambda: {"value": 0})

    def score_plane(truth: int, mask: np.ndarray, *args: int) -> dict:
        del args
        return {"value": truth + int(np.count_nonzero(mask))}

    def add_counts(total: dict, value: dict) -> None:
        total["value"] += value["value"]

    monkeypatch.setattr(B.P, "score_plane", score_plane)
    monkeypatch.setattr(B.P, "add_counts", add_counts)
    monkeypatch.setattr(B.P, "metrics", lambda value: dict(value))

    grouped = B.score_blocks(blocks, masks, audits, tmp_path)
    separate = []
    for block in blocks:
        block_id = block["block_id"]
        separate.extend(
            B.score_blocks(
                [block], {block_id: masks[block_id]}, {block_id: audits[block_id]}, tmp_path
            )
        )
    separate.sort(key=lambda row: row["block_id"])

    assert B.P.canonical_json(separate) == B.P.canonical_json(grouped)


def test_streaming_releases_each_decoded_block_before_loading_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Loaded:
        pass

    blocks = [
        {"block_id": "block-c"},
        {"block_id": "block-a"},
        {"block_id": "block-b"},
    ]
    paths = {block["block_id"]: tmp_path / block["block_id"] for block in blocks}
    for block_id, path in paths.items():
        path.write_bytes(block_id.encode("ascii"))
    array_refs = {
        block_id: (path, B.P.sha256_file(path)) for block_id, path in paths.items()
    }
    references: list[weakref.ReferenceType] = []
    loaded_ids: list[str] = []

    def load(path: Path, block: dict, manifest_hash: str) -> Loaded:
        del path, manifest_hash
        gc.collect()
        if references:
            assert references[-1]() is None
        value = Loaded()
        references.append(weakref.ref(value))
        loaded_ids.append(block["block_id"])
        return value

    def build(
        selected_blocks: list[dict], selected_arrays: dict, candidates: tuple[str, ...]
    ) -> tuple[dict, dict]:
        block_id = selected_blocks[0]["block_id"]
        assert list(selected_arrays) == [block_id]
        assert isinstance(selected_arrays[block_id], Loaded)
        assert candidates == tuple(B.CANDIDATE_CONFIGS)
        return {block_id: {"corrected_fixed": object()}}, {block_id: {}}

    def score(
        selected_blocks: list[dict], masks: dict, audits: dict, labels_root: Path
    ) -> list[dict]:
        del masks, audits, labels_root
        return [{"block_id": selected_blocks[0]["block_id"]}]

    monkeypatch.setattr(B.P, "_load_block_arrays", load)
    monkeypatch.setattr(B, "build_masks", build)
    monkeypatch.setattr(B, "score_blocks", score)
    rows = B.score_blocks_streaming(
        blocks, array_refs, B.CANDIDATE_CONFIGS, tmp_path
    )
    gc.collect()

    assert loaded_ids == ["block-c", "block-a", "block-b"]
    assert [row["block_id"] for row in rows] == ["block-a", "block-b", "block-c"]
    assert all(reference() is None for reference in references)


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


def test_implementation_hash_is_line_ending_portable(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert B.canonical_lf_sha256(lf) == B.canonical_lf_sha256(crlf)
