from __future__ import annotations

import copy
import gc
import io
import json
import os
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


def grouped_block(block_id: str, scroll: str, z0: int, z_stratum: int = 0) -> dict:
    return {
        "block_id": block_id,
        "scroll": scroll,
        "z_stratum": z_stratum,
        "geometry": {"score_local_l1": [z0, z0 + 64, 0, 64, 0, 64]},
    }


def test_scroll_z0_jobs_are_an_exact_unfragmented_partition() -> None:
    blocks = [
        grouped_block("b", "PHerc1203", 64),
        grouped_block("d", "PHerc0139", 192),
        grouped_block("a", "PHerc0139", 64),
        grouped_block("c", "PHerc1203", 64),
    ]
    jobs = B.partition_blocks_by_scroll_z0(blocks)

    assert [key for key, _ in jobs] == [
        ("PHerc0139", 64),
        ("PHerc0139", 192),
        ("PHerc1203", 64),
    ]
    assert [[block["block_id"] for block in group] for _, group in jobs] == [
        ["a"],
        ["d"],
        ["b", "c"],
    ]
    assert sorted(block["block_id"] for _, group in jobs for block in group) == [
        "a",
        "b",
        "c",
        "d",
    ]
    with pytest.raises(ValueError, match="duplicate grouped block_id"):
        B.partition_blocks_by_scroll_z0([blocks[0], copy.deepcopy(blocks[0])])


def test_amendment04_failure_receipt_and_superseded_chain_are_bound() -> None:
    repo = Path(B.__file__).resolve().parent
    prior = B.validate_superseded_lock_chain(repo)
    receipt = B.validate_preoutcome_failure(repo)

    assert prior["content_sha256"] == B.SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256
    assert receipt["content_sha256"] == (
        "9e6c3d8776532fd0c6ffb1a205f5ff1375c7af4c92a62f9956c9eca60212b7b1"
    )
    assert receipt["development_scoring_started"] is True
    assert receipt["development_scoring_completed"] is None
    assert receipt["holdout_opened"] is None
    assert receipt["selected_candidate_inspected"] is False
    assert receipt["bridge_outcomes_seen"] is False


def test_amendment04_rejects_any_unlisted_scientific_change() -> None:
    repo = Path(B.__file__).resolve().parent
    prior = B.validate_superseded_lock_chain(repo)
    amended = copy.deepcopy(prior)
    amended["execution_revision"] = B.EXECUTION_REVISION
    amended["implementation_commit"] = "1" * 40
    amended["implementation_files_sha256"] = {"example": "2" * 64}
    amended["resource_amendment"] = {
        "allowed_top_level_differences_from_superseded_lock": list(
            B.ALLOWED_SUPERSEDED_LOCK_DIFFERENCES
        )
    }
    amended["content_sha256"] = B._content_sha(amended)
    B.validate_scientific_identity_with_superseded_lock(amended, prior)

    changed = copy.deepcopy(amended)
    changed["analysis"]["bootstrap_seed"] += 1
    changed["content_sha256"] = B._content_sha(changed)
    with pytest.raises(SystemExit, match="scientific field changed: analysis"):
        B.validate_scientific_identity_with_superseded_lock(changed, prior)

    missing = copy.deepcopy(amended)
    del missing["status"]
    missing["content_sha256"] = B._content_sha(missing)
    with pytest.raises(SystemExit, match="top-level lock schema differs"):
        B.validate_scientific_identity_with_superseded_lock(missing, prior)


def scoring_job_multiset(blocks: list[dict]) -> Counter:
    jobs = Counter()
    for block in blocks:
        z0 = block["geometry"]["score_local_l1"][0]
        for k in range(B.P.SCORE_SIZE_L1):
            if (z0 + k) % B.P.Z_SAMPLE_STEP == 0:
                jobs[(block["block_id"], block["scroll"], z0 + k, k)] += 1
    return jobs


def test_frozen_split_group_jobs_preserve_exact_multiset_and_plane_counts() -> None:
    repo = Path(__file__).resolve().parent
    lock = B.load_hashed_json(
        repo / "results" / "physical_bridge_split" / "protocol_lock_amendment_01.json"
    )
    manifest = B.load_hashed_json(repo / lock["source_manifest_path"])

    for split, expected_unique_planes in (("dev", 304), ("holdout", 288)):
        blocks = B._blocks_for_split(manifest, lock, split)
        grouped_jobs = Counter()
        grouped_planes = set()
        for (scroll, z0), group in B.partition_blocks_by_scroll_z0(blocks):
            grouped_jobs.update(scoring_job_multiset(group))
            planes = {
                (scroll, z0 + k)
                for k in range(B.P.SCORE_SIZE_L1)
                if (z0 + k) % B.P.Z_SAMPLE_STEP == 0
            }
            assert len(planes) == 16
            assert grouped_planes.isdisjoint(planes)
            grouped_planes.update(planes)

        legacy_jobs = scoring_job_multiset(blocks)
        assert grouped_jobs == legacy_jobs
        assert sum(grouped_jobs.values()) == 32 * 16
        assert len(grouped_planes) == expected_unique_planes


def fake_group_row(block: dict, candidates: list[str]) -> dict:
    arms = {"corrected_fixed": {}}
    audits = {}
    for candidate in candidates:
        control = B.matched_budget_arm(candidate)
        arms[candidate] = {}
        arms[control] = {}
        audits[candidate] = {}
        audits[control] = {}
    return {
        "block_id": block["block_id"],
        "scroll": block["scroll"],
        "z_stratum": block["z_stratum"],
        "arms": arms,
        "audits": audits,
    }


def current_worker_lock() -> dict:
    repo = Path(B.__file__).resolve().parent
    return {
        "content_sha256": "a" * 64,
        "implementation_commit": B.git_output(repo, "rev-parse", "HEAD"),
        "implementation_files_sha256": {
            name: B.canonical_lf_sha256(repo / name) for name in B.IMPLEMENTATION_FILES
        },
        "resource_amendment": {
            "implementation_binary_files_sha256": B.implementation_binary_hashes(repo)
        },
    }


def test_group_scorer_loads_all_arrays_then_builds_and_scores_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    blocks = [
        grouped_block("shared-b", "PHerc0139", 64),
        grouped_block("shared-a", "PHerc0139", 64),
    ]
    array_refs = {}
    for block in blocks:
        path = tmp_path / f"{block['block_id']}.npz"
        path.write_bytes(block["block_id"].encode("ascii"))
        array_refs[block["block_id"]] = (path, B.P.sha256_file(path))
    loaded: list[str] = []
    calls = {"build": 0, "score": 0}

    def load(path: Path, block: dict, manifest_hash: str) -> dict:
        assert path == array_refs[block["block_id"]][0]
        assert manifest_hash == B.SOURCE_MANIFEST_CONTENT_SHA256
        loaded.append(block["block_id"])
        return {"corrected": block["block_id"]}

    def build(selected_blocks: list[dict], arrays: dict, candidates: tuple[str, ...]):
        calls["build"] += 1
        assert selected_blocks == blocks
        assert list(arrays) == ["shared-b", "shared-a"]
        assert loaded == ["shared-b", "shared-a"]
        assert candidates == (candidate,)
        return (
            {block["block_id"]: {"corrected_fixed": object()} for block in blocks},
            {block["block_id"]: {} for block in blocks},
        )

    def score(selected_blocks: list[dict], masks: dict, audits: dict, labels: Path):
        calls["score"] += 1
        assert selected_blocks == blocks
        assert set(masks) == {"shared-a", "shared-b"}
        assert set(audits) == {"shared-a", "shared-b"}
        assert labels == tmp_path
        return [
            fake_group_row(block, [candidate])
            for block in sorted(blocks, key=lambda item: item["block_id"])
        ]

    monkeypatch.setattr(B.P, "_load_block_arrays", load)
    monkeypatch.setattr(B, "build_masks", build)
    monkeypatch.setattr(B, "score_blocks", score)
    rows = B.score_block_group(blocks, array_refs, [candidate], tmp_path)

    assert calls == {"build": 1, "score": 1}
    assert [row["block_id"] for row in rows] == ["shared-a", "shared-b"]


def test_group_ipc_rejects_rehashed_schema_and_binding_changes(tmp_path: Path) -> None:
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    block = grouped_block("block-a", "PHerc0139", 64)
    array_path = tmp_path / "block-a.npz"
    array_path.write_bytes(b"frozen")
    request = B._worker_request(
        ("PHerc0139", 64),
        [block],
        {"block-a": (array_path, B.P.sha256_file(array_path))},
        (candidate,),
        tmp_path,
        current_worker_lock(),
    )
    B._validate_worker_request(request)

    extra = copy.deepcopy(request)
    extra["unexpected"] = True
    extra["content_sha256"] = B._content_sha(extra)
    with pytest.raises(ValueError, match="unexpected schema"):
        B._validate_worker_request(extra)

    launched_pid = os.getpid() + 100_000
    response = B._worker_response(request, [fake_group_row(block, [candidate])])
    response["worker_pid"] = launched_pid
    response["content_sha256"] = B._content_sha(response)
    assert B._validate_worker_response(response, request, launched_pid) == response["rows"]

    rebound = copy.deepcopy(response)
    rebound["request_content_sha256"] = "0" * 64
    rebound["content_sha256"] = B._content_sha(rebound)
    with pytest.raises(SystemExit, match="request_content_sha256 mismatch"):
        B._validate_worker_response(rebound, request, launched_pid)

    wrong_lock = copy.deepcopy(response)
    wrong_lock["protocol_lock_content_sha256"] = "b" * 64
    wrong_lock["content_sha256"] = B._content_sha(wrong_lock)
    with pytest.raises(SystemExit, match="protocol_lock_content_sha256 mismatch"):
        B._validate_worker_response(wrong_lock, request, launched_pid)

    wrong_binary = copy.deepcopy(response)
    wrong_binary["implementation_binary_files_sha256"][B.BOUNDED_WATERSHED_BINARY] = (
        "0" * 64
    )
    wrong_binary["content_sha256"] = B._content_sha(wrong_binary)
    with pytest.raises(
        SystemExit,
        match="implementation_binary_files_sha256 mismatch",
    ):
        B._validate_worker_response(wrong_binary, request, launched_pid)

    drifted_request = copy.deepcopy(request)
    drifted_request["implementation_binary_files_sha256"][B.BOUNDED_WATERSHED_BINARY] = (
        "0" * 64
    )
    drifted_request["content_sha256"] = B._content_sha(drifted_request)
    with pytest.raises(SystemExit, match="implementation binary drift"):
        B._verify_worker_implementation_bindings(drifted_request)

    corrupted = copy.deepcopy(response)
    corrupted["rows"][0]["block_id"] = "other"
    with pytest.raises(SystemExit, match="content SHA mismatch"):
        B._validate_worker_response(corrupted, request, launched_pid)


@pytest.mark.parametrize(
    ("stdout", "message"),
    [
        ("", "returned no response"),
        ("not-json", "returned malformed JSON"),
        ("[]", "returned a non-object response"),
        ("{}\n", "returned noncanonical JSON"),
        ("{}", "response has an unexpected schema"),
    ],
)
def test_pipe_ipc_rejects_missing_malformed_or_noncanonical_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    message: str,
) -> None:
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    block = grouped_block("block-a", "PHerc0139", 64)
    array_path = tmp_path / "block-a.npz"
    array_path.write_bytes(b"frozen")

    def launch(request: dict) -> dict:
        assert request["protocol_lock_content_sha256"] == "a" * 64
        return {
            "pid": os.getpid() + 30_000,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
        }

    monkeypatch.setattr(B, "_launch_group_worker", launch)
    with pytest.raises(SystemExit, match=message):
        B.score_blocks_spawned(
            [block],
            {"block-a": (array_path, B.P.sha256_file(array_path))},
            [candidate],
            tmp_path,
            current_worker_lock(),
        )


def test_worker_rechecks_implementation_after_computation_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    block = grouped_block("block-a", "PHerc0139", 64)
    array_path = tmp_path / "block-a.npz"
    array_path.write_bytes(b"frozen")
    request = B._worker_request(
        ("PHerc0139", 64),
        [block],
        {"block-a": (array_path, B.P.sha256_file(array_path))},
        (candidate,),
        tmp_path,
        current_worker_lock(),
    )
    checks = 0

    def verify(value: dict) -> None:
        nonlocal checks
        assert value == request
        checks += 1
        if checks == 2:
            raise SystemExit("group worker implementation drift: physical_bridge_split.py")

    monkeypatch.setattr(B, "_verify_worker_implementation_bindings", verify)
    monkeypatch.setattr(B.os, "getppid", lambda: request["parent_pid"])
    monkeypatch.setattr(
        B,
        "score_block_group",
        lambda blocks, refs, candidates, labels: [fake_group_row(block, [candidate])],
    )
    stdin = io.StringIO(B.P.canonical_json(request))
    stdout = io.StringIO()
    monkeypatch.setattr(B.sys, "stdin", stdin)
    monkeypatch.setattr(B.sys, "stdout", stdout)

    with pytest.raises(SystemExit, match="implementation drift"):
        B.group_worker_command(SimpleNamespace())

    assert checks == 2
    assert stdout.getvalue() == ""


def test_result_boundary_reverifies_lock_manifest_implementation_and_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "protocol_lock.json"
    expected_lock_sha = "1" * 64
    expected_manifest_sha = "2" * 64
    expected_head = "3" * 40
    lock = {"content_sha256": expected_lock_sha}
    calls: list[str] = []

    def load(path: Path) -> dict:
        assert path == lock_path
        calls.append("load")
        return lock

    def verify_files(repo: Path, value: dict) -> dict:
        assert repo == tmp_path
        assert value is lock
        calls.append("files")
        return {"content_sha256": expected_manifest_sha}

    def verify_head(repo: Path, value: dict, path: Path) -> str:
        assert (repo, value, path) == (tmp_path, lock, lock_path)
        calls.append("head")
        return expected_head

    monkeypatch.setattr(B, "load_hashed_json", load)
    monkeypatch.setattr(B, "verify_protocol_files", verify_files)
    monkeypatch.setattr(B, "verify_public_freeze", verify_head)
    B.reverify_before_result_write(
        tmp_path,
        lock_path,
        expected_lock_sha,
        expected_manifest_sha,
        expected_head,
    )
    assert calls == ["load", "files", "head"]

    monkeypatch.setattr(B, "verify_public_freeze", lambda repo, value, path: "4" * 40)
    with pytest.raises(SystemExit, match="public freeze HEAD changed"):
        B.reverify_before_result_write(
            tmp_path,
            lock_path,
            expected_lock_sha,
            expected_manifest_sha,
            expected_head,
        )


def test_spawned_jobs_are_serial_and_use_a_fresh_pid_per_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    blocks = [
        grouped_block("same-z-b", "PHerc1203", 64),
        grouped_block("next-z", "PHerc1203", 192),
        grouped_block("same-z-a", "PHerc1203", 64),
    ]
    array_refs = {}
    for block in blocks:
        path = tmp_path / f"{block['block_id']}.npz"
        path.write_bytes(block["block_id"].encode("ascii"))
        array_refs[block["block_id"]] = (path, B.P.sha256_file(path))

    active = 0
    max_active = 0
    pids: list[int] = []
    requested_groups: list[tuple[tuple[str, int], list[str]]] = []

    def launch(request: dict) -> dict:
        nonlocal active, max_active
        assert active == 0
        active += 1
        max_active = max(max_active, active)
        group, request_blocks, _, candidates, _ = B._validate_worker_request(request)
        pid = os.getpid() + 10_000 + len(pids)
        pids.append(pid)
        requested_groups.append((group, list(request["block_ids"])))
        rows = [
            fake_group_row(block, list(candidates))
            for block in sorted(request_blocks, key=lambda item: item["block_id"])
        ]
        response = B._worker_response(request, rows)
        response["worker_pid"] = pid
        response["content_sha256"] = B._content_sha(response)
        active -= 1
        return {
            "pid": pid,
            "returncode": 0,
            "stdout": B.P.canonical_json(response),
            "stderr": "",
        }

    monkeypatch.setattr(B, "_launch_group_worker", launch)
    rows = B.score_blocks_spawned(
        blocks, array_refs, [candidate], tmp_path, current_worker_lock()
    )

    assert max_active == 1
    assert len(pids) == len(set(pids)) == 2
    assert requested_groups == [
        (("PHerc1203", 64), ["same-z-b", "same-z-a"]),
        (("PHerc1203", 192), ["next-z"]),
    ]
    assert [row["block_id"] for row in rows] == ["next-z", "same-z-a", "same-z-b"]


def test_failed_child_stops_before_next_group_without_disk_ipc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    blocks = [
        grouped_block("first", "PHerc0139", 64),
        grouped_block("never-started", "PHerc0139", 192),
    ]
    array_refs = {}
    for block in blocks:
        path = tmp_path / f"{block['block_id']}.npz"
        path.write_bytes(block["block_id"].encode("ascii"))
        array_refs[block["block_id"]] = (path, B.P.sha256_file(path))
    calls = 0

    def fail(request: dict) -> dict:
        nonlocal calls
        assert request["kind"] == B.GROUP_WORKER_REQUEST_KIND
        calls += 1
        return {"pid": os.getpid() + 20_000, "returncode": 9, "stdout": "", "stderr": "boom"}

    monkeypatch.setattr(B, "_launch_group_worker", fail)
    with pytest.raises(SystemExit, match="failed with exit 9"):
        B.score_blocks_spawned(
            blocks, array_refs, [candidate], tmp_path, current_worker_lock()
        )

    assert calls == 1
    assert "tempfile" not in B.__dict__


def test_grouped_child_rows_equal_separate_legacy_rows_for_shared_z_group(
    tmp_path: Path,
) -> None:
    labels_root = tmp_path / "labels"
    labels_root.mkdir()
    for config in B.P.SCROLLS.values():
        label_path = labels_root / config["label_store"]
        B.zarr.open(
            str(label_path),
            mode="w",
            shape=(64, 144, 160),
            chunks=(1, 144, 160),
            dtype="u1",
            fill_value=1,
        )
    candidate = next(iter(B.CANDIDATE_CONFIGS))
    blocks = []
    array_refs = {}
    for block_id, score_x0, extent_x0 in (
        ("shared-z-left", 8, 0),
        ("shared-z-right", 88, 80),
    ):
        extent = [0, 64, 0, 144, extent_x0, extent_x0 + 80]
        block = {
            "block_id": block_id,
            "scroll": "PHerc0139",
            "z_stratum": 0,
            "geometry": {
                "score_local_l1": [0, 64, 72, 136, score_x0, score_x0 + 64],
                "prediction_extent_local_l1": extent,
                "prediction_extent_global_l1": extent,
            },
        }
        blocks.append(block)
        array_path = tmp_path / f"{block_id}.npz"
        shape = (
            B.P.SCORE_SIZE_L1,
            B.P.SCORE_SIZE_L1 + B.P.NULL_SHIFT_L1 + 2 * B.P.METRIC_HALO_L1,
            B.P.SCORE_SIZE_L1 + 2 * B.P.METRIC_HALO_L1,
        )
        metadata = {
            "schema_version": 1,
            "manifest_content_sha256": B.SOURCE_MANIFEST_CONTENT_SHA256,
            "block_id": block_id,
            "prediction_extent_global_l1": extent,
        }
        corrected = np.zeros(shape, dtype=np.float32)
        corrected[20:40, 70:90, 10:30] = 0.9
        corrected[20:40, 70:90, 50:70] = 0.9
        corrected[30, 80, 30:51] = 0.25
        np.savez_compressed(
            array_path,
            baseline_l1=np.zeros(shape, dtype=np.uint8),
            corrected_pmax_l1=corrected,
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        array_refs[block_id] = (array_path, B.P.sha256_file(array_path))

    separate_legacy = B.score_blocks_streaming(
        blocks, array_refs, [candidate], labels_root
    )
    grouped_child = B.score_blocks_spawned(
        blocks, array_refs, [candidate], labels_root, current_worker_lock()
    )

    assert B.P.canonical_json(grouped_child) == B.P.canonical_json(separate_legacy)


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
