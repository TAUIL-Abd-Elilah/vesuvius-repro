#!/usr/bin/env python3
"""Frozen dev/holdout evaluation for probability-aware surface bridge splitting.

The source probabilities are the 64 completed blocks from the public released-baseline
comparison.  This evaluator refuses partial inference, assigns four blocks per scroll/z stratum
to development and four to untouched holdout, evaluates the small frozen grid on development,
and scores at most one selected configuration on holdout.  Every candidate is also compared with
an exact-mass probability-ranking control so deletion alone cannot manufacture the claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import skimage
import zarr

import physical_normalization_ab as P
import run_physical_released_baseline_comparison as R
from probability_bridge_split import BridgeSplitConfig, split_probability_bridges


PROTOCOL_ID = "physical_probability_bridge_split_v1"
LOCK_STATUS = "preregistered_before_bridge_outcomes"
PUBLIC_BRANCH = "physical-multithreshold-repair"
EXECUTION_REVISION = 2
SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256 = (
    "fc54fc80ecd5c5c976804e709d2ac1e460c89ce4d3fc9d337542aca2435a3bb1"
)
PREOUTCOME_FAILURE_RECEIPT_PATH = (
    "results/physical_bridge_split/preoutcome_failure_01.json"
)
PREOUTCOME_FAILURE_STDERR_PATH = (
    "results/physical_bridge_split/preoutcome_failure_01.stderr.txt"
)
SOURCE_MANIFEST_CONTENT_SHA256 = (
    "567a18faa1c8ca7e743c9240133f4200e67e3085823dd4795c4518e3e0e65ac0"
)
SOURCE_PROTOCOL_LOCK_CONTENT_SHA256 = (
    "97368b402aa313ad6f24442671da0182302530d1ecb0479b0e40d174d855d8b9"
)
SOURCE_PROTOCOL_LOCK_PATH = "results/physical_released_baseline_comparison/protocol_lock.json"
SOURCE_PUBLIC_FREEZE_COMMIT = "c62fd475b6f7df716e828b3a774e304e7cf43176"
SPLIT_SEED = "physical-probability-bridge-split-v1-2026-08-11"
BOOTSTRAP_SEED = 20260812
BOOTSTRAP_DRAWS = 10_000
DEV_BLOCKS_PER_STRATUM = 4
HOLDOUT_BLOCKS_PER_STRATUM = 4
FAR37_MARGIN = 0.002
RECALL37_MARGIN = -0.002
MIN_EDITED_BLOCKS_TOTAL = 4
MIN_EDITED_BLOCKS_PER_SCROLL = 1


CANDIDATE_CONFIGS: dict[str, BridgeSplitConfig] = {
    "conservative_p45_s70": BridgeSplitConfig(
        persistence_threshold=0.45,
        seed_threshold=0.70,
        cut_ceiling=0.30,
        min_seed_voxels=128,
        min_output_component_voxels=512,
        max_removed_fraction=0.002,
    ),
    "balanced_p40_s65": BridgeSplitConfig(
        persistence_threshold=0.40,
        seed_threshold=0.65,
        cut_ceiling=0.35,
        min_seed_voxels=128,
        min_output_component_voxels=512,
        max_removed_fraction=0.005,
    ),
    "sensitive_p35_s60": BridgeSplitConfig(
        persistence_threshold=0.35,
        seed_threshold=0.60,
        cut_ceiling=0.30,
        min_seed_voxels=64,
        min_output_component_voxels=256,
        max_removed_fraction=0.010,
    ),
}


IMPLEMENTATION_FILES = (
    "PHYSICAL_BRIDGE_SPLIT_AMENDMENT_01.md",
    "PHYSICAL_BRIDGE_SPLIT_PREREG.md",
    "physical_bridge_split.py",
    "physical_normalization_ab.py",
    "probability_bridge_split.py",
    "requirements.txt",
    "results/physical_bridge_split/protocol_lock.json",
    PREOUTCOME_FAILURE_RECEIPT_PATH,
    PREOUTCOME_FAILURE_STDERR_PATH,
    SOURCE_PROTOCOL_LOCK_PATH,
    "run_physical_released_baseline_comparison.py",
    "test_physical_bridge_split.py",
    "test_probability_bridge_split.py",
)


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def _content_sha(value: dict[str, Any]) -> str:
    unhashed = {key: item for key, item in value.items() if key != "content_sha256"}
    return P.sha256_bytes(P.canonical_json(unhashed).encode("utf-8"))


def canonical_lf_bytes(path: Path) -> bytes:
    text = path.read_bytes().decode("utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return canonical.encode("utf-8")


def canonical_lf_sha256(path: Path) -> str:
    """Hash UTF-8 text with platform-independent LF line endings."""

    return P.sha256_bytes(canonical_lf_bytes(path))


def runtime_versions() -> dict[str, str]:
    """Return the exact runtime versions that can affect deterministic morphology."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-image": skimage.__version__,
        "zarr": zarr.__version__,
    }


def load_hashed_json(path: Path) -> dict[str, Any]:
    value = P.load_json(path)
    recorded = value.get("content_sha256")
    actual = _content_sha(value)
    if recorded != actual:
        raise SystemExit(f"{path}: content SHA mismatch: {recorded} != {actual}")
    return value


def validate_preoutcome_failure(repo: Path) -> dict[str, Any]:
    prior_lock = load_hashed_json(repo / "results/physical_bridge_split/protocol_lock.json")
    if prior_lock.get("content_sha256") != SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256:
        raise SystemExit("superseded bridge protocol lock content mismatch")
    receipt = load_hashed_json(repo / PREOUTCOME_FAILURE_RECEIPT_PATH)
    expected = {
        "schema_version": 1,
        "kind": "preoutcome_execution_failure",
        "protocol_id": PROTOCOL_ID,
        "public_head": "2bb4782dab6bff4c714db18b13f32ed3e4f91dff",
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "prior_protocol_lock_content_sha256": (
            SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256
        ),
        "result_path": "results/physical_bridge_split/result.json",
        "result_existed_after_exit": False,
        "failure_class": "MemoryError",
        "failure_phase": "build_masks/split_probability_bridges/skimage.watershed",
        "last_frame": "skimage/segmentation/heap_general.pxi:111 heappush",
        "development_scoring_started": False,
        "development_selection_started": False,
        "holdout_opened": False,
        "bridge_outcomes_seen": False,
        "partial_candidate_masks_may_have_existed_in_memory": True,
        "partial_candidate_masks_persisted": False,
        "partial_candidate_masks_inspected": False,
        "stdout_bytes": 0,
        "stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "observed_stderr_bytes": 1774,
        "observed_stderr_sha256": (
            "b8ee774b6c78bb045e0404b3d869e38ea2d3d489d104622a91b6aa560942d03d"
        ),
        "stderr_path": PREOUTCOME_FAILURE_STDERR_PATH,
        "stderr_canonical_lf_bytes": 1743,
        "stderr_canonical_lf_sha256": (
            "abd281349132dca2c5dddab82a299048bd4ad8a48f274f3e719262dbdcf4b4ff"
        ),
    }
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise SystemExit(f"pre-outcome failure receipt {key} mismatch")
    failure_stderr = repo / PREOUTCOME_FAILURE_STDERR_PATH
    failure_stderr_bytes = canonical_lf_bytes(failure_stderr)
    if (
        len(failure_stderr_bytes) != receipt["stderr_canonical_lf_bytes"]
        or P.sha256_bytes(failure_stderr_bytes)
        != receipt["stderr_canonical_lf_sha256"]
    ):
        raise SystemExit("pre-outcome stderr binding mismatch")
    return receipt


def _split_rank(block_id: str) -> str:
    return hashlib.sha256(f"{SPLIT_SEED}|{block_id}".encode("ascii")).hexdigest()


def split_assignments(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for block in manifest["blocks"]:
        grouped[(str(block["scroll"]), int(block["z_stratum"]))].append(block)

    assignments: dict[str, list[dict[str, Any]]] = {"dev": [], "holdout": []}
    expected_groups = {(scroll, z) for scroll in P.SCROLLS for z in range(P.Z_STRATA)}
    if set(grouped) != expected_groups:
        raise ValueError("source manifest does not contain every scroll/z stratum")
    for (scroll, z_stratum), blocks in sorted(grouped.items()):
        ranked = sorted(
            blocks, key=lambda block: (_split_rank(block["block_id"]), block["block_id"])
        )
        expected = DEV_BLOCKS_PER_STRATUM + HOLDOUT_BLOCKS_PER_STRATUM
        if len(ranked) != expected:
            raise ValueError(
                f"{scroll} z{z_stratum}: expected {expected} blocks, got {len(ranked)}"
            )
        for split, selected in (
            ("dev", ranked[:DEV_BLOCKS_PER_STRATUM]),
            ("holdout", ranked[DEV_BLOCKS_PER_STRATUM:]),
        ):
            assignments[split].extend(
                {
                    "block_id": block["block_id"],
                    "scroll": scroll,
                    "z_stratum": z_stratum,
                    "split_rank_sha256": _split_rank(block["block_id"]),
                }
                for block in selected
            )
    for split in assignments:
        assignments[split].sort(key=lambda item: item["block_id"])
    return assignments


def build_protocol_lock(repo: Path, manifest_path: Path) -> dict[str, Any]:
    if git_output(repo, "status", "--porcelain=v1"):
        raise SystemExit("refusing to build protocol lock from a dirty worktree")
    manifest = load_hashed_json(manifest_path)
    if manifest["content_sha256"] != SOURCE_MANIFEST_CONTENT_SHA256:
        raise SystemExit("unexpected source manifest content hash")
    source_lock = R.load_protocol_lock(repo / SOURCE_PROTOCOL_LOCK_PATH)
    if source_lock["content_sha256"] != SOURCE_PROTOCOL_LOCK_CONTENT_SHA256:
        raise SystemExit("unexpected source protocol lock content hash")
    missing = [name for name in IMPLEMENTATION_FILES if not (repo / name).is_file()]
    if missing:
        raise SystemExit(f"missing implementation files: {missing}")
    failure_receipt = validate_preoutcome_failure(repo)

    value: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": LOCK_STATUS,
        "execution_revision": EXECUTION_REVISION,
        "registration_evidence": "public git commit containing this protocol lock",
        "bridge_outcomes_seen_at_lock": False,
        "resource_amendment": {
            "kind": "memory_bounded_block_streaming",
            "supersedes_protocol_lock_content_sha256": (
                SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256
            ),
            "preoutcome_failure_receipt_path": PREOUTCOME_FAILURE_RECEIPT_PATH,
            "preoutcome_failure_receipt_content_sha256": failure_receipt[
                "content_sha256"
            ],
            "scientific_protocol_changed": False,
            "changed_execution_only": (
                "validate, load, build, and score one source block at a time"
            ),
        },
        "public_branch": PUBLIC_BRANCH,
        "implementation_commit": git_output(repo, "rev-parse", "HEAD"),
        "source_public_freeze_commit": SOURCE_PUBLIC_FREEZE_COMMIT,
        "source_manifest_path": str(manifest_path.relative_to(repo)).replace("\\", "/"),
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "source_manifest_file_sha256": P.sha256_file(manifest_path),
        "source_protocol_lock_path": SOURCE_PROTOCOL_LOCK_PATH,
        "source_protocol_lock_content_sha256": SOURCE_PROTOCOL_LOCK_CONTENT_SHA256,
        "inference_requirement": {
            "completed_blocks": 64,
            "completion_status": "inference_complete_unscored",
            "partial_scoring_forbidden": True,
        },
        "split": {
            "seed": SPLIT_SEED,
            "method": "SHA256(seed|block_id), four dev and four holdout per scroll/z stratum",
            "assignments": split_assignments(manifest),
        },
        "baseline": {
            "source": "corrected_pmax_l1",
            "strict_threshold": 0.2,
            "description": "exact PR #1386 plans-driven CT-normalized probability",
        },
        "matched_budget_control": {
            "method": (
                "within each block, keep exactly the candidate output voxel count by "
                "descending probability rank; flat-index order breaks probability ties"
            ),
            "population": "complete prediction extent used by the physical metric",
            "uses_labels": False,
            "exact_voxel_count": True,
        },
        "candidates": {name: asdict(config) for name, config in CANDIDATE_CONFIGS.items()},
        "development_selection": {
            "score_all_candidates_on_dev_only": True,
            "minimum_edited_blocks_total": MIN_EDITED_BLOCKS_TOTAL,
            "minimum_edited_blocks_per_scroll": MIN_EDITED_BLOCKS_PER_SCROLL,
            "pooled_point_skill_delta_mean": "strictly_positive",
            "point_skill_delta_mean_each_scroll": "nonnegative",
            "far37_delta_margin_each_scroll": FAR37_MARGIN,
            "recall37_delta_margin_each_scroll": RECALL37_MARGIN,
            "same_requirements_against_matched_budget_control": True,
            "ranking": [
                "largest minimum pooled point-skill delta mean across fixed and matched controls",
                "largest pooled point-skill delta mean against matched-budget control",
                "smallest summed far37 delta",
                "fewest removed voxels",
                "lexicographic candidate name",
            ],
            "if_none_pass": "stop_without_scoring_holdout",
        },
        "holdout_success_gate": {
            "score_selected_candidate_only": True,
            "minimum_edited_blocks_total": MIN_EDITED_BLOCKS_TOTAL,
            "minimum_edited_blocks_per_scroll": MIN_EDITED_BLOCKS_PER_SCROLL,
            "pooled_stratified_point_skill_delta_ci95_low": "strictly_positive",
            "point_skill_delta_mean_each_scroll": "strictly_positive",
            "far37_delta_margin_each_scroll": FAR37_MARGIN,
            "recall37_delta_margin_each_scroll": RECALL37_MARGIN,
            "same_requirements_against_matched_budget_control": True,
            "required_point_blocks_each_scroll": 16,
        },
        "analysis": {
            "primary_metric": "recall_37um_minus_shifted_null_recall_37um",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_groups": "equal weight over scroll x z-stratum",
            "z_sample_step": P.Z_SAMPLE_STEP,
            "no_invalid_3d_cavity_metric": True,
        },
        "runtime_versions": runtime_versions(),
        "implementation_hash_mode": "utf8_canonical_lf_v1",
        "implementation_files_sha256": {
            name: canonical_lf_sha256(repo / name) for name in IMPLEMENTATION_FILES
        },
    }
    value["content_sha256"] = _content_sha(value)
    return value


def verify_public_freeze(repo: Path, lock: dict[str, Any], lock_path: Path) -> str:
    status = git_output(repo, "status", "--porcelain=v1")
    if status:
        raise SystemExit("public freeze worktree is dirty:\n" + status)
    if git_output(repo, "branch", "--show-current") != lock["public_branch"]:
        raise SystemExit("wrong public branch")
    head = git_output(repo, "rev-parse", "HEAD")
    try:
        upstream = git_output(repo, "rev-parse", "@{upstream}")
    except subprocess.CalledProcessError as exc:
        raise SystemExit("public branch has no upstream") from exc
    if head != upstream:
        raise SystemExit(f"local HEAD {head} is not public upstream {upstream}")
    tracked = git_output(repo, "ls-files", "--error-unmatch", str(lock_path.relative_to(repo)))
    if not tracked:
        raise SystemExit("protocol lock is not tracked")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", lock["implementation_commit"], head],
        cwd=repo,
    ).returncode:
        raise SystemExit("implementation commit is not an ancestor of public freeze")
    return head


def verify_protocol_files(repo: Path, lock: dict[str, Any]) -> dict[str, Any]:
    if lock.get("protocol_id") != PROTOCOL_ID or lock.get("status") != LOCK_STATUS:
        raise SystemExit("unexpected bridge-split protocol lock")
    if lock.get("source_protocol_lock_content_sha256") != SOURCE_PROTOCOL_LOCK_CONTENT_SHA256:
        raise SystemExit("source protocol lock binding drift")
    if lock.get("source_protocol_lock_path") != SOURCE_PROTOCOL_LOCK_PATH:
        raise SystemExit("source protocol lock path drift")
    if lock.get("source_public_freeze_commit") != SOURCE_PUBLIC_FREEZE_COMMIT:
        raise SystemExit("source public freeze binding drift")
    amendment = lock.get("resource_amendment", {})
    if (
        lock.get("execution_revision") != EXECUTION_REVISION
        or amendment.get("kind") != "memory_bounded_block_streaming"
        or amendment.get("supersedes_protocol_lock_content_sha256")
        != SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256
        or amendment.get("scientific_protocol_changed") is not False
    ):
        raise SystemExit("unexpected resource amendment")
    failure_receipt = validate_preoutcome_failure(repo)
    if (
        amendment.get("preoutcome_failure_receipt_path")
        != PREOUTCOME_FAILURE_RECEIPT_PATH
        or amendment.get("preoutcome_failure_receipt_content_sha256")
        != failure_receipt["content_sha256"]
    ):
        raise SystemExit("pre-outcome failure receipt binding drift")
    if lock.get("runtime_versions") != runtime_versions():
        raise SystemExit(
            f"runtime version drift: {runtime_versions()} != {lock.get('runtime_versions')}"
        )
    if lock.get("implementation_hash_mode") != "utf8_canonical_lf_v1":
        raise SystemExit("unexpected implementation hash mode")
    if set(lock.get("implementation_files_sha256", {})) != set(IMPLEMENTATION_FILES):
        raise SystemExit("implementation file set drift")
    for name, expected in sorted(lock["implementation_files_sha256"].items()):
        path = repo / name
        if not path.is_file() or canonical_lf_sha256(path) != expected:
            raise SystemExit(f"implementation drift: {name}")
    manifest_path = repo / lock["source_manifest_path"]
    if P.sha256_file(manifest_path) != lock["source_manifest_file_sha256"]:
        raise SystemExit("source manifest file drift")
    manifest = load_hashed_json(manifest_path)
    if manifest["content_sha256"] != lock["source_manifest_content_sha256"]:
        raise SystemExit("source manifest content drift")
    if split_assignments(manifest) != lock["split"]["assignments"]:
        raise SystemExit("dev/holdout assignment drift")
    if {name: asdict(config) for name, config in CANDIDATE_CONFIGS.items()} != lock["candidates"]:
        raise SystemExit("candidate grid drift")
    return manifest


def verify_completed_inference(
    out_root: Path, manifest: dict[str, Any], lock: dict[str, Any]
) -> dict[str, tuple[Path, str]]:
    completion_path = out_root / "inference_completion_receipt.json"
    if not completion_path.is_file():
        raise SystemExit(
            "all-64 inference completion receipt is missing; partial scoring forbidden"
        )
    completion = P.load_json(completion_path)
    expected_header = {
        "schema_version": 1,
        "protocol_id": R.PROTOCOL_ID,
        "implementation_revision": R.IMPLEMENTATION_REVISION,
        "status": "inference_complete_unscored",
        "causal_claim_allowed": False,
        "completed_blocks": 64,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "protocol_lock_content_sha256": SOURCE_PROTOCOL_LOCK_CONTENT_SHA256,
        "public_freeze_commit": SOURCE_PUBLIC_FREEZE_COMMIT,
    }
    for key, expected in expected_header.items():
        if completion.get(key) != expected:
            raise SystemExit(f"inference completion {key} mismatch")

    records = completion.get("records")
    if not isinstance(records, list) or len(records) != 64:
        raise SystemExit("completion receipt must contain exactly 64 ordered records")
    expected_ids = [block["block_id"] for block in manifest["blocks"]]
    if [record.get("block_id") for record in records] != expected_ids:
        raise SystemExit("completion receipt block order or identity mismatch")
    completion_records = {record["block_id"]: record for record in records}

    expected_arrays = {(out_root / block["array_file"]).resolve() for block in manifest["blocks"]}
    expected_receipts = {
        (out_root / block["receipt_file"]).resolve() for block in manifest["blocks"]
    }
    expected_cleanups = {
        (out_root / "cleanup" / f"{block['block_id']}.json").resolve()
        for block in manifest["blocks"]
    }
    actual_arrays = {path.resolve() for path in (out_root / "arrays").glob("*.npz")}
    actual_receipts = {
        path.resolve()
        for path in (out_root / "receipts").glob("*.json")
        if ".attempt-" not in path.name
    }
    actual_cleanups = {path.resolve() for path in (out_root / "cleanup").glob("*.json")}
    if (
        actual_arrays != expected_arrays
        or actual_receipts != expected_receipts
        or actual_cleanups != expected_cleanups
    ):
        raise SystemExit("final artifact directories do not contain exactly the frozen 64 blocks")

    array_refs: dict[str, tuple[Path, str]] = {}
    for block in manifest["blocks"]:
        block_id = block["block_id"]
        array_path = out_root / block["array_file"]
        receipt_path = out_root / block["receipt_file"]
        cleanup_path = out_root / "cleanup" / f"{block_id}.json"
        record = completion_records[block_id]
        for path, key in (
            (array_path, "array_sha256"),
            (receipt_path, "receipt_sha256"),
            (cleanup_path, "cleanup_sha256"),
        ):
            if not path.is_file() or P.sha256_file(path) != record.get(key):
                raise SystemExit(f"{block_id}: {key} mismatch")
        receipt = P.load_json(receipt_path)
        cleanup = P.load_json(cleanup_path)
        inference = receipt.get("inference", {})
        if (
            receipt.get("status") != "complete"
            or receipt.get("block_id") != block_id
            or receipt.get("public_freeze_commit") != SOURCE_PUBLIC_FREEZE_COMMIT
            or receipt.get("manifest_content_sha256") != SOURCE_MANIFEST_CONTENT_SHA256
            or receipt.get("corrected_villa_commit") != P.PR1386_COMMIT
            or receipt.get("array_file_sha256") != record["array_sha256"]
            or inference.get("predict_returncode") != 0
            or inference.get("blend_returncode") != 0
            or inference.get("required_normalization_log_token")
            != R.MODEL_NORMALIZATION_LOG_TOKEN
            or inference.get("required_normalization_log_token_found") is not True
            or cleanup.get("schema_version") != 1
            or cleanup.get("protocol_id") != R.PROTOCOL_ID
            or cleanup.get("implementation_revision") != R.IMPLEMENTATION_REVISION
            or cleanup.get("block_id") != block_id
            or cleanup.get("attempt") != receipt.get("attempt")
            or cleanup.get("final_receipt_sha256") != record["receipt_sha256"]
            or cleanup.get("array_file_sha256") != record["array_sha256"]
            or cleanup.get("text_logs_retained") is not True
        ):
            raise SystemExit(f"{block_id}: final receipt binding mismatch")
        verified_arrays = P._load_block_arrays(
            array_path, block, SOURCE_MANIFEST_CONTENT_SHA256
        )
        del verified_arrays
        array_refs[block_id] = (array_path, record["array_sha256"])
    gc.collect()
    return array_refs


def verify_label_inputs(labels_root: Path, manifest: dict[str, Any]) -> None:
    """Bind scoring to the same released archives and extracted-store metadata as the source."""

    for scroll, record in manifest["inputs"]["labels"].items():
        archive = labels_root / record["archive"]["name"]
        if not archive.is_file() or archive.stat().st_size != record["archive"]["bytes"]:
            raise SystemExit(f"{scroll}: label archive byte size changed")
        if P.sha256_file(archive) != record["archive"]["sha256"]:
            raise SystemExit(f"{scroll}: label archive SHA changed")
        store = labels_root / record["store"]
        if not store.is_dir():
            raise SystemExit(f"{scroll}: extracted label store is missing")
        if P.sha256_file(store / ".zarray") != record["zarray_sha256"]:
            raise SystemExit(f"{scroll}: label .zarray changed")
        if P.sha256_file(store / ".zattrs") != record["zattrs_sha256"]:
            raise SystemExit(f"{scroll}: label .zattrs changed")
        opened = P._open_zarr(store)
        if list(map(int, opened.shape)) != list(map(int, record["shape_l1"])):
            raise SystemExit(f"{scroll}: opened label shape changed")


def _stratified_bootstrap(
    values_by_group: dict[str, list[float]], seed: int
) -> dict[str, float | int | None]:
    groups = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in sorted(values_by_group.items())
        if values
    }
    if not groups:
        return {"n": 0, "groups": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    draws = []
    means = []
    for values in groups.values():
        draws.append(
            rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
        )
        means.append(float(values.mean()))
    pooled = np.stack(draws, axis=1).mean(axis=1)
    lo, hi = np.quantile(pooled, [0.025, 0.975])
    return {
        "n": int(sum(len(values) for values in groups.values())),
        "groups": len(groups),
        "mean": float(np.mean(means)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def _mean_delta(
    rows: Iterable[dict[str, Any]], candidate: str, reference: str, metric: str
) -> float | None:
    values = []
    for row in rows:
        baseline = row["arms"][reference].get(metric)
        proposed = row["arms"][candidate].get(metric)
        if baseline is not None and proposed is not None:
            values.append(float(proposed) - float(baseline))
    return float(np.mean(values)) if values else None


def compare_candidate(
    rows: list[dict[str, Any]], candidate: str, reference: str, seed: int
) -> dict[str, Any]:
    point_groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        baseline = row["arms"][reference]["point_skill"]
        proposed = row["arms"][candidate]["point_skill"]
        if baseline is not None and proposed is not None:
            point_groups[f"{row['scroll']}:z{row['z_stratum']}"].append(
                float(proposed) - float(baseline)
            )

    by_scroll: dict[str, dict[str, Any]] = {}
    for scroll_index, scroll in enumerate(P.SCROLLS):
        selected = [row for row in rows if row["scroll"] == scroll]
        scroll_groups = {
            name: values for name, values in point_groups.items() if name.startswith(f"{scroll}:")
        }
        by_scroll[scroll] = {
            "point_skill_delta": _stratified_bootstrap(
                scroll_groups, seed + 100 + scroll_index
            ),
            "far37_fraction_delta_macro_mean": _mean_delta(
                selected, candidate, reference, "pred_far37_fraction"
            ),
            "recall37_delta_macro_mean": _mean_delta(
                selected, candidate, reference, "recall_37um"
            ),
            "arc_fully_missed_delta_macro_mean": _mean_delta(
                selected, candidate, reference, "arc_fully_missed"
            ),
            "point_blocks": sum(
                row["arms"][candidate]["point_skill"] is not None for row in selected
            ),
            "edited_blocks": sum(
                row["audits"][candidate]["removed_voxels"] > 0 for row in selected
            ),
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "pooled_point_skill_delta": _stratified_bootstrap(point_groups, seed),
        "by_scroll": by_scroll,
        "edited_blocks_total": sum(
            row["audits"][candidate]["removed_voxels"] > 0 for row in rows
        ),
        "removed_voxels_total": sum(
            int(row["audits"][candidate]["removed_voxels"]) for row in rows
        ),
    }


def development_gates(comparisons: dict[str, Any]) -> dict[str, bool]:
    comparison = comparisons["vs_fixed"]
    matched = comparisons["vs_matched_budget"]
    by_scroll = comparison["by_scroll"]
    matched_by_scroll = matched["by_scroll"]
    gates = {
        "edited_enough_blocks": comparison["edited_blocks_total"] >= MIN_EDITED_BLOCKS_TOTAL,
        "edited_each_scroll": all(
            by_scroll[scroll]["edited_blocks"] >= MIN_EDITED_BLOCKS_PER_SCROLL
            for scroll in P.SCROLLS
        ),
        "pooled_point_mean_positive": (
            comparison["pooled_point_skill_delta"]["mean"] is not None
            and comparison["pooled_point_skill_delta"]["mean"] > 0
        ),
        "point_mean_nonnegative_each_scroll": all(
            by_scroll[scroll]["point_skill_delta"]["mean"] is not None
            and by_scroll[scroll]["point_skill_delta"]["mean"] >= 0
            for scroll in P.SCROLLS
        ),
        "far37_noninferior_each_scroll": all(
            by_scroll[scroll]["far37_fraction_delta_macro_mean"] is not None
            and by_scroll[scroll]["far37_fraction_delta_macro_mean"] <= FAR37_MARGIN
            for scroll in P.SCROLLS
        ),
        "recall37_noninferior_each_scroll": all(
            by_scroll[scroll]["recall37_delta_macro_mean"] is not None
            and by_scroll[scroll]["recall37_delta_macro_mean"] >= RECALL37_MARGIN
            for scroll in P.SCROLLS
        ),
        "matched_pooled_point_mean_positive": (
            matched["pooled_point_skill_delta"]["mean"] is not None
            and matched["pooled_point_skill_delta"]["mean"] > 0
        ),
        "matched_point_mean_nonnegative_each_scroll": all(
            matched_by_scroll[scroll]["point_skill_delta"]["mean"] is not None
            and matched_by_scroll[scroll]["point_skill_delta"]["mean"] >= 0
            for scroll in P.SCROLLS
        ),
        "matched_far37_noninferior_each_scroll": all(
            matched_by_scroll[scroll]["far37_fraction_delta_macro_mean"] is not None
            and matched_by_scroll[scroll]["far37_fraction_delta_macro_mean"] <= FAR37_MARGIN
            for scroll in P.SCROLLS
        ),
        "matched_recall37_noninferior_each_scroll": all(
            matched_by_scroll[scroll]["recall37_delta_macro_mean"] is not None
            and matched_by_scroll[scroll]["recall37_delta_macro_mean"] >= RECALL37_MARGIN
            for scroll in P.SCROLLS
        ),
    }
    gates["eligible"] = all(gates.values())
    return gates


def select_development_candidate(
    comparisons: dict[str, dict[str, Any]]
) -> tuple[str | None, dict[str, dict[str, bool]]]:
    gates = {name: development_gates(value) for name, value in comparisons.items()}
    eligible = [name for name in CANDIDATE_CONFIGS if gates[name]["eligible"]]
    if not eligible:
        return None, gates

    def rank(name: str) -> tuple[float, float, float, int, str]:
        fixed = comparisons[name]["vs_fixed"]
        matched = comparisons[name]["vs_matched_budget"]
        fixed_point = float(fixed["pooled_point_skill_delta"]["mean"])
        matched_point = float(matched["pooled_point_skill_delta"]["mean"])
        far = sum(
            float(fixed["by_scroll"][scroll]["far37_fraction_delta_macro_mean"])
            for scroll in P.SCROLLS
        )
        return (
            -min(fixed_point, matched_point),
            -matched_point,
            far,
            int(fixed["removed_voxels_total"]),
            name,
        )

    return min(eligible, key=rank), gates


def holdout_gates(comparisons: dict[str, Any]) -> dict[str, bool]:
    comparison = comparisons["vs_fixed"]
    matched = comparisons["vs_matched_budget"]
    by_scroll = comparison["by_scroll"]
    matched_by_scroll = matched["by_scroll"]
    gates = {
        "edited_enough_blocks": comparison["edited_blocks_total"] >= MIN_EDITED_BLOCKS_TOTAL,
        "edited_each_scroll": all(
            by_scroll[scroll]["edited_blocks"] >= MIN_EDITED_BLOCKS_PER_SCROLL
            for scroll in P.SCROLLS
        ),
        "pooled_point_ci_low_positive": (
            comparison["pooled_point_skill_delta"]["ci95_low"] is not None
            and comparison["pooled_point_skill_delta"]["ci95_low"] > 0
        ),
        "point_mean_positive_each_scroll": all(
            by_scroll[scroll]["point_skill_delta"]["mean"] is not None
            and by_scroll[scroll]["point_skill_delta"]["mean"] > 0
            for scroll in P.SCROLLS
        ),
        "far37_noninferior_each_scroll": all(
            by_scroll[scroll]["far37_fraction_delta_macro_mean"] is not None
            and by_scroll[scroll]["far37_fraction_delta_macro_mean"] <= FAR37_MARGIN
            for scroll in P.SCROLLS
        ),
        "recall37_noninferior_each_scroll": all(
            by_scroll[scroll]["recall37_delta_macro_mean"] is not None
            and by_scroll[scroll]["recall37_delta_macro_mean"] >= RECALL37_MARGIN
            for scroll in P.SCROLLS
        ),
        "all_16_point_blocks_each_scroll": all(
            by_scroll[scroll]["point_blocks"] == 16 for scroll in P.SCROLLS
        ),
        "matched_pooled_point_ci_low_positive": (
            matched["pooled_point_skill_delta"]["ci95_low"] is not None
            and matched["pooled_point_skill_delta"]["ci95_low"] > 0
        ),
        "matched_point_mean_positive_each_scroll": all(
            matched_by_scroll[scroll]["point_skill_delta"]["mean"] is not None
            and matched_by_scroll[scroll]["point_skill_delta"]["mean"] > 0
            for scroll in P.SCROLLS
        ),
        "matched_far37_noninferior_each_scroll": all(
            matched_by_scroll[scroll]["far37_fraction_delta_macro_mean"] is not None
            and matched_by_scroll[scroll]["far37_fraction_delta_macro_mean"] <= FAR37_MARGIN
            for scroll in P.SCROLLS
        ),
        "matched_recall37_noninferior_each_scroll": all(
            matched_by_scroll[scroll]["recall37_delta_macro_mean"] is not None
            and matched_by_scroll[scroll]["recall37_delta_macro_mean"] >= RECALL37_MARGIN
            for scroll in P.SCROLLS
        ),
        "matched_all_16_point_blocks_each_scroll": all(
            matched_by_scroll[scroll]["point_blocks"] == 16 for scroll in P.SCROLLS
        ),
    }
    gates["primary_claim_passes"] = all(gates.values())
    return gates


def _blocks_for_split(
    manifest: dict[str, Any], lock: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    wanted = {item["block_id"] for item in lock["split"]["assignments"][split]}
    blocks = [block for block in manifest["blocks"] if block["block_id"] in wanted]
    if len(blocks) != 32:
        raise SystemExit(f"{split}: expected 32 blocks, got {len(blocks)}")
    return blocks


def matched_budget_arm(candidate: str) -> str:
    return f"{candidate}__probability_rank_mass_control"


def probability_rank_mass_control(
    probability: np.ndarray, target_positive_voxels: int, low_threshold: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """Keep the highest-probability baseline voxels at the candidate's exact mass.

    Probability ties are resolved by increasing C-order flat index.  This comparator asks
    whether bridge location adds value beyond simply deleting the same number of the baseline's
    least-confident voxels; it never consults labels or physical outcomes.
    """

    p = np.asarray(probability)
    if p.ndim not in (2, 3) or not np.issubdtype(p.dtype, np.floating):
        raise ValueError("probability-rank control requires a floating 2D or 3D array")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability values must be finite and in [0, 1]")
    baseline = p > low_threshold
    eligible = np.flatnonzero(baseline.ravel())
    target = int(target_positive_voxels)
    if target < 0 or target > len(eligible):
        raise ValueError(
            f"target_positive_voxels {target} is outside baseline mass {len(eligible)}"
        )
    values = p.ravel()[eligible]
    ranked = np.lexsort((eligible, -values))
    selected = eligible[ranked[:target]]
    control = np.zeros(p.size, dtype=bool)
    control[selected] = True
    control = control.reshape(p.shape)
    if int(np.count_nonzero(control)) != target or np.any(control & ~baseline):
        raise AssertionError("probability-rank mass control contract failed")
    return control, {
        "schema_version": 1,
        "kind": "probability_rank_mass_control",
        "tie_break": "increasing_C_order_flat_index",
        "baseline_positive_voxels": int(len(eligible)),
        "target_positive_voxels": target,
        "realized_positive_voxels": int(np.count_nonzero(control)),
        "removed_voxels": int(len(eligible) - target),
        "uses_labels": False,
    }


def build_masks(
    blocks: list[dict[str, Any]],
    arrays: dict[str, dict[str, Any]],
    candidates: Iterable[str],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, dict[str, Any]]]]:
    masks: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, dict[str, Any]]] = {}
    for block in blocks:
        block_id = block["block_id"]
        probability = arrays[block_id]["corrected"]
        masks[block_id] = {"corrected_fixed": probability > P.FIXED_THRESHOLD}
        audits[block_id] = {}
        for name in candidates:
            result = split_probability_bridges(probability, CANDIDATE_CONFIGS[name])
            masks[block_id][name] = result.mask
            audits[block_id][name] = result.audit
            control_name = matched_budget_arm(name)
            control, control_audit = probability_rank_mass_control(
                probability,
                int(np.count_nonzero(result.mask)),
                CANDIDATE_CONFIGS[name].low_threshold,
            )
            masks[block_id][control_name] = control
            audits[block_id][control_name] = control_audit
    return masks, audits


def score_blocks(
    blocks: list[dict[str, Any]],
    masks: dict[str, dict[str, np.ndarray]],
    audits: dict[str, dict[str, dict[str, Any]]],
    labels_root: Path,
) -> list[dict[str, Any]]:
    arm_names = tuple(next(iter(masks.values())).keys())
    counts = {
        block["block_id"]: {arm: P.blank_counts() for arm in arm_names}
        for block in blocks
    }
    jobs: dict[tuple[str, int], list[tuple[dict[str, Any], int]]] = defaultdict(list)
    for block in blocks:
        z0 = int(block["geometry"]["score_local_l1"][0])
        for k in range(P.SCORE_SIZE_L1):
            if (z0 + k) % P.Z_SAMPLE_STEP == 0:
                jobs[(block["scroll"], z0 + k)].append((block, k))

    stores = {
        scroll: P._open_zarr(labels_root / config["label_store"])
        for scroll, config in P.SCROLLS.items()
    }
    for (scroll, z), plane_jobs in sorted(jobs.items()):
        truth = P.prepare_truth_plane(np.asarray(stores[scroll][z, :, :], dtype=np.uint8))
        for block, k in plane_jobs:
            geometry = block["geometry"]
            _, _, score_y0, _, score_x0, _ = geometry["score_local_l1"]
            _, _, extent_y0, _, extent_x0, _ = geometry["prediction_extent_local_l1"]
            for arm in arm_names:
                value = P.score_plane(
                    truth,
                    masks[block["block_id"]][arm][k],
                    score_y0,
                    score_x0,
                    extent_y0,
                    extent_x0,
                )
                P.add_counts(counts[block["block_id"]][arm], value)

    rows = []
    for block in blocks:
        block_id = block["block_id"]
        rows.append(
            {
                "block_id": block_id,
                "scroll": block["scroll"],
                "z_stratum": block["z_stratum"],
                "arms": {arm: P.metrics(value) for arm, value in counts[block_id].items()},
                "audits": audits[block_id],
            }
        )
    return sorted(rows, key=lambda row: row["block_id"])


def score_blocks_streaming(
    blocks: list[dict[str, Any]],
    array_refs: dict[str, tuple[Path, str]],
    candidates: Iterable[str],
    labels_root: Path,
) -> list[dict[str, Any]]:
    """Run the unchanged operator and scorer with at most one decoded block resident.

    Per-block counts are independent integers.  Streaming changes only object lifetime and
    label-plane I/O order; masks, audits, metrics, row order, and downstream statistics are the
    same as the original all-block execution.
    """

    candidate_names = tuple(candidates)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        block_id = block["block_id"]
        if block_id not in array_refs:
            raise SystemExit(f"{block_id}: verified array reference is missing")
        array_path, expected_sha256 = array_refs[block_id]
        if P.sha256_file(array_path) != expected_sha256:
            raise SystemExit(f"{block_id}: array changed after all-64 verification")
        block_arrays = P._load_block_arrays(
            array_path, block, SOURCE_MANIFEST_CONTENT_SHA256
        )
        masks, audits = build_masks(
            [block], {block_id: block_arrays}, candidate_names
        )
        block_rows = score_blocks([block], masks, audits, labels_root)
        if len(block_rows) != 1 or block_rows[0]["block_id"] != block_id:
            raise AssertionError(f"{block_id}: streaming scorer row contract failed")
        rows.extend(block_rows)
        del block_rows, masks, audits, block_arrays
        gc.collect()
    return sorted(rows, key=lambda row: row["block_id"])


def score_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    result_path = Path(args.result).resolve()
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite existing bridge result: {result_path}")
    lock_path = Path(args.lock).resolve()
    lock = load_hashed_json(lock_path)
    manifest = verify_protocol_files(repo, lock)
    public_freeze_commit = verify_public_freeze(repo, lock, lock_path)
    array_refs = verify_completed_inference(
        Path(args.out_root).resolve(), manifest, lock
    )
    labels_root = Path(args.labels_root).resolve()
    verify_label_inputs(labels_root, manifest)

    dev_blocks = _blocks_for_split(manifest, lock, "dev")
    dev_rows = score_blocks_streaming(
        dev_blocks, array_refs, CANDIDATE_CONFIGS, labels_root
    )
    dev_comparisons = {
        name: {
            "vs_fixed": compare_candidate(
                dev_rows, name, "corrected_fixed", BOOTSTRAP_SEED + index * 1000
            ),
            "vs_matched_budget": compare_candidate(
                dev_rows,
                name,
                matched_budget_arm(name),
                BOOTSTRAP_SEED + index * 1000 + 500,
            ),
        }
        for index, name in enumerate(CANDIDATE_CONFIGS)
    }
    selected, dev_gates = select_development_candidate(dev_comparisons)

    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "protocol_lock_content_sha256": lock["content_sha256"],
        "public_freeze_commit": public_freeze_commit,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "development": {
            "per_block": dev_rows,
            "comparisons": dev_comparisons,
            "gates": dev_gates,
            "selected_candidate": selected,
        },
    }
    if selected is None:
        result["status"] = "closed_no_candidate_passed_development"
        result["holdout"] = {"scored": False}
    else:
        holdout_blocks = _blocks_for_split(manifest, lock, "holdout")
        holdout_rows = score_blocks_streaming(
            holdout_blocks, array_refs, [selected], labels_root
        )
        holdout_comparison = {
            "vs_fixed": compare_candidate(
                holdout_rows, selected, "corrected_fixed", BOOTSTRAP_SEED + 50_000
            ),
            "vs_matched_budget": compare_candidate(
                holdout_rows,
                selected,
                matched_budget_arm(selected),
                BOOTSTRAP_SEED + 50_500,
            ),
        }
        gates = holdout_gates(holdout_comparison)
        result["status"] = (
            "positive_holdout" if gates["primary_claim_passes"] else "negative_holdout"
        )
        result["holdout"] = {
            "scored": True,
            "selected_candidate": selected,
            "per_block": holdout_rows,
            "comparison": holdout_comparison,
            "gates": gates,
        }
    result["content_sha256"] = _content_sha(result)
    P.write_json(result_path, result)
    if load_hashed_json(result_path)["content_sha256"] != result["content_sha256"]:
        raise SystemExit("bridge result failed post-write integrity verification")
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_candidate": selected,
                "holdout_gates": result.get("holdout", {}).get("gates"),
                "content_sha256": result["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def verify_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    lock_path = Path(args.lock).resolve()
    lock = load_hashed_json(lock_path)
    manifest = verify_protocol_files(repo, lock)
    head = verify_public_freeze(repo, lock, lock_path)
    print(
        P.canonical_json(
            {
                "status": "verified_public_unscored",
                "head": head,
                "lock": lock["content_sha256"],
                "manifest": manifest["content_sha256"],
            }
        )
    )


def plan_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    output = Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing protocol lock: {output}")
    value = build_protocol_lock(repo, Path(args.manifest).resolve())
    P.write_json(output, value)
    verified = load_hashed_json(output)
    if verified["content_sha256"] != value["content_sha256"]:
        raise SystemExit("protocol lock failed post-write integrity verification")
    print(
        P.canonical_json(
            {
                "status": "protocol_lock_written_unscored",
                "path": str(output),
                "content_sha256": value["content_sha256"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="write a protocol lock from a clean implementation commit")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=plan_command)

    verify = sub.add_parser("verify", help="verify the committed public protocol freeze")
    verify.add_argument("--lock", required=True)
    verify.set_defaults(func=verify_command)

    score = sub.add_parser("score", help="score dev, select once, then score one holdout arm")
    score.add_argument("--lock", required=True)
    score.add_argument("--labels-root", required=True)
    score.add_argument("--out-root", required=True)
    score.add_argument("--result", required=True)
    score.set_defaults(func=score_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
