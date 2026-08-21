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
import os
import platform
import subprocess
import sys
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
from bounded_watershed import (
    BOUNDED_WATERSHED_BINARY,
    BOUNDED_WATERSHED_BINARY_SHA256,
    EXPECTED_HEAP_ITEM_BYTES,
    HEAP_CAPACITY_ITEMS,
    MIN_FREE_AFTER_FULL_HEAP_BYTES,
    cleanup_heap_files_for_pid,
    cleanup_stale_heap_files,
)
from probability_bridge_split import BridgeSplitConfig, split_probability_bridges


PROTOCOL_ID = "physical_probability_bridge_split_v1"
LOCK_STATUS = "preregistered_before_bridge_outcomes"
PUBLIC_BRANCH = "physical-multithreshold-repair"
EXECUTION_REVISION = 5
SUPERSEDED_PROTOCOL_LOCK_PATH = (
    "results/physical_bridge_split/protocol_lock_amendment_03.json"
)
SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256 = (
    "b2b483197755e4c5533cdacf066d7a415d33e55670df3518b46ac21cc0db5e19"
)
PREOUTCOME_FAILURE_RECEIPT_PATH = (
    "results/physical_bridge_split/preoutcome_failure_04.json"
)
PREOUTCOME_FAILURE_STDERR_PATH = (
    "results/physical_bridge_split/preoutcome_failure_04.stderr.txt"
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
GROUP_WORKER_SCHEMA_VERSION = 2
GROUP_WORKER_REQUEST_KIND = "physical_bridge_split_group_request"
GROUP_WORKER_RESPONSE_KIND = "physical_bridge_split_group_response"
ALLOWED_SUPERSEDED_LOCK_DIFFERENCES = (
    "content_sha256",
    "execution_revision",
    "implementation_commit",
    "implementation_files_sha256",
    "resource_amendment",
)


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
    ".gitignore",
    "_bounded_watershed_cy.pyx",
    "bounded_watershed.py",
    "build_bounded_watershed.py",
    "PHYSICAL_BRIDGE_SPLIT_AMENDMENT_01.md",
    "PHYSICAL_BRIDGE_SPLIT_AMENDMENT_02.md",
    "PHYSICAL_BRIDGE_SPLIT_AMENDMENT_03.md",
    "PHYSICAL_BRIDGE_SPLIT_AMENDMENT_04.md",
    "PHYSICAL_BRIDGE_SPLIT_PREREG.md",
    "THIRD_PARTY_NOTICES.md",
    "physical_bridge_split.py",
    "physical_normalization_ab.py",
    "probability_bridge_split.py",
    "requirements.txt",
    "results/physical_bridge_split/preoutcome_failure_01.json",
    "results/physical_bridge_split/preoutcome_failure_01.stderr.txt",
    "results/physical_bridge_split/preoutcome_failure_02.json",
    "results/physical_bridge_split/preoutcome_failure_02.stderr.txt",
    "results/physical_bridge_split/preoutcome_failure_03.json",
    "results/physical_bridge_split/preoutcome_failure_03.stderr.txt",
    "results/physical_bridge_split/protocol_lock_amendment_01.json",
    "results/physical_bridge_split/protocol_lock_amendment_02.json",
    "results/physical_bridge_split/protocol_lock_amendment_03.json",
    "results/physical_bridge_split/protocol_lock.json",
    PREOUTCOME_FAILURE_RECEIPT_PATH,
    PREOUTCOME_FAILURE_STDERR_PATH,
    SOURCE_PROTOCOL_LOCK_PATH,
    "run_physical_released_baseline_comparison.py",
    "test_physical_bridge_split.py",
    "test_bounded_watershed.py",
    "test_probability_bridge_split.py",
)

IMPLEMENTATION_BINARY_FILES = (BOUNDED_WATERSHED_BINARY,)


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def _content_sha(value: dict[str, Any]) -> str:
    unhashed = {key: item for key, item in value.items() if key != "content_sha256"}
    return P.sha256_bytes(P.canonical_json(unhashed).encode("utf-8"))


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


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


def _validate_failure_artifact(
    repo: Path,
    receipt_path: str,
    expected_content_sha256: str,
    stderr_path: str,
    label: str,
) -> dict[str, Any]:
    receipt = load_hashed_json(repo / receipt_path)
    if receipt.get("content_sha256") != expected_content_sha256:
        raise SystemExit(f"{label} failure receipt mismatch")
    if receipt.get("stderr_path") != stderr_path:
        raise SystemExit(f"{label} failure stderr path mismatch")
    stderr_bytes = canonical_lf_bytes(repo / stderr_path)
    if (
        len(stderr_bytes) != receipt.get("stderr_canonical_lf_bytes")
        or P.sha256_bytes(stderr_bytes) != receipt.get("stderr_canonical_lf_sha256")
    ):
        raise SystemExit(f"{label} failure stderr binding mismatch")
    return receipt


def _validate_amendment_01_chain(repo: Path) -> dict[str, Any]:
    lock_path = "results/physical_bridge_split/protocol_lock_amendment_01.json"
    lock_sha256 = "4e02b00859b894326ef9fddb0a21ea525e285713c49aa2a57953d78760ae0136"
    prior_lock = load_hashed_json(repo / lock_path)
    if prior_lock.get("content_sha256") != lock_sha256:
        raise SystemExit("amendment-01 protocol lock content mismatch")
    expected_amendment = {
        "kind": "memory_bounded_block_streaming",
        "supersedes_protocol_lock_content_sha256": (
            "fc54fc80ecd5c5c976804e709d2ac1e460c89ce4d3fc9d337542aca2435a3bb1"
        ),
        "preoutcome_failure_receipt_path": (
            "results/physical_bridge_split/preoutcome_failure_01.json"
        ),
        "preoutcome_failure_receipt_content_sha256": (
            "5fcf76f1971199a8f736bcb63e8dcd3648d32d06a8862f89e6d381219fc22e90"
        ),
        "scientific_protocol_changed": False,
        "changed_execution_only": (
            "validate, load, build, and score one source block at a time"
        ),
    }
    if prior_lock.get("resource_amendment") != expected_amendment:
        raise SystemExit("amendment-01 resource binding mismatch")
    _validate_failure_artifact(
        repo,
        "results/physical_bridge_split/preoutcome_failure_01.json",
        "5fcf76f1971199a8f736bcb63e8dcd3648d32d06a8862f89e6d381219fc22e90",
        "results/physical_bridge_split/preoutcome_failure_01.stderr.txt",
        "amendment-01",
    )
    return prior_lock


def _validate_amendment_02_chain(repo: Path) -> dict[str, Any]:
    """Validate the immutable amendment-02 -> amendment-01 failure chain."""

    amendment_01 = _validate_amendment_01_chain(repo)
    lock_path = "results/physical_bridge_split/protocol_lock_amendment_02.json"
    lock_sha256 = "40af2c70862763a2bd9c64324058418aae107fe57e613ef5df6bde7076555e7e"
    prior_lock = load_hashed_json(repo / lock_path)
    if prior_lock.get("content_sha256") != lock_sha256:
        raise SystemExit("amendment-02 protocol lock content mismatch")
    expected_amendment = {
        "allowed_top_level_differences_from_superseded_lock": list(
            ALLOWED_SUPERSEDED_LOCK_DIFFERENCES
        ),
        "changed_execution_only": (
            "score each exact scroll/z0 group in a fresh child process, one child at a time"
        ),
        "kind": "memory_bounded_fresh_process_per_scroll_z0_group",
        "preoutcome_failure_receipt_content_sha256": (
            "a7cdde195f28459a5dc10f3ad206ca278ca59b0f660d0c29f3ed05e4bfed0348"
        ),
        "preoutcome_failure_receipt_path": (
            "results/physical_bridge_split/preoutcome_failure_02.json"
        ),
        "scientific_protocol_changed": False,
        "supersedes_protocol_lock_content_sha256": amendment_01["content_sha256"],
        "supersedes_protocol_lock_path": (
            "results/physical_bridge_split/protocol_lock_amendment_01.json"
        ),
        "worker_group_key": ["scroll", "geometry.score_local_l1[0]"],
        "worker_parallelism": 1,
    }
    if prior_lock.get("resource_amendment") != expected_amendment:
        raise SystemExit("amendment-02 resource binding mismatch")
    if set(prior_lock) != set(amendment_01):
        raise SystemExit("amendment-02 top-level lock schema differs from amendment-01")
    for key in sorted(set(prior_lock) - set(ALLOWED_SUPERSEDED_LOCK_DIFFERENCES)):
        if P.canonical_json(prior_lock[key]) != P.canonical_json(amendment_01[key]):
            raise SystemExit(f"amendment-02 scientific field changed: {key}")
    _validate_failure_artifact(
        repo,
        "results/physical_bridge_split/preoutcome_failure_02.json",
        "a7cdde195f28459a5dc10f3ad206ca278ca59b0f660d0c29f3ed05e4bfed0348",
        "results/physical_bridge_split/preoutcome_failure_02.stderr.txt",
        "amendment-02",
    )
    return prior_lock


def validate_superseded_lock_chain(repo: Path) -> dict[str, Any]:
    """Validate the immutable amendment-03 -> 02 -> 01 failure chain."""

    amendment_02 = _validate_amendment_02_chain(repo)
    prior_lock = load_hashed_json(repo / SUPERSEDED_PROTOCOL_LOCK_PATH)
    if prior_lock.get("content_sha256") != SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256:
        raise SystemExit("superseded amendment-03 protocol lock content mismatch")
    expected_amendment = {
        "allowed_top_level_differences_from_superseded_lock": list(
            ALLOWED_SUPERSEDED_LOCK_DIFFERENCES
        ),
        "changed_execution_only": (
            "replace the doubling watershed pointer heap with the same event stream in a "
            "fixed-capacity file-backed direct-item heap"
        ),
        "event_order_changed": False,
        "heap_capacity_allocation_estimate_formula": (
            "min(max_items, max(min_items, mask_voxels * neighbor_count * "
            "marker_label_max + marker_voxels))"
        ),
        "heap_capacity_estimate_is_event_bound": False,
        "heap_capacity_exhaustion": "abort_without_result",
        "heap_capacity_items_max": 1_500_000_000,
        "heap_capacity_items_min": 1_000_000,
        "heap_item_bytes": 32,
        "heap_storage": "file_backed_mmap_direct_item_binary_heap",
        "heap_stale_file_recovery": "remove_only_after_owning_pid_has_exited",
        "implementation_binary_files_sha256": {
            "_bounded_watershed_cy.pyd": (
                "a14824fb65f5c9e7ad2ee859cc2e6a91de27dde5c18cebb3eb6bc388771b6767"
            )
        },
        "kind": "allocation_bounded_bit_equivalent_watershed_heap",
        "minimum_free_after_full_heap_bytes": 8_589_934_592,
        "preoutcome_failure_receipt_content_sha256": (
            "51f48a66d1992a9bb70175bd614346ac8c84e77c72110688a8a9fc1519e9e691"
        ),
        "preoutcome_failure_receipt_path": (
            "results/physical_bridge_split/preoutcome_failure_03.json"
        ),
        "scientific_protocol_changed": False,
        "signed_age_semantics": "pinned_scikit_image_int32_cast_including_wrap",
        "supersedes_protocol_lock_content_sha256": amendment_02["content_sha256"],
        "supersedes_protocol_lock_path": (
            "results/physical_bridge_split/protocol_lock_amendment_02.json"
        ),
        "watershed_reference": "scikit-image 0.26.0",
        "worker_group_key": ["scroll", "geometry.score_local_l1[0]"],
        "worker_parallelism": 1,
    }
    if prior_lock.get("resource_amendment") != expected_amendment:
        raise SystemExit("amendment-03 resource binding mismatch")
    if set(prior_lock) != set(amendment_02):
        raise SystemExit("amendment-03 top-level lock schema differs from amendment-02")
    for key in sorted(set(prior_lock) - set(ALLOWED_SUPERSEDED_LOCK_DIFFERENCES)):
        if P.canonical_json(prior_lock[key]) != P.canonical_json(amendment_02[key]):
            raise SystemExit(f"amendment-03 scientific field changed: {key}")
    _validate_failure_artifact(
        repo,
        "results/physical_bridge_split/preoutcome_failure_03.json",
        "51f48a66d1992a9bb70175bd614346ac8c84e77c72110688a8a9fc1519e9e691",
        "results/physical_bridge_split/preoutcome_failure_03.stderr.txt",
        "amendment-03",
    )
    return prior_lock


def validate_preoutcome_failure(repo: Path) -> dict[str, Any]:
    validate_superseded_lock_chain(repo)
    receipt = load_hashed_json(repo / PREOUTCOME_FAILURE_RECEIPT_PATH)
    expected = {
        "attempt_id": "physical_bridge_split_attempt8",
        "bridge_outcomes_seen": False,
        "development_comparison_started": None,
        "development_comparisons_inspected": False,
        "development_comparisons_may_have_existed_in_memory": True,
        "development_comparisons_persisted": False,
        "development_scoring_completed": None,
        "development_scoring_started": True,
        "development_selection_started": None,
        "execution_phase_at_failure": "unknown_development_or_holdout",
        "execution_revision": 4,
        "failed_worker_exit_code": 1,
        "failed_worker_group": {
            "scroll": "PHerc1203",
            "score_local_l1_z0": 64,
        },
        "failed_worker_pid": 15972,
        "failed_worker_response_completed": False,
        "failure_class": "MemoryError",
        "failure_phase": "build_masks/split_probability_bridges/bounded_watershed",
        "failure_scope": "fresh_group_worker",
        "holdout_opened": None,
        "kind": "preoutcome_execution_failure",
        "last_frame": "_bounded_watershed_cy.pyx:80 heappush",
        "observed_stderr_bytes": 2167,
        "observed_stderr_sha256": (
            "5f6c1e3c1c275dcddd321d3aad10fb751d9411a68e0d6287718d12ff058dd09c"
        ),
        "partial_candidate_masks_inspected": False,
        "partial_candidate_masks_may_have_existed_in_memory": True,
        "partial_candidate_masks_persisted": False,
        "partial_development_rows_inspected": False,
        "partial_development_rows_may_have_existed_in_memory": True,
        "partial_development_rows_persisted": False,
        "partial_holdout_rows_inspected": False,
        "partial_holdout_rows_may_have_existed_in_memory": True,
        "partial_holdout_rows_persisted": False,
        "phase_ambiguity_reason": (
            "failed group key occurs in both frozen splits and no progress artifact was persisted"
        ),
        "prior_protocol_lock_content_sha256": SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256,
        "prior_protocol_lock_path": SUPERSEDED_PROTOCOL_LOCK_PATH,
        "protocol_id": PROTOCOL_ID,
        "public_head": "74900651b833a1d803d3ef713b13fd5ed8aaab4e",
        "result_existed_after_exit": False,
        "result_path": "results/physical_bridge_split/result.json",
        "schema_version": 1,
        "selected_candidate_inspected": False,
        "selected_candidate_may_have_existed_in_memory": True,
        "selected_candidate_persisted": False,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "stderr_canonical_lf_bytes": 2132,
        "stderr_canonical_lf_sha256": (
            "fea8f50165e648799fd605459a470058bb6aa75e0df43b580610b9046c55d89d"
        ),
        "stderr_path": PREOUTCOME_FAILURE_STDERR_PATH,
        "stdout_bytes": 0,
        "stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
    }
    if receipt.get("content_sha256") != (
        "9e6c3d8776532fd0c6ffb1a205f5ff1375c7af4c92a62f9956c9eca60212b7b1"
    ):
        raise SystemExit("pre-outcome failure receipt content mismatch")
    if set(receipt) != set(expected) | {"content_sha256"}:
        raise SystemExit("pre-outcome failure receipt has an unexpected schema")
    for key, expected_value in expected.items():
        if receipt[key] != expected_value:
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


def validate_scientific_identity_with_superseded_lock(
    lock: dict[str, Any], prior_lock: dict[str, Any]
) -> None:
    """Permit only the enumerated execution/publication fields to differ from amendment 03."""

    allowed = set(ALLOWED_SUPERSEDED_LOCK_DIFFERENCES)
    if set(lock) != set(prior_lock):
        raise SystemExit("amendment-04 top-level lock schema differs from amendment 03")
    recorded = lock.get("resource_amendment", {}).get(
        "allowed_top_level_differences_from_superseded_lock"
    )
    if recorded != list(ALLOWED_SUPERSEDED_LOCK_DIFFERENCES):
        raise SystemExit("amendment-04 allowed top-level difference list mismatch")
    for key in sorted(set(lock) - allowed):
        if P.canonical_json(lock[key]) != P.canonical_json(prior_lock[key]):
            raise SystemExit(f"amendment-04 scientific field changed: {key}")


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


def implementation_binary_hashes(repo: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in IMPLEMENTATION_BINARY_FILES:
        path = repo / name
        if not path.is_file():
            raise SystemExit(f"missing implementation binary: {name}")
        hashes[name] = P.sha256_file(path)
    expected = {BOUNDED_WATERSHED_BINARY: BOUNDED_WATERSHED_BINARY_SHA256}
    if hashes != expected:
        raise SystemExit("bounded watershed binary binding mismatch")
    return hashes


def current_resource_amendment(
    repo: Path,
    failure_receipt: dict[str, Any],
) -> dict[str, Any]:
    return {
        "allowed_top_level_differences_from_superseded_lock": list(
            ALLOWED_SUPERSEDED_LOCK_DIFFERENCES
        ),
        "changed_execution_only": (
            "replace the nonbinding per-call heap estimate with one fixed full-capacity "
            "native sparse mapping before the unchanged compiled event stream"
        ),
        "compiled_event_core_changed": False,
        "disk_free_preflight_bytes": (
            HEAP_CAPACITY_ITEMS * EXPECTED_HEAP_ITEM_BYTES
            + MIN_FREE_AFTER_FULL_HEAP_BYTES
        ),
        "event_order_changed": False,
        "heap_capacity_is_event_bound": False,
        "heap_capacity_exhaustion": "abort_without_result",
        "heap_capacity_items_every_production_call": HEAP_CAPACITY_ITEMS,
        "heap_file_logical_bytes": HEAP_CAPACITY_ITEMS * EXPECTED_HEAP_ITEM_BYTES,
        "heap_item_bytes": EXPECTED_HEAP_ITEM_BYTES,
        "heap_sparse_allocation_reserves_disk_space": False,
        "heap_sparse_size_sequence_windows": [
            "FSCTL_SET_SPARSE",
            "SetFilePointerEx",
            "SetEndOfFile",
            "mmap_ACCESS_WRITE",
        ],
        "heap_storage": "native_sparse_file_backed_mmap_direct_item_binary_heap",
        "heap_stale_file_recovery": "remove_only_after_owning_pid_has_exited",
        "implementation_binary_files_sha256": implementation_binary_hashes(repo),
        "kind": "fixed_full_capacity_sparse_bit_equivalent_watershed_heap",
        "minimum_free_after_full_heap_bytes": MIN_FREE_AFTER_FULL_HEAP_BYTES,
        "signed_age_semantics": "pinned_scikit_image_int32_cast_including_wrap",
        "preoutcome_failure_receipt_content_sha256": failure_receipt[
            "content_sha256"
        ],
        "preoutcome_failure_receipt_path": PREOUTCOME_FAILURE_RECEIPT_PATH,
        "scientific_protocol_changed": False,
        "supersedes_protocol_lock_content_sha256": (
            SUPERSEDED_PROTOCOL_LOCK_CONTENT_SHA256
        ),
        "supersedes_protocol_lock_path": SUPERSEDED_PROTOCOL_LOCK_PATH,
        "watershed_reference": "scikit-image 0.26.0",
        "worker_group_key": ["scroll", "geometry.score_local_l1[0]"],
        "worker_parallelism": 1,
    }


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
    prior_lock = validate_superseded_lock_chain(repo)
    failure_receipt = validate_preoutcome_failure(repo)

    value: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": LOCK_STATUS,
        "execution_revision": EXECUTION_REVISION,
        "registration_evidence": "public git commit containing this protocol lock",
        "bridge_outcomes_seen_at_lock": False,
        "resource_amendment": current_resource_amendment(repo, failure_receipt),
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
    validate_scientific_identity_with_superseded_lock(value, prior_lock)
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
    prior_lock = validate_superseded_lock_chain(repo)
    failure_receipt = validate_preoutcome_failure(repo)
    expected_amendment = current_resource_amendment(repo, failure_receipt)
    if (
        lock.get("execution_revision") != EXECUTION_REVISION
        or lock.get("resource_amendment") != expected_amendment
    ):
        raise SystemExit("unexpected resource amendment")
    validate_scientific_identity_with_superseded_lock(lock, prior_lock)
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


def score_block_group(
    blocks: list[dict[str, Any]],
    array_refs: dict[str, tuple[Path, str]],
    candidates: Iterable[str],
    labels_root: Path,
) -> list[dict[str, Any]]:
    """Load and score one exact scroll/z0 group, sharing each truth-plane read."""

    grouped = partition_blocks_by_scroll_z0(blocks)
    if len(grouped) != 1:
        raise ValueError("a child must receive exactly one scroll/z0 group")
    candidate_names = tuple(candidates)
    arrays: dict[str, dict[str, Any]] = {}
    masks: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for block in blocks:
            block_id = block["block_id"]
            if block_id not in array_refs:
                raise SystemExit(f"{block_id}: verified array reference is missing")
            array_path, expected_sha256 = array_refs[block_id]
            if P.sha256_file(array_path) != expected_sha256:
                raise SystemExit(f"{block_id}: array changed after all-64 verification")
            arrays[block_id] = P._load_block_arrays(
                array_path, block, SOURCE_MANIFEST_CONTENT_SHA256
            )
        masks, audits = build_masks(blocks, arrays, candidate_names)
        rows = score_blocks(blocks, masks, audits, labels_root)
        expected_ids = sorted(block["block_id"] for block in blocks)
        if [row["block_id"] for row in rows] != expected_ids:
            raise AssertionError("group scorer row contract failed")
        return rows
    finally:
        audits.clear()
        masks.clear()
        arrays.clear()
        gc.collect()


def _block_group_key(block: dict[str, Any]) -> tuple[str, int]:
    """Return the exact label-plane group that owns a block's scoring jobs."""

    try:
        scroll = block["scroll"]
        score_box = block["geometry"]["score_local_l1"]
    except (KeyError, TypeError) as exc:
        raise ValueError("block is missing its scroll/score geometry") from exc
    if not isinstance(scroll, str) or scroll not in P.SCROLLS:
        raise ValueError(f"invalid block scroll: {scroll!r}")
    if not isinstance(score_box, list) or len(score_box) != 6:
        raise ValueError("score_local_l1 must be a six-integer list")
    if any(type(value) is not int for value in score_box):
        raise ValueError("score_local_l1 must contain only integers")
    return scroll, score_box[0]


def partition_blocks_by_scroll_z0(
    blocks: list[dict[str, Any]],
) -> list[tuple[tuple[str, int], list[dict[str, Any]]]]:
    """Partition every block exactly once without splitting a scroll/z0 group."""

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for block in blocks:
        block_id = block.get("block_id")
        if not isinstance(block_id, str) or not block_id:
            raise ValueError("every grouped block must have a nonempty string block_id")
        if block_id in seen:
            raise ValueError(f"duplicate grouped block_id: {block_id}")
        seen.add(block_id)
        grouped[_block_group_key(block)].append(block)
    return [(key, grouped[key]) for key in sorted(grouped)]


def _worker_request(
    group_key: tuple[str, int],
    blocks: list[dict[str, Any]],
    array_refs: dict[str, tuple[Path, str]],
    candidates: tuple[str, ...],
    labels_root: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    if not blocks:
        raise ValueError("a group worker request cannot be empty")
    grouped = partition_blocks_by_scroll_z0(blocks)
    if len(grouped) != 1 or grouped[0][0] != group_key:
        raise ValueError("group worker request does not match its exact scroll/z0 key")
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("group worker candidates must be nonempty and unique")
    unknown = [name for name in candidates if name not in CANDIDATE_CONFIGS]
    if unknown:
        raise ValueError(f"unknown group worker candidates: {unknown}")

    block_ids = [block["block_id"] for block in blocks]
    refs = []
    for block_id in block_ids:
        if block_id not in array_refs:
            raise SystemExit(f"{block_id}: verified array reference is missing")
        path, expected_sha256 = array_refs[block_id]
        path = path.resolve()
        if not _is_lower_hex(expected_sha256, 64):
            raise ValueError(f"{block_id}: invalid array SHA-256")
        refs.append(
            {
                "block_id": block_id,
                "path": str(path),
                "sha256": expected_sha256,
            }
        )

    request: dict[str, Any] = {
        "schema_version": GROUP_WORKER_SCHEMA_VERSION,
        "kind": GROUP_WORKER_REQUEST_KIND,
        "protocol_id": PROTOCOL_ID,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "protocol_lock_content_sha256": lock.get("content_sha256"),
        "implementation_commit": lock.get("implementation_commit"),
        "implementation_files_sha256": lock.get("implementation_files_sha256"),
        "implementation_binary_files_sha256": lock.get("resource_amendment", {}).get(
            "implementation_binary_files_sha256"
        ),
        "runtime_versions": runtime_versions(),
        "parent_pid": os.getpid(),
        "group": {"scroll": group_key[0], "z0": group_key[1]},
        "block_ids": block_ids,
        "blocks": blocks,
        "array_refs": refs,
        "candidate_names": list(candidates),
        "labels_root": str(labels_root.resolve()),
    }
    request["content_sha256"] = _content_sha(request)
    return request


def _validate_worker_request(
    request: dict[str, Any],
) -> tuple[
    tuple[str, int],
    list[dict[str, Any]],
    dict[str, tuple[Path, str]],
    tuple[str, ...],
    Path,
]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "source_manifest_content_sha256",
        "protocol_lock_content_sha256",
        "implementation_commit",
        "implementation_files_sha256",
        "implementation_binary_files_sha256",
        "runtime_versions",
        "parent_pid",
        "group",
        "block_ids",
        "blocks",
        "array_refs",
        "candidate_names",
        "labels_root",
        "content_sha256",
    }
    if set(request) != expected_keys:
        raise ValueError("group worker request has an unexpected schema")
    if request["content_sha256"] != _content_sha(request):
        raise ValueError("group worker request content SHA mismatch")
    expected_header = {
        "schema_version": GROUP_WORKER_SCHEMA_VERSION,
        "kind": GROUP_WORKER_REQUEST_KIND,
        "protocol_id": PROTOCOL_ID,
        "source_manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
        "runtime_versions": runtime_versions(),
    }
    for key, expected in expected_header.items():
        if request[key] != expected:
            raise ValueError(f"group worker request {key} mismatch")
    if not _is_lower_hex(request["protocol_lock_content_sha256"], 64):
        raise ValueError("group worker request has an invalid protocol-lock SHA-256")
    if not _is_lower_hex(request["implementation_commit"], 40):
        raise ValueError("group worker request has an invalid implementation commit")
    implementation_hashes = request["implementation_files_sha256"]
    if not isinstance(implementation_hashes, dict) or set(implementation_hashes) != set(
        IMPLEMENTATION_FILES
    ):
        raise ValueError("group worker request has an invalid implementation file set")
    if any(not _is_lower_hex(value, 64) for value in implementation_hashes.values()):
        raise ValueError("group worker request has an invalid implementation file SHA-256")
    binary_hashes = request["implementation_binary_files_sha256"]
    if (
        not isinstance(binary_hashes, dict)
        or set(binary_hashes) != set(IMPLEMENTATION_BINARY_FILES)
        or any(not _is_lower_hex(value, 64) for value in binary_hashes.values())
    ):
        raise ValueError("group worker request has invalid implementation binaries")
    if type(request["parent_pid"]) is not int or request["parent_pid"] <= 0:
        raise ValueError("group worker request has an invalid parent PID")

    group = request["group"]
    if not isinstance(group, dict) or set(group) != {"scroll", "z0"}:
        raise ValueError("group worker request has an invalid group")
    if (
        not isinstance(group["scroll"], str)
        or group["scroll"] not in P.SCROLLS
        or type(group["z0"]) is not int
    ):
        raise ValueError("group worker request has an invalid scroll/z0 key")
    group_key = (group["scroll"], group["z0"])

    blocks = request["blocks"]
    block_ids = request["block_ids"]
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("group worker request has no blocks")
    if not isinstance(block_ids, list) or block_ids != [
        block.get("block_id") if isinstance(block, dict) else None for block in blocks
    ]:
        raise ValueError("group worker request block_ids do not bind its blocks")
    grouped = partition_blocks_by_scroll_z0(blocks)
    if len(grouped) != 1 or grouped[0][0] != group_key:
        raise ValueError("group worker request fragments a scroll/z0 group")

    raw_refs = request["array_refs"]
    if not isinstance(raw_refs, list) or len(raw_refs) != len(block_ids):
        raise ValueError("group worker request has invalid array references")
    array_refs: dict[str, tuple[Path, str]] = {}
    for block_id, ref in zip(block_ids, raw_refs):
        if not isinstance(ref, dict) or set(ref) != {"block_id", "path", "sha256"}:
            raise ValueError("group worker request has a malformed array reference")
        if ref["block_id"] != block_id:
            raise ValueError("group worker array reference order mismatch")
        path = Path(ref["path"])
        sha256 = ref["sha256"]
        if not path.is_absolute():
            raise ValueError("group worker array paths must be absolute")
        if not _is_lower_hex(sha256, 64):
            raise ValueError("group worker array reference has an invalid SHA-256")
        array_refs[block_id] = (path, sha256)

    raw_candidates = request["candidate_names"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("group worker request has no candidates")
    if any(not isinstance(name, str) for name in raw_candidates):
        raise ValueError("group worker candidate names must be strings")
    candidates = tuple(raw_candidates)
    if len(set(candidates)) != len(candidates) or any(
        name not in CANDIDATE_CONFIGS for name in candidates
    ):
        raise ValueError("group worker request has invalid candidates")

    labels_root = Path(request["labels_root"])
    if not labels_root.is_absolute():
        raise ValueError("group worker labels root must be absolute")
    return group_key, blocks, array_refs, candidates, labels_root


def _verify_worker_implementation_bindings(request: dict[str, Any]) -> None:
    """Independently bind a fresh worker to the frozen implementation on disk."""

    repo = Path(__file__).resolve().parent
    implementation_commit = request["implementation_commit"]
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation_commit, "HEAD"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise SystemExit("group worker implementation commit is not in public HEAD")
    for name, expected in sorted(request["implementation_files_sha256"].items()):
        path = repo / name
        if not path.is_file() or canonical_lf_sha256(path) != expected:
            raise SystemExit(f"group worker implementation drift: {name}")
    for name, expected in sorted(
        request["implementation_binary_files_sha256"].items()
    ):
        path = repo / name
        if not path.is_file() or P.sha256_file(path) != expected:
            raise SystemExit(f"group worker implementation binary drift: {name}")


def _worker_response(
    request: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema_version": GROUP_WORKER_SCHEMA_VERSION,
        "kind": GROUP_WORKER_RESPONSE_KIND,
        "protocol_id": PROTOCOL_ID,
        "request_content_sha256": request["content_sha256"],
        "protocol_lock_content_sha256": request["protocol_lock_content_sha256"],
        "implementation_commit": request["implementation_commit"],
        "implementation_files_sha256": request["implementation_files_sha256"],
        "implementation_binary_files_sha256": request[
            "implementation_binary_files_sha256"
        ],
        "runtime_versions": runtime_versions(),
        "parent_pid": request["parent_pid"],
        "worker_pid": os.getpid(),
        "group": request["group"],
        "block_ids": request["block_ids"],
        "candidate_names": request["candidate_names"],
        "rows": rows,
    }
    response["content_sha256"] = _content_sha(response)
    return response


def _validate_worker_response(
    response: dict[str, Any], request: dict[str, Any], launched_pid: int
) -> list[dict[str, Any]]:
    expected_keys = {
        "schema_version",
        "kind",
        "protocol_id",
        "request_content_sha256",
        "protocol_lock_content_sha256",
        "implementation_commit",
        "implementation_files_sha256",
        "implementation_binary_files_sha256",
        "runtime_versions",
        "parent_pid",
        "worker_pid",
        "group",
        "block_ids",
        "candidate_names",
        "rows",
        "content_sha256",
    }
    if set(response) != expected_keys:
        raise SystemExit("group worker response has an unexpected schema")
    if response["content_sha256"] != _content_sha(response):
        raise SystemExit("group worker response content SHA mismatch")
    expected_bindings = {
        "schema_version": GROUP_WORKER_SCHEMA_VERSION,
        "kind": GROUP_WORKER_RESPONSE_KIND,
        "protocol_id": PROTOCOL_ID,
        "request_content_sha256": request["content_sha256"],
        "protocol_lock_content_sha256": request["protocol_lock_content_sha256"],
        "implementation_commit": request["implementation_commit"],
        "implementation_files_sha256": request["implementation_files_sha256"],
        "implementation_binary_files_sha256": request[
            "implementation_binary_files_sha256"
        ],
        "runtime_versions": runtime_versions(),
        "parent_pid": request["parent_pid"],
        "worker_pid": launched_pid,
        "group": request["group"],
        "block_ids": request["block_ids"],
        "candidate_names": request["candidate_names"],
    }
    for key, expected in expected_bindings.items():
        if response[key] != expected:
            raise SystemExit(f"group worker response {key} mismatch")
    if type(launched_pid) is not int or launched_pid <= 0 or launched_pid == os.getpid():
        raise SystemExit("group worker did not run in a fresh child process")

    rows = response["rows"]
    if not isinstance(rows, list):
        raise SystemExit("group worker response rows must be a list")
    expected_ids = sorted(request["block_ids"])
    if [row.get("block_id") if isinstance(row, dict) else None for row in rows] != expected_ids:
        raise SystemExit("group worker response rows do not match the requested blocks")
    blocks_by_id = {block["block_id"]: block for block in request["blocks"]}
    expected_arms = {"corrected_fixed"}
    expected_audits: set[str] = set()
    for candidate in request["candidate_names"]:
        expected_arms.update({candidate, matched_budget_arm(candidate)})
        expected_audits.update({candidate, matched_budget_arm(candidate)})
    for row in rows:
        if set(row) != {"block_id", "scroll", "z_stratum", "arms", "audits"}:
            raise SystemExit("group worker response row has an unexpected schema")
        block = blocks_by_id[row["block_id"]]
        if row["scroll"] != block["scroll"] or row["z_stratum"] != block["z_stratum"]:
            raise SystemExit("group worker response row metadata mismatch")
        if not isinstance(row["arms"], dict) or set(row["arms"]) != expected_arms:
            raise SystemExit("group worker response row arm set mismatch")
        if not isinstance(row["audits"], dict) or set(row["audits"]) != expected_audits:
            raise SystemExit("group worker response row audit set mismatch")
    return rows


def _launch_group_worker(request: dict[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_score-group-worker",
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        try:
            stdout, stderr = process.communicate(P.canonical_json(request))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    finally:
        if process.poll() is not None:
            cleanup_heap_files_for_pid(process.pid)
    return {
        "pid": process.pid,
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def _worker_failure_detail(value: str, limit: int = 4000) -> str:
    value = value.strip()
    return value[-limit:] if value else "<empty>"


def score_blocks_spawned(
    blocks: list[dict[str, Any]],
    array_refs: dict[str, tuple[Path, str]],
    candidates: Iterable[str],
    labels_root: Path,
    lock: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score exact scroll/z0 jobs in serial, short-lived child processes.

    A child must exit successfully and return one strictly bound, hashed response before the
    next child is created.  Thus native watershed allocations cannot accumulate across groups.
    """

    candidate_names = tuple(candidates)
    jobs = partition_blocks_by_scroll_z0(blocks)
    rows: list[dict[str, Any]] = []
    for group_key, group_blocks in jobs:
        request = _worker_request(
            group_key,
            group_blocks,
            array_refs,
            candidate_names,
            labels_root,
            lock,
        )
        launched = _launch_group_worker(request)
        if launched["returncode"] != 0:
            raise SystemExit(
                f"group {group_key!r} worker PID {launched['pid']} failed with exit "
                f"{launched['returncode']}; stderr: "
                f"{_worker_failure_detail(launched['stderr'])}"
            )
        if launched["stderr"]:
            raise SystemExit(
                f"group {group_key!r} worker PID {launched['pid']} emitted unexpected "
                f"stderr: {_worker_failure_detail(launched['stderr'])}"
            )
        stdout = launched["stdout"]
        if not stdout:
            raise SystemExit(
                f"group {group_key!r} worker PID {launched['pid']} returned no response"
            )
        try:
            response = json.loads(stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise SystemExit(
                f"group {group_key!r} worker PID {launched['pid']} returned malformed JSON"
            ) from exc
        if not isinstance(response, dict):
            raise SystemExit(
                f"group {group_key!r} worker PID {launched['pid']} returned a non-object response"
            )
        if stdout != P.canonical_json(response):
            raise SystemExit(
                f"group {group_key!r} worker PID {launched['pid']} returned noncanonical JSON"
            )
        rows.extend(_validate_worker_response(response, request, launched["pid"]))

    expected_ids = sorted(block["block_id"] for block in blocks)
    rows.sort(key=lambda row: row["block_id"])
    if [row["block_id"] for row in rows] != expected_ids:
        raise SystemExit("spawned group jobs did not return the exact requested partition")
    return rows


def group_worker_command(args: argparse.Namespace) -> None:
    del args
    request_text = sys.stdin.read()
    if not request_text:
        raise SystemExit("group worker received no request")
    try:
        request = json.loads(request_text)
    except json.JSONDecodeError as exc:
        raise SystemExit("group worker request is malformed JSON") from exc
    if not isinstance(request, dict):
        raise SystemExit("group worker request must be a JSON object")
    if request_text != P.canonical_json(request):
        raise SystemExit("group worker request is not canonical JSON")
    _, blocks, array_refs, candidates, labels_root = _validate_worker_request(request)
    if hasattr(os, "getppid") and os.getppid() != request["parent_pid"]:
        raise SystemExit("group worker parent PID mismatch")
    _verify_worker_implementation_bindings(request)
    rows = score_block_group(blocks, array_refs, candidates, labels_root)
    _verify_worker_implementation_bindings(request)
    response = _worker_response(request, rows)
    sys.stdout.write(P.canonical_json(response))
    sys.stdout.flush()


def reverify_before_result_write(
    repo: Path,
    lock_path: Path,
    expected_lock_content_sha256: str,
    expected_manifest_content_sha256: str,
    expected_public_head: str,
) -> None:
    """Recheck every public binding after workers finish and before any result is written."""

    lock = load_hashed_json(lock_path)
    if lock["content_sha256"] != expected_lock_content_sha256:
        raise SystemExit("protocol lock changed during bridge scoring")
    manifest = verify_protocol_files(repo, lock)
    if manifest["content_sha256"] != expected_manifest_content_sha256:
        raise SystemExit("source manifest changed during bridge scoring")
    head = verify_public_freeze(repo, lock, lock_path)
    if head != expected_public_head:
        raise SystemExit("public freeze HEAD changed during bridge scoring")


def score_command(args: argparse.Namespace) -> None:
    repo = Path(__file__).resolve().parent
    result_path = Path(args.result).resolve()
    if result_path.exists():
        raise SystemExit(f"refusing to overwrite existing bridge result: {result_path}")
    cleanup_stale_heap_files()
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
    dev_rows = score_blocks_spawned(
        dev_blocks, array_refs, CANDIDATE_CONFIGS, labels_root, lock
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
        holdout_rows = score_blocks_spawned(
            holdout_blocks, array_refs, [selected], labels_root, lock
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
    reverify_before_result_write(
        repo,
        lock_path,
        lock["content_sha256"],
        manifest["content_sha256"],
        public_freeze_commit,
    )
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
    worker = sub.add_parser("_score-group-worker", help=argparse.SUPPRESS)
    worker.set_defaults(func=group_worker_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
