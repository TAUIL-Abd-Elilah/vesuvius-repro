from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from probability_bridge_split import BridgeSplitConfig, split_probability_bridges


TEST_CONFIG = BridgeSplitConfig(
    low_threshold=0.2,
    persistence_threshold=0.4,
    seed_threshold=0.6,
    cut_ceiling=0.35,
    min_seed_voxels=8,
    min_output_component_voxels=16,
    max_removed_fraction=0.10,
)


def two_blocks_with_bridge(bridge_probability: float = 0.25) -> np.ndarray:
    p = np.zeros((15, 15, 25), dtype=np.float32)
    p[4:11, 4:11, 2:8] = 0.9
    p[4:11, 4:11, 17:23] = 0.9
    p[7, 7, 8:17] = bridge_probability
    return p


def component_count(mask: np.ndarray) -> int:
    structure = ndi.generate_binary_structure(mask.ndim, mask.ndim)
    return int(ndi.label(mask, structure=structure)[1])


def test_low_confidence_bridge_is_split_without_touching_seeds() -> None:
    p = two_blocks_with_bridge()
    result = split_probability_bridges(p, TEST_CONFIG)

    assert component_count(p > TEST_CONFIG.low_threshold) == 1
    assert component_count(result.mask) == 2
    assert result.audit["accepted_components"] == 1
    assert result.audit["removed_voxels"] > 0
    assert np.all(result.mask[p > TEST_CONFIG.seed_threshold])
    assert not np.any(result.mask & ~(p > TEST_CONFIG.low_threshold))
    assert np.all(p[result.removed] <= TEST_CONFIG.cut_ceiling)


def test_persistent_bridge_is_not_split() -> None:
    p = two_blocks_with_bridge(bridge_probability=0.5)
    result = split_probability_bridges(p, TEST_CONFIG)
    assert np.array_equal(result.mask, p > TEST_CONFIG.low_threshold)
    assert result.audit["accepted_components"] == 0


def test_seam_above_cut_ceiling_fails_closed() -> None:
    p = two_blocks_with_bridge(bridge_probability=0.36)
    result = split_probability_bridges(p, TEST_CONFIG)
    assert np.array_equal(result.mask, p > TEST_CONFIG.low_threshold)
    assert result.audit["accepted_components"] == 0
    assert result.audit["rejected"]["empty_seam"] == 1


def test_editable_mask_cannot_produce_partial_cut() -> None:
    p = two_blocks_with_bridge()
    editable = np.ones(p.shape, dtype=bool)
    editable[:, :, 11:14] = False
    result = split_probability_bridges(p, TEST_CONFIG, editable_mask=editable)
    assert np.array_equal(result.mask, p > TEST_CONFIG.low_threshold)
    assert not result.removed.any()
    assert sum(result.audit["rejected"].values()) == 1


def test_tiny_seed_does_not_authorize_a_split() -> None:
    p = two_blocks_with_bridge()
    config = BridgeSplitConfig(
        low_threshold=0.2,
        persistence_threshold=0.4,
        seed_threshold=0.6,
        cut_ceiling=0.35,
        min_seed_voxels=400,
        min_output_component_voxels=16,
        max_removed_fraction=0.1,
    )
    result = split_probability_bridges(p, config)
    assert np.array_equal(result.mask, p > config.low_threshold)
    assert result.audit["components_with_multiple_markers"] == 0


def test_edit_budget_rejects_large_seam() -> None:
    p = two_blocks_with_bridge()
    config = BridgeSplitConfig(
        low_threshold=0.2,
        persistence_threshold=0.4,
        seed_threshold=0.6,
        cut_ceiling=0.35,
        min_seed_voxels=8,
        min_output_component_voxels=16,
        max_removed_fraction=1e-6,
    )
    result = split_probability_bridges(p, config)
    assert np.array_equal(result.mask, p > config.low_threshold)
    assert result.audit["rejected"]["edit_budget"] == 1


def test_deterministic_and_idempotent_relative_to_probability() -> None:
    p = two_blocks_with_bridge()
    first = split_probability_bridges(p, TEST_CONFIG)
    second = split_probability_bridges(p.copy(), TEST_CONFIG)
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.removed, second.removed)
    assert first.audit == second.audit


def test_contained_crop_has_same_edit() -> None:
    small = two_blocks_with_bridge()
    padded = np.zeros((25, 25, 35), dtype=np.float32)
    padded[5:20, 5:20, 5:30] = small
    small_result = split_probability_bridges(small, TEST_CONFIG)
    padded_result = split_probability_bridges(padded, TEST_CONFIG)
    assert np.array_equal(small_result.mask, padded_result.mask[5:20, 5:20, 5:30])


def test_empty_and_single_sheet_are_noops() -> None:
    empty = np.zeros((8, 8, 8), dtype=np.float32)
    assert not split_probability_bridges(empty, TEST_CONFIG).mask.any()

    single = np.zeros((12, 12, 12), dtype=np.float32)
    single[3:9, 3:9, 5:7] = 0.9
    result = split_probability_bridges(single, TEST_CONFIG)
    assert np.array_equal(result.mask, single > TEST_CONFIG.low_threshold)


@pytest.mark.parametrize(
    "probability",
    [
        np.array([[[np.nan]]], dtype=np.float32),
        np.array([[[-0.1]]], dtype=np.float32),
        np.array([[[1.1]]], dtype=np.float32),
    ],
)
def test_invalid_probability_rejected(probability: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite and in"):
        split_probability_bridges(probability, TEST_CONFIG)


def test_invalid_threshold_order_rejected() -> None:
    config = BridgeSplitConfig(cut_ceiling=0.5, persistence_threshold=0.4)
    with pytest.raises(ValueError, match="thresholds must satisfy"):
        split_probability_bridges(np.zeros((3, 3, 3), dtype=np.float32), config)


def test_editable_shape_rejected() -> None:
    with pytest.raises(ValueError, match="editable_mask shape"):
        split_probability_bridges(
            np.zeros((3, 3, 3), dtype=np.float32),
            TEST_CONFIG,
            editable_mask=np.zeros((2, 2, 2), dtype=bool),
        )
