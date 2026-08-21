from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from skimage.morphology._util import _offsets_to_raveled_neighbors, _validate_connectivity
from skimage.segmentation import watershed as reference_watershed
from skimage.segmentation._watershed import _validate_inputs
from skimage.util import crop

import bounded_watershed as B
import probability_bridge_split as P

TEST_HEAP_CAPACITY_ITEMS = 1_000_000


def test_binary_and_heap_layout_are_frozen() -> None:
    binary = Path(B.__file__).resolve().parent / B.BOUNDED_WATERSHED_BINARY
    assert binary.is_file()
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == (
        B.BOUNDED_WATERSHED_BINARY_SHA256
    )
    assert B.heap_file_bytes(1) == B.EXPECTED_HEAP_ITEM_BYTES == 24
    assert B.HEAP_CAPACITY_ITEMS == 2_666_666_666
    assert B.heap_file_bytes() == 63_999_999_984
    assert B._core.heap_item_layout() == {
        "size": 24,
        "value": 0,
        "age": 8,
        "index": 12,
        "source": 16,
    }


def test_compact_index_guard_has_exact_signed_int32_boundary() -> None:
    assert B._validate_compact_index_space(
        (B.MAX_RAVELED_ELEMENTS,), (0,)
    ) == B.MAX_RAVELED_ELEMENTS
    assert (
        B._core.validate_raveled_element_count(B.MAX_RAVELED_ELEMENTS)
        == B.MAX_RAVELED_ELEMENTS
    )
    with pytest.raises(ValueError, match="signed-int32 index range"):
        B._validate_compact_index_space((B.MAX_RAVELED_ELEMENTS + 1,), (0,))
    with pytest.raises(ValueError, match="signed-int32 index range"):
        B._core.validate_raveled_element_count(B.MAX_RAVELED_ELEMENTS + 1)


def test_frozen_probability_arrays_are_far_inside_compact_index_range() -> None:
    repo = Path(B.__file__).resolve().parent
    manifest = json.loads(
        (repo / "results" / "physical_normalization_ab" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    padded_sizes = []
    for block in manifest["blocks"]:
        extent = block["geometry"]["prediction_extent_local_l1"]
        shape = tuple(extent[i + 1] - extent[i] for i in (0, 2, 4))
        padded_sizes.append(B._validate_compact_index_space(shape, (1, 1, 1)))
    assert max(padded_sizes) == 790_152
    assert max(padded_sizes) < B.MAX_RAVELED_ELEMENTS


def test_default_uses_fixed_full_capacity_without_estimator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacities: list[int] = []

    @contextmanager
    def storage(capacity_items: int, *, scratch_directory: Path):
        capacities.append(capacity_items)
        assert scratch_directory == tmp_path
        yield bytearray(24_000)

    monkeypatch.setattr(B, "_heap_storage", storage)
    image = np.zeros((3, 3), dtype=np.float32)
    markers = np.zeros_like(image, dtype=np.int32)
    markers[0, 0] = 1
    markers[2, 2] = 2
    actual = B.watershed(
        image,
        markers=markers,
        mask=np.ones_like(image, dtype=bool),
        watershed_line=True,
        _scratch_directory=tmp_path,
    )
    expected = reference_watershed(
        image,
        markers=markers,
        mask=np.ones_like(image, dtype=bool),
        watershed_line=True,
    )
    assert np.array_equal(actual, expected)
    assert capacities == [B.HEAP_CAPACITY_ITEMS]


def test_native_sparse_heap_file_is_sized_and_cleaned(tmp_path: Path) -> None:
    capacity_items = 4096
    with B._heap_storage(capacity_items, scratch_directory=tmp_path) as storage:
        assert len(storage) == B.heap_file_bytes(capacity_items)
        files = list(tmp_path.iterdir())
        assert len(files) == 1
        if sys.platform == "win32":
            assert files[0].stat().st_file_attributes & stat.FILE_ATTRIBUTE_SPARSE_FILE
        storage[0] = 17
        storage[-1] = 23
    assert list(tmp_path.iterdir()) == []


def test_random_plateau_partitions_match_scikit_image(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260821)
    values = np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=np.float32)
    checked = 0
    for case in range(300):
        ndim = 2 if case % 2 == 0 else 3
        shape = tuple(int(rng.integers(2, 7)) for _ in range(ndim))
        image = rng.choice(values, size=shape)
        mask = rng.random(shape) > 0.15
        eligible = np.flatnonzero(mask)
        if eligible.size < 2:
            continue
        marker_count = min(eligible.size, int(rng.integers(2, 5)))
        chosen = rng.choice(eligible, size=marker_count, replace=False)
        markers = np.zeros(shape, dtype=np.int32)
        markers.flat[chosen] = np.arange(1, marker_count + 1, dtype=np.int32)
        connectivity = int(rng.integers(1, ndim + 1))
        watershed_line = bool(case % 3)

        expected = reference_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            watershed_line=watershed_line,
        )
        actual = B.watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            watershed_line=watershed_line,
            _heap_capacity_items=TEST_HEAP_CAPACITY_ITEMS,
            _scratch_directory=tmp_path,
        )
        assert np.array_equal(actual, expected), (case, shape, connectivity)
        checked += 1

    assert checked >= 290
    assert list(tmp_path.iterdir()) == []


def test_multi_voxel_repeated_label_markers_match_scikit_image(
    tmp_path: Path,
) -> None:
    image = np.zeros((9, 10, 11), dtype=np.float32)
    image[4, :, :] = -0.25
    mask = np.ones_like(image, dtype=bool)
    mask[0, 0, 0] = False
    markers = np.zeros_like(image, dtype=np.int32)
    markers[1:3, 1:4, 1:3] = 1
    markers[6:8, 6:9, 7:10] = 2
    markers[1, 8, 9] = 1

    for connectivity in (1, 2, 3):
        for watershed_line in (False, True):
            expected = reference_watershed(
                image,
                markers=markers,
                mask=mask,
                connectivity=connectivity,
                watershed_line=watershed_line,
            )
            actual = B.watershed(
                image,
                markers=markers,
                mask=mask,
                connectivity=connectivity,
                watershed_line=watershed_line,
                _heap_capacity_items=TEST_HEAP_CAPACITY_ITEMS,
                _scratch_directory=tmp_path,
            )
            assert np.array_equal(actual, expected)
    assert list(tmp_path.iterdir()) == []


def test_compact_source_indices_match_scikit_image_with_compactness(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(2026082102)
    image = rng.normal(size=(8, 9, 10)).astype(np.float32)
    mask = np.ones_like(image, dtype=bool)
    mask[0, 0, 0] = False
    markers = np.zeros_like(image, dtype=np.int32)
    markers[1, 1, 1] = 1
    markers[6, 7, 8] = 2
    markers[2, 6, 4] = 3
    for watershed_line in (False, True):
        expected = reference_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=3,
            compactness=0.125,
            watershed_line=watershed_line,
        )
        actual = B.watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=3,
            compactness=0.125,
            watershed_line=watershed_line,
            _heap_capacity_items=TEST_HEAP_CAPACITY_ITEMS,
            _scratch_directory=tmp_path,
        )
        assert np.array_equal(actual, expected)
    assert list(tmp_path.iterdir()) == []


def _direct_core_watershed(
    image: np.ndarray,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int,
    storage: bytearray,
) -> np.ndarray:
    image, markers, mask = _validate_inputs(image, markers, mask, connectivity)
    connectivity, offset = _validate_connectivity(image.ndim, connectivity, None)
    pad_width = [(padding, padding) for padding in offset]
    image = np.pad(image, pad_width, mode="constant")
    mask = np.pad(mask, pad_width, mode="constant").ravel()
    output = np.pad(markers, pad_width, mode="constant")
    neighbors = _offsets_to_raveled_neighbors(
        image.shape,
        connectivity,
        center=offset,
    )
    strides = np.array(image.strides, dtype=np.intp) // image.itemsize
    B._core.watershed_raveled_bounded(
        image.ravel(),
        np.flatnonzero(output),
        neighbors,
        mask,
        strides,
        0.0,
        output.ravel(),
        True,
        storage,
    )
    return crop(output, pad_width, copy=True)


def test_compiled_event_stream_matches_5000_random_cases() -> None:
    storage = bytearray(24_000_000)
    rng = np.random.default_rng(782601)
    values = np.array([-1.0, -0.5, 0.0, 0.25, 1.0], dtype=np.float32)
    checked = 0
    for case in range(5_000):
        ndim = 2 if case % 2 == 0 else 3
        shape = tuple(int(rng.integers(2, 8)) for _ in range(ndim))
        image = rng.choice(values, size=shape)
        mask = rng.random(shape) > 0.12
        eligible = np.flatnonzero(mask)
        if eligible.size < 2:
            continue
        marker_count = min(eligible.size, int(rng.integers(2, 6)))
        chosen = rng.choice(eligible, size=marker_count, replace=False)
        markers = np.zeros(shape, dtype=np.int32)
        markers.flat[chosen] = np.arange(1, marker_count + 1, dtype=np.int32)
        connectivity = int(rng.integers(1, ndim + 1))
        expected = reference_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            watershed_line=True,
        )
        actual = _direct_core_watershed(
            image,
            markers,
            mask,
            connectivity,
            storage,
        )
        assert np.array_equal(actual, expected), (case, shape, connectivity)
        checked += 1
    assert checked >= 4_990


def test_forced_capacity_failure_is_fail_closed_and_cleans(tmp_path: Path) -> None:
    image = np.zeros((5, 5, 5), dtype=np.float32)
    markers = np.zeros_like(image, dtype=np.int32)
    markers[1, 1, 1] = 1
    markers[3, 3, 3] = 2
    with pytest.raises(MemoryError, match="heap capacity exhausted"):
        B.watershed(
            image,
            markers=markers,
            mask=np.ones_like(image, dtype=bool),
            connectivity=3,
            watershed_line=True,
            _heap_capacity_items=2,
            _scratch_directory=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_repeated_run_is_byte_identical(tmp_path: Path) -> None:
    image = np.zeros((11, 13, 15), dtype=np.float32)
    image[:, 6, :] = -0.25
    markers = np.zeros_like(image, dtype=np.int32)
    markers[2, 2, 2] = 1
    markers[8, 10, 12] = 2
    outputs = [
        B.watershed(
            image,
            markers=markers,
            mask=np.ones_like(image, dtype=bool),
            connectivity=3,
            watershed_line=True,
            _heap_capacity_items=TEST_HEAP_CAPACITY_ITEMS,
            _scratch_directory=tmp_path,
        )
        for _ in range(3)
    ]
    assert all(output.tobytes() == outputs[0].tobytes() for output in outputs[1:])
    assert outputs[0].tobytes() == reference_watershed(
        image,
        markers=markers,
        mask=np.ones_like(image, dtype=bool),
        connectivity=3,
        watershed_line=True,
    ).tobytes()


def test_bridge_mask_and_audit_match_reference_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probability = np.zeros((15, 15, 25), dtype=np.float32)
    probability[4:11, 4:11, 2:9] = 0.9
    probability[4:11, 4:11, 16:23] = 0.9
    probability[7, 7, 8:17] = 0.25
    config = P.BridgeSplitConfig(
        min_seed_voxels=8,
        min_output_component_voxels=8,
        max_removed_fraction=0.05,
        connectivity=3,
    )

    monkeypatch.setattr(P, "watershed", reference_watershed)
    reference = P.split_probability_bridges(probability, config)

    def bounded_for_test(*args, **kwargs):
        return B.watershed(
            *args,
            **kwargs,
            _heap_capacity_items=TEST_HEAP_CAPACITY_ITEMS,
            _scratch_directory=tmp_path,
        )

    monkeypatch.setattr(P, "watershed", bounded_for_test)
    actual = P.split_probability_bridges(probability, config)
    assert actual.mask.tobytes() == reference.mask.tobytes()
    assert actual.removed.tobytes() == reference.removed.tobytes()
    assert actual.audit == reference.audit
    assert list(tmp_path.iterdir()) == []


def test_parent_pid_cleanup_targets_only_owned_prefix() -> None:
    fake_pid = 2_000_000_001
    B.SCRATCH_DIRECTORY.mkdir(parents=True, exist_ok=True)
    owned = B.SCRATCH_DIRECTORY / (
        f"{B.HEAP_FILE_PREFIX}{fake_pid}-unit{B.HEAP_FILE_SUFFIX}"
    )
    unrelated = B.SCRATCH_DIRECTORY / "unrelated.keep"
    owned.write_bytes(b"owned")
    unrelated.write_bytes(b"keep")
    try:
        B.cleanup_heap_files_for_pid(fake_pid)
        assert not owned.exists()
        assert unrelated.read_bytes() == b"keep"
    finally:
        if owned.exists():
            owned.unlink()
        if unrelated.exists():
            unrelated.unlink()


def test_startup_cleanup_removes_only_exited_pid_files(tmp_path: Path) -> None:
    dead_pid = 2_000_000_001
    dead = tmp_path / (
        f"{B.HEAP_FILE_PREFIX}{dead_pid}-dead{B.HEAP_FILE_SUFFIX}"
    )
    live = tmp_path / (
        f"{B.HEAP_FILE_PREFIX}{os.getpid()}-live{B.HEAP_FILE_SUFFIX}"
    )
    unrelated = tmp_path / "unrelated.keep"
    dead.write_bytes(b"dead")
    live.write_bytes(b"live")
    unrelated.write_bytes(b"keep")

    B.cleanup_stale_heap_files(tmp_path)

    assert not dead.exists()
    assert live.read_bytes() == b"live"
    assert unrelated.read_bytes() == b"keep"
