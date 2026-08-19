#!/usr/bin/env python3
"""Locked physical metrics for a native ScrollFiesta binary mask.

This module is deliberately array-only.  It does not discover predictions or
truth on disk, so callers must bind those inputs before invoking ``score_mask``.
"""

from __future__ import annotations

import math
from itertools import product
from typing import Mapping

import numpy as np
from scipy import ndimage


SCHEMA = "crossscan-scrollfiesta-physical-score-v1"
METRIC_LOCK_CONTENT_SHA256 = (
    "70c29b370b1f6ca2bb7f6d78eb284e456187056d2ed7efb86c7b5950e976f42c"
)
TRUTH_SHAPE_L1 = (128, 128, 128)
MASK_SHAPE_L0 = (256, 256, 256)
UPSAMPLE = 2
CUBE_SIZE_L0 = 128
MINIMUM_COMPONENT_VOXELS_L0 = 504
MATCH_TOLERANCE_L0_CHEBYSHEV = 2
CONNECTIVITY = 26
_FULL_NEIGHBOURHOOD = np.ones((3, 3, 3), dtype=bool)
_COMPONENT_KEYS = (
    "truth_components",
    "prediction_components",
    "missed",
    "spurious",
    "split_excess",
    "merger_excess",
    "total_component_errors",
)


def _expanded_truth(bits_l1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bits = np.asarray(bits_l1)
    if bits.shape != TRUTH_SHAPE_L1 or bits.dtype != np.uint8:
        raise ValueError(
            f"truth bits must be a uint8 {TRUTH_SHAPE_L1} array; "
            f"got {bits.shape} {bits.dtype}"
        )
    valid = (bits & 1) != 0
    material = (bits & 2) != 0
    recto = (bits & 8) != 0
    positive_l1 = valid & recto
    supervised_l1 = positive_l1 | (valid & ~material & ~recto)

    def expand(value: np.ndarray) -> np.ndarray:
        result = value
        for axis in range(3):
            result = np.repeat(result, UPSAMPLE, axis=axis)
        return result

    return expand(positive_l1), expand(supervised_l1)


def _prediction_mask(mask_l0: np.ndarray) -> np.ndarray:
    value = np.asarray(mask_l0)
    if value.shape != MASK_SHAPE_L0 or value.dtype != np.uint8:
        raise ValueError(
            f"prediction must be a uint8 {MASK_SHAPE_L0} array; "
            f"got {value.shape} {value.dtype}"
        )
    if np.any((value != 0) & (value != 255)):
        raise ValueError("prediction values must be exactly 0 or 255")
    return value == 255


def _classification_metrics(
    prediction: np.ndarray, positive: np.ndarray, supervised: np.ndarray
) -> dict:
    true_positive = int(np.count_nonzero(prediction & positive))
    false_positive = int(np.count_nonzero(prediction & supervised & ~positive))
    false_negative = int(np.count_nonzero(~prediction & positive))
    selected_ignored = int(np.count_nonzero(prediction & ~supervised))
    predicted = int(np.count_nonzero(prediction))
    positive_count = int(np.count_nonzero(positive))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    dice_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive_supervised": false_positive,
        "false_negative": false_negative,
        "selected_ignored": selected_ignored,
        "prediction_foreground_voxels_l0": predicted,
        "truth_positive_voxels_l0": positive_count,
        "prediction_foreground_fraction": float(predicted / prediction.size),
        "precision": float(
            true_positive / precision_denominator if precision_denominator else 0.0
        ),
        "recall": float(
            true_positive / recall_denominator if recall_denominator else 0.0
        ),
        "dice": float(
            2 * true_positive / dice_denominator if dice_denominator else 0.0
        ),
    }


def _surviving_labels(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(mask, structure=_FULL_NEIGHBOURHOOD)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    surviving = np.flatnonzero(sizes >= MINIMUM_COMPONENT_VOXELS_L0)
    surviving = surviving[surviving != 0]
    remap = np.zeros(count + 1, dtype=np.int32)
    remap[surviving] = np.arange(1, len(surviving) + 1, dtype=np.int32)
    return remap[labels], int(len(surviving))


def _expanded_slice(
    item: tuple[slice, slice, slice], shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(
            max(0, axis_slice.start - MATCH_TOLERANCE_L0_CHEBYSHEV),
            min(axis_size, axis_slice.stop + MATCH_TOLERANCE_L0_CHEBYSHEV),
        )
        for axis_slice, axis_size in zip(item, shape)
    )


def _component_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth_labels, truth_count = _surviving_labels(truth)
    prediction_labels, prediction_count = _surviving_labels(prediction)
    truth_degree = np.zeros(truth_count, dtype=np.int64)
    prediction_degree = np.zeros(prediction_count, dtype=np.int64)

    if truth_count and prediction_count:
        for truth_index, item in enumerate(ndimage.find_objects(truth_labels), 1):
            if item is None:
                continue
            region_slice = _expanded_slice(item, truth.shape)
            truth_region = truth_labels[region_slice] == truth_index
            near_truth = ndimage.binary_dilation(
                truth_region,
                structure=_FULL_NEIGHBOURHOOD,
                iterations=MATCH_TOLERANCE_L0_CHEBYSHEV,
                border_value=0,
            )
            neighbours = np.unique(prediction_labels[region_slice][near_truth])
            neighbours = neighbours[neighbours != 0]
            truth_degree[truth_index - 1] = len(neighbours)
            if len(neighbours):
                prediction_degree[neighbours - 1] += 1

    missed = int(np.count_nonzero(truth_degree == 0))
    spurious = int(np.count_nonzero(prediction_degree == 0))
    split_excess = int(np.maximum(truth_degree - 1, 0).sum())
    merger_excess = int(np.maximum(prediction_degree - 1, 0).sum())
    return {
        "truth_components": truth_count,
        "prediction_components": prediction_count,
        "missed": missed,
        "spurious": spurious,
        "split_excess": split_excess,
        "merger_excess": merger_excess,
        "total_component_errors": (
            missed + spurious + split_excess + merger_excess
        ),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(
        mask, structure=_FULL_NEIGHBOURHOOD, border_value=0
    )
    return mask & ~eroded


def _surface_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    truth_surface = _surface(truth)
    prediction_surface = _surface(prediction)
    truth_count = int(np.count_nonzero(truth_surface))
    prediction_count = int(np.count_nonzero(prediction_surface))
    if not truth_count or not prediction_count:
        raise ValueError("surface distance requires nonempty truth and prediction surfaces")

    occupied = truth_surface | prediction_surface
    crop = []
    for axis in range(3):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        coordinates = np.flatnonzero(np.any(occupied, axis=reduce_axes))
        crop.append(slice(int(coordinates[0]), int(coordinates[-1]) + 1))
    crop = tuple(crop)
    del occupied
    truth_surface = truth_surface[crop]
    prediction_surface = prediction_surface[crop]

    distance = ndimage.distance_transform_edt(~prediction_surface)
    truth_to_prediction = distance[truth_surface]
    del distance
    distance = ndimage.distance_transform_edt(~truth_surface)
    prediction_to_truth = distance[prediction_surface]
    del distance
    symmetric = np.concatenate((truth_to_prediction, prediction_to_truth))
    del truth_to_prediction, prediction_to_truth
    symmetric /= UPSAMPLE
    median = float(np.median(symmetric, overwrite_input=True))
    p95 = float(np.percentile(symmetric, 95, overwrite_input=True))
    return {
        "truth_surface_voxels_l0": truth_count,
        "prediction_surface_voxels_l0": prediction_count,
        "symmetric_median_l1": median,
        "symmetric_p95_l1": p95,
    }


def _cube_slices() -> list[tuple[tuple[int, int, int], tuple[slice, ...]]]:
    if MASK_SHAPE_L0 != (2 * CUBE_SIZE_L0,) * 3:
        raise ValueError("locked mask shape must contain exactly eight equal cubes")
    result = []
    for index in product(range(2), repeat=3):
        result.append(
            (
                index,
                tuple(
                    slice(axis * CUBE_SIZE_L0, (axis + 1) * CUBE_SIZE_L0)
                    for axis in index
                ),
            )
        )
    return result


def score_mask(mask_l0: np.ndarray, truth_bits_l1: np.ndarray) -> dict:
    """Score one locked 0/255 level-0 mask against level-1 physical truth."""
    prediction = _prediction_mask(mask_l0)
    positive, supervised = _expanded_truth(truth_bits_l1)
    if positive.shape != MASK_SHAPE_L0:
        raise ValueError("expanded truth does not have the locked native shape")

    cubes = []
    aggregate = {key: 0 for key in _COMPONENT_KEYS}
    for index, cube_slice in _cube_slices():
        components = _component_metrics(
            positive[cube_slice], prediction[cube_slice]
        )
        surface = _surface_metrics(positive[cube_slice], prediction[cube_slice])
        for key in _COMPONENT_KEYS:
            aggregate[key] += components[key]
        cubes.append(
            {
                "cube_index_zyx": list(index),
                "components": components,
                "surface_distance": surface,
            }
        )

    return {
        "schema_version": SCHEMA,
        "metric_lock_content_sha256": METRIC_LOCK_CONTENT_SHA256,
        "classification": _classification_metrics(
            prediction, positive, supervised
        ),
        "component_aggregate": aggregate,
        "surface_distance_complete_box": _surface_metrics(positive, prediction),
        "cubes": cubes,
    }


def _nonnegative_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{description} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{description} must be a nonnegative integer")
    return result


def _finite_nonnegative(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{description} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{description} must be a finite nonnegative number")
    return result


def _acceptance_values(score: Mapping, description: str) -> dict:
    if not isinstance(score, Mapping):
        raise ValueError(f"{description} score must be a mapping")
    if score.get("schema_version") != SCHEMA:
        raise ValueError(f"{description} score has the wrong schema")
    if score.get("metric_lock_content_sha256") != METRIC_LOCK_CONTENT_SHA256:
        raise ValueError(f"{description} score has the wrong metric lock")
    components = score.get("component_aggregate")
    complete = score.get("surface_distance_complete_box")
    cubes = score.get("cubes")
    if not isinstance(components, Mapping) or not isinstance(complete, Mapping):
        raise ValueError(f"{description} score is missing aggregate metrics")
    if not isinstance(cubes, list) or len(cubes) != 8:
        raise ValueError(f"{description} score must contain exactly eight cubes")
    merger = _nonnegative_integer(
        components.get("merger_excess"), f"{description} merger_excess"
    )
    errors = _nonnegative_integer(
        components.get("total_component_errors"),
        f"{description} total_component_errors",
    )
    complete_median = _finite_nonnegative(
        complete.get("symmetric_median_l1"),
        f"{description} complete-box median",
    )
    cube_medians: dict[tuple[int, int, int], float] = {}
    for position, cube in enumerate(cubes):
        if not isinstance(cube, Mapping):
            raise ValueError(f"{description} cube {position} must be a mapping")
        raw_index = cube.get("cube_index_zyx")
        if (
            not isinstance(raw_index, list)
            or len(raw_index) != 3
            or any(isinstance(v, bool) or not isinstance(v, int) for v in raw_index)
        ):
            raise ValueError(f"{description} cube {position} has an invalid index")
        index = tuple(raw_index)
        if index not in set(product(range(2), repeat=3)) or index in cube_medians:
            raise ValueError(f"{description} cube {position} has an invalid index")
        surface = cube.get("surface_distance")
        if not isinstance(surface, Mapping):
            raise ValueError(f"{description} cube {position} is missing surface metrics")
        cube_medians[index] = _finite_nonnegative(
            surface.get("symmetric_median_l1"),
            f"{description} cube {position} median",
        )
    return {
        "merger_excess": merger,
        "total_component_errors": errors,
        "complete_median": complete_median,
        "cube_medians": cube_medians,
    }


def evaluate_acceptance(
    baseline_fixed: Mapping,
    candidate_fixed: Mapping,
    candidate_matched_mass: Mapping,
) -> dict:
    """Evaluate the two locked physical gates (meshing is audited separately)."""
    baseline = _acceptance_values(baseline_fixed, "baseline fixed")
    fixed = _acceptance_values(candidate_fixed, "candidate fixed")
    matched = _acceptance_values(candidate_matched_mass, "candidate matched-mass")
    order = list(product(range(2), repeat=3))

    matched_mergers = matched["merger_excess"] <= baseline["merger_excess"]
    matched_errors = (
        matched["total_component_errors"]
        <= baseline["total_component_errors"]
    )
    matched_complete = matched["complete_median"] < baseline["complete_median"]
    improved = sum(
        matched["cube_medians"][index] < baseline["cube_medians"][index]
        for index in order
    )
    matched_gate = {
        "merger_excess_nonincrease": bool(matched_mergers),
        "total_component_errors_nonincrease": bool(matched_errors),
        "complete_box_median_strictly_lower": bool(matched_complete),
        "strictly_improved_cube_count": int(improved),
        "at_least_five_of_eight_cubes_strictly_improved": improved >= 5,
    }
    matched_gate["pass"] = bool(
        all(matched_gate[key] for key in (
            "merger_excess_nonincrease",
            "total_component_errors_nonincrease",
            "complete_box_median_strictly_lower",
            "at_least_five_of_eight_cubes_strictly_improved",
        ))
    )

    fixed_mergers = fixed["merger_excess"] <= baseline["merger_excess"]
    fixed_errors = (
        fixed["total_component_errors"] <= baseline["total_component_errors"]
    )
    deltas = [
        fixed["cube_medians"][index] - baseline["cube_medians"][index]
        for index in order
    ]
    fixed_cube_gate = all(delta <= 0.5 for delta in deltas)
    fixed_gate = {
        "merger_excess_nonincrease": bool(fixed_mergers),
        "total_component_errors_nonincrease": bool(fixed_errors),
        "cube_median_deltas_l1": [float(value) for value in deltas],
        "maximum_cube_median_delta_l1": float(max(deltas)),
        "every_cube_median_delta_at_most_0_5_l1": bool(fixed_cube_gate),
    }
    fixed_gate["pass"] = bool(
        fixed_mergers and fixed_errors and fixed_cube_gate
    )
    return {
        "matched_mass": matched_gate,
        "fixed_threshold": fixed_gate,
        "pass": bool(matched_gate["pass"] and fixed_gate["pass"]),
    }
