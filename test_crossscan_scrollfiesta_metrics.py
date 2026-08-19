from __future__ import annotations

import copy
import unittest
from itertools import product
from unittest import mock

import numpy as np

import crossscan_scrollfiesta_metrics as M


def expanded(value: np.ndarray) -> np.ndarray:
    result = value
    for axis in range(3):
        result = np.repeat(result, 2, axis=axis)
    return result


def acceptance_score(
    merger_excess: int,
    total_errors: int,
    complete_median: float,
    cube_medians: list[float],
) -> dict:
    return {
        "schema_version": M.SCHEMA,
        "metric_lock_content_sha256": M.METRIC_LOCK_CONTENT_SHA256,
        "component_aggregate": {
            "merger_excess": merger_excess,
            "total_component_errors": total_errors,
        },
        "surface_distance_complete_box": {
            "symmetric_median_l1": complete_median,
        },
        "cubes": [
            {
                "cube_index_zyx": list(index),
                "surface_distance": {"symmetric_median_l1": median},
            }
            for index, median in zip(product(range(2), repeat=3), cube_medians)
        ],
    }


class PhysicalMetricTests(unittest.TestCase):
    def test_score_mask_uses_native_expansion_and_cube_local_components(self):
        truth = np.zeros((8, 8, 8), dtype=np.uint8)
        for z, y, x in product(range(2), repeat=3):
            truth[4 * z + 1, 4 * y + 1, 4 * x + 1] = 9
            truth[4 * z + 2, 4 * y + 2, 4 * x + 2] = 1
        prediction = np.where(expanded((truth & 8) != 0), 255, 0).astype(np.uint8)
        with mock.patch.multiple(
            M,
            TRUTH_SHAPE_L1=(8, 8, 8),
            MASK_SHAPE_L0=(16, 16, 16),
            CUBE_SIZE_L0=8,
            MINIMUM_COMPONENT_VOXELS_L0=8,
        ):
            score = M.score_mask(prediction, truth)
        self.assertEqual(score["classification"]["true_positive"], 64)
        self.assertEqual(score["classification"]["false_positive_supervised"], 0)
        self.assertEqual(score["classification"]["selected_ignored"], 0)
        self.assertEqual(score["classification"]["precision"], 1.0)
        self.assertEqual(score["classification"]["recall"], 1.0)
        self.assertEqual(score["classification"]["dice"], 1.0)
        self.assertEqual(score["component_aggregate"]["truth_components"], 8)
        self.assertEqual(score["component_aggregate"]["prediction_components"], 8)
        self.assertEqual(score["component_aggregate"]["total_component_errors"], 0)
        self.assertEqual(
            score["surface_distance_complete_box"]["symmetric_median_l1"], 0.0
        )
        self.assertEqual(len(score["cubes"]), 8)
        self.assertTrue(
            all(
                cube["surface_distance"]["symmetric_p95_l1"] == 0.0
                for cube in score["cubes"]
            )
        )

    def test_classification_counts_selected_ignored_separately(self):
        prediction = np.asarray([1, 1, 1, 0], dtype=bool)
        positive = np.asarray([1, 0, 0, 1], dtype=bool)
        supervised = np.asarray([1, 1, 0, 1], dtype=bool)
        result = M._classification_metrics(prediction, positive, supervised)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive_supervised"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["selected_ignored"], 1)
        self.assertEqual(result["dice"], 0.5)

    def test_bipartite_degrees_report_mergers_and_splits(self):
        truth = np.zeros((9, 9, 9), dtype=bool)
        prediction = np.zeros_like(truth)
        truth[4, 4, 2] = True
        truth[4, 4, 6] = True
        prediction[4, 4, 3:6] = True
        with mock.patch.object(M, "MINIMUM_COMPONENT_VOXELS_L0", 1):
            merged = M._component_metrics(truth, prediction)
        self.assertEqual(merged["merger_excess"], 1)
        self.assertEqual(merged["split_excess"], 0)
        self.assertEqual(merged["total_component_errors"], 1)

        truth.fill(False)
        prediction.fill(False)
        truth[4, 4, 3:6] = True
        prediction[4, 4, 2] = True
        prediction[4, 4, 6] = True
        with mock.patch.object(M, "MINIMUM_COMPONENT_VOXELS_L0", 1):
            split = M._component_metrics(truth, prediction)
        self.assertEqual(split["split_excess"], 1)
        self.assertEqual(split["merger_excess"], 0)
        self.assertEqual(split["total_component_errors"], 1)

    def test_surface_distance_is_symmetric_euclidean_in_level_one_units(self):
        truth = np.zeros((7, 7, 7), dtype=bool)
        prediction = np.zeros_like(truth)
        truth[3, 3, 2] = True
        prediction[3, 3, 4] = True
        result = M._surface_metrics(truth, prediction)
        self.assertEqual(result["symmetric_median_l1"], 1.0)
        self.assertEqual(result["symmetric_p95_l1"], 1.0)

    def test_score_rejects_nonbinary_or_wrong_dtype_prediction(self):
        truth = np.zeros(M.TRUTH_SHAPE_L1, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "uint8"):
            M.score_mask(np.zeros(M.MASK_SHAPE_L0, dtype=bool), truth)
        prediction = np.zeros(M.MASK_SHAPE_L0, dtype=np.uint8)
        prediction.flat[0] = 1
        with self.assertRaisesRegex(ValueError, "0 or 255"):
            M.score_mask(prediction, truth)
        prediction.flat[0] = 0
        with self.assertRaisesRegex(ValueError, "truth bits must be a uint8"):
            M.score_mask(prediction, truth.astype(np.uint16))


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.baseline = acceptance_score(2, 5, 2.0, [1.0] * 8)
        self.fixed = acceptance_score(2, 5, 2.1, [1.5] * 8)
        self.matched = acceptance_score(
            1, 4, 1.9, [0.9, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0]
        )

    def test_locked_boundaries_pass(self):
        result = M.evaluate_acceptance(self.baseline, self.fixed, self.matched)
        self.assertTrue(result["pass"])
        self.assertTrue(result["matched_mass"]["pass"])
        self.assertEqual(result["matched_mass"]["strictly_improved_cube_count"], 5)
        self.assertTrue(result["fixed_threshold"]["pass"])
        self.assertEqual(
            result["fixed_threshold"]["maximum_cube_median_delta_l1"], 0.5
        )

    def test_strict_matched_median_and_fixed_cube_limit_are_enforced(self):
        matched = copy.deepcopy(self.matched)
        matched["surface_distance_complete_box"]["symmetric_median_l1"] = 2.0
        self.assertFalse(
            M.evaluate_acceptance(self.baseline, self.fixed, matched)[
                "matched_mass"
            ]["pass"]
        )
        fixed = copy.deepcopy(self.fixed)
        fixed["cubes"][3]["surface_distance"]["symmetric_median_l1"] = 1.50001
        self.assertFalse(
            M.evaluate_acceptance(self.baseline, fixed, self.matched)[
                "fixed_threshold"
            ]["pass"]
        )

    def test_nonfinite_or_incomplete_scores_fail_closed(self):
        malformed = copy.deepcopy(self.matched)
        malformed["cubes"][0]["surface_distance"]["symmetric_median_l1"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            M.evaluate_acceptance(self.baseline, self.fixed, malformed)
        malformed = copy.deepcopy(self.matched)
        malformed["cubes"].pop()
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            M.evaluate_acceptance(self.baseline, self.fixed, malformed)


if __name__ == "__main__":
    unittest.main()
