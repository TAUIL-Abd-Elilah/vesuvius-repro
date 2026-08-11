#!/usr/bin/env python3
"""Regression tests for the preregistered physical normalization A/B."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import physical_normalization_ab as P
import run_physical_normalization_ab as R


class PhysicalNormalizationABTests(unittest.TestCase):
    def make_truth(self) -> tuple[P.TruthPlane, int, int, int, int]:
        raw = np.zeros((224, 192), dtype=np.uint8)
        score_y0, score_x0 = 128, 64
        extent_y0 = score_y0 - P.NULL_SHIFT_L1 - P.METRIC_HALO_L1
        extent_x0 = score_x0 - P.METRIC_HALO_L1
        raw[score_y0 : score_y0 + 64, score_x0 : score_x0 + 64] |= 1
        # One truth arc, 64 pixels long, wholly inside one 64x64 global tile.
        raw[score_y0 + 12, score_x0 : score_x0 + 64] |= 2 | 4
        return (
            P.prepare_truth_plane(raw, with_side=False),
            score_y0,
            score_x0,
            extent_y0,
            extent_x0,
        )

    def blank_prediction(self) -> np.ndarray:
        return np.zeros(
            (
                P.SCORE_SIZE_L1 + P.NULL_SHIFT_L1 + 2 * P.METRIC_HALO_L1,
                P.SCORE_SIZE_L1 + 2 * P.METRIC_HALO_L1,
            ),
            dtype=bool,
        )

    def test_geometry_has_frozen_extent_and_blend_interior(self) -> None:
        geometry = P.block_geometry((128, 256, 384), (1000, 2000, 3000))
        ext = geometry["prediction_extent_global_l1"]
        self.assertEqual([ext[1] - ext[0], ext[3] - ext[2], ext[5] - ext[4]], [64, 144, 80])
        inference = geometry["inference_bbox_l0"]
        self.assertEqual(
            [
                inference[1] - inference[0],
                inference[3] - inference[2],
                inference[5] - inference[4],
            ],
            [256, 416, 288],
        )
        l0_extent = geometry["prediction_extent_global_l0"]
        self.assertEqual(inference[0], l0_extent[0] - 64)
        self.assertEqual(inference[1], l0_extent[1] + 64)

    def test_empty_prediction_remains_in_truth_denominator(self) -> None:
        truth, sy, sx, ey, ex = self.make_truth()
        result = P.score_plane(truth, self.blank_prediction(), sy, sx, ey, ex, with_side=False)
        self.assertEqual(result["n_centerline"], 64)
        self.assertEqual(result["hit1"], 0)
        self.assertEqual(result["hit2"], 0)
        self.assertEqual(result["hit3"], 0)
        self.assertEqual(result["n_arcs"], 1)
        self.assertEqual(result["arc_hit"], 0)
        self.assertEqual(result["arc_gone"], 1)
        metric = P.metrics(result)
        self.assertEqual(metric["recall_37um"], 0.0)
        self.assertEqual(metric["arc_fully_missed"], 1.0)

    def test_sparse_failure_lowers_aggregate_instead_of_disappearing(self) -> None:
        truth, sy, sx, ey, ex = self.make_truth()
        empty = P.score_plane(truth, self.blank_prediction(), sy, sx, ey, ex, with_side=False)
        hit_pred = self.blank_prediction()
        local_y = sy + 12 - ey
        local_x = sx - ex
        hit_pred[local_y, local_x : local_x + 64] = True
        hit = P.score_plane(truth, hit_pred, sy, sx, ey, ex, with_side=False)
        aggregate = P.blank_counts()
        P.add_counts(aggregate, hit)
        P.add_counts(aggregate, empty)
        self.assertEqual(aggregate["n_centerline"], 128)
        self.assertEqual(aggregate["hit2"], 64)
        self.assertEqual(P.metrics(aggregate)["recall_37um"], 0.5)

    def test_shifted_null_uses_upstream_prediction(self) -> None:
        truth, sy, sx, ey, ex = self.make_truth()
        pred = self.blank_prediction()
        target_local_y = sy + 12 - ey
        source_local_y = target_local_y - P.NULL_SHIFT_L1
        local_x = sx - ex
        pred[source_local_y, local_x : local_x + 64] = True
        result = P.score_plane(truth, pred, sy, sx, ey, ex, with_side=False)
        self.assertEqual(result["hit2"], 0)
        self.assertEqual(result["null_hit2"], 64)
        self.assertEqual(result["null_arc_hit"], 1)

    def test_false_positive_distance_guardrail(self) -> None:
        truth, sy, sx, ey, ex = self.make_truth()
        pred = self.blank_prediction()
        # Valid but far from the single material line.
        pred[sy + 50 - ey, sx + 10 - ex] = True
        result = P.score_plane(truth, pred, sy, sx, ey, ex, with_side=False)
        self.assertEqual(result["pred_valid"], 1)
        self.assertEqual(result["pred_far2"], 1)
        self.assertEqual(result["pred_far4"], 1)

    def test_matched_mass_exact_when_values_are_distinct(self) -> None:
        threshold, count = P.choose_matched_mass_threshold(
            np.array([0.1, 0.2, 0.3, 0.4]), 2
        )
        self.assertEqual(count, 2)
        self.assertEqual(int((np.array([0.1, 0.2, 0.3, 0.4]) > threshold).sum()), 2)

    def test_matched_mass_tie_is_conservative(self) -> None:
        values = np.array([0.1, 0.5, 0.5, 0.9])
        threshold, count = P.choose_matched_mass_threshold(values, 2)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(count, 1)

    def test_matched_mass_rejects_invalid_probabilities(self) -> None:
        with self.assertRaises(ValueError):
            P.choose_matched_mass_threshold(np.array([0.2, np.nan]), 1)
        with self.assertRaises(ValueError):
            P.choose_matched_mass_threshold(np.array([-0.1, 0.2]), 1)

    def test_stratified_bootstrap_equal_weights_groups(self) -> None:
        result = P._stratified_bootstrap({"easy": [0.0, 0.0], "hard": [2.0]}, 7)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 1.0)
        self.assertEqual(result["ci95_low"], 1.0)
        self.assertEqual(result["ci95_high"], 1.0)

    def test_l0_max_pool_matches_binary_any(self) -> None:
        array = np.zeros((4, 6, 8), dtype=np.float32)
        array[1, 3, 5] = 0.9
        pooled = R.max_pool_l0_to_l1(array)
        self.assertEqual(pooled.shape, (2, 3, 4))
        self.assertAlmostEqual(float(pooled[0, 1, 2]), 0.9, places=6)
        self.assertEqual(int((pooled > 0.2).sum()), 1)

    def test_l0_max_pool_rejects_odd_shape(self) -> None:
        with self.assertRaises(ValueError):
            R.max_pool_l0_to_l1(np.zeros((3, 4, 4)))

    def test_candidate_selection_is_hash_ordered_per_stratum(self) -> None:
        candidates = []
        for stratum in range(P.Z_STRATA):
            for index in range(P.BLOCKS_PER_Z_STRATUM + 3):
                candidates.append(
                    {
                        "scroll": "S",
                        "rank_sha256": f"{index:064x}",
                        "z_stratum": stratum,
                        "local_origin_l1": [stratum * 128, index * 128, 128],
                        "geometry": {},
                        "label_stats": {},
                    }
                )
        selected = P.select_candidates(candidates)
        self.assertEqual(len(selected), P.Z_STRATA * P.BLOCKS_PER_Z_STRATUM)
        for stratum in range(P.Z_STRATA):
            ranks = [c["rank_sha256"] for c in selected if c["z_stratum"] == stratum]
            self.assertEqual(ranks, [f"{i:064x}" for i in range(P.BLOCKS_PER_Z_STRATUM)])

    def test_block_npz_is_fail_closed_on_metadata(self) -> None:
        block = {
            "block_id": "block-1",
            "geometry": {"prediction_extent_global_l1": [1, 2, 3, 4, 5, 6]},
        }
        shape = (64, 144, 80)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "block.npz"
            metadata = {
                "schema_version": 1,
                "manifest_content_sha256": "abc",
                "block_id": "wrong",
                "prediction_extent_global_l1": [1, 2, 3, 4, 5, 6],
            }
            np.savez_compressed(
                path,
                baseline_l1=np.zeros(shape, dtype=np.uint8),
                corrected_pmax_l1=np.zeros(shape, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata)),
            )
            with self.assertRaises(SystemExit):
                P._load_block_arrays(path, block, "abc")

    def test_normal_field_recovers_stripe_orientation(self) -> None:
        yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
        for angle in (0.0, 30.0, 77.0, 120.0):
            a = np.deg2rad(angle)
            ny, nx, coherence = P.normal_field(
                np.sin((yy * np.cos(a) + xx * np.sin(a)) * 2.0)
            )
            dot = abs(ny[100, 100] * np.cos(a) + nx[100, 100] * np.sin(a))
            self.assertGreater(dot, 0.97)
            self.assertGreater(coherence[100, 100], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
