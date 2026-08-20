from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from label_placement_oriented import (
    STEP,
    block_bootstrap_ci,
    label_run_centres,
    load_axis,
    orient_offsets,
    portable_path,
    ridge_from_profile,
    sample_profiles,
    sample_bootstrap_ci,
)
from ridge_residual import ridge_offset


class OrientationCorrectionTests(unittest.TestCase):
    def test_public_metadata_path_is_portable(self) -> None:
        self.assertEqual(portable_path(Path(__file__)), Path(__file__).name)

    def test_sign_flip_keeps_physical_result(self) -> None:
        normals = np.array([[0.0, 1.0, 0.0], [0.0, -0.6, 0.8]], dtype=np.float64)
        reference = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        raw = np.array([1.25, -0.75])
        corrected_a, oriented_a, alignment_a, _ = orient_offsets(normals, raw, reference)
        corrected_b, oriented_b, alignment_b, _ = orient_offsets(-normals, -raw, reference)
        np.testing.assert_allclose(corrected_a, corrected_b)
        np.testing.assert_allclose(oriented_a, oriented_b)
        np.testing.assert_allclose(alignment_a, alignment_b)
        np.testing.assert_allclose(normals * raw[:, None], oriented_a * corrected_a[:, None])

    def test_axis_is_sorted_before_interpolation(self) -> None:
        rows = [[float(i), 2000.0 + i, 4000.0 - i] for i in range(241)]
        rows[10], rows[11] = rows[11], rows[10]
        axis_bytes = "\n".join(", ".join(str(v) for v in row) for row in rows).encode()
        meta_bytes = (
            b'{"height":7888,"slices":14376,"uuid":"20230205180739",'
            b'"voxelsize":7.91,"width":8096}'
        )
        axis, info = load_axis(axis_bytes, meta_bytes)
        self.assertTrue(np.all(np.diff(axis[:, 0]) > 0))
        self.assertEqual(info["raw_negative_z_steps"], 1)

    def test_label_run_centre_and_truncation(self) -> None:
        ts = np.arange(-1.0, 1.0 + 1e-9, STEP, dtype=np.float32)
        profile = np.zeros((3, len(ts)), dtype=np.float32)
        profile[0, 3:6] = 1  # centre -0.25, 0, +0.25 -> 0
        profile[1, :5] = 1   # exits left edge
        profile[2, 5:] = 1   # centre is not labelled
        centre, truncated = label_run_centres(profile, ts)
        self.assertEqual(centre[0], 0.0)
        self.assertTrue(np.isnan(centre[1]))
        self.assertTrue(np.isnan(centre[2]))
        np.testing.assert_array_equal(truncated, [False, True, True])

    def test_parabolic_ridge_and_edges(self) -> None:
        ts = np.arange(-1.0, 1.0 + 1e-9, STEP, dtype=np.float32)
        profile = np.stack([-(ts - 0.125) ** 2, -(ts + 0.4) ** 2, ts], axis=0)
        ridge, edge = ridge_from_profile(profile, ts)
        self.assertAlmostEqual(ridge[0], 0.125, places=6)
        self.assertAlmostEqual(ridge[1], -0.4, places=6)
        self.assertTrue(np.isnan(ridge[2]))
        np.testing.assert_array_equal(edge, [False, False, True])

    def test_profile_helper_matches_original_ridge_offset(self) -> None:
        rng = np.random.default_rng(4)
        ct = rng.normal(100, 20, size=(24, 24, 24)).astype(np.float32)
        lab = np.zeros_like(ct, dtype=np.uint8)
        points = rng.uniform(7, 17, size=(12, 3)).astype(np.float32)
        normals = rng.normal(size=(12, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        expected, _ = ridge_offset(ct, points, normals, half=4.0, step=STEP)
        ts, profiles, _ = sample_profiles(ct, lab, points, normals)
        take = np.abs(ts) <= 4.0 + 1e-7
        observed, _ = ridge_from_profile(profiles[:, take], ts[take])
        np.testing.assert_array_equal(np.isnan(observed), np.isnan(expected))
        np.testing.assert_allclose(observed, expected, atol=1e-7, rtol=1e-7)

    def test_bootstraps_are_deterministic(self) -> None:
        values = np.array([-1.0, 0.0, 2.0, 4.0])
        self.assertEqual(sample_bootstrap_ci(values, "unit"), sample_bootstrap_ci(values, "unit"))
        blocks = ["0,0,0", "0,0,0", "1,0,0", "2,0,0"]
        self.assertEqual(
            block_bootstrap_ci(values, blocks, "unit"),
            block_bootstrap_ci(values, blocks, "unit"),
        )


if __name__ == "__main__":
    unittest.main()
