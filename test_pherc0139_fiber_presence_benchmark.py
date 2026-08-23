"""Synthetic tests for the locked PHerc0139 fiber-presence benchmark."""

from __future__ import annotations

import hashlib
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError, URLError

import numpy as np

import pherc0139_fiber_presence_benchmark as benchmark


def planar_xyz(rows: int, columns: int) -> np.ndarray:
    row, column = np.mgrid[:rows, :columns]
    return np.stack((2.0 * column, 3.0 * row, np.full_like(row, 5.0)), axis=-1).astype(np.float32)


class SurfaceSelectionTests(unittest.TestCase):
    def test_target_transformed_reference_url_uses_short_date_prefix(self) -> None:
        lock = {
            "reference_url_template": (
                "https://example.test/{segment}/mesh/"
                "{segment_short}-on-20260102150214-2.399um.tifxyz/{file}"
            )
        }
        url = benchmark.reference_url(
            lock, "20250108000000-w025_2025010863", "x.tif"
        )
        self.assertEqual(
            url,
            "https://example.test/20250108000000-w025_2025010863/mesh/"
            "20250108000000-on-20260102150214-2.399um.tifxyz/x.tif",
        )

    def test_one_first_valid_cycle_point_per_uv_bin(self) -> None:
        xyz = planar_xyz(12, 18)
        seed = "synthetic-seed"
        segment = "segment-a"
        selected = benchmark.select_surface_samples(xyz, segment, seed, 2, 3)

        self.assertEqual(len(selected), 6)
        self.assertEqual([(sample.bin_y, sample.bin_x) for sample in selected], [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ])
        for sample in selected:
            row_start = max(1, sample.bin_y * xyz.shape[0] // 2)
            row_stop = min(xyz.shape[0] - 1, (sample.bin_y + 1) * xyz.shape[0] // 2)
            col_start = max(1, sample.bin_x * xyz.shape[1] // 3)
            col_stop = min(xyz.shape[1] - 1, (sample.bin_x + 1) * xyz.shape[1] // 3)
            width = col_stop - col_start
            size = (row_stop - row_start) * width
            start, _ = benchmark.bin_cycle(seed, segment, sample.bin_y, sample.bin_x, size)
            expected_row = row_start + start // width
            expected_column = col_start + start % width
            self.assertEqual((sample.row, sample.column), (expected_row, expected_column))

    def test_cycle_stride_is_coprime_and_visits_every_bin_index(self) -> None:
        for size in range(2, 50):
            start, stride = benchmark.bin_cycle("seed", "segment", 3, 7, size)
            self.assertEqual(__import__("math").gcd(stride, size), 1)
            visited = {(start + index * stride) % size for index in range(size)}
            self.assertEqual(visited, set(range(size)))

    def test_invalid_neighbor_disqualifies_candidate(self) -> None:
        xyz = planar_xyz(7, 7)
        xyz[2, 3, :] = -1.0
        selected = benchmark.select_surface_samples(xyz, "segment", "seed", 1, 1)
        disqualified = {(2, 3), (1, 3), (3, 3), (2, 2), (2, 4)}
        self.assertNotIn((selected[0].row, selected[0].column), disqualified)


class NormalTests(unittest.TestCase):
    def test_central_difference_normal_for_plane(self) -> None:
        xyz = planar_xyz(7, 7)
        normal = benchmark.central_difference_normal(xyz, 3, 3)
        self.assertIsNotNone(normal)
        np.testing.assert_allclose(normal, np.array([0.0, 0.0, -1.0]), atol=1e-12)

    def test_normal_requires_four_valid_neighbors(self) -> None:
        xyz = planar_xyz(7, 7)
        xyz[3, 4, 0] = np.nan
        self.assertIsNone(benchmark.central_difference_normal(xyz, 3, 3))

    def test_query_transform_is_one_eighth_scale_then_prediction_voxel_offset(self) -> None:
        sample = benchmark.SurfaceSample(
            row=1,
            column=2,
            bin_y=0,
            bin_x=0,
            selection_sha256="0" * 64,
            xyz=(8.0, 16.0, 24.0),
            normal_xyz=(0.0, 0.0, 1.0),
        )
        queries = benchmark.build_normal_queries([sample], 0.125, [-12, 0, 12])
        np.testing.assert_allclose(queries[0], np.array([
            [-9.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
            [15.0, 2.0, 1.0],
        ]))


class TrilinearInterpolationTests(unittest.TestCase):
    def test_linear_field_is_interpolated_exactly_and_bounds_are_strict(self) -> None:
        z, y, x = np.mgrid[:5, :6, :7]
        field = (z + 2.0 * y + 3.0 * x).astype(np.float64)
        points = np.array([
            [1.25, 2.5, 3.125],
            [0.0, 0.0, 0.0],
            [4.0, 2.0, 2.0],
            [-0.01, 2.0, 2.0],
        ])
        values = benchmark.trilinear_sample(field, points)
        self.assertAlmostEqual(values[0], 1.25 + 2.0 * 2.5 + 3.0 * 3.125, places=12)
        self.assertEqual(values[1], 0.0)
        self.assertTrue(np.isnan(values[2]))
        self.assertTrue(np.isnan(values[3]))

    def test_decode_divisor_is_applied_after_interpolation(self) -> None:
        field = np.full((3, 3, 3), 255, dtype=np.uint8)
        value = benchmark.trilinear_sample(field, np.array([[1.0, 1.0, 1.0]]), 255.0)[0]
        self.assertEqual(value, 1.0)


class PrimaryGateTests(unittest.TestCase):
    GATE = {
        "minimum_analyzable_segments": 30,
        "minimum_positive_segments": 24,
        "minimum_median_segment_delta": 0.02,
    }

    def test_positive_finite_set_gate_has_no_p_value(self) -> None:
        deltas = [0.04] * 24 + [-0.01] * 6
        result = benchmark.evaluate_primary_gate(deltas, self.GATE)
        self.assertEqual(result["decision"], "LOCALIZATION_SUPPORTED")
        self.assertEqual(result["positive_segments"], 24)
        self.assertNotIn("one_sided_exact_sign_p", result)
        self.assertEqual(result["inference"], "descriptive finite-set gate; no p-value")

    def test_scientific_gate_failure_is_null(self) -> None:
        deltas = [0.03] * 23 + [-0.01] * 6 + [0.0]
        result = benchmark.evaluate_primary_gate(deltas, self.GATE)
        self.assertEqual(result["decision"], "LOCALIZATION_NOT_SUPPORTED")
        self.assertEqual(result["zero_segments"], 1)
        self.assertFalse(result["checks"]["minimum_positive_segments"])


class OrientationTests(unittest.TestCase):
    def test_nx_ny_decoder_and_tangent_angles(self) -> None:
        vectors = benchmark.decode_fiber_directions(
            np.array([128.0, 255.0]), np.array([128.0, 128.0])
        )
        np.testing.assert_allclose(vectors[0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(vectors[1], [1.0, 0.0, 0.0], atol=1e-12)
        angles = benchmark.tangent_plane_angles_degrees(
            vectors, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        )
        np.testing.assert_allclose(angles, [90.0, 0.0], atol=1e-12)

    def test_decoder_clamps_dz_then_renormalizes(self) -> None:
        vector = benchmark.decode_fiber_directions(np.array([0.0]), np.array([0.0]))[0]
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=12)
        self.assertEqual(vector[2], 0.0)

    def test_zero_center_presence_is_excluded_from_orientation_only(self) -> None:
        class ConstantSampler:
            def __init__(self, value):
                self.value = value

            def sample(self, points):
                return np.full(len(points), self.value, dtype=np.float64)

        sample = benchmark.SurfaceSample(
            row=2, column=2, bin_y=0, bin_x=0,
            selection_sha256="0" * 64,
            xyz=(200.0, 200.0, 200.0), normal_xyz=(0.0, 0.0, 1.0),
        )
        lock = {
            "sampling": {
                "normal_offsets_prediction_voxels": [-12, -8, -4, 0, 4, 8, 12],
                "reference_coordinate_scale_to_prediction": 0.125,
                "minimum_samples_per_segment": 128,
            },
            "fiber_artifact": {"channels": {
                "nx": {"decode_offset": 128, "decode_divisor": 127},
            }},
            "orientation_analysis": {
                "center_presence_subset_threshold": 0.5,
                "minimum_positive_weight_points_per_segment": 32,
            },
        }
        summary, rows = benchmark.analyze_segment(
            "segment", [sample], [5, 5], lock,
            {"presence": ConstantSampler(0.0), "nx": ConstantSampler(128.0),
             "ny": ConstantSampler(128.0)},
        )
        self.assertTrue(rows[0]["complete_profile"])
        self.assertFalse(rows[0]["orientation_valid"])
        self.assertEqual(summary["orientation"]["positive_presence_points"], 0)


class IntegrityTests(unittest.TestCase):
    def test_required_tag_must_be_annotated_and_point_to_head(self) -> None:
        expected = "a" * 40
        responses = {
            ("cat-file", "-t", "refs/tags/amendment"): "tag",
            ("rev-parse", "refs/tags/amendment"): "b" * 40,
            ("rev-parse", "refs/tags/amendment^{}"): expected,
        }

        def fake_git(*arguments):
            return responses[arguments]

        self.assertEqual(
            benchmark._annotated_tag_receipt(fake_git, "amendment", expected),
            {"name": "amendment", "object": "b" * 40, "commit": expected},
        )
        responses[("cat-file", "-t", "refs/tags/amendment")] = "commit"
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._annotated_tag_receipt(fake_git, "amendment", expected)

    def test_outcome_run_cannot_redirect_the_one_run_seal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for action in ("--run", "--resume-transport"):
                with self.subTest(action=action), self.assertRaises(SystemExit):
                    benchmark.parse_args([action, "--out-dir", directory])

    def test_only_definitive_http_404_is_a_fill_chunk(self) -> None:
        missing = HTTPError("https://example.test/chunk", 404, "Not Found", {}, None)
        server_error = HTTPError("https://example.test/chunk", 500, "Error", {}, None)
        self.assertTrue(benchmark.is_definitive_missing_chunk(missing))
        self.assertFalse(benchmark.is_definitive_missing_chunk(server_error))
        self.assertFalse(benchmark.is_definitive_missing_chunk(URLError("timeout")))

    def test_zarray_requires_numeric_zero_fill(self) -> None:
        channel = {"shape_zyx": [8, 8, 8], "chunks_zyx": [4, 4, 4], "dtype": "uint8"}
        descriptor = {
            "shape": [8, 8, 8], "chunks": [4, 4, 4], "dtype": "|u1",
            "fill_value": 0, "dimension_separator": ".",
        }
        self.assertEqual(
            benchmark._validate_zarray_descriptor(descriptor, "presence", channel), "."
        )
        with self.assertRaises(benchmark.BenchmarkError):
            benchmark._validate_zarray_descriptor(
                {**descriptor, "fill_value": 1}, "presence", channel
            )

    def test_positive_presence_missing_direction_sibling_aborts_but_zero_does_not(self) -> None:
        class ConstantSampler:
            def __init__(self, value):
                self.value = value

            def sample(self, points):
                return np.full(len(points), self.value, dtype=np.float64)

        sample = benchmark.SurfaceSample(
            row=2, column=2, bin_y=0, bin_x=0,
            selection_sha256="0" * 64,
            xyz=(40.0, 40.0, 40.0), normal_xyz=(0.0, 0.0, 1.0),
        )
        lock = {
            "reference_segments": ["segment"],
            "sampling": {
                "normal_offsets_prediction_voxels": [-12, -8, -4, 0, 4, 8, 12],
                "reference_coordinate_scale_to_prediction": 0.125,
            },
            "fiber_artifact": {"channels": {
                "nx": {"shape_zyx": [16, 16, 16], "chunks_zyx": [4, 4, 4]},
            }},
        }
        for missing_channel in ("nx", "ny"):
            missing = {"presence": set(), "nx": set(), "ny": set()}
            missing[missing_channel] = {(1, 1, 1)}
            with self.subTest(missing_channel=missing_channel):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "sibling chunk"):
                    benchmark.validate_orientation_sibling_coherence(
                        lock, {"segment": [sample]}, ConstantSampler(0.25), missing
                    )
        missing = {"presence": set(), "nx": {(1, 1, 1)}, "ny": set()}
        report = benchmark.validate_orientation_sibling_coherence(
            lock, {"segment": [sample]}, ConstantSampler(0.0), missing
        )
        self.assertEqual(report["zero_center_presence_points_excluded_from_orientation"], 1)
        self.assertEqual(report["positive_presence_missing_direction_support"], 0)

    def test_gzip_transport_hash_is_distinct_from_decoded_content(self) -> None:
        content = b'{"hello":"world"}\n'
        transport = gzip.compress(content, mtime=0)
        decoded, encoding = benchmark.decode_transport_payload(transport, "gzip")
        self.assertEqual(decoded, content)
        self.assertEqual(encoding, "gzip")
        self.assertNotEqual(benchmark.sha256_bytes(transport), benchmark.sha256_bytes(content))

    def test_outcome_marker_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = benchmark.create_outcome_marker(root, {"test": True})
            self.assertTrue(marker.is_file())
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.create_outcome_marker(root, {"test": True})

    def test_transport_resume_marker_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = benchmark.create_transport_resume_marker(root, {"test": True})
            self.assertTrue(marker.is_file())
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.create_transport_resume_marker(root, {"test": True})

    def test_second_transport_failure_is_persisted_after_resume_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark.create_outcome_marker(root, {"first": True})
            benchmark.create_transport_resume_marker(root, {"resume": True})
            path = benchmark.record_transport_resume_failure(root, OSError("network down"))
            self.assertIsNotNone(path)
            payload = __import__("json").loads(path.read_bytes())
            self.assertEqual(payload["event"], "TRANSPORT_RESUME_TECHNICAL_FAILURE")
            self.assertEqual(payload["error"]["type"], "OSError")
            self.assertTrue(payload["scientific_values_may_have_been_sampled"])
            self.assertNotIn("scientific_outcome", payload)
            self.assertEqual(
                benchmark.record_transport_resume_failure(root, OSError("again")), path
            )

    def test_mirror_inventory_digest_commits_to_all_object_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for channel in benchmark.CHANNEL_NAMES:
                channel_root = root / channel
                channel_root.mkdir(parents=True)
                (channel_root / ".zarray").write_bytes(b"array-" + channel.encode())
                (channel_root / ".zattrs").write_bytes(b"{}")
                chunk = channel_root / "1" / "2" / "3"
                chunk.parent.mkdir(parents=True)
                chunk.write_bytes(b"chunk-" + channel.encode())
            first, keys = benchmark.mirror_inventory_receipt(root)
            self.assertEqual(first["object_count"], 9)
            self.assertEqual(first["metadata_object_count"], 6)
            self.assertEqual(keys, {name: {"1/2/3"} for name in benchmark.CHANNEL_NAMES})
            (root / "presence" / "1" / "2" / "3").write_bytes(b"changed")
            second, _ = benchmark.mirror_inventory_receipt(root)
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_analysis_failure_raises_instead_of_becoming_scientific_null(self) -> None:
        class FailingSampler:
            def sample(self, _points):
                raise OSError("synthetic chunk failure")

        sample = benchmark.SurfaceSample(
            row=2, column=2, bin_y=0, bin_x=0,
            selection_sha256=hashlib.sha256(b"seed|segment|2|2").hexdigest(),
            xyz=(80.0, 80.0, 80.0), normal_xyz=(0.0, 0.0, 1.0),
        )
        lock = {
            "sampling": {
                "normal_offsets_prediction_voxels": [-12, -8, -4, 0, 4, 8, 12],
                "reference_coordinate_scale_to_prediction": 0.125,
                "minimum_samples_per_segment": 128,
            },
            "fiber_artifact": {"channels": {"nx": {"decode_offset": 128, "decode_divisor": 127}}},
            "orientation_analysis": {"center_presence_subset_threshold": 0.5},
        }
        samplers = {name: FailingSampler() for name in ("presence", "nx", "ny")}
        with self.assertRaisesRegex(OSError, "synthetic chunk failure"):
            benchmark.analyze_segment("segment", [sample], [5, 5], lock, samplers)


class PredictionMirrorIntegrationTests(unittest.TestCase):
    def test_chunk_plan_receipts_partial_zarr_and_sampling(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "outcome"
            channels = {}
            chunk_plan = {}
            for index, name in enumerate(benchmark.CHANNEL_NAMES, start=1):
                channel_source = source / name
                array = zarr.open(
                    str(channel_source), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2),
                    dtype="uint8", compressor=None, fill_value=0,
                )
                array[:2, :2, :2] = 10 * index
                array[2:, 2:, 2:] = 20 * index
                array.attrs["fixture"] = "integration"
                channels[name] = {
                    "zarr_url": f"https://example.test/{name}",
                    "zarray_sha256": benchmark.sha256_file(channel_source / ".zarray"),
                    "zattrs_sha256": benchmark.sha256_file(channel_source / ".zattrs"),
                    "shape_zyx": [4, 4, 4],
                    "chunks_zyx": [2, 2, 2],
                    "dtype": "uint8",
                }
                chunk_plan[name] = {
                    "query_count": 2,
                    "out_of_bounds_query_count": 0,
                    "chunk_coordinates": [[0, 0, 0], [1, 1, 1]],
                }
            lock = {"fiber_artifact": {"channels": channels}}
            requested_urls = []

            def fake_download(url, path, cache_root, expected_sha256=None, timeout_seconds=120.0):
                del timeout_seconds
                requested_urls.append(url)
                parts = url.rstrip("/").split("/")
                source_path = source / parts[-2] / parts[-1]
                if parts[-2] == "presence" and parts[-1] == "1.1.1":
                    raise AssertionError("pinned first-attempt 404 was requested again")
                if parts[-2] == "nx" and parts[-1] == "1.1.1":
                    raise HTTPError(url, 404, "Not Found", {}, None)
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, path)
                return benchmark._receipt(path, url, cache_root, expected_sha256)

            with mock.patch.object(benchmark, "download_cached", side_effect=fake_download):
                arrays, receipts, missing = benchmark.mirror_prediction_channels(
                    lock,
                    chunk_plan,
                    output,
                    timeout_seconds=1.0,
                    workers=2,
                    pinned_missing_chunks={
                        "presence": {(1, 1, 1)}, "nx": set(), "ny": set()
                    },
                )

            self.assertEqual(missing["presence"], {(1, 1, 1)})
            self.assertEqual(missing["nx"], {(1, 1, 1)})
            self.assertEqual(receipts["missing_fill_chunk_counts"]["presence"], 1)
            statuses = {item["status"] for item in receipts["chunks"]["presence"]}
            self.assertEqual(
                statuses, {"stored_object", "pinned_first_attempt_http_404_fill"}
            )
            self.assertNotIn("https://example.test/presence/1.1.1", requested_urls)
            sampler = benchmark.ChunkedArraySampler(arrays["presence"], 1.0, 4)
            np.testing.assert_allclose(
                sampler.sample(np.asarray([[0.5, 0.5, 0.5], [2.0, 2.0, 2.0]])),
                [10.0, 0.0],
            )


class VisualPanelTests(unittest.TestCase):
    def test_panel_contains_four_fixed_views(self) -> None:
        summary = {
            "raster_shape_rc": [5, 5],
            "median_profile": [0.0, 0.1, 0.2, 0.8, 0.2, 0.1, 0.0],
        }
        point = {
            "row": 2,
            "column": 2,
            "complete_profile": True,
            "orientation_valid": True,
            "presence_p0": 0.8,
            "delta": 0.4,
            "tangent_angle_degrees": 10.0,
        }
        payload = benchmark.render_panel(
            "synthetic", summary, [point], [-12, -8, -4, 0, 4, 8, 12]
        )
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
        width = int.from_bytes(payload[16:20], "big")
        height = int.from_bytes(payload[20:24], "big")
        self.assertEqual((width, height), (1310, 350))


if __name__ == "__main__":
    unittest.main()
