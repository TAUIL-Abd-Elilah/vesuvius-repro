from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import crossscan_highres_review as H


class PyramidCoordinateTests(unittest.TestCase):
    def test_level_coordinate_roundtrip_uses_voxel_centres(self) -> None:
        values = np.asarray([-0.25, 0.0, 3.5, 99.125])
        for level in range(6):
            np.testing.assert_allclose(
                H.l0_coordinate_to_level(H.level_coordinate_to_l0(values, level), level),
                values,
            )
        self.assertEqual(float(H.level_coordinate_to_l0(0, 1)), 0.5)
        self.assertEqual(float(H.level_coordinate_to_l0(0, 4)), 7.5)

    def test_ph1203_coordinates_reproduce_label_caster_algebra(self) -> None:
        matrix = np.asarray([
            [1.02, 0.01, -0.02],
            [-0.01, 0.99, 0.03],
            [0.02, -0.01, 1.01],
        ])
        offset = np.asarray([4.0, -3.0, 8.0])
        high_l1 = np.asarray([402.0, 169.0, 287.0])
        high_l3 = high_l1 / 4.0
        low_l1 = matrix @ high_l3 + 2.0 * offset + 0.5
        coordinates = H.registered_source_coordinates(
            "PHerc1203", matrix, offset, low_l1, 0,
            source_level=1, size_l1=1, pixels_per_l1=1,
        )
        np.testing.assert_allclose(coordinates[:, 0, 0], high_l1)

    def test_registered_plane_identity_alignment(self) -> None:
        high_l2_origin = np.asarray([11.25, 20.0, 30.0])
        # PHerc0139's caster reads L4; M=I,t=0 means low_L1=high_L4+0.5.
        low_l1_origin = high_l2_origin / 4.0 + 0.5
        coordinates = H.registered_source_coordinates(
            "PHerc0139", np.eye(3), np.zeros(3), low_l1_origin, 0,
            source_level=2, size_l1=2, pixels_per_l1=2,
        )
        self.assertEqual(coordinates.shape, (3, 4, 4))
        # The upstream pyramids scale indices without inserting centre offsets.
        self.assertTrue(np.allclose(coordinates[0], 11.25))
        np.testing.assert_allclose(coordinates[1, :, 0], [19.0, 21.0, 23.0, 25.0])
        np.testing.assert_allclose(coordinates[2, 0, :], [29.0, 31.0, 33.0, 35.0])


class ArrayReader:
    def __init__(self, value: np.ndarray):
        self.value = value

    def read_roi(self, start, stop):
        first = np.asarray(start, dtype=int)
        last = np.asarray(stop, dtype=int)
        return self.value[tuple(slice(first[a], last[a]) for a in range(3))]


class SamplingAndRenderingTests(unittest.TestCase):
    def test_registered_plane_trilinear_sample(self) -> None:
        zz, yy, xx = np.mgrid[:12, :14, :16]
        volume = (3 * zz + 5 * yy + 7 * xx).astype(np.float32)
        coordinates = np.asarray([
            [[4.25, 4.25], [4.25, 4.25]],
            [[5.5, 5.5], [6.5, 6.5]],
            [[7.5, 8.5], [7.5, 8.5]],
        ])
        sampled, record = H.sample_registered_plane(ArrayReader(volume), coordinates)
        expected = 3 * coordinates[0] + 5 * coordinates[1] + 7 * coordinates[2]
        np.testing.assert_allclose(sampled, expected, atol=1e-5)
        self.assertEqual(record["interpolation"], "trilinear order=1 in zyx voxel-centre coordinates")

    def test_renderer_labels_proxy_and_writes_real_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.png"
            ct = np.zeros((256, 256, 256), dtype=np.uint8)
            ct[:, 64:192, 64:192] = 100
            source = np.indices((256, 256)).sum(axis=0).astype(np.float32) % 256
            bits = np.ones((64, 64), dtype=np.uint8)
            bits[16:48, 20:44] |= 8
            initial = np.full((64, 64), 0.1, dtype=np.float32)
            initial[20:45, 22:42] = 0.8
            candidate = np.full((64, 64), 0.1, dtype=np.float32)
            candidate[16:48, 20:44] = 0.8
            H.render_case_panel(
                path, "case", "PHerc1203", ct, source, bits, initial, candidate,
                4.806, 2.38,
            )
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertGreater(image.width, 1000)
                self.assertGreater(image.height, 500)


def write_pack(root: Path) -> dict:
    cases = []
    for scroll in H.SOURCE_SCANS:
        for stratum in range(4):
            for score_slice in H.REVIEW_SLICES_L1:
                case_id = f"{scroll}-z{stratum}"
                panel_id = f"{case_id}-k{score_slice:02d}"
                panel = root / "panels" / f"{panel_id}.png"
                panel.parent.mkdir(parents=True, exist_ok=True)
                panel.write_bytes(f"panel-{panel_id}".encode("ascii"))
                cases.append({
                    "panel_id": panel_id,
                    "case_id": case_id,
                    "scroll": scroll,
                    "z_stratum": stratum,
                    "score_slice_l1": score_slice,
                    "label_caster_reproduction": {
                        "valid_voxels": 1,
                        "mismatches_inside_valid": 0,
                    },
                    "panel": H._file_record(panel, f"panels/{panel.name}"),
                })
    value = {
        "schema_version": H.PACK_SCHEMA,
        "status": "RENDERED_NOT_HUMAN_REVIEWED",
        "review_slices_l1": list(H.REVIEW_SLICES_L1),
        "label_caster_reproduction": {
            "panels": len(cases),
            "panels_with_valid_voxels": len(cases),
            "valid_voxels": len(cases),
            "mismatches_inside_valid": 0,
        },
        "cases": cases,
        "tool": H._file_record(Path(H.__file__).resolve(), Path(H.__file__).name),
    }
    value["content_sha256"] = H._content_hash(value)
    (root / "review_pack.json").write_text(
        json.dumps(value, indent=2), encoding="utf-8"
    )
    return value


def valid_review(pack: dict) -> dict:
    reviewed = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": H.REVIEW_SCHEMA,
        "review_pack_content_sha256": pack["content_sha256"],
        "reviewer": "human-reviewer",
        "reviewed_utc": reviewed,
        "acknowledgement": H.REVIEW_ACKNOWLEDGEMENT,
        "release_recommendation": "RELEASE_WITH_AGREEMENT_ONLY",
        "supported_panel_ids": [],
        "cases": [{
            "panel_id": case["panel_id"],
            "case_id": case["case_id"],
            "score_slice_l1": case["score_slice_l1"],
            "panel_sha256": case["panel"]["sha256"],
            "registration_alignment": "PASS",
            "initial_disagreement_image_support": "MIXED",
            "candidate_change_image_support": "MIXED",
            "notes": "Reviewed at full resolution.",
        } for case in pack["cases"]],
    }


class ReviewReceiptTests(unittest.TestCase):
    def test_valid_agreement_only_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = write_pack(root)
            review = valid_review(pack)
            path = root / "human_review.json"
            path.write_text(json.dumps(review), encoding="utf-8")
            checked, payload = H.validate_human_review(root, path)
            self.assertEqual(checked["release_recommendation"], "RELEASE_WITH_AGREEMENT_ONLY")
            self.assertEqual(checked["content_sha256"], H._content_hash(checked))
            self.assertEqual(payload, path.read_bytes())

    def test_named_claim_requires_matching_supported_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = write_pack(root)
            review = valid_review(pack)
            supported = review["cases"][0]
            review["release_recommendation"] = "RELEASE_WITH_NAMED_IMAGE_SUPPORTED_CASES"
            review["supported_panel_ids"] = [supported["panel_id"]]
            path = root / "human_review.json"
            path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not fully supported"):
                H.validate_human_review(root, path)
            supported["initial_disagreement_image_support"] = "SUPPORTED"
            supported["candidate_change_image_support"] = "CORRECTION"
            path.write_text(json.dumps(review), encoding="utf-8")
            checked, _ = H.validate_human_review(root, path)
            self.assertEqual(checked["supported_panel_ids"], [supported["panel_id"]])

    def test_failed_alignment_blocks_release_and_panel_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = write_pack(root)
            review = valid_review(pack)
            review["cases"][3]["registration_alignment"] = "FAIL"
            path = root / "human_review.json"
            path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "blocks release"):
                H.validate_human_review(root, path)
            panel = root / pack["cases"][0]["panel"]["path"]
            panel.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "panel identity mismatch"):
                H.load_review_pack(root)

    def test_receipt_must_cover_exact_ordered_panel_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = write_pack(root)
            review = valid_review(pack)
            review["cases"].reverse()
            path = root / "human_review.json"
            path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact fixed panel hashes in order"):
                H.validate_human_review(root, path)

    def test_release_requires_two_alignment_passes_per_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = write_pack(root)
            review = valid_review(pack)
            first_case = review["cases"][0]["case_id"]
            for case in review["cases"]:
                if case["case_id"] == first_case:
                    case["registration_alignment"] = "UNCERTAIN"
            path = root / "human_review.json"
            path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "two image-alignment passes"):
                H.validate_human_review(root, path)


if __name__ == "__main__":
    unittest.main()
