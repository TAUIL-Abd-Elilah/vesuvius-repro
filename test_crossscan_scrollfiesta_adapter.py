from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import zarr

import crossscan_scrollfiesta_adapter as A


class ProbabilityZarrTests(unittest.TestCase):
    def probability(self) -> np.ndarray:
        z, y, x = np.indices(A.SHAPE, dtype=np.float32)
        return np.ascontiguousarray((z + 2 * y + 3 * x) / 1530.0, dtype=np.float32)

    def test_ome_zarr_roundtrip_and_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "probability.zarr"
            original = self.probability()
            receipt = A.export_probability_zarr(original, root)
            recovered, origin = A.read_probability_zarr(root)
            self.assertTrue(np.array_equal(original, recovered))
            self.assertTrue(np.array_equal(zarr.open_array(root / "0", mode="r")[:], original))
            self.assertEqual(origin, A.DEFAULT_ORIGIN)
            self.assertEqual(receipt["status"], "PASS")
            self.assertEqual(receipt["format"], "OME-NGFF Zarr v2")
            self.assertEqual(len(receipt["chunk_records"]), 8)
            self.assertTrue(receipt["roundtrip_equal"])
            self.assertEqual(receipt["content_sha256"], A.content_hash(receipt))
            self.assertEqual(
                A.verify_probability_export(root)["content_sha256"],
                receipt["content_sha256"],
            )
            attrs = json.loads((root / ".zattrs").read_text(encoding="utf-8"))
            transforms = attrs["multiscales"][0]["datasets"][0][
                "coordinateTransformations"
            ]
            self.assertEqual(transforms[1]["translation"], [3840.0, 3712.0, 1344.0])

    def test_resume_requires_matching_chunk_and_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "probability.zarr"
            value = self.probability()
            # Build a legitimate partial export containing exactly one pair.
            root.mkdir()
            for relative, payload in A._metadata(A.DEFAULT_ORIGIN).items():
                A._create_bytes(root / relative, payload)
            index, payload = next(iter(A._chunk_records(value)))
            name = ".".join(str(v) for v in index)
            record = {
                "index_zyx": list(index),
                "path": f"0/{name}",
                "bytes": len(payload),
                "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                "input_probability_sha256": A.sha256_array(value),
            }
            record["content_sha256"] = A.content_hash(record)
            A._create_bytes(root / "0" / name, payload)
            A._create_bytes(root / "receipts" / f"{name}.json", A._json_bytes(record))
            receipt = A.export_probability_zarr(value, root, resume=True)
            self.assertEqual(receipt["resource_measurements"]["resumed_chunks"], 1)
            self.assertEqual(len(receipt["chunk_records"]), 8)

    def test_resume_rejects_tampered_or_orphan_chunk(self) -> None:
        value = self.probability()
        for mode in ("tampered", "orphan"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp) / "probability.zarr"
                root.mkdir()
                for relative, payload in A._metadata(A.DEFAULT_ORIGIN).items():
                    A._create_bytes(root / relative, payload)
                index, payload = next(iter(A._chunk_records(value)))
                name = ".".join(str(v) for v in index)
                if mode == "tampered":
                    payload = b"x" + payload[1:]
                A._create_bytes(root / "0" / name, payload)
                if mode == "tampered":
                    record = {
                        "index_zyx": list(index),
                        "path": f"0/{name}",
                        "bytes": len(payload),
                        "sha256": "0" * 64,
                        "input_probability_sha256": A.sha256_array(value),
                    }
                    record["content_sha256"] = A.content_hash(record)
                    A._create_bytes(
                        root / "receipts" / f"{name}.json", A._json_bytes(record)
                    )
                with self.assertRaises(ValueError):
                    A.export_probability_zarr(value, root, resume=True)

    def test_rejects_completed_overwrite_and_bad_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "probability.zarr"
            value = self.probability()
            A.export_probability_zarr(value, root)
            with self.assertRaises(FileExistsError):
                A.export_probability_zarr(value, root, resume=True)
            bad = value.copy()
            bad[0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "non-finite"):
                A.export_probability_zarr(bad, Path(tmp) / "bad.zarr")

    def test_final_verifier_rejects_post_export_chunk_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "probability.zarr"
            A.export_probability_zarr(self.probability(), root)
            chunk = root / "0" / "0.0.0"
            payload = bytearray(chunk.read_bytes())
            payload[0] ^= 1
            chunk.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "chunk hash mismatch"):
                A.verify_probability_export(root)


class MaskTests(unittest.TestCase):
    def test_fixed_threshold_is_inclusive(self) -> None:
        value = np.zeros(A.SHAPE, dtype=np.float32)
        value.flat[:3] = [0.19, 0.20, 0.21]
        mask = A.fixed_threshold_mask(value, 0.2)
        self.assertEqual(mask.flat[:4].tolist(), [0, 255, 255, 0])

    def test_matched_mass_is_exact_and_c_order_stable(self) -> None:
        value = np.zeros(A.SHAPE, dtype=np.float32)
        value.flat[:6] = [0.8, 0.9, 0.8, 0.7, 0.8, 0.1]
        mask = A.matched_mass_mask(value, 4)
        self.assertEqual(np.flatnonzero(mask).tolist(), [0, 1, 2, 4])
        self.assertEqual(int(np.count_nonzero(mask)), 4)


class GridTests(unittest.TestCase):
    def test_materializes_exact_eight_cube_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            for z in (3840, 3968):
                for y in (3712, 3840):
                    for x in (1344, 1472):
                        cube = np.full((128, 128, 128), (z + y + x) % 251, dtype=np.uint8)
                        tifffile.imwrite(
                            raw / f"z{z:05d}_y{y:05d}_x{x:05d}.tif",
                            cube,
                            photometric="minisblack",
                            compression=None,
                            rowsperstrip=128,
                        )
            mask = np.zeros(A.SHAPE, dtype=np.uint8)
            mask[:, :, 127:129] = 255
            output = root / "grid"
            manifest = A.materialize_scrollfiesta_grid(
                mask, raw, output, arm="fixed"
            )
            self.assertEqual(len(list((output / "cubes_PRED").glob("*.tif"))), 8)
            self.assertEqual(len(list((output / "cubes_RAW").glob("*.tif"))), 8)
            self.assertEqual(len(manifest["files"]), 16)
            self.assertEqual(manifest["foreground_voxels"], int(np.count_nonzero(mask)))
            self.assertEqual(manifest["bbox_l0_zyx"], [3840, 4096, 3712, 3968, 1344, 1600])
            self.assertEqual(manifest["content_sha256"], A.content_hash(manifest))
            restored = tifffile.imread(
                output / "cubes_PRED" / "z03840_y03712_x01344.tif"
            )
            self.assertEqual(restored.shape, (128, 128, 128))
            self.assertEqual(restored.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
