from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numcodecs
import numpy as np
import zarr

import verify_physical_label_semantics as V


class CensusTests(unittest.TestCase):
    def test_counts_bits_and_containment_from_decoded_arrays(self) -> None:
        block = np.array([[[
            0,
            V.BITS["valid"],
            V.BITS["valid"] | V.BITS["material"],
            V.BITS["valid"] | V.BITS["material"] | V.BITS["centerline"],
            V.BITS["valid"] | V.BITS["recto_band"],
            V.BITS["valid"] | V.BITS["material"] | V.BITS["boundary_poor"],
            V.BITS["material"],
            V.BITS["centerline"],
            V.BITS["recto_band"],
            V.BITS["boundary_poor"],
        ]]], dtype=np.uint8)
        result = V.census_blocks([block])
        self.assertEqual(result["counts"], {
            "window_voxels": 10,
            "valid": 5,
            "material": 4,
            "centerline": 2,
            "recto_band": 2,
            "boundary_poor": 2,
        })
        self.assertEqual(result["containment"], {
            "material_not_valid": 1,
            "centerline_not_material": 1,
            "recto_not_material": 2,
            "boundary_poor_not_material": 1,
            "boundary_poor_not_valid": 1,
        })

    def test_rejects_encoded_bytes_or_wrong_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "3-D uint8"):
            V.census_blocks([b"compressed bytes are not decoded labels"])
        with self.assertRaisesRegex(ValueError, "3-D uint8"):
            V.census_blocks([np.zeros((2, 2, 2), dtype=np.int16)])

    def test_expected_contract_pins_both_public_label_archives(self) -> None:
        self.assertEqual(set(V.EXPECTED), {"PHerc0139", "PHerc1203"})
        self.assertEqual(
            V.EXPECTED["PHerc1203"]["counts"]["boundary_poor"],
            3077792558,
        )
        self.assertEqual(
            V.EXPECTED["PHerc0139"]["tar"]["sha256"],
            "42fe53b760c2c9347d9f215bafa68beec8e96121d03549dab56a52a9a0a9e8dd",
        )

    def test_receipt_rejects_a_self_hashed_false_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = {}
            for scroll, expected in V.EXPECTED.items():
                records[scroll] = {
                    "tar": expected["tar"],
                    "extracted_tree": {"root": expected["path"], **expected["tree"]},
                    "decoded_zarr": {
                        "path": expected["path"],
                        "shape": expected["shape"],
                        "chunks": expected["chunks"],
                        "dtype": "uint8",
                        "counts": expected["counts"],
                        "containment": expected["containment"],
                        "fractions": V.fractions_from_counts(expected["counts"]),
                    },
                }
            value = {
                "schema_version": V.SCHEMA,
                "status": "PASS",
                "upstream_census_commit": V.UPSTREAM_CENSUS_COMMIT,
                "upstream_census_url": V.UPSTREAM_CENSUS_URL,
                "correction_url": V.CORRECTION_URL,
                "records": records,
            }
            value["content_sha256"] = V.content_hash(value)
            path = Path(tmp) / "audit.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            V.validate_audit_receipt(path)
            value["records"]["PHerc1203"]["decoded_zarr"]["fractions"][
                "material_of_valid"
            ] = 0.1
            value["content_sha256"] = V.content_hash(value)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fractions mismatch"):
                V.validate_audit_receipt(path)


class CompressedTarBindingTests(unittest.TestCase):
    def test_real_blosc_zarr_is_tar_bound_then_codec_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture.zarr"
            data = np.zeros((8, 8, 8), dtype=np.uint8)
            flat = data.reshape(-1)
            pattern = np.array([
                0,
                V.BITS["valid"],
                V.BITS["valid"] | V.BITS["material"],
                V.BITS["valid"] | V.BITS["material"] | V.BITS["centerline"],
                V.BITS["valid"] | V.BITS["recto_band"],
                V.BITS["valid"] | V.BITS["material"] | V.BITS["boundary_poor"],
            ], dtype=np.uint8)
            flat[:] = np.resize(pattern, flat.shape)
            array = zarr.open_array(
                str(root),
                mode="w",
                shape=data.shape,
                chunks=(4, 4, 4),
                dtype="uint8",
                compressor=numcodecs.Blosc(cname="zstd", clevel=5),
                zarr_format=2,
                dimension_separator="/",
            )
            array.attrs["bits"] = dict(V.BITS)
            array[:] = data
            tar_path = Path(tmp) / "fixture.tar"
            with tarfile.open(tar_path, mode="w") as archive:
                archive.add(root, arcname=root.name)

            binding = V.verify_tar_tree(tar_path, root, root.name)
            self.assertGreater(binding["file_count"], 2)
            census = V.census_blocks([data])
            expected = {
                "shape": list(data.shape),
                "chunks": [4, 4, 4],
                "bits": dict(V.BITS),
                "counts": census["counts"],
                "containment": census["containment"],
            }
            decoded = V.verify_array(root, expected)
            self.assertEqual(decoded["counts"], census["counts"])

            chunk = next(
                path for path in root.rglob("*")
                if path.is_file() and not path.name.startswith(".")
            )
            encoded = chunk.read_bytes()
            self.assertNotEqual(encoded, data[:4, :4, :4].tobytes(order="C"))
            raw = np.zeros(64, dtype=np.uint8)
            source = np.frombuffer(encoded[:64], dtype=np.uint8)
            raw[:source.size] = source
            self.assertNotEqual(
                V.census_blocks([raw.reshape(4, 4, 4)])["counts"],
                V.census_blocks([data[:4, :4, :4]])["counts"],
            )

            original = encoded
            chunk.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "differ|size"):
                V.verify_tar_tree(tar_path, root, root.name)
            chunk.write_bytes(original)
            extra = root / "unexpected"
            extra.write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "universe"):
                V.verify_tar_tree(tar_path, root, root.name)

    def test_rejects_unsafe_and_link_tar_members(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "fixture.zarr"
            root.mkdir()
            for name, kind in (("../escape", "file"), ("fixture.zarr/link", "link")):
                with self.subTest(name=name), tarfile.open(
                    Path(tmp) / f"{kind}.tar", mode="w"
                ) as archive:
                    info = tarfile.TarInfo(name)
                    if kind == "link":
                        info.type = tarfile.SYMTYPE
                        info.linkname = "target"
                        archive.addfile(info)
                    else:
                        payload = b"x"
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                with self.assertRaises(ValueError):
                    V.verify_tar_tree(Path(tmp) / f"{kind}.tar", root, root.name)


if __name__ == "__main__":
    unittest.main()
