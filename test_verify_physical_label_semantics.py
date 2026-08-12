from __future__ import annotations

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
