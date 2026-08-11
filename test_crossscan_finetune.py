#!/usr/bin/env python3
from __future__ import annotations

import copy
import unittest

import numpy as np

import crossscan_finetune as cf


def synthetic_candidates() -> list[dict]:
    out = []
    zs = [192, 704, 1216, 1728]
    fractions = [0.20, 0.60, 0.90, 0.98]
    for stratum, z in enumerate(zs):
        for bin_id, fraction in enumerate(fractions):
            for i in range(25):
                flat = bin_id * 25 + i
                y = 512 + (flat % 16) * 128
                x = 512 + (flat // 16) * 128
                out.append(
                    {
                        "local_origin_l1": [z, y, x],
                        "z_stratum": stratum,
                        "label_stats": {
                            "valid_fraction": 0.90,
                            "recto_count": 8192,
                            "boundary_poor_fraction_of_material": fraction,
                        },
                    }
                )
    return out


def synthetic_primary(scroll: str) -> list[dict]:
    if scroll == cf.TRAIN_SCROLL:
        zs = [192, 704, 1216, 1728]
    else:
        zs = [64, 320, 704, 1088]
    out = []
    for stratum, z in enumerate(zs):
        for i in range(8):
            origin = [z, 256 + i * 128, 256]
            block = {
                "block_id": f"{scroll}-s{stratum}-i{i}",
                "scroll": scroll,
                "z_stratum": stratum,
                "local_origin_l1": origin,
                "label_stats": {
                    "valid_fraction": 0.9,
                    "recto_count": 10000,
                    "boundary_poor_fraction_of_material": 0.5,
                },
                "array_file": f"arrays/{scroll}-s{stratum}-i{i}.npz",
            }
            out.append(cf._primary_case(block))
    return out


def synthetic_plan() -> dict:
    allocations = cf.allocate_cases(synthetic_candidates(), set())
    primary = {
        cf.TRAIN_SCROLL: synthetic_primary(cf.TRAIN_SCROLL),
        cf.SAFETY_SCROLL: synthetic_primary(cf.SAFETY_SCROLL),
    }
    allocations["primary"] = primary
    folds = copy.deepcopy(cf.FOLDS)
    bare = {
        "schema_version": "test",
        "protocol_version": cf.PROTOCOL_VERSION,
        "status": cf.PLAN_STATUS,
        "folds": folds,
        "cases": allocations,
    }
    return cf._with_content_hash(bare)


class TargetTests(unittest.TestCase):
    def test_physical_target_truth_table(self) -> None:
        labels = np.asarray([0, 1, 3, 9, 11, 8], dtype=np.uint8)
        got = cf.physical_target_l1(labels)
        np.testing.assert_array_equal(got, np.asarray([2, 0, 2, 1, 1, 2], np.uint8))

    def test_target_upsampling_is_nearest_neighbor(self) -> None:
        target = np.asarray([[[0, 1], [2, 0]]], dtype=np.uint8)
        got = cf.upsample_target_l0(target)
        self.assertEqual(got.shape, (2, 4, 4))
        for z in range(target.shape[0]):
            for y in range(target.shape[1]):
                for x in range(target.shape[2]):
                    block = got[2*z:2*z+2, 2*y:2*y+2, 2*x:2*x+2]
                    self.assertTrue(np.all(block == target[z, y, x]))

    def test_target_rejects_non_integer_labels(self) -> None:
        with self.assertRaises(TypeError):
            cf.physical_target_l1(np.zeros((2, 2, 2), dtype=np.float32))


class SamplingTests(unittest.TestCase):
    def test_difficulty_bin_boundaries(self) -> None:
        values = [0.0, 0.3999, 0.4, 0.7999, 0.8, 0.9499, 0.95, 1.0]
        self.assertEqual([cf.difficulty_bin(v) for v in values], [0, 0, 1, 1, 2, 2, 3, 3])

    def test_allocation_is_balanced_disjoint_and_order_invariant(self) -> None:
        candidates = synthetic_candidates()
        a = cf.allocate_cases(candidates, set())
        b = cf.allocate_cases(list(reversed(candidates)), set())
        for role, per_bin in (
            ("train", cf.TRAIN_PER_BIN_PER_STRATUM),
            ("internal_validation", cf.VALIDATION_PER_BIN_PER_STRATUM),
            ("pilot", cf.PILOT_PER_BIN_PER_STRATUM),
        ):
            self.assertEqual(
                [c["case_id"] for c in a[role]],
                [c["case_id"] for c in b[role]],
            )
            for stratum in range(4):
                for bin_id in range(4):
                    self.assertEqual(
                        sum(c["z_stratum"] == stratum and c["difficulty_bin"] == bin_id
                            for c in a[role]),
                        per_bin,
                    )
        origins = [tuple(c["local_origin_l1"]) for cases in a.values() for c in cases]
        self.assertEqual(len(origins), len(set(origins)))

    def test_primary_origins_are_excluded(self) -> None:
        candidates = synthetic_candidates()
        excluded = tuple(candidates[0]["local_origin_l1"])
        allocations = cf.allocate_cases(candidates, {excluded})
        origins = {
            tuple(c["local_origin_l1"])
            for cases in allocations.values() for c in cases
        }
        self.assertNotIn(excluded, origins)


class GeometryAndPlanTests(unittest.TestCase):
    def test_box_geometry(self) -> None:
        origin = [192, 1024, 1536]
        self.assertEqual(
            cf._box_from_score_origin(origin, cf.TRAIN_CROP_SIZE_L1),
            [176, 272, 1008, 1104, 1520, 1616],
        )
        self.assertEqual(
            cf._box_from_score_origin(origin, cf.EVAL_CONTEXT_SIZE_L1),
            [160, 288, 992, 1120, 1504, 1632],
        )
        self.assertTrue(cf.boxes_intersect([0, 2, 0, 2, 0, 2], [1, 3, 1, 3, 1, 3]))
        self.assertFalse(cf.boxes_intersect([0, 2, 0, 2, 0, 2], [2, 4, 0, 2, 0, 2]))

    def test_full_synthetic_plan_passes(self) -> None:
        result = cf.validate_plan(synthetic_plan())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["train_cases"], 256)
        self.assertEqual(result["internal_validation_cases"], 32)
        self.assertEqual(result["pilot_cases"], 32)
        self.assertEqual(result["primary_cases"], 64)

    def test_content_hash_mutation_is_detected(self) -> None:
        plan = synthetic_plan()
        plan["cases"]["train"][0]["case_id"] = "mutated"
        with self.assertRaisesRegex(ValueError, "content hash"):
            cf.validate_plan(plan)

    def test_train_eval_label_overlap_is_detected(self) -> None:
        plan = synthetic_plan()
        fold = cf.FOLDS["even"]
        train_case = next(
            c for c in plan["cases"]["train"]
            if c["z_stratum"] in fold["train_z_strata"]
        )
        held_case = next(
            c for c in plan["cases"]["primary"][cf.TRAIN_SCROLL]
            if c["z_stratum"] in fold["held_out_z_strata"]
        )
        train_case["training_label_box_local_l1"] = held_case["score_box_local_l1"]
        plan = cf._with_content_hash(plan)
        with self.assertRaisesRegex(ValueError, "intersect held-out"):
            cf.validate_plan(plan)


if __name__ == "__main__":
    unittest.main(verbosity=2)
