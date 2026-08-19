#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import crossscan_finetune as C
import run_crossscan_finetune as R
import score_crossscan_finetune as S


ROOT = Path(__file__).resolve().parent
PLAN = C.load_json(ROOT / "results/crossscan_finetune/plan.json")


class MetricTests(unittest.TestCase):
    def test_average_precision_exact_examples(self) -> None:
        self.assertAlmostEqual(
            S.average_precision(
                np.asarray([1, 0, 1]), np.asarray([0.9, 0.8, 0.7])
            ),
            (1.0 + 2.0 / 3.0) / 2.0,
        )
        self.assertAlmostEqual(
            S.average_precision(np.asarray([1, 0]), np.asarray([0.5, 0.5])),
            0.5,
        )

    def test_truth_masks_follow_training_semantics(self) -> None:
        bits = np.zeros((64, 64, 64), dtype=np.uint8)
        bits[0, 0, 0] = 1
        bits[0, 0, 1] = 3
        bits[0, 0, 2] = 9
        bits[0, 0, 3] = 11
        masks = S.truth_masks(bits)
        self.assertTrue(masks["negative"][0, 0, 0])
        self.assertTrue(masks["ignored_valid"][0, 0, 1])
        self.assertTrue(masks["positive"][0, 0, 2])
        self.assertTrue(masks["positive"][0, 0, 3])
        self.assertEqual(int(masks["supervised"].sum()), 3)

    def test_stable_top_n_breaks_ties_by_input_order(self) -> None:
        got = S.stable_top_n(np.asarray([0.5, 0.7, 0.7, 0.2]), 2)
        np.testing.assert_array_equal(got, [False, True, True, False])
        got = S.stable_top_n(np.asarray([0.7, 0.7, 0.7]), 2)
        np.testing.assert_array_equal(got, [True, True, False])

    def test_binary_metrics_keep_ignored_mass_separate(self) -> None:
        pred = np.asarray([1, 1, 1, 0], dtype=bool)
        pos = np.asarray([1, 0, 0, 1], dtype=bool)
        sup = np.asarray([1, 1, 0, 1], dtype=bool)
        got = S.binary_metrics(pred, pos, sup)
        self.assertEqual(got["true_positive"], 1)
        self.assertEqual(got["false_positive_supervised"], 1)
        self.assertEqual(got["false_negative"], 1)
        self.assertEqual(got["selected_ignored"], 1)
        self.assertAlmostEqual(got["dice"], 0.5)

    def test_pooled_comparison_improves_when_recto_is_ranked_first(self) -> None:
        bits = np.ones((64, 64, 64), dtype=np.uint8)
        bits[:, :4, :] = 9
        initial = np.full(bits.shape, 0.5, dtype=np.float32)
        candidate = np.full(bits.shape, 0.1, dtype=np.float32)
        candidate[(bits & 8) != 0] = 0.9
        case = {"block_id": "x", "z_stratum": 0}
        got = S._pooled_arrays([case], [bits], [initial], [candidate])
        self.assertGreater(got["average_precision_delta"], 0.5)
        self.assertEqual(got["candidate_average_precision"], 1.0)


class InferenceIdentityTests(unittest.TestCase):
    def test_fold_mapping_is_complementary(self) -> None:
        self.assertEqual([S.fold_for_stratum(PLAN, s) for s in range(4)],
                         ["odd", "even", "odd", "even"])

    def test_training_checkpoint_is_hashed_once_per_scoring_process(self) -> None:
        S._TRAINING_CHECKPOINT_SHA_CACHE.clear()
        lock = {"content_sha256": "lock"}
        with mock.patch.object(
            R, "load_training_receipt",
            return_value=({"checkpoint": {"sha256": "checkpoint"}}, Path("unused")),
        ) as loader:
            first = S.training_checkpoint_sha256(Path("data"), PLAN, lock, 40, 2000, "even")
            second = S.training_checkpoint_sha256(Path("data"), PLAN, lock, 40, 2000, "even")
        self.assertEqual((first, second), ("checkpoint", "checkpoint"))
        loader.assert_called_once()

    def test_prediction_loader_verifies_both_file_and_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = {"content_sha256": "lock"}
            case_id = "case"
            out = R.prediction_root(root, "initial", "pilot", None, None, None)
            out.mkdir(parents=True)
            probability = np.linspace(0, 1, 64 ** 3, dtype=np.float32).reshape(64, 64, 64)
            metadata = {
                "case_id": case_id,
                "kind": "initial",
                "scope": "pilot",
                "seed": None,
                "steps": None,
                "fold": None,
                "checkpoint_sha256": PLAN["inputs"]["model"][
                    "fold_0/checkpoint_best.pth"
                ]["sha256"],
                "plan_content_sha256": PLAN["content_sha256"],
                "execution_lock_content_sha256": "lock",
            }
            array_path = out / f"{case_id}.npz"
            R.atomic_save_npz(
                array_path,
                probability_l1=probability,
                metadata_json=np.asarray(C.canonical_json(metadata)),
            )
            receipt = {
                "schema_version": "crossscan-inference-case-v1",
                "status": "PASS",
                **metadata,
                "plan_content_sha256": PLAN["content_sha256"],
                "execution_lock_content_sha256": "lock",
                "array_file": R.file_record(array_path),
                "probability_array_sha256": R.array_sha256(probability),
                "probability_min": float(probability.min()),
                "probability_max": float(probability.max()),
            }
            C.write_json(out / f"{case_id}.json", R._with_content_hash(receipt))
            got = S.load_prediction(root, PLAN, lock, case_id, "initial", "pilot")
            np.testing.assert_array_equal(got, probability)
            probability[0, 0, 0] = 0.25
            R.atomic_save_npz(
                array_path,
                probability_l1=probability,
                metadata_json=np.asarray(C.canonical_json(metadata)),
            )
            with self.assertRaisesRegex(ValueError, "file hash"):
                S.load_prediction(root, PLAN, lock, case_id, "initial", "pilot")

            probability64 = probability.astype(np.float64)
            R.atomic_save_npz(
                array_path,
                probability_l1=probability64,
                metadata_json=np.asarray(C.canonical_json(metadata)),
            )
            receipt["array_file"] = R.file_record(array_path)
            receipt["probability_array_sha256"] = R.array_sha256(probability64)
            receipt["probability_min"] = float(probability64.min())
            receipt["probability_max"] = float(probability64.max())
            C.write_json(out / f"{case_id}.json", R._with_content_hash(receipt))
            with self.assertRaisesRegex(ValueError, "dtype"):
                S.load_prediction(root, PLAN, lock, case_id, "initial", "pilot")


class VerdictTests(unittest.TestCase):
    def test_machine_plan_fixes_inferential_steps_at_4000(self) -> None:
        self.assertEqual(
            PLAN["training"]["inferential_steps"], C.PILOT_RETRY_STEPS
        )

    def test_pilot_attempt_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            C.write_json(root / "pilot_attempt_steps-2000.json", {"sentinel": 1})
            with self.assertRaisesRegex(SystemExit, "attempt already exists"):
                S.score_pilot(root, {}, {}, 2000)

    def test_2000_step_pilot_pass_authorizes_frozen_4000_step_inference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = {
                "overall": {"average_precision_delta": C.PILOT_AP_GATE},
                "by_z_stratum": {
                    str(index): {"average_precision_delta": 0.0}
                    for index in range(C.Z_STRATA)
                },
            }
            plan = {"content_sha256": "plan", "cases": {"pilot": []}}
            lock = {"content_sha256": "lock"}
            with mock.patch.object(S, "load_comparison", return_value=result):
                attempt = S.score_pilot(root, plan, lock, C.PILOT_STEPS)
            verdict = C.load_json(root / "pilot_verdict.json")
            self.assertEqual(attempt["decision"], "PASS")
            self.assertEqual(attempt["steps"], C.PILOT_STEPS)
            self.assertEqual(verdict["status"], "PASS")
            self.assertEqual(verdict["selected_steps"], C.PILOT_RETRY_STEPS)
            self.assertEqual(verdict["attempt_content_sha256"], attempt["content_sha256"])
            self.assertEqual(
                verdict["content_sha256"], C.content_hash_without_field(verdict)
            )

    def test_t_summary_and_positive_bucket(self) -> None:
        primary = S.t_summary([0.011, 0.012, 0.013, 0.014, 0.015, 0.016])
        safety = S.t_summary([0.0, 0.001, -0.001, 0.0, 0.001, -0.001])
        self.assertEqual(S.outcome_bucket(primary, safety), "POSITIVE_DEPLOYABLE")
        safety["mean"] = -0.006
        self.assertEqual(
            S.outcome_bucket(primary, safety), "POSITIVE_WITH_SAFETY_REGRESSION"
        )

    def test_regression_and_inconclusive_buckets(self) -> None:
        regression = S.t_summary([-0.011, -0.012, -0.013, -0.014, -0.015, -0.016])
        safety = S.t_summary([0, 0, 0, 0, 0, 0])
        self.assertEqual(S.outcome_bucket(regression, safety), "REGRESSION")
        mixed = S.t_summary([-0.02, 0.02, -0.02, 0.02, -0.02, 0.02])
        self.assertEqual(S.outcome_bucket(mixed, safety), "INCONCLUSIVE_UNDERPOWERED")


class VisualTests(unittest.TestCase):
    def test_fixed_composite_is_written_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = {
                "block_id": "visual-case",
                "scroll": C.TRAIN_SCROLL,
                "z_stratum": 0,
            }
            plan = {
                "cases": {
                    "train": [], "internal_validation": [], "pilot": [],
                    "primary": {C.TRAIN_SCROLL: [case], C.SAFETY_SCROLL: []},
                },
                "folds": {
                    "odd": {"held_out_z_strata": [0]},
                    "even": {"held_out_z_strata": [1, 2, 3]},
                },
            }
            lock = {"resolved_protocol": {"visual_cases": [{
                "case_id": "visual-case",
                "scroll": C.TRAIN_SCROLL,
                "z_stratum": 0,
                "score_slice_l1": 32,
            }]}}
            ct = np.zeros((256, 256, 256), dtype=np.uint8)
            truth = np.ones((64, 64, 64), dtype=np.uint8)
            probability = np.full((64, 64, 64), 0.25, dtype=np.float32)
            with (
                mock.patch.object(
                    R, "verify_evaluation_case", return_value=(ct, truth, {})
                ),
                mock.patch.object(S, "load_prediction", return_value=probability),
            ):
                records = S.make_figures(root, plan, lock, 2000)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["rows"], [40, 41, 42, 43, 44, 45, "six_seed_mean"])
            panel = root / records[0]["file"]["path"]
            self.assertTrue(panel.is_file())
            self.assertEqual(R.file_record(panel)["sha256"], records[0]["file"]["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
