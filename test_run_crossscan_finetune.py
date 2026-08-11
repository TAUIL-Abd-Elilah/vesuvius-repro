#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import crossscan_finetune as C
import run_crossscan_finetune as R


ROOT = Path(__file__).resolve().parent
PLAN = C.load_json(ROOT / "results/crossscan_finetune/plan.json")


class GeometryAndSelectionTests(unittest.TestCase):
    def test_frozen_case_counts(self) -> None:
        self.assertEqual(len(R.training_cases(PLAN)), 288)
        self.assertEqual(len(R.evaluation_cases(PLAN)), 96)
        self.assertEqual(len(R._case_map(PLAN)), 384)

    def test_visual_cases_are_fixed_and_balanced(self) -> None:
        first = R.frozen_visual_cases(PLAN)
        second = R.frozen_visual_cases(PLAN)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 8)
        self.assertEqual(
            {(v["scroll"], v["z_stratum"]) for v in first},
            {(scroll, s) for scroll in (C.TRAIN_SCROLL, C.SAFETY_SCROLL) for s in range(4)},
        )
        self.assertTrue(all(v["score_slice_l1"] == 32 for v in first))

    def test_inference_case_routing(self) -> None:
        self.assertEqual(len(R.select_inference_cases(PLAN, "pilot", None)), 32)
        self.assertEqual(len(R.select_inference_cases(PLAN, "pilot", "even")), 16)
        self.assertEqual(len(R.select_inference_cases(PLAN, "primary", None)), 32)
        self.assertEqual(len(R.select_inference_cases(PLAN, "primary", "odd")), 16)
        self.assertEqual(len(R.select_inference_cases(PLAN, "safety", "even")), 32)
        self.assertEqual(R.fold_index("even"), 0)
        self.assertEqual(R.fold_index("odd"), 1)


class ArrayTests(unittest.TestCase):
    def test_max_pool(self) -> None:
        value = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        got = R.max_pool_l0_to_l1(value)
        expected = np.asarray([
            [[21, 23], [29, 31]],
            [[53, 55], [61, 63]],
        ], dtype=np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_normalization_matches_ct_rule(self) -> None:
        plans = {
            "foreground_intensity_properties_per_channel": {
                "0": {
                    "percentile_00_5": 10.0,
                    "percentile_99_5": 20.0,
                    "mean": 15.0,
                    "std": 5.0,
                }
            }
        }
        value = np.asarray([0, 10, 15, 20, 30], dtype=np.uint8)
        np.testing.assert_allclose(
            R.normalize_ct(value, plans),
            np.asarray([-1, -1, 0, 1, 1], dtype=np.float32),
        )

    def test_validate_ct(self) -> None:
        value = np.zeros((2, 3, 4), dtype=np.uint8)
        self.assertIs(R.validate_ct(value, (2, 3, 4)), value)
        with self.assertRaises(ValueError):
            R.validate_ct(value.astype(np.float32), (2, 3, 4))


class FilesAndConfigurationTests(unittest.TestCase):
    def test_one_case_training_materialization_round_trip(self) -> None:
        import tifffile
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_root = root / "truth"
            store_path = labels_root / C.SCROLLS[C.TRAIN_SCROLL]["label_store"]
            store = zarr.open(
                str(store_path), mode="w", shape=(96, 96, 96),
                chunks=(32, 32, 32), dtype="u1", zarr_format=2,
            )
            bits = np.ones((96, 96, 96), dtype=np.uint8)
            bits[:, 40:56, :] = 11
            store[:] = bits
            ct = np.arange(192, dtype=np.uint8)[:, None, None]
            ct = np.broadcast_to(ct, (192, 192, 192)).copy()
            case = {
                "case_id": "synthetic-train",
                "scroll": C.TRAIN_SCROLL,
                "role": "train",
                "z_stratum": 0,
                "difficulty_bin": 0,
                "training_label_box_local_l1": [0, 96, 0, 96, 0, 96],
                "training_ct_box_global_l0": [0, 192, 0, 192, 0, 192],
            }
            plan = {
                "content_sha256": "plan",
                "inputs": {"ct_urls": {C.TRAIN_SCROLL: "synthetic"}},
                "cases": {"train": [case], "internal_validation": []},
                "folds": {
                    "even": {"train_case_ids": [case["case_id"]], "internal_validation_case_ids": []},
                    "odd": {"train_case_ids": [], "internal_validation_case_ids": []},
                },
            }
            lock = {"content_sha256": "lock"}
            pre = root / "nnUNet_preprocessed" / R.DATASET_NAME
            pre.mkdir(parents=True)
            C.write_json(pre / f"{R.PLANS_IDENTIFIER}.json", {"sentinel": 1})
            with mock.patch.object(R, "open_remote_zarr", return_value=ct):
                result = R.materialize_training(plan, lock, labels_root, root, 1)
            self.assertEqual(result, {"training_cases": 1, "status": "PASS"})
            image = root / "nnUNet_raw" / R.DATASET_NAME / "imagesTr/synthetic-train_0000.tif"
            label = root / "nnUNet_raw" / R.DATASET_NAME / "labelsTr/synthetic-train.tif"
            np.testing.assert_array_equal(tifffile.imread(image), ct)
            target = tifffile.imread(label)
            self.assertEqual(target.shape, (192, 192, 192))
            self.assertGreater(int((target == 0).sum()), 0)
            self.assertGreater(int((target == 1).sum()), 0)
            receipt = C.load_json(
                root / "materialization/training_receipts/synthetic-train.json"
            )
            self.assertEqual(receipt["plan_content_sha256"], "plan")
            self.assertEqual(receipt["execution_lock_content_sha256"], "lock")
            self.assertEqual(
                receipt["content_sha256"], C.content_hash_without_field(receipt)
            )

    def test_one_case_evaluation_materialization_round_trip(self) -> None:
        import zarr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_root = root / "truth"
            store_path = labels_root / C.SCROLLS[C.TRAIN_SCROLL]["label_store"]
            store = zarr.open(
                str(store_path), mode="w", shape=(64, 64, 64),
                chunks=(32, 32, 32), dtype="u1", zarr_format=2,
            )
            bits = np.ones((64, 64, 64), dtype=np.uint8)
            bits[:, 20:30, :] = 11
            store[:] = bits
            ct = np.arange(256, dtype=np.uint8)[:, None, None]
            ct = np.broadcast_to(ct, (256, 256, 256)).copy()
            case = {
                "block_id": "synthetic-primary",
                "scroll": C.TRAIN_SCROLL,
                "z_stratum": 0,
                "evaluation_ct_box_global_l0": [0, 256, 0, 256, 0, 256],
                "score_box_local_l1": [0, 64, 0, 64, 0, 64],
            }
            plan = {
                "content_sha256": "plan",
                "inputs": {"ct_urls": {C.TRAIN_SCROLL: "synthetic"}},
                "cases": {
                    "train": [], "internal_validation": [], "pilot": [],
                    "primary": {
                        C.TRAIN_SCROLL: [case], C.SAFETY_SCROLL: [],
                    },
                },
            }
            lock = {"content_sha256": "lock"}
            with mock.patch.object(R, "open_remote_zarr", return_value=ct):
                result = R.materialize_evaluation(plan, lock, labels_root, root, 1)
            self.assertEqual(result, {"evaluation_cases": 1, "status": "PASS"})
            got_ct, got_truth, receipt = R.verify_evaluation_case(
                plan, lock, root, case["block_id"]
            )
            np.testing.assert_array_equal(got_ct, ct)
            np.testing.assert_array_equal(got_truth, bits)
            self.assertEqual(receipt["ct_box_global_l0"], case["evaluation_ct_box_global_l0"])
            truth_path = root / "evaluation/synthetic-primary/truth_bits_l1.npy"
            with truth_path.open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x00")
            with self.assertRaisesRegex(ValueError, "truth file hash"):
                R.verify_evaluation_case(plan, lock, root, case["block_id"])

    def test_atomic_array_and_json_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = np.arange(12, dtype=np.int16).reshape(3, 4)
            R.atomic_save_npy(root / "a.npy", value)
            np.testing.assert_array_equal(np.load(root / "a.npy", allow_pickle=False), value)
            R.atomic_save_npz(root / "b.npz", x=value)
            with np.load(root / "b.npz", allow_pickle=False) as payload:
                np.testing.assert_array_equal(payload["x"], value)
            R.atomic_write_json(root / "c.json", {"x": 1})
            self.assertEqual(json.loads((root / "c.json").read_text()), {"x": 1})
            self.assertFalse(any(p.name.endswith(".tmp") for p in root.iterdir()))

    def test_dataset_configuration_preserves_m7_architecture_and_sets_batch_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            source_plans = {
                "dataset_name": "Dataset100_VesuviusSurface",
                "plans_name": "nnUNetResEncUNetLPlans",
                "image_reader_writer": "Tiff3DIO",
                "configurations": {"3d_fullres": {"batch_size": 3, "sentinel": 7}},
            }
            C.write_json(model / "plans.json", source_plans)
            R.seed_dataset_plans(PLAN, model, root)
            R.write_dataset_configuration(PLAN, root)
            copied = C.load_json(
                root / "nnUNet_preprocessed" / R.DATASET_NAME
                / f"{R.PLANS_IDENTIFIER}.json"
            )
            self.assertEqual(copied["dataset_name"], R.DATASET_NAME)
            self.assertEqual(copied["configurations"]["3d_fullres"]["batch_size"], 1)
            self.assertEqual(copied["configurations"]["3d_fullres"]["sentinel"], 7)
            dataset = C.load_json(root / "nnUNet_raw" / R.DATASET_NAME / "dataset.json")
            self.assertEqual(dataset["numTraining"], 288)
            self.assertEqual(dataset["labels"]["ignore"], 2)
            splits = C.load_json(
                root / "nnUNet_preprocessed" / R.DATASET_NAME / "splits_final.json"
            )
            self.assertEqual([len(s["train"]) for s in splits], [128, 128])
            self.assertEqual([len(s["val"]) for s in splits], [16, 16])

    def test_preprocessed_receipt_detects_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "nnUNet_preprocessed" / R.DATASET_NAME
            folder = base / "synthetic_data"
            folder.mkdir(parents=True)
            C.write_json(base / f"{R.PLANS_IDENTIFIER}.json", {
                "configurations": {
                    R.CONFIGURATION: {"data_identifier": "synthetic_data"},
                },
            })
            artifact = folder / "case.b2nd"
            artifact.write_bytes(b"locked")
            plan = {
                "content_sha256": "plan",
                "cases": {"train": [], "internal_validation": []},
            }
            lock = {"content_sha256": "lock"}
            receipt = R._with_content_hash({
                "schema_version": "crossscan-preprocessing-v1",
                "status": "PASS",
                "plan_content_sha256": "plan",
                "execution_lock_content_sha256": "lock",
                "dataset_name": R.DATASET_NAME,
                "configuration": R.CONFIGURATION,
                "case_count": 0,
                "files": R.preprocessed_file_records(root),
            })
            C.write_json(root / "preprocessing_receipt.json", receipt)
            self.assertEqual(
                R.verify_preprocessed(plan, lock, root)["content_sha256"],
                receipt["content_sha256"],
            )
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differ from receipt"):
                R.verify_preprocessed(plan, lock, root)

    def test_training_receipt_uses_locked_relative_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lock = {"content_sha256": "lock"}
            preprocessing = R._with_content_hash({"status": "PASS"})
            C.write_json(root / "preprocessing_receipt.json", preprocessing)
            checkpoint = R.training_output_folder(root, 40, 2000, "even") / "checkpoint_final.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            config = root / "training/steps-2000/seed-40/even/training_config.json"
            config.write_bytes(b"{}\n")
            receipt = {
                "schema_version": "crossscan-training-run-v1",
                "status": "PASS",
                "plan_content_sha256": PLAN["content_sha256"],
                "execution_lock_content_sha256": "lock",
                "seed": 40,
                "fold": "even",
                "steps": 2000,
                "epochs": 50,
                "preprocessing_receipt_content_sha256": preprocessing["content_sha256"],
                "initial_checkpoint_sha256": PLAN["inputs"]["model"][
                    "fold_0/checkpoint_best.pth"
                ]["sha256"],
                "training_config": {
                    "relative_path": R.relative_data_path(root, config),
                    **R.file_record(config),
                },
                "checkpoint": {
                    "relative_path": R.relative_data_path(root, checkpoint),
                    **R.file_record(checkpoint),
                },
            }
            receipt_path = root / "training_receipts/seed-40-even-steps-2000.json"
            C.write_json(receipt_path, R._with_content_hash(receipt))
            got, got_path = R.load_training_receipt(
                PLAN, lock, root, 40, 2000, "even"
            )
            self.assertEqual(got_path, checkpoint)
            self.assertEqual(got["checkpoint"]["sha256"], R.file_record(checkpoint)["sha256"])
            receipt["checkpoint"]["relative_path"] = "../escape.pth"
            C.write_json(receipt_path, R._with_content_hash(receipt))
            with self.assertRaisesRegex(ValueError, "path mismatch"):
                R.load_training_receipt(PLAN, lock, root, 40, 2000, "even")

    def test_pilot_authorization_is_content_hashed_and_step_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verdict = R._with_content_hash({"status": "PASS", "selected_steps": 2000})
            C.write_json(root / "pilot_verdict.json", verdict)
            self.assertEqual(R.require_pilot_authorization(root, 2000)["status"], "PASS")
            with self.assertRaises(SystemExit):
                R.require_pilot_authorization(root, 4000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
