from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

import export_crossscan_release as E
import predict_crossscan_probability_ensemble as P


class ProbabilityEnsembleTests(unittest.TestCase):
    def test_returns_exact_probability_mean_not_logit_mean(self) -> None:
        first = torch.tensor([
            [[[8.0, -1.0]]],
            [[[0.0, 1.0]]],
        ])
        second = torch.tensor([
            [[[0.0, 2.0]]],
            [[[2.0, -6.0]]],
        ])
        expected = (
            torch.softmax(first, dim=0) + torch.softmax(second, dim=0)
        ) / 2
        result = torch.softmax(P.probability_mean_as_logits([first, second]), dim=0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-7, rtol=0))
        logit_mean = torch.softmax((first + second) / 2, dim=0)
        self.assertFalse(torch.allclose(result, logit_mean, atol=1e-4, rtol=0))

    def test_rejects_empty_or_invalid_ensemble(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            P.probability_mean_as_logits([])
        with self.assertRaisesRegex(ValueError, "two-class"):
            P.probability_mean_as_logits([torch.zeros(3, 2)])
        invalid = torch.zeros(2, 1)
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid"):
            P.probability_mean_as_logits([invalid])


class CheckpointExportTests(unittest.TestCase):
    def test_export_removes_training_state_and_preserves_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pth"
            destination = root / "release" / "checkpoint_final.pth"
            weight = torch.arange(512 * 512, dtype=torch.float32).reshape(512, 512)
            torch.save({
                "network_weights": {"weight": weight, "weight_alias": weight},
                "optimizer_state": {
                    "state": {0: {"exp_avg": torch.ones_like(weight),
                                  "exp_avg_sq": torch.ones_like(weight)}}
                },
                "grad_scaler_state": {"scale": 1},
                "logging": {"loss": [1.0]},
                "_best_ema": 0.5,
                "current_epoch": 50,
                "init_args": {"configuration": E.R.CONFIGURATION, "device": "cuda"},
                "trainer_name": E.FINAL_TRAINER,
                "inference_allowed_mirroring_axes": None,
            }, source)
            metadata = {
                "schema_version": "test",
                "seed": 40,
                "training_fold": "even",
                "release_fold": 0,
            }
            result = E.export_checkpoint(source, destination, metadata)
            exported = torch.load(destination, map_location="cpu", weights_only=False)
            self.assertEqual(set(exported), {
                "network_weights", "current_epoch", "init_args", "trainer_name",
                "inference_allowed_mirroring_axes", "release_metadata",
            })
            self.assertEqual(exported["trainer_name"], "nnUNetTrainer")
            self.assertEqual(exported["init_args"], {"configuration": "3d_fullres"})
            self.assertEqual(exported["release_metadata"], metadata)
            self.assertTrue(torch.equal(exported["network_weights"]["weight"], weight))
            self.assertTrue(torch.equal(
                exported["network_weights"]["weight_alias"], weight
            ))
            self.assertLess(result["checkpoint"]["bytes"], result["source_checkpoint"]["bytes"])
            self.assertEqual(result["removed_training_state"], [
                "optimizer_state", "grad_scaler_state", "logging", "_best_ema"
            ])

    def test_rejects_wrong_trainer_or_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "out.pth"
            for trainer, configuration, message in (
                ("wrong", "3d_fullres", "source trainer"),
                (E.FINAL_TRAINER, "2d", "configuration"),
            ):
                source = root / f"{trainer}-{configuration}.pth"
                torch.save({
                    "network_weights": {"weight": torch.ones(2)},
                    "optimizer_state": {},
                    "trainer_name": trainer,
                    "init_args": {"configuration": configuration},
                }, source)
                with self.assertRaisesRegex(ValueError, message):
                    E.export_checkpoint(source, destination, {})


class ManifestTests(unittest.TestCase):
    def test_manifest_content_hash_is_order_stable(self) -> None:
        value = {"status": "PASS", "models": [1, 2], "content_sha256": "ignored"}
        first = P.content_hash(value)
        second = P.content_hash({"models": [1, 2], "status": "PASS"})
        self.assertEqual(first, second)
        encoded = P.canonical_json({"b": 2, "a": 1})
        self.assertEqual(encoded, '{"a":1,"b":2}')

    def test_load_manifest_rejects_wrong_ensemble_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            value = {
                "status": "PASS",
                "ensemble": {"aggregation": "logit mean", "fold_count": 12,
                             "mirroring": False},
            }
            value["content_sha256"] = P.content_hash(value)
            (root / "release_manifest.json").write_text(
                json.dumps(value), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "does not describe"):
                P.load_release_manifest(root)

    def test_integrity_check_covers_models_and_supporting_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def record(relative: str, payload: bytes) -> dict:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                return {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": P.sha256_file(path),
                }

            models = []
            release_fold = 0
            for seed in range(40, 46):
                for training_fold in ("even", "odd"):
                    checkpoint = record(
                        f"model/fold_{release_fold}/checkpoint_final.pth",
                        f"checkpoint-{release_fold}".encode("ascii"),
                    )
                    models.append({
                        "seed": seed,
                        "training_fold": training_fold,
                        "release_fold": release_fold,
                        "checkpoint": checkpoint,
                    })
                    release_fold += 1
            plans = record("model/plans.json", b"plans")
            evidence = record("evidence/final_result.json", b"result")
            tooling = record("README.md", b"card")
            manifest = {
                "status": "PASS",
                "ensemble": {
                    "aggregation": "arithmetic mean of class probabilities",
                    "fold_count": 12,
                    "mirroring": False,
                },
                "models": models,
                "model_files": {"plans": plans},
                "artifacts": [evidence],
                "tooling": [tooling],
            }
            manifest["content_sha256"] = P.content_hash(manifest)
            (root / "release_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            loaded = P.load_release_manifest(root)
            self.assertEqual(P.verify_release_files(root, loaded), list(range(12)))
            (root / "model" / "plans.json").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "release file hash mismatch"):
                P.verify_release_files(root, loaded)


if __name__ == "__main__":
    unittest.main()
