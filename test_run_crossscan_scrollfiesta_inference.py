from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import crossscan_scrollfiesta_adapter as A
import run_crossscan_scrollfiesta_inference as I


class _FakeNetwork:
    def __init__(self, owner) -> None:
        self.owner = owner

    def load_state_dict(self, parameters, strict=True) -> None:
        self.owner.current_probability = float(parameters)


class _FakePredictor:
    def __init__(self, probabilities) -> None:
        self.list_of_parameters = list(probabilities)
        self.current_probability = float(self.list_of_parameters[0])
        self.network = _FakeNetwork(self)

    def initialize_from_trained_model_folder(self, *args, **kwargs) -> None:
        return None

    def predict_sliding_window_return_logits(self, tensor):
        probability = self.current_probability
        shape = tuple(A.CONTEXT_SHAPE)
        return torch.stack((
            torch.full(shape, math.log1p(-probability), dtype=torch.float32),
            torch.full(shape, math.log(probability), dtype=torch.float32),
        ))


class LockedInferenceBehaviorTests(unittest.TestCase):
    def shape_patches(self):
        return (
            mock.patch.object(A, "CONTEXT_SHAPE", (6, 6, 6)),
            mock.patch.object(A, "SHAPE", (4, 4, 4)),
        )

    def test_baseline_retains_the_centered_class_probability(self) -> None:
        predictor = _FakePredictor([0.7])
        plans = {"foreground_intensity_properties_per_channel": {"0": A.NORMALIZATION_CONTRACT}}
        context = np.arange(6**3, dtype=np.uint8).reshape(6, 6, 6)
        patches = self.shape_patches()
        with patches[0], patches[1], mock.patch.object(
            I.R, "load_network_for_inference", return_value=(predictor, plans)
        ), mock.patch.object(
            I.R, "normalize_ct", return_value=context.astype(np.float32)
        ), mock.patch.object(I, "_configure_runtime", return_value=A.DETERMINISM_CONTRACT):
            result = I._baseline_probability(Path("unused"), context)
        self.assertEqual(result.shape, (4, 4, 4))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, 0.7, atol=1e-6))

    def test_candidate_uses_exactly_twelve_member_probability_mean(self) -> None:
        probabilities = np.linspace(0.1, 0.9, 12).tolist()
        predictor = _FakePredictor(probabilities)
        context = np.zeros((6, 6, 6), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "model").mkdir()
            (model / "model/plans.json").write_text(json.dumps({
                "foreground_intensity_properties_per_channel": {
                    "0": A.NORMALIZATION_CONTRACT
                }
            }), encoding="utf-8")
            patches = self.shape_patches()
            with patches[0], patches[1], mock.patch.object(
                I.P, "make_predictor", return_value=predictor
            ), mock.patch.object(
                I.R, "normalize_ct", return_value=context.astype(np.float32)
            ), mock.patch.object(I, "_configure_runtime", return_value=A.DETERMINISM_CONTRACT):
                result = I._candidate_probability(model, context)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.allclose(result, np.mean(probabilities), atol=1e-6))

    def test_candidate_rejects_eleven_or_thirteen_members(self) -> None:
        context = np.zeros((6, 6, 6), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "model").mkdir()
            (model / "model/plans.json").write_text(json.dumps({
                "foreground_intensity_properties_per_channel": {
                    "0": A.NORMALIZATION_CONTRACT
                }
            }), encoding="utf-8")
            for count in (11, 13):
                with self.subTest(count=count):
                    predictor = _FakePredictor([0.5] * count)
                    patches = self.shape_patches()
                    with patches[0], patches[1], mock.patch.object(
                        I.P, "make_predictor", return_value=predictor
                    ), mock.patch.object(
                        I.R, "normalize_ct", return_value=context.astype(np.float32)
                    ), mock.patch.object(
                        I, "_configure_runtime", return_value=A.DETERMINISM_CONTRACT
                    ), self.assertRaisesRegex(RuntimeError, "exactly twelve"):
                        I._candidate_probability(model, context)

    def test_runtime_configuration_overrides_ambient_compile_and_cudnn_flags(self) -> None:
        old_compile = os.environ.get("nnUNet_compile")
        old_benchmark = torch.backends.cudnn.benchmark
        old_deterministic = torch.backends.cudnn.deterministic
        old_enabled = torch.backends.cudnn.enabled
        old_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        old_precision = torch.get_float32_matmul_precision()
        old_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        try:
            os.environ["nnUNet_compile"] = "1"
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
                torch.cuda, "set_device"
            ):
                self.assertEqual(I._configure_runtime(), A.DETERMINISM_CONTRACT)
        finally:
            if old_compile is None:
                os.environ.pop("nnUNet_compile", None)
            else:
                os.environ["nnUNet_compile"] = old_compile
            torch.backends.cudnn.benchmark = old_benchmark
            torch.backends.cudnn.deterministic = old_deterministic
            torch.backends.cudnn.enabled = old_enabled
            torch.backends.cuda.matmul.allow_tf32 = old_cuda_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
            torch.set_float32_matmul_precision(old_precision)
            torch.use_deterministic_algorithms(old_deterministic_algorithms)


if __name__ == "__main__":
    unittest.main()
