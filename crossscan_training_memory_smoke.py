#!/usr/bin/env python3
"""Run one synthetic ResEnc-L AdamW step to prove the locked recipe fits this GPU."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_DEFAULT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DEFAULT))

import crossscan_finetune as C
import run_crossscan_finetune as R


MINIMUM_FREE_GPU_BYTES = 20 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    villa_root = args.villa_root.resolve()
    labels_root = args.labels_root.resolve()
    model_dir = args.model_dir.resolve()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite memory-smoke result: {out}")

    lock, plan = R.verify_runtime(
        repo,
        repo / "results/crossscan_finetune/execution_lock.json",
        repo / "results/crossscan_finetune/plan.json",
        villa_root,
        labels_root,
        repo / "results/physical_normalization_ab/manifest.json",
        model_dir,
    )

    if "nnunetv2.training.nnUNetTrainer.nnUNetTrainer" in sys.modules:
        raise RuntimeError("nnUNetTrainer imported before isolated paths were configured")

    with tempfile.TemporaryDirectory(prefix="crossscan-training-memory-smoke-") as tmp:
        root = Path(tmp).resolve()
        raw = root / "nnUNet_raw"
        preprocessed = root / "nnUNet_preprocessed"
        results = root / "nnUNet_results"
        for path in (raw, preprocessed, results):
            path.mkdir(parents=True)
        os.environ["nnUNet_raw"] = str(raw)
        os.environ["nnUNet_preprocessed"] = str(preprocessed)
        os.environ["nnUNet_results"] = str(results)
        R.set_deterministic_seed(C.PILOT_SEED)

        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        free_before, total = torch.cuda.mem_get_info()
        if free_before < MINIMUM_FREE_GPU_BYTES:
            raise RuntimeError(
                f"only {free_before / 1024**3:.2f} GiB GPU memory free before smoke"
            )

        class CrossScanPhysicalTrainer(nnUNetTrainer):
            def configure_optimizers(self):
                optimizer = torch.optim.AdamW(
                    self.network.parameters(),
                    lr=self.initial_lr,
                    weight_decay=self.weight_decay,
                )
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=max(self.num_epochs - 1, 1),
                    eta_min=R.MINIMUM_LR,
                )
                return optimizer, scheduler

        CrossScanPhysicalTrainer.__name__ = R.TRAINER_NAME
        plans = C.load_json(model_dir / "plans.json")
        plans["dataset_name"] = R.DATASET_NAME
        plans["plans_name"] = R.PLANS_IDENTIFIER
        plans["image_reader_writer"] = "Tiff3DIO"
        plans["configurations"][R.CONFIGURATION]["batch_size"] = 1
        # Match write_dataset_configuration exactly. The source model uses the
        # same labels, but its reader and case count describe its old dataset.
        dataset = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "surface": 1, "ignore": 2},
            "numTraining": len(R.training_cases(plan)),
            "file_ending": ".tif",
            "overwrite_image_reader_writer": "Tiff3DIO",
        }
        config_path = root / "training_config.json"
        C.write_json(config_path, {
            "project_name": "crossscan-training-memory-smoke",
            "dataset": R.DATASET_NAME,
            "wandb_enabled": 0,
            "num_epochs": C.PILOT_STEPS // R.ITERATIONS_PER_EPOCH,
            "initial_lr": R.INITIAL_LR,
            "weight_decay": R.WEIGHT_DECAY,
            "num_iterations_per_epoch": R.ITERATIONS_PER_EPOCH,
            "num_val_iterations_per_epoch": R.VALIDATION_ITERATIONS_PER_EPOCH,
            "oversample_foreground_percent": 0.5,
            "enable_deep_supervision": True,
        })
        trainer = CrossScanPhysicalTrainer(
            plans=plans,
            configuration=R.CONFIGURATION,
            fold=0,
            dataset_json=dataset,
            unpack_dataset=False,
            device=torch.device("cuda"),
            yaml_config_path=str(config_path),
        )
        trainer_hyperparameters = R.coerce_locked_training_hyperparameters(trainer)
        trainer.initialize()
        checkpoint = model_dir / "fold_0" / "checkpoint_best.pth"
        checkpoint_load = R.load_frozen_pretrained_weights(
            trainer.network,
            checkpoint,
            plan["inputs"]["model"]["fold_0/checkpoint_best.pth"],
            verbose=True,
        )

        patch = tuple(int(v) for v in trainer.configuration_manager.patch_size)
        if patch != (192, 192, 192) or trainer.batch_size != 1:
            raise RuntimeError(f"unexpected smoke geometry: patch={patch}, batch={trainer.batch_size}")
        probe = torch.zeros((1, 1, *patch), dtype=torch.float32, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", enabled=True):
            probe_outputs = trainer.network(probe)
        if not isinstance(probe_outputs, (list, tuple)):
            raise RuntimeError("deep-supervision network did not return multiple outputs")
        output_shapes = [list(value.shape) for value in probe_outputs]
        del probe_outputs, probe
        torch.cuda.empty_cache()

        targets = []
        for shape in output_shapes:
            target = torch.zeros((shape[0], 1, *shape[2:]), dtype=torch.int64)
            z = max(0, shape[2] // 2)
            target[:, :, z:z + 1, :, :] = 1
            targets.append(target)
        data = torch.randn((1, 1, *patch), dtype=torch.float32)

        torch.cuda.reset_peak_memory_stats()
        free_at_step, _ = torch.cuda.mem_get_info()
        result = trainer.train_step({"data": data, "target": targets})
        torch.cuda.synchronize()
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        free_after, _ = torch.cuda.mem_get_info()
        loss = float(result["loss"])
        if not math.isfinite(loss):
            raise RuntimeError(f"synthetic training loss is nonfinite: {loss}")

        receipt = R._with_content_hash({
            "schema_version": "crossscan-training-memory-smoke-v1",
            "status": "PASS",
            "created_utc": R.utc_now(),
            "synthetic_only": True,
            "external_input_hashes_verified": True,
            "physical_label_tensors_loaded": False,
            "evaluation_predictions_created": False,
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
            "implementation_commit": lock["implementation"]["commit"],
            "smoke_script": R.file_record(Path(__file__).resolve()),
            "model_checkpoint": R.file_record(checkpoint),
            "checkpoint_load": checkpoint_load,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_bytes": int(total),
            "gpu_free_before_bytes": int(free_before),
            "gpu_free_at_step_bytes": int(free_at_step),
            "gpu_free_after_bytes": int(free_after),
            "peak_allocated_bytes": int(peak_allocated),
            "peak_reserved_bytes": int(peak_reserved),
            "parameter_count": int(sum(p.numel() for p in trainer.network.parameters())),
            "plans_sha256": C.sha256_bytes(C.canonical_json(plans).encode("ascii")),
            "dataset_json_sha256": C.sha256_bytes(
                C.canonical_json(dataset).encode("ascii")
            ),
            "patch_size": list(patch),
            "batch_size": int(trainer.batch_size),
            "optimizer": "AdamW",
            "trainer_hyperparameters": trainer_hyperparameters,
            "deep_supervision_output_shapes": output_shapes,
            "synthetic_loss": loss,
            "environment": R.runtime_environment(villa_root),
        })
        C.write_json(out, receipt)
        print(C.canonical_json(receipt))


if __name__ == "__main__":
    main()
