#!/usr/bin/env python3
"""Build a verified, optimizer-free release of a completed cross-scan run."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import crossscan_finetune as C
import crossscan_highres_review as H
import crossscan_release_publication as U
import predict_crossscan_probability_ensemble as P
import run_crossscan_finetune as R
import verify_physical_label_semantics as V


RELEASE_SCHEMA = "crossscan-model-release-v1"
FINAL_TRAINER = "CrossScanPhysicalTrainer"
INFERENCE_TRAINER = "nnUNetTrainer"


def load_hashed(path: Path) -> dict[str, Any]:
    value = C.load_json(path)
    if value.get("content_sha256") != C.content_hash_without_field(value):
        raise ValueError(f"content hash mismatch: {path}")
    return value


def relative_record(root: Path, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {"path": relative, **R.file_record(path)}


def copy_artifact(source: Path, staging: Path, relative: str) -> dict[str, Any]:
    destination = (staging / relative).resolve()
    destination.relative_to(staging.resolve())
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if R.file_record(destination) != R.file_record(source):
        raise RuntimeError(f"copied artifact changed: {source}")
    return relative_record(staging, destination)


def validate_network_state(state: object) -> dict[str, int]:
    import torch

    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint network_weights is not a nonempty mapping")
    tensors = 0
    elements = 0
    unique_storages: dict[tuple[int, int], int] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("network state must map strings to tensors")
        if not torch.isfinite(value).all():
            raise ValueError(f"network tensor is nonfinite: {key}")
        tensors += 1
        elements += int(value.numel())
        storage = value.untyped_storage()
        unique_storages[(storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return {
        "state_tensor_count": tensors,
        "state_logical_elements": elements,
        "state_unique_storage_bytes": int(sum(unique_storages.values())),
    }


def export_checkpoint(
    source: Path, destination: Path, release_metadata: dict[str, Any]
) -> dict[str, Any]:
    import torch

    if destination.exists():
        raise FileExistsError(destination)
    source_record = R.file_record(source)
    saved = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if saved.get("trainer_name") != FINAL_TRAINER:
        raise ValueError(
            f"source trainer {saved.get('trainer_name')} != {FINAL_TRAINER}"
        )
    init_args = saved.get("init_args")
    if not isinstance(init_args, dict) or init_args.get("configuration") != R.CONFIGURATION:
        raise ValueError("source checkpoint configuration mismatch")
    state = saved.get("network_weights")
    state_summary = validate_network_state(state)
    exported = {
        "network_weights": state,
        "current_epoch": saved.get("current_epoch"),
        "init_args": {"configuration": R.CONFIGURATION},
        "trainer_name": INFERENCE_TRAINER,
        "inference_allowed_mirroring_axes": None,
        "release_metadata": copy.deepcopy(release_metadata),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(exported, temporary)
    temporary.replace(destination)

    checked = torch.load(destination, map_location="cpu", weights_only=False, mmap=True)
    if set(checked) != set(exported):
        raise RuntimeError("exported checkpoint key set changed")
    if checked.get("trainer_name") != INFERENCE_TRAINER:
        raise RuntimeError("exported checkpoint trainer is not standard nnUNetTrainer")
    if checked.get("init_args") != {"configuration": R.CONFIGURATION}:
        raise RuntimeError("exported checkpoint configuration changed")
    if checked.get("release_metadata") != release_metadata:
        raise RuntimeError("exported checkpoint release metadata changed")
    checked_state = checked.get("network_weights")
    if list(checked_state) != list(state):
        raise RuntimeError("exported checkpoint state keys changed")
    for key in state:
        if not torch.equal(state[key], checked_state[key]):
            raise RuntimeError(f"exported network tensor changed: {key}")
    destination_record = R.file_record(destination)
    if destination_record["bytes"] >= source_record["bytes"]:
        raise RuntimeError("optimizer-free checkpoint did not shrink")
    del checked_state, checked, state, saved, exported
    gc.collect()
    return {
        "source_checkpoint": source_record,
        "checkpoint": destination_record,
        "removed_training_state": [
            "optimizer_state", "grad_scaler_state", "logging", "_best_ema"
        ],
        **state_summary,
    }


def validate_standard_nnunet_model(model_root: Path, folds: list[int]) -> None:
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    os.environ.setdefault("nnUNet_compile", "0")
    for fold in folds:
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=torch.device("cpu"),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_root), use_folds=(fold,), checkpoint_name="checkpoint_final.pth"
        )
        if predictor.trainer_name != INFERENCE_TRAINER:
            raise RuntimeError(f"fold {fold} did not load with standard nnUNetTrainer")
        predictor.network.load_state_dict(predictor.list_of_parameters[0], strict=True)
        del predictor
        gc.collect()


def verify_license_provenance(
    plan: dict[str, Any], base_model_card: Path, tooling_license: Path,
) -> None:
    if plan.get("inputs", {}).get("label_license") != P.RELEASE_LICENSES[
        "fine_tuned_checkpoints_and_derived_evidence"
    ]:
        raise ValueError("physical-label license differs from the release contract")
    if not base_model_card.is_file() or "\nlicense: apache-2.0\n" not in (
        "\n" + base_model_card.read_text(encoding="utf-8")
    ):
        raise ValueError("base-model card does not declare license: apache-2.0")
    if not tooling_license.is_file():
        raise ValueError("tooling LICENSE is missing")
    license_text = tooling_license.read_text(encoding="utf-8")
    required_mit_terms = (
        "MIT License",
        "Permission is hereby granted, free of charge",
        'THE SOFTWARE IS PROVIDED "AS IS"',
    )
    if not all(term in license_text for term in required_mit_terms):
        raise ValueError("tooling LICENSE does not contain the MIT license terms")


def model_license_notice(plan: dict[str, Any]) -> str:
    label_license = plan.get("inputs", {}).get("label_license")
    if label_license != P.RELEASE_LICENSES[
        "fine_tuned_checkpoints_and_derived_evidence"
    ]:
        raise ValueError("physical-label license differs from the release contract")
    label_release = plan.get("inputs", {}).get("label_release")
    if not isinstance(label_release, str) or not label_release.startswith("https://"):
        raise ValueError("plan does not identify the physical-label release")
    return f"""# Model and evidence license

The fine-tuned checkpoints and result-derived evidence in this package are released under
**{label_license}**:
<https://creativecommons.org/licenses/by-nc/4.0/>.

They were trained and evaluated with physical-label volumes from:
<{label_release}>. Those labels and their derived measurements inherit CC BY-NC 4.0 from
the underlying Vesuvius Challenge scan data.

The initial `scrollprize/surface_m7_nnunet` checkpoint declares **Apache-2.0**. That base
license and attribution are retained in `evidence/BASE_MODEL_README.md`; it does not remove
the CC BY-NC 4.0 conditions on these fine-tuned weights and derived evidence. The included
release-tooling source files remain **MIT**-licensed; their complete license text is in
`TOOLING_LICENSE.txt`.
"""


def model_card(
    result: dict[str, Any], plan: dict[str, Any], records: list[dict[str, Any]],
    visual_review: dict[str, Any] | None = None,
) -> str:
    primary = result["primary_summary"]
    safety = result["safety_summary"]
    label_release = plan["inputs"]["label_release"]
    review_line = ""
    if visual_review is not None:
        supported = visual_review.get("supported_panel_ids", [])
        review_line = (
            "- Independent-scan human review: "
            f"**{visual_review['release_recommendation']}**; "
            f"{len(supported)} fixed panels authorized for named image-supported wording.\n"
        )
    return f"""---
license: cc-by-nc-4.0
library_name: nnunet
pipeline_tag: image-segmentation
tags:
  - vesuvius-challenge
  - surface-detection
  - nnunet
  - registered-scan-derived-labels
---

# Cross-scan registered-label surface-m7 fine-tunes

This is an optimizer-free inference release generated from the public, outcome-blind
cross-scan experiment. The terminal frozen outcome is **{result['status']}**.

## Frozen evidence

- Primary PHerc1203 cross-fitted AP delta: **{primary['mean']:+.6f}** across six
  training seeds (95% t interval {primary['ci95'][0]:+.6f} to
  {primary['ci95'][1]:+.6f}; two-sided p={primary['two_sided_p']:.6g};
  {primary['positive_seeds']}/6 positive seeds).
- Untouched PHerc0139 safety AP delta: **{safety['mean']:+.6f}** (95% t interval
  {safety['ci95'][0]:+.6f} to {safety['ci95'][1]:+.6f}).
- Result content SHA-256: `{result['content_sha256']}`.
- Models: {len(records)} checkpoints = seeds 40..45 x complementary even/odd
  z-stratum folds. No best seed was selected.
{review_line}

The primary endpoint predicts every PHerc1203 evaluation block only with the fold that
did not train on its z stratum. The safety endpoint is on a different scroll and averages
the two folds within each seed. These are sampled blocks scored against registered
high-resolution-scan-derived reference masks, not a whole-scroll reading result or an
organizer-ground-truth evaluation.

## Inference

The `model/` directory is standard nnU-Net layout with folds 0..11. The exported
checkpoints retain network weights exactly and remove optimizer/gradient-scaler/logging
state. `trainer_name` is mapped from the experiment-local dynamic trainer to the standard
`nnUNetTrainer`; the architecture and weights are unchanged.

nnU-Net's default multi-fold path averages logits. The experiment's visual ensemble uses
an arithmetic mean of probabilities, so use the included runner when that distinction
matters:

```bash
python predict_crossscan_probability_ensemble.py \
  --release-dir . --input-dir INPUT_TIFFS --output-dir OUTPUT --save-probabilities
```

This loads one network on the GPU and applies the 12 parameter sets sequentially. Expect
roughly 5 GB of host RAM for checkpoint parameters plus normal nnU-Net working memory.

For the preregistered PHerc0139-to-ScrollFiesta comparison, use the included
`run_crossscan_scrollfiesta_inference.py` and `CROSSSCAN_SCROLLFIESTA_ADAPTER.md`. That path
rehashes all twelve checkpoints, the fixed public CT context, positive result, decoded-label
semantic audit, execution lock, pinned Python/Torch/CUDA/nnU-Net environment, complete local
import closure, and all three downstream arms; the adapter does not accept a bare caller-labelled
probability array. The model source, pinned Villa worktree, and shared RAW carve remain explicit
external inputs and are reverified rather than trusted from an embedded assertion.

## Intended use and limits

Use as a research surface-probability model at the m7 plans' native spacing. Preserve the
model's CT normalization metadata. Do not call the sampled AP result whole-scroll quality,
ink detection, surface tracing, or proof that a scroll can be read. The complete result,
fixed visual panels, plan, lock, and training receipts are in `evidence/`.
The separate `evidence/highres_review/` pack registers the independently acquired scan
behind every fixed panel and binds the human claim-boundary review. A general model-error
correction claim is not implied by the machine bucket.

## Attribution and license

Fine-tuned from `scrollprize/surface_m7_nnunet`, the nnU-Net component of the first-place
Kaggle Vesuvius surface-detection solution. Its base checkpoint declares Apache-2.0 and
that attribution is retained. The fine-tuned checkpoints and result-derived evidence are
**CC BY-NC 4.0**, matching the physical-label release at <{label_release}>. The included
tooling remains MIT-licensed. See `MODEL_LICENSE.md` and the copied base-model card for the
separate terms, and `TOOLING_LICENSE.txt` for the included tooling's MIT terms; upstream
terms continue to govern their respective inputs.
"""


def technical_report(
    result: dict[str, Any], plan: dict[str, Any], lock: dict[str, Any],
    records: list[dict[str, Any]],
    visual_review: dict[str, Any] | None = None,
) -> str:
    """Render the sealed result without hand-transcribing any quantitative field."""

    seeds = list(C.INFERENTIAL_SEEDS)
    if [row.get("seed") for row in result.get("seed_rows", [])] != seeds:
        raise ValueError("technical report requires the six frozen seed rows in order")
    if set(result.get("comparisons", {})) != {str(seed) for seed in seeds}:
        raise ValueError("technical report requires exactly the six frozen comparisons")
    if len(records) != 2 * len(seeds):
        raise ValueError("technical report requires all twelve release checkpoints")
    visual_fields = ("case_id", "scroll", "z_stratum", "score_slice_l1")
    expected_visuals = [
        tuple(visual[field] for field in visual_fields)
        for visual in lock["resolved_protocol"]["visual_cases"]
    ]
    actual_visuals = [
        tuple(visual[field] for field in visual_fields)
        for visual in result.get("figures", [])
    ]
    if actual_visuals != expected_visuals or len(actual_visuals) != 8:
        raise ValueError("technical report requires the eight locked visual panels in order")
    primary = result["primary_summary"]
    safety = result["safety_summary"]

    seed_rows = []
    for row in result["seed_rows"]:
        seed_rows.append(
            f"| {row['seed']} | {row['primary_initial_ap']:.6f} | "
            f"{row['primary_finetuned_ap']:.6f} | {row['primary_delta']:+.6f} | "
            f"{row['safety_initial_ap']:.6f} | {row['safety_finetuned_ap']:.6f} | "
            f"{row['safety_delta']:+.6f} |"
        )

    subgroup_rows = []
    for stratum in range(C.Z_STRATA):
        primary_values = [
            float(result["comparisons"][str(seed)]["primary"]
                  ["by_z_stratum"][str(stratum)]["average_precision_delta"])
            for seed in seeds
        ]
        safety_values = [
            float(result["comparisons"][str(seed)]["safety"]
                  ["by_z_stratum"][str(stratum)]["average_precision_delta"])
            for seed in seeds
        ]
        subgroup_rows.append(
            f"| z{stratum} | {sum(primary_values) / len(primary_values):+.6f} | "
            f"{sum(value > 0 for value in primary_values)}/6 | "
            f"{sum(safety_values) / len(safety_values):+.6f} | "
            f"{sum(value > 0 for value in safety_values)}/6 |"
        )

    first_primary = result["comparisons"][str(seeds[0])]["primary"]
    difficulty_keys = set(first_primary["by_difficulty_bin"])
    if any(
        set(result["comparisons"][str(seed)]["primary"]["by_difficulty_bin"])
        != difficulty_keys
        for seed in seeds[1:]
    ):
        raise ValueError("physical-difficulty bins differ across frozen seeds")
    difficulty_rows = []
    for key in sorted(first_primary["by_difficulty_bin"], key=int):
        values = [
            float(result["comparisons"][str(seed)]["primary"]
                  ["by_difficulty_bin"][key]["average_precision_delta"])
            for seed in seeds
        ]
        difficulty_rows.append(
            f"| {key} | {sum(values) / len(values):+.6f} | "
            f"{min(values):+.6f} | {max(values):+.6f} | "
            f"{sum(value > 0 for value in values)}/6 |"
        )

    figures = []
    for index, figure in enumerate(result["figures"], 1):
        source_name = Path(figure["file"]["path"]).name
        figures.append(
            f"### Fixed panel {index}: `{figure['case_id']}`\n\n"
            f"Scroll `{figure['scroll']}`, z stratum {figure['z_stratum']}, "
            f"preselected slice {figure['score_slice_l1']}.\n\n"
            f"![Fixed cross-scan panel {index}](evidence/figures/{source_name})"
        )

    fold_lines = []
    for name in ("even", "odd"):
        fold = plan["folds"][name]
        fold_lines.append(
            f"- **{name}:** train z strata {fold['train_z_strata']}; held-out z strata "
            f"{fold['held_out_z_strata']}; {len(fold['train_case_ids'])} training and "
            f"{len(fold['internal_validation_case_ids'])} internal-validation cases."
        )

    review_section = ""
    if visual_review is not None:
        supported = visual_review.get("supported_panel_ids", [])
        supported_text = ", ".join(f"`{panel_id}`" for panel_id in supported) or "none"
        review_section = f"""
## Registered independent-scan review

All eight fixed cases were rendered against the separately acquired scan and reviewed by
`{visual_review['reviewer']}` at `{visual_review['reviewed_utc']}`. The signed recommendation
is **{visual_review['release_recommendation']}**. Fixed panels authorized for narrowly named,
image-supported correction wording: {supported_text}.

The complete panel hashes, source-scan chunk hashes, transform identities, reviewer notes,
and proxy-not-ground-truth acknowledgement are under `evidence/highres_review/`. Cases not
listed above support only the registered-proxy-agreement claim.
"""

    return f"""# Cross-scan registered-label fine-tuning: sealed technical report

Generated from `final_result.json`; no metric in this report is manually entered.

## Result

The frozen terminal bucket is **{result['status']}**. Fine-tuning the released surface-m7
model on external, model-independent PHerc1203 recto reference masks derived from a
separately acquired high-resolution scan changed held-out, cross-fitted pooled average
precision by **{primary['mean']:+.6f}** on average over six
training seeds. The 95% seed-level t interval is **[{primary['ci95'][0]:+.6f},
{primary['ci95'][1]:+.6f}]**, the two-sided p-value is
**{primary['two_sided_p']:.6g}**, and **{primary['positive_seeds']}/6** seed effects are
positive. On the untouched PHerc0139 safety scroll, the mean AP delta is
**{safety['mean']:+.6f}**, with 95% interval **[{safety['ci95'][0]:+.6f},
{safety['ci95'][1]:+.6f}]**.

This is a sampled registered-label-agreement result, not an organizer-ground-truth,
whole-scroll segmentation, or reading claim.

## Contribution

- A direct intervention on a core virtual-unwrapping stage: the released m7 surface model.
- 288 registered-reference-label training/internal-validation crops selected without model output.
- Complementary spatial cross-fitting, a separate seed-39 learnability gate, six frozen
  inferential seeds, 32 held-out PHerc1203 blocks, and 32 untouched PHerc0139 blocks.
- Eight visual panels selected and locked before any model outcome.
- {len(records)} optimizer-free standard nnU-Net checkpoints, with no best-seed selection.

## Prospective provenance

| artifact | content SHA-256 |
|---|---|
| plan | `{plan['content_sha256']}` |
| execution lock | `{lock['content_sha256']}` |
| pilot verdict | `{result['pilot_verdict_content_sha256']}` |
| final result | `{result['content_sha256']}` |

The plan and implementation were public before materialization, training, inference, or
scoring. Selected training length: **{result['selected_steps']} optimizer steps**. Primary
effect gate: **{result['gates']['primary_effect']:+.3f} AP**; minimum positive seeds:
**{result['gates']['minimum_positive_seeds']}/6**; two-sided alpha:
**{result['gates']['alpha_two_sided']}**; untouched-scroll safety mean-delta floor:
**-{result['gates']['safety_noninferiority_margin']:.3f} AP**.

## Spatial cross-fitting

{chr(10).join(fold_lines)}

Every PHerc1203 evaluation block is scored only with the complementary fold that did not
train on its z stratum. PHerc0139 was never used for fine-tuning, model selection, retry
selection, or threshold selection; its two fold predictions are averaged within seed.

## Six-seed results

| seed | primary initial AP | primary fine-tuned AP | primary delta | safety initial AP | safety fine-tuned AP | safety delta |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(seed_rows)}

The seed is the inferential unit. Evaluation voxels and blocks are not treated as
independent replicates.

## Descriptive z-stratum consistency

| stratum | primary mean delta | positive primary seeds | safety mean delta | positive safety seeds |
|---|---:|---:|---:|---:|
{chr(10).join(subgroup_rows)}

These subgroup rows are descriptive; the frozen decision uses the pooled primary endpoint.

## Descriptive physical-difficulty consistency

| difficulty bin | mean primary delta | minimum seed delta | maximum seed delta | positive seeds |
|---:|---:|---:|---:|---:|
{chr(10).join(difficulty_rows)}

Difficulty bins were fixed from physical-label geometry before outcome. They are not
post-hoc model-error bins.

## Fixed visual evidence

Each panel shows CT, the registered scan-derived reference mask, initial m7 probability,
fine-tuned probability,
probability additions, and removals for all six seeds plus their mean. All eight are shown.

{chr(10).join(figures)}

{review_section}

## Reusable release

`model/` contains release folds 0..11 in standard nnU-Net layout, ordered as seeds 40..45
and even then odd within each seed. Export preserves network tensors exactly while removing
optimizer, gradient-scaler, logger, and best-EMA state. The included probability-ensemble
runner avoids silently substituting nnU-Net's usual logit averaging for the experiment's
probability-space visual ensemble.

## Limitations

- Evaluation covers 64 fixed 64-cubed L1 score blocks, not either complete scroll.
- The primary endpoint is agreement with an automated scan-derived recto reference mask
  with supervised negatives; it is not official or human ground truth and does not
  measure ink detection, mesh topology, surface ordering, or readable text.
- PHerc1203 primary estimates are cross-fitted within one scroll. PHerc0139 is the separate
  safety scroll, not a claim of universal cross-scroll generalization.
- Seed-level uncertainty measures optimization variability under this recipe, not the full
  uncertainty over all papyri, scanners, or physical-label constructions.
- The released twelve-model probability ensemble is computationally expensive; no smaller
  model was selected after inspecting results.

## Reproduction

See `evidence/` for the sealed plan, lock, pilot, final result, training receipts, and every
fixed panel. `release_manifest.json` binds every copied artifact and checkpoint by SHA-256.
The public training/materialization/scoring implementation remains the authoritative source
for exact commands and environment identity.

## License

The fine-tuned checkpoints and result-derived evidence are **CC BY-NC 4.0**, matching the
physical-label release used by the frozen plan. The base m7 checkpoint declares
**Apache-2.0**, and the included release tooling remains **MIT**-licensed. See
`MODEL_LICENSE.md`, `evidence/BASE_MODEL_README.md`, and `TOOLING_LICENSE.txt` for the
separated provenance and complete tooling terms.
"""


def verify_release_files(root: Path, manifest: dict[str, Any]) -> None:
    records = []
    for model in manifest["models"]:
        records.append(model["checkpoint"])
    records.extend(manifest["artifacts"])
    records.extend(manifest["tooling"])
    records.extend(manifest.get("reports", []))
    records.extend((manifest["model_files"][name] for name in sorted(manifest["model_files"])))
    for record in records:
        path = R.resolve_data_path(root, record["path"])
        if R.file_record(path) != {
            "bytes": record["bytes"], "sha256": record["sha256"]
        }:
            raise RuntimeError(f"release file differs from manifest: {path}")


def build_release(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    data = args.data_root.resolve()
    model_dir = args.model_dir.resolve()
    out = args.out.resolve()
    staging = out.with_name(out.name + ".tmp")
    if out.exists() or staging.exists():
        raise FileExistsError(f"release output already exists: {out} or {staging}")

    lock, plan = R.verify_runtime(
        repo,
        repo / "results/crossscan_finetune/execution_lock.json",
        repo / "results/crossscan_finetune/plan.json",
        args.villa_root.resolve(),
        args.labels_root.resolve(),
        repo / "results/physical_normalization_ab/manifest.json",
        model_dir,
    )
    preprocessing = R.verify_preprocessed(plan, lock, data)
    verdict = R.require_any_pilot_authorization(
        data, plan["content_sha256"], lock["content_sha256"]
    )
    result = load_hashed(data / "final_result.json")
    semantic_audit, semantic_audit_payload = V.validate_audit_receipt(
        args.semantic_audit.resolve()
    )
    review_root = args.highres_review_root.resolve()
    review_pack = H.load_review_pack(review_root)
    visual_review, visual_review_payload = H.validate_human_review(
        review_root, args.highres_review_receipt.resolve()
    )
    base_model_card = model_dir / "README.md"
    tooling_license = repo / "LICENSE"
    verify_license_provenance(plan, base_model_card, tooling_license)
    expected_result = {
        "schema_version": "crossscan-final-result-v1",
        "status": "POSITIVE_DEPLOYABLE",
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "pilot_verdict_content_sha256": verdict["content_sha256"],
        "selected_steps": verdict["selected_steps"],
    }
    mismatches = [
        key for key, value in expected_result.items() if result.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "release is allowed only for the frozen POSITIVE_DEPLOYABLE result; "
            f"mismatches={mismatches}"
        )
    review_mismatches = [
        key for key, value in {
            "plan_content_sha256": plan["content_sha256"],
            "execution_lock_content_sha256": lock["content_sha256"],
            "final_result_content_sha256": result["content_sha256"],
            "selected_steps": result["selected_steps"],
        }.items() if review_pack.get(key) != value
    ]
    if review_mismatches:
        raise ValueError(
            "high-resolution review pack does not bind this release; "
            f"mismatches={review_mismatches}"
        )
    if visual_review["release_recommendation"] == "DO_NOT_RELEASE":
        raise ValueError("human independent-scan review blocks public release")

    staging.mkdir(parents=True)
    model_root = staging / "model"
    model_root.mkdir()
    preprocessed = data / "nnUNet_preprocessed" / R.DATASET_NAME
    plans_source = preprocessed / f"{R.PLANS_IDENTIFIER}.json"
    dataset_source = preprocessed / "dataset.json"
    shutil.copy2(plans_source, model_root / "plans.json")
    shutil.copy2(dataset_source, model_root / "dataset.json")

    records = []
    release_fold = 0
    for seed in C.INFERENTIAL_SEEDS:
        for training_fold in ("even", "odd"):
            receipt, checkpoint = R.load_training_receipt(
                plan, lock, data, seed, int(result["selected_steps"]), training_fold
            )
            destination = (
                model_root / f"fold_{release_fold}" / "checkpoint_final.pth"
            )
            metadata = {
                "schema_version": "crossscan-inference-checkpoint-v1",
                "seed": seed,
                "training_fold": training_fold,
                "release_fold": release_fold,
                "source_checkpoint_sha256": receipt["checkpoint"]["sha256"],
                "final_result_content_sha256": result["content_sha256"],
                "execution_lock_content_sha256": lock["content_sha256"],
            }
            exported = export_checkpoint(checkpoint, destination, metadata)
            records.append({
                "seed": seed,
                "training_fold": training_fold,
                "release_fold": release_fold,
                "training_receipt_content_sha256": receipt["content_sha256"],
                **exported,
                "checkpoint": relative_record(staging, destination),
            })
            release_fold += 1

    evidence = []
    for source, relative in (
        (data / "final_result.json", "evidence/final_result.json"),
        (data / "pilot_verdict.json", "evidence/pilot_verdict.json"),
        (data / "preprocessing_receipt.json", "evidence/preprocessing_receipt.json"),
        (repo / "results/crossscan_finetune/plan.json", "evidence/plan.json"),
        (repo / "results/crossscan_finetune/execution_lock.json", "evidence/execution_lock.json"),
        (base_model_card, "evidence/BASE_MODEL_README.md"),
        (args.semantic_audit.resolve(), "evidence/physical_label_semantic_audit.json"),
        (review_root / "review_pack.json", "evidence/highres_review/review_pack.json"),
        (args.highres_review_receipt.resolve(), "evidence/highres_review/human_review.json"),
    ):
        evidence.append(copy_artifact(source, staging, relative))
    for attempt in sorted(data.glob("pilot_attempt_steps-*.json")):
        load_hashed(attempt)
        evidence.append(copy_artifact(
            attempt, staging, f"evidence/{attempt.name}"
        ))
    for seed in C.INFERENTIAL_SEEDS:
        for fold in ("even", "odd"):
            name = f"seed-{seed}-{fold}-steps-{result['selected_steps']}.json"
            evidence.append(copy_artifact(
                data / "training_receipts" / name,
                staging,
                f"evidence/training_receipts/{name}",
            ))
    for figure in result["figures"]:
        source = R.resolve_data_path(data, figure["file"]["path"])
        if R.file_record(source) != {
            "bytes": figure["file"]["bytes"],
            "sha256": figure["file"]["sha256"],
        }:
            raise ValueError(f"final-result figure hash mismatch: {source}")
        evidence.append(copy_artifact(
            source, staging, f"evidence/figures/{source.name}"
        ))
    for case in review_pack["cases"]:
        source = R.resolve_data_path(review_root, case["panel"]["path"])
        if R.file_record(source) != {
            "bytes": case["panel"]["bytes"],
            "sha256": case["panel"]["sha256"],
        }:
            raise ValueError(f"high-resolution review panel hash mismatch: {source}")
        evidence.append(copy_artifact(
            source, staging, f"evidence/highres_review/panels/{source.name}"
        ))

    tooling = []
    for name in (
        "export_crossscan_release.py",
        "crossscan_release_publication.py",
        "test_crossscan_release_publication.py",
        "predict_crossscan_probability_ensemble.py",
        "crossscan_scrollfiesta_adapter.py",
        "run_crossscan_scrollfiesta_inference.py",
        "run_crossscan_scrollfiesta_downstream.py",
        "crossscan_scrollfiesta_metrics.py",
        "crossscan_scrollfiesta_obj.py",
        "run_crossscan_finetune.py",
        "crossscan_finetune.py",
        "score_crossscan_finetune.py",
        "crossscan_highres_review.py",
        "physical_normalization_ab.py",
        "verify_physical_label_semantics.py",
        "crossscan_scrollfiesta_downstream_lock.json",
        "crossscan_scrollfiesta_metric_lock.json",
        "CROSSSCAN_SCROLLFIESTA_DOWNSTREAM_PREREG.md",
        "CROSSSCAN_SCROLLFIESTA_ADAPTER.md",
        "CROSSSCAN_FINETUNE_PREREG.md",
        "CROSSSCAN_FINETUNE_AMENDMENT_08.md",
        "CROSSSCAN_HIGHRES_REVIEW.md",
        "CROSSSCAN_RELEASE_PUBLICATION.md",
    ):
        tooling.append(copy_artifact(repo / name, staging, name))
    tooling.append(copy_artifact(
        tooling_license, staging, "TOOLING_LICENSE.txt"
    ))

    card = staging / "README.md"
    card.write_text(
        model_card(result, plan, records, visual_review), encoding="utf-8"
    )
    report = staging / "TECHNICAL_REPORT.md"
    report.write_text(
        technical_report(result, plan, lock, records, visual_review), encoding="utf-8"
    )
    license_notice = staging / "MODEL_LICENSE.md"
    license_notice.write_text(model_license_notice(plan), encoding="utf-8")
    reports = [
        relative_record(staging, card),
        relative_record(staging, report),
        relative_record(staging, license_notice),
    ]

    model_files = {
        "plans": relative_record(staging, model_root / "plans.json"),
        "dataset": relative_record(staging, model_root / "dataset.json"),
    }
    manifest = {
        "schema_version": RELEASE_SCHEMA,
        "status": "PASS",
        "created_utc": R.utc_now(),
        "plan_content_sha256": plan["content_sha256"],
        "execution_lock_content_sha256": lock["content_sha256"],
        "preprocessing_receipt_content_sha256": preprocessing["content_sha256"],
        "pilot_verdict_content_sha256": verdict["content_sha256"],
        "final_result_content_sha256": result["content_sha256"],
        "semantic_audit_content_sha256": semantic_audit["content_sha256"],
        "semantic_audit_file_sha256": C.sha256_bytes(semantic_audit_payload),
        "highres_review_pack_content_sha256": review_pack["content_sha256"],
        "highres_review_receipt_content_sha256": visual_review["content_sha256"],
        "highres_review_receipt_file_sha256": C.sha256_bytes(visual_review_payload),
        "highres_review_recommendation": visual_review["release_recommendation"],
        "highres_review_supported_panel_ids": visual_review["supported_panel_ids"],
        "outcome": result["status"],
        "selected_steps": result["selected_steps"],
        "base_model": plan["inputs"]["model"],
        "licenses": P.RELEASE_LICENSES,
        "ensemble": {
            "aggregation": "arithmetic mean of class probabilities",
            "fold_count": 12,
            "mirroring": False,
        },
        "models": records,
        "model_files": model_files,
        "artifacts": evidence,
        "tooling": tooling,
        "reports": reports,
    }
    manifest["content_sha256"] = C.content_hash_without_field(manifest)
    R.atomic_write_json(staging / "release_manifest.json", manifest)
    P.validate_release_manifest_value(manifest)
    verify_release_files(staging, manifest)
    validate_standard_nnunet_model(model_root, list(range(12)))
    U.write_release_checksums(staging)
    staging.replace(out)
    verified = load_hashed(out / "release_manifest.json")
    P.validate_release_manifest_value(verified)
    verify_release_files(out, verified)
    U.validate_release(out)
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--villa-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--highres-review-root", type=Path, required=True)
    parser.add_argument("--highres-review-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(C.canonical_json(build_release(args)))


if __name__ == "__main__":
    main()
