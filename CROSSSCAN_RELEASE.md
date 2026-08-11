# Cross-scan model release path

Status: **public pre-outcome release tooling**. No pilot, final model result, or
fine-tuned weight is claimed here. The exporter refuses to run unless the frozen terminal
result is exactly `POSITIVE_DEPLOYABLE` and matches the public plan, execution lock, pilot
authorization, preprocessing receipt, twelve training receipts, and checkpoint hashes.

## What this adds

`export_crossscan_release.py` turns the six seeds x two complementary folds into one
standard nnU-Net model folder with folds 0..11. It:

- verifies the locked runtime and every source checkpoint before export;
- preserves every network tensor exactly;
- removes optimizer, gradient-scaler, logger, and best-EMA state;
- maps the experiment-local dynamic trainer name to standard `nnUNetTrainer` for inference;
- copies the result, fixed figures, plan, lock, pilot record, and training receipts;
- derives a technical report directly from the sealed result, including every seed,
  fixed subgroup, and preselected visual panel, with no hand-copied metrics;
- writes a content-hashed manifest, a CC BY-NC 4.0 model card and explicit
  license notice, while preserving the base model's separate Apache-2.0 attribution; and
- reloads every exported fold through nnU-Net's official model-folder API with a strict
  state-dict load before sealing the release.

There is no best-seed selection. The release order is fixed as seeds 40 through 45, with
the even fold followed by the odd fold for each seed.

`predict_crossscan_probability_ensemble.py` supplies the inference distinction that the
standard nnU-Net multi-fold path does not: it averages **class probabilities**, whereas
nnU-Net normally averages logits. The runner loads one network on the GPU and applies the
twelve parameter sets sequentially. Its default integrity check hashes every checkpoint
against the release manifest before inference.

## Real compatibility smoke

The release layout was exercised before any cross-scan outcome using the frozen public m7
base checkpoint:

- source SHA-256:
  `17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e`;
- source size: 820,473,701 bytes;
- optimizer-free export size: 409,644,269 bytes; and
- official `nnUNetPredictor.initialize_from_trained_model_folder`, standard
  `nnUNetTrainer` discovery, and strict loading of every network tensor: **PASS**.

This proves format compatibility and storage reduction, not model efficacy. The fine-tuned
checkpoints do not exist yet.

## Verification

```bash
python -m unittest -v \
  test_crossscan_release.py \
  test_crossscan_finetune.py \
  test_run_crossscan_finetune.py \
  test_score_crossscan_finetune.py
```

The current focused suite has 50 tests. It includes a counterexample proving that the probability
ensemble is not silently equivalent to logit averaging and an export round trip proving
that training state is removed while aliased network tensors remain identical. It also
checks that the generated report contains all six seed rows, all fixed strata, and all
eight preselected panels, refuses a reordered seed result, and fails if the label,
base-model, tooling, or release-manifest license contract drifts.

After a qualifying final result, build the local release with:

```bash
python export_crossscan_release.py \
  --villa-root /path/to/villa \
  --labels-root /path/to/physical_truth \
  --model-dir /path/to/surface_m7_nnunet \
  --data-root /path/to/crossscan_finetune_v1 \
  --out /path/to/crossscan_release
```

Then run the probability ensemble on nnU-Net-formatted inputs:

```bash
python predict_crossscan_probability_ensemble.py \
  --release-dir /path/to/crossscan_release \
  --input-dir /path/to/input_tiffs \
  --output-dir /path/to/output \
  --save-probabilities
```

## Scope

The prospective primary endpoint is PHerc1203 cross-fitted block evaluation; the safety
endpoint is an untouched PHerc0139 scroll. Neither is a whole-scroll reading claim. If the
pilot stops, or the final bucket is null, inconclusive, regressive, or safety-regressive,
this exporter fails closed rather than creating a deployable model card.
