# Cross-scan fine-tuning: pre-outcome implementation amendment 02

Status: **discovered before real dataset materialization, preprocessing, training, or any
experiment prediction**.

The first public execution lock omitted `dataset_fingerprint.json`. Villa's nnU-Net preprocessor
does not require that file, so the omission escaped the preprocessing-path tests, but
`nnUNetTrainer.on_train_start()` copies it unconditionally. Training under the first lock would
therefore have stopped before its first optimization step. The waiting handoff was stopped before
it created the cross-scan data root. No experiment outcome, model, or prediction existed when this
amendment was written.

The affected lock has content SHA-256
`b1a3cf2fba4c354bd5fa99297d2f6cf206f3c9dff6bba7544d0233a7d053a7b1`. It is retained as a
superseded provenance artifact. A replacement lock at the canonical `execution_lock.json` path
binds this amendment and the corrected implementation before execution resumes.

## Correct fingerprint provenance

Preprocessing now runs villa's standard `DatasetFingerprintExtractor` on all 288 materialized
training/validation cases. It uses nnU-Net's fixed per-case sampling seed 1234 and its standard
100,000,000-voxel total foreground sampling budget. A content-hashed intent is written first, so
an interrupted extraction can only resume with the same plan, execution lock, training-receipt
aggregate, sampling parameters, and process count. The completed fingerprint and its summary are
then bound by a content-hashed receipt. Preprocessing v2 binds that receipt and hashes the complete
preprocessed tree; every training run re-verifies the chain.

The fingerprint describes the new physical-truth dataset. It does **not** trigger replanning.
Architecture, patch size, spacing, CT normalization parameters, and all other plans remain the
frozen released surface-m7 values. Thus the new intensity statistics are provenance metadata, not
a post hoc normalization change.

## Real integration gate

Before the replacement execution lock is created, `crossscan_preprocess_smoke.py` must execute the
official fingerprint extractor and official default preprocessor on a deterministic synthetic
case using the real released m7 plans. The gate also performs the exact fingerprint file copy that
blocked trainer startup. Its content-hashed public result is an execution-locked artifact.

This correction changes no physical label mapping, case, fold, seed, training duration, model
hyperparameter, augmentation, endpoint, statistical test, effect threshold, safety gate, visual
case, or permitted pilot adaptation.
