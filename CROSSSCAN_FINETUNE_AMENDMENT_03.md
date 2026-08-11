# Cross-scan fine-tuning: pre-training implementation amendment 03

Status: **discovered after materialization and preprocessing, but before any model training,
pilot prediction, experiment prediction, score, or outcome inspection**.

The public execution lock with content SHA-256
`9a685b5c2842c9ccb34699c6254c5da1ff834824d51bdbbb515483812e51f746` completed all 384
case materializations and preprocessing of all 288 training/validation cases. Its preprocessing
receipt has content SHA-256
`d6dcb8ac44a89f9561a7daae411d4f5a36ce4bed88775ece0697e83698dffdc1`.
The subsequent synthetic-only GPU memory gate stopped while constructing AdamW, before its first
synthetic optimization step. It produced no PASS receipt. The data root contained no training
directory, training receipt, checkpoint, prediction, score, pilot result, or final result when
this amendment was written.

The affected lock is retained at
`results/crossscan_finetune/execution_lock.superseded-20260811-pretraining.json`. The completed
data root is retained as failed pre-training provenance and is not rebound to the replacement
lock. A fresh data root must materialize and preprocess under the replacement lock before any
training may begin.

## Correct the JSON-to-PyYAML numeric boundary

The runtime writes the preregistered AdamW weight decay `0.00001` as JSON scientific notation
`1e-05`. Villa's bundled `nnUNetTrainer` reads the JSON file through `yaml.safe_load`; in the
locked PyYAML environment that token resolves to the string `"1e-05"`. AdamW rejects the string
while validating `weight_decay`, so the trainer cannot initialize.

Immediately after the trainer reads its configuration and before `trainer.initialize()`, the
runtime now converts `initial_lr` and `weight_decay` to floats only after rejecting booleans,
non-numeric values, non-finite values, and any value not exactly equal to the preregistered
`0.0001` and `0.00001`. This is a representation repair, not a hyperparameter adaptation. The
validated numeric values are recorded in every training receipt.

A public synthetic GPU memory smoke uses the same helper and the same released m7 architecture,
checkpoint, 192-cubed patch, batch size one, AdamW optimizer, and deep-supervision loss path. Its
receipt binds the unchanged script, checkpoint, execution lock, plan, plans, dataset metadata,
GPU memory measurements, output shapes, and finite synthetic loss. It does not load a physical
label tensor or create an evaluation prediction.

The replacement lock binds this amendment, the corrected runtime, the public memory smoke, and a
regression test that traverses the actual JSON-write/PyYAML-read boundary. This correction changes
no label mapping, case, fold, seed, duration, optimizer value, augmentation, normalization,
endpoint, effect threshold, statistical test, safety gate, visual case, or permitted pilot
adaptation.
