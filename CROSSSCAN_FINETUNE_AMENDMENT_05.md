# Cross-scan fine-tuning: pre-execution implementation amendment 05

Status: **discovered after the hardened synthetic GPU preflight, before creating a v3 data root,
materializing a v3 case, training, predicting, scoring, or inspecting any outcome**.

The hardened execution lock with content SHA-256
`65549866725f03ea74647a1f23248d70888ccec90ebaba3211ec39857fb5ae5e` passed public-runtime
verification, all 58 local tests, and the real synthetic-only 192-cubed GPU preflight. The
preflight loaded the exact frozen checkpoint through the restricted safe-global policy, built
AdamW with the frozen numeric values, completed one synthetic optimization step with finite loss,
and produced no physical-label tensor or evaluation prediction.

That lock was not used to create a data root. It is retained byte-for-byte at
`results/crossscan_finetune/execution_lock.withdrawn-20260811-release-attributes.json`.

## Keep the execution lock compatible with the positive-only release branch

The unexecuted lock included `.gitattributes` among its exact implementation files. The execution
branch's attributes correctly force the two locked smoke scripts to LF, but the positive-only
release branch intentionally adds further LF rules for its exporter, ensemble helper, tests, and
workflow. Binding the entire attributes file would therefore make the release branch fail runtime
verification even when every experiment implementation byte matched the lock.

The next lock omits `.gitattributes` from its implementation-file map. The public attributes file
still forces both smoke scripts to LF, and each smoke script remains individually bound by exact
byte length and SHA-256. This removes only a branch-composition false rejection; it weakens no
runtime, input, model, receipt, or outcome check. The retained withdrawn lock proves the change
occurred before any v3 materialization or outcome-generating execution.

This compatibility correction changes no label mapping, case, fold, seed, duration, optimizer
value, checkpoint policy, augmentation, normalization, endpoint, effect threshold, statistical
test, safety gate, visual case, or permitted pilot adaptation.
