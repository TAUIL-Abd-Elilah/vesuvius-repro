# Cross-scan fine-tuning: pre-outcome implementation amendment 07

Status: **discovered by a static scorer/watcher audit during the first v3 seed-39 pilot training
fold, before that fold completed and before any fine-tuned prediction, pilot attempt, pilot
verdict, physical-truth score, primary inference, final result, or release existed**.

The v3 execution used lock content SHA-256
`20b13277ec289d1f4726be4e3cb52327815dd4c7853c2884bdb7fee4304ea247` (whole-file SHA-256
`eb98ca2b07f4c9c238f3a6826ed12cbf918cbba6c2ff8d977c30934866df4c16`). It completed
materialization, preprocessing, the independent byte-equivalence audit, initial-model pilot
inference, and the synthetic-only memory gate. No prediction value or efficacy metric was
inspected. During seed 39's first 2,000-step fold, a static review found the mismatch below. The
exact training process was stopped at its epoch-37 marker, before checkpoint completion or pilot
scoring. All five v3 launchers then exited fail-closed. The partial v3 root is retained; it contains
no `pilot_attempt_steps-*.json`, `pilot_verdict.json`, `final_result.json`, or release root.

The superseded lock is retained byte-for-byte at
`results/crossscan_finetune/execution_lock.superseded-20260811-preverdict-step-selection-v3.json`.

## Align implementation with the already-frozen preregistration

The public preregistration has always stated: “A pass fixes 4,000 steps for every inferential
run.” The scorer instead wrote the passing pilot attempt's own duration into `selected_steps`.
Consequently, a pass on the first 2,000-step attempt would have authorized six 2,000-step
inferential seeds, contrary to the preregistration. The runbook repeated that implementation
mistake. No such verdict was created.

The corrected scorer keeps the frozen pilot logic unchanged: seed 39 is evaluated first at 2,000
steps; only a failure permits the single 4,000-step retry. A pass at either permitted pilot attempt
now authorizes exactly 4,000 steps for every inferential seed. Both runtime authorization paths
independently reject any passing verdict whose `selected_steps` is not 4,000. The machine plan now
records `inferential_steps=4000`, the runbook matches the preregistration, and regression tests
exercise the previously missing 2,000-step-PASS-to-4,000-step authorization case and rejection of
a self-hashed but invalid 2,000-step inferential verdict.

This correction changes no case, label mapping, fold, seed, pilot attempt duration, retry rule,
optimizer value, checkpoint, augmentation, normalization, endpoint, threshold, statistical test,
safety gate, or visual case. It repairs only the inferential duration selected after a passing
pilot and is locked before any pilot outcome is generated or inspected.
