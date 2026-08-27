# PHercParis4 family-balanced patch sampler screen

Frozen before either fit was launched.

- Public input snapshot: `https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/`
- Fit window: z `[10500, 11500)`
- Seed: `17`
- Steps: `5000`
- Baseline: current area^0.5 verified-patch sampler
- Treatment: same sampler, retaining every patch, with aggregate draw mass of IDs matching `^band-seed` capped at `0.25`
- All non-patch annotation and dense-volume inputs are disabled in both arms. This is a focused screening experiment, not a production-quality full-input fit.

Primary metric: point-weighted satisfaction on the newly published `eval_fibers/` collection, which is not used for training by either arm. Report the paired per-fiber delta and bootstrap 95% confidence interval. This is held-out geometric consistency, not human ground truth: the fibers share upstream model provenance.

Secondary metrics: within-family satisfied area for `band-seed` and non-`band-seed` verified patches, plus median per-patch satisfaction. These are in-sample diagnostics.

Success gate: treatment improves held-out point-weighted fiber satisfaction by at least 2 absolute percentage points, with a paired-fiber bootstrap 95% confidence interval above zero, while `band-seed` satisfied-area fraction falls by no more than 1 point. If the gate passes, repeat with three seeds and/or a production-length run. If it fails, do not claim fit-quality improvement.
