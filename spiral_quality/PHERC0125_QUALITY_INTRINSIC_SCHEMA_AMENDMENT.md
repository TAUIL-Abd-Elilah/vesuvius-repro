# PHerc0125 paired-quality intrinsic-schema amendment

Frozen 2026-08-10 01:23 Africa/Casablanca after the paired-quality evaluator
completed its CT work and stopped while adapting the pinned intrinsic checker's
return dictionary, and before any quality report, stored raw CT sample, CT
statistic, bootstrap interval, intrinsic result, quality decision, PHerc0211
dataset, or public outcome existed.

## Observed interface fact

The pinned `spiralcheck` source at commit
`d1b50e2957409a870225fb9f5dcc5e25f7a0f9da` defines
`IntrinsicReport.to_dict()` with `n_bins_checked`, `n_inflated`,
`n_violations`, `violated_bin_fraction`, `n_collapsed`, and
`collapsed_bin_fraction`, but no `inflated_bin_fraction`. The already frozen
quality evaluator attempted to read that absent convenience field and stopped
with `KeyError: 'inflated_bin_fraction'`. It wrote no evaluation report.

## Frozen schema-normalization rule

For each baseline and final intrinsic dictionary returned by that exact pinned
checker:

1. Require `n_bins_checked` to be an integer greater than zero.
2. Require each of `n_violations`, `n_collapsed`, and `n_inflated` to be an
   integer in the closed interval `[0, n_bins_checked]`.
3. Require each checker-provided `violated_bin_fraction` and
   `collapsed_bin_fraction` to be finite and equal, within the evaluator's
   existing floating-point tolerance, to its corresponding count divided by
   `n_bins_checked`.
4. Derive exactly
   `inflated_bin_fraction = n_inflated / n_bins_checked`, require it to be
   finite and in `[0, 1]`, and add it to the evaluator's normalized copy of
   the intrinsic dictionary.
5. Use only these normalized copies in the already frozen baseline-versus-final
   intrinsic comparison, report, and verifier.

This repairs only the adapter between the pinned checker and the evaluator.
It changes no fit or geometry artifact, angular alignment, CT sample or
statistic, bootstrap, intrinsic-checker inputs, threshold, decision rule,
claim boundary, downstream authorization rule, or public artifact. No
previously computed CT or intrinsic values were inspected or persisted before
this rule was frozen.
