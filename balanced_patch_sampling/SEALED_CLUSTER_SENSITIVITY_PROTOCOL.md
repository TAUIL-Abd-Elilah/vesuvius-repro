# Pre-result family-cluster sensitivity for the sealed PHercParis4 comparison

Frozen while the seed-17 baseline scorer was still running and before either
seed-17 `report.json` existed. This supplements, and does not replace or amend,
`SEALED_PATCH_PROTOCOL.md` or its frozen gates.

## Why this sensitivity exists

The frozen primary bootstrap resamples patch IDs inside the two reporting
families (`band-seed` and `other`). The split manifest also records
`family_of` relationships for derived patch names. In the pre-result held-out
scope, eligible `other` patches are therefore not all independent source
units. A cluster bootstrap is a stricter uncertainty check.

The cap value was selected using development fits made before the held-out
split was created. No held-out score was inspected during selection, but the
development fits consumed the full public patch collection. Results must be
described as an untouched-outcome, post-selection holdout check, not as a
pristine independent confirmation.

## Frozen inputs and population

- Split manifest SHA-256:
  `9a1b226ebde3854728adfeb6f21513026c0cc49d948cc52684d6ee96e3819f31`.
- Held-out scope file SHA-256:
  `aeb342353687473bcff9ff7182e591831fcf94014ad9ba4ff6bb4f10c01881e8`.
- Seed-17 runner source SHA-256:
  `c33f956a55dc7f4ebe0b5007bfdead231eec65813c97c691b42fc2e86878c3c1`.
- Baseline and treatment reports must pass the original comparator's exact
  pairing, metadata, leakage-audit, and unseen-count checks.
- Eligibility remains at least eight unseen points per patch. No patch or
  cluster may be selected or removed based on either arm's score.

## Cluster-bootstrap sensitivity

For each eligible patch, define its source cluster as
`split_manifest.family_of[patch_id]` when that mapping exists, otherwise use
the patch ID itself. Keep the original two reporting families: IDs beginning
with `band-seed` and all other IDs.

Within each reporting family, draw that family's observed number of source
clusters uniformly with replacement. Every time a cluster is drawn, include
all of its eligible patch rows together. Compute the baseline and treatment
scores using the original unseen-point weights, take their paired difference,
then average the two reporting-family differences equally. Use NumPy's
`default_rng(20260827)`, 20,000 draws, and the 2.5/97.5 percentiles.

The point estimate remains the frozen primary estimate. Call its uncertainty
**cluster-robust** only when the treatment-minus-baseline point estimate is
positive and this sensitivity interval's lower endpoint is above zero. If it
crosses zero, report that the frozen patch-level result is uncertain under
source-family dependence even if the original gate passes.

## Family-specific proximity diagnostic

For each reporting family and arm, calculate the point-weighted unseen
fraction within six voxels from the per-patch report rows:

`sum(n_points * frac_within_tau) / sum(n_points)`.

Report treatment-minus-baseline deltas for both families. Claim “no
family-specific proximity regression” only if both deltas are at least
`-0.01`. This diagnostic is additional context, not a retroactive change to
the frozen pooled-distance gate.

Per-family pooled distance quantiles cannot be reconstructed exactly from the
published per-patch summaries and will not be approximated by averaging
quantiles. Any later exact per-family distance-quantile run must be labelled
post-result exploratory.
