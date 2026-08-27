# Sealed PHercParis4 patch-sampling comparison

Frozen before creating or scoring the held-out split.

## Selection and scope

- Code under test: villa commit `17dad916c`.
- The `0.75` cap was selected using a development screen on the full public
  patch set plus the public `eval_fibers/` consistency check. Those fibers are
  withheld from direct training but share upstream provenance and may overlap
  patch geometry; they are not the sealed test below.
- Dataset snapshot:
  `https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/`, local index
  SHA-256 `29f975b1f1615bf7b11e58ec88c77d17c3d7603c4f70e43175e1eb31d65159c2`.
- Evaluation tool: `Nicodol/spiralcheck` commit
  `d1b50e2957409a870225fb9f5dcc5e25f7a0f9da`.

## Sealed split

Run `spiralcheck split` over the complete 89,237-patch snapshot with holdout
fraction `0.20` and seed `20260827`. The tool groups derived-name families,
merges byte-identical geometry, stratifies by z, and writes content and
geometry hashes. Neither arm may consume the held-out side.

## Fit arms

Both arms use z `[10500, 11500)`, seed `17`, 5,000 steps, area exponent `0.5`,
and only verified-patch supervision; all non-patch annotation and dense-volume
inputs are disabled. This is a controlled screening fit, not a production
whole-scroll result.

- Baseline: historical sampler (no family cap).
- Treatment: retain all fit-side patches and cap the aggregate draw mass of
  IDs matching `^band-seed` at `0.75`.

Export the plain (not patch-spliced) meshes from each final checkpoint. Score
the identical held-out side with `spiralcheck score --variant plain`, passing
both the split manifest and actual fit-side directory. Quote only the
`unseen` results after its geometric leakage audit.

## Frozen decision rule

The primary metric is family-balanced unseen sheet consistency: compute the
point-weighted unseen sheet-consistency fraction separately for `band-seed`
and all other patches, then average the two family values equally. Use 20,000
paired bootstrap resamples within each family with seed `20260827`.

The treatment passes only if:

1. the primary delta is positive and its 95% bootstrap interval is above zero;
2. the `band-seed` unseen sheet-consistency point estimate falls by no more
   than 3 absolute percentage points; and
3. pooled unseen fraction-within-6-voxels falls by no more than 1 point.

Also report distance p50/p90/p99, normal-angle p90, both family results,
intrinsic checks, run time, and every failure. If the gate fails, make no
held-out-quality claim.
