# Physical normalization A/B: preregistered protocol

Status: publicly frozen protocol v2, implementation r3; no real model arm has been scored
against either physical label volume. Protocol v1 was superseded before inference after the
truth-only power audit documented in `PHYSICAL_NORMALIZATION_AB_AMENDMENT_01.md`. All 64
coordinates are unchanged. The initial v2 runner was superseded after a pre-prediction Zarr
compatibility failure documented in `PHYSICAL_NORMALIZATION_AB_AMENDMENT_02.md`; this changed
runtime store creation only, not the statistical protocol. A second pre-score failure and
Windows-safe blend entrypoint are documented in `PHYSICAL_NORMALIZATION_AB_AMENDMENT_03.md`.

## Question

Does fixing `vesuvius.predict` to honor the `CTNormalization` declared by the public
`surface_m7` checkpoint improve its output against physical, model-independent truth?

This connects two independent August findings:

- [villa#1364](https://github.com/ScrollPrize/villa/issues/1364) found that the production
  wrapper silently used per-volume instance z-scoring instead of the checkpoint's plans.
  [villa#1386](https://github.com/ScrollPrize/villa/pull/1386) fixes that path and has an
  independent 16-volume fidelity check.
- [villa#1382](https://github.com/ScrollPrize/villa/pull/1382) provides physical truth from
  registered higher-resolution scans of PHerc0139 and PHerc1203. Its published m7 audit
  explicitly identifies #1364 as an upstream confounder and calls a corrected rerun the
  natural follow-up.

The higher-resolution scans are used only as evaluation truth. Both inference arms read the
same public 9.362 um scans. No high-resolution voxel is supplied to the model.

## Frozen inputs

The machine-readable manifest records every digest and URL. The intended inputs are:

- `scrollprize/surface_m7_nnunet`, fold 0 `checkpoint_best.pth`, with the model's own
  `plans.json` and `dataset.json`.
- PHerc0139 low-resolution volume `20250728140407-9.362um-1.2m-113keV-masked.zarr`, and
  published prediction `20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr`.
- PHerc1203 low-resolution volume `20250820131727-9.362um-1.2m-113keV-masked.zarr`, and
  published prediction `20250820131727-surface-20260413222639-surface-m7-L0-th0.2.zarr`.
- Physical-label release `v1.0` from
  `7jycwjmbfn-eng/pherc0139-physical-audit`: PHerc0139 archive SHA-256
  `42fe53b760c2c9347d9f215bafa68beec8e96121d03549dab56a52a9a0a9e8dd`; PHerc1203
  archive SHA-256 `32a09f6081342b0f015b258ec577d0296ff23a55892af9785689d8a55bff344c`.
- Corrected implementation at villa PR #1386 head
  `f74929a643095ce422ea4d9b70c25ae2b233a000`.
- Broken-path reproduction tree at `94ba215963afb6216e380fe2c86131fa5e724c3b`, which
  carries the previously validated global-grid/retry fixes but predates plans-driven
  normalization.
- Physical metric definitions are tied to villa PR #1382 head
  `5408c48d9db0558a78118d24fe9919ee63b204ee`.

All local files must match the manifest before a run is authorized. A dirty implementation
worktree, a changed model, an uncommitted manifest, a missing prediction chunk, or a
normalization log that does not explicitly resolve to `ct` is a hard failure.

## Arms

1. **Published baseline.** The public threshold-0.2 m7 binary artifact. On one manifest-frozen
   sentinel block per scroll, the old instance-zscore code path must reproduce this artifact
   at Dice >= 0.999 before the baseline is interpreted as the broken-normalization arm.
2. **Corrected fixed-threshold arm (primary).** PR #1386, no `--normalization` override,
   threshold 0.2, TTA disabled, fold 0, batch size 1. The log must state that model-declared
   `ct` replaces the CLI default.
3. **Corrected matched-mass arm (control).** One threshold per scroll is selected without
   looking at material or centerline truth: it minimizes the difference between the corrected
   and published positive counts over the manifest-frozen valid mask. Ties choose the higher,
   more conservative threshold. This asks whether any gain survives an equal prediction
   budget.

The fixed-threshold arm is primary because it is the operational behavior changed by #1386.
The matched-mass arm prevents a denser output from winning by construction.

## Frozen sampling

The full physical windows are too large for a timely two-arm GPU rerun. Sampling is therefore
fixed before inference and depends only on the released label bits, never on either prediction.

- Grid: candidate 64^3 level-1 score cubes on a fixed 128-voxel lattice. Origins remain
  multiples of 64 so arc tiles match #1382.
- Eligibility: at least 5% `valid` voxels and at least 256 centerline voxels on the same
  every-fourth-z phase used by #1382. These are truth-availability requirements, not model
  outcomes.
- Coverage: split each label window into four equal z strata and choose eight candidates per
  stratum (32 per scroll) by ascending SHA-256 of the frozen seed, scroll ID, and coordinates.
  No replacement is allowed after outcomes exist.
- Separation: the 128-voxel lattice leaves at least 64 level-1 voxels between scored cubes.
- Context: predictions include an 8-voxel metric halo and the complete 64-voxel upstream
  region needed for the 1.2 mm shifted null. Inference requests an additional 64 level-0
  voxels on every face so all scored values are in the blended interior.
- Visuals: the first hash-ranked selected block in each z stratum, at the label-only slice
  with the most centerline voxels, is fixed in the manifest before inference.

The PHerc1203 label's `boundary_poor` bit covers most material, while its centerline bit exists
only where sheets remain separable. Results are therefore explicitly conditional on the
released, physically resolvable centerlines. `boundary_poor` coverage is reported for every
block and as a frozen secondary subgroup; it is not used to pick favorable examples.

## Metrics

The scorer corrects a fail-open behavior in the current #1382 harness: a truth slice must not
leave the denominator merely because `pb.sum() < 100`. Empty and sparse predictions remain in
the evaluation and receive zero hits. Prediction shape, dtype, coordinates, hashes, and block
completeness are checked before scoring.

Metrics reproduce #1382 where applicable:

- point recall within 19, 37, and 56 um;
- arc recall at 37 um and fully missed 1.2 mm arc tiles;
- the same prediction shifted +64 level-1 voxels in y as a density/null control;
- inward-side fraction and full-population null/ideal ceiling as secondary diagnostics;
- predicted-positive count within `valid`;
- fractions of valid predicted positives farther than 37 and 75 um from physical material.

Every point and arc denominator is derived from truth alone and is identical across arms.
No slice is excluded based on prediction density. Aggregate counts and paired per-block values
are both retained.

## Primary analysis and decision rule

The primary block statistic is:

`point_skill = recall_37um - shifted_null_recall_37um`.

For each scroll, compare corrected fixed-threshold minus published baseline on the same 32
blocks. Report the macro mean difference and a deterministic 10,000-draw paired block
bootstrap 95% interval. Also report a pooled, z-stratified paired bootstrap in which each
scroll has equal weight.

The original v1 draft named arc skill as primary. Before any real-arm inference, an empty-
prediction rehearsal measured truth support only and found arc-bearing blocks in 26/32
PHerc0139 samples but just 4/32 PHerc1203 samples (7 arcs total there). The common primary was
therefore amended to null-controlled point skill, which has a frozen minimum of 256 sampled
centerline voxels per block. Arc skill remains a prespecified secondary endpoint, strong on
PHerc0139 and descriptive on PHerc1203.

A positive core-improvement claim requires all of the following:

1. mean point-skill improvement is positive on both scrolls;
2. the pooled paired-bootstrap 95% interval excludes zero;
3. the fraction of predictions farther than 37 um from material does not worsen by more than
   1.0 percentage point on either scroll;
4. corrected matched-mass point skill is non-inferior on both scrolls;
5. all 64 frozen blocks complete, or a pre-outcome technical failure is repaired and rerun
   under a receipt that preserves the same block and parameters.

Arc skill and fully missed arcs, 19/56 um point recall, 75 um false positives, side placement,
prediction mass, boundary-poor subgroup results, and fixed visual overlays are secondary.
They cannot rescue a failed primary rule. Arc results must always carry the number of
arc-bearing blocks; no cross-scroll arc claim is allowed from the sparse PHerc1203 support.

## No-tuning and stop rules

- After the public v2 replacement manifest, the model, normalization, threshold 0.2,
  selection seed, samples, metric tolerances, null displacement, bootstrap seed, and decision
  rule cannot change. The v1-to-v2 amendment is preserved in public history and was based only
  on empty-prediction truth support before any real-arm inference.
- The matched-mass threshold may use prediction mass and the released `valid` bit only. It may
  not inspect material, centerline, recto, boundary-poor, or any score.
- A failed block may be rerun only for an objective technical failure (network error, nonzero
  process exit, missing chunk, or failed provenance check). Its prior receipt is preserved.
- No block can be replaced. No qualitative figure can be selected after outcomes; the figure
  coordinates and z slices are in the manifest.
- If the fixed branch does not resolve to CT normalization, the sentinel does not reproduce
  the public artifact, or any input digest changes, scoring is forbidden.
- A negative result is published as negative; it is not used to tune a second protocol on
  these physical windows.

## Commands

Exact commands are emitted into the manifest and run receipts. The intended sequence is:

```text
python physical_normalization_ab.py plan ...
python test_physical_normalization_ab.py
git commit && git push                    # public freeze before outcomes
python run_physical_normalization_ab.py verify ...
python run_physical_normalization_ab.py sentinel ...
python run_physical_normalization_ab.py run ...
python physical_normalization_ab.py score ...
python physical_normalization_ab.py figures ...
```

The public freeze commit, implementation-file hashes, environment versions, commands,
stdout/stderr hashes, elapsed times, and every generated array hash are included in receipts.

Both arms run under one frozen Zarr compatibility bootstrap. Under Zarr 3 it restores the
former `zarr.Blosc -> numcodecs.Blosc` alias and requests Zarr format 2 whenever the legacy
writer supplies a v2 compressor. Under Zarr 2 those operations preserve native behavior. The
bootstrap is applied to inference and blending and its SHA-256 is recorded in every receipt.
Blending imports its module by canonical package name so Windows worker-spawn can resolve its
functions; this entrypoint has its own frozen SHA-256 and regression test.
