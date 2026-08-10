# PHerc0125 paired-quality CT block-batching amendment

Frozen 2026-08-10 01:09 Africa/Casablanca after the aligned quality evaluator
stopped at its frozen CT-block resource cap, and before any quality report,
stored raw CT sample, CT statistic, bootstrap interval, intrinsic report,
quality decision, PHerc0211 dataset, or public outcome existed.

## Observed resource fact

The already frozen 20 sites and 400 paired rays were selected successfully.
For site 1, the single axis-aligned block enclosing all baseline and final rays
had shape `[152, 766, 304]`, or 35,395,328 voxels. That exceeds the registered
32,000,000-voxel cap. The evaluator failed before writing a report. The cap is
not raised and the site/ray sample is not changed.

## Frozen batching rule

For each already selected neighbourhood, preserve its stored lexicographic
cell order and greedily form consecutive ray batches:

1. Start with the first unassigned cell. Add the next cell while the one block
   enclosing both arms' complete rays, including the existing six-voxel
   margin and volume clipping, contains at most 32,000,000 voxels.
2. When the next cell would exceed the cap, finalize the non-empty current
   batch and start a new batch at that cell. A single paired ray that exceeds
   the cap remains a hard failure.
3. Compute planned clipped bounds before any block fetch. Require the fetched
   block shape to equal the plan, remain non-empty, have every dimension at
   least 12, and remain at or below the original cap.
4. Evaluate every original ray once, restore the exact original cell order,
   and retain neighbourhood ID as the cluster-bootstrap unit. Record each
   batch's site, index, half-open cell-index range, shape and voxel count.

This changes only CT I/O batching. It changes no fit or geometry artifact,
angular alignment, scroll, CT volume/level, site/ray identity, profile reach,
step, detector, statistic, bootstrap, threshold, intrinsic guard, decision or
claim boundary. No batched block or resampled geometry becomes a published fit
artifact.

