# Label-placement orientation correction: frozen analysis plan

Frozen on 2026-08-20 before reading the corrected outcome.

This audit corrects the signed label-placement result in `results/label_placement.json` and
ScrollPrize/villa#193. The old `+0.0077 vox` aggregate used Hessian eigenvectors whose signs were
not tied to a shared physical direction. It has no population-level directional meaning. This
correction is prompted by ZandieckNol's public review; it is not an independent discovery.

## Fixed inputs

- The same 892 public CT/label TIFF pairs used by the original analysis.
- The pinned `results/overlap/overlap_report.json`, which independently maps 189 non-overlapping
  320-cube samples to Scroll1A in absolute `(z,y,x)` voxel coordinates.
- The public `umbilicus-scroll1a_zyx.txt` from the PHercParis4 volume package. The run records its
  URL and SHA-256. The Discord metadata discussion on 2026-08-14 correctly warns that an axis
  without volume identity, dimensions, and scale is ambiguous. The containing volume package
  binds Scroll1A to UUID `20230205180739`, shape `(14376,7888,8096)` in `(z,y,x)`, and 7.91-um
  voxels; the run records and hashes that `meta.json` alongside the axis.
- Seed 0, 600 selected label voxels per sample, and a minimum of 150 valid label runs per sample.
- Primary profile corridor `+/-4 vox` at 0.25-voxel steps. Sensitivity corridors: `+/-3`, `+/-6`,
  and `+/-8 vox`.

## Cohorts

1. **Original replay subset.** Recreate the original seed-0 sampling stream and first 30 eligible
   samples. Analyse the nine of those 30 that have an independent Scroll1A mapping. Confirm that
   their un-oriented `+/-4` medians reproduce the published rows before correcting their signs.
2. **Mapped expansion.** Analyse all 189 mapped Scroll1A samples. Sampling is deterministic per
   sample, so resuming cannot alter later samples.

The first cohort adjudicates the recoverable part of the old result. The second is a new mapped
cohort and will not be described as the original corpus. The other 21 original samples remain
magnitude-only because no independent global coordinates are available.

## Physical orientation

For local point `p=(z,y,x)`, use mapped coordinate `g=lo+p`. The public file is an unsorted list of
unique-z control points, not an ordered 3D path: the upstream Khartes/evolutor reader sorts this
exact format, and the source reference says it is specified at original scale. Sort by z, linearly
interpolate `(y_axis,x_axis)` at `g_z`, and define the inward reference
`r=(0,y_axis-g_y,x_axis-g_x)`. Flip Hessian normal `n` when `n dot r < 0`, so every oriented normal
points inward. Corrected scalar offset is the original offset multiplied by the same sign. Positive
means the CT ridge is inward of the label-run centre. The correction vector, landing point, and
`abs(offset)` must remain numerically unchanged.

Because a sign is fragile for nearly tangential normals, repeat the summaries after requiring
`abs(n dot unit(r)) >= 0.25` and `>= 0.50`. These are fixed sensitivity analyses.

## Fixed summaries

- Use the sample/cube as the aggregation unit.
- Per cohort, corridor, and alignment threshold, report the median and q10/q90 of per-sample
  signed medians, median per-sample absolute offset, counts, ridge-edge/NaN rate, label-run
  truncation rate, and radial-alignment distribution.
- Report deterministic 95% percentile intervals from 10,000 sample bootstraps.
- Report a 10,000-replicate cluster bootstrap over fixed 1,280-voxel `(z,y,x)` spatial blocks.
- No directional hypothesis or p-value is registered. This is a corrective audit.

## Interpretation gate

- Withdraw the un-oriented `+0.0077 vox` result and “global label snapping has nothing to
  correct” regardless of the corrected outcome.
- The sign-invariant landing point, synthetic landing validation, and `abs(offset)` calculations
  stand, subject to their existing real-data and axis-aligned limitations.
- This audit alone is not a new prize submission. Discord guidance on 2026-08-16 is explicit that
  useful work must compare against existing methods or visually demonstrate better output. A new
  submission therefore requires generated labels and a positive held-out comparison against
  independent physical or human truth. A null or sensitivity-unstable result stops this path.
