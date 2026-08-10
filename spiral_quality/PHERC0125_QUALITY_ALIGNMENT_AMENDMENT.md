# PHerc0125 paired-quality angular-alignment amendment

Frozen 2026-08-10 01:00 Africa/Casablanca after the first quality-evaluator
attempt stopped at its pre-sampling geometry-shape check, and before any CT ray,
CT statistic, bootstrap interval, intrinsic report, quality decision, PHerc0211
dataset, or public outcome existed.

## Observed compatibility fact

The exact frozen step-1 and step-15,000 preview artifacts have the same 121
winding IDs (10 through 130), the same 61-row axial grid and the same declared
grid scale. Their per-winding column counts differ because Villa exports the
normalized angular axis at a fit-dependent resolution. Step 1 has 42,935 total
columns, with winding widths 53 through 656; step 15,000 has 39,458 total
columns, with widths 49 through 603. Every winding differs, with final/baseline
width ratio approximately 0.919 at the median.

This is an expected native representation difference, not a scientific result.
Villa's pinned `merge_concat_runs.py` explicitly documents that the column axis
is normalized angular (theta) position sampled at run-dependent resolution, and
that grids describing the same winding must be resampled to a common normalized
column grid before per-cell comparison. The original evaluator SHA-256
`64e8e71c140a9e9e17152897af0abe63c6a709639a01e58b8a9dd59137c65090`
instead required equal native shapes and exited with
`baseline/final winding shape differs: 10` before constructing a CT volume.

## Frozen alignment rule

This amendment changes only the representation alignment needed to execute the
already registered paired sampling. It changes no scroll, evidence artifact,
fit, seed, CT volume/level, site/ray count, reach, step, bootstrap, statistic,
threshold, intrinsic guard, decision rule, or claim boundary.

For CT sampling only:

1. Require identical winding IDs, identical row count, at least two columns per
   winding, finite declared scales, and matching scales between arms.
2. For each winding independently, set the common column count to the smaller
   of its baseline and final native column counts. This never invents a finer
   grid than either arm.
3. Map common column `j` to normalized theta coordinate
   `j / (common_columns - 1)`, including both endpoints exactly. In each arm,
   map that coordinate to `(native_columns - 1) * theta`.
4. Linearly interpolate the two adjacent native 3-D vertices. A common vertex
   is valid only when every required bracketing native vertex is valid and
   finite; an exact integer source location requires only that one vertex.
5. Recompute surface normals from each aligned grid using the pinned
   `sheetcheck.Surface.normals` implementation. Then form the registered
   baseline/final common-normal-valid intersection and choose the same seeded
   winding/row/common-column cells for both arms.
6. Record every native and common per-winding shape plus the alignment rule in
   the machine report, bind this amendment by SHA-256, and make the report
   verifier reject any mismatch.

The whole-family `spiralcheck` intrinsic reports remain on the original,
unresampled artifacts. No resampled geometry is published as a fit output.

## Fail-closed launcher repair

The launcher must also require that a successful evaluator process created the
expected report before reading it. If the child exits without a report, the
launcher reports the evaluator stderr instead of misreporting only a missing
path. Existing fit and preview artifacts remain immutable.
