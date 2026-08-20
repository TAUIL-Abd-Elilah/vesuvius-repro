# Amendment 03: cell eligibility and spatial clusters

Frozen on 2026-08-20 before any corrected placement outcome was computed.

The 150-valid-run minimum applies separately to every corridor and radial-alignment threshold
cell. A sample with fewer than 150 retained runs in a cell is excluded from that cell, and each
summary reports both excluded-sample counts and retained-point counts. This prevents a strict
alignment sensitivity from being summarized from a small, unstable subset.

For the spatial sensitivity, assign each mapped cube to a fixed 1,280-voxel `(z,y,x)` block using
`floor(cube_global_centre / 1280)`. Resample the unique blocks with replacement; every selected
block contributes all of its member cubes, including repeated contributions when that block is
drawn more than once. The sample bootstrap and spatial-block bootstrap use independent,
deterministically keyed seed-0 streams for each cohort/corridor/threshold cell.

The original-replay mapped subset and the 189-cube expansion overlap and are never pooled or
treated as independent. No other input, outcome, or interpretation rule changes.
