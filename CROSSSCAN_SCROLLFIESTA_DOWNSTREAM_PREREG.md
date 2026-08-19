# Cross-scan → ScrollFiesta downstream preregistration

Status: **locked before any terminal v4 pilot or final result was inspected**.

Created: 2026-08-12 00:32:47 UTC. This document adds a separate system-level gate; it does not
change the frozen cross-scan primary endpoint, cases, seeds, model, or promotion thresholds.
The machine-readable source of truth is `crossscan_scrollfiesta_downstream_lock.json`; v2 content
SHA-256 is `06142dc819c193a462f37d08a4769024c41ab551411013d40dda72db148457f6`.

Amendment v2 was frozen at 2026-08-12 00:40:24 UTC while the seed-39 odd-fold process and all
three watchers were still alive and before any pilot verdict, final result, or release manifest
existed or was inspected. It supersedes v1 content SHA-256
`709740293e44aef3b616eb1d8f520f8614131f58870a4a21ce88fc95c8fb2491`. A pinned-source audit
showed that direct `grid_pipeline` success does not prove a clean weld: it can treat a weld-audit
failure as a nonfatal warning; `pinch_verts` was missing from v1; and `grid_weld` treats residual
same-direction pairs as cosmetic while v1 imposed an unexplained candidate-only zero threshold.
No threshold was changed using model or baseline output.

## Why this exists

The registered cross-scan experiment can establish a held-out PHerc1203 block effect and
PHerc0139 non-inferiority in pooled AP. It cannot establish fewer mergers, sheet switches, or
tracing failures because non-recto material is ignored by that endpoint. `POSITIVE_DEPLOYABLE`
therefore means release-eligible under the registered block endpoint, not downstream-safe.

This prospective extension asks one narrower system question: on a fixed 2×2×2 grid from the
completely untouched PHerc0139 scroll, does the treatment improve the physical surface at equal
foreground mass, and does that improvement survive ScrollFiesta meshing and seam welding?

## Frozen region and provenance

Implementation integrity note (2026-08-12, after the pilot gate): the already frozen public RAW
context was read once from the URI and box below and pinned by canonical uint8 array SHA-256
`b654ba8428beb5f24378efb2d9e8f5d516cec60410871682bdbff5535b6b665f`. This adds no region,
threshold, arm, or acceptance change; it only prevents a different CT payload from being passed
off as the preregistered input.

The grid is anchored at `PHerc0139-z0192-y1280-x0192`, the z-stratum-0 visual case already selected
by the public pre-outcome hash rule. The local level-1 label box is
`[192:320,1280:1408,192:320]`; the corresponding world level-0 box is
`[3840:4096,3712:3968,1344:1600]`. It contains eight adjacent 128³ level-0 cubes. No model
prediction selected the region. Pre-outcome truth counts are 1,674,807 valid, 1,241,241 material,
and 444,149 recto level-1 voxels; its boundary-poor-material bit count is zero, so this is a dense
untouched-scroll weld test, not evidence about annotated boundary-poor regions.

PR #1382 authored and supplies the physical labels under CC BY-NC 4.0. Our contribution is the
locked model comparison, cloud-native adapter, independent metrics, and consumer run.

## Frozen arms

- Baseline: released m7 fold-0 checkpoint, SHA-256
  `17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e`.
- Candidate: only if the registered final bucket is `POSITIVE_DEPLOYABLE`; equal-weight mean of
  all twelve untouched-PHerc0139 predictions (six seeds × even/odd folds).
- Operational mask: `probability >= 0.2`, encoded as uint8 0/255.
- Spend-controlled mask: select exactly the baseline operational foreground count from the
  candidate by stable descending probability and then C-order index.

Inference uses the exact PR #1386 `plans.json` normalization, Gaussian sliding-window blending,
tile step 0.5, and no mirroring. It reads the locked 384³ context and retains the central 256³
box. Probability is exported as float32 OME-NGFF Zarr v2 with 128³ chunks, explicit world
translation, resumable hash receipts, round-trip reader tests, and resource measurements.

## Frozen consumer

ScrollFiesta is pinned to the clean ABI-fixed PR #12 head
`f0d9d2e54823e7ba2460725e81290eead8ed6e5e`. The exact `cube_mesh`, `grid_pipeline`, and
`grid_weld` binary sizes/hashes and argument vector are in the machine lock. Baseline fixed,
candidate fixed, and candidate matched-mass grids use identical RAW CT cubes and settings.

## Nondegenerate topology and system gates

The retired `binary_fill_holes` metric is forbidden: an open sheet does not enclose a 3-D cavity.
Instead, 26-connected components of at least 63 level-1 voxels are matched within one level-1
voxel. We report missed/spurious components, splits, mergers, symmetric surface-distance median
and p95, precision, recall, Dice, and foreground mass.

The headline spend-controlled gate requires no additional aggregate mergers or total component
errors, a lower aggregate symmetric median distance, and improvement in at least five of eight
cubes. The fixed-threshold safety gate permits no additional aggregate mergers/errors and no
single-cube median-distance regression above 0.5 level-1 voxel.

All three direct ScrollFiesta pipeline commands must exit zero, but the independent checker also
parses each final OBJ and weld report because pipeline success is not a topology certificate.
Candidate arms must process every baseline cube, introduce no garbage rejection or crash, produce
an eight-cube weld with correct world extent, and report zero non-manifold edges and zero pinch
vertices. Each candidate arm must have no more same-direction pairs than baseline; zero remains a
separately reported stricter clean-mesh result rather than the comparative pass gate.

Seam survival is checked directly. From the final welded OBJ, an internal-seam unpaired edge is an
edge with one incident face whose midpoint lies within one level-0 voxel of the internal cube plane
`z=3968`, `y=3840`, or `x=1472`, excluding the one-voxel band at the six outer box faces. The union
and each plane are reported; neither candidate arm may exceed baseline. This avoids treating all
raw unpaired edges as failures, because legitimate open-sheet outer boundaries are unpaired.
Fixed orthogonal CT/truth/prediction cross-sections and a fixed welded-mesh camera are emitted
without replacement or cherry-picking.

Passing supports only a bounded untouched-PHerc0139 probability-to-ScrollFiesta meshing/weld
claim. It does not prove every-scroll utility, lasagna/Spiral improvement, universal topology
repair, or deployment safety outside this region. Failure is published and blocks the downstream
claim.
