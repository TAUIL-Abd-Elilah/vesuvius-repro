# PHerc0139 fiber-field surface sanity check — preregistration

Status: **frozen at `2026-08-23T00:25:57Z` before reading any
prediction-array value**. The exact lock, reference manifest, runner, tests,
and environment are published under the immutable annotated tag
`pherc0139-fiber-sanity-prereg-v1`. Before this freeze I read only
catalog/provenance documents, Zarr descriptors, and reference coordinate
files. I did not sample `presence`, `nx`, or `ny`.

## Question and scope

On the 37 public PHerc0139 surface maps available before this inference, does
the new public fiber field (a) place higher presence scores on the mapped
surface than at fixed normal offsets and (b) predict directions tangent to the
surface?

This is a real-scroll **surface-coincidence and tangent-plausibility sanity
check**. The maps are not human fiber traces. A generic sheet detector or any
tangent vector can pass parts of the check. It cannot establish fiber-tracing
accuracy, better downstream unrolling, or ink recovery.

## Frozen inputs

All machine-readable choices are in
`pherc0139_fiber_presence_lock.json`. Exact hashes of every reference file and
the deterministic sample coordinates are in
`pherc0139_fiber_reference_manifest.json`.

- Scroll and target volume: `PHerc0139`, `20260102150214`, 2.399 µm base
  voxels.
- Fiber inference: run `20260801084232`, generated
  `2026-08-18T16:41:32.243382Z` from source level L1 (4.798 µm voxels).
- Tested arrays: uint8 `presence`, `nx`, and `ny` at L3. Their stored spacing
  is 19.192 µm. “4.8 µm prediction” describes the inference input, not the
  published output grid.
- Presence is decoded as `uint8 / 255` and called a score, not a calibrated
  probability. `nx` and `ny` use the producer's compact hemisphere encoding.
- The reference universe is the 37 catalog-listed segments in the lock. Each
  uses the target-specific transformed map
  `<short-id>-on-20260102150214-2.399um.tifxyz`, in target base-voxel
  coordinates. The title segment is excluded because it has no catalogued
  transformed coordinate map.

The original low-resolution maps are not used: multiplying those coordinates
by four is only approximate and can misregister the first control offset. The
target coordinates convert to L3 by exactly `1 / 8 = 0.125`, with XYZ input
reordered to ZYX array indices and no added half-voxel translation.

The inference record does not publish the exact checkpoint training set.
Therefore this check cannot rule out overlap between model development data
and PHerc0139, even though all reference segments predate the inference.

## Deterministic sampling

For each locked segment:

1. Verify the pinned `meta.json`, `x.tif`, `y.tif`, and `z.tif` bytes.
2. A coordinate is valid only when all three values are finite and
   non-negative. Its four one-pixel neighbours must also be valid.
3. Estimate the surface normal as the normalized cross product of centered
   row and column differences. Degenerate normals are invalid.
4. Divide the UV raster into 16 × 32 bins using integer
   `floor(k * length / bins)` edges, clipped to the one-pixel interior. For a
   bin with `N` row-major candidates, hash
   `seed|segment|bin_y|bin_x` as UTF-8. The first eight digest bytes choose a
   start modulo `N`; the next eight choose a stride, advanced cyclically to
   the first value in `[1,N-1]` coprime with `N`. Traverse that permutation
   and take its first valid, non-degenerate point. This produces at most 512
   spatially distributed points per map without consulting predictions.
5. Convert each XYZ center to prediction ZYX coordinates by 0.125. Apply the
   unit-normal offsets only after conversion to the L3 grid.
6. Trilinearly sample presence at `-12, -8, -4, 0, +4, +8, +12` L3 voxels,
   equal to `-230.304, -153.536, -76.768, 0, +76.768, +153.536, +230.304`
   µm. The full eight-voxel interpolation cube must be in bounds.
7. Sample `nx` and `ny` at the center only. Every requested object is recorded
   with either its content hash or a definitive HTTP 404. Because the pinned
   Zarr descriptors declare fill value zero, a missing presence chunk is a
   zero score. A point contributes to orientation only when its center
   presence is positive and all `nx`/`ny` interpolation-support chunks exist;
   positive presence without direction support is reported as sibling
   incoherence. Other network or software failures abort rather than becoming
   scientific exclusions.

A map is analyzable for localization only if at least 128 complete profiles
remain. No failed or missing map is replaced. An even-count median is the
arithmetic mean of its two central values.

The offsets may intersect nearby papyrus sheets. Therefore the endpoint means
only “more score at these mapped surfaces than at these exact offsets.” A null
result is not evidence that the fiber model fails generally.

## Primary descriptive endpoint

For each point:

`delta = presence(0) - mean(presence(-12,-8,-4,+4,+8,+12))`.

The map value is the median point delta. The finite-set result is
`LOCALIZATION_SUPPORTED` only if all three preregistered gates pass:

1. at least 30 maps are analyzable;
2. at least 24 analyzable maps have positive median delta; and
3. the median of map deltas is at least 0.02 on the decoded [0,1] score scale.

Otherwise it is `LOCALIZATION_NOT_SUPPORTED`. No p-value is used: maps from
one scroll can overlap and are not independent random population samples.

## Secondary tangent-plausibility endpoint

At each valid center, decode
`dx=(nx-128)/127`, `dy=(ny-128)/127`,
`dz=sqrt(max(0,1-dx²-dy²))`, then renormalize. The sign-invariant deviation
from the local tangent plane is
`alpha = degrees(asin(clamp(abs(direction · surface_normal), 0, 1)))`;
0° is ideal and an isotropic unoriented axis has median 30°.

The preregistered map summary is the presence-score-weighted median alpha,
provided at least 32 positive-weight points have complete direction support.
A deterministic matched baseline preserves predicted directions and surface
normals but globally sorts samples by `SHA256(seed|orientation-baseline|segment|row|col)`
and rotates the normals by `floor(N/2)` before pairing. The secondary result is
`TANGENCY_SUPPORTED` only if:

1. at least 30 maps are analyzable;
2. the median of analyzable map summaries is at most 20°; and
3. at least 24 maps improve by at least 5° over their matched-baseline
   weighted median.

The unweighted result over all positive-presence, direction-valid centers and
a fixed center-presence-score `>= 0.5` subset with its coverage are reported
descriptively. This endpoint tests only tangency to a known sheet, not the
direction along a particular fiber.

## Visual outputs

Exactly 12 map IDs and their selection hashes are materialized in the lock
before outcome access. Fixed-scale panels show UV sample locations colored by
center presence ([0,1]), localization delta ([-1,1]), and tangent deviation
([0°,90°]), plus the seven-offset median profile ([0,1]). Panels are
descriptive quality control and cannot override either numeric result. No
panel may be replaced after inspection.

## Execution and publication

The 67.4 GiB compressed prediction is not downloaded wholesale. A
reference-only preparation stage freezes all reference hashes and sample
coordinates without opening a prediction array. It selected 17,763 points;
their frozen queries require 12,338 presence chunks and 10,076 chunks from
each direction channel before accounting for sparse 404 fills. The outcome stage writes an
exclusive `OUTCOME_STARTED` seal before any prediction access, downloads only
the exact required chunks into a verified local mirror, and records every
chunk hash. Outcome mode accepts only the canonical lock and output directory,
so changing `--out-dir` cannot create another local outcome slot.

There is one outcome run after the public immutable tag. A technical failure
is reported and requires a public amendment; it is never converted into a
null. Scientifically valid positive, partial, null, or negative results are
published without retuning, filtering, panel substitution, or rerunning.
