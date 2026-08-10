# Paired quality evaluation for the PHerc0125 native Spiral fit

Frozen 2026-08-09 21:28 Africa/Casablanca, while the guarded native-fit
watcher was still waiting, before the PHerc0125 dataset root existed, and
before any baseline, final fit, CT sample or intrinsic result existed.

## Reuse and scope

This evaluation composes, rather than reimplements, the exact MIT revisions:

- `DomRusso2/sheetcheck` commit
  `7d53893abcc6cc7c0542e483c7266d75ea930885` for CT ray support and nearest
  detected-sheet offset primitives;
- `Nicodol/spiralcheck` commit
  `d1b50e2957409a870225fb9f5dcc5e25f7a0f9da` for whole-family radial,
  collapsed/inflated-spacing and validity checks.

`sheetcheck`'s periodicity-based pitch estimator is withdrawn and must not be
imported, run, reported or used in a decision. The evaluation measures only
the registered step-1 baseline against the same run's step-15,000 output. It
does not establish letters, reading, physical winding sense, generalization,
maintainer adoption or a prize outcome.

## Frozen inputs

- Native-fit preregistration SHA-256
  `a8af1607cdb930d13e2c26b1b1e5aaaca7d46a0abea1394234ab6ada9253b504`.
- Villa head `07bb743eb0382e4d94217f49128b126c4b0a9682`.
- PHerc0125 plan SHA-256
  `611b0d2a35fd4699cba09aa179d09525d44e633f8a25f12774b5e178d0371dd5`.
- Manual curve SHA-256
  `458e6ecfcdefef4a1cbd7baa859fd68907f5ea3b18de6f11589893019b5e663f`.
- Masked CT
  `PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr`, pyramid
  level 1, hence 18.724 micrometres per evaluated voxel.
- Baseline is exactly milestone 1 in the complete smoke evidence; final is
  exactly milestone 15,000 in the complete production evidence. Every preview
  file and the runner's canonical tree hash must revalidate before sampling.

## Paired CT sampling

Use seed `20260809`, ray reach 700 micrometres in each direction, and step 0.5
level-1 voxel. Split Villa's combined TIFXYZ by its declared winding column
ranges before calculating normals, so no concatenation seam becomes a normal.

Form the intersection of baseline/final normal-valid cells per winding. Draw
20 area-weighted neighbourhood centres uniformly over that intersection,
rejecting repeat centres and centres whose 13 by 13 windows overlap on the
same winding. Each accepted 13 by 13 window must contain at least five common
valid cells. Select at most 20 of its common cells with a deterministic RNG
derived from the global seed, winding and centre. The exact same winding,row,
column cells are evaluated in both arms.

For each neighbourhood, fetch one CT block that encloses both arms' rays and
refuse empty, undersized or greater-than-32,000,000-voxel blocks. Record every
sampled profile, its SHA-256, support, gap-structure availability and absolute
nearest-sheet offset. Do not calculate pitch or planarity.

## Paired intervals and CT decisions

Compute these paired differences, positive favorable:

1. final minus baseline support;
2. baseline minus final absolute offset in micrometres;
3. final minus baseline gap-structure indicator.

Use the paired median for support and offset and paired mean for gap structure.
For each quantity, generate 10,000 deterministic cluster-bootstrap draws by
resampling the 20 neighbourhoods with replacement and pooling their selected
rays. A quantity is sufficient only with at least eight contributing
neighbourhoods and 50 paired rays. All three quantities must be sufficient.

An improvement is present when at least one 95% interval is strictly above
zero. A significant CT degradation is present when any interval is strictly
below zero. A quantitative CT-improvement decision requires an improvement,
no significant CT degradation, and no material intrinsic regression.

## Intrinsic regression guard

Run unmodified `spiralcheck intrinsic_report` defaults on both complete
winding families with the frozen manual curve. Require identical winding IDs
and positive finite median pitch. The final result has a material intrinsic
regression if any condition holds:

- radial-violation fraction increases by more than 0.001 absolute;
- collapsed or inflated spacing fraction increases by more than 0.005
  absolute;
- mean winding validity decreases by more than 0.02 absolute;
- any one winding's validity decreases by more than 0.05 absolute;
- final/baseline median-pitch ratio lies outside the inclusive `[0.5, 2.0]`
  engineering stability range.

These are deterministic safety tolerances, not accuracy confidence intervals.
PHerc0211 execution may be authorized only when all CT measurements are
sufficient, no CT interval is significantly adverse, and the intrinsic guard
passes. PHerc0125 need not itself improve for PHerc0211 to run, because the
registered two-scroll gate permits the other scroll to supply the favorable
result. Step 30,000 remains a separate decision under the native-fit protocol.

## Publication boundary

The machine report may authorize only the narrow phrase "improved paired CT
support/offset under the preregistered sample" when its quantitative decision
is true. It never authorizes a reading, letters, winding-sense, cross-scroll,
adoption or prize claim. Recheck public overlap and visually inspect the bound
artifacts before any publication.
