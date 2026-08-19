# Protocol amendment 01: common primary endpoint power

Date: 2026-08-11. This amendment was made before any broken-path sentinel inference, corrected
inference, matched-mass threshold, real-arm physical score, or model comparison existed.

## Trigger

Protocol v1 froze 32 label-only blocks per scroll and proposed null-controlled arc skill as the
common primary endpoint. A required empty-prediction rehearsal then exercised the production
scorer over all 64 blocks. Both synthetic arms were identically empty; the run could reveal
truth denominators and software failures, but no model outcome or arm difference.

The rehearsal found:

| scroll | blocks with >=1 arc | arcs | sampled centerline voxels |
|---|---:|---:|---:|
| PHerc0139 | 26 / 32 | 249 | 106,628 |
| PHerc1203 | 4 / 32 | 7 | 38,022 |

PHerc1203 has only one arc-bearing block in each z stratum. This follows from the released
physical label itself: centerlines exist only where individual sheets remain separable, and a
#1382 arc additionally requires at least 20 connected centerline pixels inside one 64x64 tile.
The v1 requirement of 24 arc-bearing blocks was therefore impossible before either model was
run.

Evidence:

- v1 manifest content SHA-256:
  `38a4c9888d0e35ab2cb913b8bc2117701f211ea5e5abb4fb0e3043c5a256d9a2`
- empty-synthetic scorer result content SHA-256:
  `f525b4d599ec99a8db08d5fe5c1f33e9bd63b49b7a630645fcbea425183c336b`
- machine-readable summary: `results/physical_normalization_ab/truth_power_audit.json`

## Amendment

The 64 coordinates, label eligibility, model, arms, thresholds, null displacement, false-
positive guardrail, visual selection, and bootstrap design remain unchanged.

The common primary endpoint becomes:

`point_skill = recall_37um - shifted_null_recall_37um`.

Every frozen block has at least 256 sampled physical centerline voxels by construction, so the
paired block endpoint is supported on 32/32 blocks on both scrolls. The shifted null addresses
the density confound that makes raw radius recall misleading on compressed PHerc1203.

Arc skill and fully missed arcs remain prespecified secondary endpoints. They are powered on
PHerc0139 and descriptive only on PHerc1203; every report must show its arc/block denominator.

The v2 success gate is otherwise unchanged: positive fixed-threshold point-skill change on
both scrolls, pooled z-stratified paired-bootstrap 95% interval above zero, <=1 percentage-
point worsening in far-37-um prediction fraction on either scroll, and nonnegative matched-
mass point-skill change on both scrolls.

No sample was added, removed, or replaced. This is a power correction based only on physical
truth availability, not a response to model performance.
