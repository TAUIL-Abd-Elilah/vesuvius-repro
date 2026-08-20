# Label-placement orientation correction: result

Result generated on 2026-08-20 from source commit `0f9e2d98cda8cc68c0568bdc0642e39b2fc84c06`.
The design was frozen publicly before the corrected outcome in the preregistration and three
amendments on this branch.

## Correction

ZandieckNol's objection is correct. The Hessian eigenvectors in the historical analysis had
arbitrary signs, so the reported population signed median `+0.0077 vox` had no common physical
direction. I withdraw that number as a directional result and withdraw the claims that annotation
centres already sit on the CT ridge, that there is no systematic displacement, and that global
label snapping has nothing to correct.

The historical seed-0 analysis replays exactly: all 30 row counts and four-decimal signed and
absolute medians match. Only 9 of those 30 have independent global Scroll1A coordinates, so the
other 21 cannot be physically oriented from this reference.

## Frozen primary result

Normals point inward toward the pinned Scroll1A axis. Corrected offset is the CT-ridge coordinate
minus the label-run-centre coordinate along that inward normal. Positive means ridge inward of
label centre; negative means ridge outward of label centre.

On all 189 independently mapped Scroll1A cubes, at the frozen `+/-4 vox` corridor and no alignment
filter:

- median of cube signed medians: **-0.5150 vox**;
- q10/q90 of cube medians: **-0.9042 / -0.0816 vox**;
- sample-bootstrap 95% interval: **[-0.5659, -0.4591] vox**;
- 1,280-voxel spatial-block-bootstrap interval: **[-0.6570, -0.3921] vox** over 38 blocks;
- **186/189** cube medians are negative;
- median of cube median absolute offsets: **1.1562 vox**.

Under this estimator, the CT intensity ridge is therefore usually outward of the label-run centre;
equivalently, the label centre is inward of the ridge. The recoverable 9-cube historical subset
agrees descriptively: `-0.4926 vox`, sample interval `[-0.7779, -0.2123]`.

Direction remains negative in every frozen sensitivity, but magnitude is corridor-dependent:

| corridor | no filter | abs cosine >= 0.25 | abs cosine >= 0.50 |
|---|---:|---:|---:|
| +/-3 vox | -0.459 (188) | -0.482 (178) | -0.533 (140) |
| +/-4 vox | -0.515 (189) | -0.558 (188) | -0.618 (164) |
| +/-6 vox | -0.614 (189) | -0.665 (189) | -0.730 (172) |
| +/-8 vox | -0.785 (189) | -0.817 (189) | -0.891 (179) |

Parentheses give cubes meeting the frozen 150-retained-run minimum in that cell. At the primary
cell, 55,061/113,400 sampled points remain valid. Ridge-edge and label-run-truncation rates are
32.34% and 35.35%; they overlap and must not be added. Across cubes, the q10/median/q90 of each
cube's median absolute radial cosine are 0.476/0.574/0.674.

## Limits

This is a correction, not evidence for applying a universal half-voxel snap. The magnitude moves
with corridor width, absolute offset includes real-CT estimator error, and this mapped cohort is
Scroll1A only rather than all 892 pairs. The 9-cube replay subset overlaps the 189-cube expansion
and is not an independent replication. The intervals are descriptive cube/block bootstraps, not a
registered hypothesis test.

Most importantly, Villa's [VC guidance](https://github.com/ScrollPrize/villa/blob/main/scrollprize.org/docs/06_tutorial_VC.md#L282)
says segmentation should ideally lie on the inside, ink-facing sheet surface. An inward label
centre relative to the CT intensity ridge can therefore be intentional rather than an annotation
error. Demonstrating a useful label correction still requires generated labels and a positive
held-out comparison against physical or human truth and existing tooling.

## Reproduction

- Machine-readable result: `results/label_placement_oriented.json`
- Result SHA-256: `8397821b21f226650457b69ff8888e093fcbbdfc4ea8aa4be8710d73589d15c8`
- Axis SHA-256: `84785853ad918e98bf241656b2dea80bae6b77303a13d57805f9d7854d391cc9`
- Volume metadata SHA-256: `d34e437ca3404aa5f7faaaaa731927ce7adfadf84376dfc3a587c400a40d2520`
- Tests: `python -m unittest -v test_label_placement_oriented.py` (8 passed)

```text
python label_placement_oriented.py \
  --images /path/to/images \
  --labels /path/to/labels
```

After the first numerical run, commits `45d294b`, `04a90f0`, and `0f9e2d9` changed only portable
metadata, corrected explanatory wording, added the official interpretation limit and alignment
summary, and made JSON key ordering deterministic. The final result above was recomputed from a
fresh checkpoint after those changes. No sample, estimator, corridor, threshold, eligibility rule,
or statistic changed.
