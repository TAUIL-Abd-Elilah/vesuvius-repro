# Where m7 misses: the paired within-volume comparison

One JSON per volume from `bench_m7_recall.py`, same as `results/m7_recall/` but with an
extra `local_missed_vs_found` block, and with precision computed correctly (label class 2,
`ignore`, excluded from scoring).

## Why this exists

Volume-level properties do not explain where the model fails. Thickness, inter-sheet
spacing, CT contrast, fragmentation and flatness were measured for all 868 volumes
(`results/miss_map/`) and regressed against recall: **all nine jointly explain 9.9% of the
variance**, and the fitted model spans 77–92% predicted recall against an actual 55–97%.
Averaging over a volume destroys the signal, because the effect is local.

## What the paired comparison shows

For each volume, the *missed* sheet voxels are compared against the *found* sheet voxels in
that same volume, so each volume is its own control and no volume-level confound can
produce a signal.

| property | missed vs found |
|---|---|
| **CT intensity** | **missed are 10.3% darker** (median 96 vs 107) |
| consistency | **161 of 201 volumes (80.1%)**, sign test **p = 2×10⁻¹⁸** |
| local CT texture | weakly lower (61% of volumes) |
| local sheet thickness | no difference |
| component size | no difference |

Dose-dependent, which is what distinguishes a cause from a static bias:

| volumes by recall | median effect | darker in |
|---|---|---|
| recall < 70% | −0.708 σ | 90% |
| 70–90% | −0.475 σ | 84% |
| > 90% | −0.163 σ | 71% |

**The model finds the bright parts of a sheet and misses the faint parts.** Not thickness,
not fragmentation, not crowding.

## Caveats

`delta_z` is the difference of medians in units of the found-group standard deviation, so
it is comparable across volumes with different intensity scales.

These volumes are almost certainly m7's **own training data** (see
`results/m7_recall/README.md`), so this is a statement about fit, not generalisation.
"It misses the faint parts of what it was trained on" is supported; "it will miss faint
sheet on unseen scrolls" is not tested here.
