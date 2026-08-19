# Physical normalization A/B: fail-closed sentinel result

Status: **closed without running the corrected arm**.

This is the outcome of the baseline-reproduction gate frozen in
[`PHYSICAL_NORMALIZATION_AB_PREREG.md`](PHYSICAL_NORMALIZATION_AB_PREREG.md). The protocol
required the public m7 artifact to be reproduced at Dice >= 0.999 on one frozen 64^3 L1
score cube from **each** scroll before any inference with PR #1386 was allowed.

The implementation, inputs, blocks, and cutoff were public at commit
`a30a06bf0cfabed73a7fad5bf7e897dcbb427d60`; the manifest content SHA-256 was
`567a18faa1c8ca7e743c9240133f4200e67e3085823dd4795c4518e3e0e65ac0`.

## Frozen-gate outcome

| Scroll | Published positives | Reproduced positives | Disagree | Dice | Required | Gate |
|---|---:|---:|---:|---:|---:|---|
| PHerc0139 | 63,572 | 63,575 | 19 | 0.9998505667 | 0.999 | pass |
| PHerc1203 | 95,815 | 95,844 | 199 | 0.9989616976 | 0.999 | **fail** |

Because the conjunction failed, `run_physical_normalization_ab.py run` remains
fail-closed and the corrected arm was not run. The cutoff was not rounded, relaxed, or
redefined after observing the result.

## Diagnosis, not a rescue analysis

The PHerc1203 discrepancy is confined to the decision boundary: all 199 disagreeing
voxels are within 0.0060142 probability of the frozen 0.2 threshold; 169 are within
0.001 and 47 within 0.0001. There are 85 published-only and 114 reproduced-only voxels.
This is consistent with the same small numerical residue previously observed for this
exact L0 artifact in the collection audit (Dice 0.9996302 on a different central 128^3
region, all 421 differences within 0.01), but it does not satisfy the prospectively
chosen gate on this cube.

A post-outcome threshold sweep is reported only to rule out a harmless-looking rescue:
over 0.18..0.22 in steps of 1e-5, the best result is Dice 0.9989875585 at 0.20002 with
194 disagreements, still below the gate. That threshold is not used anywhere.

The exact machine-readable outcome is
[`results/physical_normalization_ab/sentinel_outcome.json`](results/physical_normalization_ab/sentinel_outcome.json).
Any later comparison that uses the released artifact directly must be registered as a
different, operational estimand and must not be described as this causal A/B.
