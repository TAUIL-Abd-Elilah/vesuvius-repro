# PHercParis4 family-cap replication addendum

Published while the first sealed comparison was still in its byte-level split
verification phase and before either sealed mesh had been exported or scored.
No sealed outcome was available when these rules were written.

## Frozen inputs

- Villa code: `17dad916c79266f6a19f76abc507bb8b95c63a9b`.
- SpiralCheck: `d1b50e2957409a870225fb9f5dcc5e25f7a0f9da`.
- Split manifest SHA-256:
  `9a1b226ebde3854728adfeb6f21513026c0cc49d948cc52684d6ee96e3819f31`.
- The original protocol, arms, z range, 5,000-step budget, plain-mesh export,
  leakage audit, metrics, and three gates remain unchanged.

## Added fixed seeds

Run both baseline and cap-0.75 arms at optimizer seeds `23` and `101`, in
addition to the already-running seed `17`. Run and report all four additional
fits regardless of the seed-17 result; do not replace a failed seed or change
the cap after seeing sealed scores.

Each seed is evaluated separately on the same sealed held-out split. Report
every seed's primary delta, paired within-family 95% bootstrap interval,
band-seed delta, and pooled unseen within-6-voxel delta.

## Replication interpretation

A repeatability claim requires all three seed-level primary point estimates to
be positive and all three seed-level noninferiority gates to pass. The original
seed-17 frozen gate remains the sole confirmatory test; seeds 23 and 101 measure
training variability and are not treated as additional independent held-out
datasets. Report the mean and full range across seeds without hiding failures.
