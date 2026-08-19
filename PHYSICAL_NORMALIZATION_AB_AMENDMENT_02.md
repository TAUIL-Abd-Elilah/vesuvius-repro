# Protocol amendment 02: pre-prediction Zarr writer compatibility

Date: 2026-08-11. This is a technical implementation amendment. No broken-path prediction,
corrected prediction, physical score, matched-mass threshold, or arm comparison existed.

## Trigger

The first real sentinel invocation used public manifest
`ef84062fab6adf29539e67bd6f0ac9e2354c2850dfd60e615a95eec26123a703` at public commit
`e1b5d8107c2176724ef31434f8909c382cc5c69d`. It loaded the frozen m7 checkpoint and enumerated
the fixed PHerc0139 ROI, then failed before creating an output store or evaluating a patch:

`AttributeError: module 'zarr' has no attribute 'Blosc'`.

The attempt's `logits` directory contained zero files. Machine-readable provenance is in
`results/physical_normalization_ab/sentinel_failure_01.json`.

## Cause and repair

The pre-fix villa commit used by the baseline sentinel predates villa's Zarr-3 output-store
repair. It calls the Zarr-2-era `zarr.Blosc` re-export, and after restoring that name Zarr 3
must also be told to create format-2 arrays when given a v2 `numcodecs` compressor. Both
behaviors are native under Zarr 2, the environment in which the published pipeline originally
ran.

Implementation r2 therefore launches inference and blending through one hashed bootstrap
that:

1. aliases `zarr.Blosc` to `numcodecs.Blosc` only when the re-export is absent; and
2. adds `zarr_format=2` only when a compressor is supplied without an explicit format.

The bootstrap is applied identically to the broken and corrected arms. A subprocess regression
test creates a compressed store under the current Zarr 3 environment and asserts that its
metadata is format 2 with a Blosc compressor.

## Frozen invariants

No coordinate, input, model checkpoint, villa commit, normalization, threshold, metric,
bootstrap seed, decision gate, or visual rule changed. The failed attempt remains immutable;
the replacement public manifest must preserve all 64 block records and every input byte.
