# Protocol amendment 03: Windows-safe blending entrypoint

Date: 2026-08-11. This is a technical implementation amendment. No sentinel Dice, physical
score, matched-mass threshold, or arm comparison existed.

## Trigger

Implementation r2 successfully ran all 150 frozen PHerc0139 inference patches under public
manifest `20e3349c08f30df357eaba8f83249bef794fb972f15f53c6bafeaaff54127a98`.
Blending then failed before processing its first output chunk:

`PicklingError: Can't pickle <function _init_worker ...>: it's not found as __main__._init_worker`.

The Zarr compatibility bootstrap used `runpy.run_module(..., run_name='__main__')` for both
entrypoints. That is harmless for inference, but on Windows the blend process pool must pickle
module-level worker functions. Naming those functions as members of `__main__` made them
unresolvable in spawned children.

Attempt 3 produced a complete 2.64 GiB logits tree (150 patches) but only the two metadata files
of the merged store. It produced no extracted probability extent and no Dice. Exact hashes are
in `results/physical_normalization_ab/sentinel_failure_02.json`.

## Repair

Implementation r3 retains the identical Zarr bootstrap but imports
`vesuvius.models.run.blending` by its canonical package name and calls `blending.main()`.
Therefore `_init_worker` remains
`vesuvius.models.run.blending._init_worker`, which Windows spawn can import. The explicit
`--num_workers` value is the same runner setting already frozen for inference.

A regression test executes the exact blend bootstrap against a temporary package whose main
function uses a spawn-context process pool; it must successfully pickle and execute a
module-level worker. A no-score diagnostic on attempt 3's completed logits also returned zero
with the canonical import. For strongest provenance, the formal sentinel does not reuse that
intermediate: its next attempt starts again from the frozen CT ROI.

## Frozen invariants

No coordinate, input, checkpoint, villa commit, normalization, threshold, metric, bootstrap
seed, decision gate, or visual rule changed. Both failed attempts remain immutable and the
replacement manifest must again preserve all 64 blocks and all inputs byte-for-byte.
