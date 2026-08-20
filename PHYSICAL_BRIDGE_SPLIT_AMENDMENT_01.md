# Physical probability-bridge split: execution amendment 01

Status: **published before any development score or physical bridge outcome**.

The original all-block scorer failed with `MemoryError` inside scikit-image watershed while
`build_masks` was still constructing development masks. The traceback text is preserved with
canonical LF line endings in `results/physical_bridge_split/preoutcome_failure_01.stderr.txt` and
bound by the machine-readable failure receipt beside it; that receipt also records the exact
observed byte hash. `score_blocks` had not been called, development selection had not
started, holdout was unopened, stdout was empty, and no result file existed after exit. Partial
candidate masks may have existed transiently in memory before the exception, but none was
persisted or inspected.

This amendment changes execution lifetime only:

- all 64 inference artifacts are still hash-checked and decoded before scoring is allowed, but
  each decoded validation object is released immediately;
- development and any conditionally opened holdout are then loaded, transformed, and scored one
  registered block at a time;
- each block still uses the unchanged bridge operator, fixed baseline, candidate order,
  exact-mass control, physical scorer, and audit structure;
- rows are sorted by the same block ID before the unchanged comparisons, bootstrap, selection,
  and gates run.

No threshold, candidate, connectivity, split, seed, input, metric, control, ranking, margin,
bootstrap, success gate, or interpretation rule changes. The original protocol lock remains
immutable. A new amendment lock binds this execution revision and supersedes it only for running
the scorer.
