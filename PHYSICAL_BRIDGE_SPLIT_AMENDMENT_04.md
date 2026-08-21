# Physical probability-bridge split: execution amendment 04

Status: **prepared after a terminal unobserved-outcome execution failure and before any
bridge outcome was exposed or inspected**.

The amendment-03 scorer ended naturally when its deterministic per-call allocation estimate
was exhausted in the bounded watershed heap. The exact traceback is preserved in
`results/physical_bridge_split/preoutcome_failure_04.stderr.txt` and bound by the adjacent
machine-readable receipt. Stdout was empty and no result file existed. The failed
`(PHerc1203, 64)` key occurs in both frozen splits, and the anonymous-pipe protocol persisted
no progress record, so the receipt conservatively records development completion, selection,
and holdout opening as unknown. No comparison, selected candidate, partial row, candidate mask,
or bridge outcome was persisted, exposed, or inspected.

Amendment 03 correctly described its capacity formula as an estimate rather than an event
bound. Attempt 8 reached that estimate before reaching its 1,500,000,000-item hard cap. This
amendment removes the estimate from production capacity selection. Every production watershed
call receives one fixed 2,000,000,000-item logical mapping: 64,000,000,000 bytes for the pinned
32-byte event layout. The hard cap is still not claimed to bound live or cumulative events;
exhaustion aborts without a result.

The existing compiled event core and its raw SHA-256 are unchanged. On Windows, the wrapper
creates the mapping with `FSCTL_SET_SPARSE`, `SetFilePointerEx`, `SetEndOfFile`, and writable
`mmap`, avoiding the physical zero-fill performed by this runtime's `os.ftruncate`. The scorer
requires at least the full logical size plus an 8 GiB free-space reserve before each call and
never silently lowers capacity. Sparse allocation does not reserve space against concurrent
writers, so free disk remains an execution requirement. Heap files retain the worker-PID name,
ordinary cleanup, parent cleanup, and dead-PID startup recovery from amendment 03.

A pre-freeze target-volume smoke created and mapped the full 64,000,000,000-byte sparse file in
under 1 millisecond, touched the first, middle, and final byte, allocated 327,680 physical bytes,
and removed the file. Tests verify native sparse marking and sizing, full-capacity selection,
forced exhaustion, cleanup, repeated-label marker regions, randomized differential equivalence,
and the unchanged compiled event stream against scikit-image 0.26.0.

No threshold, candidate, connectivity, split, seed, input, metric, control, row content,
ranking, margin, bootstrap, success gate, or interpretation rule changes. Amendments 01 through
03 and all four failure receipts remain immutable. The amendment-04 lock permits only the same
five execution/publication fields to differ from amendment 03 and requires every other
top-level field to be canonical-JSON identical.
