# Physical probability-bridge split: execution amendment 05

Status: **prepared after a terminal unobserved-outcome execution failure and before any
bridge outcome was exposed or inspected**.

The amendment-04 scorer ended naturally when the fixed 2,000,000,000-item watershed heap was
exhausted in the `(PHerc1203, 960)` worker. The exact traceback is preserved in
`results/physical_bridge_split/preoutcome_failure_05.stderr.txt` and bound by the adjacent
machine-readable receipt. Stdout was empty and no result file existed. That worker key occurs
in both frozen splits, and the anonymous-pipe protocol persisted no progress record, so the
receipt conservatively records development completion, selection, and holdout opening as
unknown. No comparison, selected candidate, partial row, candidate mask, or bridge outcome was
persisted, exposed, or inspected.

This amendment changes only the exact heap representation. The event `index` and `source`
payloads change from native signed pointer width to signed 32-bit integers. `value` remains
float64, `age` remains signed int32 with the pinned scikit-image wrap behavior, and heap counts,
positions, index arithmetic, and the age accumulator remain native `Py_ssize_t`. Natural
alignment gives a 24-byte item with fields at byte offsets 0, 8, 12, and 16. The comparator,
event state machine, marker order, neighbor order, cost propagation, and watershed-line logic
are unchanged.

The wrapper checks the padded raveled element count before allocating the heap, and the compiled
core repeats that check before any narrowing cast. It allows at most 2,147,483,648 elements, so
the largest valid flat index is signed-int32 maximum. Every frozen probability block has 737,280
voxels and 790,152 elements after the one-voxel 3D watershed padding, with maximum flat index
790,151. Component crops are no larger. The narrowed payload is therefore lossless for every
frozen production call.

Every production call receives 2,666,666,666 item slots. At 24 bytes each, the logical sparse
mapping is 63,999,999,984 bytes: 16 bytes smaller than amendment 04 while providing 33.3% more
live event slots. The scorer requires that mapping plus an 8 GiB free-space reserve and never
silently lowers capacity. This capacity is not claimed to bound live or cumulative events;
exhaustion still aborts without a result. Native sparse allocation does not reserve disk space,
so free disk remains an execution requirement.

The compiled extension is rebuilt and pinned at raw SHA-256
`90d1a8d2b444922458c963477481eebe66c47abbcd82aab41e57268e21ac3611`.
Pre-freeze synthetic validation
confirmed the 24-byte layout and exact offsets, the signed-int32 boundary in both wrapper and
compiled core, 5,000 random direct-core partitions against scikit-image 0.26.0, random wrapper
partitions, repeated-label markers, nonzero compactness, bridge masks and audit records,
byte-identical repeats, forced exhaustion, and cleanup. A target-size Windows smoke mapped the
63,999,999,984-byte sparse file, touched its first, middle, and final bytes, allocated 196,608
physical bytes, and removed it in under 0.004 seconds.

No threshold, candidate, connectivity, split, seed, source input, metric, matched-budget control,
ranking rule, margin, bootstrap, success gate, or interpretation changes. This amendment is
valid only if its implementation commit and a new amendment-05 protocol lock are public before
scoring resumes.
