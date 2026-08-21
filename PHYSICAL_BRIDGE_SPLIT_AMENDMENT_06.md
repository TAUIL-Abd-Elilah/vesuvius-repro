# Physical probability-bridge split: final execution amendment 06

Status: **prepared after a terminal unobserved-outcome execution failure and before any
bridge outcome was exposed or inspected**.

The amendment-05 scorer ended naturally when its fixed 2,666,666,666-item watershed heap was
exhausted in the `(PHerc1203, 960)` worker. The exact traceback is preserved in
`results/physical_bridge_split/preoutcome_failure_06.stderr.txt` and bound by the adjacent
machine-readable receipt. Stdout was empty and no result file existed. That worker key occurs
in both frozen splits, and the anonymous-pipe protocol persisted no progress record, so the
receipt conservatively records development completion, selection, and holdout opening as
unknown. No comparison, selected candidate, partial row, candidate mask, or bridge outcome was
persisted, exposed, or inspected.

This is the final capacity amendment for this protocol. It changes only the representation of
the exact noncompact watershed event cost. Every frozen call uses `compactness == 0`. In that
case each event cost is exactly one value from the private padded float64 image: a marker starts
with its own image index; propagation inherits the parent's cost index only when the neighbor's
image value is strictly smaller, otherwise it keeps the neighbor index. By induction this is
the same float64 value assigned by the pinned scikit-image state machine. The comparator reads
that immutable image value and applies the unchanged value-then-signed-age comparison. No cost
is rounded or recomputed.

Each naturally aligned event is therefore four signed-int32 fields: `cost_index`, `age`,
`index`, and `source`, at byte offsets 0, 4, 8, and 12. Heap counts, positions, index arithmetic,
and the age accumulator remain native `Py_ssize_t`; stored age retains the pinned signed-int32
wrap behavior. The existing padded-element guard covers all three index payloads and permits at
most 2,147,483,648 elements. Every frozen probability block has 737,280 voxels and 790,152
elements after one-voxel 3D padding. Both the wrapper and compiled core reject nonzero
compactness, with the wrapper rejecting it before heap allocation. Frozen probability inputs
are finite by the unchanged operator validation.

Every production call receives 4,000,000,000 slots. At 16 bytes each, the native sparse mapping
is exactly 64,000,000,000 bytes and requires an additional 8 GiB free-space reserve. This is
50% more slots than amendment 05 but is still not claimed to bound live or cumulative events.
Sparse allocation does not reserve disk space. If this heap is exhausted, scoring aborts
without a result and this protocol terminates: there will be no further capacity amendment.

The compiled extension is pinned at raw SHA-256
`93dfd77cc857cfa1d67e5dd1f2d1865aae8f010a5ff309c999eb2ccf6dd7841a`.
Pre-freeze synthetic validation covers the 16-byte layout and offsets, signed-int32 boundary,
direct comparator and shared propagation selector including adjacent float64 values, equal
values, signed zero, NaN, and signed-age extremes, 5,000 random direct-core partitions against
scikit-image 0.26.0, random wrapper partitions, repeated-label markers, bridge masks and audit
records, byte-identical repeats, forced exhaustion, compactness rejection, and cleanup. A
target-size Windows smoke mapped the exact 64,000,000,000-byte sparse file, touched its first,
middle, and final bytes, allocated 196,608 physical bytes, and removed it in under 0.005 seconds.

No threshold, candidate, connectivity, split, seed, source input, metric, matched-budget control,
ranking rule, margin, bootstrap, success gate, or interpretation changes. This amendment is
valid only if its implementation commit and an amendment-06 protocol lock are public before
scoring resumes.
