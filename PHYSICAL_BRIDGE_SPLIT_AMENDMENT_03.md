# Physical probability-bridge split: execution amendment 03

Status: **prepared after a terminal unobserved-outcome execution failure and before any
bridge outcome was exposed or inspected**.

The amendment-02 scorer ended naturally with `MemoryError` in the scikit-image watershed
heap. The exact traceback is preserved in
`results/physical_bridge_split/preoutcome_failure_03.stderr.txt` and bound by the adjacent
machine-readable receipt. Stdout was empty and no result file existed. The failed
`(PHerc1203, 960)` key occurs in both frozen splits, and the anonymous-pipe protocol persisted
no progress record, so the receipt conservatively records development completion, selection,
and holdout opening as unknown. Comparisons, a selected candidate, and partial rows may have
existed only in process memory; none was persisted, exposed, or inspected.

The failure was an allocator discontinuity rather than a requested scientific change.
scikit-image 0.26 stores each 32-byte watershed event plus an 8-byte pointer in a contiguous
binary heap and doubles both arrays. Crossing 1,048,576,000 live events therefore requests an
immediate additional allocation of roughly 39 GiB even when only one more event is needed.

This amendment retains the complete amendment-02 execution protocol and changes only that
heap representation:

- the scikit-image 0.26 watershed state machine is transcribed without changing its event
  fields, float cost, signed 32-bit age, comparator, marker order, neighbor order, push/pop
  order, cost propagation, output-on-pop timing, or watershed-line rule;
- heap payloads are swapped directly in the same logical array positions instead of swapping
  pointers to a second payload array;
- the direct-item array is backed by a temporary memory-mapped file, so it does not require a
  contiguous doubling RAM allocation;
- capacity is fixed before each call by the public allocation-estimate formula
  `min(1,500,000,000, max(1,000,000, mask voxels * neighbor count * maximum marker label +
  marker voxels))`. This estimate is not claimed to bound live or cumulative events; reaching
  its 48,000,000,000-byte hard cap aborts without a result. The pinned signed-32-bit age cast,
  including wrap, is unchanged. Insufficient scratch space, malformed storage, binary drift,
  or cleanup failure also aborts without a result;
- the compiled Windows/Python 3.14 extension is committed and bound by raw SHA-256. Its Cython
  source, build script, wrapper, differential tests, and exact binary hash are also bound;
- only one fresh scroll/z0 worker runs at a time. Heap files use a worker-PID prefix, are removed
  by the wrapper on every ordinary exit, and are independently removed by the parent after the
  worker has terminated. A later scorer start removes strict-name leftovers only after verifying
  that their owning PID has exited.

Differential tests compare the complete partition, seam, final mask, and audit against the
pinned scikit-image function on plateau-heavy randomized 2-D and 3-D cases, all supported
connectivities, forced capacity failure, repeated runs, and the existing bridge-split fixtures.

No threshold, candidate, connectivity, split, seed, input, metric, control, row content,
ranking, margin, bootstrap, success gate, or interpretation rule changes. Amendments 01 and 02
and all three failure receipts remain immutable. The amendment-03 lock permits only the same
five execution/publication fields to differ from amendment 02 and requires every other
top-level field to be canonical-JSON identical.
