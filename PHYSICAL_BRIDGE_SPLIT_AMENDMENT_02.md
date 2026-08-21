# Physical probability-bridge split: execution amendment 02

Status: **prepared after a terminal pre-outcome failure and before any bridge outcome was
exposed**.

The amendment-01 block-streaming scorer ended naturally with `MemoryError` inside
scikit-image watershed. The traceback is preserved with canonical LF line endings in
`results/physical_bridge_split/preoutcome_failure_02.stderr.txt` and bound by the adjacent
machine-readable receipt. Development scoring had started but had not completed, development
comparison and selection had not started, holdout was unopened, stdout was empty, and no result
file existed after exit. Partial rows and candidate masks may have existed transiently in memory;
none was persisted or inspected.

The failure came from native allocator high-water memory remaining committed across sequential
blocks. Python reference deletion and garbage collection did not return that memory reliably.
This amendment therefore changes the process lifetime only:

- all 64 inference artifacts and their receipts are still verified before scoring is allowed;
- blocks in the active split are partitioned by the exact `(scroll, score z0)` key already fixed
  in the source manifest;
- one fresh child process scores one complete group at a time, using the unchanged array loader,
  bridge operator, candidate order, exact-mass control, truth preparation, and physical scorer;
- only one child may run at once, and it must exit successfully before the next group starts, so
  its native heap is returned to the operating system;
- a strict hashed request/response contract travels only through anonymous stdin/stdout pipes;
  no partial outcome rows are written to IPC files, and missing, partial, duplicate, unexpected,
  malformed, or non-canonical responses abort without writing a result;
- each request and response binds the protocol-lock hash, implementation commit, and complete
  implementation-file hash map; the child rechecks those files before scoring and before
  returning, and the parent re-verifies the public freeze immediately before result creation;
- returned rows are sorted by the same block ID before the unchanged comparisons, bootstrap,
  development selection, and holdout gates run;
- holdout workers are launched only if development selects a candidate, and receive only that
  selected candidate.

Grouping restores within-z plane reuse without changing the job multiset: the frozen development
split has 19 groups and 304 unique planes, while holdout has 18 groups and 288 unique planes.
Every registered block still appears exactly once in its split and every `(scroll, z, block, k)`
scoring job is unchanged.

No threshold, candidate, connectivity, split, seed, input, metric, control, row content, ranking,
margin, bootstrap, success gate, or interpretation rule changes. Amendment 01 remains immutable.
The amendment-02 lock binds this execution revision and explicitly requires every scientific
top-level field to be canonical-JSON identical to amendment 01.
