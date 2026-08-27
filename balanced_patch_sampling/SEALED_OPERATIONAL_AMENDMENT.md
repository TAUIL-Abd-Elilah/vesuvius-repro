# Sealed PHercParis4 scoring operational amendment

Frozen before any held-out score was produced: **2026-08-27T11:20:48Z**.

This amendment does not change the split, checkpoints, arms, seeds, metric,
bootstrap, gates, or held-out patch set in `SEALED_PATCH_PROTOCOL.md`. It
corrects which directory is supplied to SpiralCheck's `--fit-inputs` leakage
audit so that it contains the patches the fitter actually consumed.

## Why the first scoring launch was stopped

The first seed-17 scoring launch supplied all 71,421 directories on the fit
side of the sealed split to `--fit-inputs`. That is broader than the fit's
actual input set because Villa applies its half-open z ROI and patch erosion
while loading.

SpiralCheck loaded the full 17,816-directory held-out side without a load
warning, then encountered this structurally empty fit-side surface:

`auto_grown_20260522202321410_sel_20260522_202629_23`

Its two z samples are approximately 12,230.027 and 12,209.842, both outside
the frozen `[10500, 11500)` fit interval. Villa therefore rejected it in the
z prefilter before decoding x/y and never used it. SpiralCheck correctly would
have refused a leakage audit over the broader directory instead of silently
ignoring the malformed surface. The launch was stopped before it wrote either
a baseline report or any held-out metric. Its export and console log are kept
as an aborted operational attempt, not as a result.

## Exact fit-input audit view

Both seed-17 fit logs record the same deterministic loader accounting:

- 71,421 fit-side directories
- 62,697 rejected by the z ROI prefilter
- 182 moved outside the ROI by the frozen erosion rule
- 8,542 loaded and fitted

The baseline fit's `satisfied_fitted.json` records exactly those 8,542 unique
input IDs. Sort the IDs, join them with `"\n"`, append a final newline, and
hash the UTF-8 bytes. The expected SHA-256 is:

`48ab9630e757cbf6483da0fc9fff8eb8b0410099a93a56fb75d7784adacc1a10`

The treatment began from the same 8,542 IDs. Its later theta-lift rejection of
one patch cannot make an audit of the full initial 8,542-set optimistic; using
the union is conservative for both arms.

Before scoring, the runner will now require an audit view whose directory
names exactly equal that recorded set, require every name to be assigned to
the fit side of the unchanged sealed manifest, and re-hash every view member
against the manifest geometry hash. SpiralCheck remains pinned to commit
`d1b50e2957409a870225fb9f5dcc5e25f7a0f9da` and receives this exact view via
`--fit-inputs` without `--allow-input-load-errors`.

The scored patch directory remains the complete 17,816-directory sealed
held-out side. Both arms will be rerun from their already-frozen checkpoints
into a new output directory. The aborted launch cannot be resumed or reused.

For optimizer seeds 23 and 101, the same procedure applies, and each seed's
fit artifact must independently reproduce the expected input-ID set before
its held-out scoring begins.

## Second implementation note (2026-08-27T11:37:37Z)

Still before any held-out report or metric existed, source review found that
the pinned SpiralCheck implementation treats both z endpoints as inclusive.
The fit and frozen protocol use the half-open interval `[10500, 11500)`.
Therefore the unchanged pinned scorer will receive the equivalent floating
interval `10500,11499.999999999998`, where the upper value is
`nextafter(11500, -infinity)`. This excludes coordinates equal to 11500 while
retaining every representable coordinate below it. Report validation will
require the literal CLI string in `meta.z_range` and the parsed numeric pair in
`heldout_aggregate.z_range`.

The materialization preflight will also independently replay the pinned Villa
loader over all 71,421 fit-side directories, with z range `[10500,11500)`, one
erosion cell, and no UUID filter. It must reproduce exactly 62,697 z-prefilter
drops, 182 post-erosion drops, zero load failures, and the same 8,542-ID set.
For each arm the authoritative consumed set is the union of IDs in
`satisfied_fitted.json` and basenames in `non_liftable_patches.txt`; this keeps
the treatment patch that was consumed before being rejected around step 1,200.
The two arm unions must be identical. Every hard-linked file is checked
against both the sealed manifest's content hash and geometry hash.

## Held-out point-scope freeze (2026-08-27T12:00:09Z)

Before loading either fitted model, a separate model-independent preflight
will strict-load all 17,816 held-out patches with the pinned SpiralCheck source
and record each patch ID plus its number of quad centers in the corrected z
window. It will retain zero-point rows, require zero load failures, and publish
a canonical sorted `patch_id<TAB>n_points<LF>` hash. The scoring runner will
refuse either arm unless its scored/skipped IDs, per-patch point counts, and
total point count exactly match that pre-result scope manifest. It will also
require baseline and treatment to have identical per-patch unseen-point counts
and identical fit-input leakage profiles before running the frozen comparator.
