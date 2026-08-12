# Cross-scan probability to ScrollFiesta adapter

Status: outcome-blind implementation of the downstream interface locked in
`crossscan_scrollfiesta_downstream_lock.json` v2 (content SHA-256
`06142dc819c193a462f37d08a4769024c41ab551411013d40dda72db148457f6`). It does
not authorize candidate inference or a downstream-improvement claim. Those remain contingent on
the registered final result being `POSITIVE_DEPLOYABLE`.

The adapter deliberately ends at ScrollFiesta's documented `cubes_PRED/` and `cubes_RAW/`
interface. It does not duplicate ScrollFiesta's mesher, weld, or audit code.

## Semantic source-label gate

Villa issue #191 exposed a downstream tool that reproducibly hashed the right PHerc1203 archive
while reading its Blosc-compressed chunks as raw label bytes. Byte provenance was correct and the
scientific values were not. Before releasing a model trained from these labels, run the exact
decoded-volume gate:

```powershell
python verify_physical_label_semantics.py `
  --labels-root D:\path\to\physical_truth `
  --output D:\path\to\physical_label_semantic_audit.json
```

The command:

- verifies the two release-v1.0 tarball sizes and SHA-256 values;
- opens each Zarr through its declared codec and requires 3-D `uint8` blocks;
- counts all 30,427,840,512 decoded voxels;
- requires the exact bit-plane and containment census published at upstream commit
  `4277710452578e802c23244a6ab385a048aed100`; and
- creates a content-hashed, no-replace `PASS` receipt.

This gate cites the [upstream census](https://github.com/7jycwjmbfn-eng/pherc0139-physical-audit/blob/4277710452578e802c23244a6ab385a048aed100/results/bit_census.json)
and the [compressed-byte correction](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5261207396).
It does not use the retracted thickness map.

## Export a probability ROI

The input is the locked central `256 x 256 x 256` foreground-probability ROI, not a thresholded
mask. `.npy` is accepted directly. An `.npz` must contain a unique array or use `--npz-key`; a
two-class `2 x 256 x 256 x 256` array selects class 1.

```powershell
python crossscan_scrollfiesta_adapter.py export-zarr `
  --probability candidate_probability.npy `
  --source-receipt final_result.json `
  --origin 3840 3712 1344 `
  --output candidate_probability.zarr
```

The output is OME-NGFF Zarr v2 with axes `z,y,x`, float32 probabilities, `128^3` uncompressed
chunks, unit level-0 scale, and explicit world translation. Every chunk has a sidecar SHA-256
receipt bound to the complete input probability hash. A final immutable receipt binds the exact
chunk universe and a byte-for-byte round trip. Existing output is rejected. `--resume` accepts
only an incomplete store whose metadata and every existing chunk/receipt pair match the current
input; a completed store remains immutable.

## Prepare the locked RAW cubes

Reuse ScrollFiesta PR #12 at commit
`f0d9d2e54823e7ba2460725e81290eead8ed6e5e`, rather than adding another cloud reader. Its existing
`carve_grid_tifs.py` can carve the RAW source while its temporary PRED output is ignored:

```powershell
python <scrollfiesta>\python\scripts\carve_grid_tifs.py `
  --pred-zarr s3://vesuvius-challenge-open-data/PHerc0139/representations/predictions/surfaces/20250728140407-surface-20260413222639-surface-m7-L0-th0.2.zarr `
  --raw-zarr s3://vesuvius-challenge-open-data/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr `
  --bbox 3840 4096 3712 3968 1344 1600 `
  --out locked_raw_carve
```

## Materialize a native grid

Fixed threshold:

```powershell
python crossscan_scrollfiesta_adapter.py make-grid `
  --probability-zarr candidate_probability.zarr `
  --raw-cubes locked_raw_carve\cubes_RAW `
  --arm fixed --threshold 0.2 `
  --output candidate_fixed_grid
```

Matched mass uses the foreground count recorded by the baseline fixed arm:

```powershell
python crossscan_scrollfiesta_adapter.py make-grid `
  --probability-zarr candidate_probability.zarr `
  --raw-cubes locked_raw_carve\cubes_RAW `
  --arm matched-mass --foreground-count <BASELINE_N> `
  --output candidate_matched_grid
```

The matched-mass rule is stable descending probability followed by C-order index for exact ties.
Each grid contains all eight native `128^3` uint8 TIFFs in both trees, with filenames at their
world origins. PRED is exactly 0/255. RAW inputs are shape/dtype checked, copied byte-for-byte,
and hash bound. The grid manifest is content hashed and the output directory is no-replace.

Only after registered promotion may the three frozen arms be passed to the pinned
`grid_pipeline.exe`. Pipeline exit zero is not a topology certificate: the v2 downstream lock
also requires independent OBJ and weld-report checks for non-manifold edges, pinch vertices,
same-direction pairs, world extent, and internal-seam unpaired edges.

## Tests

```powershell
python -m unittest -v `
  test_crossscan_scrollfiesta_adapter.py `
  test_verify_physical_label_semantics.py
```

The focused suite covers OME-Zarr compatibility and round trips, interrupted resume, orphan and
tampered chunks, immutable completion, inclusive thresholding, stable matched mass, exact native
TIFF layout, decoded bit census logic, and rejection of encoded/non-`uint8` input.
