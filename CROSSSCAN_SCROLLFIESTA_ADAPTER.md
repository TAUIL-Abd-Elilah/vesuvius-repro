# Cross-scan probability to ScrollFiesta adapter

Status: implementation of the outcome-blind downstream protocol in
`crossscan_scrollfiesta_downstream_lock.json` v2 (content SHA-256
`06142dc819c193a462f37d08a4769024c41ab551411013d40dda72db148457f6`). Candidate
execution and any improvement claim remain forbidden unless the registered terminal result is
`POSITIVE_DEPLOYABLE`. Implementation details are frozen by
`crossscan_scrollfiesta_metric_lock.json` (content SHA-256
`70c29b370b1f6ca2bb7f6d78eb284e456187056d2ed7efb86c7b5950e976f42c`).

The adapter ends at ScrollFiesta's documented `cubes_PRED/` and `cubes_RAW/` interface. The
separate downstream executor invokes and audits the pinned ScrollFiesta mesher rather than
duplicating it.

## Trust chain

A `PASS` grid set proves this complete pre-meshing chain:

1. The two physical-label tarballs match their pinned identities, every extracted compressed-Zarr
   byte matches the tar member, and all 30,427,840,512 voxels match the codec-decoded upstream
   census.
2. The RAW/inference context is the fixed public PHerc0139 level-0 Zarr window
   `[3776:4160,3648:4032,1280:1664]`. Its canonical uint8 array SHA-256 is
   `b654ba8428beb5f24378efb2d9e8f5d516cec60410871682bdbff5535b6b665f`; the
   canonical array SHA-256 of each of its eight retained RAW cubes is also pinned and rechecked
   from every final TIFF.
3. Baseline inference rehashes the released m7 fold-0 checkpoint. Candidate inference rehashes
   the promoted release manifest and every one of its twelve checkpoints. Both use the exact m7
   normalization, tile step `0.5`, Gaussian blending, and no mirroring. Manifest paths must be the
   exact checkpoint/plans/dataset paths the runner opens. Candidate accumulation is a float64
   arithmetic mean of exactly twelve class probabilities.
4. The retained `256^3` probability, every Zarr chunk, all eight RAW TIFFs, all 24 PRED TIFFs, the
   positive result, execution lock, semantic audit, model source, and the complete local Python
   import closure are content/hash bound.
5. A final set verifier requires exactly `baseline-fixed`, `candidate-fixed`, and
   `candidate-matched-mass`; identical RAW, context, promotion, and semantic sources; the same
   candidate probability in both candidate arms; identical pinned software, nnU-Net tree, CUDA
   device/driver, determinism settings, and tooling bytes across both inferences; and byte-for-byte
   equality between all PRED TIFFs and masks recomputed from the verified probability stores.

The RAW window fingerprint is an input-integrity addition, not an outcome-selected region or a
changed acceptance gate. The region and all scientific gates remain those frozen in v2.

## 1. Validate label semantics

```powershell
python verify_physical_label_semantics.py `
  --labels-root D:\path\to\physical_truth `
  --output D:\evidence\physical_label_semantic_audit.json
```

The validator cites the corrected upstream census commit
`4277710452578e802c23244a6ab385a048aed100` and the
[Villa issue #191 correction](https://github.com/ScrollPrize/villa/issues/191#issuecomment-5261207396).
It rejects the compressed-byte failure mode; it does not use the retracted thickness map.

The model release command requires this receipt explicitly:

```powershell
python export_crossscan_release.py `
  --villa-root D:\path\to\villa `
  --labels-root D:\path\to\physical_truth `
  --model-dir D:\path\to\surface_m7 `
  --data-root D:\path\to\completed_run `
  --semantic-audit D:\evidence\physical_label_semantic_audit.json `
  --out D:\release\crossscan_model
```

## 2. Carve one shared public CT source

```powershell
python crossscan_scrollfiesta_adapter.py carve-raw `
  --output locked_pherc0139_raw
```

The source URI, level, metadata, context/grid boxes, and array hash are non-configurable. The
command writes one `context.npy`, exactly eight native `128^3` RAW TIFFs, and a no-replace receipt.
The verifier reopens every TIFF and compares it to the correct slice of the pinned context.

## 3. Run the two locked inferences

Both commands require the same positive terminal receipt, semantic audit, and original pre-outcome
execution lock. `--villa-root` must be the clean pinned Villa commit
`94ba215963afb6216e380fe2c86131fa5e724c3b` with nnU-Net tree
`24941bfa19e7239db6458287c2a39b9ad4bd7f4a`. The runner rejects any different locked package
versions, forces `nnUNet_compile=0`, `cudnn.benchmark=False`, and
`cudnn.deterministic=True`, explicitly locks the source environment's TF32/matmul flags, and
records GPU UUID/driver identity. `--model-role` only selects the
execution path; subsequent verifiers derive and recheck the role from the model bytes and receipt.

```powershell
python run_crossscan_scrollfiesta_inference.py `
  --model-role baseline `
  --model-source D:\models\surface_m7 `
  --raw-carve locked_pherc0139_raw `
  --promotion-receipt D:\completed_run\final_result.json `
  --execution-lock results\crossscan_finetune\execution_lock.json `
  --villa-root D:\path\to\pinned-villa `
  --semantic-audit D:\evidence\physical_label_semantic_audit.json `
  --output baseline_inference

python run_crossscan_scrollfiesta_inference.py `
  --model-role candidate `
  --model-source D:\release\crossscan_model `
  --raw-carve locked_pherc0139_raw `
  --promotion-receipt D:\completed_run\final_result.json `
  --execution-lock results\crossscan_finetune\execution_lock.json `
  --villa-root D:\path\to\pinned-villa `
  --semantic-audit D:\evidence\physical_label_semantic_audit.json `
  --output candidate_inference
```

Each output contains the exact float32 probability, complete provenance receipts, and copies of
all eight local modules imported by the inference/validation path. Reverification deliberately
also requires the external model source and shared RAW carve so their full bytes—not merely an
embedded assertion—are rechecked.

## 4. Export immutable probability stores

```powershell
python crossscan_scrollfiesta_adapter.py export-zarr `
  --inference-run baseline_inference `
  --model-source D:\models\surface_m7 `
  --raw-carve locked_pherc0139_raw `
  --output baseline_probability.zarr

python crossscan_scrollfiesta_adapter.py export-zarr `
  --inference-run candidate_inference `
  --model-source D:\release\crossscan_model `
  --raw-carve locked_pherc0139_raw `
  --output candidate_probability.zarr
```

The OME-NGFF Zarr v2 stores use axes `z,y,x`, float32, `128^3` uncompressed chunks, unit level-0
scale, and fixed translation `[3840,3712,1344]`. Chunk sidecars and the final receipt bind the
entire probability and provenance universe. `--resume` accepts only a matching incomplete export;
a completed store is immutable.

## 5. Materialize and jointly verify all three grids

```powershell
python crossscan_scrollfiesta_adapter.py make-grid `
  --probability-zarr baseline_probability.zarr `
  --raw-carve locked_pherc0139_raw `
  --arm baseline-fixed `
  --output baseline_fixed_grid

python crossscan_scrollfiesta_adapter.py make-grid `
  --probability-zarr candidate_probability.zarr `
  --raw-carve locked_pherc0139_raw `
  --arm candidate-fixed `
  --output candidate_fixed_grid

python crossscan_scrollfiesta_adapter.py make-grid `
  --probability-zarr candidate_probability.zarr `
  --raw-carve locked_pherc0139_raw `
  --arm candidate-matched-mass `
  --baseline-fixed-manifest baseline_fixed_grid\manifest.json `
  --output candidate_matched_grid

python crossscan_scrollfiesta_adapter.py verify-grid-set `
  --baseline-probability-zarr baseline_probability.zarr `
  --candidate-probability-zarr candidate_probability.zarr `
  --baseline-fixed-grid baseline_fixed_grid `
  --candidate-fixed-grid candidate_fixed_grid `
  --candidate-matched-grid candidate_matched_grid `
  --output crossscan_scrollfiesta_grid_set.json
```

The fixed rule is inclusive `probability >= 0.2`. Matched mass derives `N` from the fully verified
baseline grid; callers cannot provide it. All TIFFs are uint8, PRED is exactly 0/255, and every
output is create-no-replace.

Only the set receipt may advance to the pinned `grid_pipeline.exe`. Exit zero is not a topology
certificate: the downstream lock additionally requires independent OBJ/weld checks for
non-manifold edges, pinch vertices, same-direction pairs, world extent, and internal-seam
unpaired edges.

## 6. Execute and seal the three-arm downstream result

This step is Windows-only because the registered consumer artifacts are pinned `.exe` files. It
must not run for the candidate unless the upstream registered outcome is `POSITIVE_DEPLOYABLE`.

```powershell
python run_crossscan_scrollfiesta_downstream.py run `
  --grid-set-receipt crossscan_scrollfiesta_grid_set.json `
  --baseline-probability baseline_probability.zarr `
  --candidate-probability candidate_probability.zarr `
  --baseline-grid baseline_fixed_grid `
  --candidate-fixed-grid candidate_fixed_grid `
  --candidate-matched-grid candidate_matched_grid `
  --labels-root D:\path\to\physical_truth `
  --semantic-audit D:\evidence\physical_label_semantic_audit.json `
  --binary-dir D:\path\to\ScrollFiesta\build\Release `
  --renderer-script D:\path\to\ScrollFiesta\scripts\render_mesh.py `
  --metric-lock crossscan_scrollfiesta_metric_lock.json `
  --output crossscan_scrollfiesta_downstream

python run_crossscan_scrollfiesta_downstream.py verify `
  --output crossscan_scrollfiesta_downstream `
  --expected-content-sha256 <published-terminal-content-sha256>
```

Before execution, the runner validates the pinned executables, renderer, physical truth, grid-set
receipt, and exact production Python/NumPy/SciPy stack. It then makes private verified snapshots
of all grids, tools, and truth; both scoring and meshing consume those same immutable bytes. The
child environment is a minimal allowlist, each pipeline has a 12-hour timeout, and each renderer
has a 2-hour timeout. A failure or timeout in one arm is sealed while the other arms still run.

Every arm requires exactly eight clean summaries and logs, all final cube meshes, a decodable
fixed-camera PNG, and an independently parsed final OBJ/weld report. The physical scorer applies
the frozen component and surface-distance gates; the OBJ auditor independently recomputes edge
incidence, world span, and internal-seam counts. Fixed centre cross-sections and mesh views cannot
be replaced by outcome-selected panels.

The terminal receipt binds the exact regular-file universe and rejects symlinks, junctions, and
special entries. Verification recomputes every physical score and PASS-arm mesh audit from the
staged inputs. Supplying the externally published terminal digest prevents a self-consistent
receipt rewrite. A scientifically valid sealed negative result returns exit code 1; preserve and
publish it rather than rerunning selectively.

## Tests

```powershell
python -m unittest -v `
  test_crossscan_release.py `
  test_run_crossscan_scrollfiesta_inference.py `
  test_crossscan_scrollfiesta_adapter.py `
  test_crossscan_scrollfiesta_metrics.py `
  test_crossscan_scrollfiesta_obj.py `
  test_run_crossscan_scrollfiesta_downstream.py `
  test_verify_physical_label_semantics.py
```

The tests include minimal/fabricated promotion rejection, exact executable model-path and byte
revalidation, false semantic-fraction rejection, RAW/checkpoint/chunk tampering, strict float32
inference-file semantics, real Blosc tar/Zarr decoding, deterministic runtime setup, centered
baseline cropping, exact twelve-member probability averaging, stable matched mass, cross-arm
source/runtime/tool equality, and recomputation of all three masks from the final Zarr stores.
The downstream tests additionally cover exact metric boundaries, split/merger matching,
independent edge/seam auditing, malformed OBJ/report rejection, poison-environment removal,
command timeouts, corrupt visual rejection, exact artifact hashes, and a self-consistent forged
PASS. Hosted CI runs these contract and fixture tests on Linux and Windows; it cannot execute the
real pipeline because the three pinned external executables are not stored in this repository.
