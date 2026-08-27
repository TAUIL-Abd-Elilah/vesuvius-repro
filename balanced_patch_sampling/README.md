# PHercParis4 family-balanced patch sampling: reproduction package

This source-only package records how to reproduce the controlled development
screen and the pre-registered sealed comparison. It deliberately contains **no
patch data, fibers, checkpoints, meshes, fit outputs, or outcome JSON**.

## Frozen provenance

- Public data: <https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/>
- Public verified-patch index SHA-256:
  `29f975b1f1615bf7b11e58ec88c77d17c3d7603c4f70e43175e1eb31d65159c2`
  (89,237 patch directories; 84,316 `band-seed*`, 4,921 other; 213 eval-fiber
  JSON files in the observed snapshot).
- Villa code under test (the commit is presently in the contributor fork, not
  upstream):
  [`17dad916c79266f6a19f76abc507bb8b95c63a9b`](https://github.com/TAUIL-Abd-Elilah/villa/commit/17dad916c79266f6a19f76abc507bb8b95c63a9b).
- Sealed evaluator: `Nicodol/spiralcheck`
  [`d1b50e2957409a870225fb9f5dcc5e25f7a0f9da`](https://github.com/Nicodol/spiralcheck/commit/d1b50e2957409a870225fb9f5dcc5e25f7a0f9da).
- This package was prepared from `vesuvius-repro` commit
  `6b757b9e6f9efe2d0f808671cad89c5ad95882a3`; the sampler/evaluator source is
  intentionally external and must be checked out at the Villa commit above.

The two public pre-outcome registrations are the authoritative timestamped
records:

1. [Sealed PHercParis4 patch-sampling comparison](https://gist.github.com/TAUIL-Abd-Elilah/235a47482c6959420ca4592b0e22ed4f)
2. [Family-cap replication addendum](https://gist.github.com/TAUIL-Abd-Elilah/fb185ed973673c382042014891d90b1e)

## What the package contains

- `PREREGISTERED_SCREEN.md`: development-screen protocol frozen before its two
  primary arms ran.
- `SEALED_PATCH_PROTOCOL.md` and `SEALED_REPLICATION_PROTOCOL.md`: verbatim
  frozen protocol copies; their public gists above are the authoritative
  timestamped records.
- `config_screen_*.json`: immutable input overrides for each named screen arm.
- `config_sealed_{baseline,cap075}_seed{17,23,101}.json`: all six frozen
  sealed-fit overrides.
- `fetch_public_inputs.py`: resume-safe downloader for the public inputs.
- `run_screen.ps1`: parameterized runner that writes each arm to a new output
  directory.
- `run_sealed.ps1`: portable, hash-recording sealed fit runner (one arm/seed).
- `run_sealed_spiralcheck.py`, `compare_sealed_patch_reports.py`, and
  `summarize_sealed_replicates.py`: portable export/scoring, strict per-seed
  comparison, and three-seed summarization tooling. No result is stored here.

## Reproduce a development arm

1. Check out Villa at the frozen commit and apply the family-cap sampler change
   under review. Install its documented Spiral dependencies.
2. Fetch public inputs (approximately 3.2 GiB in the observed snapshot):

   ```powershell
   python balanced_patch_sampling/fetch_public_inputs.py --output <public-input-directory>
   ```

3. Run an arm. This runner refuses to reuse an output directory:

   ```powershell
   powershell -ExecutionPolicy Bypass -File balanced_patch_sampling/run_screen.ps1 `
     -Arm baseline -VillaSpiralDir <villa-checkout>/spiral-fitting `
     -PythonExe <python-executable> -DatasetDir <public-input-directory> `
     -OutputRoot <development-output-directory>
   ```

The `cap025` primary treatment retains every patch and sets aggregate draw mass
for `^band-seed` to 0.25. `cap075`, `cap080`, and `noband` are included solely
as named exploratory configurations; they are not part of the original
development decision.

## Sealed reproduction (six fits, then scoring)

Obtain the complete public snapshot, create the sealed split exactly as in
`SEALED_PATCH_PROTOCOL.md`, and use only its fit side for all six fits. The
replication addendum fixes seeds 17, 23, and 101 for both arms.

```powershell
powershell -ExecutionPolicy Bypass -File balanced_patch_sampling/run_sealed.ps1 `
  -Arm baseline -Seed 17 -VillaWorktree <villa-checkout> `
  -PythonExe <python-executable> -FitDatasetDir <sealed-fit-dataset> `
  -SplitManifest <split-manifest.json> -OutputRoot <sealed-output-directory>
```

After both checkpoints for one seed exist, export and score them with explicit
paths; this script verifies the code commit, split, checkpoint configuration,
and report compatibility before producing a comparison:

```powershell
python balanced_patch_sampling/run_sealed_spiralcheck.py `
  --baseline-checkpoint <baseline-checkpoint> --treatment-checkpoint <treatment-checkpoint> `
  --spiral-fitting <villa-checkout>/spiral-fitting --spiralcheck-source <spiralcheck-checkout> `
  --spiralcheck-python <python-executable> --manifest <split-manifest.json> `
  --heldout-patches <heldout-patches> --fit-inputs <fit-side-patches> `
  --umbilicus <umbilicus.json> --output-root <new-score-output-directory> `
  --optimizer-seed 17
```

Run the comparator self-test before a new environment and summarize the three
per-seed `sealed_comparison.json` files only after all six fits are complete.

## Interpretation limits

The screen disables non-patch annotation and dense-volume inputs. Public
`eval_fibers/` are not direct fit inputs, but share upstream provenance and may
overlap geometry. They are a geometric-consistency diagnostic, not independent
ground truth, absolute-winding accuracy, or a production whole-scroll result.

## Sealed results: intentionally blank

Do not fill this section from a development screen. After a sealed run, record
the split-manifest hash, exact commands and commits, meshes, score artifacts,
leakage audit, all seed-level metrics, and failures here or in a separately
versioned result document.

- Split manifest: `TBD after sealed split creation`
- Baseline artifacts: `TBD`
- Treatment artifacts: `TBD`
- Sealed score / decision: `TBD`
