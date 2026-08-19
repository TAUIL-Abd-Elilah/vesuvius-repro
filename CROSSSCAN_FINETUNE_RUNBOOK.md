# Cross-scan fine-tuning runbook

This runbook executes the public pre-outcome protocol. The commands fail closed on a dirty or
unpushed branch, changed plan/code/input hash, changed nnU-Net tree or Python package version,
missing pilot gate, partial output, or attempted overwrite.

Use a CUDA environment in which villa's nnU-Net fork imports successfully. The frozen local run
uses Python 3.14.6, Torch 2.13.0+cu126, CUDA 12.6, and an RTX 3090. Reserve roughly 50 GB for raw
crops, preprocessed arrays, checkpoints, predictions, and receipts. Generated datasets,
checkpoints, and label-derived results are CC BY-NC 4.0.

The examples use PowerShell variables only for readability:

```powershell
$workspace = 'D:\path\to\workspace'
$repo = Join-Path $workspace 'vesuvius-repro-crossscan-finetune'
$villa = Join-Path $workspace 'villa'
$labels = Join-Path $workspace 'physical_truth'
$model = Join-Path $workspace 'model_m7'
$data = Join-Path $workspace 'crossscan_finetune_v1'
$python = 'C:\path\to\python.exe'
Set-Location $repo
```

## 1. Verify the public lock

Run all offline tests and verify every local model/label/source hash:

```powershell
& $python -m unittest -v test_crossscan_finetune.py test_run_crossscan_finetune.py test_score_crossscan_finetune.py
& $python crossscan_finetune.py verify results/crossscan_finetune/plan.json --labels-root $labels --source-manifest results/physical_normalization_ab/manifest.json --model-dir $model
```

The execution lock includes `results/crossscan_finetune/preprocess_smoke.json`, produced by a
real one-case run of villa's fingerprint extractor and default preprocessor with the released m7
plans. To reproduce that integration check without overwriting the public receipt:

```powershell
$smoke = Join-Path ([IO.Path]::GetTempPath()) ("crossscan-smoke-" + [guid]::NewGuid() + ".json")
& $python crossscan_preprocess_smoke.py --villa-root $villa --model-dir $model --out $smoke
```

The maintainer creates the execution lock only after its implementation commit is public. Once
results/crossscan_finetune/execution_lock.json is committed, every command below verifies it.

## 2. Materialize and preprocess outcome-blind data

This reads the fixed public CT boxes and local released labels. It creates 288 nnU-Net cases and
96 cached evaluation contexts. It does not load a model or prediction.

```powershell
& $python run_crossscan_finetune.py materialize --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --scope all
& $python run_crossscan_finetune.py preprocess --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --num-processes 4
```

Both stages are resumable by content-hashed per-case receipts. A mismatching partial case is an
error, not silently overwritten. Before preprocessing, the runtime creates the standard nnU-Net
fingerprint from all 288 cases with the fixed 100,000,000-voxel sampling budget. Its intent and
receipt bind the training receipts, while the released m7 plans remain frozen (no replanning).
Preprocessing hashes its complete case dataset; every training run verifies the fingerprint and
preprocessing chain before loading a batch.

## 3. Run the pilot only

First cache the initial model on the 32 pilot cubes:

```powershell
& $python run_crossscan_finetune.py infer --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --kind initial --scope pilot
```

Train seed 39 for 2,000 steps in both complementary folds, then infer only their held-out pilot
cases:

```powershell
foreach ($fold in @('even','odd')) {
  & $python run_crossscan_finetune.py train --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --seed 39 --steps 2000 --fold $fold
  & $python run_crossscan_finetune.py infer --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --kind finetuned --scope pilot --seed 39 --steps 2000 --fold $fold
}
& $python score_crossscan_finetune.py pilot --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --steps 2000
```

If the 2,000-step scorer returns PASS, `pilot_verdict.json` fixes the preregistered 4,000 steps for
every inferential run. If it returns RETRY_REQUIRED, repeat both folds and pilot inference at
4,000 steps, then score once at 4,000; a pass there fixes the same 4,000-step inferential recipe.
The runtime refuses the retry unless the 2,000-step decision authorizes it. A second failure writes
TARGET-UNLEARNABLE and blocks every primary command.

## 4. Run all six inferential seeds

Set the step count to the content-hashed pilot verdict. Training a seed with any other count is
refused.

```powershell
$steps = (Get-Content -Raw (Join-Path $data 'pilot_verdict.json') | ConvertFrom-Json).selected_steps
if ($steps -ne 4000) { throw 'pilot verdict did not select the frozen inferential recipe' }
foreach ($seed in 40..45) {
  foreach ($fold in @('even','odd')) {
    & $python run_crossscan_finetune.py train --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --seed $seed --steps $steps --fold $fold
  }
}
```

Only after that pilot PASS, create the initial primary/safety predictions and every held-out
fine-tuned prediction:

```powershell
& $python run_crossscan_finetune.py infer --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --kind initial --scope primary
& $python run_crossscan_finetune.py infer --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --kind initial --scope safety
foreach ($seed in 40..45) {
  foreach ($fold in @('even','odd')) {
    & $python run_crossscan_finetune.py infer --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --kind finetuned --scope primary --seed $seed --steps $steps --fold $fold
    & $python run_crossscan_finetune.py infer --villa-root $villa --labels-root $labels --model-dir $model --data-root $data --kind finetuned --scope safety --seed $seed --steps $steps --fold $fold
  }
}
```

Primary routing uses exactly one cross-fitted model per block. Safety routing deliberately runs
both fold models; their probabilities are averaged within seed before scoring.

## 5. Score once and publish every bucket

```powershell
& $python score_crossscan_finetune.py final --villa-root $villa --labels-root $labels --model-dir $model --data-root $data
```

The scorer validates every array, coordinate, checkpoint, and receipt, writes final_result.json
only after all eight fixed composite panels succeed, and refuses a second verdict. Each composite
shows all six seeds and their mean with separate additions/removals; the result hashes every
panel. Publish the pilot attempt(s), pilot verdict, training/inference receipts, six-seed table,
block metrics, figures, and final result whether the bucket is positive, null, regression,
underpowered, or target-unlearnable.
