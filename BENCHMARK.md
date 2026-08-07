# A surface-model benchmark on the 892 public volumes, split by provenance

Scoring code: [`surface_bench.py`](surface_bench.py). Null controls: `--validate`, no model or
GPU required.

## ⚠ Read this before quoting any number

**This is not a held-out benchmark.** m7's own `dataset.json` reports `numTraining: 786` with
shapes `[320, 314, 314]` — the same scale and label scheme as this 892-volume set. The close
counts are strong evidence of substantial overlap, but the fingerprint does not expose public
sample identifiers, so **exact membership is unknown**. Two volumes (`sample_00853`,
`sample_00854`) are provably outside it, because `hash_public_volumes.py` finds their crop
shape nowhere in the fingerprint. Treat m7's numbers here as *training-set-contaminated* and
use them as a reference point, not as evidence of generalisation. A model you train yourself
and score here has no such problem, provided you respect the split.

**Class 2 is the majority of a volume.** The labels are `{0: background, 1: surface,
2: ignore}` and `ignore` runs ~59% of a typical volume. It is excluded from scoring
everywhere. Folding it into background counts predictions in unscored regions as false
positives and understates precision badly — an earlier version of our own harness did exactly
that, and its precision figures are marked `precision_INVALID` in `results/m7_recall/`.

## The point of the benchmark: there are two populations, not one

@Jinhojeong's normalized-cross-correlation table (villa#191) locates **189 of the 892** volumes
inside Scroll1A. The published m7 model behaves like two different models across that line:

| population | n | median recall |
|---|---|---|
| locates on Scroll1A | 174 | **0.777** |
| locates nowhere searched | 681 | **0.918** |

That gap has survived elimination of **leakage** (p=0.80), **label artifact**, **fused
geometry** (Jinhojeong's independent 892-volume run, p=0.33) and **labelled-sheet density**
(survives quintile matching). It is unexplained. A single pooled score hides it entirely, so
every endpoint is reported per population as well as pooled.

## Endpoints

Per volume, never per voxel — the volume is the unit of analysis.

| endpoint | what it is |
|---|---|
| `recall` | of voxels labelled sheet, the fraction found, at threshold 0.2 (m7's published operating point) |
| `precision` | of scored predicted voxels, the fraction on labelled sheet |
| **`precision_lift`** | precision ÷ base rate. **Random = 1.0**, perfect ≈ 6. This is the honest one — raw precision looks bad on a 0.17 base rate even for a good model |
| `pred_positive_fraction` | share of the scored region predicted sheet |
| **`budget_recall`** | recall at a **matched predicted-positive budget** of 0.12, threshold chosen per volume |
| `pred_on_empty_ct` | guardrail: share of predicted sheet on CT that is identically zero |

**Why `budget_recall` exists.** Recall at a fixed threshold confounds discrimination with
calibration: two models that rank voxels identically but differ in confidence post different
recall at 0.2. At a matched budget both spend the same amount of "sheet", so a difference that
survives is a difference in ranking. If you report only one number, report this one.

## The benchmark ships with its own nulls

`python surface_bench.py --validate` — no model, no GPU. Measured on real labels:

| predictor | precision lift | recall |
|---|---|---|
| uniform random | **1.0000** | **0.7999** (= 1 − 0.2, as it must) |
| perfect | 1/base_rate (≈ 5.1–8.1) | 1.0 |

If the random arm does not land at 1.000, the scoring is broken and no number from this file
means anything. A benchmark whose own nulls are unknown cannot separate a result from a bug.

## m7 reference numbers

60 volumes, all from the located (hard) population, full endpoint suite
(`results/surface_bench_m7.json`):

| | value |
|---|---|
| recall @0.2 | 0.7740 |
| precision | 0.4459 |
| **precision lift** | **2.65×** |
| predicted-positive fraction | 0.3006 |
| **recall at matched 0.12 budget** | **0.4176** |

So on the population it struggles with, m7 is 2.65× better than chance against a ceiling near
6, and at a matched spend recovers **42%** of the labelled sheet. Recall-only numbers for all
**855** scored volumes, both populations, are in `results/m7_recall/`.

## What is already known not to work

Published so nobody spends a week re-deriving them. All are preregistered or controlled.

- **The asserted label margin is not mislabelled sheet.** m7's false positives enrich near
  sheet (3.10× at 1 voxel, 1.85× at 2, 1.01× at 3, then depleted) — but that is **proximity,
  not a misplaced boundary**. The shell profile decays geometrically with no step, and shell 1
  sits *below* the trend fitted to shells 2–5 (p=3.2e-09). Direction is not special either.
  `m7_margin_fp.py`, `PREREGISTER_margin.md` amendment 2.
- **Precision is not recoverable by any decision rule.** Ridge non-maximum suppression beats
  the best global threshold at matched recall by +0.0136 — and a **random-direction control
  gets +0.0130 of it**. m7's field is diffuse, not a crisp ridge under a fat coat. Any fix must
  come from the model or its training data. `m7_ridge_nms.py`.
- **Predicted-vs-CT thickness is not measurable this way.** The FWHM ratio is a free parameter
  of the profile window (0.94 → 0.62 as the half-width goes 3 → 8). Needs a window-invariant
  definition. `m7_thickness.py`.
- **Intensity augmentation for faint sheet made things worse** (July, published as a negative).
- **Label thinning** was abandoned on its own controls before any training: the estimator
  inflates by +0.286 vox from voxelisation alone. `PREREGISTER_labelthin.md`.

## Usage

```
python surface_bench.py --validate
python surface_bench.py --pred <dir of sample_XXXXX.npy probability volumes>
```

Predictions are float probability volumes in `[0,1]`, named to match the sample. They may be
full-size or a centred crop — the label is centre-cropped to match, so a model predicting only
an interior region is scored fairly on that region.

## Limitations, stated plainly

- The full endpoint suite is currently computed on **60 volumes, all located**. The `other`
  population has recall-only coverage (`results/m7_recall/`, 855 volumes) because producing
  probability maps for it costs GPU time we have not spent.
- **3 seeds** is the basis for any noise floor quoted from the arms experiments — a usable
  estimate, not a tight one.
- m7's numbers are contaminated by probable training overlap (top of this file).
- `bench_m7_recall.py` regenerates m7's predictions from scratch and **required fixing on
  2026-08-07**: as published it named an environment that can no longer import villa, and
  villa's inference exits 0 on failure (villa#1360), so its return-code checks passed over a
  dead stage. It now verifies artefacts instead.
