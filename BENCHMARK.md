# A surface-model benchmark on the 892 public volumes, split by provenance

Scoring code: [`surface_bench.py`](surface_bench.py). Null controls: `--validate`, no model or
GPU required.

## ⛔ CORRECTION, 2026-08-07 — every m7 number below was produced with the WRONG NORMALIZATION

**`vesuvius.predict` defaults to `--normalization instance_zscore`. m7's `plans.json` declares
`normalization_schemes: ['CTNormalization']` with mean 87.544, std 47.744. The nnU-Net loading
path never reads the plans**, so the checkpoint is fed per-volume z-scoring instead of the CT
normalization it was trained on. `inference.py` sets `normalization_scheme='instance_zscore'` in
the constructor and then takes `self.model_normalization_scheme or self.normalization_scheme`;
`_checkpoint_normalization_scheme` only inspects the checkpoint, so for an nnU-Net
`checkpoint_best.pth` the model scheme stays `None` and the default wins.

**Found by @Jinhojeong** (villa#1364), on villa#193. Reproduced here independently — CT-normalized
input with `--normalization none`, 4 volumes of this cohort:

| | instance_zscore (**what produced every number below**) | CT-normalized |
|---|---|---|
| median recall | 0.803 | **0.940** |
| median precision | 0.467 | **0.742** |

**Consequences, stated plainly:**

1. ⛔ **The m7 reference table is wrong.** recall 0.7740 / precision 0.4459 / lift 2.65 /
   budget_recall 0.4176 all understate m7 substantially. **Do not quote them.**
2. ⛔ **The two-population gap below is substantially an ARTIFACT of this default, not a
   property of the model.** With plans normalization @Jinhojeong measures **0.914 vs 0.923**
   across the same split, against the 0.777 / 0.918 reported here. The mechanism is intensity:
   the located cohort's centred-cube mean is ~135 against a training fingerprint of 87.5 ± 47.7,
   a full sigma out, while the non-located sit within half a sigma — and instance z-scoring
   diverges from CT normalization exactly as a volume's own statistics leave that fingerprint.
   **We eliminated leakage, label artifact, fused geometry and density and called the gap
   unexplained. The cause was in the inference wrapper and we never questioned it.**
3. ✅ **What survives:** the scoring code, the validated nulls, the budget-matched endpoint, the
   seed-variance findings (those concern our own trained proxy, not m7), and the margin
   conclusion — which holds on @Jinhojeong's correctly-normalized predictions and holds *more*
   strongly. The label-placement negatives used labels and CT only, so they are unaffected.
4. ⚠ **Anyone running an nnU-Net checkpoint through `vesuvius.predict` has this problem**, and
   there is no flag for it: the CLI offers `instance_zscore`, `global_zscore`, `instance_minmax`
   and `none`, and the `ct` scheme raises because `inference.py` never passes the intensity
   properties. The workaround is to CT-normalize the input yourself and pass `--normalization none`.

The reference numbers will be replaced once the cohort is re-scored under plans normalization.
They are left visible rather than deleted so the size of the error stays checkable.

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
| locates on Scroll1A | 174 | **0.777** ⛔ artifact — see correction at top |
| locates nowhere searched | 681 | **0.918** ⛔ artifact — see correction at top |

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

⛔ **WRONG — produced under instance_zscore. See the correction at the top of this file.**


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

## Resolution: what difference this benchmark can actually see

Two different questions, with very different answers. Conflating them will make you believe a
result that isn't there.

### Comparing two FIXED prediction sets — significance is nearly free

Paired per volume over the 60 cached m7 volumes, comparing m7 against itself at a slightly
shifted threshold:

| shifted threshold | median Δrecall | Wilcoxon p |
|---|---|---|
| 0.205 | −0.0025 | 1.6e-11 |
| 0.210 | −0.0050 | 1.6e-11 |
| 0.220 | −0.0100 | 1.6e-11 |
| 0.300 | −0.0469 | 1.6e-11 |

**p is identical at every effect size, including a difference of a quarter of one percent.**
Raising a threshold lowers recall on essentially every volume, so the paired sign test is
unanimous 60/60 and p sits at its floor for that n. Wilcoxon here is measuring *consistency of
sign*, not size of effect.

⚠ **So do not report a p-value as evidence that a change matters.** On this benchmark any
consistent change is "significant". **Report the magnitude and judge it against a
pre-committed floor.** That is why `PREREGISTER_margin.md` fixed a magnitude floor of 0.01
median recall *regardless of p* before any arm ran — and it is the single most important thing
to carry over into your own comparison.

### Testing an intervention by RETRAINING — the floor is much higher

Training-run variance dwarfs the effects people want to measure. Two seeds of the **same** arm,
identical configuration, 200 epochs each:

| | median recall @0.2 | pred-positive |
|---|---|---|
| arm A seed 0 | 0.6982 | 0.5617 |
| arm A seed 1 | 0.8415 | 0.6775 |

**A spread of 0.143 between two runs of the identical setup**, with the operating point swinging
0.56 → 0.68. For comparison, the A-vs-B difference at seed 0 was +0.020 — **seven times smaller
than the noise between two runs of one arm.**

Across all three seeds of arm A: **sd 0.0892, range 0.1637** — nine times the 0.01 magnitude
floor that study had registered.

**⚠ And the unit of analysis matters more than the seed count.** Pairing by *volume* treats the
174 volumes as independent replicates. They are not, with respect to run-level noise: a whole run
sits high or low together, so one offset is counted 174 times. That produced

> primary recall @0.2 — A 0.8009, B 0.7805, delta −0.0205, **p = 2.15e-05**

from seed-level differences of **+0.020, −0.055, +0.009** (mean −0.0087, 95% CI −0.074…+0.056).
An effect cannot be significant and also reverse sign across seeds. **When run-level variance
dominates, the seed is the unit of analysis, not the volume.**

### ⭐ The fix: score at a matched predicted-positive budget, not a fixed threshold

Most of that variance is *calibration*, not capability — pred-positive swung 0.56 → 0.73 across
seeds of one arm. Scoring each volume at the threshold that spends a fixed budget removes it:

| arm A across seeds | recall @0.2 | `budget_recall` |
|---|---|---|
| values | 0.6982, 0.8415, 0.8619 | 0.1752, 0.1785, 0.1828 |
| **sd** | **0.0892** | **0.0038** |

**A 23× reduction in run-to-run noise, from the same six runs.** This is the single most
practical thing to take from this file: **if you are testing a training-data intervention here,
report `budget_recall`.** At a fixed threshold, calibration variance will swamp your effect —
it swamped ours, and it did so while producing a p-value of 2e-05 in the wrong direction.

⚠ **Even so, three seeds gives sem ≈ 0.023 on the primary.** Reducing the noise 23× makes the
endpoint usable; it does not make n=3 sufficient. Budget seeds against the effect you expect.

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

The 892-volume set is **not vendored here** — it is the public Kaggle data. Point at it with:

```
VESUVIUS_DATA=/path/to/kaggle python surface_bench.py --validate
```

expecting `images/` and `labels/` beneath that directory. Without it the script exits 2 with
the paths it looked in, rather than failing later on an individual sample.

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
