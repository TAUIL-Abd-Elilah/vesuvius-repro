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
2. ⛔ **The two-population RECALL gap below was an ARTIFACT of this default, not a property of
   the model.** On the full 892 under plans normalization the split reads **0.9140 vs 0.9201**
   — a gap of 0.006 against the 0.141 reported here. (An earlier 60-volume draw gave 0.914 vs
   0.923; the full-population figures supersede it.) The mechanism is intensity:
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

## Split by provenance — the recall gap was an artifact, the precision gap is not

@Jinhojeong's normalized-cross-correlation table (villa#191) locates **189 of the 892** volumes
inside Scroll1A. Under the wrong normalization the published m7 model looked like two different
models across that line. **It is not. The recall gap was the normalization.**

⭐ **FULL POPULATION, 2026-08-08.** @Jinhojeong re-scored **all 892** under the CTNormalization
constants m7's own `plans.json` declares, at this benchmark's geometry
([villa#193](https://github.com/ScrollPrize/villa/issues/193#issuecomment-5223464638); rows,
runner and finalizer in his diagnostic repo at `ae64423`, `results/shell892` and
`scripts/shell892`). These supersede every m7 figure this file carried before:

| population | n | median recall | median precision | precision lift | pred-positive fraction |
|---|---|---|---|---|---|
| locates on Scroll1A | 174 | **0.9140** | **0.6935** | 4.03× | 0.2129 |
| locates nowhere searched | 679 | **0.9201** | **0.7559** | 6.41× | 0.1430 |
| all | 853 | **0.9194** | **0.7463** | — | 0.1538 |

⚠ **Read the volume counts before comparing them to ours.** Each column is a median over the rows
that define *that* column, so the denominator moves within a row: `pred_positive_fraction` is
defined on volumes carrying no labelled sheet, recall and precision are not. Hence 174 / 679 / 853
rather than our 174 / 681 / 855. The two extra volumes on our non-located side
(`sample_00075`, `sample_00127`) carry 4 and 7 labelled sheet voxels and nothing predicted, so
recall is defined and 0.0 while precision is not — **the non-located recall median reads 0.9201
over 681 as well as over 679.** 39 rows carry a degenerate status with a reason string rather
than being dropped: 32 with `n_scored` zero (the whole inner cube is class 2), 5 with scored
background but no labelled sheet, plus that pair.

⛔ **THE RECALL GAP IS GONE: 0.9140 against 0.9201 is 0.006, where we published 0.777 against
0.918 — a gap of 0.141.** It survived our elimination of **leakage** (p=0.80), **label
artifact**, **fused geometry** (Jinhojeong's independent 892-volume run, p=0.33) and
**labelled-sheet density** (quintile matching), and we reported it as unexplained. It was the
`instance_zscore` inference default the whole time.

The reason the split *looked* like two populations is that it is nearly a proxy for intensity:
the located cohort's centred-cube mean is ~135 against m7's training fingerprint of 87.5 ± 47.7,
while the non-located sit within half a sigma of it, and per-volume z-scoring departs from CT
normalization exactly as a volume's statistics leave that fingerprint.

⭐ **But do not now conclude the split is meaningless — it moved from recall to precision.**
Precision runs **0.6935 against 0.7559** and precision lift **4.03× against 6.41×**, on base
rates of 0.163 and 0.117. **Reporting per population still earns its place**; a pooled score
would have hidden the original artifact too, and that is how it was found. What is retracted is
the claim that m7 *recalls* less on the located population, not the practice of splitting.

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

**Quote the full-population table above.** The reference figures for m7 are now the 174 / 679 /
853 medians, not the 60-volume draw this file used to lead with.

The 60-volume run below is kept for one purpose only: it is **our** implementation on **our**
hardware, and it is what makes the correction checkable end to end. Both columns are the same
code and the same volumes; only the input normalization differs
(`results/surface_bench_m7_ctnorm.json` and the superseded `surface_bench_m7.json`).

| 60 located volumes | ⛔ instance_zscore (villa default) | ✅ **CT normalization (what m7 was trained on)** |
|---|---|---|
| recall @0.2 | 0.7740 | **0.9109** |
| precision | 0.4459 | **0.6680** |
| **precision lift** | 2.65× | **3.75×** |
| predicted-positive fraction | 0.3006 | 0.2495 |
| **recall at matched 0.12 budget** | 0.4176 | **0.5730** |

**m7 is a much better model than this benchmark originally reported** — on the full 174 located
volumes, precision lift **4.03×** against a ceiling near 6, while predicting *less* (0.2129
against the broken path's 0.3006). The wrong normalization was making it both blinder and more
liberal.

⭐ **What the corroboration is, precisely.** On the **same 60 volumes**, two separate
implementations with separate checkpoint fetches give **0.9109 / 0.6680** (ours) against
**0.9138 / 0.678** (Jinhojeong's). That is a genuine cross-implementation check and it is why
these numbers can be trusted where the previous ones could not.

⚠ **It is not, however, a check of the 60 against the population, and we previously implied it
was.** Jinhojeong's located 60 are a *subset* of the 174, not an independent draw, and he
characterises that draw as the bright, low-contrast end of the located set. On the full 174 his
precision moves 0.678 → **0.6935** while the recall stands. So read the 60-volume agreement as
evidence about *implementations*, and the 174 / 679 / 853 table as evidence about *m7*.

⚠ **Per-volume agreement is looser than the medians suggest, and that is expected.** Across the
same 60, the per-volume absolute recall difference between the two implementations has a median
of 0.006 with a tail to 0.13, and only 29 of 60 sit inside 0.005. Jinhojeong also re-ran 120
volumes inside the full pass and found label-side quantities reproduced exactly while prediction
counts moved by a few parts in 1e-5 (worst case 8.8e-4 relative on `n_fp`), with only 5 of 120
bit-identical — **fp16 sliding-window nondeterminism, not a code difference.** A second-decimal
per-volume disagreement is not by itself a sign that something is wrong.

⚠ **Budget-calibrated columns are not directly comparable between the two paths.**
`surface_bench.py` takes the exact 0.88 quantile of the scored probabilities; Jinhojeong searches
a 2000-bin histogram for the first bin at or below a 0.12 predicted-positive share. The
thresholds land within one bin of each other, but the endpoints are not bounded that tightly —
our rule always spends the full budget where a grid cannot, and 9 of his 853 come in under 0.10.

Recall-only numbers for all **855** scored volumes in `results/m7_recall/` are **still from the
broken path** and are retained only so the size of the error stays checkable. **Do not quote
them.** Our own regeneration under CT normalization (`m7_renorm.py --all`) is in progress as an
independent check of the table above.

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

- ✅ **Resolved 2026-08-08 for m7.** The full endpoint suite now covers all **853** scorable
  volumes across both populations, contributed by @Jinhojeong. Our own full-suite run remains at
  **60 volumes, all located**; the `other` population had recall-only coverage
  (`results/m7_recall/`, 855 volumes, broken path) because producing probability maps for it
  costs GPU time we had not spent. **This limitation still applies to any model that is not m7** —
  scoring a new checkpoint across both populations is still an unspent GPU cost.
- ⚠ **The full-population table is a contributed result, not one we reproduced.** It ships with
  per-volume counts, provenance hashes and a runner, so it is checkable; we have not yet checked
  it. Treat the 60-volume column as the part this repo stands behind directly.
- **3 seeds** is the basis for any noise floor quoted from the arms experiments — a usable
  estimate, not a tight one.
- m7's numbers are contaminated by probable training overlap (top of this file).
- `bench_m7_recall.py` regenerates m7's predictions from scratch and **required fixing on
  2026-08-07**: as published it named an environment that can no longer import villa, and
  villa's inference exits 0 on failure (villa#1360), so its return-code checks passed over a
  dead stage. It now verifies artefacts instead.
