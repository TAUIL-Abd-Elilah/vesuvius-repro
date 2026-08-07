# Preregistration — does the surface label's asserted margin hurt the model?

**Written before any training arm was run,** and committed before the first result file, so the
dates are checkable against each other. This is bet 2. Bet 1 (label thinning,
`PREREGISTER_labelthin.md`) was abandoned on its own controls before any arm; the same
abandon discipline applies here.

Author: TAUIL Abd Elilah. Date: 2026-08-06.

---

## 1. The observation

The m7 label set has three classes, `{0: background, 1: surface, 2: ignore}`, and *ignore* is
a large share of a typical volume — between 0.13 and 0.78 in the eight volumes sampled so far.
So the annotators had a way to say "I decline to commit here", and used it heavily.

**They did not use it at the sheet boundary.** Sampling the voxel immediately outside each
labelled sheet run along the across-sheet normal (15,924 margins, 8 volumes,
`results/label_margin_classes.json`):

| class | share |
|---|---|
| **0 — background, asserted not-sheet** | **97.1%** |
| 2 — ignore | 2.9% |
| 1 — surface | 0.0% |

Meanwhile the CT sheet's median FWHM is **3.5 voxels** (`results/ct_sheet_thickness.json`) and
the published labels' implied true thickness is about **3.05** — measured 3.335 minus the
**+0.286** inflation that voxelisation alone produces in this estimator
(`results/thickness_control.json`, 16 synthetic sheets, 4 tilts).

**So on roughly half a voxel of what the CT calls sheet, the label positively asserts
background.** A model that predicts there is punished in training and scored as a false
positive at evaluation.

**Hypothesis (H1).** Relabelling that margin from `background` to `ignore` — never to
`surface` — produces a better surface model than training on the labels as published.

**Why `ignore` and not dilation.** Dilating to `surface` asserts sheet where we are not
certain there is sheet, and FWHM on a smoothed profile is a generous definition of where
papyrus ends. `ignore` only withdraws a claim the annotation was not entitled to make. If the
margin really is background, withdrawing it costs a little supervision and the arms come out
flat; it cannot teach the model something false.

**The null we expect to have to take seriously.** Segmentation networks are frequently robust
to a sub-voxel boundary convention, and this is the second intervention we have tried on this
model. The first (July's intensity augmentation) was a clean negative and bet 1 never reached
training. Base rate for our interventions here is 0 for 1, with 1 abandoned.

## 2. Gate zero — run before anything else

**The 97.1% is from 8 volumes, and the ignore-fraction varies from 0.13 to 0.78 between
them.** If margin class correlates with how heavily a volume was annotated, that number could
be carried by a few densely-labelled volumes and would not describe the set.

**Gate:** measure margin class per volume over **at least 200 volumes**, reporting the
distribution across volumes rather than a pooled percentage.

- **Proceed only if the median per-volume class-0 margin share is ≥ 0.80.**
- If it is below that, or if it is strongly bimodal, H1 is not about the label set as a whole
  and this preregistration is closed unamended.

## 3. Arms

Images are never modified. Only label classes change, and only within the margin.

| arm | labels |
|---|---|
| **A — baseline** | published labels, unmodified |
| **B — margin ignored** | voxels within **1** voxel outside a labelled sheet run, along the across-sheet normal, whose class is 0, are set to 2 (*ignore*) |

One margin width, fixed. A second width is exploratory and cannot become the headline.

## 4. Data and split

892 public labelled volumes, held locally at `data/kaggle/{images,labels}`.

Split **by provenance**, using @Jinhojeong's localisation table pinned in `overlap_report.json`
and sha256-checked:

- **Test: the 174 volumes that locate on Scroll1A** and are scored by `bench_m7_recall.py` —
  the population the current model does worst on.
- **Train/val: sampled from the 681 that locate nowhere searched**, disjoint from test.

Assignment is written to a file and committed before training starts.

## 5. Endpoints

⚠ **The trap: the intervention changes the labels, so scoring against the changed labels
proves nothing.** All endpoints below are computed against the **unmodified published labels**
or against the CT, never against arm B's relabelled set.

**Primary — recall on the test set, scored exactly as `bench_m7_recall.py` scores it today**,
with class 2 excluded from scoring as it already is. H1 predicts B > A. This metric is
unaffected by the relabelling because the margin was class 0 and becomes class 2, i.e. it
leaves the scored set for *both* arms identically when the same mask is applied.

**Co-primary — surface localisation error against the CT.** Median distance from the predicted
sheet's centroid to the CT ridge along the across-sheet normal. Thickness- and label-free.

**Secondary — predicted positive fraction.** Reported because the motivating observation was
that the model over-predicts where it fails (0.266 against 0.136). Direction is not predicted
in advance and it is not a success criterion.

**Guardrail — false positives in confidently-empty CT.** Fraction of predicted sheet lying on
CT that is identically zero, reusing `pred_over_empty_ct.py`. **If B raises this by more than
2 points against A, B fails regardless of the primary**: teaching the model to be less certain
about "not sheet" must not teach it to predict into nothing.

## 6. Statistics, fixed now

- **3 seeds per arm.** Volume is the unit of analysis, never voxel.
- Paired across test volumes, Wilcoxon signed rank, two-sided.
- **Magnitude floor: a median recall gain below 0.01 (1 point) is reported as null regardless
  of p.** The located population sits at 0.777 against 0.918; a gain that does not dent that
  gap is not the result this study is looking for.
- Every per-volume row published, not just aggregates.

## 7. Abandon conditions

1. **Gate zero fails** (§2) — closed, unamended.
2. **The transform does not do what it claims** — if relabelling does not move the margin
   class distribution on held-out volumes, stop before training.
3. **Guardrail fires** — empty-CT false positives up more than 2 points.
4. **Nothing at 3 seeds** — below the magnitude floor is a negative and is published as one.
   No extra seeds, no second margin width, no new metric.
5. **Compute** — if a full arm will not fit the time to the 31 August deadline, the study is
   reported as not run rather than quietly run smaller.

## 8. Published either way

Transform, split file, per-volume results for every arm, and the write-up — positive, negative
or abandoned. July's faint-sheet ablation was published as a negative, bet 1 was published as
an abandonment, and this gets the same treatment.

---

*Registered before the first arm. Deviations are recorded as dated amendments below, never by
editing the text above.*

## Amendments

### Amendment 1 — 2026-08-07 — the proxy model at the registered budget cannot test H1

**Decided on arm A alone. No arm B run existed, and no A-vs-B comparison had been computed,
when this was written.** Arm A seed 0 was the first and only completed run.

**What arm A seed 0 showed.** At the registered budget (80 epochs × 60 iters × batch 2, 4800
steps), scored over all 174 test volumes:

| | value |
|---|---|
| median predicted-positive fraction, scored region | **0.817** |
| sheet base rate in the scored region (class 1 / (class 0 + class 1)) | **0.118** |
| median precision | **0.175** |
| median recall | 0.925 |

**Median precision 0.175 is 1.48× the base rate of 0.118.** A model that predicted *everything*
would score recall 1.00 and precision 0.118. So the recall of 0.925 is bought almost entirely by
predicting 82% of the scored region as sheet, and the model has very little discrimination to
detect a change in. This is the same artifact `ablate_faint_sheet.compare()` was written to
catch — recall traded for precision by a bias shift — and it makes a margin effect worth ~0.44×
the sheet volume undetectable in either direction.

**Changed — training configuration only:**

1. Budget 80 → **200 epochs** (12,000 steps). The loss was still noisy and flat at 80.
2. Class weights: inverse frequency → **inverse *square-root* frequency**. Inverse frequency put
   a 7.5× ratio on class 1 over class 0 and drove the collapse.
3. Model checkpoints are **saved**, so any later threshold question can be asked without
   retraining. Not saving them is why this needed a rerun rather than a re-score.

**Added — one co-primary endpoint, and the reason it is not optional.** Arm B changes the class
frequencies of the training labels, so the two arms will have *different operating points* by
construction. Recall at a fixed threshold therefore confounds discrimination with calibration,
and a "gain" for arm B could be pure bias shift. So:

> **Co-primary: recall at a matched predicted-positive budget.** For each run, the decision
> threshold is chosen on the **val set** (the 100 volumes already reserved and so far unused) as
> the threshold whose predicted-positive fraction equals **0.12** — the sheet base rate in the
> scored region, fixed here in advance. That threshold is then applied to the test set.

It is the same control §6's magnitude floor and July's headroom normalisation exist to provide,
applied properly rather than after the fact.

> **Clarification, same day, written before the calibration was implemented.** Above I called
> this calibration "label-free". That was imprecise and is corrected here rather than by editing
> it. Measuring a predicted-positive fraction *in the scored region* requires the scored mask,
> which does require labels. The property the argument actually needs — and has — is that the
> mask comes from the **unmodified** published labels (`EVAL_LABELS`), exactly as every other
> endpoint here does, so it is **identical for both arms and arm B's relabelling cannot reach
> it**. Arm-invariant, not label-free. The registered target of 0.12 is unchanged.

**Explicitly unchanged:** the hypothesis, the two arms, the margin width, the provenance split
and its file, the registered primary (recall at threshold 0.2, class 2 excluded), the
localisation and predicted-positive-fraction endpoints, the empty-CT guardrail and its 2-point
kill condition, 3 seeds, Wilcoxon paired on volumes, and the 0.01 magnitude floor.

**On abandon condition 4 ("no new metric").** That condition governs what happens *after* a null
at 3 seeds — it forbids going fishing once the registered analysis has come back empty. No arm
comparison has been run. This is a repair to the instrument before the experiment, recorded
before the fact, not a search for a result after one.

### Amendment 2 — 2026-08-07 — the motivating premise fails a control on m7. The arms still run.

**Recorded while the arms were training and before any A/B comparison existed.**

`m7_margin_fp.py` asked §1's question of the published model over 60 held-out volumes: are m7's
scored false positives concentrated in the asserted margin? The headline said yes — median
enrichment **3.10**, 98% of volumes above the null, **26% of m7's scored false positives inside
the margin**.

**The control says that is proximity, not the label boundary.** Enrichment by Euclidean shell
from the labelled sheet:

| shell (vox) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| median enrichment | 3.103 | 1.851 | 1.012 | 0.588 | 0.306 |
| ratio to previous | — | 0.597 | 0.547 | 0.581 | 0.520 |

The ratios are near-constant: a clean geometric decay with **no kink at shell 1**. H1 predicts
an *excess* at shell 1, because the CT sheet is supposed to stop just past the label. Fitting
shells 2–5 per volume and extrapolating back to shell 1 gives observed/predicted **0.755**
(median), with only **10%** of volumes showing any excess, Wilcoxon **p = 3.2e-09** — shell 1 is
significantly **below** the trend, not above it.

Two further points, both against H1:

- **Direction is not special either.** The normal-restricted margin enriches at **3.104**, the
  full Euclidean shell 1 at **3.103**. If the across-sheet direction carried the effect, the
  normal margin should beat its containing shell. It does not.
- What survives is descriptive and worth keeping: **m7's errors are boundary errors.** Enriched
  within 2 voxels of labelled sheet, at the null by 3, and *depleted* beyond (0.59, 0.31). m7
  does not hallucinate sheet in open space; it makes sheets slightly too fat.

**The arms are NOT abandoned, and this is deliberate.** §7 lists four abandon conditions and
"the motivating observation weakened" is not among them. Stopping a preregistered experiment
because the prior turned against it is precisely the behaviour preregistration exists to
prevent, and the registered question — does relabelling the margin help — remains a real
empirical question that this control does not answer. The arms run to 3 seeds and are published
under the registered rule.

**What this does mean:** the expected outcome is now a null, and it will be reported as one.
Any positive arm result must be read against a motivating premise that has already failed its
control on the production model, and that caveat belongs in the write-up whatever the arms say.
