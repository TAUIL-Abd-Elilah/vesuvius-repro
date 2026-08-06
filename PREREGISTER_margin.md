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

*(none yet)*
