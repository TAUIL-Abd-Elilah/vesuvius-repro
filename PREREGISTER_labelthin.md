# Preregistration — do thinner surface labels make a better surface model?

**Written before any training arm was run.** Nothing below was chosen after seeing a result.
Committed to `vesuvius-repro` before the first arm starts, so the commit date is checkable
against the first result file.

Author: TAUIL Abd Elilah. Date: 2026-08-06.

---

## 1. Why this question

Two measurements of ours, made for other reasons, point at the same place.

**The published surface labels are thicker than the sheet they mark.** Over 40 volumes
(`measure_label_drift.py`, `results/label_drift_centroid_n40.json`): mean label thickness
**3.335 voxels**, against a sheet that is about **2.4 voxels** at this resolution. The labels
are not *displaced* — mean signed centroid drift is **0.0002 voxels**, i.e. zero — they are
simply fat.

**And the model over-predicts exactly where it does worst.** On the 174 volumes that locate
on Scroll1A, median recall is **0.777** against **0.918** elsewhere, while
`pred_positive_fraction` runs **0.266** against **0.136**. Four explanations for that gap are
already dead: leakage (p = 0.80), a label artifact in the "fewer labels" direction (they carry
*more* labelled sheet), fused geometry (@Jinhojeong's 892-volume census, p = 0.33, powered),
and labelled-sheet density (gap survives quintile matching, 0.135 against 0.142 unmatched).

**Hypothesis (H1).** A model trained on labels thinned toward the sheet centre predicts sheet
that is better localised and less over-inclusive than the same model trained on the published
labels, without losing true sheet.

**The null we expect to have to take seriously.** Thickness is a property of the annotation,
not of the CT, and nnU-Net may already be robust to it. Our July intensity-augmentation
ablation on the same benchmark came back cleanly negative, so the base rate for our
interventions on this model is currently 0 for 1.

## 2. Arms

Labels are modified; images are never touched.

| arm | label transform |
|---|---|
| **A — baseline** | published labels, unmodified |
| **B — centroid-thinned, w = 3.0 vox** | along the across-sheet normal, keep the central 3.0 voxels of each label run, **position preserved** |
| **C — centroid-thinned, w = 2.4 vox** | as B, at the measured sheet thickness |

**Why position is preserved and not snapped.** Snapping each label onto its local CT peak is
the obvious alternative and we are deliberately not making it the primary arm. Our own
measurement says there is nothing systematic to correct (signed drift ≈ 0), and §00.0o C2
recorded that the per-voxel offsets are sign-random and that the estimator can lock onto a
*neighbouring* sheet where the pitch is small. Moving labels on that basis risks teaching the
model the wrong sheet. If B or C wins, a snapped arm becomes a follow-up question, not a
rescue.

**w is a preregistered pair, not a swept parameter.** 3.0 and 2.4 are fixed here. If a third
value is ever run it is reported as exploratory and cannot be the headline.

## 3. Data and split

892 public labelled volumes, held locally (`data/kaggle/{images,labels}`).

Split **by provenance, not at random**, using @Jinhojeong's localisation table pinned at
`overlap_report.json` (sha256 checked):

- **Test set: all 174 volumes that locate on Scroll1A and are scored by `bench_m7_recall.py`.**
  This is the population the model already fails on, so it is where an improvement has to show.
- **Train/val: sampled from the 681 that locate nowhere searched**, disjoint from test.

Volumes are assigned once, before training, and the assignment file is committed.

## 4. Endpoints

⚠ **The trap this section exists to close: the intervention changes the labels, so the
evaluation must not use the thinned labels.** A thinned-label model predicts thinner sheet and
would score worse against fat ground truth and better against thin ground truth, purely by
construction. Neither is evidence. So the primary endpoint is thickness-independent and the
conservative one is reported beside it.

**Primary (thickness-independent): surface localisation error.** For predicted sheet voxels,
the signed distance along the across-sheet normal from the predicted sheet's centroid to the
CT ridge, using the same Hessian/profile machinery as `measure_label_drift.py`. Reported as
median |error| over the test set. H1 predicts B and C beat A.

**Co-primary: predicted thickness against measured sheet thickness.** Median predicted sheet
run-length along the normal, compared to the CT-derived sheet thickness in the same place.
H1 predicts A over-predicts thickness and B/C sit closer to the measured sheet.

**Secondary, conservative: Dice and recall against the ORIGINAL published labels**, for both
arms. This is biased *against* B and C by construction — a thinner prediction cannot fully
cover a fat label. It is reported so the cost of the intervention on the conventional metric
is visible rather than hidden. **A drop here is expected and is not on its own a failure.**

**Guardrail: recall of sheet *centres*.** Fraction of label-run centroids covered by the
prediction. This is the "did we lose true sheet" check, and it is thickness-independent.
**If B or C loses more than 2 points of centre recall against A, the arm fails regardless of
what the localisation endpoint says.**

## 5. Statistics, fixed in advance

- **3 seeds per arm.** Volume is the unit of analysis, not voxel.
- Paired comparison across test volumes (each volume scored under every arm), Wilcoxon signed
  rank, two-sided.
- **Significance is not the bar; magnitude is.** A localisation improvement below **0.25
  voxels** median is reported as null regardless of p, because it is smaller than the
  measurement's own resolution on a 2.4-voxel sheet.
- Every per-volume row is published, not just aggregates.

## 6. What would make me abandon this

Stated now so it cannot be renegotiated later:

1. **The transform does not do what it claims.** If thinning does not measurably reduce label
   thickness toward w on held-out volumes, stop before training.
2. **The guardrail fires.** Centre recall down more than 2 points → the arm is dead.
3. **Nothing at 3 seeds.** If neither B nor C clears 0.25 voxels on the primary, the result is
   a negative and gets published as one. **No extra seeds, no new w, no new metric.**
4. **Compute.** If a full arm cannot be trained in the time available, the study is reported
   as not run rather than run smaller and quietly.

## 7. What gets published either way

The transform, the split file, every arm's per-volume results, and the write-up — positive,
negative or abandoned. The July faint-sheet ablation was published as a negative and this gets
the same treatment.

---

*Registered before the first arm. Any deviation from this document is recorded as a dated
amendment below rather than by editing the text above.*

## Amendments

*(none yet)*
