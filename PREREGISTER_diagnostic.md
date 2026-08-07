# Preregistration — can surface-prediction failure be attributed to the scan, the model, or the label?

**Written before any implementation**, and committed before the first result file, so the dates
are checkable against each other. This is bet 3. Bet 1 (label thinning) was abandoned on its own
controls before any arm. Bet 2 (margin relabelling) reached six training runs and returned a
null, its premise already killed by a proximity control. The same discipline applies here.

Author: TAUIL Abd Elilah. Date: 2026-08-07.

---

## 1. The problem, in the team's own words

`scrollprize.org/2026_open_problems` states this gap twice — once for surface prediction, once
as a cross-cutting meta-problem:

> "No clear mechanism to distinguish scanner limitations from model limitations in any given
> region"

> "we do not always know which part of the pipeline is limiting us; the limiting factor has to
> be assessed scroll by scroll"

and, separately: *"better diagnostics matter just as much as better models."*

@anshu231 built exactly this for **ink** (Inkdx, `#ink-detection`), citing the same page. Nobody
has built it for **surfaces**. That is the gap this bet addresses.

**Hypothesis (H1).** A CT-derived, model-free measure of *local sheet resolvability* separates
regions where a surface model fails because the scan does not resolve a sheet from regions where
it fails despite the scan resolving one — and that separation is **real**, meaning it predicts
the failures of a model it was not computed from.

## 2. ⚠ The trap this bet exists to avoid

**A three-way attribution is trivially constructible and worthless.** If "scan-limited" is
defined as *"the CT ridge is weak"* and "model-limited" as *"m7 fails where the ridge is
strong"*, then every voxel receives a label by construction, nothing can be wrong, and the
output is a taxonomy dressed as a finding. Bet 2's headline died to a version of this: an
enrichment that looked like a discovery and was arithmetic.

So the deliverable is **not** the taxonomy. It is the falsifiable claim underneath:

> **If a region is genuinely scan-limited, then an INDEPENDENT surface model must fail there
> too. If it is model-limited, independent models should disagree.**

That is testable, it can come out flat, and it is the only thing here worth calling a result.

## 3. The two models

Both published by the project:

| | |
|---|---|
| **A** | `surface_m7_nnunet` (nnU-Net, the production checkpoint) |
| **B** | `surface_recto_3dunet` (3D U-Net, independent architecture and training) |

⚠ **Both are run with CT normalization**, applied to the input with `--normalization none`
handed to the wrapper. `vesuvius.predict` defaults to `instance_zscore` and the nnU-Net path
never reads the scheme from `plans.json` (villa#1364, found by @Jinhojeong). Every m7 number we
published before 7 Aug was wrong because of this, and running B under the default would repeat
the mistake with a different model.

If B's own plans declare a different scheme, B gets **its own** declared scheme, not m7's.

## 4. The resolvability measure

Model-free and label-free, computed from CT alone at a sampled point:

- the across-sheet direction from the CT Hessian (most-negative-curvature eigenvector);
- the intensity profile along it;
- **resolvability = ridge prominence**, the peak-to-shoulder contrast of that profile,
  normalised by local noise.

⚠ **Built on ridge prominence, never on a half-max width.** The predicted/CT FWHM ratio moves
0.94 → 0.62 purely as the profile half-width goes 3 → 8, because a wider window keeps lowering
the baseline. A width is a free parameter of the window; a ridge is a location.
(`m7_thickness.py`, discarded for exactly this.)

## 5. Gate zero — run before anything else

**Does resolvability vary at all?** If the measure is near-constant across the 892 volumes there
is nothing to stratify and the bet is closed unamended.

- **Proceed only if the interquartile range of per-volume median resolvability spans at least a
  factor of 1.5.**
- **And only if resolvability is not merely a restatement of labelled-sheet density**: if
  `|Spearman(resolvability, sheet fraction)| > 0.7`, the measure is a density proxy, and the
  bet is closed unamended.

## 6. Primary endpoint

Stratify scored voxels into **resolvability quintiles**. Within each quintile compute

> **cross-model failure agreement** = of the labelled sheet voxels that model A misses, the
> fraction that model B also misses.

**H1 predicts agreement is HIGH in the lowest-resolvability quintile and LOWER in the highest.**
Two models failing together where the CT is poor is what "scan-limited" means; two models
failing on different voxels where the CT is good is what "model-limited" means.

- **Primary statistic: agreement in Q1 minus agreement in Q5**, per volume, paired.
- **Magnitude floor: a difference below 0.10 is reported as null regardless of p.**
- Volume is the unit; ⚠ but see §8 — if a run-level variance term appears, the unit is the run.

## 7. Controls, fixed now

1. ⭐ **Random-stratification control.** Repeat the whole endpoint with volumes assigned to
   quintiles at random. **If the random control reproduces the Q1−Q5 gap, resolvability is
   adding nothing** and the result is an artifact of stratifying a skewed quantity. This is the
   control bet 2 lacked until its premise had already been published.
2. **Density matching.** Recompute the primary within matched sheet-fraction strata. If the
   Q1−Q5 gap vanishes under matching, the measure is density in disguise.
3. **Base-rate control.** Model B's overall recall differs from A's; agreement will therefore
   drift with B's operating point. Agreement is computed with **B thresholded to match A's
   per-volume predicted-positive fraction**, so both spend the same budget.
4. **Null of the measure itself.** A uniform-random resolvability field must give a Q1−Q5 gap
   of 0 ± noise. If it does not, the stratification code is wrong.

## 8. Statistics

- Paired per volume, Wilcoxon signed rank, two-sided.
- ⚠ **`BENCHMARK.md` records that on this benchmark significance is nearly free**: comparing two
  fixed prediction sets, p = 1.6e-11 at every effect size down to Δrecall 0.0025, because paired
  changes are unanimous in sign. **A p-value is not evidence that a difference matters here.**
  The magnitude floor decides.
- Both models are fixed checkpoints, so there is no training-seed variance. If any run-level
  variance term is introduced later, the unit of analysis becomes the run, not the volume —
  bet 2's amendment 4 was caused by getting exactly this wrong.

## 9. Abandon conditions

1. **Gate zero fails** (§5) — closed, unamended.
2. **Random-stratification control reproduces the gap** — the measure is not carrying the
   result; closed.
3. **Gap below the 0.10 floor** — published as a negative. No extra strata, no second measure,
   no new endpoint.
4. **Density matching removes the gap** — reported as "resolvability is sheet density", which is
   a real if unexciting finding, not a rescue.
5. **Compute** — if either model cannot be run over the cohort in the time to 31 August, the
   study is reported as not run rather than quietly run smaller.

## 10. Published either way

Measure, strata, per-volume rows for both models, and the write-up — positive, negative or
abandoned. July's faint-sheet ablation went out as a negative, bet 1 as an abandonment, bet 2 as
a null with four amendments including one recording that its own preregistration named the wrong
unit of analysis. This gets the same treatment.

⚠ **What this bet is not.** It does not improve a surface model, and it should not be presented
as if it does. July's prize data is unambiguous that diagnostics and tools were paid at the
$1,000 tier while a working integrated pipeline took $20,000. This is a diagnostic. It is
proposed because it addresses a gap the team has named twice and because it is the thing we can
actually finish by 31 August — not because it is a headline.

---

*Registered before implementation. Deviations are recorded as dated amendments below, never by
editing the text above.*

## Amendments

### Amendment 1 — 2026-08-07 — GATE ZERO FAILED. Bet 3 is closed under abandon condition 1.

Run on 100 volumes, 500 labelled sheet voxels each (`results/resolvability_gate.json`):

| gate | measured | required | |
|---|---|---|---|
| 1 — per-volume spread | **q75/q25 = 1.244** | ≥ 1.5 | **FAIL** |
| 2 — not a density proxy | Spearman = **+0.190** | \|ρ\| ≤ 0.7 | PASS |

**Closed unamended, per abandon condition 1.** The threshold is not being moved.

**⚠ A defect in this preregistration, recorded because it is mine.** §5 gates on the spread of
**per-volume medians**, while §6's primary endpoint stratifies **voxels**. Those are different
scales and I registered the wrong one. The post-mortem measurement is unambiguous: within a
volume the q75/q25 ratio of voxel-level resolvability is **2.551**, which would have passed
comfortably. So the gate as written tested a quantity the study does not depend on.

**That does not rescue the bet, and here is why it matters more than the gate.**

Voxel-level CNR over 9,600 sampled labelled voxels: median **271.8**, q05 **66.4**, and only
**2.9%** below 50, **0.4%** below 20. **The scan resolves the sheet essentially everywhere a
label exists.**

That is a selection effect built into the design, not a tuning problem. Resolvability was
sampled *at labelled sheet voxels*, and a labelled voxel is by construction a place where a
human could see a sheet well enough to annotate it. **Sampling there conditions on
annotatability, so it cannot find scan-limited regions — they are exactly the regions the
annotator skipped.** This is the same shape as the "labels avoid ambiguous regions" complaint on
the open problems page, arriving from the other direction.

**The fundamental obstacle, which closes the line rather than the run.** To identify a
scan-limited region you must know a sheet is present *despite* the scan being too poor to
resolve it. That knowledge cannot come from the scan, and it cannot come from the labels, since
labels are absent for precisely that reason. It requires an external source of "a sheet is here"
— @Jinhojeong's incompleteness candidates, a neighbouring-wrap geometric prior, or a
higher-resolution scan of the same region. **We have none of those, so the attribution H1
proposes is not identifiable with the data available to us.**

**Any continuation needs a NEW preregistration, not an amendment to this one**, because it would
change the sampling frame, the gate and the identifying assumption — that is a different study,
and folding it in here would let a failed design quietly become a successful one.

**What survives and is worth keeping:** `resolvability.py`, its CNR measure, the second-difference
noise estimator, and the finding that labelled sheet in this set sits at a median CNR of ~272
with under 3% below 50. That last number is a useful fact about the label set on its own — it
says the 892-volume labels are drawn overwhelmingly from high-contrast sheet, which bears
directly on how far a model trained on them should be expected to generalise to compressed or
hazed regions.

**Elapsed: one afternoon.** Which is the point of gate zero.
