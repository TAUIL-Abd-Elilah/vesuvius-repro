# Preregistration — does a distance-transform representation preserve sheet topology better than binary segmentation?

**Written before any implementation**, committed before the first result file. This is bet 4.
Bet 1 was abandoned on its own controls; bet 2 ran six training arms and returned a null; bet 3
closed at gate zero in an afternoon. The same discipline applies, and the abandon conditions are
meant to be used.

Author: TAUIL Abd Elilah. Date: 2026-08-07.

---

## 1. Why this bet

`scrollprize.org/2026_open_problems` states the call to action directly:

> "Create datasets with labels better localized on the papyrus' recto **or train ML models that
> can better preserve the sheets' topology**"

and names what is missing under surface prediction:

> "reaching a high level of topologic accuracy in **densely packed regions**, regions affected by
> high curvature or in spots where the papyrus is damaged"

with alternative representations — specifically a **Distance Transform rather than binary
segmentation** — named as deserving investigation and not implemented.

**Nobody in the community is training surface models.** As of 7 Aug, villa carries no issue or PR
on distance-transform or topology-preserving surface prediction, and every August community
contribution is tooling, meshing, VC3D or fiber infrastructure. The work needs a GPU, labelled
volumes, and a way to prove an improvement; that combination is rare, which is why the gap is open.

**Our own measurements point here.** m7's probability field is diffuse: its precision is **not
recoverable by any decision rule**, neither thresholding (0.2→0.5 buys precision 0.446→0.509 at a
cost of recall 0.774→0.643) nor ridge non-maximum suppression (+0.0136, against +0.0130 for a
random-direction control). If the output cannot be fixed downstream, the fix has to be upstream —
in the model or its representation.

**Hypothesis (H1).** A model trained to regress a **signed distance to the sheet surface**
preserves sheet topology better — specifically, merges fewer distinct nearby sheets — than the
same model trained on binary segmentation, at matched capacity, data, schedule and evaluation
budget.

**Mechanism, stated so it can be wrong.** Binary segmentation of a ~3-voxel sheet is ill-posed at
the boundary: the target flips class across one voxel and the network hedges, which is what a
diffuse field looks like. Two sheets a few voxels apart fall inside one another's hedged band and
the mask bridges them. A signed distance field is smooth and single-valued everywhere, and two
nearby sheets induce **distinct** fields with a ridge between them, so the zero-crossing can
separate what a mask merges.

## 2. Arms

Identical architecture, data, split, schedule, optimiser and seeds. **Only the target and loss
differ.**

| arm | target | loss | decision rule |
|---|---|---|---|
| **A — binary** | 3-class labels as published | weighted CE + soft Dice (bet 2's recipe) | `sigmoid(l1-l0) > t` |
| **B — DT** | signed distance to the sheet surface, clipped to ±D voxels, normalised to [-1,1] | masked L1 on the CT-supported region | `predicted distance < 0` |

⚠ **Class 2 (`ignore`) is excluded from the loss in both arms**, exactly as it is excluded from
scoring. It is ~59% of a volume and supervising on it would differ between arms for reasons
unrelated to representation.

## 3. Primary endpoint — the topology claim, not a recall claim

**Merger rate.** Take pairs of **distinct connected components** of labelled sheet whose closest
approach is ≤ 6 voxels — the densely packed regime the page names. For each pair, ask whether the
thresholded prediction **connects** them into one component. 

> **merger rate = fraction of close labelled-component pairs that the prediction merges.**
> **H1 predicts B < A. Lower is better.**

- **Magnitude floor: an absolute reduction below 0.05 is reported as null regardless of p.**
- Per volume, paired between arms.

**⚠ Scored at a MATCHED PREDICTED-POSITIVE BUDGET.** A model that predicts less merges less, for
free. Both arms are thresholded per volume to spend the same predicted-positive fraction (0.12,
as in `BENCHMARK.md`), so a merger difference cannot be bought by being conservative. Without
this the endpoint is meaningless.

**Co-primary — `budget_recall`** from `surface_bench.py`, so the result is comparable to bet 2
and to the corrected m7 reference (0.5730). Topology must not be won by losing the sheet.

**Secondary, reported not predicted:** precision lift, localisation error, and the empty-CT
guardrail.

## 4. Gate zero — is there a densely-packed regime in this data at all?

⚠ **This bet is about merging nearby sheets. If the 892-volume set rarely contains two distinct
labelled sheets within a few voxels, there is nothing to measure and the study is about
nothing.** Our own earlier probe is a warning: closing-based contact fraction at radius 2–3 was
only 0.001–0.006 of sheet volume.

Over **≥100 volumes**, count pairs of distinct labelled sheet components whose closest approach
is ≤ 6 voxels.

- **Proceed only if the median volume contains ≥ 5 such pairs**, and at least 60% of volumes
  contain ≥ 1.
- Below that, closed unamended: the dataset has no densely-packed regime and the hypothesis is
  untestable here.

## 5. Controls, fixed now

1. ⭐ **Matched-budget scoring** (§3). Non-optional; without it "predicts less" wins.
2. **Identical everything but the target.** Same seeds, same pool schedule, same architecture,
   same steps. Any other difference invalidates the comparison.
3. **Degenerate-arm check.** If either arm's predicted-positive fraction at its own natural
   threshold falls outside [0.02, 0.60], that arm is degenerate and the pair is discarded — bet
   2's arm A predicted 0.817 of the scored region and could not test anything.
4. **Merger-rate null.** A uniform-random prediction at the same budget must give a merger rate
   near 1.0 (it connects everything), and a perfect prediction near 0. If the metric does not
   bracket, the implementation is wrong.

## 6. Statistics

- **3 seeds per arm.** Paired per volume, Wilcoxon two-sided.
- ⚠ **The unit of analysis is the SEED if run-level variance is material.** Bet 2's amendment 4
  exists because volume-level pairing counted one run-level offset 174 times and produced
  p = 2.15e-05 for an effect whose sign reversed across seeds. **Before reading any A/B number,
  arm A's across-seed sd is computed and reported.** If it exceeds the magnitude floor, the
  comparison is reported as underpowered regardless of what the paired test says.
- ⚠ **A p-value is not evidence here.** On this benchmark, comparing two fixed prediction sets
  gives p = 1.6e-11 at every effect size down to Δrecall 0.0025, because paired changes are
  unanimous in sign. The magnitude floor decides.
- **Prediction registered now:** bet 2's arm A had across-seed sd **0.0892** on recall. H1's
  mechanism implies the DT arm should be *more* stable, since it has no class imbalance and no
  calibration drift. **If arm B's across-seed sd is not below arm A's, that is evidence against
  the mechanism** even if the merger rate moves.

## 7. Abandon conditions

1. **Gate zero fails** (§4) — closed, unamended.
2. **Either arm degenerate** (§5.3) — closed, reported.
3. **Merger-rate reduction below 0.05** — published as a negative. No extra seeds, no second
   distance clip, no new endpoint.
4. **Topology bought by losing sheet** — if B's `budget_recall` falls more than 0.05 below A's,
   B fails regardless of merger rate.
5. **Compute** — GPU is shared with the user's own work. If six runs will not fit before
   31 August, the study is reported as not run rather than quietly run smaller.

## 8. Published either way

Transform, split, per-volume rows for both arms, checkpoints, and the write-up — positive,
negative or abandoned, as with July's ablation, bet 1, bet 2 and bet 3.

⚠ **What this bet is and is not.** It does not claim to beat m7, which is a 4000-epoch nnU-Net
against our small UNet at lift 1.2 versus 3.75. It tests the **representation**, at matched
capacity, which is what the page asks be investigated. A positive result is a recipe and a
finding, not a production model.

---

*Registered before implementation. Deviations are recorded as dated amendments below, never by
editing the text above.*

## Amendments

### Amendment 1 — 2026-08-07 — GATE ZERO FAILED. Bet 4 is closed under abandon condition 1.

100 volumes, components ≥100 voxels, closest approach by boundary KD-tree
(`results/dt_gate.json`):

| gate | measured | required | |
|---|---|---|---|
| 1 — median close pairs per volume | **1.0** | ≥ 5 | **FAIL** |
| 2 — volumes with ≥ 1 close pair | **57.0%** | ≥ 60% | **FAIL** |

Median **7** components per volume, mean **1.37** close pairs, max 8. **Closed unamended.** The
thresholds are not being moved, and the probe on 10 volumes that showed the same thing was not
treated as the gate — the registered 100 were run.

**The densely-packed regime this bet is about barely exists in this dataset.** With a median of
one close pair per volume and 43% of volumes containing none, a merger rate cannot be estimated
per volume, and the primary endpoint has nothing to measure.

**⚠ Why, and it is the same root cause that closed bet 3.** These volumes carry a median of 7
labelled components but almost no *close* ones. That is the signature of labels that trace pieces
of **one** surface — fragmented by the crop and by annotation gaps — rather than labels that
capture **adjacent wraps**. The neighbouring sheet, a few voxels away, is simply not labelled. So
the configuration the hypothesis is about is present in the CT and absent from the ground truth.

Bet 3 died because resolvability sampled at labelled voxels conditions on annotatability. Bet 4
dies because merging measured between labelled components conditions on both sheets having been
annotated. **Both are the same defect seen from different angles: this label set does not contain
the failure modes the open problems page cares about.**

⭐ **That is the finding worth keeping from bets 2, 3 and 4 together.** Three independent
measurements, three different quantities, one conclusion:

| measurement | result |
|---|---|
| label placement (bet 2 follow-up) | centres already on the CT ridge, signed offset **+0.008 vox** |
| sheet resolvability (bet 3 gate) | median CNR **272**, only **2.9%** below 50 |
| adjacent-wrap proximity (bet 4 gate) | median **1** close pair per volume, 43% have none |

**The 892-volume public label set is drawn overwhelmingly from high-contrast, well-separated,
correctly-placed sheet.** It supports recall and precision studies. It does **not** support
studies of compressed regions, adjacent-wrap merging, or scan-limited areas — the failures the
team names — because those regions are systematically absent from the labels rather than merely
rare in them. That is a concrete, evidenced statement of the page's own *"densely labeled 3D
training data is the bottleneck"* and *"labels ... avoid ambiguous regions"*, and it explains why
a model trained on this set should not be expected to generalise into exactly the regions that
matter.

**What a continuation would need**, and it is not an amendment to this bet: labels that include
adjacent wraps. Candidates are @Jinhojeong's incompleteness candidates, VC3D-traced multi-wrap
segments, or synthetic phantoms with known separation. Any of those is a different study with a
different sampling frame and needs its own preregistration.

**Elapsed: under two hours from registration to closure.** Which is what gate zero is for, and
the second time today.
