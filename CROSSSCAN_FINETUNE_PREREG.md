# Cross-scan physical-truth fine-tuning

Status: **pre-outcome design lock**. No pilot model, fine-tuned checkpoint, or prediction from
this experiment exists at this status. The outcome-executing training, inference, and scoring
scripts must be public and machine-locked before the pilot starts.

## Question

Does fine-tuning the released surface-m7 nnU-Net on model-independent PHerc1203 recto labels
improve its predictions on spatially held-out physical truth, without degrading performance on
the completely untouched PHerc0139 scroll?

This is a core surface-model experiment, not another analytic audit. A null or regression will
be published under the same protocol as a positive result.

## Why this is not the existing retraining result

The closest public experiment is Jinhojeong's six-seed v1-versus-v2.0 weld-boundary A/B on
PHerc1218. It trained on two model-derived instance trees and found a tight null: mean paired
delta -0.0020 with a 95% interval of [-0.0075, +0.0036]. That result is an important prior, but
it is not this intervention.

Here the positive labels come from an independently registered 2.403 um physical scan of
PHerc1203. They do not depend on m7, a repair solver, or either arm's predictions. The source
release explicitly identifies model training as the missing next test. GitHub code and issue
searches through 2026-08-11 found no public training run using labels1203_L1.zarr or
labels0139_L1.zarr. If an earlier equivalent run is found before execution, this protocol stops
for a duplicate review.

## Inputs and licensing

- Initial model: the released surface-m7 fold-0 checkpoint, architecture, plans, and intensity
  statistics, all byte-hashed in the machine plan.
- Training truth: labels1203_L1.zarr from physical-audit release v1.0, registered from the
  2.403 um scan into the public 9.362 um PHerc1203 frame with 2.38 um median held-out error.
- Safety truth: labels0139_L1.zarr from the same release, registered independently with 4.09 um
  median held-out error.
- CT: the exact public level-0 volumes named by the source label metadata.
- Primary blocks: the 32 PHerc1203 and 32 PHerc0139 blocks already selected from truth alone in
  the public physical-normalization manifest. They cannot be replaced.

The code is MIT. Labels, materialized training crops, numerical results derived from them, and
fine-tuned checkpoints are released as CC BY-NC 4.0. This matches both the source-data license
and the live Vesuvius Challenge rule that training datasets be published under CC BY-NC 4.0.

## Training target

The released uint8 label bits are converted at level 1 to the existing m7 class convention:

- class 1: valid AND recto_band;
- class 0: valid AND NOT material AND NOT recto_band; and
- ignore class 2: every other voxel.

The recto rule is applied last. This preserves the source label's intended one-voxel dilation
just outside material. Material away from the measured recto band is ignored rather than
invented as background. The target is upsampled to level 0 by exact 2x nearest-neighbor repeats
on all three axes. No prediction is used to create or filter a target.

## Outcome-blind sampling

The planner reuses the source manifest's level-1 candidate lattice. Eligible training candidates
must have at least 50% valid voxels and at least 4,096 recto-band voxels in the central 64-cubed
score cube. Every primary coordinate is excluded.

Within each of four z strata, candidates are split into four fixed difficulty bins using the
fraction of material marked boundary_poor: [0, 0.40), [0.40, 0.80), [0.80, 0.95), and
[0.95, 1]. A public SHA-256 rank allocates, per stratum and bin:

- 16 training cases;
- 2 internal-validation cases; and
- 2 pilot-gate cases.

Thus each z stratum has 64 training, 8 internal-validation, and 8 pilot cases. A training case is
a 96-cubed level-1 label crop paired with the exact 192-cubed level-0 CT crop. Candidate spacing
is 128 level-1 voxels, so allocated label crops do not overlap. The machine validator also proves
that no training label crop intersects a pilot or primary score cube for the model that scores it.

## Spatial cross-fitting

Two complementary models are trained for every seed:

- even model: train on z strata 0 and 2; score strata 1 and 3;
- odd model: train on z strata 1 and 3; score strata 0 and 2.

Each model receives 128 training cases and 16 internal-validation cases. The final epoch is used
regardless of internal-validation performance; the validation set is only a training-health
diagnostic. Combining the complementary held-out predictions yields all 32 PHerc1203 primary
blocks without ever scoring a block with a model trained on its z stratum.

PHerc0139 is never used for target construction, sampling, training, step selection, or the pilot.
For its safety endpoint, the even and odd probabilities are averaged within seed before scoring.

## Frozen recipe and one pilot adaptation

The model is the full m7 3d_fullres ResidualEncoderUNet initialized from the exact released best
checkpoint. Training uses 192-cubed patches, batch size 1, standard nnU-Net augmentation and deep
supervision, mixed precision, AdamW at 1e-4 with weight decay 1e-5, and cosine decay to 1e-6.
Python, NumPy, and Torch receive the same declared seed; augmentation runs single-process; cuDNN
benchmarking is disabled and deterministic behavior is requested. Every receipt records hardware,
software, hashes, command, seed, and final checkpoint hash.

Before the six inferential seeds, seed 39 trains both folds for 2,000 optimizer steps. It is scored
only on the 32 separate pilot cases. The trainability gate passes if pooled physical-truth average
precision improves by at least 0.005 and no z stratum regresses by more than 0.005. If it fails,
the only allowed adaptation is to restart both seed-39 models from the original checkpoint for
4,000 steps. A pass fixes 4,000 steps for every inferential run. A second failure is published as
TARGET-UNLEARNABLE and no primary block is opened. No learning rate, target, crop, sampler,
endpoint, or gate may change.

If the pilot passes, both folds are trained from scratch for seeds 40 through 45. Pilot checkpoints
are not inferential models.

## Inference and endpoints

Both the initial and fine-tuned checkpoints use the same local deployment harness: a 256-cubed
level-0 CT context, 192-cubed sliding windows, 50% overlap, Gaussian blending, plan-derived
CTNormalization, and no test-time mirroring. Only the central 128-cubed level-0 region is reduced
by 2x2x2 max pooling to the frozen 64-cubed level-1 score cube.

The primary endpoint is pooled supervised-voxel average precision on PHerc1203. Positives are
valid recto-band voxels; negatives are valid non-material, non-recto voxels; all remaining voxels
are excluded. Average precision is threshold-free and is computed over the same 32 blocks for the
initial model and every seed's cross-fitted prediction. Seed, not voxel or block, is the
inferential unit.

Primary success requires all of the following:

1. mean fine-tuned-minus-initial average-precision delta at least +0.010;
2. at least five of six seed deltas positive;
3. a two-sided one-sample t test over the six deltas below 0.05;
4. all 32 PHerc1203 blocks contributing; and
5. mean PHerc0139 safety delta no worse than -0.005.

The +0.010 gate is a practical effect requirement, not merely a significance cutoff. With six
seeds and the closest published across-seed standard deviations near 0.0045 to 0.0053, an effect
around 0.007 to 0.008 is the approximate 80%-power range; the declared gate is stricter.

Secondary endpoints are recto recall, precision, and Dice at (a) each seed's threshold matched to
the initial model's threshold-0.2 positive mass and (b) the fixed threshold 0.2. Results are also
reported by z stratum and difficulty bin. Exact matched-mass tie handling is frozen in the scoring
implementation before the pilot.

## Visual and reproducibility gates

Before scoring, every prediction receipt, array shape, dtype, finite range, checkpoint hash, and
case coordinate must validate. Hash-selected slices fixed by the plan show CT, physical truth,
initial prediction, fine-tuned prediction, additions, and removals for every reported seed summary.
Panels are generated for all preselected visual cases; none can be removed for looking unfavorable.

Outcome buckets are POSITIVE-DEPLOYABLE, POSITIVE-WITH-SAFETY-REGRESSION, NULL, REGRESSION,
INCONCLUSIVE-UNDERPOWERED, TARGET-UNLEARNABLE, or TECHNICAL-FAILURE. Every bucket publishes the
six-seed table, confidence interval, block-level machine output, receipts, and visual panels.

## Scope limits

This tests one architecture, two physical-label volumes, one target construction, and a short
fine-tuning regime. A positive result establishes held-out improvement under that regime; it does
not prove full-scroll tracing improvement or replace downstream GrowPatch/spiral evaluation. A
null does not invalidate the physical labels. PHerc1203's recto band covers only a minority of its
material because much of the scroll is fused, so the claim remains limited to regions where the
physical scan supports this target and to transfer measured on the untouched safety scroll.
