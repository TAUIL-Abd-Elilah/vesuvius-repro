# Cross-scan fine-tuning: pre-outcome implementation amendment 01

Status: **before materialization, pilot training, or any experiment prediction**.

The design commit and case plan fixed the scientific question, truth mapping, candidate cases,
folds, seeds, pilot adaptation, endpoint, and effect gates. This amendment resolves details that
the outcome-blind plan intentionally left to the public execution implementation. It changes no
case, target, fold, seed, threshold, effect gate, or permitted adaptation.

## Public two-step execution lock

The training, inference, scoring, and test files are committed and pushed first. The freeze
command then records their byte hashes, the plan's file and content hashes, and the exact clean
villa nnU-Net Git tree. That generated execution lock is committed and pushed in a second commit.
Every state-changing command requires a clean public branch containing that lock and refuses any
implementation hash drift.

## Batch-one foreground sampling

nnU-Net's default positional oversampling rule rounds a fraction across items in a batch. At
batch size one, 0.5 would collapse to either zero or one rather than produce a 50% mixture. The
locked trainer therefore uses the loader's supported probabilistic oversampling rule at 0.5,
with single-process augmentation and the declared seed. The spatial and intensity augmentations,
deep supervision, loss, and all other sampling behavior remain the standard nnU-Net path.

## Exact endpoint and matched-mass ties

Average precision uses the standard non-interpolated precision-recall definition and aggregates
all equal-probability voxels at one threshold. Randomized tests, including tied scores, must match
scikit-learn to numerical precision before freeze.

For the matched-mass secondary endpoint, the initial budget is the number of valid voxels at
probability at least 0.2. Candidate voxels are ordered by descending probability with a stable
sort; ties therefore resolve by frozen plan case order and then C-order voxel index. Exactly the
initial count is selected. This tie rule is truth-blind.

## Visual cases

The design requires hash-selected, non-cherry-picked panels but the first machine plan did not
materialize the eight IDs. The execution lock fixes one existing primary block per scroll and z
stratum by the smallest SHA-256 of the literal string crossscan-visual-v1 plus the block ID. The
displayed score-cube slice is index 32 for every case. Thus the plan's block pool never changes,
and the exact IDs and slices become public before the pilot. Each fixed panel contains all six
seed predictions plus their mean, with separate probability-addition and probability-removal
views. The final result records each panel's byte hash.

## Content-hashed execution chain

Materialization, preprocessing, training, and inference receipts are content-hashed and tied to
the same plan and execution lock. Training rehashes the complete preprocessed dataset. Inference
rehashes its materialized CT/truth case, verifies the frozen coordinates, and verifies the source
checkpoint receipt before accepting either a new or resumed prediction. Checkpoint paths are
data-root-relative so published receipts do not depend on one machine's directory layout.

## Fully specified outcome buckets

POSITIVE-DEPLOYABLE and POSITIVE-WITH-SAFETY-REGRESSION use the conjunction in the original
preregistration. REGRESSION is its negative-direction counterpart: mean delta at most -0.010,
at least five of six negative seeds, and two-sided p below 0.05. NULL requires the mean magnitude
below 0.010 and the entire 95% t interval inside [-0.010, +0.010]. Every remaining completed
result is INCONCLUSIVE-UNDERPOWERED. Pilot failure after the single permitted 4,000-step retry
remains TARGET-UNLEARNABLE.

The machine result includes the pooled endpoint, every individual block, z stratum, and
difficulty bin for every seed. Pilot attempts, final results, and fixed visual outputs refuse
overwrite.

Primary and safety inference cannot start until a content-hashed pilot PASS tied to the same plan
and execution lock exists. The scorer refuses to overwrite either a pilot verdict or final result.

## Truth-support audit before freeze

All selected label crops were opened once without CT or predictions to verify that the declared
loss and endpoint are defined. All 288 training/validation crops contain both supervised classes;
the smallest level-1 crop has 2,658 background and 12,275 recto voxels. The 32 pilot cubes contain
1,064,375 positives and 1,295,421 negatives; PHerc1203 primary has 845,894 and 878,626; untouched
PHerc0139 has 2,104,399 and 3,106,112. Every individual evaluation cube contains both classes,
and every z stratum has nonzero positive and negative support. These are truth-only eligibility
checks, not model outcomes, and no case was added, removed, or reordered after measuring them.
