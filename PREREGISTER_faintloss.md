# Pre-registration — `faintloss` arm

**Written 2026-07-29, BEFORE any training run of this arm existed.** July's ablation file
records that its criteria were set after looking at results; this exists so that cannot
happen twice. Nothing below may be edited after the first run completes.

## Hypothesis

The published surface model under-recalls *faint* sheet: missed voxels are ~10.3% darker
than found voxels inside the same volume (161 of 201 volumes), and a model trained from
scratch with flips only recalls **0.32** of the darkest GT-sheet tercile against **0.75** of
the brightest — a 2.3x deficit reproduced in every one of three seeds.

The `faint` augmentation arm attacked this by making training sheet fainter and **failed**:
dark recall unmoved (−0.016), bright recall 0.75 → 0.35. Diagnosis: it shifted the whole
training intensity distribution darker and the model lost the bright regime.

`faintloss` attacks the same measurement from the optimisation side. It does not modify the
input at all. Per-voxel CE multiplier where the label is sheet:

    m = 1 + FAINT_ALPHA * (1 - intensity),  FAINT_ALPHA = 2.0

so the darkest sheet voxel carries 3x the weight of the brightest. The 3:1 ratio is anchored
to the measured 2.3x tercile deficit, **not** selected by trying values.

## Validated before launch

- `alpha = 0` reproduces `CrossEntropyLoss(weight=w, reduction='mean')` **exactly**
  (1.4056029 vs 1.4056028). Without this the arms would differ by more than the intervention.
- Darkening sheet voxels changes the loss; darkening background voxels does not.
- Gradient is finite and non-zero under autocast.
- The weight is a function of **input intensity**, available at test time, so it is a
  reweighting and not a label leak. Nothing changes at inference.

## The bar — fixed now

Baseline is the existing 3 seeds (sheet Dice 0.1560 ±0.0016; recall dark 0.3233 ±0.0312 /
mid 0.5777 ±0.0337 / bright 0.7474 ±0.0425). `faintloss` runs 3 seeds, same split, schedule,
architecture, head.

**SUCCESS requires BOTH:**
1. dark-tercile recall gain **> 0.03** (the across-seed spread), and
2. bright recall and sheet Dice **not degraded beyond their own spreads**
   (bright ≥ 0.705, Dice ≥ 0.154).

**Outcomes that are NOT success, and must be reported as written here:**
- Gain in dark **and** comparable gain in mid+bright → generic regularisation, not this
  hypothesis. Say so.
- Gain in bright only → contradicts the hypothesis. Say so.
- Dark gain inside the spread → not demonstrated. It is a null.
- Dark gain with bright/Dice degraded → a trade, not a fix. Report both numbers together
  and do not lead with the gain.

**`--compare` decides.** It refuses any difference smaller than the across-seed spread.

## What goes in the July submission

**Only outcome 1 above, and only with all three numbers quoted together.** Anything else is
either omitted or stated as a second negative. A positive from a single seed, or a gain that
required degrading the bright mass, does not go in the submission at all.

Caveat that travels regardless: 96 of 892 volumes, 100 epochs, weak absolute baseline
(sheet Dice 0.156). A null here can be a floor effect, exactly as in the distance-transform
ablation.
