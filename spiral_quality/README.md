# Spiral fit quality: a falling loss hid geometric collapse

A production Spiral fit on one bounded PHerc0125 slab reduced its optimization loss by
**82.30%**, yet a paired raw-CT and intrinsic-geometry gate rejected it. The CT comparison was
neutral and median pitch collapsed to **20.43%** of its starting value.

This is a negative result and an evaluation reference, not a successful fit.

## Registered result

Scope: PHerc0125 full-resolution z `[9984,10880)`, 896 of 20,840 slices (**4.30%**). This is not
a complete-height or whole-scroll fit.

| measure | step 1 / baseline | step 15,000 / final |
|---|---:|---:|
| optimization loss | 111.705109 | 19.777014 |
| median pitch | 15.990415 | **3.267604** |
| collapsed spacing bins | 0 | **1,817 / 55,194** |
| inflated spacing bins | 0 | **8,669 / 55,194** |
| radial-order violations | 0 | **2,455 / 55,194** |

The registered paired CT intervals all span zero:

| paired measure | estimate | 95% cluster-bootstrap CI |
|---|---:|---:|
| support, final minus baseline | +0.064370 | [-0.132227, +0.200788] |
| absolute-offset reduction, µm | +12.002879 | [-4.063081, +21.038129] |
| gap structure, final minus baseline | -0.154167 | [-0.358333, +0.029167] |

Four intrinsic alerts fired: collapsed spacing, inflated spacing, pitch-scale instability and
radial-order regression. All six machine authorization decisions are false. In particular, this
does **not** authorize an accuracy claim, the next scroll, letters/reading, physical winding
direction, a prize claim or public “improved fit” wording.

## Reproduce the released result

The result JSON includes all 400 registered paired rays and 800 raw CT profiles. The release
verifier rehashes every float32 profile, recomputes the three 10,000-draw cluster-bootstrap
intervals, recomputes all intrinsic deltas/alerts, and derives the all-false decision gate:

```bash
python spiral_quality/verify_spiral_quality_release.py
```

The evaluator's focused unit suite is also included:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest -q spiral_quality/test_evaluate_native_spiral_quality.py
```

The exact evaluator is experiment-pinned: it is a reference implementation of the paired gate,
not yet a turnkey any-scroll CLI. A full raw-data replay additionally requires the original
tifxyz fit previews, the public CT volume, and the pinned `sheetcheck` and `spiralcheck` commits.
Those large tifxyz artifacts are not bundled. The complete statistical result *is* reproducible
from the released raw profiles and intrinsic records.

The two full runner-evidence JSON files are also omitted because their metadata contains
machine-local paths. `PHerc0125_loss_summary.json` extracts only the two loss fields, warnings and
validation status and binds the original runner records by SHA-256. The quality report carries the
same source digests.

## What this contributes

Optimization loss is self-referential: it can improve while the fitted winding family folds or
compresses. This gate couples CT support/offset/gap intervals to independent pitch, spacing,
radial-order and validity guards. On this real production run, those guards caught severe
self-fitting that the loss curve concealed.

That addresses a concrete failure mode described by Spiral users: poor fits can look convincing,
especially without ink. It does not yet meet the stronger Villa #1353 bar that a diagnostic must
help produce an improved fit; this release demonstrates detection and stopping only.

## Credit and non-duplication

- CT primitives are from [`DomRusso2/sheetcheck`](https://github.com/DomRusso2/sheetcheck), pinned
  at `7d53893abcc6cc7c0542e483c7266d75ea930885`.
- Intrinsic geometry is from [`Nicodol/spiralcheck`](https://github.com/Nicodol/spiralcheck), pinned
  at `d1b50e2957409a870225fb9f5dcc5e25f7a0f9da`.
- Phase/surf-SDT and current phase defaults are upstream Villa work in #1203 and #1274; this result
  claims neither method.
- Villa [#1380](https://github.com/ScrollPrize/villa/pull/1380) evaluates synthetic phantoms against
  exact truth and correlates them with real-scroll rankings. It is complementary: this release
  evaluates one failed production fit directly against real CT plus intrinsic geometry.
- Villa [#1382](https://github.com/ScrollPrize/villa/pull/1382) and its linked physical-audit
  repository evaluate binary surface-prediction volumes against cross-resolution truth labels on
  PHerc0139 and PHerc1203, including shifted-null controls. That is also complementary: it audits
  upstream surface predictions, whereas this release compares the geometry and CT agreement of
  two states of a production Spiral winding fit.

No code from #1380, #1382 or the reviewed consumer-GPU project is copied here.

## Files

- `evaluate_native_spiral_quality.py` — exact SHA-256-bound evaluator used for the report.
- `test_evaluate_native_spiral_quality.py` — 16 focused tests.
- `results/PHerc0125_native_fit_quality.json` — raw profiles, summaries and decisions.
- `results/PHerc0125_loss_summary.json` — path-free loss/provenance extract.
- `verify_spiral_quality_release.py` — offline recomputation and integrity check.
- the preregistration and four outcome-blind amendments that governed the run.

The repository is MIT licensed. No maintainer adoption, merge or prize outcome is implied.
