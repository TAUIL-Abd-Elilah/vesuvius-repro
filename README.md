# vesuvius-repro — are the published surface predictions reproducible?

Every scroll's published surface prediction comes from a model run whose weights are
public (`hf://scrollprize/surface_m7_nnunet`), against CT that is also public. So the
artifacts everyone builds on *should* be regenerable from public inputs alone. Nobody
had checked.

This checks it, for any scroll, and reports how close the match is and — when it is not
close — what class of explanation is still open.

**Headline: they do reproduce.** Six scrolls sampled across both CT pyramid levels come
back at Dice 0.9996–0.9999, with the residual disagreement confined to voxels sitting on
the decision boundary. One scroll does not: **PHerc. Paris 4**, the Title-prize scroll.

## Results

Scored on the interior of a 256³ region (64 voxels trimmed per face), thresholded at the
0.2 the published artifact is named for.

| scroll | CT level | Dice | disagreeing voxels | verdict |
|---|---|---|---|---|
| PHerc0500P2 | L2 | **1.0000** | **0** | reproduced exactly |
| PHerc1203   | L2 | 0.9999 |    81 (0.004%) | reproduced |
| PHerc0332   | L2 | 0.9998 |   134 (0.006%) | reproduced |
| PHerc0139   | L0 | 0.9997 |   266 (0.013%) | reproduced |
| PHerc0125   | L0 | 0.9997 |   254 (0.012%) | reproduced |
| PHerc0175A  | L0 | 0.9997 |   254 (0.012%) | reproduced |
| PHerc0846A  | L2 | 0.9997 |   281 (0.013%) | reproduced |
| PHerc0211   | L0 | 0.9996 |   384 (0.018%) | reproduced |
| PHerc0841   | L2 | 0.9996 |   404 (0.019%) | reproduced |
| PHerc0191   | L0 | 0.9989 | 1,267 (0.060%) | reproduced |
| **PHercParis4** | **L2** | **0.8907** | **108,044 (5.15%)** | **not reproduced** |
| **PHercParis4** (2nd region) | **L2** | **0.8425** | **158,236 (7.55%)** | **not reproduced** |

Ten scrolls, five at each CT level. In every reproduced case **100% of the differing
voxels lie within 0.01 of the threshold** — that is what float16 storage and autocast
leave behind, and nothing structural remains. One region came back exact to the voxel.

One region of PHerc0846A (z 2460, 36% positive) is worth a note: there the model produced
an unusually flat output — logits spanning [-1.9, 6.5] against [-4.3, 18.3] on a healthy
region, so 97% of voxels cleared the 0.2 threshold. A second region of the *same scroll*
reproduced at 0.9997, so this is a property of that region, not a reproduction failure.
It is recorded here rather than dropped because a region where the published model loses
confidence is worth someone's attention.

### The PHerc. Paris 4 exception

Two independent regions fail, and the usual suspects are ruled out by measurement:

| hypothesis | test | result |
|---|---|---|
| wrong source volume | shape match across all 5 of its volumes and their pyramids | only `20260411134726` L2 matches the prediction; ruled out |
| spatial offset | Dice over a ±2 voxel shift search in all 3 axes | peaks at **(0,0,0)**; ruled out |
| step size | rerun at 0.5 / 0.75 / 1.0 | 0.8907 / 0.8467 / 0.7483 — the default is best; ruled out |
| calibration or precision | Dice over every threshold 0.01–0.95 | headroom **+0.0005**; ruled out |
| the region | a second, independent region | also fails; ruled out |

PHerc. Paris 4 is also the only scroll carrying a second surface model
(`surface-recto-2um-ps256`, a different run), so it visibly receives bespoke treatment.

We are **not** claiming the published prediction is wrong. The claim is narrower and
checkable: it is not reproducible from the published inputs, while six other scrolls are.
If the run used a resampled volume or a configuration that is not in the bucket,
publishing that would close the gap.

## The catalogue

`catalog_predictions.py` walks the public bucket and resolves, for every scroll, which
published predictions exist, which CT volume each names, and which pyramid level each
actually sits on (by shape, not by trusting the filename).

```
scrolls scanned                    : 45
published surface predictions      : 42   across 36 scrolls
declared CT level matches the grid : 42   (L0=28, L2=14)
declared level WRONG               : 0
model families                     : surface-m7 (run 20260413222639) x41
                                     surface-recto-2um-ps256 (run 20260413141734) x1
```

Two things worth knowing that fall out of it:

* **The `L<k>` token in the filenames is accurate on all 42.** The ~9 µm scans are
  predicted at level 0 and the ~2.4 µm scans at level 2, i.e. the model has one working
  resolution (~9.6 µm) and the level is chosen to hit it. You can rely on the token.
* **Nine scrolls have public CT but no published surface prediction**: PHerc0172,
  PHerc1667, PHerc1667Cr1Fr3, PHerc51Cr4Fr8, PHercParis1Fr34, PHercParis1Fr39,
  PHercParis2Fr143, PHercParis2Fr47, PHercParis3.
* **Five scrolls are predicted twice**, once from a 9 µm scan and once from a 2.4 µm scan:
  PHerc0500P2, PHerc0814, PHerc0841, PHerc0846A, PHerc1203.

## How the measurements work

Two things do most of the work, and both are designed to *bound* explanations rather than
suggest them.

**A sweep over every threshold.** Re-thresholding is strictly more generous than any global
precision or calibration change, so the best Dice over all thresholds is an upper bound on
what fp16 autocast, float16 storage or a threshold ambiguity could ever buy. On PHerc0139
before the grid was fixed, that bound was 0.8019 against 0.8001 achieved — which killed the
"it's fp16" hypothesis analytically, before spending any GPU time on it.

**Where the disagreement sits relative to the threshold.** A voxel far from the decision
boundary cannot be flipped by a small perturbation, so the share of disagreement lying far
away is a floor on what has to be explained structurally.

`diagnose_structure.py` adds fingerprints that separate the remaining structural causes:
disagreement against position modulo the patch stride (a grid or blending artifact is
phase-locked), against distance from the region faces (an edge effect rises at the faces),
and against CT intensity (an input difference tracks the data).

## Usage

```bash
# 1. catalogue what is published and what level it lives on
python catalog_predictions.py --out catalog.json

# 2. verify one scroll end to end: pick a region, predict, blend, score
python run_verification.py --scroll PHerc0125 --model m7

# scrolls predicted twice need the level
python run_verification.py --scroll PHerc1203 --model m7 --level 2

# 3. score a blended store you already have
python verify_region.py --scroll PHerc0139 --model m7 \
    --ours outputs/repro/merged.zarr --bbox 8000:8256,3200:3456,3200:3456

# 4. if it did not reproduce, fingerprint the difference
python diagnose_structure.py --scroll PHercParis4 \
    --ours outputs/paris4/merged.zarr --bbox 6600:6856,3200:3456,4032:4288
```

Only public data is touched, anonymously — no credentials and no terms acceptance. The
catalogue step reads metadata only, a few KB per scroll.

## Requirements

The `vesuvius` package (for `run_verification.py`; the scoring tools need only numpy,
zarr and scipy), plus these fixes to `vesuvius.predict`, without which a region-scoped
run does not match a full-volume run and, on the pinned dependencies, does not run at all:

| PR | what it fixes |
|---|---|
| [#1247](https://github.com/ScrollPrize/villa/pull/1247) | `--bbox` took its patch grid from the region instead of the volume, so an ROI run silently disagreed with a full run (Dice 0.80 rather than 0.9997) |
| [#1238](https://github.com/ScrollPrize/villa/pull/1238) | `predict` could not create output stores under the pinned zarr |
| [#1239](https://github.com/ScrollPrize/villa/pull/1239) | `--device cpu` |
| [#1240](https://github.com/ScrollPrize/villa/pull/1240) | `Volume.meta()` crash on `--verbose` |
| [#1244](https://github.com/ScrollPrize/villa/pull/1244) | one dropped connection discarded a whole run; it recovered 23 transient failures during the runs behind this table |
| [#1241](https://github.com/ScrollPrize/villa/pull/1241) | `--bbox` itself (merged) |

## Licence

MIT.
