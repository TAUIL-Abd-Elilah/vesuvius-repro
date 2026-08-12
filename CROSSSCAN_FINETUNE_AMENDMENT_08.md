# Cross-scan fine-tuning: pre-outcome claim-boundary amendment 08

Status: **recorded after maintainer review of Villa PR #1382 and before any terminal final
result, candidate downstream execution, or fine-tuned release existed or was inspected**.
At `2026-08-12T23:03:53Z`, the resource-gated final watcher was idle, only seed 40's two
training receipts existed, `final_result.json` did not exist, and no release root existed.
No prediction value or efficacy metric was opened for this amendment.

## Correct label provenance and terminology

Villa maintainer @bruniss closed PR #1382 at `2026-08-12T21:54:06Z` and noted that
PHerc1203 has no ground-truth labels. That review exposed an overclaim in this experiment's
legacy `physical truth` wording.

The external label release is independent of surface-m7 and of this experiment, and it is
derived from a separately acquired 2.403 um PHerc1203 scan registered into the public
9.362 um frame. It is not, however, organizer-issued ground truth or human annotation. The
upstream construction in `code/pass10_1203.py` is automated:

- the material threshold is selected from the intensity-histogram valley and material is
  `hiT > threshold` inside an eroded valid mask;
- centerline bits are local maxima of a material-mask distance transform; and
- recto-band bits use a smoothed-CT normal field, orient the normal radially from the slice
  centroid, test for air 1.5 voxels inward, and dilate the result once.

Accordingly, the canonical first-use description is:

> external, model-independent recto reference masks derived from a separately acquired
> 2.403 um PHerc1203 scan and registered into the public 9.362 um frame.

Short form: **registered high-resolution-scan-derived reference masks**. These are
scan-derived proxy labels, not official/human ground truth.

## Interpretation of the frozen experiment

The frozen cases, folds, seeds, endpoints, thresholds, tests, and gates do not change. This
amendment changes only what those measurements can establish:

- the primary endpoint is agreement with held-out registered PHerc1203 reference masks;
- the safety endpoint is agreement with independently registered PHerc0139 reference masks;
- a positive endpoint shows that the treatment learns and transfers this automated label
  construction better under the registered measurement; and
- it does **not**, by itself, prove that model errors were corrected or that a surface is
  physically correct.

Legacy `truth`, `physical_truth`, and `POSITIVE_DEPLOYABLE` tokens in frozen files remain
byte-for-byte historical schema identifiers. They do not override this claim boundary, and
the last token is only the name of the registered machine-verdict bucket.

Any positive machine verdict may authorize the already-preregistered diagnostic downstream
run, but no public model-improvement or deployment claim is permitted without image-backed
review of all fixed panels against the registered high-resolution scan. Generalization still
requires organizer-run hidden evaluation; downstream topology evidence alone is not a
substitute for that review.
