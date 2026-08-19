# Fixed independent-scan visual review

This is a post-verdict evidence step, not part of the frozen efficacy decision. It exists
because agreement with an automated registered mask is not, by itself, proof that a model
error was corrected.

The renderer uses the eight cases fixed in the public execution lock and the prospectively
fixed within-case slices `k=16,32,48`. Slice 32 is the original machine-panel slice; 16 and
48 prevent a label-caster slab boundary from making the only image for a case empty. No
slice is selected from model output. For each case/slice it shows:

1. the public 9.362 µm CT context and fixed score-cube locator;
2. the separately acquired scan registered into that frame at approximately 4.5–4.8 µm;
3. the automated scan-derived proxy contour on that image;
4. the initial m7 and six-seed fine-tuned contours on the same image;
5. initial/fine-tuned probabilities and their difference.

Before writing a panel, the renderer also resamples the exact source pyramid level used by
the upstream label caster and requires its thresholded intensities to reproduce every
material bit inside the released valid mask. This is a fail-closed coordinate-convention
check, not an accuracy score.

Every panel says that the reference is an automated proxy, not organizer-issued or human
ground truth. The renderer never changes a case, score, threshold, seed, checkpoint, or
terminal bucket. It records the exact transform, source URLs, Zarr metadata, downloaded
chunk hashes, interpolation coordinates, panel hashes, plan, lock, and final-result identity.
The manifest also binds the public renderer repository, exact clean commit, and canonical
Git source bytes. Its CRLF-to-LF-only canonicalization makes the identity invariant to a
normal Windows or Linux checkout while retaining every other source byte.

## Inputs

- A completed cross-scan data root containing the verified prediction receipts and
  `final_result.json`.
- The exact upstream physical-audit checkout at commit
  `b24e028178f2c8720ba1d16ac53d5f0b6ac00da7`. The renderer pins both exact label-caster
  NPZ transforms by SHA-256 and separately pins PHerc0139's published L0 JSON transform
  for its six-landmark forward check.
- A cache directory for raw, uncompressed public scan chunks.
- A new output directory. Existing output is never overwritten.

The independent scan sources are:

| scroll | source scan | displayed pyramid | registration held-out median |
|---|---|---:|---:|
| PHerc0139 | 20260413113053, 1.129 µm acquisition | level 2, ≈4.516 µm | 4.09 µm |
| PHerc1203 | 20260319130212, 2.403 µm acquisition | level 1, ≈4.806 µm | 2.38 µm |

## Render

```powershell
python crossscan_highres_review.py render `
  --repo D:\path\to\vesuvius-repro `
  --data-root D:\data\crossscan_finetune_v4 `
  --transform-root D:\path\to\pherc0139-physical-audit `
  --cache-root D:\data\crossscan_highres_review_cache `
  --out D:\data\crossscan_highres_review
```

Do not run this command before the terminal result exists. Rendering reads predictions for
the fixed panels; it does not read or summarize efficacy metrics.

The output contains 24 PNGs (eight locked cases × three fixed slices), `review_pack.json`, and
`human_review_TEMPLATE.json`. The manifest is transitive through the panel and downloaded
source-chunk hashes. Rendering rejects an uncommitted renderer, and verification resolves
the recorded commit's Git blob and rejects any canonical source-identity mismatch.

An input-only rehearsal on 2026-08-13 checked all 24 slice registrations against 89,605
released valid-mask voxels and found zero material-bit mismatches. Two original `k=32`
machine-panel slices have no valid proxy voxels because they lie on label-caster slab
boundaries; their prospectively fixed `k=16` and `k=48` companions are caster-verifiable.

## Human review

Copy `human_review_TEMPLATE.json` to a new path, view every PNG at full resolution, and fill
the reviewer, UTC time, per-panel judgments, and nonempty notes. Keep the acknowledgement
exactly as written. A release recommendation requires at least two alignment passes among
the three fixed slices for every case. The allowed recommendations are:

- `RELEASE_WITH_AGREEMENT_ONLY`: the images were reviewed, but public wording remains a
  registered-proxy-agreement claim.
- `RELEASE_WITH_NAMED_IMAGE_SUPPORTED_CASES`: only the explicitly listed fixed cases may be
  described as image-supported corrections. Each listed case must have passing alignment,
  a supported initial disagreement, and a supported candidate correction.
- `DO_NOT_RELEASE`: a registration failure or other visual problem blocks release.

Validate the completed receipt:

```powershell
python crossscan_highres_review.py verify `
  --review-root D:\data\crossscan_highres_review `
  --receipt D:\data\crossscan_highres_review\human_review.json
```

The verifier rejects reordered or omitted cases, changed PNGs, stale pack hashes, future or
missing review times, unnamed reviewers, unsupported claim escalation, and any release
recommendation when a panel's registration is marked failed.

## Claim boundary

The safe general claim is: the treatment changed held-out agreement with automated,
registered high-resolution-scan-derived reference masks. Neither a positive machine verdict
nor topology output proves physical correctness. A model-error correction claim must be
limited to named fixed cases supported by the human receipt; hidden organizer evaluation is
still required for generalization.
