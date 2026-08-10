# Native full-scroll Spiral fits on prize scrolls without a published native fit

Frozen 2026-08-09 after metadata-only acquisition planning and before this
experiment downloads any PHerc0211 normal chunk, materializes either complete
slab, or starts a fit.

## Prior work and non-duplication boundary

This experiment implements a task proposed publicly by Bruniss on 2026-08-06:
run Spiral on prize-scroll volumes with public Lasagna normals and/or tracks,
using the public manual umbilici. As of the updated 15-channel Discord archive
through 2026-08-08, the team had not yet run Spiral on the 13 competition
scrolls, and no full-scroll native Spiral fit on PHerc0125, PHerc0211, or
PHerc0826 was published. Current GitHub rechecks on 2026-08-09 found no
competing public implementation or full-scroll native Spiral fit for these
three scrolls. Public local VC3D-grown patches are addressed in the dated
overlap amendment below; they do not make the full-scroll task a duplicate.

The following adjacent work is reused or credited rather than rebuilt:

- Villa's native `fit_session`, `spiral_runtime`, resident-pool packer, and
  production `fit_spiral` implementation;
- Sean Johnson's public manual curves, described as good enough for a fit;
- `Nicodol/spiralcheck` at
  `d1b50e2957409a870225fb9f5dcc5e25f7a0f9da` for output-mesh intrinsic checks;
- `DomRusso2/sheetcheck` at
  `7d53893abcc6cc7c0542e483c7266d75ea930885` for CT-support measurements;
- `aistae/scroll-truth` at
  `f75224423ae6b570f714cee2b07a3d1f1a43042c` as the independent-CT
  same-wrap reference on Paris 4. It is not directly reusable here because its
  released reference is Paris 4-only;
- Villa PR #1380's phantom evaluation. It is complementary synthetic evidence,
  not real-scroll truth for these fits.
- Jeff Chen's public `jeff-j-chen/vesuvius` repository at
  `3cb86a6bb1781354fa31c4395d277b3da6f4e2dd` documents local VC3D-grown and
  merged PHerc0211/PHerc0826 surface patches used as ink-model test inputs. It
  is prior local-surface work, not a full-scroll native Spiral fit. The
  repository has no declared license, so its code and artifacts are not copied
  into this experiment or used as quantitative reference data.

The registered automatic-umbilicus experiment is closed in
`SPIRAL_UMBILICUS_TERMINATION_RECORD.md`. PHerc0211 is used here with the public
manual curve, not as an estimator held-out. No result from this experiment may
be pooled into the terminated estimator gate.

## Question and claim boundary

Can current native Villa Spiral, using a public manual curve and bounded public
Lasagna normal/gradient evidence, produce a finite, structurally sane, more
CT-supported full-scroll fit on a prize scroll without a published full-scroll
native Spiral fit?

This is a surface-fitting experiment. It does not claim letters, legibility,
ink, a correct physical winding direction, or automatic umbilicus estimation.
`CW` is initially only a mirror convention: the dense normal loss uses an
unoriented absolute-dot term, and the physical winding sense has not been
measured. Direction-sensitive publication requires measuring or flipping it.

## Frozen scrolls, windows, and acquisition plans

Run in this order. PHerc0211 is not materialized unless PHerc0125 passes the
technical smoke gate.

| role | scroll | full-res z range | stored z chunks | plan SHA-256 | compressed bytes |
|---|---|---:|---|---|---:|
| development/smoke | PHerc0125 | [9984, 10880) | 78--84 | `611b0d2a35fd4699cba09aa179d09525d44e633f8a25f12774b5e178d0371dd5` | 515,313,347 |
| second-scroll replication | PHerc0211 | [9216, 10112) | 72--78 | `4ab157d2ee70d36dc00007754ff6a99c75067d295846269b740099964ab21fb4` | 358,280,712 |

Both plans were built twice from live S3 listings and reproduced byte for byte.
They require identical sparse chunk keys across `nx`, `ny`, and `grad_mag`,
simple ETags, exact manifest and `.zarray` hashes, and seven stored z chunks.
Metadata planning fetched no normal value.

Public inputs:

- PHerc0125 Lasagna stem `20250821151825-lasagna-20260419180421`, manifest
  SHA-256 `90be7bc4da9d9d2305a37f38947934ea708931a53517fa792cd1bb05f65caf00`,
  manual-curve SHA-256
  `458e6ecfcdefef4a1cbd7baa859fd68907f5ea3b18de6f11589893019b5e663f`;
- PHerc0211 Lasagna stem `20250821151803-lasagna-20260419180421`, manifest
  SHA-256 `559e59ea44574280e900ad04bd9157f2df867d77016ac34859a7f43c8c3b2970`,
  manual-curve SHA-256
  `aa07c0cba4458def8092fb54e85299558a038ced408851366463f3a38655595c`;
- CT volumes are the exact masked 9.362-micron scans
  `20250821151825-9.362um-1.2m-113keV-masked.zarr` and
  `20250821151803-9.362um-1.2m-113keV-masked.zarr`.

The materializer is frozen at SHA-256
`f0edc52deea90701fe36130729a0a54f3eb9484c65c9653d85e32e8f8b9bb01a`.
Its focused suite passes 9/9. It verifies every downloaded object against size
and ETag, records SHA-256, rejects stale extra chunks, force-rebuilds native
resident pools, asks Villa's packer to compare 64 random voxels per pool back
to the source Zarr, validates sidecar shapes and file sizes, and emits native
API-v15 requests. Materialization is not a fit result.

Pre-execution runner freeze, still before either slab is materialized or any
fit is started: `run_native_spiral_slab.py` SHA-256
`e6b2b0d5531f382570adc6f620cd2eb22d0d2d4a87d8c45609c765d636556ab8`.
Its focused suite passes 4/4; the combined materializer/runner suite passes
13/13. The runner binds the request and clean Villa revision, calls Villa's
native validator and resident session, runs absolute milestones, rejects
non-finite metrics/warnings, validates every preview plane and winding range,
validates every saved checkpoint with Villa's own container validator, hashes
the complete preview tree, and writes atomic progress/evidence records. It
does not decide whether the fit is accurate.

The exact upstream base was checked out as a clean detached worktree before
materialization. With the frozen Python/dependency overlay, its sparse-cache,
headless-session, and dense-spacing suites pass 59/59 with one expected skip.
This directly exercises absent sparse bricks, z-ROI loading, native preview
publication, patch-disabled validation, and the spacing modes used here.

## Frozen fit configuration

Use current upstream Villa base
`07bb743eb0382e4d94217f49128b126c4b0a9682` unless a later upstream commit is
audited and recorded before the first fit. The request uses:

- patches disabled;
- dense normal weight 100;
- gradient-magnitude dense spacing weight 12;
- umbilicus weight 1.25;
- symmetric-Dirichlet weight 10;
- minimum-spacing weight 2;
- all shell, patch, PCL, track, phase/count/density/attachment weights zero;
- `dense_spacing_mode=grad_mag`, encode scale 1000, factor 1.0 as declared by
  these exact manifests;
- outer winding 130, gap expander 133, initial radial gap 16;
- seed 1, sparse-CUDA storage, Lasagna group 2 and scale 4.

The shell index remains required even with shell loss zero because current
dense-normal, dense-spacing, Dirichlet, and minimum-spacing paths use it to
bound the modelled family. Gap-expander 133 avoids the upstream warning for an
outer index of 130. The exact generated request has already passed Villa's
native resolver/config validator with zero errors for both plans.

## Execution and gates

### A. Materialization gate

For each scroll, require:

1. exact frozen plan and manual-reference hashes;
2. all `3 * object_keys_per_channel` objects present with the frozen sizes and
   ETags and no unplanned chunk file;
3. source-verifying packer exit zero for both resident pools;
4. normal/gradient sidecar shapes equal the pinned Lasagna array shape;
5. materialization evidence says `fit_executed: false` and
   `physical_winding_sense_measured: false`.

### B. Native 300-step smoke gate

The session is loaded from the generated API-v15 request. Export a preview
after step 1 as the frozen pre-fit baseline, then continue to exactly step 300.
Require:

1. native validation remains empty immediately before GPU allocation;
2. the resident session reaches `Ready`, completes exactly 300 iterations, and
   returns to `Paused` without an error;
3. total and component losses at the first reported step and step 300 are
   finite, with no non-finite-gradient warning;
4. a finite, parseable checkpoint is saved;
5. both step-1 and step-300 preview manifests and TIFXYZ coordinates parse,
   contain the declared winding range, and contain no finite-value violation;
6. measured peak RAM/VRAM and wall time are recorded;
7. a new user workload aborts only this experiment at a safe boundary.

This is a compatibility gate, not evidence of a good final fit.

### C. Production milestones

Only after the smoke passes, continue the same seed/configuration to step
15,000. Bruniss reported that most of his fits are close to their final state
by about 15,000 steps, while progress often continues to 30,000. Save a
checkpoint and preview at 15,000. Continue to 30,000 only if all losses remain
finite and the independent checks below show no material regression at 15,000.

Do not tune the loss weights, z window, initial gap, seed, or scroll after
seeing a fit. A materially different configuration requires a new dated
registration and is reported separately.

### D. Independent quality checks

Pretty previews are not a gate. The updated Discord record explicitly warns
that a poor Spiral fit can look solid and that fit constraints can fight each
other. Therefore run both checks below on the step-1 and step-15,000 outputs:

1. **Raw-CT support:** use the exact pinned `sheetcheck` code against each
   scroll's masked CT, level 1, with a frozen deterministic sample shared by
   both outputs. Use 20 surface neighborhoods, at most 20 rays each, 700 microns
   reach, 0.5 level-voxel step, seed 20260809. Record support, gap-structure
   fraction, absolute offset, and every raw sample. Pitch is explicitly
   excluded because `sheetcheck` withdrew that estimator.
2. **Intrinsic whole-family checks:** run the exact pinned `spiralcheck`
   intrinsic mode on winding-separated copies of the same preview geometry.
   Record radial monotonicity, collapsed/inflated spacing, and validity. These
   are ground-truth-free safety checks, not accuracy proof.

A **quantitative CT-improvement claim** is allowed only when the final-minus-
baseline paired support improvement has a 95% bootstrap interval strictly
above zero, or paired absolute-offset reduction has a 95% interval strictly
below zero, and no intrinsic alert materially worsens. If that test does not
pass, report the fit and measurements as a technical outcome only; do not call
it an accuracy improvement.

### E. Two-scroll public-outcome gate

The $20k-shaped outcome requires both PHerc0125 and PHerc0211 to pass the
materialization and native technical gates, produce step-15,000 checkpoints and
previews, and receive the independent checks. At least one scroll must pass the
quantitative CT-improvement rule, and neither may show a significant CT-support
decrease or a catastrophic intrinsic regression.

If only one scroll succeeds, publish it narrowly and do not say the method
generalizes. If both fail quality, preserve the negative result and stop. A
successful geometric fit still does not establish visible letters or reading;
First Letters requires a separate ink-detection and human-verification path.

## Resource and publication rules

- Start GPU work only after three consecutive checks with no competing user
  workload, at least 12 GiB free system RAM, and at least 20 GiB free GPU RAM.
- Metadata planning and bounded download may run earlier only if they do not
  compete with a user workload; native packing and all fitting wait for the
  safe window.
- Hash every request, code revision, checkpoint, preview manifest, evaluator
  input, and evaluator report. Never overwrite a completed run directory.
- Recheck public GitHub and Discord immediately before publication for a newer
  fit or overlapping work.
- Credit every reused project above. Do not present this as inventing Spiral,
  manual umbilici, resident pools, or CT-support evaluation.

## Pre-execution overlap amendment — 2026-08-09 21:08 Africa/Casablanca

This amendment was recorded while only the unrelated PHercParis4 diagnostic
was materializing and before any PHerc0125/PHerc0211 native-fit normal slab or
fit existed. It narrows wording and records newly found prior work; it changes
no scroll, z window, configuration, seed, milestone or quality threshold.

- A fresh GitHub search for PHerc0125/PHerc0211 code, issues and pull requests
  found no public full-scroll native Spiral fit or competing launcher.
- `jeff-j-chen/vesuvius` at exact commit
  `3cb86a6bb1781354fa31c4395d277b3da6f4e2dd` contains a PHerc0211 merged
  VC3D-grown patch documented as a 360 by 326 tifxyz grid and a PHerc0826
  merged patch documented as 475 by 227. The PHerc0211 summary records two
  merged local surfaces; the repository README describes these as ink-model
  test segments. This disproves any broad claim that these scrolls have no
  surface/unrolling work, but does not overlap the bounded full-scroll native
  Spiral outcome registered here.
- The repository declares no license through GitHub, and its tracked x/y/z TIF
  entries are 131-byte Git-LFS pointers rather than the underlying arrays.
  Therefore none of its implementation or geometry is vendored, redistributed
  or treated as ground truth. It is credited as adjacent prior work. A future
  maintainer-authorized comparison could be registered separately if usable
  licensed bytes become available.

## Guarded-execution amendment — 2026-08-09 21:14 Africa/Casablanca

This amendment was recorded before the PHerc0125 dataset root existed and
before any PHerc0125 native fit had started. It operationalizes the already
frozen PHerc0125 run; it changes no input, z window, fit configuration, seed,
15,000-step quality endpoint, evaluator or claim threshold.

- Execution waits until the existing PHercParis4/PHerc0332 diagnostic queue
  reports its exact terminal `complete` state. Immediately before every child
  process it also requires three consecutive 30-second checks with no detected
  competing competition workload, at least 12 GiB free system memory and at
  least 20 GiB free GPU-0 memory.
- The launcher hash-binds this preregistration, both input plans/references,
  the materializer, the native runner and the clean Villa revision. It
  revalidates every downloaded object, generated request, milestone evidence,
  preview and checkpoint before trusting an existing stage.
- A newly appearing competing workload stops only the launcher's owned child
  process tree. Interrupted materialization may resume only through the same
  manifest-verifying materializer. An ambiguous partial native milestone is
  not accepted or overwritten.
- In addition to the registered step-1/300 smoke and step-15,000 endpoint, the
  unchanged run saves technical continuation checkpoints at steps 1,000 and
  5,000. These are recovery boundaries only: they are not inspected for tuning,
  scroll selection or a quality claim.
- The automatic launcher stops in `quality-pending` after the validated
  15,000-step artifact. It does not authorize PHerc0211, step 30,000, public
  accuracy wording or a prize claim. Those remain conditional on the frozen
  raw-CT and intrinsic evaluations above.

## Queue-priority amendment — 2026-08-09 21:17 Africa/Casablanca

This final ordering amendment was recorded before the PHerc0125 dataset root
existed or any native-fit process had started. A live audit found that the
lower-priority PHerc0332 all-100 reliability rerun had produced only 71/100
summary rows, while its separate failed-19 recovery was already complete and
validated 19/19. Letting that long CPU rerun gate the prize-scroll experiment
would invert the evidence-based August priority.

The first bullet of the guarded-execution amendment is therefore superseded
only as follows: PHercParis4 external-reference validation must finish first,
then the exact hash-pinned queue producer enters `native-fit-released` and
parks all remaining PHerc0332 work. The PHerc0125 launcher accepts only that
exact handoff (or the queue's exact already-complete terminal receipt), while
retaining the same three-check RAM/GPU/competition gate before every child.
The queue does not automatically resume PHerc0332 when step 15,000 finishes;
it records `native-quality-pending` and leaves resources to the frozen raw-CT
and intrinsic evaluation. No PHerc0332 result is deleted or overwritten.
