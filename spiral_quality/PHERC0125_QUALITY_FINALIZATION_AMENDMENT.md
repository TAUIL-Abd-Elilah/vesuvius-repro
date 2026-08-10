# PHerc0125 paired-quality finalization amendment

Frozen 2026-08-10 01:38 Africa/Casablanca after attempt 4 wrote an unverified
report and its mandatory self-verifier rejected it, and before any decision
boolean, CT statistic, bootstrap interval, public wording, downstream
authorization, or prize forecast from that artifact was inspected or used.

The rejected artifact is preserved as
`PHerc0125_native_fit_quality.failed-unverified-attempt4.json`, SHA-256
`5b6489e62bec0a5c293047e5f646a48671cff1bdeac8978f1e5678038c59aa9a`.
It is not an accepted quality report.

## Observed deterministic-finalization fact

The only intrinsic-summary mismatch reported by a recursive structural diff
was `deltas.mean_validity`: the stored value was
`0.018814695950939875` and the post-JSON recomputation was
`0.018814695950939986`, a difference of `-1.1102230246251565e-16`.
Canonical JSON sorts string winding keys lexicographically, while the initial
in-memory dictionary retained numeric winding insertion order. Floating-point
summation order therefore changed despite identical underlying values.

## Frozen finalization rules

1. Compute baseline and final mean winding validity over winding IDs sorted in
   ascending numeric order. Require the two winding-ID sets to be identical
   and non-empty as before. This makes the stored computation and verifier
   recomputation order invariant under canonical JSON key sorting.
2. Keep exact report equality in the verifier. Do not widen a scientific or
   verification tolerance to accept the rejected artifact.
3. After the final CT block is evaluated, close both S3 filesystem resources
   created by the pinned `sheetcheck` path: the Zarr volume store filesystem
   and its cached metadata filesystem, using their registered asynchronous
   client creators on the owning event loops. Resource-finalization failure is
   a hard evaluator failure. No filesystem is closed before all registered CT
   rays have been evaluated.

This changes no geometry, site, ray, CT value, bootstrap, intrinsic-checker
input, count, fraction, tolerance, decision threshold, claim boundary, or
downstream authorization rule. Attempt 5 must recompute the complete report;
the rejected attempt-4 artifact may not be edited, promoted, or used as an
input.
