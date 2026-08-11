# Cross-scan fine-tuning: pre-execution implementation amendment 06

Status: **discovered by public CI after publishing the release-compatible lock, before running a
final-lock GPU preflight, creating a v3 data root, materializing a v3 case, training, predicting,
scoring, or inspecting any v3 outcome**.

The release-compatible execution lock with content SHA-256
`2853ecb55324609971643cadc6afbe745678da5179b7e485c9c544a5ab1b9d2a` passed all 58 execution
tests and was merged into the release branch, where all 72 tests passed locally. The two public
release CI invocations then exposed a portability defect in the new checkpoint-loader unit test:
the test tried to resolve its mock target by importing `nnunetv2`, although the intentionally
minimal release CI environment does not install nnU-Net. Production execution uses the pinned
Villa environment and was not implicated.

That lock was not used to create a v3 data root or any v3 result. It is retained byte-for-byte at
`results/crossscan_finetune/execution_lock.withdrawn-20260811-ci-portability.json` (whole-file
SHA-256 `39cf8cce7468ad308b9da4f54e71b3e932e85d710d04183a8cddb6f8ce02dc6d`).

## Make the unit test dependency-independent

The checkpoint-loader test now installs a minimal in-memory `nnunetv2` module hierarchy for the
duration of the test and injects a mock `load_pretrained_weights` function. This exercises the
same production import and verifies the exact call, safe-global context, allowlist receipt, frozen
checkpoint hash rejection, and unexpected-global rejection without requiring nnU-Net to be
installed in CI. No production loader, checkpoint policy, or experiment behavior changes.

The next lock additionally binds this amendment and the byte-identical withdrawn lock. This
portability correction changes no label mapping, case, fold, seed, duration, optimizer value,
checkpoint allowlist, augmentation, normalization, endpoint, effect threshold, statistical test,
safety gate, visual case, or permitted pilot adaptation.
