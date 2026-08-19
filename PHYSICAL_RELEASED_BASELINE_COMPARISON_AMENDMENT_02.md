# Physical released-baseline comparison: Amendment 02

Status: **frozen after one revision-2 inference run and before blending, a corrected
probability array, or any physical score**.

## Observed pre-score failure

Revision 2 ran the first preregistered block,
`PHerc0139-z0064-y1280-x1472`, from CT through all 150 predictor patches. The inherited
runner then failed closed before blending because its required normalization proof token
was stale:

- required: `Using model-declared normalization 'ct' instead of CLI value 'instance_zscore'.`
- observed: `Using model-declared normalization 'ct' instead of CLI/default 'instance_zscore'.`

The observed line is the exact message emitted by the frozen PR #1386 commit. This is a
proof-token compatibility error, not a change to model inference, normalization, blocks,
metrics, thresholds, or gates.

The failed attempt is retained. It produced 155 files under `logits/` totaling
2,832,480,979 bytes, but did not start blending and did not create a corrected probability
array, final block array, or physical score. Its bindings are:

- public protocol commit: `ea3f537570e02cdc95d285a89e005eacc1435360`
- revision-2 lock: `044e5640103f1ff8189fc31fb414962b2a16dcedbdd3942dee83b15378e6c84f`
- protocol receipt: `c6db260c09f160ebcb3648e8235e59f53db3db4ac0f3cf2eedb5a4ce91a770ef`
- failed attempt receipt: `53e331a502134b8d719d6221cb0e81bd601e28839b5d4459fb67ac295c8d55d1`
- predictor stdout: `9d861d59bb86cb3b844d77367a10dba263481a583c09b3149e12c6a0ef4fe386`
- predictor stderr: `df7ebd02e32f7f42823afae3b5dc4c607e6799dbe510d15c94d0ee6444fd8440`
- logits tree SHA: `bb624884856d770d4f22880b24dd8d07054a7378177230124735431f00b43dc3`
  using `sha256(relative_path + NUL + size + NUL + file_sha256 + LF)` over
  lexicographically sorted POSIX-style relative paths

## Narrow repair

Revision 3 changes only the model-execution behavior needed here: the operational proof
token is set to the exact observed PR #1386 line. The wrapper installs it only while
invoking one inherited block run and restores the original constant afterward. Revision 3
writes to `physical_released_baseline_comparison_r3` and reruns from CT; revision-2 logits
are not salvaged or scored.

Revision 3 also adds a checkout-only `.gitattributes` policy for every byte-hashed protocol
file: generated protocol files are pinned to LF, while the already-frozen source manifest
is pinned to its recorded CRLF bytes. This prevents platform line-ending conversion from
invalidating the same public Git content. It does not affect inference or scoring.

All frozen scientific choices remain unchanged, including the 64 truth-only-selected
blocks, exact commits and model, no TTA, model-declared `ct` normalization, thresholds,
controls, bootstrap, success conjunction, and cleanup policy.
