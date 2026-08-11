# Physical labels: released-baseline operational comparison

Status: **preregistered after the causal sentinel failure and before any corrected physical
outcome**.

## Why this is a separate protocol

The earlier causal normalization A/B required an old-path reproduction Dice of at least
0.999 on both scrolls. PHerc0139 passed, but PHerc1203 scored 0.9989616976, so that
protocol correctly stopped before PR #1386 inference. Its cutoff remains failed and will
not be weakened.

This protocol asks a different operational question:

> On the same truth-only-selected physical blocks, is the prediction produced by the exact
> PR #1386 pipeline better than the binary m7 artifact users can download today?

The released artifact is the baseline by definition; it does not need to be regenerated.
A positive result would support deploying the exact candidate over the released artifact.
It would **not** prove that normalization is the sole causal difference, because small
runtime, storage, and historical inference differences cannot be excluded.

## Evidence known at lock time

- The public baseline aggregate scores reported by villa PR #1382 were already known.
- The old-path sentinel outcomes and their threshold-edge diagnosis were known and are
  published in `PHYSICAL_NORMALIZATION_AB_SENTINEL_RESULT.md`.
- No corrected probability array has been generated for any of the 64 physical blocks.
- No corrected arm has been scored against either physical label volume.

## Frozen inputs and blocks

This protocol reuses `results/physical_normalization_ab/manifest.json` byte-for-byte:

- 64 deterministic 64^3 L1 score cubes, 32 per scroll and eight per z stratum;
- coordinates selected from released label bits only, before any real-arm inference;
- the same label archives, CT volumes, released m7 artifacts, m7 checkpoint, and hashes;
- corrected villa PR #1386 at
  `f74929a643095ce422ea4d9b70c25ae2b233a000`; and
- physical metric definitions tied to villa PR #1382 at
  `5408c48d9db0558a78118d24fe9919ee63b204ee`.

No block may be added, removed, or replaced. PHerc1203's sparse arc support remains
descriptive; the powered cross-scroll primary is point skill.

## Arms, metrics, and gates

The two operational arms are:

1. the released threshold-0.2 binary m7 artifact; and
2. PR #1386 with normalization resolved from the model plans (`ct`), no normalization
   override, no TTA, and the frozen m7 checkpoint.

Both the fixed 0.2 candidate and truth-blind matched-mass control are scored exactly as in
the source manifest. The primary quantity is `recall_37um - shifted_null_recall_37um`.
The frozen success conjunction is unchanged:

1. positive fixed-threshold point skill on each scroll;
2. pooled paired-block 95% bootstrap CI for the point-skill delta strictly above zero;
3. fixed-threshold far-37-um fraction no more than one percentage point worse;
4. matched-mass point-skill delta nonnegative on each scroll; and
5. all 32 blocks per scroll contribute to the point metric.

All secondary metrics, arc denominators, side diagnostics, figures, bootstrap seed, and
matched-mass tie rule remain unchanged. Empty predictions remain in denominators.

## No tuning and interpretation

- No threshold, block, model setting, metric, null shift, or gate can change after this
  public lock.
- A technical failure may be retried with its failed receipt preserved. A completed block
  cannot be replaced.
- The scorer runs once after all 64 signed block receipts exist. A negative result is
  published as negative.
- The result must be called a **released-baseline operational comparison**, never the
  failed causal normalization A/B.
- A positive result permits the narrow claim that the exact candidate outperforms the
  released artifact on these registered physical blocks. It does not isolate cause.

## Storage and reproducibility

Each completed block stores the released binary extent, corrected float32 probabilities,
commands, environment, code/input hashes, and stdout/stderr hashes. After the final array
and receipt validate, the runner deletes only the heavy temporary `logits/` and
`merged.zarr/` directories for that attempt. Text logs, receipts, final arrays, and cleanup
receipts remain. Failed attempts are never cleaned automatically.

The machine lock is
`results/physical_released_baseline_comparison/protocol_lock.json`. Exact sequence:

```text
python test_physical_released_baseline_comparison.py
git commit && git push
python run_physical_released_baseline_comparison.py verify ...
python run_physical_released_baseline_comparison.py run ...
python physical_normalization_ab.py score ...
python physical_normalization_ab.py figures ...
```

The runner refuses a dirty worktree, an unpushed branch head, a changed lock or source
manifest, a changed failed-sentinel record, implementation drift, wrong villa commit,
changed model/labels, or changed remote shapes.
