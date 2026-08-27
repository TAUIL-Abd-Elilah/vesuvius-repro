# Operational correction after the seed-17 baseline score

This record is being published after the seed-17 baseline scorer completed,
but before any treatment scorer was started and before any seed-17 comparison
existed.

At `2026-08-27T16:43:16.882255Z`, the sealed runner rejected the completed
baseline report because its metadata recorded SpiralCheck as
`0.0.0+unknown`, while the frozen requirement is `0.4.0`. The scorer was
launched from the exact pinned source checkout, but that checkout was not
installed as a Python distribution in the scoring environment. Consequently,
`importlib.metadata.version("spiralcheck")` could not resolve package metadata.
The source checkout itself remained clean at commit
`d1b50e2957409a870225fb9f5dcc5e25f7a0f9da`, whose `pyproject.toml` declares
version `0.4.0`.

The failure correctly stopped the runner before treatment scoring. The failed
output root and its baseline report will be retained unchanged. A baseline
summary is present in that failure log, so the rerun is explicitly
**partially unblinded**; no treatment result or comparison was available.

## Frozen corrective action

1. Install the exact clean pinned checkout into the existing scoring virtual
   environment with editable package metadata and no dependency changes:

   `python -m pip install --no-deps -e C:\VesuviusProgressTools\spiralcheck`

2. Verify that the imported module resolves to that checkout, distribution
   metadata reports `0.4.0`, the Git commit remains the pinned commit, and the
   checkout remains clean.
3. Rerun the complete sealed runner in a fresh
   `sealed-evaluation-seed17-v3` output root. Both baseline and treatment will
   be rescored; no report will be copied or edited.

The checkpoints, exported-mesh procedure, split, patch scope, fit-input audit,
z range, SpiralCheck source, runner, comparator, statistical gates, and
treatment remain unchanged. This correction supplies missing distribution
metadata only. Seed 23 and seed 101 will use the same corrected environment.

