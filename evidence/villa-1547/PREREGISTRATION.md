# PHerc0139 overlap-magnitude replay

Registered before the Villa replay on 2026-08-23.

- Implementation: `TAUIL-Abd-Elilah/villa` commit `ee5aa5bc84faaff14701b3ba4c8e8b27540cac76`
- Input source: `nerln/vesuvius-ladder` commit `7ae4dffe5ffe91c508c26699543320b8e51f6b03`
- Inputs: original intermediate TIFXYZ for PHerc0139 windows w044, w045, w046, and w047
- Parameters: tolerance 2, point stride 1, target-index sampling stride 1
- Metric: directed fraction of queried source points within tolerance of a target surface

Pass gates:

1. w045 to w046 and w046 to w045 are both above 0.75.
2. Every direction for the adjacent controls w044/w045 and w046/w047 is below 0.05.
3. The smaller positive divided by the largest control is above 15.
4. Reports from one and four workers are byte-identical.
5. Every generated `overlapping.json` is byte-identical with and without `--report-json`.

Public timestamp: https://github.com/ScrollPrize/villa/issues/1547#issuecomment-5386343015
