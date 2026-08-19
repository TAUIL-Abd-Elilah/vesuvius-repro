# Released-baseline comparison amendment 01: uint8 binary encoding

Status: **frozen before model inference and before any corrected physical outcome**.

The first revision-1 dry-run on `PHerc0139-z0064-y1280-x1472` stopped while reading the
released baseline, before constructing or invoking the model command. The inherited runner
required literal values in `{0,1}`. The public artifact's frozen extent is uint8 with exactly:

| value | voxels |
|---:|---:|
| 0 | 4,856,580 |
| 255 | 1,041,660 |

There were no other values. The failed attempt receipt SHA-256 is
`54b620f8d19d4292b1ee19aa3596f5480ffa30197805682c2fde309379b2df4a`; its error is
`published baseline contains values outside {0,1}` and it contains no inference field.
The authorized revision-1 protocol receipt SHA-256 is
`9143b5279c80d5065f3b1ee4ecf31687e9ffa5d458c91c4d40d340ff058d9165`.

## Narrow implementation repair

Revision 2 wraps only the released prediction array. Each read must contain values drawn
from `{0,1,255}`; otherwise the run fails. Accepted values are canonicalized with
`value != 0` to uint8 `{0,1}` before the unchanged inherited check and max pooling.

This is representation normalization, not a prediction or scoring change. It is equivalent
to the boolean conversion already used by the causal sentinel and maps both common binary
encodings identically. CT input, model output, blocks, threshold, metrics, matched-mass rule,
bootstrap, and gates are unchanged.

Revision 2 uses the new output namespace `physical_released_baseline_comparison_r2` so the
revision-1 failure remains immutable. No corrected probability array existed when this
amendment was written.
