# Amendment 01: bind the volume and treat the axis as a 3D polyline

Frozen on 2026-08-20 before any corrected placement outcome was computed.

Two input checks performed after the initial plan require a mechanical amendment:

1. Discord discussion from 2026-08-14 warned that umbilicus files need an exact volume identity,
   dimensions, and scale. The axis URL's containing package supplies Scroll1A metadata at
   `volumes/20230205180739/meta.json`: 14,376 slices, height 7,888, width 8,096, voxel size
   7.91 um. These are the same coordinate bounds used by the mapped Scroll1A crops. The run will
   pin both URLs and both SHA-256 values.
2. `umbilicus-scroll1a_zyx.txt` contains an ordered 3D polyline whose z coordinate reverses 47
   times. The initial plan's same-z interpolation is therefore undefined as a representation of
   that path. The physical reference is amended to the vector from each global point to the
   closest point on the full ordered 3D polyline.

No outcome has been read. Cohorts, sampling, corridors, alignment thresholds, aggregation,
bootstrap counts, and the interpretation gate are unchanged.

