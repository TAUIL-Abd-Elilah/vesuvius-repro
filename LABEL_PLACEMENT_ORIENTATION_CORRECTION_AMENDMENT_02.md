# Amendment 02: follow the upstream sorted-z axis convention

Frozen on 2026-08-20 before any corrected placement outcome was computed. This supersedes only
Amendment 01's ordered-3D-polyline interpretation; its exact-volume binding remains in force.

The raw axis file has 47 negative consecutive z steps, which initially suggested an ordered 3D
path. A source audit shows that interpretation is wrong:

- `KhartesViewer/evolutor`'s `readUmbilicus` reads the CSV rows and calls `zyxs.sort()`.
- `pmh47/spiral-fitting/scroll1_umbilicus.py` embeds this exact control list and states that it is
  specified at original scale.
- All 241 z coordinates are unique. Sorted by z, the controls follow a smooth same-scan axis;
  treating file order as segments instead creates several artificial jumps over 1,000 voxels.

The analysis will therefore sort controls by z and linearly interpolate y and x at each point's
global z. The inward reference is `(0, y_axis-y, x_axis-x)`. Axis-order handling is recorded in the
result. No outcome, cohort, sampling, corridor, threshold, aggregation, or interpretation rule is
changed.

