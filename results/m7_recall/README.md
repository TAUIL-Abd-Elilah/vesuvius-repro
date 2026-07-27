# m7 recall benchmark — read this before using the numbers

One JSON per volume from `bench_m7_recall.py`, over the 868 public Kaggle
surface-detection volumes. 826 have labelled sheet in the scored interior.

## `recall` is valid. `precision_INVALID` is not.

The labels have three classes and m7's own `dataset.json` names them
`{0: background, 1: surface, 2: ignore}`. Class 2 is the **majority of a typical
volume (~59%)** and must be excluded from scoring.

The run that produced these files folded `ignore` into background, so predictions
landing in unscored regions were counted as false positives. That understates
precision badly, and the field is renamed `precision_INVALID` rather than deleted so
the record of the error survives.

**Recall is unaffected** — it only ever looks at class-1 voxels — so every recall
figure in these files stands, including the headline: median 90.8%, a quarter of
volumes below 80%, and 41 of 3,400 labelled sheet components barely recovered.

## These are not held-out numbers

The same `dataset.json` says `numTraining: 786`, shapes `[320, 314, 314]`, same label
scheme, same `.tif`/`SimpleTiffIO` format. **This is almost certainly m7's own
training data.** The benchmark is therefore a fit-quality measurement, not a
generalisation one — which makes the low-recall tail more notable, not less.
