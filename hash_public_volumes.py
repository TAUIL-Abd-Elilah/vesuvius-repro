"""Publish content hashes for the 868 public Kaggle surface volumes.

Purpose. @Jinhojeong pointed out on ScrollPrize/villa#191 that the containment question
-- are the 868 public volumes inside the 1,754-pair
`Dataset059_s1_s4_s5_patches_frangiedt`? -- closes exactly if both sides hash their
volumes, since both are public artifacts. This computes our side. Nothing here needs
their data, a GPU, or the network.

Why a naive checksum would have answered the question wrong. The public volumes are
320^3 on disk. The mirror was described as 300^3. Hashing raw file contents across two
packagings with different padding returns zero matches whatever the truth is, and zero
matches reads as "no overlap" rather than "packed differently". So the exact hash is
published as one key among several, cheapest and strictest first:

  raw_sha256      sha256 over the array exactly as stored: dtype uint8, C order,
                  axes as read. Matches only a byte-identical packaging.
  content_sha256  the same hash taken after cropping to the nonzero bounding box.
                  Invariant to how much zero padding a release carries, which is the
                  difference a 320^3-vs-300^3 packaging would produce.
  content_shape   the crop-to-nonzero extent. This is the quantity nnU-Net records as
                  `shapes_after_crop` in dataset_fingerprint.json, so it is the one
                  field that can be compared against a published model fingerprint
                  without holding that model's data at all.

The label volumes are hashed separately and must not be used as the containment key.
The public labels carry three values (0 background, 1 sheet, 2 ignore) while the mirror
was described as carrying a binary surface band; those cannot hash equal even for
identical source material. They are published for completeness.

Usage:
    python hash_public_volumes.py --images data/kaggle/images --labels data/kaggle/labels
    python hash_public_volumes.py --verify        # re-hash a sample, confirm determinism
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile

HERE = Path(__file__).resolve().parent
EXPECTED_LABEL_VALUES = {0, 1, 2}


def _sha256(array: np.ndarray) -> str:
    """Hash a uint8 array by its voxels, independent of how the file stored them.

    Fed one slice at a time. `.tobytes()` on the whole volume would copy another 32 MB
    per hash, which matters when this has to share a machine with a training run.
    """
    if array.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {array.dtype}")
    digest = hashlib.sha256()
    for plane in array:
        digest.update(np.ascontiguousarray(plane).tobytes())
    return digest.hexdigest()


def _nonzero_bbox(array: np.ndarray):
    """Inclusive-exclusive bounds of the nonzero content, or None if the volume is empty.

    Projects onto each axis rather than calling np.nonzero, which on a nearly-dense 320^3
    volume would materialise ~31M index triples (several hundred MB) to answer a question
    about six numbers.
    """
    bounds = []
    for axis in range(array.ndim):
        others = tuple(i for i in range(array.ndim) if i != axis)
        present = np.nonzero(array.any(axis=others))[0]
        if present.size == 0:
            return None
        bounds.append((int(present[0]), int(present[-1]) + 1))
    return tuple(bounds)


def fingerprint(path: Path, kind: str) -> dict:
    array = tifffile.imread(str(path))

    # Rule 8 of the project notes: a truncated download still opens and still reshapes.
    # Validate the shape rather than the existence, and refuse to hash a partial read.
    if array.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3-D volume, got shape {array.shape}")
    if array.dtype != np.uint8:
        raise ValueError(f"{path.name}: expected uint8, got {array.dtype}")

    record = {
        "sample": path.stem,
        "kind": kind,
        "stored_shape": list(array.shape),
        "raw_sha256": _sha256(array),
    }

    bbox = _nonzero_bbox(array)
    if bbox is None:
        record["content_shape"] = None
        record["content_sha256"] = None
        record["nonzero_bbox"] = None
    else:
        # A view, not a copy; _sha256 walks it a plane at a time.
        crop = array[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        record["nonzero_bbox"] = [list(b) for b in bbox]
        record["content_shape"] = list(crop.shape)
        record["content_sha256"] = _sha256(crop)

    if kind == "label":
        record["label_values"] = sorted(int(v) for v in np.unique(array))
    return record


def run(images: Path, labels: Path, out: Path, limit: int | None) -> dict:
    image_files = sorted(images.glob("*.tif"))
    label_files = sorted(labels.glob("*.tif"))
    if limit:
        image_files, label_files = image_files[:limit], label_files[:limit]

    image_stems = [p.stem for p in image_files]
    if image_stems != [p.stem for p in label_files]:
        raise SystemExit("image and label file names do not correspond one to one")
    print(f"hashing {len(image_files)} image/label pairs from {images} and {labels}",
          flush=True)

    records, started = [], time.time()
    for index, (image_path, label_path) in enumerate(zip(image_files, label_files), 1):
        image_record = fingerprint(image_path, "image")
        label_record = fingerprint(label_path, "label")
        records.append({"sample": image_path.stem,
                        "image": image_record, "label": label_record})
        if index % 50 == 0 or index == len(image_files):
            rate = index / (time.time() - started)
            print(f"  {index}/{len(image_files)}  {rate:.1f}/s", flush=True)

    payload = {
        "description": "Content hashes for the public Kaggle surface volumes. "
                       "See the module docstring for which key answers which question.",
        "source": "Vesuvius Challenge public surface-labelling set",
        "n_pairs": len(records),
        "partial": bool(limit),
        "hash": "sha256 over uint8 voxels, C order, axes as read",
        "records": records,
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return payload


def report(payload: dict) -> None:
    records = payload["records"]
    images = [r["image"] for r in records]
    labels = [r["label"] for r in records]

    print("\n" + "=" * 72)
    print("CENSUS")
    print("=" * 72)
    print("stored shapes      :", dict(Counter(tuple(r["stored_shape"]) for r in images)))
    print("content max-dim    :",
          dict(sorted(Counter(max(r["content_shape"]) for r in images if r["content_shape"]).items())))
    print("distinct raw hashes:", len({r["raw_sha256"] for r in images}), f"of {len(images)}")
    print("distinct content   :", len({r["content_sha256"] for r in images}), f"of {len(images)}")

    values = Counter(tuple(r["label_values"]) for r in labels)
    print("label value sets   :", dict(values))
    unexpected = {v for combo in values for v in combo} - EXPECTED_LABEL_VALUES
    if unexpected:
        print(f"  !! unexpected label values present: {sorted(unexpected)}")

    duplicates = [h for h, n in Counter(r["raw_sha256"] for r in images).items() if n > 1]
    if duplicates:
        print(f"\n!! {len(duplicates)} raw hash collisions inside the public set itself:")
        for h in duplicates[:5]:
            same = [r["sample"] for r in images if r["raw_sha256"] == h]
            print("   ", h[:16], same)
    else:
        print("\nno duplicate volumes inside the public set")


def compare_to_fingerprint(payload: dict, fingerprint_path: Path) -> None:
    """Compare a model's shapes_after_crop against our crop-to-nonzero shapes.

    A published nnU-Net fingerprint records shapes and no voxel content, so shapes are the
    only comparison available against a model whose data we do not hold. Two comparisons
    are printed, and only the first should be quoted.

    Largest-dimension buckets (robust). Both sides are quantised to one number per volume,
    which survives the few voxels of border that differ between packagings.

    Exact 3-tuples (diagnostic only, NOT evidence). These are packaging-sensitive: this
    model's fingerprint reports (320, 314, 314) where the public release crops to
    (320, 320, 320), so an exact-tuple mismatch is what two packagings of the SAME volume
    look like. Reading it as evidence about containment would manufacture a false negative.

    Neither direction proves containment: a shape is not an identifier. The test has real
    power in one direction only -- a bucket occurring more often in training than in the
    public set means the training set is not contained in the public set.
    """
    data = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    train_shapes = [tuple(s) for s in data["shapes_after_crop"]]
    public_shapes = [tuple(r["image"]["content_shape"]) for r in payload["records"]
                     if r["image"]["content_shape"]]

    print("\n" + "=" * 72)
    print(f"SHAPE COMPARISON vs {fingerprint_path}")
    print("=" * 72)
    print(f"model training cases : {len(train_shapes)}")
    print(f"public volumes       : {len(public_shapes)}")
    if payload.get("partial"):
        print("\n  !! PARTIAL RUN - do not quote anything below. Every figure here is a"
              "\n     sample, and on this data samples have reversed sign before.")

    train_bucket = Counter(max(s) for s in train_shapes)
    public_bucket = Counter(max(s) for s in public_shapes)
    print("\n-- largest-dimension buckets (the robust comparison) --")
    print(f"   {'bucket':>8}  {'training':>9}  {'public':>7}  {'public-training':>15}")
    for bucket in sorted(set(train_bucket) | set(public_bucket)):
        t, p = train_bucket[bucket], public_bucket[bucket]
        print(f"   {bucket:>8}  {t:>9}  {p:>7}  {p - t:>+15}")

    short = {b: (t, public_bucket[b]) for b, t in train_bucket.items() if t > public_bucket[b]}
    if short:
        deficit = sum(t - p for t, p in short.values())
        print(f"\n   NOT CONTAINED: {deficit} training cases sit in buckets the public set "
              f"cannot supply.")
    else:
        residual = sum(public_bucket.values()) - sum(train_bucket.values())
        print(f"\n   Consistent with containment; {residual} public volumes are surplus to "
              f"the training count.")
        print("   Consistency is not proof. A bucket is not an identifier, and this says "
              "nothing\n   about WHICH volumes are surplus - only how many could be.")

    exact_train, exact_public = Counter(train_shapes), Counter(public_shapes)
    overlap = sum((exact_train & exact_public).values())
    print(f"\n-- exact 3-tuples (diagnostic only, packaging-sensitive) --")
    print(f"   distinct shapes: training {len(exact_train)}, public {len(exact_public)}; "
          f"{overlap} cases share an exact shape")
    print("   Do not read a mismatch here as evidence about containment.")


def verify(payload: dict, images: Path, labels: Path, sample: int) -> None:
    """Re-hash a random subset from disk and confirm the published values reproduce."""
    rng = np.random.default_rng(0)
    picks = rng.choice(len(payload["records"]), size=min(sample, len(payload["records"])),
                       replace=False)
    print(f"\nre-hashing {len(picks)} volumes to confirm determinism")
    for i in picks:
        record = payload["records"][int(i)]
        for kind, folder in (("image", images), ("label", labels)):
            fresh = fingerprint(folder / f"{record['sample']}.tif", kind)
            for key in ("raw_sha256", "content_sha256", "content_shape"):
                if fresh[key] != record[kind][key]:
                    raise SystemExit(f"MISMATCH {record['sample']} {kind} {key}")
    print("all re-hashed volumes reproduce exactly")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="data/kaggle/images")
    ap.add_argument("--labels", default="data/kaggle/labels")
    ap.add_argument("--out", default=str(HERE / "results" / "public_volume_hashes.json"))
    ap.add_argument("--limit", type=int, default=None, help="hash only the first N pairs")
    ap.add_argument("--fingerprint", default=None,
                    help="an nnU-Net dataset_fingerprint.json to run the sub-multiset test against")
    ap.add_argument("--verify", type=int, default=0,
                    help="re-hash N volumes from an existing output and confirm they reproduce")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.verify:
        payload = json.loads(out.read_text(encoding="utf-8"))
        verify(payload, Path(args.images), Path(args.labels), args.verify)
    else:
        payload = run(Path(args.images), Path(args.labels), out, args.limit)
        report(payload)

    if args.fingerprint:
        compare_to_fingerprint(payload, Path(args.fingerprint))
    return 0


if __name__ == "__main__":
    sys.exit(main())
