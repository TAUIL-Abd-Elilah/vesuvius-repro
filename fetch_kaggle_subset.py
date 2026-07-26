"""Fetch a subset of the public Kaggle surface-detection training pairs.

Data lives in the public HF bucket `scrollprize/datasets` (anonymous, no token):
    surfaces/kaggle/images/sample_NNNNN.tif   320^3 uint8 CT
    surfaces/kaggle/labels/sample_NNNNN.tif   320^3 uint8, classes {0,1,2}

Downloads are resumable and retried, because a mid-stream SSL/connection drop on a
long transfer is routine (same failure mode as ScrollPrize/villa#1244).

Usage:
    python fetch_kaggle_subset.py --n 200 --out data/kaggle
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BUCKET = "https://huggingface.co/buckets/scrollprize/datasets/resolve/surfaces/kaggle"


def fetch(url: str, dest: Path, tries: int = 40) -> bool:
    """curl with resume; returns True once dest exists and curl reports success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl", "-sS", "-L", "-C", "-",
        "--retry", str(tries), "--retry-all-errors", "--retry-delay", "2",
        "--connect-timeout", "30", "--max-time", "1800",
        "-o", str(dest), url,
    ]
    return subprocess.run(cmd).returncode == 0


def intact(path: Path) -> bool:
    """Is this a complete 320^3 volume, or a truncated download?

    A size threshold is not enough. Transfers on this link cut off mid-stream, and a
    short TIFF still OPENS - it just reshapes to (320, 320) instead of (320, 320, 320),
    so a partial volume is silently accepted and trained on. Only the shape settles it.
    Same check as _intact() in ablate_distance_transform.py.

    Opened through a context manager on purpose: the handle must be closed before the
    caller can delete a bad file. On Windows an open handle makes unlink() raise
    PermissionError(WinError 32), which killed a run of this script mid-download.
    """
    try:
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            return tuple(tf.series[0].shape) == (320, 320, 320)
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of sample pairs")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data/kaggle"))
    ap.add_argument("--verify-existing", action="store_true",
                    help="shape-check pairs already on disk instead of trusting them")
    args = ap.parse_args()

    ok = skipped = failed = refetched = 0
    for i in range(args.start, args.start + args.n):
        name = f"sample_{i:05d}.tif"
        img, lab = args.out / "images" / name, args.out / "labels" / name
        if img.exists() and lab.exists():
            # Cheap gate first; the shape check opens the file, so only pay for it when
            # asked. Either way a pair that fails is re-fetched rather than skipped.
            good = (img.stat().st_size > 1_000_000
                    and (not args.verify_existing or (intact(img) and intact(lab))))
            if good:
                skipped += 1
                continue
            print(f"  truncated on disk, re-fetching: {name}", flush=True)
            img.unlink(missing_ok=True)
            lab.unlink(missing_ok=True)
            refetched += 1

        got = fetch(f"{BUCKET}/labels/{name}", lab) and fetch(f"{BUCKET}/images/{name}", img)
        if got and intact(img) and intact(lab):
            ok += 1
        else:
            failed += 1
            # Leave nothing half-written behind: a partial file that survives would be
            # skipped by the size gate on the next run and never repaired.
            img.unlink(missing_ok=True)
            lab.unlink(missing_ok=True)
            print(f"  failed: {name}", file=sys.stderr)
        if (ok + skipped + failed) % 10 == 0:
            print(f"  {ok} fetched, {skipped} present, {failed} failed", flush=True)

    print(f"done: {ok} fetched, {skipped} present, {failed} failed, "
          f"{refetched} replaced -> {args.out}")


if __name__ == "__main__":
    main()
