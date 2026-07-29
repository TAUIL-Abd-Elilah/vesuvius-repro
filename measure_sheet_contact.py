#!/usr/bin/env python3
"""How much labelled sheet sits across a thin gap from another sheet?

WHY THIS EXISTS. The faint-sheet ablation (`ablate_faint_sheet.py`) was a clean negative:
dark-tercile recall did not move while bright recall fell 0.75 -> 0.35. On ScrollPrize/villa
issue #191, @Jinhojeong proposed a mechanism -- the attenuation is geometry-blind, so it
also darkens *contacts*, where two sheets sit close enough that intensity has already
stopped separating them. Those cases are unresolvable by construction, so the arm would be
spending capacity on them while degrading the bright mass that was working. They report
models merge sheets at sub-4-voxel contacts.

That predicts a fix: attenuate only where the sheet is NOT in contact. This script measures
whether that fix could bite, BEFORE spending GPU time on it, by asking what fraction of
labelled sheet is actually in contact at a given gap.

METHOD. A morphological closing fills any gap thinner than 2*r voxels; whatever the closing
adds was a thin gap, and the sheet voxels bordering it are the contacts. Reported over
random crops from the same public labelled volumes the ablation trains on.

THE RADIUS IS NOT TUNED. It is swept, and the row that matters is the one matching the
externally reported sub-4-voxel figure. Picking r to produce a convenient contact fraction
would be exactly the "threshold chosen by eye" this project has a rule against.

    python measure_sheet_contact.py [--crops 40] [--volumes 96]
"""
import argparse
import random

import numpy as np

import ablate_faint_sheet as A


def contact_mask(m: np.ndarray, r: int) -> np.ndarray:
    from scipy.ndimage import binary_closing, binary_dilation, generate_binary_structure
    st = generate_binary_structure(3, 1)
    gap = binary_closing(m, structure=st, iterations=r) & ~m
    if not gap.any():
        return np.zeros_like(m)
    return binary_dilation(gap, structure=st, iterations=r) & m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", type=int, default=40)
    ap.add_argument("--volumes", type=int, default=A.N_VOLUMES)
    ap.add_argument("--min-sheet", type=int, default=200)
    args = ap.parse_args()

    tr, _ = A.load_split(args.volumes)
    vols = A.Volumes(tr)
    rng = random.Random(0)
    crops, tries = [], 0
    while len(crops) < args.crops and tries < args.crops * 20:
        tries += 1
        _, lb = vols.crop(rng, "baseline", A.CROP)
        m = (lb == 1)
        if m.sum() >= args.min_sheet:
            crops.append(m)

    print("crops %d   mean sheet occupancy %.3f"
          % (len(crops), float(np.mean([m.mean() for m in crops]))))
    print()
    print("  %-9s %11s %11s %10s" % ("radius", "closes gap", "mean frac", "median"))
    for r in (1, 2, 3, 5, 8, 12):
        fr = [contact_mask(m, r).sum() / m.sum() for m in crops]
        print("  %-9d %8d vox %11.3f %10.3f" % (r, 2 * r, np.mean(fr), np.median(fr)))
    print()
    print("Row to read is the 4-voxel one: that is the externally reported contact scale.")


if __name__ == "__main__":
    main()
