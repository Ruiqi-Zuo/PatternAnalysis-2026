"""
Data audit of the HipMRI 2D slices (keras_slices_data), used as feasibility evidence.

Reports for each split: slice / patient / scan counts, slice shapes, how many slices
contain prostate, prostate area statistics, prostate-bearing slices per scan and the
pixel fraction of every class. Fails if any patient appears in more than one split.

Usage:
    python scripts/audit2d.py --root ~/data/keras_slices_data

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import collections
import os
import sys

import numpy as np

# make the project modules importable when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset import (CLASS_NAMES, NUM_CLASSES, PROSTATE, SPLITS,  # noqa: E402
                     HipMRI2DDataset, check_patient_disjoint, load_nifti_slice)


def audit_split(ds):
    """Print the audit of one split and return its per-class pixel counts."""
    shapes = collections.Counter()
    counts = np.zeros(NUM_CLASSES)
    areas = []                                  # prostate area of every prostate slice
    per_scan = collections.defaultdict(int)     # (case, week) -> number of prostate slices
    scans = set()
    for i in range(len(ds)):
        seg = np.rint(load_nifti_slice(os.path.join(ds.seg_dir, ds.seg_names[i]))).astype(int)
        shapes[seg.shape] += 1
        counts += np.bincount(seg.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        scan = ds.meta[i][:2]
        scans.add(scan)
        area = int((seg == PROSTATE).sum())
        if area:
            areas.append(area)
            per_scan[scan] += 1

    areas = np.array(areas)
    frac = counts / counts.sum()
    span = list(per_scan.values())
    print(f"[{ds.split}] slices={len(ds)} patients={len(ds.case_ids)} scans={len(scans)} "
          f"ids={sorted(ds.case_ids)}")
    print(f"  slice shapes: {dict(shapes)}")
    print(f"  slices with prostate: {len(areas)} ({len(areas) / len(ds):.1%}); "
          f"prostate pixel fraction: {frac[PROSTATE]:.3%}")
    print(f"  prostate area (px): min={areas.min()} median={np.median(areas):.0f} max={areas.max()}; "
          f"< 100 px: {(areas < 100).mean():.1%}")
    print(f"  prostate slices per scan: min={min(span)} median={np.median(span):.0f} max={max(span)}")
    print("  class pixel fraction: " + ", ".join(f"{n}={f:.2%}" for n, f in zip(CLASS_NAMES, frac)))
    return counts


def main():
    parser = argparse.ArgumentParser(description="Audit the HipMRI 2D slice data")
    parser.add_argument("--root", default=os.path.expanduser("~/data/keras_slices_data"))
    args = parser.parse_args()

    datasets = [HipMRI2DDataset(args.root, split) for split in SPLITS]
    check_patient_disjoint(*datasets)
    print("Patient-disjoint check passed.")

    total = sum(audit_split(ds) for ds in datasets)
    print(f"[all] slices={sum(len(ds) for ds in datasets)}; class pixel fraction: "
          + ", ".join(f"{n}={f:.2%}" for n, f in zip(CLASS_NAMES, total / total.sum())))


if __name__ == "__main__":
    main()
