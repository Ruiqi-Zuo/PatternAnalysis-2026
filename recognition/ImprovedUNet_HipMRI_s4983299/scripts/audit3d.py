"""
Data audit of the HipMRI 3D volumes (semantic_MRs / semantic_labels_only), used as
feasibility evidence.

Reports volume shapes, voxel spacing and orientation, class fractions, the prostate's
axial extent, volumes and patients per split (same patient split as the 2D slices), and
whether every organ fits inside the fixed training crop CROP_3D.

Usage:
    python scripts/audit3d.py --root ~/data/HipMRI_3D
    python scripts/audit3d.py --root /home/groups/comp3710/HipMRI_Study_open   # on Rangpur

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import collections
import os
import sys

import nibabel as nib
import numpy as np

# make the project modules importable when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset import (CLASS_NAMES, CROP_3D, NUM_CLASSES, PROSTATE,  # noqa: E402
                     parse_volume_name, split_of_case)

ORGANS = (3, 4, 5)  # bladder, rectum, prostate


def bounding_box(mask):
    """(min, max) voxel index of a boolean mask along each axis, or None if empty."""
    if not mask.any():
        return None
    return [(int(idx.min()), int(idx.max())) for idx in np.nonzero(mask)]


def main():
    parser = argparse.ArgumentParser(description="Audit the HipMRI 3D volumes")
    parser.add_argument("--root", default=os.path.expanduser("~/data/HipMRI_3D"))
    args = parser.parse_args()

    mr_dir = os.path.join(args.root, "semantic_MRs")
    seg_dir = os.path.join(args.root, "semantic_labels_only")
    mr_names = sorted(n for n in os.listdir(mr_dir) if ".nii" in n)
    print(f"{len(mr_names)} MR volumes in {mr_dir}")

    shapes, spacings, orientations = collections.Counter(), collections.Counter(), collections.Counter()
    counts = np.zeros(NUM_CLASSES)
    prostate_frac, prostate_slices = [], []
    volumes_per_split = collections.Counter()
    patients_per_split = collections.defaultdict(set)
    outside_crop = []

    for n in mr_names:
        case, week = parse_volume_name(n)
        split = split_of_case(case)
        volumes_per_split[split] += 1
        patients_per_split[split].add(case)

        img = nib.load(os.path.join(seg_dir, n.replace("_LFOV", "_SEMANTIC")))
        shapes[img.shape] += 1
        spacings[tuple(round(float(z), 2) for z in img.header.get_zooms())] += 1
        orientations["".join(nib.aff2axcodes(img.affine))] += 1

        seg = np.rint(img.get_fdata(dtype=np.float32)).astype(np.uint8)
        counts += np.bincount(seg.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
        prostate = seg == PROSTATE
        prostate_frac.append(prostate.mean())
        # last axis runs inferior -> superior: count the axial slices that hold prostate
        prostate_slices.append(int(prostate.any(axis=(0, 1)).sum()))

        # does every organ lie inside the training crop?
        for label in ORGANS:
            box = bounding_box(seg == label)
            if box is None:
                continue
            for axis, ((lo, hi), crop) in enumerate(zip(box, CROP_3D)):
                if lo < crop.start or hi >= crop.stop:
                    outside_crop.append(f"{n} {CLASS_NAMES[label]} axis {axis}: {lo}-{hi}")

    print(f"label shapes: {dict(shapes)}")
    print(f"voxel spacing (mm): {dict(spacings)}")
    print(f"orientation (axis codes): {dict(orientations)}")
    print("class voxel fraction: " + ", ".join(
        f"{name}={f:.2%}" for name, f in zip(CLASS_NAMES, counts / counts.sum())))
    print(f"prostate voxel fraction per volume: median {np.median(prostate_frac):.3%}, "
          f"min {min(prostate_frac):.3%}, max {max(prostate_frac):.3%}")
    print(f"prostate-bearing axial slices per volume: min {min(prostate_slices)}, "
          f"median {np.median(prostate_slices):.0f}, max {max(prostate_slices)}")
    for split in ("train", "validate", "test"):
        print(f"{split:8s}: {volumes_per_split[split]:3d} volumes, "
              f"{len(patients_per_split[split]):2d} patients {sorted(patients_per_split[split])}")
    print(f"organ voxels outside CROP_3D {CROP_3D}: {len(outside_crop)} cases")
    for line in outside_crop:
        print(f"  {line}")


if __name__ == "__main__":
    main()
