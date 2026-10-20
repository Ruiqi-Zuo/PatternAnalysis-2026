"""
Data loading and preprocessing for the HipMRI 2D prostate slices (keras_slices_data).

Responsibilities:
    - load 2D Nifti image / segmentation slices
    - unify slice shape and normalise intensities per slice
    - verify the official train / validate / test split is patient-disjoint
    - data augmentation applied identically to image and mask

Expected directory layout (as provided on Rangpur):
    keras_slices_data/
        keras_slices_train/      case_004_week_0_slice_0.nii.gz ...
        keras_slices_seg_train/  seg_004_week_0_slice_0.nii.gz  ...
        keras_slices_validate/   keras_slices_seg_validate/
        keras_slices_test/       keras_slices_seg_test/

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

# Label indices in the segmentation masks.
# 3 / 4 identified from anatomy (per-slice label profiles): label 4 starts exactly where the
# prostate ends (superior to the base) -> bladder; label 3 runs posterior to the prostate
# and continues below the apex -> rectum. Slice index increases from inferior to superior.
CLASS_NAMES = ["background", "body", "bone", "rectum", "bladder", "prostate"]
NUM_CLASSES = len(CLASS_NAMES)
PROSTATE = 5

# ~99% of slices are 256 x 128; the rest (256 x 144) are cropped to this size
TARGET_SHAPE = (256, 128)

SPLITS = ("train", "validate", "test")
_NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii(?:\.gz)?$")


def parse_slice_name(fname):
    """Return (case, week, slice) parsed from a HipMRI slice filename."""
    m = _NAME_RE.search(fname)
    if m is None:
        raise ValueError(f"Unexpected filename: {fname}")
    return tuple(int(g) for g in m.groups())


def load_nifti_slice(path):
    """Load a single 2D Nifti slice as a float32 array of shape (H, W)."""
    return nib.load(path).get_fdata(dtype=np.float32)


def center_crop_or_pad(arr, shape=TARGET_SHAPE):
    """
    Centre-crop or zero-pad a 2D array to `shape`.

    Used instead of resizing so that mask labels stay integers and the
    image / mask pixel correspondence is preserved exactly.
    """
    out = np.zeros(shape, dtype=arr.dtype)
    src, dst = [], []
    for size, target in zip(arr.shape, shape):
        if size >= target:  # crop
            start = (size - target) // 2
            src.append(slice(start, start + target))
            dst.append(slice(0, target))
        else:  # pad
            start = (target - size) // 2
            src.append(slice(0, size))
            dst.append(slice(start, start + size))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def normalise(img, eps=1e-6):
    """Per-slice z-score normalisation (scanner intensity ranges vary between scans)."""
    return (img - img.mean()) / (img.std() + eps)


def build_augmentation():
    """
    Geometric augmentation applied jointly to image and mask.

    Wrapping tensors as tv_tensors.Image / tv_tensors.Mask makes torchvision use
    bilinear interpolation for the image and nearest-neighbour for the mask.
    The 256-pixel (vertical) axis is patient left-right, so a vertical flip is an
    anatomically plausible mirror; the anterior-posterior axis is never flipped.
    """
    return v2.Compose([
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    ])


class HipMRI2DDataset(Dataset):
    """
    HipMRI 2D slice dataset for one split.

    Each item is (image, mask, meta):
        image: float32 tensor (1, 256, 128), z-score normalised
        mask:  int64 tensor (256, 128) with labels 0..5
        meta:  dict with "case", "week", "slice" (used for per-slice error analysis)
    """

    def __init__(self, root, split, augment=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.img_dir = os.path.join(root, f"keras_slices_{split}")
        self.seg_dir = os.path.join(root, f"keras_slices_seg_{split}")
        self.split = split

        self.img_names = sorted(f for f in os.listdir(self.img_dir) if ".nii" in f)
        # image "case_XXX_..." pairs with mask "seg_XXX_..."
        self.seg_names = ["seg_" + n.split("_", 1)[1] for n in self.img_names]
        self.meta = [parse_slice_name(n) for n in self.img_names]

        missing = [s for s in self.seg_names if not os.path.exists(os.path.join(self.seg_dir, s))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} masks missing in {self.seg_dir}, e.g. {missing[0]}")

        self.transform = build_augmentation() if augment else None

    @property
    def case_ids(self):
        """Set of patient IDs in this split."""
        return {case for case, _, _ in self.meta}

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = load_nifti_slice(os.path.join(self.img_dir, self.img_names[idx]))
        seg = load_nifti_slice(os.path.join(self.seg_dir, self.seg_names[idx]))
        img = center_crop_or_pad(img)
        seg = center_crop_or_pad(seg)

        image = tv_tensors.Image(torch.from_numpy(img).unsqueeze(0))
        mask = tv_tensors.Mask(torch.from_numpy(np.rint(seg).astype(np.int64)))

        # Augment before normalising so out-of-image areas are filled with 0 (= background)
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        image = normalise(image.as_subclass(torch.Tensor))
        mask = mask.as_subclass(torch.Tensor)

        case, week, slc = self.meta[idx]
        return image, mask, {"case": case, "week": week, "slice": slc}


def check_patient_disjoint(*datasets):
    """
    Assert that no patient appears in more than one split.

    The official HipMRI split is used as-is; this check documents that it is
    free of cross-patient leakage (multiple weekly scans per patient exist).
    """
    for i, a in enumerate(datasets):
        for b in datasets[i + 1:]:
            shared = a.case_ids & b.case_ids
            if shared:
                raise RuntimeError(f"Patient leakage between {a.split} and {b.split}: {sorted(shared)}")


def get_dataloaders(root, batch_size=16, num_workers=4):
    """Build (train, validate, test) DataLoaders; only the training set is augmented."""
    train_set = HipMRI2DDataset(root, "train", augment=True)
    val_set = HipMRI2DDataset(root, "validate")
    test_set = HipMRI2DDataset(root, "test")
    check_patient_disjoint(train_set, val_set, test_set)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def _save_sample_figure(dataset, path, n=4):
    """Save image / mask / overlay panels for slices that contain prostate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    colours = ["black", "tab:gray", "wheat", "tab:blue", "tab:green", "tab:red"]
    cmap = ListedColormap(colours)

    # pick slices with the largest prostate area, from different scans
    picks, seen = [], set()
    areas = []
    for i in range(len(dataset)):
        seg = load_nifti_slice(os.path.join(dataset.seg_dir, dataset.seg_names[i]))
        areas.append((int((np.rint(seg) == PROSTATE).sum()), i))
    for area, i in sorted(areas, reverse=True):
        scan = dataset.meta[i][:2]
        if area > 0 and scan not in seen:
            picks.append(i)
            seen.add(scan)
        if len(picks) == n:
            break

    fig, axes = plt.subplots(len(picks), 3, figsize=(7, 3.2 * len(picks)))
    for row, i in enumerate(picks):
        image, mask, meta = dataset[i]
        img, msk = image[0].numpy(), mask.numpy()
        title = f"case {meta['case']:03d} wk {meta['week']} slice {meta['slice']}"
        axes[row, 0].imshow(img, cmap="gray")
        axes[row, 0].set_title(title, fontsize=8)
        axes[row, 1].imshow(msk, cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[row, 1].set_title("mask", fontsize=8)
        axes[row, 2].imshow(img, cmap="gray")
        axes[row, 2].imshow(np.ma.masked_where(msk < 3, msk), cmap=cmap, vmin=0,
                            vmax=NUM_CLASSES - 1, alpha=0.5, interpolation="nearest")
        axes[row, 2].set_title("overlay (labels 3-5)", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    handles = [Patch(color=c, label=f"{k}: {name}") for k, (c, name) in enumerate(zip(colours, CLASS_NAMES))]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check the HipMRI 2D data pipeline")
    parser.add_argument("--root", default=os.path.expanduser("~/data/keras_slices_data"))
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(args.root, num_workers=0)
    for name, loader in zip(SPLITS, (train_loader, val_loader, test_loader)):
        ds = loader.dataset
        print(f"{name:9s} slices={len(ds):6d}  patients={len(ds.case_ids):2d}  ids={sorted(ds.case_ids)}")
    print("Patient-disjoint check passed.")

    image, mask, meta = next(iter(train_loader))
    print(f"batch image {tuple(image.shape)} {image.dtype}, mask {tuple(mask.shape)} {mask.dtype}")
    print(f"image mean {image.mean():.3f} std {image.std():.3f}; mask labels {mask.unique().tolist()}")

    os.makedirs(args.out, exist_ok=True)
    fig_path = os.path.join(args.out, "samples.png")
    _save_sample_figure(val_loader.dataset, fig_path)
    print(f"Saved {fig_path}")
