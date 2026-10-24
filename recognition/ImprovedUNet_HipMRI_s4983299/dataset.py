"""
Data loading and preprocessing for HipMRI prostate segmentation.

2D (baseline): coronal slices from keras_slices_data
    - load 2D Nifti image / segmentation slices
    - unify slice shape and normalise intensities per slice
    - verify the official train / validate / test split is patient-disjoint
    - data augmentation applied identically to image and mask

3D (Improved U-Net): full volumes from semantic_MRs / semantic_labels_only
    - fixed, label-independent crop to 128 x 144 x 128 at original resolution
    - per-volume normalisation, cached as .npy after the first load
    - same patient split as the 2D data, so both models see identical patients
    - 3D augmentation (left-right flip, small rotation / scale, intensity jitter)

Expected directory layouts (as provided on Rangpur):
    keras_slices_data/
        keras_slices_train/      case_004_week_0_slice_0.nii.gz ...
        keras_slices_seg_train/  seg_004_week_0_slice_0.nii.gz  ...
        keras_slices_validate/   keras_slices_seg_validate/
        keras_slices_test/       keras_slices_seg_test/
    HipMRI_3D/
        semantic_MRs/            B006_Week0_LFOV.nii.gz ...
        semantic_labels_only/    B006_Week0_SEMANTIC.nii.gz ...

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import math
import os
import re

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2

# Label indices in the segmentation masks.
# Verified against the 3D volumes (LPS orientation): each 2D slice k is the coronal plane
# flip_LR(volume[:, 98 + k, :]), so the slice index runs anterior -> posterior, the vertical
# image axis is left-right and the horizontal axis runs inferior -> superior.
# Label 3 lies superior/anterior to the prostate (bladder); label 4 lies posterior to it and
# spans above and below it (rectum).
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
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
    anatomically plausible mirror; the inferior-superior (horizontal) axis is never flipped.
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


# --------------------------------------------------------------------------- #
# 3D volumes
# --------------------------------------------------------------------------- #
# Patient split mirrors the official keras_slices_data split (verified: the numeric part
# of "B006_Week0" is the 2D case ID and every 2D slice is a plane of the matching volume).
VAL_CASES = frozenset({36, 37, 38, 39})
TEST_CASES = frozenset({40, 41, 42})

# Fixed crop in voxel indices of the (L, P, S)-oriented 256 x 256 x 128 volumes.
# Chosen from organ bounding boxes over all 211 volumes: bladder, rectum and prostate lie
# inside in every volume (only K019_Week1, the single 144-slice train volume, loses the top
# 3 bladder slices). The crop does not depend on the labels, so it is valid at inference.
CROP_3D = (slice(64, 192), slice(48, 192), slice(0, 128))
VOLUME_SHAPE = (128, 144, 128)

_VOL_RE = re.compile(r"[A-Z](\d+)_Week(\d+)_")


def parse_volume_name(fname):
    """Return (case, week) parsed from a HipMRI 3D filename such as B006_Week0_LFOV.nii.gz."""
    m = _VOL_RE.match(fname)
    if m is None:
        raise ValueError(f"Unexpected filename: {fname}")
    return int(m.group(1)), int(m.group(2))


def split_of_case(case):
    """Split name for a patient ID, identical to the 2D official split."""
    if case in TEST_CASES:
        return "test"
    if case in VAL_CASES:
        return "validate"
    return "train"


def load_volume(mr_path, seg_path):
    """Load one volume pair, crop it and z-score normalise the image over the crop."""
    img = nib.load(mr_path).get_fdata(dtype=np.float32)[CROP_3D]
    seg = np.rint(nib.load(seg_path).get_fdata(dtype=np.float32)[CROP_3D]).astype(np.uint8)
    return normalise(img).astype(np.float32), seg


def augment_volume(image, mask, max_rot_deg=10.0, scale_range=(0.9, 1.1)):
    """
    Random 3D augmentation for one volume (image (1, D, H, W), mask (D, H, W)).

    - left-right flip (axis D is patient left-right), p = 0.5
    - small rotation about each axis and isotropic scaling, applied with one
      affine grid: trilinear for the image, nearest for the mask
    - intensity scale / shift jitter on the (normalised) image
    """
    if torch.rand(1).item() < 0.5:
        image, mask = image.flip(1), mask.flip(0)

    angles = (torch.rand(3) * 2 - 1) * math.radians(max_rot_deg)
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    scale = torch.empty(1).uniform_(*scale_range).item()
    theta = torch.zeros(1, 3, 4)
    theta[0, :, :3] = (rz @ ry @ rx) / scale

    grid = F.affine_grid(theta, (1, 1, *mask.shape), align_corners=False)
    # pad image with its minimum (= background intensity after normalisation)
    fill = image.min()
    image = F.grid_sample((image - fill).unsqueeze(0), grid, mode="bilinear",
                          padding_mode="zeros", align_corners=False)[0] + fill
    mask = F.grid_sample(mask[None, None].float(), grid, mode="nearest",
                         padding_mode="zeros", align_corners=False)[0, 0].long()

    image = image * torch.empty(1).uniform_(0.9, 1.1) + torch.empty(1).uniform_(-0.1, 0.1)
    return image, mask


class HipMRI3DDataset(Dataset):
    """
    HipMRI 3D volume dataset for one split.

    Each item is (image, mask, meta):
        image: float32 tensor (1, 128, 144, 128), axes (L-R, A-P, I-S)
        mask:  int64 tensor (128, 144, 128) with labels 0..5
        meta:  dict with "case", "week"

    Cropped, normalised volumes are cached as .npy in `cache_dir` on first use,
    because decompressing the Nifti files dominates loading time.
    """

    def __init__(self, root, split, augment=False, cache_dir=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.split = split
        self.augment = augment
        mr_dir = os.path.join(root, "semantic_MRs")
        seg_dir = os.path.join(root, "semantic_labels_only")
        self.cache_dir = cache_dir or os.path.join(root, "cache_crop")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.items = []  # (case, week, mr_path, seg_path)
        for n in sorted(os.listdir(mr_dir)):
            case, week = parse_volume_name(n)
            if split_of_case(case) != split:
                continue
            seg_name = n.replace("_LFOV", "_SEMANTIC")
            seg_path = os.path.join(seg_dir, seg_name)
            if not os.path.exists(seg_path):
                raise FileNotFoundError(seg_path)
            self.items.append((case, week, os.path.join(mr_dir, n), seg_path))

    @property
    def case_ids(self):
        """Set of patient IDs in this split."""
        return {case for case, *_ in self.items}

    def _cached(self, idx):
        case, week, mr_path, seg_path = self.items[idx]
        stem = os.path.join(self.cache_dir, f"{case:03d}_week{week}")
        if not os.path.exists(stem + "_seg.npy"):
            img, seg = load_volume(mr_path, seg_path)
            np.save(stem + "_img.npy", img)
            np.save(stem + "_seg.npy", seg)  # written last: marks the pair as complete
        return np.load(stem + "_img.npy"), np.load(stem + "_seg.npy")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, seg = self._cached(idx)
        image = torch.from_numpy(img).unsqueeze(0)
        mask = torch.from_numpy(seg.astype(np.int64))
        if self.augment:
            image, mask = augment_volume(image, mask)
        case, week, _, _ = self.items[idx]
        return image, mask, {"case": case, "week": week}


def get_dataloaders_3d(root, batch_size=2, num_workers=4, cache_dir=None):
    """Build (train, validate, test) DataLoaders of 3D volumes; only training is augmented."""
    train_set = HipMRI3DDataset(root, "train", augment=True, cache_dir=cache_dir)
    val_set = HipMRI3DDataset(root, "validate", cache_dir=cache_dir)
    test_set = HipMRI3DDataset(root, "test", cache_dir=cache_dir)
    check_patient_disjoint(train_set, val_set, test_set)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, drop_last=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader


def _save_volume_figure(dataset, path, idx=0, augmented=None):
    """Axial / coronal / sagittal planes through the prostate centroid of one volume."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["black", "tab:gray", "wheat", "tab:blue", "tab:green", "tab:red"])
    rows = [("original", dataset[idx])]
    if augmented is not None:
        rows.append(("augmented", augmented))
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.6 * len(rows)), squeeze=False)
    for r, (label, (image, mask, meta)) in enumerate(rows):
        img, msk = image[0].numpy(), mask.numpy()
        c = np.argwhere(msk == PROSTATE).mean(0).astype(int) if (msk == PROSTATE).any() \
            else np.array(msk.shape) // 2
        planes = [
            ("axial (S=%d)" % c[2], img[:, :, c[2]].T, msk[:, :, c[2]].T),
            ("coronal (P=%d)" % c[1], img[:, c[1], :].T, msk[:, c[1], :].T),
            ("sagittal (L=%d)" % c[0], img[c[0], :, :].T, msk[c[0], :, :].T),
        ]
        for ax, (title, im, m) in zip(axes[r], planes):
            ax.imshow(im, cmap="gray", origin="lower")
            ax.imshow(np.ma.masked_where(m < 3, m), cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1,
                      alpha=0.5, origin="lower", interpolation="nearest")
            ax.set_title(f"{label} case {meta['case']:03d} wk {meta['week']}: {title}", fontsize=8)
            ax.axis("off")
    fig.suptitle("blue bladder, green rectum, red prostate | axial: posterior up; "
                 "coronal/sagittal: superior up; sagittal: posterior right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


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
    parser = argparse.ArgumentParser(description="Sanity-check the HipMRI data pipelines")
    parser.add_argument("--mode", choices=("2d", "3d"), default="2d")
    parser.add_argument("--root", default=None,
                        help="default: ~/data/keras_slices_data (2d) or ~/data/HipMRI_3D (3d)")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.mode == "3d":
        import time
        root = args.root or os.path.expanduser("~/data/HipMRI_3D")
        t0 = time.time()
        train_loader, val_loader, test_loader = get_dataloaders_3d(root, num_workers=0)
        for name, loader in zip(SPLITS, (train_loader, val_loader, test_loader)):
            ds = loader.dataset
            print(f"{name:9s} volumes={len(ds):4d}  patients={len(ds.case_ids):2d}  ids={sorted(ds.case_ids)}")
            for i in range(len(ds)):  # builds the .npy cache on first run
                ds._cached(i)
        print(f"Patient-disjoint check passed. Cache ready in {time.time() - t0:.0f}s")

        t0 = time.time()
        image, mask, meta = next(iter(train_loader))
        print(f"batch image {tuple(image.shape)} {image.dtype}, mask {tuple(mask.shape)} {mask.dtype} "
              f"({time.time() - t0:.2f}s incl. augmentation)")
        print(f"mask labels {mask.unique().tolist()}")
        val_set = val_loader.dataset
        lost = [m for m in (val_set[i][1] for i in range(len(val_set))) if (m == PROSTATE).sum() == 0]
        print(f"validation volumes without prostate after cropping: {len(lost)}")

        torch.manual_seed(1)
        aug = train_loader.dataset
        sample = aug[0]
        aug.augment = False
        fig_path = os.path.join(args.out, "samples_3d.png")
        _save_volume_figure(aug, fig_path, idx=0, augmented=sample)
        print(f"Saved {fig_path}")
        raise SystemExit

    root = args.root or os.path.expanduser("~/data/keras_slices_data")
    train_loader, val_loader, test_loader = get_dataloaders(root, num_workers=0)
    for name, loader in zip(SPLITS, (train_loader, val_loader, test_loader)):
        ds = loader.dataset
        print(f"{name:9s} slices={len(ds):6d}  patients={len(ds.case_ids):2d}  ids={sorted(ds.case_ids)}")
    print("Patient-disjoint check passed.")

    image, mask, meta = next(iter(train_loader))
    print(f"batch image {tuple(image.shape)} {image.dtype}, mask {tuple(mask.shape)} {mask.dtype}")
    print(f"image mean {image.mean():.3f} std {image.std():.3f}; mask labels {mask.unique().tolist()}")

    fig_path = os.path.join(args.out, "samples.png")
    _save_sample_figure(val_loader.dataset, fig_path)
    print(f"Saved {fig_path}")
