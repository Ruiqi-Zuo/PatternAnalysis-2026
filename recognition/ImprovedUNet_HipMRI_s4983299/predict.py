"""
Example usage of the trained models: load saved weights, segment test cases, print
per-class Dice and save visualisations (and optionally the predicted masks as Nifti).

Usage:
    # 3D Improved U-Net on the first 3 test volumes (Dice at full resolution + figures)
    python predict.py --model improved3d --n 3

    # 2D U-Net baseline on 4 test slices that contain prostate
    python predict.py --model unet --n 4

    # deploy on any HipMRI-style MR volume (no label needed); writes <out>/<name>_pred.nii.gz
    # with the same geometry as the input, ready to open in a viewer such as 3D Slicer
    python predict.py --model improved3d --input path/to/B040_Week0_LFOV.nii.gz --save-nifti

Outputs (in --out, default predictions/): one PNG per case and optional *_pred.nii.gz.

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import os
import time

import nibabel as nib
import numpy as np
import torch

from dataset import (CLASS_NAMES, CROP_3D, NUM_CLASSES, PROSTATE, TARGET_SHAPE, VOLUME_SHAPE,
                     HipMRI2DDataset, HipMRI3DDataset, center_crop_or_pad, load_nifti_slice, normalise)
from modules import UNet2D, ImprovedUNet3D

MODELS = {"unet": UNet2D, "improved3d": ImprovedUNet3D}
DEFAULT_ROOT = {"unet": "~/data/keras_slices_data", "improved3d": "~/data/HipMRI_3D"}
ORGAN_COLOURS = {3: (0.2, 0.4, 1.0), 4: (0.1, 0.8, 0.2), 5: (1.0, 0.1, 0.1)}  # bladder, rectum, prostate


def load_model(name, ckpt, device):
    """Build a model and load trained weights (as saved by train.py)."""
    model = MODELS[name](in_channels=1, num_classes=NUM_CLASSES).to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


@torch.no_grad()
def _forward(model, x, device):
    """Argmax prediction and wall-clock time of one forward pass."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred = model(x.to(device)).argmax(1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred.cpu().numpy().astype(np.uint8), time.time() - t0


def predict_volume(model, mr_path, device):
    """
    Segment a full HipMRI MR volume.

    The volume is cropped exactly as in training, normalised, segmented, and the
    prediction is pasted back into an array with the input's shape (background outside
    the crop). Returns (prediction, cropped normalised image, nibabel image, seconds).
    """
    nii = nib.load(mr_path)
    image = normalise(nii.get_fdata(dtype=np.float32)[CROP_3D])
    pred_crop, seconds = _forward(model, torch.from_numpy(image)[None, None], device)
    pred = np.zeros(nii.shape, dtype=np.uint8)
    pred[CROP_3D] = pred_crop[0]
    return pred, image, nii, seconds


def predict_slice(model, path, device):
    """Segment one 2D slice. Returns (prediction, normalised image, seconds)."""
    image = normalise(center_crop_or_pad(load_nifti_slice(path)))
    pred, seconds = _forward(model, torch.from_numpy(image)[None, None], device)
    return pred[0], image, seconds


def per_class_dice(pred, gt):
    """Dice for every foreground class (NaN if a class is absent from both)."""
    out = []
    for c in range(1, NUM_CLASSES):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        out.append(2.0 * (p & g).sum() / denom if denom else float("nan"))
    return out


def _overlay(ax, img, lab, title, origin="lower"):
    """Grey image with bladder / rectum / prostate drawn in colour."""
    ax.imshow(img, cmap="gray", origin=origin, vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
    if lab is not None:
        rgba = np.zeros(lab.shape + (4,))
        for c, colour in ORGAN_COLOURS.items():
            rgba[lab == c] = (*colour, 0.5)
        ax.imshow(rgba, origin=origin, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_volume_figure(image, gt, pred, title, path):
    """Rows: ground truth (if available) and prediction; columns: axial, coronal, sagittal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = gt if gt is not None and (gt == PROSTATE).any() else pred
    centre = np.argwhere(ref == PROSTATE).mean(0).astype(int) if (ref == PROSTATE).any() \
        else np.array(ref.shape) // 2
    l, p, s = centre
    planes = [  # (name, plane extractor, imshow origin); arrays are (L, P, S)
        (f"axial S={s} (anterior up)", lambda v: v[:, :, s].T, "upper"),
        (f"coronal P={p} (superior up)", lambda v: v[:, p, :].T, "lower"),
        (f"sagittal L={l} (superior up)", lambda v: v[l, :, :].T, "lower"),
    ]
    rows = ([("ground truth", gt)] if gt is not None else []) + [("prediction", pred)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.6 * len(rows)), squeeze=False)
    for r, (rname, lab) in enumerate(rows):
        for c, (pname, extract, origin) in enumerate(planes):
            _overlay(axes[r, c], extract(image), extract(lab), f"{rname}: {pname}", origin)
    fig.suptitle(f"{title}   (blue bladder, green rectum, red prostate)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_slice_figure(samples, path):
    """One row per 2D slice: image, ground truth, prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(samples), 3, figsize=(7, 3.3 * len(samples)), squeeze=False)
    for r, (name, image, gt, pred) in enumerate(samples):
        _overlay(axes[r, 0], image, None, name, "upper")
        _overlay(axes[r, 1], image, gt, "ground truth", "upper")
        _overlay(axes[r, 2], image, pred, "prediction", "upper")
    fig.suptitle("2D U-Net (coronal slices; blue bladder, green rectum, red prostate)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_row(name, dice, seconds):
    cells = " ".join(f"{d:6.3f}" for d in dice) if dice is not None else "  (no label)"
    print(f"{name:32s} {cells}  {seconds * 1000:7.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="Segment HipMRI scans with a trained model")
    parser.add_argument("--model", choices=MODELS, default="improved3d")
    parser.add_argument("--ckpt", default=None, help="default: runs/<model>_seed0/best.pth")
    parser.add_argument("--root", default=None, help="dataset root (default depends on --model)")
    parser.add_argument("--split", default="test", choices=("train", "validate", "test"))
    parser.add_argument("--n", type=int, default=3, help="number of volumes / slices to segment")
    parser.add_argument("--input", nargs="+", help="segment these Nifti files instead of the dataset")
    parser.add_argument("--save-nifti", action="store_true", help="also write predictions as .nii.gz")
    parser.add_argument("--out", default="predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or f"runs/{args.model}_seed0/best.pth"
    root = os.path.expanduser(args.root or DEFAULT_ROOT[args.model])
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.model, ckpt, device)
    print(f"{args.model} loaded from {ckpt} on {device}")
    # warm-up pass so the reported times exclude one-off CUDA / cuDNN initialisation
    _forward(model, torch.zeros(1, 1, *(VOLUME_SHAPE if args.model == "improved3d" else TARGET_SHAPE)), device)
    if args.model == "improved3d":
        print("Dice inside the 128x144x128 pelvic crop the model segments (bladder, rectum and "
              "prostate lie inside it; body / bone outside it are not segmented)")
    print(f"{'case':32s} " + " ".join(f"{n[:6]:>6s}" for n in CLASS_NAMES[1:]) + "  inference")

    # ---- 3D: whole volumes ----
    if args.model == "improved3d":
        if args.input:
            jobs = [(os.path.basename(p).split(".")[0], p, None) for p in args.input]
        else:
            items = HipMRI3DDataset(root, args.split).items[:args.n]
            jobs = [(f"{args.split}_case{c:03d}_week{w}", mr, seg) for c, w, mr, seg in items]

        for name, mr_path, seg_path in jobs:
            pred, image, nii, seconds = predict_volume(model, mr_path, device)
            gt = np.rint(nib.load(seg_path).get_fdata()).astype(np.uint8) if seg_path else None
            print_row(name, per_class_dice(pred[CROP_3D], gt[CROP_3D]) if gt is not None else None, seconds)
            save_volume_figure(image, gt[CROP_3D] if gt is not None else None, pred[CROP_3D],
                               name, os.path.join(args.out, f"{name}.png"))
            if args.save_nifti:
                out_nii = nib.Nifti1Image(pred, nii.affine)
                out_nii.set_data_dtype(np.uint8)
                nib.save(out_nii, os.path.join(args.out, f"{name}_pred.nii.gz"))

    # ---- 2D: individual coronal slices ----
    else:
        if args.input:
            jobs = [(os.path.basename(p).split(".")[0], p, None) for p in args.input]
        else:
            ds = HipMRI2DDataset(root, args.split)
            jobs = []
            for i in range(len(ds)):  # demonstrate on slices that actually contain prostate
                seg_path = os.path.join(ds.seg_dir, ds.seg_names[i])
                if (np.rint(load_nifti_slice(seg_path)) == PROSTATE).any():
                    jobs.append((ds.img_names[i].split(".")[0], os.path.join(ds.img_dir, ds.img_names[i]), seg_path))
                if len(jobs) == args.n:
                    break

        samples = []
        for name, img_path, seg_path in jobs:
            pred, image, seconds = predict_slice(model, img_path, device)
            gt = np.rint(center_crop_or_pad(load_nifti_slice(seg_path))).astype(np.uint8) if seg_path else None
            print_row(name, per_class_dice(pred, gt) if gt is not None else None, seconds)
            samples.append((name, image, gt, pred))
            if args.save_nifti:
                nib.save(nib.Nifti1Image(pred, np.eye(4)), os.path.join(args.out, f"{name}_pred.nii.gz"))
        save_slice_figure(samples, os.path.join(args.out, f"{args.model}_{args.split}_slices.png"))

    print(f"Saved outputs to {args.out}/")


if __name__ == "__main__":
    main()
