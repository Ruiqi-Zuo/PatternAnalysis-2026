"""
Training, validation, testing and saving of the segmentation models.

Usage:
    python train.py --model unet --epochs 30          # baseline 2D U-Net
    python train.py --root ~/data/keras_slices_data --out runs/unet_seed0

Outputs (in --out):
    best.pth        weights with the highest validation prostate Dice (mean per scan)
    history.json    per-epoch losses / metrics, resource usage, final test metrics
    curves.png      training loss and validation Dice curves

Model is imported from modules.py, data loaders from dataset.py.
The test split is only evaluated once, with the best validation checkpoint.

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES, PROSTATE
from modules import UNet2D, DiceCELoss, count_parameters

MODELS = {"unet": UNet2D}


def set_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one training epoch and return the mean loss."""
    model.train()
    total, n = 0.0, 0
    for image, mask, _ in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(image), mask)
        loss.backward()
        optimizer.step()

        total += loss.item() * image.size(0)
        n += image.size(0)
    return total / n


def _dice(tp, fp, fn):
    """Dice from counts; NaN when the class is absent from both prediction and truth."""
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """
    Evaluate on a split.

    Returns a dict with:
        loss:                 mean loss
        dice / iou:           per-class scores over the whole split (pixels pooled)
        prostate_dice_scan:   prostate Dice averaged over scans (each scan = one 3D volume),
                              matching the per-volume acceptance criteria
    """
    model.eval()
    total, n = 0.0, 0
    tp = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    scan_counts = defaultdict(lambda: [0.0, 0.0, 0.0])  # (case, week) -> prostate tp, fp, fn

    for image, mask, meta in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(image)
        total += loss_fn(logits, mask).item() * image.size(0)
        n += image.size(0)

        pred = logits.argmax(dim=1)
        for c in range(NUM_CLASSES):
            p, g = pred == c, mask == c
            tp[c] += (p & g).sum().item()
            fp[c] += (p & ~g).sum().item()
            fn[c] += (~p & g).sum().item()

        # per-slice prostate counts, accumulated into their scan
        p, g = pred == PROSTATE, mask == PROSTATE
        s_tp = (p & g).sum(dim=(1, 2)).cpu()
        s_fp = (p & ~g).sum(dim=(1, 2)).cpu()
        s_fn = (~p & g).sum(dim=(1, 2)).cpu()
        for i in range(image.size(0)):
            key = (int(meta["case"][i]), int(meta["week"][i]))
            scan_counts[key][0] += s_tp[i].item()
            scan_counts[key][1] += s_fp[i].item()
            scan_counts[key][2] += s_fn[i].item()

    dice = [_dice(tp[c].item(), fp[c].item(), fn[c].item()) for c in range(NUM_CLASSES)]
    iou = [(tp[c] / (tp[c] + fp[c] + fn[c])).item() if (tp[c] + fp[c] + fn[c]) > 0 else float("nan")
           for c in range(NUM_CLASSES)]
    scan_dice = [_dice(*v) for v in scan_counts.values()]
    return {
        "loss": total / n,
        "dice": dice,
        "iou": iou,
        "prostate_dice_scan": float(np.nanmean(scan_dice)),
    }


def format_metrics(m):
    """One-line summary of per-class Dice."""
    per_class = " ".join(f"{name[:4]} {d:.3f}" for name, d in zip(CLASS_NAMES[1:], m["dice"][1:]))
    return f"loss {m['loss']:.4f} | {per_class} | prostate/scan {m['prostate_dice_scan']:.3f}"


def plot_curves(history, path):
    """Plot training / validation loss and validation Dice (prostate and all organs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("Dice + CE loss"); ax1.set_title("Loss"); ax1.legend()

    val_dice = np.array(history["val_dice"])  # (epochs, classes)
    for c in range(1, NUM_CLASSES):
        ax2.plot(epochs, val_dice[:, c], label=CLASS_NAMES[c], lw=2.5 if c == PROSTATE else 1)
    ax2.plot(epochs, history["val_prostate_dice_scan"], "k--", label="prostate (mean per scan)")
    ax2.axhline(0.75, color="grey", ls=":", lw=1)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1)
    ax2.set_title("Validation Dice"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a HipMRI 2D segmentation model")
    parser.add_argument("--model", choices=MODELS, default="unet")
    parser.add_argument("--root", default=os.path.expanduser("~/data/keras_slices_data"))
    parser.add_argument("--out", default="runs/unet_seed0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(args.root, args.batch_size, args.num_workers)
    model = MODELS[args.model](in_channels=1, num_classes=NUM_CLASSES).to(device)
    loss_fn = DiceCELoss(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"model {args.model}: {count_parameters(model):,} parameters, device {device}")
    print(f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} / test {len(test_loader.dataset)} slices")

    history = defaultdict(list)
    history["args"] = vars(args)
    best_score, best_epoch = -1.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_dice"].append(val["dice"])
        history["val_prostate_dice_scan"].append(val["prostate_dice_scan"])
        history["epoch_time"].append(elapsed)

        flag = ""
        if val["prostate_dice_scan"] > best_score:
            best_score, best_epoch = val["prostate_dice_scan"], epoch
            torch.save(model.state_dict(), os.path.join(args.out, "best.pth"))
            flag = " *"
        print(f"[{epoch:3d}/{args.epochs}] {elapsed:5.1f}s train {train_loss:.4f} | val {format_metrics(val)}{flag}",
              flush=True)

    plot_curves(history, os.path.join(args.out, "curves.png"))

    # Final test evaluation with the best validation checkpoint (test split used only here)
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pth"), map_location=device))
    test = evaluate(model, test_loader, loss_fn, device)
    print(f"best epoch {best_epoch} (val prostate/scan {best_score:.3f})")
    print(f"TEST {format_metrics(test)}")
    print("TEST IoU " + " ".join(f"{n[:4]} {v:.3f}" for n, v in zip(CLASS_NAMES[1:], test["iou"][1:])))

    history["best_epoch"] = best_epoch
    history["test"] = test
    history["parameters"] = count_parameters(model)
    if device.type == "cuda":
        history["peak_train_vram_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"peak VRAM {history['peak_train_vram_gb']:.2f} GB, "
              f"mean epoch {np.mean(history['epoch_time']):.1f}s")
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
