"""
Training, validation, testing and saving of the segmentation models.

Usage:
    # baseline 2D U-Net on coronal slices
    python train.py --model unet --epochs 30 --out runs/unet_seed0

    # 3D Improved U-Net on cropped volumes (bf16 mixed precision on GPU)
    python train.py --model improved3d --epochs 20 --out runs/improved3d_seed0

    # Rangpur: data from the course directory, cache in $HOME, resume after interruption
    python train.py --model improved3d --root /home/groups/comp3710/HipMRI_Study_open \
        --cache-dir ~/cache/hipmri3d_crop --resume

Outputs (in --out):
    best.pth        weights with the highest validation prostate Dice (mean per scan)
    last.pth        full training state of the latest epoch (for --resume)
    history.json    per-epoch losses / metrics, resource usage, final test metrics
    curves.png      training loss and validation Dice curves
    train.log       copy of the console output

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

from dataset import get_dataloaders, get_dataloaders_3d, CLASS_NAMES, NUM_CLASSES, PROSTATE
from modules import UNet2D, ImprovedUNet3D, DiceCELoss, count_parameters

# model name -> (class, is_3d, default data root, default batch size, default lr)
MODELS = {
    "unet": (UNet2D, False, "~/data/keras_slices_data", 16, 1e-3),
    "improved3d": (ImprovedUNet3D, True, "~/data/HipMRI_3D", 2, 5e-4),
}


class Logger:
    """Print to the console and append to a log file (flushed every line)."""

    def __init__(self, path):
        self.path = path

    def __call__(self, msg):
        print(msg, flush=True)
        with open(self.path, "a") as f:
            f.write(msg + "\n")


def set_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device, enabled):
    """bf16 autocast on CUDA (no-op otherwise); bf16 needs no gradient scaler."""
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                          enabled=enabled and device.type == "cuda")


def train_one_epoch(model, loader, loss_fn, optimizer, device, amp):
    """Run one training epoch and return the mean loss."""
    model.train()
    total, n = 0.0, 0
    for image, mask, _ in loader:
        image = image.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp):
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
def evaluate(model, loader, loss_fn, device, amp=True):
    """
    Evaluate on a split (2D slices or 3D volumes).

    Returns a dict with:
        loss:                 mean loss
        dice / iou:           per-class scores over the whole split (voxels pooled)
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
        with autocast(device, amp):
            logits = model(image)
            total += loss_fn(logits, mask).item() * image.size(0)
        n += image.size(0)

        pred = logits.argmax(dim=1)
        for c in range(NUM_CLASSES):
            p, g = pred == c, mask == c
            tp[c] += (p & g).sum().item()
            fp[c] += (p & ~g).sum().item()
            fn[c] += (~p & g).sum().item()

        # per-sample prostate counts, accumulated into their scan
        spatial = tuple(range(1, pred.dim()))
        p, g = pred == PROSTATE, mask == PROSTATE
        s_tp = (p & g).sum(dim=spatial).cpu()
        s_fp = (p & ~g).sum(dim=spatial).cpu()
        s_fn = (~p & g).sum(dim=spatial).cpu()
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
    ax2.axhline(0.7, color="grey", ls=":", lw=1)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("Dice"); ax2.set_ylim(0, 1)
    ax2.set_title("Validation Dice"); ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train a HipMRI segmentation model")
    parser.add_argument("--model", choices=MODELS, default="unet")
    parser.add_argument("--root", default=None, help="data directory (default depends on --model)")
    parser.add_argument("--cache-dir", default=None, help="3D only: where cropped volumes are cached")
    parser.add_argument("--out", default=None, help="default: runs/<model>_seed<seed>")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true", help="disable bf16 mixed precision")
    parser.add_argument("--resume", action="store_true", help="continue from <out>/last.pth if present")
    args = parser.parse_args()

    model_cls, is_3d, default_root, default_bs, default_lr = MODELS[args.model]
    args.root = os.path.expanduser(args.root or default_root)
    args.cache_dir = os.path.expanduser(args.cache_dir) if args.cache_dir else None
    args.batch_size = args.batch_size or default_bs
    args.lr = args.lr or default_lr
    args.out = args.out or f"runs/{args.model}_seed{args.seed}"
    amp = not args.no_amp

    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    log = Logger(os.path.join(args.out, "train.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_3d:
        train_loader, val_loader, test_loader = get_dataloaders_3d(
            args.root, args.batch_size, args.num_workers, cache_dir=args.cache_dir)
    else:
        train_loader, val_loader, test_loader = get_dataloaders(args.root, args.batch_size, args.num_workers)

    model = model_cls(in_channels=1, num_classes=NUM_CLASSES).to(device)
    loss_fn = DiceCELoss(NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = defaultdict(list)
    best_score, best_epoch, start_epoch = -1.0, 0, 1
    last_path = os.path.join(args.out, "last.pth")
    if args.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        saved_epochs = state["history"]["args"]["epochs"]
        if saved_epochs != args.epochs:
            # the cosine schedule length is fixed at the start; changing it would leave lr = 0
            raise SystemExit(f"--epochs {args.epochs} differs from the resumed run ({saved_epochs})")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history.update(state["history"])
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["rng_cpu"].cpu())  # map_location moved it to the GPU
        log(f"resumed from epoch {state['epoch']} (best {best_score:.3f} at epoch {best_epoch})")
    history["args"] = vars(args)

    unit = "volumes" if is_3d else "slices"
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    log(f"model {args.model}: {count_parameters(model):,} parameters | {gpu} | bf16 {amp}")
    log(f"train {len(train_loader.dataset)} / val {len(val_loader.dataset)} / "
        f"test {len(test_loader.dataset)} {unit} | batch {args.batch_size} | lr {args.lr}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, amp)
        val = evaluate(model, val_loader, loss_fn, device, amp)
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
        torch.save({
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": dict(history),
            "best_score": best_score, "best_epoch": best_epoch, "rng_cpu": torch.get_rng_state(),
        }, last_path)
        log(f"[{epoch:3d}/{args.epochs}] {elapsed:5.1f}s train {train_loss:.4f} | val {format_metrics(val)}{flag}")

    plot_curves(history, os.path.join(args.out, "curves.png"))

    # Final test evaluation with the best validation checkpoint (test split used only here)
    model.load_state_dict(torch.load(os.path.join(args.out, "best.pth"), map_location=device))
    t0 = time.time()
    test = evaluate(model, test_loader, loss_fn, device, amp)
    infer_time = (time.time() - t0) / len(test_loader.dataset)
    log(f"best epoch {best_epoch} (val prostate/scan {best_score:.3f})")
    log(f"TEST {format_metrics(test)}")
    log("TEST IoU " + " ".join(f"{n[:4]} {v:.3f}" for n, v in zip(CLASS_NAMES[1:], test["iou"][1:])))

    history["best_epoch"] = best_epoch
    history["test"] = test
    history["parameters"] = count_parameters(model)
    history["test_seconds_per_" + unit[:-1]] = infer_time
    if device.type == "cuda":
        history["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        log(f"peak VRAM {history['peak_vram_gb']:.2f} GiB (this session), "
            f"mean epoch {np.mean(history['epoch_time']):.1f}s, test {infer_time * 1000:.1f} ms/{unit[:-1]}")
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
