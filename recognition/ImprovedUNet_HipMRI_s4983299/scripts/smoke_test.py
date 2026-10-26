"""
Smoke test of the baseline 2D U-Net, used as feasibility evidence.

Trains UNet2D for a few epochs (fp32, no checkpoint) and reports on the validation split:
    - per-class Dice after every epoch
    - prostate Dice in the apex / central / base regions (last epoch)
    - prostate Dice of the anterior / central / posterior coronal slices (last epoch)
    - predicted prostate pixels on prostate-free slices (last epoch)
    - peak GPU memory while training, time per epoch and inference time per slice

Geometry (see dataset.py): a 2D slice is a coronal plane, the slice index runs anterior ->
posterior and the horizontal image axis runs inferior -> superior. Apex and base are
therefore columns of the slices, not slice indices: they are the EDGE most inferior /
superior prostate-bearing axial planes of each scan plus everything beyond them, exactly
as in evaluate.py. The coronal grouping is kept because the first version of this script
reported it under the wrong names "apex" (anterior) and "base" (posterior).

Pooled Dice = 2 * sum|P & G| / (sum|P| + sum|G|) over all slices or scans of a group.
Per-scan Dice is computed per (case, week) and then averaged.

Usage:
    python scripts/smoke_test.py --root ~/data/keras_slices_data --epochs 5

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import collections
import os
import sys
import time

import numpy as np
import torch

# make the project modules importable when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset import CLASS_NAMES, NUM_CLASSES, PROSTATE, get_dataloaders  # noqa: E402
from modules import DiceCELoss, UNet2D, count_parameters  # noqa: E402

EDGE = 3  # slices counted as an edge region, as N_EDGE in evaluate.py


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training set; returns the mean loss."""
    model.train()
    total = 0.0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """
    Predict the whole loader.

    Returns:
        dice:    pooled Dice per class, tensor (NUM_CLASSES,)
        slices:  per slice (scan, slice index, |P&G|, |P|+|G|, |G|) for the prostate
        columns: per scan a (3, W) tensor of prostate |P&G|, |P|, |G| summed over each
                 column (= axial plane) of all its slices
    """
    model.eval()
    inter = torch.zeros(NUM_CLASSES)
    denom = torch.zeros(NUM_CLASSES)
    slices, columns = [], {}
    for x, y, meta in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for c in range(NUM_CLASSES):
            p, g = pred == c, y == c
            inter[c] += (p & g).sum()
            denom[c] += p.sum() + g.sum()

        p, g = pred == PROSTATE, y == PROSTATE
        for b in range(len(x)):
            scan = (int(meta["case"][b]), int(meta["week"][b]))
            pb, gb = p[b], g[b]
            slices.append((scan, int(meta["slice"][b]), int((pb & gb).sum()),
                           int(pb.sum() + gb.sum()), int(gb.sum())))
            col = torch.stack([(pb & gb).sum(0), pb.sum(0), gb.sum(0)])  # sum over left-right
            columns[scan] = columns[scan] + col if scan in columns else col
    return 2 * inter / denom.clamp(min=1), slices, columns


def axial_region_dice(columns):
    """
    Prostate Dice in the apex / central / base regions of every scan.

    Returns {region: (pooled Dice over all scans, list of per-scan Dice)}.
    """
    pooled = collections.defaultdict(lambda: [0, 0])
    per_scan = collections.defaultdict(list)
    for inter, pred, gt in columns.values():
        z = torch.nonzero(gt).flatten()
        if len(z) == 0:
            continue
        lo, hi = int(z.min()), int(z.max())
        regions = {"apex": slice(0, lo + EDGE),
                   "central": slice(lo + EDGE, hi - EDGE + 1),
                   "base": slice(hi - EDGE + 1, None)}
        for name, cols in regions.items():
            i, d = int(inter[cols].sum()), int(pred[cols].sum() + gt[cols].sum())
            pooled[name][0] += i
            pooled[name][1] += d
            if d:
                per_scan[name].append(2 * i / d)
    return {name: (2 * pooled[name][0] / max(pooled[name][1], 1), per_scan[name])
            for name in ("apex", "central", "base")}


def coronal_edge_dice(slices):
    """
    Prostate Dice of the EDGE most anterior / posterior prostate-bearing coronal slices of
    each scan, pooled over scans. Prostate-free slices form their own group, where every
    predicted prostate pixel is a false positive.

    Returns {group: [|P&G|, |P|+|G|, number of slices]}.
    """
    extent = collections.defaultdict(list)
    for scan, k, _, _, gt in slices:
        if gt:
            extent[scan].append(k)

    groups = collections.defaultdict(lambda: [0, 0, 0])
    for scan, k, i, d, gt in slices:
        if gt == 0:
            name = "no prostate"
            groups[name][0] += d
        else:
            lo, hi = min(extent[scan]), max(extent[scan])
            name = "anterior" if k < lo + EDGE else "posterior" if k > hi - EDGE else "central"
            groups[name][0] += i
            groups[name][1] += d
        groups[name][2] += 1
    return groups


@torch.no_grad()
def inference_latency(model, device, shape=(1, 1, 256, 128), warmup=20, runs=200):
    """Mean fp32 forward time (ms) for one slice and the peak memory (MiB) during it."""
    model.eval()
    x = torch.randn(*shape, device=device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        model(x)
    torch.cuda.synchronize()
    return (time.time() - start) / runs * 1000, torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    parser = argparse.ArgumentParser(description="Smoke test of the baseline 2D U-Net")
    parser.add_argument("--root", default=os.path.expanduser("~/data/keras_slices_data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required: the memory and timing figures are GPU measurements.")
    device = "cuda"
    torch.manual_seed(args.seed)

    train_loader, val_loader, _ = get_dataloaders(args.root, batch_size=args.batch_size,
                                                  num_workers=args.num_workers)
    model = UNet2D().to(device)
    loss_fn = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"UNet2D: {count_parameters(model):,} parameters | {torch.cuda.get_device_name()} | "
          f"torch {torch.__version__} | train {len(train_loader.dataset)} / "
          f"val {len(val_loader.dataset)} slices")

    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        torch.cuda.synchronize()
        seconds = time.time() - start
        dice, slices, columns = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: loss {loss:.4f}, train {seconds:.0f} s | val Dice "
              + " ".join(f"{CLASS_NAMES[c]}={dice[c]:.3f}" for c in range(1, NUM_CLASSES)))
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    print(f"peak training memory (max_memory_allocated, batch {args.batch_size}): "
          f"{peak:.0f} MiB = {peak / 1024:.2f} GiB")

    print(f"\nvalidation prostate Dice by region, epoch {args.epochs} "
          f"(apex/base = {EDGE} most inferior/superior axial planes and beyond):")
    for name, (pooled, per_scan) in axial_region_dice(columns).items():
        print(f"  {name:8s} pooled {pooled:.3f} | per-scan mean {np.mean(per_scan):.3f} "
              f"± {np.std(per_scan):.3f} (n={len(per_scan)})")

    print(f"validation prostate Dice of the {EDGE} outer coronal slices, epoch {args.epochs}:")
    groups = coronal_edge_dice(slices)
    for name in ("anterior", "central", "posterior"):
        i, d, n = groups[name]
        print(f"  {name:9s} slices={n:4d} pooled {2 * i / max(d, 1):.3f}")
    fp, _, n = groups["no prostate"]
    print(f"  {n} prostate-free slices: {fp} predicted prostate pixels")

    ms, mib = inference_latency(model, device)
    print(f"\ninference (fp32, batch 1): {ms:.2f} ms per slice, peak memory {mib:.0f} MiB")


if __name__ == "__main__":
    main()
