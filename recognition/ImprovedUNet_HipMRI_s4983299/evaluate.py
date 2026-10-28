"""
Volumetric comparison of the 2D U-Net baseline and the 3D Improved U-Net on the test split,
plus an automatic failure-case autopsy.

Both models are scored on the same 9 test volumes, in the same 3D crop, at the original
voxel resolution:
    - the 2D model predicts the 60 coronal slices of each scan; slice k is placed back at
      volume[:, 98 + k, :] (with the left-right flip used by keras_slices_data)
    - the 3D model predicts the cropped volume directly

Prostate metrics per volume (acceptance criteria A1-A3 in the README):
    dice                overall Dice
    dice_apex / _base   Dice over the 3 most inferior / superior prostate-bearing axial slices,
                        plus any axial slice beyond them (catches over-extension)
    dice_central        Dice over the remaining axial slices
    dice_anterior /     the same along the anterior-posterior axis: the 3 most anterior / posterior
      _posterior        prostate-bearing coronal planes and beyond. The 2D model segments coronal
                        slices, so these are its slice transitions, where it sees no context
    hd95                95th percentile symmetric surface distance (mm)
    components          number of 26-connected predicted prostate components
Error breakdown (clinical meaning):
    apex/base FN, FP    missed / extra prostate volume (mL) in the apex / base regions
    missed_apex_mm      how far above the true apex the prediction stops (negative = extends below)
    missed_base_mm      how far below the true base the prediction stops (negative = extends above)
    leak_bladder/rectum prostate predicted inside the bladder / rectum (mL): toxicity risk
All classes:
    Dice and IoU of body, bone, bladder, rectum and prostate on the voxels both models predict (the
    60 coronal planes of the 2D slices inside the crop), and for the 3D model also over the
    whole crop

Several checkpoints per model (one per training seed) are evaluated separately; tables show
each run, the run average and the gap between runs, so model differences can be compared with
run-to-run variation. Per-volume metrics, failure-case selection and the profile plot use the
run average; failure figures show the predictions of the first run.

Usage:
    python evaluate.py --ckpt2d runs/unet_seed0/best.pth runs/unet_seed1/best.pth \
                       --ckpt3d runs/improved3d_seed0/best.pth runs/improved3d_seed1/best.pth \
                       --out results/final

Outputs (in --out): comparison.txt, comparison.json, apex_base_profile.png, failure_*.png

Author: Ruiqi Zuo (s4983299)
"""

import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import DataLoader

from dataset import CLASS_NAMES, CROP_3D, NUM_CLASSES, PROSTATE, HipMRI2DDataset, HipMRI3DDataset
from modules import UNet2D, ImprovedUNet3D

BLADDER, RECTUM = 3, 4
SLICE_OFFSET = 98   # keras slice k == volume[:, 98 + k, :] (left-right flipped)
N_EDGE = 3          # axial slices counted as apex / base
FULL_SHAPE = (256, 256, 128)
MODEL_2D, MODEL_3D = "2D U-Net", "3D Improved U-Net"


# --------------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------------- #
def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def predict_2d_volumes(model, root2d, device, batch_size=60):
    """
    Predict every 2D test slice and re-assemble them into cropped 3D volumes.

    Returns {(case, week): (pred_volume, gt_volume)} in CROP_3D coordinates, with the
    re-assembled 2D ground truth kept for a consistency check against the 3D labels.
    Voxels outside the 60-slice coronal window are set to background (0); in the test
    split the prostate lies entirely inside that window.
    """
    ds = HipMRI2DDataset(root2d, "test")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    vols = {}
    for image, mask, meta in loader:
        with _autocast(device):
            pred = model(image.to(device)).argmax(1).cpu().numpy().astype(np.uint8)
        for i in range(image.size(0)):
            key = (int(meta["case"][i]), int(meta["week"][i]))
            if key not in vols:
                vols[key] = (np.zeros(FULL_SHAPE, np.uint8), np.zeros(FULL_SHAPE, np.uint8))
            k = int(meta["slice"][i])
            vols[key][0][:, SLICE_OFFSET + k, :] = pred[i][::-1, :]
            vols[key][1][:, SLICE_OFFSET + k, :] = mask[i].numpy()[::-1, :]
    return {k: (p[CROP_3D], g[CROP_3D]) for k, (p, g) in vols.items()}


@torch.no_grad()
def predict_3d_volumes(model, root3d, device):
    """Predict every cropped 3D test volume. Returns {(case, week): (pred, gt, image)}."""
    ds = HipMRI3DDataset(root3d, "test")
    out = {}
    for i in range(len(ds)):
        image, mask, meta = ds[i]
        with _autocast(device):
            pred = model(image[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
        out[(meta["case"], meta["week"])] = (pred, mask.numpy().astype(np.uint8), image[0].numpy())
    return out


@torch.no_grad()
def time_inference(model, x, device, repeats=10):
    """Pure forward-pass latency (s) and peak inference VRAM (GiB) for input x."""
    x = x.to(device)
    with _autocast(device):
        model(x)  # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else float("nan")
    return (time.time() - t0) / repeats, vram


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def dice(p, g):
    """Dice of two boolean arrays; NaN if both are empty."""
    denom = p.sum() + g.sum()
    return 2.0 * (p & g).sum() / denom if denom > 0 else float("nan")


def hd95(p, g, spacing):
    """95th percentile symmetric surface distance in mm (inf if one mask is empty)."""
    if not p.any() or not g.any():
        return float("inf")
    sp = p ^ ndimage.binary_erosion(p)
    sg = g ^ ndimage.binary_erosion(g)
    dist_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return float(np.percentile(np.concatenate([dist_to_g[sp], dist_to_p[sg]]), 95))


def prostate_extent(mask, axis=2):
    """
    (lowest, highest) index along `axis` that contains the mask. Axis 2 runs inferior ->
    superior (axial slices), axis 1 anterior -> posterior (coronal planes).
    """
    z = np.nonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))[0]
    return (int(z.min()), int(z.max())) if len(z) else (None, None)


def prostate_metrics(pred, gt, spacing):
    """All per-volume prostate metrics."""
    p, g = pred == PROSTATE, gt == PROSTATE
    lo, hi = prostate_extent(g)
    apex = np.r_[0:lo + N_EDGE]                 # apex slices + everything below
    base = np.r_[hi - N_EDGE + 1:g.shape[2]]    # base slices + everything above
    central = np.r_[lo + N_EDGE:hi - N_EDGE + 1]
    alo, ahi = prostate_extent(g, axis=1)       # the same edges across coronal planes
    anterior = np.r_[0:alo + N_EDGE]
    posterior = np.r_[ahi - N_EDGE + 1:g.shape[1]]
    ml = float(np.prod(spacing)) / 1000.0       # mL per voxel
    plo, phi = prostate_extent(p)

    def region_errors(z):
        pz, gz = p[:, :, z], g[:, :, z]
        return float((gz & ~pz).sum() * ml), float((pz & ~gz).sum() * ml)

    apex_fn, apex_fp = region_errors(apex)
    base_fn, base_fp = region_errors(base)
    _, n_comp = ndimage.label(p, structure=np.ones((3, 3, 3)))
    return {
        "dice": dice(p, g),
        "dice_apex": dice(p[:, :, apex], g[:, :, apex]),
        "dice_central": dice(p[:, :, central], g[:, :, central]),
        "dice_base": dice(p[:, :, base], g[:, :, base]),
        "dice_anterior": dice(p[:, anterior], g[:, anterior]),
        "dice_posterior": dice(p[:, posterior], g[:, posterior]),
        "hd95": hd95(p, g, spacing),
        "components": int(n_comp),
        "apex_fn_ml": apex_fn, "apex_fp_ml": apex_fp,
        "base_fn_ml": base_fn, "base_fp_ml": base_fp,
        "missed_apex_mm": (plo - lo) * spacing[2] if plo is not None else float("nan"),
        "missed_base_mm": (hi - phi) * spacing[2] if phi is not None else float("nan"),
        "leak_bladder_ml": float((p & (gt == BLADDER)).sum() * ml),
        "leak_rectum_ml": float((p & (gt == RECTUM)).sum() * ml),
        "prostate_ml": float(g.sum() * ml),
        "prostate_slices": int(hi - lo + 1),
        "profile": [dice(p[:, :, s], g[:, :, s]) for s in range(lo, hi + 1)],
    }


def class_dice(pred, gt, region):
    """Dice of every foreground class (body ... prostate) inside the voxel index `region`."""
    p, g = pred[region], gt[region]
    return [dice(p == c, g == c) for c in range(1, NUM_CLASSES)]


def voxel_spacing(root3d, case, week):
    """Voxel size (mm) of a volume, read from its Nifti header."""
    seg_dir = os.path.join(root3d, "semantic_labels_only")
    name = next(n for n in os.listdir(seg_dir) if n[1:].startswith(f"{case:03d}_Week{week}_"))
    return tuple(float(z) for z in nib.load(os.path.join(seg_dir, name)).header.get_zooms())


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
SUMMARY_KEYS = ["dice", "dice_apex", "dice_central", "dice_base", "dice_anterior", "dice_posterior",
                "hd95", "components"]
SUMMARY_LABELS = ["Dice", "apex", "central", "base", "anterior", "posterior", "HD95 mm", "comp."]
ERROR_KEYS = ["apex_fn_ml", "apex_fp_ml", "base_fn_ml", "base_fp_ml",
              "missed_apex_mm", "missed_base_mm", "leak_bladder_ml", "leak_rectum_ml"]


def summarise(results, keys):
    """Mean and std over volumes for each scalar metric."""
    out = {}
    for k in keys:
        v = np.array([r[k] for r in results.values()], dtype=float)
        out[k] = (float(np.nanmean(v)), float(np.nanstd(v)))
    return out


def summarise_classes(runs):
    """
    Per-class Dice and IoU from a list of runs ({key: [Dice per class]}): mean and std over
    volumes of the run-averaged score, and the gap between the Dice means of the first two runs.
    IoU is derived per volume from Dice: IoU = Dice / (2 - Dice).
    """
    d = np.array([[r[k] for k in sorted(r)] for r in runs], dtype=float)  # (runs, volumes, classes)
    out = {}
    for prefix, score in (("", d), ("iou_", d / (2 - d))):
        per_volume = np.nanmean(score, axis=0)
        out[prefix + "mean"] = np.nanmean(per_volume, axis=0).tolist()
        out[prefix + "std"] = np.nanstd(per_volume, axis=0).tolist()
    run_means = np.nanmean(d, axis=1)
    out["gap"] = (np.abs(run_means[0] - run_means[1]) if len(runs) > 1 else np.full(d.shape[2], np.nan)).tolist()
    return out


def average_seeds(seed_results):
    """Average per-volume metrics over training seeds (list of {key: metrics} dicts)."""
    out = {}
    for key in seed_results[0]:
        runs = [r[key] for r in seed_results]
        avg = {k: float(np.nanmean([m[k] for m in runs])) for k in runs[0] if k != "profile"}
        avg["profile"] = np.nanmean([m["profile"] for m in runs], axis=0).tolist()
        avg["prostate_slices"] = runs[0]["prostate_slices"]
        out[key] = avg
    return out


def plot_profile(all_results, path, bins=10):
    """Mean prostate Dice along the gland, binned from apex (0) to base (1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8))
    centres = (np.arange(bins) + 0.5) / bins
    for name, results in all_results.items():
        sums, counts = np.zeros(bins), np.zeros(bins)
        for r in results.values():
            prof = np.array(r["profile"], dtype=float)
            pos = np.linspace(0, 1, len(prof), endpoint=False) + 0.5 / len(prof)
            idx = np.minimum((pos * bins).astype(int), bins - 1)
            ok = ~np.isnan(prof)
            np.add.at(sums, idx[ok], prof[ok])
            np.add.at(counts, idx[ok], 1)
        ax.plot(centres, sums / np.maximum(counts, 1), "o-", label=name)
    ax.set_xlabel("relative position in prostate (0 = apex, 1 = base)")
    ax.set_ylabel("mean axial-slice Dice")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Prostate Dice from apex to base (test volumes)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_failure_cases(all_results, n=4):
    """
    Pick representative failure cases, centred on the proposed 3D model: its worst apex and
    worst base volumes, the baseline's worst apex volume, and the volume where 3D falls
    furthest behind 2D at apex or base. Returns [(key, reason), ...] without duplicates.
    """
    r2, r3 = all_results[MODEL_2D], all_results[MODEL_3D]
    keys = sorted(r2)
    rules = [
        ("worst 3D apex", lambda k: r3[k]["dice_apex"]),
        ("worst 2D apex", lambda k: r2[k]["dice_apex"]),
        ("worst 3D base", lambda k: r3[k]["dice_base"]),
        ("largest 3D regression vs 2D", lambda k: -max(r2[k]["dice_apex"] - r3[k]["dice_apex"],
                                                         r2[k]["dice_base"] - r3[k]["dice_base"])),
    ]
    picked = []
    for reason, score in rules:
        for k in sorted(keys, key=score):
            if k not in [p[0] for p in picked]:
                picked.append((k, reason))
                break
    return picked[:n]


def plot_failure_case(key, reason, image, gt, preds, results, path, margin=10):
    """
    Multi-panel comparison for one volume.

    Rows: sagittal plane through the prostate centre (shows apex and base together),
    axial plane at the apex, axial plane at the base. Columns: MR, ground truth,
    2D prediction, 3D prediction. Prediction panels colour prostate voxels as
    true positive (yellow), false positive (red) and false negative (cyan).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    g = gt == PROSTATE
    union = g | np.any([p == PROSTATE for p in preds.values()], axis=0)
    box = [(max(int(i.min()) - margin, 0), min(int(i.max()) + margin + 1, s))
           for i, s in zip(np.nonzero(union), union.shape)]
    lo, hi = prostate_extent(g)
    centre_l = int(np.round(np.nonzero(g)[0].mean()))
    vmin, vmax = np.percentile(image, [1, 99])

    def most_different_slice(candidates):
        """Axial slice among `candidates` where the models' prostate Dice differ most."""
        def gap(s):
            d = [dice(p[:, :, s] == PROSTATE, g[:, :, s]) for p in preds.values()]
            return np.nan_to_num(max(d) - min(d))
        return max(candidates, key=gap)

    s_apex = most_different_slice(range(lo, lo + N_EDGE))
    s_base = most_different_slice(range(hi - N_EDGE + 1, hi + 1))

    # each view: (title, function extracting a 2D plane from a volume)
    views = [
        (f"sagittal (L={centre_l})", lambda v: v[centre_l, box[1][0]:box[1][1], box[2][0]:box[2][1]].T),
        (f"axial apex (S={s_apex}, apex at {lo})", lambda v: v[box[0][0]:box[0][1], box[1][0]:box[1][1], s_apex].T),
        (f"axial base (S={s_base}, base at {hi})", lambda v: v[box[0][0]:box[0][1], box[1][0]:box[1][1], s_base].T),
    ]

    def organ_rgba(lab):
        rgba = np.zeros(lab.shape + (4,))
        rgba[lab == BLADDER] = (0.2, 0.4, 1.0, 0.45)
        rgba[lab == RECTUM] = (0.1, 0.8, 0.2, 0.45)
        rgba[lab == PROSTATE] = (1.0, 0.1, 0.1, 0.55)
        return rgba

    def error_rgba(pred_plane, gt_plane):
        p, t = pred_plane == PROSTATE, gt_plane == PROSTATE
        rgba = np.zeros(p.shape + (4,))
        rgba[p & t] = (1.0, 0.9, 0.0, 0.55)
        rgba[p & ~t] = (1.0, 0.0, 0.0, 0.85)
        rgba[~p & t] = (0.0, 0.9, 1.0, 0.85)
        return rgba

    cols = ["MR", "ground truth"] + list(preds)
    fig, axes = plt.subplots(len(views), len(cols), figsize=(3.0 * len(cols), 3.4 * len(views)))
    for r, (vtitle, extract) in enumerate(views):
        img_plane, gt_plane = extract(image), extract(gt)
        for c, col in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(img_plane, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
            if col == "ground truth":
                ax.imshow(organ_rgba(gt_plane), origin="lower", interpolation="nearest")
            elif col in preds:
                ax.imshow(error_rgba(extract(preds[col]), gt_plane), origin="lower", interpolation="nearest")
                ax.contour(gt_plane == BLADDER, levels=[0.5], colors="royalblue", linewidths=0.6)
                ax.contour(gt_plane == RECTUM, levels=[0.5], colors="limegreen", linewidths=0.6)
            title = vtitle if c == 0 else col
            if col in preds:
                m = results[col][key]
                region = "dice" if r == 0 else ("dice_apex" if r == 1 else "dice_base")
                title += f"\n{region.replace('dice_', '') if r else 'overall'} Dice {m[region]:.3f}"
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    legend = [Patch(color=(1, 0.9, 0), label="TP"), Patch(color="red", label="FP (extra)"),
              Patch(color=(0, 0.9, 1), label="FN (missed)"), Patch(color="royalblue", label="bladder"),
              Patch(color="limegreen", label="rectum")]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"case {key[0]:03d} week {key[1]} - {reason}   "
                 "(sagittal: superior up, posterior right; axial: posterior up)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def format_report(seed_results, all_results, class_results, resources, failures, ckpts):
    """Human-readable comparison tables (all_results = per-volume metrics averaged over seeds)."""
    lines = []
    header = f"{'model':18s} {'seed':>5s} " + " ".join(f"{n:>12s}" for n in SUMMARY_LABELS)
    lines += ["mean ± std over the 9 test volumes, per training run and averaged over runs:",
              header, "-" * len(header)]
    dice_keys = [k for k in SUMMARY_KEYS if k.startswith("dice")]

    def row(name, label, results):
        s = summarise(results, SUMMARY_KEYS)
        return f"{name:18s} {label:>5s} " + " ".join(
            f"{s[k][0]:6.3f}±{s[k][1]:5.3f}" if k != "components" else f"{s[k][0]:4.1f}±{s[k][1]:4.1f}"
            for k in SUMMARY_KEYS)

    for name, runs in seed_results.items():
        for i, results in enumerate(runs):
            lines.append(row(name, str(i), results))
        if len(runs) > 1:
            lines.append(row(name, "mean", all_results[name]))
            gap = {k: abs(summarise(runs[0], [k])[k][0] - summarise(runs[1], [k])[k][0]) for k in dice_keys}
            lines.append(f"{'':18s} {'|0-1|':>5s} " + " ".join(f"{gap[k]:12.3f}" for k in dice_keys))
    lines += ["", "all classes: Dice (first row) and IoU per volume, mean ± std over volumes, averaged over runs;",
              "|0-1| = Dice gap between runs (2D window = the 60 coronal planes of the 2D slices inside the crop,",
              "predicted by both models):"]
    header = f"{'model':37s} " + " ".join(f"{n:>13s}" for n in CLASS_NAMES[1:])
    lines += [header, "-" * len(header)]
    for name, s in class_results.items():
        lines.append(f"{name:37s} " + " ".join(f"{m:6.3f}±{sd:5.3f}" for m, sd in zip(s["mean"], s["std"])))
        lines.append(f"{'':31s} {'IoU':>5s} " + " ".join(f"{m:6.3f}±{sd:5.3f}"
                                                         for m, sd in zip(s["iou_mean"], s["iou_std"])))
        lines.append(f"{'':31s} {'|0-1|':>5s} " + " ".join(f"{g:13.3f}" for g in s["gap"]))
    lines += ["", "error breakdown (mean over volumes and runs):"]
    header = (f"{'model':18s} {'apexFN':>7s} {'apexFP':>7s} {'baseFN':>7s} {'baseFP':>7s} "
              f"{'missApex':>9s} {'missBase':>9s} {'->bladder':>9s} {'->rectum':>9s}")
    lines += [header, f"{'':18s} {'(mL)':>7s} {'(mL)':>7s} {'(mL)':>7s} {'(mL)':>7s} "
                      f"{'(mm)':>9s} {'(mm)':>9s} {'(mL)':>9s} {'(mL)':>9s}"]
    for name, results in all_results.items():
        s = summarise(results, ERROR_KEYS)
        lines.append(f"{name:18s} " + " ".join(
            f"{s[k][0]:{9 if 'mm' in k or 'leak' in k else 7}.2f}" for k in ERROR_KEYS))
    lines += ["", "per volume: Dice / apex / base / HD95 mm / components / missed apex mm"]
    for key in sorted(all_results[MODEL_2D]):
        row = [f"  case {key[0]:03d} wk {key[1]} ({all_results[MODEL_2D][key]['prostate_slices']:2d} sl)"]
        for name in all_results:
            r = all_results[name][key]
            row.append(f"{name.split()[0]}: {r['dice']:.3f}/{r['dice_apex']:.3f}/{r['dice_base']:.3f}/"
                       f"{r['hd95']:4.1f}/{r['components']:.1f}/{r['missed_apex_mm']:+5.1f}")
        lines.append("  ".join(row))
    lines += ["", "per patient (mean prostate Dice / apex / base):"]
    for case in sorted({k[0] for k in all_results[MODEL_2D]}):
        row = [f"  patient {case:03d} ({sum(k[0] == case for k in all_results[MODEL_2D])} scans)"]
        for name, results in all_results.items():
            sub = [r for k, r in results.items() if k[0] == case]
            row.append(f"{name.split()[0]}: " + "/".join(
                f"{np.nanmean([r[m] for r in sub]):.3f}" for m in ("dice", "dice_apex", "dice_base")))
        lines.append("  ".join(row))
    lines.append("")
    for name, r in resources.items():
        lines.append(f"{name:18s} params {r['params']:,}  inference {r['s_per_volume'] * 1000:.0f} ms/volume  "
                     f"peak inference VRAM {r['infer_vram_gb']:.2f} GiB")
    lines += ["", "failure cases (chosen on run-averaged metrics, figures show run 0): "
              + ", ".join(f"case {k[0]:03d} wk {k[1]} ({why})" for k, why in failures)]
    lines += ["", "checkpoints:"] + [f"  {n} run {i}: {c}" for n, cs in ckpts.items() for i, c in enumerate(cs)]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare 2D U-Net and 3D Improved U-Net in 3D")
    parser.add_argument("--ckpt2d", nargs="+", default=["runs/unet_seed0/best.pth"],
                        help="one checkpoint per training run (seed)")
    parser.add_argument("--ckpt3d", nargs="+", default=["runs/improved3d_seed0/best.pth"])
    parser.add_argument("--root2d", default=os.path.expanduser("~/data/keras_slices_data"))
    parser.add_argument("--root3d", default=os.path.expanduser("~/data/HipMRI_3D"))
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = {MODEL_2D: args.ckpt2d, MODEL_3D: args.ckpt3d}

    def load(cls, path):
        model = cls().to(device).eval()
        model.load_state_dict(torch.load(path, map_location=device))
        return model

    # predictions of every run; run 0 also provides images / labels for figures
    preds = {MODEL_2D: [], MODEL_3D: []}
    for path in args.ckpt3d:
        vols3d = predict_3d_volumes(load(ImprovedUNet3D, path), args.root3d, device)
        preds[MODEL_3D].append({k: v[0] for k, v in vols3d.items()})
    gts = {k: v[1] for k, v in vols3d.items()}
    images = {k: v[2] for k, v in vols3d.items()}

    window = slice(SLICE_OFFSET - CROP_3D[1].start, SLICE_OFFSET - CROP_3D[1].start + 60)
    for path in args.ckpt2d:
        vols2d = predict_2d_volumes(load(UNet2D, path), args.root2d, device)
        preds[MODEL_2D].append({k: v[0] for k, v in vols2d.items()})
        # sanity check: re-assembled 2D labels must equal the 3D labels inside the 2D window
        for key, (_, gt2d) in vols2d.items():
            assert np.array_equal(gt2d[:, window, :], gts[key][:, window, :]), f"label mismatch {key}"
            g = gts[key] == PROSTATE
            assert not g[:, :window.start].any() and not g[:, window.stop:].any(), f"prostate outside window {key}"
    print("2D->3D re-assembly verified: labels identical, prostate inside the 2D window for all test volumes")

    spacing = {key: voxel_spacing(args.root3d, *key) for key in gts}
    seed_results = {name: [{key: prostate_metrics(pred, gts[key], spacing[key]) for key, pred in sorted(run.items())}
                           for run in runs]
                    for name, runs in preds.items()}
    all_results = {name: average_seeds(runs) for name, runs in seed_results.items()}

    # all classes on identical voxels; the 3D model is also scored over the whole crop
    in_window, whole = (slice(None), window, slice(None)), (slice(None),) * 3
    class_results = {
        f"{MODEL_2D} (2D window)": [{k: class_dice(p, gts[k], in_window) for k, p in run.items()}
                                    for run in preds[MODEL_2D]],
        f"{MODEL_3D} (2D window)": [{k: class_dice(p, gts[k], in_window) for k, p in run.items()}
                                    for run in preds[MODEL_3D]],
        f"{MODEL_3D} (whole crop)": [{k: class_dice(p, gts[k], whole) for k, p in run.items()}
                                     for run in preds[MODEL_3D]],
    }
    class_results = {name: summarise_classes(runs) for name, runs in class_results.items()}

    # resources: pure forward time per test volume (60 slices for 2D, one crop for 3D)
    m2d, m3d = load(UNet2D, args.ckpt2d[0]), load(ImprovedUNet3D, args.ckpt3d[0])
    t2d, v2d = time_inference(m2d, torch.randn(60, 1, 256, 128), device)
    t3d, v3d = time_inference(m3d, torch.randn(1, 1, 128, 144, 128), device)
    resources = {
        MODEL_2D: {"params": sum(p.numel() for p in m2d.parameters()), "s_per_volume": t2d, "infer_vram_gb": v2d},
        MODEL_3D: {"params": sum(p.numel() for p in m3d.parameters()), "s_per_volume": t3d, "infer_vram_gb": v3d},
    }

    failures = select_failure_cases(all_results)
    run0 = {name: runs[0] for name, runs in seed_results.items()}
    for key, reason in failures:
        plot_failure_case(key, reason, images[key], gts[key], {n: preds[n][0][key] for n in preds},
                          run0, os.path.join(args.out, f"failure_{key[0]:03d}_wk{key[1]}.png"))
    plot_profile(all_results, os.path.join(args.out, "apex_base_profile.png"))

    report = format_report(seed_results, all_results, class_results, resources, failures, ckpts)
    print(report)
    with open(os.path.join(args.out, "comparison.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump({"checkpoints": ckpts,
                   "summary": {n: summarise(r, SUMMARY_KEYS + ERROR_KEYS) for n, r in all_results.items()},
                   "summary_per_run": {n: [summarise(r, SUMMARY_KEYS + ERROR_KEYS) for r in runs]
                                       for n, runs in seed_results.items()},
                   "class_dice": {n: dict(s, classes=CLASS_NAMES[1:]) for n, s in class_results.items()},
                   "resources": resources,
                   "failure_cases": [{"case": k[0], "week": k[1], "reason": why} for k, why in failures],
                   "per_volume": {n: {f"{c:03d}_week{w}": r for (c, w), r in res.items()}
                                  for n, res in all_results.items()}}, f, indent=2)
    print(f"Saved results to {args.out}/")


if __name__ == "__main__":
    main()
