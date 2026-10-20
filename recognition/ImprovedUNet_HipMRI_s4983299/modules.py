"""
Model components for HipMRI 2D prostate segmentation.

Contains:
    - UNet2D:          baseline 2D U-Net (Ronneberger et al., 2015)
    - ImprovedUNet2D:  2D Improved U-Net with residual context modules,
                       localisation modules and deep supervision (Isensee et al., 2017)
    - DiceLoss:        multi-class soft Dice loss
    - DiceCELoss:      soft Dice + cross-entropy (used for training)

All components are implemented in PyTorch only (no NumPy).

Author: Ruiqi Zuo (s4983299)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Baseline: 2D U-Net
# --------------------------------------------------------------------------- #
class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x 2, spatial size preserved by padding=1."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """
    Baseline 2D U-Net (Ronneberger et al., 2015).

    Encoder: 4 x (DoubleConv -> MaxPool), channels base, 2*base, 4*base, 8*base.
    Bottleneck: DoubleConv with 16*base channels.
    Decoder: 4 x (ConvTranspose 2x2 upsample -> concat skip -> DoubleConv).
    Differences from the original paper: padded convolutions (output size = input size)
    and BatchNorm, which are standard modern choices.

    Input (B, in_channels, H, W) with H, W divisible by 16 (here 256 x 128).
    Output raw logits (B, num_classes, H, W).
    """

    def __init__(self, in_channels=1, num_classes=6, base=32):
        super().__init__()
        chs = [base, base * 2, base * 4, base * 8]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.encoders.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(chs[-1], chs[-1] * 2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = chs[-1] * 2
        for c in reversed(chs):
            self.upsamples.append(nn.ConvTranspose2d(prev, c, kernel_size=2, stride=2))
            self.decoders.append(DoubleConv(c * 2, c))  # c from upsample + c from skip
            prev = c

        self.head = nn.Conv2d(chs[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))

        return self.head(x)


# --------------------------------------------------------------------------- #
# Improved U-Net
# --------------------------------------------------------------------------- #
# TODO: Improved U-Net
class ImprovedUNet2D(nn.Module):
    pass


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss.

        Dice_c = (2 * sum(p_c * g_c) + eps) / (sum(p_c) + sum(g_c) + eps)
        loss   = 1 - mean_c(Dice_c)

    p_c are softmax probabilities, g_c the one-hot ground truth; sums run over the
    whole batch and all pixels, so a class absent from one slice does not give an
    undefined 0/0 term. Each class counts equally regardless of its size, which
    counteracts the extreme imbalance (prostate ~0.4% of pixels).

    Args:
        num_classes: number of segmentation labels.
        include_background: whether label 0 contributes to the mean.
        eps: smoothing constant.
    """

    def __init__(self, num_classes=6, include_background=False, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.eps = eps

    def forward(self, logits, target):
        # logits (B, C, H, W), target (B, H, W) int64
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).to(probs.dtype)

        dims = (0, 2, 3)  # sum over batch and spatial dims -> one value per class
        intersection = (probs * one_hot).sum(dims)
        denominator = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (denominator + self.eps)

        if not self.include_background:
            dice = dice[1:]
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    """
    Weighted sum of soft Dice loss and cross-entropy.

    Cross-entropy gives stable per-pixel gradients early in training; Dice
    directly optimises region overlap for the small structures.
    """

    def __init__(self, num_classes=6, dice_weight=1.0, ce_weight=1.0):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits, target):
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * self.ce(logits, target)


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: shapes, parameter count, loss sanity and one backward pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet2D().to(device)
    x = torch.randn(4, 1, 256, 128, device=device)
    y = torch.randint(0, 6, (4, 256, 128), device=device)

    logits = model(x)
    print(f"UNet2D output {tuple(logits.shape)}, params {count_parameters(model):,}")

    loss_fn = DiceCELoss()
    loss = loss_fn(logits, y)
    loss.backward()
    print(f"DiceCE loss on random input: {loss.item():.4f}")

    # a perfect prediction must give Dice loss ~0
    perfect = F.one_hot(y, 6).permute(0, 3, 1, 2).float() * 100
    print(f"Dice loss for perfect logits: {DiceLoss()(perfect, y).item():.6f}")
