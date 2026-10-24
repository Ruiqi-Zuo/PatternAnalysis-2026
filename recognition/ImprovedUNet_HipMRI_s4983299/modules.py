"""
Model components for HipMRI prostate segmentation.

Contains:
    - UNet2D:          baseline 2D U-Net (Ronneberger et al., 2015)
    - ImprovedUNet3D:  3D Improved U-Net with pre-activation residual context modules,
                       localisation modules and deep supervision (Isensee et al., 2017)
    - DiceLoss:        multi-class soft Dice loss (2D or 3D)
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
def conv_norm_act(in_ch, out_ch, kernel_size=3, stride=1):
    """Conv3d -> InstanceNorm3d -> LeakyReLU(0.01)."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
        nn.InstanceNorm3d(out_ch, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ContextModule(nn.Module):
    """
    Pre-activation residual block: (IN -> LReLU -> Conv3) -> Dropout -> (IN -> LReLU -> Conv3).

    The caller adds its input back (x + context(x)), as in the paper.
    """

    def __init__(self, ch, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
            nn.Dropout3d(dropout),
            nn.InstanceNorm3d(ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class UpsamplingModule(nn.Module):
    """Nearest-neighbour x2 upscale followed by a 3x3x3 conv that halves the channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = conv_norm_act(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.up(x))


class LocalisationModule(nn.Module):
    """3x3x3 conv on the concatenated features, then 1x1x1 conv halving the channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(conv_norm_act(in_ch, in_ch), conv_norm_act(in_ch, out_ch, kernel_size=1))

    def forward(self, x):
        return self.block(x)


class ImprovedUNet3D(nn.Module):
    """
    3D Improved U-Net (Isensee et al., 2017, "Brain tumor segmentation and radiomics
    survival prediction: contribution to the BRATS 2017 challenge").

    Encoder (5 levels, channels base * [1, 2, 4, 8, 16]):
        level 1: conv3 -> x + context(x)
        level k: stride-2 conv3 (downsampling) -> x + context(x)
    Decoder:
        upsampling module -> concat skip -> localisation module (levels 4..2)
        level 1: concat skip -> conv3
    Deep supervision: 1x1x1 segmentation layers at decoder levels 3, 2 and 1; the coarser
        maps are upscaled and added element-wise to the finer ones, so gradients reach the
        deeper decoder levels directly (helps small structures such as the prostate).

    InstanceNorm is used instead of BatchNorm because 3D batches hold only 1-2 volumes.
    Input (B, in_channels, D, H, W) with D, H, W divisible by 16; output logits
    (B, num_classes, D, H, W).
    """

    def __init__(self, in_channels=1, num_classes=6, base=16, dropout=0.3):
        super().__init__()
        chs = [base * m for m in (1, 2, 4, 8, 16)]

        # encoder
        self.stem = conv_norm_act(in_channels, chs[0])
        self.down = nn.ModuleList([conv_norm_act(chs[i - 1], chs[i], stride=2) for i in range(1, 5)])
        self.context = nn.ModuleList([ContextModule(c, dropout) for c in chs])

        # decoder: up[i] goes from level 4-i+1 to level 4-i (0-based: 4->3, 3->2, 2->1, 1->0)
        self.up = nn.ModuleList([UpsamplingModule(chs[i], chs[i - 1]) for i in (4, 3, 2, 1)])
        self.loc = nn.ModuleList([LocalisationModule(chs[i] * 2, chs[i]) for i in (3, 2, 1)])
        self.final_conv = conv_norm_act(chs[0] * 2, chs[0] * 2)

        # deep supervision heads (levels 3, 2 and full resolution)
        self.seg3 = nn.Conv3d(chs[2], num_classes, 1)
        self.seg2 = nn.Conv3d(chs[1], num_classes, 1)
        self.seg1 = nn.Conv3d(chs[0] * 2, num_classes, 1)
        self.upscale = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

    def forward(self, x):
        # encoder
        x = self.stem(x)
        x = x + self.context[0](x)
        skips = [x]
        for down, ctx in zip(self.down, self.context[1:]):
            x = down(x)
            x = x + ctx(x)
            skips.append(x)
        x = skips.pop()  # bottleneck (level 5)

        # decoder levels 4, 3, 2 with localisation modules
        seg_maps = []
        for up, loc in zip(self.up[:3], self.loc):
            x = loc(torch.cat([skips.pop(), up(x)], dim=1))
            seg_maps.append(x)

        # level 1 (full resolution)
        x = self.final_conv(torch.cat([skips.pop(), self.up[3](x)], dim=1))

        # deep supervision: seg maps from level 3 and 2 are upscaled and summed into level 1
        _, level3, level2 = seg_maps
        out = self.upscale(self.seg3(level3)) + self.seg2(level2)
        return self.upscale(out) + self.seg1(x)


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
        # logits (B, C, *spatial), target (B, *spatial) int64; works for 2D and 3D
        probs = F.softmax(logits.float(), dim=1)
        one_hot = F.one_hot(target, self.num_classes).movedim(-1, 1).to(probs.dtype)

        dims = (0, *range(2, logits.dim()))  # sum over batch and spatial dims -> one value per class
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
    # Smoke test: shapes, parameter count, loss sanity, one training step and its cost
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = DiceCELoss()

    def profile(model, shape, batch, steps=3):
        """Run a few bf16 training steps on random data; return output shape and costs."""
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(batch, 1, *shape, device=device)
        y = torch.randint(0, 6, (batch, *shape), device=device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for step in range(steps + 1):  # first step is warm-up
            if step == 1:
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / steps
        vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device == "cuda" else float("nan")
        return tuple(logits.shape), loss.item(), dt, vram

    for name, model, shape, batch in [
        ("UNet2D", UNet2D(), (256, 128), 16),
        ("ImprovedUNet3D", ImprovedUNet3D(), (128, 144, 128), 2),
    ]:
        out, loss, dt, vram = profile(model, shape, batch)
        print(f"{name:15s} out {out}  params {count_parameters(model):,}  "
              f"loss {loss:.3f}  {dt:.3f}s/step  peak VRAM {vram:.2f} GiB (bf16, batch {batch})")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # a perfect prediction must give Dice loss ~0 (2D and 3D)
    for shape in [(2, 64, 32), (1, 16, 16, 16)]:
        y = torch.randint(0, 6, shape)
        perfect = F.one_hot(y, 6).movedim(-1, 1).float() * 100
        print(f"Dice loss for perfect logits, target {shape}: {DiceLoss()(perfect, y).item():.6f}")
