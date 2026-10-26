"""
GPU memory and speed of the 3D Improved U-Net (modules.ImprovedUNet3D), used as the
resource estimate in the feasibility review.

For each input size, batch size and precision it runs a few Adam training steps with
DiceCELoss on random data (peak memory, time per step) and a few batch-1 forward passes
(peak memory, time per volume). Sizes are voxels along (left-right, anterior-posterior,
inferior-superior): the training crop VOLUME_SHAPE and the full 256 x 256 x 128 volume.

On Windows / WSL the driver can spill past the card's memory into system RAM instead of
raising an out-of-memory error; such runs still finish but are many times slower.

Usage:
    python scripts/profile3d.py

Author: Ruiqi Zuo (s4983299)
"""

import os
import sys
import time

import torch

# make the project modules importable when run from any directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dataset import VOLUME_SHAPE  # noqa: E402
from modules import DiceCELoss, ImprovedUNet3D, count_parameters  # noqa: E402

FULL_SHAPE = (256, 256, 128)
# (input shape, batch size, bf16 autocast); the first row is train.py's 3D configuration
CONFIGS = [
    (VOLUME_SHAPE, 2, True),
    (VOLUME_SHAPE, 2, False),
    (FULL_SHAPE, 1, True),
    (FULL_SHAPE, 1, False),
]


def train_cost(model, shape, batch, bf16, steps=3):
    """Peak memory (GiB) and seconds per training step, or None when out of memory."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DiceCELoss()
    x = torch.randn(batch, 1, *shape, device="cuda")
    y = torch.randint(0, 6, (batch, *shape), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(steps + 1):  # step 0 is a warm-up
            if step == 1:
                torch.cuda.synchronize()
                start = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2 ** 30, (time.time() - start) / steps
    except torch.OutOfMemoryError:
        return None


@torch.no_grad()
def inference_cost(model, shape, bf16, repeats=3):
    """Peak memory (GiB) and seconds per volume for batch-1 inference, or None when out of memory."""
    model.eval()
    x = torch.randn(1, 1, *shape, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            model(x)  # warm-up
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                model(x)
            torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2 ** 30, (time.time() - start) / repeats
    except torch.OutOfMemoryError:
        return None


def main():
    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required.")
    props = torch.cuda.get_device_properties(0)
    print(f"ImprovedUNet3D: {count_parameters(ImprovedUNet3D()):,} parameters | {props.name}, "
          f"{props.total_memory / 2 ** 30:.1f} GiB | torch {torch.__version__}")

    def fmt(result, unit):
        return "out of memory" if result is None else f"{result[0]:5.2f} GiB, {result[1]:6.2f} {unit}"

    for shape, batch, bf16 in CONFIGS:
        # a fresh model per configuration, so leftover gradients do not count towards the peak
        model = ImprovedUNet3D().cuda()
        train = train_cost(model, shape, batch, bf16)
        infer = inference_cost(model, shape, bf16)
        print(f"{'x'.join(map(str, shape)):>11s} batch {batch} {'bf16' if bf16 else 'fp32'} | "
              f"train {fmt(train, 's/step')} | inference {fmt(infer, 's/volume')}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
