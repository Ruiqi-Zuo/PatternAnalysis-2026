"""
Model components for HipMRI 2D prostate segmentation.

Contains:
    - UNet2D:          baseline 2D U-Net (Ronneberger et al., 2015)
    - ImprovedUNet2D:  2D Improved U-Net with residual context modules,
                       localisation modules and deep supervision (Isensee et al., 2017)
    - DiceLoss:        multi-class soft Dice loss

All components are implemented in PyTorch only (no NumPy).

Author: Ruiqi Zuo (s4983299)
"""

import torch
import torch.nn as nn


# TODO: baseline 2D U-Net
class UNet2D(nn.Module):
    pass


# TODO: Improved U-Net
class ImprovedUNet2D(nn.Module):
    pass


# TODO: soft Dice loss
class DiceLoss(nn.Module):
    pass
