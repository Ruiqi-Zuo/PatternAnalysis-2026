"""
Training, validation, testing and saving of the segmentation models.

Usage (planned):
    python train.py --model improved   # Improved U-Net
    python train.py --model unet       # baseline 2D U-Net

Model is imported from modules.py, data loaders from dataset.py.
Loss and Dice curves are plotted during training.

Author: Ruiqi Zuo (s4983299)
"""

from modules import UNet2D, ImprovedUNet2D, DiceLoss
from dataset import get_dataloaders


# TODO: one training epoch
def train_one_epoch():
    pass


# TODO: validation / test evaluation (per-class Dice, IoU)
def evaluate():
    pass


def main():
    # TODO: parse args, build model, train loop, save best weights, plot curves
    pass


if __name__ == "__main__":
    main()
