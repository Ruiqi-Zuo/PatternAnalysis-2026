"""
Example usage of the trained model.

Loads saved weights, runs inference on test cases, prints Dice / IoU results
and saves visualisations (input, ground truth, prediction, FP/FN overlay).

Author: Ruiqi Zuo (s4983299)
"""

from modules import UNet2D, ImprovedUNet3D
from dataset import get_dataloaders


def main():
    # TODO: load weights, run inference, report metrics, save figures
    pass


if __name__ == "__main__":
    main()
