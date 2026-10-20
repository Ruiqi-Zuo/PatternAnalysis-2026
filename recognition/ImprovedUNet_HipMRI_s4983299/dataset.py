"""
Data loading and preprocessing for the HipMRI 2D prostate slices (keras_slices_data).

Responsibilities:
    - load 2D Nifti image / segmentation slices
    - intensity normalisation and one-hot mask encoding
    - patient-level train / validation / test splitting (no cross-patient leakage)
    - data augmentation applied identically to image and mask

Author: Ruiqi Zuo (s4983299)
"""

import torch
from torch.utils.data import Dataset, DataLoader


# TODO: dataset class for HipMRI 2D slices
class HipMRI2DDataset(Dataset):
    pass


# TODO: patient-level split
def patient_level_split():
    pass


# TODO: build train / val / test DataLoaders
def get_dataloaders():
    pass
