"""
This file contains the data loader for loading and preprocessing your data
"""

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

NUM_CLASSES = 6

class Prostate3DDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, subset=None, num_classes=NUM_CLASSES):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes

        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".nii", ".nii.gz"))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith((".nii", ".nii.gz"))])

        mask_map = {}
        for f in self.mask_files:
            base = f.replace("_SEMANTIC.nii.gz", "").replace("_SEMANTIC.nii", "")
            mask_map[base] = f

        paired = []
        for f in self.image_files:
            base = f.replace("_LFOV.nii.gz", "").replace("_LFOV.nii", "")
            if base in mask_map:
                paired.append((f, mask_map[base]))

        if subset:
            paired = paired[:subset]

        self.pairs = paired
        print(f"📦 Loaded {len(self.pairs)} paired volumes from {image_dir}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_file, mask_file = self.pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image = image_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.int64)

        image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        image = torch.tensor(image).unsqueeze(0)

        mask = torch.tensor(mask, dtype=torch.long)
        one_hot = torch.nn.functional.one_hot(mask, num_classes=self.num_classes)
        one_hot = one_hot.permute(3, 0, 1, 2).float()

        if self.transform:
            image, one_hot = self.transform(image, one_hot)

        return image, one_hot
