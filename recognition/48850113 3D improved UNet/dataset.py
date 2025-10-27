"""
This file contains the data loader for loading and preprocessing the data.
It can also apply 3D augmentation (random rotations) to the images and masks.
"""

import os
import torch
import nibabel as nib
import numpy as np
import random
import scipy.ndimage
from torch.utils.data import Dataset

# Target shape for all images (depth, height, width)
TARGET_SHAPE = (64, 128, 128)
NUM_CLASSES = 6

def resize_volume(volume, target_shape=TARGET_SHAPE):
    """
    Resize 3D image volume to a target shape using linear interpolation.

    Parameters:
    Volume: 3D image volume [D, H, W]
    target_shape: Desired output shape as (D, H, W)

    Returns:
    numpy.ndarray: Resized 3D images
    """
    # Compute scaling factors for each dimension
    factors = (
        target_shape[0] / volume.shape[0], # Depth scaling
        target_shape[1] / volume.shape[1], # Height scaling
        target_shape[2] / volume.shape[2], # Width scaling
    )
    # Resize using linear iterpolation
    volume_resized = scipy.ndimage.zoom(volume, factors, order=1)
    return volume_resized

def resize_mask(mask, target_shape=TARGET_SHAPE):
    """
    Resize mask to a target shape using nearest neighbour interpolation.

    Parameters:
    mask: 3D masks [D, H, W]
    target_shape: Desired output shape as (D, H, W)

    Returns:
    numpy.ndarray: Resized 3D mask (labels for classes are preserved for one-hot encoding)
    """
    # Compute scaling factors for each dimension
    factors = (
        target_shape[0] / mask.shape[0], # Depth scaling
        target_shape[1] / mask.shape[1], # Height scaling
        target_shape[2] / mask.shape[2], # Width scaling
    )

    # Resize using nearest neighbour interpolation to preserve labels for classes
    mask_resized = scipy.ndimage.zoom(mask, factors, order=0)
    return mask_resized.astype(np.int64)

class Prostate3DDataset(Dataset):
    """
    Dataset class for 3D U-Net model.
    Loads paired 3D MRI images and corresponding segmentation masks from specified directories.
    Can apply transformations (e.g. augmentation) and one-hot encode masks for segmentation wil multiple classes.
    Used to create data loaders for the training, validation and testing phases.
    """
    def __init__(self, image_dir, mask_dir, transform=None, subset=None, num_classes=NUM_CLASSES):
        """
        Initialise the dataset class by loading and pairing images with their masks.

        Parameters:
        image_dir: Path to directory contatining the imgaes.
        mask_dir: Path to directory containing the masks.
        transform: Transformation function to apply to both the images and masks.
        subset: Number of samples to load from the dataset.
        num_classes: Number of segmentation classes for one-hot encoding.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.num_classes = num_classes

        # Get all image and mask filenames
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".nii", ".nii.gz"))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith((".nii", ".nii.gz"))])

        # Create a dictionary mapping of mask filenames by case number
        mask_map = {}
        for f in self.mask_files:
            base = f.replace("_SEMANTIC.nii.gz", "").replace("_SEMANTIC.nii", "")
            mask_map[base] = f

        # Pair each image with its corresponding mask using case numbers
        paired = []
        for f in self.image_files:
            base = f.replace("_LFOV.nii.gz", "").replace("_LFOV.nii", "")
            if base in mask_map:
                paired.append((f, mask_map[base]))

        # Limit dataset size to subset size if provided
        if subset:
            paired = paired[:subset]

        self.pairs = paired
        print(f"📦 Loaded {len(self.pairs)} paired volumes from {image_dir}")

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
        int: Number of paired samples avaliable in the dataset.
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image and its corresponding segmentation mask.

        Parameters:
        idx: Index of sample to retrieve.

        Returns:
        tuple: (image, one_hot_mask)
        - image: Normalised MRI volume tensor with shape [1, D, H, w]
            - 1 for number of channels (1 for grayscale)
            - D for depth (number of slices)
            - H for height (pixels per slice)
            - W for width (pixels per slice)
        - one_hot_mask: One-hot encoded segmentation mask tensor with shape [C, D, H, W]
            - C for number of classes for one-hot encoding
            - D for depth
            - H for height
            - W for width
        """
        image_file, mask_file = self.pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Load Nifti image and mask
        image_nii = nib.load(image_path)
        mask_nii = nib.load(mask_path)

        image = image_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.int64)

        # Downsample images and masks to the target shape
        image = resize_volume(image)
        mask = resize_mask(mask)

        # Normalise image to zero mean and unit variance
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)

        # Convert image to tensor and add channel dimension
        image = torch.tensor(image).unsqueeze(0)

        # Convert mask to tensor and one-hot encode
        mask = torch.tensor(mask, dtype=torch.long) # [1, D, H, W]
        one_hot = torch.nn.functional.one_hot(mask, num_classes=self.num_classes)
        # Permute dimentions from [D, H, W, C] to [C, D, H, W] since PyTorch expects 
        # C first for 3D conv layers
        one_hot = one_hot.permute(3, 0, 1, 2).float()

        # Apply transformation if given
        if self.transform:
            image, one_hot = self.transform(image, one_hot)

        return image, one_hot

# Random rotation augmentation for 3D images
def random_rotate(image, mask):
    """
    Rotate a 3D image and its corresponding mask by a random multiple of 90 
    degrees along a randomly chosen plane.

    Parameters:
    image: 3D image tensor of shape [C, D, H, W]
    mask: One-hot encoded mask tensort of shape [C, D, H, W]

    Returns:
    tuple: Rotated image and mask tensors
    """
    # Randomly choose a number of 90 degree rotations (0, 90, 180, 270)
    k = random.choice([0, 1, 2, 3])

    # Randomly choose a plane to rotate along
    # (2, 3) = height-width plane, (1, 3) = depth-width plane, 
    # (1, 2) = depth-height plane
    axis = random.choice([(2,3), (1,3), (1,2)])

    # Apply the rotation to both image and mask
    image = torch.rot90(image, k, dims=axis)
    mask = torch.rot90(mask, k, dims=axis)
    return image, mask

class Random3DTransform:
    """
    Callable class for applying random 3D transformations to an image and mask pair.
    Has a 50% change of applying a random rotation.

    Parameters:
    image: 3D image tensor of shape [C, D, H, W]
    mask: One-hot encoded mask tensort of shape [C, D, H, W]

    Returns: 
    tuple: Transformed image and mask tensors
    """
    def __call__(self, image, mask):
        if random.random() > 0.5:
            image, mask = random_rotate(image, mask)
        return image, mask