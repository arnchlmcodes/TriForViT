"""
data_loader.py - Dataset & DataLoader for Image Forgery Detection

Supports loading images from the following datasets:
  - CASIA v2.0   (copy-move, splicing)
  - IMD2020      (inpainting, removal, mixed)
  - Columbia     (splicing — high-quality uncompressed)
  - COVERAGE     (copy-move — realistic scenarios)

Augmentation pipeline includes post-processing robustness transforms
as described in the presentation (JPEG recompression, Gaussian blur,
noise injection, rescaling).
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Forgery label mapping
# ---------------------------------------------------------------------------

FORGERY_TYPES = {
    "authentic": 0,
    "copy_move": 1,
    "splicing": 2,
    "inpainting": 3,
}

BINARY_LABELS = {
    "authentic": 0,
    "tampered": 1,
}


# ---------------------------------------------------------------------------
# Augmentation / robustness transforms
# ---------------------------------------------------------------------------

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Training transforms with post-processing robustness augmentations.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ], p=0.3),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Evaluation transforms (deterministic).
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset Wrapper for Splits
# ---------------------------------------------------------------------------

class TransformSubset(Dataset):
    """Wraps a dataset subset and applies a specific transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        image, binary_label, forgery_label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, binary_label, forgery_label

    def __len__(self):
        return len(self.subset)


# ---------------------------------------------------------------------------
# Generic Forgery Dataset
# ---------------------------------------------------------------------------

class ForgeryDataset(Dataset):
    """
    Generic dataset class for image forgery detection.

    Expected directory layout (example for CASIA):
        data/raw/CASIA_v2/
            authentic/
                Au_xxx.jpg
            tampered/
                Tp_xxx.jpg

    You will need to adapt the label-parsing logic per dataset once
    the raw data is downloaded and organised.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        binary_only: bool = True,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.binary_only = binary_only

        self.samples = []  # list of (image_path, binary_label, forgery_type_label)
        self._scan_directory()

    def _scan_directory(self):
        """Scan root_dir for images and infer labels from folder names."""
        for label_folder in sorted(self.root_dir.iterdir()):
            if not label_folder.is_dir():
                continue
            folder_name = label_folder.name.lower()

            # Ignore ground truth mask folders
            if "groundtruth" in folder_name:
                continue

            # Determine labels from folder name
            if "authentic" in folder_name or "au" in folder_name:
                binary_label = BINARY_LABELS["authentic"]
                forgery_label = FORGERY_TYPES["authentic"]
            else:
                binary_label = BINARY_LABELS["tampered"]
                # Default to splicing; override per-dataset as needed
                if "copy" in folder_name or "move" in folder_name:
                    forgery_label = FORGERY_TYPES["copy_move"]
                elif "inpaint" in folder_name:
                    forgery_label = FORGERY_TYPES["inpainting"]
                else:
                    forgery_label = FORGERY_TYPES["splicing"]

            for img_path in label_folder.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".bmp"):
                    self.samples.append((str(img_path), binary_label, forgery_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, binary_label, forgery_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, binary_label, forgery_label


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    root_dir: str,
    batch_size: int = 32,
    is_train: bool = True,
    img_size: int = 224,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for the given dataset directory.

    Args:
        root_dir: Path to the dataset folder (e.g. 'data/raw/CASIA_v2').
        batch_size: Batch size.
        is_train: If True, apply training augmentations.
        img_size: Resize images to this resolution.
        num_workers: Number of data-loading workers.
    """
    tfm = get_train_transforms(img_size) if is_train else get_eval_transforms(img_size)
    dataset = ForgeryDataset(root_dir, transform=tfm)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        # Keeps worker processes alive between epochs instead of tearing
        # them down and re-spawning each time — meaningful savings when
        # there are many short epochs. No-op / ignored when num_workers=0.
        persistent_workers=(num_workers > 0),
    )

def create_split_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1)
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates Train, Validation, and Test DataLoaders from a dataset directory.
    Automatically applies training transforms to Train, and eval transforms to Val/Test.
    """
    # 1. Load full dataset without transforms (returns raw PIL images)
    full_dataset = ForgeryDataset(root_dir, transform=None)

    # 2. Split dataset
    total_size = len(full_dataset)
    train_size = int(split_ratio[0] * total_size)
    val_size = int(split_ratio[1] * total_size)
    test_size = total_size - train_size - val_size

    # Use a fixed generator for reproducible splits
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # 3. Apply appropriate transforms via TransformSubset wrapper
    train_dataset = TransformSubset(train_subset, transform=get_train_transforms(img_size))
    val_dataset = TransformSubset(val_subset, transform=get_eval_transforms(img_size))
    test_dataset = TransformSubset(test_subset, transform=get_eval_transforms(img_size))

    # 4. Create DataLoaders
    persistent = (num_workers > 0)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=persistent
    )

    return train_loader, val_loader, test_loader