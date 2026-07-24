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
from torch.utils.data import Dataset, DataLoader
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
        self.transform = transform or get_eval_transforms()
        self.binary_only = binary_only

        self.samples = []  # list of (image_path, binary_label, forgery_type_label)
        self._scan_directory()

    def _scan_directory(self):
        """Scan root_dir for images and infer labels from folder names."""
        for label_folder in sorted(self.root_dir.iterdir()):
            if not label_folder.is_dir():
                continue
            folder_name = label_folder.name.lower()

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

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
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