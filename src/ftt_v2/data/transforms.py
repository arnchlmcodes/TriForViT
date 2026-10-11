"""
transforms.py — Augmentation and preprocessing transforms for FTT v2.

Training: geometric + photometric augmentations (same transform for image and mask).
Validation/Test: resize + normalize only.
"""

from __future__ import annotations

import io
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter


def _resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Resize image with bilinear interpolation via PIL."""
    pil = Image.fromarray(img)
    pil = pil.resize((size, size), Image.BILINEAR)
    return np.array(pil)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resize mask with NEAREST interpolation, then binarize."""
    pil = Image.fromarray(mask)
    pil = pil.resize((size, size), Image.NEAREST)
    out = np.array(pil).astype(np.float32)
    out = (out > 0.5).astype(np.float32)
    return out


def _random_horizontal_flip(
    img: np.ndarray, mask: np.ndarray, p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() < p:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    return img, mask


def _random_resized_crop(
    img: np.ndarray, mask: np.ndarray,
    scale: Tuple[float, float] = (0.6, 1.0),
    target_size: int = 512, p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() >= p:
        return img, mask
    h, w = img.shape[:2]
    s = random.uniform(scale[0], scale[1])
    crop_h, crop_w = int(h * s), int(w * s)
    top = random.randint(0, max(h - crop_h, 0))
    left = random.randint(0, max(w - crop_w, 0))
    img = img[top:top + crop_h, left:left + crop_w]
    mask = mask[top:top + crop_h, left:left + crop_w]
    img = _resize_image(img, target_size)
    mask = _resize_mask(mask, target_size)
    return img, mask


def _jpeg_recompression(img: np.ndarray, quality_range: Tuple[int, int] = (50, 100), p: float = 0.3) -> np.ndarray:
    if random.random() >= p:
        return img
    quality = random.randint(quality_range[0], quality_range[1])
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _gaussian_blur(img: np.ndarray, kernels: List[int] = None, p: float = 0.2) -> np.ndarray:
    if random.random() >= p:
        return img
    if kernels is None:
        kernels = [3, 5, 7]
    k = random.choice(kernels)
    pil = Image.fromarray(img)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=k // 2))
    return np.array(pil)


def _gaussian_noise(img: np.ndarray, std_range: Tuple[float, float] = (5.0, 25.0), p: float = 0.2) -> np.ndarray:
    if random.random() >= p:
        return img
    std = random.uniform(std_range[0], std_range[1])
    noise = np.random.randn(*img.shape) * std
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def normalize_tensor(
    img_tensor: torch.Tensor,
    mean: List[float],
    std: List[float],
) -> torch.Tensor:
    """Normalize a (3, H, W) float tensor with mean and std."""
    m = torch.tensor(mean, dtype=img_tensor.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=img_tensor.dtype).view(3, 1, 1)
    return (img_tensor - m) / s


class TrainTransform:
    """Training augmentation pipeline for FTT v2.

    Applies the same geometric transform to image AND mask.
    """

    def __init__(
        self,
        image_size: int = 512,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(
        self, img: np.ndarray, mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        img : (H, W, 3) uint8
        mask : (H, W) float32 in {0, 1}

        Returns
        -------
        img_tensor : (3, image_size, image_size) float32, normalized
        mask_tensor : (1, image_size, image_size) float32 in {0, 1}
        """
        # Resize first
        img = _resize_image(img, self.image_size)
        mask = _resize_mask(mask, self.image_size)

        # Geometric (applied to both)
        img, mask = _random_horizontal_flip(img, mask)
        img, mask = _random_resized_crop(
            img, mask, scale=(0.6, 1.0), target_size=self.image_size, p=0.5,
        )

        # Photometric (image only)
        img = _jpeg_recompression(img, p=0.3)
        img = _gaussian_blur(img, p=0.2)
        img = _gaussian_noise(img, p=0.2)

        # To tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = normalize_tensor(img_tensor, self.mean, self.std)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor


class ValTransform:
    """Validation / test transform: resize + normalize only."""

    def __init__(
        self,
        image_size: int = 512,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ):
        self.image_size = image_size
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

    def __call__(
        self, img: np.ndarray, mask: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img = _resize_image(img, self.image_size)
        mask = _resize_mask(mask, self.image_size)

        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = normalize_tensor(img_tensor, self.mean, self.std)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return img_tensor, mask_tensor
