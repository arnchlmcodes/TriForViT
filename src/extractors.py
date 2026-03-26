"""
extractors.py - Feature Extraction for Three-Stream Tokenization

This module implements the three forensic feature extraction pipelines:
1. Spatial Stream:  Raw pixel patch → linear projection
2. Frequency Stream: FFT magnitude spectrum of each patch
3. Noise Residual Stream: Gaussian high-pass filtered noise residual (SRM-style)

References:
- Fridrich & Kodovsky (2012) - Rich Models for Steganalysis (SRM filters)
- Dosovitskiy et al. (2021) - ViT patch embedding
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frequency-domain feature extraction (Stream 2)
# ---------------------------------------------------------------------------

def compute_fft_magnitude(patch: torch.Tensor) -> torch.Tensor:
    """
    Compute the FFT magnitude spectrum for an image patch.

    Args:
        patch: Tensor of shape (B, C, H, W) — a batch of image patches.

    Returns:
        Magnitude spectrum of shape (B, C, H, W).
    """
    # 2D FFT over spatial dimensions, shift DC component to centre
    fft_result = torch.fft.fft2(patch, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
    magnitude = torch.abs(fft_shifted)
    # Log-scale for numerical stability
    magnitude = torch.log1p(magnitude)
    return magnitude


# ---------------------------------------------------------------------------
# Noise-residual feature extraction (Stream 3)
# ---------------------------------------------------------------------------

def _build_srm_highpass_kernel(ksize: int = 5) -> torch.Tensor:
    """
    Build a simple Gaussian high-pass kernel (approximation of SRM filters).

    Args:
        ksize: Kernel size (must be odd).

    Returns:
        High-pass kernel of shape (1, 1, ksize, ksize).
    """
    sigma = ksize / 3.0
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    gaussian = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()

    # High-pass = identity impulse − low-pass
    identity = torch.zeros(ksize, ksize)
    identity[ksize // 2, ksize // 2] = 1.0
    hp_kernel = identity - gaussian
    return hp_kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)


def compute_noise_residual(
    patch: torch.Tensor, ksize: int = 5
) -> torch.Tensor:
    """
    Extract the noise residual of an image patch using a Gaussian HP filter.

    Args:
        patch: Tensor of shape (B, C, H, W).
        ksize: Kernel size for the high-pass filter.

    Returns:
        Noise residual of shape (B, C, H, W).
    """
    kernel = _build_srm_highpass_kernel(ksize).to(patch.device)
    # Apply per-channel
    channels = []
    for c in range(patch.shape[1]):
        filtered = F.conv2d(
            patch[:, c : c + 1, :, :],
            kernel,
            padding=ksize // 2,
        )
        channels.append(filtered)
    return torch.cat(channels, dim=1)


# ---------------------------------------------------------------------------
# Patch extraction helper
# ---------------------------------------------------------------------------

def extract_patches(image: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Split an image into non-overlapping patches.

    Args:
        image: Tensor of shape (B, C, H, W).
        patch_size: Side length of each square patch (default 16 for ViT-Base).

    Returns:
        Patches of shape (B, N, C, patch_size, patch_size)
        where N = (H // patch_size) * (W // patch_size).
    """
    B, C, H, W = image.shape
    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Image dims ({H}×{W}) must be divisible by patch_size ({patch_size})"
    )
    patches = image.unfold(2, patch_size, patch_size).unfold(
        3, patch_size, patch_size
    )
    # patches shape: (B, C, nH, nW, patch_size, patch_size)
    patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4)  # (B, N, C, pH, pW)
    return patches
