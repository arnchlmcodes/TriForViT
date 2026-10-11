"""
visualize_batch.py — Visualize a batch of samples from the dataset.

Saves 16 samples (image, mask overlay, labels) to outputs/debug/batch_vis.png.

Usage:
    python scripts/visualize_batch.py \
        --manifest data/manifests/train.csv \
        --output outputs/debug/batch_vis.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.ftt_v2.data.dataset import ForensicDataset
from src.ftt_v2.data.transforms import ValTransform
from src.ftt_v2 import TYPES


def denormalize(img_tensor, mean, std):
    """Denormalize a (3, H, W) tensor for visualization."""
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (img_tensor * s + m).clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize batch samples")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/debug/batch_vis.png")
    parser.add_argument("--n", type=int, default=16, help="Number of samples")
    args = parser.parse_args()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = ValTransform(image_size=512, mean=mean, std=std)
    dataset = ForensicDataset(args.manifest, transform=transform)

    n = min(args.n, len(dataset))
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols * 2, figsize=(ncols * 8, nrows * 4))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    indices = np.random.choice(len(dataset), n, replace=False)

    for idx_i, data_idx in enumerate(indices):
        sample = dataset[int(data_idx)]
        row = idx_i // ncols
        col = (idx_i % ncols) * 2

        # Image
        img = denormalize(sample["image"], mean, std).permute(1, 2, 0).numpy()
        axes[row, col].imshow(img)
        is_forged = sample["is_forged"].item()
        type_str = []
        for i, t in enumerate(TYPES):
            if sample["types"][i] > 0.5:
                type_str.append(t)
        label = "FORGED" if is_forged > 0.5 else "AUTH"
        if type_str:
            label += f" ({', '.join(type_str)})"
        axes[row, col].set_title(label, fontsize=9, color="red" if is_forged > 0.5 else "green")
        axes[row, col].axis("off")

        # Mask overlay
        mask = sample["mask"].squeeze(0).numpy()
        overlay = img.copy()
        overlay[:, :, 0] = np.where(mask > 0.5, 1.0, overlay[:, :, 0])
        overlay[:, :, 1] = np.where(mask > 0.5, 0.0, overlay[:, :, 1])
        overlay[:, :, 2] = np.where(mask > 0.5, 0.0, overlay[:, :, 2])
        blended = img * 0.6 + overlay * 0.4
        axes[row, col + 1].imshow(np.clip(blended, 0, 1))
        axes[row, col + 1].set_title("Mask overlay", fontsize=9)
        axes[row, col + 1].axis("off")

    # Hide unused axes
    for i in range(n, nrows * ncols):
        row = i // ncols
        col = (i % ncols) * 2
        axes[row, col].axis("off")
        axes[row, col + 1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved batch visualization to {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
