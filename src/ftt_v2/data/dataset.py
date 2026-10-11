"""
dataset.py — Manifest-based dataset for FTT v2.

Reads CSV manifests with columns:
  image_id, path, mask_path, is_forged, copy_move, splicing, inpainting,
  type_valid, source_id, dataset, width, height, file_format

Returns a dict per sample:
  image     (3, 512, 512) float normalized
  mask      (1, 512, 512) float {0, 1}
  is_forged  scalar float
  types     (3,) float
  type_valid (3,) float
  source_id  str
  dataset    str
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .. import TYPES


class ForensicDataset(Dataset):
    """Manifest-based forgery detection dataset with mask loading.

    Parameters
    ----------
    manifest_path : str
        Path to CSV manifest file.
    transform : callable or None
        Transform that accepts (img: np.ndarray, mask: np.ndarray) and
        returns (img_tensor, mask_tensor).
    image_root : str or None
        If provided, ``path`` and ``mask_path`` in the manifest are
        treated as relative to this root.
    """

    def __init__(
        self,
        manifest_path: str,
        transform: Optional[Callable] = None,
        image_root: Optional[str] = None,
    ):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform
        self.image_root = image_root
        self._validate_columns()

    def _validate_columns(self) -> None:
        required = {
            "image_id", "path", "is_forged",
            "copy_move", "splicing", "inpainting", "type_valid",
        }
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Manifest missing columns: {missing}. "
                f"Available: {list(self.df.columns)}"
            )

    def _resolve_path(self, p: str) -> str:
        if self.image_root and not os.path.isabs(p):
            return os.path.join(self.image_root, p)
        return p

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Load image
        img_path = self._resolve_path(str(row["path"]))
        img = np.array(Image.open(img_path).convert("RGB"))

        # Load mask
        mask_path = row.get("mask_path", "")
        if pd.notna(mask_path) and str(mask_path).strip():
            mask_path = self._resolve_path(str(mask_path))
            mask_pil = Image.open(mask_path).convert("L")
            mask = np.array(mask_pil).astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)
        else:
            # Authentic image: zero mask
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        # Apply transform
        if self.transform:
            img_tensor, mask_tensor = self.transform(img, mask)
        else:
            # Default: just convert to tensor
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # Labels
        is_forged = float(row["is_forged"])
        types = torch.tensor([
            float(row["copy_move"]),
            float(row["splicing"]),
            float(row["inpainting"]),
        ], dtype=torch.float32)

        # type_valid can be a single value (applied to all) or per-type
        tv = row["type_valid"]
        type_valid = torch.tensor([float(tv)] * len(TYPES), dtype=torch.float32)

        source_id = str(row.get("source_id", ""))
        dataset_name = str(row.get("dataset", ""))

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "is_forged": torch.tensor(is_forged, dtype=torch.float32),
            "types": types,
            "type_valid": type_valid,
            "source_id": source_id,
            "dataset": dataset_name,
        }


class BalancedForgeryTypeSampler(Sampler):
    """Class-balanced sampler ensuring roughly equal proportion of
    authentic and each forgery type per epoch.

    Oversamples minority classes to match the majority class count.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        self.rng = np.random.RandomState(seed)

        # Group indices
        self.authentic_idx = df.index[df["is_forged"] == 0].tolist()
        self.cm_idx = df.index[df["copy_move"] == 1].tolist()
        self.sp_idx = df.index[df["splicing"] == 1].tolist()
        self.ip_idx = df.index[df["inpainting"] == 1].tolist()

        # Target: max group size
        groups = [self.authentic_idx, self.cm_idx, self.sp_idx, self.ip_idx]
        self.target_size = max(len(g) for g in groups) if groups else 0
        self.total = self.target_size * 4

    def __len__(self) -> int:
        return self.total

    def __iter__(self):
        groups = [self.authentic_idx, self.cm_idx, self.sp_idx, self.ip_idx]
        all_indices = []
        for g in groups:
            if len(g) == 0:
                continue
            repeated = list(g) * (self.target_size // max(len(g), 1) + 1)
            self.rng.shuffle(repeated)
            all_indices.extend(repeated[:self.target_size])

        self.rng.shuffle(all_indices)
        return iter(all_indices)
