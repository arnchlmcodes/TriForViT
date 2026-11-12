"""
manifest.py — Manifest schema and validation for FTT v2.

Every dataset split is stored as a CSV manifest with standardized columns:
  image_id, path, mask_path, is_forged, copy_move, splicing, inpainting,
  type_valid, source_id, dataset, width, height, file_format
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

MANIFEST_COLUMNS = [
    "image_id",
    "path",
    "mask_path",
    "is_forged",
    "copy_move",
    "splicing",
    "inpainting",
    "type_valid",
    "source_id",
    "dataset",
    "width",
    "height",
    "file_format",
]


def validate_manifest(df: pd.DataFrame) -> List[str]:
    """Validate that a DataFrame conforms to the FTT v2 manifest spec.

    Returns a list of error strings (empty if valid).
    """
    errors = []

    # Check required columns
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if not errors:
        # Check binary labels
        for col in ["is_forged", "copy_move", "splicing", "inpainting", "type_valid"]:
            invalid = ~df[col].isin([0, 1])
            if invalid.any():
                errors.append(f"Column '{col}' contains non-binary values in {invalid.sum()} rows")

        # Check authentic rows consistency
        auth = df[df["is_forged"] == 0]
        if len(auth) > 0:
            if (auth["copy_move"] != 0).any() or (auth["splicing"] != 0).any() or (auth["inpainting"] != 0).any():
                errors.append("Authentic images (is_forged=0) must have copy_move=0, splicing=0, inpainting=0")

    return errors


def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Save a DataFrame as a standardized manifest CSV."""
    errors = validate_manifest(df)
    if errors:
        raise ValueError(f"Manifest validation failed:\n" + "\n".join(errors))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[MANIFEST_COLUMNS].to_csv(path, index=False)


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load and validate a manifest CSV."""
    df = pd.read_csv(path)
    # Fill empty mask_path with empty string for authentic images
    if "mask_path" in df.columns:
        df["mask_path"] = df["mask_path"].fillna("")
    return df
