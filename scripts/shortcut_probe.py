"""
shortcut_probe.py — Detect metadata-based shortcuts in the dataset.

Trains logistic regression on metadata-only features to predict is_forged.
If AUC > 0.65, the data is biased.

Usage:
    python scripts/shortcut_probe.py \
        --train data/manifests/train.csv \
        --val data/manifests/val.csv
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


def extract_metadata_features(df: pd.DataFrame) -> np.ndarray:
    """Extract metadata-only features (no pixel content).

    Features: width, height, aspect_ratio, file_format_encoded
    """
    features = []

    # Width, height
    if "width" in df.columns and "height" in df.columns:
        features.append(df["width"].fillna(0).values.reshape(-1, 1))
        features.append(df["height"].fillna(0).values.reshape(-1, 1))
        # Aspect ratio
        w = df["width"].fillna(1).values.astype(float)
        h = df["height"].fillna(1).values.astype(float)
        features.append((w / np.maximum(h, 1)).reshape(-1, 1))
    else:
        # Use dummy values if not available
        features.append(np.zeros((len(df), 1)))
        features.append(np.zeros((len(df), 1)))
        features.append(np.ones((len(df), 1)))

    # File format encoding
    if "file_format" in df.columns:
        le = LabelEncoder()
        encoded = le.fit_transform(df["file_format"].fillna("unknown").astype(str))
        features.append(encoded.reshape(-1, 1))
    else:
        features.append(np.zeros((len(df), 1)))

    return np.hstack(features)


def run_shortcut_probe(train_path: str, val_path: str) -> float:
    """Train logistic regression on metadata and return val AUC."""
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    X_train = extract_metadata_features(train_df)
    y_train = train_df["is_forged"].values

    X_val = extract_metadata_features(val_df)
    y_val = val_df["is_forged"].values

    # Handle case where features are all constant
    if X_train.std(axis=0).max() == 0:
        print("[WARNING] All metadata features are constant. AUC = 0.5 (random).")
        return 0.5

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_val)[:, 1]

    if len(np.unique(y_val)) < 2:
        print("[WARNING] Only one class in validation. Cannot compute AUC.")
        return 0.5

    auc = roc_auc_score(y_val, y_pred)
    return auc


def main():
    parser = argparse.ArgumentParser(description="Shortcut probe")
    parser.add_argument("--train", type=str, required=True, help="Train manifest CSV")
    parser.add_argument("--val", type=str, required=True, help="Val manifest CSV")
    args = parser.parse_args()

    auc = run_shortcut_probe(args.train, args.val)
    print(f"\n{'='*50}")
    print(f"  Shortcut Probe AUC: {auc:.4f}")
    print(f"{'='*50}")

    if auc > 0.65:
        print("\n[FAIL] AUC > 0.65 → metadata shortcuts detected!")
        print("  Fix: re-encode authentic images to match forged pipeline,")
        print("  match resolution distributions, and re-run.")
        sys.exit(1)
    else:
        print("\n[PASS] AUC <= 0.65 → no significant metadata shortcuts.")


if __name__ == "__main__":
    main()
