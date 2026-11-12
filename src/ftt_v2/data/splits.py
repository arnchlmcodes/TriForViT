"""
splits.py — Source-disjoint dataset splitting utilities for FTT v2.

Splits manifests by source_id to prevent data leakage between train,
validation, and test partitions. Also supports perceptual-hash grouping
when explicit source IDs are unavailable.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import pandas as pd


def split_by_source_id(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Split a manifest into train, val, and optional test_A partitions by source_id.

    All samples sharing the same source_id will be placed in the same split.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest containing a ``source_id`` column.
    val_ratio : float
        Fraction of unique source_ids assigned to validation.
    test_ratio : float
        Fraction of unique source_ids assigned to test.
    seed : int
        RNG seed for deterministic splitting.

    Returns
    -------
    dict with keys 'train', 'val', and optionally 'test_A'.
    """
    rng = random.Random(seed)

    # Group unique source IDs
    unique_sources = sorted(list(df["source_id"].dropna().unique()))
    rng.shuffle(unique_sources)

    n_total = len(unique_sources)
    n_val = int(round(n_total * val_ratio))
    n_test = int(round(n_total * test_ratio))

    val_sources = set(unique_sources[:n_val])
    test_sources = set(unique_sources[n_val : n_val + n_test])
    train_sources = set(unique_sources[n_val + n_test :])

    splits = {
        "train": df[df["source_id"].isin(train_sources)].copy().reset_index(drop=True),
        "val": df[df["source_id"].isin(val_sources)].copy().reset_index(drop=True),
    }

    if test_ratio > 0:
        splits["test_A"] = df[df["source_id"].isin(test_sources)].copy().reset_index(drop=True)

    # Verification: assert disjoint
    assert len(val_sources & train_sources) == 0, "Train and Val share source_ids!"
    if test_ratio > 0:
        assert len(val_sources & test_sources) == 0, "Val and Test share source_ids!"
        assert len(train_sources & test_sources) == 0, "Train and Test share source_ids!"

    return splits


def group_by_phash(
    image_paths: List[str],
    max_hamming_dist: int = 6,
) -> Dict[str, str]:
    """Group near-duplicate images by perceptual hash (imagehash.phash).

    Returns a mapping from image_path to a cluster group_id.
    """
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        raise ImportError("imagehash and pillow are required for phash grouping.")

    hashes = {}
    for p in image_paths:
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img)
                hashes[p] = h
        except Exception:
            continue

    # Simple greedy clustering
    clusters: List[Tuple[str, List[str]]] = []  # (cluster_id, [paths])
    path_to_cluster = {}

    for path, h in hashes.items():
        matched = False
        for cluster_id, members in clusters:
            rep_path = members[0]
            rep_hash = hashes[rep_path]
            if (h - rep_hash) <= max_hamming_dist:
                members.append(path)
                path_to_cluster[path] = cluster_id
                matched = True
                break
        if not matched:
            new_id = f"cluster_{len(clusters)}"
            clusters.append((new_id, [path]))
            path_to_cluster[path] = new_id

    return path_to_cluster
