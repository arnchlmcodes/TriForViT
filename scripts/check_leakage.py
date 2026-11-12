"""
check_leakage.py — Verify source-disjoint splits have no leakage.

Asserts that source_id intersection between train, val, and test is empty.
Prints counts per split by type.

Usage:
    python scripts/check_leakage.py --manifest-dir data/manifests
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd

from src.ftt_v2 import TYPES


def check_leakage(manifest_dir: str) -> bool:
    """Check for source ID leakage between splits.

    Returns True if no leakage found.
    """
    manifest_dir = Path(manifest_dir)
    splits = {}

    for name in ["train", "val", "test_A"]:
        path = manifest_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            splits[name] = df

    if len(splits) < 2:
        print(f"[WARNING] Found only {len(splits)} splits. Need at least 2 to check leakage.")
        return True

    # Check pairwise source_id intersection
    split_names = list(splits.keys())
    all_clean = True

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name_i, name_j = split_names[i], split_names[j]
            ids_i = set(splits[name_i]["source_id"].dropna().astype(str))
            ids_j = set(splits[name_j]["source_id"].dropna().astype(str))
            overlap = ids_i & ids_j

            if overlap:
                print(f"[LEAKAGE] {len(overlap)} shared source_ids between {name_i} and {name_j}")
                print(f"  Examples: {list(overlap)[:5]}")
                all_clean = False
            else:
                print(f"[OK] No overlap between {name_i} and {name_j}")

    # Print counts per split by type
    print(f"\n{'='*60}")
    print(f"{'Split':<10} {'Total':>8} {'Auth':>8} {'CM':>8} {'SP':>8} {'IP':>8}")
    print(f"{'='*60}")
    for name, df in splits.items():
        total = len(df)
        auth = (df["is_forged"] == 0).sum()
        cm = (df["copy_move"] == 1).sum()
        sp = (df["splicing"] == 1).sum()
        ip = (df["inpainting"] == 1).sum()
        print(f"{name:<10} {total:>8} {auth:>8} {cm:>8} {sp:>8} {ip:>8}")
    print(f"{'='*60}")

    return all_clean


def main():
    parser = argparse.ArgumentParser(description="Check data leakage")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests")
    args = parser.parse_args()

    clean = check_leakage(args.manifest_dir)
    if not clean:
        print("\n[FAIL] Source leakage detected! Fix splits before training.")
        sys.exit(1)
    else:
        print("\n[PASS] No source leakage detected.")


if __name__ == "__main__":
    main()
