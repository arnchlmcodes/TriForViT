"""
test_leakage.py — Pytest tests for data leakage checks.

Verifies that check_leakage correctly identifies clean and leaked splits.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import pandas as pd
import pytest

from scripts.check_leakage import check_leakage


class TestLeakageCheck:
    """Test source_id leakage detection between splits."""

    def test_clean_splits(self):
        """Disjoint source_ids return True (clean)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_train = pd.DataFrame({
                "image_id": ["t1", "t2"],
                "source_id": ["src_1", "src_2"],
                "is_forged": [1, 0],
                "copy_move": [1, 0],
                "splicing": [0, 0],
                "inpainting": [0, 0],
            })
            df_val = pd.DataFrame({
                "image_id": ["v1", "v2"],
                "source_id": ["src_3", "src_4"],
                "is_forged": [1, 0],
                "copy_move": [0, 0],
                "splicing": [1, 0],
                "inpainting": [0, 0],
            })
            df_train.to_csv(os.path.join(tmpdir, "train.csv"), index=False)
            df_val.to_csv(os.path.join(tmpdir, "val.csv"), index=False)

            assert check_leakage(tmpdir) is True

    def test_leaked_splits(self):
        """Overlapping source_ids return False (leakage detected)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            df_train = pd.DataFrame({
                "image_id": ["t1", "t2"],
                "source_id": ["src_1", "src_2"],
                "is_forged": [1, 0],
                "copy_move": [1, 0],
                "splicing": [0, 0],
                "inpainting": [0, 0],
            })
            df_val = pd.DataFrame({
                "image_id": ["v1", "v2"],
                "source_id": ["src_2", "src_4"],  # src_2 leaks!
                "is_forged": [1, 0],
                "copy_move": [0, 0],
                "splicing": [1, 0],
                "inpainting": [0, 0],
            })
            df_train.to_csv(os.path.join(tmpdir, "train.csv"), index=False)
            df_val.to_csv(os.path.join(tmpdir, "val.csv"), index=False)

            assert check_leakage(tmpdir) is False
