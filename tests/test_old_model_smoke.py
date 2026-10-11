"""
test_old_model_smoke.py — Smoke test for the existing ThreeStreamViT model.

Ensures the old model pipeline is not broken by FTT v2 additions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch


class TestOldModelSmoke:
    """Smoke test: the existing ThreeStreamViT can be imported and called."""

    def test_import_model(self):
        """ThreeStreamViT can be imported."""
        from model import ThreeStreamViT
        assert ThreeStreamViT is not None

    def test_instantiate(self):
        """ThreeStreamViT can be instantiated."""
        try:
            from model import ThreeStreamViT
            model = ThreeStreamViT(img_size=224, patch_size=16)
            assert model is not None
        except ImportError as e:
            # timm might not be installed — acceptable for smoke test
            if "timm" in str(e):
                pytest.skip("timm not installed")
            raise

    def test_forward_shape(self):
        """ThreeStreamViT forward produces correct output shapes."""
        try:
            from model import ThreeStreamViT
            model = ThreeStreamViT(img_size=224, patch_size=16)
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                det_logits, cls_logits = model(x)
            assert det_logits.shape == (1, 2)
            assert cls_logits.shape == (1, 3)
        except ImportError as e:
            if "timm" in str(e):
                pytest.skip("timm not installed")
            raise
