"""
test_ckpt.py — Unit tests for checkpoint save/load and model save/load.

Tests: save -> load -> identical outputs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import pytest
import torch

from src.ftt_v2.forensic_model import ForensicModel


class TestModelSaveLoad:
    """Test ForensicModel.save() and ForensicModel.load()."""

    def _make_cfg(self):
        return {
            "model": {"name": "ftt_v2", "image_size": 512},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": False},
            "decoder": {"norm": "gn"},
            "det": {"enabled": True, "use_mask_stats": True},
            "type_head": {"enabled": True},
            "cm": {"enabled": False},
        }

    def test_save_load_identical_outputs(self):
        """Model outputs are identical after save -> load."""
        cfg = self._make_cfg()
        model1 = ForensicModel(cfg)
        model1.eval()

        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out1 = model1(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            model1.save(tmpdir)
            model2 = ForensicModel.load(tmpdir)
            model2.eval()

            with torch.no_grad():
                out2 = model2(x)

        assert torch.allclose(out1.mask_logits, out2.mask_logits, atol=1e-5)
        assert torch.allclose(out1.det_logit, out2.det_logit, atol=1e-5)
        assert torch.allclose(out1.type_logits, out2.type_logits, atol=1e-5)

    def test_config_json_exists(self):
        """Save creates config.json."""
        cfg = self._make_cfg()
        model = ForensicModel(cfg)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "config.json"))
            assert os.path.exists(os.path.join(tmpdir, "weights.pt"))

    def test_eval_deterministic(self):
        """Two eval passes give identical outputs (no dropout/BN noise)."""
        cfg = self._make_cfg()
        model = ForensicModel(cfg)
        model.eval()

        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1.mask_logits, out2.mask_logits)
        assert torch.allclose(out1.det_logit, out2.det_logit)
        assert torch.allclose(out1.type_logits, out2.type_logits)


class TestCheckpointUtils:
    """Test checkpoint save/load utilities."""

    def test_save_load_checkpoint(self):
        """Full checkpoint save and restore."""
        from src.ftt_v2.utils.ckpt import save_checkpoint, load_checkpoint

        cfg = {
            "model": {"name": "ftt_v2", "image_size": 512},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": False},
            "decoder": {"norm": "gn"},
            "det": {"enabled": False},
            "type_head": {"enabled": False},
            "cm": {"enabled": False},
        }
        model = ForensicModel(cfg)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pt")
            save_checkpoint(path, model, optimizer, None, None, epoch=5, best_score=0.8, config=cfg)

            model2 = ForensicModel(cfg)
            opt2 = torch.optim.AdamW(model2.trainable_parameters(), lr=1e-3)
            info = load_checkpoint(path, model2, opt2)

            assert info["epoch"] == 5
            assert info["best_score"] == 0.8
