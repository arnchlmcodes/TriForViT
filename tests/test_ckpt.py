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

    def test_end_of_training_load_best_and_save(self):
        """Verify the exact end-of-training call path used in train.py:
        load_checkpoint(str(best_pt), model, device=str(device)) followed by model.save().
        """
        from pathlib import Path
        from src.ftt_v2.utils.ckpt import save_checkpoint, load_checkpoint

        cfg = {
            "model": {"name": "ftt_v2", "image_size": 512},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": False},
            "decoder": {"norm": "gn"},
            "det": {"enabled": True, "use_mask_stats": True},
            "type_head": {"enabled": True},
            "cm": {"enabled": False},
        }
        model = ForensicModel(cfg)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
        device = torch.device("cpu")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            best_pt = output_dir / "best.pt"
            save_checkpoint(
                str(best_pt),
                model, optimizer, None, None,
                epoch=0, best_score=0.85, config=cfg,
            )
            assert best_pt.exists()

            # Exact call shape used at end of train.py
            ckpt_info = load_checkpoint(str(best_pt), model, device=str(device))
            assert ckpt_info["best_score"] == 0.85

            model_dir = output_dir / "model"
            model.save(str(model_dir))
            assert (model_dir / "weights.pt").exists()
            assert (model_dir / "config.json").exists()

            # Load exported model back and ensure strict load passes
            loaded = ForensicModel.load(str(model_dir))
            assert loaded is not None

    def test_load_checkpoint_strict_mismatch_raises(self):
        """Verify load_checkpoint raises RuntimeError on missing trainable keys."""
        from src.ftt_v2.utils.ckpt import save_checkpoint, load_checkpoint

        cfg = {
            "model": {"name": "ftt_v2", "image_size": 512},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": False},
            "decoder": {"norm": "gn"},
            "det": {"enabled": True},
            "type_head": {"enabled": True},
            "cm": {"enabled": False},
        }
        model = ForensicModel(cfg)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad_ckpt.pt")
            save_checkpoint(path, model, optimizer, None, None, epoch=1, best_score=0.5, config=cfg)

            # Corrupt state dict: delete a trainable head key
            ckpt = torch.load(path, weights_only=False)
            keys_to_delete = [k for k in ckpt["model_state_dict"] if "det_head" in k]
            for k in keys_to_delete:
                del ckpt["model_state_dict"][k]
            torch.save(ckpt, path)

            # Attempting to load onto model should raise RuntimeError
            with pytest.raises(RuntimeError, match="Checkpoint mismatch in load_checkpoint"):
                load_checkpoint(path, model)

    def test_load_checkpoint_restore_rng_behavior(self):
        """Verify load_checkpoint handles restore_rng flag and CPU conversion for RNG tensors."""
        from src.ftt_v2.utils.ckpt import save_checkpoint, load_checkpoint

        cfg = {
            "model": {"name": "ftt_v2", "image_size": 512},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": False},
            "decoder": {"norm": "gn"},
            "det": {"enabled": True},
            "type_head": {"enabled": True},
            "cm": {"enabled": False},
        }
        model = ForensicModel(cfg)
        optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rng_ckpt.pt")
            save_checkpoint(path, model, optimizer, None, None, epoch=1, best_score=0.5, config=cfg)

            # Test 1: restore_rng=False does not alter current RNG state
            initial_torch_state = torch.random.get_rng_state()
            load_checkpoint(path, model, restore_rng=False)
            assert torch.equal(torch.random.get_rng_state(), initial_torch_state)

            # Test 2: restore_rng=True with tensor rng state succeeds and ensures CPU tensor passed
            ckpt = torch.load(path, weights_only=False)
            # Ensure rng tensor is present
            ckpt["rng_state"]["torch"] = torch.random.get_rng_state()
            torch.save(ckpt, path)
            info = load_checkpoint(path, model, restore_rng=True)
            assert info["epoch"] == 1

