"""
test_backbone.py — Unit tests for the FTT v2 backbone.

Tests run on CPU with the dummy backbone (no HF token needed).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.ftt_v2.backbone import DummyViTBackbone, load_backbone, split_tokens


class TestSplitTokens:
    """Tests for the split_tokens helper."""

    def test_basic_shapes(self):
        """split_tokens returns correct shapes."""
        B, d = 2, 384
        n_h, n_w = 32, 32
        n_prefix = 1  # CLS only
        N = n_h * n_w
        S = n_prefix + N

        hidden = torch.randn(B, S, d)
        cls, patches = split_tokens(hidden, n_h, n_w, n_prefix)

        assert cls.shape == (B, d)
        assert patches.shape == (B, d, n_h, n_w)

    def test_patch_count(self):
        """Patch count == 1024 for 512x512 with patch_size=16."""
        B, d = 1, 384
        n_h, n_w = 32, 32
        n_prefix = 1
        N = n_h * n_w

        assert N == 1024

        hidden = torch.randn(B, n_prefix + N, d)
        cls, patches = split_tokens(hidden, n_h, n_w, n_prefix)
        assert patches.shape[2] * patches.shape[3] == 1024

    def test_patch_spatial_ordering(self):
        """Patch (r, c) sits at index r*32+c after reshape."""
        B, d = 1, 384
        n_h, n_w = 32, 32
        n_prefix = 1
        N = n_h * n_w

        hidden = torch.randn(B, n_prefix + N, d)
        # Set a specific patch
        r, c = 5, 10
        idx = r * n_w + c
        hidden[:, n_prefix + idx, :] = 42.0

        cls, patches = split_tokens(hidden, n_h, n_w, n_prefix)
        assert torch.allclose(patches[0, :, r, c], torch.tensor(42.0))

    def test_with_register_tokens(self):
        """Works with n_prefix > 1 (CLS + register tokens)."""
        B, d = 2, 384
        n_h, n_w = 32, 32
        n_register = 4
        n_prefix = 1 + n_register
        N = n_h * n_w
        S = n_prefix + N

        hidden = torch.randn(B, S, d)
        cls, patches = split_tokens(hidden, n_h, n_w, n_prefix)

        assert cls.shape == (B, d)
        assert patches.shape == (B, d, n_h, n_w)

    def test_wrong_count_raises(self):
        """Mismatched patch count raises assertion."""
        B, d = 1, 384
        hidden = torch.randn(B, 10, d)  # Too few tokens
        with pytest.raises(AssertionError):
            split_tokens(hidden, 32, 32, 1)


class TestDummyBackbone:
    """Tests for the DummyViTBackbone."""

    def test_output_shapes(self):
        """Dummy backbone produces correct output shapes."""
        model = DummyViTBackbone(embed_dim=384, depth=12, patch_size=16)
        x = torch.randn(2, 3, 512, 512)
        out = model(x, output_hidden_states=True)

        assert "last_hidden_state" in out
        B, S, d = out["last_hidden_state"].shape
        assert B == 2
        assert d == 384
        n_prefix = model.n_prefix
        expected_patches = (512 // 16) ** 2
        assert S == n_prefix + expected_patches

    def test_hidden_states(self):
        """output_hidden_states returns a tuple of states."""
        model = DummyViTBackbone(embed_dim=384, depth=12, patch_size=16)
        x = torch.randn(1, 3, 512, 512)
        out = model(x, output_hidden_states=True)
        assert "hidden_states" in out
        assert isinstance(out["hidden_states"], tuple)
        assert len(out["hidden_states"]) >= 2

    def test_frozen(self):
        """All dummy backbone parameters are frozen."""
        model = DummyViTBackbone()
        for p in model.parameters():
            assert not p.requires_grad

    def test_config_dict(self):
        """config_dict returns expected keys."""
        model = DummyViTBackbone(embed_dim=384, depth=12)
        cfg = model.config_dict
        assert cfg["hidden_size"] == 384
        assert cfg["num_hidden_layers"] == 12


class TestLoadBackbone:
    """Tests for the load_backbone dispatcher."""

    def test_dummy(self):
        model, info = load_backbone("dummy")
        assert info["hidden_size"] == 384
        assert info["n_prefix"] >= 1

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            load_backbone("unknown_model")

    def test_custom_embed_dim(self):
        model, info = load_backbone("dummy", embed_dim=768, depth=12)
        assert info["hidden_size"] == 768

    def test_missing_backbone_name_raises(self):
        from src.ftt_v2.forensic_model import ForensicModel
        cfg = {"model": {"name": "ftt_v2", "image_size": 128}, "backbone": {}}
        with pytest.raises(ValueError, match="Missing 'backbone.name' in configuration"):
            ForensicModel(cfg)

    @pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="Requires HF_TOKEN environment variable")
    def test_real_dinov3_vitb16(self):
        """Integration test for real DINOv3 ViT-B backbone when HF_TOKEN is provided."""
        from src.ftt_v2.lora import apply_lora, find_qkv_target_modules

        model, info = load_backbone("dinov3_vitb16")
        assert info["hidden_size"] == 768
        assert info["num_hidden_layers"] == 12
        assert info["n_prefix"] >= 1

        targets = find_qkv_target_modules(model)
        assert len(targets) > 0

        lora_model = apply_lora(model, r=64, alpha=128, target_modules=targets)
        backbone_total = sum(p.numel() for p in lora_model.parameters())
        trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
        ratio = trainable / backbone_total
        # ViT-B should have ~4% trainable parameters (well under 10%)
        assert ratio < 0.10
        assert ratio > 0.01

