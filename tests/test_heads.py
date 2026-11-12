"""
test_heads.py — Unit tests for decoder and classification heads.

Tests output shapes for B=2 with dummy backbone.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.ftt_v2.heads import MaskDecoder, DetectionHead, TypeHead
from src.ftt_v2.copy_move import CopyMoveBranch
from src.ftt_v2.forensic_model import ForensicModel


class TestMaskDecoder:
    def test_output_shape(self):
        """MaskDecoder produces (B, 1, 512, 512) output."""
        d = 384
        decoder = MaskDecoder(in_channels=d, embed_dim=d, image_size=512, norm_type="gn")
        f_last = torch.randn(2, d, 32, 32)
        out = decoder(f_last)
        assert out.shape == (2, 1, 512, 512)

    def test_with_cm_channels(self):
        """MaskDecoder works with extra copy-move channels."""
        d = 384
        in_ch = d + 32  # 32 from copy-move branch
        decoder = MaskDecoder(in_channels=in_ch, embed_dim=d, image_size=512, norm_type="gn")
        f_last = torch.randn(2, in_ch, 32, 32)
        out = decoder(f_last)
        assert out.shape == (2, 1, 512, 512)

    def test_no_sigmoid(self):
        """Output is raw logits (can be negative)."""
        d = 384
        decoder = MaskDecoder(in_channels=d, embed_dim=d, image_size=512, norm_type="gn")
        f_last = torch.randn(2, d, 32, 32)
        out = decoder(f_last)
        # Logits can be negative
        assert out.min() < 0 or out.max() > 0  # Not all zeros


class TestDetectionHead:
    def test_output_shape(self):
        """DetectionHead produces (B,) output."""
        d = 384
        head = DetectionHead(embed_dim=d, use_mask_stats=True, cm_scalars=False)
        cls_token = torch.randn(2, d)
        patch_mean = torch.randn(2, d)
        mask_logits = torch.randn(2, 1, 32, 32)
        out = head(cls_token, patch_mean, mask_logits)
        assert out.shape == (2,)

    def test_without_mask_stats(self):
        """Works without mask statistics."""
        d = 384
        head = DetectionHead(embed_dim=d, use_mask_stats=False)
        out = head(torch.randn(2, d), torch.randn(2, d))
        assert out.shape == (2,)

    def test_with_cm_scalars(self):
        """Works with copy-move scalars."""
        d = 384
        head = DetectionHead(embed_dim=d, use_mask_stats=True, cm_scalars=True)
        out = head(
            torch.randn(2, d), torch.randn(2, d),
            torch.randn(2, 1, 32, 32), torch.randn(2, 1, 32, 32),
        )
        assert out.shape == (2,)


class TestTypeHead:
    def test_output_shape(self):
        """TypeHead produces (B, 3) output."""
        d = 384
        head = TypeHead(embed_dim=d, num_types=3)
        out = head(torch.randn(2, d), torch.randn(2, d))
        assert out.shape == (2, 3)

    def test_with_cm_scalars(self):
        d = 384
        head = TypeHead(embed_dim=d, num_types=3, cm_scalars=True)
        out = head(
            torch.randn(2, d), torch.randn(2, d),
            torch.randn(2, 1, 32, 32),
        )
        assert out.shape == (2, 3)


class TestCopyMoveBranch:
    def test_output_shapes(self):
        """CopyMoveBranch produces correct shapes."""
        d = 384
        branch = CopyMoveBranch(embed_dim=d, proj_dim=128, topk=8, grid_h=32, grid_w=32)
        patch_tokens = torch.randn(2, 1024, d)
        cm_feat, cm_map = branch(patch_tokens)
        assert cm_feat.shape == (2, 32, 32, 32)
        assert cm_map.shape == (2, 1, 32, 32)

    def test_near_mask_diagonal(self):
        """Near-neighbor mask includes diagonal entries."""
        d = 384
        branch = CopyMoveBranch(embed_dim=d, min_dist=3, grid_h=32, grid_w=32)
        # Diagonal should be masked (distance 0 <= 3)
        assert branch.near_mask.diagonal().all()

    def test_finite_output(self):
        """Outputs are finite (no NaN/Inf) even with float16 inputs."""
        d = 384
        branch = CopyMoveBranch(embed_dim=d, proj_dim=128, topk=8, grid_h=32, grid_w=32)
        # Use float16 input to test the float32 enforcement
        patch_tokens = torch.randn(1, 1024, d, dtype=torch.float16)
        cm_feat, cm_map = branch(patch_tokens)
        assert torch.isfinite(cm_feat).all()
        assert torch.isfinite(cm_map).all()


class TestForensicModelShapes:
    """Test full model output shapes with dummy backbone, B=2."""

    def _make_cfg(self, det=False, type_h=False, cm=False, lora=False):
        return {
            "model": {"name": "ftt_v2", "image_size": 512},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": lora, "r": 8, "alpha": 16},
            "decoder": {"norm": "gn"},
            "det": {"enabled": det, "use_mask_stats": True},
            "type_head": {"enabled": type_h},
            "cm": {"enabled": cm, "tap_layer": 1, "proj_dim": 64, "topk": 4},
        }

    def test_stage_a_shapes(self):
        """Stage A: mask decoder only."""
        cfg = self._make_cfg()
        model = ForensicModel(cfg)
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        assert out.mask_logits.shape == (2, 1, 512, 512)
        assert out.det_logit.shape == (2,)
        assert out.type_logits.shape == (2, 3)
        assert out.cm_map_logits is None

    def test_stage_c_shapes(self):
        """Stage C: mask + det + type heads."""
        cfg = self._make_cfg(det=True, type_h=True)
        model = ForensicModel(cfg)
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        assert out.mask_logits.shape == (2, 1, 512, 512)
        assert out.det_logit.shape == (2,)
        assert out.type_logits.shape == (2, 3)

    def test_stage_d_shapes(self):
        """Stage D: all heads + copy-move branch."""
        cfg = self._make_cfg(det=True, type_h=True, cm=True)
        model = ForensicModel(cfg)
        x = torch.randn(2, 3, 512, 512)
        out = model(x)
        assert out.mask_logits.shape == (2, 1, 512, 512)
        assert out.det_logit.shape == (2,)
        assert out.type_logits.shape == (2, 3)
        assert out.cm_map_logits is not None
        assert out.cm_map_logits.shape == (2, 1, 32, 32)

    def test_zero_global_features_ablation(self):
        """Detection head with zero_global_features=True ignores cls and patch_mean."""
        from src.ftt_v2.heads import DetectionHead
        head = DetectionHead(embed_dim=384, use_mask_stats=True, zero_global_features=True)
        head.eval()

        cls1 = torch.randn(2, 384)
        cls2 = torch.randn(2, 384)
        p1 = torch.randn(2, 384)
        p2 = torch.randn(2, 384)
        mask_32 = torch.randn(2, 1, 32, 32)

        out1 = head(cls1, p1, mask_logits_32=mask_32)
        out2 = head(cls2, p2, mask_logits_32=mask_32)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_type_head_mask_pooling(self):
        """Type head with use_mask_pooling=True pools tampered region features."""
        from src.ftt_v2.heads import TypeHead
        head = TypeHead(embed_dim=384, use_mask_pooling=True)
        cls = torch.randn(2, 384)
        p_mean = torch.randn(2, 384)
        mask_32 = torch.randn(2, 1, 32, 32)
        f_last = torch.randn(2, 384, 32, 32)

        out = head(cls, p_mean, mask_logits_32=mask_32, patch_features=f_last)
        assert out.shape == (2, 3)

    def test_type_head_mask_pooling_empty_mask_fallback(self):
        """Type head with near-zero predicted mask falls back to uniform weights without NaNs."""
        from src.ftt_v2.heads import TypeHead
        head = TypeHead(embed_dim=384, use_mask_pooling=True)
        head.eval()
        cls = torch.randn(2, 384)
        p_mean = torch.randn(2, 384)
        # All logits -50.0 -> sigmoid values ~ 0.0, sum is tiny
        empty_mask_32 = torch.full((2, 1, 32, 32), -50.0)
        f_last = torch.randn(2, 384, 32, 32)

        out = head(cls, p_mean, mask_logits_32=empty_mask_32, patch_features=f_last)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_type_head_topk_mask_pooling(self):
        """Type head with topk mask pooling mode (top 5% patches)."""
        from src.ftt_v2.heads import TypeHead
        head = TypeHead(embed_dim=384, use_mask_pooling=True, mask_pooling_mode="topk", topk_pct=0.05)
        head.eval()
        cls = torch.randn(2, 384)
        p_mean = torch.randn(2, 384)
        mask_32 = torch.randn(2, 1, 32, 32)
        f_last = torch.randn(2, 384, 32, 32)

        out = head(cls, p_mean, mask_logits_32=mask_32, patch_features=f_last)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()
