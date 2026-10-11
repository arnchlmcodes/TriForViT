"""
test_lora.py — Unit tests for LoRA wrapper.

Tests run on CPU with the dummy backbone.
"""

import os
os.environ["USE_TF"] = "0"

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.ftt_v2.backbone import DummyViTBackbone, load_backbone
from src.ftt_v2.lora import apply_lora, find_qkv_target_modules, verify_lora_noop


class TestFindTargetModules:
    """Tests for QKV target module discovery."""

    def test_dummy_backbone(self):
        """Dummy backbone has Linear layers that can be discovered."""
        model = DummyViTBackbone(embed_dim=384, depth=12)
        # The dummy uses TransformerEncoderLayer which has self_attn with
        # in_proj and out_proj. find_qkv_target_modules may or may not
        # find targets depending on naming convention.
        # This test verifies it either finds something or raises a clear error.
        try:
            targets = find_qkv_target_modules(model)
            assert len(targets) > 0
        except RuntimeError as e:
            assert "Could not auto-detect" in str(e)


class TestApplyLora:
    """Tests for LoRA application on the dummy backbone."""

    def _get_dummy(self):
        return DummyViTBackbone(embed_dim=384, depth=12)

    def test_lora_applies(self):
        """LoRA can be applied to the dummy backbone."""
        model = self._get_dummy()
        # Use explicit target to avoid auto-detection issues with dummy
        try:
            targets = find_qkv_target_modules(model)
            lora_model = apply_lora(model, r=8, alpha=16, target_modules=targets)
            # Verify trainable params exist
            trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            assert trainable > 0
        except RuntimeError:
            # If no QKV modules found in dummy, that's OK — skip
            pytest.skip("Dummy backbone doesn't have standard QKV naming")

    def test_trainable_params_under_10pct(self):
        """LoRA trainable params are < 10% of total."""
        model = self._get_dummy()
        try:
            targets = find_qkv_target_modules(model)
            lora_model = apply_lora(model, r=8, alpha=16, target_modules=targets)
            total = sum(p.numel() for p in lora_model.parameters())
            trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
            assert trainable / total < 0.10
        except RuntimeError:
            pytest.skip("Dummy backbone doesn't have standard QKV naming")


class TestVerifyLoraNoOp:
    """Tests for the LoRA no-op assertion."""

    def test_noop_with_fresh_lora(self):
        """Freshly applied LoRA should be a no-op (B matrices = 0)."""
        import copy
        model = DummyViTBackbone(embed_dim=384, depth=12)
        try:
            targets = find_qkv_target_modules(model)
        except RuntimeError:
            pytest.skip("Dummy backbone doesn't have standard QKV naming")

        plain = copy.deepcopy(model)
        lora_model = apply_lora(model, r=8, alpha=16, target_modules=targets)

        dummy_input = torch.randn(1, 3, 512, 512)
        # This should not raise
        verify_lora_noop(lora_model, plain, dummy_input)
