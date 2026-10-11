"""
test_losses.py — Unit tests for FTT v2 losses.

Tests edge band computation, loss finiteness, and gradient accumulation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from src.ftt_v2.losses import compute_edge_band, FTTv2Loss


class TestEdgeBand:
    """Tests for the edge band computation."""

    def test_square_mask_band_area(self):
        """Edge band on a toy square mask has expected area > 0."""
        # 64x64 mask with a 20x20 square in the center
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 22:42, 22:42] = 1.0

        band = compute_edge_band(mask, band_px=7)
        assert band.shape == mask.shape
        assert band.sum() > 0
        # Band should be smaller than the whole mask
        assert band.sum() < mask.shape[2] * mask.shape[3]
        # Band should be along edges (inside the dilation-erosion zone)
        assert band.max() == 1.0
        assert band.min() == 0.0

    def test_empty_mask_zero_band(self):
        """All-zero mask (authentic) gives zero-area band."""
        mask = torch.zeros(1, 1, 64, 64)
        band = compute_edge_band(mask, band_px=7)
        assert band.sum() == 0

    def test_full_mask(self):
        """All-ones mask has no internal edge boundary (zero band)."""
        mask = torch.ones(1, 1, 64, 64)
        band = compute_edge_band(mask, band_px=7)
        assert band.sum() == 0

    def test_batch(self):
        """Works with batch > 1."""
        mask = torch.zeros(4, 1, 64, 64)
        mask[0, :, 10:30, 10:30] = 1.0
        mask[2, :, 5:50, 5:50] = 1.0
        band = compute_edge_band(mask, band_px=7)
        assert band.shape == (4, 1, 64, 64)


class TestFTTv2Loss:
    """Tests for the combined loss function."""

    def _make_inputs(self, B=2, H=64, W=64, forged_mask=True):
        mask_logits = torch.randn(B, 1, H, W, requires_grad=True)
        if forged_mask:
            mask_gt = torch.zeros(B, 1, H, W)
            mask_gt[:, :, 10:30, 10:30] = 1.0
        else:
            mask_gt = torch.zeros(B, 1, H, W)
        det_logit = torch.randn(B, requires_grad=True)
        is_forged = torch.ones(B) if forged_mask else torch.zeros(B)
        type_logits = torch.randn(B, 3, requires_grad=True)
        type_targets = torch.zeros(B, 3)
        if forged_mask:
            type_targets[:, 0] = 1.0  # copy_move
        type_valid = torch.ones(B, 3)
        return mask_logits, mask_gt, det_logit, is_forged, type_logits, type_targets, type_valid

    def test_all_losses_finite(self):
        """All loss components are finite on random inputs."""
        loss_fn = FTTv2Loss()
        args = self._make_inputs(forged_mask=True)
        losses = loss_fn(*args, det_enabled=True, type_enabled=True)

        for name, val in losses.items():
            assert torch.isfinite(val), f"Loss '{name}' is not finite: {val}"

    def test_authentic_finite(self):
        """All-zero mask (authentic) gives finite loss and zero edge loss."""
        loss_fn = FTTv2Loss()
        args = self._make_inputs(forged_mask=False)
        losses = loss_fn(*args, det_enabled=True, type_enabled=True)

        for name, val in losses.items():
            assert torch.isfinite(val), f"Loss '{name}' is not finite: {val}"

        # Edge loss should be 0 for authentic (empty band)
        assert losses["edge"].item() == 0.0

    def test_backward(self):
        """Loss allows backward pass."""
        loss_fn = FTTv2Loss()
        args = self._make_inputs()
        losses = loss_fn(*args, det_enabled=True, type_enabled=True)
        losses["total"].backward()
        # Check gradients exist
        assert args[0].grad is not None  # mask_logits
        assert args[2].grad is not None  # det_logit
        assert args[4].grad is not None  # type_logits

    def test_accum_steps_divides(self):
        """Loss is divided by accum_steps."""
        loss_fn_1 = FTTv2Loss(accum_steps=1)
        loss_fn_4 = FTTv2Loss(accum_steps=4)

        args = self._make_inputs()
        l1 = loss_fn_1(*args, det_enabled=True, type_enabled=True)
        l4 = loss_fn_4(*args, det_enabled=True, type_enabled=True)

        # Total loss should be divided by accum_steps
        # Note: only total is divided, not individual components
        ratio = l1["total"].item() / max(l4["total"].item(), 1e-8)
        assert abs(ratio - 4.0) < 0.01, f"Expected ratio ~4, got {ratio}"

    def test_type_valid_masking(self):
        """type_valid=0 excludes those entries from type loss."""
        loss_fn = FTTv2Loss()
        args = list(self._make_inputs())
        # Set all type_valid to 0
        args[6] = torch.zeros(2, 3)
        losses = loss_fn(*args, det_enabled=True, type_enabled=True)
        assert losses["type"].item() == 0.0

    def test_no_sigmoid_before_logits_loss(self):
        """Verify the loss uses raw logits (BCEWithLogits), not sigmoid + BCE."""
        loss_fn = FTTv2Loss()
        # Large positive logits should give low loss for target=1
        mask_logits = torch.full((1, 1, 32, 32), 10.0, requires_grad=True)
        mask_gt = torch.ones(1, 1, 32, 32)
        det_logit = torch.tensor([10.0], requires_grad=True)
        is_forged = torch.tensor([1.0])
        type_logits = torch.full((1, 3), 10.0, requires_grad=True)
        type_targets = torch.ones(1, 3)
        type_valid = torch.ones(1, 3)

        losses = loss_fn(mask_logits, mask_gt, det_logit, is_forged,
                         type_logits, type_targets, type_valid,
                         det_enabled=True, type_enabled=True)
        # BCE and detection losses should be very small for matching inputs
        assert losses["bce"].item() < 0.01
        assert losses["det"].item() < 0.01
