"""
losses.py — Loss functions for FTT v2.

L_total = L_seg + lambda_det * L_det + lambda_type * L_type

L_seg = L_bce + lambda_edge * L_edge
L_det = BCEWithLogits(det_logit, is_forged)
L_type = BCEWithLogits(type_logits, type_targets) [masked by type_valid]
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_edge_band(
    mask: torch.Tensor,
    band_px: int = 7,
) -> torch.Tensor:
    """Compute the edge band around the GT mask boundary.

    Uses max-pool for dilation and min-pool (neg-maxpool-neg) for erosion.

    Parameters
    ----------
    mask : (B, 1, H, W)
        Binary ground-truth mask in {0, 1}.
    band_px : int
        Width of the band in pixels.

    Returns
    -------
    band : (B, 1, H, W)
        Binary band mask where the edge region is 1.
    """
    k = band_px
    p = k // 2

    # Dilation: maxpool
    dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=p)
    # Erosion: -maxpool(-mask)
    eroded = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=p)
    # Band = dilated - eroded
    band = dilated - eroded
    return band.clamp(0, 1)


class FTTv2Loss(nn.Module):
    """Combined loss for FTT v2.

    Parameters
    ----------
    lambda_edge : float
        Weight for edge loss (default: 20, from paper).
    edge_band_px : int
        Width of edge band in pixels (default: 7).
    lambda_det : float
        Weight for detection loss (default: 1.0).
    lambda_type : float
        Weight for type classification loss (default: 1.0).
    type_pos_weight : list of float or None
        Per-class positive weights for type BCEWithLogits.
    accum_steps : int
        Number of gradient accumulation steps. Loss is divided by this.
    """

    def __init__(
        self,
        lambda_edge: float = 20.0,
        edge_band_px: int = 7,
        lambda_det: float = 1.0,
        lambda_type: float = 1.0,
        type_pos_weight: list | None = None,
        accum_steps: int = 1,
    ):
        super().__init__()
        self.lambda_edge = lambda_edge
        self.edge_band_px = edge_band_px
        self.lambda_det = lambda_det
        self.lambda_type = lambda_type
        self.accum_steps = accum_steps

        if type_pos_weight is not None:
            self.register_buffer(
                "type_pos_weight",
                torch.tensor(type_pos_weight, dtype=torch.float32),
            )
        else:
            self.type_pos_weight = None

    def forward(
        self,
        mask_logits: torch.Tensor,
        mask_gt: torch.Tensor,
        det_logit: torch.Tensor,
        is_forged: torch.Tensor,
        type_logits: torch.Tensor,
        type_targets: torch.Tensor,
        type_valid: torch.Tensor,
        det_enabled: bool = True,
        type_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        mask_logits : (B, 1, H, W)
            Predicted mask logits.
        mask_gt : (B, 1, H, W)
            Ground-truth binary mask {0, 1}.
        det_logit : (B,)
            Image-level detection logit.
        is_forged : (B,)
            Ground-truth binary label {0, 1} float.
        type_logits : (B, 3)
            Predicted type logits.
        type_targets : (B, 3)
            Ground-truth type targets {0, 1} float.
        type_valid : (B, 3)
            Mask for valid type labels (1 = valid, 0 = exclude).
        det_enabled : bool
            Whether to compute detection loss.
        type_enabled : bool
            Whether to compute type loss.

        Returns
        -------
        losses : dict
            Keys: ``total``, ``seg``, ``bce``, ``edge``, ``det``, ``type``.
        """
        device = mask_logits.device
        losses = {}

        # ---- Segmentation loss ----
        # L_bce: BCEWithLogits over all pixels, mean
        bce_map = F.binary_cross_entropy_with_logits(
            mask_logits, mask_gt, reduction="none",
        )
        L_bce = bce_map.mean()
        losses["bce"] = L_bce

        # L_edge: weighted BCE in edge band
        band = compute_edge_band(mask_gt, self.edge_band_px)
        band_sum = band.sum()
        # Guard: max(sum(band), 1) to prevent NaN on authentic images
        L_edge = (band * bce_map).sum() / band_sum.clamp(min=1.0)
        losses["edge"] = L_edge

        L_seg = L_bce + self.lambda_edge * L_edge
        losses["seg"] = L_seg

        # ---- Detection loss ----
        if det_enabled:
            L_det = F.binary_cross_entropy_with_logits(
                det_logit, is_forged,
            )
        else:
            L_det = torch.tensor(0.0, device=device)
        losses["det"] = L_det

        # ---- Type loss ----
        if type_enabled:
            # Masked BCEWithLogits: only compute for valid type labels
            type_bce = F.binary_cross_entropy_with_logits(
                type_logits, type_targets,
                pos_weight=self.type_pos_weight,
                reduction="none",
            )
            # Mask invalid entries
            valid_mask = type_valid.float()
            valid_count = valid_mask.sum()
            if valid_count > 0:
                L_type = (type_bce * valid_mask).sum() / valid_count
            else:
                L_type = torch.tensor(0.0, device=device)
        else:
            L_type = torch.tensor(0.0, device=device)
        losses["type"] = L_type

        # ---- Total loss ----
        L_total = L_seg + self.lambda_det * L_det + self.lambda_type * L_type
        # Divide by accumulation steps
        L_total = L_total / self.accum_steps
        losses["total"] = L_total

        return losses
