"""
heads.py — Decoder and classification heads for FTT v2.

Contains:
  - MaskDecoder: conv-based pixel-level localization decoder
  - DetectionHead: image-level forgery detection (binary logit)
  - TypeHead: multi-label forgery type classification (3 logits)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, channels: int) -> nn.Module:
    """Create a normalization layer.

    Parameters
    ----------
    norm_type : str
        ``"bn"`` for BatchNorm2d, ``"gn"`` for GroupNorm (8 groups).
    channels : int
        Number of channels / features.
    """
    if norm_type == "bn":
        return nn.BatchNorm2d(channels)
    elif norm_type == "gn":
        groups = min(8, channels)
        return nn.GroupNorm(groups, channels)
    else:
        raise ValueError(f"Unknown norm type: {norm_type!r}")


# ---------------------------------------------------------------------------
# Mask Decoder (Section 3.3.1)
# ---------------------------------------------------------------------------

class MaskDecoder(nn.Module):
    """Conv-based mask decoder: F_last → mask_logits.

    Architecture (from the plan):
        Conv3x3(in → d/2) + Norm + ReLU
        Conv3x3(d/2 → d/4) + Norm + ReLU
        Conv1x1(d/4 → 1)
        Bilinear upsample to image_size × image_size

    Output is raw logits (no sigmoid).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        image_size: int = 512,
        norm_type: str = "bn",
    ):
        super().__init__()
        d = embed_dim
        mid1 = d // 2
        mid2 = d // 4

        self.conv1 = nn.Conv2d(in_channels, mid1, 3, padding=1)
        self.norm1 = _make_norm(norm_type, mid1)
        self.conv2 = nn.Conv2d(mid1, mid2, 3, padding=1)
        self.norm2 = _make_norm(norm_type, mid2)
        self.conv3 = nn.Conv2d(mid2, 1, 1)
        self.image_size = image_size

    def forward(self, f_last: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        f_last : (B, in_channels, n_h, n_w)
            Patch-level feature map (e.g., 32×32 for 512 input with patch 16).

        Returns
        -------
        mask_logits : (B, 1, image_size, image_size)
        """
        x = F.relu(self.norm1(self.conv1(f_last)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)  # (B, 1, n_h, n_w)
        # Upsample to full image resolution
        mask_logits = F.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        return mask_logits


# ---------------------------------------------------------------------------
# Detection Head (Section 3.3.2)
# ---------------------------------------------------------------------------

class DetectionHead(nn.Module):
    """Image-level forgery detection head.

    Input: concat(cls_token, mean(patch_tokens)) → 2d features.
    Optionally includes detached mask statistics (s_max, s_topk).

    Output: det_logit of shape (B,).
    """

    def __init__(
        self,
        embed_dim: int,
        use_mask_stats: bool = True,
        cm_scalars: bool = False,
    ):
        super().__init__()
        in_dim = 2 * embed_dim  # cls + mean_patches
        if use_mask_stats:
            in_dim += 2  # s_max, s_topk
        if cm_scalars:
            in_dim += 2  # cm s_max, cm s_topk

        self.use_mask_stats = use_mask_stats
        self.cm_scalars = cm_scalars

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_mean: torch.Tensor,
        mask_logits_32: torch.Tensor | None = None,
        cm_map_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_token : (B, d)
        patch_mean : (B, d)
        mask_logits_32 : (B, 1, n_h, n_w) or None
            Low-res mask logits (before upsampling). Detached.
        cm_map_logits : (B, 1, n_h, n_w) or None
            Copy-move map logits. Detached.

        Returns
        -------
        det_logit : (B,)
        """
        features = [cls_token, patch_mean]

        if self.use_mask_stats and mask_logits_32 is not None:
            ml = mask_logits_32.detach()
            s_max = ml.flatten(1).max(dim=1).values.unsqueeze(1)  # (B, 1)
            # Top 1% mean
            flat = ml.flatten(1)  # (B, N)
            k = max(1, flat.shape[1] // 100)
            topk_vals = flat.topk(k, dim=1).values
            s_topk = topk_vals.mean(dim=1, keepdim=True)  # (B, 1)
            features.extend([s_max, s_topk])

        if self.cm_scalars and cm_map_logits is not None:
            cm = cm_map_logits.detach()
            cm_max = cm.flatten(1).max(dim=1).values.unsqueeze(1)
            cm_flat = cm.flatten(1)
            cm_k = max(1, cm_flat.shape[1] // 100)
            cm_topk = cm_flat.topk(cm_k, dim=1).values.mean(dim=1, keepdim=True)
            features.extend([cm_max, cm_topk])

        z = torch.cat(features, dim=1)
        return self.mlp(z).squeeze(1)  # (B,)


# ---------------------------------------------------------------------------
# Type Head (Section 3.3.3)
# ---------------------------------------------------------------------------

class TypeHead(nn.Module):
    """Multi-label forgery type classification head.

    Multi-label (BCEWithLogits), NOT softmax.
    Output: 3 logits in TYPES order [copy_move, splicing, inpainting].
    """

    def __init__(
        self,
        embed_dim: int,
        num_types: int = 3,
        cm_scalars: bool = False,
    ):
        super().__init__()
        in_dim = 2 * embed_dim  # cls + mean_patches
        if cm_scalars:
            in_dim += 2  # cm s_max, cm s_topk

        self.cm_scalars = cm_scalars

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_types),
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_mean: torch.Tensor,
        cm_map_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        cls_token : (B, d)
        patch_mean : (B, d)
        cm_map_logits : (B, 1, n_h, n_w) or None

        Returns
        -------
        type_logits : (B, 3)
        """
        features = [cls_token, patch_mean]

        if self.cm_scalars and cm_map_logits is not None:
            cm = cm_map_logits.detach()
            cm_max = cm.flatten(1).max(dim=1).values.unsqueeze(1)
            cm_flat = cm.flatten(1)
            cm_k = max(1, cm_flat.shape[1] // 100)
            cm_topk = cm_flat.topk(cm_k, dim=1).values.mean(dim=1, keepdim=True)
            features.extend([cm_max, cm_topk])

        z = torch.cat(features, dim=1)
        return self.mlp(z)
