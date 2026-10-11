"""
copy_move.py — Copy-move self-correlation branch for FTT v2.

Explicit self-correlation inspired by BusterNet and Forensim.
Operates on early-layer patch tokens to detect copy-move forgeries.

CRITICAL: Do NOT pixel-supervise cm_map_logits with the ground-truth mask.
The match map lights up BOTH the source and pasted regions, but GT masks
usually mark only the pasted one. Let it learn only through downstream losses.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CopyMoveBranch(nn.Module):
    """Self-correlation branch for copy-move detection.

    Takes early-layer patch tokens → affinity matrix → per-patch
    descriptors → small conv head → cm_feat + cm_map_logits.

    Runs entirely in float32 with autocast disabled.
    """

    def __init__(
        self,
        embed_dim: int,
        proj_dim: int = 128,
        min_dist: int = 3,
        topk: int = 8,
        tau_init: float = 0.1,
        grid_h: int = 32,
        grid_w: int = 32,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.min_dist = min_dist
        self.topk = topk
        self.grid_h = grid_h
        self.grid_w = grid_w
        N = grid_h * grid_w

        # Projection to lower dim + L2 normalize
        self.proj = nn.Linear(embed_dim, proj_dim)

        # Learnable log-temperature: learn log(tau) so tau stays positive
        # and is not pulled toward zero by weight decay.
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

        # Near-neighbor mask: precomputed (N, N) boolean
        # near[i,j] = True if Chebyshev distance <= min_dist
        ys, xs = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
        coords = torch.stack([ys.flatten(), xs.flatten()], dim=1)  # (N, 2)
        near = (coords[:, None] - coords[None]).abs().amax(dim=-1) <= min_dist
        self.register_buffer("near_mask", near)

        # Small conv head on top-k descriptors
        self.conv_head = nn.Sequential(
            nn.Conv2d(topk, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.cm_map_conv = nn.Conv2d(32, 1, 1)

    @torch.autocast(device_type="cuda", enabled=False)
    @torch.autocast(device_type="cpu", enabled=False)
    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        patch_tokens : (B, N, d)
            Patch tokens from an early transformer layer.

        Returns
        -------
        cm_feat : (B, 32, grid_h, grid_w)
            Feature map to concatenate with F_last for the mask decoder.
        cm_map_logits : (B, 1, grid_h, grid_w)
            Copy-move match-map logits (NOT to be pixel-supervised).
        """
        # Force float32
        patch_tokens = patch_tokens.float()

        B, N, d = patch_tokens.shape
        assert N <= 4096, f"N={N} too large for dense affinity. Implement chunked top-k."
        assert N == self.grid_h * self.grid_w, (
            f"Expected N={self.grid_h * self.grid_w}, got {N}"
        )

        # 1. Project + L2 normalize
        Fn = self.proj(patch_tokens)  # (B, N, proj_dim)
        Fn = F.normalize(Fn, p=2, dim=-1)

        # 2. Affinity: A = Fn @ Fn^T / tau (tau = exp(log_tau), clamped > 0)
        tau = self.log_tau.exp().clamp(min=0.01)
        A = torch.bmm(Fn, Fn.transpose(1, 2)) / tau  # (B, N, N)

        # 3. Mask out self and nearby patches (use finite value, never -inf)
        A = A.masked_fill(self.near_mask.unsqueeze(0), -1e4)

        # 4. Mutual-match normalization: P = sqrt(softmax_row * softmax_col)
        P_row = F.softmax(A, dim=-1)   # (B, N, N)
        P_col = F.softmax(A, dim=-2)   # (B, N, N)
        P = torch.sqrt(P_row * P_col + 1e-8)  # add eps to avoid sqrt(0)

        # 5. Top-k per patch along last dim
        topk_vals, _ = P.topk(self.topk, dim=-1)  # (B, N, k)

        # 6. Reshape to spatial grid
        topk_map = topk_vals.permute(0, 2, 1)  # (B, k, N)
        topk_map = topk_map.reshape(B, self.topk, self.grid_h, self.grid_w)

        # 7. Small conv head
        cm_feat = self.conv_head(topk_map)  # (B, 32, grid_h, grid_w)
        cm_map_logits = self.cm_map_conv(cm_feat)  # (B, 1, grid_h, grid_w)

        return cm_feat, cm_map_logits
