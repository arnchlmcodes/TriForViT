"""
model.py - Three-Stream Tokenization Vision Transformer

Architecture (from Team 39 presentation):
  Input Image → Patch Extraction →
    ├─ Spatial Token  (linear projection of raw pixels)
    ├─ Frequency Token  (FFT magnitude → linear projection)
    └─ Noise Residual Token  (HP-filtered residual → linear projection)
  → Cross-Attention Fusion → Transformer Encoder →
    ├─ Forgery Detection  (binary: authentic / tampered)
    └─ Forgery-Type Classification  (copy-move / splicing / inpainting)
"""

import torch
import torch.nn as nn
import math

from extractors import compute_fft_magnitude, compute_noise_residual


# ---------------------------------------------------------------------------
# Stream Embedding
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Flatten a patch and project to embedding dim (shared architecture for all streams)."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: (B, N, C, pH, pW)
        Returns:
            tokens: (B, N, D)
        """
        B, N = patches.shape[:2]
        flat = patches.reshape(B, N, -1)
        return self.proj(flat)


# ---------------------------------------------------------------------------
# Cross-Attention Fusion Layer
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Fuse three stream tokens per patch via multi-head cross-attention.
    Input:  T_i = [z_s, z_f, z_n]  ∈ R^{3 × D}   for each patch i
    Output: z*_i ∈ R^{D}
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        z_spatial: torch.Tensor,
        z_freq: torch.Tensor,
        z_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_spatial, z_freq, z_noise: each (B, N, D)
        Returns:
            fused: (B, N, D)
        """
        B, N, D = z_spatial.shape
        # Stack streams → (B*N, 3, D)
        stacked = torch.stack([z_spatial, z_freq, z_noise], dim=2)
        stacked = stacked.view(B * N, 3, D)

        attn_out, _ = self.attn(stacked, stacked, stacked)
        attn_out = self.norm(attn_out)

        # Mean-pool across 3 stream tokens → single fused token per patch
        fused = attn_out.mean(dim=1)  # (B*N, D)
        return fused.view(B, N, D)


# ---------------------------------------------------------------------------
# Transformer Encoder (lightweight placeholder — swap with timm/ViT later)
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Standard Transformer encoder block stack."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


# ---------------------------------------------------------------------------
# Full Model: ThreeStreamViT
# ---------------------------------------------------------------------------

class ThreeStreamViT(nn.Module):
    """
    Three-Stream Tokenization Vision Transformer for Image Forgery
    Detection and Classification.

    Hierarchical strategy:
      Head 1 — Binary detection  (Authentic vs Tampered)
      Head 2 — Forgery-type classification  (Copy-Move / Splicing / Inpainting)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        num_forgery_types: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # --- Stream embeddings ---
        self.spatial_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.freq_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.noise_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

        # --- Fusion ---
        self.fusion = CrossAttentionFusion(embed_dim, num_heads=8)

        # --- Positional embedding + CLS token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Transformer ---
        self.transformer = TransformerEncoder(embed_dim, depth, num_heads)

        # --- Classification heads ---
        self.detection_head = nn.Linear(embed_dim, 2)        # authentic / tampered
        self.classification_head = nn.Linear(embed_dim, num_forgery_types)

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, C, H, W)
        Returns:
            detection_logits: (B, 2)
            classification_logits: (B, num_forgery_types)
        """
        from extractors import extract_patches

        patches = extract_patches(images, self.patch_size)  # (B, N, C, pH, pW)

        # --- Three streams ---
        z_s = self.spatial_embed(patches)

        B, N, C, pH, pW = patches.shape
        freq_patches = compute_fft_magnitude(patches.reshape(-1, C, pH, pW))
        freq_patches = freq_patches.reshape(B, N, C, pH, pW)
        z_f = self.freq_embed(freq_patches)

        noise_patches = compute_noise_residual(patches.reshape(-1, C, pH, pW))
        noise_patches = noise_patches.reshape(B, N, C, pH, pW)
        z_n = self.noise_embed(noise_patches)

        # --- Cross-attention fusion ---
        fused = self.fusion(z_s, z_f, z_n)  # (B, N, D)

        # Prepend CLS token + add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, fused], dim=1)
        tokens = tokens + self.pos_embed

        # --- Transformer ---
        encoded = self.transformer(tokens)

        # CLS token output
        cls_out = encoded[:, 0]

        detection_logits = self.detection_head(cls_out)
        classification_logits = self.classification_head(cls_out)

        return detection_logits, classification_logits
