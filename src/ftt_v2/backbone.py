"""
backbone.py — DINOv3 ViT backbone loader for FTT v2.

Supports:
  - ``dinov3_vits16``: loads from HuggingFace ``facebook/dinov3-vits16-pretrain-lvd1689m``
  - ``dummy``: a tiny frozen random transformer with the same output layout,
    so unit tests run on CPU without the gated model.

The backbone is always fully frozen (``requires_grad=False``). LoRA is applied
separately in ``lora.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Token splitting helper
# ---------------------------------------------------------------------------

def split_tokens(
    hidden: torch.Tensor,
    n_h: int,
    n_w: int,
    n_prefix: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a transformer output sequence into CLS and patch tokens.

    Parameters
    ----------
    hidden : (B, S, d)
        Full sequence including CLS, register tokens and patch tokens.
    n_h, n_w : int
        Grid dimensions for the patch tokens (height, width in patches).
    n_prefix : int
        Number of prefix tokens (CLS + registers) before the patch tokens.

    Returns
    -------
    cls : (B, d)
        The CLS token.
    patches : (B, d, n_h, n_w)
        Patch tokens reshaped to a 2-D feature map.
    """
    cls = hidden[:, 0]  # (B, d)
    patch_tokens = hidden[:, n_prefix: n_prefix + n_h * n_w]  # (B, N, d)
    B, N, d = patch_tokens.shape
    assert N == n_h * n_w, (
        f"Expected {n_h * n_w} patch tokens, got {N} "
        f"(n_prefix={n_prefix}, seq_len={hidden.shape[1]})"
    )
    patches = patch_tokens.permute(0, 2, 1).reshape(B, d, n_h, n_w)
    return cls, patches


# ---------------------------------------------------------------------------
# Dummy backbone for CPU unit tests
# ---------------------------------------------------------------------------

class DummyViTBackbone(nn.Module):
    """Tiny frozen random ViT-like backbone with the same output API.

    Does NOT learn anything. Outputs random but deterministic features
    given the same input and seed.
    """

    def __init__(self, embed_dim: int = 384, depth: int = 12,
                 n_register: int = 0, patch_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.n_register = n_register
        self.patch_size = patch_size
        # n_prefix = 1 (CLS) + n_register
        self.n_prefix = 1 + n_register

        # Tiny linear projection (replaces real patch embedding)
        self.proj = nn.Linear(3 * patch_size * patch_size, embed_dim)
        # Simple transformer blocks (tiny)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=6, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=min(depth, 2))
        self.norm = nn.LayerNorm(embed_dim)

        # CLS and register tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        if n_register > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, n_register, embed_dim) * 0.02
            )

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass mimicking HuggingFace DINOv3 output format.

        Parameters
        ----------
        pixel_values : (B, 3, H, W)
        output_hidden_states : bool

        Returns
        -------
        dict with keys:
            - ``last_hidden_state``: (B, S, d)
            - ``hidden_states``: tuple of (B, S, d) if requested
        """
        B, C, H, W = pixel_values.shape
        n_h = H // self.patch_size
        n_w = W // self.patch_size
        N = n_h * n_w

        # Patchify
        patches = pixel_values.unfold(2, self.patch_size, self.patch_size) \
                              .unfold(3, self.patch_size, self.patch_size)
        # (B, C, n_h, n_w, p, p)
        patches = patches.contiguous().reshape(B, C, N, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).reshape(B, N, -1)  # (B, N, C*p*p)
        patch_embeds = self.proj(patches)  # (B, N, d)

        # Prepend CLS + registers
        cls = self.cls_token.expand(B, -1, -1)
        tokens = [cls]
        if self.n_register > 0:
            tokens.append(self.register_tokens.expand(B, -1, -1))
        tokens.append(patch_embeds)
        x = torch.cat(tokens, dim=1)  # (B, S, d)

        hidden_states = [x] if output_hidden_states else []

        # Run through encoder
        # For dummy, we treat the whole encoder as one step and record
        # intermediate states by running layers individually
        for layer in self.encoder.layers:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.norm(x)

        result = {"last_hidden_state": x}
        if output_hidden_states:
            result["hidden_states"] = tuple(hidden_states)
        return result

    @property
    def config_dict(self) -> dict:
        return {
            "hidden_size": self.embed_dim,
            "num_hidden_layers": self.depth,
            "patch_size": self.patch_size,
            "n_register": self.n_register,
        }


# ---------------------------------------------------------------------------
# DINOv3 backbone loader
# ---------------------------------------------------------------------------

def load_dinov3_backbone(
    hf_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
) -> Tuple[nn.Module, dict]:
    """Load the DINOv3 ViT backbone from HuggingFace.

    Returns
    -------
    model : nn.Module
        The backbone model with all parameters frozen.
    info : dict
        ``hidden_size``, ``num_hidden_layers``, ``patch_size``,
        ``n_register``, ``n_prefix``, ``mean``, ``std``.
    """
    import os
    from transformers import AutoModel, AutoImageProcessor

    token = os.environ.get("HF_TOKEN", None)

    model = AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True)
    processor = AutoImageProcessor.from_pretrained(hf_id, token=token, trust_remote_code=True)

    # Freeze all backbone parameters
    for p in model.parameters():
        p.requires_grad = False

    # Extract config
    cfg = model.config
    hidden_size = cfg.hidden_size
    num_layers = cfg.num_hidden_layers
    patch_size = cfg.patch_size

    # Detect register tokens
    # DINOv3 models may have register tokens. Check the config.
    n_register = getattr(cfg, "num_register_tokens", 0)
    n_prefix = 1 + n_register  # CLS + registers

    # Read normalization from processor
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])

    info = {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "patch_size": patch_size,
        "n_register": n_register,
        "n_prefix": n_prefix,
        "mean": list(mean),
        "std": list(std),
    }

    return model, info


def load_backbone(name: str, **kwargs) -> Tuple[nn.Module, dict]:
    """Load backbone by name.

    Parameters
    ----------
    name : str
        ``"dinov3_vits16"`` or ``"dummy"``

    Returns
    -------
    model, info
    """
    if name == "dummy":
        embed_dim = kwargs.get("embed_dim", 384)
        depth = kwargs.get("depth", 12)
        n_register = kwargs.get("n_register", 0)
        patch_size = kwargs.get("patch_size", 16)
        model = DummyViTBackbone(
            embed_dim=embed_dim, depth=depth,
            n_register=n_register, patch_size=patch_size,
        )
        info = {
            "hidden_size": embed_dim,
            "num_hidden_layers": depth,
            "patch_size": patch_size,
            "n_register": n_register,
            "n_prefix": 1 + n_register,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        return model, info
    elif name == "dinov3_vits16":
        hf_id = kwargs.get("hf_id", "facebook/dinov3-vits16-pretrain-lvd1689m")
        return load_dinov3_backbone(hf_id)
    else:
        raise ValueError(f"Unknown backbone: {name!r}. Use 'dinov3_vits16' or 'dummy'.")
