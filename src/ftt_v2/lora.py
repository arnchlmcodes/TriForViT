"""
lora.py — LoRA wrapper for FTT v2 backbone.

Applies LoRA adapters to Q, K, V projections of every self-attention layer
in the DINOv3 backbone using the ``peft`` library.
"""

from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")

from typing import List, Optional

import torch
import torch.nn as nn


def find_qkv_target_modules(backbone: nn.Module) -> List[str]:
    """Discover the names of Q, K, V linear layers in the backbone.

    Supports both separate Q/K/V (``query``, ``key``, ``value``) and
    fused ``qkv`` projections. Searches the full module tree.

    Returns
    -------
    target_names : list of str
        Module name patterns suitable for ``peft.LoraConfig.target_modules``.
    """
    names = set()
    for name, module in backbone.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        basename = name.split(".")[-1]
        # Separate Q, K, V (transformers-style DINOv2/v3)
        if basename in ("query", "key", "value", "q_proj", "k_proj", "v_proj"):
            names.add(basename)
        # Fused QKV
        if basename in ("qkv",):
            names.add(basename)
    if not names:
        raise RuntimeError(
            "Could not auto-detect Q/K/V target modules in the backbone. "
            "Please inspect `backbone.named_modules()` and set "
            "`lora.target_modules` manually."
        )
    return sorted(names)


def apply_lora(
    backbone: nn.Module,
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    depth: Optional[int] = None,
) -> nn.Module:
    """Wrap the backbone with LoRA adapters on Q/K/V projections.

    Parameters
    ----------
    backbone : nn.Module
        The frozen DINOv3 backbone.
    r : int
        LoRA rank.
    alpha : int
        LoRA alpha (scaling).
    dropout : float
        LoRA dropout.
    target_modules : list of str or None
        Target module name patterns. Auto-detected if None.
    depth : int or None
        Expected number of transformer layers (for assertion).

    Returns
    -------
    backbone : nn.Module
        The backbone with LoRA adapters applied (via peft).
    """
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = find_qkv_target_modules(backbone)

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    backbone = get_peft_model(backbone, config)

    # --- Assertions (Section 3.2 of the plan) ---
    _verify_lora(backbone, target_modules, depth)

    return backbone


def _verify_lora(
    model: nn.Module,
    target_modules: list,
    depth: Optional[int],
) -> None:
    """Run the four LoRA assertions from Section 3.2 of the plan."""

    # Count LoRA-wrapped modules
    lora_module_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            lora_module_count += 1

    # (1) Number of LoRA-wrapped modules
    if depth is not None:
        # Separate Q/K/V → 3*depth modules; fused QKV → depth modules
        is_fused = any(t in ("qkv",) for t in target_modules)
        n_targets = len(target_modules)
        expected = depth * n_targets if not is_fused else depth
        assert lora_module_count == expected, (
            f"Expected {expected} LoRA modules "
            f"({n_targets} targets × {depth} layers), got {lora_module_count}"
        )

    # (2) Every trainable backbone parameter name contains 'lora_'
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    non_lora_trainable = [n for n in trainable_names if "lora_" not in n]
    assert len(non_lora_trainable) == 0, (
        f"Found trainable non-LoRA params: {non_lora_trainable[:5]}"
    )

    # (3) Trainable params > 0 and < 10% of total
    # Note: ViT-S (d=384) with r=64 on q/k/v across 12 layers gives ~7.6%,
    # while ViT-B (d=768) gives ~4%. The 10% bound covers both.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params > 0, "No trainable parameters after LoRA"
    ratio = trainable_params / total_params
    assert ratio < 0.10, (
        f"Trainable params = {ratio:.2%} of total, expected < 10%"
    )


def verify_lora_noop(
    backbone_lora: nn.Module,
    backbone_plain: nn.Module,
    dummy_input: torch.Tensor,
) -> None:
    """Assert that LoRA starts as a no-op (B matrices init to zero).

    Parameters
    ----------
    backbone_lora : nn.Module
        Backbone with freshly-applied LoRA (B matrices should be zero).
    backbone_plain : nn.Module
        The same backbone without LoRA.
    dummy_input : torch.Tensor
        A (B, 3, H, W) input tensor.
    """
    backbone_lora.eval()
    backbone_plain.eval()
    with torch.no_grad():
        out_lora = backbone_lora(dummy_input)
        out_plain = backbone_plain(dummy_input)

    # Handle dict outputs
    if isinstance(out_lora, dict):
        h_lora = out_lora["last_hidden_state"]
        h_plain = out_plain["last_hidden_state"]
    else:
        h_lora = out_lora[0] if isinstance(out_lora, (tuple, list)) else out_lora
        h_plain = out_plain[0] if isinstance(out_plain, (tuple, list)) else out_plain

    assert torch.allclose(h_lora, h_plain, atol=1e-5), (
        f"LoRA is NOT a no-op at init! "
        f"max diff = {(h_lora - h_plain).abs().max().item():.6e}"
    )
