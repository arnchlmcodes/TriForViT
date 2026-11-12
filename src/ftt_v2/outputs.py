"""
outputs.py — Model output dataclass for FTT v2.

All training, evaluation and inference code uses these field names only.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class ModelOutput(NamedTuple):
    """Structured output from ForensicModel.forward().

    Inherits from NamedTuple so PyTorch nn.DataParallel can gather
    outputs across multiple GPUs (e.g. Kaggle 2x T4) without TypeError.

    Fields
    ------
    mask_logits : (B, 1, 512, 512)
        Per-pixel forged logit (before sigmoid).
    det_logit : (B,)
        Image-level forged logit (before sigmoid).
    type_logits : (B, 3)
        Multi-label logits in TYPES order [copy_move, splicing, inpainting].
    cm_map_logits : (B, 1, 32, 32) or None
        Copy-move match-map logits (only when copy-move branch is enabled).
    """

    mask_logits: torch.Tensor
    det_logit: torch.Tensor
    type_logits: torch.Tensor
    cm_map_logits: Optional[torch.Tensor] = None
