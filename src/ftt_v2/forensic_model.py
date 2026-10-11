"""
forensic_model.py — Main FTT v2 model wrapper.

Assembles backbone, LoRA, mask decoder, detection head, type head,
and optional copy-move branch into a single nn.Module.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from . import TYPES
from .backbone import load_backbone, split_tokens
from .copy_move import CopyMoveBranch
from .heads import DetectionHead, MaskDecoder, TypeHead
from .lora import apply_lora
from .outputs import ModelOutput


def _get_git_hash() -> str:
    """Get current git commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


class ForensicModel(nn.Module):
    """FTT v2: DINOv3 + LoRA multi-task image forgery model.

    forward(images) → ModelOutput with:
        mask_logits  (B, 1, 512, 512)
        det_logit    (B,)
        type_logits  (B, 3)
        cm_map_logits (B, 1, 32, 32) or None
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # ---- Backbone ----
        backbone_name = cfg.get("backbone", {}).get("name", "dummy")
        backbone_kwargs = {
            k: v for k, v in cfg.get("backbone", {}).items() if k != "name"
        }
        self.backbone, self.backbone_info = load_backbone(backbone_name, **backbone_kwargs)
        self.d = self.backbone_info["hidden_size"]
        self.patch_size = self.backbone_info["patch_size"]
        self.n_prefix = self.backbone_info["n_prefix"]
        self.image_size = cfg.get("model", {}).get("image_size", 512)
        self.n_h = self.image_size // self.patch_size
        self.n_w = self.image_size // self.patch_size

        # ---- LoRA ----
        lora_cfg = cfg.get("lora", {})
        self.lora_enabled = lora_cfg.get("enabled", True)
        if self.lora_enabled:
            target_modules = lora_cfg.get("target_modules", None)
            self.backbone = apply_lora(
                self.backbone,
                r=lora_cfg.get("r", 64),
                alpha=lora_cfg.get("alpha", 128),
                dropout=lora_cfg.get("dropout", 0.0),
                target_modules=target_modules,
                depth=self.backbone_info.get("num_hidden_layers"),
            )

        # ---- Copy-move branch ----
        cm_cfg = cfg.get("cm", {})
        self.cm_enabled = cm_cfg.get("enabled", False)
        self.cm_branch: Optional[CopyMoveBranch] = None
        if self.cm_enabled:
            self.cm_tap_layer = cm_cfg.get("tap_layer", 4)
            self.cm_branch = CopyMoveBranch(
                embed_dim=self.d,
                proj_dim=cm_cfg.get("proj_dim", 128),
                min_dist=cm_cfg.get("min_dist", 3),
                topk=cm_cfg.get("topk", 8),
                tau_init=cm_cfg.get("tau_init", 0.1),
                grid_h=self.n_h,
                grid_w=self.n_w,
            )

        # ---- Mask decoder ----
        decoder_cfg = cfg.get("decoder", {})
        decoder_in = self.d + (32 if self.cm_enabled else 0)
        self.mask_decoder = MaskDecoder(
            in_channels=decoder_in,
            embed_dim=self.d,
            image_size=self.image_size,
            norm_type=decoder_cfg.get("norm", "bn"),
        )

        # ---- Detection head ----
        det_cfg = cfg.get("det", {})
        self.det_enabled = det_cfg.get("enabled", False)
        self.det_head: Optional[DetectionHead] = None
        if self.det_enabled:
            self.det_head = DetectionHead(
                embed_dim=self.d,
                use_mask_stats=det_cfg.get("use_mask_stats", True),
                cm_scalars=self.cm_enabled,
            )

        # ---- Type head ----
        type_cfg = cfg.get("type_head", {})
        self.type_enabled = type_cfg.get("enabled", False)
        self.type_head: Optional[TypeHead] = None
        if self.type_enabled:
            self.type_head = TypeHead(
                embed_dim=self.d,
                num_types=len(TYPES),
                cm_scalars=self.cm_enabled,
            )

        # ---- LayerNorm for patch tokens if needed ----
        self.patch_norm = nn.LayerNorm(self.d)

    def forward(self, images: torch.Tensor) -> ModelOutput:
        """
        Parameters
        ----------
        images : (B, 3, image_size, image_size)
            Normalized input images.

        Returns
        -------
        ModelOutput
        """
        B = images.shape[0]

        # ---- Backbone forward ----
        need_hidden = self.cm_enabled
        backbone_out = self.backbone(
            pixel_values=images,
            output_hidden_states=need_hidden,
        )

        # Handle peft-wrapped output (may return a ModelOutput-like object)
        if hasattr(backbone_out, "last_hidden_state"):
            last_hidden = backbone_out.last_hidden_state
            hidden_states = getattr(backbone_out, "hidden_states", None)
        elif isinstance(backbone_out, dict):
            last_hidden = backbone_out["last_hidden_state"]
            hidden_states = backbone_out.get("hidden_states", None)
        else:
            raise TypeError(f"Unexpected backbone output type: {type(backbone_out)}")

        # ---- Split tokens ----
        cls_token, f_last = split_tokens(
            last_hidden, self.n_h, self.n_w, self.n_prefix,
        )
        # Apply LayerNorm to patch features
        # f_last is (B, d, n_h, n_w) → need (B, n_h*n_w, d) for norm
        f_last_flat = f_last.flatten(2).permute(0, 2, 1)  # (B, N, d)
        f_last_flat = self.patch_norm(f_last_flat)
        patch_mean = f_last_flat.mean(dim=1)  # (B, d)
        f_last = f_last_flat.permute(0, 2, 1).reshape(B, self.d, self.n_h, self.n_w)

        # ---- Copy-move branch ----
        cm_feat = None
        cm_map_logits = None
        if self.cm_enabled and self.cm_branch is not None:
            assert hidden_states is not None, (
                "Copy-move branch requires output_hidden_states=True"
            )
            # Tap early layer
            early_hidden = hidden_states[self.cm_tap_layer]
            # Extract only patch tokens from early hidden state
            early_patches = early_hidden[:, self.n_prefix:self.n_prefix + self.n_h * self.n_w]
            cm_feat, cm_map_logits = self.cm_branch(early_patches)

        # ---- Mask decoder ----
        if cm_feat is not None:
            decoder_input = torch.cat([f_last, cm_feat], dim=1)
        else:
            decoder_input = f_last

        mask_logits = self.mask_decoder(decoder_input)  # (B, 1, H, W)

        # ---- Low-res mask logits for detection head ----
        # Before upsampling, the decoder produces at (n_h, n_w) resolution
        # We pass the already-upsampled version; compute stats on low-res
        mask_logits_32 = F.interpolate(
            mask_logits, size=(self.n_h, self.n_w),
            mode="bilinear", align_corners=False,
        )

        # ---- Detection head ----
        if self.det_enabled and self.det_head is not None:
            det_logit = self.det_head(
                cls_token, patch_mean, mask_logits_32, cm_map_logits,
            )
        else:
            # Default: derive from mask (max logit)
            det_logit = mask_logits.flatten(1).max(dim=1).values

        # ---- Type head ----
        if self.type_enabled and self.type_head is not None:
            type_logits = self.type_head(cls_token, patch_mean, cm_map_logits)
        else:
            # Default: zeros (no type prediction)
            type_logits = torch.zeros(B, len(TYPES), device=images.device)

        return ModelOutput(
            mask_logits=mask_logits,
            det_logit=det_logit,
            type_logits=type_logits,
            cm_map_logits=cm_map_logits,
        )

    def trainable_parameters(self):
        """Return only trainable parameters (LoRA + heads)."""
        return [p for p in self.parameters() if p.requires_grad]

    def param_report(self) -> Dict[str, int]:
        """Print and return parameter counts by component."""
        counts = {}

        # Backbone (frozen)
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        counts["backbone_frozen"] = backbone_total - backbone_trainable
        counts["backbone_lora"] = backbone_trainable

        # Heads
        for name, module in [
            ("mask_decoder", self.mask_decoder),
            ("det_head", self.det_head),
            ("type_head", self.type_head),
            ("cm_branch", self.cm_branch),
            ("patch_norm", self.patch_norm),
        ]:
            if module is not None:
                counts[name] = sum(p.numel() for p in module.parameters())

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        counts["total"] = total
        counts["trainable"] = trainable

        print("=" * 50)
        print("Parameter Report")
        print("=" * 50)
        for k, v in counts.items():
            print(f"  {k:25s}: {v:>12,}")
        print("=" * 50)

        return counts

    def save(self, path: str) -> None:
        """Save LoRA adapter weights + head weights + config as JSON + git hash.

        Never pickles the whole model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Collect trainable state dict (LoRA + heads)
        trainable_state = {
            k: v for k, v in self.state_dict().items()
            if any(
                tag in k
                for tag in ("lora_", "mask_decoder", "det_head", "type_head",
                            "cm_branch", "patch_norm")
            )
        }
        torch.save(trainable_state, path / "weights.pt")

        # Config
        save_meta = {
            "config": self.cfg,
            "backbone_info": self.backbone_info,
            "git_hash": _get_git_hash(),
        }
        with open(path / "config.json", "w") as f:
            json.dump(save_meta, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ForensicModel":
        """Rebuild model from saved config and load weights.

        Parameters
        ----------
        path : str
            Directory containing ``config.json`` and ``weights.pt``.
        device : str

        Returns
        -------
        ForensicModel
        """
        path = Path(path)

        with open(path / "config.json") as f:
            meta = json.load(f)

        cfg = meta["config"]
        model = cls(cfg)

        weights = torch.load(path / "weights.pt", map_location=device)
        # Load only matching keys (partial load)
        model_state = model.state_dict()
        for k, v in weights.items():
            if k in model_state:
                model_state[k] = v
        model.load_state_dict(model_state)
        model.to(device)

        return model
