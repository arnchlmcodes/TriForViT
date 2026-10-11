"""
ckpt.py — Checkpoint save / resume utilities for FTT v2.

Saves: model (trainable weights), optimizer, scheduler, scaler, epoch, RNG states.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    best_score: float,
    config: dict,
) -> None:
    """Save a full training checkpoint for resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect trainable state dict only
    trainable_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    # Also include buffers from heads (e.g., BN running stats)
    state_dict = model.state_dict()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_score": best_score,
        "config": config,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.amp.GradScaler] = None,
    device: str = "cpu",
) -> dict:
    """Load checkpoint and restore all states.

    Returns
    -------
    info : dict
        Contains ``epoch``, ``best_score``, ``config``.
    """
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    # Restore RNG states
    rng = ckpt.get("rng_state", {})
    if "python" in rng:
        random.setstate(rng["python"])
    if "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if "torch" in rng:
        torch.random.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])

    return {
        "epoch": ckpt["epoch"],
        "best_score": ckpt.get("best_score", 0.0),
        "config": ckpt.get("config", {}),
    }
