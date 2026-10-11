"""
train.py — FTT v2 training loop.

Usage:
    python -m src.ftt_v2.train --config configs/ftt_v2_pilot_stageA.yaml

Supports:
  - Gradient accumulation
  - Mixed precision (fp16 via GradScaler on GPU)
  - Cosine LR schedule with warmup
  - Periodic evaluation and best-checkpoint selection
  - Full resume from checkpoint
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

# ---- Local imports ----
from src.ftt_v2 import TYPES
from src.ftt_v2.forensic_model import ForensicModel
from src.ftt_v2.losses import FTTv2Loss
from src.ftt_v2.metrics import (
    compute_image_metrics,
    compute_pixel_metrics,
    compute_type_metrics,
    compute_validation_score,
)
from src.ftt_v2.utils.seed import seed_everything, log_run_info
from src.ftt_v2.utils.ckpt import save_checkpoint, load_checkpoint
from src.ftt_v2.data.dataset import ForensicDataset, BalancedForgeryTypeSampler
from src.ftt_v2.data.transforms import TrainTransform, ValTransform


def build_dataloaders(cfg: dict, backbone_info: dict) -> tuple:
    """Build train and validation data loaders from manifests."""
    data_cfg = cfg.get("data", {})
    manifest_dir = data_cfg.get("manifest_dir", "data/manifests")
    image_size = cfg.get("model", {}).get("image_size", 512)
    mean = backbone_info.get("mean", [0.485, 0.456, 0.406])
    std = backbone_info.get("std", [0.229, 0.224, 0.225])

    train_transform = TrainTransform(image_size=image_size, mean=mean, std=std)
    val_transform = ValTransform(image_size=image_size, mean=mean, std=std)

    train_manifest = os.path.join(manifest_dir, "train.csv")
    val_manifest = os.path.join(manifest_dir, "val.csv")

    train_ds = ForensicDataset(train_manifest, transform=train_transform)
    val_ds = ForensicDataset(val_manifest, transform=val_transform)

    train_cfg = cfg.get("train", {})
    micro_batch = train_cfg.get("micro_batch", 8)
    num_workers = data_cfg.get("num_workers", 4)

    # Balanced sampler
    sampler = None
    if data_cfg.get("balanced_sampler", True):
        sampler = BalancedForgeryTypeSampler(train_ds.df, seed=train_cfg.get("seed", 42))

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def build_optimizer(model: ForensicModel, cfg: dict):
    train_cfg = cfg.get("train", {})
    lr = train_cfg.get("lr", 3e-4)
    wd = train_cfg.get("weight_decay", 0.01)
    params = model.trainable_parameters()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    return optimizer


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    train_cfg = cfg.get("train", {})
    epochs = train_cfg.get("epochs", 20)
    warmup_epochs = train_cfg.get("warmup_epochs", 5)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(
    model: ForensicModel,
    val_loader: DataLoader,
    loss_fn: FTTv2Loss,
    device: torch.device,
    cfg: dict,
) -> Dict[str, float]:
    """Run validation and compute all metrics."""
    model.eval()
    det_scores, det_labels = [], []
    type_scores_list, type_targets_list, type_valid_list = [], [], []
    pred_masks_forged, gt_masks_forged = [], []
    pred_masks_auth = []
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        mask_gt = batch["mask"].to(device)
        is_forged = batch["is_forged"].to(device)
        type_targets = batch["types"].to(device)
        type_valid = batch["type_valid"].to(device)

        output = model(images)

        losses = loss_fn(
            output.mask_logits, mask_gt,
            output.det_logit, is_forged,
            output.type_logits, type_targets, type_valid,
            det_enabled=cfg.get("det", {}).get("enabled", False),
            type_enabled=cfg.get("type_head", {}).get("enabled", False),
        )
        total_loss += losses["total"].item()
        n_batches += 1

        # Collect predictions
        det_prob = torch.sigmoid(output.det_logit).cpu().numpy()
        det_scores.append(det_prob)
        det_labels.append(is_forged.cpu().numpy())

        type_prob = torch.sigmoid(output.type_logits).cpu().numpy()
        type_scores_list.append(type_prob)
        type_targets_list.append(type_targets.cpu().numpy())
        type_valid_list.append(type_valid.cpu().numpy())

        mask_prob = torch.sigmoid(output.mask_logits).squeeze(1).cpu().numpy()
        is_f = is_forged.cpu().numpy()
        for i in range(len(is_f)):
            if is_f[i] > 0.5:
                pred_masks_forged.append(mask_prob[i])
                gt_masks_forged.append(mask_gt[i].squeeze(0).cpu().numpy())
            else:
                pred_masks_auth.append(mask_prob[i])

    # Aggregate
    det_scores = np.concatenate(det_scores) if det_scores else np.array([])
    det_labels = np.concatenate(det_labels) if det_labels else np.array([])
    type_scores_all = np.concatenate(type_scores_list) if type_scores_list else np.zeros((0, 3))
    type_targets_all = np.concatenate(type_targets_list) if type_targets_list else np.zeros((0, 3))
    type_valid_all = np.concatenate(type_valid_list) if type_valid_list else np.zeros((0, 3))
    pred_masks_forged = np.array(pred_masks_forged) if pred_masks_forged else np.zeros((0, 512, 512))
    gt_masks_forged = np.array(gt_masks_forged) if gt_masks_forged else np.zeros((0, 512, 512))
    pred_masks_auth = np.array(pred_masks_auth) if pred_masks_auth else np.zeros((0, 512, 512))

    results = {"val_loss": total_loss / max(n_batches, 1)}

    # Image-level
    if len(det_scores) > 0:
        img_m = compute_image_metrics(det_scores, det_labels)
        results.update({f"img_{k}": v for k, v in img_m.items()})

    # Pixel-level (forged only)
    if len(pred_masks_forged) > 0:
        pix_m = compute_pixel_metrics(pred_masks_forged, gt_masks_forged)
        results.update(pix_m)
    else:
        results["pixel_f1"] = 0.0

    # Type-level
    if len(type_scores_all) > 0:
        type_m = compute_type_metrics(type_scores_all, type_targets_all, type_valid_all)
        results.update(type_m)

    # False alarm area on authentic
    if len(pred_masks_auth) > 0:
        from src.ftt_v2.metrics import compute_false_alarm_area
        results["false_alarm_area"] = compute_false_alarm_area(pred_masks_auth)

    # Validation score for checkpoint selection
    pixel_f1 = results.get("pixel_f1", 0.0)
    image_auc = results.get("img_roc_auc", 0.0)
    type_f1s = [results.get(f"f1_{t}", 0.0) for t in TYPES]
    type_macro_f1 = float(np.nanmean([f for f in type_f1s if not np.isnan(f)])) if type_f1s else 0.0
    results["val_score"] = compute_validation_score(pixel_f1, image_auc, type_macro_f1)

    return results


def train_one_epoch(
    model: ForensicModel,
    train_loader: DataLoader,
    loss_fn: FTTv2Loss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    cfg: dict,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation and AMP."""
    model.train()
    train_cfg = cfg.get("train", {})
    micro_batch = train_cfg.get("micro_batch", 8)
    effective_batch = train_cfg.get("effective_batch", 64)
    accum_steps = max(effective_batch // micro_batch, 1)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    use_amp = (train_cfg.get("amp", "fp16") == "fp16") and device.type == "cuda"

    total_loss = 0.0
    n_steps = 0

    # Assertion: no non-LoRA backbone param has requires_grad=True
    for name, p in model.backbone.named_parameters():
        if p.requires_grad and "lora_" not in name:
            raise RuntimeError(
                f"Non-LoRA backbone param {name} has requires_grad=True! "
                "This would fully fine-tune the backbone."
            )

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        mask_gt = batch["mask"].to(device)
        is_forged = batch["is_forged"].to(device)
        type_targets = batch["types"].to(device)
        type_valid = batch["type_valid"].to(device)

        # Forward with AMP
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(images)
            losses = loss_fn(
                output.mask_logits, mask_gt,
                output.det_logit, is_forged,
                output.type_logits, type_targets, type_valid,
                det_enabled=cfg.get("det", {}).get("enabled", False),
                type_enabled=cfg.get("type_head", {}).get("enabled", False),
            )

        loss = losses["total"]

        # Check for NaN
        if torch.isnan(loss):
            print(f"[WARNING] NaN loss at epoch {epoch}, step {step}. Skipping.")
            optimizer.zero_grad()
            continue

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Optimizer step every accum_steps
        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.trainable_parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_loss += loss.item() * accum_steps  # Undo the /accum in the loss
        n_steps += 1

    return {
        "train_loss": total_loss / max(n_steps, 1),
        "lr": optimizer.param_groups[0]["lr"],
    }


def main():
    parser = argparse.ArgumentParser(description="FTT v2 Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/ftt_v2", help="Output directory")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("train", {})
    seed = train_cfg.get("seed", 42)
    seed_everything(seed)
    run_info = log_run_info(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # Build model
    model = ForensicModel(cfg)
    model.to(device)
    model.param_report()

    # Data
    train_loader, val_loader = build_dataloaders(cfg, model.backbone_info)

    # Optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    accum_steps = max(
        train_cfg.get("effective_batch", 64) // train_cfg.get("micro_batch", 8), 1
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader) // accum_steps)

    # Scaler for AMP
    use_amp = (train_cfg.get("amp", "fp16") == "fp16") and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    # Loss
    loss_cfg = cfg.get("loss", {})
    type_pos_weight = cfg.get("type_head", {}).get("pos_weight", None)
    loss_fn = FTTv2Loss(
        lambda_edge=loss_cfg.get("lambda_edge", 20.0),
        edge_band_px=loss_cfg.get("edge_band_px", 7),
        lambda_det=loss_cfg.get("lambda_det", 1.0),
        lambda_type=loss_cfg.get("lambda_type", 1.0),
        type_pos_weight=type_pos_weight,
        accum_steps=accum_steps,
    ).to(device)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({"config": cfg, "run_info": run_info}, f, indent=2)

    # Resume
    start_epoch = 0
    best_score = 0.0
    if args.resume:
        ckpt_info = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, str(device),
        )
        start_epoch = ckpt_info["epoch"] + 1
        best_score = ckpt_info["best_score"]
        print(f"[RESUME] from epoch {start_epoch}, best_score={best_score:.4f}")

    # Training loop
    epochs = train_cfg.get("epochs", 20)
    eval_every = train_cfg.get("eval_every", 4)

    print(f"\n{'='*60}")
    print(f"Starting training: {epochs} epochs, eval every {eval_every}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            scaler, device, cfg, epoch,
        )
        elapsed = time.time() - t0

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"lr={train_metrics['lr']:.2e} "
            f"time={elapsed:.1f}s"
        )

        # Save last checkpoint
        save_checkpoint(
            str(output_dir / "last.pt"),
            model, optimizer, scheduler, scaler,
            epoch, best_score, cfg,
        )

        # Periodic evaluation
        if (epoch + 1) % eval_every == 0 or (epoch + 1) == epochs:
            val_metrics = validate(model, val_loader, loss_fn, device, cfg)

            print(f"  [VAL] ", end="")
            for k, v in sorted(val_metrics.items()):
                if isinstance(v, float):
                    print(f"{k}={v:.4f} ", end="")
            print()

            # Best checkpoint
            val_score = val_metrics.get("val_score", 0.0)
            if val_score > best_score:
                best_score = val_score
                save_checkpoint(
                    str(output_dir / "best.pt"),
                    model, optimizer, scheduler, scaler,
                    epoch, best_score, cfg,
                )
                print(f"  [BEST] New best score: {best_score:.4f}")

            # Save validation metrics
            with open(output_dir / f"val_metrics_epoch{epoch+1}.json", "w") as f:
                json.dump(val_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training complete. Best score: {best_score:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")

    # Save model via ForensicModel.save() for inference
    model.save(str(output_dir / "model"))


if __name__ == "__main__":
    main()
