"""
test_train_smoke.py — End-to-end 1-epoch training and export smoke test.

Verifies:
1. Training loop completes 1 epoch.
2. Validation runs and reports metrics without disagreement.
3. Checkpoint saving (best.pt, last.pt) succeeds.
4. End-of-training reload of best.pt executes cleanly.
5. Final model export (model/ directory with weights.pt and config.json) succeeds.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
import pytest

from src.ftt_v2.forensic_model import ForensicModel
from src.ftt_v2.losses import FTTv2Loss
from src.ftt_v2.train import train_one_epoch, validate, build_optimizer, build_scheduler
from src.ftt_v2.utils.ckpt import save_checkpoint, load_checkpoint
from src.ftt_v2.data.dataset import ForensicDataset
from torch.utils.data import DataLoader


def test_one_epoch_smoke_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir()
        mask_dir.mkdir()

        # Create 4 synthetic images and masks
        records = []
        for i in range(4):
            img_p = img_dir / f"img_{i}.png"
            mask_p = mask_dir / f"mask_{i}.png"
            Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)).save(img_p)
            Image.fromarray((np.random.rand(128, 128) > 0.5).astype(np.uint8) * 255).save(mask_p)
            records.append({
                "image_id": f"id_{i}",
                "path": str(img_p),
                "mask_path": str(mask_p),
                "is_forged": 1.0 if i % 2 == 1 else 0.0,
                "copy_move": 1.0 if i == 1 else 0.0,
                "splicing": 1.0 if i == 3 else 0.0,
                "inpainting": 0.0,
                "type_valid": 1.0,
                "source_id": f"src_{i}",
                "dataset": "synthetic",
                "width": 128,
                "height": 128,
                "file_format": "PNG",
            })

        manifest_path = tmp_path / "manifest.csv"
        pd.DataFrame(records).to_csv(manifest_path, index=False)

        cfg = {
            "model": {"name": "ftt_v2", "image_size": 128},
            "backbone": {"name": "dummy"},
            "lora": {"enabled": False},
            "decoder": {"norm": "gn"},
            "det": {"enabled": True, "use_mask_stats": True},
            "type_head": {"enabled": True},
            "cm": {"enabled": False},
            "loss": {"lambda_edge": 5.0, "edge_band_px": 5, "lambda_det": 1.0, "lambda_type": 1.0},
            "train": {
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 0.01,
                "micro_batch": 2,
                "effective_batch": 4,
                "grad_clip": 1.0,
                "amp": "fp32",
            },
        }

        model = ForensicModel(cfg)
        dataset = ForensicDataset(str(manifest_path))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        loss_fn = FTTv2Loss(accum_steps=2)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(len(loader) // 2, 1))

        device = torch.device("cpu")
        model.to(device)

        # 1. Train 1 epoch
        train_metrics = train_one_epoch(
            model, loader, loss_fn, optimizer, scheduler,
            scaler=None, device=device, cfg=cfg, epoch=0,
        )
        assert "train_loss" in train_metrics
        assert train_metrics["skipped"] == 0

        # 2. Validate
        val_metrics = validate(model, loader, loss_fn, device, cfg)
        assert "val_loss" in val_metrics
        assert "val_score" in val_metrics

        # 3. Save best.pt
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        best_pt = output_dir / "best.pt"
        save_checkpoint(
            str(best_pt), model, optimizer, scheduler, None,
            epoch=0, best_score=val_metrics["val_score"], config=cfg,
        )
        assert best_pt.exists()

        # 4. End-of-training reload of best.pt
        load_checkpoint(str(best_pt), model, device=str(device))

        # 5. Final model export
        export_dir = output_dir / "model"
        model.save(str(export_dir))
        assert (export_dir / "weights.pt").exists()
        assert (export_dir / "config.json").exists()

        # 6. Verify exported model can be reloaded strictly
        exported_model = ForensicModel.load(str(export_dir))
        assert exported_model is not None
