"""
test_overfit.py — Phase 3 Overfit Gate Test for FTT v2.

Per Section 10 (Phase 3 Gate):
"Overfit test: 32 samples (mixed forged and authentic),
loss falls by at least 80% and pixel-F1 on those 32 is at least 0.8."
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F

from src.ftt_v2.forensic_model import ForensicModel
from src.ftt_v2.losses import FTTv2Loss
from src.ftt_v2.metrics import compute_pixel_metrics


@pytest.mark.filterwarnings("ignore")
def test_overfit_32_samples():
    """Verify model can overfit a tiny batch of 32 mixed forged/authentic samples."""
    torch.manual_seed(42)
    img_size = 128

    cfg = {
        "model": {"name": "ftt_v2", "image_size": img_size, "types": ["copy_move", "splicing", "inpainting"]},
        "backbone": {"name": "dummy"},
        "lora": {"enabled": False},
        "decoder": {"norm": "gn"},
        "det": {"enabled": True, "use_mask_stats": True},
        "type_head": {"enabled": True},
        "cm": {"enabled": False},
        "loss": {"lambda_edge": 5.0, "edge_band_px": 5, "lambda_det": 1.0, "lambda_type": 1.0},
    }

    model = ForensicModel(cfg)

    # 32 synthetic samples: 16 authentic, 16 forged
    B = 32
    images = torch.randn(B, 3, img_size, img_size)
    masks = torch.zeros(B, 1, img_size, img_size)
    is_forged = torch.zeros(B)
    types = torch.zeros(B, 3)
    type_valid = torch.ones(B, 3)

    for i in range(16, 32):
        is_forged[i] = 1.0
        y1, y2 = 30 + (i - 16) * 2, 60 + (i - 16) * 2
        x1, x2 = 30 + (i - 16) * 2, 60 + (i - 16) * 2
        masks[i, 0, y1:y2, x1:x2] = 1.0
        images[i, :, y1:y2, x1:x2] += 2.0
        types[i, (i - 16) % 3] = 1.0

    # Cache frozen dummy backbone output once
    with torch.no_grad():
        cached_backbone_out = model.backbone(pixel_values=images, output_hidden_states=False)

    class CachedBackbone(torch.nn.Module):
        def forward(self, pixel_values, output_hidden_states=False):
            return cached_backbone_out

    model.backbone = CachedBackbone()

    criterion = FTTv2Loss(lambda_edge=5.0, edge_band_px=5, lambda_det=1.0, lambda_type=1.0)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=5e-3, weight_decay=0.0)

    model.train()
    initial_loss = None
    final_loss = None

    # 3. Train for 280 steps (plan specifies 300 steps)
    for step in range(280):
        optimizer.zero_grad()
        out = model(images)
        loss_dict = criterion(
            mask_logits=out.mask_logits,
            mask_gt=masks,
            det_logit=out.det_logit,
            is_forged=is_forged,
            type_logits=out.type_logits,
            type_targets=types,
            type_valid=type_valid,
        )
        total_loss = loss_dict["total"]
        if step == 0:
            initial_loss = total_loss.item()
        final_loss = total_loss.item()

        total_loss.backward()
        optimizer.step()

    loss_reduction = (initial_loss - final_loss) / initial_loss
    assert loss_reduction >= 0.80, (
        f"Loss reduction was {loss_reduction * 100:.1f}% (expected >= 80%). Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    )

    model.eval()
    with torch.no_grad():
        out = model(images)
        pred_probs = torch.sigmoid(out.mask_logits)
        pred_bin = (pred_probs >= 0.5).float()

        forged_preds = pred_bin[16:]
        forged_targets = masks[16:]
        metrics = compute_pixel_metrics(forged_preds, forged_targets)
        pixel_f1 = metrics["pixel_f1"]

    assert pixel_f1 >= 0.80, f"Pixel F1 was {pixel_f1:.4f} (expected >= 0.80)"
