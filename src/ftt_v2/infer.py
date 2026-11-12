"""
infer.py — FTT v2 inference script.

Usage:
    python -m src.ftt_v2.infer \
        --model-dir outputs/ftt_v2/model \
        --image path/to/image.jpg \
        --thresholds outputs/ftt_v2/eval/thresholds.json \
        --output outputs/ftt_v2/predictions

Provides both CLI and programmatic ``predict()`` API.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

from src.ftt_v2 import TYPES
from src.ftt_v2.forensic_model import ForensicModel
from src.ftt_v2.data.transforms import ValTransform


def predict(
    model: ForensicModel,
    image_path: str,
    thresholds: Optional[dict] = None,
    device: Optional[torch.device] = None,
    output_dir: Optional[str] = None,
) -> dict:
    """Run inference on a single image.

    Parameters
    ----------
    model : ForensicModel
        A loaded model in eval mode.
    image_path : str
        Path to the input image.
    thresholds : dict or None
        Contains ``tau_det`` and ``tau_type`` keys.
    device : torch.device or None
    output_dir : str or None
        If provided, saves mask and overlay PNGs.

    Returns
    -------
    result : dict
        JSON-serializable result.
    """
    if device is None:
        device = next(model.parameters()).device

    tau_det = thresholds.get("tau_det", 0.5) if thresholds else 0.5
    tau_type_dict = thresholds.get("tau_type", {}) if thresholds else {}
    tau_type = [tau_type_dict.get(t, 0.5) for t in TYPES]

    # Load and preprocess image
    image_size = model.image_size
    mean = model.backbone_info.get("mean", [0.485, 0.456, 0.406])
    std = model.backbone_info.get("std", [0.229, 0.224, 0.225])
    transform = ValTransform(image_size=image_size, mean=mean, std=std)

    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    dummy_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.float32)

    img_tensor, _ = transform(img_np, dummy_mask)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)

    p_forged = float(torch.sigmoid(output.det_logit).item())
    forged = p_forged >= tau_det

    type_probs_raw = torch.sigmoid(output.type_logits).squeeze(0).cpu().numpy()
    type_probs = {t: float(type_probs_raw[i]) for i, t in enumerate(TYPES)}

    # Predicted types (only when forged)
    predicted_types = []
    if forged:
        for i, t in enumerate(TYPES):
            if type_probs_raw[i] >= tau_type[i]:
                predicted_types.append(t)

    # Mask
    mask_prob = torch.sigmoid(output.mask_logits).squeeze().cpu().numpy()
    if not forged:
        mask_prob = np.zeros_like(mask_prob)

    result = {
        "forged": forged,
        "p_forged": p_forged,
        "type_probs": type_probs,
        "predicted_types": predicted_types,
        "mask_png": None,
        "overlay_png": None,
    }

    # Save outputs
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(image_path).stem

        # Mask PNG
        mask_uint8 = (mask_prob * 255).astype(np.uint8)
        mask_path = output_dir / f"{stem}_mask.png"
        Image.fromarray(mask_uint8).save(str(mask_path))
        result["mask_png"] = str(mask_path)

        # Overlay PNG
        overlay = _create_overlay(img_pil, mask_prob, image_size)
        overlay_path = output_dir / f"{stem}_overlay.png"
        overlay.save(str(overlay_path))
        result["overlay_png"] = str(overlay_path)

    return result


def _create_overlay(
    img_pil: Image.Image,
    mask_prob: np.ndarray,
    image_size: int,
) -> Image.Image:
    """Create a red-overlay visualization of the mask on the image."""
    img_resized = img_pil.resize((image_size, image_size), Image.BILINEAR)
    img_arr = np.array(img_resized).astype(np.float32)

    # Red overlay where mask > 0.5
    mask_binary = (mask_prob > 0.5).astype(np.float32)
    alpha = 0.4

    overlay = img_arr.copy()
    overlay[:, :, 0] = np.where(
        mask_binary > 0,
        img_arr[:, :, 0] * (1 - alpha) + 255 * alpha,
        img_arr[:, :, 0],
    )
    overlay[:, :, 1] = np.where(
        mask_binary > 0,
        img_arr[:, :, 1] * (1 - alpha),
        img_arr[:, :, 1],
    )
    overlay[:, :, 2] = np.where(
        mask_binary > 0,
        img_arr[:, :, 2] * (1 - alpha),
        img_arr[:, :, 2],
    )

    return Image.fromarray(overlay.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="FTT v2 Inference")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to image or directory of images")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to thresholds.json")
    parser.add_argument("--output", type=str, default="outputs/ftt_v2/predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ForensicModel.load(args.model_dir, device=str(device))
    model.to(device)
    model.eval()

    # Load thresholds
    thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)

    # Process images
    image_path = Path(args.image)
    if image_path.is_dir():
        image_files = sorted([
            p for p in image_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        ])
    else:
        image_files = [image_path]

    results = []
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        result = predict(model, str(img_path), thresholds, device, args.output)
        result["image"] = str(img_path)
        results.append(result)

        status = "FORGED" if result["forged"] else "AUTHENTIC"
        print(f"  → {status} (p={result['p_forged']:.3f})")
        if result["predicted_types"]:
            print(f"  → Types: {', '.join(result['predicted_types'])}")

    # Save all results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
