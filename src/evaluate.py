"""
evaluate.py - Evaluation & Explainability for Three-Stream ViT

Computes:
  - Binary detection metrics  (accuracy, precision, recall, F1)
  - Forgery-type classification metrics
  - Grad-CAM attention heatmaps for forensic explainability

Usage:
    python src/evaluate.py --data_dir data/raw/CASIA_v2 --checkpoint results/checkpoints/model_final.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import ThreeStreamViT
from data_loader import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Three-Stream ViT")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to evaluation dataset root")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    """Run inference and collect predictions."""
    model.eval()
    all_det_preds, all_det_labels = [], []
    all_cls_preds, all_cls_labels = [], []

    for images, binary_labels, forgery_labels in loader:
        images = images.to(device)
        det_logits, cls_logits = model(images)

        det_preds = det_logits.argmax(dim=1).cpu()
        cls_preds = cls_logits.argmax(dim=1).cpu()

        all_det_preds.extend(det_preds.numpy())
        all_det_labels.extend(binary_labels.numpy())

        # Only record classification for tampered samples
        tampered_mask = binary_labels == 1
        if tampered_mask.sum() > 0:
            all_cls_preds.extend(cls_preds[tampered_mask].numpy())
            all_cls_labels.extend(forgery_labels[tampered_mask].numpy())

    return (
        np.array(all_det_preds), np.array(all_det_labels),
        np.array(all_cls_preds), np.array(all_cls_labels),
    )


def print_metrics(det_preds, det_labels, cls_preds, cls_labels):
    """Print evaluation metrics."""
    print("=" * 60)
    print("BINARY DETECTION (Authentic vs Tampered)")
    print("=" * 60)
    print(classification_report(
        det_labels, det_preds,
        target_names=["Authentic", "Tampered"],
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(det_labels, det_preds))

    if len(cls_preds) > 0:
        print("\n" + "=" * 60)
        print("FORGERY TYPE CLASSIFICATION")
        print("=" * 60)
        print(classification_report(
            cls_labels, cls_preds,
            target_names=["Copy-Move", "Splicing", "Inpainting"],
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(cls_labels, cls_preds))


# ---------------------------------------------------------------------------
# Grad-CAM (placeholder — to be fully implemented during experiments)
# ---------------------------------------------------------------------------

def compute_gradcam(model, image, target_class=None):
    """
    Compute Grad-CAM heatmap for the given image.

    This is a placeholder skeleton. Full implementation requires hooking
    into an intermediate layer of the Transformer encoder.

    Args:
        model: ThreeStreamViT model.
        image: Single image tensor (1, C, H, W).
        target_class: Target class index for Grad-CAM. If None, uses predicted class.

    Returns:
        heatmap: numpy array (H, W) normalised to [0, 1].
    """
    # TODO: Register forward/backward hooks on the last transformer layer
    # TODO: Compute gradients w.r.t. target class
    # TODO: Weight feature maps by mean gradient → ReLU → normalise
    raise NotImplementedError(
        "Grad-CAM implementation pending. "
        "See Selvaraju et al. (2017) for reference."
    )


def main():
    args = parse_args()

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    model = ThreeStreamViT(img_size=args.img_size).to(args.device)
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    loader = create_dataloader(
        args.data_dir,
        batch_size=args.batch_size,
        is_train=False,
        img_size=args.img_size,
    )

    det_preds, det_labels, cls_preds, cls_labels = evaluate(model, loader, args.device)
    print_metrics(det_preds, det_labels, cls_preds, cls_labels)


if __name__ == "__main__":
    main()
