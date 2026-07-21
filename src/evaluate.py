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
    parser.add_argument("--gradcam", action="store_true",
                        help="Generate and save Grad-CAM visualizations for first few tampered images")
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

def compute_gradcam(model, image, target_class=None, device="cpu"):
    """
    Compute Grad-CAM heatmap for the given image.

    Args:
        model: ThreeStreamViT model.
        image: Single image tensor (1, C, H, W).
        target_class: Target class index for Grad-CAM. If None, uses predicted class.
        device: Device to run inference on.

    Returns:
        heatmap: numpy array (H, W) normalised to [0, 1].
        pred_class: Predicted class index.
    """
    model.eval()
    
    # Store activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        # output is shape (B, num_patches + 1, D)
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        # grad_output is a tuple; first element is gradient w.r.t. output
        gradients.append(grad_output[0].detach())

    # Get the last layer of the transformer encoder
    target_layer = model.transformer.encoder.layers[-1]
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    # Enable gradient computation
    image = image.to(device).requires_grad_(True)
    det_logits, _ = model(image)

    pred_class = det_logits.argmax(dim=1).item()
    if target_class is None:
        target_class = pred_class

    # Target class score
    score = det_logits[0, target_class]
    
    # Zero gradients and backprop
    model.zero_grad()
    score.backward()

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    if not activations or not gradients:
        raise RuntimeError("Failed to capture activations or gradients. Check hook registration.")

    # Shape: (1, num_patches + 1, D)
    act = activations[0]
    grad = gradients[0]

    # Exclude CLS token: shape (1, N, D)
    act = act[:, 1:, :]
    grad = grad[:, 1:, :]

    # Compute channel weights: mean of gradients over tokens (N)
    weights = grad.mean(dim=1)  # (1, D)
    
    # Weighted sum of activations
    cam = (weights.unsqueeze(1) * act).sum(dim=-1)  # (1, N)
    cam = F.relu(cam)  # Apply ReLU

    # Normalise
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    cam = cam.squeeze(0).cpu().numpy()  # (N,)

    # Reshape to 2D grid
    num_patches = cam.shape[0]
    grid_size = int(np.sqrt(num_patches))
    heatmap_grid = cam.reshape(grid_size, grid_size)

    # Upscale to original image size (img_size, img_size)
    from PIL import Image
    heatmap_img = Image.fromarray((heatmap_grid * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((image.shape[2], image.shape[3]), resample=Image.BILINEAR)
    heatmap = np.array(heatmap_img) / 255.0

    return heatmap, pred_class


def save_gradcam_visualization(original_img_path, heatmap, pred_class, output_path):
    """Overlay heatmap on original image and save it."""
    import matplotlib.pyplot as plt
    from PIL import Image

    orig_img = Image.open(original_img_path).convert("RGB").resize((224, 224))
    orig_np = np.array(orig_img) / 255.0

    plt.figure(figsize=(10, 5))
    
    # Original
    plt.subplot(1, 2, 1)
    plt.imshow(orig_np)
    plt.title("Original Image")
    plt.axis("off")

    # Grad-CAM overlay
    plt.subplot(1, 2, 2)
    plt.imshow(orig_np)
    # Overlay heatmap with transparency
    plt.imshow(heatmap, cmap="jet", alpha=0.5)
    plt.title(f"Grad-CAM (Pred: {'Tampered' if pred_class == 1 else 'Authentic'})")
    plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved Grad-CAM visualization to: {output_path}")


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

    if args.gradcam:
        print("\n[INFO] Generating Grad-CAM visualizations...")
        dataset = loader.dataset
        # Find first 3 tampered images
        tampered_count = 0
        for i in range(len(dataset)):
            img_path, binary_label, forgery_label = dataset.samples[i]
            if binary_label == 1: # Tampered
                # Get raw tensor image
                image, _, _ = dataset[i]
                image = image.unsqueeze(0).to(args.device) # Add batch dimension
                
                heatmap, pred_class = compute_gradcam(model, image, target_class=1, device=args.device)
                
                output_filename = f"gradcam_tampered_{tampered_count}.png"
                output_path = os.path.join("results", "figures", output_filename)
                
                save_gradcam_visualization(img_path, heatmap, pred_class, output_path)
                
                tampered_count += 1
                if tampered_count >= 3:
                    break


if __name__ == "__main__":
    main()
