"""
train.py - Training script for Three-Stream ViT

Hierarchical training strategy:
  Stage 1: Binary Forgery Detection  (Authentic vs Tampered)
  Stage 2: Forgery-Type Classification  (Copy-Move / Splicing / Inpainting)

Usage:
    python src/train.py --data_dir data/raw/CASIA_v2 --epochs 50
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import ThreeStreamViT
from data_loader import create_split_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train Three-Stream ViT")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (e.g. data/raw/CASIA_v2)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--save_dir", type=str, default="results/checkpoints")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (default: on when CUDA is available)")
    parser.add_argument("--no_amp", dest="amp", action="store_false",
                        help="Disable automatic mixed precision")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion_det, criterion_cls, device, scaler, use_amp):
    """Train for one epoch with joint detection + classification loss."""
    model.train()
    total_loss = 0.0
    correct_det = 0
    total = 0

    for images, binary_labels, forgery_labels in loader:
        images = images.to(device, non_blocking=True)
        binary_labels = binary_labels.to(device, non_blocking=True)
        forgery_labels = forgery_labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")
        with torch.autocast(device_type=device_type, enabled=use_amp):
            det_logits, cls_logits = model(images)

            loss_det = criterion_det(det_logits, binary_labels)
            # Only compute classification loss on tampered samples
            tampered_mask = binary_labels == 1
            if tampered_mask.sum() > 0:
                # Subtract 1 because tampered forgery_labels are 1, 2, 3 but model outputs 3 logits (0, 1, 2)
                loss_cls = criterion_cls(cls_logits[tampered_mask], forgery_labels[tampered_mask] - 1)
            else:
                loss_cls = torch.tensor(0.0, device=device)

            loss = loss_det + 0.5 * loss_cls  # weighted sum

        if use_amp and device_type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = det_logits.argmax(dim=1)
        correct_det += (preds == binary_labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct_det / total if total > 0 else 0
    return avg_loss, accuracy

@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion_det, criterion_cls, device, use_amp):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct_det = 0
    total = 0
    device_type = "cuda" if "cuda" in device else ("mps" if "mps" in device else "cpu")

    for images, binary_labels, forgery_labels in loader:
        images = images.to(device, non_blocking=True)
        binary_labels = binary_labels.to(device, non_blocking=True)
        forgery_labels = forgery_labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device_type, enabled=use_amp):
            det_logits, cls_logits = model(images)

            loss_det = criterion_det(det_logits, binary_labels)
            tampered_mask = binary_labels == 1
            if tampered_mask.sum() > 0:
                loss_cls = criterion_cls(cls_logits[tampered_mask], forgery_labels[tampered_mask] - 1)
            else:
                loss_cls = torch.tensor(0.0, device=device)

            loss = loss_det + 0.5 * loss_cls

        total_loss += loss.item() * images.size(0)
        preds = det_logits.argmax(dim=1)
        correct_det += (preds == binary_labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct_det / total if total > 0 else 0
    return avg_loss, accuracy


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    use_amp = args.amp and (args.device.startswith("cuda") or args.device.startswith("mps"))

    print(f"[INFO] Device: {args.device}  (CUDA available: {torch.cuda.is_available()})")
    print(f"[INFO] Mixed precision (AMP): {'on' if use_amp else 'off'}")
    print(f"[INFO] Loading data from: {args.data_dir}")

    train_loader, val_loader, test_loader = create_split_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )

    model = ThreeStreamViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
    ).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion_det = nn.CrossEntropyLoss()
    criterion_cls = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"[INFO] Starting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion_det, criterion_cls,
            args.device, scaler, use_amp,
        )
        val_loss, val_acc = evaluate_one_epoch(
            model, val_loader, criterion_det, criterion_cls, args.device, use_amp
        )
        scheduler.step()

        print(f"Epoch [{epoch}/{args.epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": train_loss,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(args.save_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()