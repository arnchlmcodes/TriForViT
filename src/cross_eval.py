"""
cross_eval.py - Cross-Dataset Evaluation Pipeline

This script evaluates a model trained on a source dataset (e.g. CASIA v2)
on a different target dataset (e.g. COVERAGE or Columbia) to test
cross-dataset generalizability and domain transfer.

Usage:
    python src/cross_eval.py --checkpoint results/checkpoints/model_final.pth --test_dir data/raw/Columbia
"""

import argparse
import sys
import os
import torch
import numpy as np

# Ensure src path is accessible
sys.path.insert(0, os.path.dirname(__file__))

from model import ThreeStreamViT
from data_loader import create_dataloader
from evaluate import evaluate, print_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Dataset Evaluation Pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Path to target evaluation dataset directory")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 65)
    print("  Cross-Dataset Evaluation (Domain Generalization)")
    print("=" * 65)
    print(f"[INFO] Target Test Directory: {args.test_dir}")
    print(f"[INFO] Checkpoint Path      : {args.checkpoint}")
    print(f"[INFO] Device               : {args.device}")

    # Load Model
    model = ThreeStreamViT(img_size=args.img_size).to(args.device)
    
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return
        
    state_dict = torch.load(args.checkpoint, map_location=args.device)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    
    model.eval()

    # Load target dataloader
    try:
        loader = create_dataloader(
            args.test_dir,
            batch_size=args.batch_size,
            is_train=False,
            img_size=args.img_size
        )
    except Exception as e:
        print(f"[ERROR] Failed to load data from {args.test_dir}: {e}")
        return

    if len(loader.dataset) == 0:
        print(f"[ERROR] No images found in evaluation directory: {args.test_dir}")
        return

    print(f"[INFO] Loaded {len(loader.dataset)} target test images.")
    
    # Run evaluation
    det_preds, det_labels, cls_preds, cls_labels = evaluate(model, loader, args.device)
    
    print("\n--- Domain Generalization Performance ---")
    print_metrics(det_preds, det_labels, cls_preds, cls_labels)
    print("=" * 65)

if __name__ == "__main__":
    main()
