"""
evaluate.py — FTT v2 evaluation script.

Runs on any manifest. Outputs JSON + CSV + a printed table.

Usage:
    python -m src.ftt_v2.evaluate \
        --model-dir outputs/ftt_v2/model \
        --manifest data/manifests/val.csv \
        --output outputs/ftt_v2/eval_val
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.ftt_v2 import TYPES
from src.ftt_v2.forensic_model import ForensicModel
from src.ftt_v2.metrics import (
    compute_best_threshold_f1,
    compute_false_alarm_area,
    compute_image_metrics,
    compute_pixel_metrics,
    compute_type_metrics,
    compute_validation_score,
)
from src.ftt_v2.data.dataset import ForensicDataset
from src.ftt_v2.data.transforms import ValTransform


def find_optimal_thresholds(
    det_scores: np.ndarray,
    det_labels: np.ndarray,
    type_scores: np.ndarray,
    type_targets: np.ndarray,
    type_valid: np.ndarray,
) -> dict:
    """Find optimal thresholds on validation data.

    tau_det: threshold that maximizes image-level F1.
    tau_type: per-class threshold that maximizes per-class F1.
    """
    from sklearn.metrics import f1_score as sk_f1

    # Detection threshold
    best_f1, best_tau = 0.0, 0.5
    for tau in np.arange(0.05, 0.96, 0.05):
        preds = (det_scores >= tau).astype(int)
        f1 = sk_f1(det_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    tau_type = []
    for i in range(len(TYPES)):
        valid = type_valid[:, i].astype(bool)
        if valid.sum() == 0:
            tau_type.append(0.5)
            continue
        best_ti, best_fi = 0.5, 0.0
        for tau in np.arange(0.05, 0.96, 0.05):
            preds = (type_scores[valid, i] >= tau).astype(int)
            f1 = sk_f1(type_targets[valid, i], preds, zero_division=0)
            if f1 > best_fi:
                best_fi = f1
                best_ti = float(tau)
        tau_type.append(best_ti)

    return {
        "tau_det": best_tau,
        "tau_type": {t: tau_type[i] for i, t in enumerate(TYPES)},
    }


@torch.no_grad()
def run_evaluation(
    model: ForensicModel,
    data_loader: DataLoader,
    device: torch.device,
    thresholds: Optional[dict] = None,
    dataset_name: str = "unknown",
) -> Dict[str, float]:
    """Run full evaluation on a data loader.

    Parameters
    ----------
    model : ForensicModel
    data_loader : DataLoader
    device : torch.device
    thresholds : dict or None
        Pre-computed thresholds from validation. If None, uses 0.5 defaults.
    dataset_name : str
    """
    model.eval()
    tau_det = thresholds.get("tau_det", 0.5) if thresholds else 0.5
    tau_type_dict = thresholds.get("tau_type", {}) if thresholds else {}
    tau_type = [tau_type_dict.get(t, 0.5) for t in TYPES]

    det_scores, det_labels = [], []
    type_scores_list, type_targets_list, type_valid_list = [], [], []
    pred_masks_forged, gt_masks_forged = [], []
    pred_masks_auth = []
    latencies = []

    for batch in data_loader:
        images = batch["image"].to(device)
        mask_gt = batch["mask"]
        is_forged = batch["is_forged"]
        type_targets = batch["types"]
        type_valid = batch["type_valid"]

        t0 = time.time()
        output = model(images)
        latencies.append(time.time() - t0)

        det_prob = torch.sigmoid(output.det_logit).cpu().numpy()
        det_scores.append(det_prob)
        det_labels.append(is_forged.numpy())

        type_prob = torch.sigmoid(output.type_logits).cpu().numpy()
        type_scores_list.append(type_prob)
        type_targets_list.append(type_targets.numpy())
        type_valid_list.append(type_valid.numpy())

        mask_prob = torch.sigmoid(output.mask_logits).squeeze(1).cpu().numpy()
        is_f = is_forged.numpy()
        for i in range(len(is_f)):
            if is_f[i] > 0.5:
                pred_masks_forged.append(mask_prob[i])
                gt_masks_forged.append(mask_gt[i].squeeze(0).numpy())
            else:
                pred_masks_auth.append(mask_prob[i])

    # Aggregate
    det_scores = np.concatenate(det_scores)
    det_labels = np.concatenate(det_labels)
    type_scores_all = np.concatenate(type_scores_list) if type_scores_list else np.zeros((0, 3))
    type_targets_all = np.concatenate(type_targets_list) if type_targets_list else np.zeros((0, 3))
    type_valid_all = np.concatenate(type_valid_list) if type_valid_list else np.zeros((0, 3))

    results = {"dataset": dataset_name, "n_images": len(det_scores)}

    # Image-level
    img_m = compute_image_metrics(det_scores, det_labels, tau_det)
    results.update({f"img_{k}": v for k, v in img_m.items()})

    # Pixel-level (forged only)
    if len(pred_masks_forged) > 0:
        pred_arr = np.array(pred_masks_forged)
        gt_arr = np.array(gt_masks_forged)
        pix_m = compute_pixel_metrics(pred_arr, gt_arr)
        results.update(pix_m)
        best_m = compute_best_threshold_f1(pred_arr, gt_arr)
        results.update(best_m)
    else:
        results["pixel_f1"] = 0.0
        results["pixel_iou"] = 0.0

    # False alarm on authentic
    if len(pred_masks_auth) > 0:
        results["false_alarm_area"] = compute_false_alarm_area(
            np.array(pred_masks_auth)
        )
    else:
        results["false_alarm_area"] = 0.0

    # Type-level
    if len(type_scores_all) > 0:
        type_m = compute_type_metrics(
            type_scores_all, type_targets_all, type_valid_all, tau_type,
        )
        results.update(type_m)

    # Timing
    results["median_latency_ms"] = float(np.median(latencies) * 1000) if latencies else 0.0

    # Param counts
    report = model.param_report()
    results["params_frozen"] = report.get("backbone_frozen", 0)
    results["params_lora"] = report.get("backbone_lora", 0)
    results["params_trainable"] = report.get("trainable", 0)
    results["params_total"] = report.get("total", 0)

    return results


def print_results_table(results: dict) -> None:
    """Pretty-print results."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results: {results.get('dataset', '?')}")
    print(f"  Images: {results.get('n_images', 0)}")
    print(f"{'='*60}")
    for k, v in sorted(results.items()):
        if k in ("dataset",):
            continue
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>10.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="FTT v2 Evaluation")
    parser.add_argument("--model-dir", type=str, required=True, help="Model directory")
    parser.add_argument("--manifest", type=str, required=True, help="Manifest CSV path")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--thresholds", type=str, default=None, help="Path to thresholds.json")
    parser.add_argument("--dataset-name", type=str, default="eval", help="Dataset name for reporting")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--find-thresholds", action="store_true",
                        help="Find optimal thresholds (only on validation data)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ForensicModel.load(args.model_dir, device=str(device))
    model.to(device)
    model.eval()

    # Data
    image_size = model.image_size
    mean = model.backbone_info.get("mean", [0.485, 0.456, 0.406])
    std = model.backbone_info.get("std", [0.229, 0.224, 0.225])
    transform = ValTransform(image_size=image_size, mean=mean, std=std)
    dataset = ForensicDataset(args.manifest, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Load or find thresholds
    thresholds = None
    if args.thresholds:
        with open(args.thresholds) as f:
            thresholds = json.load(f)
    elif args.find_thresholds:
        print("[INFO] Finding optimal thresholds on this data...")
        # Need a quick pass
        det_s, det_l, ts, tt, tv = [], [], [], [], []
        for batch in loader:
            images = batch["image"].to(device)
            with torch.no_grad():
                output = model(images)
            det_s.append(torch.sigmoid(output.det_logit).cpu().numpy())
            det_l.append(batch["is_forged"].numpy())
            ts.append(torch.sigmoid(output.type_logits).cpu().numpy())
            tt.append(batch["types"].numpy())
            tv.append(batch["type_valid"].numpy())

        thresholds = find_optimal_thresholds(
            np.concatenate(det_s), np.concatenate(det_l),
            np.concatenate(ts), np.concatenate(tt), np.concatenate(tv),
        )
        print(f"  Found thresholds: {thresholds}")

    # Run evaluation
    results = run_evaluation(model, loader, device, thresholds, args.dataset_name)
    print_results_table(results)

    # Save outputs
    output_dir = Path(args.output) if args.output else Path(args.model_dir) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"results_{args.dataset_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save as CSV too
    df = pd.DataFrame([results])
    df.to_csv(output_dir / f"results_{args.dataset_name}.csv", index=False)

    if thresholds:
        with open(output_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
