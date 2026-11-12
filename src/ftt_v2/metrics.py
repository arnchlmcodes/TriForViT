"""
metrics.py — Evaluation metrics for FTT v2.

Image-level: ROC-AUC, precision, recall, F1, FPR@95TPR
Type-level:  per-class AP and F1, confusion matrix
Pixel-level: F1, IoU (forged images only), false-alarm area on authentic
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from . import TYPES


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Image-level metrics
# ---------------------------------------------------------------------------

def compute_image_metrics(
    det_scores: np.ndarray,
    det_labels: np.ndarray,
    tau_det: float = 0.5,
) -> Dict[str, float]:
    det_scores = _to_numpy(det_scores)
    det_labels = _to_numpy(det_labels)
    """Compute image-level detection metrics.

    Parameters
    ----------
    det_scores : (N,)
        Sigmoid probabilities for each image.
    det_labels : (N,)
        Ground-truth binary labels {0, 1}.
    tau_det : float
        Decision threshold.

    Returns
    -------
    metrics : dict
    """
    preds = (det_scores >= tau_det).astype(int)
    tp = ((preds == 1) & (det_labels == 1)).sum()
    fp = ((preds == 1) & (det_labels == 0)).sum()
    fn = ((preds == 0) & (det_labels == 1)).sum()
    tn = ((preds == 0) & (det_labels == 0)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    # ROC-AUC
    if len(np.unique(det_labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(det_labels, det_scores))
        # FPR at 95% TPR
        fpr, tpr, _ = roc_curve(det_labels, det_scores)
        idx = np.searchsorted(tpr, 0.95)
        if idx < len(fpr):
            metrics["fpr_at_95tpr"] = float(fpr[idx])
        else:
            metrics["fpr_at_95tpr"] = float(fpr[-1])
    else:
        metrics["roc_auc"] = float("nan")
        metrics["fpr_at_95tpr"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Type-level metrics
# ---------------------------------------------------------------------------

def compute_type_metrics(
    type_scores: np.ndarray,
    type_targets: np.ndarray,
    type_valid: np.ndarray,
    tau_type: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute per-type AP and F1."""
    type_scores = _to_numpy(type_scores)
    type_targets = _to_numpy(type_targets)
    type_valid = _to_numpy(type_valid)

    if tau_type is None:
        tau_type = [0.5] * len(TYPES)

    metrics = {}
    for i, t in enumerate(TYPES):
        valid = type_valid[:, i].astype(bool)
        if valid.sum() == 0:
            metrics[f"ap_{t}"] = float("nan")
            metrics[f"f1_{t}"] = float("nan")
            continue

        scores_i = type_scores[valid, i]
        targets_i = type_targets[valid, i]

        if len(np.unique(targets_i)) > 1:
            metrics[f"ap_{t}"] = float(average_precision_score(targets_i, scores_i))
        else:
            metrics[f"ap_{t}"] = float("nan")

        preds_i = (scores_i >= tau_type[i]).astype(int)
        metrics[f"f1_{t}"] = float(f1_score(targets_i, preds_i, zero_division=0))

    return metrics


# ---------------------------------------------------------------------------
# Pixel-level metrics (forged images only)
# ---------------------------------------------------------------------------

def compute_pixel_metrics(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute pixel-level F1 and IoU on forged images only."""
    pred_masks = _to_numpy(pred_masks)
    gt_masks = _to_numpy(gt_masks)
    if pred_masks.ndim == 4 and pred_masks.shape[1] == 1:
        pred_masks = pred_masks.squeeze(1)
    if gt_masks.ndim == 4 and gt_masks.shape[1] == 1:
        gt_masks = gt_masks.squeeze(1)

    preds_binary = (pred_masks >= threshold).astype(np.float32)

    # Per-image then average
    f1_list = []
    iou_list = []
    for i in range(len(pred_masks)):
        pred = preds_binary[i].flatten()
        gt = gt_masks[i].flatten()

        tp = (pred * gt).sum()
        fp = (pred * (1 - gt)).sum()
        fn = ((1 - pred) * gt).sum()

        precision = tp / max(tp + fp, 1e-8)
        recall = tp / max(tp + fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        iou = tp / max(tp + fp + fn, 1e-8)

        f1_list.append(f1)
        iou_list.append(iou)

    return {
        "pixel_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "pixel_iou": float(np.mean(iou_list)) if iou_list else 0.0,
    }


def compute_best_threshold_f1(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> Dict[str, float]:
    """Find the threshold that maximizes pixel-F1 (labelled 'optimistic')."""
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    best_f1 = 0.0
    best_thr = 0.5
    for thr in thresholds:
        m = compute_pixel_metrics(pred_masks, gt_masks, threshold=thr)
        if m["pixel_f1"] > best_f1:
            best_f1 = m["pixel_f1"]
            best_thr = thr

    return {"best_pixel_f1": float(best_f1), "best_threshold": float(best_thr)}


def compute_false_alarm_area(
    pred_masks: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Mean fraction of pixels predicted forged on authentic images."""
    preds_binary = (pred_masks >= threshold).astype(np.float32)
    if len(preds_binary) == 0:
        return 0.0
    fractions = [p.mean() for p in preds_binary]
    return float(np.mean(fractions))


# ---------------------------------------------------------------------------
# Validation score for checkpoint selection
# ---------------------------------------------------------------------------

def compute_validation_score(
    pixel_f1: float,
    image_auc: float,
    type_macro_f1: float,
) -> float:
    """Combined validation score: mean of pixel-F1, image-AUC, type-macro-F1.

    Used for checkpoint selection.
    """
    # Handle NaN values
    values = []
    for v in [pixel_f1, image_auc, type_macro_f1]:
        if not np.isnan(v):
            values.append(v)
    return float(np.mean(values)) if values else 0.0
