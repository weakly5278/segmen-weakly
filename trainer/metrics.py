"""Classification and localization metrics."""

from typing import Dict

import numpy as np


def binary_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute accuracy, sensitivity, precision, and F1-score."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    eps = 1e-8
    return {
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "sensitivity": float(tp / (tp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "f1": float((2 * tp) / (2 * tp + fp + fn + eps)),
    }


def dice_iou(mask_true, mask_pred) -> Dict[str, float]:
    """Compute Dice and IoU for binary localization masks."""
    mask_true = np.asarray(mask_true).astype(bool)
    mask_pred = np.asarray(mask_pred).astype(bool)
    intersection = np.logical_and(mask_true, mask_pred).sum()
    union = np.logical_or(mask_true, mask_pred).sum()
    pred_sum = mask_pred.sum()
    true_sum = mask_true.sum()
    eps = 1e-8
    return {
        "dice": float((2.0 * intersection) / (pred_sum + true_sum + eps)),
        "iou": float(intersection / (union + eps)),
    }
