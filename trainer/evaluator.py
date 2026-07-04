"""Evaluation and inference routines for LMCF-Net."""

import os
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from postprocessing import refine_blm
from .metrics import binary_classification_metrics, dice_iou


@torch.no_grad()
def evaluate(model, loader, device, cfg, save_outputs: bool = False, output_dir: str = None) -> Dict[str, float]:
    """Evaluate classification and localization performance."""
    model.eval()
    if save_outputs and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    y_true, y_pred = [], []
    dice_scores, iou_scores = [], []

    for batch_idx, batch in enumerate(loader):
        prior = batch["prior"].to(device, non_blocking=True)
        current = batch["current"].to(device, non_blocking=True)
        labels = batch["label"].cpu().numpy()

        outputs = model(prior, current)
        probs = F.softmax(outputs["logits"], dim=1)
        pred = torch.argmax(probs, dim=1).cpu().numpy()
        cancer_probs = probs[:, 1].detach().cpu().numpy()
        score_maps = outputs["score_map"].detach().cpu().numpy()[:, 0]

        y_true.extend(labels.tolist())
        y_pred.extend(pred.tolist())

        for i, score_map in enumerate(score_maps):
            gate_prob = cancer_probs[i] if cfg.use_classification_gate else None
            refined = refine_blm(
                score_map,
                threshold=cfg.threshold,
                min_component_area=cfg.min_component_area,
                closing_kernel=cfg.closing_kernel,
                cancer_probability=gate_prob,
                cancer_probability_threshold=cfg.cancer_probability_threshold,
            )

            if "mask" in batch:
                gt_mask = batch["mask"][i].numpy()[0]
                metrics = dice_iou(gt_mask > 0.5, refined > 0)
                dice_scores.append(metrics["dice"])
                iou_scores.append(metrics["iou"])

            if save_outputs and output_dir is not None:
                name = _output_name(batch, batch_idx, i)
                cv2.imwrite(os.path.join(output_dir, f"{name}_score.png"), (score_map * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(output_dir, f"{name}_blm.png"), (refined * 255).astype(np.uint8))

    cls_metrics = binary_classification_metrics(y_true, y_pred)
    cls_metrics["dice"] = float(np.mean(dice_scores)) if dice_scores else 0.0
    cls_metrics["iou"] = float(np.mean(iou_scores)) if iou_scores else 0.0
    return cls_metrics


def _output_name(batch, batch_idx: int, item_idx: int) -> str:
    """Create a stable output filename for saved masks."""
    if "patient_id" in batch:
        return str(batch["patient_id"][item_idx])
    return f"case_{batch_idx:04d}_{item_idx:02d}"
