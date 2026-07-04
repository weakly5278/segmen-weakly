"""Morphological operations used for BLM refinement."""

import cv2
import numpy as np


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components with area below ``min_area``."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    refined = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            refined[labels == label_id] = 1
    return refined


def morphological_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Connect nearby fragments and fill small gaps in a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
