"""Largest connected component selection."""

import cv2
import numpy as np


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Retain only the largest foreground component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)
