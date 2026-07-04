"""Thresholding utilities for localization maps."""

import numpy as np


def threshold_map(score_map: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a continuous score map into a binary mask."""
    return (score_map >= threshold).astype(np.uint8)
