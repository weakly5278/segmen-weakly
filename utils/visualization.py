"""Visualization utilities for qualitative localization results."""

import os

import matplotlib.pyplot as plt
import numpy as np


def save_qualitative_panel(prior, current, score_map, binary_mask, save_path: str, gt_mask=None):
    """Save a compact prior-current-localization visualization panel."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    images = [prior, current, score_map, binary_mask]
    titles = ["Prior", "Current", "Score Map", "BLM"]
    if gt_mask is not None:
        images.insert(2, gt_mask)
        titles.insert(2, "GT")

    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3))
    if len(images) == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(np.squeeze(image), cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
