"""Spatial smoothness regularization for weak localization maps."""

import torch


def smoothness_loss(score_map: torch.Tensor) -> torch.Tensor:
    """Penalize abrupt differences between neighboring score-map pixels."""
    dx = torch.abs(score_map[:, :, 1:, :] - score_map[:, :, :-1, :]).mean()
    dy = torch.abs(score_map[:, :, :, 1:] - score_map[:, :, :, :-1]).mean()
    return dx + dy
