"""Weighted cross-entropy for image-level supervision."""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Compute class-weighted cross-entropy from image-level labels."""
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=logits.dtype, device=logits.device)
    return F.cross_entropy(logits, labels, weight=weight)
