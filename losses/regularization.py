"""Model regularization utilities."""

import torch
from torch import nn


def l2_regularization(model: nn.Module) -> torch.Tensor:
    """Compute L2 regularization over trainable weight tensors."""
    total = None
    for parameter in model.parameters():
        if parameter.requires_grad:
            value = torch.sum(parameter ** 2)
            total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total
