"""Weak localization head for producing lesion score maps."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LocalizationHead(nn.Module):
    """Generate a weak localization score map from fused temporal features."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]):
        # score_low is used by the smoothness term at the aligned feature scale;
        # score_map is upsampled to the original mammogram resolution.
        score_low = torch.sigmoid(self.conv(x))
        score_map = F.interpolate(score_low, size=output_size, mode="bilinear", align_corners=False)
        return score_low, score_map
