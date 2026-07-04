"""Learnable Multi-scale Discrepancy Block (LMDB)."""

import torch
from torch import nn


class LearnableMultiScaleDiscrepancyBlock(nn.Module):
    """Learn temporal discrepancy features from prior-current representations.

    Instead of using fixed subtraction, LMDB concatenates the prior and current
    feature maps along the channel dimension and learns a discrepancy mapping
    through stacked convolution, normalization, and nonlinearity.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, prior_feature: torch.Tensor, current_feature: torch.Tensor) -> torch.Tensor:
        # Channel-wise concatenation doubles the channel dimension and preserves
        # spatial correspondence between the prior and current feature maps.
        x = torch.cat([prior_feature, current_feature], dim=1)
        return self.activation(self.main(x) + self.skip(x))
