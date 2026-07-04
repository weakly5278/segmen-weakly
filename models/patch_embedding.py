"""Patch embedding layer for mammogram inputs."""

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert a full-resolution mammogram into patch-level feature maps.

    A convolution with stride equal to the patch size reduces the input spatial
    resolution from ``H x W`` to ``H/patch_size x W/patch_size`` while expanding
    the channel dimension.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 64, patch_size: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.proj(x)))
