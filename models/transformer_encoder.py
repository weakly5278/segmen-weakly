"""Shared hierarchical transformer encoder for prior-current mammograms."""

from typing import List, Tuple

import torch
from torch import nn

from .patch_embedding import PatchEmbedding
from .transformer_block import SpatialTransformerBlock


class EncoderStage(nn.Module):
    """One hierarchical encoder stage with optional downsampling."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        depth: int,
        mlp_ratio: float,
        dropout: float,
        downsample: bool,
    ):
        super().__init__()
        self.downsample = nn.Identity() if not downsample else nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
        )
        dim = in_dim if not downsample else out_dim
        self.blocks = nn.Sequential(
            *[
                SpatialTransformerBlock(
                    dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        return self.blocks(x)


class SharedHierarchicalTransformerEncoder(nn.Module):
    """Four-stage hierarchical encoder used for both prior and current images."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dims: Tuple[int, int, int, int] = (64, 128, 320, 512),
        num_heads: Tuple[int, int, int, int] = (2, 4, 8, 8),
        depths: Tuple[int, int, int, int] = (1, 1, 1, 1),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dims[0], patch_size=4)
        self.stage1 = EncoderStage(embed_dims[0], embed_dims[0], num_heads[0], depths[0], mlp_ratio, dropout, False)
        self.stage2 = EncoderStage(embed_dims[0], embed_dims[1], num_heads[1], depths[1], mlp_ratio, dropout, True)
        self.stage3 = EncoderStage(embed_dims[1], embed_dims[2], num_heads[2], depths[2], mlp_ratio, dropout, True)
        self.stage4 = EncoderStage(embed_dims[2], embed_dims[3], num_heads[3], depths[3], mlp_ratio, dropout, True)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Returned features correspond to H/4, H/8, H/16, and H/32 scales.
        x = self.patch_embed(x)
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]
