"""Multi-scale feature alignment for LMDB discrepancy maps."""

from typing import List, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MultiScaleFeatureAlignment(nn.Module):
    """Project discrepancy features to a shared channel and spatial space."""

    def __init__(self, in_channels: Sequence[int], alignment_dim: int = 256):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, alignment_dim, kernel_size=1) for ch in in_channels]
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        # All scales are aligned to the highest discrepancy-map resolution.
        target_size = features[0].shape[-2:]
        aligned = []
        for feature, projection in zip(features, self.projections):
            feature = projection(feature)
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(feature)
        return aligned
