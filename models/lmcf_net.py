"""LMCF-Net: weakly supervised lesion localization from prior-current mammograms."""

from typing import Dict

import torch
from torch import nn

from .classifier_head import ClassificationHead
from .feature_alignment import MultiScaleFeatureAlignment
from .lmdb import LearnableMultiScaleDiscrepancyBlock
from .localization_head import LocalizationHead
from .tcfm import TemporalCrossAttentionFusionModule
from .transformer_encoder import SharedHierarchicalTransformerEncoder


class LMCFNet(nn.Module):
    """Learnable Multi-scale Discrepancy and Cross-attention Fusion Network.

    The model accepts a prior mammogram and a current mammogram. A shared
    hierarchical transformer encoder extracts multi-scale features from both
    examinations. Stage-wise LMDB modules learn temporal discrepancy features,
    which are aligned and fused by TCFM before branching into classification
    and localization heads.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dims=(64, 128, 320, 512),
        num_heads=(2, 4, 8, 8),
        depths=(1, 1, 1, 1),
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        alignment_dim: int = 256,
        attention_dim: int = 256,
    ):
        super().__init__()

        # The same encoder instance is applied to both prior and current
        # mammograms, enforcing a common feature space for temporal comparison.
        self.encoder = SharedHierarchicalTransformerEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # One LMDB is used per encoder stage to learn scale-specific temporal
        # discrepancy representations.
        self.lmdb = nn.ModuleList(
            [LearnableMultiScaleDiscrepancyBlock(ch, ch) for ch in embed_dims]
        )

        self.alignment = MultiScaleFeatureAlignment(embed_dims, alignment_dim)
        self.tcfm = TemporalCrossAttentionFusionModule(
            alignment_dim=alignment_dim,
            attention_dim=attention_dim,
            num_heads=max(1, min(8, attention_dim // 32)),
            dropout=dropout,
        )
        self.classifier = ClassificationHead(alignment_dim, num_classes=num_classes)
        self.localizer = LocalizationHead(alignment_dim)

    def forward(self, prior: torch.Tensor, current: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run LMCF-Net on paired mammograms.

        Args:
            prior: Tensor of shape ``B x 1 x H x W``.
            current: Tensor of shape ``B x 1 x H x W``.

        Returns:
            Dictionary containing image-level logits, localization score maps,
            fused features, discrepancy features, and aligned features.
        """
        image_size = prior.shape[-2:]

        prior_features = self.encoder(prior)
        current_features = self.encoder(current)

        # Stage-wise temporal discrepancy modeling.
        discrepancies = [
            block(fp, fc)
            for block, fp, fc in zip(self.lmdb, prior_features, current_features)
        ]

        aligned = self.alignment(discrepancies)
        fused = self.tcfm(aligned)

        logits = self.classifier(fused)
        score_low, score_map = self.localizer(fused, output_size=image_size)

        return {
            "logits": logits,
            "score_low": score_low,
            "score_map": score_map,
            "fused": fused,
            "discrepancies": discrepancies,
            "aligned": aligned,
        }
