"""Temporal Cross-Attention Fusion Module (TCFM)."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class TemporalCrossAttentionFusionModule(nn.Module):
    """Fuse aligned multi-scale discrepancy features through cross-attention.

    The deepest aligned discrepancy feature provides the query, while lower-level
    discrepancy features form the keys and values. This allows high-level
    temporal semantics to select informative fine-scale temporal changes.
    """

    def __init__(
        self,
        alignment_dim: int = 256,
        attention_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        kv_size: Tuple[int, int] = (32, 32),
    ):
        super().__init__()
        self.kv_size = kv_size
        self.q_proj = nn.Linear(alignment_dim, attention_dim)
        self.k_proj = nn.Linear(alignment_dim * 3, attention_dim)
        self.v_proj = nn.Linear(alignment_dim * 3, attention_dim)
        self.attn = nn.MultiheadAttention(attention_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.norm2 = nn.LayerNorm(attention_dim)
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 4, attention_dim),
            nn.Dropout(dropout),
        )
        self.out_proj = nn.Conv2d(attention_dim, alignment_dim, kernel_size=1)

    def forward(self, aligned_features: List[torch.Tensor]) -> torch.Tensor:
        d1, d2, d3, d4 = aligned_features
        b, _, h, w = d4.shape

        # Query: highest-level aligned discrepancy feature.
        q0 = d4.flatten(2).transpose(1, 2)

        # Keys and values: concatenated lower-level aligned discrepancy features.
        kv_map = torch.cat([d1, d2, d3], dim=1)
        if self.kv_size is not None:
            kv_map = F.adaptive_avg_pool2d(kv_map, self.kv_size)
        kv = kv_map.flatten(2).transpose(1, 2)

        q = self.q_proj(q0)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        attn_out, _ = self.attn(q, k, v, need_weights=False)
        t1 = self.norm1(q + attn_out)
        t2 = self.norm2(t1 + self.ffn(t1))

        fused = t2.transpose(1, 2).reshape(b, -1, h, w)
        return self.out_proj(fused)
