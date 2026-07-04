"""Spatial transformer blocks used by the hierarchical encoder."""

import torch
import torch.nn.functional as F
from torch import nn


def _window_partition(x: torch.Tensor, window_size: int):
    """Partition a feature map into non-overlapping local windows."""
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))

    hp, wp = x.shape[-2:]
    x = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = windows.view(-1, window_size * window_size, c)
    return windows, hp, wp, pad_h, pad_w


def _window_reverse(
    windows: torch.Tensor,
    batch_size: int,
    channels: int,
    padded_h: int,
    padded_w: int,
    window_size: int,
    pad_h: int,
    pad_w: int,
):
    """Reconstruct the feature map after window-based attention."""
    x = windows.view(
        batch_size,
        padded_h // window_size,
        padded_w // window_size,
        window_size,
        window_size,
        channels,
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(batch_size, channels, padded_h, padded_w)
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


class SpatialTransformerBlock(nn.Module):
    """Window-based transformer block for 2D medical image features."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        window_size: int = 8,
    ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = min(self.window_size, h, w)

        windows, hp, wp, pad_h, pad_w = _window_partition(x, ws)

        # Self-attention is applied within local windows to preserve efficiency
        # for high-resolution mammograms.
        attn_in = self.norm1(windows)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        windows = windows + attn_out
        windows = windows + self.mlp(self.norm2(windows))

        return _window_reverse(windows, b, c, hp, wp, ws, pad_h, pad_w)
