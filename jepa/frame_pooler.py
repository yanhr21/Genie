"""
FramePooler: pool DiT token states to per-frame JEPA representations.
"""

from typing import Literal

import torch
import torch.nn as nn


class FramePooler(nn.Module):
    def __init__(
        self,
        dit_dim: int = 2048,
        out_dim: int = 512,
        pooling_mode: Literal["mean", "attn"] = "attn",
        num_queries: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dit_dim = int(dit_dim)
        self.out_dim = int(out_dim)
        self.pooling_mode = str(pooling_mode).lower()
        self.num_queries = max(1, int(num_queries))
        self.num_heads = max(1, int(num_heads))

        self.proj = nn.Linear(self.dit_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)

        if self.pooling_mode not in {"mean", "attn"}:
            raise ValueError(f"Unsupported pooling_mode={pooling_mode}")
        if self.pooling_mode == "attn":
            if self.out_dim % self.num_heads != 0:
                raise ValueError(f"out_dim={self.out_dim} must be divisible by num_heads={self.num_heads}")
            self.query_tokens = nn.Parameter(torch.empty(self.num_queries, self.out_dim))
            self.query_norm = nn.LayerNorm(self.out_dim)
            self.attn = nn.MultiheadAttention(self.out_dim, self.num_heads, batch_first=True)
            self.ffn = nn.Sequential(
                nn.LayerNorm(self.out_dim),
                nn.Linear(self.out_dim, 4 * self.out_dim),
                nn.GELU(),
                nn.Linear(4 * self.out_dim, self.out_dim),
            )
            self.out_proj = nn.Linear(self.num_queries * self.out_dim, self.out_dim)
            self.out_norm = nn.LayerNorm(self.out_dim)
            self._init_attn_weights()

    def _init_attn_weights(self):
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        bsz, thw, dim = hidden_states.shape
        expected = int(num_frames) * int(height) * int(width)
        if thw != expected:
            raise ValueError(f"Token count mismatch: {thw} != {expected}")
        if dim != self.dit_dim:
            raise ValueError(f"Hidden dim mismatch: {dim} != {self.dit_dim}")

        x = hidden_states.reshape(bsz, int(num_frames), int(height) * int(width), dim)
        x = self.proj(x)
        x = self.norm(x)
        if self.pooling_mode == "mean":
            return x.mean(dim=2)

        tokens = x.reshape(bsz * int(num_frames), int(height) * int(width), self.out_dim)
        queries = self.query_tokens.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(self.query_norm(queries), tokens, tokens, need_weights=False)
        pooled = pooled + self.ffn(pooled)
        pooled = pooled.reshape(bsz, int(num_frames), self.num_queries * self.out_dim)
        pooled = self.out_proj(pooled)
        pooled = self.out_norm(pooled)
        return pooled
