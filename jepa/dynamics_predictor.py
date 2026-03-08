"""
LatentDynamicsPredictor: predict next-frame JEPA representation from current + action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_prefix_mask(query_len: int, key_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    q_idx = torch.arange(query_len, device=device).unsqueeze(1)
    k_idx = torch.arange(key_len, device=device).unsqueeze(0)
    mask = torch.zeros(query_len, key_len, device=device, dtype=dtype)
    mask = mask.masked_fill(k_idx > q_idx, -1e4)
    return mask.unsqueeze(0).unsqueeze(0)


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        cross_dim: int,
        mlp_ratio: float = 4.0,
        causal_actions: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.causal_actions = bool(causal_actions)

        self.norm1 = nn.LayerNorm(self.dim)
        self.self_attn_qkv = nn.Linear(self.dim, 3 * self.dim)
        self.self_attn_out = nn.Linear(self.dim, self.dim)

        self.norm2 = nn.LayerNorm(self.dim)
        self.cross_attn_q = nn.Linear(self.dim, self.dim)
        self.cross_attn_ctx_norm = nn.LayerNorm(cross_dim)
        self.cross_attn_kv = nn.Linear(cross_dim, 2 * self.dim)
        self.cross_attn_out = nn.Linear(self.dim, self.dim)

        self.norm3 = nn.LayerNorm(self.dim)
        hidden_dim = int(self.dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, steps, dim = x.shape
        ctx_steps = context.shape[1]

        h = self.norm1(x)
        qkv = self.self_attn_qkv(h).reshape(bsz, steps, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, steps, dim)
        x = x + self.self_attn_out(attn_out)

        h = self.norm2(x)
        q = self.cross_attn_q(h).reshape(bsz, steps, self.num_heads, self.head_dim).transpose(1, 2)
        context = self.cross_attn_ctx_norm(context)
        kv = self.cross_attn_kv(context).reshape(bsz, ctx_steps, 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn_mask = None
        if self.causal_actions:
            attn_mask = _build_prefix_mask(steps, ctx_steps, device=q.device, dtype=q.dtype)
        cross_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        cross_out = cross_out.transpose(1, 2).reshape(bsz, steps, dim)
        x = x + self.cross_attn_out(cross_out)

        x = x + self.ffn(self.norm3(x))
        return x


class LatentDynamicsPredictor(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        action_dim: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        causal_actions: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    self.dim,
                    num_heads=int(num_heads),
                    cross_dim=self.dim,
                    mlp_ratio=mlp_ratio,
                    causal_actions=causal_actions,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.pred_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.pred_head[-1].weight)
        nn.init.zeros_(self.pred_head[-1].bias)

    def forward(self, frame_repr: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_proj(actions)
        x = frame_repr
        for block in self.blocks:
            x = block(x, context=action_emb)
        return self.pred_head(x)
