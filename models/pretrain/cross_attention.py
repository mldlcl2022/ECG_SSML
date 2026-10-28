from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim: int,
                 kv_dim: int,
                 output_dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=qkv_bias)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, output_dim),
            nn.Dropout(drop_out_rate)
        )

    def forward(self, query, kv):

        q = self.to_q(query)   # [B, T1, H*D]
        k = self.to_k(kv) # [B, T2, H*D]
        v = self.to_v(kv) # [B, T2, H*D]

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # [B, H, T1, T2]
        attn = self.attend(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                                # [B, H, T1, D]
        out = rearrange(out, 'b h n d -> b n (h d)')               # [B, T1, H*D]
        out = self.to_out(out)

        return out

class CrossPreNorm(nn.Module):
    def __init__(self,
                 query_dim: int,
                 kv_dim: int,
                 fn: nn.Module):
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.fn = fn

    def forward(self, query, kv):
        return self.fn(self.norm_q(query), self.norm_kv(kv))

class PreNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    MLP Module with GELU activation fn + dropout.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_out_rate),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.Dropout(drop_out_rate))

    def forward(self, x):
        return self.net(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 query_dim: int,
                 kv_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 heads: int = 8,
                 dim_head: int = 32,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        attn = CrossAttention(query_dim=query_dim,
                                kv_dim=kv_dim,
                                output_dim=output_dim,
                                heads=heads,
                                dim_head=dim_head,
                                qkv_bias=qkv_bias,
                                drop_out_rate=drop_out_rate,
                                attn_drop_out_rate=attn_drop_out_rate)
        
        self.attn = CrossPreNorm(query_dim=query_dim,
                                 kv_dim=kv_dim,
                                 fn=attn)

        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim,
                         drop_out_rate=drop_out_rate)

        self.ff = PreNorm(dim=output_dim,
                          fn=ff)

        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, query, kv):
        x = self.droppath1(self.attn(query, kv)) + query
        x = self.droppath2(self.ff(x)) + x
        return x