"""Diff from model.py (standard GPT):
- No positional embedding.
- Each attention head gets its own sliding-window size T_h, log-uniformly spaced
  from min_window to block_size. Per-head SDPA mask; no per-head storage change.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def _log_uniform_windows(min_w, max_w, n):
    """n window sizes log-uniformly spaced between min_w and max_w (endpoints included)."""
    log_min = math.log2(min_w)
    log_max = math.log2(max_w)
    return tuple(
        int(round(2 ** (log_min + (log_max - log_min) * i / (n - 1))))
        for i in range(n)
    )


@dataclass
class GPTConfig:
    block_size: int = 1024 # also the max per-head window
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12 #nh
    n_embd: int = 768
    # Per-head sliding window sizes (length must equal n_heads). If None,
    # defaults to log-uniform from min_window to block_size.
    head_windows: tuple = None
    min_window: int = 2

    def __post_init__(self):
        if self.head_windows is None:
            self.head_windows = _log_uniform_windows(self.min_window, self.block_size, self.n_heads)
        assert len(self.head_windows) == self.n_heads
        assert max(self.head_windows) <= self.block_size
        assert min(self.head_windows) >= 1


class MultiScaleAttention(nn.Module):
    """Causal self-attention where each head has its own sliding-window size T_h.
    No positional embedding: distance info emerges from *which head* can see each
    token (small T_h = local, large T_h = global)."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        # Per-head window sizes. Buffer so it moves with the model and gets saved.
        self.register_buffer(
            'head_windows',
            torch.tensor(config.head_windows, dtype=torch.long),
            persistent=True,
        )

    def forward(self, x):
        B, T, C = x.size()
        head_size = C // self.n_heads

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, head_size).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_size).transpose(1, 2)

        # Per-head causal sliding-window mask of shape (n_heads, T, T):
        # head h's query at position i attends to keys in [i - W_h + 1, i].
        # SDPA broadcasts this mask across the batch dim.
        pos = torch.arange(T, device=q.device)
        dist = pos.view(1, T, 1) - pos.view(1, 1, T) # (1, T, T): query_pos - key_pos
        W = self.head_windows.view(self.n_heads, 1, 1)
        mask = (dist >= 0) & (dist < W) # causal + within window

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiScaleAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Multi-scale-head GPT. Each attention head gets its own sliding window;
    geometrically-spaced windows collectively encode distance information,
    so no explicit positional embedding is needed."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList(Block(config) for _ in range(config.n_layers)),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.transformer['wte'](idx) # (B, T, n_embd) — NoPE
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
