import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024 # token length limit
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12 #nh
    n_embd: int = 768
    rope_base: float = 10000.0 # RoPE frequency base (GPT-NeoX / Llama convention)


def _precompute_rope(head_dim, max_seq_len, base):
    """Build cos/sin tables for split-half RoPE.
    Returns (cos, sin), each of shape (max_seq_len, head_dim).
    Split-half convention: dim i and dim i+half share the same angle θ_i,
    so the half-dim frequencies get tiled along the last axis."""
    assert head_dim % 2 == 0
    half = head_dim // 2
    # θ_i = base^(-2i/head_dim), i in [0, half) — geometric falloff over head dims.
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)              # (max_seq_len, half)
    freqs_full = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return freqs_full.cos(), freqs_full.sin()


def _rotate_half(x):
    """For x = [x1, x2] split on the last dim, return [-x2, x1].
    This is the 'imaginary part' multiplier in the complex-rotation view of RoPE."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    """Apply RoPE rotation. x: (B, H, T, head_dim). cos, sin: (T, head_dim).
    Equivalent to a per-position 2D rotation on each (x_i, x_{i+half}) pair."""
    return x * cos + _rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd

    def forward(self, x, cos, sin):
        B, T, C = x.size()
        head_dim = C // self.n_heads

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, head_dim).transpose(1, 2) # (B, nh, T, hd)
        k = k.view(B, T, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_dim).transpose(1, 2)

        # Rotate Q and K (V is left alone — RoPE only injects position into the QK dot product).
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """Standard causal GPT with RoPE replacing learned positional embeddings.
    Full-attention (not sliding window) — meant as a direct baseline against
    the multi-scale model and the original standard GPT."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList(Block(config) for _ in range(config.n_layers)),
            'ln_f': nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight sharing (same as model.py)
        self.transformer['wte'].weight = self.lm_head.weight

        # RoPE cos/sin tables — same for every layer, recomputable from config,
        # so persistent=False (not saved in checkpoints).
        head_dim = config.n_embd // config.n_heads
        cos, sin = _precompute_rope(head_dim, config.block_size, config.rope_base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."

        x = self.transformer['wte'](idx) # (B, T, n_embd) — no wpe; RoPE goes into Q/K inside attn
        cos = self.rope_cos[:T] # (T, head_dim) — slice to current seq len
        sin = self.rope_sin[:T]
        for block in self.transformer['h']:
            x = block(x, cos, sin)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
