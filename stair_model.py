"""Diff from model.py (standard GPT):
- Pair-sum input embedding: x[i] = wte(idx[i - 1]) + wte(idx[i]), replacing positional embedding.
- KV cache carry: each forward() returns new K/V for the chunk.
- Adds a K/V only module above the top block to supply K[L] for layer L-1's cache slot
  (since there's no real layer above the deepest one).
- Every query sees recent W/2 from K[ℓ] AND older W/2 from K[ℓ+1] (W = window = block_size),
  independent of intra-chunk position.
- Requires chunk_size <= block_size/2 for the strict semantics (asserted).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024 # also the attention window size W; expected chunk size at training is W/2
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12 #nh
    n_embd: int = 768


class StairAttention(nn.Module):
    """Causal attention with strict per-query W/2 + W/2 stair routing (W = window size).
    Recent W/2 positions use K[ℓ] (this layer); older W/2 use K[ℓ+1] (one layer deeper).
    The deeper-layer K comes from a previous chunk's layer ℓ+1 forward, so the
    caller must (a) chunk inputs into pieces of size ≤ W/2 and (b) feed both
    low_cache (K[ℓ], V[ℓ] from previous chunk) and high_cache (K[ℓ+1], V[ℓ+1]
    from previous chunks) into each forward."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        assert config.block_size % 2 == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.window_size = config.block_size

    def forward(self, x, cache=None):
        """
        x: (B, T_new, C) — current chunk; assume T_new = W/2 in training (asserted ≤ in GPT).
        cache: optional (low_new, high_old, high_new) — three raw chunk K/V tensors:
                 low_new:  (K[ℓ],   V[ℓ])   from previous chunk N-1
                 high_new: (K[ℓ+1], V[ℓ+1]) from previous chunk N-1
                 high_old: (K[ℓ+1], V[ℓ+1]) from previous chunk N-2, or None during warmup (N=1)
               None entirely for the first chunk of a shard (N=0). low_new and high_new
               are both from chunk N-1; high_old is from chunk N-2.
               Each tensor is W/2 positions — no concat/trim at chunk boundaries.
        Returns:
          y: (B, T_new, C)
          (k, v): this chunk's K[ℓ], V[ℓ] — caller routes into the next chunk's cache.
        """
        B, T_new, C = x.size()
        head_size = C // self.n_heads
        W = self.window_size
        half_W = W // 2

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T_new, self.n_heads, head_size).transpose(1, 2) # (B, nh, T_new, hd)
        k = k.view(B, T_new, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T_new, self.n_heads, head_size).transpose(1, 2)

        if cache is None:
            # First chunk of a shard: no cross-chunk history. Standard causal within chunk.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self._merge_heads(y, B, T_new, C), (k, v)

        low_new, high_old, high_new = cache
        K_low, V_low = low_new
        K_high_new, V_high_new = high_new
        T_low = K_low.size(2)

        # Assemble K_combined with one cat instead of building K_low then K_combined.
        # K_high positions go first, then K_low + live k.
        if high_old is None:
            # Warmup (chunk N=1): chunk N-2 doesn't exist. K_high is just N-1's K[ℓ+1].
            K_combined = torch.cat([K_high_new, K_low, k], dim=2)
            V_combined = torch.cat([V_high_new, V_low, v], dim=2)
        else:
            # Steady state: K_high = [N-2's K[ℓ+1], N-1's K[ℓ+1]] (contiguous absolute positions).
            K_high_old, V_high_old = high_old
            K_combined = torch.cat([K_high_old, K_high_new, K_low, k], dim=2)
            V_combined = torch.cat([V_high_old, V_high_new, V_low, v], dim=2)
        T_high_total = K_combined.size(2) - T_low - T_new

        # Build mask. Anchor positions relative to current chunk start = 0.
        # K_high portion (first T_high_total cols) covers absolute positions [-T_high_total, -1].
        # K_low portion (remaining cols)         covers absolute positions [-T_low, T_new - 1].
        # Stair rule per query q:
        #   recent W/2 (use K_low):  0 <= dist < half_W
        #   older W/2  (use K_high): half_W <= dist < W
        q_pos = torch.arange(T_new, device=q.device)

        high_pos = torch.arange(-T_high_total, 0, device=q.device)
        high_dist = q_pos.unsqueeze(1) - high_pos.unsqueeze(0) # (T_new, T_high_total)
        high_mask = (high_dist >= half_W) & (high_dist < W)

        low_pos = torch.arange(-T_low, T_new, device=q.device)
        low_dist = q_pos.unsqueeze(1) - low_pos.unsqueeze(0) # (T_new, T_low + T_new)
        low_mask = (low_dist >= 0) & (low_dist < half_W)

        mask = torch.cat([high_mask, low_mask], dim=1)

        y = F.scaled_dot_product_attention(q, K_combined, V_combined, attn_mask=mask)
        return self._merge_heads(y, B, T_new, C), (k, v)

    @staticmethod
    def _merge_heads(y, B, T_new, C):
        return y.transpose(1, 2).contiguous().view(B, T_new, C)


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
        self.attn = StairAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, cache=None):
        attn_out, new_kv = self.attn(self.ln_1(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv


class KVOnly(nn.Module):
    """K/V-only projection above the top block. Produces K[L], V[L] from the final
    block's output, used as 'one layer deeper' K for the deepest block's next-chunk
    high_cache. No Q/SDPA/output/residual — just LN + KV projection."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd)
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        head_size = C // self.n_heads
        kv = self.c_kv(self.ln(x))
        k, v = kv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_heads, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_heads, head_size).transpose(1, 2)
        return k, v


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.kv_only = KVOnly(config) # supplies K[L] for layer L-1's high_cache

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, caches=None, prev_tokens=None):
        """
        idx: (B, T_new) — this chunk's new token ids. T_new must be <= block_size/2.
        targets: (B, T_new) — pair-sum next-next-token targets; targets[i] is the token
                 to predict from the pair ending at idx[i] (same alignment as skewed_model).
                 None at eval skips loss.
        caches: list of n_layers entries, each a tuple (low_new, high_old, high_new).
                Each member is a (K, V) tuple of raw chunk tensors (W/2 positions), except
                high_old which is None during the chunk-N=1 warmup. Pass None (whole list)
                for the first chunk of a shard. Caches accumulate gradient across sub-chunks
                for BPTT; reset to None at the start of each optimizer step (do not carry
                across optimizer.step() boundaries).
        prev_tokens: (B, 1) — last token of previous chunk, or shard seed.
        Returns:
          logits: (B, T_new, vocab_size)
          loss: scalar cross-entropy, or None
          new_caches: list of n_layers (low_new, high_old, high_new) for the next call.
                      All entries are W/2-sized raw chunk tensors (no cat or trim).
          new_prev_tokens: (B, 1) — last token of idx, to feed forward.
        """
        assert prev_tokens is not None, "prev_tokens (seed or carry-over) is required."
        B, T_new = idx.size()
        W = self.config.block_size
        assert T_new <= W // 2, f"chunk size {T_new} exceeds block_size/2 = {W//2}; strict stair invariant breaks."

        # Pair-sum input (same as skewed_model / sliding_cache_model).
        full = torch.cat([prev_tokens, idx], dim=1)             # (B, 1 + T_new)
        tok_emb = self.wte(full)                                # (B, 1 + T_new, n_embd)
        x = tok_emb[:, :-1, :] + tok_emb[:, 1:, :]              # (B, T_new, n_embd)

        # Forward through layers, collecting each layer's K[ℓ], V[ℓ].
        computed_kv = []
        for layer_idx, block in enumerate(self.h):
            layer_cache = None if caches is None else caches[layer_idx]
            x, (k_new, v_new) = block(x, cache=layer_cache)
            computed_kv.append((k_new, v_new))

        # K[L], V[L] for the deepest layer's high_cache routing.
        k_top, v_top = self.kv_only(x)

        # Logits branch.
        x_out = self.ln_f(x)
        logits = self.lm_head(x_out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Build new caches for the next chunk — no cat, no trim, just pointer shuffles.
        # Each cache entry: (low_new, high_old, high_new) of raw chunk K/V tensors.
        # Shift rule: previous chunk's high_new becomes the new high_old; the new low_new
        # and high_new are this chunk's K[ℓ] and K[ℓ+1] (or kv_only output for top layer).
        new_caches = []
        n_layers = len(self.h)
        for layer_idx in range(n_layers):
            k_l, v_l = computed_kv[layer_idx]
            if layer_idx + 1 < n_layers:
                k_above, v_above = computed_kv[layer_idx + 1]
            else:
                k_above, v_above = k_top, v_top

            next_low_new = (k_l, v_l)
            next_high_new = (k_above, v_above)
            if caches is None:
                next_high_old = None  # next chunk (N=1) has no chunk N-2 to draw from
            else:
                # this chunk's incoming high_new shifts forward into the high_old slot
                _, _, prev_high_new = caches[layer_idx]
                next_high_old = prev_high_new

            new_caches.append((next_low_new, next_high_old, next_high_new))

        new_prev_tokens = idx[:, -1:]
        return logits, loss, new_caches, new_prev_tokens
