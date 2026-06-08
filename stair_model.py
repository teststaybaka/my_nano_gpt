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
from torch.utils.checkpoint import checkpoint
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
        x: (B, T_new, C) — current chunk; require T_new <= W/2 for strict stair semantics.
        cache: optional ((K_low_prev, V_low_prev), (K_high_prev, V_high_prev)) — both halves
               are always present together (set by GPT.forward in lockstep). None only for
               the first chunk of a shard.
                 low_prev:  K[ℓ], V[ℓ] from previous chunk (steady-state size W/2 - 1).
                 high_prev: K[ℓ+1], V[ℓ+1] from previous chunks (steady-state size W - 1).
        Returns:
          y: (B, T_new, C)
          (k, v): this chunk's K[ℓ], V[ℓ] — caller routes to next chunk's caches.
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

        (K_low_prev, V_low_prev), (K_high_prev, V_high_prev) = cache

        # K_low = previous chunk's K[ℓ] cache + this chunk's live K[ℓ]
        K_low = torch.cat([K_low_prev, k], dim=2)
        V_low = torch.cat([V_low_prev, v], dim=2)
        T_low_prev = K_low_prev.size(2)
        T_high = K_high_prev.size(2)

        # Stack K_high then K_low for a single SDPA call.
        K_combined = torch.cat([K_high_prev, K_low], dim=2)
        V_combined = torch.cat([V_high_prev, V_low], dim=2)

        # Build mask. Anchor positions relative to current chunk start = 0.
        # K_high_prev covers absolute positions [-T_high, -1]
        # K_low       covers absolute positions [-T_low_prev, T_new - 1]
        # Stair rule per query q:
        #   recent W/2 (use K_low):  positions [q - half_W + 1, q]      → 0 <= dist < half_W
        #   older W/2  (use K_high): positions [q - W + 1, q - half_W]  → half_W <= dist < W
        q_pos = torch.arange(T_new, device=q.device)

        high_pos = torch.arange(-T_high, 0, device=q.device)
        high_dist = q_pos.unsqueeze(1) - high_pos.unsqueeze(0) # (T_new, T_high)
        high_mask = (high_dist >= half_W) & (high_dist < W)

        low_pos = torch.arange(-T_low_prev, T_new, device=q.device)
        low_dist = q_pos.unsqueeze(1) - low_pos.unsqueeze(0) # (T_new, T_low)
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


def _trim(t, max_size):
    """Trim a (B, nh, T, hd) tensor to its last max_size positions along dim 2."""
    return t if t.size(2) <= max_size else t[:, :, -max_size:, :]


class GPT(nn.Module):
    """Stair-routed cache GPT with pair-sum input. W = window = block_size.
    Per layer, two caches threaded across chunks:
      - low_cache:  K[ℓ], V[ℓ] from previous chunk    (steady-state size W/2 - 1)
      - high_cache: K[ℓ+1], V[ℓ+1] from previous ~2 chunks (steady-state size W - 1)
    Strict per-query W/2 stair: recent W/2 from K[ℓ], older W/2 from K[ℓ+1].
    Requires chunk_size <= W/2 (=block_size/2) for the strict semantics."""

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
        caches: list of n_layers entries, each a tuple ((K_low, V_low), (K_high, V_high)).
                Pass None (whole list) for the first chunk of a shard; subsequent calls
                receive the new_caches returned by the previous call. Caches accumulate
                gradient across sub-chunks for BPTT; reset to None at the start of each
                optimizer step (do not carry across optimizer.step() boundaries).
        prev_tokens: (B, 1) — last token of previous chunk, or shard seed.
        Returns:
          logits: (B, T_new, vocab_size)
          loss: scalar cross-entropy, or None
          new_caches: list of n_layers ((K_low, V_low), (K_high, V_high)) for the next call.
                      Sizes trimmed to (W/2 - 1) for low and (W - 1) for high.
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
        # Activation checkpointing: don't keep block internals (attention scores,
        # MLP hidden, etc.) for backward — recompute them by re-running the block
        # forward. Trades ~33% compute for ~10x activation memory at long BPTT.
        computed_kv = []
        for layer_idx, block in enumerate(self.h):
            layer_cache = None if caches is None else caches[layer_idx]
            x, (k_new, v_new) = checkpoint(block, x, layer_cache, use_reentrant=False)
            computed_kv.append((k_new, v_new))

        # K[L], V[L] for the deepest layer's high_cache routing.
        k_top, v_top = self.kv_only(x)

        # Logits branch.
        x_out = self.ln_f(x)
        logits = self.lm_head(x_out)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Build new caches for the next chunk.
        new_caches = []
        n_layers = len(self.h)
        for layer_idx in range(n_layers):
            k_l, v_l = computed_kv[layer_idx]
            if layer_idx + 1 < n_layers:
                k_above, v_above = computed_kv[layer_idx + 1]
            else:
                k_above, v_above = k_top, v_top

            if caches is None:
                ext_low_k, ext_low_v = k_l, v_l
                ext_high_k, ext_high_v = k_above, v_above
            else:
                (old_low_k, old_low_v), (old_high_k, old_high_v) = caches[layer_idx]
                ext_low_k = torch.cat([old_low_k, k_l], dim=2)
                ext_low_v = torch.cat([old_low_v, v_l], dim=2)
                ext_high_k = torch.cat([old_high_k, k_above], dim=2)
                ext_high_v = torch.cat([old_high_v, v_above], dim=2)
            max_low = W // 2 - 1
            max_high = W - 1
            new_low = (_trim(ext_low_k, max_low), _trim(ext_low_v, max_low))
            new_high = (_trim(ext_high_k, max_high), _trim(ext_high_v, max_high))
            new_caches.append((new_low, new_high))

        new_prev_tokens = idx[:, -1:]
        return logits, loss, new_caches, new_prev_tokens
