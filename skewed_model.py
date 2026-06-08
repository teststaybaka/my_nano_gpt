"""Diff from model.py (standard GPT):
- Pair-sum input embedding: x[i] = wte(idx[i - 1]) + wte(idx[i]), replacing positional embedding.
- Each attention head has a sliding window of size W = block_size, attending to the last W K/V pairs.
- KV cache carry and is skewed: each forward() returns new K/V for the chunk, moves back by one layer
  and stacks them on top of the next step's K/V pairs. layer ℓ's queries attend to previous chunk's
  K[ℓ+1] (one layer deeper) instead of previous chunk's K[ℓ].
- Adds a K/V only module above the top block to supply K[L] for layer L-1's cache slot
  (since there's no real layer above the deepest one).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # also the sliding-window size and KV cache size
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12 #nh
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_heads == 0
    # key, query, value projections for all heads
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.n_heads = config.n_heads
    self.n_embd = config.n_embd
    # sliding-window size in tokens; also the size of the KV cache.
    self.window_size = config.block_size

  def forward(self, x, kv_cache=None):
    """
    x: (B, T_new, C) — chunk's input embeddings at this layer.
    kv_cache: optional (K_cache, V_cache), each (B, n_heads, T_cache, head_size).
              Skew lives in the caller: the cache passed here came from a
              DIFFERENT layer of the previous chunk (layer index ℓ+1, or the
              top K/V projection for the deepest layer). This module is agnostic.
              Caller must detach these if no gradient should flow through them.
              Pass None for the prefill chunk.
    Returns:
      y: (B, T_new, C)
      (k, v): newly computed K, V for this chunk at this layer, each
              (B, n_heads, T_new, head_size). Caller routes these into the
              next chunk's cache slot (skew = layer ℓ → next chunk's layer ℓ-1).
    """
    B, T_new, C = x.size() # batch size, new-chunk length, embedding dim (n_embd)
    head_size = C // self.n_heads

    # qkv for the new chunk only — old positions' K/V come from kv_cache
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    q = q.view(B, T_new, self.n_heads, head_size).transpose(1, 2) # (B, nh, T_new, hs)
    k = k.view(B, T_new, self.n_heads, head_size).transpose(1, 2) # (B, nh, T_new, hs)
    v = v.view(B, T_new, self.n_heads, head_size).transpose(1, 2) # (B, nh, T_new, hs)

    if kv_cache is None:
      # Prefill chunk: no cached context, standard causal attention within the chunk.
      y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    else:
      # Sliding-window over [K_cache, K_new].
      K_cache, V_cache = kv_cache
      T_cache = K_cache.size(2)
      K_all = torch.cat([K_cache, k], dim=2) # (B, nh, T_cache + T_new, hs)
      V_all = torch.cat([V_cache, v], dim=2)

      # Build a boolean mask of shape (T_new, T_cache + T_new):
      # query at intra-chunk index q_i (global pos = chunk_start + q_i) attends
      # to K_all indices [q_i + T_cache - W + 1, q_i + T_cache].
      # broadcasts across B and n_heads when passed to SDPA.
      W = self.window_size
      q_pos = torch.arange(T_new, device=q.device).unsqueeze(1)            # (T_new, 1)
      k_pos = torch.arange(T_cache + T_new, device=q.device).unsqueeze(0)  # (1, T_cache + T_new)
      mask = (k_pos >= q_pos + T_cache - W + 1) & (k_pos <= q_pos + T_cache)

      y = F.scaled_dot_product_attention(q, K_all, V_all, attn_mask=mask)

    y = y.transpose(1, 2).contiguous().view(B, T_new, C) # re-assemble heads
    y = self.c_proj(y)
    return y, (k, v)


class MLP(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x


class Block(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x, kv_cache=None):
    attn_out, new_kv = self.attn(self.ln_1(x), kv_cache=kv_cache)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, new_kv


class KVOnly(nn.Module):
  """K/V-only projection at the top of the stack. Produces K[L], V[L] from the
  final block's output for the next chunk's deepest layer to attend to.
  No Q, no SDPA, no output projection, no residual. Output shape matches every
  other layer's K/V so the cache list is uniform."""
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln = nn.LayerNorm(config.n_embd)
    self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd)
    self.n_heads = config.n_heads
    self.n_embd = config.n_embd

  def forward(self, x):
    B, T, C = x.size()
    head_size = C // self.n_heads
    kv = self.c_kv(self.ln(x))                              # (B, T, 2*n_embd)
    k, v = kv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_heads, head_size).transpose(1, 2)  # (B, nh, T, hs)
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
    self.kv_only = KVOnly(config)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None, kv_caches=None, prev_tokens=None):
    """
    idx: (B, T_new) — this chunk's new token ids.
    targets: (B, T_new) — next-token targets aligned with idx. Pass None at eval to
             skip the cross-entropy computation; loss will be returned as None.
    kv_caches: list of n_layers (K, V) entries, each (B, n_heads, T_cache, head_size),
               or None for the prefill chunk. Skew routing:
                 kv_caches[ℓ] is consumed by THIS chunk's layer ℓ, and was
                 produced by the PREVIOUS chunk's layer ℓ+1 (for ℓ < L-1) or
                 by the previous chunk's top K/V projection (for ℓ = L-1).
               Caller must detach between optimizer steps.
    prev_tokens: (B, 1) — last token from the previous chunk, or the seed.
    Returns:
      logits: (B, T_new, vocab_size)
      loss: scalar cross-entropy averaged over all T_new positions, or None if targets is None.
      new_kv_caches: list of n_layers (K, V) for the next call (NOT detached).
                     Skew packing:
                       new_kv_caches[ℓ-1] = this chunk's layer ℓ output (for ℓ ≥ 1)
                       new_kv_caches[L-1] = this chunk's top K/V projection output
                     This chunk's layer 0 K/V is discarded (only used intra-chunk).
      new_prev_tokens: (B, 1) — last token of idx, to pass into the next forward().
    """
    assert prev_tokens is not None, "prev_tokens (seed or carry-over) is required."
    B, T_new = idx.size()
    n_layers = len(self.h)

    # Pair-sum input embedding — same scheme as sliding_cache_model.
    full = torch.cat([prev_tokens, idx], dim=1)             # (B, 1 + T_new)
    tok_emb = self.wte(full)                                # (B, 1 + T_new, n_embd)
    x = tok_emb[:, :-1, :] + tok_emb[:, 1:, :]              # (B, T_new, n_embd)

    # Thread the skewed cache through the stack.
    new_kv_caches = [None] * n_layers
    for layer_idx, block in enumerate(self.h):
      layer_cache = None if kv_caches is None else kv_caches[layer_idx]
      x, (k_new, v_new) = block(x, kv_cache=layer_cache)
      # layer 0's K/V is consumed intra-chunk only; layers 1..L-1 carry forward.
      if layer_idx >= 1:
        new_kv_caches[layer_idx - 1] = (k_new, v_new)

    # Top K/V projection — fills the last cache slot so the next chunk's
    # deepest layer has something to attend to.
    k_top, v_top = self.kv_only(x)
    new_kv_caches[n_layers - 1] = (k_top, v_top)

    # Logits branch (separate LN from the K/V branch — both consume the same x).
    x = self.ln_f(x)
    logits = self.lm_head(x)                                # (B, T_new, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    new_prev_tokens = idx[:, -1:]                           # carry last token to next chunk
    return logits, loss, new_kv_caches, new_prev_tokens
