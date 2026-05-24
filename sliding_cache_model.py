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
    x: (B, T_new, C) — chunk's input embeddings.
    kv_cache: optional (K_cache, V_cache), each (B, n_heads, T_cache, head_size).
              Caller must detach these if no gradient should flow through them.
              Pass None for the prefill chunk.
    Returns:
      y: (B, T_new, C)
      (k, v): newly computed K, V for this chunk, each (B, n_heads, T_new, head_size).
              Caller uses these to build the next step's KV cache.
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


class GPT(nn.Module):
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

  def forward(self, idx, targets, kv_caches=None, prev_tokens=None):
    """
    idx: (B, T_new) — this chunk's new token ids.
    targets: (B, T_new) — next-token targets aligned with idx, i.e., targets[i] is the
             token that the pair ending at idx[i] should predict.
    kv_caches: list of (K, V) per layer, each (B, n_heads, T_cache, head_size). None for
               the first chunk of a shard. Caller must detach between optimizer steps.
    prev_tokens: (B, 1) — last token from the previous chunk, or the seed for the first chunk.
                  Must not be None — call seed() first.
    Returns:
      logits: (B, T_new, vocab_size)
      loss: scalar cross-entropy averaged over all T_new positions
      new_kv_caches: list of (K, V) per layer for next call (NOT detached)
      new_prev_tokens: (B, 1) — last token of idx, to pass into the next forward()
    """
    assert prev_tokens is not None, "Call seed() before the first forward() on a shard."
    B, T_new = idx.size()

    # Build pair-sum input. Concat carry-over with the new chunk, sum adjacent embeddings.
    # For k=2: full[0] = prev, full[1..T_new] = idx[0..T_new-1].
    # x[i] = wte(full[i]) + wte(full[i+1]) for i in [0, T_new) → pair at position i uses
    # (full[i], full[i+1]) = (prev_or_prev_idx, current_idx), predicting the token after current.
    full = torch.cat([prev_tokens, idx], dim=1)             # (B, 1 + T_new)
    tok_emb = self.transformer['wte'](full)                 # (B, 1 + T_new, n_embd)
    x = tok_emb[:, :-1, :] + tok_emb[:, 1:, :]              # (B, T_new, n_embd)

    # Thread the KV cache through the stack of blocks, one (K, V) per layer.
    new_kv_caches = []
    for layer_idx, block in enumerate(self.transformer.h):
      layer_cache = None if kv_caches is None else kv_caches[layer_idx]
      x, new_kv = block(x, kv_cache=layer_cache)
      new_kv_caches.append(new_kv)

    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)                                # (B, T_new, vocab_size)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    new_prev_tokens = idx[:, -1:]                           # carry last token to next chunk
    return logits, loss, new_kv_caches, new_prev_tokens
