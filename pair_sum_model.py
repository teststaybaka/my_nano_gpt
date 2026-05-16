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

class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_heads == 0
    # key, query, value projections for all heads
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    # regularization
    self.n_heads = config.n_heads
    self.n_embd = config.n_embd

    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                  .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    # calculate query, key, values for all heads in batch and move head forward to be the batch dimension
    # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
    # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=768 channels in the Transformer
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) # (B, nh, T, hs)
    # attention (materializes the large (T,T) matrix for all queries and keys)

    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    # output projection
    y = self.c_proj(y)
    return y

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

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class GPT(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.config = config
    self.transformer = nn.ModuleDict({
        'wte': nn.Embedding(config.vocab_size, config.n_embd),
        'h': nn.ModuleList(Block(config) for _ in range(config.n_layers)),
        'ln_f': nn.LayerNorm(config.n_embd)
    })
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # init weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    # idx is of shape (B, T); needs T >= 2 to form at least one pair
    B, T = idx.size()
    assert T <= self.config.block_size, "Cannot forward, model block size is exhausted."
    assert T >= 2, "Need at least 2 tokens to form a pair."

    tok_emb = self.transformer['wte'](idx) # (B, T, n_embd)
    # Pair-sum: x[i] = tok_emb[i] + tok_emb[i+1] for i in 0..T-2
    x = tok_emb[:, :-1, :] + tok_emb[:, 1:, :] # (B, T-1, n_embd)

    for block in self.transformer.h:
      x = block(x)
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) # (B, T-1, vocab_size)

    loss = None
    if targets is not None:
      # logits[:, i] predicts the token AFTER pair[i] = original seq position i+2.
      # Data loader provides targets[i] = seq[i+1], so shift by 1 to align: use targets[:, 1:].
      loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, 1:].reshape(-1))
    return logits, loss
