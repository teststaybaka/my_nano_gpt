import time
import math
import torch
import tiktoken
from model import GPT, GPTConfig

class DataLoader:
  def __init__(self, B, T):
    self.B = B
    self.T = T

    enc = tiktoken.get_encoding("gpt2")
    with open('input.txt', 'r') as f:
       text = f.read()
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)

    # Initialize for epoch-based batching
    self.current_position = 0
    self._reset_epoch()

  def _reset_epoch(self):
    """Apply random offset, chunk tokens into lists of size T+1, then shuffle the chunks"""
    # Apply random initial offset (0 to T)
    offset = torch.randint(0, self.T + 1, (1,)).item()

    # Start from offset and chunk into sequences of T+1 tokens
    tokens_from_offset = self.tokens[offset:]
    num_chunks = len(tokens_from_offset) // (self.T + 1)

    # Truncate to fit exact number of chunks
    usable_tokens = tokens_from_offset[:num_chunks * (self.T + 1)]

    # Reshape into chunks of size (T+1)
    chunks = usable_tokens.reshape(num_chunks, self.T + 1)

    # Shuffle the chunks
    perm = torch.randperm(num_chunks)
    self.shuffled_chunks = chunks[perm]

    self.current_position = 0

  def get_batch(self):
    # Check if we have enough chunks left for a full batch
    if self.current_position + self.B > len(self.shuffled_chunks):
      self._reset_epoch()

    # Get next B chunks
    batch_chunks = self.shuffled_chunks[self.current_position:self.current_position + self.B]
    self.current_position += self.B

    # Split into x and y
    x = batch_chunks[:, :self.T]  # First T tokens of each chunk
    y = batch_chunks[:, 1:self.T+1]  # Tokens 1 to T+1 (shifted by 1)

    x = x.to('cuda')
    y = y.to('cuda')
    return x, y

def configure_optimizer(model, weight_decay, betas):
  param_dict = {pn: p for pn, p in model.named_parameters()}
  param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
  decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
  no_decay_params = [p for n, p in param_dict.items() if p.ndim < 2]
  optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': no_decay_params, 'weight_decay': 0.0}
  ]
  optimizer = torch.optim.AdamW(optim_groups, lr=1e-3, betas=betas, eps=1e-8, fused=True)
  return optimizer

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision('high')

dataloader = DataLoader(B=4, T=1024)

model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to('cuda')
model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 10
max_iters = 50
def get_lr(it):
  # 1. linear warmup for warmup_iters steps
  if it < warmup_iters:
    return max_lr * (it + 1) / warmup_iters
  # 2. cosine decay down to min_lr for the rest of the steps
  if it >= max_iters:
    return min_lr
  # 3. cosine decay
  decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

optimizer = configure_optimizer(model, weight_decay=1e-1, betas=(0.9, 0.95))
for i in range(max_iters):
  t0 = time.time()
  optimizer.zero_grad()
  x, y = dataloader.get_batch()
  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    logits, loss = model(x, y)
  loss.backward()
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(i)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  torch.cuda.synchronize()
  t1 = time.time()
  dt = (t1 - t0) * 1000
  tokens_per_sec = (dataloader.T * dataloader.B) / (dt / 1000)
  print(f"step {i}: loss {loss.item()}, lr {lr:.6f}, norm {norm:.4f}, dt {dt:.2f}ms, {tokens_per_sec:.2f} tokens/sec")

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# while x.size(1) < max_length:
#   # forward the model to get the logits
#   with torch.no_grad():
#     logits = model(x) # (B, T, vocab_size)
#     # take the logits at the last position
#     logits = logits[:, -1, :] # (B, vocab_size)
#     # get the probabilities
#     probs = F.softmax(logits, dim=-1) # (B, vocab_size)
#     # do top-k sampling of 50 (huggingface pipeline default)
#     # topk_probs here becomes (5, 50), tok_indices becomes (5, 50)
#     topk_probs, tok_indices = torch.topk(probs, k=50, dim=-1)
#     # select a token from the top-k candidates
#     ix = torch.multinomial(topk_probs, num_samples=1) # (5, 1)
#     # gather the token indices
#     tok = torch.gather(tok_indices, 1, ix) # (5, 1)
#     # append to the sequence and continue
#     x = torch.cat((x, tok), dim=1) # (5, T + 1)

# # print the generated text
# for i in range(num_return_sequences):
#   tokens = x[i, :max_length].tolist()
#   decoded = enc.decode(tokens)
#   print(">", decoded)
