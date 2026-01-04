import os
import time
import math
import torch
import tiktoken
from model import GPT, GPTConfig
from fine_web_data_loader import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

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

distributed = int(os.environ.get('RANK', '-1')) != -1
if distributed:
  init_process_group(backend='nccl')
  rank = int(os.environ['RANK'])
  local_rank = int(os.environ['LOCAL_RANK'])
  world_size = int(os.environ['WORLD_SIZE'])
  device = f'cuda:{local_rank}'
  torch.cuda.set_device(device)
  is_master = (rank == 0)
else:
  rank = 0
  local_rank = 0
  world_size = 1
  is_master = True
  device = 'cuda'

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

torch.set_float32_matmul_precision('high')

total_batch_size = 524288 # 2**19 tokens
B = 4  # batch size
T = 1024 # context length
grad_accum_steps = total_batch_size // (B * T * world_size)
if is_master:
  print(f"total desired batch size: {total_batch_size}, world size: {world_size}, grad_accum_steps: {grad_accum_steps}")

train_dataloader = DataLoader(B=B, T=T, process_rank=rank, process_count=world_size, split="train")

model = GPT(GPTConfig(vocab_size=50304))
model.eval()
model.to(device)
model = torch.compile(model)
if distributed:
  model = DDP(model, device_ids=[local_rank])
raw_model = model.module if distributed else model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 715
max_iters = 19073
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

optimizer = configure_optimizer(raw_model, weight_decay=1e-1, betas=(0.9, 0.95))
for i in range(max_iters):
  t0 = time.time()
  # Valuation every 100 steps
  if i % 100 == 0:
    model.eval()
    val_dataloader = DataLoader(B=B, T=T, process_rank=rank, process_count=world_size, split="val")
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_dataloader.get_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
      if distributed:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
      if is_master:
        print(f"step {i}: validation loss {val_loss_accum.item()}")

  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_dataloader.get_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
      logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    if distributed:
      model.requires_backward_grad_sync = (micro_step == grad_accum_steps - 1)
    loss.backward()
  if distributed:
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(i)
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()
  torch.cuda.synchronize()
  t1 = time.time()
  dt = (t1 - t0) * 1000
  tokens_per_sec = (train_dataloader.T * train_dataloader.B * grad_accum_steps * world_size) / (dt / 1000)
  if is_master:
    print(f"step {i}: loss {loss_accum.item()}, lr {lr:.6f}, norm {norm:.4f}, dt {dt:.2f}ms, {tokens_per_sec:.2f} tokens/sec")

if distributed:
  destroy_process_group()

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
