import os
import time
import math
import torch
import argparse
from reversed_pos_model import GPT, GPTConfig
from tiny_shakespear_data_loader import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
args = parser.parse_args()

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
# Only clear log file if not resuming
if not args.resume:
  with open(log_file, "w") as f:
    pass

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

def save_checkpoint(step, is_master):
  if not is_master or step == 0:
    return
  checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
  checkpoint = {
      'model': raw_model.state_dict(),
      'config': raw_model.config,
      'step': step,
  }
  torch.save(checkpoint, checkpoint_path)
  print(f"Saved checkpoint to {checkpoint_path}")

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

total_batch_size = 16384 # 2**14 tokens
B = 64  # batch size
T = 256 # context length
grad_accum_steps = total_batch_size // (B * T * world_size)
if is_master:
  print(f"total desired batch size: {total_batch_size}, world size: {world_size}, grad_accum_steps: {grad_accum_steps}")

train_dataloader = DataLoader(B=B, T=T, process_rank=rank, process_count=world_size)

model = GPT(GPTConfig(block_size=T, vocab_size=50304, n_layers=6, n_heads=6, n_embd=384))
model.to(device)
model = torch.compile(model)
if distributed:
  model = DDP(model, device_ids=[local_rank])
raw_model = model.module if distributed else model

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_iters = 100
max_iters = 5000
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

# Resume from checkpoint if specified
start_step = 0
if args.resume:
  if os.path.exists(args.resume):
    if is_master:
      print(f"Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    raw_model.load_state_dict(checkpoint['model'])
    # No optimizer state loaded for simplicity
    start_step = checkpoint['step'] + 1  # Start from next step
    if is_master:
      print(f"Resumed from step {checkpoint['step']}, starting at step {start_step}")
  else:
    if is_master:
      print(f"Checkpoint not found: {args.resume}, starting from scratch")

log_interval = 50
checkpoint_interval = 1000
for i in range(start_step, max_iters):
  t0 = time.time()
  if i % checkpoint_interval == 0:
    save_checkpoint(i, is_master)

  model.train()
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
    if i % log_interval == 0:
      with open(log_file, "a") as f:
        f.write(f"{i} train {loss_accum.item():.4f}\n")

save_checkpoint(max_iters, is_master)

if distributed:
  destroy_process_group()
