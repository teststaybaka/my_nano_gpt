import os
import time
import math
import torch
from sliding_cache_model import GPT, GPTConfig
from data_loader import DataLoader

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
  pass

def configure_optimizer(model, weight_decay, betas):
  param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
  decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
  no_decay_params = [p for n, p in param_dict.items() if p.ndim < 2]
  optim_groups = [
      {'params': decay_params, 'weight_decay': weight_decay},
      {'params': no_decay_params, 'weight_decay': 0.0}
  ]
  return torch.optim.AdamW(optim_groups, lr=1e-3, betas=betas, eps=1e-8, fused=True)

def validation(model, B, T, sub_chunks_per_batch, T_dataloader, device, step):
  val_dataloader = DataLoader(B=B, T=T_dataloader, process_rank=0, process_count=1, split="val")
  val_loss_accum = 0.0
  val_loss_steps = 20
  with torch.no_grad():
    for _ in range(val_loss_steps):
      x, y = val_dataloader.get_batch()
      x, y = x.to(device), y.to(device)

      prev_tokens = x[:, 0:1]
      kv_caches = None
      sub_loss_total = 0.0
      for sub in range(sub_chunks_per_batch):
        start = 1 + sub * T
        idx     = x[:, start : start + T]
        targets = y[:, start : start + T]
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          _, loss, kv_caches, prev_tokens = model(idx, targets, kv_caches, prev_tokens)
        sub_loss_total = sub_loss_total + loss

      val_loss_accum += (sub_loss_total / sub_chunks_per_batch / val_loss_steps).detach()

  print(f"step {step}: validation loss {val_loss_accum.item():.4f}")
  with open(log_file, "a") as f:
    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

device = 'cuda'
torch.manual_seed(42)
if torch.cuda.is_available():
  torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision('high')

# Hyperparameters
total_batch_size = 524288             # tokens producing loss per optimizer step (2**19)
B = 16                                # parallel walkers
T = 1024                              # model chunk size per forward
context_length = 4096                 # per-batch BPTT span
sub_chunks_per_batch = context_length // T
grad_accum_steps = total_batch_size // (B * context_length)
assert grad_accum_steps * B * context_length == total_batch_size, \
    f"total_batch_size ({total_batch_size}) must be divisible by B * context_length ({B * context_length})"
print(f"total_batch_size: {total_batch_size}, B: {B}, context_length: {context_length}, "
      f"sub_chunks_per_batch: {sub_chunks_per_batch}, grad_accum_steps: {grad_accum_steps}")

T_dataloader = context_length + 1     # +1 for the seed token at position 0
train_dataloader = DataLoader(B=B, T=T_dataloader, process_rank=0, process_count=1, split="train")

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_iters = 715
max_iters = 19073 * 2
def get_lr(it):
  if it < warmup_iters:
    return max_lr * (it + 1) / warmup_iters
  if it >= max_iters:
    return min_lr
  decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  return min_lr + coeff * (max_lr - min_lr)

optimizer = configure_optimizer(model, weight_decay=1e-1, betas=(0.9, 0.95))

for i in range(max_iters):
  t0 = time.time()
  if i % 500 == 0:
    model.eval()
    validation(model, B, T, sub_chunks_per_batch, T_dataloader, device, i)

  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0

  for micro_step in range(grad_accum_steps):
    x, y = train_dataloader.get_batch()
    x, y = x.to(device), y.to(device)

    # Seed pair-sum carry-over from the first token of each walker's context.
    prev_tokens = x[:, 0:1]
    kv_caches = None
    sub_loss_total = 0.0

    # BPTT across sub_chunks_per_batch chunks: KV cache and prev_tokens carry
    # forward with gradient attached, so loss.backward() at the end reaches all of them.
    for sub in range(sub_chunks_per_batch):
      start = 1 + sub * T
      idx     = x[:, start : start + T]
      targets = y[:, start : start + T]
      with torch.autocast(device_type=device, dtype=torch.bfloat16):
        _, loss, kv_caches, prev_tokens = model(idx, targets, kv_caches, prev_tokens)
      sub_loss_total = sub_loss_total + loss

    # Average over sub-chunks, then scale for grad accumulation so the accumulated
    # gradient is the mean loss over all (sub_chunk, micro_step) positions.
    micro_loss = sub_loss_total / sub_chunks_per_batch / grad_accum_steps
    micro_loss.backward()
    loss_accum += micro_loss.detach()

  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  lr = get_lr(i)
  for pg in optimizer.param_groups:
    pg['lr'] = lr
  optimizer.step()
  torch.cuda.synchronize()
  t1 = time.time()
  dt = (t1 - t0) * 1000
  tokens_per_sec = total_batch_size / (dt / 1000)
  print(f"step {i}: loss {loss_accum.item():.4f}, lr {lr:.6f}, norm {norm:.4f}, dt {dt:.2f}ms, {tokens_per_sec:.2f} tokens/sec")

model.eval()
validation(model, B, T, sub_chunks_per_batch, T_dataloader, device, max_iters)

checkpoint_path = os.path.join(log_dir, f"model_{max_iters:05d}.pt")
torch.save({
    'model': model.state_dict(),
    'config': model.config,
    'step': max_iters,
}, checkpoint_path)
print(f"Saved checkpoint to {checkpoint_path}")
