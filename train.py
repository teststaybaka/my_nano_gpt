import os
import time
import math
import torch
import tiktoken
import json
from model import GPT, GPTConfig
from fine_web_data_loader import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F

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

def valiation(model, B, T, device, rank, world_size, distributed, is_master, step):
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
      print(f"step {step}: validation loss {val_loss_accum.item()}")

def evaluate_hellaswag(model, device, rank, world_size, distributed, is_master, step):
  """Evaluate model on HellaSwag dataset, parallelized across GPUs"""

  # Load HellaSwag data (all processes load it)
  hellaswag_path = "hellaswag_data/hellaswag_val.jsonl"
  if not os.path.exists(hellaswag_path):
    if is_master:
      print(f"HellaSwag data not found at {hellaswag_path}, skipping evaluation")
    return

  with open(hellaswag_path, 'r') as f:
    examples = [json.loads(line) for line in f]

  enc = tiktoken.get_encoding("gpt2")

  num_correct = 0
  num_total = 0
  num_printed = 0  # Track how many examples we've printed

  with torch.no_grad():
    # Each GPU processes every world_size-th example
    for idx in range(rank, len(examples), world_size):
      example = examples[idx]
      ctx = example['ctx']
      endings = example['endings']
      label = int(example['label'])

      # Tokenize context once
      ctx_tokens = enc.encode(ctx)

      # Evaluate each ending
      ending_losses = []
      for ending in endings:
        # Create full sequence: context + ending
        full_text = ctx + " " + ending
        tokens = enc.encode(full_text)

        # Skip if sequence is too long for model
        if len(tokens) > 1024:
          ending_losses.append(float('inf'))
          continue

        # Convert to tensor
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

        # Forward pass
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
          logits, _ = model(tokens_tensor)

        # Calculate loss only on ending tokens (mask out context)
        # logits[0, i] predicts token at position i+1
        # To predict ending tokens (positions ctx_len to end), use logits[ctx_len-1:-1]
        ctx_len = len(ctx_tokens)
        shift_logits = logits[0, ctx_len-1:-1, :]  # Predictions for ending
        shift_labels = tokens_tensor[0, ctx_len:]  # Target ending tokens

        # Calculate average cross entropy loss over ending tokens
        loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
        ending_losses.append(loss.item())

      # Choose ending with lowest loss as model's prediction
      predicted = ending_losses.index(min(ending_losses))
      if predicted == label:
        num_correct += 1
        # Print first 3 correct examples from master process for inspection
        if is_master and num_printed < 3:
          print(f"\n--- Correct Example {num_printed + 1} ---")
          print(f"Context: {ctx}")
          for i, ending in enumerate(endings):
            marker = "✓ CORRECT" if i == label else ""
            print(f"  [{i}] (loss={ending_losses[i]:.4f}) {ending} {marker}")
          print(f"Model chose: {predicted}, Label: {label}")
          num_printed += 1
      num_total += 1

  # Gather results from all GPUs
  num_correct_tensor = torch.tensor(num_correct, dtype=torch.long, device=device)
  num_total_tensor = torch.tensor(num_total, dtype=torch.long, device=device)

  if distributed:
    dist.all_reduce(num_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)

  # Calculate and print accuracy (on master)
  if is_master:
    accuracy = num_correct_tensor.item() / num_total_tensor.item()
    print(f"step {step}: HellaSwag accuracy: {num_correct_tensor.item()}/{num_total_tensor.item()} = {accuracy:.4f}")

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
    valiation(model, B, T, device, rank, world_size, distributed, is_master, i)
    evaluate_hellaswag(model, device, rank, world_size, distributed, is_master, i)

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

valiation(model, B, T, device, rank, world_size, distributed, is_master, max_iters)
evaluate_hellaswag(model, device, rank, world_size, distributed, is_master, max_iters)

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
