import os
import time
import torch
import argparse
from model import GPT, GPTConfig
from ultrachat_data_loader import DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained checkpoint')
args = parser.parse_args()

log_dir = "finetune_log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, "w") as f:
    pass

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

# Training config
total_batch_size = 524288
B = 32
T = 1024
grad_accum_steps = total_batch_size // (B * T * world_size)
max_iters = 976  # 2 epochs with 255M tokens
lr = 6e-5

if is_master:
    print(f"total batch size: {total_batch_size}, grad_accum_steps: {grad_accum_steps}")
    print(f"max_iters: {max_iters}, lr: {lr}")

# Load pretrained model
if is_master:
    print(f"Loading pretrained checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
config = GPTConfig(**checkpoint['config'])
model = GPT(config)
model.load_state_dict(checkpoint['model'])
model.to(device)
model = torch.compile(model)
if distributed:
    model = DDP(model, device_ids=[local_rank])
raw_model = model.module if distributed else model

# Data loaders
train_dataloader = DataLoader(B=B, T=T, process_rank=rank, process_count=world_size, split="train")
val_dataloader = DataLoader(B=B, T=T, process_rank=rank, process_count=world_size, split="val")

# Optimizer with constant LR
optimizer = torch.optim.AdamW(raw_model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)

def validation(step):
    model.eval()
    val_dataloader._reset_epoch()
    with torch.no_grad():
        val_loss_accum = 0.0
        val_loss_steps = 50
        for _ in range(val_loss_steps):
            x, y = val_dataloader.get_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
            val_loss_accum += loss.detach() / val_loss_steps
        if distributed:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if is_master:
            print(f"step {step}: val loss {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

def save_checkpoint(step):
    if not is_master or step == 0:
        return
    # Strip _orig_mod. prefix from compiled model
    state_dict = raw_model.state_dict()
    unwrapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            unwrapped_state_dict[k[len('_orig_mod.'):]] = v
        else:
            unwrapped_state_dict[k] = v
    checkpoint = {
        'model': unwrapped_state_dict,
        'config': raw_model.config.__dict__,
        'step': step,
    }
    path = os.path.join(log_dir, f"model_{step:05d}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

# Training loop
for i in range(max_iters):
    t0 = time.time()

    if i % 100 == 0:
        validation(i)
        save_checkpoint(i)

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
    optimizer.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tokens_per_sec = total_batch_size / (dt / 1000)

    if is_master:
        print(f"step {i}: loss {loss_accum.item():.4f}, norm {norm:.4f}, dt {dt:.2f}ms, {tokens_per_sec:.0f} tokens/sec")

# Final validation and save
validation(max_iters)
save_checkpoint(max_iters)

if distributed:
    destroy_process_group()
