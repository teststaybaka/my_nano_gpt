"""
HellaSwag eval for pair-sum models with the (idx, targets, kv_caches, prev_tokens)
forward signature — i.e. sliding_cache_model and skewed_model.

Swap the import line below to switch which model is evaluated. Both share the
same forward contract, so the same eval code works for either.

Usage:
  python eval_hellaswag.py <checkpoint.pt>
  python eval_hellaswag.py <checkpoint.pt> --max-examples 200    # debug
"""
import os
import json
import argparse
import torch
import tiktoken
import torch.nn.functional as F

# === Swap this line to switch model ===
from sliding_cache_model import GPT, GPTConfig
# from skewed_model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
parser.add_argument('--max-examples', type=int, default=None,
                    help='Limit number of examples (for quick debugging)')
args = parser.parse_args()

device = 'cuda'
torch.set_float32_matmul_precision('high')

# --- Load checkpoint ---
print(f"Loading checkpoint: {args.checkpoint}")
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

# Rebuild config using this script's GPTConfig (avoids unpickling issues if the
# saved config object was a different module's GPTConfig). Field names match
# across all pair-sum model variants.
saved = checkpoint['config']
config = GPTConfig(
    block_size=saved.block_size,
    vocab_size=saved.vocab_size,
    n_layers=saved.n_layers,
    n_heads=saved.n_heads,
    n_embd=saved.n_embd,
)
model = GPT(config).to(device)

# Training script wraps the model in torch.compile() and saves the wrapper's
# state_dict, which prefixes every key with '_orig_mod.'. Strip it for loading
# into an uncompiled model.
state_dict = checkpoint['model']
state_dict = {k.removeprefix('_orig_mod.'): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()
print(f"Loaded model from step {checkpoint['step']}")

# --- Load HellaSwag ---
hellaswag_path = "hellaswag_data/hellaswag_val.jsonl"
if not os.path.exists(hellaswag_path):
    raise FileNotFoundError(f"HellaSwag data not found at {hellaswag_path}")
with open(hellaswag_path, 'r') as f:
    examples = [json.loads(line) for line in f]
if args.max_examples is not None:
    examples = examples[:args.max_examples]
print(f"Loaded {len(examples)} HellaSwag examples")

enc = tiktoken.get_encoding("gpt2")

num_correct = 0
num_total = 0
num_printed = 0  # print a few correct examples for inspection

with torch.no_grad():
    for idx_ex, example in enumerate(examples):
        ctx = example['ctx']
        endings = example['endings']
        label = int(example['label'])
        ctx_tokens = enc.encode(ctx)

        ending_losses = []
        for ending in endings:
            full_text = ctx + " " + ending
            tokens = enc.encode(full_text)

            # Skip if sequence is too long for this model's window.
            # HellaSwag examples are short so this almost never triggers.
            if len(tokens) > config.block_size:
                ending_losses.append(float('inf'))
                continue

            tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            # Pair-sum model: first token seeds prev_tokens, rest go in idx.
            # The model internally builds full = cat([prev_tokens, idx]) and
            # forms x[i] = wte(full[i]) + wte(full[i+1]), so logits[i] predicts
            # tokens[i+2]. (One position earlier than the standard model's
            # logits[i] → tokens[i+1].)
            prev_tokens = tokens_tensor[:, 0:1]
            idx_input = tokens_tensor[:, 1:]
            targets_dummy = idx_input  # internal loss is ignored

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _, _, _ = model(idx_input, targets_dummy,
                                        kv_caches=None, prev_tokens=prev_tokens)

            # Slice logits aligned with ending tokens.
            # ending tokens are tokens[ctx_len..N-1]; they are predicted by
            # logits[ctx_len-2 .. N-3]. With -1 endpoint that's [ctx_len-2:-1].
            ctx_len = len(ctx_tokens)
            shift_logits = logits[0, ctx_len-2:-1, :]
            shift_labels = tokens_tensor[0, ctx_len:]

            loss = F.cross_entropy(shift_logits, shift_labels, reduction='mean')
            ending_losses.append(loss.item())

        predicted = ending_losses.index(min(ending_losses))
        if predicted == label:
            num_correct += 1
            if num_printed < 3:
                print(f"\n--- Correct Example {num_printed + 1} ---")
                print(f"Context: {ctx}")
                for i, ending in enumerate(endings):
                    marker = "<-- CORRECT" if i == label else ""
                    print(f"  [{i}] (loss={ending_losses[i]:.4f}) {ending} {marker}")
                print(f"Model chose: {predicted}, Label: {label}")
                num_printed += 1
        num_total += 1

        if (idx_ex + 1) % 500 == 0:
            running_acc = num_correct / num_total
            print(f"  [{idx_ex+1}/{len(examples)}] running accuracy: {running_acc:.4f}")

accuracy = num_correct / num_total
print(f"\nFinal HellaSwag accuracy: {num_correct}/{num_total} = {accuracy:.4f}")
