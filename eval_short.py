"""Short-context evals: HellaSwag (4-way MCQ commonsense completion) and
LAMBADA (last-word cloze) on a trained checkpoint.

Supports all model families via --model:
  standard          → model.py                 forward(idx, targets=None) → (logits, loss)
  rope              → rope_model.py            forward(idx, targets=None) → (logits, loss)
  multi_scale       → multi_scale_model        forward(idx, targets=None) → (logits, loss)
  pair_sum          → pair_sum_model.py        forward(idx, targets=None) → (logits, loss)
  sliding           → sliding_cache_model      forward(idx, targets, kv_caches, prev_tokens)
  skewed            → skewed_model             forward(idx, targets, kv_caches, prev_tokens)
  stair             → stair_model              forward(idx, targets, caches,    prev_tokens)
                      (requires T_new ≤ block_size/2 per forward — we chunk internally)
  rope_sliding      → rope_sliding_cache_model forward(idx, targets, kv_caches) → 3-tuple
  rope_stair        → rope_stair_model         forward(idx, targets, caches)    → 3-tuple
                      (requires T_new ≤ block_size/2 — we chunk internally)

Standard/rope/multi_scale and the next-token cache models: logits[i] predicts
seq[i+1]; the pair-sum four predict seq[i+2]. We hide that asymmetry behind
compute_logp_per_token().

Sequences in HellaSwag/LAMBADA all fit in one block_size=1024 window. For most
families that means one forward; stair chunks at W/2 internally.

Usage:
  python eval_short.py <checkpoint.pt> --model {standard|rope|multi_scale|pair_sum|sliding|skewed|stair}
                                       [--task {val|hellaswag|lambada|all}]
                                       [--max-examples N]
                                       [--val-context-length 4096]

--task val recomputes the training-recipe FineWeb-edu validation loss on the
loaded checkpoint; it should match the training log's latest val line. Run it
first when numbers look wrong — a large gap means the checkpoint was trained
with different model code than this eval is importing.
"""
import os
import json
import math
import argparse
import torch
import tiktoken
import torch.nn.functional as F
from glob import glob


# ---- model dispatch -----------------------------------------------------

def load_model_module(name):
    """Returns (GPT_cls, GPTConfig_cls, family) where family ∈
    {'standard', 'pair_sum', 'cache', 'stair_cache'}. Family controls forward
    signature and logit-shift convention; underlying model class is the variant."""
    if name == 'standard':
        from model import GPT, GPTConfig
        return GPT, GPTConfig, 'standard'
    if name == 'rope':
        from rope_model import GPT, GPTConfig
        return GPT, GPTConfig, 'standard'  # same forward signature & alignment
    if name == 'multi_scale':
        from multi_scale_model import GPT, GPTConfig
        return GPT, GPTConfig, 'standard'  # same forward signature & alignment
    if name == 'pair_sum':
        from pair_sum_model import GPT, GPTConfig
        return GPT, GPTConfig, 'pair_sum'
    if name == 'sliding':
        from sliding_cache_model import GPT, GPTConfig
        return GPT, GPTConfig, 'cache'
    if name == 'skewed':
        from skewed_model import GPT, GPTConfig
        return GPT, GPTConfig, 'cache'
    if name == 'stair':
        from stair_model import GPT, GPTConfig
        return GPT, GPTConfig, 'stair_cache'
    if name == 'rope_sliding':
        from rope_sliding_cache_model import GPT, GPTConfig
        return GPT, GPTConfig, 'nt_cache'        # next-token cache: 3-tuple, no prev_tokens
    if name == 'rope_stair':
        from rope_stair_model import GPT, GPTConfig
        return GPT, GPTConfig, 'nt_stair_cache'  # next-token stair: 3-tuple, chunk at W/2
    raise SystemExit(f"unknown --model {name!r}; choose from standard|rope|multi_scale|pair_sum|"
                     f"sliding|skewed|stair|rope_sliding|rope_stair")


def load_checkpoint(checkpoint_path, GPT, GPTConfig, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = checkpoint['config']
    # Older checkpoints stored config as a plain dict; newer ones store the dataclass
    # object. Rebuild with this script's GPTConfig either way to dodge pickle mismatch.
    def _get(attr):
        return saved[attr] if isinstance(saved, dict) else getattr(saved, attr)
    config = GPTConfig(
        block_size=_get('block_size'),
        vocab_size=_get('vocab_size'),
        n_layers=_get('n_layers'),
        n_heads=_get('n_heads'),
        n_embd=_get('n_embd'),
    )
    model = GPT(config).to(device)
    # Strip torch.compile's '_orig_mod.' prefix if present.
    state_dict = {k.removeprefix('_orig_mod.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict)
    # Sanity check: every bias inits at zero and drifts once it receives gradients,
    # so a tensor that is still exactly all-zero after training was never reached by
    # the forward graph at train time (e.g. the stair_model c_proj bug) — i.e. the
    # checkpoint was trained with different model code than this eval is using.
    dead = [k for k, v in state_dict.items()
            if v.dtype.is_floating_point and v.numel() > 0 and not v.any()]
    if dead:
        print("WARNING: all-zero parameters in checkpoint — these were never trained,")
        print("likely a checkpoint/model-code mismatch; eval results will be garbage:")
        for k in dead:
            print(f"  {k}")
    model.eval()
    print(f"Loaded model from step {checkpoint['step']}")
    return model, config


# ---- per-position log-prob (the alignment-hiding core) ------------------

def compute_logp_per_token(model, family, tokens, device):
    """Returns (logps, greedy, first_scored).

    `tokens` is (1, N). For every absolute position i ≥ first_scored:
      logps[i]  = log P(tokens[0, i] | tokens[0, :i])
      greedy[i] = argmax over vocab at the same conditional.
    For i < first_scored, both arrays carry sentinel values (0.0 and -1).

    Family alignment:
      standard    : feed tokens[:-1], logits[i] predicts tokens[i+1], first_scored=1
      pair_sum    : feed tokens     , logits[i] predicts tokens[i+2], first_scored=2
      cache       : feed prev=tokens[:,:1], idx=tokens[:,1:],
                    logits[i] predicts tokens[i+2], first_scored=2
      stair_cache : same alignment as cache, but model asserts T_new ≤ W/2 so we
                    chunk idx into W/2 pieces and thread caches across them.
      nt_cache    : feed tokens[:-1] in one forward (fits one window),
                    logits[i] predicts tokens[i+1], first_scored=1
      nt_stair_cache : same alignment as nt_cache, but chunked at W/2 with
                    caches threaded across chunks.
    """
    N = tokens.size(1)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        if family == 'standard':
            logits, _ = model(tokens[:, :-1])
            shift_logits = logits[0]                     # (N-1, V), logits[i] → tokens[i+1]
            first_scored = 1
        elif family == 'pair_sum':
            logits, _ = model(tokens)
            shift_logits = logits[0, :N-2, :]            # logits[i] → tokens[i+2], drop last
            first_scored = 2
        elif family == 'cache':
            prev_tokens = tokens[:, 0:1]
            idx = tokens[:, 1:]
            logits, _, _, _ = model(idx, prev_tokens=prev_tokens)
            shift_logits = logits[0, :N-2, :]            # logits[i] → tokens[i+2], drop last
            first_scored = 2
        elif family == 'stair_cache':
            # Stair requires chunking at W/2. Positional call so the kwarg name
            # difference (caches vs kv_caches) doesn't matter.
            chunk_size = model.config.block_size // 2
            prev_tokens = tokens[:, 0:1]
            idx = tokens[:, 1:]
            M = idx.size(1)
            cache_state = None
            all_logits = []
            pos = 0
            while pos < M:
                end = min(pos + chunk_size, M)
                chunk_logits, _, cache_state, prev_tokens = model(
                    idx[:, pos:end], None, cache_state, prev_tokens)
                all_logits.append(chunk_logits)
                pos = end
            logits = torch.cat(all_logits, dim=1)         # (1, M, V); M = N-1
            shift_logits = logits[0, :N-2, :]             # logits[i] → tokens[i+2], drop last
            first_scored = 2
        elif family == 'nt_cache':
            # Sequence fits one block_size window: single forward, no cache needed.
            logits, _, _ = model(tokens[:, :-1])
            shift_logits = logits[0]                      # (N-1, V), logits[i] → tokens[i+1]
            first_scored = 1
        elif family == 'nt_stair_cache':
            # Stair requires chunking at W/2; thread caches across chunks.
            chunk_size = model.config.block_size // 2
            idx = tokens[:, :-1]
            M = idx.size(1)
            cache_state = None
            all_logits = []
            pos = 0
            while pos < M:
                end = min(pos + chunk_size, M)
                chunk_logits, _, cache_state = model(idx[:, pos:end], None, cache_state)
                all_logits.append(chunk_logits)
                pos = end
            logits = torch.cat(all_logits, dim=1)         # (1, M, V); M = N-1
            shift_logits = logits[0]                      # logits[i] → tokens[i+1]
            first_scored = 1
        else:
            raise ValueError(family)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    labels = tokens[0, first_scored:N]
    selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    greedy = shift_logits.argmax(dim=-1)

    full_logps = torch.zeros(N, device=device)
    full_logps[first_scored:] = selected
    full_greedy = torch.full((N,), -1, dtype=torch.long, device=device)
    full_greedy[first_scored:] = greedy
    return full_logps, full_greedy, first_scored


# ---- FineWeb-edu validation loss (training-recipe replica) ---------------

def eval_fineweb_val(model, family, config, device, val_steps=20, context_length=4096):
    """Recomputes the training scripts' FineWeb-edu validation loss on the loaded
    checkpoint. Should reproduce the latest `val` line of the training log (up to
    compile/kernel noise); a big gap means checkpoint/model-code mismatch.

    Recipes matched per family:
      standard/pair_sum        : train.py — B=32, one model(x, y) per block_size batch
      cache/stair_cache        : sliding_cache_train.py — B=16, BPTT span of
                                 `context_length` chunked at block_size (stair: /2),
                                 caches + prev_tokens threaded, seed token at x[:,0]
      nt_cache/nt_stair_cache  : cache_train.py — same, next-token aligned: no seed,
                                 chunks start at 0, 3-tuple forward
    """
    if not glob(os.path.join("fineweb_shards", "fineweb_val_*.npy")):
        print("  SKIP fineweb val — fineweb_shards/fineweb_val_*.npy not found "
              "(run prepare_fine_web_data.py)")
        return None
    from fine_web_data_loader import DataLoader

    W = config.block_size
    val_loss_accum = 0.0
    if family in ('standard', 'pair_sum'):
        B, T = 32, W
        loader = DataLoader(B=B, T=T, process_rank=0, process_count=1, split="val")
        print(f"FineWeb val (train.py recipe): B={B}, T={T}, {val_steps} batches")
        with torch.no_grad():
            for _ in range(val_steps):
                x, y = loader.get_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss.detach() / val_steps
    else:
        B = 16
        T = W if family in ('cache', 'nt_cache') else W // 2
        assert context_length % T == 0, f"context_length {context_length} not divisible by chunk {T}"
        sub_chunks = context_length // T
        pair_seed = family in ('cache', 'stair_cache')  # pair-sum carry needs a seed token
        T_dataloader = context_length + 1 if pair_seed else context_length
        loader = DataLoader(B=B, T=T_dataloader, process_rank=0, process_count=1, split="val")
        print(f"FineWeb val (cache recipe): B={B}, chunk T={T}, context={context_length}, "
              f"{val_steps} batches, seed={pair_seed}")
        with torch.no_grad():
            for _ in range(val_steps):
                x, y = loader.get_batch()
                x, y = x.to(device), y.to(device)
                cache_state = None
                prev_tokens = x[:, 0:1] if pair_seed else None
                sub_loss_total = 0.0
                for sub in range(sub_chunks):
                    start = (1 if pair_seed else 0) + sub * T
                    idx = x[:, start:start + T]
                    targets = y[:, start:start + T]
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        if pair_seed:
                            _, loss, cache_state, prev_tokens = model(idx, targets, cache_state, prev_tokens)
                        else:
                            _, loss, cache_state = model(idx, targets, cache_state)
                    sub_loss_total = sub_loss_total + loss
                val_loss_accum += (sub_loss_total / sub_chunks / val_steps).detach()

    val_loss = val_loss_accum.item()
    print(f"FineWeb-edu validation loss: {val_loss:.4f}")
    return val_loss


# ---- HellaSwag ----------------------------------------------------------

def eval_hellaswag(model, family, config, enc, device, max_examples=None):
    path = "hellaswag_data/hellaswag_val.jsonl"
    if not os.path.exists(path):
        print(f"  SKIP HellaSwag — {path} not found (run prepare_evals.py hellaswag)")
        return None
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    if max_examples is not None:
        examples = examples[:max_examples]
    print(f"HellaSwag: {len(examples)} examples")

    correct = 0
    with torch.no_grad():
        for idx_ex, example in enumerate(examples):
            ctx = example['ctx']
            endings = example['endings']
            label = int(example['label'])
            ctx_tokens = enc.encode(ctx)
            ctx_len = len(ctx_tokens)

            ending_avg_nll = []
            for ending in endings:
                tokens = enc.encode(ctx + " " + ending)
                if len(tokens) > config.block_size or ctx_len < 2:
                    ending_avg_nll.append(float('inf'))
                    continue
                tokens_t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logps, _, first_scored = compute_logp_per_token(model, family, tokens_t, device)
                # Score only ending tokens (positions [ctx_len .. N-1]).
                # If first_scored > ctx_len, the first few ending tokens are unscored;
                # for HellaSwag in practice ctx is long enough that first_scored=2 ≪ ctx_len.
                start = max(ctx_len, first_scored)
                ending_logps = logps[start:]
                if ending_logps.numel() == 0:
                    ending_avg_nll.append(float('inf'))
                    continue
                ending_avg_nll.append(-ending_logps.mean().item())

            predicted = ending_avg_nll.index(min(ending_avg_nll))
            if predicted == label:
                correct += 1

            if (idx_ex + 1) % 500 == 0:
                acc = correct / (idx_ex + 1)
                print(f"  [{idx_ex+1}/{len(examples)}] running acc: {acc:.4f}")

    acc = correct / len(examples)
    print(f"HellaSwag accuracy: {correct}/{len(examples)} = {acc:.4f}")
    return acc


# ---- LAMBADA ------------------------------------------------------------

def eval_lambada(model, family, config, enc, device, max_examples=None):
    path = "lambada_data/lambada_test.jsonl"
    if not os.path.exists(path):
        print(f"  SKIP LAMBADA — {path} not found (run prepare_evals.py lambada)")
        return None
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    if max_examples is not None:
        examples = examples[:max_examples]
    print(f"LAMBADA: {len(examples)} examples")

    correct = 0
    total_target_nll = 0.0
    total_target_tokens = 0
    skipped = 0

    with torch.no_grad():
        for idx_ex, example in enumerate(examples):
            text = example['text'].strip()
            # Last whitespace-delimited word is the target.
            prefix, _, _ = text.rpartition(' ')
            if not prefix:
                skipped += 1
                continue
            tokens_full = enc.encode(text)
            tokens_prefix = enc.encode(prefix)
            # Verify prefix invariance (rare BPE edge case can violate this).
            if tokens_full[:len(tokens_prefix)] != tokens_prefix:
                skipped += 1
                continue
            target_start = len(tokens_prefix)
            if target_start >= len(tokens_full):
                skipped += 1
                continue
            if len(tokens_full) > config.block_size:
                skipped += 1
                continue

            tokens_t = torch.tensor(tokens_full, dtype=torch.long, device=device).unsqueeze(0)
            logps, greedy, first_scored = compute_logp_per_token(model, family, tokens_t, device)

            start = max(target_start, first_scored)
            target_logps = logps[start:]
            if target_logps.numel() == 0:
                skipped += 1
                continue

            # Accuracy: every target-token argmax matches.
            target_labels = tokens_t[0, start:]
            target_greedy = greedy[start:]
            if torch.equal(target_greedy, target_labels):
                correct += 1

            total_target_nll += -target_logps.sum().item()
            total_target_tokens += target_logps.numel()

            if (idx_ex + 1) % 500 == 0:
                running_acc = correct / (idx_ex + 1 - skipped) if (idx_ex + 1 - skipped) > 0 else 0.0
                print(f"  [{idx_ex+1}/{len(examples)}] running acc: {running_acc:.4f}, skipped: {skipped}")

    n_eval = len(examples) - skipped
    acc = correct / n_eval if n_eval > 0 else 0.0
    ppl = float('inf') if total_target_tokens == 0 else \
          math.exp(total_target_nll / total_target_tokens)
    print(f"LAMBADA accuracy: {correct}/{n_eval} = {acc:.4f}  (skipped {skipped})")
    print(f"LAMBADA target-token perplexity: {ppl:.4f}")
    return acc, ppl


# ---- entrypoint ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--model', type=str, required=True,
                        choices=['standard', 'rope', 'multi_scale', 'pair_sum', 'sliding', 'skewed', 'stair',
                                 'rope_sliding', 'rope_stair'])
    parser.add_argument('--task', type=str, default='all',
                        choices=['val', 'hellaswag', 'lambada', 'all'])
    parser.add_argument('--val-context-length', type=int, default=4096,
                        help='BPTT span for cache-family fineweb val (match the '
                             'training run, e.g. 32768 for the 32x models).')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Limit examples per task (debug).')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    GPT, GPTConfig, family = load_model_module(args.model)
    model, config = load_checkpoint(args.checkpoint, GPT, GPTConfig, device)
    enc = tiktoken.get_encoding("gpt2")

    print(f"Family: {family}  |  block_size: {config.block_size}\n")

    if args.task in ('val', 'all'):
        eval_fineweb_val(model, family, config, device,
                         context_length=args.val_context_length)
        print()
    if args.task in ('hellaswag', 'all'):
        eval_hellaswag(model, family, config, enc, device, args.max_examples)
        print()
    if args.task in ('lambada', 'all'):
        eval_lambada(model, family, config, enc, device, args.max_examples)


if __name__ == "__main__":
    main()
