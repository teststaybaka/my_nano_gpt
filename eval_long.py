"""Long-context perplexity eval on PG19 validation books.

Two chunking strategies depending on model family:

  standard / rope / multi_scale / pair_sum  → sliding-window stride. Each window
      is block_size tokens; subsequent windows advance by --stride tokens. Each
      token is scored from the window where it has maximum in-window left context.

  sliding / skewed   → cache carry-forward. Chunks of block_size, with
      kv_caches and prev_tokens threaded forward, just like training.

  stair              → cache carry-forward with chunks of block_size/2 (stair's
      strict per-query W/2 stair requires T_new ≤ W/2 per forward).

Reports per-length perplexity over the first --lengths tokens of each
qualifying book, so you get a curve showing context utility across
families and over distances.

Usage:
  python eval_long.py <checkpoint.pt> --model {standard|rope|multi_scale|pair_sum|sliding|skewed|stair}
                                       [--lengths 1024,2048,4096,8192,16384]
                                       [--stride 512]
                                       [--max-books N]
"""
import os
import json
import math
import argparse
import torch
import tiktoken
import torch.nn.functional as F

from eval_short import load_model_module, load_checkpoint


# ---- cache family: carry kv_caches & prev_tokens forward ----------------

def eval_long_cache(model, tokens_t, T, device):
    """tokens_t: (1, N) long. T is the per-forward chunk size
    (block_size for sliding/skewed, block_size/2 for stair).

    Returns (nll_buf, scored_buf) of shape (N,) each, where
      nll_buf[p]    = -log P(tokens[p] | tokens[:p]) for scored positions, 0 otherwise
      scored_buf[p] = True iff position p was scored.

    Cache-family alignment (with prev_tokens = tokens[:,pos-1:pos], idx = tokens[:,pos:end]):
      logits[i] predicts absolute position (pos + i + 1).
    The very first 2 positions are never scorable (pair-sum cold start).

    Model is called positionally: model(idx, None, cache_state, prev_tokens) — this
    works for both sliding/skewed (kwarg kv_caches) and stair (kwarg caches).
    """
    N = tokens_t.size(1)
    nll_buf = torch.zeros(N, device=device)
    scored_buf = torch.zeros(N, dtype=torch.bool, device=device)

    cache_state = None  # kv_caches or caches depending on model — opaque here
    prev_tokens = tokens_t[:, 0:1]
    pos = 1  # absolute position of first new idx token

    while pos < N:
        end = min(pos + T, N)
        idx = tokens_t[:, pos:end]
        T_new = end - pos
        if T_new < 1:
            break
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, _, cache_state, prev_tokens = model(idx, None, cache_state, prev_tokens)
        # logits[i] predicts absolute position pos+i+1. Valid iff pos+i+1 < N
        # AND iff i < T_new (so we don't run past returned logits).
        max_i = min(T_new - 1, N - pos - 2)
        if max_i >= 0:
            sel = logits[0, :max_i+1, :].float()
            labels = tokens_t[0, pos+1:pos+max_i+2]
            log_probs = F.log_softmax(sel, dim=-1)
            nlls = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            nll_buf[pos+1:pos+max_i+2] = nlls
            scored_buf[pos+1:pos+max_i+2] = True
        pos = end

    return nll_buf, scored_buf


# ---- standard / pair_sum: sliding-window stride -------------------------

def eval_long_stride(model, family, tokens_t, W, stride, device):
    """Sliding-window perplexity. Each absolute position is scored exactly once,
    from the earliest window in which it sits at local index ≥ W-stride
    (giving it ≥ W-stride tokens of in-window left context). Window 0 also
    scores its early positions, which have less context but no alternative.

    Alignment per family:
      standard : logits[i] predicts chunk[i+1] → absolute window_start+i+1
      pair_sum : logits[i] predicts chunk[i+2] → absolute window_start+i+2
    """
    assert family in ('standard', 'pair_sum')
    N = tokens_t.size(1)
    first_scored = 1 if family == 'standard' else 2
    nll_buf = torch.zeros(N, device=device)
    scored_buf = torch.zeros(N, dtype=torch.bool, device=device)

    k = 0
    while True:
        window_start = k * stride
        if window_start >= N:
            break
        window_end = min(window_start + W, N)
        T_actual = window_end - window_start
        if T_actual < first_scored + 1:
            break

        chunk = tokens_t[:, window_start:window_end]
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            if family == 'standard':
                logits, _ = model(chunk)                       # (1, T_actual, V)
                shift_logits = logits[0, :T_actual-1, :]       # drop last (predicts unknown)
                pred_abs_start = window_start + 1
            else:  # pair_sum
                logits, _ = model(chunk)                       # (1, T_actual-1, V)
                shift_logits = logits[0, :T_actual-2, :]
                pred_abs_start = window_start + 2

        # Determine which absolute positions to score from this window.
        if k == 0:
            score_abs_start = pred_abs_start
        else:
            # Only score positions that sit in the last `stride` slots of the window
            # — i.e., positions p with (p - window_start) >= W - stride.
            score_abs_start = max(pred_abs_start, window_start + (W - stride))
        score_abs_end = window_end  # exclusive

        local_start = score_abs_start - pred_abs_start
        local_end = min(score_abs_end - pred_abs_start, shift_logits.size(0))

        if local_start < local_end:
            sel = shift_logits[local_start:local_end].float()
            labels = tokens_t[0, score_abs_start:score_abs_start + (local_end - local_start)]
            log_probs = F.log_softmax(sel, dim=-1)
            nlls = -log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            nll_buf[score_abs_start:score_abs_start + nlls.size(0)] = nlls
            scored_buf[score_abs_start:score_abs_start + nlls.size(0)] = True

        if window_end >= N:
            break
        k += 1

    return nll_buf, scored_buf


# ---- main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to .pt checkpoint')
    parser.add_argument('--model', required=True,
                        choices=['standard', 'rope', 'multi_scale', 'pair_sum', 'sliding', 'skewed', 'stair'])
    parser.add_argument('--lengths', default='1024,2048,4096,8192,16384',
                        help='Comma-separated lengths at which to report perplexity.')
    parser.add_argument('--stride', type=int, default=512,
                        help='Stride for standard/pair_sum sliding-window eval (ignored for cache).')
    parser.add_argument('--max-books', type=int, default=None,
                        help='Limit to first N qualifying books (debug).')
    args = parser.parse_args()

    lengths = sorted({int(x) for x in args.lengths.split(',')})
    max_L = max(lengths)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_float32_matmul_precision('high')

    GPT, GPTConfig, family = load_model_module(args.model)
    model, config = load_checkpoint(args.checkpoint, GPT, GPTConfig, device)
    print(f"Family: {family}  |  block_size: {config.block_size}")
    if family in ('standard', 'pair_sum'):
        if args.stride > config.block_size:
            raise SystemExit(f"--stride ({args.stride}) cannot exceed block_size ({config.block_size})")
        print(f"Strategy: sliding-window stride (W={config.block_size}, S={args.stride})")
    elif family == 'stair_cache':
        print(f"Strategy: cache carry-forward (chunk T={config.block_size // 2}, stair W/2)")
    else:
        print(f"Strategy: cache carry-forward (chunk T={config.block_size})")
    print(f"Lengths: {lengths}\n")

    path = "pg19_data/pg19_val.jsonl"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run `python prepare_evals.py pg19` first")
    with open(path) as f:
        books = [json.loads(line) for line in f]

    enc = tiktoken.get_encoding("gpt2")

    # Tokenize on demand; skip books too short for the shortest target length.
    total_nll = {L: 0.0 for L in lengths}
    total_n = {L: 0 for L in lengths}
    books_used = {L: 0 for L in lengths}

    processed = 0
    for book_idx, book in enumerate(books):
        tokens = enc.encode(book['text'])[:max_L]
        if len(tokens) < lengths[0]:
            continue  # too short for any target length
        tokens_t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            if family == 'cache':
                nll_buf, scored_buf = eval_long_cache(model, tokens_t, config.block_size, device)
            elif family == 'stair_cache':
                nll_buf, scored_buf = eval_long_cache(model, tokens_t, config.block_size // 2, device)
            else:
                nll_buf, scored_buf = eval_long_stride(
                    model, family, tokens_t, config.block_size, args.stride, device)

        for L in lengths:
            if len(tokens) < L:
                continue
            sub_scored = scored_buf[:L]
            sub_nll = nll_buf[:L]
            n = int(sub_scored.sum().item())
            if n == 0:
                continue
            total_nll[L] += float(sub_nll[sub_scored].sum().item())
            total_n[L] += n
            books_used[L] += 1

        processed += 1
        print(f"  book {book_idx+1}/{len(books)}: "
              f"{len(tokens):,} tokens used; cumulative books={processed}")
        if args.max_books is not None and processed >= args.max_books:
            break

    print()
    print(f"{'Length':>10} {'#books':>8} {'#tokens':>14} {'avg NLL':>10} {'PPL':>10}")
    for L in lengths:
        n = total_n[L]
        if n == 0:
            print(f"{L:>10} {books_used[L]:>8} {n:>14}     (no data)")
            continue
        avg = total_nll[L] / n
        ppl = math.exp(avg)
        print(f"{L:>10} {books_used[L]:>8} {n:>14,} {avg:>10.4f} {ppl:>10.4f}")


if __name__ == "__main__":
    main()
