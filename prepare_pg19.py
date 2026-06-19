import os
import argparse
import numpy as np
import tiktoken
from datasets import load_dataset
import multiprocessing as mp

def init_worker():
    global enc
    enc = tiktoken.get_encoding("gpt2")

def tokenize_book(book):
    EOT = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
    tokens = enc.encode(book['text'], allowed_special={'<|endoftext|>'})
    return [EOT] + tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='pg19_shards')
    parser.add_argument('--tokens-per-shard', type=int, default=100_000_000)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tokens_per_shard = args.tokens_per_shard
    nprocs = mp.cpu_count()

    print(f"Loading emozilla/pg19 train split…")
    ds = load_dataset("emozilla/pg19", split="train")
    print(f"  {len(ds)} books")
    print(f"Output directory: {output_dir}")
    print(f"Tokens per shard: {tokens_per_shard:,}")
    print(f"Shard 0 = validation, rest = training")
    print(f"Worker processes: {nprocs}")
    print()

    shard_idx = 0
    token_buffer = []

    with mp.Pool(nprocs, initializer=init_worker) as pool:
        for i, tokens in enumerate(pool.imap(tokenize_book, ds, chunksize=16)):
            token_buffer.extend(tokens)

            while len(token_buffer) >= tokens_per_shard:
                shard_tokens = token_buffer[:tokens_per_shard]
                token_buffer = token_buffer[tokens_per_shard:]

                if shard_idx == 0:
                    filename = f"pg19_val_{shard_idx:04d}.npy"
                else:
                    filename = f"pg19_train_{shard_idx:04d}.npy"

                filepath = os.path.join(output_dir, filename)
                np.save(filepath, np.array(shard_tokens, dtype=np.uint16))
                print(f"Saved {filename}: {len(shard_tokens):,} tokens")
                shard_idx += 1

            if (i + 1) % 1000 == 0:
                print(f"  processed {i+1}/{len(ds)} books, buffer {len(token_buffer):,} tokens")

    if token_buffer:
        if shard_idx == 0:
            filename = f"pg19_val_{shard_idx:04d}.npy"
        else:
            filename = f"pg19_train_{shard_idx:04d}.npy"

        filepath = os.path.join(output_dir, filename)
        np.save(filepath, np.array(token_buffer, dtype=np.uint16))
        print(f"Saved {filename}: {len(token_buffer):,} tokens (final shard)")
        shard_idx += 1

    print(f"\nDone! Created {shard_idx} shards.")

if __name__ == "__main__":
    main()
