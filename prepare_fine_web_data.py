import os
import numpy as np
import tiktoken
from datasets import load_dataset
import multiprocessing as mp

# Global encoder for multiprocessing workers
def init_worker():
    """Initialize encoder in each worker process"""
    global enc
    enc = tiktoken.get_encoding("gpt2")

def tokenize_doc(doc):
    """Tokenize a single document with EOT prepended"""
    EOT = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]  # 50256
    tokens = enc.encode(doc['text'], allowed_special={'<|endoftext|>'})
    return [EOT] + tokens  # Prepend EOT to each document

def main():
    # Configuration
    dataset_name = "HuggingFaceFW/fineweb-edu"
    sample_name = "sample-10BT"
    output_dir = "fineweb_shards"
    os.makedirs(output_dir, exist_ok=True)

    tokens_per_shard = 100_000_000  # 100M tokens per shard
    nprocs = mp.cpu_count()
    batch_size = 16  # Process 16 docs at a time in parallel

    print(f"Loading dataset: {dataset_name}/{sample_name}")
    print(f"Output directory: {output_dir}")
    print(f"Tokens per shard: {tokens_per_shard:,}")
    print(f"Shard 0 = validation, rest = training")
    print(f"Worker processes: {nprocs}")
    print(f"Process until dataset is exhausted...")
    print()

    # Load dataset (streaming to avoid loading everything into memory)
    ds = load_dataset(dataset_name, name=sample_name, split="train", streaming=True)

    # Initialize tracking variables
    shard_idx = 0
    token_buffer = []

    with mp.Pool(nprocs, initializer=init_worker) as pool:
        # Process entire dataset with imap, chunksize controls batching
        for tokens in pool.imap(tokenize_doc, ds, chunksize=batch_size):
            token_buffer.extend(tokens)

            # Save shards when buffer has enough tokens
            while len(token_buffer) >= tokens_per_shard:
                # Extract one shard worth of tokens
                shard_tokens = token_buffer[:tokens_per_shard]
                token_buffer = token_buffer[tokens_per_shard:]

                # Determine filename (first shard is validation)
                if shard_idx == 0:
                    filename = f"fineweb_val_{shard_idx:04d}.npy"
                else:
                    filename = f"fineweb_train_{shard_idx:04d}.npy"

                # Save as numpy array in uint16 format
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, np.array(shard_tokens, dtype=np.uint16))

                print(f"Saved {filename}: {len(shard_tokens):,} tokens")

                shard_idx += 1

        # Save any remaining tokens as final shard
        if token_buffer:
            if shard_idx == 0:
                filename = f"fineweb_val_{shard_idx:04d}.npy"
            else:
                filename = f"fineweb_train_{shard_idx:04d}.npy"

            filepath = os.path.join(output_dir, filename)
            np.save(filepath, np.array(token_buffer, dtype=np.uint16))
            print(f"Saved {filename}: {len(token_buffer):,} tokens (final shard)")
            shard_idx += 1

    print(f"\nDone! Created {shard_idx} shards.")

if __name__ == "__main__":
    main()
