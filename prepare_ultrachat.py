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

def tokenize_conversation(example):
    """Tokenize a conversation with EOT after each assistant message"""
    EOT = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]  # 50256

    all_tokens = []
    for msg in example['messages']:
        role = msg['role'].capitalize()
        content = msg['content']
        text = f"{role}: {content}\n"
        tokens = enc.encode(text)
        all_tokens.extend(tokens)
        if role == "Assistant":
            all_tokens.append(EOT)

    return all_tokens

def process_split(split_name, output_path, nprocs, batch_size):
    """Process a single split of the dataset"""
    print(f"\nProcessing {split_name} split...")

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split_name, streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    EOT = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]
    all_tokens = [EOT]  # Start with EOT

    with mp.Pool(nprocs, initializer=init_worker) as pool:
        for tokens in pool.imap(tokenize_conversation, ds, chunksize=batch_size):
            all_tokens.extend(tokens)

    np.save(output_path, np.array(all_tokens, dtype=np.uint16))
    print(f"Saved {output_path}: {len(all_tokens):,} tokens")

def main():
    output_dir = "ultrachat_shards"
    os.makedirs(output_dir, exist_ok=True)

    nprocs = mp.cpu_count()
    batch_size = 16

    print(f"Loading dataset: HuggingFaceH4/ultrachat_200k")
    print(f"Output directory: {output_dir}")
    print(f"Worker processes: {nprocs}")

    process_split("train_sft", os.path.join(output_dir, "ultrachat_train.npy"), nprocs, batch_size)
    process_split("test_sft", os.path.join(output_dir, "ultrachat_val.npy"), nprocs, batch_size)

    print(f"\nDone!")

if __name__ == "__main__":
    main()
