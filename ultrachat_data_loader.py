import numpy as np
import torch

class DataLoader:
    def __init__(self, B, T, process_rank, process_count, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.process_count = process_count
        self.split = split

        # Load tokens from numpy file
        if split == "train":
            filepath = "ultrachat_shards/ultrachat_train.npy"
        else:
            filepath = "ultrachat_shards/ultrachat_val.npy"

        self.tokens = torch.tensor(np.load(filepath), dtype=torch.long)
        print(f"Loaded {len(self.tokens):,} tokens from {filepath}")

        self._reset_epoch()

    def _reset_epoch(self):
        """Chunk tokens into lists of size T+1. Shuffle for train, deterministic for val."""
        # Apply random offset only for training
        if self.split == "train":
            offset = torch.randint(0, self.T + 1, (1,)).item()
        else:
            offset = 0

        # Start from offset and chunk into sequences of T+1 tokens
        tokens_from_offset = self.tokens[offset:]
        num_chunks = len(tokens_from_offset) // (self.T + 1)

        # Truncate to fit exact number of chunks
        usable_tokens = tokens_from_offset[:num_chunks * (self.T + 1)]

        # Reshape into chunks of size (T+1)
        self.chunks = usable_tokens.reshape(num_chunks, self.T + 1)

        # Shuffle only for training
        if self.split == "train":
            perm = torch.randperm(num_chunks)
            self.chunks = self.chunks[perm]

        self.current_position = self.B * self.process_rank

    def get_batch(self):
        # Check if we have enough chunks left for a full batch
        if self.current_position + self.B > len(self.chunks):
            self._reset_epoch()

        # Get next B chunks
        batch_chunks = self.chunks[self.current_position:self.current_position + self.B]
        self.current_position += self.B * self.process_count

        # Split into x and y
        x = batch_chunks[:, :self.T]  # First T tokens of each chunk
        y = batch_chunks[:, 1:self.T + 1]  # Tokens 1 to T+1 (shifted by 1)
        return x, y
