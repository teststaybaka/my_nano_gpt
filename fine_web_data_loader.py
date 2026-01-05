import os
import numpy as np
import torch
from glob import glob

class DataLoader:
    def __init__(self, B, T, process_rank, process_count, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.process_count = process_count
        self.split = split

        # Find all shard files for this split
        shard_dir = "fineweb_shards"
        if split == "train":
            pattern = os.path.join(shard_dir, "fineweb_train_*.npy")
        else:  # val
            pattern = os.path.join(shard_dir, "fineweb_val_*.npy")

        self.shard_files = sorted(glob(pattern))
        assert len(self.shard_files) > 0, f"No shard files found for split '{split}' in {shard_dir}"

        # Initialize token buffer (can span multiple shards)
        self.token_buffer = torch.tensor([], dtype=torch.long)
        self.current_shard_idx = -1  # Haven't loaded any shard yet

        # Set starting position for this process
        self.current_position = self.B * self.T * self.process_rank

    def _load_next_shard(self):
        """Trim consumed tokens, load next shard, and reset position"""
        # Trim all consumed tokens before loading new shard
        if self.current_position > 0:
            if self.current_position >= len(self.token_buffer):
                self.current_position -= len(self.token_buffer)
                self.token_buffer = torch.tensor([], dtype=torch.long)
            else:
                self.token_buffer = self.token_buffer[self.current_position:]
                self.current_position = 0

        # Load next shard and append to buffer
        next_shard_idx = (self.current_shard_idx + 1) % len(self.shard_files)
        shard_path = self.shard_files[next_shard_idx]
        shard_tokens = torch.from_numpy(np.load(shard_path)).long()
        self.token_buffer = torch.cat([self.token_buffer, shard_tokens])
        self.current_shard_idx = next_shard_idx
        print(f"Rank {self.process_rank}: Loaded {os.path.basename(shard_path)} ({len(shard_tokens):,} tokens, buffer now {len(self.token_buffer):,} tokens)")

    def get_batch(self):
        """
        Returns consecutive batches:
        x[0] = tokens[pos:pos+T],     y[0] = tokens[pos+1:pos+T+1]
        x[1] = tokens[pos+T:pos+2T],  y[1] = tokens[pos+T+1:pos+2T+1]
        ...
        """
        # Ensure we have enough tokens in buffer from current position.
        # self.current_position can be updated during _load_next_shard(), so we need to re-calculate the needed length.
        while len(self.token_buffer) < self.current_position + self.B * self.T + 1:
            self._load_next_shard()

        # Extract consecutive batch from buffer using view (more efficient)
        x = self.token_buffer[self.current_position:self.current_position + self.B * self.T].view(self.B, self.T)
        y = self.token_buffer[self.current_position + 1:self.current_position + self.B * self.T + 1].view(self.B, self.T)

        # Advance position for next batch
        # Each process advances by B*T*process_count to avoid overlap
        self.current_position += self.B * self.T * self.process_count

        return x, y
