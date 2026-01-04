import tiktoken
import torch

class DataLoader:
  def __init__(self, B, T, process_rank, process_count):
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.process_count = process_count

    enc = tiktoken.get_encoding("gpt2")
    with open('input.txt', 'r') as f:
       text = f.read()
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)

    # Initialize for epoch-based batching
    self.current_position = 0
    self._reset_epoch()

  def _reset_epoch(self):
    """Apply random offset, chunk tokens into lists of size T+1, then shuffle the chunks"""
    # Apply random initial offset (0 to T)
    offset = torch.randint(0, self.T + 1, (1,)).item()

    # Start from offset and chunk into sequences of T+1 tokens
    tokens_from_offset = self.tokens[offset:]
    num_chunks = len(tokens_from_offset) // (self.T + 1)

    # Truncate to fit exact number of chunks
    usable_tokens = tokens_from_offset[:num_chunks * (self.T + 1)]

    # Reshape into chunks of size (T+1)
    chunks = usable_tokens.reshape(num_chunks, self.T + 1)

    # Shuffle the chunks
    perm = torch.randperm(num_chunks)
    self.shuffled_chunks = chunks[perm]

    self.current_position = self.B * self.process_rank

  def get_batch(self):
    # Check if we have enough chunks left for a full batch
    if self.current_position + self.B > len(self.shuffled_chunks):
      self._reset_epoch()

    # Get next B chunks
    batch_chunks = self.shuffled_chunks[self.current_position:self.current_position + self.B]
    self.current_position += self.B * self.process_count

    # Split into x and y
    x = batch_chunks[:, :self.T]  # First T tokens of each chunk
    y = batch_chunks[:, 1:self.T+1]  # Tokens 1 to T+1 (shifted by 1)
    return x, y
