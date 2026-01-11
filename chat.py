import torch
import tiktoken
from model import GPT, GPTConfig
import torch.nn.functional as F

def load_model(checkpoint_path, device):
  """Load model from checkpoint"""
  print(f"Loading model from {checkpoint_path}...")
  checkpoint = torch.load(checkpoint_path, map_location=device)

  # Create model from saved config (config is a dict)
  config = GPTConfig(**checkpoint['config'])
  model = GPT(config)

  # Handle compiled model checkpoint (strip _orig_mod. prefix)
  state_dict = checkpoint['model']
  unwrapped_state_dict = {}
  for k, v in state_dict.items():
      if k.startswith('_orig_mod.'):
          unwrapped_state_dict[k[len('_orig_mod.'):]] = v
      else:
          unwrapped_state_dict[k] = v
  model.load_state_dict(unwrapped_state_dict)
  model.to(device)
  model.eval()

  print(f"Loaded model from step {checkpoint['step']}")
  return model

def generate(model, enc, prompt, max_new_tokens=100, top_k=50, device='cuda'):
  """Generate text given a prompt"""
  # Tokenize prompt
  tokens = enc.encode(prompt)
  tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

  # Generate tokens
  with torch.no_grad():
    for _ in range(max_new_tokens):
      # Crop to max context length if needed
      tokens_cond = tokens if tokens.size(1) <= 1024 else tokens[:, -1024:]
      # forward the model to get the logits
      logits, _ = model(tokens_cond) # (B, T, vocab_size)
      # take the logits at the last position
      logits = logits[:, -1, :] # (B, vocab_size)
      # get the probabilities
      probs = F.softmax(logits, dim=-1) # (B, vocab_size)
      # do top-k sampling of 50 (huggingface pipeline default)
      # topk_probs here becomes (5, 50), tok_indices becomes (1, 50)
      topk_probs, tok_indices = torch.topk(probs, k=top_k, dim=-1)
      # select a token from the top-k candidates
      ix = torch.multinomial(topk_probs, num_samples=1) # (1, 1)
      # gather the token indices
      next_token = torch.gather(tok_indices, 1, ix) # (1, 1)
      # append to the sequence and continue
      tokens = torch.cat((tokens, next_token), dim=1) # (1, T + 1)
      if next_token.item() == enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]:
          break

  # Decode and return
  generated_tokens = tokens[0].tolist()
  return enc.decode(generated_tokens)

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  checkpoint_path = "log/model_clean.pt"

  # Load model and tokenizer
  model = load_model(checkpoint_path, device)
  enc = tiktoken.get_encoding("gpt2")

  print("\n" + "="*50)
  print("GPT Chat - Type your prompt and press Enter")
  print("Commands: 'quit' to exit")
  print("="*50 + "\n")

  max_new_tokens = 100

  while True:
    try:
      prompt = input("You: ").strip()
      if not prompt:
        continue

      if prompt.lower() == 'quit':
        print("Goodbye!")
        break

      # Handle max tokens command
      if prompt.lower().startswith('max='):
        try:
          max_new_tokens = int(prompt.split('=')[1])
          print(f"Max new tokens set to {max_new_tokens}")
          continue
        except ValueError:
          print("Invalid max tokens value")
          continue

      # Generate response
      output = generate(
        model, enc, prompt,
        max_new_tokens=max_new_tokens,
        device=device
      )
      print(f"\nGPT: {output}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break

if __name__ == "__main__":
  main()
