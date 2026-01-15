import torch
import tiktoken
from model import GPT, GPTConfig
import torch.nn.functional as F

def load_model(checkpoint_path, device):
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = GPTConfig(**checkpoint['config'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"Loaded model from step {checkpoint.get('step', 'unknown')}")
    return model

def generate(model, enc, prompt, max_new_tokens=500, top_k=50, device='cuda'):
    # Format as chat: User: <prompt>\nAssistant:
    formatted = f"User: {prompt}\nAssistant:"
    tokens = enc.encode(formatted)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    eot_token = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})[0]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max context length if needed
            tokens_cond = tokens if tokens.size(1) <= 1024 else tokens[:, -1024:]

            logits, _ = model(tokens_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            next_token = torch.gather(topk_indices, 1, ix)

            tokens = torch.cat((tokens, next_token), dim=1)

            if next_token.item() == eot_token:
                break

    # Decode and extract assistant response
    output = enc.decode(tokens[0].tolist())
    # Extract just the assistant's response
    response = output.split("Assistant:")[-1].strip()
    # Remove EOT token if present
    response = response.replace("<|endoftext|>", "").strip()
    return response

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = "finetuned_final/model.pt"

    model = load_model(checkpoint_path, device)
    enc = tiktoken.get_encoding("gpt2")

    print("\n" + "="*50)
    print("Chat - Type your message and press Enter")
    print("Commands: 'quit' to exit, 'max=N' to set max tokens")
    print("="*50 + "\n")

    max_new_tokens = 500

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break

            if user_input.lower().startswith('max='):
                try:
                    max_new_tokens = int(user_input.split('=')[1])
                    print(f"Max tokens set to {max_new_tokens}")
                    continue
                except ValueError:
                    print("Invalid value")
                    continue

            response = generate(model, enc, user_input, max_new_tokens=max_new_tokens, device=device)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
