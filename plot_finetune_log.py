import matplotlib.pyplot as plt

steps, losses = [], []

with open('finetuned_final/log.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            steps.append(int(parts[0]))
            losses.append(float(parts[2]))

plt.figure(figsize=(8, 5))
plt.plot(steps, losses, marker='o')
plt.xlabel('Step')
plt.ylabel('Validation Loss')
plt.title('Fine-tuning Validation Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('finetuned_final/finetune_loss.png')
