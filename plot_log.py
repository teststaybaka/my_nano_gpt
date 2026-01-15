import matplotlib.pyplot as plt

val_steps, val_losses = [], []
hella_steps, hella_accs = [], []

with open('pretrained_final/log.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        step = int(parts[0])
        metric = parts[1]
        value = float(parts[2])
        if metric == 'val':
            val_steps.append(step)
            val_losses.append(value)
        elif metric == 'hella':
            hella_steps.append(step)
            hella_accs.append(value)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(val_steps, val_losses)
ax1.set_xlabel('Step')
ax1.set_ylabel('Loss')
ax1.set_title('Validation Loss')

ax2.plot(hella_steps, hella_accs)
ax2.set_xlabel('Step')
ax2.set_ylabel('Accuracy')
ax2.set_title('HellaSwag Accuracy')

plt.tight_layout()
plt.savefig('pretrained_final/training_curves.png')
