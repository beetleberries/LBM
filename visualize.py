import numpy as np
import matplotlib.pyplot as plt

X_psd = np.load("data/X_psd.npy")
X_tokens = np.load("data/X_tokens_vq.npy")

plt.figure()
plt.imshow(X_psd[0], aspect='auto', cmap='viridis')
plt.colorbar(label='Log Power (dB)')
plt.title("PSD Heatmap (Sample 0)")
plt.xlabel("Frequency Bin")
plt.ylabel("EEG Channel")
plt.show()

plt.figure(figsize=(10, 2))
plt.plot(X_tokens[0], lw=1)
plt.title("Token Sequence (Sample 0)")
plt.xlabel("Time Step")
plt.ylabel("Token Index")
plt.show()

plt.figure()
plt.hist(X_tokens.flatten(), bins=256)
plt.title("Token Frequency Distribution")
plt.xlabel("Token Index")
plt.ylabel("Count")
plt.show()
