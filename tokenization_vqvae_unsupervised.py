import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

vqvae_path = os.path.join(os.getcwd(), "unsupervised", "VQ-VAE")
sys.path.append(vqvae_path)
from vqvae import VQVAE

X = np.load("X_psd_multi.npy")  # (samples, 40, 53)
y = np.load("y_label_multi.npy")  # (samples,)
print("Original PSD shape:", X.shape)

X_tensor = torch.tensor(X[:, :3], dtype=torch.float32)
X_tensor = X_tensor.unsqueeze(-1).repeat(1, 1, 1, X.shape[2])  # (N, 3, 53, 53)
print("Converted for VQ-VAE input:", X_tensor.shape)

dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

model = VQVAE(
    num_hiddens=128,
    num_residual_layers=2,
    num_residual_hiddens=32,
    num_embeddings=256,
    embedding_dim=64,
    commitment_cost=0.25,
    decay=0.99
)
model.eval()

token_seqs = []
valid_labels = []

with torch.no_grad():
    for i, (batch,) in enumerate(dataloader):
        z_e = model.encode(batch)
        _, _, _, encodings = model._vq(z_e)

        B, C, H, W = z_e.shape
        try:
            tokens = torch.argmax(encodings, dim=1).view(B, H * W)
            token_seqs.append(tokens.cpu().numpy())
            start = i * dataloader.batch_size
            valid_labels.extend(y[start : start + B])
        except Exception as e:
            print(f"⚠️ Skipping batch {i}: {e}")

X_tokens = np.vstack(token_seqs)
y_filtered = np.array(valid_labels)

np.save("X_tokens_vqvae.npy", X_tokens)
np.save("y_label_multi_filtered.npy", y_filtered)

print(f"✅ Token shape: {X_tokens.shape}, Labels: {y_filtered.shape}")
