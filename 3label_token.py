import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

vqvae_path = os.path.join(os.getcwd(), "VQ-VAE")
sys.path.append(vqvae_path)
from vqvae import VQVAE

# === 3 클래스 PSD + 라벨 불러오기 ===
X = np.load("X_psd_label3.npy")  # (samples, 40, 53)
y = np.load("y_label_3class.npy")  # (samples,)
print("Loaded PSD shape:", X.shape, "Labels shape:", y.shape)

# === 채널 3개로 줄이고 (3, 53, 53)로 확장 ===
X_tensor = torch.tensor(X[:, :3], dtype=torch.float32)
X_tensor = X_tensor.unsqueeze(-1).repeat(1, 1, 1, X.shape[2])  # (N, 3, 53, 53)
print("Converted for VQ-VAE input:", X_tensor.shape)

# === DataLoader 구성 ===
dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# === VQ-VAE 모델 생성 ===
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

# === 토크나이징 ===
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
            print(f"Skipping batch {i}: {e}")

# === 저장 ===
X_tokens = np.vstack(token_seqs)
y_filtered = np.array(valid_labels)

np.save("X_tokens_vqvae_3label.npy", X_tokens)
np.save("y_label_3class_filtered.npy", y_filtered)

print(f"Token shape: {X_tokens.shape}")
print(f"Label shape: {y_filtered.shape}")
print("Saved tokenized data for 3-class EEG classification.")
