import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from vqvae import VQVAE

# 2. 데이터 로드
X = np.load("X_psd_binary.npy")  # shape: (samples, channels, freqs)
y = np.load("y_label_binary.npy")

# 3. VQ-VAE 입력 형태로 변환 (EEG → 3x53x53 이미지처럼)
X_tensor = torch.tensor(X[:, :3], dtype=torch.float32)  # RGB처럼 3채널만 사용
X_tensor = X_tensor.unsqueeze(-1).repeat(1, 1, 1, X.shape[2])  # (N, 3, F, F)

dataset = TensorDataset(X_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# 4. VQ-VAE 모델 선언
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

# 5. 토크나이즈 (인코딩 인덱스 추출)
all_token_seqs = []

with torch.no_grad():
    for batch in dataloader:
        x = batch[0]  # (B, 3, F, F)
        z_e = model.encode(x)
        _, _, _, encodings = model._vq(z_e)  # (B*H*W, K)
        encoding_indices = torch.argmax(encodings, dim=1)  # (B*H*W,)
        B = x.size(0)
        HW = encoding_indices.shape[0] // B
        encoding_indices = encoding_indices.view(B, HW)
        all_token_seqs.append(encoding_indices.cpu().numpy())

# 6. 시퀀스 길이 맞추기
min_len = min(seq.shape[1] for seq in all_token_seqs)
X_tokens = np.vstack([seq[:, :min_len] for seq in all_token_seqs])
y_filtered = y[:X_tokens.shape[0]]

# 7. 저장
np.save("X_tokens_vqvae_binary.npy", X_tokens)
np.save("y_label_binary_filtered.npy", y_filtered)

print("✅ saved: X_tokens_vqvae_binary.npy, y_label_binary_filtered.npy")
