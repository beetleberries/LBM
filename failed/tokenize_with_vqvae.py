import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.getcwd(), "VQ-VAE"))
from vqvae import VQVAE

# === 1. 데이터 로드 ===
X = np.load("X_psd_multi.npy")  # (samples, channels, freqs)
y = np.load("y_label_multi.npy")

print(f"✅ Loaded X: {X.shape}, y: {y.shape}")

# === 2. 3채널 입력으로 확장 (이미지 형태로 변형) ===
X_tensor = torch.tensor(X[:, :3], dtype=torch.float32)   # (N, 3, 53)
X_tensor = X_tensor.unsqueeze(-1)                         # (N, 3, 53, 1)
X_tensor = X_tensor.repeat(1, 1, 1, 53)                   # → (N, 3, 53, 53)

# === 3. DataLoader 준비 ===
dataset = TensorDataset(X_tensor, torch.tensor(y, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# === 4. VQ-VAE 모델 초기화 ===
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

# === 5. VQ-VAE 토크나이징 ===
all_token_seqs = []
all_labels = []

success_count = 0
fail_count = 0

with torch.no_grad():
    for i, (x, y_batch) in enumerate(dataloader):
        try:
            z_e = model.encode(x)  # (B, C, H, W)
            _, z_q, _, encodings = model._vq(z_e)  # encodings: (B*H*W, K)

            # 👉 encoding indices to (B, H*W)
            encoding_indices = torch.argmax(encodings, dim=1)
            batch_size, _, h, w = z_e.shape
            encoding_indices = encoding_indices.view(batch_size, h * w)

            for idx, y in zip(encoding_indices, y_batch):
                all_token_seqs.append(idx.cpu().numpy())  # (H*W,)
                all_labels.append(y.item())

            success_count += len(y_batch)

        except Exception as e:
            print(f"❌ Encoding failed for batch {i}: {e}")
            fail_count += len(y_batch)

print(f"✅ 토크나이징 성공 샘플 수: {success_count}")
print(f"❌ 토크나이징 실패 샘플 수: {fail_count}")
print(f"🔍 토큰 시퀀스 개수: {len(all_token_seqs)}")
print(f"🔍 레이블 개수: {len(all_labels)}")

# === 6. 저장 (모든 시퀀스 길이 맞추기) ===
try:
    min_len = min(seq.shape[0] for seq in all_token_seqs)
    print(f"📏 최소 토큰 길이: {min_len}")

    X_tokens = np.stack([seq[:min_len] for seq in all_token_seqs])  # (N, min_len)
    y_tokens = np.array(all_labels)

    print("X_tokens shape:", X_tokens.shape)
    print("y_tokens shape:", y_tokens.shape)

    if X_tokens.shape[0] != y_tokens.shape[0]:
        print("⚠️ 길이 불일치 발생! 저장 생략 또는 확인 필요.")
    else:
        np.save("X_tokens_vqvae.npy", X_tokens)
        np.save("y_label_multi_filtered.npy", y_tokens)
        print("✅ 저장 완료: X_tokens_vqvae.npy, y_label_multi_filtered.npy")
except Exception as e:
    print(f"❌ 저장 중 오류 발생: {e}")
