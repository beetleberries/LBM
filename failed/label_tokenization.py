import numpy as np
from sklearn.cluster import KMeans

# 1
X = np.load("X_psd_multi.npy")  # shape: (samples, channels, freqs)
y = np.load("y_label_multi.npy")

print("data load complete:", X.shape)

# 2
X_flat = X.reshape(X.shape[0], X.shape[1], -1)  # shape: (samples, 40, 53)
print("Flattened shape:", X_flat.shape)

# 3
num_codes = 256
all_vectors = X_flat.reshape(-1, X_flat.shape[-1])
print("KMeans clustering...")

kmeans = KMeans(n_clusters=num_codes, random_state=42)
kmeans.fit(all_vectors)

# 4
token_sequences = []
for sample in X_flat:
    tokens = kmeans.predict(sample)  # 각 채널 벡터 → 토큰 인덱스
    token_sequences.append(tokens)

token_sequences = np.array(token_sequences)  # shape: (samples, seq_len)
print("Token sequences shape:", token_sequences.shape)

# 5
np.save("X_tokens_vq_multi.npy", token_sequences)
np.save("y_label_multi.npy", y)
print("saved: X_tokens_vq_multi.npy")
