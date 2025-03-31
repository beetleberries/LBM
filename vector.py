import numpy as np
from sklearn.cluster import KMeans

X = np.load("data/X_psd.npy")
y = np.load("data/y_label.npy")

print("Loaded X:", X.shape)

X_flat = X.reshape(X.shape[0], X.shape[1], -1)
all_vectors = X_flat.reshape(-1, X_flat.shape[-1])

print("Training KMeans...")
kmeans = KMeans(n_clusters=256, random_state=42)
kmeans.fit(all_vectors)

token_sequences = np.array([kmeans.predict(sample) for sample in X_flat])

np.save("data/X_tokens_vq.npy", token_sequences)
print("Saved: data/X_tokens_vq.npy", token_sequences.shape)
