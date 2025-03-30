import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

# load preprocessed data
X = np.load("X_psd.npy")  # shape: (samples, channels, freqs)
y = np.load("y_label.npy")

print("data loaded:", X.shape)

# Flatten (channel × frequency) → vector sequence
X_flat = X.reshape(X.shape[0], X.shape[1], -1)  # (samples, seq_len, dim)
print("Flattened shape:", X_flat.shape)

# VQ (Vector Quantization)
num_codes = 256  
all_vectors = X_flat.reshape(-1, X_flat.shape[-1]) 

print("KMeans training..")
kmeans = KMeans(n_clusters=num_codes, random_state=42)
kmeans.fit(all_vectors)
codebook = kmeans.cluster_centers_

# Token sequence
token_sequences = []
for sample in X_flat:
    tokens = kmeans.predict(sample)  # vector to closest codebook index
    token_sequences.append(tokens)

token_sequences = np.array(token_sequences) 
print("Token sequence shape:", token_sequences.shape)

# save
np.save("X_tokens_vq.npy", token_sequences)
np.save("y_label.npy", y)  
print("saved: X_tokens_vq.npy")
