# demo.py

import os
import sys
import torch
import numpy as np
import json
from torch.nn.functional import softmax

# --- Path Setup ---
sys.path.append("D:/capstone/LBM")

from model_v3 import VQVAE, EEGTransformerClassifier
from training import preprocess_demo

# --- Constants ---
SET_FILE = "D:/capstone/LBM/dataset_unsupervised/s01_051017m.set"
VQ_MODEL_PATH = "D:/capstone/LBM/training/vq_vae_model_v3.pth"
T_MODEL_PATH = "D:/capstone/LBM/training/transformer_classifier_v3.pth"
OUTPUT_JSON = "D:/capstone/LBM/training/output/s01_051017m_vigilance_results.json"

# --- Model Hyperparameters ---
VQ_EMBEDDING_DIM = 256
VQ_NUM_EMBEDDINGS = 512
VQ_COMMITMENT_COST = 0.25
VQ_DECAY = 0.99
NUM_CLASSES = 3
LABELS = ["alert", "transition", "drowsy"]

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- EEG Preprocessing ---
print("Running preprocessing on:", SET_FILE)
epochs_data, _ = preprocess_demo.run_pipeline(SET_FILE)  # shape: (N, C, T)

# --- Load VQ-VAE and Extract Embeddings ---
vq_model = VQVAE(input_channels=epochs_data.shape[1],
                 embedding_dim=VQ_EMBEDDING_DIM,
                 num_embeddings=VQ_NUM_EMBEDDINGS,
                 commitment_cost=VQ_COMMITMENT_COST,
                 decay=VQ_DECAY).to(device)
vq_model.load_state_dict(torch.load(VQ_MODEL_PATH, map_location=device))
vq_model.eval()

with torch.no_grad():
    data_tensor = torch.tensor(epochs_data, dtype=torch.float32).to(device)
    encoded_seqs = vq_model.encode(data_tensor)  # shape: (N, seq_len)

# --- Load Transformer Classifier ---
transformer_model = EEGTransformerClassifier(num_embeddings=VQ_NUM_EMBEDDINGS + 1,
                                             embedding_dim=VQ_EMBEDDING_DIM,
                                             nhead=4,
                                             num_layers=3,
                                             dim_feedforward=128,
                                             num_classes=NUM_CLASSES,
                                             dropout=0.2).to(device)
transformer_model.load_state_dict(torch.load(T_MODEL_PATH, map_location=device))
transformer_model.eval()

# --- Perform Inference ---
predicted_labels_list = []
output_results = []

with torch.no_grad():
    for i in range(encoded_seqs.shape[0]):
        input_seq = encoded_seqs[i].unsqueeze(0).to(device)  # shape: (1, seq_len)
        logits = transformer_model(input_seq)  # shape: (1, num_classes)
        probs = softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_label = LABELS[pred_idx]
        predicted_labels_list.append(pred_label)

        output_results.append({
            "epoch_index": i,
            "predicted_label": pred_label,
            "probabilities": {
                LABELS[j]: float(probs[j]) for j in range(NUM_CLASSES)
            }
        })

# --- Save All Predictions to JSON ---
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output_results, f, indent=2)

# --- Output Latest Driver State ---
latest_state = predicted_labels_list[-1]
print("✅ Current driver state:", latest_state)

summary_json_path = "D:/capstone/LBM/training/output/latest_state.json"
with open(summary_json_path, 'w') as f:
    json.dump({"latest_state": latest_state}, f, indent=2)

print(f"\nLatest driver state saved to: {summary_json_path}")
