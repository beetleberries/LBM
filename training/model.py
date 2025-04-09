import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns # For plotting confusion matrix
import math

# --- Configuration ---
# Data Paths (MODIFY THESE)

SAVE_DIR = './processed_data' # Create this directory if it doesn't exist
EPOCHS_DATA_PATH = os.path.join(SAVE_DIR, 'processed_epochs.npy')
LABELS_PATH = os.path.join(SAVE_DIR, 'processed_labels.npy')


# Model Hyperparameters (Tune these!)
# VQ-VAE
VQ_EMBEDDING_DIM = 64       # Dimension of each codebook vector
VQ_NUM_EMBEDDINGS = 128     # Size of the codebook (K)
VQ_COMMITMENT_COST = 0.25   # Beta term in the VQ loss
VQ_DECAY = 0.99             # For EMA updates

# Transformer
T_DIM = VQ_EMBEDDING_DIM    # Dimension for Transformer (usually matches VQ embedding)
T_NHEAD = 4                 # Number of attention heads
T_NUMLAYERS = 3             # Number of Transformer encoder layers
T_DIM_FEEDFORWARD = 256     # Dimension of feedforward layers in Transformer
T_DROPOUT = 0.1

# Training Hyperparameters
NUM_CLASSES = 3            # alert, transition, drowsy
BATCH_SIZE = 32
VQ_LR = 5e-5
VQ_EPOCHS = 20 # Adjust based on convergence
T_LR = 1e-4
T_EPOCHS = 30 # Adjust based on convergence
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42 # For reproducibility

# Paths for saving models
VQ_MODEL_SAVE_PATH = './training/vq_vae_model.pth'
T_MODEL_SAVE_PATH = './training/transformer_classifier.pth'

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. Vector Quantizer Layer ---
# Often uses Exponential Moving Average (EMA) for stable codebook updates
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize the codebook embeddings
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

        # EMA tracking variables (not part of state_dict, managed manually)
        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros_like(self.embedding.weight.data)) # Use zeros_like

    def forward(self, inputs):
        # inputs: (Batch, Features, Dim) -> flatten to (Batch * Features, Dim)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances: (N, D) vs (K, D) -> (N, K)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Find closest encodings (indices)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1) # one-hot

        # Quantize and un-flatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # EMA codebook update during training
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace smoothing to avoid zero counts
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

            # Update embedding weights
            self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))


        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # Encoder loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # Codebook loss (using EMA)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))) # codebook usage

        # Reshape encoding_indices back to spatial structure for Transformer input
        # (Batch * Height * Width) -> (Batch, Height * Width) or similar
        # The exact shape depends on the encoder output before flattening
        indices_for_transformer = encoding_indices.view(input_shape[0], -1) # Simple flatten for now

        return loss, quantized, perplexity, indices_for_transformer

    def get_code_indices(self, flat_input):
         # Simplified version for inference/encoding phase
         distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
         encoding_indices = torch.argmin(distances, dim=1)
         return encoding_indices


# --- 2. VQ-VAE Model ---
class VQVAE(nn.Module):
    def __init__(self, input_channels, embedding_dim, num_embeddings, commitment_cost, decay):
        super(VQVAE, self).__init__()
        self.input_channels = input_channels # Num EEG channels

        # Example Encoder (adjust architecture based on your data timepoints)
        # Input: (Batch, Channels, Timepoints)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1), # Halves timepoints
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), # Halves timepoints again
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1), # No change
            nn.ReLU(),
            nn.Conv1d(256, embedding_dim, kernel_size=1, stride=1) # Project to embedding_dim
        )
        # Output: (Batch, embedding_dim, Timepoints / 4)

        self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)

        # Example Decoder (mirrors encoder)
        # Input: (Batch, embedding_dim, Timepoints / 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1), # Doubles timepoints
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1) # Doubles timepoints
        )
        # Output: (Batch, Channels, Timepoints)

    def forward(self, x):
        z = self.encoder(x)
        # Permute for VQ layer: (Batch, Dim, Features) -> (Batch, Features, Dim)
        z_permuted = z.permute(0, 2, 1).contiguous()
        vq_loss, quantized, perplexity, indices = self.vq_layer(z_permuted)
        # Permute back for decoder: (Batch, Features, Dim) -> (Batch, Dim, Features)
        quantized_permuted = quantized.permute(0, 2, 1).contiguous()
        x_recon = self.decoder(quantized_permuted)
        return vq_loss, x_recon, perplexity, indices

    # --- Methods for the second stage (encoding data for Transformer) ---
    def encode(self, x):
        """ Encodes input x to latent codes (indices). """
        z = self.encoder(x)
        z_permuted = z.permute(0, 2, 1).contiguous()
        flat_z = z_permuted.reshape(-1, VQ_EMBEDDING_DIM)
        indices = self.vq_layer.get_code_indices(flat_z)
        # Reshape indices back to sequence: (Batch, Features) where Features = Timepoints / 4
        indices_seq = indices.view(x.shape[0], -1)
        return indices_seq


# --- 3. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, embedding_dim)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# --- 4. Transformer Classifier Model ---
class EEGTransformerClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(EEGTransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Embedding layer for the discrete codes from VQ-VAE
        self.code_embedding = nn.Embedding(num_embeddings, embedding_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Standard Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                                    batch_first=True) # Use batch_first=True
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Classification head
        self.classifier_head = nn.Linear(embedding_dim, num_classes)

        # CLS token - learnable parameter
        # Add 1 to num_embeddings for the CLS token's unique ID
        self.cls_token_id = num_embeddings # Use the next available index
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim)) # (1, 1, Dim)

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.code_embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, code_indices):
        # code_indices shape: (Batch, Sequence Length)
        batch_size = code_indices.shape[0]
        seq_len = code_indices.shape[1]

        # Add CLS token embedding at the beginning of each sequence
        cls_tokens = self.cls_embedding.expand(batch_size, -1, -1) # (Batch, 1, Dim)

        # Embed the code indices
        embedded_codes = self.code_embedding(code_indices) # (Batch, Seq Len, Dim)

        # Concatenate CLS token and embedded codes
        x = torch.cat((cls_tokens, embedded_codes), dim=1) # (Batch, 1 + Seq Len, Dim)

        # Add positional encoding - requires shape (Seq Len + 1, Batch, Dim)
        # Transpose for positional encoding, then transpose back
        x = x.transpose(0, 1) # (1 + Seq Len, Batch, Dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # (Batch, 1 + Seq Len, Dim) - Ready for batch_first Transformer

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x) # (Batch, 1 + Seq Len, Dim)

        # Use the output corresponding to the CLS token for classification
        cls_output = transformer_output[:, 0, :] # (Batch, Dim)

        # Final classification
        logits = self.classifier_head(cls_output) # (Batch, Num Classes)
        return logits

# --- 5. Dataset Class ---
class EEGDataset(Dataset):
    def __init__(self, data, labels, label_map):
        self.data = torch.tensor(data, dtype=torch.float32)
        # Map string labels to integers
        self.labels = torch.tensor([label_map[lbl] for lbl in labels], dtype=torch.long)
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Dataset Class for Transformer (takes code indices) ---
class EncodedEEGDataset(Dataset):
    def __init__(self, encoded_data_indices, labels): # labels should already be integers
        self.encoded_data = torch.tensor(encoded_data_indices, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded_data[idx], self.labels[idx]


# --- 6. Training and Evaluation Functions ---

def train_vqvae(model, dataloader, optimizer, epochs, device):
    model.train()
    print("--- Starting VQ-VAE Training ---")
    recon_criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_vq_loss = 0.0
        total_recon_loss = 0.0
        total_perplexity = 0.0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity, _ = model(data)
            recon_loss = recon_criterion(data_recon, data)
            loss = vq_loss + recon_loss # Combine VQ loss and reconstruction loss

            loss.backward()
            optimizer.step()

            total_vq_loss += vq_loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity.item()

            #if batch_idx % 10 == 0:
                #print(f"VQ Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                #      f"Loss: {loss.item():.4f} (VQ: {vq_loss.item():.4f}, Recon: {recon_loss.item():.4f}) | "
                #      f"Perplexity: {perplexity.item():.2f}")

        avg_vq_loss = total_vq_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_perplexity = total_perplexity / len(dataloader)
        print(f"VQ Epoch {epoch+1} Avg Loss: {(avg_vq_loss + avg_recon_loss):.4f} (VQ: {avg_vq_loss:.4f}, Recon: {avg_recon_loss:.4f}, Avg Perplexity: {avg_perplexity:.2f})")


    print("--- VQ-VAE Training Finished ---")
    if (avg_perplexity < 0.1 * VQ_NUM_EMBEDDINGS):
        print(f" perplexity too low {avg_perplexity} to attempt training transformer, should be at least 10% of codebook size: {VQ_NUM_EMBEDDINGS}")
        exit()


def encode_data_for_transformer(vq_model, dataloader, device):
    vq_model.eval()
    all_indices = []
    all_labels = []
    print("--- Encoding data using trained VQ-VAE ---")
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            indices_seq = vq_model.encode(data) # Shape: (Batch, Seq Len)
            all_indices.append(indices_seq.cpu().numpy())
            all_labels.append(labels.numpy()) # Keep original labels

    all_indices = np.concatenate(all_indices, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Encoded data shape (indices): {all_indices.shape}")
    print(f"Labels shape: {all_labels.shape}")
    return all_indices, all_labels


def train_transformer(model, train_loader, val_loader, optimizer, criterion, epochs, device, label_map):
    model.train()
    print("--- Starting Transformer Training ---")
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_idx, (codes, labels) in enumerate(train_loader):
            codes = codes.to(device)
            labels = labels.to(device) # Already integer labels
            optimizer.zero_grad()

            outputs = model(codes) # Get logits
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

            #if batch_idx % 10 == 0:
            #    print(f"T Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(train_targets, train_preds)

        # --- Validation Phase ---
        val_loss, val_accuracy, _, _ = evaluate_transformer(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}, Val Loss  : {val_loss:.4f} | Val Acc  : {val_accuracy:.4f}")

        # Save model if validation accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print(f"*** New best validation accuracy: {best_val_accuracy:.4f}. Saving model... ***")
            torch.save(model.state_dict(), T_MODEL_SAVE_PATH)

    print("--- Transformer Training Finished ---")


def evaluate_transformer(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for codes, labels in dataloader:
            codes = codes.to(device)
            labels = labels.to(device)

            outputs = model(codes)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)

    return avg_loss, accuracy, all_targets, all_preds

# --- 7. Main Execution ---
if __name__ == "__main__":
    # --- Load Prepared Data ---
    print("Loading preprocessed data...")
    try:
        # Ensure shapes are (num_epochs, num_channels, num_timepoints)
        epochs_data = np.load(EPOCHS_DATA_PATH)
        labels = np.load(LABELS_PATH, allow_pickle=True) # Allow pickle if labels are strings
        print(f"Loaded epochs data shape: {epochs_data.shape}")
        print(f"Loaded labels shape: {labels.shape}")
        # Example: (1000, 30, 500) -> 1000 epochs, 30 channels, 500 timepoints
        n_channels = epochs_data.shape[1]
        n_timepoints = epochs_data.shape[2]

    except FileNotFoundError:
        print(f"Error: Data files not found at {EPOCHS_DATA_PATH} or {LABELS_PATH}")
        print("Please run the previous script to generate and save these files.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- Create Label Map ---
    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) != NUM_CLASSES:
         print(f"Warning: Found {len(unique_labels)} unique labels, but expected {NUM_CLASSES}. Check labels.")
         print(f"Unique labels found: {unique_labels}")
         # Adjust NUM_CLASSES if necessary, but ensure it matches your intent
         # NUM_CLASSES = len(unique_labels)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    reverse_label_map = {v: k for k, v in label_map.items()}
    print(f"Label Map: {label_map}")


    # --- Create Datasets and Dataloaders (for VQ-VAE training) ---
    full_dataset = EEGDataset(epochs_data, labels, label_map)
    # Note: We use the full dataset for unsupervised VQ-VAE training here.
    # A separate split could be used if desired.
    vq_dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Initialize and Train VQ-VAE ---
    vq_model = VQVAE(input_channels=n_channels,
                     embedding_dim=VQ_EMBEDDING_DIM,
                     num_embeddings=VQ_NUM_EMBEDDINGS,
                     commitment_cost=VQ_COMMITMENT_COST,
                     decay=VQ_DECAY).to(device)

    vq_optimizer = optim.AdamW(vq_model.parameters(), lr=VQ_LR)

    # Train the VQ-VAE
    train_vqvae(vq_model, vq_dataloader, vq_optimizer, VQ_EPOCHS, device)

    # Save the trained VQ-VAE
    print(f"Saving trained VQ-VAE model to {VQ_MODEL_SAVE_PATH}")
    torch.save(vq_model.state_dict(), VQ_MODEL_SAVE_PATH)

    # --- Encode the Full Dataset using Trained VQ-VAE ---
    # Create a dataloader *without* shuffling to keep data and labels aligned
    full_dataloader_no_shuffle = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    encoded_indices, original_int_labels = encode_data_for_transformer(vq_model, full_dataloader_no_shuffle, device)

    # --- Prepare Data for Transformer Training (Split Encoded Data) ---
    indices_train, indices_test, labels_train, labels_test = train_test_split(
        encoded_indices,
        original_int_labels, # Use the integer labels
        test_size=TEST_SPLIT_RATIO,
        random_state=RANDOM_SEED,
        stratify=original_int_labels # Stratify to keep class distribution similar
    )

    print(f"Encoded Train data shape: {indices_train.shape}, Labels: {labels_train.shape}")
    print(f"Encoded Test data shape: {indices_test.shape}, Labels: {labels_test.shape}")

    # Create datasets and dataloaders for the Transformer
    transformer_train_dataset = EncodedEEGDataset(indices_train, labels_train)
    transformer_test_dataset = EncodedEEGDataset(indices_test, labels_test)

    transformer_train_loader = DataLoader(transformer_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    transformer_test_loader = DataLoader(transformer_test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    # --- Initialize and Train Transformer ---
    # Add 1 to num_embeddings for the CLS token ID
    transformer_model = EEGTransformerClassifier(num_embeddings=VQ_NUM_EMBEDDINGS + 1,
                                                 embedding_dim=T_DIM,
                                                 nhead=T_NHEAD,
                                                 num_layers=T_NUMLAYERS,
                                                 dim_feedforward=T_DIM_FEEDFORWARD,
                                                 num_classes=NUM_CLASSES,
                                                 dropout=T_DROPOUT).to(device)

    t_optimizer = optim.AdamW(transformer_model.parameters(), lr=T_LR)
    # Use CrossEntropyLoss for multi-class classification
    t_criterion = nn.CrossEntropyLoss()

    # Train the Transformer (using train/test split loaders)
    train_transformer(transformer_model, transformer_train_loader, transformer_test_loader,
                      t_optimizer, t_criterion, T_EPOCHS, device, label_map)

    # --- Final Evaluation on Test Set ---
    print("\n--- Evaluating Final Transformer Model on Test Set ---")
    # Load the best model saved during training
    try:
        transformer_model.load_state_dict(torch.load(T_MODEL_SAVE_PATH, map_location=device))
        print("Loaded best saved Transformer model state.")
    except FileNotFoundError:
        print("Warning: Best model checkpoint not found. Evaluating with the final state.")
    except Exception as e:
         print(f"Error loading model checkpoint: {e}. Evaluating with final state.")


    test_loss, test_accuracy, test_targets, test_preds = evaluate_transformer(
        transformer_model, transformer_test_loader, t_criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --- Detailed Classification Report ---
    target_names = [reverse_label_map[i] for i in range(NUM_CLASSES)]
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names, zero_division=0))

    # --- Confusion Matrix ---
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set')
    plt.show()

    print("\nScript finished.")