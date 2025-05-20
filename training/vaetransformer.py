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
import argparse

# --- Configuration ---
# Data Paths (MODIFY THESE IF NEEDED)
EPOCHS_DATA_PATH = "D:/EEG/processed/stimlocked_psd_features_labeled_features.npy"
LABELS_PATH = "D:/EEG/processed/stimlocked_psd_features_labeled_labels.npy"

# --- Global Lists for Plotting ---
vae_train_recon_losses = []
vae_train_kl_losses = []
vae_val_recon_losses = []
vae_val_kl_losses = []
tf_train_accuracies = [] # Renamed from tf_test_accuracies for clarity
tf_val_accuracies = []


# Model Hyperparameters (Tune these!)
# VAE
VAE_LATENT_DIM = 128        # Dimension of the latent space (per time step of encoder output)
VAE_BETA_KL = 1.0           # Weight for KL divergence term in VAE loss (can be tuned, e.g. 0.001, 0.01, 0.1, 1)
VAE_LR = 1e-4
VAE_EPOCHS = 100 # Adjust based on convergence

# Transformer
T_DIM = VAE_LATENT_DIM      # Dimension for Transformer (matches VAE latent dim)
T_NHEAD = 8                 # Number of attention heads
T_NUMLAYERS = 6             # Number of Transformer encoder layers
T_DIM_FEEDFORWARD = 256     # Dimension of feedforward layers in Transformer
T_DROPOUT = 0.2
T_LR = 1e-4
T_EPOCHS = 150 # Adjust based on convergence

# Training Hyperparameters
NUM_CLASSES = 3            # alert, transition, drowsy
BATCH_SIZE = 64
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42 # For reproducibility

# Paths for saving models
VAE_MODEL_SAVE_PATH = './training/vae_model.pth' # Updated path
T_MODEL_SAVE_PATH = './training/vae_transformer_classifier.pth' # Updated path

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. VAE Model ---
class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, kl_beta=1.0):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.kl_beta = kl_beta

        # Encoder
        # Input: (Batch, Channels, Timepoints)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1), # Halves timepoints
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), # Halves timepoints again
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1), # No change in timepoints
            nn.ReLU()
            # Output: (Batch, 256, Timepoints / 4)
        )

        # These layers will output parameters for each "pixel" or "time step" of the convolved feature map
        self.conv_mu = nn.Conv1d(256, latent_dim, kernel_size=1, stride=1)
        self.conv_log_var = nn.Conv1d(256, latent_dim, kernel_size=1, stride=1)
        # Output of conv_mu/conv_log_var: (Batch, latent_dim, Timepoints / 4)

        # Decoder
        # Input: (Batch, latent_dim, Timepoints / 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1), # Doubles timepoints
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1) # Doubles timepoints
            # Optional: nn.Tanh() or nn.Sigmoid() if data is normalized to [-1,1] or [0,1]
        )
        # Output: (Batch, Channels, Timepoints)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder_conv(x)    # (Batch, 256, ReducedTimepoints)
        mu = self.conv_mu(h)        # (Batch, latent_dim, ReducedTimepoints)
        log_var = self.conv_log_var(h)  # (Batch, latent_dim, ReducedTimepoints)

        z = self.reparameterize(mu, log_var) # (Batch, latent_dim, ReducedTimepoints)
        x_recon = self.decoder(z)

        # Reconstruction Loss (MSE) - averages over all elements
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')

        # KL Divergence
        # Sum over latent dimensions (latent_dim and ReducedTimepoints), then mean over batch
        # kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=list(range(1, mu.ndim)))
        # Corrected KLD sum dimensions. mu and log_var are (B, D, T_reduced)
        # We want to sum over D and T_reduced
        kld_elements = 1 + log_var - mu.pow(2) - log_var.exp()
        kld = -0.5 * torch.sum(kld_elements, dim=[1, 2]) # Sum over latent_dim and ReducedTimepoints
        kld_loss = torch.mean(kld) # Average over batch

        total_loss = recon_loss + self.kl_beta * kld_loss

        # Latent representation for transformer (using mu for stability)
        # Permute mu to (Batch, ReducedTimepoints, latent_dim)
        mu_for_transformer = mu.permute(0, 2, 1).contiguous()

        return total_loss, x_recon, recon_loss, kld_loss, mu_for_transformer

    def encode(self, x):
        """ Encodes input x to a sequence of latent mean vectors (mu). """
        h = self.encoder_conv(x)
        mu = self.conv_mu(h) # (Batch, latent_dim, ReducedTimepoints)
        # Permute for transformer: (Batch, ReducedTimepoints, latent_dim)
        mu_permuted = mu.permute(0, 2, 1).contiguous()
        return mu_permuted


# --- 2. Positional Encoding (Unchanged) ---
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

# --- 3. Transformer Classifier Model (Adapted for continuous latent vectors) ---
class EEGTransformerClassifier(nn.Module):
    def __init__(self, latent_dim, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(EEGTransformerClassifier, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim # This is T_DIM

        self.pos_encoder = PositionalEncoding(latent_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier_head = nn.Linear(latent_dim, num_classes)

        self.cls_embedding = nn.Parameter(torch.randn(1, 1, latent_dim)) # (1, 1, Dim)

        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        # No code_embedding to initialize
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, latent_sequences):
        # latent_sequences shape: (Batch, Sequence Length, latent_dim)
        batch_size = latent_sequences.shape[0]

        cls_tokens = self.cls_embedding.expand(batch_size, -1, -1) # (Batch, 1, latent_dim)
        x = torch.cat((cls_tokens, latent_sequences), dim=1) # (Batch, 1 + Seq Len, latent_dim)

        x = x.transpose(0, 1) # (1 + Seq Len, Batch, latent_dim)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1) # (Batch, 1 + Seq Len, latent_dim)

        transformer_output = self.transformer_encoder(x) # (Batch, 1 + Seq Len, latent_dim)
        cls_output = transformer_output[:, 0, :] # (Batch, latent_dim)
        logits = self.classifier_head(cls_output) # (Batch, Num Classes)
        return logits

# --- 4. Dataset Classes ---
class EEGDataset(Dataset): # For VAE training (Unchanged from VQ-VAE version)
    def __init__(self, data, labels, label_map):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor([label_map[lbl] for lbl in labels], dtype=torch.long)
        self.label_map = label_map
        self.reverse_label_map = {v: k for k, v in label_map.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class LatentEEGDataset(Dataset): # For Transformer training with continuous latent vectors
    def __init__(self, latent_data, labels): # labels should already be integers
        self.latent_data = torch.tensor(latent_data, dtype=torch.float32) # Data is float
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latent_data[idx], self.labels[idx]


# --- 5. Training and Evaluation Functions ---

def train_vae(model, train_loader, val_loader, optimizer, epochs, device):
    model.train()
    print("--- Starting VAE Training ---")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        total_train_recon_loss = 0.0
        total_train_kld_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            vae_loss, _, recon_loss, kld_loss, _ = model(data)

            vae_loss.backward()
            optimizer.step()

            total_train_loss += vae_loss.item()
            total_train_recon_loss += recon_loss.item()
            total_train_kld_loss += kld_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon_loss = total_train_recon_loss / len(train_loader)
        avg_train_kld_loss = total_train_kld_loss / len(train_loader)

        # Store training metrics
        vae_train_recon_losses.append(avg_train_recon_loss)
        vae_train_kl_losses.append(avg_train_kld_loss)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        total_val_recon_loss = 0.0
        total_val_kld_loss = 0.0

        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                vae_loss, _, recon_loss, kld_loss, _ = model(data)

                total_val_loss += vae_loss.item()
                total_val_recon_loss += recon_loss.item()
                total_val_kld_loss += kld_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_recon_loss = total_val_recon_loss / len(val_loader)
        avg_val_kld_loss = total_val_kld_loss / len(val_loader)

        # Store validation metrics
        vae_val_recon_losses.append(avg_val_recon_loss)
        vae_val_kl_losses.append(avg_val_kld_loss)

        print(f"VAE Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} (Recon: {avg_train_recon_loss:.4f}, KL: {avg_train_kld_loss:.4f}) | "
              f"Val Loss  : {avg_val_loss:.4f} (Recon: {avg_val_recon_loss:.4f}, KL: {avg_val_kld_loss:.4f})")

    print("--- VAE Training Finished ---")


def encode_data_for_transformer_vae(vae_model, dataloader, device):
    vae_model.eval()
    all_latent_vectors = []
    all_labels = []
    print("--- Encoding data using trained VAE ---")
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)
            # encode returns mu_permuted: (Batch, ReducedTimepoints, latent_dim)
            latent_seq = vae_model.encode(data)
            all_latent_vectors.append(latent_seq.cpu().numpy())
            all_labels.append(labels.numpy())

    all_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Encoded latent data shape: {all_latent_vectors.shape}")
    print(f"Labels shape: {all_labels.shape}")
    return all_latent_vectors, all_labels


# train_transformer and evaluate_transformer remain largely the same,
# but will use LatentEEGDataset and the adapted EEGTransformerClassifier.
# The input `codes` to train_transformer will now be `latent_sequences`.
def train_transformer(model, train_loader, val_loader, optimizer, criterion, epochs, device, label_map):
    model.train()
    print("--- Starting Transformer Training ---")
    best_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_preds = []
        train_targets = []

        for batch_idx, (latent_sequences, labels) in enumerate(train_loader):
            latent_sequences = latent_sequences.to(device) # These are continuous vectors
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(latent_sequences)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(train_targets, train_preds)

        val_loss, val_accuracy, _, _ = evaluate_transformer(model, val_loader, criterion, device)

        tf_train_accuracies.append(train_accuracy) # Storing train accuracy
        tf_val_accuracies.append(val_accuracy)
        print(f"TF Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")

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
        for latent_sequences, labels in dataloader: # Input is latent_sequences
            latent_sequences = latent_sequences.to(device)
            labels = labels.to(device)

            outputs = model(latent_sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_targets, all_preds)
    return avg_loss, accuracy, all_targets, all_preds

# --- 6. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE and Transformer for EEG classification.')
    parser.add_argument('--load-vae', action='store_true', help='Load pre-trained VAE model instead of training.') # Updated
    parser.add_argument('--load-transformer', action='store_true', help='Load pre-trained Transformer model.')
    args = parser.parse_args()
    print(f"Arguments: Load VAE={args.load_vae}, Load Transformer={args.load_transformer}")

    print("Loading preprocessed data...")
    try:
        epochs_data = np.load(EPOCHS_DATA_PATH)
        labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"Loaded epochs data shape: {epochs_data.shape}") # (num_epochs, num_channels, num_timepoints)
        print(f"Loaded labels shape: {labels.shape}")
        n_channels = epochs_data.shape[1]
        # n_timepoints = epochs_data.shape[2] # Not strictly needed by model __init__ if using full conv
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    unique_labels = sorted(list(set(labels)))
    if len(unique_labels) != NUM_CLASSES:
         print(f"Warning: Found {len(unique_labels)} unique labels, but expected {NUM_CLASSES}. Unique: {unique_labels}")
         # NUM_CLASSES = len(unique_labels) # Optionally adjust
    label_map = {label: i for i, label in enumerate(unique_labels)}
    reverse_label_map = {v: k for k, v in label_map.items()}
    print(f"Label Map: {label_map}")

    # Split data for VAE training and validation
    vae_train_data, vae_val_data, vae_train_labels, vae_val_labels = train_test_split(
        epochs_data, labels, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=labels
    )
    print(f"VAE Train data shape: {vae_train_data.shape}")
    print(f"VAE Validation data shape: {vae_val_data.shape}")

    vae_train_dataset = EEGDataset(vae_train_data, vae_train_labels, label_map)
    vae_val_dataset = EEGDataset(vae_val_data, vae_val_labels, label_map)
    vae_train_loader = DataLoader(vae_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    vae_val_loader = DataLoader(vae_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize VAE
    vae_model = VAE(input_channels=n_channels,
                    latent_dim=VAE_LATENT_DIM,
                    kl_beta=VAE_BETA_KL).to(device)
    vae_optimizer = optim.AdamW(vae_model.parameters(), lr=VAE_LR)

    if args.load_vae:
        if os.path.exists(VAE_MODEL_SAVE_PATH):
            print(f"Loading pre-trained VAE model from {VAE_MODEL_SAVE_PATH}")
            vae_model.load_state_dict(torch.load(VAE_MODEL_SAVE_PATH, map_location=device))
            vae_model.eval()
        else:
            print(f"Error: --load-vae specified, but model file not found at {VAE_MODEL_SAVE_PATH}. Exiting.")
            exit()
    else:
        train_vae(vae_model, vae_train_loader, vae_val_loader, vae_optimizer, VAE_EPOCHS, device)
        print(f"Saving trained VAE model to {VAE_MODEL_SAVE_PATH}")
        torch.save(vae_model.state_dict(), VAE_MODEL_SAVE_PATH)

    # Encode the Full Dataset using Trained VAE
    full_dataset = EEGDataset(epochs_data, labels, label_map) # Uses original string labels, maps internally
    full_dataloader_no_shuffle = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # encode_data_for_transformer_vae returns latent vectors and original *integer mapped* labels
    # The labels from full_dataset.__getitem__ are already integers due to label_map
    # So we need to get those mapped integer labels directly from the dataset if we want to pass them
    # to encode_data_for_transformer_vae, or let it handle it.
    # The current encode_data_for_transformer_vae expects labels from dataloader, which are already mapped.
    latent_vectors, original_int_labels = encode_data_for_transformer_vae(vae_model, full_dataloader_no_shuffle, device)


    # Prepare Data for Transformer Training (Split Encoded Data)
    latent_train, latent_test, labels_train, labels_test = train_test_split(
        latent_vectors, original_int_labels,
        test_size=TEST_SPLIT_RATIO, random_state=RANDOM_SEED, stratify=original_int_labels
    )

    # Class Weights for imbalanced datasets (for Transformer loss)
    print("\nCalculating class weights for weighted loss...")
    label_counts = np.bincount(original_int_labels, minlength=NUM_CLASSES)
    total_samples = len(original_int_labels)
    class_weights = total_samples / (NUM_CLASSES * label_counts + 1e-9)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class counts: {label_counts}")
    print(f"Calculated class weights: {class_weights_tensor.cpu().numpy()}")


    print(f"Encoded Train latent data shape: {latent_train.shape}, Labels: {labels_train.shape}")
    print(f"Encoded Test latent data shape: {latent_test.shape}, Labels: {labels_test.shape}")

    transformer_train_dataset = LatentEEGDataset(latent_train, labels_train)
    transformer_test_dataset = LatentEEGDataset(latent_test, labels_test)

    from torch.utils.data import WeightedRandomSampler
    print("\nConfiguring WeightedRandomSampler for balanced batches...")
    labels_train_np = np.array(labels_train)
    class_counts_train = np.bincount(labels_train_np, minlength=NUM_CLASSES)
    sample_weights = np.array([1.0 / class_counts_train[lbl] if class_counts_train[lbl] > 0 else 0 for lbl in labels_train_np]) # Avoid div by zero
    sample_weights_tensor = torch.DoubleTensor(sample_weights)
    train_sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor), replacement=True)
    print(f"Training class counts: {class_counts_train}")

    transformer_train_loader = DataLoader(transformer_train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    # transformer_train_loader = DataLoader(transformer_train_dataset, batch_size=BATCH_SIZE, shuffle=True) # If not using sampler
    transformer_test_loader = DataLoader(transformer_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize and Train Transformer
    transformer_model = EEGTransformerClassifier(latent_dim=T_DIM, # Use T_DIM which is VAE_LATENT_DIM
                                                 nhead=T_NHEAD,
                                                 num_layers=T_NUMLAYERS,
                                                 dim_feedforward=T_DIM_FEEDFORWARD,
                                                 num_classes=NUM_CLASSES,
                                                 dropout=T_DROPOUT).to(device)
    t_optimizer = optim.AdamW(transformer_model.parameters(), lr=T_LR)
    # Consider using weighted loss if sampler is not used or if further balancing is needed
    # t_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    t_criterion = nn.CrossEntropyLoss() # Using sampler, so standard CE loss is often fine

    if args.load_transformer:
        if os.path.exists(T_MODEL_SAVE_PATH):
            print(f"Loading pre-trained Transformer model from {T_MODEL_SAVE_PATH}")
            transformer_model.load_state_dict(torch.load(T_MODEL_SAVE_PATH, map_location=device))
            transformer_model.eval()
        else:
            print(f"Error: --load-transformer specified, but model file not found at {T_MODEL_SAVE_PATH}. Exiting.")
            exit()
    else:
        print("Training Transformer model...")
        train_transformer(transformer_model, transformer_train_loader, transformer_test_loader,
                          t_optimizer, t_criterion, T_EPOCHS, device, label_map)

    # Final Evaluation on Test Set
    print("\n--- Evaluating Final Transformer Model on Test Set ---")
    try:
        transformer_model.load_state_dict(torch.load(T_MODEL_SAVE_PATH, map_location=device))
        print("Loaded best saved Transformer model state for final evaluation.")
    except Exception as e:
         print(f"Could not load best Transformer model state ({e}). Evaluating with current model state.")

    test_loss, test_accuracy, test_targets, test_preds = evaluate_transformer(
        transformer_model, transformer_test_loader, t_criterion, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    target_names = [reverse_label_map.get(i, f"Class {i}") for i in range(NUM_CLASSES)] # Handle missing keys if any
    print("\nClassification Report:")
    print(classification_report(test_targets, test_preds, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_targets, test_preds, labels=list(range(NUM_CLASSES))) # Ensure all classes appear
    print(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set (VAE-Transformer)')
    plt.savefig('vae_transformer_confusion_matrix.png')
    plt.show()

    # Plotting
    plt.figure(figsize=(18, 6))

    # VAE Losses
    plt.subplot(1, 3, 1)
    plt.plot(vae_train_recon_losses, label='Train Recon Loss')
    plt.plot(vae_val_recon_losses, label='Val Recon Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss')
    plt.title('VAE Reconstruction Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(vae_train_kl_losses, label='Train KL Loss')
    plt.plot(vae_val_kl_losses, label='Val KL Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('KL Divergence')
    plt.title('VAE KL Divergence')
    plt.legend()
    plt.grid(True)

    # Transformer Accuracy Plot
    plt.subplot(1, 3, 3)
    plt.plot(tf_train_accuracies, label='Train Accuracy')
    plt.plot(tf_val_accuracies, label='Val Accuracy', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Transformer Classifier Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vae_transformer_training_plots.png')
    plt.show()

    print("\nScript finished.")