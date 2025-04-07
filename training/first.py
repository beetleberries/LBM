import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from torch.nn import functional as F
from dotenv import load_dotenv

# --- Configuration ---
# EEG Specific
SFREQ = 500  # Replace with your dataset's sampling frequency (Hz)
N_CHANNELS = 33 # Replace with the number of EEG channels you want to use
ARTIFACT_FREQS = (50, 60) # Frequencies for notch filter (e.g., power line noise)
FILTER_L_FREQ = 1.0
FILTER_H_FREQ = 40.0
ICA_N_COMPONENTS = 20 # Number of ICA components (can be float for variance explained)
ICA_RANDOM_STATE = 42 # For reproducibility

# VQ-VAE Specific
VQVAE_INPUT_DIM = N_CHANNELS # Input dimension for VQ-VAE (number of channels)
VQVAE_SEGMENT_LEN_SEC = 1.0 # Length of EEG segments fed into VQ-VAE (in seconds)
VQVAE_HIDDEN_DIM = 128
VQVAE_EMBEDDING_DIM = 64  # Dimension of each codebook vector (latent space per timestep)
VQVAE_NUM_EMBEDDINGS = 512 # Size of the codebook (number of discrete tokens)
VQVAE_COMMITMENT_COST = 0.25

# Transformer Specific
TRANSFORMER_EMBEDDING_DIM = VQVAE_EMBEDDING_DIM # Must match VQ-VAE embedding dim if using its vectors directly
TRANSFORMER_NHEAD = 4 #8
TRANSFORMER_NUM_ENCODER_LAYERS = 2 #6
TRANSFORMER_DIM_FEEDFORWARD = 64 # 512
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_SEQUENCE_LENGTH = 20 #50  How many VQ tokens to use as input sequence

# Training Specific
BATCH_SIZE_VQVAE = 32
BATCH_SIZE_TRANSFORMER = 32
LEARNING_RATE_VQVAE = 1e-4
LEARNING_RATE_TRANSFORMER = 1e-4
EPOCHS_VQVAE = 5 # Adjust as needed
EPOCHS_TRANSFORMER = 2 # Adjust as needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# Data Specific
# Placeholder - replace with your actual file path or logic to find files
EEG_FILE_PATH = 'path/to/your/eeg_data.fif' # Example for .fif file
load_dotenv()
EEG_FILE_PATH = os.getenv("EEG_EXAMPLE", EEG_FILE_PATH)


# --- MNE Preprocessing and Artifact Removal ---

def preprocess_eeg_data(raw_file_path, sfreq, l_freq, h_freq, notch_freqs,
                        ica_n_components, ica_random_state,
                        eog_channels=None):
    """
    Loads EEG data, applies standard preprocessing, and uses ICA for artifact removal.
    Args:
        raw_file_path (str): Path to the EEG data file (e.g., .fif, .edf).
        sfreq (int): Expected sampling frequency.
        l_freq (float): Low cutoff frequency for bandpass filter.
        h_freq (float): High cutoff frequency for bandpass filter.
        notch_freqs (tuple): Frequencies for notch filter.
        ica_n_components (int or float): Number of ICA components.
        ica_random_state (int): Random seed for ICA.
        eog_channels (list, optional): Names of EOG channels for automated ICA component finding. Defaults to None.
    Returns:
        np.ndarray: Preprocessed EEG data (channels, timepoints).
        mne.Info: MNE info object associated with the data.
    """
    try:
        # Load data - adjust reader based on your file type
        if raw_file_path.endswith('.fif'):
            raw = mne.io.read_raw_fif(raw_file_path, preload=True)
        else:
            raise ValueError("Unsupported file format. Please adapt the script.")

        print(f"Original data shape: {raw.get_data().shape}")
        print(f"Original sfreq: {raw.info['sfreq']}")

        # --- Basic Preprocessing ---
        # Resample if necessary (optional, depends on model requirements)
        # if raw.info['sfreq'] != sfreq:
        #     print(f"Resampling data to {sfreq} Hz...")
        #     raw.resample(sfreq, npad='auto')

        # Select only EEG channels (adjust channel type/names as needed)
        raw.pick_types(meg=False, eeg=True, stim=False, eog=eog_channels is not None, ecg=False, exclude='bads')
        if eog_channels:
             raw.set_channel_types({ch: 'eog' for ch in eog_channels})

        print(f"Data shape after channel selection: {raw.get_data().shape}")

        # Apply band-pass filter
        print(f"Applying band-pass filter ({l_freq}-{h_freq} Hz)...")
        raw.filter(l_freq, h_freq, fir_design='firwin', skip_by_annotation='edge')

        # Apply notch filter for power line noise
        if notch_freqs:
            print(f"Applying notch filter ({notch_freqs} Hz)...")
            raw.notch_filter(freqs=notch_freqs, fir_design='firwin')

        # --- ICA for Artifact Removal ---
        print("Setting up ICA...")
        ica = mne.preprocessing.ICA(n_components=ica_n_components,
                                    random_state=ica_random_state,
                                    max_iter='auto')

        # Fit ICA (use a copy to avoid filtering original data for ICA fit if needed)
        # High-pass filtering data for ICA improves stationarity assumption
        raw_filt_ica = raw.copy().filter(l_freq=1.0, h_freq=None, fir_design='firwin')
        print("Fitting ICA...")
        ica.fit(raw_filt_ica)
        del raw_filt_ica # Free memory

        print("Identifying artifact components (EOG/ECG)...")
        # Find EOG components automatically if EOG channels provided
        eog_indices, eog_scores = [], []
        if eog_channels:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)
            print(f"Found {len(eog_indices)} EOG component(s): {eog_indices}")
            ica.exclude.extend(eog_indices)

        # Find ECG components (requires ECG channel or creating ECG epochs)
        # ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation', threshold=0.8)
        # print(f"Found {len(ecg_indices)} ECG component(s): {ecg_indices}")
        # ica.exclude.extend(ecg_indices)

        # --- !!! IMPORTANT: Manual Inspection Recommended !!! ---
        # It's highly recommended to visually inspect components
        print("Plotting ICA components. Please visually inspect and select bad components.")
        ica.plot_sources(raw, show_scrollbars=False, title='ICA Components - Inspect and Close')
        # ica.plot_components() # Plot topomaps
        # After visual inspection, manually add bad indices to ica.exclude if needed
        # Example: ica.exclude.extend([1, 5, 10]) # Add indices of bad components

        print(f"Excluding ICA components: {ica.exclude}")

        # Apply ICA to the original filtered data
        print("Applying ICA to remove artifacts...")
        raw_cleaned = ica.apply(raw.copy()) # Apply to a copy

        # --- Extract Data ---
        data_cleaned = raw_cleaned.get_data()
        print(f"Cleaned data shape: {data_cleaned.shape}")

        return data_cleaned, raw_cleaned.info

    except FileNotFoundError:
        print(f"Error: EEG file not found at {raw_file_path}")
        return None, None
    except Exception as e:
        import traceback
        print(f"An error occurred during preprocessing:")
        print(traceback.format_exc())
        return None, None

# --- VQ-VAE Model ---

class VectorQuantizer(nn.Module):
    """
    Improved VectorQuantizer module with straight-through estimation.
    Reference: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: (Batch, Channels, Time) -> (Batch*Time, Channels=embedding_dim)
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        # Calculate distances: (N, E) x (E, K) -> (N, K)
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        # Find closest encodings: (N,)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Convert indices to one-hot vectors (optional, can work with indices directly)
        # encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten using embedding vectors: (N, E)
        quantized = self.embedding(encoding_indices).view(input_shape) # (B, C, T)


        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs) # Encoder loss
        q_latent_loss = F.mse_loss(quantized, inputs.detach()) # Codebook loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # Reshape encoding_indices back to (Batch, Time)
        encoding_indices_reshaped = encoding_indices.view(input_shape[0], input_shape[2]) # Assuming Time is the last dim after potential transpose

        return quantized, loss, encoding_indices_reshaped # Return indices for Transformer

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(True),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, embedding_dim, kernel_size=1, stride=1) # Map to embedding dim
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
             nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
             nn.BatchNorm1d(hidden_dim),
             ResidualBlock(hidden_dim, hidden_dim),
             ResidualBlock(hidden_dim, hidden_dim),
             nn.ConvTranspose1d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
             nn.BatchNorm1d(hidden_dim // 2),
             nn.ReLU(True),
             nn.ConvTranspose1d(hidden_dim // 2, output_dim, kernel_size=4, stride=2, padding=1),
             # nn.Tanh() # Or Sigmoid depending on data normalization
        )

    def forward(self, x):
        return self.net(x)

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_dim, input_dim) # output_dim matches input_dim

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq_layer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, encoding_indices

# --- Transformer Model ---

class PositionalEncoding(nn.Module):
    """ From https://pytorch.org/tutorials/beginner/transformer_tutorial.html """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EEGTransformerPredictor(nn.Module):
    def __init__(self, num_tokens, embedding_dim, nhead, dim_feedforward, num_encoder_layers, dropout, sequence_length):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout, batch_first=True), # Use batch_first=True
            num_encoder_layers
        )
        self.d_model = embedding_dim
        self.decoder = nn.Linear(embedding_dim, num_tokens) # Predict next token index

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len] - Indices from VQ-VAE
            src_mask: Tensor, shape [seq_len, seq_len] - Optional mask

        Returns:
            output Tensor of shape [batch_size, seq_len, num_tokens] (logits)
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        # Permute for positional encoding if batch_first=False expected: [seq_len, batch_size, embedding_dim]
        # src = src.permute(1, 0, 2)
        # src = self.pos_encoder(src)
        # src = src.permute(1, 0, 2) # Permute back if needed

        # If batch_first=True for TransformerEncoderLayer:
        src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2) # Apply pos encoding correctly

        if src_mask is None:
             # Create causal mask for autoregressive prediction
             size = src.size(1) # sequence length
             src_mask = nn.Transformer.generate_square_subsequent_mask(size).to(src.device)


        output = self.transformer_encoder(src, mask=src_mask)
        output = self.decoder(output) # [batch_size, seq_len, num_tokens]
        return output


# --- Datasets ---

class EEGDatasetVQVAE(Dataset):
    def __init__(self, eeg_data, segment_length_samples):
        self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
        self.segment_length = segment_length_samples
        self.num_channels, self.total_length = self.eeg_data.shape
        self.num_segments = self.total_length // self.segment_length

        if self.num_segments == 0:
             raise ValueError("Data length is shorter than segment length.")

        print(f"VQVAE Dataset: Total length={self.total_length}, Seg length={self.segment_length}, Num segments={self.num_segments}")


    def __len__(self):
        # Return number of *possible* start points for segments
        # Use overlap for more data: return self.total_length - self.segment_length + 1
        return self.num_segments # Non-overlapping for simplicity here

    def __getitem__(self, idx):
        # Non-overlapping segments
        start = idx * self.segment_length
        end = start + self.segment_length
        segment = self.eeg_data[:, start:end]

        # Overlapping segments:
        # start = idx
        # end = start + self.segment_length
        # segment = self.eeg_data[:, start:end]

        return segment


class EEGTokenSequenceDataset(Dataset):
    def __init__(self, token_sequence, sequence_length):
        """
        Args:
            token_sequence (torch.Tensor): Flat tensor of token indices (output from VQVAE).
            sequence_length (int): Length of input sequence for Transformer.
        """
        self.token_sequence = token_sequence
        self.seq_len = sequence_length

        if len(token_sequence) <= sequence_length:
             raise ValueError("Token sequence length must be greater than Transformer sequence length.")

    def __len__(self):
        # Number of sequences we can create
        return len(self.token_sequence) - self.seq_len

    def __getitem__(self, idx):
        # Input sequence: tokens from idx to idx + seq_len - 1
        input_seq = self.token_sequence[idx : idx + self.seq_len]
        # Target token: the token immediately following the input sequence
        target_token = self.token_sequence[idx + self.seq_len]
        return input_seq, target_token


# --- Training Functions ---

def train_vqvae(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    print("Starting VQ-VAE Training...")
    for epoch in range(epochs):
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            x_recon, vq_loss, _ = model(data)

            recon_loss = criterion(x_recon, data) # Reconstruction loss (e.g., MSE)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()

            if batch_idx % 50 == 0:
                 print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                       f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f})")

        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_vq_loss = total_vq_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed ---")
        print(f"  Avg Loss: {avg_loss:.4f} (Avg Recon: {avg_recon_loss:.4f}, Avg VQ: {avg_vq_loss:.4f})")
        print("-" * 30)

    print("VQ-VAE Training Finished.")


def train_transformer(model, dataloader, optimizer, criterion, epochs, device, sequence_length):
    model.train()
    print("Starting Transformer Training...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(dataloader):
            # src: [batch_size, seq_len], tgt: [batch_size]
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()

            # Generate causal mask
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(device)

            # Get model output (logits): [batch_size, seq_len, num_tokens]
            output = model(src, src_mask)

            # We only need to predict the *next* token after the sequence.
            # So, we take the output corresponding to the *last* input token.
            # Output shape: [batch_size, seq_len, num_tokens] -> [batch_size, num_tokens]
            output_last_token = output[:, -1, :]

            # Target shape needs to be [batch_size] for CrossEntropyLoss
            loss = criterion(output_last_token, tgt)

            loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 20 == 0:
                 print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed ---")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print("-" * 30)

    print("Transformer Training Finished.")

# --- Tokenization Function ---

def tokenize_data_with_vqvae(model, eeg_data, segment_length_samples, batch_size, device):
    model.eval()
    all_indices = []
    dataset = EEGDatasetVQVAE(eeg_data, segment_length_samples)
    # Use a dataloader for efficient batch processing, even if shuffle=False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Tokenizing data using trained VQ-VAE...")
    with torch.no_grad():
        for data_batch in dataloader:
            data_batch = data_batch.to(device)
            _, _, indices = model(data_batch) # indices shape: [batch_size, time_steps_in_segment_after_encoding]
            # Flatten the indices from the batch and time dimensions
            all_indices.append(indices.cpu().view(-1)) # Flatten to 1D

    token_sequence = torch.cat(all_indices)
    print(f"Tokenization complete. Generated sequence of length: {len(token_sequence)}")
    return token_sequence


# --- Main Execution ---

if __name__ == "__main__":

    # 1. Preprocess Data
    print("Step 1: Preprocessing EEG Data...")
    # Replace 'eog_channel_name' if you have dedicated EOG channels
    # eog_ch_names = ['EOG1', 'EOG2'] # Example
    eog_ch_names = None # Set to None if no dedicated EOG channels

    preprocessed_data, info = preprocess_eeg_data(
        raw_file_path=EEG_FILE_PATH,
        sfreq=SFREQ,
        l_freq=FILTER_L_FREQ,
        h_freq=FILTER_H_FREQ,
        notch_freqs=ARTIFACT_FREQS,
        ica_n_components=ICA_N_COMPONENTS,
        ica_random_state=ICA_RANDOM_STATE,
        eog_channels=eog_ch_names
    )

    if preprocessed_data is None:
        print("Exiting due to preprocessing error.")
        exit()

    # Optional: Normalize data (e.g., channel-wise z-score)
    mean = np.mean(preprocessed_data, axis=1, keepdims=True)
    std = np.std(preprocessed_data, axis=1, keepdims=True)
    preprocessed_data = (preprocessed_data - mean) / (std + 1e-6) # Add epsilon for stability


    # 2. Train VQ-VAE (or load pre-trained)
    print("\nStep 2: VQ-VAE Setup and Training...")
    vqvae_segment_samples = int(VQVAE_SEGMENT_LEN_SEC * SFREQ) # Calculate samples per segment

    # --- Adjust segment length based on VQ-VAE architecture ---
    # The encoder downsamples (stride=2 twice). Input length needs to be divisible
    # by total stride (e.g., 4 here) for clean reconstruction shapes.
    # A simple check:
    if vqvae_segment_samples % 4 != 0:
        original_len = vqvae_segment_samples
        vqvae_segment_samples = (vqvae_segment_samples // 4) * 4
        print(f"Warning: Adjusting VQ-VAE segment length from {original_len} to {vqvae_segment_samples} samples for divisibility by encoder stride.")
        if vqvae_segment_samples == 0:
             raise ValueError("Segment length too short after stride adjustment.")


    vqvae_dataset = EEGDatasetVQVAE(preprocessed_data, vqvae_segment_samples)
    vqvae_dataloader = DataLoader(vqvae_dataset, batch_size=BATCH_SIZE_VQVAE, shuffle=True, num_workers=4) # Use num_workers

    vqvae_model = VQVAE(
        input_dim=N_CHANNELS,
        hidden_dim=VQVAE_HIDDEN_DIM,
        embedding_dim=VQVAE_EMBEDDING_DIM,
        num_embeddings=VQVAE_NUM_EMBEDDINGS,
        commitment_cost=VQVAE_COMMITMENT_COST
    ).to(DEVICE)

    vqvae_optimizer = optim.Adam(vqvae_model.parameters(), lr=LEARNING_RATE_VQVAE)
    vqvae_criterion = nn.MSELoss() # Reconstruction loss

    # --- Training ---
    train_vqvae(vqvae_model, vqvae_dataloader, vqvae_optimizer, vqvae_criterion, EPOCHS_VQVAE, DEVICE)

    # Optional: Save the trained VQ-VAE model
    torch.save(vqvae_model.state_dict(), 'training/vqvae_eeg_model.pth')
    # To load: vqvae_model.load_state_dict(torch.load('vqvae_eeg_model.pth'))


    # 3. Tokenize Data using Trained VQ-VAE
    print("\nStep 3: Tokenizing Data...")
    token_sequence = tokenize_data_with_vqvae(
        vqvae_model,
        preprocessed_data, # Use the same normalized data
        vqvae_segment_samples,
        BATCH_SIZE_VQVAE, # Can use a larger batch size for inference
        DEVICE
    )


    # 4. Train Transformer (or load pre-trained)
    print("\nStep 4: Transformer Setup and Training...")
    transformer_dataset = EEGTokenSequenceDataset(token_sequence, TRANSFORMER_SEQUENCE_LENGTH)
    # Use drop_last=True if the last batch is smaller than sequence length requirements
    transformer_dataloader = DataLoader(transformer_dataset, batch_size=BATCH_SIZE_TRANSFORMER, shuffle=True, drop_last=True)

    transformer_model = EEGTransformerPredictor(
        num_tokens=VQVAE_NUM_EMBEDDINGS, # Vocabulary size is the codebook size
        embedding_dim=TRANSFORMER_EMBEDDING_DIM,
        nhead=TRANSFORMER_NHEAD,
        dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
        num_encoder_layers=TRANSFORMER_NUM_ENCODER_LAYERS,
        dropout=TRANSFORMER_DROPOUT,
        sequence_length=TRANSFORMER_SEQUENCE_LENGTH
    ).to(DEVICE)

    transformer_optimizer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE_TRANSFORMER)
    # Use CrossEntropyLoss for classification (predicting the next token index)
    transformer_criterion = nn.CrossEntropyLoss()

    # --- Training ---
    train_transformer(transformer_model, transformer_dataloader, transformer_optimizer, transformer_criterion, EPOCHS_TRANSFORMER, DEVICE, TRANSFORMER_SEQUENCE_LENGTH)

    # Optional: Save the trained Transformer model
    torch.save(transformer_model.state_dict(), 'training/transformer_eeg_predictor.pth')
    # To load: transformer_model.load_state_dict(torch.load('transformer_eeg_predictor.pth'))

    print("\nScript Finished.")

    # --- Example: Generating next token prediction ---
    # print("\nExample Prediction:")
    # transformer_model.eval()
    # vqvae_model.eval() # Keep VQVAE in eval mode if needed for input prep

    # # Get a sample sequence from the token dataset
    # sample_input_seq, actual_next_token = transformer_dataset[0] # Get first sequence
    # sample_input_seq = sample_input_seq.unsqueeze(0).to(DEVICE) # Add batch dimension and move to device

    # with torch.no_grad():
    #     output_logits = transformer_model(sample_input_seq)
    #     # Get logits for the last token in the sequence -> prediction for the *next* token
    #     next_token_logits = output_logits[:, -1, :]
    #     predicted_token_index = torch.argmax(next_token_logits, dim=1).item()

    # print(f"Input Sequence (first 10 tokens): {sample_input_seq.squeeze().cpu().numpy()[:10]}...")
    # print(f"Actual Next Token Index: {actual_next_token.item()}")
    # print(f"Predicted Next Token Index: {predicted_token_index}")