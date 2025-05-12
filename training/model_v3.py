# model_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- VectorQuantizerEMA ---
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.zeros_like(self.embedding.weight.data))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        if self.training:
            encodings_for_ema = encodings.detach()
            flat_input_for_ema = flat_input.detach()

            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings_for_ema, 0)

            with torch.no_grad():
                n = torch.sum(self.ema_cluster_size)
                self.ema_cluster_size = (
                    (self.ema_cluster_size + self.epsilon)
                    / (n + self.num_embeddings * self.epsilon) * n)

                dw = torch.matmul(encodings_for_ema.t(), flat_input_for_ema)
                self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw

                self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        with torch.no_grad():
            avg_probs = torch.mean(encodings, dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        indices_for_transformer = encoding_indices.view(input_shape[0], -1)
        return loss, quantized, perplexity, indices_for_transformer

    def get_code_indices(self, flat_input):
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices


# --- VQ-VAE Model ---
class VQVAE(nn.Module):
    def __init__(self, input_channels, embedding_dim, num_embeddings, commitment_cost, decay):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, embedding_dim, kernel_size=1, stride=1)
        )

        self.vq_layer = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, input_channels, kernel_size=4, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        z_permuted = z.permute(0, 2, 1).contiguous()
        vq_loss, quantized, perplexity, indices = self.vq_layer(z_permuted)
        quantized_permuted = quantized.permute(0, 2, 1).contiguous()
        x_recon = self.decoder(quantized_permuted)
        return vq_loss, x_recon, perplexity, indices

    def encode(self, x):
        z = self.encoder(x)
        z_permuted = z.permute(0, 2, 1).contiguous()
        flat_z = z_permuted.reshape(-1, self.vq_layer.embedding_dim)
        indices = self.vq_layer.get_code_indices(flat_z)
        indices_seq = indices.view(x.shape[0], -1)
        return indices_seq


# --- Positional Encoding ---
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# --- Transformer Classifier ---
class EEGTransformerClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, nhead, num_layers, dim_feedforward, num_classes, dropout=0.1):
        super(EEGTransformerClassifier, self).__init__()
        self.embedding_dim = embedding_dim

        self.code_embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead,
                                                    dim_feedforward=dim_feedforward, dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.classifier_head = nn.Linear(embedding_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.code_embedding.weight.data.uniform_(-initrange, initrange)
        self.classifier_head.bias.data.zero_()
        self.classifier_head.weight.data.uniform_(-initrange, initrange)

    def forward(self, code_indices):
        batch_size = code_indices.shape[0]
        cls_tokens = self.cls_embedding.expand(batch_size, -1, -1)
        embedded_codes = self.code_embedding(code_indices)
        x = torch.cat((cls_tokens, embedded_codes), dim=1)

        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)

        transformer_output = self.transformer_encoder(x)
        cls_output = transformer_output[:, 0, :]
        logits = self.classifier_head(cls_output)
        return logits
