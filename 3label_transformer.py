import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === Load tokenized data ===
X = np.load("X_tokens_vqvae_3label.npy")
y = np.load("y_label_3class_filtered.npy")  # 3-class filtered label

# === Convert to tensor ===
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# === Dataset and split ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=8)

# === Transformer model ===
class EEGTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len, n_class):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64)
        self.pos = nn.Parameter(torch.randn(1, seq_len, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls = nn.Linear(64, n_class)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        return self.cls(x.mean(dim=1))

# === Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGTransformer(vocab_size=256, seq_len=X.shape[1], n_class=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === Train ===
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# === Evaluation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in val_loader:
        xb = xb.to(device)
        pred = model(xb).argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy())
        y_pred.extend(pred)

# === Metrics ===
target_names = ["Drowsy", "Normal", "Distracted"]
label_ids = [0, 1, 2]  
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

print("Accuracy:", acc)
print("F1-score:", f1)
print("\n", classification_report(
    y_true, y_pred,
    labels=label_ids,
    target_names=target_names,
    zero_division=0
))

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred, labels=label_ids)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix (3-label)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix_3label.png")

# === Save classification report ===
with open("classification_report_3label.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1-score (weighted): {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(
        y_true, y_pred,
        labels=label_ids,
        target_names=target_names,
        zero_division=0
    ))

print("Saved classification report to classification_report_3label.txt")
